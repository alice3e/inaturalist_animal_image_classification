# %%
%pip install chromadb

# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile
from pathlib import Path
from tqdm import tqdm
import json
import chromadb  # Заменили faiss на chromadb
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")


# %% [markdown]
# ## 1. Загрузка датасета

# %%
# Загружаем CSV с информацией об изображениях
df = pd.read_csv('../data/balanced_animals_dataset.csv')
print(f"Всего изображений: {len(df)}")
print(f"\nКлассы: {df['scientific_name'].unique()}")
print(f"\nРаспределение по классам:\n{df['scientific_name'].value_counts()}")

# %% [markdown]
# ## 2. Создание экстрактора эмбеддингов

# %%
class EmbeddingExtractor(nn.Module):
    """Модель для извлечения эмбеддингов из EfficientNet V2 M"""
    
    def __init__(self, base_model):
        super().__init__()
        # Для EfficientNet: features + avgpool
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def load_embedding_extractor(checkpoint_path):
    """Загружает модель и создает экстрактор эмбеддингов"""
    
    # Загружаем checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    params = checkpoint['params']
    
    print(f"Загружена модель: {params['model']['name']}")
    print(f"Количество классов: {params['model']['num_classes']}")
    
    # Создаем базовую модель
    model_name = params['model']['name']
    num_classes = params['model']['num_classes']
    
    if model_name == 'efficientnet_v2_m':
        weights = getattr(models.EfficientNet_V2_M_Weights, params['model']['pretrained'])
        base_model = models.efficientnet_v2_m(weights=weights)
        num_features = base_model.classifier[1].in_features
        base_model.classifier[1] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Модель {model_name} не поддерживается")
    
    # Загружаем веса
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Создаем экстрактор эмбеддингов (без классификационного слоя)
    embedding_extractor = EmbeddingExtractor(base_model)
    embedding_extractor.eval()
    embedding_extractor = embedding_extractor.to(device)
    
    # Определяем размерность эмбеддинга
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_output = embedding_extractor(dummy_input)
        embedding_dim = dummy_output.shape[1]
    
    print(f"Размерность эмбеддинга: {embedding_dim}")
    
    return embedding_extractor, embedding_dim, checkpoint

# %%
# Загружаем модель
model_path = Path('../models/best_model.pth')
embedding_extractor, embedding_dim, checkpoint = load_embedding_extractor(model_path)

# Извлекаем маппинги классов
idx_to_label = checkpoint['idx_to_label']
label_to_idx = checkpoint['label_to_idx']

# %% [markdown]
# ## 3. Подготовка DataLoader

# %%
class ImageDataset(Dataset):
    """Dataset для загрузки изображений"""
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.base_path = Path('../animal_images')
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        species = row['scientific_name'].replace(' ', '_')
        img_path = self.base_path / species / f"{row['uuid']}.jpg"
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Ошибка загрузки {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        # Возвращаем изображение и метаданные
        metadata = {
            'uuid': row['uuid'],
            'scientific_name': row['scientific_name'],
            'path': str(img_path.relative_to(Path('../'))),
            'index': idx
        }
        
        return image, metadata


# Трансформации (такие же как при валидации)
transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Создаем dataset и dataloader
dataset = ImageDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Создан DataLoader с {len(dataset)} изображениями")

# %% [markdown]
# ## 4. Извлечение эмбеддингов

# %%
def extract_embeddings(model, dataloader, device):
    """Извлекает эмбеддинги для всех изображений"""
    
    all_embeddings = []
    all_metadata = []
    
    model.eval()
    
    with torch.no_grad():
        for images, metadata_batch in tqdm(dataloader, desc="Извлечение эмбеддингов"):
            images = images.to(device)
            
            # Получаем эмбеддинги
            embeddings = model(images)
            
            embeddings_np = embeddings.cpu().numpy().astype(np.float32)
            embeddings_np = np.ascontiguousarray(embeddings_np)
            all_embeddings.append(embeddings_np)
            
            # Сохраняем метаданные
            for i in range(len(metadata_batch['uuid'])):
                meta = {
                    'uuid': metadata_batch['uuid'][i],
                    'scientific_name': metadata_batch['scientific_name'][i],
                    'path': metadata_batch['path'][i],
                    'embedding_index': len(all_metadata)
                }
                all_metadata.append(meta)
    
    # Объединяем все эмбеддинги
    all_embeddings = np.vstack(all_embeddings)
    all_embeddings = np.ascontiguousarray(all_embeddings, dtype=np.float32)
    
    print(f"\nИзвлечено эмбеддингов: {all_embeddings.shape}")
    print(f"Метаданных: {len(all_metadata)}")
    print(f"Тип данных: {all_embeddings.dtype}")
    print(f"Contiguous: {all_embeddings.flags['C_CONTIGUOUS']}")
    
    return all_embeddings, all_metadata


# %%
# Извлекаем эмбеддинги
embeddings, metadata = extract_embeddings(embedding_extractor, dataloader, device)

print(f"\nФорма массива эмбеддингов: {embeddings.shape}")
print(f"Тип данных: {embeddings.dtype}")
print(f"\nПример метаданных первого изображения:")
print(json.dumps(metadata[0], indent=2, ensure_ascii=False))

# %% [markdown]
# ## 5. Построение chroma индекса

# %%
def build_chromadb_collection(embeddings, metadata, collection_name="animal_embeddings"):
    """Строит ChromaDB коллекцию для поиска похожих векторов
    
    Args:
        embeddings: numpy array с эмбеддингами (n_samples, embedding_dim)
        metadata: список словарей с метаданными для каждого изображения
        collection_name: название коллекции
    
    Returns:
        client: ChromaDB клиент
        collection: ChromaDB коллекция
    """
    
    print(f"Создание ChromaDB коллекции '{collection_name}'...")
    
    # Создаем persistent клиент (сохраняется на диск)
    chroma_path = Path('../embeddings/chromadb')
    chroma_path.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    # Удаляем коллекцию если существует (для чистого старта)
    try:
        client.delete_collection(name=collection_name)
        print("Старая коллекция удалена")
    except:
        pass
    
    # Создаем новую коллекцию с косинусным расстоянием
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # cosine, l2, или ip
    )
    
    # Подготавливаем данные для ChromaDB
    ids = [meta['uuid'] for meta in metadata]
    embeddings_list = embeddings.tolist()
    
    # Подготавливаем метаданные (ChromaDB требует строковые значения)
    metadatas = [
        {
            'scientific_name': meta['scientific_name'],
            'path': meta['path'],
            'embedding_index': str(meta['embedding_index'])
        }
        for meta in metadata
    ]
    
    # Добавляем данные батчами (ChromaDB рекомендует батчи ~40000)
    batch_size = 1000
    num_batches = (len(ids) + batch_size - 1) // batch_size
    
    print(f"Добавление {len(ids)} векторов в {num_batches} батчах...")
    
    for i in tqdm(range(0, len(ids), batch_size), desc="Загрузка в ChromaDB"):
        batch_end = min(i + batch_size, len(ids))
        
        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings_list[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
    
    print(f"✓ Коллекция создана. Количество векторов: {collection.count()}")
    
    return client, collection


# %%
# Строим ChromaDB коллекцию
client, collection = build_chromadb_collection(embeddings, metadata)

# %% [markdown]
# ## 6. Тестирование поиска

# %%
def test_search(collection, metadata, test_idx=0, k=10):
    """Тестирует поиск похожих изображений в ChromaDB"""
    
    # Получаем UUID тестового изображения
    query_uuid = metadata[test_idx]['uuid']
    
    print(f"\nТестовое изображение (индекс {test_idx}):")
    print(f"  UUID: {metadata[test_idx]['uuid']}")
    print(f"  Класс: {metadata[test_idx]['scientific_name']}")
    print(f"  Путь: {metadata[test_idx]['path']}")
    
    # Получаем эмбеддинг из коллекции
    query_result = collection.get(
        ids=[query_uuid],
        include=['embeddings']
    )
    
    query_embedding = query_result['embeddings'][0]
    
    # Ищем k ближайших соседей
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=['metadatas', 'distances']
    )
    
    print(f"\nТоп-{k} похожих изображений:")
    for i, (uuid, distance, meta) in enumerate(zip(
        results['ids'][0], 
        results['distances'][0], 
        results['metadatas'][0]
    )):
        # Для косинусного расстояния: similarity = 1 - distance
        similarity = (1 - distance) * 100
        print(f"  {i+1}. Похожесть: {similarity:.2f}% | Класс: {meta['scientific_name']} | UUID: {uuid}")
    
    return results


# %%
# Тестируем поиск на нескольких примерах
print("=" * 80)
print("ТЕСТ 1: Первое изображение")
print("=" * 80)
test_search(collection, metadata, test_idx=0, k=10)

print("\n" + "=" * 80)
print("ТЕСТ 2: Случайное изображение")
print("=" * 80)
random_idx = np.random.randint(0, len(metadata))
test_search(collection, metadata, test_idx=random_idx, k=10)


# %% [markdown]
# ## 7. Сохранение индекса и метаданных

# %%
# Создаем директорию для сохранения
embeddings_dir = Path('../embeddings')
embeddings_dir.mkdir(exist_ok=True)

# ChromaDB уже сохранен в persistent mode
print(f"✓ ChromaDB сохранен в: {embeddings_dir / 'chromadb'}")

# Сохраняем метаданные
metadata_dict = {
    'images': metadata,
    'metadata': {
        'total_images': len(metadata),
        'embedding_dim': embedding_dim,
        'model': checkpoint['params']['model']['name'],
        'use_cosine_similarity': True,
        'created_at': datetime.now().isoformat(),
        'classes': list(idx_to_label.values()),
        'collection_name': 'animal_embeddings'
    }
}

metadata_path = embeddings_dir / 'image_metadata.json'
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
print(f"✓ Метаданные сохранены: {metadata_path}")

# Опционально: сохраняем сами эмбеддинги (для анализа)
embeddings_path = embeddings_dir / 'embeddings.npy'
np.save(embeddings_path, embeddings)
print(f"✓ Эмбеддинги сохранены: {embeddings_path}")

# Статистика размеров
import shutil
chroma_size = sum(f.stat().st_size for f in (embeddings_dir / 'chromadb').rglob('*') if f.is_file())
print(f"\nВсе файлы сохранены в директории: {embeddings_dir}")
print(f"Размер ChromaDB: {chroma_size / 1024 / 1024:.2f} MB")
print(f"Размер метаданных: {metadata_path.stat().st_size / 1024:.2f} KB")
print(f"Размер эмбеддингов: {embeddings_path.stat().st_size / 1024 / 1024:.2f} MB")


# %% [markdown]
# ## 8. Проверка загрузки

# %%
# Проверяем, что ChromaDB можно загрузить обратно
print("Проверка загрузки ChromaDB...")

# Загружаем существующий клиент
loaded_client = chromadb.PersistentClient(path=str(embeddings_dir / 'chromadb'))
loaded_collection = loaded_client.get_collection(name="animal_embeddings")
print(f"✓ ChromaDB загружен. Количество векторов: {loaded_collection.count()}")

with open(metadata_path, 'r', encoding='utf-8') as f:
    loaded_metadata = json.load(f)
print(f"✓ Метаданные загружены. Количество изображений: {loaded_metadata['metadata']['total_images']}")

loaded_embeddings = np.load(embeddings_path)
print(f"✓ Эмбеддинги загружены. Форма: {loaded_embeddings.shape}")

print("\n✅ Все файлы успешно сохранены и загружены!")

# Тестовый поиск на загруженной коллекции
print("\n" + "=" * 80)
print("ТЕСТ: Поиск в загруженной коллекции")
print("=" * 80)
test_idx = 100
test_search(loaded_collection, loaded_metadata['images'], test_idx=test_idx, k=5)


# %% [markdown]
# ## 9. Статистика и визуализация

# %%
# Статистика по классам
class_counts = {}
for meta in metadata:
    class_name = meta['scientific_name']
    class_counts[class_name] = class_counts.get(class_name, 0) + 1

print("Распределение изображений по классам:")
for class_name, count in sorted(class_counts.items()):
    print(f"  {class_name}: {count} изображений")

print(f"\nВсего классов: {len(class_counts)}")
print(f"Всего изображений: {sum(class_counts.values())}")

# %% [markdown]
# ## Готово! 🎉
# 
# Индекс эмбеддингов успешно построен и сохранен. Теперь можно использовать его в веб-приложении для поиска похожих изображений.
# 
# **Созданные файлы:**
# - `embeddings/faiss_index.bin` - FAISS индекс для быстрого поиска
# - `embeddings/image_metadata.json` - метаданные изображений
# - `embeddings/embeddings.npy` - сохраненные эмбеддинги (опционально)
# 
# **Следующие шаги:**
# 1. Создать модули для веб-приложения (`embedding_extractor.py`, `vector_db.py`, `similarity_search.py`)
# 2. Интегрировать поиск похожих изображений в UI
# 3. Протестировать работу системы


