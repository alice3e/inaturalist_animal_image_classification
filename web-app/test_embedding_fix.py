"""
Тестовый скрипт для проверки работы embedding_extractor с ResNet50
"""

import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

from embedding_extractor import create_embedding_extractor
from PIL import Image
import torch

def test_embedding_extractor():
    """Тестирует создание экстрактора эмбеддингов"""
    
    # Путь к модели
    model_path = Path(__file__).parent.parent / "models" / "best_model.pth"
    
    if not model_path.exists():
        print(f"❌ Модель не найдена: {model_path}")
        return False
    
    try:
        # Создаем экстрактор
        print("Загрузка модели...")
        embedding_extractor, embedding_dim, params = create_embedding_extractor(
            model_path, 
            device=torch.device("cpu")  # Используем CPU для теста
        )
        
        print(f"✅ Модель загружена успешно")
        print(f"   Тип модели: {params['model']['name']}")
        print(f"   Размерность эмбеддинга: {embedding_dim}")
        
        # Проверяем, что модель может обработать изображение
        print("\nТестирование извлечения эмбеддинга...")
        dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        from embedding_extractor import extract_embedding_from_image
        embedding = extract_embedding_from_image(
            dummy_image,
            embedding_extractor,
            params,
            torch.device("cpu"),
            normalize=True
        )
        
        print(f"✅ Эмбеддинг извлечен успешно")
        print(f"   Форма эмбеддинга: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedding_extractor()
    sys.exit(0 if success else 1)