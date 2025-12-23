"""
Модуль для поиска похожих изображений с использованием ChromaDB
"""

from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import torch

from embedding_extractor import create_embedding_extractor, extract_embedding_from_image
from vector_db import VectorDatabase


class SimilaritySearch:
    """Класс для поиска похожих изображений с ChromaDB"""
    
    def __init__(
        self,
        model_path: Path,
        chroma_path: Path,
        metadata_path: Path,
        collection_name: str = "animal_embeddings",
        device: Optional[torch.device] = None
    ):
        """
        Инициализация системы поиска
        
        Args:
            model_path: путь к файлу модели
            chroma_path: путь к директории ChromaDB
            metadata_path: путь к метаданным
            collection_name: название коллекции в ChromaDB
            device: устройство для вычислений
        """
        # Загружаем модель для извлечения эмбеддингов
        self.embedding_model, self.embedding_dim, self.params = create_embedding_extractor(
            model_path, device
        )
        self.device = device if device else torch.device("cpu")
        
        # Загружаем векторную БД
        self.vector_db = VectorDatabase(
            chroma_path=chroma_path,
            metadata_path=metadata_path,
            collection_name=collection_name
        )
        
        print(f"✓ Модель загружена (embedding_dim={self.embedding_dim})")
        print(f"✓ ChromaDB загружена ({self.vector_db.collection.count()} векторов)")
    
    def find_similar(
        self,
        query_image: Image.Image,
        top_k: int = 10,
        return_paths: bool = True
    ) -> List[Dict]:
        """
        Находит похожие изображения
        
        Args:
            query_image: изображение для поиска
            top_k: количество похожих изображений
            return_paths: возвращать ли полные пути к файлам
            
        Returns:
            Список словарей с информацией о похожих изображениях:
            [
                {
                    'image_path': 'path/to/image.jpg',
                    'similarity': 95.5,
                    'scientific_name': 'Canis lupus',
                    'uuid': '...',
                    'distance': 0.05
                },
                ...
            ]
        """
        # Извлекаем эмбеддинг из запроса
        query_embedding = extract_embedding_from_image(
            query_image,
            self.embedding_model,
            self.params,
            self.device,
            normalize=True
        )
        
        # Ищем похожие
        distances, uuids, results = self.vector_db.search(
            query_embedding,
            k=top_k
        )
        
        # Форматируем результаты
        formatted_results = []
        for result in results:
            formatted_result = {
                'image_path': result['path'],
                'similarity': result['similarity'],
                'scientific_name': result['scientific_name'],
                'uuid': result['uuid'],
                'distance': result['distance']
            }
            
            # Добавляем полный путь если нужно
            if return_paths:
                full_path = Path(result['path'])
                if not full_path.is_absolute():
                    # Предполагаем, что путь относительно корня проекта
                    full_path = Path(__file__).parent.parent / result['path']
                formatted_result['full_path'] = str(full_path)
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """Возвращает статистику по коллекции"""
        return self.vector_db.get_stats()
    
    def find_similar_by_path(
        self,
        image_path: Path,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Находит похожие изображения по пути к файлу
        
        Args:
            image_path: путь к изображению
            top_k: количество похожих изображений
            
        Returns:
            Список похожих изображений
        """
        image = Image.open(image_path).convert('RGB')
        return self.find_similar(image, top_k)
    
    def find_similar_by_uuid(
        self,
        uuid: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Находит похожие изображения по UUID существующего изображения
        
        Args:
            uuid: UUID изображения из базы данных
            top_k: количество похожих изображений
            
        Returns:
            Список похожих изображений
        """
        # Получаем эмбеддинг по UUID
        image_meta = self.vector_db.get_image_by_uuid(uuid)
        if not image_meta:
            raise ValueError(f"Изображение с UUID '{uuid}' не найдено")
        
        # Получаем путь к изображению
        image_path = Path(__file__).parent.parent / image_meta['path']
        return self.find_similar_by_path(image_path, top_k)


def create_similarity_search(
    model_path: Optional[Path] = None,
    chroma_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    collection_name: str = "animal_embeddings"
) -> SimilaritySearch:
    """
    Создает экземпляр SimilaritySearch с путями по умолчанию
    
    Args:
        model_path: путь к модели (по умолчанию ../models/best_model.pth)
        chroma_path: путь к ChromaDB (по умолчанию ../embeddings/chromadb)
        metadata_path: путь к метаданным (по умолчанию ../embeddings/image_metadata.json)
        collection_name: название коллекции
        
    Returns:
        Экземпляр SimilaritySearch
    """
    base_dir = Path(__file__).parent.parent
    
    if model_path is None:
        model_path = base_dir / "models" / "best_model.pth"
    
    if chroma_path is None:
        chroma_path = base_dir / "embeddings" / "chromadb"
    
    if metadata_path is None:
        metadata_path = base_dir / "embeddings" / "image_metadata.json"
    
    return SimilaritySearch(
        model_path=model_path,
        chroma_path=chroma_path,
        metadata_path=metadata_path,
        collection_name=collection_name
    )