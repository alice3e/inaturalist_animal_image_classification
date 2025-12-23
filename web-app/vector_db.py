"""
Модуль для работы с векторной базой данных ChromaDB
"""

import chromadb
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime


class VectorDatabase:
    """Класс для работы с ChromaDB индексом и метаданными изображений"""
    
    def __init__(
        self,
        chroma_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        collection_name: str = "animal_embeddings"
    ):
        """
        Инициализация векторной БД
        
        Args:
            chroma_path: путь к директории ChromaDB
            metadata_path: путь к JSON файлу с метаданными
            collection_name: название коллекции в ChromaDB
        """
        self.client = None
        self.collection = None
        self.metadata = None
        self.embedding_dim = None
        self.collection_name = collection_name
        
        if chroma_path and metadata_path:
            self.load(chroma_path, metadata_path)
    
    def create_collection(
        self,
        embedding_dim: int,
        use_cosine: bool = True
    ) -> None:
        """
        Создает новую ChromaDB коллекцию
        
        Args:
            embedding_dim: размерность эмбеддингов
            use_cosine: использовать ли косинусное расстояние
        """
        self.embedding_dim = embedding_dim
        
        # Создаем persistent клиент
        if not hasattr(self, 'chroma_path'):
            raise ValueError("Не указан путь для ChromaDB")
        
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        
        # Удаляем коллекцию если существует
        try:
            self.client.delete_collection(name=self.collection_name)
        except:
            pass
        
        # Создаем новую коллекцию
        if use_cosine:
            space = "cosine"
        else:
            space = "l2"
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": space}
        )
        
        self.metadata = {
            'images': [],
            'metadata': {
                'total_images': 0,
                'embedding_dim': embedding_dim,
                'use_cosine_similarity': use_cosine,
                'created_at': datetime.now().isoformat(),
                'collection_name': self.collection_name
            }
        }
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        image_metadata: List[Dict],
        batch_size: int = 1000
    ) -> None:
        """
        Добавляет эмбеддинги в коллекцию
        
        Args:
            embeddings: массив эмбеддингов [N, embedding_dim]
            image_metadata: список метаданных для каждого изображения
            batch_size: размер батча для добавления
        """
        if self.collection is None:
            raise ValueError("Коллекция не создана. Используйте create_collection() или load()")
        
        # Подготавливаем данные для ChromaDB
        ids = [meta['uuid'] for meta in image_metadata]
        embeddings_list = embeddings.tolist()
        
        # Подготавливаем метаданные
        metadatas = [
            {
                'scientific_name': meta['scientific_name'],
                'path': meta['path'],
                'embedding_index': str(meta.get('embedding_index', i))
            }
            for i, meta in enumerate(image_metadata)
        ]
        
        # Добавляем данные батчами
        num_batches = (len(ids) + batch_size - 1) // batch_size
        
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings_list[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
        
        # Обновляем метаданные
        if self.metadata is None:
            self.metadata = {
                'images': [],
                'metadata': {
                    'total_images': 0,
                    'embedding_dim': self.embedding_dim,
                    'use_cosine_similarity': True,
                    'created_at': datetime.now().isoformat(),
                    'collection_name': self.collection_name
                }
            }
        self.metadata['images'] = image_metadata
        self.metadata['metadata']['total_images'] = len(image_metadata)
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> Tuple[List[float], List[str], List[Dict]]:
        """
        Ищет k ближайших соседей для запроса
        
        Args:
            query_embedding: эмбеддинг запроса [1, embedding_dim]
            k: количество ближайших соседей
            
        Returns:
            distances: расстояния до ближайших соседей
            indices: UUID найденных изображений
            results: метаданные найденных изображений
        """
        if self.collection is None:
            raise ValueError("Коллекция не загружена")
        
        # Ищем похожие
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            include=['metadatas', 'distances']
        )
        
        # Форматируем результаты
        distances = results['distances'][0] if results['distances'] else []
        uuids = results['ids'][0] if results['ids'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        # Создаем список результатов
        formatted_results = []
        for i, (uuid, distance, meta) in enumerate(zip(uuids, distances, metadatas)):
            result = {
                'uuid': uuid,
                'distance': float(distance),
                'scientific_name': meta['scientific_name'],
                'path': meta['path'],
                'embedding_index': int(meta['embedding_index'])
            }
            
            # Конвертируем расстояние в процент похожести
            # Для косинусного расстояния: similarity = 1 - distance
            similarity = (1 - distance) * 100
            result['similarity'] = float(similarity)
            
            formatted_results.append(result)
        
        return distances, uuids, formatted_results
    
    def save(
        self,
        chroma_path: Path,
        metadata_path: Path
    ) -> None:
        """
        Сохраняет коллекцию и метаданные на диск
        
        Args:
            chroma_path: путь для сохранения ChromaDB
            metadata_path: путь для сохранения метаданных
        """
        if self.collection is None:
            raise ValueError("Коллекция не создана")
        
        # ChromaDB автоматически сохраняется в persistent mode
        # Просто сохраняем метаданные
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def load(
        self,
        chroma_path: Path,
        metadata_path: Path
    ) -> None:
        """
        Загружает коллекцию и метаданные с диска
        
        Args:
            chroma_path: путь к директории ChromaDB
            metadata_path: путь к метаданным
        """
        self.chroma_path = chroma_path
        
        # Загружаем ChromaDB клиент
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        
        # Получаем коллекцию
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception as e:
            # Проверяем, существует ли коллекция
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            if self.collection_name not in collection_names:
                raise ValueError(f"Коллекция '{self.collection_name}' не найдена. Доступные коллекции: {collection_names}")
            else:
                raise ValueError(f"Не удалось загрузить коллекцию '{self.collection_name}': {e}")
        
        # Загружаем метаданные
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.embedding_dim = self.metadata['metadata']['embedding_dim']
    
    def get_stats(self) -> Dict:
        """Возвращает статистику по коллекции"""
        if self.collection is None or self.metadata is None:
            return {}
        
        stats = {
            'total_vectors': self.collection.count(),
            'embedding_dim': self.embedding_dim,
            'total_images': self.metadata['metadata']['total_images'],
            'use_cosine': self.metadata['metadata'].get('use_cosine_similarity', True),
            'created_at': self.metadata['metadata'].get('created_at', 'unknown'),
            'collection_name': self.collection_name
        }
        
        # Статистика по классам
        class_counts = {}
        for img in self.metadata['images']:
            class_name = img.get('scientific_name', 'unknown')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        stats['classes'] = class_counts
        
        return stats
    
    def get_image_by_uuid(self, uuid: str) -> Optional[Dict]:
        """Получает метаданные изображения по UUID"""
        if self.metadata:
            for img in self.metadata['images']:
                if img.get('uuid') == uuid:
                    return img
        return None
    
    def get_all_images(self) -> List[Dict]:
        """Возвращает метаданные всех изображений"""
        if self.metadata:
            return self.metadata['images']
        return []