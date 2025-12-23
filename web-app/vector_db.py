"""
Модуль для работы с векторной базой данных FAISS
"""

import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime


class VectorDatabase:
    """Класс для работы с FAISS индексом и метаданными изображений"""
    
    def __init__(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None
    ):
        """
        Инициализация векторной БД
        
        Args:
            index_path: путь к FAISS индексу
            metadata_path: путь к JSON файлу с метаданными
        """
        self.index = None
        self.metadata = None
        self.embedding_dim = None
        
        if index_path and metadata_path:
            self.load(index_path, metadata_path)
    
    def create_index(
        self,
        embedding_dim: int,
        use_cosine: bool = True
    ) -> None:
        """
        Создает новый FAISS индекс
        
        Args:
            embedding_dim: размерность эмбеддингов
            use_cosine: использовать ли косинусное расстояние
        """
        self.embedding_dim = embedding_dim
        
        if use_cosine:
            # IndexFlatIP для косинусного расстояния (Inner Product)
            # Требует нормализации векторов
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            # IndexFlatL2 для Euclidean расстояния
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        self.metadata = {
            'images': [],
            'metadata': {
                'total_images': 0,
                'embedding_dim': embedding_dim,
                'use_cosine_similarity': use_cosine,
                'created_at': datetime.now().isoformat()
            }
        }
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        image_metadata: List[Dict],
        normalize: bool = True
    ) -> None:
        """
        Добавляет эмбеддинги в индекс
        
        Args:
            embeddings: массив эмбеддингов [N, embedding_dim]
            image_metadata: список метаданных для каждого изображения
            normalize: нормализовать ли векторы (для косинусного расстояния)
        """
        if self.index is None:
            raise ValueError("Индекс не создан. Используйте create_index() или load()")
        
        # Нормализуем векторы для косинусного расстояния
        if normalize:
            faiss.normalize_L2(embeddings)
        
        # Добавляем в индекс
        self.index.add(embeddings)
        
        # Обновляем метаданные
        start_idx = len(self.metadata['images'])
        for i, meta in enumerate(image_metadata):
            meta['embedding_index'] = start_idx + i
            self.metadata['images'].append(meta)
        
        self.metadata['metadata']['total_images'] = len(self.metadata['images'])
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Ищет k ближайших соседей для запроса
        
        Args:
            query_embedding: эмбеддинг запроса [1, embedding_dim]
            k: количество ближайших соседей
            normalize: нормализовать ли вектор запроса
            
        Returns:
            distances: расстояния до ближайших соседей
            indices: индексы ближайших соседей
            results: метаданные найденных изображений
        """
        if self.index is None:
            raise ValueError("Индекс не загружен")
        
        # Нормализуем вектор запроса
        if normalize:
            faiss.normalize_L2(query_embedding)
        
        # Выполняем поиск
        distances, indices = self.index.search(query_embedding, k)
        
        # Получаем метаданные для найденных изображений
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata['images']):
                result = self.metadata['images'][idx].copy()
                result['distance'] = float(distances[0][i])
                
                # Конвертируем расстояние в процент похожести
                if self.metadata['metadata'].get('use_cosine_similarity', True):
                    # Для косинусного расстояния (Inner Product после нормализации)
                    # Значение от -1 до 1, конвертируем в проценты
                    similarity = (distances[0][i] + 1) / 2 * 100
                else:
                    # Для L2 расстояния конвертируем в похожесть
                    # Чем меньше расстояние, тем больше похожесть
                    similarity = max(0, 100 - distances[0][i] * 10)
                
                result['similarity'] = float(similarity)
                results.append(result)
        
        return distances, indices, results
    
    def save(
        self,
        index_path: Path,
        metadata_path: Path
    ) -> None:
        """
        Сохраняет индекс и метаданные на диск
        
        Args:
            index_path: путь для сохранения FAISS индекса
            metadata_path: путь для сохранения метаданных
        """
        if self.index is None:
            raise ValueError("Индекс не создан")
        
        # Создаем директории если нужно
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем индекс
        faiss.write_index(self.index, str(index_path))
        
        # Сохраняем метаданные
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def load(
        self,
        index_path: Path,
        metadata_path: Path
    ) -> None:
        """
        Загружает индекс и метаданные с диска
        
        Args:
            index_path: путь к FAISS индексу
            metadata_path: путь к метаданным
        """
        # Загружаем индекс
        self.index = faiss.read_index(str(index_path))
        
        # Загружаем метаданные
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.embedding_dim = self.metadata['metadata']['embedding_dim']
    
    def get_stats(self) -> Dict:
        """Возвращает статистику по индексу"""
        if self.index is None or self.metadata is None:
            return {}
        
        stats = {
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'total_images': self.metadata['metadata']['total_images'],
            'use_cosine': self.metadata['metadata'].get('use_cosine_similarity', True),
            'created_at': self.metadata['metadata'].get('created_at', 'unknown')
        }
        
        # Статистика по классам
        class_counts = {}
        for img in self.metadata['images']:
            class_name = img.get('scientific_name', 'unknown')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        stats['classes'] = class_counts
        
        return stats
    
    def get_image_by_index(self, index: int) -> Optional[Dict]:
        """Получает метаданные изображения по индексу"""
        if self.metadata and 0 <= index < len(self.metadata['images']):
            return self.metadata['images'][index]
        return None
    
    def get_all_images(self) -> List[Dict]:
        """Возвращает метаданные всех изображений"""
        if self.metadata:
            return self.metadata['images']
        return []