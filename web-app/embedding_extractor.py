"""
Модуль для извлечения эмбеддингов из изображений
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import numpy as np


class EmbeddingExtractor(nn.Module):
    """Модель для извлечения эмбеддингов из различных архитектур"""
    
    def __init__(self, base_model, model_type='efficientnet'):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'efficientnet':
            # Для EfficientNet: features + avgpool
            self.features = base_model.features
            self.avgpool = base_model.avgpool
        elif model_type == 'resnet':
            # Для ResNet: все слои кроме fc
            self.conv1 = base_model.conv1
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool
            self.layer1 = base_model.layer1
            self.layer2 = base_model.layer2
            self.layer3 = base_model.layer3
            self.layer4 = base_model.layer4
            self.avgpool = base_model.avgpool
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        
    def forward(self, x):
        if self.model_type == 'efficientnet':
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        elif self.model_type == 'resnet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        
        return x


def create_embedding_extractor(
    checkpoint_path: Path,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, int, dict]:
    """
    Создает экстрактор эмбеддингов из обученной модели
    
    Args:
        checkpoint_path: путь к файлу модели (.pth)
        device: устройство для вычислений (CPU/GPU/MPS)
        
    Returns:
        embedding_extractor: модель для извлечения эмбеддингов
        embedding_dim: размерность эмбеддинга
        params: параметры модели
    """
    
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    print(f"Используемое устройство: {device}")
    
    # Загружаем checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    params = checkpoint['params']
    
    # Создаем базовую модель
    model_name = params['model']['name']
    num_classes = params['model']['num_classes']
    
    if model_name == 'efficientnet_v2_m':
        weights = getattr(models.EfficientNet_V2_M_Weights, params['model']['pretrained'])
        base_model = models.efficientnet_v2_m(weights=weights)
        num_features = int(base_model.classifier[1].in_features)
        base_model.classifier[1] = nn.Linear(num_features, num_classes)
        model_type = 'efficientnet'
        
    elif model_name == 'resnet50':
        weights = getattr(models.ResNet50_Weights, params['model']['pretrained'])
        base_model = models.resnet50(weights=weights)
        num_features = int(base_model.fc.in_features)
        base_model.fc = nn.Linear(num_features, num_classes)
        model_type = 'resnet'
        
    elif model_name == 'resnet101':
        weights = getattr(models.ResNet101_Weights, params['model']['pretrained'])
        base_model = models.resnet101(weights=weights)
        num_features = int(base_model.fc.in_features)
        base_model.fc = nn.Linear(num_features, num_classes)
        model_type = 'resnet'
        
    else:
        raise ValueError(f"Модель {model_name} не поддерживается")
    
    # Загружаем веса
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Создаем экстрактор эмбеддингов (без классификационного слоя)
    embedding_extractor = EmbeddingExtractor(base_model, model_type=model_type)
    embedding_extractor.eval()
    embedding_extractor = embedding_extractor.to(device)
    
    # Убеждаемся, что все параметры модели на правильном устройстве
    for param in embedding_extractor.parameters():
        param.data = param.data.to(device)
    
    # Определяем размерность эмбеддинга
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_output = embedding_extractor(dummy_input)
        embedding_dim = dummy_output.shape[1]
    
    return embedding_extractor, embedding_dim, params


def extract_embedding_from_image(
    image: Image.Image,
    model: nn.Module,
    params: dict,
    device: torch.device,
    normalize: bool = True
) -> np.ndarray:
    """
    Извлекает эмбеддинг из одного изображения
    
    Args:
        image: PIL Image объект
        model: модель для извлечения эмбеддингов
        params: параметры модели (для трансформаций)
        device: устройство для вычислений
        normalize: нормализовать ли эмбеддинг для косинусного расстояния
        
    Returns:
        embedding: numpy array с эмбеддингом
    """
    
    # Создаем трансформации (такие же как при валидации)
    transform = transforms.Compose([
        transforms.Resize(params['augmentation']['resize']),
        transforms.CenterCrop(params['augmentation']['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Применяем трансформации
    image_tensor = transform(image)
    # Явно указываем, что это torch.Tensor
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image_tensor)}")
    image_tensor = image_tensor.unsqueeze(0)  # Добавляем batch dimension
    print(f"Изображение до перемещения: device={image_tensor.device}")
    image_tensor = image_tensor.to(device)  # Перемещаем на то же устройство, что и модель
    print(f"Изображение после перемещения: device={image_tensor.device}")
    print(f"Модель: device={next(model.parameters()).device}")
    
    # Извлекаем эмбеддинг
    model.eval()
    with torch.no_grad():
        embedding = model(image_tensor)
    
    # Конвертируем в numpy
    embedding_np = embedding.cpu().numpy()
    
    # Нормализуем для косинусного расстояния
    if normalize:
        norm = np.linalg.norm(embedding_np, axis=1, keepdims=True)
        embedding_np = embedding_np / (norm + 1e-8)
    
    return embedding_np