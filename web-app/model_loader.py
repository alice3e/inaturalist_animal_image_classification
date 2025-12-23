import torch
from torchvision import models
import torch.nn as nn
from pathlib import Path


def load_model(model_path):
    """
    Загружает обученную модель из файла .pth
    
    Args:
        model_path: путь к файлу модели
        
    Returns:
        model: загруженная модель
        idx_to_label: маппинг индексов в названия классов
        params: параметры модели
        device: устройство для инференса
    """
    # Определяем устройство
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Загружаем checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Извлекаем параметры
    params = checkpoint['params']
    idx_to_label = checkpoint['idx_to_label']
    
    # Создаем архитектуру модели
    model = create_model_architecture(params)
    
    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Переводим в режим оценки
    model.eval()
    model = model.to(device)
    
    return model, idx_to_label, params, device


def create_model_architecture(params):
    """
    Создает архитектуру модели на основе параметров
    
    Args:
        params: словарь с параметрами модели
        
    Returns:
        model: модель с правильной архитектурой
    """
    model_name = params['model']['name']
    num_classes = params['model']['num_classes']
    
    if model_name == 'resnet50':
        weights = getattr(models.ResNet50_Weights, params['model']['pretrained'])
        model = models.resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_name == 'resnet101':
        weights = getattr(models.ResNet101_Weights, params['model']['pretrained'])
        model = models.resnet101(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_name == 'efficientnet_v2_m':
        weights = getattr(models.EfficientNet_V2_M_Weights, params['model']['pretrained'])
        model = models.efficientnet_v2_m(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        
    elif model_name == 'convnext_base':
        weights = getattr(models.ConvNeXt_Base_Weights, params['model']['pretrained'])
        model = models.convnext_base(weights=weights)
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, num_classes)
        
    elif model_name == 'vit_b_16':
        weights = getattr(models.ViT_B_16_Weights, params['model']['pretrained'])
        model = models.vit_b_16(weights=weights)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")
    
    return model