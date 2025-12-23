import torch
from torchvision import transforms
from PIL import Image


def predict_image(image, model, idx_to_label, params, device):
    """
    Выполняет предсказание для изображения
    
    Args:
        image: PIL Image объект
        model: загруженная модель
        idx_to_label: маппинг индексов в названия классов
        params: параметры модели
        device: устройство для инференса
        
    Returns:
        predicted_class: название предсказанного класса
        confidence: уверенность в процентах
        all_probabilities: словарь со всеми вероятностями
    """
    # Создаем трансформации для валидации (как в обучении)
    transform = transforms.Compose([
        transforms.Resize(params['augmentation']['resize']),
        transforms.CenterCrop(params['augmentation']['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Применяем трансформации
    image_tensor = transform(image).unsqueeze(0)  # Добавляем batch dimension
    image_tensor = image_tensor.to(device)
    
    # Выполняем предсказание
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    # Получаем название класса
    predicted_idx = predicted_idx.item()
    predicted_class = idx_to_label[predicted_idx]
    confidence_percent = confidence.item() * 100
    
    # Создаем словарь со всеми вероятностями
    all_probabilities = {}
    probs = probabilities[0].cpu().numpy()
    for idx, prob in enumerate(probs):
        class_name = idx_to_label[idx]
        all_probabilities[class_name] = float(prob * 100)
    
    return predicted_class, confidence_percent, all_probabilities