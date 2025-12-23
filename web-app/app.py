import streamlit as st
import torch
from PIL import Image
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

from model_loader import load_model
from inference import predict_image

# Настройка страницы
st.set_page_config(
    page_title="Классификация волков",
    page_icon="🐺",
    layout="centered"
)

# Заголовок
st.title("🐺 Классификация видов волков")
st.markdown("Загрузите изображение волков для определения его вида")

# Инициализация модели (кэшируется)
@st.cache_resource
def initialize_model():
    # Получаем путь к модели относительно текущей директории или абсолютный путь
    current_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    model_path = current_dir.parent / "models" / "best_model.pth"
    
    # Если файл не найден, пробуем альтернативный путь
    if not model_path.exists():
        model_path = Path.cwd() / "models" / "best_model.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
    
    return load_model(model_path)

try:
    model, idx_to_label, params, device = initialize_model()
    st.success("✅ Модель успешно загружена")
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

# Загрузка изображения
uploaded_file = st.file_uploader(
    "Выберите изображение", 
    type=["jpg", "jpeg", "png"],
    help="Поддерживаются форматы: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Отображение загруженного изображения
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Загруженное изображение", use_container_width=True)
    
    # Кнопка для предсказания
    if st.button("🔍 Определить вид", type="primary", use_container_width=True):
        with st.spinner("Анализ изображения..."):
            try:
                # Получение предсказания
                predicted_class, confidence, all_probabilities = predict_image(
                    image, model, idx_to_label, params, device
                )
                
                with col2:
                    # Результат предсказания
                    st.markdown("### Результат")
                    st.markdown(f"**Вид:** `{predicted_class}`")
                    st.markdown(f"**Уверенность:** `{confidence:.1f}%`")
                    
                    # Прогресс-бар для уверенности
                    st.progress(confidence / 100)
                
                # Показать все вероятности
                st.markdown("---")
                st.markdown("### Вероятности для всех классов")
                
                # Сортируем по убыванию вероятности
                sorted_probs = sorted(
                    all_probabilities.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                for class_name, prob in sorted_probs:
                    st.markdown(f"**{class_name}:** {prob:.2f}%")
                    st.progress(prob / 100)
                
            except Exception as e:
                st.error(f"❌ Ошибка при предсказании: {e}")
                st.exception(e)

# Информация о классах
with st.expander("ℹ️ Информация о классах"):
    st.markdown("Модель обучена распознавать следующие виды:")
    for idx, label in sorted(idx_to_label.items()):
        st.markdown(f"- **{label}** (класс {idx})")

# Футер
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <small>Модель: EfficientNet V2 M | Обучена на датасете волков</small>
    </div>
    """,
    unsafe_allow_html=True
)