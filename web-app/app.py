import streamlit as st
import torch
from PIL import Image
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

from model_loader import load_model
from inference import predict_image
from similarity_search import create_similarity_search

# Настройка страницы
st.set_page_config(
    page_title="Классификация животных",
    page_icon="🐺",
    layout="centered"
)

# Создаем вкладки
tab1, tab2 = st.tabs(["🔍 Классификация", "📸 Поиск похожих"])

# Инициализация модели (кэшируется)
@st.cache_resource
def initialize_classification_model():
    # Получаем путь к модели относительно текущей директории или абсолютный путь
    current_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    model_path = current_dir.parent / "models" / "best_model.pth"
    
    # Если файл не найден, пробуем альтернативный путь
    if not model_path.exists():
        model_path = Path.cwd() / "models" / "best_model.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
    
    return load_model(model_path)

# Инициализация поиска похожих (кэшируется)
@st.cache_resource
def initialize_similarity_search():
    try:
        return create_similarity_search()
    except Exception as e:
        st.error(f"❌ Ошибка загрузки системы поиска: {e}")
        return None

# Заголовок для вкладки классификации
with tab1:
    st.title("🐺 Классификация видов животных")
    st.markdown("Загрузите изображение животного для определения его вида")
    
    try:
        model, idx_to_label, params, device = initialize_classification_model()
        st.success("✅ Модель классификации успешно загружена")
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        st.stop()
    
    # Загрузка изображения
    uploaded_file = st.file_uploader(
        "Выберите изображение", 
        type=["jpg", "jpeg", "png"],
        help="Поддерживаются форматы: JPG, JPEG, PNG",
        key="classification_uploader"
    )
    
    if uploaded_file is not None:
        # Отображение загруженного изображения
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Загруженное изображение", use_container_width=True)
        
        # Кнопка для предсказания
        if st.button("🔍 Определить вид", type="primary", use_container_width=True, key="classify_btn"):
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

# Заголовок для вкладки поиска похожих
with tab2:
    st.title("📸 Поиск похожих изображений")
    st.markdown("Загрузите изображение для поиска визуально похожих изображений из базы данных")
    
    similarity_search = initialize_similarity_search()
    
    if similarity_search is None:
        st.error("❌ Система поиска похожих изображений недоступна")
        st.info("Убедитесь, что файлы эмбеддингов созданы в директории `embeddings/`")
        st.stop()
    
    # Статистика базы данных
    stats = similarity_search.get_stats()
    st.success(f"✅ База данных загружена ({stats['total_images']} изображений)")
    
    with st.expander("📊 Статистика базы данных"):
        st.json(stats)
    
    # Загрузка изображения
    uploaded_file = st.file_uploader(
        "Выберите изображение для поиска", 
        type=["jpg", "jpeg", "png"],
        help="Поддерживаются форматы: JPG, JPEG, PNG",
        key="similarity_uploader"
    )
    
    if uploaded_file is not None:
        # Отображение загруженного изображения
        image = Image.open(uploaded_file).convert('RGB')
        
        st.image(image, caption="Загруженное изображение", use_container_width=True)
        
        # Настройки поиска
        col1, col2 = st.columns([1, 1])
        with col1:
            top_k = st.slider("Количество похожих изображений", 1, 20, 10)
        with col2:
            show_similarity = st.checkbox("Показывать процент похожести", value=True)
        
        # Кнопка для поиска
        if st.button("🔍 Найти похожие", type="primary", use_container_width=True, key="similarity_btn"):
            with st.spinner("Поиск похожих изображений..."):
                try:
                    # Поиск похожих изображений
                    similar_images = similarity_search.find_similar(
                        image, 
                        top_k=top_k,
                        return_paths=True
                    )
                    
                    st.markdown("---")
                    st.markdown(f"### Топ-{top_k} похожих изображений")
                    
                    # Отображаем результаты в сетке
                    cols = st.columns(3)
                    for i, result in enumerate(similar_images):
                        col_idx = i % 3
                        
                        with cols[col_idx]:
                            try:
                                # Пытаемся загрузить и отобразить изображение
                                similar_image = Image.open(result['full_path']).convert('RGB')
                                st.image(
                                    similar_image, 
                                    caption=f"{result['scientific_name']} ({result['similarity']:.1f}%)" if show_similarity else result['scientific_name'],
                                    use_container_width=True
                                )
                                
                                # Дополнительная информация при наведении
                                with st.expander("ℹ️ Подробнее"):
                                    st.markdown(f"**UUID:** `{result['uuid']}`")
                                    st.markdown(f"**Похожесть:** `{result['similarity']:.2f}%`")
                                    st.markdown(f"**Расстояние:** `{result['distance']:.4f}`")
                                    st.markdown(f"**Путь:** `{result['image_path']}`")
                                
                            except Exception as e:
                                st.error(f"❌ Ошибка загрузки изображения: {e}")
                                st.markdown(f"**{result['scientific_name']}** ({result['similarity']:.1f}%)")
                                st.markdown(f"*Файл не найден: {result['full_path']}*")
                    
                    # Сводная статистика
                    st.markdown("---")
                    st.markdown("### Сводная статистика")
                    
                    avg_similarity = sum(r['similarity'] for r in similar_images) / len(similar_images)
                    max_similarity = max(r['similarity'] for r in similar_images)
                    min_similarity = min(r['similarity'] for r in similar_images)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Средняя похожесть", f"{avg_similarity:.1f}%")
                    col2.metric("Максимальная похожесть", f"{max_similarity:.1f}%")
                    col3.metric("Минимальная похожесть", f"{min_similarity:.1f}%")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при поиске похожих изображений: {e}")
                    st.exception(e)

# Футер
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <small>Модель: EfficientNet V2 M | Обучена на датасете животных</small>
    </div>
    """,
    unsafe_allow_html=True
)