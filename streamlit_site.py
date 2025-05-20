import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors

# Настройка кастомной палитры цветов
custom_palette = {
    0: (0, 0, 0),  # черный
    1: (255, 0, 0),  # синий
    2: (42, 42, 165),  # коричневый
    3: (55, 175, 212),  # золотой
    4: (0, 255, 0),  # зеленый
    5: (128, 128, 128),  # серый
    6: (0, 165, 255),  # оранжевый
    7: (0, 0, 255),  # красный
    8: (192, 192, 192),  # серебряный
    9: (230, 0, 255),  # фиолетовый
    10: (255, 255, 255),  # белый
    11: (0, 255, 255)  # желтый
}


@st.cache_resource
def load_model():
    colors = Colors()
    colors.palette = custom_palette
    model = YOLO('D:/git/ITMO/ML/ML_3/train_yolov8n/weights/best.pt')
    
    return model

# Заголовок приложения
st.title("Детекция объектов с YOLOv8")
st.markdown("Загрузите изображение для обнаружения объектов")

# Загрузка файла
uploaded_file = st.file_uploader(
    "Выберите изображение...",
    type=["jpg", "jpeg", "png"],
    help="Поддерживаются форматы JPG, JPEG, PNG. Максимальный размер 200MB"
)

if uploaded_file is not None:
    with st.spinner('Обрабатываем изображение...'):
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        if image_np.shape[-1] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        model = load_model()
        results = model(image_np)
        im_array = results[0].plot(line_width=2)
        result_image = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

        boxes = results[0].boxes
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_names = [model.names[id] for id in class_ids]

        results_df = {
            "Класс": class_names,
            "Уверенность": [f"{conf:.2f}" for conf in confidences]
        }

    st.success("✅ Обработка завершена!")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(
            result_image,
            caption="Результат детекции объектов",
            use_container_width=True
        )

    with col2:
        st.subheader("Обнаруженные объекты")
        for name, conf in zip(results_df["Класс"], results_df["Уверенность"]):
            st.markdown(f"- **{name.capitalize()}**: {conf}")

        st.download_button(
            label="Скачать результат",
            data=cv2.imencode('.jpg', im_array)[1].tobytes(),
            file_name="result_detection.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

else:
    st.info("👆 Пожалуйста, загрузите изображение в формате JPG, JPEG или PNG")