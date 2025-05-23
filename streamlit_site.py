import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Настройка кастомной палитры цветов
custom_palette = {
    0:  [0, 0, 0],        # black
    1:  [255, 0, 0],      # blue
    2:  [42, 42, 165],    # brown
    3:  [55, 175, 212],   # gold
    4:  [0, 255, 0],      # green
    5:  [128, 128, 128],  # grey
    6:  [0, 165, 255],    # orange
    7:  [0, 0, 255],      # red
    8:  [192, 192, 192],  # silver
    9:  [230, 0, 255],    # violet
    10: [255, 255, 255],  # white
    11: [0, 255, 255]     # yellow
}

@st.cache_resource
def load_model():
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
        # 'image' in BGR.
        image = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        
        model = load_model()
        results = model(image, iou=0.2, conf=0.1)
        names = results[0].names          

        boxes = results[0].boxes.xyxy.cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()
        class_names = [model.names[id] for id in class_ids]
        
        custom_palette = {
            0:  [0, 0, 0],        # black
            1:  [255, 0, 0],      # blue
            2:  [42, 42, 165],    # brown
            3:  [55, 175, 212],   # gold
            4:  [0, 255, 0],      # green
            5:  [128, 128, 128],  # grey
            6:  [0, 165, 255],    # orange
            7:  [0, 0, 255],      # red
            8:  [192, 192, 192],  # silver
            9:  [230, 0, 255],    # violet
            10: [255, 255, 255],  # white
            11: [0, 255, 255]     # yellow
        }

        # Draw predictions manually
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            cls_id = class_ids[i]
            conf = confidences[i]
            label = f"{names[cls_id]} {conf:.2f}"

            # Get color from custom color map
            color = custom_palette.get(cls_id)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

            # Draw label background + text
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Background rectangle for label
            cv2.rectangle(image, (x1, y1 - 25), (x1 + text_width, y1), color, -1)

            # White text on top
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (200, 200, 200), thickness) 
      
        results_df = {
            "Класс": class_names,
            "Уверенность": [f"{conf:.2f}" for conf in confidences]
        }
        #im_array = results[0].plot(line_width=2)
        result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
            data=cv2.imencode('.jpg', image)[1].tobytes(),
            file_name="result_detection.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

else:
    st.info("👆 Пожалуйста, загрузите изображение в формате JPG, JPEG или PNG")