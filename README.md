# Resistor color-band detector

## Перечень основных файлов:
- 'ML_3_train.ipynb' - блокнот с обучением моделей.
- 'inference.ipynb' - блокнот с локальным инференсом.
- 'streamlit_site.py' - файл web-интерфейса.

Итоговая модель построена на базе **YOLOv8n**.

Original             |  Detected
:-------------------------:|:-------------------------:
![Original](/readme_images/test_1_origin.jpg)  |  ![Detected](/readme_images/test_1_result.jpg)

Original             |  Detected
:-------------------------:|:-------------------------:
![Original](/readme_images/test_2_origin.jpg)  |  ![Detected](/readme_images/test_2_result.jpg)

## Авторы
- [Горбунов Роман](https://github.com/romangorbunov91), R4160
- [Иваненко Станислава](https://github.com/smthCreate), R4160
- [Волынец Глеб](https://github.com/glebvol12), R4160
- [Давыдов Игорь](https://github.com/TriglCr), R4197

## Исходный датасет
https://universe.roboflow.com/resistorv1/resistor-value-training/dataset/8#
## Reference
- [Explore Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/#performance-metrics)
- [Обучение YOLOv8 для задачи детекции](https://rutube.ru/video/20dcbc7489d0059da28b2053f6a4b8bd/)
- [Обучение YOLOv8 для задачи инстанс сегментации (YOLOv8-seg)](https://rutube.ru/video/ed03b1558d5dff48e9dc077938866bf7/)