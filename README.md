# Resistor color band detector (ML_3)

## Перечень основных файлов:
- 'ML_3_train.ipynb' - блокнот с обучением моделей.
- 'inference.ipynb' - блокнот с локальным инференсом.
- 'streamlit_site.py' - файл web-интерфейса.

Итоговая модель построена на базе **YOLOv8n**.

Original             |  Detected
:-------------------------:|:-------------------------:
![Original](readme_images\test_1_origin.jpg)  |  ![Detected](readme_images\test_1_result.jpg)

Original             |  Detected
:-------------------------:|:-------------------------:
![Original](readme_images\test_2_origin.jpg)  |  ![Detected](readme_images\test_2_result.jpg)

## Авторы
- [Горбунов Роман](https://github.com/romangorbunov91), R4160
- [Иваненко Станислава](https://github.com/smthCreate), R4160
- [Волынец Глеб](https://github.com/glebvol12), R4160
- [Давыдов Игорь](https://github.com/TriglCr), R4197

## Исходный датасет
https://universe.roboflow.com/resistorv1/resistor-value-training/dataset/8#
## Reference
https://docs.ultralytics.com/models/yolov8/#performance-metrics