!pip install ultralytics
!pip install ipywidgets

import zipfile, os

zip_path = "Car Scratch and Dent.v5i.yolov8.zip"  # замени на имя своего файла
extract_path = "/content/dataset"

# Распаковка
os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ Датасет распакован в:", extract_path)

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # лёгкая модель
model.train(data="/content/dataset/data.yaml", epochs=30, imgsz=640)

from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

trained_model = YOLO("runs/detect/train/weights/best.pt")

test_image_path = "test.jpg"  # загрузи своё фото в Jupyter и напиши имя файла
results = trained_model(test_image_path)

# Покажем картинку с предсказаниями
res_img = results[0].plot()
plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

