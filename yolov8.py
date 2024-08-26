from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.predict('apple.mp4',save=True,conf=0.4)

