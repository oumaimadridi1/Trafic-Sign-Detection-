import sys
from ultralytics import YOLO
YOLO(sys.argv[1]).predict(sys.argv[2], conf=0.4, imgsz=960, save=True, name="preview", project="runs")
print("Open runs/preview to see the images with boxes.")