import os
import cv2
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(PROJECT_ROOT, "runs", "detect", "safety_v1-4_run2b_yolov8s_800", "weights", "best.pt")
video_path = os.path.join(PROJECT_ROOT, "data_collection", "videos", "4048038451-preview.mp4")

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    results = model.track(frame, imgsz=800, conf=0.50, tracker="bytetrack.yaml", persist=True, verbose=False)[0]
    print("ByteTrack tracking results:", results.boxes)
    if results.boxes.id is not None:
        print("Track IDs:", results.boxes.id.cpu().numpy())
    else:
        print("No Track IDs assigned")
