import os
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

video_dir = Path("data_collection/more")
model_path = Path("runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt")

model = YOLO(str(model_path))

video_files = sorted(list(video_dir.glob("*.mp4")))

scene_reports = {}

for v_idx, v_path in enumerate(video_files, 1):
    cap = cv2.VideoCapture(str(v_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sample 10 frames evenly
    sample_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    
    person_counts = []
    helmet_counts = []
    mask_counts = []
    
    for f_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        results = model.predict(frame, imgsz=800, conf=0.15, verbose=False)[0]
        p_cnt = 0
        h_cnt = 0
        m_cnt = 0
        for box in results.boxes:
            c = int(box.cls[0].cpu().numpy())
            if c == 2: p_cnt += 1
            elif c == 0: h_cnt += 1
            elif c == 1: m_cnt += 1
        person_counts.append(p_cnt)
        helmet_counts.append(h_cnt)
        mask_counts.append(m_cnt)
    
    cap.release()
    
    scene_reports[v_path.name] = {
        "v_idx": f"video{v_idx:02d}",
        "avg_persons": round(float(np.mean(person_counts)), 1),
        "avg_helmets": round(float(np.mean(helmet_counts)), 1),
        "avg_masks": round(float(np.mean(mask_counts)), 1),
        "total_frames": total_frames,
        "fps": round(fps, 2)
    }
    print(f"Video {v_idx} ({v_path.name}): Avg Persons={scene_reports[v_path.name]['avg_persons']}, Avg Helmets={scene_reports[v_path.name]['avg_helmets']}, Avg Masks={scene_reports[v_path.name]['avg_masks']}")

with open("data_collection/more/metadata/scene_analysis.json", "w") as f:
    json.dump(scene_reports, f, indent=2)
