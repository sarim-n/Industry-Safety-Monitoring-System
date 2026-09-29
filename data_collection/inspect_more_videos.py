import os
import cv2
import json
import csv
from pathlib import Path

video_dir = Path("data_collection/more")
metadata_dir = video_dir / "metadata"
metadata_dir.mkdir(parents=True, exist_ok=True)

video_files = sorted(list(video_dir.glob("*.mp4")))

metadata_list = []

for v_path in video_files:
    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        print(f"Error opening {v_path}")
        continue
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0
    
    # Read first frame, middle frame, last frame to inspect content
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
    ret, frame = cap.read()
    cap.release()
    
    info = {
        "filename": v_path.name,
        "width": w,
        "height": h,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration, 2),
        "path": str(v_path)
    }
    metadata_list.append(info)
    print(f"Video: {v_path.name} | Resolution: {w}x{h} | FPS: {fps:.2f} | Total Frames: {total_frames} | Duration: {duration:.2f}s")

# Write metadata CSV and JSON
csv_path = metadata_dir / "video_metadata.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "width", "height", "fps", "total_frames", "duration_sec", "path"])
    writer.writeheader()
    writer.writerows(metadata_list)

json_path = metadata_dir / "video_metadata.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(metadata_list, f, indent=2)

print(f"\nSaved metadata to {csv_path} and {json_path}")
