import os
import glob
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vid_dir = os.path.join(PROJECT_ROOT, "data_collection", "videos")
videos = sorted(glob.glob(os.path.join(vid_dir, "*.mp4")))

print(f"Found {len(videos)} videos in {vid_dir}:")
for v in videos:
    cap = cv2.VideoCapture(v)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"  {os.path.basename(v):<35} | {frames:<5} frames | {fps:<4.1f} FPS | {w}x{h}")
