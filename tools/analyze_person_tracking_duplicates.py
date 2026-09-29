"""
Live Video & Tracking Person Duplicate Diagnostic Script (Phase 7.6)
=====================================================================
Analyzes frame-by-frame tracking and rendering layer for duplicate person boxes
on live video streams and webcam sequences.
THIS SCRIPT IS DIAGNOSIS ONLY. DO NOT ALTER CODE OR CONFIG.
"""

import os
import sys
import cv2
import csv
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.safety.ppe_association import PPEAssociationConfig, PPEAssociator, box_iou
from src.safety.temporal_confirmation import TemporalConfirmationConfig, TemporalConfirmationEngine

def main():
    model_path = os.path.join(PROJECT_ROOT, "runs", "detect", "safety_v1-4_run2b_yolov8s_800", "weights", "best.pt")
    video_path = os.path.join(PROJECT_ROOT, "data_collection", "videos", "4048038451-preview.mp4")

    output_dir = Path(os.path.join(PROJECT_ROOT, "reports", "person_duplicate_analysis_v1"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("   LIVE TRACKING PERSON DUPLICATE DIAGNOSIS (PHASE 7.6)")
    print("==================================================================")
    print(f"  Model Path  : {model_path}")
    print(f"  Video Path  : {video_path}")

    model = YOLO(model_path)
    associator = PPEAssociator(PPEAssociationConfig(person_conf=0.50, helmet_conf=0.25, mask_conf=0.20))
    temp_engine = TemporalConfirmationEngine(TemporalConfirmationConfig(
        confirmation_frames=5,
        min_track_iou=0.30,
        max_missed_frames=10,
        fps=30.0,
        alert_cooldown_seconds=5.0,
        uncertain_breaks_streak=True
    ))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_index = 0
    tracking_records = []
    suspicious_frames = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_index += 1

        # STAGE 1: Raw YOLO
        results = model.predict(frame, imgsz=800, conf=0.05, verbose=False)[0]
        raw_dets = []
        for box in results.boxes:
            c = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            raw_dets.append({'cls': c, 'conf': conf, 'box': xyxy})

        # Count raw YOLO person boxes >= 0.50
        yolo_person_boxes = [d['box'] for d in raw_dets if d['cls'] == 2 and d['conf'] >= 0.50]

        # Check raw YOLO duplicate person boxes in this frame
        yolo_dup_count = 0
        for i in range(len(yolo_person_boxes)):
            for j in range(i + 1, len(yolo_person_boxes)):
                if box_iou(yolo_person_boxes[i], yolo_person_boxes[j]) >= 0.50:
                    yolo_dup_count += 1

        # STAGE 2: PPE Association
        person_states = associator.process_detections(raw_dets, frame_w, frame_h)

        # STAGE 3: Temporal Tracking
        events = temp_engine.process_frame(person_states, frame_index=frame_index - 1, timestamp_sec=(frame_index - 1) / fps)

        active_tracks = temp_engine.active_tracks

        # Check active track IoU overlaps in this frame
        track_dup_count = 0
        for i in range(len(active_tracks)):
            for j in range(i + 1, len(active_tracks)):
                if box_iou(list(active_tracks[i].last_bbox), list(active_tracks[j].last_bbox)) >= 0.50:
                    track_dup_count += 1

        record = {
            'frame_index': frame_index,
            'yolo_person_count': len(yolo_person_boxes),
            'yolo_dup_count': yolo_dup_count,
            'state_count': len(person_states),
            'active_track_count': len(active_tracks),
            'track_dup_count': track_dup_count,
            'active_track_ids': [t.track_id for t in active_tracks]
        }
        tracking_records.append(record)

        if yolo_dup_count > 0 or track_dup_count > 0 or len(active_tracks) > len(yolo_person_boxes):
            suspicious_frames.append(record)

            # Save diagnostic frame image
            annotated = frame.copy()
            for t in active_tracks:
                bx1, by1, bx2, by2 = [int(v) for v in t.last_bbox]
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 0, 255), 2)
                cv2.putText(annotated, f"Track #{t.track_id} (mf={t.missed_frames})", (bx1, max(20, by1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            save_p = output_dir / f"frame_{frame_index:04d}_tracking_dup.jpg"
            cv2.imwrite(str(save_p), annotated)

    cap.release()

    total_frames = len(tracking_records)
    frames_with_yolo_dups = sum(1 for r in tracking_records if r['yolo_dup_count'] > 0)
    frames_with_track_dups = sum(1 for r in tracking_records if r['track_dup_count'] > 0)
    frames_with_extra_tracks = sum(1 for r in tracking_records if r['active_track_count'] > r['yolo_person_count'])

    print("\n==================================================================")
    print("   LIVE TRACKING DUPLICATE ANALYSIS SUMMARY")
    print("==================================================================")
    print(f"Total Video Frames Processed : {total_frames}")
    print(f"Frames with Raw YOLO Duplicates (IoU >= 0.50): {frames_with_yolo_dups} ({frames_with_yolo_dups/max(1, total_frames)*100:.1f}%)")
    print(f"Frames with Track IoU Overlaps (IoU >= 0.50)  : {frames_with_track_dups} ({frames_with_track_dups/max(1, total_frames)*100:.1f}%)")
    print(f"Frames with Extra Active Tracks (missed_frames): {frames_with_extra_tracks} ({frames_with_extra_tracks/max(1, total_frames)*100:.1f}%)")

    if suspicious_frames:
        print("\n[SUSPICIOUS FRAMES LOG (First 10)]")
        for s in suspicious_frames[:10]:
            print(f"  Frame {s['frame_index']:4d} | YOLO Persons: {s['yolo_person_count']} | Active Tracks: {s['active_track_count']} (IDs: {s['active_track_ids']}) | YOLO Dups: {s['yolo_dup_count']} | Track Dups: {s['track_dup_count']}")

    # Write CSV
    csv_track_path = output_dir / "tracking_duplicate_analysis.csv"
    with open(csv_track_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['frame_index', 'yolo_person_count', 'yolo_dup_count', 'state_count', 'active_track_count', 'track_dup_count', 'active_track_ids'])
        writer.writeheader()
        writer.writerows(tracking_records)

    print(f"\n[TRACKING LOG SAVED] {csv_track_path}")

if __name__ == '__main__':
    main()
