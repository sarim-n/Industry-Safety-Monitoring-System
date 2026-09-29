"""
ByteTrack Experimental Evaluation Script (Phase 7.6 / Tracker Comparison v1)
=============================================================================
ISOLATED EXPERIMENT ONLY. DO NOT REPLACE PRODUCTION TRACKER. DO NOT MUTATE PRODUCTION CODE.

Compares:
A. Current Custom IoU Tracker (TemporalConfirmationEngine)
B. ByteTrack (Ultralytics BYTETracker)

Evaluates on video stream: data_collection/videos/4048038451-preview.mp4 (301 frames)
Model: runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import csv
import math
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.safety.ppe_association import PPEAssociationConfig, PPEAssociator, PersonPPEState, box_iou
from src.safety.temporal_confirmation import TemporalConfirmationConfig, TemporalConfirmationEngine, TemporaryWorkerTrack

def main():
    model_path = os.path.join(PROJECT_ROOT, "runs", "detect", "safety_v1-4_run2b_yolov8s_800", "weights", "best.pt")
    video_path = os.path.join(PROJECT_ROOT, "data_collection", "videos", "4048038451-preview.mp4")

    output_dir = Path(os.path.join(PROJECT_ROOT, "reports", "tracker_comparison_v1"))
    viz_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("   BYTETRACK EXPERIMENTAL EVALUATION (PHASE 7.6 / V1)")
    print("==================================================================")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    frame_h, frame_w = frames[0].shape[:2]
    fps_video = 30.0

    print(f"[INFO] Loaded {total_frames} video frames ({frame_w}x{frame_h} @ {fps_video} FPS)")

    person_conf = 0.50
    helmet_conf = 0.25
    mask_conf = 0.20

    associator = PPEAssociator(PPEAssociationConfig(
        person_conf=person_conf, helmet_conf=helmet_conf, mask_conf=mask_conf
    ))

    # =========================================================================
    # EXPERIMENT A: CURRENT CUSTOM TRACKER
    # =========================================================================
    print("\n[INFO] Running Tracker A: Current Custom IoU Tracker...")
    current_engine = TemporalConfirmationEngine(TemporalConfirmationConfig(
        confirmation_frames=5, min_track_iou=0.30, max_missed_frames=10, fps=fps_video
    ))

    tracker_a_records = []
    track_id_history_a = {} # track_id -> list of frame indices where active
    safety_history_a = {}   # track_id -> list of safety statuses

    t0_a = time.time()

    for idx, frame in enumerate(frames):
        res = model.predict(frame, imgsz=800, conf=0.05, verbose=False)[0]
        raw_dets = []
        person_boxes = []
        for box in res.boxes:
            c = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            raw_dets.append({'cls': c, 'conf': conf, 'box': xyxy})
            if c == 2 and conf >= person_conf:
                person_boxes.append(xyxy)

        person_states = associator.process_detections(raw_dets, frame_w, frame_h)
        events = current_engine.process_frame(person_states, frame_index=idx, timestamp_sec=idx / fps_video)

        active_tracks = current_engine.active_tracks

        raw_dup_count = 0
        for i in range(len(person_boxes)):
            for j in range(i + 1, len(person_boxes)):
                if box_iou(person_boxes[i], person_boxes[j]) >= 0.50:
                    raw_dup_count += 1

        track_dup_count = 0
        for i in range(len(active_tracks)):
            for j in range(i + 1, len(active_tracks)):
                if box_iou(list(active_tracks[i].last_bbox), list(active_tracks[j].last_bbox)) >= 0.50:
                    track_dup_count += 1

        active_ids = []
        for trk in active_tracks:
            tid = trk.track_id
            active_ids.append(tid)
            if tid not in track_id_history_a:
                track_id_history_a[tid] = []
                safety_history_a[tid] = []
            track_id_history_a[tid].append(idx)

            matched_status = "UNKNOWN"
            for st in person_states:
                if box_iou(list(trk.last_bbox), list(st.person_bbox)) >= 0.50:
                    matched_status = st.safety_status
                    break
            safety_history_a[tid].append(matched_status)

        tracker_a_records.append({
            'frame_index': idx,
            'person_det_count': len(person_boxes),
            'raw_dup_count': raw_dup_count,
            'active_track_count': len(active_tracks),
            'track_dup_count': track_dup_count,
            'active_track_ids': active_ids,
            'person_states': person_states
        })

    t_elapsed_a = time.time() - t0_a
    fps_a = total_frames / max(1e-6, t_elapsed_a)
    latency_a_ms = (t_elapsed_a / total_frames) * 1000.0

    # =========================================================================
    # EXPERIMENT B: BYTETRACK
    # =========================================================================
    print("\n[INFO] Running Tracker B: ByteTrack (Ultralytics BYTETracker)...")
    
    model.predictor = None # reset tracker persistence

    tracker_b_records = []
    track_id_history_b = {}
    safety_history_b = {}

    bytetrack_engine = TemporalConfirmationEngine(TemporalConfirmationConfig(
        confirmation_frames=5, min_track_iou=0.30, max_missed_frames=10, fps=fps_video
    ))

    t0_b = time.time()

    for idx, frame in enumerate(frames):
        results = model.track(frame, imgsz=800, conf=0.05, tracker="bytetrack.yaml", persist=True, verbose=False)[0]

        raw_dets = []
        person_boxes_bt = []
        bt_track_map = {}

        for box in results.boxes:
            c = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            raw_dets.append({'cls': c, 'conf': conf, 'box': xyxy})

            if c == 2 and conf >= person_conf:
                person_boxes_bt.append(xyxy)
                if box.id is not None:
                    bt_id = int(box.id[0].cpu().numpy())
                    bt_track_map[tuple(round(v, 1) for v in xyxy)] = bt_id

        person_states = associator.process_detections(raw_dets, frame_w, frame_h)

        for st in person_states:
            st_key = tuple(round(v, 1) for v in st.person_bbox)
            if st_key in bt_track_map:
                st.person_id = bt_track_map[st_key]

        events = bytetrack_engine.process_frame(person_states, frame_index=idx, timestamp_sec=idx / fps_video)

        for st in person_states:
            if hasattr(st, 'person_id') and st.person_id is not None:
                bt_id = st.person_id
                for trk in bytetrack_engine.active_tracks:
                    if box_iou(list(trk.last_bbox), list(st.person_bbox)) >= 0.50:
                        trk.track_id = bt_id
                        break

        active_tracks_b = bytetrack_engine.active_tracks

        raw_dup_count_b = 0
        for i in range(len(person_boxes_bt)):
            for j in range(i + 1, len(person_boxes_bt)):
                if box_iou(person_boxes_bt[i], person_boxes_bt[j]) >= 0.50:
                    raw_dup_count_b += 1

        track_dup_count_b = 0
        for i in range(len(active_tracks_b)):
            for j in range(i + 1, len(active_tracks_b)):
                if box_iou(list(active_tracks_b[i].last_bbox), list(active_tracks_b[j].last_bbox)) >= 0.50:
                    track_dup_count_b += 1

        active_ids_b = [t.track_id for t in active_tracks_b]
        for trk in active_tracks_b:
            tid = trk.track_id
            if tid not in track_id_history_b:
                track_id_history_b[tid] = []
                safety_history_b[tid] = []
            track_id_history_b[tid].append(idx)

            matched_status = "UNKNOWN"
            for st in person_states:
                if box_iou(list(trk.last_bbox), list(st.person_bbox)) >= 0.50:
                    matched_status = st.safety_status
                    break
            safety_history_b[tid].append(matched_status)

        tracker_b_records.append({
            'frame_index': idx,
            'person_det_count': len(person_boxes_bt),
            'raw_dup_count': raw_dup_count_b,
            'active_track_count': len(active_tracks_b),
            'track_dup_count': track_dup_count_b,
            'active_track_ids': active_ids_b,
            'person_states': person_states
        })

    t_elapsed_b = time.time() - t0_b
    fps_b = total_frames / max(1e-6, t_elapsed_b)
    latency_b_ms = (t_elapsed_b / total_frames) * 1000.0

    def compute_summary_metrics(records, track_id_history, safety_history, fps_val, latency_val):
        det_counts = [r['person_det_count'] for r in records]
        act_tracks = [r['active_track_count'] for r in records]
        dup_det_frames = sum(1 for r in records if r['raw_dup_count'] > 0)
        dup_track_frames = sum(1 for r in records if r['track_dup_count'] > 0)

        total_dets = sum(det_counts)
        unique_tids = list(track_id_history.keys())
        num_unique_tids = len(unique_tids)

        avg_act_tracks = np.mean(act_tracks)
        max_act_tracks = np.max(act_tracks)

        lifetimes = [len(frames_list) for frames_list in track_id_history.values()]
        avg_lifetime = float(np.mean(lifetimes)) if lifetimes else 0.0
        max_lifetime = int(np.max(lifetimes)) if lifetimes else 0

        fragmentations = 0
        recreations = 0
        for tid, f_list in track_id_history.items():
            f_sorted = sorted(f_list)
            gaps = [f_sorted[i+1] - f_sorted[i] for i in range(len(f_sorted)-1)]
            num_gaps = sum(1 for g in gaps if g > 1)
            fragmentations += num_gaps
            if num_gaps > 0:
                recreations += num_gaps

        safety_counts = {'SAFE': 0, 'NO_HELMET': 0, 'NO_MASK': 0, 'NO_HELMET_AND_MASK': 0, 'UNCERTAIN': 0}
        safety_switches = 0

        for r in records:
            for st in r['person_states']:
                s = st.safety_status
                if s in safety_counts:
                    safety_counts[s] += 1

        for tid, s_list in safety_history.items():
            for i in range(len(s_list) - 1):
                if s_list[i] != s_list[i+1] and s_list[i] != 'UNKNOWN' and s_list[i+1] != 'UNKNOWN':
                    safety_switches += 1

        return {
            'total_frames': len(records),
            'total_person_detections': total_dets,
            'unique_track_ids': num_unique_tids,
            'avg_active_tracks': round(float(avg_act_tracks), 2),
            'max_active_tracks': int(max_act_tracks),
            'frames_with_dup_person_boxes': dup_det_frames,
            'frames_with_dup_track_ids': dup_track_frames,
            'track_fragmentation': fragmentations,
            'track_recreations': recreations,
            'avg_track_lifetime': round(avg_lifetime, 1),
            'max_track_lifetime': max_lifetime,
            'fps': round(fps_val, 1),
            'latency_ms': round(latency_val, 2),
            'safety_safe': safety_counts['SAFE'],
            'safety_no_helmet': safety_counts['NO_HELMET'],
            'safety_no_mask': safety_counts['NO_MASK'],
            'safety_no_helmet_and_mask': safety_counts['NO_HELMET_AND_MASK'],
            'safety_uncertain': safety_counts['UNCERTAIN'],
            'safety_status_switches': safety_switches
        }

    sum_a = compute_summary_metrics(tracker_a_records, track_id_history_a, safety_history_a, fps_a, latency_a_ms)
    sum_b = compute_summary_metrics(tracker_b_records, track_id_history_b, safety_history_b, fps_b, latency_b_ms)

    csv_path = output_dir / "tracker_comparison.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Current_Custom_Tracker", "ByteTrack"])
        for k in sum_a.keys():
            writer.writerow([k, sum_a[k], sum_b[k]])

    print(f"\n[SUCCESS] Tracker Comparison CSV written to: {csv_path}")

    print("\n==================================================================")
    print("        TRACKER COMPARISON EXPERIMENT RESULTS SUMMARY")
    print("==================================================================")
    print(f"{'Metric':<32} | {'Current Custom Tracker':<22} | {'ByteTrack':<12}")
    print("-" * 72)
    for k in sum_a.keys():
        print(f"{k:<32} | {str(sum_a[k]):<22} | {str(sum_b[k]):<12}")

    print("\n[INFO] Generating representative comparison visualizations...")

    def render_annotated_frame(frame, records_at_idx, title_text):
        img = frame.copy()
        cv2.putText(img, title_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        for st in records_at_idx['person_states']:
            bx1, by1, bx2, by2 = [int(v) for v in st.person_bbox]
            pid = getattr(st, 'person_id', None)
            color = (0, 255, 0) if st.safety_status == 'SAFE' else ((0, 0, 255) if 'NO' in st.safety_status else (255, 255, 0))
            cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
            
            lbl = f"Worker {st.safety_status}"
            if pid is not None:
                lbl = f"ID #{pid} {st.safety_status}"
            cv2.putText(img, lbl, (bx1, max(20, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return img

    key_frames = [25, 75, 145, 220]
    for kf_idx in key_frames:
        if kf_idx < total_frames:
            img_a = render_annotated_frame(frames[kf_idx], tracker_a_records[kf_idx], f"Frame {kf_idx} | Current Custom Tracker")
            img_b = render_annotated_frame(frames[kf_idx], tracker_b_records[kf_idx], f"Frame {kf_idx} | ByteTrack")

            side_by_side = np.hstack([img_a, img_b])
            cv2.imwrite(str(viz_dir / f"frame_{kf_idx:04d}_tracker_comparison.jpg"), side_by_side)

    print(f"[SUCCESS] Visualizations saved in: {viz_dir}")

if __name__ == '__main__':
    main()
