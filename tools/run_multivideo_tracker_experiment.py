"""
Multi-Video Tracker Comparison Experiment Script (Phase 7.6 / Tracker Comparison v2)
======================================================================================
Evaluates Current Custom Tracker vs Ultralytics ByteTrack across 6 representative videos:
1. 4048038451-preview.mp4
2. 8482302-hd_1920_1080_25fps.mp4
3. 4017518657-preview.mp4
4. 19832490-hd_1920_1080_25fps.mp4
5. no safety.mp4
6. helmet+mask+gloves.mp4

ISOLATED EXPERIMENT ONLY. DO NOT REPLACE PRODUCTION TRACKER. DO NOT MUTATE PRODUCTION CODE.
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

from src.safety.ppe_association import PPEAssociationConfig, PPEAssociator, box_iou
from src.safety.temporal_confirmation import TemporalConfirmationConfig, TemporalConfirmationEngine

def main():
    model_path = os.path.join(PROJECT_ROOT, "runs", "detect", "safety_v1-4_run2b_yolov8s_800", "weights", "best.pt")
    vid_dir = os.path.join(PROJECT_ROOT, "data_collection", "videos")

    output_dir = Path(os.path.join(PROJECT_ROOT, "reports", "tracker_comparison_v2"))
    viz_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("   MULTI-VIDEO TRACKER COMPARISON EXPERIMENT (PHASE 7.6 / V2)")
    print("==================================================================")

    model = YOLO(model_path)

    target_video_names = [
        "4048038451-preview.mp4",
        "8482302-hd_1920_1080_25fps.mp4",
        "4017518657-preview.mp4",
        "19832490-hd_1920_1080_25fps.mp4",
        "no safety.mp4",
        "helmet+mask+gloves.mp4"
    ]

    selected_video_paths = []
    for name in target_video_names:
        p = os.path.join(vid_dir, name)
        if os.path.exists(p):
            selected_video_paths.append(p)
        else:
            print(f"[WARN] Video not found: {p}")

    print(f"[INFO] Selected {len(selected_video_paths)} videos for evaluation:")
    for v_path in selected_video_paths:
        print(f"  - {os.path.basename(v_path)}")

    person_conf = 0.50
    helmet_conf = 0.25
    mask_conf = 0.20

    associator = PPEAssociator(PPEAssociationConfig(
        person_conf=person_conf, helmet_conf=helmet_conf, mask_conf=mask_conf
    ))

    per_video_results = []

    for v_path in selected_video_paths:
        v_name = os.path.basename(v_path)
        print(f"\n==================================================================")
        print(f" PROCESSING VIDEO: {v_name}")
        print(f"==================================================================")

        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open {v_path}")
            continue

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
        cap.release()

        n_frames = len(frames)
        frame_h, frame_w = frames[0].shape[:2]
        fps_vid = 30.0
        print(f"[INFO] Read {n_frames} frames ({frame_w}x{frame_h})")

        # ---------------------------------------------------------------------
        # RUN TRACKER A: CURRENT CUSTOM TRACKER
        # ---------------------------------------------------------------------
        print(f"[INFO] Evaluating Custom Tracker on {v_name}...")
        engine_a = TemporalConfirmationEngine(TemporalConfirmationConfig(
            confirmation_frames=5, min_track_iou=0.30, max_missed_frames=10, fps=fps_vid
        ))

        t_yolo_a, t_track_a, t_ppe_a, t_temp_a = 0.0, 0.0, 0.0, 0.0
        records_a = []
        track_hist_a = {}
        status_hist_a = {}
        confirmed_events_a = 0

        t0_a = time.perf_counter()

        for idx, frame in enumerate(frames):
            # 1. YOLO Inference
            t_s = time.perf_counter()
            res = model.predict(frame, imgsz=800, conf=0.05, verbose=False)[0]
            t_yolo_a += (time.perf_counter() - t_s)

            raw_dets = []
            person_boxes = []
            for box in res.boxes:
                c = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                raw_dets.append({'cls': c, 'conf': conf, 'box': xyxy})
                if c == 2 and conf >= person_conf:
                    person_boxes.append(xyxy)

            # 2. PPE Association
            t_s = time.perf_counter()
            person_states = associator.process_detections(raw_dets, frame_w, frame_h)
            t_ppe_a += (time.perf_counter() - t_s)

            # 3. Temporal Tracking & Confirmation
            t_s = time.perf_counter()
            events = engine_a.process_frame(person_states, frame_index=idx, timestamp_sec=idx / fps_vid)
            t_temp_a += (time.perf_counter() - t_s)
            confirmed_events_a += len(events)

            active_tracks = engine_a.active_tracks

            # Raw duplicate person detections
            raw_dup_pairs = 0
            for i in range(len(person_boxes)):
                for j in range(i + 1, len(person_boxes)):
                    if box_iou(person_boxes[i], person_boxes[j]) >= 0.50:
                        raw_dup_pairs += 1

            # Duplicate active tracks
            track_dup_pairs = 0
            for i in range(len(active_tracks)):
                for j in range(i + 1, len(active_tracks)):
                    if box_iou(list(active_tracks[i].last_bbox), list(active_tracks[j].last_bbox)) >= 0.50:
                        track_dup_pairs += 1

            act_ids = []
            for trk in active_tracks:
                tid = trk.track_id
                act_ids.append(tid)
                if tid not in track_hist_a:
                    track_hist_a[tid] = []
                    status_hist_a[tid] = []
                track_hist_a[tid].append(idx)

                st_status = "UNKNOWN"
                for st in person_states:
                    if box_iou(list(trk.last_bbox), list(st.person_bbox)) >= 0.50:
                        st_status = st.safety_status
                        break
                status_hist_a[tid].append(st_status)

            records_a.append({
                'p_dets': len(person_boxes),
                'raw_dup_pairs': raw_dup_pairs,
                'act_tracks': len(active_tracks),
                'track_dup_pairs': track_dup_pairs,
                'act_ids': act_ids
            })

        t_total_a = time.perf_counter() - t0_a
        fps_a = n_frames / max(1e-6, t_total_a)
        latency_a_ms = (t_total_a / max(1, n_frames)) * 1000.0

        # ---------------------------------------------------------------------
        # RUN TRACKER B: ULTRALYTICS BYTETRACK
        # ---------------------------------------------------------------------
        print(f"[INFO] Evaluating ByteTrack on {v_name}...")
        model.predictor = None # Reset tracker state

        engine_b = TemporalConfirmationEngine(TemporalConfirmationConfig(
            confirmation_frames=5, min_track_iou=0.30, max_missed_frames=10, fps=fps_vid
        ))

        t_yolo_b, t_track_b, t_ppe_b, t_temp_b = 0.0, 0.0, 0.0, 0.0
        records_b = []
        track_hist_b = {}
        status_hist_b = {}
        confirmed_events_b = 0

        t0_b = time.perf_counter()

        for idx, frame in enumerate(frames):
            # 1. YOLO + ByteTrack
            t_s = time.perf_counter()
            results = model.track(frame, imgsz=800, conf=0.05, tracker="bytetrack.yaml", persist=True, verbose=False)[0]
            t_yolo_b += (time.perf_counter() - t_s)

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

            # 2. PPE Association
            t_s = time.perf_counter()
            person_states = associator.process_detections(raw_dets, frame_w, frame_h)
            t_ppe_b += (time.perf_counter() - t_s)

            for st in person_states:
                st_key = tuple(round(v, 1) for v in st.person_bbox)
                if st_key in bt_track_map:
                    st.person_id = bt_track_map[st_key]

            # 3. Temporal Confirmation
            t_s = time.perf_counter()
            events = engine_b.process_frame(person_states, frame_index=idx, timestamp_sec=idx / fps_vid)
            t_temp_b += (time.perf_counter() - t_s)
            confirmed_events_b += len(events)

            for st in person_states:
                if hasattr(st, 'person_id') and st.person_id is not None:
                    bt_id = st.person_id
                    for trk in engine_b.active_tracks:
                        if box_iou(list(trk.last_bbox), list(st.person_bbox)) >= 0.50:
                            trk.track_id = bt_id
                            break

            active_tracks_b = engine_b.active_tracks

            raw_dup_pairs_b = 0
            for i in range(len(person_boxes_bt)):
                for j in range(i + 1, len(person_boxes_bt)):
                    if box_iou(person_boxes_bt[i], person_boxes_bt[j]) >= 0.50:
                        raw_dup_pairs_b += 1

            track_dup_pairs_b = 0
            for i in range(len(active_tracks_b)):
                for j in range(i + 1, len(active_tracks_b)):
                    if box_iou(list(active_tracks_b[i].last_bbox), list(active_tracks_b[j].last_bbox)) >= 0.50:
                        track_dup_pairs_b += 1

            act_ids_b = [t.track_id for t in active_tracks_b]
            for trk in active_tracks_b:
                tid = trk.track_id
                if tid not in track_hist_b:
                    track_hist_b[tid] = []
                    status_hist_b[tid] = []
                track_hist_b[tid].append(idx)

                st_status = "UNKNOWN"
                for st in person_states:
                    if box_iou(list(trk.last_bbox), list(st.person_bbox)) >= 0.50:
                        st_status = st.safety_status
                        break
                status_hist_b[tid].append(st_status)

            records_b.append({
                'p_dets': len(person_boxes_bt),
                'raw_dup_pairs': raw_dup_pairs_b,
                'act_tracks': len(active_tracks_b),
                'track_dup_pairs': track_dup_pairs_b,
                'act_ids': act_ids_b
            })

        t_total_b = time.perf_counter() - t0_b
        fps_b = n_frames / max(1e-6, t_total_b)
        latency_b_ms = (t_total_b / max(1, n_frames)) * 1000.0

        # Calculate metrics for video
        def calc_video_metrics(recs, track_hist, status_hist, total_t, confirmed_evts):
            total_p_dets = sum(r['p_dets'] for r in recs)
            dup_p_frames = sum(1 for r in recs if r['raw_dup_pairs'] > 0)
            dup_track_frames = sum(1 for r in recs if r['track_dup_pairs'] > 0)
            dup_p_pairs_tot = sum(r['raw_dup_pairs'] for r in recs)
            dup_track_pairs_tot = sum(r['track_dup_pairs'] for r in recs)

            u_ids = len(track_hist)
            act_trks = [r['act_tracks'] for r in recs]
            avg_act = float(np.mean(act_trks)) if act_trks else 0.0
            max_act = int(np.max(act_trks)) if act_trks else 0

            ltimes = [len(lst) for lst in track_hist.values()]
            avg_ltime = float(np.mean(ltimes)) if ltimes else 0.0
            max_ltime = int(np.max(ltimes)) if ltimes else 0

            frags = 0
            for tid, flst in track_hist.items():
                fs = sorted(flst)
                gaps = [fs[i+1] - fs[i] for i in range(len(fs)-1)]
                frags += sum(1 for g in gaps if g > 1)

            switches = 0
            safe_to_viol = 0
            viol_to_safe = 0
            uncert_trans = 0

            viol_set = {"NO_HELMET", "NO_MASK", "NO_HELMET_AND_MASK"}

            for tid, slst in status_hist.items():
                for i in range(len(slst) - 1):
                    s1, s2 = slst[i], slst[i+1]
                    if s1 != s2 and s1 != 'UNKNOWN' and s2 != 'UNKNOWN':
                        switches += 1
                        if s1 == 'SAFE' and s2 in viol_set:
                            safe_to_viol += 1
                        elif s1 in viol_set and s2 == 'SAFE':
                            viol_to_safe += 1
                        if s1 == 'UNCERTAIN' or s2 == 'UNCERTAIN':
                            uncert_trans += 1

            return {
                'total_p_dets': total_p_dets,
                'dup_p_frames': dup_p_frames,
                'dup_p_pairs_tot': dup_p_pairs_tot,
                'u_ids': u_ids,
                'avg_act': round(avg_act, 2),
                'max_act': max_act,
                'avg_ltime': round(avg_ltime, 1),
                'max_ltime': max_ltime,
                'frags': frags,
                'dup_track_frames': dup_track_frames,
                'dup_track_pairs_tot': dup_track_pairs_tot,
                'switches': switches,
                'safe_to_viol': safe_to_viol,
                'viol_to_safe': viol_to_safe,
                'uncert_trans': uncert_trans,
                'confirmed_evts': confirmed_evts,
                'total_time_sec': round(total_t, 3),
                'latency_ms': round((total_t / max(1, len(recs))) * 1000.0, 2),
                'fps': round(len(recs) / max(1e-6, total_t), 1)
            }

        m_a = calc_video_metrics(records_a, track_hist_a, status_hist_a, t_total_a, confirmed_events_a)
        m_b = calc_video_metrics(records_b, track_hist_b, status_hist_b, t_total_b, confirmed_events_b)

        per_video_results.append({
            'video': v_name,
            'frames': n_frames,
            'custom': m_a,
            'bytetrack': m_b,
            'timing_a': {'yolo': t_yolo_a, 'track': t_track_a, 'ppe': t_ppe_a, 'temp': t_temp_a},
            'timing_b': {'yolo': t_yolo_b, 'track': t_track_b, 'ppe': t_ppe_b, 'temp': t_temp_b}
        })

        # Save visual comparison frames for this video
        key_idx = min(30, n_frames - 1)
        img_a = frames[key_idx].copy()
        cv2.putText(img_a, f"{v_name} | Custom Tracker (IDs: {records_a[key_idx]['act_ids']})", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        img_b = frames[key_idx].copy()
        cv2.putText(img_b, f"{v_name} | ByteTrack (IDs: {records_b[key_idx]['act_ids']})", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        sbs = np.hstack([img_a, img_b])
        cv2.imwrite(str(viz_dir / f"{v_name}_comparison.jpg"), sbs)

    # -------------------------------------------------------------------------
    # AGGREGATE RESULTS & CSV GENERATION
    # -------------------------------------------------------------------------
    csv_path = output_dir / "multivideo_tracker_comparison.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "video", "frames",
            "custom_detections", "bytetrack_detections",
            "custom_unique_ids", "bytetrack_unique_ids",
            "custom_avg_active_tracks", "bytetrack_avg_active_tracks",
            "custom_max_active_tracks", "bytetrack_max_active_tracks",
            "custom_duplicate_person_frames", "bytetrack_duplicate_person_frames",
            "custom_duplicate_track_frames", "bytetrack_duplicate_track_frames",
            "custom_fragmentations", "bytetrack_fragmentations",
            "custom_avg_track_lifetime", "bytetrack_avg_track_lifetime",
            "custom_status_switches", "bytetrack_status_switches",
            "custom_confirmed_events", "bytetrack_confirmed_events",
            "custom_avg_latency_ms", "bytetrack_avg_latency_ms",
            "custom_fps", "bytetrack_fps"
        ])

        for res in per_video_results:
            ca, cb = res['custom'], res['bytetrack']
            writer.writerow([
                res['video'], res['frames'],
                ca['total_p_dets'], cb['total_p_dets'],
                ca['u_ids'], cb['u_ids'],
                ca['avg_act'], cb['avg_act'],
                ca['max_act'], cb['max_act'],
                ca['dup_p_frames'], cb['dup_p_frames'],
                ca['dup_track_frames'], cb['dup_track_frames'],
                ca['frags'], cb['frags'],
                ca['avg_ltime'], cb['avg_ltime'],
                ca['switches'], cb['switches'],
                ca['confirmed_evts'], cb['confirmed_evts'],
                ca['latency_ms'], cb['latency_ms'],
                ca['fps'], cb['fps']
            ])

    print(f"\n[SUCCESS] Multi-video CSV written to: {csv_path}")

    # Print Summary Table
    print("\n==================================================================")
    print("      MULTI-VIDEO TRACKER COMPARISON RESULTS SUMMARY")
    print("==================================================================")
    print(f"{'Video Name':<32} | {'Frames':<6} | {'Custom IDs':<10} | {'ByteTrk IDs':<11} | {'Custom DupTrkF':<14} | {'ByteTrk DupTrkF':<15} | {'Custom FPS':<10} | {'ByteTrk FPS':<11}")
    print("-" * 125)
    for res in per_video_results:
        ca, cb = res['custom'], res['bytetrack']
        print(f"{res['video']:<32} | {res['frames']:<6} | {ca['u_ids']:<10} | {cb['u_ids']:<11} | {ca['dup_track_frames']:<14} | {cb['dup_track_frames']:<15} | {ca['fps']:<10.1f} | {cb['fps']:<11.1f}")

    # Totals across all videos
    tot_frames = sum(r['frames'] for r in per_video_results)
    tot_custom_ids = sum(r['custom']['u_ids'] for r in per_video_results)
    tot_bt_ids = sum(r['bytetrack']['u_ids'] for r in per_video_results)
    tot_custom_dup_tf = sum(r['custom']['dup_track_frames'] for r in per_video_results)
    tot_bt_dup_tf = sum(r['bytetrack']['dup_track_frames'] for r in per_video_results)
    tot_custom_switches = sum(r['custom']['switches'] for r in per_video_results)
    tot_bt_switches = sum(r['bytetrack']['switches'] for r in per_video_results)

    avg_custom_fps = np.mean([r['custom']['fps'] for r in per_video_results])
    avg_bt_fps = np.mean([r['bytetrack']['fps'] for r in per_video_results])

    print("-" * 125)
    print(f"{'TOTAL / OVERALL AGGREGATE':<32} | {tot_frames:<6} | {tot_custom_ids:<10} | {tot_bt_ids:<11} | {tot_custom_dup_tf:<14} | {tot_bt_dup_tf:<15} | {avg_custom_fps:<10.1f} | {avg_bt_fps:<11.1f}")
    print(f"Total Status Switches Across All Videos: Custom = {tot_custom_switches}, ByteTrack = {tot_bt_switches} ({round((1 - tot_bt_switches/max(1, tot_custom_switches))*100, 1)}% reduction)")

    print(f"\n[SUCCESS] Visualizations saved in: {viz_dir}")

if __name__ == '__main__':
    main()
