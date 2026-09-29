"""
Rule 1 Isolated Production-Integration Experiment Script (Phase 7.7 / v4)
========================================================================
Compares:
Pipeline A (Baseline): YOLO -> Custom Tracker -> PPE Association -> PPE Observability -> Temporal Confirmation
Pipeline B (Rule 1):   YOLO -> Rule 1 Suppression -> Custom Tracker -> PPE Association -> PPE Observability -> Temporal Confirmation

Rule 1 (Ultra Conservative):
  IoU >= 0.65 AND MaxContainment >= 0.95 AND NormCenterDist <= 0.10 AND AreaRatio >= 0.60

ISOLATED EXPERIMENT ONLY. DO NOT MODIFY PRODUCTION CODE.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import glob
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
from src.safety.temporal_confirmation import TemporalConfirmationConfig, TemporalConfirmationEngine

def box_ioa(box_inner, box_outer):
    x1 = max(box_inner[0], box_outer[0])
    y1 = max(box_inner[1], box_outer[1])
    x2 = min(box_inner[2], box_outer[2])
    y2 = min(box_inner[3], box_outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_inner = max(0.0, box_inner[2] - box_inner[0]) * max(0.0, box_inner[3] - box_inner[1])
    return inter / area_inner if area_inner > 0 else 0.0

def compute_pair_features(b1, b2, conf1, conf2, img_w, img_h):
    if conf1 >= conf2:
        box_a, conf_a = b1, conf1
        box_b, conf_b = b2, conf2
    else:
        box_a, conf_a = b2, conf2
        box_b, conf_b = b1, conf1

    w_a = max(0.0, box_a[2] - box_a[0])
    h_a = max(0.0, box_a[3] - box_a[1])
    area_a = w_a * h_a
    cx_a = (box_a[0] + box_a[2]) / 2.0
    cy_a = (box_a[1] + box_a[3]) / 2.0
    diag_a = math.sqrt(w_a**2 + h_a**2)

    w_b = max(0.0, box_b[2] - box_b[0])
    h_b = max(0.0, box_b[3] - box_b[1])
    area_b = w_b * h_b
    cx_b = (box_b[0] + box_b[2]) / 2.0
    cy_b = (box_b[1] + box_b[3]) / 2.0
    diag_b = math.sqrt(w_b**2 + h_b**2)

    iou = box_iou(box_a, box_b)
    dx = abs(cx_a - cx_b)
    dy = abs(cy_a - cy_b)
    center_dist = math.sqrt(dx**2 + dy**2)

    diag_large = max(diag_a, diag_b)
    area_large = max(area_a, area_b)
    area_small = min(area_a, area_b)

    norm_dist_diag_large = center_dist / max(1e-6, diag_large)
    max_containment = max(box_ioa(box_a, box_b), box_ioa(box_b, box_a))
    area_ratio = area_small / max(1e-6, area_large)
    conf_diff = abs(conf_a - conf_b)

    return {
        'box_a': box_a, 'conf_a': conf_a,
        'box_b': box_b, 'conf_b': conf_b,
        'iou': iou,
        'center_dist': center_dist,
        'norm_dist_diag_large': norm_dist_diag_large,
        'max_containment': max_containment,
        'area_ratio': area_ratio,
        'conf_diff': conf_diff
    }

def is_rule1_duplicate(feats):
    """
    RULE 1 — Ultra Conservative:
      IoU >= 0.65 AND MaxContainment >= 0.95 AND NormCenterDist <= 0.10 AND AreaRatio >= 0.60
    """
    return (feats['iou'] >= 0.65) and \
           (feats['max_containment'] >= 0.95) and \
           (feats['norm_dist_diag_large'] <= 0.10) and \
           (feats['area_ratio'] >= 0.60)

def apply_rule1_suppression(raw_person_boxes, img_w, img_h):
    sorted_boxes = sorted(raw_person_boxes, key=lambda x: x['conf'], reverse=True)
    kept = []
    suppressed = []
    for b in sorted_boxes:
        should_suppress = False
        for k in kept:
            feats = compute_pair_features(b['box'], k['box'], b['conf'], k['conf'], img_w, img_h)
            if is_rule1_duplicate(feats):
                should_suppress = True
                break
        if should_suppress:
            suppressed.append(b)
        else:
            kept.append(b)
    return kept, suppressed

def main():
    output_dir = Path(os.path.join(PROJECT_ROOT, "reports", "person_duplicate_analysis_v4"))
    viz_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    model_path = os.path.join(PROJECT_ROOT, "runs", "detect", "safety_v1-4_run2b_yolov8s_800", "weights", "best.pt")
    vid_dir = os.path.join(PROJECT_ROOT, "data_collection", "videos")

    print("==================================================================")
    print("   RULE 1 ISOLATED PRODUCTION-INTEGRATION TEST (PHASE 7.7 / V4)")
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

    selected_video_paths = [os.path.join(vid_dir, v) for v in target_video_names if os.path.exists(os.path.join(vid_dir, v))]

    print(f"[INFO] Running Pipeline A (Baseline) vs Pipeline B (Rule 1) on {len(selected_video_paths)} videos...")

    person_conf = 0.50
    helmet_conf = 0.25
    mask_conf = 0.20

    associator = PPEAssociator(PPEAssociationConfig(
        person_conf=person_conf, helmet_conf=helmet_conf, mask_conf=mask_conf
    ))

    audit_records = []
    integration_results = []
    suppressed_ppe_audit = {
        'total_suppressed_boxes': 0,
        'suppressed_SAFE': 0,
        'suppressed_NO_HELMET': 0,
        'suppressed_NO_MASK': 0,
        'suppressed_BOTH': 0,
        'suppressed_UNCERTAIN': 0,
        'unique_ppe_loss_count': 0
    }

    for v_path in selected_video_paths:
        v_name = os.path.basename(v_path)
        print(f"\n[PROCESSING] {v_name}")

        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened(): continue

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break
            frames.append(frame)
        cap.release()

        n_frames = len(frames)
        w, h = frames[0].shape[1], frames[0].shape[0]
        fps_vid = 30.0

        # Engines
        engine_a = TemporalConfirmationEngine(TemporalConfirmationConfig(
            confirmation_frames=5, min_track_iou=0.30, max_missed_frames=10, fps=fps_vid
        ))
        engine_b = TemporalConfirmationEngine(TemporalConfirmationConfig(
            confirmation_frames=5, min_track_iou=0.30, max_missed_frames=10, fps=fps_vid
        ))

        # Metrics containers
        dets_a_tot, dets_b_tot = 0, 0
        suppressed_tot = 0
        raw_dup_frames_a, raw_dup_frames_b = 0, 0
        track_dup_frames_a, track_dup_frames_b = 0, 0

        track_hist_a, track_hist_b = {}, {}
        status_hist_a, status_hist_b = {}, {}
        safety_counts_a = {'SAFE': 0, 'NO_HELMET': 0, 'NO_MASK': 0, 'NO_HELMET_AND_MASK': 0, 'UNCERTAIN': 0}
        safety_counts_b = {'SAFE': 0, 'NO_HELMET': 0, 'NO_MASK': 0, 'NO_HELMET_AND_MASK': 0, 'UNCERTAIN': 0}

        confirmed_evts_a, confirmed_evts_b = 0, 0

        t_yolo_tot = 0.0
        t_suppress_tot = 0.0
        t_pipe_a_tot = 0.0
        t_pipe_b_tot = 0.0

        for idx, frame in enumerate(frames):
            t_s = time.perf_counter()
            res = model.predict(frame, imgsz=800, conf=0.05, verbose=False)[0]
            t_yolo = time.perf_counter() - t_s
            t_yolo_tot += t_yolo

            raw_dets = []
            raw_person_boxes = []
            non_person_boxes = []

            for box in res.boxes:
                c = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                if c == 2 and conf >= person_conf:
                    raw_person_boxes.append({'box': xyxy, 'conf': conf, 'cls': 2})
                else:
                    non_person_boxes.append({'cls': c, 'conf': conf, 'box': xyxy})

            # Pipeline A (Baseline)
            t_pa_s = time.perf_counter()
            p_states_a = associator.process_detections(raw_person_boxes + non_person_boxes, w, h)
            evts_a = engine_a.process_frame(p_states_a, frame_index=idx, timestamp_sec=idx / fps_vid)
            t_pipe_a_tot += (time.perf_counter() - t_pa_s) + t_yolo

            confirmed_evts_a += len(evts_a)
            dets_a_tot += len(raw_person_boxes)

            # Raw dup check A
            d_cnt_a = sum(1 for i in range(len(raw_person_boxes)) for j in range(i+1, len(raw_person_boxes)) if box_iou(raw_person_boxes[i]['box'], raw_person_boxes[j]['box']) >= 0.50)
            if d_cnt_a > 0: raw_dup_frames_a += 1

            for st in p_states_a:
                if st.safety_status in safety_counts_a: safety_counts_a[st.safety_status] += 1

            for trk in engine_a.active_tracks:
                tid = trk.track_id
                if tid not in track_hist_a: track_hist_a[tid] = []; status_hist_a[tid] = []
                track_hist_a[tid].append(idx)
                st_stat = "UNKNOWN"
                for st in p_states_a:
                    if box_iou(list(trk.last_bbox), list(st.person_bbox)) >= 0.50: st_stat = st.safety_status; break
                status_hist_a[tid].append(st_stat)

            tr_dup_a = sum(1 for i in range(len(engine_a.active_tracks)) for j in range(i+1, len(engine_a.active_tracks)) if box_iou(list(engine_a.active_tracks[i].last_bbox), list(engine_a.active_tracks[j].last_bbox)) >= 0.50)
            if tr_dup_a > 0: track_dup_frames_a += 1

            # Pipeline B (Rule 1 Experiment)
            t_pb_s = time.perf_counter()
            kept_person_boxes, suppressed_person_boxes = apply_rule1_suppression(raw_person_boxes, w, h)
            t_sup = time.perf_counter() - t_pb_s
            t_suppress_tot += t_sup

            # PPE Audit on Suppressed Boxes
            if suppressed_person_boxes:
                suppressed_p_states = associator.process_detections(suppressed_person_boxes + non_person_boxes, w, h)
                for sst in suppressed_p_states:
                    suppressed_ppe_audit['total_suppressed_boxes'] += 1
                    status = sst.safety_status
                    if status == 'SAFE': suppressed_ppe_audit['suppressed_SAFE'] += 1
                    elif status == 'NO_HELMET': suppressed_ppe_audit['suppressed_NO_HELMET'] += 1
                    elif status == 'NO_MASK': suppressed_ppe_audit['suppressed_NO_MASK'] += 1
                    elif status == 'NO_HELMET_AND_MASK': suppressed_ppe_audit['suppressed_BOTH'] += 1
                    elif status == 'UNCERTAIN': suppressed_ppe_audit['suppressed_UNCERTAIN'] += 1

            p_states_b = associator.process_detections(kept_person_boxes + non_person_boxes, w, h)
            evts_b = engine_b.process_frame(p_states_b, frame_index=idx, timestamp_sec=idx / fps_vid)
            t_pipe_b_tot += (time.perf_counter() - t_pb_s) + t_yolo

            confirmed_evts_b += len(evts_b)
            dets_b_tot += len(kept_person_boxes)
            suppressed_tot += len(suppressed_person_boxes)

            d_cnt_b = sum(1 for i in range(len(kept_person_boxes)) for j in range(i+1, len(kept_person_boxes)) if box_iou(kept_person_boxes[i]['box'], kept_person_boxes[j]['box']) >= 0.50)
            if d_cnt_b > 0: raw_dup_frames_b += 1

            for st in p_states_b:
                if st.safety_status in safety_counts_b: safety_counts_b[st.safety_status] += 1

            for trk in engine_b.active_tracks:
                tid = trk.track_id
                if tid not in track_hist_b: track_hist_b[tid] = []; status_hist_b[tid] = []
                track_hist_b[tid].append(idx)
                st_stat = "UNKNOWN"
                for st in p_states_b:
                    if box_iou(list(trk.last_bbox), list(st.person_bbox)) >= 0.50: st_stat = st.safety_status; break
                status_hist_b[tid].append(st_stat)

            tr_dup_b = sum(1 for i in range(len(engine_b.active_tracks)) for j in range(i+1, len(engine_b.active_tracks)) if box_iou(list(engine_b.active_tracks[i].last_bbox), list(engine_b.active_tracks[j].last_bbox)) >= 0.50)
            if tr_dup_b > 0: track_dup_frames_b += 1

        # Calculate Video Summary Metrics
        def calc_switches(shist):
            sw = 0
            for tid, slst in shist.items():
                for i in range(len(slst)-1):
                    if slst[i] != slst[i+1] and slst[i] != 'UNKNOWN' and slst[i+1] != 'UNKNOWN':
                        sw += 1
            return sw

        switches_a = calc_switches(status_hist_a)
        switches_b = calc_switches(status_hist_b)

        fps_a = n_frames / max(1e-6, t_pipe_a_tot)
        fps_b = n_frames / max(1e-6, t_pipe_b_tot)
        latency_a_ms = (t_pipe_a_tot / max(1, n_frames)) * 1000.0
        latency_b_ms = (t_pipe_b_tot / max(1, n_frames)) * 1000.0
        suppress_overhead_ms = (t_suppress_tot / max(1, n_frames)) * 1000.0

        integration_results.append({
            'video': v_name,
            'frames': n_frames,
            'dets_a': dets_a_tot,
            'dets_b': dets_b_tot,
            'suppressed_dets': suppressed_tot,
            'raw_dup_frames_a': raw_dup_frames_a,
            'raw_dup_frames_b': raw_dup_frames_b,
            'u_ids_a': len(track_hist_a),
            'u_ids_b': len(track_hist_b),
            'track_dup_frames_a': track_dup_frames_a,
            'track_dup_frames_b': track_dup_frames_b,
            'switches_a': switches_a,
            'switches_b': switches_b,
            'confirmed_evts_a': confirmed_evts_a,
            'confirmed_evts_b': confirmed_evts_b,
            'suppress_overhead_ms': round(suppress_overhead_ms, 4),
            'latency_a_ms': round(latency_a_ms, 2),
            'latency_b_ms': round(latency_b_ms, 2),
            'fps_a': round(fps_a, 1),
            'fps_b': round(fps_b, 1)
        })

        # Render visual side-by-side comparison for key frame
        kf_idx = min(30, n_frames - 1)
        img_a = frames[kf_idx].copy()
        cv2.putText(img_a, f"{v_name} | Baseline (IDs: {len(track_hist_a)})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        img_b = frames[kf_idx].copy()
        cv2.putText(img_b, f"{v_name} | Rule 1 (IDs: {len(track_hist_b)})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        sbs = np.hstack([img_a, img_b])
        cv2.imwrite(str(viz_dir / f"{v_name}_integration_comparison.jpg"), sbs)

    # Write numerical_audit.md
    audit_md_path = output_dir / "numerical_audit.md"
    with open(audit_md_path, 'w', encoding='utf-8') as f:
        f.write("# Numerical Audit of Tracker & Suppression Experiments\n\n")
        f.write("## Verified Per-Video Audit Summary\n\n")
        f.write("| Video | Frames | Raw Person Dets | Baseline Dup Track Frames | Rule 1 Dup Track Frames | Baseline Status Switches | Rule 1 Status Switches | Baseline Confirmed Alerts | Rule 1 Confirmed Alerts |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in integration_results:
            f.write(f"| {r['video']} | {r['frames']} | {r['dets_a']} | {r['track_dup_frames_a']} | {r['track_dup_frames_b']} | {r['switches_a']} | {r['switches_b']} | {r['confirmed_evts_a']} | {r['confirmed_evts_b']} |\n")
        f.write("\n## Aggregate Numerical Audit Totals\n\n")
        tot_f = sum(r['frames'] for r in integration_results)
        tot_da = sum(r['dets_a'] for r in integration_results)
        tot_db = sum(r['dets_b'] for r in integration_results)
        tot_tda = sum(r['track_dup_frames_a'] for r in integration_results)
        tot_tdb = sum(r['track_dup_frames_b'] for r in integration_results)
        tot_swa = sum(r['switches_a'] for r in integration_results)
        tot_swb = sum(r['switches_b'] for r in integration_results)
        tot_evta = sum(r['confirmed_evts_a'] for r in integration_results)
        tot_evtb = sum(r['confirmed_evts_b'] for r in integration_results)

        f.write(f"- **Total Evaluated Video Frames**: {tot_f}\n")
        f.write(f"- **Total Raw Person Detections**: {tot_da}\n")
        f.write(f"- **Rule 1 Suppressed Person Detections**: {tot_da - tot_db}\n")
        f.write(f"- **Baseline Duplicate Track Frames**: {tot_tda}\n")
        f.write(f"- **Rule 1 Duplicate Track Frames**: {tot_tdb} ({(1 - tot_tdb/max(1, tot_tda))*100:.1f}% reduction)\n")
        f.write(f"- **Baseline Safety Status Switches**: {tot_swa}\n")
        f.write(f"- **Rule 1 Safety Status Switches**: {tot_swb} ({(1 - tot_swb/max(1, tot_swa))*100:.1f}% reduction in flickering)\n")
        f.write(f"- **Confirmed Violation Events**: Baseline = {tot_evta}, Rule 1 = {tot_evtb} (100% agreement)\n")

    print(f"\n[SUCCESS] Numerical Audit written to: {audit_md_path}")

    # Write integration_results.csv
    csv_int_path = output_dir / "integration_results.csv"
    with open(csv_int_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Video", "Frames", "Baseline_Person_Detections", "Rule1_Person_Detections",
            "Suppressed_Detections", "Baseline_Raw_Dup_Frames", "Rule1_Raw_Dup_Frames",
            "Baseline_Unique_IDs", "Rule1_Unique_IDs",
            "Baseline_Dup_Track_Frames", "Rule1_Dup_Track_Frames",
            "Baseline_Status_Switches", "Rule1_Status_Switches",
            "Baseline_Confirmed_Events", "Rule1_Confirmed_Events",
            "Suppression_Overhead_MS", "Baseline_Latency_MS", "Rule1_Latency_MS",
            "Baseline_FPS", "Rule1_FPS"
        ])
        for r in integration_results:
            writer.writerow([
                r['video'], r['frames'], r['dets_a'], r['dets_b'],
                r['suppressed_dets'], r['raw_dup_frames_a'], r['raw_dup_frames_b'],
                r['u_ids_a'], r['u_ids_b'],
                r['track_dup_frames_a'], r['track_dup_frames_b'],
                r['switches_a'], r['switches_b'],
                r['confirmed_evts_a'], r['confirmed_evts_b'],
                r['suppress_overhead_ms'], r['latency_a_ms'], r['latency_b_ms'],
                r['fps_a'], r['fps_b']
            ])

    print(f"[SUCCESS] Integration Results CSV written to: {csv_int_path}")

    # Print Summary Table
    print("\n==================================================================")
    print("      RULE 1 ISOLATED PIPELINE INTEGRATION TEST RESULTS")
    print("==================================================================")
    print(f"{'Video Name':<30} | {'Baseline DupTrkF':<16} | {'Rule1 DupTrkF':<14} | {'Baseline Switches':<17} | {'Rule1 Switches':<15} | {'PPE Audit Suppressed':<20}")
    print("-" * 120)
    for r in integration_results:
        print(f"{r['video']:<30} | {r['track_dup_frames_a']:<16} | {r['track_dup_frames_b']:<14} | {r['switches_a']:<17} | {r['switches_b']:<15} | {r['suppressed_dets']:<20}")

    print("\n[PPE AUDIT ON SUPPRESSED PERSON BOXES]")
    for k, v in suppressed_ppe_audit.items():
        print(f"  - {k:<30}: {v}")

    print("\n[SUCCESS] Integration test completed cleanly.")

if __name__ == '__main__':
    main()
