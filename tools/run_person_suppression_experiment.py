"""
Person Duplicate Suppression Experiment Script
================================================
Evaluates person-class-only duplicate suppression thresholds (0.40, 0.45, 0.50, 0.55, 0.60)
against Baseline (no suppression) on:
1. Validation dataset (136 images, 386 GT persons)
2. Live video sequence (data_collection/videos/4048038451-preview.mp4)

THIS IS AN EXPERIMENT ONLY. DO NOT MODIFY PRODUCTION CODE.
"""

import os
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

from src.safety.ppe_association import PPEAssociationConfig, PPEAssociator, box_iou
from src.safety.temporal_confirmation import TemporalConfirmationConfig, TemporalConfirmationEngine

def xywh2xyxy(box, w, h):
    cx, cy, bw, bh = box
    x1 = max(0.0, (cx - bw / 2.0) * w)
    y1 = max(0.0, (cy - bh / 2.0) * h)
    x2 = min(float(w), (cx + bw / 2.0) * w)
    y2 = min(float(h), (cy + bh / 2.0) * h)
    return [x1, y1, x2, y2]

def suppress_person_duplicates(person_boxes, iou_threshold):
    """
    Greedy NMS for person boxes only.
    person_boxes: list of dicts {'box': [x1, y1, x2, y2], 'conf': float}
    Returns kept person boxes.
    """
    sorted_boxes = sorted(person_boxes, key=lambda x: x['conf'], reverse=True)
    kept = []
    for b in sorted_boxes:
        should_suppress = False
        for k in kept:
            if box_iou(b['box'], k['box']) >= iou_threshold:
                should_suppress = True
                break
        if not should_suppress:
            kept.append(b)
    return kept

def evaluate_val_set(model, val_img_dir, val_lbl_dir, iou_thresholds, person_conf=0.50):
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.*")))
    
    # Pre-run YOLO on all val images to cache predictions for baseline
    # to guarantee identical underlying YOLO predictions across threshold evaluations
    cached_predictions = []
    
    print(f"[INFO] Caching YOLO predictions for {len(val_images)} validation images...")
    for img_path in val_images:
        img_name = os.path.basename(img_path)
        lbl_path = os.path.join(val_lbl_dir, os.path.splitext(img_name)[0] + ".txt")
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        gt_persons = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5 and int(parts[0]) == 2:
                        gt_persons.append(xywh2xyxy([float(x) for x in parts[1:]], w, h))
                        
        results = model.predict(img, imgsz=800, conf=person_conf, verbose=False)[0]
        raw_person_boxes = []
        for box in results.boxes:
            c = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            if c == 2 and conf >= person_conf:
                raw_person_boxes.append({'box': xyxy, 'conf': conf, 'img_name': img_name})
                
        cached_predictions.append({
            'img_path': img_path,
            'img_name': img_name,
            'img_w': w,
            'img_h': h,
            'gt_persons': gt_persons,
            'raw_person_boxes': raw_person_boxes
        })
        
    total_gt_persons = sum(len(item['gt_persons']) for item in cached_predictions)
    print(f"[INFO] Total Ground Truth Persons: {total_gt_persons}")

    # First, analyze baseline legitimate overlaps and true duplicates
    # A candidate pair is baseline predicted boxes (p1, p2) with IoU >= 0.50
    baseline_candidate_pairs = []
    for item in cached_predictions:
        preds = item['raw_person_boxes']
        gt_boxes = item['gt_persons']
        n = len(preds)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = preds[i], preds[j]
                iou = box_iou(p1['box'], p2['box'])
                if iou >= 0.50:
                    # Match p1 to GT
                    match_1 = -1
                    best_iou1 = 0.0
                    for g_idx, gp in enumerate(gt_boxes):
                        giou = box_iou(p1['box'], gp)
                        if giou > best_iou1:
                            best_iou1 = giou
                            match_1 = g_idx
                    if best_iou1 < 0.50:
                        match_1 = -1

                    # Match p2 to GT
                    match_2 = -1
                    best_iou2 = 0.0
                    for g_idx, gp in enumerate(gt_boxes):
                        giou = box_iou(p2['box'], gp)
                        if giou > best_iou2:
                            best_iou2 = giou
                            match_2 = g_idx
                    if best_iou2 < 0.50:
                        match_2 = -1

                    category = "UNKNOWN"
                    if match_1 >= 0 and match_2 >= 0:
                        if match_1 == match_2:
                            category = "TRUE_DUPLICATE"
                        else:
                            category = "LEGITIMATE_OVERLAP"
                    elif match_1 >= 0 or match_2 >= 0:
                        category = "PARTIAL_FP"
                    else:
                        category = "FALSE_POSITIVE"

                    baseline_candidate_pairs.append({
                        'img_name': item['img_name'],
                        'p1': p1,
                        'p2': p2,
                        'iou': iou,
                        'category': category,
                        'match_1': match_1,
                        'match_2': match_2
                    })

    baseline_true_dups = [p for p in baseline_candidate_pairs if p['category'] == "TRUE_DUPLICATE"]
    baseline_legit_overlaps = [p for p in baseline_candidate_pairs if p['category'] == "LEGITIMATE_OVERLAP"]
    print(f"[INFO] Baseline Duplicate Pairs (IoU>=0.50): Total={len(baseline_candidate_pairs)}, TrueDups={len(baseline_true_dups)}, LegitOverlaps={len(baseline_legit_overlaps)}")

    results_summary = []

    # Run evaluation for Baseline (None) and each threshold in iou_thresholds
    configs_to_eval = [('BASELINE', None)] + [(f"IoU < {t:.2f}", t) for t in iou_thresholds]

    for label, thresh in configs_to_eval:
        total_pred_persons = 0
        all_remaining_pairs = []
        true_dups_remaining = 0
        legit_overlaps_suppressed = 0

        total_tp = 0
        total_fp = 0
        total_fn = 0
        gt_matched_counts = [0] * total_gt_persons # count of pred boxes matching each GT person

        # Track suppressed boxes to check legitimate overlap suppression
        for item in cached_predictions:
            raw_boxes = item['raw_person_boxes']
            gt_boxes = item['gt_persons']

            if thresh is None:
                kept_boxes = raw_boxes
            else:
                kept_boxes = suppress_person_duplicates(raw_boxes, thresh)

            total_pred_persons += len(kept_boxes)

            # Check remaining duplicate pairs (IoU >= 0.50) in kept boxes
            nk = len(kept_boxes)
            for i in range(nk):
                for j in range(i + 1, nk):
                    kiou = box_iou(kept_boxes[i]['box'], kept_boxes[j]['box'])
                    if kiou >= 0.50:
                        # match to GT
                        m1, m2 = -1, -1
                        b1, b2 = 0.0, 0.0
                        for g_idx, gp in enumerate(gt_boxes):
                            g1 = box_iou(kept_boxes[i]['box'], gp)
                            if g1 > b1: b1 = g1; m1 = g_idx
                            g2 = box_iou(kept_boxes[j]['box'], gp)
                            if g2 > b2: b2 = g2; m2 = g_idx
                        if b1 < 0.50: m1 = -1
                        if b2 < 0.50: m2 = -1

                        if m1 >= 0 and m2 >= 0 and m1 == m2:
                            true_dups_remaining += 1

            # Match kept boxes to GT persons for Precision / Recall / F1 / GT Missed
            gt_detected = [False] * len(gt_boxes)
            matched_gt_indices = set()

            # Greedy matching for TP count: assign highest IoU >= 0.50 first
            gt_matched_for_item = [0] * len(gt_boxes)
            
            for kb in kept_boxes:
                best_g_idx = -1
                best_g_iou = 0.0
                for g_idx, gp in enumerate(gt_boxes):
                    giou = box_iou(kb['box'], gp)
                    if giou > best_g_iou:
                        best_g_iou = giou
                        best_g_idx = g_idx

                if best_g_iou >= 0.50 and best_g_idx >= 0:
                    gt_matched_for_item[best_g_idx] += 1
                    if not gt_detected[best_g_idx]:
                        gt_detected[best_g_idx] = True
                        total_tp += 1
                    else:
                        # Extra box matching already-detected GT count as FP (duplicate prediction)
                        total_fp += 1
                else:
                    total_fp += 1

            # Unmatched GT persons count as FN
            fn_for_item = sum(1 for det in gt_detected if not det)
            total_fn += fn_for_item

            # Store gt_matched_for_item into overall list
            gt_matched_counts.extend(gt_matched_for_item)

            # Check legitimate overlap pairs suppression
            if thresh is not None:
                # Find which baseline legitimate overlap pairs in this image had a box suppressed
                kept_box_ids = set(id(b) for b in kept_boxes)
                for b_pair in baseline_legit_overlaps:
                    if b_pair['img_name'] == item['img_name']:
                        p1_kept = id(b_pair['p1']) in kept_box_ids
                        p2_kept = id(b_pair['p2']) in kept_box_ids
                        if not (p1_kept and p2_kept):
                            legit_overlaps_suppressed += 1

        # Calculate metrics
        precision = total_tp / max(1, (total_tp + total_fp))
        recall = total_tp / max(1, (total_tp + total_fn))
        f1 = (2 * precision * recall) / max(1e-6, (precision + recall))

        # Calculate duplicate pairs remaining (IoU >= 0.50)
        total_dup_pairs_remaining = 0
        for item in cached_predictions:
            if thresh is None:
                kb_list = item['raw_person_boxes']
            else:
                kb_list = suppress_person_duplicates(item['raw_person_boxes'], thresh)
            for i in range(len(kb_list)):
                for j in range(i + 1, len(kb_list)):
                    if box_iou(kb_list[i]['box'], kb_list[j]['box']) >= 0.50:
                        total_dup_pairs_remaining += 1

        # Calculate average person boxes per GT person
        # sum of boxes matching any GT person / total GT persons
        total_boxes_matching_gt = sum(gt_matched_counts)
        avg_boxes_per_gt = total_boxes_matching_gt / max(1, total_gt_persons)

        results_summary.append({
            'threshold_label': label,
            'thresh_val': thresh,
            'total_pred_persons': total_pred_persons,
            'duplicate_pairs_remaining': total_dup_pairs_remaining,
            'true_duplicate_pairs_remaining': true_dups_remaining,
            'legit_overlaps_suppressed': legit_overlaps_suppressed,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'gt_persons_missed': total_fn,
            'avg_boxes_per_gt': avg_boxes_per_gt,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        })

    return cached_predictions, baseline_candidate_pairs, results_summary

def evaluate_video(model, video_path, iou_thresholds, output_dir, person_conf=0.50):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = []
    print(f"[INFO] Reading video frames from {video_path}...")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()

    print(f"[INFO] Total video frames loaded: {len(frames)}")

    # Pre-run raw YOLO inference on video frames to get raw detections
    print("[INFO] Running raw YOLO inference on video frames...")
    cached_video_dets = []
    for idx, frame in enumerate(frames):
        res = model.predict(frame, imgsz=800, conf=0.05, verbose=False)[0]
        raw_dets = []
        for box in res.boxes:
            c = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            raw_dets.append({'cls': c, 'conf': conf, 'box': xyxy})
        cached_video_dets.append(raw_dets)

    configs = [('BASELINE', None)] + [(f"IoU < {t:.2f}", t) for t in iou_thresholds]
    video_summary = []

    before_after_dir = output_dir / "before_after"
    before_after_dir.mkdir(parents=True, exist_ok=True)

    for label, thresh in configs:
        associator = PPEAssociator(PPEAssociationConfig(person_conf=person_conf, helmet_conf=0.25, mask_conf=0.20))
        temp_engine = TemporalConfirmationEngine(TemporalConfirmationConfig(
            confirmation_frames=5,
            min_track_iou=0.30,
            max_missed_frames=10,
            fps=fps_video,
            alert_cooldown_seconds=5.0,
            uncertain_breaks_streak=True
        ))

        frames_with_dup_persons = 0
        total_active_tracks_sum = 0
        all_track_ids_created = set()
        frames_with_dup_tracks = 0

        t_start = time.time()

        for idx, (frame, raw_dets) in enumerate(zip(frames, cached_video_dets)):
            # Separate person vs non-person
            person_boxes = [d for d in raw_dets if d['cls'] == 2 and d['conf'] >= person_conf]
            non_person_boxes = [d for d in raw_dets if d['cls'] != 2 or d['conf'] < person_conf]

            if thresh is None:
                kept_persons = person_boxes
            else:
                # Apply person-only duplicate suppression
                kept_persons_struct = suppress_person_duplicates(
                    [{'box': d['box'], 'conf': d['conf'], 'raw': d} for d in person_boxes],
                    thresh
                )
                kept_persons = [item['raw'] for item in kept_persons_struct]

            # Count duplicate person boxes in this frame (IoU >= 0.50)
            dup_count = 0
            for i in range(len(kept_persons)):
                for j in range(i + 1, len(kept_persons)):
                    if box_iou(kept_persons[i]['box'], kept_persons[j]['box']) >= 0.50:
                        dup_count += 1
            if dup_count > 0:
                frames_with_dup_persons += 1

            # Combine back with non-person boxes for PPE association
            combined_dets = kept_persons + non_person_boxes

            # Stage 2: PPE Association
            person_states = associator.process_detections(combined_dets, frame_w, frame_h)

            # Stage 3: Temporal confirmation
            events = temp_engine.process_frame(person_states, frame_index=idx, timestamp_sec=idx / fps_video)

            active_tracks = temp_engine.active_tracks
            total_active_tracks_sum += len(active_tracks)

            for trk in active_tracks:
                all_track_ids_created.add(trk.track_id)

            # Check duplicate track IDs (active tracks with IoU >= 0.50)
            track_dup_in_frame = 0
            for i in range(len(active_tracks)):
                for j in range(i + 1, len(active_tracks)):
                    if box_iou(list(active_tracks[i].last_bbox), list(active_tracks[j].last_bbox)) >= 0.50:
                        track_dup_in_frame += 1
            if track_dup_in_frame > 0:
                frames_with_dup_tracks += 1

            # Save visual before/after comparison frames for selected key frames
            # Let's pick frame 50, 100, 150, 200, 250
            if idx in [45, 95, 145, 195, 245]:
                annotated = frame.copy()
                for trk in active_tracks:
                    bx1, by1, bx2, by2 = [int(v) for v in trk.last_bbox]
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 255) if thresh else (0, 0, 255), 2)
                    cv2.putText(annotated, f"Track #{trk.track_id}", (bx1, max(20, by1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if thresh else (0, 0, 255), 2)
                
                tag = "baseline" if thresh is None else f"suppressed_iou{thresh:.2f}"
                cv2.imwrite(str(before_after_dir / f"frame_{idx:04d}_{tag}.jpg"), annotated)

        t_elapsed = time.time() - t_start
        fps_measured = len(frames) / max(1e-6, t_elapsed)
        latency_ms = (t_elapsed / max(1, len(frames))) * 1000.0

        avg_active_tracks = total_active_tracks_sum / max(1, len(frames))

        video_summary.append({
            'threshold_label': label,
            'thresh_val': thresh,
            'frames_with_dup_persons': frames_with_dup_persons,
            'avg_active_tracks': round(avg_active_tracks, 2),
            'unique_track_ids_created': len(all_track_ids_created),
            'frames_with_dup_tracks': frames_with_dup_tracks,
            'fps_measured': round(fps_measured, 1),
            'latency_ms': round(latency_ms, 2)
        })

    return video_summary

def main():
    model_path = os.path.join(PROJECT_ROOT, "runs", "detect", "safety_v1-4_run2b_yolov8s_800", "weights", "best.pt")
    val_img_dir = os.path.join(PROJECT_ROOT, "training_dataset_v2", "images", "val")
    val_lbl_dir = os.path.join(PROJECT_ROOT, "training_dataset_v2", "labels", "val")
    video_path = os.path.join(PROJECT_ROOT, "data_collection", "videos", "4048038451-preview.mp4")

    output_dir = Path(os.path.join(PROJECT_ROOT, "reports", "person_duplicate_analysis_v1"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("   PERSON DUPLICATE SUPPRESSION EXPERIMENT")
    print("==================================================================")

    model = YOLO(model_path)
    iou_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]

    # Evaluate Validation Set
    cached_preds, baseline_candidate_pairs, val_summary = evaluate_val_set(
        model, val_img_dir, val_lbl_dir, iou_thresholds, person_conf=0.50
    )

    # Evaluate Live Video
    video_summary = evaluate_video(
        model, video_path, iou_thresholds, output_dir, person_conf=0.50
    )

    # Combine results into CSV
    csv_path = output_dir / "suppression_experiment.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Threshold_Label", "Threshold_Val",
            "Val_Total_Pred_Persons", "Val_Duplicate_Pairs_Remaining", "Val_True_Duplicate_Pairs_Remaining",
            "Val_Legit_Overlaps_Suppressed", "Val_Precision", "Val_Recall", "Val_F1",
            "Val_GT_Persons_Missed", "Val_Avg_Boxes_Per_GT",
            "Video_Frames_With_Dup_Persons", "Video_Avg_Active_Tracks", "Video_Unique_Track_IDs",
            "Video_Frames_With_Dup_Tracks", "Video_FPS", "Video_Latency_MS"
        ])

        for val_s, vid_s in zip(val_summary, video_summary):
            writer.writerow([
                val_s['threshold_label'],
                val_s['thresh_val'] if val_s['thresh_val'] is not None else "Baseline",
                val_s['total_pred_persons'],
                val_s['duplicate_pairs_remaining'],
                val_s['true_duplicate_pairs_remaining'],
                val_s['legit_overlaps_suppressed'],
                round(val_s['precision'], 4),
                round(val_s['recall'], 4),
                round(val_s['f1'], 4),
                val_s['gt_persons_missed'],
                round(val_s['avg_boxes_per_gt'], 4),
                vid_s['frames_with_dup_persons'],
                vid_s['avg_active_tracks'],
                vid_s['unique_track_ids_created'],
                vid_s['frames_with_dup_tracks'],
                vid_s['fps_measured'],
                vid_s['latency_ms']
            ])

    print(f"\n[SUCCESS] Experiment CSV written to: {csv_path}")

    # Print summary table to console
    print("\n==================================================================")
    print("              EXPERIMENT RESULTS TABLE")
    print("==================================================================")
    print(f"{'Setting':<12} | {'PredP':<5} | {'DupRem':<6} | {'TrueDup':<7} | {'LegitSupp':<9} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'GTMissed':<8} | {'AvgB/GT':<7}")
    print("-" * 95)
    for s in val_summary:
        print(f"{s['threshold_label']:<12} | {s['total_pred_persons']:<5} | {s['duplicate_pairs_remaining']:<6} | {s['true_duplicate_pairs_remaining']:<7} | {s['legit_overlaps_suppressed']:<9} | {s['precision']:<6.4f} | {s['recall']:<6.4f} | {s['f1']:<6.4f} | {s['gt_persons_missed']:<8} | {s['avg_boxes_per_gt']:<7.4f}")

    print("\n[VIDEO SUMMARY]")
    print(f"{'Setting':<12} | {'DupFrames':<9} | {'AvgTracks':<9} | {'TrackIDs':<8} | {'DupTrackFrames':<14} | {'FPS':<6} | {'Latency(ms)':<11}")
    print("-" * 85)
    for s in video_summary:
        print(f"{s['threshold_label']:<12} | {s['frames_with_dup_persons']:<9} | {s['avg_active_tracks']:<9} | {s['unique_track_ids_created']:<8} | {s['frames_with_dup_tracks']:<14} | {s['fps_measured']:<6.1f} | {s['latency_ms']:<11.2f}")

if __name__ == '__main__':
    main()
