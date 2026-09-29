"""
Threshold Sensitivity Analysis for MASK_CONF (Phase 7.6)
=========================================================
Evaluates MASK_CONF thresholds: 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
Reports TP, FP, FN, Precision, Recall, F1, and Face-Region Mask FPs.
DO NOT RETRAIN, DO NOT MODIFY PROJECT CONFIG OR THRESHOLDS.
"""

import os
import sys
import glob
import cv2
import csv
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def xywh2xyxy(box, w, h):
    cx, cy, bw, bh = box
    x1 = max(0.0, (cx - bw / 2.0) * w)
    y1 = max(0.0, (cy - bh / 2.0) * h)
    x2 = min(float(w), (cx + bw / 2.0) * w)
    y2 = min(float(h), (cy + bh / 2.0) * h)
    return [x1, y1, x2, y2]

def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def box_ioa(inner_box, outer_box):
    x1 = max(inner_box[0], outer_box[0])
    y1 = max(inner_box[1], outer_box[1])
    x2 = min(inner_box[2], outer_box[2])
    y2 = min(inner_box[3], outer_box[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    inner_area = max(0.0, inner_box[2] - inner_box[0]) * max(0.0, inner_box[3] - inner_box[1])
    return inter / inner_area if inner_area > 0 else 0.0


def main():
    model_path = os.path.join(PROJECT_ROOT, "runs", "detect", "safety_v1-4_run2b_yolov8s_800", "weights", "best.pt")
    val_img_dir = os.path.join(PROJECT_ROOT, "training_dataset_v2", "images", "val")
    val_lbl_dir = os.path.join(PROJECT_ROOT, "training_dataset_v2", "labels", "val")

    output_dir = Path(os.path.join(PROJECT_ROOT, "reports", "mask_fp_analysis_v1"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("   MASK_CONF THRESHOLD SENSITIVITY ANALYSIS (VAL SET: 136 IMAGES)")
    print("==================================================================")

    model = YOLO(model_path)
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.*")))

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    # Store all frame detections first at conf=0.05
    frame_data = []

    for img_path in val_images:
        img_name = os.path.basename(img_path)
        lbl_path = os.path.join(val_lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # GT parsing
        gt_persons = []
        gt_masks = []

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c = int(parts[0])
                        box = xywh2xyxy([float(x) for x in parts[1:]], w, h)
                        if c == 2:
                            gt_persons.append(box)
                        elif c == 1:
                            gt_masks.append(box)

        results = model.predict(img, imgsz=800, conf=0.05, verbose=False)[0]

        pred_masks = []
        pred_persons = []

        for box in results.boxes:
            c = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()

            if c == 1:
                pred_masks.append({'box': xyxy, 'conf': conf})
            elif c == 2:
                pred_persons.append({'box': xyxy, 'conf': conf})

        frame_data.append({
            'img_name': img_name,
            'gt_masks': gt_masks,
            'gt_persons': gt_persons,
            'pred_masks': pred_masks,
            'pred_persons': pred_persons
        })

    # Evaluate each threshold
    results_table = []

    for t in thresholds:
        tp = 0
        fp = 0
        fn = 0
        face_fp = 0
        total_gt_masks = 0

        for frame in frame_data:
            gt_masks = frame['gt_masks']
            gt_persons = frame['gt_persons']
            pred_masks = [pm for pm in frame['pred_masks'] if pm['conf'] >= t]
            pred_persons = frame['pred_persons']

            total_gt_masks += len(gt_masks)
            gt_matched = [False] * len(gt_masks)

            # Match pred masks to GT masks by highest IoU >= 0.30
            sorted_pred = sorted(pred_masks, key=lambda x: x['conf'], reverse=True)

            for pm in sorted_pred:
                p_box = pm['box']
                best_iou = 0.0
                best_gt_idx = -1

                for g_idx, gm in enumerate(gt_masks):
                    if not gt_matched[g_idx]:
                        iou = box_iou(p_box, gm)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx

                if best_iou >= 0.30 and best_gt_idx >= 0:
                    tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp += 1
                    # Check face overlap
                    target_persons = gt_persons if gt_persons else [p['box'] for p in pred_persons]
                    is_face_fp = False

                    for person_box in target_persons:
                        p_w = max(1.0, person_box[2] - person_box[0])
                        p_h = max(1.0, person_box[3] - person_box[1])

                        head_roi = [
                            person_box[0],
                            person_box[1],
                            person_box[2],
                            person_box[1] + 0.45 * p_h
                        ]

                        ioa_head = box_ioa(p_box, head_roi)
                        ioa_person = box_ioa(p_box, person_box)

                        if ioa_head > 0.15 or ioa_person > 0.40:
                            is_face_fp = True
                            break

                    if is_face_fp:
                        face_fp += 1

            fn += (len(gt_masks) - sum(gt_matched))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        res = {
            'threshold': t,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'face_fp': face_fp
        }
        results_table.append(res)

    print("\n[THRESHOLD SENSITIVITY ANALYSIS RESULTS]")
    print(f"{'Threshold':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'Face FP':<8}")
    print("-" * 80)

    for r in results_table:
        print(f"{r['threshold']:<10.2f} | {r['tp']:<6} | {r['fp']:<6} | {r['fn']:<6} | {r['precision']:<10.4f} | {r['recall']:<10.4f} | {r['f1']:<10.4f} | {r['face_fp']:<8}")

    # Write CSV
    csv_path = output_dir / "mask_conf_sensitivity_analysis.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['MASK_CONF', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1', 'Face_Region_Mask_FP'])
        for r in results_table:
            writer.writerow([r['threshold'], r['tp'], r['fp'], r['fn'], round(r['precision'], 4), round(r['recall'], 4), round(r['f1'], 4), r['face_fp']])

    print(f"\n[CSV SAVED] {csv_path}")

if __name__ == '__main__':
    main()
