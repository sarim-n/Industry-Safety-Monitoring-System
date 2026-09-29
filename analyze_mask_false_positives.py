"""
Mask False Positive Diagnosis & Analysis Script (Phase 7.6)
=============================================================
Performs comprehensive diagnostic analysis of mask false positives on human faces.
DO NOT RETRAIN, DO NOT MODIFY THRESHOLDS, DO NOT ALTER DATASET.
"""

import os
import sys
import glob
import math
import cv2
import json
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
    """Intersection over Inner Box Area."""
    x1 = max(inner_box[0], outer_box[0])
    y1 = max(inner_box[1], outer_box[1])
    x2 = min(inner_box[2], outer_box[2])
    y2 = min(inner_box[3], outer_box[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    inner_area = max(0.0, inner_box[2] - inner_box[0]) * max(0.0, inner_box[3] - inner_box[1])
    return inter / inner_area if inner_area > 0 else 0.0

def get_percentiles(arr):
    if len(arr) == 0:
        return {'count': 0, 'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'p25': 0, 'p75': 0}
    s_arr = sorted(arr)
    return {
        'count': len(s_arr),
        'min': float(np.min(s_arr)),
        'max': float(np.max(s_arr)),
        'mean': float(np.mean(s_arr)),
        'median': float(np.median(s_arr)),
        'p25': float(np.percentile(s_arr, 25)),
        'p75': float(np.percentile(s_arr, 75))
    }

def get_buckets(arr):
    buckets = {
        "0.20-0.30": 0,
        "0.30-0.40": 0,
        "0.40-0.50": 0,
        "0.50-0.60": 0,
        "0.60-0.70": 0,
        "0.70-0.80": 0,
        "0.80-0.90": 0,
        "0.90-1.00": 0
    }
    for val in arr:
        if 0.20 <= val < 0.30:
            buckets["0.20-0.30"] += 1
        elif 0.30 <= val < 0.40:
            buckets["0.30-0.40"] += 1
        elif 0.40 <= val < 0.50:
            buckets["0.40-0.50"] += 1
        elif 0.50 <= val < 0.60:
            buckets["0.50-0.60"] += 1
        elif 0.60 <= val < 0.70:
            buckets["0.60-0.70"] += 1
        elif 0.70 <= val < 0.80:
            buckets["0.70-0.80"] += 1
        elif 0.80 <= val < 0.90:
            buckets["0.80-0.90"] += 1
        elif 0.90 <= val <= 1.00:
            buckets["0.90-1.00"] += 1
    return buckets


def main():
    model_path = os.path.join(PROJECT_ROOT, "runs", "detect", "safety_v1-4_run2b_yolov8s_800", "weights", "best.pt")
    val_img_dir = os.path.join(PROJECT_ROOT, "training_dataset_v2", "images", "val")
    val_lbl_dir = os.path.join(PROJECT_ROOT, "training_dataset_v2", "labels", "val")

    train_img_dir = os.path.join(PROJECT_ROOT, "training_dataset_v2", "images", "train")
    train_lbl_dir = os.path.join(PROJECT_ROOT, "training_dataset_v2", "labels", "train")

    output_dir = Path(os.path.join(PROJECT_ROOT, "reports", "mask_fp_analysis_v1"))
    fp_img_dir = output_dir / "mask_fp_images"
    sheets_dir = output_dir / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    fp_img_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("   MASK FALSE POSITIVE DIAGNOSIS & ANALYSIS (PHASE 7.6)")
    print("==================================================================")
    print(f"  Model Path  : {model_path}")
    print(f"  Val Images  : {val_img_dir}")
    print(f"  Output Dir  : {output_dir}")

    # Load Model
    model = YOLO(model_path)

    # 1. Dataset Balance Analysis
    def analyze_dataset_labels(lbl_dir):
        total_helmets = 0
        total_masks = 0
        total_persons = 0
        person_with_mask = 0
        person_without_mask = 0

        lbl_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
        for lf in lbl_files:
            boxes = []
            with open(lf, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c = int(parts[0])
                        boxes.append(c)

            c_helmets = boxes.count(0)
            c_masks = boxes.count(1)
            c_persons = boxes.count(2)

            total_helmets += c_helmets
            total_masks += c_masks
            total_persons += c_persons

            # Estimate persons with masks in image
            if c_persons > 0:
                m_count = min(c_persons, c_masks)
                person_with_mask += m_count
                person_without_mask += (c_persons - m_count)

        return {
            'total_files': len(lbl_files),
            'helmets': total_helmets,
            'masks': total_masks,
            'persons': total_persons,
            'person_with_mask': person_with_mask,
            'person_without_mask': person_without_mask
        }

    train_stats = analyze_dataset_labels(train_lbl_dir)
    val_stats = analyze_dataset_labels(val_lbl_dir)

    print("\n[DATASET BALANCE ANALYSIS]")
    print(f"  Train Set: {train_stats['total_files']} images | Persons: {train_stats['persons']} | Masks: {train_stats['masks']} | Helmets: {train_stats['helmets']}")
    print(f"    - Persons WITH Mask    : {train_stats['person_with_mask']} ({train_stats['person_with_mask']/max(1, train_stats['persons'])*100:.1f}%)")
    print(f"    - Persons WITHOUT Mask : {train_stats['person_without_mask']} ({train_stats['person_without_mask']/max(1, train_stats['persons'])*100:.1f}%)")
    print(f"    - Mask / Person Ratio  : {train_stats['masks']/max(1, train_stats['persons']):.2f}")

    print(f"  Val Set  : {val_stats['total_files']} images | Persons: {val_stats['persons']} | Masks: {val_stats['masks']} | Helmets: {val_stats['helmets']}")
    print(f"    - Persons WITH Mask    : {val_stats['person_with_mask']} ({val_stats['person_with_mask']/max(1, val_stats['persons'])*100:.1f}%)")
    print(f"    - Persons WITHOUT Mask : {val_stats['person_without_mask']} ({val_stats['person_without_mask']/max(1, val_stats['persons'])*100:.1f}%)")

    # 2. Run Inference on Validation Set
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.*")))
    
    true_mask_confs = []
    face_fp_mask_confs = []
    non_face_fp_mask_confs = []
    
    fp_records = []
    cropped_fp_images = []

    for img_path in val_images:
        img_name = os.path.basename(img_path)
        lbl_path = os.path.join(val_lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Load GT
        gt_persons = []
        gt_masks = []
        gt_helmets = []

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
                        elif c == 0:
                            gt_helmets.append(box)

        # Run YOLO Inference (conf=0.05 to capture full confidence spectrum)
        results = model.predict(img, imgsz=800, conf=0.05, verbose=False)[0]

        pred_masks = []
        pred_persons = []

        for box in results.boxes:
            c = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()

            if c == 1:  # Mask
                pred_masks.append({'box': xyxy, 'conf': conf})
            elif c == 2: # Person
                pred_persons.append({'box': xyxy, 'conf': conf})

        # Match Pred Masks against GT Masks
        gt_masks_matched = [False] * len(gt_masks)

        for pm in pred_masks:
            p_box = pm['box']
            p_conf = pm['conf']

            # Check matching with GT mask
            best_iou = 0.0
            matched_gt_idx = -1
            for idx, gm in enumerate(gt_masks):
                iou = box_iou(p_box, gm)
                if iou > best_iou:
                    best_iou = iou
                    matched_gt_idx = idx

            if best_iou >= 0.30:
                # True Positive Mask
                gt_masks_matched[matched_gt_idx] = True
                if p_conf >= 0.20:
                    true_mask_confs.append(p_conf)
            else:
                # False Positive Candidate
                # Check if it overlaps a person's head/face region
                # Head/Face region defined as top 45% of person box
                nearest_person = None
                min_dist = float('inf')
                overlaps_face = False
                position_label = "other"

                # Compare against GT persons or Pred persons
                target_persons = gt_persons if gt_persons else [p['box'] for p in pred_persons]
                
                for person_box in target_persons:
                    p_w = max(1.0, person_box[2] - person_box[0])
                    p_h = max(1.0, person_box[3] - person_box[1])

                    # Define Head/Face ROI (top 45% of person box)
                    head_roi = [
                        person_box[0],
                        person_box[1],
                        person_box[2],
                        person_box[1] + 0.45 * p_h
                    ]

                    # Face ROI specific: mouth/nose region (y1 + 0.08*h to y1 + 0.38*h)
                    face_roi = [
                        person_box[0] + 0.15 * p_w,
                        person_box[1] + 0.08 * p_h,
                        person_box[2] - 0.15 * p_w,
                        person_box[1] + 0.38 * p_h
                    ]

                    # Check IoA of mask box with head/face ROI
                    ioa_head = box_ioa(p_box, head_roi)
                    ioa_person = box_ioa(p_box, person_box)
                    iou_person = box_iou(p_box, person_box)

                    if ioa_head > 0.15 or ioa_person > 0.40:
                        overlaps_face = True
                        nearest_person = person_box

                        # Determine bbox position relative to face
                        m_center_y = (p_box[1] + p_box[3]) / 2.0
                        norm_y = (m_center_y - person_box[1]) / p_h

                        if 0.15 <= norm_y <= 0.32:
                            position_label = "nose_and_mouth"
                        elif 0.22 <= norm_y <= 0.38:
                            position_label = "lower_face_mouth"
                        elif 0.05 <= norm_y < 0.20:
                            position_label = "upper_face_eyes"
                        elif 0.35 < norm_y <= 0.50:
                            position_label = "chin_neck"
                        else:
                            position_label = "entire_face"
                        break

                if overlaps_face:
                    if p_conf >= 0.20:
                        face_fp_mask_confs.append(p_conf)

                    # Compute dimensions & relative box stats
                    mask_w = p_box[2] - p_box[0]
                    mask_h = p_box[3] - p_box[1]

                    p_bbox_str = f"[{nearest_person[0]:.1f},{nearest_person[1]:.1f},{nearest_person[2]:.1f},{nearest_person[3]:.1f}]" if nearest_person else "N/A"
                    p_w = nearest_person[2] - nearest_person[0] if nearest_person else 0.0
                    p_h = nearest_person[3] - nearest_person[1] if nearest_person else 0.0

                    rel_w = mask_w / max(1.0, p_w)
                    rel_h = mask_h / max(1.0, p_h)

                    record = {
                        'image_name': img_name,
                        'mask_conf': p_conf,
                        'mask_bbox': [round(v, 1) for v in p_box],
                        'mask_width': round(mask_w, 1),
                        'mask_height': round(mask_h, 1),
                        'nearest_person_bbox': p_bbox_str,
                        'rel_width': round(rel_w, 3),
                        'rel_height': round(rel_h, 3),
                        'overlaps_face': overlaps_face,
                        'position_label': position_label,
                        'img_width': w,
                        'img_height': h
                    }
                    fp_records.append(record)

                    # Render debug image crop
                    if p_conf >= 0.20:
                        annotated = img.copy()
                        # Draw GT persons
                        for gp in gt_persons:
                            cv2.rectangle(annotated, (int(gp[0]), int(gp[1])), (int(gp[2]), int(gp[3])), (255, 0, 0), 2)
                        # Draw GT masks
                        for gm in gt_masks:
                            cv2.rectangle(annotated, (int(gm[0]), int(gm[1])), (int(gm[2]), int(gm[3])), (0, 255, 0), 2)
                        # Draw FP Mask (Red)
                        cv2.rectangle(annotated, (int(p_box[0]), int(p_box[1])), (int(p_box[2]), int(p_box[3])), (0, 0, 255), 3)
                        txt = f"MASK FP: {p_conf:.2f} ({position_label})"
                        cv2.putText(annotated, txt, (int(p_box[0]), max(20, int(p_box[1]) - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        # Crop around person/head with margin
                        crop_x1 = max(0, int(p_box[0] - mask_w * 1.5))
                        crop_y1 = max(0, int(p_box[1] - mask_h * 1.5))
                        crop_x2 = min(w, int(p_box[2] + mask_w * 1.5))
                        crop_y2 = min(h, int(p_box[3] + mask_h * 1.5))

                        crop_img = annotated[crop_y1:crop_y2, crop_x1:crop_x2]
                        if crop_img.size > 0:
                            save_path = fp_img_dir / f"fp_{len(fp_records):03d}_{p_conf:.2f}_{img_name}"
                            cv2.imwrite(str(save_path), crop_img)
                            cropped_fp_images.append({'path': str(save_path), 'conf': p_conf, 'img_name': img_name, 'pos': position_label})
                else:
                    if p_conf >= 0.20:
                        non_face_fp_mask_confs.append(p_conf)

    # 3. Confidence Distribution Statistics
    true_stats = get_percentiles(true_mask_confs)
    fp_stats = get_percentiles(face_fp_mask_confs)

    true_buckets = get_buckets(true_mask_confs)
    fp_buckets = get_buckets(face_fp_mask_confs)

    print("\n==================================================================")
    print("   CONFIDENCE DISTRIBUTION COMPARISON (MASK_CONF >= 0.20)")
    print("==================================================================")
    print(f"True Mask Detections (TP)     : Count={true_stats['count']}, Mean={true_stats['mean']:.3f}, Median={true_stats['median']:.3f}, P25={true_stats['p25']:.3f}, P75={true_stats['p75']:.3f}, Max={true_stats['max']:.3f}")
    print(f"Face-Region Mask False Pos   : Count={fp_stats['count']}, Mean={fp_stats['mean']:.3f}, Median={fp_stats['median']:.3f}, P25={fp_stats['p25']:.3f}, P75={fp_stats['p75']:.3f}, Max={fp_stats['max']:.3f}")

    print("\n[CONFIDENCE BUCKET BREAKDOWN]")
    print(f"{'Bucket':<12} | {'True Mask (TP)':<16} | {'Face Mask FP':<16}")
    print("-" * 50)
    for b in true_buckets.keys():
        print(f"{b:<12} | {true_buckets[b]:<16} | {fp_buckets[b]:<16}")

    # Write CSVs
    # a) mask_fp_analysis.csv
    csv_fp_path = output_dir / "mask_fp_analysis.csv"
    with open(csv_fp_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'mask_conf', 'mask_bbox', 'mask_width', 'mask_height', 'nearest_person_bbox', 'rel_width', 'rel_height', 'overlaps_face', 'position_label', 'img_width', 'img_height'])
        writer.writeheader()
        writer.writerows(fp_records)

    # b) confidence_analysis.csv
    csv_conf_path = output_dir / "confidence_analysis.csv"
    with open(csv_conf_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Count', 'Min', 'Max', 'Mean', 'Median', 'P25', 'P75'])
        writer.writerow(['True Masks (TP)', true_stats['count'], round(true_stats['min'], 3), round(true_stats['max'], 3), round(true_stats['mean'], 3), round(true_stats['median'], 3), round(true_stats['p25'], 3), round(true_stats['p75'], 3)])
        writer.writerow(['Face Mask FP', fp_stats['count'], round(fp_stats['min'], 3), round(fp_stats['max'], 3), round(fp_stats['mean'], 3), round(fp_stats['median'], 3), round(fp_stats['p25'], 3), round(fp_stats['p75'], 3)])
        writer.writerow([])
        writer.writerow(['Bucket', 'True Masks (TP)', 'Face Mask FP'])
        for b in true_buckets.keys():
            writer.writerow([b, true_buckets[b], fp_buckets[b]])

    # 4. Generate Contact Sheets for Visual Inspection
    if cropped_fp_images:
        cell_w, cell_h = 240, 240
        cols = 5
        rows = math.ceil(len(cropped_fp_images) / cols)
        sheet_img = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for i, item in enumerate(cropped_fp_images):
            r = i // cols
            c = i % cols
            c_img = cv2.imread(item['path'])
            if c_img is not None:
                resized = cv2.resize(c_img, (cell_w, cell_h))
                sheet_img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized

        cv2.imwrite(str(sheets_dir / "mask_fp_contact_sheet_01.jpg"), sheet_img)
        print(f"\n[CONTACT SHEET CREATED] {sheets_dir / 'mask_fp_contact_sheet_01.jpg'}")

    print("\n[DIAGNOSTIC ANALYSIS COMPLETE]")

if __name__ == '__main__':
    main()
