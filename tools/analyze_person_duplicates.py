"""
Person Duplicate Bounding Box Diagnostic Script (Phase 7.6)
============================================================
Rigorous diagnostic analysis of duplicate person bounding boxes.
THIS SCRIPT IS DIAGNOSIS ONLY. DO NOT RETRAIN, DO NOT ALTER MODEL OR THRESHOLDS.
"""

import os
import sys
import glob
import math
import cv2
import csv
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.safety.ppe_association import BBox, compute_intersection

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

    output_dir = Path(os.path.join(PROJECT_ROOT, "reports", "person_duplicate_analysis_v1"))
    sheets_dir = output_dir / "duplicate_contact_sheets"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("   PERSON DUPLICATE BOUNDING BOX DIAGNOSIS (PHASE 7.6)")
    print("==================================================================")
    print(f"  Model Path  : {model_path}")
    print(f"  Val Images  : {val_img_dir}")
    print(f"  Output Dir  : {output_dir}")

    model = YOLO(model_path)
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.*")))

    person_conf_thresh = 0.50
    candidate_pairs = []
    
    total_images_processed = len(val_images)
    total_gt_persons_all = 0
    total_pred_persons_all = 0

    for img_path in val_images:
        img_name = os.path.basename(img_path)
        lbl_path = os.path.join(val_lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Load GT Person boxes
        gt_persons = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5 and int(parts[0]) == 2:
                        box = xywh2xyxy([float(x) for x in parts[1:]], w, h)
                        gt_persons.append(box)

        total_gt_persons_all += len(gt_persons)

        # Run YOLO inference
        results = model.predict(img, imgsz=800, conf=person_conf_thresh, verbose=False)[0]

        pred_persons = []
        for box in results.boxes:
            c = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            if c == 2 and conf >= person_conf_thresh:
                pred_persons.append({'box': xyxy, 'conf': conf})

        total_pred_persons_all += len(pred_persons)

        # Compare every pair of predicted person boxes
        n_preds = len(pred_persons)
        for i in range(n_preds):
            for j in range(i + 1, n_preds):
                p1 = pred_persons[i]
                p2 = pred_persons[j]

                iou = box_iou(p1['box'], p2['box'])
                if iou >= 0.50:
                    # Order by higher confidence box A, lower confidence box B
                    if p1['conf'] >= p2['conf']:
                        box_a, conf_a = p1['box'], p1['conf']
                        box_b, conf_b = p2['box'], p2['conf']
                    else:
                        box_a, conf_a = p2['box'], p2['conf']
                        box_b, conf_b = p1['box'], p1['conf']

                    conf_diff = conf_a - conf_b

                    # Match against GT Person boxes
                    gt_match_a = -1
                    best_iou_a = 0.0
                    for g_idx, gp in enumerate(gt_persons):
                        giou = box_iou(box_a, gp)
                        if giou > best_iou_a:
                            best_iou_a = giou
                            gt_match_a = g_idx

                    gt_match_b = -1
                    best_iou_b = 0.0
                    for g_idx, gp in enumerate(gt_persons):
                        giou = box_iou(box_b, gp)
                        if giou > best_iou_b:
                            best_iou_b = giou
                            gt_match_b = g_idx

                    # Categorize duplicate pair
                    if gt_match_a >= 0 and gt_match_b >= 0:
                        if gt_match_a == gt_match_b:
                            category = "TRUE_DUPLICATE"
                        else:
                            category = "LEGITIMATE_OVERLAP"
                    elif gt_match_a >= 0 or gt_match_b >= 0:
                        # Check if B is a fragment of A
                        ioa_b_in_a = box_ioa(box_b, box_a)
                        if ioa_b_in_a > 0.60:
                            category = "PARTIAL_FRAGMENT"
                        else:
                            category = "FALSE_POSITIVE"
                    else:
                        category = "FALSE_POSITIVE"

                    candidate = {
                        'image_name': img_name,
                        'iou': round(iou, 4),
                        'conf_a': round(conf_a, 4),
                        'conf_b': round(conf_b, 4),
                        'conf_diff': round(conf_diff, 4),
                        'box_a': [round(v, 1) for v in box_a],
                        'box_b': [round(v, 1) for v in box_b],
                        'category': category,
                        'gt_match_a': gt_match_a,
                        'gt_match_b': gt_match_b,
                        'img_w': w,
                        'img_h': h
                    }
                    candidate_pairs.append(candidate)

                    # Save debug visualization
                    annotated = img.copy()
                    # Draw GT persons in Green
                    for gp in gt_persons:
                        cv2.rectangle(annotated, (int(gp[0]), int(gp[1])), (int(gp[2]), int(gp[3])), (0, 255, 0), 2)

                    # Draw Box A in Red, Box B in Cyan
                    cv2.rectangle(annotated, (int(box_a[0]), int(box_a[1])), (int(box_a[2]), int(box_a[3])), (0, 0, 255), 3)
                    cv2.rectangle(annotated, (int(box_b[0]), int(box_b[1])), (int(box_b[2]), int(box_b[3])), (255, 255, 0), 2)

                    txt = f"IoU={iou:.2f} | A={conf_a:.2f} B={conf_b:.2f} | {category}"
                    cv2.putText(annotated, txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    save_name = f"dup_iou{iou:.2f}_{category}_{img_name}"
                    save_path = output_dir / save_name
                    cv2.imwrite(str(save_path), annotated)

    # Buckets Analysis
    def analyze_bucket(min_iou):
        filtered = [c for c in candidate_pairs if c['iou'] >= min_iou]
        unique_imgs = set(c['image_name'] for c in filtered)
        true_dups = [c for c in filtered if c['category'] == 'TRUE_DUPLICATE']
        legit_overlaps = [c for c in filtered if c['category'] == 'LEGITIMATE_OVERLAP']
        fragments = [c for c in filtered if c['category'] == 'PARTIAL_FRAGMENT']
        fps = [c for c in filtered if c['category'] == 'FALSE_POSITIVE']

        avg_iou = np.mean([c['iou'] for c in filtered]) if filtered else 0.0
        max_iou = np.max([c['iou'] for c in filtered]) if filtered else 0.0
        avg_conf_a = np.mean([c['conf_a'] for c in filtered]) if filtered else 0.0
        avg_conf_b = np.mean([c['conf_b'] for c in filtered]) if filtered else 0.0
        avg_diff = np.mean([c['conf_diff'] for c in filtered]) if filtered else 0.0

        return {
            'min_iou': min_iou,
            'total_pairs': len(filtered),
            'unique_images': len(unique_imgs),
            'true_duplicates': len(true_dups),
            'legit_overlaps': len(legit_overlaps),
            'fragments': len(fragments),
            'false_positives': len(fps),
            'avg_iou': round(float(avg_iou), 4),
            'max_iou': round(float(max_iou), 4),
            'avg_conf_a': round(float(avg_conf_a), 4),
            'avg_conf_b': round(float(avg_conf_b), 4),
            'avg_diff': round(float(avg_diff), 4)
        }

    b90 = analyze_bucket(0.90)
    b80 = analyze_bucket(0.80)
    b70 = analyze_bucket(0.70)
    b50 = analyze_bucket(0.50)

    print("\n==================================================================")
    print("   RAW YOLO PERSON DUPLICATE SUMMARY (VAL SET: 136 IMAGES)")
    print("==================================================================")
    print(f"Total Validation Images      : {total_images_processed}")
    print(f"Total Ground-Truth Persons   : {total_gt_persons_all}")
    print(f"Total Predicted Persons      : {total_pred_persons_all}")
    print(f"Total Duplicate Pairs (>=0.5): {len(candidate_pairs)}")

    print("\n[DUPLICATE BUCKET BREAKDOWN]")
    print(f"{'IoU Threshold':<14} | {'Pairs':<6} | {'Images':<8} | {'Same-GT True Dup':<18} | {'Legit Overlap':<14} | {'Avg Diff':<10}")
    print("-" * 80)
    for b in [b90, b80, b70, b50]:
        print(f"IoU >= {b['min_iou']:<7.2f} | {b['total_pairs']:<6} | {b['unique_images']:<8} | {b['true_duplicates']:<18} | {b['legit_overlaps']:<14} | {b['avg_diff']:<10.4f}")

    # Write duplicate_candidates.csv
    csv_cand_path = output_dir / "duplicate_candidates.csv"
    with open(csv_cand_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'iou', 'conf_a', 'conf_b', 'conf_diff', 'box_a', 'box_b', 'category', 'gt_match_a', 'gt_match_b', 'img_w', 'img_h'])
        writer.writeheader()
        writer.writerows(candidate_pairs)

    print(f"\n[CSV WRITTEN] {csv_cand_path}")

    # Create Contact Sheet for high IoU candidate duplicates
    high_iou_dups = [c for c in candidate_pairs if c['iou'] >= 0.70]
    if high_iou_dups:
        cell_w, cell_h = 320, 240
        cols = 4
        rows = math.ceil(len(high_iou_dups) / cols)
        sheet_img = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for i, item in enumerate(high_iou_dups):
            r = i // cols
            c = i % cols
            img_p = output_dir / f"dup_iou{item['iou']:.2f}_{item['category']}_{item['image_name']}"
            if img_p.exists():
                c_img = cv2.imread(str(img_p))
                if c_img is not None:
                    resized = cv2.resize(c_img, (cell_w, cell_h))
                    sheet_img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized

        sheet_save_path = sheets_dir / "person_duplicate_contact_sheet_01.jpg"
        cv2.imwrite(str(sheet_save_path), sheet_img)
        print(f"[CONTACT SHEET CREATED] {sheet_save_path}")

if __name__ == '__main__':
    main()
