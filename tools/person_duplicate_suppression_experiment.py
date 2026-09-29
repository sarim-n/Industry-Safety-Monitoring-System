"""
Person Duplicate Suppression Experiment Script (v3)
===================================================
Evaluates Person-Class-Only Duplicate Suppression rules placed AFTER YOLO inference
and BEFORE Current Custom Tracker & PPE Association.

Dataset: training_dataset_v2 (136 validation images, 386 GT persons)
Videos: 6 representative videos (2,190 frames)
Model: runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt

ISOLATED DIAGNOSTIC EXPERIMENT ONLY. DO NOT MODIFY PRODUCTION CODE.
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

from src.safety.ppe_association import PPEAssociationConfig, PPEAssociator, box_iou
from src.safety.temporal_confirmation import TemporalConfirmationConfig, TemporalConfirmationEngine

def xywh2xyxy(box, w, h):
    cx, cy, bw, bh = box
    x1 = max(0.0, (cx - bw / 2.0) * w)
    y1 = max(0.0, (cy - bh / 2.0) * h)
    x2 = min(float(w), (cx + bw / 2.0) * w)
    y2 = min(float(h), (cy + bh / 2.0) * h)
    return [x1, y1, x2, y2]

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
    diag_small = min(diag_a, diag_b)
    w_large = max(w_a, w_b)
    h_large = max(h_a, h_b)
    w_small = min(w_a, w_b)
    h_small = min(h_a, h_b)
    area_large = max(area_a, area_b)
    area_small = min(area_a, area_b)

    norm_dist_diag_large = center_dist / max(1e-6, diag_large)
    norm_dist_diag_small = center_dist / max(1e-6, diag_small)
    norm_dist_img_w = dx / max(1.0, float(img_w))
    norm_dist_img_h = dy / max(1.0, float(img_h))

    ioa_a_in_b = box_ioa(box_a, box_b)
    ioa_b_in_a = box_ioa(box_b, box_a)
    max_containment = max(ioa_a_in_b, ioa_b_in_a)
    min_containment = min(ioa_a_in_b, ioa_b_in_a)

    area_ratio = area_small / max(1e-6, area_large)
    width_ratio = w_small / max(1e-6, w_large)
    height_ratio = h_small / max(1e-6, h_large)

    ar_a = w_a / max(1e-6, h_a)
    ar_b = w_b / max(1e-6, h_b)
    ar_diff = abs(ar_a - ar_b)

    inter_x = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    inter_y = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    horiz_overlap_ratio = inter_x / max(1e-6, w_small)
    vert_overlap_ratio = inter_y / max(1e-6, h_small)

    conf_diff = abs(conf_a - conf_b)

    return {
        'box_a': box_a, 'conf_a': conf_a,
        'box_b': box_b, 'conf_b': conf_b,
        'iou': iou,
        'w_a': w_a, 'h_a': h_a, 'area_a': area_a, 'cx_a': cx_a, 'cy_a': cy_a,
        'w_b': w_b, 'h_b': h_b, 'area_b': area_b, 'cx_b': cx_b, 'cy_b': cy_b,
        'center_dist': center_dist,
        'norm_dist_diag_large': norm_dist_diag_large,
        'norm_dist_diag_small': norm_dist_diag_small,
        'norm_dist_img_w': norm_dist_img_w,
        'norm_dist_img_h': norm_dist_img_h,
        'ioa_a_in_b': ioa_a_in_b,
        'ioa_b_in_a': ioa_b_in_a,
        'max_containment': max_containment,
        'min_containment': min_containment,
        'area_ratio': area_ratio,
        'width_ratio': width_ratio,
        'height_ratio': height_ratio,
        'ar_diff': ar_diff,
        'horiz_overlap_ratio': horiz_overlap_ratio,
        'vert_overlap_ratio': vert_overlap_ratio,
        'conf_diff': conf_diff
    }

def get_stats(vals):
    if not vals:
        return {'min': 0.0, 'p25': 0.0, 'median': 0.0, 'mean': 0.0, 'p75': 0.0, 'max': 0.0}
    arr = np.array(vals)
    return {
        'min': float(np.min(arr)),
        'p25': float(np.percentile(arr, 25)),
        'median': float(np.median(arr)),
        'mean': float(np.mean(arr)),
        'p75': float(np.percentile(arr, 75)),
        'max': float(np.max(arr))
    }

def apply_suppression_rule(raw_person_boxes, rule_fn, img_w, img_h):
    """
    Greedy suppression for person-class boxes.
    raw_person_boxes: list of dicts {'box': [x1, y1, x2, y2], 'conf': float}
    Returns kept person boxes.
    """
    sorted_boxes = sorted(raw_person_boxes, key=lambda x: x['conf'], reverse=True)
    kept = []
    for b in sorted_boxes:
        should_suppress = False
        for k in kept:
            feats = compute_pair_features(b['box'], k['box'], b['conf'], k['conf'], img_w, img_h)
            if rule_fn(feats):
                should_suppress = True
                break
        if not should_suppress:
            kept.append(b)
    return kept

def main():
    output_dir = Path(os.path.join(PROJECT_ROOT, "reports", "person_duplicate_analysis_v3"))
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    contact_dir = output_dir / "contact_sheets"
    viz_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    model_path = os.path.join(PROJECT_ROOT, "runs", "detect", "safety_v1-4_run2b_yolov8s_800", "weights", "best.pt")
    val_img_dir = os.path.join(PROJECT_ROOT, "training_dataset_v2", "images", "val")
    val_lbl_dir = os.path.join(PROJECT_ROOT, "training_dataset_v2", "labels", "val")
    vid_dir = os.path.join(PROJECT_ROOT, "data_collection", "videos")

    print("==================================================================")
    print("   PERSON DUPLICATE SUPPRESSION EXPERIMENT (PHASE 7.6 / V3)")
    print("==================================================================")

    model = YOLO(model_path)
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.*")))
    person_conf = 0.50

    # -------------------------------------------------------------------------
    # STEP 2 & 3: REPRODUCE BASELINE & BUILD CANDIDATE PAIR DATASET
    # -------------------------------------------------------------------------
    print(f"[INFO] Caching YOLO predictions for {len(val_images)} validation images...")
    cached_val_preds = []
    total_gt_persons = 0
    total_pred_persons = 0
    candidate_pairs = []

    for img_path in val_images:
        img_name = os.path.basename(img_path)
        lbl_path = os.path.join(val_lbl_dir, os.path.splitext(img_name)[0] + ".txt")
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        gt_persons = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5 and int(parts[0]) == 2:
                        gt_persons.append(xywh2xyxy([float(x) for x in parts[1:]], w, h))

        total_gt_persons += len(gt_persons)

        results = model.predict(img, imgsz=800, conf=person_conf, verbose=False)[0]
        raw_person_boxes = []
        for box in results.boxes:
            c = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            if c == 2 and conf >= person_conf:
                raw_person_boxes.append({'box': xyxy, 'conf': conf})

        total_pred_persons += len(raw_person_boxes)

        cached_val_preds.append({
            'img_path': img_path,
            'img_name': img_name,
            'img_w': w,
            'img_h': h,
            'gt_persons': gt_persons,
            'raw_person_boxes': raw_person_boxes
        })

        n = len(raw_person_boxes)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = raw_person_boxes[i], raw_person_boxes[j]
                iou = box_iou(p1['box'], p2['box'])
                if iou >= 0.50:
                    feats = compute_pair_features(p1['box'], p2['box'], p1['conf'], p2['conf'], w, h)

                    match_a, best_iou_a = -1, 0.0
                    for g_idx, gp in enumerate(gt_persons):
                        giou = box_iou(feats['box_a'], gp)
                        if giou > best_iou_a:
                            best_iou_a = giou
                            match_a = g_idx
                    if best_iou_a < 0.50: match_a = -1

                    match_b, best_iou_b = -1, 0.0
                    for g_idx, gp in enumerate(gt_persons):
                        giou = box_iou(feats['box_b'], gp)
                        if giou > best_iou_b:
                            best_iou_b = giou
                            match_b = g_idx
                    if best_iou_b < 0.50: match_b = -1

                    if match_a >= 0 and match_b >= 0:
                        category = "TRUE_DUPLICATE" if match_a == match_b else "LEGITIMATE_OVERLAP"
                    elif match_a >= 0 or match_b >= 0:
                        category = "PARTIAL_FP"
                    else:
                        category = "FALSE_POSITIVE"

                    candidate_pairs.append({
                        'img_name': img_name,
                        'img_path': img_path,
                        'category': category,
                        'match_a': match_a,
                        'match_b': match_b,
                        'img_w': w,
                        'img_h': h,
                        'gt_persons': gt_persons,
                        **feats
                    })

    true_dups = [c for c in candidate_pairs if c['category'] == "TRUE_DUPLICATE"]
    legit_overlaps = [c for c in candidate_pairs if c['category'] == "LEGITIMATE_OVERLAP"]
    partial_fps = [c for c in candidate_pairs if c['category'] == "PARTIAL_FP"]
    fps_list = [c for c in candidate_pairs if c['category'] == "FALSE_POSITIVE"]

    print("\n==================================================================")
    print("   REPRODUCED VALIDATION BASELINE SUMMARY")
    print("==================================================================")
    print(f"Total Validation Images          : {len(val_images)}")
    print(f"Total Ground-Truth Persons       : {total_gt_persons}")
    print(f"Total Predicted Persons          : {total_pred_persons}")
    print(f"Total Candidate Duplicate Pairs  : {len(candidate_pairs)} (IoU >= 0.50)")
    print(f"  - TRUE_DUPLICATE Pairs         : {len(true_dups)} ({len(true_dups)/max(1, len(candidate_pairs))*100:.1f}%)")
    print(f"  - LEGITIMATE_OVERLAP Pairs     : {len(legit_overlaps)} ({len(legit_overlaps)/max(1, len(candidate_pairs))*100:.1f}%)")
    print(f"  - PARTIAL_FP Pairs             : {len(partial_fps)} ({len(partial_fps)/max(1, len(candidate_pairs))*100:.1f}%)")
    print(f"  - FALSE_POSITIVE Pairs         : {len(fps_list)} ({len(fps_list)/max(1, len(candidate_pairs))*100:.1f}%)")

    # Write baseline.csv
    csv_base_path = output_dir / "baseline.csv"
    with open(csv_base_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["total_validation_images", len(val_images)])
        writer.writerow(["total_gt_persons", total_gt_persons])
        writer.writerow(["total_pred_persons", total_pred_persons])
        writer.writerow(["candidate_duplicate_pairs_iou50", len(candidate_pairs)])
        writer.writerow(["true_duplicate_pairs", len(true_dups)])
        writer.writerow(["legitimate_overlap_pairs", len(legit_overlaps)])
        writer.writerow(["partial_fp_pairs", len(partial_fps)])
        writer.writerow(["false_positive_pairs", len(fps_list)])

    print(f"[SUCCESS] Baseline CSV written to: {csv_base_path}")

    # -------------------------------------------------------------------------
    # STEP 4: VISUAL INSPECTION & CONTACT SHEETS
    # -------------------------------------------------------------------------
    print("\n[INFO] Generating visual contact sheets and crops...")
    def render_crop(c, tag):
        img = cv2.imread(c['img_path'])
        if img is None: return None
        box_a, box_b = [int(v) for v in c['box_a']], [int(v) for v in c['box_b']]

        for gp in c['gt_persons']:
            cv2.rectangle(img, (int(gp[0]), int(gp[1])), (int(gp[2]), int(gp[3])), (0, 255, 0), 2)

        cv2.rectangle(img, (box_a[0], box_a[1]), (box_a[2], box_a[3]), (0, 0, 255), 3)
        cv2.rectangle(img, (box_b[0], box_b[1]), (box_b[2], box_b[3]), (255, 255, 0), 2)

        cv2.putText(img, f"{tag} | IoU={c['iou']:.2f} | ConfA={c['conf_a']:.2f} ConfB={c['conf_b']:.2f}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f"NDist={c['norm_dist_diag_large']:.2f} | MaxCont={c['max_containment']:.2f} | AreaR={c['area_ratio']:.2f}",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        min_x = max(0, min(box_a[0], box_b[0]) - 40)
        min_y = max(0, min(box_a[1], box_b[1]) - 40)
        max_x = min(c['img_w'], max(box_a[2], box_b[2]) + 40)
        max_y = min(c['img_h'], max(box_a[3], box_b[3]) + 40)

        crop = img[min_y:max_y, min_x:max_x]
        return cv2.resize(crop, (400, 300))

    td_selected = sorted(true_dups, key=lambda x: x['iou'], reverse=True)[:10]
    lo_selected = sorted(legit_overlaps, key=lambda x: x['iou'], reverse=True)[:10]

    if td_selected:
        crops_td = [render_crop(c, "TRUE_DUP") for c in td_selected if render_crop(c, "TRUE_DUP") is not None]
        if crops_td:
            cols = 5
            rows = math.ceil(len(crops_td) / cols)
            sheet_td = np.zeros((rows * 300, cols * 400, 3), dtype=np.uint8)
            for idx, crp in enumerate(crops_td):
                r, col = idx // cols, idx % cols
                sheet_td[r*300:(r+1)*300, col*400:(col+1)*400] = crp
            cv2.imwrite(str(contact_dir / "true_duplicates_contact_sheet.jpg"), sheet_td)

    if lo_selected:
        crops_lo = [render_crop(c, "LEGIT_OVERLAP") for c in lo_selected if render_crop(c, "LEGIT_OVERLAP") is not None]
        if crops_lo:
            cols = 5
            rows = math.ceil(len(crops_lo) / cols)
            sheet_lo = np.zeros((rows * 300, cols * 400, 3), dtype=np.uint8)
            for idx, crp in enumerate(crops_lo):
                r, col = idx // cols, idx % cols
                sheet_lo[r*300:(r+1)*300, col*400:(col+1)*400] = crp
            cv2.imwrite(str(contact_dir / "legitimate_overlaps_contact_sheet.jpg"), sheet_lo)

    # -------------------------------------------------------------------------
    # STEP 5 & 6: DESIGN CONSERVATIVE MULTI-SIGNAL RULES & EVALUATION
    # -------------------------------------------------------------------------
    print("\n[INFO] Evaluating multi-signal candidate suppression rules on validation set...")

    # Define Candidate Rules
    # Rule 1: Ultra Conservative (Zero Worker Loss Target)
    def rule_1(f):
        return (f['iou'] >= 0.65) and (f['max_containment'] >= 0.95) and (f['norm_dist_diag_large'] <= 0.10) and (f['area_ratio'] >= 0.60)

    # Rule 2: High Safety (Max 2 Worker Loss Target)
    def rule_2(f):
        return (f['iou'] >= 0.55) and (f['max_containment'] >= 0.85) and (f['norm_dist_diag_large'] <= 0.15) and (f['area_ratio'] >= 0.50)

    # Rule 3: Balanced
    def rule_3(f):
        return (f['iou'] >= 0.50) and (f['max_containment'] >= 0.80) and (f['norm_dist_diag_large'] <= 0.20) and (f['area_ratio'] >= 0.40)

    # Rule 4: Confidence-Gated (High IoU + Strong Containment + Significant Conf Diff)
    def rule_4(f):
        return (f['iou'] >= 0.55) and (f['max_containment'] >= 0.85) and (f['conf_diff'] >= 0.15) and (f['area_ratio'] >= 0.45)

    candidate_rule_definitions = [
        ('BASELINE', None, "No suppression (Raw YOLO)"),
        ('RULE_1 (Ultra Conservative)', rule_1, "IoU>=0.65 & MaxCont>=0.95 & NDist<=0.10 & AreaR>=0.60"),
        ('RULE_2 (High Safety)', rule_2, "IoU>=0.55 & MaxCont>=0.85 & NDist<=0.15 & AreaR>=0.50"),
        ('RULE_3 (Balanced)', rule_3, "IoU>=0.50 & MaxCont>=0.80 & NDist<=0.20 & AreaR>=0.40"),
        ('RULE_4 (Conf-Gated)', rule_4, "IoU>=0.55 & MaxCont>=0.85 & ConfDiff>=0.15 & AreaR>=0.45")
    ]

    tot_td = max(1, len(true_dups))
    tot_lo = max(1, len(legit_overlaps))

    candidate_rules_eval = []

    for label, r_fn, desc in candidate_rule_definitions:
        total_pred_p = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0

        td_removed = 0
        lo_suppressed = 0
        partial_fp_affected = 0
        gt_matched_counts = []

        # Evaluate on validation dataset
        for item in cached_val_preds:
            raw_boxes = item['raw_person_boxes']
            gt_boxes = item['gt_persons']
            w, h = item['img_w'], item['img_h']

            if r_fn is None:
                kept_boxes = raw_boxes
            else:
                kept_boxes = apply_suppression_rule(raw_boxes, r_fn, w, h)

            total_pred_p += len(kept_boxes)

            gt_detected = [False] * len(gt_boxes)
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
                        total_fp += 1
                else:
                    total_fp += 1

            fn_for_item = sum(1 for det in gt_detected if not det)
            total_fn += fn_for_item
            gt_matched_counts.extend(gt_matched_for_item)

        # Evaluate candidate pair removal/suppression counts
        if r_fn is not None:
            for c in candidate_pairs:
                if r_fn(c):
                    if c['category'] == "TRUE_DUPLICATE":
                        td_removed += 1
                    elif c['category'] == "LEGITIMATE_OVERLAP":
                        lo_suppressed += 1
                    elif c['category'] in ("PARTIAL_FP", "FALSE_POSITIVE"):
                        partial_fp_affected += 1

        precision = total_tp / max(1, (total_tp + total_fp))
        recall = total_tp / max(1, (total_tp + total_fn))
        f1 = (2 * precision * recall) / max(1e-6, (precision + recall))
        avg_boxes_per_gt = sum(gt_matched_counts) / max(1, total_gt_persons)

        td_rem_pct = (td_removed / tot_td) * 100.0
        lo_pres_pct = ((tot_lo - lo_suppressed) / tot_lo) * 100.0

        candidate_rules_eval.append({
            'rule_label': label,
            'description': desc,
            'total_pred_p': total_pred_p,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'gt_missed': total_fn,
            'td_removed': td_removed,
            'td_rem_pct': round(td_rem_pct, 1),
            'lo_suppressed': lo_suppressed,
            'lo_pres_pct': round(lo_pres_pct, 1),
            'partial_fp_affected': partial_fp_affected,
            'avg_boxes_per_gt': round(avg_boxes_per_gt, 4),
            'r_fn': r_fn
        })

    # Write candidate_rules.csv
    csv_rules_path = output_dir / "candidate_rules.csv"
    with open(csv_rules_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rule_Label", "Description", "Total_Pred_Persons", "Precision", "Recall", "F1",
            "GT_Persons_Missed", "True_Dups_Removed", "True_Dup_Removal_Pct",
            "Legit_Overlaps_Suppressed", "Legit_Worker_Preservation_Pct",
            "Partial_FP_Affected", "Avg_Boxes_Per_GT"
        ])
        for r in candidate_rules_eval:
            writer.writerow([
                r['rule_label'], r['description'], r['total_pred_p'], r['precision'], r['recall'], r['f1'],
                r['gt_missed'], r['td_removed'], r['td_rem_pct'],
                r['lo_suppressed'], r['lo_pres_pct'],
                r['partial_fp_affected'], r['avg_boxes_per_gt']
            ])

    print(f"[SUCCESS] Candidate Rules CSV written to: {csv_rules_path}")

    print("\n==================================================================")
    print("      CANDIDATE SUPPRESSION RULES EVALUATION TABLE (VAL SET)")
    print("==================================================================")
    print(f"{'Rule':<28} | {'PredP':<5} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'TDRem%':<7} | {'LOSupp':<6} | {'LOPres%':<7} | {'GTMissed':<8}")
    print("-" * 105)
    for r in candidate_rules_eval:
        print(f"{r['rule_label']:<28} | {r['total_pred_p']:<5} | {r['precision']:<6.4f} | {r['recall']:<6.4f} | {r['f1']:<6.4f} | {r['td_rem_pct']:<7.1f} | {r['lo_suppressed']:<6} | {r['lo_pres_pct']:<7.1f} | {r['gt_missed']:<8}")

    # -------------------------------------------------------------------------
    # STEP 7, 8, 9 & 10: MULTI-VIDEO EVALUATION WITH CURRENT CUSTOM TRACKER
    # -------------------------------------------------------------------------
    target_videos = [
        "4048038451-preview.mp4",
        "8482302-hd_1920_1080_25fps.mp4",
        "4017518657-preview.mp4",
        "19832490-hd_1920_1080_25fps.mp4",
        "no safety.mp4",
        "helmet+mask+gloves.mp4"
    ]

    selected_v_paths = [os.path.join(vid_dir, v) for v in target_videos if os.path.exists(os.path.join(vid_dir, v))]

    print(f"\n[INFO] Evaluating top suppression rules vs Baseline on {len(selected_v_paths)} videos using Current Custom Tracker...")

    video_eval_results = []

    # Select rules for video evaluation: Baseline, Rule 1, Rule 2, Rule 4
    rules_for_video = [
        ('BASELINE', None),
        ('RULE_1 (Ultra Conservative)', rule_1),
        ('RULE_2 (High Safety)', rule_2),
        ('RULE_4 (Conf-Gated)', rule_4)
    ]

    for v_path in selected_v_paths:
        v_name = os.path.basename(v_path)
        cap = cv2.VideoCapture(v_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break
            frames.append(frame)
        cap.release()

        n_frames = len(frames)
        w, h = frames[0].shape[1], frames[0].shape[0]
        fps_vid = 30.0

        for r_label, r_fn in rules_for_video:
            engine = TemporalConfirmationEngine(TemporalConfirmationConfig(
                confirmation_frames=5, min_track_iou=0.30, max_missed_frames=10, fps=fps_vid
            ))
            associator_vid = PPEAssociator(PPEAssociationConfig(
                person_conf=person_conf, helmet_conf=0.25, mask_conf=0.20
            ))

            t0 = time.perf_counter()
            t_suppress = 0.0

            raw_p_dets_tot = 0
            raw_dup_frames = 0
            raw_dup_pairs_tot = 0

            track_dup_frames = 0
            track_hist = {}
            status_hist = {}
            confirmed_events = 0
            act_tracks_list = []

            for idx, frame in enumerate(frames):
                res = model.predict(frame, imgsz=800, conf=0.05, verbose=False)[0]
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

                # Apply suppression if specified
                t_s = time.perf_counter()
                if r_fn is None:
                    kept_person_boxes = raw_person_boxes
                else:
                    kept_person_boxes = apply_suppression_rule(raw_person_boxes, r_fn, w, h)
                t_suppress += (time.perf_counter() - t_s)

                raw_p_dets_tot += len(kept_person_boxes)

                # Check raw duplicate person boxes in kept detections
                d_cnt = 0
                for i in range(len(kept_person_boxes)):
                    for j in range(i + 1, len(kept_person_boxes)):
                        if box_iou(kept_person_boxes[i]['box'], kept_person_boxes[j]['box']) >= 0.50:
                            d_cnt += 1
                if d_cnt > 0:
                    raw_dup_frames += 1
                    raw_dup_pairs_tot += d_cnt

                # Combine with non-person boxes for PPE association
                combined = kept_person_boxes + non_person_boxes

                # PPE Association
                p_states = associator_vid.process_detections(combined, w, h)

                # Current Custom Tracker
                evts = engine.process_frame(p_states, frame_index=idx, timestamp_sec=idx / fps_vid)
                confirmed_events += len(evts)

                active_trks = engine.active_tracks
                act_tracks_list.append(len(active_trks))

                tr_dup_cnt = 0
                for i in range(len(active_trks)):
                    for j in range(i + 1, len(active_trks)):
                        if box_iou(list(active_trks[i].last_bbox), list(active_trks[j].last_bbox)) >= 0.50:
                            tr_dup_cnt += 1
                if tr_dup_cnt > 0:
                    track_dup_frames += 1

                for trk in active_trks:
                    tid = trk.track_id
                    if tid not in track_hist:
                        track_hist[tid] = []
                        status_hist[tid] = []
                    track_hist[tid].append(idx)

                    st_stat = "UNKNOWN"
                    for st in p_states:
                        if box_iou(list(trk.last_bbox), list(st.person_bbox)) >= 0.50:
                            st_stat = st.safety_status
                            break
                    status_hist[tid].append(st_stat)

            t_total = time.perf_counter() - t0
            fps_measured = n_frames / max(1e-6, t_total)
            latency_ms = (t_total / max(1, n_frames)) * 1000.0
            suppress_overhead_ms = (t_suppress / max(1, n_frames)) * 1000.0

            u_ids = len(track_hist)
            avg_act = float(np.mean(act_tracks_list)) if act_tracks_list else 0.0
            max_act = int(np.max(act_tracks_list)) if act_tracks_list else 0

            frags = 0
            for tid, flst in track_hist.items():
                fs = sorted(flst)
                gaps = [fs[i+1] - fs[i] for i in range(len(fs)-1)]
                frags += sum(1 for g in gaps if g > 1)

            switches = 0
            for tid, slst in status_hist.items():
                for i in range(len(slst) - 1):
                    if slst[i] != slst[i+1] and slst[i] != 'UNKNOWN' and slst[i+1] != 'UNKNOWN':
                        switches += 1

            video_eval_results.append({
                'video': v_name,
                'rule_label': r_label,
                'frames': n_frames,
                'raw_p_dets_tot': raw_p_dets_tot,
                'raw_dup_frames': raw_dup_frames,
                'raw_dup_pairs_tot': raw_dup_pairs_tot,
                'u_ids': u_ids,
                'avg_act': round(avg_act, 2),
                'max_act': max_act,
                'track_dup_frames': track_dup_frames,
                'frags': frags,
                'switches': switches,
                'confirmed_events': confirmed_events,
                'suppress_overhead_ms': round(suppress_overhead_ms, 4),
                'latency_ms': round(latency_ms, 2),
                'fps': round(fps_measured, 1)
            })

            # Save sample visualization frame
            if r_label == 'RULE_2 (High Safety)':
                k_idx = min(30, n_frames - 1)
                annotated = frames[k_idx].copy()
                for trk in engine.active_tracks:
                    bx1, by1, bx2, by2 = [int(v) for v in trk.last_bbox]
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                    cv2.putText(annotated, f"Track #{trk.track_id}", (bx1, max(20, by1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imwrite(str(viz_dir / f"{v_name}_rule2_suppressed.jpg"), annotated)

    # Write video_results.csv
    csv_vid_path = output_dir / "video_results.csv"
    with open(csv_vid_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Video", "Rule_Label", "Frames", "Raw_Person_Detections",
            "Raw_Duplicate_Frames", "Raw_Duplicate_Pairs",
            "Unique_Track_IDs", "Avg_Active_Tracks", "Max_Active_Tracks",
            "Duplicate_Track_Frames", "Track_Fragmentations",
            "Status_Switches", "Confirmed_Events",
            "Suppression_Overhead_MS", "Latency_MS", "Pipeline_FPS"
        ])
        for r in video_eval_results:
            writer.writerow([
                r['video'], r['rule_label'], r['frames'], r['raw_p_dets_tot'],
                r['raw_dup_frames'], r['raw_dup_pairs_tot'],
                r['u_ids'], r['avg_act'], r['max_act'],
                r['track_dup_frames'], r['frags'],
                r['switches'], r['confirmed_events'],
                r['suppress_overhead_ms'], r['latency_ms'], r['fps']
            ])

    print(f"\n[SUCCESS] Multi-Video Results CSV written to: {csv_vid_path}")

    # Print Multi-Video Summary Table
    print("\n==================================================================")
    print("      MULTI-VIDEO SUPPRESSION EXPERIMENT RESULTS SUMMARY")
    print("==================================================================")
    print(f"{'Video Name':<30} | {'Rule':<26} | {'RawDupF':<7} | {'TrkDupF':<7} | {'IDs':<4} | {'Switches':<8} | {'SuppOverhead':<12}")
    print("-" * 110)
    for r in video_eval_results:
        print(f"{r['video']:<30} | {r['rule_label']:<26} | {r['raw_dup_frames']:<7} | {r['track_dup_frames']:<7} | {r['u_ids']:<4} | {r['switches']:<8} | {r['suppress_overhead_ms']:<12.4f} ms")

    print("\n[SUCCESS] Experiment completed cleanly.")

if __name__ == '__main__':
    main()
