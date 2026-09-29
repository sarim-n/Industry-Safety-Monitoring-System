"""
Multi-Signal Person Duplicate Analysis & Experiment Script (v2)
================================================================
Performs comprehensive multi-signal feature extraction, ground-truth labeling,
distribution analysis, visual contact sheet generation, deterministic rule grid search,
validation detection metrics calculation, and video tracking interaction testing.

ISOLATED DIAGNOSTIC EXPERIMENT ONLY. DO NOT MUTATE PRODUCTION CODE.
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

def box_ioa(box_inner, box_outer):
    x1 = max(box_inner[0], box_outer[0])
    y1 = max(box_inner[1], box_outer[1])
    x2 = min(box_inner[2], box_outer[2])
    y2 = min(box_inner[3], box_outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_inner = max(0.0, box_inner[2] - box_inner[0]) * max(0.0, box_inner[3] - box_inner[1])
    return inter / area_inner if area_inner > 0 else 0.0

def compute_pair_features(b1, b2, conf1, conf2, img_w, img_h):
    # Ensure b1 is higher confidence, b2 is lower
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

    img_diag = math.sqrt(img_w**2 + img_h**2)

    norm_dist_diag_large = center_dist / max(1e-6, diag_large)
    norm_dist_diag_small = center_dist / max(1e-6, diag_small)
    norm_dist_img_w = dx / max(1.0, float(img_w))
    norm_dist_img_h = dy / max(1.0, float(img_h))
    norm_dist_img_diag = center_dist / max(1.0, img_diag)
    norm_dx_large_w = dx / max(1e-6, w_large)
    norm_dy_large_h = dy / max(1e-6, h_large)

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
        'box_a': box_a,
        'conf_a': conf_a,
        'box_b': box_b,
        'conf_b': conf_b,
        'iou': iou,
        'w_a': w_a,
        'h_a': h_a,
        'area_a': area_a,
        'cx_a': cx_a,
        'cy_a': cy_a,
        'w_b': w_b,
        'h_b': h_b,
        'area_b': area_b,
        'cx_b': cx_b,
        'cy_b': cy_b,
        'center_dist': center_dist,
        'norm_dist_diag_large': norm_dist_diag_large,
        'norm_dist_diag_small': norm_dist_diag_small,
        'norm_dist_img_w': norm_dist_img_w,
        'norm_dist_img_h': norm_dist_img_h,
        'norm_dist_img_diag': norm_dist_img_diag,
        'norm_dx_large_w': norm_dx_large_w,
        'norm_dy_large_h': norm_dy_large_h,
        'ioa_a_in_b': ioa_a_in_b,
        'ioa_b_in_a': ioa_b_in_a,
        'max_containment': max_containment,
        'min_containment': min_containment,
        'area_ratio': area_ratio,
        'width_ratio': width_ratio,
        'height_ratio': height_ratio,
        'ar_a': ar_a,
        'ar_b': ar_b,
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

def main():
    output_dir = Path(os.path.join(PROJECT_ROOT, "reports", "person_duplicate_analysis_v2"))
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    model_path = os.path.join(PROJECT_ROOT, "runs", "detect", "safety_v1-4_run2b_yolov8s_800", "weights", "best.pt")
    val_img_dir = os.path.join(PROJECT_ROOT, "training_dataset_v2", "images", "val")
    val_lbl_dir = os.path.join(PROJECT_ROOT, "training_dataset_v2", "labels", "val")
    video_path = os.path.join(PROJECT_ROOT, "data_collection", "videos", "4048038451-preview.mp4")

    print("==================================================================")
    print("   MULTI-SIGNAL PERSON DUPLICATE ANALYSIS (PHASE 7.6 / V2)")
    print("==================================================================")

    model = YOLO(model_path)
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.*")))
    person_conf = 0.50

    # PART 1: Run YOLO on 136 validation images and build candidate pairs dataset
    print(f"[INFO] Running YOLOv8s@800 on {len(val_images)} validation images...")
    cached_predictions = []
    total_gt_persons = 0
    total_pred_persons = 0

    candidate_dataset = []

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

        cached_predictions.append({
            'img_path': img_path,
            'img_name': img_name,
            'img_w': w,
            'img_h': h,
            'gt_persons': gt_persons,
            'raw_person_boxes': raw_person_boxes
        })

        # Extract Candidate Pairs (IoU >= 0.50)
        n = len(raw_person_boxes)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = raw_person_boxes[i], raw_person_boxes[j]
                iou = box_iou(p1['box'], p2['box'])
                if iou >= 0.50:
                    feats = compute_pair_features(p1['box'], p2['box'], p1['conf'], p2['conf'], w, h)

                    # Match p1 (box_a) to GT
                    match_a, best_iou_a = -1, 0.0
                    for g_idx, gp in enumerate(gt_persons):
                        giou = box_iou(feats['box_a'], gp)
                        if giou > best_iou_a:
                            best_iou_a = giou
                            match_a = g_idx
                    if best_iou_a < 0.50: match_a = -1

                    # Match p2 (box_b) to GT
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

                    record = {
                        'img_name': img_name,
                        'img_path': img_path,
                        'category': category,
                        'match_a': match_a,
                        'match_b': match_b,
                        'img_w': w,
                        'img_h': h,
                        'gt_persons': gt_persons,
                        **feats
                    }
                    candidate_dataset.append(record)

    # Print Verified Baseline Summary
    true_dups_list = [c for c in candidate_dataset if c['category'] == "TRUE_DUPLICATE"]
    legit_overlaps_list = [c for c in candidate_dataset if c['category'] == "LEGITIMATE_OVERLAP"]
    partial_fps_list = [c for c in candidate_dataset if c['category'] == "PARTIAL_FP"]
    fps_list = [c for c in candidate_dataset if c['category'] == "FALSE_POSITIVE"]

    print("\n==================================================================")
    print("   PART 1 & 3: VERIFIED BASELINE & CANDIDATE PAIR DATASET")
    print("==================================================================")
    print(f"Total Validation Images          : {len(val_images)}")
    print(f"Total Ground-Truth Persons       : {total_gt_persons}")
    print(f"Total Predicted Persons          : {total_pred_persons}")
    print(f"Total Candidate Duplicate Pairs  : {len(candidate_dataset)} (IoU >= 0.50)")
    print(f"  - TRUE_DUPLICATE Pairs         : {len(true_dups_list)} ({len(true_dups_list)/max(1, len(candidate_dataset))*100:.1f}%)")
    print(f"  - LEGITIMATE_OVERLAP Pairs     : {len(legit_overlaps_list)} ({len(legit_overlaps_list)/max(1, len(candidate_dataset))*100:.1f}%)")
    print(f"  - PARTIAL_FP Pairs             : {len(partial_fps_list)} ({len(partial_fps_list)/max(1, len(candidate_dataset))*100:.1f}%)")
    print(f"  - FALSE_POSITIVE Pairs         : {len(fps_list)} ({len(fps_list)/max(1, len(candidate_dataset))*100:.1f}%)")

    # PART 4: FEATURE ANALYSIS & DISTRIBUTION COMPARISON
    features_to_compare = [
        'iou', 'center_dist', 'norm_dist_diag_large', 'norm_dist_diag_small',
        'norm_dist_img_w', 'norm_dist_img_h', 'norm_dist_img_diag',
        'norm_dx_large_w', 'norm_dy_large_h', 'max_containment', 'min_containment',
        'area_ratio', 'width_ratio', 'height_ratio', 'ar_diff',
        'horiz_overlap_ratio', 'vert_overlap_ratio', 'conf_diff', 'conf_a', 'conf_b'
    ]

    csv_feature_path = output_dir / "feature_analysis.csv"
    with open(csv_feature_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Feature",
            "TD_Min", "TD_P25", "TD_Median", "TD_Mean", "TD_P75", "TD_Max",
            "LO_Min", "LO_P25", "LO_Median", "LO_Mean", "LO_P75", "LO_Max"
        ])

        for feat in features_to_compare:
            td_vals = [c[feat] for c in true_dups_list]
            lo_vals = [c[feat] for c in legit_overlaps_list]
            td_s = get_stats(td_vals)
            lo_s = get_stats(lo_vals)

            writer.writerow([
                feat,
                round(td_s['min'], 4), round(td_s['p25'], 4), round(td_s['median'], 4), round(td_s['mean'], 4), round(td_s['p75'], 4), round(td_s['max'], 4),
                round(lo_s['min'], 4), round(lo_s['p25'], 4), round(lo_s['median'], 4), round(lo_s['mean'], 4), round(lo_s['p75'], 4), round(lo_s['max'], 4)
            ])

    print(f"\n[SUCCESS] Feature Analysis CSV written to: {csv_feature_path}")

    # Write Candidate Pairs Dataset CSV
    csv_cand_path = output_dir / "candidate_pairs_v2.csv"
    if candidate_dataset:
        keys = list(candidate_dataset[0].keys())
        keys.remove('gt_persons')
        keys.remove('img_path')
        with open(csv_cand_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in candidate_dataset:
                r_copy = {k: (round(v, 4) if isinstance(v, float) else (str([round(x,1) for x in v]) if isinstance(v, list) else v)) for k, v in row.items() if k in keys}
                writer.writerow(r_copy)

    # PART 5: VISUAL INSPECTION (10 True Duplicates, 10 Legitimate Overlaps Contact Sheets)
    print("\n[INFO] Generating visual contact sheets for 10 TRUE_DUPLICATE and 10 LEGITIMATE_OVERLAP pairs...")
    td_sorted = sorted(true_dups_list, key=lambda x: x['iou'], reverse=True)
    lo_sorted = sorted(legit_overlaps_list, key=lambda x: x['iou'], reverse=True)

    selected_td = td_sorted[:10] if len(td_sorted) >= 10 else td_sorted
    selected_lo = lo_sorted[:10] if len(lo_sorted) >= 10 else lo_sorted

    def render_candidate_crop(c, label_prefix):
        img = cv2.imread(c['img_path'])
        if img is None: return None
        box_a, box_b = [int(v) for v in c['box_a']], [int(v) for v in c['box_b']]

        # Draw GT in Green
        for gp in c['gt_persons']:
            cv2.rectangle(img, (int(gp[0]), int(gp[1])), (int(gp[2]), int(gp[3])), (0, 255, 0), 2)

        # Draw Box A in Red, Box B in Cyan
        cv2.rectangle(img, (box_a[0], box_a[1]), (box_a[2], box_a[3]), (0, 0, 255), 3)
        cv2.rectangle(img, (box_b[0], box_b[1]), (box_b[2], box_b[3]), (255, 255, 0), 2)

        info_str1 = f"{label_prefix} | IoU={c['iou']:.2f} | ConfA={c['conf_a']:.2f} ConfB={c['conf_b']:.2f}"
        info_str2 = f"NDistDiag={c['norm_dist_diag_large']:.2f} | MaxCont={c['max_containment']:.2f} | AreaR={c['area_ratio']:.2f}"
        cv2.putText(img, info_str1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, info_str2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Crop around bounding union with padding
        min_x = max(0, min(box_a[0], box_b[0]) - 40)
        min_y = max(0, min(box_a[1], box_b[1]) - 40)
        max_x = min(c['img_w'], max(box_a[2], box_b[2]) + 40)
        max_y = min(c['img_h'], max(box_a[3], box_b[3]) + 40)

        crop = img[min_y:max_y, min_x:max_x]
        return cv2.resize(crop, (400, 300))

    # Save contact sheets
    if selected_td:
        td_crops = [render_candidate_crop(c, "TRUE_DUP") for c in selected_td if render_candidate_crop(c, "TRUE_DUP") is not None]
        if td_crops:
            cols = 5
            rows = math.ceil(len(td_crops) / cols)
            sheet_td = np.zeros((rows * 300, cols * 400, 3), dtype=np.uint8)
            for idx, crp in enumerate(td_crops):
                r, col = idx // cols, idx % cols
                sheet_td[r*300:(r+1)*300, col*400:(col+1)*400] = crp
            cv2.imwrite(str(contact_dir / "true_duplicates_contact_sheet.jpg"), sheet_td)

    if selected_lo:
        lo_crops = [render_candidate_crop(c, "LEGIT_OVERLAP") for c in selected_lo if render_candidate_crop(c, "LEGIT_OVERLAP") is not None]
        if lo_crops:
            cols = 5
            rows = math.ceil(len(lo_crops) / cols)
            sheet_lo = np.zeros((rows * 300, cols * 400, 3), dtype=np.uint8)
            for idx, crp in enumerate(lo_crops):
                r, col = idx // cols, idx % cols
                sheet_lo[r*300:(r+1)*300, col*400:(col+1)*400] = crp
            cv2.imwrite(str(contact_dir / "legitimate_overlaps_contact_sheet.jpg"), sheet_lo)

    # PART 6 & 7: GRID-SEARCH LIGHTWEIGHT DETERMINISTIC RULES
    print("\n[INFO] Grid-searching multi-signal deterministic rules...")

    # Define Candidate Rule Evaluator
    # A rule returns True if pair (box_a, box_b) is considered a DUPLICATE (lower conf box suppressed)
    def evaluate_rule(rule_fn):
        td_removed = 0
        lo_suppressed = 0
        for c in candidate_dataset:
            is_dup = rule_fn(c)
            if is_dup:
                if c['category'] == "TRUE_DUPLICATE":
                    td_removed += 1
                elif c['category'] == "LEGITIMATE_OVERLAP":
                    lo_suppressed += 1
        return td_removed, lo_suppressed

    grid_results = []

    iou_thresholds = [0.50, 0.55, 0.60, 0.65]
    ndist_thresholds = [0.15, 0.20, 0.25, 0.30]
    containment_thresholds = [0.70, 0.75, 0.80, 0.85]
    area_ratio_thresholds = [0.40, 0.50, 0.60]

    rule_id = 0
    total_td = max(1, len(true_dups_list))
    total_lo = max(1, len(legit_overlaps_list))

    for t_iou in iou_thresholds:
        for t_ndist in ndist_thresholds:
            for t_cont in containment_thresholds:
                for t_aratio in area_ratio_thresholds:
                    rule_id += 1
                    # Multi-signal rule: IoU >= t_iou AND (norm_dist_diag_large <= t_ndist OR max_containment >= t_cont) AND area_ratio >= t_aratio
                    def rule(c, ti=t_iou, tn=t_ndist, tc=t_cont, ta=t_aratio):
                        return (c['iou'] >= ti) and (c['norm_dist_diag_large'] <= tn or c['max_containment'] >= tc) and (c['area_ratio'] >= ta)

                    td_rem, lo_supp = evaluate_rule(rule)
                    td_rem_rate = td_rem / total_td
                    lo_pres_rate = (total_lo - lo_supp) / total_lo
                    lo_supp_rate = lo_supp / total_lo

                    grid_results.append({
                        'rule_id': rule_id,
                        't_iou': t_iou,
                        't_ndist': t_ndist,
                        't_cont': t_cont,
                        't_aratio': t_aratio,
                        'rule_str': f"IoU>={t_iou:.2f} & (NDist<={t_ndist:.2f} | MaxCont>={t_cont:.2f}) & AreaR>={t_aratio:.2f}",
                        'td_removed': td_rem,
                        'td_remaining': total_td - td_rem,
                        'lo_suppressed': lo_supp,
                        'lo_retained': total_lo - lo_supp,
                        'td_rem_rate': round(td_rem_rate, 4),
                        'lo_pres_rate': round(lo_pres_rate, 4),
                        'lo_supp_rate': round(lo_supp_rate, 4)
                    })

    # Sort rules by minimal legitimate overlap suppression, then highest true duplicate removal
    grid_results.sort(key=lambda x: (x['lo_suppressed'], -x['td_removed']))

    print(f"[INFO] Total candidate rules evaluated: {len(grid_results)}")
    print("\n[TOP 5 SAFEST MULTI-SIGNAL RULES (MINIMAL LEGIT WORKER SUPPRESSION)]")
    print(f"{'Rule Description':<65} | {'TDRem':<6} | {'LOSupp':<6} | {'LOPres%':<7} | {'TDRem%':<7}")
    print("-" * 100)
    for r in grid_results[:5]:
        print(f"{r['rule_str']:<65} | {r['td_removed']:<6} | {r['lo_suppressed']:<6} | {r['lo_pres_rate']*100:<7.1f} | {r['td_rem_rate']*100:<7.1f}")

    # PART 8 & 9: FULL VALIDATION SET PERSON DETECTION METRICS FOR TOP CANDIDATE RULES
    print("\n[INFO] Evaluating full validation set detection metrics for select candidate rules...")

    # Let's select 3 specific promising rules for detailed evaluation:
    # Rule A (High Safety): IoU >= 0.55 & NDist <= 0.15 & MaxCont >= 0.85 & AreaR >= 0.50
    # Rule B (Balanced): IoU >= 0.50 & NDist <= 0.20 & MaxCont >= 0.80 & AreaR >= 0.40
    # Rule C (Aggressive): IoU >= 0.50 & (NDist <= 0.25 | MaxCont >= 0.75) & AreaR >= 0.40

    def apply_rule_custom(raw_boxes, rule_type, img_w, img_h):
        sorted_boxes = sorted(raw_boxes, key=lambda x: x['conf'], reverse=True)
        kept = []
        for b in sorted_boxes:
            suppress = False
            for k in kept:
                feats = compute_pair_features(b['box'], k['box'], b['conf'], k['conf'], img_w, img_h)
                if rule_type == 'RULE_A':
                    is_dup = (feats['iou'] >= 0.55) and (feats['norm_dist_diag_large'] <= 0.15) and (feats['max_containment'] >= 0.85) and (feats['area_ratio'] >= 0.50)
                elif rule_type == 'RULE_B':
                    is_dup = (feats['iou'] >= 0.50) and (feats['norm_dist_diag_large'] <= 0.20) and (feats['max_containment'] >= 0.80) and (feats['area_ratio'] >= 0.40)
                elif rule_type == 'RULE_C':
                    is_dup = (feats['iou'] >= 0.50) and (feats['norm_dist_diag_large'] <= 0.25 or feats['max_containment'] >= 0.75) and (feats['area_ratio'] >= 0.40)
                else:
                    is_dup = False
                if is_dup:
                    suppress = True
                    break
            if not suppress:
                kept.append(b)
        return kept

    rules_to_eval = [
        ('BASELINE', None),
        ('RULE_A (High Safety)', 'RULE_A'),
        ('RULE_B (Balanced)', 'RULE_B'),
        ('RULE_C (Aggressive)', 'RULE_C')
    ]

    val_rule_metrics = []

    for label, r_code in rules_to_eval:
        total_pred_p = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        gt_matched_counts = []

        for item in cached_predictions:
            raw_boxes = item['raw_person_boxes']
            gt_boxes = item['gt_persons']
            w, h = item['img_w'], item['img_h']

            if r_code is None:
                kept_boxes = raw_boxes
            else:
                kept_boxes = apply_rule_custom(raw_boxes, r_code, w, h)

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

        precision = total_tp / max(1, (total_tp + total_fp))
        recall = total_tp / max(1, (total_tp + total_fn))
        f1 = (2 * precision * recall) / max(1e-6, (precision + recall))
        avg_boxes_per_gt = sum(gt_matched_counts) / max(1, total_gt_persons)

        val_rule_metrics.append({
            'label': label,
            'total_pred_p': total_pred_p,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'gt_missed': total_fn,
            'avg_boxes_per_gt': avg_boxes_per_gt,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        })

    print("\n[VALIDATION METRICS FOR CANDIDATE MULTI-SIGNAL RULES]")
    print(f"{'Setting':<22} | {'PredP':<5} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'GTMissed':<8} | {'AvgB/GT':<7}")
    print("-" * 80)
    for m in val_rule_metrics:
        print(f"{m['label']:<22} | {m['total_pred_p']:<5} | {m['precision']:<6.4f} | {m['recall']:<6.4f} | {m['f1']:<6.4f} | {m['gt_missed']:<8} | {m['avg_boxes_per_gt']:<7.4f}")

    # PART 10: TRACKING INTERACTION EVALUATION ON LIVE VIDEO (4048038451-preview.mp4)
    print("\n[INFO] Evaluating candidate rules on video sequence (301 frames)...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None: break
        frames.append(frame)
    cap.release()

    frame_w = int(frames[0].shape[1])
    frame_h = int(frames[0].shape[0])
    fps_video = 30.0

    # Cache raw video YOLO predictions
    cached_vid_dets = []
    for frame in frames:
        res = model.predict(frame, imgsz=800, conf=0.05, verbose=False)[0]
        raw_dets = []
        for box in res.boxes:
            c = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            raw_dets.append({'cls': c, 'conf': conf, 'box': xyxy})
        cached_vid_dets.append(raw_dets)

    vid_rule_metrics = []

    for label, r_code in rules_to_eval:
        associator = PPEAssociator(PPEAssociationConfig(person_conf=person_conf, helmet_conf=0.25, mask_conf=0.20))
        temp_engine = TemporalConfirmationEngine(TemporalConfirmationConfig(
            confirmation_frames=5, min_track_iou=0.30, max_missed_frames=10, fps=fps_video, alert_cooldown_seconds=5.0, uncertain_breaks_streak=True
        ))

        frames_with_dup_p = 0
        total_active_tracks_sum = 0
        unique_track_ids = set()
        frames_with_dup_tracks = 0

        for idx, (frame, raw_dets) in enumerate(zip(frames, cached_vid_dets)):
            person_boxes = [d for d in raw_dets if d['cls'] == 2 and d['conf'] >= person_conf]
            non_person_boxes = [d for d in raw_dets if d['cls'] != 2 or d['conf'] < person_conf]

            if r_code is None:
                kept_persons = person_boxes
            else:
                kept_struct = apply_rule_custom(
                    [{'box': d['box'], 'conf': d['conf'], 'raw': d} for d in person_boxes],
                    r_code, frame_w, frame_h
                )
                kept_persons = [item['raw'] for item in kept_struct]

            # Count duplicate person boxes
            dcount = 0
            for i in range(len(kept_persons)):
                for j in range(i + 1, len(kept_persons)):
                    if box_iou(kept_persons[i]['box'], kept_persons[j]['box']) >= 0.50:
                        dcount += 1
            if dcount > 0: frames_with_dup_p += 1

            combined = kept_persons + non_person_boxes
            person_states = associator.process_detections(combined, frame_w, frame_h)
            temp_engine.process_frame(person_states, frame_index=idx, timestamp_sec=idx / fps_video)

            active_tracks = temp_engine.active_tracks
            total_active_tracks_sum += len(active_tracks)
            for trk in active_tracks: unique_track_ids.add(trk.track_id)

            t_dcount = 0
            for i in range(len(active_tracks)):
                for j in range(i + 1, len(active_tracks)):
                    if box_iou(list(active_tracks[i].last_bbox), list(active_tracks[j].last_bbox)) >= 0.50:
                        t_dcount += 1
            if t_dcount > 0: frames_with_dup_tracks += 1

        vid_rule_metrics.append({
            'label': label,
            'dup_p_frames': frames_with_dup_p,
            'avg_active_tracks': round(total_active_tracks_sum / max(1, len(frames)), 2),
            'unique_track_ids': len(unique_track_ids),
            'dup_track_frames': frames_with_dup_tracks
        })

    print("\n[VIDEO TRACKING METRICS FOR CANDIDATE MULTI-SIGNAL RULES]")
    print(f"{'Setting':<22} | {'DupPFrames':<10} | {'AvgTracks':<9} | {'TrackIDs':<8} | {'DupTrackFrames':<14}")
    print("-" * 75)
    for vm in vid_rule_metrics:
        print(f"{vm['label']:<22} | {vm['dup_p_frames']:<10} | {vm['avg_active_tracks']:<9} | {vm['unique_track_ids']:<8} | {vm['dup_track_frames']:<14}")

    # PART 11: COMPUTATIONAL COST / LATENCY BENCHMARKING
    print("\n[INFO] Benchmarking multi-signal rule latency overhead...")
    sample_person_boxes = cached_predictions[0]['raw_person_boxes']
    w_s, h_s = cached_predictions[0]['img_w'], cached_predictions[0]['img_h']

    num_trials = 1000
    t0 = time.time()
    for _ in range(num_trials):
        _ = apply_rule_custom(sample_person_boxes, 'RULE_B', w_s, h_s)
    t_total = time.time() - t0
    added_latency_ms = (t_total / num_trials) * 1000.0

    print(f"Added Latency per Frame for Multi-Signal Rule: {added_latency_ms:.4f} ms ({added_latency_ms*1000:.1f} microseconds)")

    print("\n[SUCCESS] Multi-signal experiment completed cleanly.")

if __name__ == '__main__':
    main()
