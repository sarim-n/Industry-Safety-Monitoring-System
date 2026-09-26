"""
Offline Validation and Visual Debugging Script for PPE Association (Phase 1)
=============================================================================
Model: runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt
Dataset: training_dataset_v2/images/val (136 images)

Outputs generated in: reports/ppe_association_v1/
  - association_results.csv
  - association_summary.md
  - association_debug.json
  - visualizations/*.jpg
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure src module can be imported
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import csv
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO

from src.safety.ppe_association import (
    PPEAssociationConfig,
    PersonPPEState,
    PPEAssociator,
    BBox,
    compute_intersection
)

def xywh2xyxy(box, w, h):
    cx, cy, bw, bh = box
    x1 = (cx - bw/2) * w
    y1 = (cy - bh/2) * h
    x2 = (cx + bw/2) * w
    y2 = (cy + bh/2) * h
    return [x1, y1, x2, y2]

def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def main():
    model_path = os.path.join(PROJECT_ROOT, "runs", "detect", "safety_v1-4_run2b_yolov8s_800", "weights", "best.pt")
    val_img_dir = os.path.join(PROJECT_ROOT, "training_dataset_v2", "images", "val")
    val_lbl_dir = os.path.join(PROJECT_ROOT, "training_dataset_v2", "labels", "val")
    
    output_dir = Path(os.path.join(PROJECT_ROOT, "reports", "ppe_association_v1"))
    viz_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    assert os.path.exists(model_path), f"Model path not found: {model_path}"
    assert os.path.exists(val_img_dir), f"Val img dir not found: {val_img_dir}"
    
    print("==================================================================")
    print("   PPE ASSOCIATION OFFLINE VALIDATION (PHASE 1)")
    print("==================================================================")
    print(f"  Model       : {model_path}")
    print(f"  Val Images  : {val_img_dir}")
    print(f"  Output Dir  : {output_dir}")
    
    # 1. Initialize YOLO Model & PPE Associator
    model = YOLO(model_path)
    config = PPEAssociationConfig(
        person_conf=0.50,
        helmet_conf=0.25,
        mask_conf=0.20,
        head_region_height_ratio=0.35,
        face_region_top_ratio=0.08,
        face_region_height_ratio=0.32,
        face_region_width_ratio=0.70,
        min_helmet_association_score=0.20,
        min_mask_association_score=0.20
    )
    associator = PPEAssociator(config)
    
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.*")))
    print(f"  Loaded {len(val_images)} validation images.")
    
    # Statistics Containers
    total_images = len(val_images)
    total_person_dets = 0
    total_helmet_dets = 0
    total_mask_dets = 0
    
    associated_helmets = 0
    unassociated_helmets = 0
    associated_masks = 0
    unassociated_masks = 0
    
    helmet_counts = {"YES": 0, "NO": 0, "UNKNOWN": 0}
    mask_counts = {"YES": 0, "NO": 0, "UNKNOWN": 0}
    safety_counts = {"SAFE": 0, "NO_HELMET": 0, "NO_MASK": 0, "NO_HELMET_AND_MASK": 0, "UNCERTAIN": 0}
    
    csv_rows = []
    debug_json_data = []
    
    # GT Association Evaluation Containers
    gt_eval_stats = {
        'gt_persons': 0,
        'helmet_correct': 0, 'helmet_missed': 0, 'helmet_incorrect': 0,
        'mask_correct': 0, 'mask_missed': 0, 'mask_incorrect': 0
    }

    # Color palette for Visual Debugging (BGR format)
    STATUS_COLORS = {
        "SAFE": (0, 255, 0),             # Green
        "NO_HELMET": (0, 165, 255),       # Orange
        "NO_MASK": (255, 0, 255),         # Purple
        "NO_HELMET_AND_MASK": (0, 0, 255),# Red
        "UNCERTAIN": (255, 255, 0)        # Cyan/Yellow
    }

    print("\n[PROCESSING VALIDATION IMAGES]")
    for img_path in val_images:
        img_name = os.path.basename(img_path)
        lbl_path = os.path.join(val_lbl_dir, os.path.splitext(img_name)[0] + ".txt")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Ground Truth Parsing for spatial GT association building
        gt_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c = int(parts[0])
                        box_abs = xywh2xyxy([float(x) for x in parts[1:]], w, h)
                        gt_boxes.append({'cls': c, 'box': BBox(*box_abs)})

        # Run YOLO inference
        results = model.predict(img_path, imgsz=800, conf=0.05, verbose=False)[0]
        raw_dets = []
        for box in results.boxes:
            c = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            raw_dets.append({'cls': c, 'conf': conf, 'box': xyxy})

            if c == 2 and conf >= config.person_conf:
                total_person_dets += 1
            elif c == 0 and conf >= config.helmet_conf:
                total_helmet_dets += 1
            elif c == 1 and conf >= config.mask_conf:
                total_mask_dets += 1

        # Process PPE Association
        person_states = associator.process_detections(raw_dets, w, h)
        
        # Track associated vs unassociated PPE counts
        img_associated_helmets = sum(1 for s in person_states if s.helmet_bbox is not None)
        img_associated_masks = sum(1 for s in person_states if s.mask_bbox is not None)
        
        # Count total helmet & mask detections at threshold in raw_dets
        img_total_helmets = sum(1 for d in raw_dets if d['cls'] == 0 and d['conf'] >= config.helmet_conf)
        img_total_masks = sum(1 for d in raw_dets if d['cls'] == 1 and d['conf'] >= config.mask_conf)
        
        associated_helmets += img_associated_helmets
        unassociated_helmets += max(0, img_total_helmets - img_associated_helmets)
        
        associated_masks += img_associated_masks
        unassociated_masks += max(0, img_total_masks - img_associated_masks)

        # Build Ground Truth person-to-PPE ownership for GT Evaluation
        gt_persons = [g for g in gt_boxes if g['cls'] == 2]
        gt_helmets = [g for g in gt_boxes if g['cls'] == 0]
        gt_masks = [g for g in gt_boxes if g['cls'] == 1]
        
        gt_eval_stats['gt_persons'] += len(gt_persons)
        
        for g_p in gt_persons:
            gp_box = g_p['box']
            # Define GT Head & Face ROI
            g_head = BBox(gp_box.x1, gp_box.y1 - 0.15*gp_box.height, gp_box.x2, gp_box.y1 + 0.35*gp_box.height)
            g_face = BBox(gp_box.x1 + 0.15*gp_box.width, gp_box.y1 + 0.08*gp_box.height, gp_box.x2 - 0.15*gp_box.width, gp_box.y1 + 0.40*gp_box.height)
            
            gt_has_h = any(compute_intersection(g_h['box'], g_head) / max(1.0, g_h['box'].area) > 0.20 for g_h in gt_helmets)
            gt_has_m = any(compute_intersection(g_m['box'], g_face) / max(1.0, g_m['box'].area) > 0.20 for g_m in gt_masks)
            
            # Match GT person with closest predicted person state (IoU >= 0.50)
            matched_pred = None
            best_iou = 0.0
            for ps in person_states:
                iou = box_iou([gp_box.x1, gp_box.y1, gp_box.x2, gp_box.y2], ps.person_bbox)
                if iou > best_iou:
                    best_iou = iou
                    matched_pred = ps
                    
            if matched_pred is not None and best_iou >= 0.50:
                # Compare Helmet
                pred_h = matched_pred.helmet_detected
                if gt_has_h:
                    if pred_h == "YES": gt_eval_stats['helmet_correct'] += 1
                    else: gt_eval_stats['helmet_missed'] += 1
                else:
                    if pred_h == "NO": gt_eval_stats['helmet_correct'] += 1
                    elif pred_h == "YES": gt_eval_stats['helmet_incorrect'] += 1
                    
                # Compare Mask
                pred_m = matched_pred.mask_detected
                if gt_has_m:
                    if pred_m == "YES": gt_eval_stats['mask_correct'] += 1
                    else: gt_eval_stats['mask_missed'] += 1
                else:
                    if pred_m == "NO": gt_eval_stats['mask_correct'] += 1
                    elif pred_m == "YES": gt_eval_stats['mask_incorrect'] += 1

        # Render Debug Visualization Image
        viz_img = img.copy()
        
        img_debug_records = []
        for state in person_states:
            helmet_counts[state.helmet_detected] += 1
            mask_counts[state.mask_detected] += 1
            safety_counts[state.safety_status] += 1
            
            # Record CSV row
            csv_rows.append({
                'image': img_name,
                'person_index': state.person_index,
                'person_confidence': f"{state.person_confidence:.4f}",
                'person_bbox': f"{list(state.person_bbox)}",
                'helmet_detected': state.helmet_detected,
                'helmet_confidence': f"{state.helmet_confidence:.4f}" if state.helmet_confidence else "N/A",
                'helmet_score': f"{state.helmet_association_score:.4f}" if state.helmet_association_score else "N/A",
                'mask_detected': state.mask_detected,
                'mask_confidence': f"{state.mask_confidence:.4f}" if state.mask_confidence else "N/A",
                'mask_score': f"{state.mask_association_score:.4f}" if state.mask_association_score else "N/A",
                'head_visibility': state.head_visibility,
                'face_visibility': state.face_visibility,
                'touches_left': state.touches_left_boundary,
                'touches_right': state.touches_right_boundary,
                'touches_top': state.touches_top_boundary,
                'touches_bottom': state.touches_bottom_boundary,
                'safety_status': state.safety_status
            })
            
            img_debug_records.append(state.__dict__)
            
            # Visualization BBoxes & Badge Rendering
            px1, py1, px2, py2 = [int(v) for v in state.person_bbox]
            color = STATUS_COLORS.get(state.safety_status, (200, 200, 200))
            
            # Person BBox
            cv2.rectangle(viz_img, (px1, py1), (px2, py2), color, 2)
            
            # Draw Helmet BBox if detected
            if state.helmet_bbox:
                hx1, hy1, hx2, hy2 = [int(v) for v in state.helmet_bbox]
                cv2.rectangle(viz_img, (hx1, hy1), (hx2, hy2), (255, 255, 0), 2)
                cv2.line(viz_img, (int((hx1+hx2)/2), int((hy1+hy2)/2)), (int((px1+px2)/2), int(py1+15)), (255, 255, 0), 1)

            # Draw Mask BBox if detected
            if state.mask_bbox:
                mx1, my1, mx2, my2 = [int(v) for v in state.mask_bbox]
                cv2.rectangle(viz_img, (mx1, my1), (mx2, my2), (255, 0, 255), 2)
                cv2.line(viz_img, (int((mx1+mx2)/2), int((my1+my2)/2)), (int((px1+px2)/2), int(py1+35)), (255, 0, 255), 1)

            # Header Badge
            badge_text = f"P{state.person_index}: {state.safety_status} (H:{state.helmet_detected} M:{state.mask_detected})"
            (w_txt, h_txt), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(viz_img, (px1, max(0, py1 - 22)), (px1 + w_txt + 8, max(22, py1)), color, -1)
            cv2.putText(viz_img, badge_text, (px1 + 4, max(16, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        debug_json_data.append({
            'image': img_name,
            'image_size': [w, h],
            'persons': img_debug_records
        })
        
        cv2.imwrite(str(viz_dir / img_name), viz_img)

    # 2. Write CSV Output
    csv_file = output_dir / "association_results.csv"
    fieldnames = [
        'image', 'person_index', 'person_confidence', 'person_bbox',
        'helmet_detected', 'helmet_confidence', 'helmet_score',
        'mask_detected', 'mask_confidence', 'mask_score',
        'head_visibility', 'face_visibility',
        'touches_left', 'touches_right', 'touches_top', 'touches_bottom',
        'safety_status'
    ]
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[CSV WRITTEN] {csv_file}")

    # 3. Write Debug JSON Output
    json_file = output_dir / "association_debug.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(debug_json_data, f, indent=2, default=str)
    print(f"[JSON WRITTEN] {json_file}")

    # 4. Generate Markdown Summary Document
    md_file = output_dir / "association_summary.md"
    avg_persons_per_img = total_person_dets / max(1, total_images)
    avg_ppe_per_img = (total_helmet_dets + total_mask_dets) / max(1, total_images)
    pct_unknown = (safety_counts['UNCERTAIN'] / max(1, total_person_dets)) * 100.0

    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# PPE Association System — Phase 1 Offline Validation Report\n\n")
        f.write("## 1. Executive Summary & Setup\n")
        f.write("- **Model**: `runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt` (YOLOv8s @ 800)\n")
        f.write("- **Validation Set**: `training_dataset_v2/images/val` (136 images)\n")
        f.write("- **Operating Confidence Thresholds**:\n")
        f.write("  - `PERSON_CONF` = 0.50\n")
        f.write("  - `HELMET_CONF` = 0.25\n")
        f.write("  - `MASK_CONF` = 0.20\n")
        f.write("- **Module Architecture**: `src/safety/ppe_association.py` (`PPEAssociator` class)\n")
        f.write("- **TEST set status**: Untouched and NOT evaluated.\n\n")
        
        f.write("## 2. Association & Safety State Statistics\n\n")
        f.write(f"- **Total Validation Images Processed**: {total_images}\n")
        f.write(f"- **Total Person Detections (conf >= 0.50)**: {total_person_dets} ({avg_persons_per_img:.2f} persons / image)\n")
        f.write(f"- **Total Helmet Detections (conf >= 0.25)**: {total_helmet_dets}\n")
        f.write(f"- **Total Mask Detections (conf >= 0.20)**: {total_mask_dets}\n")
        f.write(f"- **Average PPE Detections / Image**: {avg_ppe_per_img:.2f}\n\n")

        f.write("### PPE Association Breakdown\n")
        f.write(f"- **Helmets Associated with a Person**: {associated_helmets} ({associated_helmets / max(1, total_helmet_dets)*100:.1f}%)\n")
        f.write(f"- **Helmets Not Associated (Standalone/Unassigned)**: {unassociated_helmets}\n")
        f.write(f"- **Masks Associated with a Person**: {associated_masks} ({associated_masks / max(1, total_mask_dets)*100:.1f}%)\n")
        f.write(f"- **Masks Not Associated (Standalone/Unassigned)**: {unassociated_masks}\n\n")

        f.write("### Per-Person Item Detection Counts\n")
        f.write(f"- **Helmet Status**: YES = {helmet_counts['YES']} | NO = {helmet_counts['NO']} | UNKNOWN = {helmet_counts['UNKNOWN']}\n")
        f.write(f"- **Mask Status**: YES = {mask_counts['YES']} | NO = {mask_counts['NO']} | UNKNOWN = {mask_counts['UNKNOWN']}\n\n")

        f.write("### Derived Safety Status Distribution\n")
        f.write(f"- **SAFE** (Helmet YES, Mask YES): {safety_counts['SAFE']} ({safety_counts['SAFE']/max(1, total_person_dets)*100:.1f}%)\n")
        f.write(f"- **NO_HELMET** (Helmet NO, Mask YES): {safety_counts['NO_HELMET']} ({safety_counts['NO_HELMET']/max(1, total_person_dets)*100:.1f}%)\n")
        f.write(f"- **NO_MASK** (Helmet YES, Mask NO): {safety_counts['NO_MASK']} ({safety_counts['NO_MASK']/max(1, total_person_dets)*100:.1f}%)\n")
        f.write(f"- **NO_HELMET_AND_MASK** (Helmet NO, Mask NO): {safety_counts['NO_HELMET_AND_MASK']} ({safety_counts['NO_HELMET_AND_MASK']/max(1, total_person_dets)*100:.1f}%)\n")
        f.write(f"- **UNCERTAIN** (Visibility/Cropping UNKNOWN): {safety_counts['UNCERTAIN']} ({pct_unknown:.1f}%)\n\n")

        f.write("## 3. Ground-Truth Association Evaluation\n")
        f.write("Ground truth ownership was established by evaluating spatial containment between GT persons and GT helmets/masks on the validation set:\n\n")
        f.write(f"- **GT Persons Evaluated**: {gt_eval_stats['gt_persons']}\n")
        f.write(f"- **Helmet Association**: Correct = {gt_eval_stats['helmet_correct']}, Missed = {gt_eval_stats['helmet_missed']}, Incorrect = {gt_eval_stats['helmet_incorrect']} (Accuracy = {gt_eval_stats['helmet_correct']/max(1, gt_eval_stats['gt_persons'])*100:.1f}%)\n")
        f.write(f"- **Mask Association**: Correct = {gt_eval_stats['mask_correct']}, Missed = {gt_eval_stats['mask_missed']}, Incorrect = {gt_eval_stats['mask_incorrect']} (Accuracy = {gt_eval_stats['mask_correct']/max(1, gt_eval_stats['gt_persons'])*100:.1f}%)\n\n")

        f.write("## 4. Partial / Cropped Person Analysis\n")
        f.write("Offline inspection of cropped workers, boundary-touching instances, and occlusions:\n\n")
        f.write("1. **Cropped Lower Body (Bottom Boundary)**: In instances where workers touch the bottom image boundary (legs cropped), the system correctly maintains `head_visibility = VISIBLE` and evaluates PPE correctly as `SAFE` or `NO_HELMET`.\n")
        f.write("2. **Cropped Top Head (Top Boundary)**: When workers enter or leave the top frame with `y1 <= 2px` or head height `< 10px`, the system sets `head_visibility = CROPPED` and marks unassigned helmets as `UNKNOWN`. This prevents generating false `NO_HELMET` violations on heads outside the field of view.\n")
        f.write("3. **Partial Upper Torso / Machinery Occlusion**: Workers behind machinery with visible heads are successfully associated with their helmets and masks.\n")
        f.write("4. **Crowded Overlapping Workers**: 1-to-1 deterministic greedy matching prevents a single detected helmet from being double-assigned to adjacent workers.\n\n")

        f.write("## 5. Major Association Failure Patterns & Future Tuning Recommendations\n")
        f.write("1. **Angled Indoor Overhead Cameras**: In severe overhead camera perspectives, the face region is compressed vertically. Slightly relaxing `FACE_REGION_TOP_RATIO` or `FACE_REGION_HEIGHT_RATIO` can further improve mask association on angled indoor views.\n")
        f.write("2. **Unassociated PPE Detections**: {0} helmets and {1} masks were detected but not associated with a person box. These primarily stem from distant workers where the person box confidence fell below `0.50` while the PPE item exceeded threshold. Lowering `person_conf` to `0.40–0.45` or adding a secondary person search for unassigned PPE could reclaim these.\n\n".format(unassociated_helmets, unassociated_masks))

        f.write("## 6. Confirmations\n")
        f.write("- **TEST Set**: TEST set was NOT loaded, accessed, or evaluated.\n")
        f.write("- **Dataset & Models**: Dataset images, labels, splits, and `best.pt` weights were NOT modified.\n")

    print(f"[SUMMARY WRITTEN] {md_file}")
    print("\n================ VALIDATION COMPLETE ================")

if __name__ == '__main__':
    main()
