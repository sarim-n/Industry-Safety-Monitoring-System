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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    PROJECT_ROOT = Path(r"C:\Users\sarim\safety monitoring")
    VAL_IMG_DIR = PROJECT_ROOT / "training_dataset_v2" / "images" / "val"
    VAL_LBL_DIR = PROJECT_ROOT / "training_dataset_v2" / "labels" / "val"

    RUN2B_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "safety_v1-4_run2b_yolov8s_800" / "weights" / "best.pt"
    RUN3_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "safety_v1-4_run3_mask_hard_negative" / "weights" / "best.pt"

    OUTPUT_DIR = PROJECT_ROOT / "reports" / "mask_fp_analysis_run3"
    VIZ_DIR = OUTPUT_DIR / "visual_comparisons"
    SHEETS_DIR = OUTPUT_DIR / "contact_sheets"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("   PHASE 8, 9, 10 — MASK FP DIAGNOSIS & SENSITIVITY ANALYSIS")
    print("==================================================================")

    model_2b = YOLO(str(RUN2B_WEIGHTS))
    model_3 = YOLO(str(RUN3_WEIGHTS))

    val_img_paths = sorted(list(VAL_IMG_DIR.glob("*.jpg")) + list(VAL_IMG_DIR.glob("*.png")))

    # Load ground truth for all val images
    gt_data = {}
    for img_p in val_img_paths:
        lbl_p = VAL_LBL_DIR / (img_p.stem + ".txt")
        gt_boxes = []
        if lbl_p.exists():
            img = cv2.imread(str(img_p))
            h, w = img.shape[:2]
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = max(0.0, (cx - bw/2.0)*w)
                        y1 = max(0.0, (cy - bh/2.0)*h)
                        x2 = min(float(w), (cx + bw/2.0)*w)
                        y2 = min(float(h), (cy + bh/2.0)*h)
                        gt_boxes.append({'cls': c, 'box': [x1, y1, x2, y2]})
        gt_data[img_p.name] = gt_boxes

    # Function to evaluate model predictions at a specific mask confidence threshold
    def eval_model_at_threshold(model, mask_conf_thresh=0.20):
        tp_mask = 0
        fp_mask = 0
        fp_face_region_mask = 0
        fn_mask = 0
        
        all_mask_confs = []
        face_fp_records = []

        for img_p in val_img_paths:
            fn = img_p.name
            gt_boxes = gt_data[fn]
            img = cv2.imread(str(img_p))
            h, w = img.shape[:2]

            gt_masks = [b['box'] for b in gt_boxes if b['cls'] == 1]
            gt_persons = [b['box'] for b in gt_boxes if b['cls'] == 2]

            # Model prediction
            res = model.predict(img, imgsz=800, conf=0.05, verbose=False)[0]

            pred_masks = []
            for box in res.boxes:
                c = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                if c == 1 and conf >= mask_conf_thresh:
                    pred_masks.append({'box': xyxy, 'conf': conf, 'matched': False})
                    all_mask_confs.append(conf)

            # Match pred masks to GT masks (IoU >= 0.40)
            gt_matched = [False] * len(gt_masks)

            for pm in pred_masks:
                best_iou = 0.0
                best_idx = -1
                for idx, gm in enumerate(gt_masks):
                    if not gt_matched[idx]:
                        iou = box_iou(pm['box'], gm)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx

                if best_iou >= 0.40 and best_idx != -1:
                    pm['matched'] = True
                    gt_matched[best_idx] = True
                    tp_mask += 1
                else:
                    fp_mask += 1
                    # Check if this FP is in a person's upper body / face region
                    in_face_region = False
                    matched_p_box = None
                    for pb in gt_persons:
                        # Upper 50% of person bbox is head/face region
                        face_roi = [pb[0], pb[1], pb[2], pb[1] + 0.50 * (pb[3] - pb[1])]
                        if box_ioa(pm['box'], face_roi) >= 0.30 or box_ioa(pm['box'], pb) >= 0.40:
                            in_face_region = True
                            matched_p_box = pb
                            break

                    if in_face_region:
                        fp_face_region_mask += 1
                        face_fp_records.append({
                            'img_name': fn,
                            'img_path': str(img_p),
                            'mask_box': pm['box'],
                            'mask_conf': pm['conf'],
                            'person_box': matched_p_box
                        })

            fn_mask += sum(1 for m in gt_matched if not m)

        prec = tp_mask / max(1, tp_mask + fp_mask)
        rec = tp_mask / max(1, tp_mask + fn_mask)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)

        return {
            'tp': tp_mask,
            'fp': fp_mask,
            'fp_face_region': fp_face_region_mask,
            'fn': fn_mask,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'all_confs': all_mask_confs,
            'face_fp_records': face_fp_records
        }

    # 1. PHASE 8: Diagnostic Analysis at MASK_CONF = 0.20
    diag_2b = eval_model_at_threshold(model_2b, mask_conf_thresh=0.20)
    diag_3 = eval_model_at_threshold(model_3, mask_conf_thresh=0.20)

    print("\n--- PHASE 8: MASK FP DIAGNOSIS (MASK_CONF = 0.20) ---")
    print(f"Run 2B Baseline : TP={diag_2b['tp']}, FP={diag_2b['fp']}, Face-FP={diag_2b['fp_face_region']}, FN={diag_2b['fn']}, Prec={diag_2b['precision']:.4f}, Rec={diag_2b['recall']:.4f}, F1={diag_2b['f1']:.4f}")
    print(f"Run 3 Experiment: TP={diag_3['tp']}, FP={diag_3['fp']}, Face-FP={diag_3['fp_face_region']}, FN={diag_3['fn']}, Prec={diag_3['precision']:.4f}, Rec={diag_3['recall']:.4f}, F1={diag_3['f1']:.4f}")
    print(f"FP Change       : FP={diag_3['fp']-diag_2b['fp']:+d}, Face-FP={diag_3['fp_face_region']-diag_2b['fp_face_region']:+d}, FN={diag_3['fn']-diag_2b['fn']:+d}")

    # 2. PHASE 9: Mask Confidence Sensitivity Analysis
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    sens_results = []

    print("\n--- PHASE 9: MASK CONFIDENCE SENSITIVITY ANALYSIS ---")
    print(f"{'Thresh':<8} | {'Run 2B Prec':<12} {'Run 2B Rec':<12} {'Run 2B FaceFP':<14} | {'Run 3 Prec':<12} {'Run 3 Rec':<12} {'Run 3 FaceFP':<14}")
    print("-" * 85)

    for th in thresholds:
        r2b = eval_model_at_threshold(model_2b, mask_conf_thresh=th)
        r3 = eval_model_at_threshold(model_3, mask_conf_thresh=th)
        sens_results.append({
            'threshold': th,
            'r2b': r2b,
            'r3': r3
        })
        print(f"{th:<8.2f} | {r2b['precision']:<12.4f} {r2b['recall']:<12.4f} {r2b['fp_face_region']:<14} | {r3['precision']:<12.4f} {r3['recall']:<12.4f} {r3['fp_face_region']:<14}")

    # Save sensitivity CSV
    sens_csv = OUTPUT_DIR / "mask_sensitivity_comparison.csv"
    with open(sens_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "run2b_tp", "run2b_fp", "run2b_face_fp", "run2b_fn", "run2b_prec", "run2b_rec", "run2b_f1",
                         "run3_tp", "run3_fp", "run3_face_fp", "run3_fn", "run3_prec", "run3_rec", "run3_f1"])
        for sr in sens_results:
            th = sr['threshold']
            b = sr['r2b']
            r = sr['r3']
            writer.writerow([th, b['tp'], b['fp'], b['fp_face_region'], b['fn'], f"{b['precision']:.4f}", f"{b['recall']:.4f}", f"{b['f1']:.4f}",
                             r['tp'], r['fp'], r['fp_face_region'], r['fn'], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}"])

    print(f"\nSaved sensitivity comparison to {sens_csv}")

    # 3. PHASE 10: Visual Comparison of Problematic Images
    print("\n--- PHASE 10: VISUAL ERROR COMPARISON ON PROBLEMATIC IMAGES ---")
    fp_imgs_2b = {r['img_name']: r for r in diag_2b['face_fp_records']}
    fp_imgs_3 = {r['img_name']: r for r in diag_3['face_fp_records']}

    all_prob_imgs = sorted(list(set(fp_imgs_2b.keys()).union(set(fp_imgs_3.keys()))))
    print(f"Total validation images with face-region mask FPs across Run 2B and Run 3: {len(all_prob_imgs)}")

    for idx, fn in enumerate(all_prob_imgs[:25], 1):
        img_p = VAL_IMG_DIR / fn
        img = cv2.imread(str(img_p))
        if img is None: continue
        h, w = img.shape[:2]

        # Predict Run 2B & Run 3
        res_2b = model_2b.predict(img, imgsz=800, conf=0.20, verbose=False)[0]
        res_3 = model_3.predict(img, imgsz=800, conf=0.20, verbose=False)[0]

        img_2b = img.copy()
        cv2.putText(img_2b, "Run 2B Baseline (imgsz=800)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for box in res_2b.boxes:
            c = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            b = box.xyxy[0].cpu().numpy().astype(int)
            color = (0, 255, 0) if c == 2 else ((0, 165, 255) if c == 0 else (255, 0, 255))
            c_name = "person" if c == 2 else ("helmet" if c == 0 else "mask")
            cv2.rectangle(img_2b, (b[0], b[1]), (b[2], b[3]), color, 2)
            cv2.putText(img_2b, f"{c_name} {conf:.2f}", (b[0], max(15, b[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        img_3 = img.copy()
        cv2.putText(img_3, "Run 3 Mask Hard-Neg Experiment", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for box in res_3.boxes:
            c = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            b = box.xyxy[0].cpu().numpy().astype(int)
            color = (0, 255, 0) if c == 2 else ((0, 165, 255) if c == 0 else (255, 0, 255))
            c_name = "person" if c == 2 else ("helmet" if c == 0 else "mask")
            cv2.rectangle(img_3, (b[0], b[1]), (b[2], b[3]), color, 2)
            cv2.putText(img_3, f"{c_name} {conf:.2f}", (b[0], max(15, b[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        side_by_side = np.hstack([img_2b, img_3])
        out_viz_path = VIZ_DIR / f"cmp_{idx:02d}_{fn}"
        cv2.imwrite(str(out_viz_path), side_by_side, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    print(f"Saved {len(all_prob_imgs[:25])} visual side-by-side comparison images to {VIZ_DIR}")

if __name__ == '__main__':
    main()
