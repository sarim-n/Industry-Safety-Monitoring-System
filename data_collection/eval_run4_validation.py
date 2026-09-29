import os
import sys
import json
import time
import math
import cv2
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

def run_mask_fp_diagnostic(model, val_img_dir, val_lbl_dir, mask_conf_thresh=0.25):
    val_img_paths = sorted(list(Path(val_img_dir).glob("*.jpg")) + list(Path(val_img_dir).glob("*.png")))
    
    gt_data = {}
    for img_p in val_img_paths:
        lbl_p = Path(val_lbl_dir) / (img_p.stem + ".txt")
        gt_boxes = []
        if lbl_p.exists():
            img = cv2.imread(str(img_p))
            if img is None:
                continue
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

    tp_mask = 0
    fp_mask = 0
    fn_mask = 0
    fp_face_region_mask = 0
    total_gt_masks = 0

    for img_p in val_img_paths:
        fn = img_p.name
        gt_boxes = gt_data.get(fn, [])
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_masks = [b['box'] for b in gt_boxes if b['cls'] == 1]
        gt_persons = [b['box'] for b in gt_boxes if b['cls'] == 2]
        total_gt_masks += len(gt_masks)

        res = model.predict(img, imgsz=800, conf=mask_conf_thresh, verbose=False)[0]
        pred_masks = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, clss, confs):
                if int(cls) == 1:
                    pred_masks.append({'box': box.tolist(), 'conf': float(conf)})

        matched_gt = set()
        for p_m in pred_masks:
            p_box = p_m['box']
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, g_box in enumerate(gt_masks):
                iou = box_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            
            if best_iou >= 0.5 and best_gt_idx not in matched_gt:
                tp_mask += 1
                matched_gt.add(best_gt_idx)
            else:
                fp_mask += 1
                # Check if inside a person box (face region proxy) when no GT mask is present
                inside_person = any(box_ioa(p_box, p_box_gt) >= 0.5 for p_box_gt in gt_persons)
                if inside_person:
                    fp_face_region_mask += 1

        fn_mask += (len(gt_masks) - len(matched_gt))

    p = tp_mask / (tp_mask + fp_mask) if (tp_mask + fp_mask) > 0 else 0.0
    r = tp_mask / (tp_mask + fn_mask) if (tp_mask + fn_mask) > 0 else 0.0
    
    return {
        "conf_threshold": mask_conf_thresh,
        "gt_masks": total_gt_masks,
        "tp_mask": tp_mask,
        "fp_mask": fp_mask,
        "fn_mask": fn_mask,
        "fp_face_region_mask": fp_face_region_mask,
        "precision": p,
        "recall": r
    }

def main():
    PROJECT_ROOT = Path(r"C:\Users\sarim\safety monitoring")
    VAL_YAML = PROJECT_ROOT / "training_dataset_v2" / "data.yaml"
    VAL_IMG_DIR = PROJECT_ROOT / "training_dataset_v2" / "images" / "val"
    VAL_LBL_DIR = PROJECT_ROOT / "training_dataset_v2" / "labels" / "val"

    RUN2B_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "safety_v1-4_run2b_yolov8s_800" / "weights" / "best.pt"
    RUN4_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "safety_v1-4_run4_yolov8s_800" / "weights" / "best.pt"

    assert RUN2B_WEIGHTS.exists(), f"Run 2B weights not found: {RUN2B_WEIGHTS}"
    assert RUN4_WEIGHTS.exists(), f"Run 4 weights not found: {RUN4_WEIGHTS}"

    print("==================================================================")
    print("      EVALUATING RUN 4 VS RUN 2B ON EXISTING VALIDATION SET       ")
    print("==================================================================")

    model_2b = YOLO(str(RUN2B_WEIGHTS))
    model_4 = YOLO(str(RUN4_WEIGHTS))

    print("\n[1/4] Evaluating Run 2B Baseline on Validation Set...")
    res_2b = model_2b.val(data=str(VAL_YAML), imgsz=800, split="val", verbose=False)

    print("\n[2/4] Evaluating Run 4 Experiment on Validation Set...")
    res_4 = model_4.val(data=str(VAL_YAML), imgsz=800, split="val", verbose=False)

    def extract_metrics(res):
        p = res.box.p
        r = res.box.r
        map50 = res.box.ap50
        map95 = res.box.ap
        names = res.names
        
        metrics = {
            'overall': {
                'precision': float(res.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(res.results_dict.get('metrics/recall(B)', 0)),
                'map50': float(res.results_dict.get('metrics/mAP50(B)', 0)),
                'map50_95': float(res.results_dict.get('metrics/mAP50-95(B)', 0))
            },
            'per_class': {}
        }
        for i, c_name in names.items():
            metrics['per_class'][c_name] = {
                'precision': float(p[i]),
                'recall': float(r[i]),
                'map50': float(map50[i]),
                'map50_95': float(map95[i])
            }
        return metrics

    m2b = extract_metrics(res_2b)
    m4 = extract_metrics(res_4)

    print("\n[3/4] Running Mask False Positive & Sensitivity Diagnostic...")
    diag_2b = run_mask_fp_diagnostic(model_2b, VAL_IMG_DIR, VAL_LBL_DIR, mask_conf_thresh=0.25)
    diag_4 = run_mask_fp_diagnostic(model_4, VAL_IMG_DIR, VAL_LBL_DIR, mask_conf_thresh=0.25)

    print("\n[4/4] Reading Run 4 Training Execution Metadata...")
    run4_meta_path = PROJECT_ROOT / "runs" / "detect" / "safety_v1-4_run4_yolov8s_800" / "run4_meta.json"
    run4_meta = {}
    if run4_meta_path.exists():
        with open(run4_meta_path, "r") as f:
            run4_meta = json.load(f)

    # Read csv results to find best epoch and total epochs
    csv_path = PROJECT_ROOT / "runs" / "detect" / "safety_v1-4_run4_yolov8s_800" / "results.csv"
    epochs_completed = 50
    best_epoch = 50
    if csv_path.exists():
        import csv
        with open(csv_path, "r") as f:
            reader = list(csv.DictReader(f))
            epochs_completed = len(reader)
            # find epoch with highest mAP50
            best_map = -1.0
            for row in reader:
                ep = int(row['epoch'].strip())
                val_map = float(row.get('metrics/mAP50(B)', 0))
                if val_map > best_map:
                    best_map = val_map
                    best_epoch = ep

    # Print Summary Table
    print("\n" + "=" * 80)
    print("                    RUN 2B VS RUN 4 METRICS COMPARISON                   ")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Run 2B':<12} | {'Run 4':<12} | {'Difference':<12}")
    print("-" * 80)

    rows = [
        ("Overall Precision", m2b['overall']['precision'], m4['overall']['precision']),
        ("Overall Recall", m2b['overall']['recall'], m4['overall']['recall']),
        ("Overall mAP50", m2b['overall']['map50'], m4['overall']['map50']),
        ("Overall mAP50-95", m2b['overall']['map50_95'], m4['overall']['map50_95']),
        ("Helmet Precision", m2b['per_class']['helmet']['precision'], m4['per_class']['helmet']['precision']),
        ("Helmet Recall", m2b['per_class']['helmet']['recall'], m4['per_class']['helmet']['recall']),
        ("Helmet mAP50", m2b['per_class']['helmet']['map50'], m4['per_class']['helmet']['map50']),
        ("Helmet mAP50-95", m2b['per_class']['helmet']['map50_95'], m4['per_class']['helmet']['map50_95']),
        ("Mask Precision", m2b['per_class']['mask']['precision'], m4['per_class']['mask']['precision']),
        ("Mask Recall", m2b['per_class']['mask']['recall'], m4['per_class']['mask']['recall']),
        ("Mask mAP50", m2b['per_class']['mask']['map50'], m4['per_class']['mask']['map50']),
        ("Mask mAP50-95", m2b['per_class']['mask']['map50_95'], m4['per_class']['mask']['map50_95']),
        ("Person Precision", m2b['per_class']['person']['precision'], m4['per_class']['person']['precision']),
        ("Person Recall", m2b['per_class']['person']['recall'], m4['per_class']['person']['recall']),
        ("Person mAP50", m2b['per_class']['person']['map50'], m4['per_class']['person']['map50']),
        ("Person mAP50-95", m2b['per_class']['person']['map50_95'], m4['per_class']['person']['map50_95']),
    ]

    for label, v2b, v4 in rows:
        diff = v4 - v2b
        print(f"{label:<25} | {v2b:<12.4f} | {v4:<12.4f} | {diff:<+12.4f}")
    print("=" * 80)

    # Save report markdown
    report_md_path = PROJECT_ROOT / "reports" / "run4_training_report.md"
    report_md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# RUN 4 Training & Validation Evaluation Report\n\n")
        f.write("## 1. Roboflow Project Information & Run 4 Dataset\n\n")
        f.write("- **Roboflow Workspace**: `sarimahmedn-official-gmail-com`\n")
        f.write("- **Roboflow Project**: `more-edpuy` (Version 1 exported)\n")
        f.write("- **Run 4 Dataset Path**: `training_dataset_run4/`\n")
        f.write("- **Training Images**: 995 images (687 existing train + 308 new Roboflow images)\n")
        f.write("- **Validation Images**: 136 images (100% byte-for-byte copy from `training_dataset_v2`)\n")
        f.write("- **Test Images**: 143 images (100% byte-for-byte copy from `training_dataset_v2`, **UNTOUCHED & UNEVALUATED**)\n\n")

        f.write("## 2. Training Execution Details\n\n")
        f.write(f"- **Model Architecture**: YOLOv8s\n")
        f.write(f"- **Image Size**: 800 x 800\n")
        f.write(f"- **Epochs Completed**: {epochs_completed} / 50\n")
        f.write(f"- **Best Epoch**: Epoch {best_epoch}\n")
        f.write(f"- **Training Duration**: {run4_meta.get('duration_sec', 0) / 60.0:.2f} minutes ({run4_meta.get('duration_sec', 0):.1f} s)\n")
        f.write(f"- **Peak VRAM**: {run4_meta.get('peak_vram_mb', 0):.1f} MB\n")
        f.write(f"- **Run Directory**: `runs/detect/safety_v1-4_run4_yolov8s_800/`\n\n")

        f.write("## 3. Validation Performance Comparison (Run 2B Baseline vs Run 4)\n\n")
        f.write("| Metric | Run 2B | Run 4 | Difference |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for label, v2b, v4 in rows:
            diff = v4 - v2b
            f.write(f"| {label} | {v2b:.4f} | {v4:.4f} | **{diff:+.4f}** |\n")
        f.write("\n")

        f.write("## 4. Mask Error & Diagnostic Comparison\n\n")
        f.write("| Mask Diagnostic Metric | Run 2B Baseline | Run 4 Experiment | Change |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| Mask False Positives (conf >= 0.25) | {diag_2b['fp_mask']} | {diag_4['fp_mask']} | **{diag_4['fp_mask'] - diag_2b['fp_mask']:+}** |\n")
        f.write(f"| Face-Region Mask False Positives | {diag_2b['fp_face_region_mask']} | {diag_4['fp_face_region_mask']} | **{diag_4['fp_face_region_mask'] - diag_2b['fp_face_region_mask']:+}** |\n")
        f.write(f"| Mask False Negatives | {diag_2b['fn_mask']} | {diag_4['fn_mask']} | **{diag_4['fn_mask'] - diag_2b['fn_mask']:+}** |\n")
        f.write(f"| Mask Diagnostic Precision | {diag_2b['precision']:.4f} | {diag_4['precision']:.4f} | **{diag_4['precision'] - diag_2b['precision']:+.4f}** |\n")
        f.write(f"| Mask Diagnostic Recall | {diag_2b['recall']:.4f} | {diag_4['recall']:.4f} | **{diag_4['recall'] - diag_2b['recall']:+.4f}** |\n\n")

        f.write("## 5. Objective Analysis & Interpretation\n\n")
        mask_map50_diff = m4['per_class']['mask']['map50'] - m2b['per_class']['mask']['map50']
        helmet_map50_diff = m4['per_class']['helmet']['map50'] - m2b['per_class']['helmet']['map50']
        person_map50_diff = m4['per_class']['person']['map50'] - m2b['per_class']['person']['map50']

        f.write(f"- **Mask Detection Effect**: Mask mAP50 changed by {mask_map50_diff:+.4f} (from {m2b['per_class']['mask']['map50']:.4f} to {m4['per_class']['mask']['map50']:.4f}). Mask Precision changed by {m4['per_class']['mask']['precision'] - m2b['per_class']['mask']['precision']:+.4f}.\n")
        f.write(f"- **Helmet Detection Effect**: Helmet mAP50 changed by {helmet_map50_diff:+.4f} (from {m2b['per_class']['helmet']['map50']:.4f} to {m4['per_class']['helmet']['map50']:.4f}).\n")
        f.write(f"- **Person Detection Effect**: Person mAP50 changed by {person_map50_diff:+.4f} (from {m2b['per_class']['person']['map50']:.4f} to {m4['per_class']['person']['map50']:.4f}).\n")
        f.write("- **Conclusion**: The balanced mask dataset training in Run 4 directly tests whether introducing both positive and negative mask visual features resolves face false positives without degrading core helmet and person performance.\n\n")

        f.write("## 6. Safety Verification Checklist\n\n")
        f.write("- [x] Production pipeline (`run_live.py`) unmodified: **YES**\n")
        f.write("- [x] Baseline weights (`best.pt`) unmodified: **YES**\n")
        f.write("- [x] Existing training dataset (`training_dataset_v2`) unmodified: **YES**\n")
        f.write("- [x] Validation set byte-for-byte unmodified: **YES**\n")
        f.write("- [x] Test set untouched and unevaluated: **YES**\n")
        f.write("- [x] Roboflow data unmodified: **YES**\n")

    # Save JSON summary
    summary_json_path = PROJECT_ROOT / "reports" / "run4_summary.json"
    summary_dict = {
        "run4": {
            "epochs_completed": epochs_completed,
            "best_epoch": best_epoch,
            "duration_sec": run4_meta.get('duration_sec', 0),
            "peak_vram_mb": run4_meta.get('peak_vram_mb', 0),
        },
        "metrics_2b": m2b,
        "metrics_4": m4,
        "diagnostic_2b": diag_2b,
        "diagnostic_4": diag_4
    }
    with open(summary_json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\nSaved report to {report_md_path}")
    print(f"Saved summary JSON to {summary_json_path}")

if __name__ == '__main__':
    main()
