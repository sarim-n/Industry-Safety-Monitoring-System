import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import numpy as np
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO

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

def xywh2xyxy(box, w, h):
    cx, cy, bw, bh = box
    x1 = (cx - bw/2) * w
    y1 = (cy - bh/2) * h
    x2 = (cx + bw/2) * w
    y2 = (cy + bh/2) * h
    return [x1, y1, x2, y2]

def main():
    model_path = r"runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt"
    val_img_dir = r"training_dataset_v2/images/val"
    val_lbl_dir = r"training_dataset_v2/labels/val"
    
    assert os.path.exists(model_path), f"Model path not found: {model_path}"
    
    model = YOLO(model_path)
    class_names = {0: 'helmet', 1: 'mask', 2: 'person'}
    
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.*")))
    conf_threshold = 0.25
    iou_threshold = 0.5
    
    # 4x4 Confusion matrix: GT (row) x Pred (col) [0:helmet, 1:mask, 2:person, 3:bg]
    cm = np.zeros((4, 4), dtype=int)
    
    source_stats = {}
    mask_fn_list = []
    helmet_fp_list = []
    person_fn_list = []
    person_fp_list = []
    
    target_sources = [
        "4048038451_preview", 
        "4017518657_preview", 
        "8482302_hd_1920_1080_25fps", 
        "19832490_hd_1920_1080_25fps", 
        "istockphoto_2258622635",
        "no_safety", 
        "gloves_mask"
    ]
    
    for img_path in val_images:
        img_name = os.path.basename(img_path)
        lbl_path = os.path.join(val_lbl_dir, os.path.splitext(img_name)[0] + ".txt")
        
        # Source group determination
        sg = "other"
        for candidate in target_sources:
            if candidate in img_name:
                sg = candidate
                break
                
        if sg not in source_stats:
            source_stats[sg] = {
                'imgs': 0, 'gt_helmet': 0, 'gt_mask': 0, 'gt_person': 0,
                'mask_fn': 0, 'helmet_fp': 0, 'person_fn': 0, 'person_fp': 0
            }
        source_stats[sg]['imgs'] += 1
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        gt_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c = int(parts[0])
                        box_norm = [float(x) for x in parts[1:]]
                        box_abs = xywh2xyxy(box_norm, w, h)
                        gt_boxes.append({'cls': c, 'box': box_abs, 'norm': box_norm, 'matched': False})
                        if c == 0: source_stats[sg]['gt_helmet'] += 1
                        elif c == 1: source_stats[sg]['gt_mask'] += 1
                        elif c == 2: source_stats[sg]['gt_person'] += 1

        results = model.predict(img_path, imgsz=800, conf=conf_threshold, verbose=False)[0]
        preds = []
        for box in results.boxes:
            c = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            preds.append({'cls': c, 'conf': conf, 'box': xyxy, 'matched': False})

        # Match Preds to GT (greedy IoU matching across all classes)
        for c in [0, 1, 2]:
            c_gts = [g for g in gt_boxes if g['cls'] == c]
            c_preds = [p for p in preds if p['cls'] == c]
            c_preds.sort(key=lambda x: x['conf'], reverse=True)
            for p in c_preds:
                best_iou = 0
                best_g = None
                for g in c_gts:
                    if not g['matched']:
                        iou = box_iou(p['box'], g['box'])
                        if iou > best_iou:
                            best_iou = iou
                            best_g = g
                if best_iou >= iou_threshold:
                    p['matched'] = True
                    best_g['matched'] = True
                    cm[c, c] += 1
                    
        for p in preds:
            if not p['matched']:
                best_iou = 0
                best_g = None
                for g in gt_boxes:
                    if not g['matched']:
                        iou = box_iou(p['box'], g['box'])
                        if iou > best_iou:
                            best_iou = iou
                            best_g = g
                if best_iou >= 0.3:
                    p['matched'] = True
                    best_g['matched'] = True
                    cm[best_g['cls'], p['cls']] += 1
                else:
                    cm[3, p['cls']] += 1
                    if p['cls'] == 0:
                        source_stats[sg]['helmet_fp'] += 1
                        helmet_fp_list.append({'img': img_name, 'sg': sg, 'pred': p, 'cause': 'bg_fp'})
                    elif p['cls'] == 2:
                        source_stats[sg]['person_fp'] += 1
                        person_fp_list.append({'img': img_name, 'sg': sg, 'pred': p, 'cause': 'bg_fp'})

        for g in gt_boxes:
            if not g['matched']:
                cm[g['cls'], 3] += 1
                if g['cls'] == 1:
                    source_stats[sg]['mask_fn'] += 1
                    mask_fn_list.append({'img': img_name, 'sg': sg, 'gt': g})
                elif g['cls'] == 0:
                    pass
                elif g['cls'] == 2:
                    source_stats[sg]['person_fn'] += 1
                    person_fn_list.append({'img': img_name, 'sg': sg, 'gt': g})

    print("=================== RUN 2B ERROR ANALYSIS REPORT ===================")
    print("\n--- CONFUSION MATRIX (GT rows x Pred cols) ---")
    print("Labels: [0: helmet, 1: mask, 2: person, 3: background]")
    print(f"       {'helmet':>8} {'mask':>8} {'person':>8} {'bg(FN)':>8}")
    labels = ['helmet', 'mask', 'person', 'background']
    for i in range(3):
        print(f"{labels[i]:<8} {cm[i,0]:8d} {cm[i,1]:8d} {cm[i,2]:8d} {cm[i,3]:8d}")
    print(f"{'bg(FP)':<8} {cm[3,0]:8d} {cm[3,1]:8d} {cm[3,2]:8d} {'-':>8}")

    print("\n--- PER SOURCE GROUP BREAKDOWN ---")
    print(f"{'Source Group':<30} {'Imgs':>5} | {'GT H':>5} {'GT M':>5} {'GT P':>5} | {'M FN':>5} {'H FP':>5} {'P FN':>5} {'P FP':>5}")
    print("-" * 80)
    for sg, st in sorted(source_stats.items()):
        print(f"{sg:<30} {st['imgs']:5d} | {st['gt_helmet']:5d} {st['gt_mask']:5d} {st['gt_person']:5d} | {st['mask_fn']:5d} {st['helmet_fp']:5d} {st['person_fn']:5d} {st['person_fp']:5d}")

    # Analyze Mask FNs by size
    print("\n--- MASK FALSE NEGATIVE ANALYSIS ---")
    total_gt_mask = sum(st['gt_mask'] for st in source_stats.values())
    print(f"Total Mask FN count: {len(mask_fn_list)} out of {total_gt_mask} GT masks ({len(mask_fn_list)/max(1, total_gt_mask)*100:.1f}%)")
    
    small_masks = [m for m in mask_fn_list if (m['gt']['norm'][2]*800)*(m['gt']['norm'][3]*800) < 32*32]
    med_masks = [m for m in mask_fn_list if 32*32 <= (m['gt']['norm'][2]*800)*(m['gt']['norm'][3]*800) < 96*96]
    large_masks = [m for m in mask_fn_list if (m['gt']['norm'][2]*800)*(m['gt']['norm'][3]*800) >= 96*96]
    print(f"  Small masks (<32x32 px at 800): {len(small_masks)} FNs ({len(small_masks)/max(1,len(mask_fn_list))*100:.1f}%)")
    print(f"  Medium masks (32x32-96x96 px):  {len(med_masks)} FNs ({len(med_masks)/max(1,len(mask_fn_list))*100:.1f}%)")
    print(f"  Large masks (>=96x96 px):       {len(large_masks)} FNs ({len(large_masks)/max(1,len(mask_fn_list))*100:.1f}%)")

    # Analyze Helmet FPs
    print("\n--- HELMET FALSE POSITIVE ANALYSIS ---")
    print(f"Total Helmet FP count: {len(helmet_fp_list)}")
    h_fp_confs = [h['pred']['conf'] for h in helmet_fp_list]
    print(f"  Confidence scores of Helmet FPs: min={min(h_fp_confs) if h_fp_confs else 0:.2f}, max={max(h_fp_confs) if h_fp_confs else 0:.2f}, mean={np.mean(h_fp_confs) if h_fp_confs else 0:.2f}")

    # Analyze Person errors
    print("\n--- PERSON ERROR ANALYSIS ---")
    total_gt_person = sum(st['gt_person'] for st in source_stats.values())
    print(f"Person FNs: {len(person_fn_list)} / {total_gt_person} GT persons ({len(person_fn_list)/max(1, total_gt_person)*100:.1f}%)")
    print(f"Person FPs: {len(person_fp_list)}")

if __name__ == '__main__':
    main()
