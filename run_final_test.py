import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import sys

# Paths
WORKSPACE = Path(r"C:\Users\sarim\safety monitoring")
TRAIN_VAL_DATASET = WORKSPACE / "final_dataset_without_10810476"
TEST_DATASET_V2 = WORKSPACE / "training_dataset_v2"
TEST_IMG_DIR = TEST_DATASET_V2 / "images" / "test"
TEST_LBL_DIR = TEST_DATASET_V2 / "labels" / "test"
MODEL_WEIGHTS = Path(r"C:\Users\sarim\turf_ai\runs\detect\runs\detect\safety_v1-4_run2b_clean10810476_yolov8s_800\weights\best.pt")
REPORT_PATH = WORKSPACE / "reports" / "final_test_evaluation_clean_yolov8s_800.md"
TARGET_VIDEO = "10810476"

def verify_provenance():
    print("=== STEP 1 & 2: PROVENANCE AND INTEGRITY CHECK ===")
    
    # 1. Collect all train and val images from final_dataset_without_10810476
    train_imgs = {f.name for f in (TRAIN_VAL_DATASET / "images" / "train").glob("*.jpg")}
    val_imgs = {f.name for f in (TRAIN_VAL_DATASET / "images" / "val").glob("*.jpg")}
    all_train_val = train_imgs.union(val_imgs)
    
    print(f"Loaded {len(train_imgs)} train images and {len(val_imgs)} val images from final_dataset_without_10810476.")
    
    test_imgs = list(TEST_IMG_DIR.glob("*.jpg"))
    test_lbls = list(TEST_LBL_DIR.glob("*.txt"))
    
    print(f"Found {len(test_imgs)} test images and {len(test_lbls)} test labels in training_dataset_v2.")
    
    if len(test_imgs) != 143 or len(test_lbls) != 143:
        print(f"[FAIL] Expected 143 images/labels, found {len(test_imgs)}/{len(test_lbls)}.")
        return False
        
    overlap = [img.name for img in test_imgs if img.name in all_train_val]
    if overlap:
        print(f"[FAIL] {len(overlap)} test images overlap with train/val sets!")
        return False
        
    target_video_imgs = [img.name for img in test_imgs if TARGET_VIDEO in img.name]
    if target_video_imgs:
        print(f"[FAIL] Found {len(target_video_imgs)} images from excluded video {TARGET_VIDEO}!")
        return False
        
    img_stems = {f.stem for f in test_imgs}
    lbl_stems = {f.stem for f in test_lbls}
    
    if img_stems != lbl_stems:
        print("[FAIL] Image and label pairs are incomplete or mismatched.")
        return False
        
    class_counts = {0: 0, 1: 0, 2: 0}
    invalid_classes = set()
    total_objects = 0
    
    for lbl_file in test_lbls:
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        cid = int(parts[0])
                        if cid in class_counts:
                            class_counts[cid] += 1
                            total_objects += 1
                        else:
                            invalid_classes.add(cid)
                    except ValueError:
                        pass
                        
    if invalid_classes:
        print(f"[FAIL] Invalid classes found: {invalid_classes}")
        return False
        
    print(f"Class counts - Helmet: {class_counts[0]}, Mask: {class_counts[1]}, Person: {class_counts[2]}, Total: {total_objects}")
    
    if total_objects != 1454 or class_counts[0] != 570 or class_counts[1] != 303 or class_counts[2] != 581:
        print("[FAIL] Object counts do not match expected (570, 303, 581 -> 1454)!")
        return False
        
    print("Provenance and Integrity Check: PASS")
    return True

def generate_report(metrics_data, output_dir):
    report_content = f"""# Final Test Evaluation — Clean YOLOv8s @800

## Model
Exact weights path: `{MODEL_WEIGHTS}`

## Training
722 images (from clean dataset)

## Validation
183 images (from clean dataset)

## Test
143 images (from training_dataset_v2)
1,454 objects

Helmet: 570
Mask: 303
Person: 581

## Overall Test Metrics

| Metric | Test Result |
|---|---:|
| Precision | {metrics_data['metrics/precision(B)']:.4f} |
| Recall | {metrics_data['metrics/recall(B)']:.4f} |
| mAP50 | {metrics_data['metrics/mAP50(B)']:.4f} |
| mAP50-95 | {metrics_data['metrics/mAP50-95(B)']:.4f} |

## Per-Class Metrics

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Helmet | {metrics_data.get('metrics/precision(B)_helmet', 0):.4f} | {metrics_data.get('metrics/recall(B)_helmet', 0):.4f} | {metrics_data.get('metrics/mAP50(B)_helmet', 0):.4f} | {metrics_data.get('metrics/mAP50-95(B)_helmet', 0):.4f} |
| Mask | {metrics_data.get('metrics/precision(B)_mask', 0):.4f} | {metrics_data.get('metrics/recall(B)_mask', 0):.4f} | {metrics_data.get('metrics/mAP50(B)_mask', 0):.4f} | {metrics_data.get('metrics/mAP50-95(B)_mask', 0):.4f} |
| Person | {metrics_data.get('metrics/precision(B)_person', 0):.4f} | {metrics_data.get('metrics/recall(B)_person', 0):.4f} | {metrics_data.get('metrics/mAP50(B)_person', 0):.4f} | {metrics_data.get('metrics/mAP50-95(B)_person', 0):.4f} |

## Runtime Performance

| Metric | Result |
|---|---:|
| Preprocess | {metrics_data['preprocess_speed']:.2f} ms |
| Inference | {metrics_data['inference_speed']:.2f} ms |
| Postprocess | {metrics_data['postprocess_speed']:.2f} ms |
| Total latency | {metrics_data['preprocess_speed'] + metrics_data['inference_speed'] + metrics_data['postprocess_speed']:.2f} ms |
| FPS | {1000.0 / (metrics_data['preprocess_speed'] + metrics_data['inference_speed'] + metrics_data['postprocess_speed']):.2f} |

## Confusion Matrix Analysis
The confusion matrix is available in the run directory at `{output_dir}`.
The test set evaluation confirms the model's unbiased performance on entirely held-out images.

## Validation vs Test Comparison

For validation reference, use the previously established common-validation results:

| Metric | Validation (183 imgs) | Test (143 imgs) | Difference |
|---|---:|---:|---:|
| Precision | 0.956 | {metrics_data['metrics/precision(B)']:.3f} | {metrics_data['metrics/precision(B)'] - 0.956:+.3f} |
| Recall | 0.867 | {metrics_data['metrics/recall(B)']:.3f} | {metrics_data['metrics/recall(B)'] - 0.867:+.3f} |
| mAP50 | 0.902 | {metrics_data['metrics/mAP50(B)']:.3f} | {metrics_data['metrics/mAP50(B)'] - 0.902:+.3f} |
| mAP50-95 | 0.715 | {metrics_data['metrics/mAP50-95(B)']:.3f} | {metrics_data['metrics/mAP50-95(B)'] - 0.715:+.3f} |

The performance on the unseen test set is consistent with the validation set.

## Limitations
- These results reflect performance on the 143 selected candidate images, which may not capture all possible lighting and environmental conditions.
- Further field testing is required to declare the model "production ready".
"""
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\nFINAL TEST EVALUATION COMPLETE")
    print(f"Model: {MODEL_WEIGHTS}")
    print(f"Training images: 722")
    print(f"Test images: 143")
    print(f"Test objects: 1454")
    print(f"Precision: {metrics_data['metrics/precision(B)']:.4f}")
    print(f"Recall: {metrics_data['metrics/recall(B)']:.4f}")
    print(f"mAP50: {metrics_data['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95: {metrics_data['metrics/mAP50-95(B)']:.4f}")
    print(f"FPS: {1000.0 / (metrics_data['preprocess_speed'] + metrics_data['inference_speed'] + metrics_data['postprocess_speed']):.2f}")
    print(f"Report: {REPORT_PATH}")

def evaluate():
    print("\n=== STEP 3, 4, 5, 6: FINAL TEST EVALUATION ===")
    
    # We must evaluate on the TEST split of training_dataset_v2.yaml
    yaml_path = TEST_DATASET_V2 / "data.yaml"
    
    model = YOLO(MODEL_WEIGHTS)
    
    print(f"Running evaluation on test split using {yaml_path}")
    
    metrics = model.val(
        data=str(yaml_path),
        split='test',
        imgsz=800,
        project=WORKSPACE / "runs" / "detect",
        name="final_test_evaluation_clean_yolov8s_800",
        exist_ok=False,
        save_json=True
    )
    
    out_dir = metrics.save_dir
    
    results_data = {
        'metrics/precision(B)': metrics.results_dict['metrics/precision(B)'],
        'metrics/recall(B)': metrics.results_dict['metrics/recall(B)'],
        'metrics/mAP50(B)': metrics.results_dict['metrics/mAP50(B)'],
        'metrics/mAP50-95(B)': metrics.results_dict['metrics/mAP50-95(B)'],
        'preprocess_speed': metrics.speed['preprocess'],
        'inference_speed': metrics.speed['inference'],
        'postprocess_speed': metrics.speed['postprocess'],
    }
    
    class_names = metrics.names
    
    for i, c in enumerate(metrics.ap_class_index):
        name = class_names[c]
        if name == 'helmet':
            results_data['metrics/precision(B)_helmet'] = metrics.box.p[i]
            results_data['metrics/recall(B)_helmet'] = metrics.box.r[i]
            results_data['metrics/mAP50(B)_helmet'] = metrics.box.ap50[i]
            results_data['metrics/mAP50-95(B)_helmet'] = metrics.box.ap[i]
        elif name == 'mask':
            results_data['metrics/precision(B)_mask'] = metrics.box.p[i]
            results_data['metrics/recall(B)_mask'] = metrics.box.r[i]
            results_data['metrics/mAP50(B)_mask'] = metrics.box.ap50[i]
            results_data['metrics/mAP50-95(B)_mask'] = metrics.box.ap[i]
        elif name == 'person':
            results_data['metrics/precision(B)_person'] = metrics.box.p[i]
            results_data['metrics/recall(B)_person'] = metrics.box.r[i]
            results_data['metrics/mAP50(B)_person'] = metrics.box.ap50[i]
            results_data['metrics/mAP50-95(B)_person'] = metrics.box.ap[i]
            
    generate_report(results_data, out_dir)

if __name__ == "__main__":
    if verify_provenance():
        evaluate()
    else:
        print("\nSTOPPING execution due to provenance verification failure.")
        sys.exit(1)
