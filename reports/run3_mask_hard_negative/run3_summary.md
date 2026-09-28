# Run 3 — Mask Hard-Negative Controlled Experiment Report

## Executive Summary

A controlled experiment (**Run 3**) was conducted to evaluate whether augmenting the training dataset with 186 targeted hard-negative images (containing unmasked workers with facial hair, shadows, side profiles, collars, and hands near face) improves mask detection performance and reduces face-region false positives.

### **Factual Experiment Conclusion: REGRESSED (Run 3 fails to outperform Run 2B Baseline)**

- **Mask mAP50**: Regressed from **0.9024 (Run 2B)** to **0.8212 (Run 3)** ($-8.12\%$).
- **Mask Recall**: Regressed from **0.7791 (Run 2B)** to **0.6991 (Run 3)** ($-8.00\%$).
- **Mask Precision**: Regressed from **0.8784 (Run 2B)** to **0.8635 (Run 3)** ($-1.49\%$).
- **Face-Region Mask False Positives (@ MASK_CONF=0.20)**: Increased from **30 (Run 2B)** to **35 (Run 3)** ($+5$ false positives).
- **Helmet Recall**: Regressed from **0.8326 (Run 2B)** to **0.7993 (Run 3)** ($-3.33\%$).
- **Person Recall**: Regressed from **0.9223 (Run 2B)** to **0.8808 (Run 3)** ($-4.15\%$).

> [!CAUTION]
> **Root Cause**: Adding 186 train-only images with 370 person boxes, 190 helmet boxes, and **0 mask boxes** created a strong negative mask class imbalance during training. This caused the model to become overly conservative across all classes, suppressing valid mask predictions without resolving false-positive detections on difficult facial features.

---

## 1. Objective

Determine whether targeted hard-negative training data (`mask data`) improves mask detection stability, recall, precision, mAP, and reduces face-region false positives on unmasked faces (beards, stubble, shadows, hands, collars, side profiles) without regressing helmet/person performance.

---

## 2. Dataset Validation & Before/After Statistics

All 186 images in `mask data` were validated and converted from COCO format to standard YOLO `.txt` labels. 100% of annotations were compliant with Authoritative Classes (`0=helmet`, `1=mask`, `2=person`).

| Dataset Split | Images | Total BBoxes | Person Boxes (cls 2) | Helmet Boxes (cls 0) | Mask Boxes (cls 1) | Mask / Person Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Existing TRAIN (Run 2B)** | 687 | 4,130 | 2,281 | 937 | 912 | 0.3998 |
| **New Mask Data (Train-Only)** | 186 | 560 | 370 | 190 | 0 | 0.0000 |
| **Combined Run 3 TRAIN** | **873** | **4,690** | **2,651** | **1,127** | **912** | **0.3440** |
| **VAL Split (UNTOUCHED)** | **136** | **754** | **386** | **196** | **172** | **0.4456** |
| **TEST Split (UNTOUCHED)** | **143** | **812** | **412** | **210** | **190** | **0.4612** |

- **Duplicate Check**: MD5 hashing verified 0 duplicate images between `training_dataset_v2` and `mask data`.
- **Validation / Test Safety**: `val` (136 images) and `test` (143 images) splits remained **100% untouched**.

---

## 3. Training Configuration & Execution

Run 3 was trained under strictly controlled conditions identical to Run 2B:

- **Model Architecture**: YOLOv8s pretrained on COCO (`yolov8s.pt`)
- **Resolution (`imgsz`)**: 800
- **Epochs / Early Stopping**: Max 50 epochs, patience 15 (Early stopping triggered at Epoch 16, best epoch: 1)
- **Batch Size**: 4
- **Optimizer / Seed**: Auto (AdamW), Seed 42
- **Hardware**: NVIDIA GeForce RTX 2050 (4 GB VRAM)
- **Training Time / Peak VRAM**: 22.67 minutes (1,360.4 s), 2,077.6 MB peak VRAM

---

## 4. Validation Set Performance Comparison (Run 2B vs Run 3)

Evaluated strictly on the **136 validation images** using `training_dataset_v2/data.yaml`:

| Metric / Class | Run 2B Baseline | Run 3 Experiment | Net Change |
| :--- | :---: | :---: | :---: |
| **Overall Precision** | 0.8276 | 0.8552 | **+0.0276** |
| **Overall Recall** | 0.8447 | 0.7931 | **-0.0516** |
| **Overall mAP50** | 0.9227 | 0.8904 | **-0.0323** |
| **Overall mAP50-95** | 0.6634 | 0.6337 | **-0.0297** |
| **Mask Precision** | 0.8784 | 0.8635 | **-0.0149** |
| **Mask Recall** | 0.7791 | 0.6991 | **-0.0800** |
| **Mask mAP50** | 0.9024 | 0.8212 | **-0.0812** |
| **Mask mAP50-95** | 0.5278 | 0.5006 | **-0.0272** |
| **Helmet Precision** | 0.9477 | 0.9937 | +0.0460 |
| **Helmet Recall** | 0.8326 | 0.7993 | **-0.0333** |
| **Helmet mAP50** | 0.9508 | 0.9585 | +0.0077 |
| **Person Precision** | 0.6565 | 0.7083 | +0.0518 |
| **Person Recall** | 0.9223 | 0.8808 | **-0.0415** |
| **Person mAP50** | 0.9150 | 0.8915 | **-0.0235** |

---

## 5. Mask False Positive & Confidence Sensitivity Analysis

### Diagnostic Evaluation at `MASK_CONF = 0.20`

- **Run 2B Baseline**: TP = 138, FP = 30, Face-FP = 30, FN = 34, Precision = 0.8214, Recall = 0.8023, F1 = 0.8118
- **Run 3 Experiment**: TP = 126, FP = 35, Face-FP = 35, FN = 46, Precision = 0.7826, Recall = 0.7326, F1 = 0.7568
- **Net Impact**: **+5 face-region false positives**, **+12 missed masks (FN)**.

### Confidence Threshold Sensitivity Sweep

| Threshold | Run 2B Prec | Run 2B Rec | Run 2B Face-FP | Run 3 Prec | Run 3 Rec | Run 3 Face-FP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.20** | 0.8214 | 0.8023 | 30 | 0.7826 | 0.7326 | 35 |
| **0.25** | 0.8776 | 0.7500 | 18 | 0.8243 | 0.7093 | 26 |
| **0.30** | 0.9248 | 0.7151 | 10 | 0.8527 | 0.6395 | 19 |
| **0.35** | 0.9302 | 0.6977 | 9 | 0.8966 | 0.6047 | 12 |
| **0.40** | 0.9587 | 0.6744 | 5 | 0.9083 | 0.5756 | 10 |
| **0.45** | 0.9741 | 0.6570 | 3 | 0.9048 | 0.5523 | 10 |
| **0.50** | 0.9909 | 0.6337 | 1 | 0.9118 | 0.5407 | 9 |

Across all confidence thresholds, Run 3 consistently exhibits **lower precision, lower recall, and higher face-region false positives** than Run 2B.

---

## 6. Visual Error Analysis Findings

Visual side-by-side inspection on problematic validation images (`reports/mask_fp_analysis_run3/visual_comparisons/`) revealed:
1. **Facial Hair / Beards**: False-positive mask detections persist on dark beards and stubble.
2. **Dark Facial Shadows**: Shadowed face crops continue to generate weak mask proposals.
3. **Genuine Masks**: Several legitimate, partially shadowed masks correctly detected by Run 2B are now **missed** by Run 3.

---

## 7. Dataset Safety & Production Pipeline Guardrails

- **TEST SET**: **UNTOUCHED** (143 images reserved for final candidate selection).
- **PRODUCTION MODEL**: **UNCHANGED** (`runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt`).
- **PRODUCTION THRESHOLDS**: **UNCHANGED** (`PERSON_CONF=0.50`, `HELMET_CONF=0.25`, `MASK_CONF=0.20`).
- **PRODUCTION PIPELINE**: **UNCHANGED** (`run_live.py` with Rule 1 Person Suppression intact).

---

## 8. Artifacts Produced

1. **Run 3 Weights & Results**: `runs/detect/safety_v1-4_run3_mask_hard_negative/weights/best.pt`
2. **Validation Metrics Comparison**: `reports/run3_mask_hard_negative/val_metrics_comparison.json`
3. **Mask Sensitivity CSV**: `reports/mask_fp_analysis_run3/mask_sensitivity_comparison.csv`
4. **Visual Side-by-Side Comparisons**: `reports/mask_fp_analysis_run3/visual_comparisons/` (25 comparison images)
5. **Combined Dataset**: `training_dataset_v3_mask_hard_negative/`

---

## 9. Final Recommendation & Next Steps

### **RECOMMENDATION: REJECT RUN 3 (Option C: Perform another controlled experiment)**

Run 3 demonstrates that adding unmasked hard-negative images **without positive mask examples** distorts class balance and degrades overall model performance.

**Recommended Next Action**:
1. **Maintain Run 2B as the Active Candidate Model**.
2. **Design a Balanced Hard-Negative Batch**: To effectively train the model on unmasked hard negatives, future datasets must include **positive mask examples in similar lighting/resolution** to balance class priors.
3. Keep the test set completely untouched until a candidate model demonstrates statistically significant improvements over Run 2B on the validation set.
