# Controlled Confidence Threshold Analysis — Run 2B (YOLOv8s @ 800)

## 1. Methodology & Validation Setup
- **Model**: `runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt`
- **Image Resolution**: 800 x 800
- **Validation Set**: `training_dataset_v2/images/val` (136 validation images, 754 ground truth instances)
- **GT Instance Counts**: `helmet` = 196, `mask` = 172, `person` = 386
- **Object Matching Rules**:
  - Ground truth class matches predicted class
  - Bounding box IoU >= 0.50
  - 1-to-1 greedy matching sorted by prediction confidence score descending
  - Unmatched predictions = FP, unmatched ground truth = FN
- **F1 Formula**: $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
- **TEST set status**: Untouched and NOT evaluated.

---

## 2. Complete Threshold Analysis Table

| Class | Confidence Threshold | TP | FP | FN | Precision | Recall | F1 Score | Total Predictions | GT Count | Delta FP vs 0.25 | Delta FN vs 0.25 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **helmet** | 0.20 | 165 | 11 | 31 | 0.9375 | 0.8418 | 0.8871 | 176 | 196 | +1 | -6 |
| **helmet** | **0.25** | **159** | **10** | **37** | **0.9408** | **0.8112** | **0.8712** | **169** | **196** | **0** | **0** |
| **helmet** | 0.30 | 154 | 7 | 42 | 0.9565 | 0.7857 | 0.8627 | 161 | 196 | -3 | +5 |
| **helmet** | 0.35 | 147 | 7 | 49 | 0.9545 | 0.7500 | 0.8400 | 154 | 196 | -3 | +12 |
| **helmet** | 0.40 | 136 | 6 | 60 | 0.9577 | 0.6939 | 0.8047 | 142 | 196 | -4 | +23 |
| **helmet** | 0.45 | 130 | 4 | 66 | 0.9701 | 0.6633 | 0.7879 | 134 | 196 | -6 | +29 |
| **helmet** | 0.50 | 124 | 3 | 72 | 0.9764 | 0.6327 | 0.7678 | 127 | 196 | -7 | +35 |
| **helmet** | 0.55 | 112 | 0 | 84 | 1.0000 | 0.5714 | 0.7273 | 112 | 196 | -10 | +47 |
| **helmet** | 0.60 | 97 | 0 | 99 | 1.0000 | 0.4949 | 0.6621 | 97 | 196 | -10 | +62 |
| | | | | | | | | | | | |
| **mask** | 0.20 | 138 | 30 | 34 | 0.8214 | 0.8023 | 0.8118 | 168 | 172 | +12 | -9 |
| **mask** | **0.25** | **129** | **18** | **43** | **0.8776** | **0.7500** | **0.8088** | **147** | **172** | **0** | **0** |
| **mask** | 0.30 | 123 | 10 | 49 | 0.9248 | 0.7151 | 0.8066 | 133 | 172 | -8 | +6 |
| **mask** | 0.35 | 120 | 9 | 52 | 0.9302 | 0.6977 | 0.7973 | 129 | 172 | -9 | +9 |
| **mask** | 0.40 | 116 | 5 | 56 | 0.9587 | 0.6744 | 0.7918 | 121 | 172 | -13 | +13 |
| **mask** | 0.45 | 113 | 3 | 59 | 0.9741 | 0.6570 | 0.7847 | 116 | 172 | -15 | +16 |
| **mask** | 0.50 | 109 | 1 | 63 | 0.9909 | 0.6337 | 0.7730 | 110 | 172 | -17 | +20 |
| **mask** | 0.55 | 106 | 1 | 66 | 0.9907 | 0.6163 | 0.7599 | 107 | 172 | -17 | +23 |
| **mask** | 0.60 | 103 | 1 | 69 | 0.9904 | 0.5988 | 0.7464 | 104 | 172 | -17 | +26 |
| | | | | | | | | | | | |
| **person** | 0.20 | 359 | 371 | 27 | 0.4918 | 0.9301 | 0.6434 | 730 | 386 | +66 | -5 |
| **person** | **0.25** | **354** | **305** | **32** | **0.5372** | **0.9171** | **0.6775** | **659** | **386** | **0** | **0** |
| **person** | 0.30 | 350 | 251 | 36 | 0.5824 | 0.9067 | 0.7092 | 601 | 386 | -54 | +4 |
| **person** | 0.35 | 348 | 202 | 38 | 0.6327 | 0.9016 | 0.7436 | 550 | 386 | -103 | +6 |
| **person** | 0.40 | 342 | 160 | 44 | 0.6813 | 0.8860 | 0.7703 | 502 | 386 | -145 | +12 |
| **person** | 0.45 | 339 | 119 | 47 | 0.7402 | 0.8782 | 0.8033 | 458 | 386 | -186 | +15 |
| **person** | 0.50 | 337 | 91 | 49 | 0.7874 | 0.8731 | 0.8280 | 428 | 386 | -214 | +17 |
| **person** | 0.55 | 329 | 74 | 57 | 0.8164 | 0.8523 | 0.8340 | 403 | 386 | -231 | +25 |
| **person** | 0.60 | 313 | 52 | 73 | 0.8575 | 0.8109 | 0.8336 | 365 | 386 | -253 | +41 |

---

## 3. Class-by-Class Observations & Tradeoff Behavior

### A. Person Class (GT = 386)
- **Baseline at conf=0.25**: Precision = `0.5372`, Recall = `0.9171`, F1 = `0.6775`, FP = `305`, FN = `32`.
- **FP Behavior**: At low confidence (0.20 – 0.25), person detections suffer from 305 to 371 background false positives on complex factory machinery, foliage, and structural reflections.
- **Threshold Impact**: Raising threshold from 0.25 to **0.50 – 0.55** drastically reduces false positives from 305 down to **91 – 74** (a **70.2% – 75.7% drop in FPs**), while retaining **85.2% – 87.3% person recall**.
- **F1 Peak**: Peak F1 score of **0.8340** is reached at **conf = 0.55** (Precision = 0.8164, Recall = 0.8523).

### B. Helmet Class (GT = 196)
- **Baseline at conf=0.25**: Precision = `0.9408`, Recall = `0.8112`, F1 = `0.8712`, FP = `10`, FN = `37`.
- **Precision / Recall Behavior**: Helmet detection maintains high precision (93.8% – 100.0%) across all thresholds. However, raising the threshold above 0.30 rapidly degrades recall (drops from 84.2% at 0.20 to 63.3% at 0.50).
- **F1 Peak**: Peak F1 score of **0.8871** occurs at **conf = 0.20** (Precision = 0.9375, Recall = 0.8418, 11 FPs, 31 FNs), with **0.25** providing F1 = **0.8712** (Precision = 0.9408, Recall = 0.8112, 10 FPs, 37 FNs).

### C. Mask Class (GT = 172)
- **Baseline at conf=0.25**: Precision = `0.8776`, Recall = `0.7500`, F1 = `0.8088`, FP = `18`, FN = `43`.
- **Precision / Recall Behavior**: At conf=0.20, mask recall reaches **80.23%** (34 FNs) with 30 FPs (F1 = `0.8118`). As confidence increases, precision improves to 97.4% at conf=0.45, but recall drops to 65.7% (59 FNs).
- **F1 Peak**: Peak F1 score of **0.8118** occurs at **conf = 0.20** (Precision = 0.8214, Recall = 0.8023, FP = 30, FN = 34), with **conf = 0.25** offering a very strong balance: Precision = `0.8776`, Recall = `0.7500`, F1 = `0.8088` (FP = 18, FN = 43).

---

## 4. Candidate Threshold Recommendations for Next Stage (PPE Association)

For downstream PPE association (`person` → `helmet`, `person` → `mask`), we recommend testing the following candidate class-specific threshold ranges:

1. **PERSON: Candidate Range = [0.45 – 0.55]**
   - **Why**: At conf=0.25, person detections generate 305 false positives. Raising the threshold to **0.45 – 0.55** drops FPs by **61.0% – 75.7%** (down to 119 – 74 FPs) while keeping person recall very high (**85.2% – 87.8%**) and maximizing Person F1 (**0.8033 – 0.8340**). This ensures PPE association receives clean person bounding boxes without background noise.

2. **HELMET: Candidate Range = [0.20 – 0.30]**
   - **Why**: Helmet precision is extremely strong even at low thresholds (94.1% at 0.25). Keeping the threshold in **0.20 – 0.30** maximizes helmet recall (**78.6% – 84.2%**) so helmets are not missed during person-helmet overlap checking.

3. **MASK: Candidate Range = [0.20 – 0.30]**
   - **Why**: Mask is the most recall-sensitive safety class. Keeping the threshold in **0.20 – 0.30** preserves high recall (**71.5% – 80.2%**) with solid precision (**82.1% – 92.5%**), giving the PPE association logic sufficient mask detections to evaluate compliance.
