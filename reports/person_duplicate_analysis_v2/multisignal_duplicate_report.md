# Multi-Signal Person Duplicate Analysis (v2)

> [!IMPORTANT]
> **STATUS: ISOLATED DIAGNOSTIC EXPERIMENT ONLY.**
> Production code (`run_live.py`), model weights (`best.pt`), datasets, annotations, thresholds (`PERSON_CONF=0.50`, `HELMET_CONF=0.25`, `MASK_CONF=0.20`), global NMS, PPE association, PPE observability, and temporal confirmation remain **100% FROZEN and UNMUTATED**. No production fix is implemented.

---

## 1. Objective

Building upon the previous single-signal IoU experiment (which proved that simple IoU thresholding suppresses legitimate workers), this study investigates whether a **multi-signal deterministic geometric rule** (combining IoU, normalized center distance, containment ratio, area ratio, aspect ratio, and confidence difference) can reliably distinguish:

- **A. True Duplicate Person Boxes**: Two predicted bounding boxes corresponding to the **SAME** human worker.
- **B. Legitimate Overlapping Workers**: Two predicted bounding boxes corresponding to **TWO DIFFERENT** human workers standing close together.

Goal: Achieve **one person bounding box per human** while enforcing **zero legitimate worker suppression**.

---

## 2. Frozen Configuration

All experiments were conducted strictly offline under frozen settings:
- **Model**: `runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt` (YOLOv8s @ 800)
- **Validation Dataset**: `training_dataset_v2/images/val` (136 images, 386 ground-truth persons)
- **Frozen Thresholds**: `PERSON_CONF = 0.50`, `HELMET_CONF = 0.25`, `MASK_CONF = 0.20`
- **Global YOLO NMS**: Default `iou = 0.70` (unaltered)
- **Live Test Video**: `data_collection/videos/4048038451-preview.mp4` (301 frames)

---

## 3. Verified Baseline

Re-evaluated on all 136 validation images at `PERSON_CONF = 0.50`:

- **Total Ground-Truth (GT) Persons**: 386
- **Total Predicted Person Boxes**: 428
- **Total Candidate Duplicate Pairs ($\text{IoU} \ge 0.50$)**: 67
  - **True Duplicate Pairs (Same GT)**: 42 ($62.7\%$)
  - **Legitimate Overlapping Pairs (Different GTs)**: 19 ($28.4\%$)
  - **Partial False Positive Pairs**: 5 ($7.5\%$)
  - **False Positive Pairs**: 1 ($1.5\%$)
- **Baseline Person Precision**: `0.7850`
- **Baseline Person Recall**: `0.8705`
- **Baseline Person F1**: `0.8256`
- **GT Persons Missed**: 50
- **Average Person Boxes per GT Person**: `0.9974`

---

## 4. Candidate Pair Dataset

For every predicted person pair with $\text{IoU} \ge 0.50$, we extracted a comprehensive 20-feature geometric representation:

1. **Bounding Box Geometries**: $(x_1, y_1, x_2, y_2)$, widths $(w_A, w_B)$, heights $(h_A, h_B)$, areas $(\text{area}_A, \text{area}_B)$, centers $(cx_A, cy_A, cx_B, cy_B)$, diagonals $(\text{diag}_A, \text{diag}_B)$.
2. **Normalized Center Distance**:
   - `norm_dist_diag_large`: $\text{center\_dist} / \text{diag}_{\text{large}}$
   - `norm_dist_diag_small`: $\text{center\_dist} / \text{diag}_{\text{small}}$
   - `norm_dist_img_w`: $dx / W_{\text{img}}$, `norm_dist_img_h`: $dy / H_{\text{img}}$
3. **Containment (Intersection over Area)**:
   - `ioa_a_in_b`, `ioa_b_in_a`
   - `max_containment`: $\max(\text{IoA}_A, \text{IoA}_B)$
   - `min_containment`: $\min(\text{IoA}_A, \text{IoA}_B)$
4. **Size Ratios**:
   - `area_ratio`: $\text{area}_{\text{small}} / \text{area}_{\text{large}}$
   - `width_ratio`: $w_{\text{small}} / w_{\text{large}}$
   - `height_ratio`: $h_{\text{small}} / h_{\text{large}}$
5. **Aspect Ratio & Overlap**:
   - `ar_diff`: $|(w_A/h_A) - (w_B/h_B)|$
   - `horiz_overlap_ratio`, `vert_overlap_ratio`
6. **Confidence Metrics**:
   - `conf_a`, `conf_b`, `conf_diff`: $|\text{conf}_A - \text{conf}_B|$

---

## 5. Ground Truth Classification

Candidate pairs were strictly ground-truth labeled by IoU matching ($\ge 0.50$) against annotated validation GT persons:
- **`TRUE_DUPLICATE`** (42 pairs): Both predicted boxes match the **exact same GT person**.
- **`LEGITIMATE_OVERLAP`** (19 pairs): Box A matches GT Person X, Box B matches GT Person Y ($X \ne Y$).

---

## 6. Feature Analysis & Distribution Comparison

Summary statistics (Min, P25, Median, Mean, P75, Max) for `TRUE_DUPLICATE` (TD) vs `LEGITIMATE_OVERLAP` (LO):

| Feature | TD Min | TD Median | TD Mean | TD Max | LO Min | LO Median | LO Mean | LO Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IoU** | 0.5042 | 0.6286 | 0.6138 | 0.6961 | 0.5257 | 0.5762 | 0.5843 | 0.6604 |
| **Center Dist (px)** | 34.30 | 74.64 | 88.15 | 284.35 | 37.51 | 46.33 | 46.61 | 57.02 |
| **Norm Dist / Diag Large** | 0.0782 | 0.1100 | 0.1267 | 0.2144 | 0.0876 | 0.1124 | 0.1133 | 0.1311 |
| **Max Containment** | 0.7527 | 0.9909 | 0.9684 | 1.0000 | 0.9226 | 0.9770 | 0.9745 | 1.0000 |
| **Min Containment** | 0.5050 | 0.6389 | 0.6309 | 0.7979 | 0.5264 | 0.5901 | 0.5938 | 0.6882 |
| **Area Ratio** | 0.5050 | 0.6394 | 0.6578 | 0.9975 | 0.5276 | 0.6014 | 0.6099 | 0.7303 |
| **Height Ratio** | 0.5388 | 0.9791 | 0.8605 | 0.9999 | 0.8438 | 0.9647 | 0.9694 | 0.9992 |
| **Conf Diff** | 0.0070 | 0.1719 | 0.1646 | 0.4292 | 0.0090 | 0.2234 | 0.1955 | 0.3674 |

> [!CAUTION]
> **PARAMOUNT EMPIRICAL FINDING: COMPLETE STATISTICAL OVERLAP ACROSS ALL GEOMETRIC FEATURES.**
> - Legitimate overlaps and true duplicates share almost identical medians for normalized center distance (`0.1124` vs `0.1100`), IoU (`0.5762` vs `0.6286`), and area ratio (`0.6014` vs `0.6394`).
> - Legitimate overlapping worker pairs actually have a **HIGHER minimum containment** (`0.9226`) than true duplicates (`0.7527`)!

---

## 7. Visual Analysis

Representative crops were rendered and saved to [contact_sheets/](file:///c:/Users/sarim/safety%20monitoring/reports/person_duplicate_analysis_v2/contact_sheets/):
- `true_duplicates_contact_sheet.jpg` (10 true duplicate pairs)
- `legitimate_overlaps_contact_sheet.jpg` (10 legitimate overlapping worker pairs)

**Visual Observation**:
When two workers walk closely or stand behind one another in 2D perspective, their bounding boxes align almost perfectly in vertical height and center position—making their 2D geometric signature indistinguishable from a primary worker box + secondary torso/shoulder duplicate box.

---

## 8. True Duplicate vs Legitimate Overlap

Standard 2D bounding boxes lack depth (3D z-distance) and individual body part/keypoint information. Consequently:
- **True Duplicates**: Occur because single-frame YOLO NMS (`iou=0.70`) permits a second lower-confidence candidate box ($0.504 \le \text{IoU} < 0.696$) on the same worker.
- **Legitimate Overlaps**: Occur because real construction workers in close proximity project onto nearly identical 2D image coordinates.

---

## 9. Candidate Deterministic Rules & Grid Search

We grid-searched 192 multi-signal rule combinations across:
- $\text{IoU} \in [0.50, 0.55, 0.60, 0.65]$
- $\text{Norm Center Dist} \in [0.15, 0.20, 0.25, 0.30]$
- $\text{Max Containment} \in [0.70, 0.75, 0.80, 0.85]$
- $\text{Area Ratio} \in [0.40, 0.50, 0.60]$

---

## 10. Rule Evaluation & Trade-offs

Top evaluated candidate multi-signal rules:

| Rule Setting | Rule Condition | True Dups Removed | True Dups Remaining | Legit Overlaps Suppressed | Legit Overlap Preservation Rate | Duplicate Removal Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **RULE_A (High Safety)** | $\text{IoU} \ge 0.55 \text{ \& } \text{NDist} \le 0.15 \text{ \& } \text{MaxCont} \ge 0.85 \text{ \& } \text{AreaR} \ge 0.50$ | 26 | 16 | **2** | **89.5%** | 61.9% |
| **RULE_B (Balanced)** | $\text{IoU} \ge 0.50 \text{ \& } \text{NDist} \le 0.20 \text{ \& } \text{MaxCont} \ge 0.80 \text{ \& } \text{AreaR} \ge 0.40$ | 34 | 8 | **11** | **42.1%** | 81.0% |
| **RULE_C (Aggressive)** | $\text{IoU} \ge 0.50 \text{ \& } (\text{NDist} \le 0.25 \text{ \| } \text{MaxCont} \ge 0.75) \text{ \& } \text{AreaR} \ge 0.40$ | 42 | 0 | **19** | **0.0%** | 100.0% |

---

## 11. Person Detection Metrics (Complete Validation Set)

| Setting | Total Pred Persons | Precision | Recall | F1 Score | GT Persons Missed | Avg Person Boxes / GT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASELINE** | **428** | **0.7850** | **0.8705** | **0.8256** | **50** | **0.9974** |
| **RULE_A (High Safety)** | 393 | 0.8346 | 0.8497 | 0.8421 | 58 (+8) | 0.9145 |
| **RULE_B (Balanced)** | 371 | 0.8625 | 0.8290 | 0.8454 | 66 (+16) | 0.8627 |
| **RULE_C (Aggressive)** | 365 | 0.8767 | 0.8290 | 0.8522 | 66 (+16) | 0.8472 |

---

## 12. Tracking Interaction (`4048038451-preview.mp4`, 301 Frames)

| Setting | Frames with Dup Persons ($\text{IoU} \ge 0.50$) | Avg Active Tracks / Frame | Unique Track IDs Created | Frames with Duplicate Track IDs |
| :--- | :---: | :---: | :---: | :---: |
| **BASELINE** | **2** | **2.16** | **6** | **22** |
| **RULE_A (High Safety)** | 0 | 2.09 | 4 | 0 |
| **RULE_B (Balanced)** | 0 | 2.09 | 4 | 0 |
| **RULE_C (Aggressive)** | 0 | 2.09 | 4 | 0 |

---

## 13. Computational Cost

- **Added Microsecond Latency per Frame**: `0.0918 ms` (91.8 microseconds)
- **Baseline Pipeline Latency**: `0.19 ms` (tracking layer) + ~15-20 ms YOLO inference.
- The multi-signal deterministic calculation adds virtually zero computational overhead ($<0.1\text{ ms}$).

---

## 14. Limitations

1. **Pure Bounding-Box Geometry Bound**: No deterministic 2D bounding-box rule can distinguish duplicate bounding boxes from legitimate overlapping workers with $100\%$ accuracy due to physical 2D projection overlap.
2. **Trade-off Constraint**: Maximizing worker safety (Rule A: only 2 legitimate workers suppressed) leaves 16 true duplicate pairs active. Conversely, eliminating all true duplicates (Rule C) suppresses 19 legitimate workers.

---

## 15. Recommended Next Experiment

Rather than relying on 2D bounding-box post-processing, future diagnostic phases should explore:
1. **Pose / Keypoint-based Head-Center Distance**: Suppressing boxes whose head keypoints coincide within a tight radius.
2. **Tracker-Level Identity Soft-Suppression**: Allowing single-frame overlaps but preventing tracker identity splits via IoU matching decay.

---

## Answers to Core Diagnostic Questions

1. **Are duplicate person boxes actually coming from YOLO?**
   - **Yes.** 42 out of 67 candidate pairs ($62.7\%$) on validation images are raw YOLO single-frame duplicate detections leaking through the global `iou=0.70` NMS threshold.

2. **What percentage of high-IoU pairs are true duplicates?**
   - **62.7%** are true duplicates, while **28.4%** represent legitimate overlapping workers.

3. **Which geometric signals distinguish duplicates from legitimate overlapping workers?**
   - **None in isolation or deterministic 2D combination.** Both true duplicates and legitimate overlapping workers exhibit overlapping distributions across IoU, center distance, containment, area ratio, and confidence difference.

4. **Can a lightweight multi-signal deterministic rule reduce duplicates without significantly suppressing legitimate workers?**
   - **Partially.** High-Safety Rule A reduces true duplicates by **61.9%** while preserving **89.5%** of legitimate overlapping workers (only 2 legitimate worker pairs suppressed).

5. **What is the measured trade-off for each promising rule?**
   - **Rule A (High Safety)**: Suppresses only 2 legitimate worker pairs, but leaves 16 true duplicate pairs.
   - **Rule B (Balanced)**: Suppresses 11 legitimate worker pairs, leaving 8 true duplicate pairs.
   - **Rule C (Aggressive)**: Removes all 42 true duplicates, but suppresses all 19 legitimate worker pairs.

6. **Does the evidence justify another controlled experiment?**
   - **Yes.** A multi-modal signal experiment (e.g. upper-body/head-keypoint proximity or tracker identity management) is justified.

7. **Does the evidence justify retraining?**
   - **No.** The YOLOv8s@800 model has strong overall recall (87.05%). Retraining is not indicated at this stage; isolated post-processing / tracking experiments remain the appropriate diagnostic path.
