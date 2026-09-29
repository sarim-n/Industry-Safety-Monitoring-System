# Person Duplicate Box Suppression Experiment Summary (v1)

> [!IMPORTANT]
> **STATUS: CONTROLLED EXPERIMENT ONLY.**
> Production code, model weights, datasets, annotations, confidence thresholds, global NMS settings, PPE association, and temporal tracking remain **UNMUTATED and FROZEN**. No production fix is committed until this experiment is reviewed.

---

## 1. Executive Summary & Diagnostic Context

Prior diagnostic analysis identified **48 true duplicate person bounding boxes** across the validation dataset, all occurring within the IoU range of $0.504 \le \text{IoU} < 0.696$ (primarily one high-confidence box + one lower-confidence box). Because standard YOLOv8s NMS operates at a global `iou=0.70`, these duplicate pairs leak through single-frame detection and are subsequently amplified by the temporal tracking stage.

To test whether a post-YOLO, pre-PPE association **person-class-only IoU suppression filter** can eliminate duplicate person boxes, we conducted an empirical sweep across five IoU thresholds: **0.40, 0.45, 0.50, 0.55, 0.60**.

When two person boxes overlap above the experimental threshold $t$:
1. The **higher-confidence box is kept**.
2. The **lower-confidence box is suppressed**.
3. Helmet and mask detections remain completely untouched.

---

## 2. Validation Set Experimental Results (136 Images, 386 GT Persons)

Evaluated on the complete validation dataset using `PERSON_CONF = 0.50`.

| Setting | Total Pred Persons | Dup Pairs Remaining ($\text{IoU} \ge 0.50$) | True Dup Pairs Remaining | Legitimate Overlaps Suppressed | Person Precision | Person Recall | Person F1 | GT Persons Missed | Avg Person Boxes / GT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASELINE** | **428** | **67** | **42** | **0** | **0.7850** | **0.8705** | **0.8256** | **50** | **0.9974** |
| **IoU < 0.40** | 353 | 0 | 0 | **19** (100%) | 0.8839 | 0.8083 | 0.8444 | 74 (+24) | 0.8212 |
| **IoU < 0.45** | 361 | 0 | 0 | **19** (100%) | 0.8809 | 0.8238 | 0.8514 | 68 (+18) | 0.8420 |
| **IoU < 0.50** | 365 | 0 | 0 | **19** (100%) | 0.8767 | 0.8290 | 0.8522 | 66 (+16) | 0.8472 |
| **IoU < 0.55** | 381 | 16 | 8 | **11** (57.9%) | 0.8609 | 0.8497 | 0.8553 | 58 (+8) | 0.8886 |
| **IoU < 0.60** | 394 | 30 | 18 | **10** (52.6%) | 0.8325 | 0.8497 | 0.8410 | 58 (+8) | 0.9145 |

---

## 3. Live Video Experimental Results (`4048038451-preview.mp4`, 301 Frames)

Evaluated on the full 301-frame live test sequence through PPE Association and Temporal Tracking.

| Setting | Frames with Duplicate Persons ($\text{IoU} \ge 0.50$) | Avg Active Tracks / Frame | Unique Track IDs Created | Frames with Duplicate Track IDs ($\text{IoU} \ge 0.50$) | Pipeline FPS | Latency / Frame |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASELINE** | **2** | **2.16** | **6** | **22** | **5332.7** | **0.19 ms** |
| **IoU < 0.40** | 0 | 2.09 | 4 | 0 | 7097.9 | 0.14 ms |
| **IoU < 0.45** | 0 | 2.09 | 4 | 0 | 6807.6 | 0.15 ms |
| **IoU < 0.50** | 0 | 2.09 | 4 | 0 | 7324.3 | 0.14 ms |
| **IoU < 0.55** | 0 | 2.09 | 4 | 0 | 7784.8 | 0.13 ms |
| **IoU < 0.60** | 0 | 2.09 | 4 | 0 | 8887.7 | 0.11 ms |

---

## 4. Critical Findings & Trade-off Analysis

> [!WARNING]
> **CRITICAL DISCOVERY: PURE BBOX IoU CANNOT DISTINGUISH TRUE DUPLICATES FROM LEGITIMATE OVERLAPPING WORKERS.**

1. **Strict Thresholds ($t \le 0.50$) Cause Severe Worker Suppression**:
   - Setting $t = 0.40, 0.45, 0.50$ successfully removes **100% of true duplicate person pairs** (0 true duplicates remaining).
   - However, it **incorrectly suppresses ALL 19 legitimate overlapping worker pairs** ($100\%$ collateral damage).
   - This causes person recall to drop sharply from **87.05% down to 80.83%–82.90%**, missing **16 to 24 real ground-truth workers** in crowded or multi-worker industrial scenes.

2. **Moderate Thresholds ($t = 0.55, 0.60$) Suffer Dual Failure**:
   - At $t = 0.55$ and $t = 0.60$, **10 to 11 legitimate overlapping worker pairs are STILL incorrectly suppressed** (missing 8 real ground-truth workers).
   - Simultaneously, **8 to 18 true duplicate person pairs STILL survive**, failing to solve the duplicate problem.

3. **Fundamental Cause**:
   - In industrial camera feeds, legitimate construction workers standing side-by-side or walking past each other routinely share bounding box overlaps of **$0.45 \le \text{IoU} < 0.65$**.
   - Because standard greedy IoU NMS relies exclusively on box coordinates and confidence scores, **it has zero identity or spatial structure awareness** to differentiate a duplicate prediction of worker A from two real workers standing close together.

---

## 5. Artifacts Created

The complete experimental output is saved in the workspace:

- **Experiment CSV**: [suppression_experiment.csv](file:///c:/Users/sarim/safety%20monitoring/reports/person_duplicate_analysis_v1/suppression_experiment.csv)
- **Before/After Visual Comparison Frames**: [before_after/](file:///c:/Users/sarim/safety%20monitoring/reports/person_duplicate_analysis_v1/before_after/)
  - `frame_0045_baseline.jpg` vs `frame_0045_suppressed_iou0.50.jpg`
  - `frame_0095_baseline.jpg` vs `frame_0095_suppressed_iou0.50.jpg`
  - `frame_0145_baseline.jpg` vs `frame_0145_suppressed_iou0.50.jpg`
  - `frame_0195_baseline.jpg` vs `frame_0195_suppressed_iou0.50.jpg`
  - `frame_0245_baseline.jpg` vs `frame_0245_suppressed_iou0.50.jpg`

---

## 6. Conclusion & Recommendation

> [!CAUTION]
> **RECOMMENDATION: DO NOT ADOPT A SIMPLE POST-YOLO PERSON IoU THRESHOLD IN PRODUCTION.**

- **No single IoU threshold in the range [0.40, 0.60] successfully removes duplicate person boxes while preserving legitimate overlapping workers.**
- Any standalone person IoU suppression filter will either:
  1. Miss real workers in multi-person industrial scenarios ($t \le 0.50$), OR
  2. Fail to eliminate duplicate person boxes ($t \ge 0.55$).

### Next Step / Architectural Alternatives for Future Phase:
To suppress true duplicates without harming legitimate workers, future work should consider multi-signal suppression (e.g., checking upper-body/head center distance, keypoint overlap, or tracker-level soft-suppression) rather than raw bounding-box IoU.

Production codebase status remains **100% UNTOUCHED and FROZEN**.
