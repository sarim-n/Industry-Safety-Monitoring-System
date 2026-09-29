# Person Duplicate Bounding Box Diagnostic Report

## 1. Executive Summary

A rigorous, evidence-based diagnostic investigation was conducted to determine why the live system produces **more than one `person` bounding box around the same human**.

### Key Findings:
- **Duplicate Person Boxes Do Occur**: 67 overlapping person box candidate pairs ($\text{IoU} \ge 0.50$) were identified across 50 unique validation images (out of 136 total images).
- **True Duplicate Proportion**: **48 pairs ($71.6\%$) are TRUE DUPLICATES** where two predicted `person` boxes correspond to the **exact same ground-truth human**.
- **Primary Root Cause (YOLO / NMS Layer)**: Ultralytics' default Non-Maximum Suppression (NMS) threshold of `iou = 0.70` allows duplicate person boxes with IoU between $0.504$ and $0.696$ to survive NMS unsuppressed. All 67 candidate duplicate pairs lie strictly in the $0.50 \le \text{IoU} < 0.70$ band (zero pairs exist at $\text{IoU} \ge 0.70$).
- **Tracker Amplification Layer**: When a duplicate YOLO person box enters `TemporalConfirmationEngine`, greedy track matching assigns one box to the primary track and spawns a **NEW Track ID** for the duplicate box. Because `max_missed_frames = 10`, the spawned duplicate track persists for up to 10 frames even after the duplicate box disappears.
- **Rendering Layer**: The OpenCV overlay and React dashboard draw all active tracks from `TemporalConfirmationEngine`, visually displaying multiple bounding boxes for the same human.

---

## 2. Validation Dataset

- **Validation Dataset Path**: `training_dataset_v2/images/val`
- **Total Validation Images**: 136 images
- **Total Ground-Truth Persons**: 386 person instances
- **Total Predicted Persons** (`PERSON_CONF >= 0.50`): 428 person instances
- **Images Containing Overlapping Person Candidates**: 50 images ($36.8\%$ of validation images)

---

## 3. Duplicate Statistics

Evaluating all predicted person bounding box pairs at `PERSON_CONF >= 0.50`:

| IoU Threshold | Total Candidate Pairs | Unique Images | Same-GT True Duplicate Pairs | Legitimate Overlap Pairs | Partial / Fragment Pairs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$\text{IoU} \ge 0.90$** | 0 | 0 | 0 | 0 | 0 |
| **$\text{IoU} \ge 0.80$** | 0 | 0 | 0 | 0 | 0 |
| **$\text{IoU} \ge 0.70$** | 0 | 0 | 0 | 0 | 0 |
| **$\text{IoU} \ge 0.50$** | **67** | **50** | **48** ($71.6\%$) | **19** ($28.4\%$) | 0 |

### Critical NMS Boundary Observation:
**Zero ($0$) duplicate pairs exist at $\text{IoU} \ge 0.70$**, proving that Ultralytics' NMS (`iou=0.70`) successfully suppresses duplicates above $0.70$ IoU. However, all 67 duplicate pairs fall in the $0.500 \le \text{IoU} \le 0.696$ window, slipping directly past default NMS.

---

## 4. Confidence Analysis

For the 48 True Duplicate Person Pairs:

| Metric | Primary Box ($Box A$) | Duplicate Box ($Box B$) | Confidence Difference ($\text{conf}_A - \text{conf}_B$) |
| :--- | :--- | :--- | :--- |
| **Mean Confidence** | **0.784** | **0.616** | **0.168** |
| **Median Confidence** | 0.793 | 0.601 | 0.170 |
| **Min Confidence** | 0.587 | 0.509 | 0.020 |
| **Max Confidence** | 0.964 | 0.830 | 0.429 |

### Distribution Pattern:
In $83.3\%$ of true duplicate cases, the pair consists of **one high-confidence primary box ($\text{conf} \ge 0.75$) and one lower-confidence secondary box ($0.51 \le \text{conf} \le 0.68$)**.

---

## 5. Visual Causes

Analysis of true duplicate visualizations (`reports/person_duplicate_analysis_v1/`) reveals three main visual causes:

1. **Full-Body vs Upper-Body Dual Predictions ($54.2\%$ of true dups)**:
   YOLO predicts one box covering the full standing/sitting person (head to feet/knees) AND a second overlapping box covering only the torso/upper body (head to waist).
2. **Bottom-Cropped Workers ($27.1\%$ of true dups)**:
   When a worker's legs are cut off at the bottom frame boundary, YOLO outputs two candidate boxes of slightly different heights around the same worker.
3. **Close-Up Workers Near Camera ($18.7\%$ of true dups)**:
   Workers close to the camera trigger two overlapping predictions due to wide shoulder/body feature coverage.

---

## 6. YOLO vs Tracking vs Rendering Layer Breakdown

```
Raw YOLO Detections (NMS Leakage: 0.50 <= IoU < 0.70)
                      ↓
PPE Association (Preserves both boxes: conf >= 0.50)
                      ↓
Temporal Confirmation (Matches Box A -> Track 1; Spawns Box B -> Track 2)
                      ↓
Rendering Layer (Draws both Track 1 and Track 2 on frame)
```

1. **YOLO Layer**: Produces two raw detections for the same human because default NMS is set to `iou = 0.70`, letting $0.50 \le \text{IoU} < 0.70$ duplicates pass.
2. **Tracking Layer**: Receives both boxes, matches the primary box to the existing track ID, and creates a new track ID for the secondary box. Retains the new track ID for up to 10 missed frames.
3. **Rendering Layer**: Renders all active tracks, displaying two overlapping bounding boxes with different track IDs over the same human.

---

## 7. Representative Evidence

### Representative Validation Images (True Duplicates):
- `19832490_hd_1920_1080_25fps_t00-96_f0024_jpg.rf.uLKsWCF1xoyowB0Sy520.jpg`: $\text{IoU}=0.688$, $\text{conf}_A=0.885$, $\text{conf}_B=0.774$. Full-body vs upper-body duplicate.
- `8482302_hd_1920_1080_25fps_t00-00_f0000_jpg.rf.nzZ3fDlKWaIBmXllCm0Q.jpg`: $\text{IoU}=0.626$, $\text{conf}_A=0.913$, $\text{conf}_B=0.619$. Standing worker duplicate.
- `4017518657_preview_t04-90_f0147_jpg.rf.KNOIXW0DLHHTTrUoVsOA.jpg`: $\text{IoU}=0.635$, $\text{conf}_A=0.887$, $\text{conf}_B=0.624$. Bottom-cropped worker duplicate.
- `gloves_mask_t04-50_f0108_jpg.rf.LvLxWrAh6Ah4mrFX6l76.jpg`: $\text{IoU}=0.658$, $\text{conf}_A=0.943$, $\text{conf}_B=0.514$. High vs low confidence duplicate.

### Representative Video Stream Frames (`4048038451-preview.mp4`):
- `Frame 8`: YOLO produces 2 person detections for a worker ($\text{IoU}=0.58$). Tracker matches `Track 1` and spawns `Track 4`.
- `Frames 9–18`: `Track 4` remains active in rendering loop for 10 frames due to `max_missed_frames = 10`.

---

## 8. Recommended Next Steps (For Future Implementation)

*Note: No changes have been implemented in this phase.*

Based on empirical evidence, recommended experiments for future implementation include:

1. **Person-Specific Non-Maximum Suppression (NMS)**:
   Apply a person-class duplicate suppression gate with a tighter IoU threshold (e.g. `person_nms_iou = 0.50`) in post-processing (`PPEAssociator` or inference call) to suppress secondary person boxes with $\text{IoU} \ge 0.50$.
2. **IoU-Based Track Merging in Temporal Engine**:
   In `TemporalConfirmationEngine`, suppress or merge new track candidate creation if a new detection overlaps an existing active track with $\text{IoU} \ge 0.50$.
