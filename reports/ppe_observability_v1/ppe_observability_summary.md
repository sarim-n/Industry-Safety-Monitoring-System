# Phase 7.5 — PPE Observability Summary Report

## 1. Problem Statement
In the Industrial Safety Monitoring System, when a camera frame presents a person detection containing only a face/head crop or a heavily cropped upper body, YOLO generates a `person` detection box (`cls=2`). Previously, the PPE association logic evaluated missing helmet and mask detections as absent (`helmet_detected="NO"`, `mask_detected="NO"`), resulting in false safety violation classifications of `NO_HELMET_AND_MASK`.

In safety-critical computer vision systems:
$$\text{Absence of Detection} \neq \text{Absence of PPE}$$
when the relevant PPE region is not sufficiently observable.

---

## 2. Root Cause Analysis
1. **Top Boundary Margin Gap**: The boundary check (`touches_top`) was defined as `p_box.y1 <= boundary_margin_px` ($3.0\text{ px}$). When a person's head was near the top frame edge ($3.0\text{ px} < y_1 \le ~30\text{ px}$), `touches_top` evaluated to `False`. However, the helmet region (`head_y1 = y1 - 0.15 * height`) extended above the top image canvas ($y < 0$), truncating the helmet observation area while `head_visibility` was incorrectly classified as `"VISIBLE"`.
2. **Face-Only / Head-Only Person Bounding Box Misclassification**: When YOLO detected a `person` box around a close-up face or head, $y_1$ did not touch the top frame edge ($y_1 > 3.0$) and box width exceeded `min_person_width_px` ($12\text{ px}$). The system marked both `head_visibility` and `face_visibility` as `"VISIBLE"`. Because a face-only crop lacks the scalp/head top region where helmets sit, no helmet detection was present, leading to `helmet_detected = "NO"` and `mask_detected = "NO"`, causing false `NO_HELMET_AND_MASK` classifications.
3. **Lack of Specific PPE Region Observability Helpers**: Observability was coupled to crude whole-person boundary flags rather than evaluating whether the specific geometric region required for helmet observation (`head_roi` top extent) or mask observation (`face_roi` vertical extent) was sufficiently visible inside the image canvas.

---

## 3. Existing Logic vs. New Observability Rule

### Existing Logic (Before Fix)
```python
# Crude boundary check
if touches_top and (p_box.y1 <= 2.0 or head_roi.height < min_head_height_px):
    head_vis = "CROPPED"
elif p_box.width < min_person_width_px:
    head_vis = "CROPPED"
else:
    head_vis = "VISIBLE"
```

### New Observability Rule (Phase 7.5 Fix)
Before classifying PPE presence/absence, the system evaluates individual region observability:

$$\text{Person Detected} \longrightarrow \text{Is PPE Region Observable?} \begin{cases} \text{NO} & \longrightarrow \text{PPE State = UNKNOWN} \longrightarrow \text{Safety = UNCERTAIN} \\ \text{YES} & \longrightarrow \text{Evaluate Detection} \longrightarrow \text{SAFE / NO\_HELMET / NO\_MASK / BOTH} \end{cases}$$

---

## 4. Geometry-Based Observability Implementation

### A. Helmet Region Observability (`is_helmet_region_observable`)
A helmet region is observable if and only if:
1. **Top Boundary Truncation**: Top offset extent $(y_1 - \text{top\_offset\_px})$ is $\ge \text{boundary\_margin\_px}$ ($3.0\text{ px}$), ensuring the scalp/head top area is inside the canvas.
2. **Aspect Ratio Check**: Person box aspect ratio $\frac{\text{height}}{\text{width}} \ge \text{max\_face\_only\_aspect\_ratio}$ ($1.15$), identifying body/torso context rather than a face-only crop.
3. **Width Check**: Person box width $\ge \text{min\_person\_width\_px}$ ($12.0\text{ px}$).
4. **Visible Head ROI Height**: Visible height of `head_roi` inside the frame canvas $\ge \text{min\_head\_height\_px}$ ($10.0\text{ px}$).

### B. Mask Region Observability (`is_mask_region_observable`)
A mask region is observable if and only if:
1. **Vertical Bounds**: `face_roi.y1` $\ge 3.0\text{ px}$ and `face_roi.y2` $\le (\text{img\_height} - 3.0\text{ px})$.
2. **Width Check**: Person box width $\ge \text{min\_person\_width\_px}$ ($12.0\text{ px}$).
3. **Visible Face ROI Height**: Visible height of `face_roi` inside the frame canvas $\ge \text{min\_face\_height\_px}$ ($8.0\text{ px}$).

### C. Safety Decision Matrix
- `helmet = YES, mask = YES` $\longrightarrow$ `SAFE`
- `helmet = NO, mask = YES` $\longrightarrow$ `NO_HELMET`
- `helmet = YES, mask = NO` $\longrightarrow$ `NO_MASK`
- `helmet = NO, mask = NO` $\longrightarrow$ `NO_HELMET_AND_MASK`
- `helmet = UNKNOWN` or `mask = UNKNOWN` $\longrightarrow$ `UNCERTAIN`

---

## 5. Preservation of Valid Partial-Person Cases
Bottom-cropped workers (e.g., workers with legs cut off at the bottom frame boundary) maintain `head_visibility = "VISIBLE"` and `face_visibility = "VISIBLE"` as long as their top head and face regions are fully inside the image canvas. Bottom cropping alone does NOT mark a worker as `UNCERTAIN`.

---

## 6. Regression Test Suite & Verification

### Unit Test Suite (`test_ppe_observability.py`)
Nine deterministic unit tests covering Cases A through I were added:

| Test Case | Description | Expected Status | Result |
| :--- | :--- | :--- | :--- |
| **Case A** | Full person visible, helmet + mask visible | `SAFE` | PASS |
| **Case B** | Full person visible, helmet absent, mask visible | `NO_HELMET` | PASS |
| **Case C** | Full person visible, helmet visible, mask absent | `NO_MASK` | PASS |
| **Case D** | Full person visible, neither PPE detected | `NO_HELMET_AND_MASK` | PASS |
| **Case E** | Face-only crop (low aspect ratio) | `UNCERTAIN` | PASS |
| **Case F** | Top boundary cropped ($y_1 \le 3.0$) | `UNCERTAIN` | PASS |
| **Case G** | Bottom half cropped, head + face clearly visible | `NO_MASK` (Normal Eval) | PASS |
| **Case H** | Face ROI height $< 8\text{ px}$ | `UNCERTAIN` | PASS |
| **Case I** | Helmet top extent $y_1 - \text{offset} < 0$ | `UNCERTAIN` | PASS |

---

## 7. Validation Results

### Offline Association Validation (`validate_ppe_association.py`)
- **Validation Dataset**: 136 images / 386 ground-truth person instances.
- **Helmet Association Accuracy**: $379/386$ ($98.19\%$).
- **Mask Association Accuracy**: $377/386$ ($97.67\%$).
- **UNCERTAIN Status Count**: Adjusted from $21$ to $23$ instances, correctly reclassifying face-only/top-cropped edge cases from false violations to `UNCERTAIN`.

### Temporal Confirmation Validation (`validate_temporal_confirmation.py`)
- **Video Sequences Evaluated**: 7 sequences (136 frames).
- **Suppressed Transient Alerts**: 185 raw transient alerts suppressed.
- **UNCERTAIN Streaks**: Correctly broke violation streaks without generating false positive confirmed alert events.

### Live Video Pipeline Benchmark (`run_live.py`)
- **Test Source**: `data_collection/videos/4048038451-preview.mp4` (301 frames).
- **Face-Only Crops**: Verified that close-up face crops evaluate to `UNCERTAIN` instead of `NO_HELMET_AND_MASK`.
- **Full Workers**: Confirmed workers with missing PPE generate valid, actionable alerts.

---

## 8. Known Limitations
1. **Severe Occlusion by Heavy Objects**: If an un-annotated heavy object (e.g. large metal beam) blocks a worker's head in a mid-body crop, geometry alone sees $y_1$ inside the frame. Additional occlusion segmentation could be explored in future work if non-frame occlusions are required.
2. **Extreme Camera Angles**: Top-down bird's-eye views where aspect ratios approach ~1.0 for standing workers should maintain custom aspect ratio bounds in configuration.
