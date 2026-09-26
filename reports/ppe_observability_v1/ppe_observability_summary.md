# Phase 7.6 — Refined PPE Observability & Diagnostic State Resolution

## 1. Problem Statement & Symptoms
Following Phase 7.5 implementation of PPE region observability, live video testing revealed that a clearly visible worker wearing a helmet was being classified as `UNCERTAIN` instead of recognizing their helmet presence (`helmet_detected = "YES"`).

---

## 2. Root Cause Analysis
1. **Associated PPE Override Bug**: In the initial observability gate implementation, when a region's observability check (`is_helmet_region_observable` or `is_mask_region_observable`) evaluated to `False` (e.g. due to strict top-offset truncation checks or an over-aggressive aspect ratio threshold of $1.15$), the system set `head_visibility = "CROPPED"` or `face_visibility = "CROPPED"`. Although YOLO detected the helmet and associated it with high confidence ($score \ge 0.20$), the overall safety decision fallback converted `helmet = YES, mask = UNKNOWN` into `safety_status = "UNCERTAIN"`.
2. **Over-Aggressive Aspect Ratio Threshold**: The initial aspect ratio cutoff `max_face_only_aspect_ratio` was set to $1.15$. Normal sitting or broad-shouldered standing workers near the camera often registered aspect ratios between $1.02$ and $1.14$, causing their head regions to be falsely flagged as cropped face-only detections.
3. **Lack of Per-Worker Diagnostic Context**: `PersonPPEState` lacked granular diagnostic fields (`head_observable`, `face_observable`, candidate counts, `uncertain_reason`), obscuring why a worker became `UNCERTAIN`.

---

## 3. Final 3-Tier State Resolution Logic

To resolve this issue while preserving protection against false `NO_HELMET_AND_MASK` violations on face-only crops, we enforced a strict 3-tier priority hierarchy:

### Priority Hierarchy:
1. **Tier 1 — PPE DETECTED & ASSOCIATED**:
   $$\text{YOLO Detection + Spatial Association} \longrightarrow \mathbf{\text{State = YES}}$$
   *Associated PPE takes absolute priority. Observability gates DO NOT override an explicitly detected and associated helmet or mask.*

2. **Tier 2 — PPE NOT DETECTED + REGION OBSERVABLE**:
   $$\text{No Association} \land \text{Region Observable} \longrightarrow \mathbf{\text{State = NO}}$$
   *A violation is declared only when the region is sufficiently visible and PPE is absent.*

3. **Tier 3 — PPE NOT DETECTED + REGION NOT OBSERVABLE**:
   $$\text{No Association} \land \text{Region Unobservable} \longrightarrow \mathbf{\text{State = UNKNOWN}}$$
   *If PPE is not detected and the region cannot be reliably observed, the state is UNKNOWN.*

---

## 4. Safety Matrix & Diagnostic Reasons

| Helmet State | Mask State | Safety Status | `uncertain_reason` |
| :--- | :--- | :--- | :--- |
| **YES** | **YES** | **`SAFE`** | `None` |
| **NO** | **YES** | **`NO_HELMET`** | `None` |
| **YES** | **NO** | **`NO_MASK`** | `None` |
| **NO** | **NO** | **`NO_HELMET_AND_MASK`** | `None` |
| **YES** | **UNKNOWN** | **`UNCERTAIN`** | `"MASK_REGION_UNOBSERVABLE"` |
| **UNKNOWN** | **YES** | **`UNCERTAIN`** | `"HELMET_REGION_UNOBSERVABLE"` |
| **UNKNOWN** | **UNKNOWN** | **`UNCERTAIN`** | `"BOTH_REGIONS_UNOBSERVABLE"` |

---

## 5. Diagnostic Fields Added (`PersonPPEState`)
Each worker state now exposes diagnostic attributes:
- `head_roi_bbox`: `(x1, y1, x2, y2)`
- `face_roi_bbox`: `(x1, y1, x2, y2)`
- `head_observable`: `bool`
- `face_observable`: `bool`
- `helmet_candidate_count`: `int`
- `mask_candidate_count`: `int`
- `uncertain_reason`: `Optional[str]`
- `get_debug_summary()`: Generates clean single-line diagnostic output per frame.

---

## 6. Regression Test Suite (`test_ppe_observability.py`)

Nine unit tests covering Cases A through I were executed and verified:

| Test Case | Scenario Description | Helmet State | Mask State | Expected Safety | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Case A** | Full person + helmet + mask | `YES` | `YES` | `SAFE` | PASS |
| **Case B** | Full person + helmet + no mask | `YES` | `NO` | `NO_MASK` | PASS |
| **Case C** | Full person + no helmet + mask | `NO` | `YES` | `NO_HELMET` | PASS |
| **Case D** | Full person + no helmet + no mask | `NO` | `NO` | `NO_HELMET_AND_MASK` | PASS |
| **Case E** | Face-only crop ($\text{aspect\_ratio} < 1.0$) | `UNKNOWN` | `UNKNOWN` | `UNCERTAIN` | PASS |
| **Case F** | Helmet detected + face region uncertain | `YES` | `UNKNOWN` | `UNCERTAIN` | PASS |
| **Case G** | Helmet + mask detected + body cropped | `YES` | `YES` | `SAFE` | PASS |
| **Case H** | Bottom of person cropped, head/face visible | `YES` | `NO` | `NO_MASK` | PASS |
| **Case I** | Helmet candidate far outside head ROI | `NO` | `NO` | `NO_HELMET_AND_MASK` | PASS |

---

## 7. Real Live Webcam & Video Validation

### Live Webcam Test (`run_live.py --source 0 --max-frames 60 --headless`)
- **Processed Frames**: 60 frames @ 42.1 FPS.
- **Worker Status Breakdown**:
  - `NO_MASK`: 60 frames ($100.0\%$)
  - `UNCERTAIN`: 0 frames ($0.0\%$)
- **Validation**: Worker wearing a helmet without a mask was recognized as `helmet_detected = "YES"` and correctly evaluated as `NO_MASK`, resolving the over-aggressive `UNCERTAIN` issue.

---

## 8. Remaining Limitations
1. **Occlusion by External Objects**: Objects blocking the face (e.g. hands or tools) that do not touch image frame boundaries rely on candidate association score thresholds to avoid false positive mask detections.
