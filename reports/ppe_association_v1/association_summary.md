# PPE Association System — Phase 1 Offline Validation Report

## 1. Executive Summary & Setup
- **Model**: `runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt` (YOLOv8s @ 800)
- **Validation Set**: `training_dataset_v2/images/val` (136 images)
- **Operating Confidence Thresholds**:
  - `PERSON_CONF` = 0.50
  - `HELMET_CONF` = 0.25
  - `MASK_CONF` = 0.20
- **Module Architecture**: `src/safety/ppe_association.py` (`PPEAssociator` class)
- **TEST set status**: Untouched and NOT evaluated.

## 2. Association & Safety State Statistics

- **Total Validation Images Processed**: 136
- **Total Person Detections (conf >= 0.50)**: 428 (3.15 persons / image)
- **Total Helmet Detections (conf >= 0.25)**: 169
- **Total Mask Detections (conf >= 0.20)**: 168
- **Average PPE Detections / Image**: 2.48

### PPE Association Breakdown
- **Helmets Associated with a Person**: 141 (83.4%)
- **Helmets Not Associated (Standalone/Unassigned)**: 28
- **Masks Associated with a Person**: 130 (77.4%)
- **Masks Not Associated (Standalone/Unassigned)**: 38

### Per-Person Item Detection Counts
- **Helmet Status**: YES = 141 | NO = 260 | UNKNOWN = 27
- **Mask Status**: YES = 130 | NO = 298 | UNKNOWN = 0

### Derived Safety Status Distribution
- **SAFE** (Helmet YES, Mask YES): 47 (11.0%)
- **NO_HELMET** (Helmet NO, Mask YES): 81 (18.9%)
- **NO_MASK** (Helmet YES, Mask NO): 94 (22.0%)
- **NO_HELMET_AND_MASK** (Helmet NO, Mask NO): 179 (41.8%)
- **UNCERTAIN** (Visibility/Cropping UNKNOWN): 27 (6.3%)

## 3. Ground-Truth Association Evaluation
Ground truth ownership was established by evaluating spatial containment between GT persons and GT helmets/masks on the validation set:

- **GT Persons Evaluated**: 386
- **Helmet Association**: Correct = 286, Missed = 35, Incorrect = 7 (Accuracy = 74.1%)
- **Mask Association**: Correct = 295, Missed = 37, Incorrect = 6 (Accuracy = 76.4%)

## 4. Partial / Cropped Person Analysis
Offline inspection of cropped workers, boundary-touching instances, and occlusions:

1. **Cropped Lower Body (Bottom Boundary)**: In instances where workers touch the bottom image boundary (legs cropped), the system correctly maintains `head_visibility = VISIBLE` and evaluates PPE correctly as `SAFE` or `NO_HELMET`.
2. **Cropped Top Head (Top Boundary)**: When workers enter or leave the top frame with `y1 <= 2px` or head height `< 10px`, the system sets `head_visibility = CROPPED` and marks unassigned helmets as `UNKNOWN`. This prevents generating false `NO_HELMET` violations on heads outside the field of view.
3. **Partial Upper Torso / Machinery Occlusion**: Workers behind machinery with visible heads are successfully associated with their helmets and masks.
4. **Crowded Overlapping Workers**: 1-to-1 deterministic greedy matching prevents a single detected helmet from being double-assigned to adjacent workers.

## 5. Major Association Failure Patterns & Future Tuning Recommendations
1. **Angled Indoor Overhead Cameras**: In severe overhead camera perspectives, the face region is compressed vertically. Slightly relaxing `FACE_REGION_TOP_RATIO` or `FACE_REGION_HEIGHT_RATIO` can further improve mask association on angled indoor views.
2. **Unassociated PPE Detections**: 28 helmets and 38 masks were detected but not associated with a person box. These primarily stem from distant workers where the person box confidence fell below `0.50` while the PPE item exceeded threshold. Lowering `person_conf` to `0.40–0.45` or adding a secondary person search for unassigned PPE could reclaim these.

## 6. Confirmations
- **TEST Set**: TEST set was NOT loaded, accessed, or evaluated.
- **Dataset & Models**: Dataset images, labels, splits, and `best.pt` weights were NOT modified.
