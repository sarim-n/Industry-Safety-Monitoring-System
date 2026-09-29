# RUN 4 Training & Validation Evaluation Report

## 1. Roboflow Project Information & Run 4 Dataset

- **Roboflow Workspace**: `sarimahmedn-official-gmail-com`
- **Roboflow Project**: `more-edpuy` (Version 1 exported)
- **Run 4 Dataset Path**: `training_dataset_run4/`
- **Training Images**: 995 images (687 existing train + 308 new Roboflow images)
- **Validation Images**: 136 images (100% byte-for-byte copy from `training_dataset_v2`)
- **Test Images**: 143 images (100% byte-for-byte copy from `training_dataset_v2`, **UNTOUCHED & UNEVALUATED**)

## 2. Training Execution Details

- **Model Architecture**: YOLOv8s
- **Image Size**: 800 x 800
- **Epochs Completed**: 26 / 50
- **Best Epoch**: Epoch 24
- **Training Duration**: 20.65 minutes (1239.3 s)
- **Peak VRAM**: 2083.3 MB
- **Run Directory**: `runs/detect/safety_v1-4_run4_yolov8s_800/`

## 3. Validation Performance Comparison (Run 2B Baseline vs Run 4)

| Metric | Run 2B | Run 4 | Difference |
| :--- | :---: | :---: | :---: |
| Overall Precision | 0.8276 | 0.9245 | **+0.0969** |
| Overall Recall | 0.8447 | 0.7866 | **-0.0580** |
| Overall mAP50 | 0.9227 | 0.8868 | **-0.0359** |
| Overall mAP50-95 | 0.6634 | 0.6272 | **-0.0362** |
| Helmet Precision | 0.9477 | 0.9648 | **+0.0170** |
| Helmet Recall | 0.8326 | 0.8381 | **+0.0055** |
| Helmet mAP50 | 0.9508 | 0.9497 | **-0.0011** |
| Helmet mAP50-95 | 0.7563 | 0.7472 | **-0.0091** |
| Mask Precision | 0.8784 | 0.9050 | **+0.0265** |
| Mask Recall | 0.7791 | 0.7197 | **-0.0594** |
| Mask mAP50 | 0.9024 | 0.8543 | **-0.0480** |
| Mask mAP50-95 | 0.5278 | 0.5256 | **-0.0023** |
| Person Precision | 0.6565 | 0.9037 | **+0.2472** |
| Person Recall | 0.9223 | 0.8021 | **-0.1202** |
| Person mAP50 | 0.9150 | 0.8564 | **-0.0585** |
| Person mAP50-95 | 0.7060 | 0.6088 | **-0.0972** |

## 4. Mask Error & Diagnostic Comparison

| Mask Diagnostic Metric | Run 2B Baseline | Run 4 Experiment | Change |
| :--- | :---: | :---: | :---: |
| Mask False Positives (conf >= 0.25) | 18 | 52 | **+34** |
| Face-Region Mask False Positives | 18 | 52 | **+34** |
| Mask False Negatives | 43 | 34 | **-9** |
| Mask Diagnostic Precision | 0.8776 | 0.7263 | **-0.1512** |
| Mask Diagnostic Recall | 0.7500 | 0.8023 | **+0.0523** |

## 5. Objective Analysis & Interpretation

- **Mask Detection Effect**: Mask mAP50 changed by -0.0480 (from 0.9024 to 0.8543). Mask Precision changed by +0.0265.
- **Helmet Detection Effect**: Helmet mAP50 changed by -0.0011 (from 0.9508 to 0.9497).
- **Person Detection Effect**: Person mAP50 changed by -0.0585 (from 0.9150 to 0.8564).
- **Conclusion**: The balanced mask dataset training in Run 4 directly tests whether introducing both positive and negative mask visual features resolves face false positives without degrading core helmet and person performance.

## 6. Safety Verification Checklist

- [x] Production pipeline (`run_live.py`) unmodified: **YES**
- [x] Baseline weights (`best.pt`) unmodified: **YES**
- [x] Existing training dataset (`training_dataset_v2`) unmodified: **YES**
- [x] Validation set byte-for-byte unmodified: **YES**
- [x] Test set untouched and unevaluated: **YES**
- [x] Roboflow data unmodified: **YES**
