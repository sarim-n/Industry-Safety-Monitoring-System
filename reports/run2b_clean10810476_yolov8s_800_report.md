# Cleaned Dataset Rerun: YOLOv8s @ 800 (Run 2B Cleaned)

## 1. Dataset Used
- **Original Dataset**: `C:\Users\sarim\safety monitoring\final_dataset`
- **Cleaned Dataset**: `C:\Users\sarim\safety monitoring\final_dataset_without_10810476`

## 2. Dataset Counts
- **Train**: 722 images (50 removed)
- **Validation**: 183 images (11 removed)
- **Test**: 0 images

## 3. Removed-Source Information
All images originating from the video `10810476-hd_1920_1080_30fps.mp4` were removed to investigate their impact on model performance, particularly regarding duplicate detections and false positives.

## 4. Training Configuration
- **Model**: YOLOv8s
- **Image Size**: 800
- **Epochs**: 50
- **Patience**: 15
- **Batch Size**: 4
- **Optimizer**: auto
- **Seed**: 42
- **AMP**: enabled
- **Workers**: 2
- **Augmentation**:
  - hsv_h = 0.015
  - hsv_s = 0.7
  - hsv_v = 0.4
  - degrees = 0
  - translate = 0.1
  - scale = 0.5
  - fliplr = 0.5
  - mosaic = 1.0
  - mixup = 0.0

## 5. Training Duration
- **Duration**: ~48.6 minutes (0.810 hours)
- **Epochs Completed**: 50/50

## 6. Peak VRAM
- **Peak VRAM**: 1.97 GB

## 7. Best Epoch
- **Best Epoch**: ~50 (based on final metric ascent)

## 8. Validation Metrics
**Overall**:
- **Precision**: 0.967
- **Recall**: 0.954
- **mAP50**: 0.986
- **mAP50-95**: 0.793
- **Inference Speed**: 8.6ms per image

## 9. Per-class Metrics
- **Helmet**:
  - Precision: 0.971
  - Recall: 0.958
  - mAP50: 0.982
  - mAP50-95: 0.799
- **Mask**:
  - Precision: 0.975
  - Recall: 0.955
  - mAP50: 0.987
  - mAP50-95: 0.712
- **Person**:
  - Precision: 0.957
  - Recall: 0.948
  - mAP50: 0.988
  - mAP50-95: 0.868

## 10. Comparison Against Original Run 2B
| Metric | Original Run 2B | Cleaned Run 2B | Difference |
|---|---|---|---|
| mAP50 | 0.9227 | 0.9860 | +0.0633 |
| mAP50-95 | 0.6634 | 0.7930 | +0.1296 |
| Helmet (mAP50) | 0.9508 | 0.9820 | +0.0312 |
| Mask (mAP50) | 0.9024 | 0.9870 | +0.0846 |
| Person (mAP50) | 0.9150 | 0.9880 | +0.0730 |

## 11. Error Analysis
Based on the drastic metric improvements, removing the `10810476` images effectively eliminated the primary source of error in the dataset:
- **Mask False Positives/Negatives:** Mask mAP50 improved by ~8.5%, indicating that the dropped validation frames contained highly ambiguous or incorrectly labeled masks.
- **Person Duplicate Detections:** Person mAP50 improved by ~7.3%, and mAP50-95 for Person is exceptionally high (0.868). This implies the `10810476` dataset was responsible for the majority of messy person detections or severe occlusions/reflections causing duplicate boxes.
- **Helmet Detections:** Improved slightly (+3.1%), but was already very high, showing it wasn't the main struggle of this specific video.

## 12. Interpretation
Removing `10810476` frames significantly boosts apparent performance. However, because we dropped 11 validation images, the remaining validation set is fundamentally "easier." The model did not magically improve its architecture; rather, the most challenging (or perhaps incorrectly annotated) visual distribution was removed from the evaluation entirely. This confirms that `10810476` was a major outlier dragging down the aggregate scores.

## 13. Limitations
**The validation set changed because 11 validation images originating from 10810476 were removed. Therefore, this run should be interpreted as a cleaned-dataset rerun rather than a perfectly controlled model-to-model comparison.**
