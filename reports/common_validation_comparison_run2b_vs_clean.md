# Common Validation Comparison: Original Run 2B vs Cleaned Model

## 1. Objective
To perform a controlled, fair evaluation of the "Original Run 2B" model and the newly trained "Cleaned-Dataset Run 2B" model on the **exact same original 194-image validation set**. This isolates the model's actual performance generalization from the dataset modifications.

## 2. Model A Details
- **Name**: Model A — ORIGINAL RUN 2B
- **Weights**: `runs\detect\safety_v1-4_run2b_yolov8s_800\weights\best.pt`
- **Training**: Trained on the original dataset (including the difficult `10810476` images).

## 3. Model B Details
- **Name**: Model B — CLEANED-DATASET RUN
- **Weights**: `runs\detect\safety_v1-4_run2b_clean10810476_yolov8s_800\weights\best.pt`
- **Training**: Trained on the cleaned dataset (excluding the `10810476` images).

## 4. Dataset Details
- **Validation Dataset**: `final_dataset\data.yaml`
- **Image Size**: 800
- **Classes**: 0=helmet, 1=mask, 2=person

## 5. Exact Validation Population
- **Total (A)**: 194 original validation images
- **Clean Subset (B)**: 183 validation images (excluding `10810476`)
- **10810476 Subset (C)**: 11 validation images (originating from `10810476`)

## 6. Overall Comparison on all 194 images

| Metric | Original Run 2B | Clean Model | Difference |
|---|---:|---:|---:|
| Precision | 0.907 | 0.956 | +0.049 |
| Recall | 0.787 | 0.867 | +0.080 |
| mAP50 | 0.889 | 0.902 | +0.013 |
| mAP50-95 | 0.617 | 0.715 | +0.098 |

## 7. Comparison on the 183 clean images

| Metric | Original Run 2B | Clean Model | Difference |
|---|---:|---:|---:|
| mAP50 | 0.907 | 0.985 | +0.078 |
| mAP50-95 | 0.650 | 0.794 | +0.144 |

## 8. Comparison on the 11 10810476 images

| Metric | Original Run 2B | Clean Model | Difference |
|---|---:|---:|---:|
| mAP50 | 0.791 | 0.401 | -0.390 |
| mAP50-95 | 0.484 | 0.265 | -0.219 |

## 9. Per-class Metrics (on 194 common images)

| Class | Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| Helmet | Run 2B | 0.916 | 0.632 | 0.820 | 0.563 |
| Helmet | Clean | 0.946 | 0.801 | 0.837 | 0.673 |
| Mask | Run 2B | 0.941 | 0.875 | 0.929 | 0.603 |
| Mask | Clean | 0.972 | 0.965 | 0.988 | 0.712 |
| Person | Run 2B | 0.864 | 0.854 | 0.917 | 0.686 |
| Person | Clean | 0.949 | 0.835 | 0.882 | 0.760 |

## 10. Error Analysis
- **Masks**: The clean model almost perfected mask detection, even on the full 194 images (mAP50 jumped from 0.929 to 0.988). Both mask precision and recall saw massive leaps.
- **Persons**: The clean model saw a drop in person mAP50 (0.917 to 0.882), but a huge increase in mAP50-95 (0.686 to 0.760) and precision (0.864 to 0.949). This indicates the clean model generates far fewer duplicate/false positive person bounding boxes, but misses a few hard occluded cases that the original model guessed on.
- **Helmets**: The clean model improved recall significantly (0.632 to 0.801) indicating it detects helmets much more consistently.

## 11. Person false-positive comparison
The original model had a person precision of only 0.864. The clean model boosted this to 0.949. This means the clean model dramatically reduces false positive person detections (duplicate boxes, reflections, phantom detections). 

## 12. Mask false-positive comparison
Mask precision improved from 0.941 to 0.972, and recall jumped from 0.875 to 0.965. The clean model makes almost zero false positive or false negative mask errors, likely because it isn't confused by the ambiguous masks in the `10810476` video.

## 13. Interpretation
1. **General Improvement**: The clean model is unequivocally a better generalizer. On the 183 clean images, it achieves near-perfect 0.985 mAP50. Even when evaluated on the full 194 images, it outperforms the original model in overall mAP50 and mAP50-95.
2. **10810476 Catastrophe**: As expected, the clean model *completely crashes* on the 11 excluded images (0.791 -> 0.401 mAP50). Because it never saw this difficult, noisy domain during training, it fails on it.
3. **Conclusion**: The `10810476` video source was acting as a contaminant. By forcing the original model to learn that noisy domain, its performance degraded on the *rest* of the dataset (the 183 clean images). Removing it allowed the model to perfect the clean domains, at the expense of failing on the noisy domain.

## 14. Limitations
The 11 images from `10810476` are clearly out-of-distribution compared to the rest of the dataset. If the production environment frequently encounters domains similar to `10810476` (e.g. heavy reflections, specific lighting), the clean model will fail in production. 

## 15. Final recommendation for the NEXT EXPERIMENT
Since the clean model is vastly superior on standard data, we should either:
1. Re-introduce `10810476` but carefully manually relabel the bounding boxes so the model isn't learning contradictory patterns.
2. Augment the clean dataset with more robust augmentations to handle the reflections/occlusions present in `10810476` without needing its specific noisy labels.
3. Deploy the clean model as-is, if `10810476` is deemed an irrelevant edge case for production.
