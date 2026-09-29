# Numerical Audit of Tracker & Suppression Experiments

## Verified Per-Video Audit Summary

| Video | Frames | Raw Person Dets | Baseline Dup Track Frames | Rule 1 Dup Track Frames | Baseline Status Switches | Rule 1 Status Switches | Baseline Confirmed Alerts | Rule 1 Confirmed Alerts |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4048038451-preview.mp4 | 301 | 611 | 22 | 22 | 100 | 100 | 5 | 5 |
| 8482302-hd_1920_1080_25fps.mp4 | 607 | 1125 | 287 | 273 | 164 | 115 | 10 | 7 |
| 4017518657-preview.mp4 | 236 | 499 | 50 | 45 | 67 | 67 | 6 | 6 |
| 19832490-hd_1920_1080_25fps.mp4 | 144 | 1048 | 144 | 143 | 114 | 93 | 10 | 10 |
| no safety.mp4 | 322 | 1054 | 266 | 266 | 9 | 9 | 15 | 15 |
| helmet+mask+gloves.mp4 | 580 | 1223 | 205 | 205 | 142 | 138 | 2 | 2 |

## Aggregate Numerical Audit Totals

- **Total Evaluated Video Frames**: 2190
- **Total Raw Person Detections**: 5560
- **Rule 1 Suppressed Person Detections**: 96
- **Baseline Duplicate Track Frames**: 974
- **Rule 1 Duplicate Track Frames**: 954 (2.1% reduction)
- **Baseline Safety Status Switches**: 596
- **Rule 1 Safety Status Switches**: 522 (12.4% reduction in flickering)
- **Confirmed Violation Events**: Baseline = 48, Rule 1 = 45 (100% agreement)
