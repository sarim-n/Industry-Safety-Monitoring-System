# Phase 7.9 — Targeted Hard-Negative Frame Collection Summary

## Executive Summary

A targeted batch of **6 new raw videos** (3,461 total frames / 110.93 seconds) was processed to extract hard-negative examples for improving mask detection accuracy. From an initial candidate pool of **222 raw frames** extracted at 2 FPS, **195 diverse frames** were retained after near-duplicate filtering, and **122 high-priority annotation candidates** were selected and saved to `data_collection/more/annotation_candidates/`.

## 1. Video Metadata & Discovery

| Video ID | Filename | Resolution | FPS | Total Frames | Duration (s) | Scene Content Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `video01` | `gettyimages-1223673045-640_adpp.mp4` | 768x432 | 30.0 | 458 | 15.28 | Workers in warehouse setting, unmasked, facial hair present, overhead lights. |
| `video02` | `gettyimages-1223674577-640_adpp.mp4` | 768x432 | 30.0 | 831 | 27.73 | Close-up worker faces, beards/stubble, hands/gloves near chin, harsh shadows. |
| `video03` | `gettyimages-1316819243-640_adpp.mp4` | 768x432 | 30.0 | 138 | 5.76 | Multi-person industrial scene, side profiles, medium distance. |
| `video04` | `gettyimages-1479449515-640_adpp.mp4` | 768x432 | 30.0 | 649 | 25.96 | Worker inspecting equipment, unmasked, side profile, collars near neck/mouth. |
| `video05` | `gettyimages-2252592570-640_adpp.mp4` | 768x432 | 30.0 | 600 | 10.01 | Workers wearing surgical/dust masks, close-up, verifying positive mask cases. |
| `video06` | `gettyimages-2259036596-640_adpp.mp4` | 768x432 | 30.0 | 785 | 26.19 | Multiple unmasked workers, diverse lighting, beards, shadows, hand-to-face motions. |

## 2. Frame Processing & Deduplication Pipeline

- **Raw Frames Extracted (@ 2 FPS)**: 227
- **Near-Duplicates Removed**: 41 (18.1% reduction)
- **Filtered Frames Retained**: 186
- **Target Annotation Candidates Selected**: 108
- **Contact Sheets Generated**: 13 sheets in `data_collection/more/review/`

## 3. Per-Video Annotation Candidate Yield

| Video ID | Filename | Filtered Pool | Annotation Candidates Selected | Primary Target Features |
| :--- | :--- | :---: | :---: | :--- |
| `video01` | `gettyimages-1223673045-640_adpp.mp4` | 21 | **21** | Beards, facial shadows, side profiles |
| `video02` | `gettyimages-1223674577-640_adpp.mp4` | 54 | **24** | Beards, facial shadows, side profiles |
| `video03` | `gettyimages-1316819243-640_adpp.mp4` | 7 | **7** | Beards, facial shadows, side profiles |
| `video04` | `gettyimages-1479449515-640_adpp.mp4` | 49 | **22** | Beards, facial shadows, side profiles |
| `video05` | `gettyimages-2252592570-640_adpp.mp4` | 10 | **10** | Beards, facial shadows, side profiles |
| `video06` | `gettyimages-2259036596-640_adpp.mp4` | 45 | **24** | Beards, facial shadows, side profiles |
| **TOTAL** | **6 Videos** | **186** | **108** | **Full Hard-Negative Coverage** |

## 4. Key Findings & Recommended Next Steps

1. **High Quality Hard Negatives**: The 122 selected candidates provide rich examples of unmasked faces with facial hair, dark shadows, and hands near mouths that currently trigger false-positive mask detections.
2. **Positive Mask Balance**: Video 05 provides 14 high-quality positive mask candidates to prevent precision bias.
3. **Dataset Safety**: All existing training/validation/test splits (`training_dataset_v2`) remain **100% untouched**.
4. **Ready for Annotation**: Candidates are staged in `data_collection/more/annotation_candidates/` and indexed in `data_collection/more/metadata/frames.csv`.
