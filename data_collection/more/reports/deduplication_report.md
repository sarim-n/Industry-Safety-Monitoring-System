# Phase 7.9 — Frame Extraction & Deduplication Statistics Report

## 1. Per-Video Extraction and Deduplication Summary

| Video ID | Source Video Filename | Duration (s) | Resolution | Total Video Frames | Raw Extracted (2 FPS) | Retained Filtered | Removed Duplicates | Deduplication Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `video01` | `gettyimages-1223673045-640_adpp.mp4` | 15.28 | 768x432 | 458 | 31 | 21 | 10 | 32.3% |
| `video02` | `gettyimages-1223674577-640_adpp.mp4` | 27.73 | 768x432 | 831 | 56 | 54 | 2 | 3.6% |
| `video03` | `gettyimages-1316819243-640_adpp.mp4` | 5.76 | 768x432 | 138 | 12 | 7 | 5 | 41.7% |
| `video04` | `gettyimages-1479449515-640_adpp.mp4` | 25.96 | 768x432 | 649 | 55 | 49 | 6 | 10.9% |
| `video05` | `gettyimages-2252592570-640_adpp.mp4` | 10.01 | 768x432 | 600 | 20 | 10 | 10 | 50.0% |
| `video06` | `gettyimages-2259036596-640_adpp.mp4` | 26.19 | 768x432 | 785 | 53 | 45 | 8 | 15.1% |
| **TOTALS** | **6 Raw Videos** | **110.93s** | **768x432** | **3461** | **227** | **186** | **41** | **18.1%** |

## 2. Deduplication Methodology & Control Parameters

- **Extraction Rate**: 2.0 FPS uniform sampling across all video sources.
- **Hash Algorithm**: 64-bit difference hash (dHash) computed on luminance channel.
- **Distance Threshold**: Hamming distance $d \le 4$ within a 5-second temporal window.
- **Conservative Over-Filtering Guard**: Preserves distinct head turns, facial shadows, glove/hand movements, side profiles, and occlusions.
