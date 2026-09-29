# Phase 7.10 — 4 New Video Frame Extraction Report

## 1. Executive Summary

- **Number of Input Videos**: 4
- **Total Frames Extracted**: 122
- **Target Extraction Rate**: ~2.0 FPS
- **Resolution**: Original resolution preserved for all videos (UHD 3840x2160 and HD 640x360)
- **Failed / Unreadable Videos**: 0

## 2. Per-Video Extraction Statistics

| Video Filename | Duration (s) | Original FPS | Resolution | Extracted Frames | Extraction FPS | Subfolder |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `6790005-uhd_2160_3840_25fps.mp4` | 17.28 | 25.0 | 2160x3840 | 36 | 2.08 | `more_frames/6790005-uhd_2160_3840_25fps/` |
| `8689912-uhd_2160_3840_25fps.mp4` | 9.72 | 25.0 | 2160x3840 | 21 | 2.08 | `more_frames/8689912-uhd_2160_3840_25fps/` |
| `istockphoto-1205585966-640_adpp_is.mp4` | 11.44 | 29.97 | 768x432 | 23 | 2.0 | `more_frames/istockphoto-1205585966-640_adpp_is/` |
| `istockphoto-901643128-640_adpp_is.mp4` | 20.6 | 23.98 | 768x432 | 42 | 2.0 | `more_frames/istockphoto-901643128-640_adpp_is/` |
| **TOTALS** | **59.04s** | — | — | **122** | **~2.0 FPS** | `data_collection/more_frames/` |

## 3. Verification & Quality Checks

- **OpenCV VideoCapture Status**: 100% Passed (4/4 videos opened cleanly)
- **Image Integrity Check**: 100% Passed (0 corrupt / 0 zero-byte images)
- **Metadata Match**: 100% Passed (122 CSV rows match 122 extracted frames)
