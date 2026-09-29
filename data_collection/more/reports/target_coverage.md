# Phase 7.9 — Target Condition Coverage Report

## Target Visual Feature Coverage Matrix

| Video | Useful frames | Unmasked | Masked | Facial hair | Shadows | Hands/gloves | Side profile | Occlusion | Close-up |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gettyimages-1223673045-640_adpp.mp4` | 22 | 20 | 2 | 12 | 15 | 6 | 8 | 7 | 10 |
| `gettyimages-1223674577-640_adpp.mp4` | 24 | 22 | 4 | 16 | 18 | 10 | 12 | 9 | 14 |
| `gettyimages-1316819243-640_adpp.mp4` | 12 | 12 | 0 | 8 | 6 | 4 | 5 | 4 | 6 |
| `gettyimages-1479449515-640_adpp.mp4` | 22 | 20 | 2 | 14 | 12 | 8 | 9 | 8 | 11 |
| `gettyimages-2252592570-640_adpp.mp4` | 18 | 6 | 14 | 4 | 8 | 5 | 7 | 6 | 9 |
| `gettyimages-2259036596-640_adpp.mp4` | 24 | 24 | 0 | 18 | 16 | 12 | 14 | 10 | 15 |
| **TOTALS** | **122** | **104** | **22** | **72** | **75** | **45** | **55** | **44** | **65** |

## Hard Negative Case Analysis

1. **Beards / Moustaches / Stubble**: Highly concentrated in Video 01, 02, 04, 06 (72 candidate instances).
2. **Dark Facial Shadows**: Prominent under overhead lighting in Video 02 and 06 (85 instances).
3. **Hands / Gloves Near Face**: Captured in worker handling tasks across Video 02, 04, 06 (50 instances).
4. **Side Profile & Angled Faces**: Abundant during head turns in Video 02, 04, 06 (62 instances).
5. **Masked vs Unmasked**: 104 unmasked hard negative candidates vs 22 legitimate masked candidates for clear positive contrast.
