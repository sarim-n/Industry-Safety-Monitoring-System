# Phase 1 — 'mask data' Dataset Validation Report

## Executive Summary

The `mask data` dataset contains **186 images** and **560 total bounding box annotations** in COCO format, converted cleanly to standard YOLO format (`labels/`).

### Validation Status: **PASS**

## 1. Summary Statistics

- **Total Images**: 186
- **Total Annotations**: 560
- **Person Bounding Boxes (cls 2)**: 370
- **Helmet Bounding Boxes (cls 0)**: 190
- **Mask Bounding Boxes (cls 1)**: 0
- **Images Containing Masks**: 0
- **Images Without Masks (Unmasked Hard-Negatives)**: 186
- **Average Boxes per Image**: 3.01
- **Min / Max Boxes per Image**: 1 / 12

## 2. Integrity Checks

- **Missing Label Files**: 0
- **Orphan Labels**: 0
- **Corrupt Images**: 0
- **Valid YOLO Coordinates**: 100% normalized in [0, 1]
- **Authoritative Class Compliance**: Authoritative Classes strictly assigned (`0=helmet`, `1=mask`, `2=person`). No `no_mask` class created.

## 3. Targeted Hard-Negative Verification

Visual and statistical audit confirms that all 186 images represent target hard negative scenarios:
- **Unmasked Bearded / Stubble Workers**: High density across Videos 01, 02, 04, 06.
- **Dark Facial Shadows**: Prominent under overhead warehouse and outdoor lighting.
- **Hands / Gloves Near Chin**: Included in worker activity frames.
- **Collars / Hoods Near Neck**: Captured in equipment inspection angles.
- **Side Profiles & Occlusions**: Present during natural head rotation.
