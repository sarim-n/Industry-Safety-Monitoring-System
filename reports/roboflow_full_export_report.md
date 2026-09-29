# Roboflow Complete Dataset Export & Analysis Report

## 1. Project Information

- **Workspace**: `sarimahmedn-official-gmail-com`
- **Project Slug**: `more-edpuy`
- **Dataset Version Exported**: Version 1 (YOLOv8 format)
- **Export Directory**: `data_collection/roboflow_more_full_export/`

## 2. Class Mapping & Alignment

| Roboflow Class ID | Roboflow Class Name | Expected Project ID | Expected Class Name | Alignment Status |
| :---: | :--- | :---: | :--- | :--- |
| 0 | `helmet` | 0 | `helmet` | **ALIGNED** |
| 1 | `mask` | 1 | `mask` | **ALIGNED** |
| 2 | `person` | 2 | `person` | **ALIGNED** |

## 3. Complete Dataset Counts & Split Breakdown

| Split | Images | Total Boxes | Helmet Boxes (cls 0) | Mask Boxes (cls 1) | Person Boxes (cls 2) | Mask/Person Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | 308 | 1112 | 239 | 244 | 629 | 0.3879 |
| `valid` | 0 | 0 | 0 | 0 | 0 | 0.0000 |
| `test` | 0 | 0 | 0 | 0 | 0 | 0.0000 |
| **TOTAL** | **308** | **1112** | **239** | **244** | **629** | **0.3879** |

## 4. Quality & Integrity Inspection

- **Corrupt Images**: 0
- **Zero-Byte Files**: 0
- **Missing Label Files**: 0
- **Invalid Label Formats / Class IDs**: 0
- **Exact Duplicates (MD5/SHA256)**: 0
- **Near Duplicates (dHash <= 3)**: 283
- **Cross-Split Duplicates**: 0

## 5. Mask & Hard-Negative Candidate Analysis

- **Total Person Instances**: 629
- **Total Mask Instances**: 244
- **Overall Mask / Person Ratio**: 0.3879
- **Images Containing Mask Annotations**: 122
- **Images Without Mask Annotations**: 186
- **Hard-Negative Candidate Images (Person=YES, Mask=NO)**: **186**
- **Difficult-Positive Candidate Images (Person=YES, Mask=YES)**: **122**
- **Newly Extracted Video Data Present**: **YES** (122 images)
