# 🦺 Industry Safety Monitoring System

A Real-Time AI Safety Monitoring System for Industrial / Factory Environments.

Detects whether workers are wearing required PPE (Personal Protective Equipment)
using a custom-trained YOLO model.

---

## 🎯 YOLO Class Mapping

| Class ID | Label  |
|----------|--------|
| 0        | helmet |
| 1        | mask   |
| 2        | person |

---

## 📁 Project Structure

```
safety monitoring/
│
├── data_collection/
│   ├── new_batch_5_videos/        ← New batch (6 videos processed)
│   │   ├── metadata/
│   │   │   ├── candidates.csv     ← All extracted frame metadata
│   │   │   └── filtered.csv      ← Kept/removed frame records
│   │   ├── review/
│   │   │   └── contact_sheets/   ← JPG contact sheets for human review
│   │   └── reports/
│   │       ├── processing_report.txt
│   │       └── extra_video_report.txt
│   │
│   ├── process_new_batch.py       ← Pipeline: 5-video batch processor
│   ├── process_extra_video.py     ← Pipeline: single extra video adder
│   ├── extract_candidates.py      ← Original frame extractor
│   ├── extract_additional_candidates.py
│   ├── filter_and_review.py       ← Dedup + contact sheet generator
│   ├── inspect_videos.py
│   ├── preview_videos.py
│   └── review_candidates.py
│
├── roboflow/
│   └── export.py                  ← Roboflow dataset utilities
│
└── isolate_unlabeled.py
```

---

## 🔄 Processing Pipeline

```
RAW VIDEOS
    ↓
FRAME EXTRACTION  (~3 FPS, deterministic filenames)
    ↓
QUALITY FILTERING  (black frames + blur detection)
    ↓
NEAR-DUPLICATE FILTERING  (dHash, Hamming ≤ 3/64 bits)
    ↓
CONTACT SHEETS  (20 frames/sheet for human review)
    ↓
HUMAN REVIEW
    ↓
MANUAL ANNOTATION via Roboflow
    ↓
YOLO TRAINING  (future)
```

---

## 📊 Dataset Status

### New Batch (6 videos)

| Video | Resolution | FPS | Final Frames | Watermark |
|-------|------------|-----|-------------|-----------|
| `5434223-hd_1920_1080_24fps.mp4` | 1920×1080 | 23.98 | 35 | — |
| `gettyimages-1286759790-640_adpp.mp4` | 768×432 | 29.97 | 23 | Getty |
| `gettyimages-1315550872-640_adpp.mp4` | 768×432 | 25.00 | 26 | Getty |
| `gettyimages-1578985062-640_adpp.mp4` | 768×432 | 23.98 | 16 | Getty |
| `gettyimages-2216765853-640_adpp.mp4` | 768×432 | 25.00 | 7  | Getty |
| `istockphoto-2258622635-640_adpp_is.mp4` | 768×432 | 50.00 | 13 | iStock |

**Total final frames (new batch): 120**

> ⚠️ Getty Images and iStock frames are included for research/development.
> Evaluate licensing before commercial use.

---

## ⚙️ Requirements

```bash
pip install opencv-python numpy
```

---

## 🚀 Usage

### Process a new batch of videos
```bash
cd data_collection
python process_new_batch.py
```

### Add a single extra video to an existing batch
```bash
cd data_collection
python process_extra_video.py
```

---

## 📌 Notes

- Do **not** annotate automatically — all annotation is done manually via Roboflow.
- Do **not** train YOLO until annotation and human review are complete.
- Raw video files and individual frame images are excluded from this repo (see `.gitignore`).
  Only scripts, metadata CSVs, reports, and contact sheets are tracked.
