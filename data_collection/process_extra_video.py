"""
================================================================================
SINGLE EXTRA VIDEO PROCESSOR
Adds ONE new video into the existing new_batch_5_videos/ batch directory.

Target video: gettyimages-1315550872-640_adpp.mp4

YOLO CLASS MAPPING:
  0 = helmet
  1 = mask
  2 = person

Output appended into (same batch dir as the 5 originals):
  data_collection/new_batch_5_videos/

NEVER TOUCHES OLD DATASET.
================================================================================
"""

import csv
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# -------------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------------

VIDEOS_FOLDER_NAME   = "new videos"
BATCH_DIR_NAME       = "new_batch_5_videos"

# The ONE new video to process
TARGET_VIDEO         = "gettyimages-1315550872-640_adpp.mp4"

SAMPLE_FPS           = 3
DHASH_SIZE           = 8
SIMILARITY_THRESHOLD = 3      # Hamming distance out of 64 bits
BLUR_THRESHOLD       = 20.0   # Laplacian variance; below = blurry
BLACK_THRESHOLD      = 10.0   # Mean pixel value; below = black

FRAMES_PER_SHEET     = 20
THUMB_WIDTH          = 320
COLS                 = 5

YOLO_CLASSES         = {0: "helmet", 1: "mask", 2: "person"}
WATERMARK_PATTERNS   = ["gettyimages", "istockphoto", "shutterstock", "pond5", "123rf"]

# -------------------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------------------

def sanitize(name):
    stem  = Path(name).stem
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_")
    return clean.lower()[:40]

def fmt_duration(sec):
    s = int(round(sec))
    return f"{s // 60}m {s % 60:02d}s"

def detect_watermark(video_name):
    low = video_name.lower()
    for wm in WATERMARK_PATTERNS:
        if wm in low:
            return wm
    return None

# -------------------------------------------------------------------------------
# IMAGE ANALYSIS
# -------------------------------------------------------------------------------

def compute_dhash(img):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (DHASH_SIZE + 1, DHASH_SIZE), interpolation=cv2.INTER_AREA)
    diff    = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()

def hamming_distance(h1, h2):
    return int(np.count_nonzero(h1 != h2))

def is_black_frame(img):
    return float(img.mean()) < BLACK_THRESHOLD

def is_blurry(img):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return lap_var < BLUR_THRESHOLD

# -------------------------------------------------------------------------------
# CONTACT SHEET
# -------------------------------------------------------------------------------

def make_contact_sheet(frames_info, images_dir, title, sheet_idx, total_sheets):
    if not frames_info:
        return None

    thumb_h = None
    for fi in frames_info:
        sample = cv2.imread(str(images_dir / fi["filename"]))
        if sample is not None:
            h, w    = sample.shape[:2]
            thumb_h = max(1, int(round(THUMB_WIDTH * h / w)))
            break
    if thumb_h is None:
        thumb_h = 180

    rows     = math.ceil(len(frames_info) / COLS)
    margin, gap, header_h = 15, 8, 55
    canvas_w = margin * 2 + COLS * THUMB_WIDTH + (COLS - 1) * gap
    canvas_h = header_h + margin * 2 + rows * thumb_h + (rows - 1) * gap
    canvas   = np.full((canvas_h, canvas_w, 3), (24, 24, 30), dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (canvas_w, header_h), (40, 40, 50), -1)
    hdr = f"Sheet {sheet_idx}/{total_sheets}  |  {title}  |  Frames: {len(frames_info)}"
    cv2.putText(canvas, hdr, (margin, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    for idx, fi in enumerate(frames_info):
        r, c = divmod(idx, COLS)
        x = margin + c * (THUMB_WIDTH + gap)
        y = header_h + margin + r * (thumb_h + gap)

        frame = cv2.imread(str(images_dir / fi["filename"]))
        if frame is None:
            thumb = np.zeros((thumb_h, THUMB_WIDTH, 3), dtype=np.uint8)
        else:
            thumb = cv2.resize(frame, (THUMB_WIDTH, thumb_h), interpolation=cv2.INTER_AREA)

        ts    = float(fi.get("timestamp_seconds", 0))
        label = f" {fi['filename'][:28]}  t={ts:.1f}s "
        fs, thick = 0.38, 1
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        bh = th + bl + 6
        cv2.rectangle(thumb, (0, thumb_h - bh), (THUMB_WIDTH, thumb_h), (0, 0, 0), -1)
        cv2.putText(thumb, label, (3, thumb_h - bl - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 230, 230), thick, cv2.LINE_AA)
        cv2.putText(thumb, title[:32], (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 80), 1, cv2.LINE_AA)
        cv2.rectangle(thumb, (0, 0), (THUMB_WIDTH - 1, thumb_h - 1), (70, 70, 90), 1)
        canvas[y:y + thumb_h, x:x + THUMB_WIDTH] = thumb

    return canvas

# -------------------------------------------------------------------------------
# CSV HELPERS  (append-safe)
# -------------------------------------------------------------------------------

def append_to_candidates_csv(rows, path):
    """Append new rows to candidates.csv; write header only if file is new."""
    headers = ["source_video", "frame_number", "timestamp_seconds",
               "filename", "width", "height", "fps"]
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)

def append_to_filtered_csv(rows, path):
    headers = ["source_video", "original_candidate", "filtered_filename",
               "kept", "removal_reason", "timestamp_seconds",
               "frame_number", "width", "height"]
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)

# -------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------

def main():
    script_dir   = Path(__file__).resolve().parent
    videos_dir   = script_dir / VIDEOS_FOLDER_NAME
    batch_root   = script_dir / BATCH_DIR_NAME

    raw_dir      = batch_root / "raw_frames"
    filtered_dir = batch_root / "filtered_frames"
    sheets_dir   = batch_root / "review" / "contact_sheets"
    meta_dir     = batch_root / "metadata"
    reports_dir  = batch_root / "reports"

    candidates_csv = meta_dir / "candidates.csv"
    filtered_csv   = meta_dir / "filtered.csv"
    report_path    = reports_dir / "extra_video_report.txt"

    # Verify batch dir already exists
    if not batch_root.exists():
        print(f"[ERROR] Batch directory does not exist: {batch_root}")
        print("        Run process_new_batch.py first.")
        raise SystemExit(1)

    for d in [raw_dir, filtered_dir, sheets_dir, meta_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Locate target video
    video_path = videos_dir / TARGET_VIDEO
    if not video_path.exists():
        print(f"[ERROR] Target video not found: {video_path}")
        raise SystemExit(1)

    print("\n" + "=" * 78)
    print("  EXTRA VIDEO PROCESSOR".center(78))
    print("  Adding to: new_batch_5_videos/".center(78))
    print("=" * 78)
    print(f"\n  Video   : {TARGET_VIDEO}")
    print(f"  Output  : {batch_root}")
    print(f"\n  YOLO class mapping:")
    for cid, cname in YOLO_CLASSES.items():
        print(f"    {cid} = {cname}")

    # ── STEP 1: Inspect ───────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("STEP 1 -- VIDEO INSPECTION".center(78))
    print("=" * 78)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {TARGET_VIDEO}")
        raise SystemExit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or total_frames <= 0 or vw <= 0 or vh <= 0:
        print(f"[ERROR] Invalid metadata for: {TARGET_VIDEO}")
        raise SystemExit(1)

    duration = total_frames / fps
    wm       = detect_watermark(TARGET_VIDEO) or "--"

    print(f"\n  Filename     : {TARGET_VIDEO}")
    print(f"  Resolution   : {vw}x{vh}")
    print(f"  FPS          : {fps:.2f}")
    print(f"  Duration     : {fmt_duration(duration)}")
    print(f"  Total frames : {total_frames}")
    print(f"  Watermark    : {wm}")

    video_info = {
        "filepath":     video_path,
        "filename":     TARGET_VIDEO,
        "width":        vw,
        "height":       vh,
        "fps":          fps,
        "total_frames": total_frames,
        "duration_sec": duration,
        "watermark":    wm,
    }

    # ── STEP 2: Extract ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("STEP 2 -- FRAME EXTRACTION".center(78))
    print("=" * 78)

    step       = max(1, int(round(fps / SAMPLE_FPS)))
    prefix     = sanitize(TARGET_VIDEO)
    candidates_csv_rows = []
    all_frames = []
    frame_idx  = 0

    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            ts       = frame_idx / fps
            img_name = f"{prefix}_t{int(ts):05d}_f{frame_idx:06d}.jpg"
            out_path = raw_dir / img_name
            if not out_path.exists():
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            row = {
                "source_video":      TARGET_VIDEO,
                "frame_number":      frame_idx,
                "timestamp_seconds": round(ts, 3),
                "filename":          img_name,
                "width":             vw,
                "height":            vh,
                "fps":               round(fps, 3),
            }
            all_frames.append(row)
            candidates_csv_rows.append(row)
        frame_idx += 1
    cap.release()

    total_candidates = len(all_frames)
    print(f"  Extracted: {total_candidates} candidate frames")

    # Check for already-processed frames in existing CSVs (resumable)
    existing_filenames = set()
    if candidates_csv.exists():
        with open(candidates_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("source_video") == TARGET_VIDEO:
                    existing_filenames.add(r["filename"])

    new_rows = [r for r in candidates_csv_rows if r["filename"] not in existing_filenames]
    if new_rows:
        append_to_candidates_csv(new_rows, candidates_csv)
        print(f"  Appended {len(new_rows)} new rows to candidates.csv")
    else:
        print("  Candidates already recorded — skipping CSV append")

    # ── STEP 3: Quality filter ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("STEP 3 -- QUALITY FILTERING".center(78))
    print("=" * 78)

    quality_kept, quality_removed = [], []
    for fi in all_frames:
        img = cv2.imread(str(raw_dir / fi["filename"]))
        if img is None:
            quality_removed.append({**fi, "removal_reason": "unreadable"})
        elif is_black_frame(img):
            quality_removed.append({**fi, "removal_reason": "black_frame"})
        elif is_blurry(img):
            quality_removed.append({**fi, "removal_reason": "blurry"})
        else:
            quality_kept.append(fi)

    print(f"  Candidates in       : {total_candidates}")
    print(f"  Quality-removed     : {len(quality_removed)}")
    print(f"  After quality filter: {len(quality_kept)}")

    filtered_csv_rows = []
    for fi in quality_removed:
        filtered_csv_rows.append({
            "source_video":      fi["source_video"],
            "original_candidate":fi["filename"],
            "filtered_filename": "",
            "kept":              "NO",
            "removal_reason":    fi.get("removal_reason", "quality"),
            "timestamp_seconds": fi["timestamp_seconds"],
            "frame_number":      fi["frame_number"],
            "width":             fi["width"],
            "height":            fi["height"],
        })

    # ── STEP 4: Near-duplicate filter ─────────────────────────────────────
    print("\n" + "=" * 78)
    print("STEP 4 -- NEAR-DUPLICATE FILTERING (dHash)".center(78))
    print("=" * 78)
    print(f"  Hamming threshold: <= {SIMILARITY_THRESHOLD} / 64 bits")

    last_hash     = None
    dedup_kept    = []
    n_dupes       = 0

    for fi in quality_kept:
        img = cv2.imread(str(raw_dir / fi["filename"]))
        if img is None:
            dedup_kept.append(fi)
            last_hash = None
            continue
        h = compute_dhash(img)
        if last_hash is None or hamming_distance(last_hash, h) > SIMILARITY_THRESHOLD:
            dedup_kept.append(fi)
            last_hash = h
        else:
            n_dupes += 1

    print(f"  After quality filter    : {len(quality_kept)}")
    print(f"  Near-duplicates removed : {n_dupes}")
    print(f"  Final retained          : {len(dedup_kept)}")

    dedup_kept_names = {fi["filename"] for fi in dedup_kept}
    for fi in quality_kept:
        if fi["filename"] not in dedup_kept_names:
            filtered_csv_rows.append({
                "source_video":      fi["source_video"],
                "original_candidate":fi["filename"],
                "filtered_filename": "",
                "kept":              "NO",
                "removal_reason":    "near_duplicate",
                "timestamp_seconds": fi["timestamp_seconds"],
                "frame_number":      fi["frame_number"],
                "width":             fi["width"],
                "height":            fi["height"],
            })

    # ── STEP 5: Copy to filtered_frames/ ──────────────────────────────────
    print("\n" + "=" * 78)
    print("STEP 5 -- COPYING TO filtered_frames/".center(78))
    print("=" * 78)

    for fi in dedup_kept:
        src = raw_dir      / fi["filename"]
        dst = filtered_dir / fi["filename"]
        if not dst.exists():
            shutil.copy2(src, dst)
        filtered_csv_rows.append({
            "source_video":      fi["source_video"],
            "original_candidate":fi["filename"],
            "filtered_filename": fi["filename"],
            "kept":              "YES",
            "removal_reason":    "",
            "timestamp_seconds": fi["timestamp_seconds"],
            "frame_number":      fi["frame_number"],
            "width":             fi["width"],
            "height":            fi["height"],
        })

    # Append to filtered.csv (skip if already recorded)
    existing_filtered = set()
    if filtered_csv.exists():
        with open(filtered_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("source_video") == TARGET_VIDEO:
                    existing_filtered.add(r["original_candidate"])

    new_filtered_rows = [r for r in filtered_csv_rows
                         if r["original_candidate"] not in existing_filtered]
    if new_filtered_rows:
        append_to_filtered_csv(new_filtered_rows, filtered_csv)
        print(f"  Appended {len(new_filtered_rows)} rows to filtered.csv")
    else:
        print("  Filtered CSV already up to date")

    print(f"  {len(dedup_kept)} frames in filtered_frames/")

    # ── STEP 6: Contact sheets ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("STEP 6 -- CONTACT SHEETS".center(78))
    print("=" * 78)

    prefix   = sanitize(TARGET_VIDEO)
    n_sheets = math.ceil(len(dedup_kept) / FRAMES_PER_SHEET)
    sheets_made = 0

    for s_idx in range(n_sheets):
        chunk      = dedup_kept[s_idx * FRAMES_PER_SHEET:(s_idx + 1) * FRAMES_PER_SHEET]
        sheet      = make_contact_sheet(chunk, filtered_dir, TARGET_VIDEO,
                                        s_idx + 1, n_sheets)
        if sheet is not None:
            sheet_name = f"{prefix}_sheet_{s_idx + 1:02d}of{n_sheets:02d}.jpg"
            cv2.imwrite(str(sheets_dir / sheet_name), sheet,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            sheets_made += 1

    print(f"  {sheets_made} contact sheet(s) saved to: {sheets_dir}")

    # ── Report ─────────────────────────────────────────────────────────────
    lines = []
    sep   = "=" * 78

    def L(s=""): lines.append(s)

    L(sep)
    L("  EXTRA VIDEO -- PROCESSING REPORT".center(78))
    L(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(78))
    L(sep)
    L()
    L("  YOLO CLASS MAPPING")
    L("    0 = helmet  |  1 = mask  |  2 = person")
    L()
    L(sep)
    L("  VIDEO")
    L(sep)
    L(f"  Filename     : {TARGET_VIDEO}")
    L(f"  Resolution   : {vw}x{vh}")
    L(f"  FPS          : {fps:.2f}")
    L(f"  Duration     : {fmt_duration(duration)}")
    L(f"  Total frames : {total_frames}")
    L(f"  Watermark    : {wm}")
    L()
    L(sep)
    L("  FRAME STATISTICS")
    L(sep)
    L(f"  Candidate frames extracted  : {total_candidates}")
    L(f"  Quality-filtered removed    : {len(quality_removed)}")
    L(f"  Near-duplicates removed     : {n_dupes}")
    L(f"  FINAL FRAMES RETAINED       : {len(dedup_kept)}")
    L()
    L(sep)
    L("  WATERMARK")
    L(sep)
    if wm != "--":
        L(f"  [!]  Watermark detected: {wm}")
        L("  Frames NOT modified. Evaluate licensing before using in training.")
    else:
        L("  None detected")
    L()
    L(sep)
    L("  OUTPUT (appended to existing batch)")
    L(sep)
    L(f"  Batch root      : {batch_root}")
    L(f"  Filtered frames : {filtered_dir}")
    L(f"  Contact sheets  : {sheets_dir}")
    L(f"  Candidates CSV  : {candidates_csv}")
    L(f"  Filtered CSV    : {filtered_csv}")
    L()
    L(sep)
    L("  EXTRA VIDEO PROCESSING COMPLETE")
    L(sep)

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print()
    print(report_text)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
