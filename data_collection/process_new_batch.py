"""
================================================================================
NEW BATCH PROCESSING PIPELINE  --  5 Raw Videos
================================================================================

YOLO CLASS MAPPING (enforced throughout):
  0 = helmet
  1 = mask
  2 = person

ALL OUTPUT GOES INTO:
  data_collection/new_batch_5_videos/

NEVER TOUCHES:
  data_collection/candidates/
  data_collection/candidates_additional/
  data_collection/filtered_candidates/
  data_collection/candidate_review/
  data_collection/review_after_filter/
  data_collection/all_candidates.csv
  data_collection/candidates.csv
  data_collection/candidates_additional.csv
  data_collection/filtered_candidates.csv
================================================================================
"""

import csv
import math
import os
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

VIDEOS_FOLDER_NAME   = "new videos"   # inside data_collection/
EXPECTED_VIDEO_COUNT = 5
BATCH_DIR_NAME       = "new_batch_5_videos"

SAMPLE_FPS           = 3      # ~3 frames per second
DHASH_SIZE           = 8
SIMILARITY_THRESHOLD = 3      # Hamming distance out of 64 bits
BLUR_THRESHOLD       = 20.0   # Laplacian variance; below = blurry
                               # Note: cinematic/smooth video footage can have lap_var ~20-40;
                               # threshold of 80 was too aggressive, 20 only catches hard motion blur
BLACK_THRESHOLD      = 10.0   # Mean pixel value; below = black

FRAMES_PER_SHEET     = 20
THUMB_WIDTH          = 320
COLS                 = 5

YOLO_CLASSES         = {0: "helmet", 1: "mask", 2: "person"}
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
WATERMARK_PATTERNS   = ["gettyimages", "istockphoto", "shutterstock", "pond5", "123rf"]

FLAGGED_CATEGORIES = {
    "road_traffic":       ["traffic", "road", "highway", "car", "vehicle", "driving"],
    "public_surveillance":["surveillance", "cctv", "public"],
    "sports":             ["sport", "football", "soccer", "basketball", "game"],
    "nature_outdoor":     ["nature", "outdoor", "park", "forest"],
}

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

def fmt_ts(sec):
    s = int(round(sec))
    return f"{s // 60:02d}:{s % 60:02d}"

def detect_watermark(video_name):
    low = video_name.lower()
    for wm in WATERMARK_PATTERNS:
        if wm in low:
            return wm
    return None

def flag_footage(video_name):
    low = video_name.lower()
    for cat, keywords in FLAGGED_CATEGORIES.items():
        for kw in keywords:
            if kw in low:
                return cat
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
# CONTACT SHEETS
# -------------------------------------------------------------------------------

def make_contact_sheet(frames_info, images_dir, title, sheet_idx, total_sheets):
    if not frames_info:
        return None

    thumb_h = None
    for fi in frames_info:
        sample = cv2.imread(str(images_dir / fi["filename"]))
        if sample is not None:
            h, w   = sample.shape[:2]
            thumb_h = max(1, int(round(THUMB_WIDTH * h / w)))
            break
    if thumb_h is None:
        thumb_h = 180

    cols     = COLS
    rows     = math.ceil(len(frames_info) / cols)
    margin, gap, header_h = 15, 8, 55

    canvas_w = margin * 2 + cols * THUMB_WIDTH + (cols - 1) * gap
    canvas_h = header_h + margin * 2 + rows * thumb_h + (rows - 1) * gap
    canvas   = np.full((canvas_h, canvas_w, 3), (24, 24, 30), dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (canvas_w, header_h), (40, 40, 50), -1)
    hdr = f"Sheet {sheet_idx}/{total_sheets}  |  {title}  |  Frames: {len(frames_info)}"
    cv2.putText(canvas, hdr, (margin, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    for idx, fi in enumerate(frames_info):
        r, c = divmod(idx, cols)
        x = margin + c * (THUMB_WIDTH + gap)
        y = header_h + margin + r * (thumb_h + gap)

        img_path = images_dir / fi["filename"]
        frame    = cv2.imread(str(img_path))
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

        src_short = fi.get("source_video", "")[:32]
        cv2.putText(thumb, src_short, (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 80), 1, cv2.LINE_AA)
        cv2.rectangle(thumb, (0, 0), (THUMB_WIDTH - 1, thumb_h - 1), (70, 70, 90), 1)
        canvas[y:y + thumb_h, x:x + THUMB_WIDTH] = thumb

    return canvas

# -------------------------------------------------------------------------------
# STEP 1: INSPECT
# -------------------------------------------------------------------------------

def inspect_videos(videos_dir):
    video_files = sorted([
        f for f in videos_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    print("\n" + "=" * 78)
    print("STEP 1 -- VIDEO INSPECTION".center(78))
    print("=" * 78)

    if len(video_files) != EXPECTED_VIDEO_COUNT:
        print(f"\n[FATAL] Expected exactly {EXPECTED_VIDEO_COUNT} video files.")
        print(f"        Found: {len(video_files)}")
        for vf in video_files:
            print(f"          {vf.name}")
        raise SystemExit(1)

    infos = []
    col_w = 42
    header = (f"{'Filename':<{col_w}} | {'Resolution':<11} | {'FPS':<7} | "
              f"{'Duration':<10} | {'Total Frames':<14} | {'Watermark':<12} | {'Flag':<15}")
    line_len = len(header)
    print("\n" + "-" * line_len)
    print(header)
    print("-" * line_len)

    for vf in video_files:
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {vf.name}")
            cap.release()
            continue

        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0 or total_frames <= 0 or w <= 0 or h <= 0:
            print(f"[WARNING] Bad metadata: {vf.name}")
            continue

        duration = total_frames / fps
        wm       = detect_watermark(vf.name) or "--"
        flag_cat = flag_footage(vf.name) or "--"

        info = {
            "filepath":     vf,
            "filename":     vf.name,
            "width":        w,
            "height":       h,
            "fps":          fps,
            "total_frames": total_frames,
            "duration_sec": duration,
            "watermark":    wm,
            "footage_flag": flag_cat,
        }
        infos.append(info)

        fn_d = vf.name if len(vf.name) <= col_w else vf.name[:col_w-3] + "..."
        print(
            f"{fn_d:<{col_w}} | {w}x{h:<8} | {fps:<7.2f} | "
            f"{fmt_duration(duration):<10} | {total_frames:<14} | {wm:<12} | {flag_cat:<15}"
        )

    print("-" * line_len)
    print(f"\nConfirmed exactly {len(infos)} video(s) found.\n")
    return infos

# -------------------------------------------------------------------------------
# STEP 2: EXTRACT FRAMES
# -------------------------------------------------------------------------------

def extract_frames(video_info, raw_dir, candidates_csv_rows):
    vf   = video_info["filepath"]
    fps  = video_info["fps"]
    step = max(1, int(round(fps / SAMPLE_FPS)))
    prefix = sanitize(vf.name)

    cap = cv2.VideoCapture(str(vf))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {vf.name}")
        return []

    extracted  = []
    frame_idx  = 0

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
                "source_video":      vf.name,
                "frame_number":      frame_idx,
                "timestamp_seconds": round(ts, 3),
                "filename":          img_name,
                "width":             video_info["width"],
                "height":            video_info["height"],
                "fps":               round(fps, 3),
            }
            extracted.append(row)
            candidates_csv_rows.append(row)

        frame_idx += 1

    cap.release()
    print(f"  [{vf.name}]  Extracted: {len(extracted)} candidate frames")
    return extracted

# -------------------------------------------------------------------------------
# STEP 3: QUALITY FILTER
# -------------------------------------------------------------------------------

def quality_filter(frames, raw_dir):
    kept, removed = [], []
    for fi in frames:
        img = cv2.imread(str(raw_dir / fi["filename"]))
        if img is None:
            removed.append({**fi, "removal_reason": "unreadable"})
            continue
        if is_black_frame(img):
            removed.append({**fi, "removal_reason": "black_frame"})
            continue
        if is_blurry(img):
            removed.append({**fi, "removal_reason": "blurry"})
            continue
        kept.append(fi)
    return kept, removed

# -------------------------------------------------------------------------------
# STEP 4: NEAR-DUPLICATE FILTER
# -------------------------------------------------------------------------------

def dedup_filter(frames, raw_dir):
    by_video = defaultdict(list)
    for fi in frames:
        by_video[fi["source_video"]].append(fi)

    kept_all      = []
    total_removed = 0

    for sv, items in by_video.items():
        last_hash = None
        for fi in items:
            img = cv2.imread(str(raw_dir / fi["filename"]))
            if img is None:
                kept_all.append(fi)
                last_hash = None
                continue

            h = compute_dhash(img)
            if last_hash is None or hamming_distance(last_hash, h) > SIMILARITY_THRESHOLD:
                kept_all.append(fi)
                last_hash = h
            else:
                total_removed += 1

    return kept_all, total_removed

# -------------------------------------------------------------------------------
# STEP 5: COPY TO FILTERED
# -------------------------------------------------------------------------------

def copy_to_filtered(frames, raw_dir, filtered_dir, filtered_csv_rows):
    for fi in frames:
        src = raw_dir  / fi["filename"]
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

# -------------------------------------------------------------------------------
# STEP 6: CONTACT SHEETS
# -------------------------------------------------------------------------------

def generate_contact_sheets(frames, filtered_dir, sheets_dir):
    by_video = defaultdict(list)
    for fi in frames:
        by_video[fi["source_video"]].append(fi)

    sheets_dir.mkdir(parents=True, exist_ok=True)
    total_sheets = 0

    for sv, items in by_video.items():
        prefix   = sanitize(sv)
        n_sheets = math.ceil(len(items) / FRAMES_PER_SHEET)
        for s_idx in range(n_sheets):
            chunk = items[s_idx * FRAMES_PER_SHEET:(s_idx + 1) * FRAMES_PER_SHEET]
            sheet = make_contact_sheet(
                frames_info=chunk,
                images_dir=filtered_dir,
                title=sv,
                sheet_idx=s_idx + 1,
                total_sheets=n_sheets,
            )
            if sheet is not None:
                sheet_name = f"{prefix}_sheet_{s_idx + 1:02d}of{n_sheets:02d}.jpg"
                cv2.imwrite(str(sheets_dir / sheet_name), sheet,
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
                total_sheets += 1

    return total_sheets

# -------------------------------------------------------------------------------
# CSV WRITERS
# -------------------------------------------------------------------------------

def write_candidates_csv(rows, path):
    headers = ["source_video", "frame_number", "timestamp_seconds",
               "filename", "width", "height", "fps"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

def write_filtered_csv(rows, path):
    headers = ["source_video", "original_candidate", "filtered_filename",
               "kept", "removal_reason", "timestamp_seconds",
               "frame_number", "width", "height"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

# -------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------

def main():
    script_dir   = Path(__file__).resolve().parent   # data_collection/
    videos_dir   = script_dir / VIDEOS_FOLDER_NAME
    batch_root   = script_dir / BATCH_DIR_NAME

    raw_dir      = batch_root / "raw_frames"
    filtered_dir = batch_root / "filtered_frames"
    review_dir   = batch_root / "review"
    sheets_dir   = review_dir / "contact_sheets"
    meta_dir     = batch_root / "metadata"
    reports_dir  = batch_root / "reports"

    candidates_csv = meta_dir / "candidates.csv"
    filtered_csv   = meta_dir / "filtered.csv"
    report_path    = reports_dir / "processing_report.txt"

    for d in [raw_dir, filtered_dir, review_dir, sheets_dir, meta_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("  NEW 5-VIDEO BATCH PROCESSING PIPELINE".center(78))
    print("  Real-Time AI Safety Monitoring System".center(78))
    print("=" * 78)
    print(f"\n  Input  : {videos_dir}")
    print(f"  Output : {batch_root}")
    print(f"\n  YOLO class mapping:")
    for cid, cname in YOLO_CLASSES.items():
        print(f"    {cid} = {cname}")

    # STEP 1
    video_infos = inspect_videos(videos_dir)

    # STEP 2
    print("\n" + "=" * 78)
    print("STEP 2 -- FRAME EXTRACTION".center(78))
    print("=" * 78)
    print(f"  Sample rate: ~{SAMPLE_FPS} FPS\n")

    candidates_csv_rows = []
    per_video_extracted = {}
    all_candidates      = []

    for vi in video_infos:
        frames = extract_frames(vi, raw_dir, candidates_csv_rows)
        per_video_extracted[vi["filename"]] = frames
        all_candidates.extend(frames)

    write_candidates_csv(candidates_csv_rows, candidates_csv)
    total_candidates = len(all_candidates)
    print(f"\n  Total candidate frames extracted: {total_candidates}")
    print(f"  Metadata written: {candidates_csv}")

    # STEP 3
    print("\n" + "=" * 78)
    print("STEP 3 -- QUALITY FILTERING".center(78))
    print("=" * 78)

    quality_kept, quality_removed = quality_filter(all_candidates, raw_dir)
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

    # STEP 4
    print("\n" + "=" * 78)
    print("STEP 4 -- NEAR-DUPLICATE FILTERING (dHash)".center(78))
    print("=" * 78)
    print(f"  Hamming threshold: <= {SIMILARITY_THRESHOLD} / 64 bits")

    dedup_kept, n_dupes_removed = dedup_filter(quality_kept, raw_dir)
    print(f"  After quality filter    : {len(quality_kept)}")
    print(f"  Near-duplicates removed : {n_dupes_removed}")
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

    # STEP 5
    print("\n" + "=" * 78)
    print("STEP 5 -- COPYING TO filtered_frames/".center(78))
    print("=" * 78)

    copy_to_filtered(dedup_kept, raw_dir, filtered_dir, filtered_csv_rows)
    write_filtered_csv(filtered_csv_rows, filtered_csv)
    print(f"  {len(dedup_kept)} frames copied to: {filtered_dir}")
    print(f"  Filtered metadata: {filtered_csv}")

    # STEP 6
    print("\n" + "=" * 78)
    print("STEP 6 -- CONTACT SHEETS".center(78))
    print("=" * 78)

    n_sheets = generate_contact_sheets(dedup_kept, filtered_dir, sheets_dir)
    print(f"  {n_sheets} contact sheet(s) saved to: {sheets_dir}")

    # Per-video final counts
    per_video_final = defaultdict(int)
    for fi in dedup_kept:
        per_video_final[fi["source_video"]] += 1

    # Build report
    watermarked = [vi for vi in video_infos if vi["watermark"] != "--"]
    flagged     = [vi for vi in video_infos if vi["footage_flag"] != "--"]
    problematic = [vi for vi in video_infos if not per_video_extracted.get(vi["filename"])]

    lines = []
    sep   = "=" * 78

    def L(s=""):
        lines.append(s)

    L(sep)
    L("  NEW 5-VIDEO BATCH -- PROCESSING REPORT".center(78))
    L(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(78))
    L(sep)
    L()
    L("  YOLO CLASS MAPPING")
    L("    0 = helmet")
    L("    1 = mask")
    L("    2 = person")
    L()
    L(sep)
    L("  1. VIDEOS FOUND & PROCESSED")
    L(sep)
    L(f"  Videos found       : {len(video_infos)}")
    L(f"  Successfully proc. : {len(video_infos) - len(problematic)}")
    L()

    col_w = 42
    hdr = (f"  {'Filename':<{col_w}} | {'Res':<11} | {'FPS':<7} | "
           f"{'Duration':<10} | {'Tot.Fr':<8} | {'Extrac':<8} | "
           f"{'Final':<7} | {'WM':<12} | {'Flag'}")
    L(hdr)
    L("  " + "-" * (len(hdr) - 2))

    for vi in video_infos:
        fn       = vi["filename"]
        fn_disp  = fn if len(fn) <= col_w else fn[:col_w-3] + "..."
        extracted= len(per_video_extracted.get(fn, []))
        final_ct = per_video_final.get(fn, 0)
        L(
            f"  {fn_disp:<{col_w}} | {vi['width']}x{vi['height']:<7} | {vi['fps']:<7.2f} | "
            f"{fmt_duration(vi['duration_sec']):<10} | {vi['total_frames']:<8} | "
            f"{extracted:<8} | {final_ct:<7} | {vi['watermark']:<12} | {vi['footage_flag']}"
        )

    L()
    L(sep)
    L("  2. FRAME STATISTICS")
    L(sep)
    L(f"  Candidate frames extracted  : {total_candidates}")
    L(f"  Quality-filtered removed    : {len(quality_removed)}")
    L(f"  Near-duplicates removed     : {n_dupes_removed}")
    L(f"  FINAL FRAMES RETAINED       : {len(dedup_kept)}")
    L()
    L("  Final frames per video:")
    for vi in video_infos:
        fn = vi["filename"]
        L(f"    {fn:60s}: {per_video_final.get(fn, 0)}")
    L()
    L(sep)
    L("  3. PROBLEMATIC VIDEOS")
    L(sep)
    if problematic:
        for vi in problematic:
            L(f"    [!]  {vi['filename']}")
    else:
        L("    None")
    L()
    L(sep)
    L("  4. WATERMARKED VIDEOS  (NOT modified)")
    L(sep)
    if watermarked:
        for vi in watermarked:
            L(f"    [!]  {vi['filename']}  (detected: {vi['watermark']})")
        L()
        L("    NOTE: Watermarks were NOT removed. Frames kept as-is.")
        L("    Evaluate licensing before using these frames in training.")
    else:
        L("    None detected")
    L()
    L(sep)
    L("  5. QUESTIONABLE / NON-INDUSTRIAL FOOTAGE")
    L(sep)
    if flagged:
        for vi in flagged:
            L(f"    [!]  {vi['filename']}  (category: {vi['footage_flag']})")
        L()
        L("    ACTION REQUIRED: Review these videos manually before annotation.")
        L("    Frames retained -- do NOT include in training if unrelated to")
        L("    industrial/factory safety contexts.")
    else:
        L("    None flagged (all filenames appear industrial-context neutral)")
    L()
    L(sep)
    L("  6. OUTPUT DIRECTORIES")
    L(sep)
    L(f"    Batch root      : {batch_root}")
    L(f"    Raw frames      : {raw_dir}")
    L(f"    Filtered frames : {filtered_dir}  <-- FINAL IMAGES FOR REVIEW")
    L(f"    Contact sheets  : {sheets_dir}")
    L(f"    Candidates CSV  : {candidates_csv}")
    L(f"    Filtered CSV    : {filtered_csv}")
    L(f"    Report          : {report_path}")
    L()
    L(sep)
    L("  7. NEXT STEPS  (DO NOT AUTOMATE)")
    L(sep)
    L("    1. Review contact sheets in:  review/contact_sheets/")
    L("    2. Manually inspect any watermarked or flagged footage")
    L("    3. Annotate retained frames via Roboflow:")
    L("         YOLO class 0 = helmet")
    L("         YOLO class 1 = mask")
    L("         YOLO class 2 = person")
    L("    4. Do NOT train YOLO until annotation + review are complete")
    L()
    L("  OLD DATASET: NOT TOUCHED -- all output is inside new_batch_5_videos/")
    L()
    L(sep)
    L("  NEW 5-VIDEO BATCH PROCESSING COMPLETE")
    L(sep)

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print()
    print(report_text)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
