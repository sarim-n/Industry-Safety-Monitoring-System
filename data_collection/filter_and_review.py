import csv
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

DHASH_SIZE = 8
# Similarity threshold: Hamming distance <= 1 bit out of 64 (removes exact/near-exact static duplicates)
SIMILARITY_THRESHOLD = 1
FRAMES_PER_SHEET = 20  # For review contact sheets
THUMB_WIDTH = 320
COLS = 5

def sanitize_filename(filename: str) -> str:
    """Sanitize original video filename for clean file prefix."""
    stem = Path(filename).stem
    clean = re.sub(r'[^a-zA-Z0-9]+', '_', stem).strip('_')
    return clean.lower()

def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS format."""
    total_sec = int(round(seconds))
    mins = total_sec // 60
    secs = total_sec % 60
    return f"{mins:02d}:{secs:02d}"

def compute_dhash(img: np.ndarray) -> np.ndarray:
    """Compute 64-bit difference hash (dHash) for an image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (DHASH_SIZE + 1, DHASH_SIZE), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()

def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """Calculate Hamming distance between two boolean bit vectors."""
    return np.count_nonzero(hash1 != hash2)

def create_filtered_contact_sheet(
    chunk_rows: list,
    filtered_dir: Path,
    source_video: str,
    sheet_index: int,
    total_sheets: int
) -> np.ndarray:
    """Generate a contact sheet for a chunk of retained candidate images."""
    if not chunk_rows:
        return None

    first_img_path = filtered_dir / chunk_rows[0]["image_name"]
    sample_img = cv2.imread(str(first_img_path))
    if sample_img is None:
        return None

    src_h, src_w = sample_img.shape[:2]
    thumb_w = THUMB_WIDTH
    thumb_h = max(1, int(round(thumb_w * src_h / src_w)))

    num_items = len(chunk_rows)
    cols = COLS
    rows = math.ceil(num_items / cols)

    margin = 15
    gap = 10
    header_height = 50

    canvas_w = margin * 2 + cols * thumb_w + (cols - 1) * gap
    canvas_h = header_height + margin * 2 + rows * thumb_h + (rows - 1) * gap

    canvas = np.full((canvas_h, canvas_w, 3), (28, 28, 32), dtype=np.uint8)

    # Header
    header_text = (
        f"Video: {source_video}  |  Filtered Review Sheet {sheet_index}/{total_sheets}  |  "
        f"Retained Frames: {num_items}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas_w, header_height), (45, 45, 52), -1)
    cv2.putText(
        canvas,
        header_text,
        (margin, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    for idx, item in enumerate(chunk_rows):
        r = idx // cols
        c = idx % cols

        x = margin + c * (thumb_w + gap)
        y = header_height + margin + r * (thumb_h + gap)

        img_path = filtered_dir / item["image_name"]
        frame_img = cv2.imread(str(img_path))

        if frame_img is None:
            resized = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
        else:
            resized = cv2.resize(frame_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        # Label Overlay
        ts_sec = float(item["timestamp_seconds"])
        label_text = f" t={ts_sec:.2f}s ({format_timestamp(ts_sec)}) | f={item['frame_number']} | Grp:{item['similarity_group_id']} "

        font_scale = 0.42
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        banner_h = th + baseline + 8
        cv2.rectangle(resized, (0, thumb_h - banner_h), (thumb_w, thumb_h), (0, 0, 0), -1)
        cv2.putText(
            resized,
            label_text,
            (4, thumb_h - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        cv2.rectangle(resized, (0, 0), (thumb_w - 1, thumb_h - 1), (80, 80, 95), 1)
        canvas[y:y + thumb_h, x:x + thumb_w] = resized

    return canvas

def run_filter_and_review_pipeline():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Input paths
    csv_pass1_path = project_root / "data_collection" / "candidates.csv"
    dir_pass1 = project_root / "data_collection" / "candidates"

    csv_pass2_path = project_root / "data_collection" / "candidates_additional.csv"
    dir_pass2 = project_root / "data_collection" / "candidates_additional"

    # Output paths
    all_candidates_csv = project_root / "data_collection" / "all_candidates.csv"
    filtered_dir = project_root / "data_collection" / "filtered_candidates"
    filtered_csv = project_root / "data_collection" / "filtered_candidates.csv"
    review_filtered_dir = project_root / "data_collection" / "review_after_filter"

    if not csv_pass1_path.exists() or not csv_pass2_path.exists():
        print("[ERROR] Required candidate CSV metadata files missing.")
        return

    # STEP 1: Combine Metadata
    all_candidates = []

    # Read Pass 1 metadata
    with open(csv_pass1_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = f"data_collection/candidates/{row['image_name']}"
            all_candidates.append({
                "image_name": row["image_name"],
                "image_path": rel_path,
                "abs_path": dir_pass1 / row["image_name"],
                "source_video": row["source_video"],
                "timestamp_seconds": float(row["timestamp_seconds"]),
                "frame_number": int(row["frame_number"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "extraction_pass": 1
            })

    # Read Pass 2 metadata
    with open(csv_pass2_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = f"data_collection/candidates_additional/{row['image_name']}"
            all_candidates.append({
                "image_name": row["image_name"],
                "image_path": rel_path,
                "abs_path": dir_pass2 / row["image_name"],
                "source_video": row["source_video"],
                "timestamp_seconds": float(row["timestamp_seconds"]),
                "frame_number": int(row["frame_number"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "extraction_pass": 2
            })

    # Sort all candidates by source video and timestamp
    all_candidates.sort(key=lambda x: (x["source_video"], x["timestamp_seconds"], x["frame_number"]))

    # Write unified all_candidates.csv
    all_headers = [
        "image_name", "image_path", "source_video", "timestamp_seconds",
        "frame_number", "width", "height", "extraction_pass"
    ]
    with open(all_candidates_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_headers)
        writer.writeheader()
        for item in all_candidates:
            writer.writerow({
                "image_name": item["image_name"],
                "image_path": item["image_path"],
                "source_video": item["source_video"],
                "timestamp_seconds": f"{item['timestamp_seconds']:.2f}",
                "frame_number": item["frame_number"],
                "width": item["width"],
                "height": item["height"],
                "extraction_pass": item["extraction_pass"]
            })

    print(f"[STEP 1] Created unified metadata: {all_candidates_csv.resolve()} ({len(all_candidates)} candidates)")

    # STEP 2: Near-Duplicate Detection
    grouped_by_video = defaultdict(list)
    for item in all_candidates:
        grouped_by_video[item["source_video"]].append(item)

    global_group_counter = 1
    filtered_records = []
    video_summary = []
    total_near_duplicates = 0
    total_retained = 0

    print(f"[STEP 2] Running near-duplicate detection (dHash Hamming distance threshold <= {SIMILARITY_THRESHOLD})...")

    for source_video, items in grouped_by_video.items():
        video_duplicates = 0
        video_retained = 0

        current_group_id = global_group_counter
        last_retained_hash = None

        for item in items:
            img = cv2.imread(str(item["abs_path"]))
            if img is None:
                # If image fails to read, retain it safely
                item["similarity_group_id"] = current_group_id
                item["selected_as_representative"] = 1
                last_retained_hash = None
                video_retained += 1
                current_group_id += 1
                filtered_records.append(item)
                continue

            img_hash = compute_dhash(img)

            if last_retained_hash is None:
                # First image in video stream
                item["similarity_group_id"] = current_group_id
                item["selected_as_representative"] = 1
                last_retained_hash = img_hash
                video_retained += 1
                filtered_records.append(item)
            else:
                dist = hamming_distance(last_retained_hash, img_hash)
                if dist <= SIMILARITY_THRESHOLD:
                    # Near-duplicate: mark as not representative
                    item["similarity_group_id"] = current_group_id
                    item["selected_as_representative"] = 0
                    video_duplicates += 1
                    filtered_records.append(item)
                else:
                    # Visually distinct: start new group and retain
                    current_group_id += 1
                    item["similarity_group_id"] = current_group_id
                    item["selected_as_representative"] = 1
                    last_retained_hash = img_hash
                    video_retained += 1
                    filtered_records.append(item)

        current_group_id += 1
        global_group_counter = current_group_id

        total_near_duplicates += video_duplicates
        total_retained += video_retained

        video_summary.append({
            "video": source_video,
            "total_candidates": len(items),
            "duplicates": video_duplicates,
            "retained": video_retained
        })

    # STEP 3: Create Review Outputs & Filtered Directory
    filtered_dir.mkdir(parents=True, exist_ok=True)
    review_filtered_dir.mkdir(parents=True, exist_ok=True)

    # Clean destination output directories
    for f_file in filtered_dir.glob("*"):
        if f_file.is_file():
            try: f_file.unlink()
            except Exception: pass

    for r_file in review_filtered_dir.glob("*"):
        if r_file.is_file():
            try: r_file.unlink()
            except Exception: pass

    # Write filtered_candidates.csv
    filtered_headers = [
        "image_name", "source_video", "timestamp_seconds",
        "frame_number", "width", "height", "similarity_group_id", "selected_as_representative"
    ]

    retained_items_by_video = defaultdict(list)

    with open(filtered_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=filtered_headers)
        writer.writeheader()

        for item in filtered_records:
            writer.writerow({
                "image_name": item["image_name"],
                "source_video": item["source_video"],
                "timestamp_seconds": f"{item['timestamp_seconds']:.2f}",
                "frame_number": item["frame_number"],
                "width": item["width"],
                "height": item["height"],
                "similarity_group_id": item["similarity_group_id"],
                "selected_as_representative": item["selected_as_representative"]
            })

            # Copy retained frames to filtered_candidates/
            if item["selected_as_representative"] == 1:
                dest_path = filtered_dir / item["image_name"]
                shutil.copy2(item["abs_path"], dest_path)
                retained_items_by_video[item["source_video"]].append(item)

    print(f"[STEP 3] Saved retained images to: {filtered_dir.resolve()}")
    print(f"[STEP 3] Saved filtered metadata CSV to: {filtered_csv.resolve()}")

    # Generate review contact sheets for retained candidates
    total_review_sheets = 0
    for source_video, items in retained_items_by_video.items():
        num_retained = len(items)
        num_sheets = math.ceil(num_retained / FRAMES_PER_SHEET)
        prefix = sanitize_filename(source_video)

        for s_idx in range(num_sheets):
            chunk = items[s_idx * FRAMES_PER_SHEET : (s_idx + 1) * FRAMES_PER_SHEET]
            sheet_img = create_filtered_contact_sheet(
                chunk_rows=chunk,
                filtered_dir=filtered_dir,
                source_video=source_video,
                sheet_index=s_idx + 1,
                total_sheets=num_sheets
            )

            if sheet_img is not None:
                sheet_filename = f"{prefix}_filtered_review_{s_idx + 1:02d}.jpg"
                sheet_path = review_filtered_dir / sheet_filename
                cv2.imwrite(str(sheet_path), sheet_img)
                total_review_sheets += 1

    print(f"[STEP 3] Saved {total_review_sheets} filtered review contact sheets to: {review_filtered_dir.resolve()}\n")

    # STEP 4 & REPORT: Summary Display
    max_fn_len = max(len(rec["video"]) for rec in video_summary) if video_summary else 32
    max_fn_len = max(max_fn_len, 32)

    header_str = (
        f"{'Source Video Filename':<{max_fn_len}} | {'Total Candidates':<18} | "
        f"{'Near-Duplicates':<16} | {'Retained Candidates':<20}"
    )
    line_len = len(header_str)

    print("=" * line_len)
    print("NEAR-DUPLICATE FILTERING & REVIEW PIPELINE SUMMARY".center(line_len))
    print("=" * line_len)
    print(header_str)
    print("-" * line_len)

    for rec in video_summary:
        print(
            f"{rec['video']:<{max_fn_len}} | "
            f"{rec['total_candidates']:<18} | "
            f"{rec['duplicates']:<16} | "
            f"{rec['retained']:<20}"
        )

    print("-" * line_len)
    print(
        f"{f'TOTAL ({len(video_summary)} videos)':<{max_fn_len}} | "
        f"{len(all_candidates):<18} | "
        f"{total_near_duplicates:<16} | "
        f"{total_retained:<20}"
    )
    print("=" * line_len + "\n")

if __name__ == "__main__":
    run_filter_and_review_pipeline()
