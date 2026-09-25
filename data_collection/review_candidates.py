import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

FRAMES_PER_SHEET = 20  # 5 cols x 4 rows
THUMB_WIDTH = 320
COLS = 5

def sanitize_filename(filename: str) -> str:
    """Sanitize original video filename for clean contact sheet file prefix."""
    stem = Path(filename).stem
    clean = re.sub(r'[^a-zA-Z0-9]+', '_', stem).strip('_')
    return clean.lower()

def format_timestamp(seconds_str: str) -> str:
    """Format seconds float string into MM:SS format."""
    sec = float(seconds_str)
    total_sec = int(round(sec))
    mins = total_sec // 60
    rem_sec = total_sec % 60
    return f"{mins:02d}:{rem_sec:02d}"

def create_candidate_contact_sheet(
    chunk_rows: list,
    candidates_dir: Path,
    source_video: str,
    sheet_index: int,
    total_sheets: int
) -> np.ndarray:
    """Create a contact sheet grid image for a chunk of candidate frame rows."""
    if not chunk_rows:
        return None

    # Read first image to determine target thumbnail aspect ratio
    first_img_path = candidates_dir / chunk_rows[0]["image_name"]
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

    # Dark background canvas
    canvas = np.full((canvas_h, canvas_w, 3), (28, 28, 32), dtype=np.uint8)

    # Draw Header Banner
    header_text = f"Video: {source_video}  |  Review Sheet {sheet_index}/{total_sheets}  |  Candidates in Sheet: {num_items}"
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

    # Place Thumbnails
    for idx, item in enumerate(chunk_rows):
        r = idx // cols
        c = idx % cols

        x = margin + c * (thumb_w + gap)
        y = header_height + margin + r * (thumb_h + gap)

        img_path = candidates_dir / item["image_name"]
        frame_img = cv2.imread(str(img_path))

        if frame_img is None:
            # Black thumbnail placeholder if image fails to load
            resized = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
        else:
            resized = cv2.resize(frame_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        # Label Overlay on thumbnail (Timestamp & Frame Number)
        ts_sec = float(item["timestamp_seconds"])
        label_text = f" t={ts_sec:.2f}s ({format_timestamp(item['timestamp_seconds'])}) | f={item['frame_number']} "

        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Draw dark semi-transparent banner at bottom of thumbnail
        banner_h = th + baseline + 8
        cv2.rectangle(resized, (0, thumb_h - banner_h), (thumb_w, thumb_h), (0, 0, 0), -1)
        cv2.putText(
            resized,
            label_text,
            (6, thumb_h - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),  # Yellow
            thickness,
            cv2.LINE_AA
        )

        # Border
        cv2.rectangle(resized, (0, 0), (thumb_w - 1, thumb_h - 1), (80, 80, 95), 1)

        canvas[y:y + thumb_h, x:x + thumb_w] = resized

    return canvas

def generate_candidate_review():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    candidates_csv_path = project_root / "data_collection" / "candidates.csv"
    candidates_dir = project_root / "data_collection" / "candidates"
    review_output_dir = project_root / "data_collection" / "candidate_review"
    index_csv_path = review_output_dir / "index.csv"

    if not candidates_csv_path.exists():
        print(f"[ERROR] candidates.csv not found at: {candidates_csv_path}")
        return

    if not candidates_dir.exists():
        print(f"[ERROR] candidates directory not found at: {candidates_dir}")
        return

    review_output_dir.mkdir(parents=True, exist_ok=True)

    # Clean old contact sheet images & index file in candidate_review directory
    for old_file in review_output_dir.glob("*"):
        if old_file.is_file():
            try:
                old_file.unlink()
            except Exception:
                pass

    # Read candidates metadata
    with open(candidates_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        candidates_data = list(reader)

    if not candidates_data:
        print("[WARNING] candidates.csv is empty.")
        return

    # Group candidates by source_video preserving insertion order
    grouped_videos = defaultdict(list)
    for row in candidates_data:
        grouped_videos[row["source_video"]].append(row)

    print(f"Loaded {len(candidates_data)} candidate frame records from {len(grouped_videos)} videos.")
    print("Generating candidate review contact sheets...\n")

    index_rows = []
    total_sheets_generated = 0
    per_video_summary = []

    for source_video, items in grouped_videos.items():
        # Sort items by frame number
        items.sort(key=lambda x: int(x["frame_number"]))
        num_candidates = len(items)

        num_sheets = math.ceil(num_candidates / FRAMES_PER_SHEET)
        prefix = sanitize_filename(source_video)

        video_sheets_count = 0

        for s_idx in range(num_sheets):
            chunk = items[s_idx * FRAMES_PER_SHEET : (s_idx + 1) * FRAMES_PER_SHEET]
            sheet_img = create_candidate_contact_sheet(
                chunk_rows=chunk,
                candidates_dir=candidates_dir,
                source_video=source_video,
                sheet_index=s_idx + 1,
                total_sheets=num_sheets
            )

            if sheet_img is not None:
                sheet_name = f"{prefix}_review_{s_idx + 1:02d}.jpg"
                sheet_save_path = review_output_dir / sheet_name
                cv2.imwrite(str(sheet_save_path), sheet_img)
                video_sheets_count += 1
                total_sheets_generated += 1

                # Record index mapping
                for item in chunk:
                    index_rows.append({
                        "contact_sheet": sheet_name,
                        "image_name": item["image_name"],
                        "source_video": item["source_video"],
                        "timestamp_seconds": item["timestamp_seconds"]
                    })

        per_video_summary.append({
            "video": source_video,
            "candidates": num_candidates,
            "sheets": video_sheets_count
        })

    # Save index.csv
    index_headers = ["contact_sheet", "image_name", "source_video", "timestamp_seconds"]
    with open(index_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=index_headers)
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"Saved review index file to: {index_csv_path.resolve()}")

    # Print Summary Table
    max_fn_len = max(len(rec["video"]) for rec in per_video_summary) if per_video_summary else 32
    max_fn_len = max(max_fn_len, 32)

    header_str = f"{'Source Video':<{max_fn_len}} | {'Candidates Count':<18} | {'Review Sheets':<14}"
    line_len = len(header_str)

    print("\n" + "=" * line_len)
    print("CANDIDATE REVIEW GENERATION SUMMARY".center(line_len))
    print("=" * line_len)
    print(header_str)
    print("-" * line_len)

    for rec in per_video_summary:
        print(
            f"{rec['video']:<{max_fn_len}} | "
            f"{rec['candidates']:<18} | "
            f"{rec['sheets']:<14}"
        )

    print("-" * line_len)
    print(
        f"{f'TOTAL ({len(per_video_summary)} videos)':<{max_fn_len}} | "
        f"{len(candidates_data):<18} | "
        f"{total_sheets_generated:<14}"
    )
    print("=" * line_len + "\n")

if __name__ == "__main__":
    generate_candidate_review()
