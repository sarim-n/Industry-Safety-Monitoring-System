import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
import cv2

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ADDITIONAL_FPS = 4  # Target sampling rate for 2nd pass (~4 FPS)

def sanitize_filename(filename: str) -> str:
    """Sanitize original video filename for clean frame file prefix."""
    stem = Path(filename).stem
    clean = re.sub(r'[^a-zA-Z0-9]+', '_', stem).strip('_')
    return clean.lower()

def format_duration(seconds: float) -> str:
    """Format seconds into 'Xm Ys' format."""
    total_sec = int(round(seconds))
    mins = total_sec // 60
    secs = total_sec % 60
    return f"{mins}m {secs}s"

def extract_additional_candidates():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    videos_dir = project_root / "data_collection" / "videos"
    existing_csv_path = project_root / "data_collection" / "candidates.csv"
    existing_candidates_dir = project_root / "data_collection" / "candidates"

    additional_candidates_dir = project_root / "data_collection" / "candidates_additional"
    additional_csv_path = project_root / "data_collection" / "candidates_additional.csv"

    if not videos_dir.exists():
        print(f"[ERROR] Videos directory does not exist: {videos_dir}")
        return

    if not existing_csv_path.exists():
        print(f"[ERROR] Existing candidates.csv does not exist: {existing_csv_path}")
        return

    additional_candidates_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous additional outputs if re-run
    for old_file in additional_candidates_dir.glob("*.jpg"):
        try:
            old_file.unlink()
        except Exception:
            pass

    # Read existing candidate metadata to prevent duplicates
    existing_frame_keys = set()
    existing_counts_per_video = defaultdict(int)

    with open(existing_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_video = row["source_video"]
            f_num = int(row["frame_number"])
            ts_sec = float(row["timestamp_seconds"])

            existing_frame_keys.add((src_video, f_num))
            existing_frame_keys.add((src_video, f"{ts_sec:.2f}"))
            existing_counts_per_video[src_video] += 1

    video_files = [
        f for f in sorted(videos_dir.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not video_files:
        print(f"[WARNING] No video files found in {videos_dir}")
        return

    print(f"Loaded {len(existing_frame_keys)//2} existing candidate records across {len(existing_counts_per_video)} videos.")
    print(f"Extracting ADDITIONAL candidate frames (~{ADDITIONAL_FPS} FPS pass) from {len(video_files)} video(s)...\n")

    additional_csv_rows = []
    total_additional_extracted = 0
    per_video_summary = []

    for filepath in video_files:
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            print(f"[WARNING] Could not open video file: {filepath.name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
            print(f"[WARNING] Invalid metadata for video: {filepath.name}")
            cap.release()
            continue

        duration_sec = total_frames / fps
        step = max(1, int(round(fps / ADDITIONAL_FPS)))
        prefix = sanitize_filename(filepath.name)

        video_additional_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                timestamp = frame_idx / fps
                ts_str = f"{timestamp:.2f}"

                # Skip duplicate frame if already extracted in Pass 1
                if (filepath.name, frame_idx) in existing_frame_keys or (filepath.name, ts_str) in existing_frame_keys:
                    frame_idx += 1
                    continue

                img_name = f"{prefix}_t{timestamp:05.2f}_f{frame_idx:04d}.jpg"
                out_path = additional_candidates_dir / img_name

                # Save original resolution frame
                cv2.imwrite(str(out_path), frame)

                additional_csv_rows.append({
                    "image_name": img_name,
                    "source_video": filepath.name,
                    "timestamp_seconds": ts_str,
                    "frame_number": frame_idx,
                    "width": width,
                    "height": height
                })

                video_additional_count += 1

            frame_idx += 1

        cap.release()
        total_additional_extracted += video_additional_count

        exist_count = existing_counts_per_video[filepath.name]
        combined_count = exist_count + video_additional_count

        per_video_summary.append({
            "video": filepath.name,
            "duration": format_duration(duration_sec),
            "existing": exist_count,
            "additional": video_additional_count,
            "total_pool": combined_count
        })

        print(f"[{filepath.name}] Existing: {exist_count} | Additional: {video_additional_count} | Total Pool: {combined_count}")

    # Write additional CSV
    csv_headers = ["image_name", "source_video", "timestamp_seconds", "frame_number", "width", "height"]
    with open(additional_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(additional_csv_rows)

    print(f"\nSaved additional candidates metadata to: {additional_csv_path.resolve()}")

    # Print Summary Table
    max_fn_len = max(len(rec["video"]) for rec in per_video_summary) if per_video_summary else 32
    max_fn_len = max(max_fn_len, 32)

    header_str = (
        f"{'Source Video Filename':<{max_fn_len}} | {'Duration':<10} | "
        f"{'Pass 1 (~2 FPS)':<15} | {'Pass 2 (~4 FPS Add)':<18} | {'Total Candidate Pool':<20}"
    )
    line_len = len(header_str)

    print("\n" + "=" * line_len)
    print("CANDIDATE EXPANSION SUMMARY (PASS 2)".center(line_len))
    print("=" * line_len)
    print(header_str)
    print("-" * line_len)

    total_exist_sum = 0
    total_add_sum = 0
    total_pool_sum = 0

    for rec in per_video_summary:
        print(
            f"{rec['video']:<{max_fn_len}} | "
            f"{rec['duration']:<10} | "
            f"{rec['existing']:<15} | "
            f"{rec['additional']:<18} | "
            f"{rec['total_pool']:<20}"
        )
        total_exist_sum += rec["existing"]
        total_add_sum += rec["additional"]
        total_pool_sum += rec["total_pool"]

    print("-" * line_len)
    print(
        f"{f'TOTAL ({len(per_video_summary)} videos)':<{max_fn_len}} | "
        f"{'-':<10} | "
        f"{total_exist_sum:<15} | "
        f"{total_add_sum:<18} | "
        f"{total_pool_sum:<20}"
    )
    print("=" * line_len + "\n")

if __name__ == "__main__":
    extract_additional_candidates()
