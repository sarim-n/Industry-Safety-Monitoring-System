import csv
import math
import os
import re
from pathlib import Path
import cv2

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SAMPLE_FPS = 2  # Extract ~2 frames per second

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

def extract_candidate_frames(videos_dir: Path, candidates_dir: Path, csv_path: Path):
    """Extract candidate frames at SAMPLE_FPS and write metadata to CSV."""
    if not videos_dir.exists():
        print(f"[ERROR] Videos directory does not exist: {videos_dir}")
        return

    candidates_dir.mkdir(parents=True, exist_ok=True)

    video_files = [
        f for f in sorted(videos_dir.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not video_files:
        print(f"[WARNING] No video files found in {videos_dir}")
        return

    print(f"Extracting candidate frames (~{SAMPLE_FPS} FPS) from {len(video_files)} video(s)...\n")

    summary_records = []
    all_csv_rows = []
    total_candidates_extracted = 0
    total_duration_all = 0.0

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
        total_duration_all += duration_sec

        step = max(1, int(round(fps / SAMPLE_FPS)))
        prefix = sanitize_filename(filepath.name)

        video_candidates_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                timestamp = frame_idx / fps
                img_name = f"{prefix}_t{timestamp:05.2f}_f{frame_idx:04d}.jpg"
                out_path = candidates_dir / img_name

                # Save original resolution image
                cv2.imwrite(str(out_path), frame)

                all_csv_rows.append({
                    "image_name": img_name,
                    "source_video": filepath.name,
                    "timestamp_seconds": f"{timestamp:.2f}",
                    "frame_number": frame_idx,
                    "width": width,
                    "height": height
                })

                video_candidates_count += 1

            frame_idx += 1

        cap.release()
        total_candidates_extracted += video_candidates_count

        summary_records.append({
            "video": filepath.name,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "duration_sec": duration_sec,
            "duration_str": format_duration(duration_sec),
            "candidates": video_candidates_count
        })

        print(f"[{filepath.name}] Duration: {format_duration(duration_sec)} | FPS: {fps:.2f} | Extracted: {video_candidates_count} frames")

    # Write Metadata CSV
    csv_headers = ["image_name", "source_video", "timestamp_seconds", "frame_number", "width", "height"]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(all_csv_rows)

    print(f"\nSaved candidates metadata to: {csv_path.resolve()}")

    # Print Summary Table
    max_fn_len = max(len(rec["video"]) for rec in summary_records) if summary_records else 32
    max_fn_len = max(max_fn_len, 32)

    header_str = f"{'Video Filename':<{max_fn_len}} | {'Resolution':<11} | {'FPS':<7} | {'Duration':<10} | {'Candidates Extracted':<20}"
    line_len = len(header_str)

    print("\n" + "=" * line_len)
    print("CANDIDATE FRAME EXTRACTION SUMMARY".center(line_len))
    print("=" * line_len)
    print(header_str)
    print("-" * line_len)

    for rec in summary_records:
        print(
            f"{rec['video']:<{max_fn_len}} | "
            f"{rec['resolution']:<11} | "
            f"{rec['fps']:<7.2f} | "
            f"{rec['duration_str']:<10} | "
            f"{rec['candidates']:<20}"
        )

    print("-" * line_len)
    tot_dur_str = format_duration(total_duration_all)
    print(
        f"{f'TOTAL ({len(summary_records)} videos)':<{max_fn_len}} | "
        f"{'-':<11} | "
        f"{'-':<7} | "
        f"{tot_dur_str:<10} | "
        f"{total_candidates_extracted:<20}"
    )
    print("=" * line_len + "\n")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    input_videos_dir = project_root / "data_collection" / "videos"
    output_candidates_dir = project_root / "data_collection" / "candidates"
    output_csv_path = project_root / "data_collection" / "candidates.csv"

    extract_candidate_frames(input_videos_dir, output_candidates_dir, output_csv_path)
