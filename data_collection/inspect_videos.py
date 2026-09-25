import math
import os
from pathlib import Path
import cv2

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

def format_duration(seconds: float) -> str:
    """Format total seconds into 'Xm Ys' format."""
    total_sec = int(round(seconds))
    minutes = total_sec // 60
    rem_sec = total_sec % 60
    return f"{minutes}m {rem_sec}s"

def inspect_videos(videos_dir: Path):
    """Find, inspect, and summarize all video files in the target directory."""
    if not videos_dir.exists():
        print(f"[WARNING] Directory not found: {videos_dir}")
        return

    # Find all supported video files (sorted for consistent output)
    video_files = [
        f for f in sorted(videos_dir.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not video_files:
        print(f"[WARNING] No video files found in {videos_dir}")
        return

    summary_data = []

    print(f"Found {len(video_files)} video(s) in '{videos_dir}'. Inspecting...\n")

    for filepath in video_files:
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            print(f"[WARNING] Could not open video file: {filepath.name}\n")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
            print(f"[WARNING] Invalid video metadata for file: {filepath.name}\n")
            continue

        duration_sec = total_frames / fps
        duration_str = format_duration(duration_sec)
        frames_05fps = int(round(duration_sec * 0.5))
        frames_1fps = int(round(duration_sec * 1.0))

        # Per-video details display
        print("=" * 60)
        print(f"Video: {filepath.name}")
        print(f"Resolution: {width} x {height}")
        print(f"FPS: {fps:.2f}")
        print(f"Frames: {total_frames}")
        print(f"Duration: {duration_sec:.2f} sec ({duration_str})")
        print(f"Frames @ 0.5 FPS: {frames_05fps}")
        print(f"Frames @ 1 FPS: {frames_1fps}")
        print("=" * 60)
        print()

        summary_data.append({
            "filename": filepath.name,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "frames": total_frames,
            "duration_sec": duration_sec,
            "duration_str": duration_str,
            "frames_05fps": frames_05fps,
            "frames_1fps": frames_1fps,
        })

    if not summary_data:
        print("[WARNING] No valid video metadata collected.")
        return

    # Print Final Summary Table
    max_fn_len = max(len(item["filename"]) for item in summary_data)
    max_fn_len = max(max_fn_len, 28)

    title = "VIDEO INSPECTION SUMMARY"
    header_line = f"{'Filename':<{max_fn_len}} | {'Resolution':<11} | {'FPS':<7} | {'Frames':<7} | {'Duration':<10} | {'@ 0.5 FPS':<9} | {'@ 1 FPS':<7}"
    separator = "-" * len(header_line)
    banner = "=" * len(header_line)

    print(banner)
    print(title.center(len(header_line)))
    print(banner)
    print(header_line)
    print(separator)

    total_duration_sec = 0.0
    total_frames_count = 0
    total_05fps = 0
    total_1fps = 0

    for item in summary_data:
        print(
            f"{item['filename']:<{max_fn_len}} | "
            f"{item['resolution']:<11} | "
            f"{item['fps']:<7.2f} | "
            f"{item['frames']:<7} | "
            f"{item['duration_str']:<10} | "
            f"{item['frames_05fps']:<9} | "
            f"{item['frames_1fps']:<7}"
        )
        total_duration_sec += item["duration_sec"]
        total_frames_count += item["frames"]
        total_05fps += item["frames_05fps"]
        total_1fps += item["frames_1fps"]

    print(separator)
    tot_dur_str = format_duration(total_duration_sec)
    tot_label = f"TOTAL ({len(summary_data)} videos)"
    print(
        f"{tot_label:<{max_fn_len}} | "
        f"{'-':<11} | "
        f"{'-':<7} | "
        f"{total_frames_count:<7} | "
        f"{tot_dur_str:<10} | "
        f"{total_05fps:<9} | "
        f"{total_1fps:<7}"
    )
    print(banner)

if __name__ == "__main__":
    # Resolve directory relative to project root or execution location
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    target_dir = project_root / "data_collection" / "videos"
    if not target_dir.exists():
        # Fallback to script_dir / "videos" or Path("data_collection/videos")
        target_dir = Path("data_collection/videos")

    inspect_videos(target_dir)
