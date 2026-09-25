import math
import os
import re
from pathlib import Path
import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
FRAMES_PER_SHEET = 16  # 4x4 grid (fits 12-20 requirement per contact sheet)
THUMB_WIDTH = 320
COLS = 4

def sanitize_filename(filename: str) -> str:
    """Create a clean prefix for preview image files."""
    stem = Path(filename).stem
    # Replace non-alphanumeric characters with underscores
    clean = re.sub(r'[^a-zA-Z0-9]+', '_', stem).strip('_')
    return clean.lower()

def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS format."""
    total_sec = int(round(seconds))
    mins = total_sec // 60
    secs = total_sec % 60
    return f"{mins:02d}:{secs:02d}"

def create_contact_sheet(
    sampled_frames: list,
    video_filename: str,
    sheet_index: int,
    total_sheets: int,
    fps: float,
    duration_sec: float
) -> np.ndarray:
    """Generate a single contact sheet image from a list of (timestamp, frame) tuples."""
    if not sampled_frames:
        return None

    # Determine thumbnail dimensions preserving original aspect ratio
    sample_img = sampled_frames[0][1]
    src_h, src_w = sample_img.shape[:2]
    thumb_w = THUMB_WIDTH
    thumb_h = max(1, int(round(thumb_w * src_h / src_w)))

    num_frames = len(sampled_frames)
    cols = COLS
    rows = math.ceil(num_frames / cols)

    margin = 15
    gap = 10
    header_height = 50

    canvas_w = margin * 2 + cols * thumb_w + (cols - 1) * gap
    canvas_h = header_height + margin * 2 + rows * thumb_h + (rows - 1) * gap

    # Dark charcoal background
    canvas = np.full((canvas_h, canvas_w, 3), (30, 30, 34), dtype=np.uint8)

    # Draw Header Banner
    header_text = (
        f"Video: {video_filename}  |  Sheet {sheet_index}/{total_sheets}  |  "
        f"FPS: {fps:.1f}  |  Duration: {format_timestamp(duration_sec)}"
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

    # Place Thumbnails
    for i, (timestamp_sec, frame) in enumerate(sampled_frames):
        row = i // cols
        col = i % cols

        x = margin + col * (thumb_w + gap)
        y = header_height + margin + row * (thumb_h + gap)

        resized = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        # Draw timestamp overlay on thumbnail
        ts_str = f" {format_timestamp(timestamp_sec)} "
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(ts_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Background box for timestamp text at bottom-right
        tx = thumb_w - tw - 6
        ty = thumb_h - 8
        cv2.rectangle(
            resized,
            (tx - 2, ty - th - 4),
            (tx + tw + 2, ty + baseline),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            resized,
            ts_str,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),  # Yellow text
            thickness,
            cv2.LINE_AA
        )

        # Thumbnail border
        cv2.rectangle(resized, (0, 0), (thumb_w - 1, thumb_h - 1), (80, 80, 90), 1)

        # Copy into canvas
        canvas[y:y + thumb_h, x:x + thumb_w] = resized

    return canvas

def generate_video_previews(videos_dir: Path, output_dir: Path):
    """Read videos from videos_dir, sample @ 1 FPS, and save contact sheets in output_dir."""
    if not videos_dir.exists():
        print(f"[ERROR] Videos directory does not exist: {videos_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    # Clean old preview files to ensure fresh output
    for old_file in output_dir.glob("*.jpg"):
        try:
            old_file.unlink()
        except Exception:
            pass

    video_files = [
        f for f in sorted(videos_dir.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not video_files:
        print(f"[WARNING] No video files found in {videos_dir}")
        return

    print(f"Processing {len(video_files)} video(s) for preview generation...\n")

    summary_records = []
    total_sheets_generated = 0

    for filepath in video_files:
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            print(f"[WARNING] Could not open video file: {filepath.name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            print(f"[WARNING] Invalid video metadata for {filepath.name}")
            cap.release()
            continue

        duration_sec = total_frames / fps
        step = max(1, int(round(fps)))  # ~1 frame per second

        sampled_frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                timestamp = frame_idx / fps
                sampled_frames.append((timestamp, frame))

            frame_idx += 1

        cap.release()

        num_sampled = len(sampled_frames)
        if num_sampled == 0:
            print(f"[WARNING] No frames sampled for {filepath.name}")
            continue

        # Split sampled frames into chunks of FRAMES_PER_SHEET
        num_sheets = math.ceil(num_sampled / FRAMES_PER_SHEET)
        prefix = sanitize_filename(filepath.name)

        generated_files = []
        for s_idx in range(num_sheets):
            chunk = sampled_frames[s_idx * FRAMES_PER_SHEET : (s_idx + 1) * FRAMES_PER_SHEET]
            sheet_img = create_contact_sheet(
                sampled_frames=chunk,
                video_filename=filepath.name,
                sheet_index=s_idx + 1,
                total_sheets=num_sheets,
                fps=fps,
                duration_sec=duration_sec
            )

            if sheet_img is not None:
                out_filename = f"{prefix}_preview_{s_idx + 1:02d}.jpg"
                out_path = output_dir / out_filename
                cv2.imwrite(str(out_path), sheet_img)
                generated_files.append(out_filename)

        sheets_count = len(generated_files)
        total_sheets_generated += sheets_count

        mins = int(duration_sec // 60)
        secs = int(round(duration_sec % 60))
        dur_str = f"{mins}m {secs}s"

        summary_records.append({
            "video": filepath.name,
            "duration": dur_str,
            "preview_frames": num_sampled,
            "sheets_count": sheets_count
        })

    if not summary_records:
        print("[WARNING] No previews generated.")
        return

    # Print Summary Table
    max_fn_len = max(len(rec["video"]) for rec in summary_records)
    max_fn_len = max(max_fn_len, 32)

    header_str = f"{'Video Filename':<{max_fn_len}} | {'Duration':<10} | {'Preview Frames':<16} | {'Contact Sheets':<14}"
    line_len = len(header_str)

    print("\n" + "=" * line_len)
    print("PREVIEW GENERATION SUMMARY".center(line_len))
    print("=" * line_len)
    print(header_str)
    print("-" * line_len)

    for rec in summary_records:
        print(
            f"{rec['video']:<{max_fn_len}} | "
            f"{rec['duration']:<10} | "
            f"{rec['preview_frames']:<16} | "
            f"{rec['sheets_count']:<14}"
        )

    print("-" * line_len)
    print(f"Total Contact Sheets Generated: {total_sheets_generated}")
    print(f"Saved Previews Location: {output_dir.resolve()}")
    print("=" * line_len + "\n")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    input_videos_dir = project_root / "data_collection" / "videos"
    output_previews_dir = project_root / "data_collection" / "video_previews"

    generate_video_previews(input_videos_dir, output_previews_dir)
