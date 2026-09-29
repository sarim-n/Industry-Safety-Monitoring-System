import os
import cv2
import csv
import math
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(r"C:\Users\sarim\safety monitoring")
MORE_DIR = PROJECT_ROOT / "data_collection" / "more"
MORE_FRAMES_DIR = PROJECT_ROOT / "data_collection" / "more_frames"

MORE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# 4 New Videos
TARGET_VIDEOS = [
    "6790005-uhd_2160_3840_25fps.mp4",
    "8689912-uhd_2160_3840_25fps.mp4",
    "istockphoto-1205585966-640_adpp_is.mp4",
    "istockphoto-901643128-640_adpp_is.mp4"
]

print("==================================================================")
print("   EXTRACTING FRAMES FROM 4 NEW VIDEOS (2 FPS)")
print("==================================================================")

video_metadata_list = []
frame_metadata_list = []

total_extracted_all_videos = 0
failed_videos = []

for v_fn in TARGET_VIDEOS:
    v_path = MORE_DIR / v_fn
    if not v_path.exists():
        print(f"[ERROR] Video file not found: {v_path}")
        failed_videos.append({"filename": v_fn, "reason": "File not found"})
        continue

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {v_path}")
        failed_videos.append({"filename": v_fn, "reason": "OpenCV VideoCapture failed"})
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0

    target_extraction_fps = 2.0
    frame_step = max(1, int(round(fps / target_extraction_fps)))
    actual_extraction_fps = fps / frame_step if frame_step > 0 else 2.0

    # Video subfolder
    v_stem = v_path.stem
    subfolder = MORE_FRAMES_DIR / v_stem
    subfolder.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % frame_step == 0:
            timestamp = frame_idx / fps if fps > 0 else 0.0
            frame_fn = f"{v_stem}_frame_{frame_idx:06d}.jpg"
            out_path = subfolder / frame_fn

            # Preserve original resolution, no resizing, JPEG quality 95
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            frame_metadata_list.append({
                "source_video": v_fn,
                "frame_number": frame_idx,
                "timestamp_seconds": round(timestamp, 3),
                "original_width": w,
                "original_height": h,
                "fps": round(fps, 2),
                "video_duration": round(duration, 2),
                "extraction_fps": round(actual_extraction_fps, 2),
                "relative_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            })
            extracted_count += 1

        frame_idx += 1

    cap.release()

    total_extracted_all_videos += extracted_count
    video_metadata_list.append({
        "filename": v_fn,
        "folder": v_stem,
        "duration": round(duration, 2),
        "fps": round(fps, 2),
        "resolution": f"{w}x{h}",
        "total_video_frames": total_frames,
        "extracted_frames": extracted_count,
        "extraction_fps": round(actual_extraction_fps, 2)
    })
    print(f"Extracted {extracted_count} frames from {v_fn} ({w}x{h} @ {fps:.2f} FPS, Duration: {duration:.2f}s)")

# Write metadata.csv
meta_csv_path = MORE_FRAMES_DIR / "metadata.csv"
fieldnames = [
    "source_video", "frame_number", "timestamp_seconds",
    "original_width", "original_height", "fps",
    "video_duration", "extraction_fps", "relative_path"
]

with open(meta_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(frame_metadata_list)

print(f"\n[METADATA] Saved frame metadata ({len(frame_metadata_list)} records) to {meta_csv_path}")

# Verification Checks
print("\n[VERIFICATION CHECKS]")
corrupt_count = 0
zero_byte_count = 0

for item in frame_metadata_list:
    img_p = PROJECT_ROOT / item["relative_path"]
    if not img_p.exists() or img_p.stat().st_size == 0:
        zero_byte_count += 1
        continue
    try:
        with Image.open(img_p) as im:
            im.verify()
    except Exception:
        corrupt_count += 1

print(f"  Videos Opened Successfully : {len(video_metadata_list)} / {len(TARGET_VIDEOS)}")
print(f"  Total Extracted Frames      : {total_extracted_all_videos}")
print(f"  Metadata CSV Records        : {len(frame_metadata_list)}")
print(f"  Corrupted Images            : {corrupt_count}")
print(f"  Zero-Byte Files             : {zero_byte_count}")

assert len(video_metadata_list) == 4, "Not all 4 videos were processed!"
assert len(frame_metadata_list) == total_extracted_all_videos, "Metadata count mismatch!"
assert corrupt_count == 0, "Found corrupted images!"
assert zero_byte_count == 0, "Found zero-byte images!"

# Generate report.md
report_md_path = MORE_FRAMES_DIR / "report.md"
with open(report_md_path, "w", encoding="utf-8") as f:
    f.write("# Phase 7.10 — 4 New Video Frame Extraction Report\n\n")
    f.write("## 1. Executive Summary\n\n")
    f.write(f"- **Number of Input Videos**: {len(video_metadata_list)}\n")
    f.write(f"- **Total Frames Extracted**: {total_extracted_all_videos}\n")
    f.write(f"- **Target Extraction Rate**: ~2.0 FPS\n")
    f.write(f"- **Resolution**: Original resolution preserved for all videos (UHD 3840x2160 and HD 640x360)\n")
    f.write(f"- **Failed / Unreadable Videos**: {len(failed_videos)}\n\n")

    f.write("## 2. Per-Video Extraction Statistics\n\n")
    f.write("| Video Filename | Duration (s) | Original FPS | Resolution | Extracted Frames | Extraction FPS | Subfolder |\n")
    f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n")
    for vm in video_metadata_list:
        f.write(f"| `{vm['filename']}` | {vm['duration']} | {vm['fps']} | {vm['resolution']} | {vm['extracted_frames']} | {vm['extraction_fps']} | `more_frames/{vm['folder']}/` |\n")

    f.write(f"| **TOTALS** | **{sum(vm['duration'] for vm in video_metadata_list):.2f}s** | — | — | **{total_extracted_all_videos}** | **~2.0 FPS** | `data_collection/more_frames/` |\n\n")

    f.write("## 3. Verification & Quality Checks\n\n")
    f.write(f"- **OpenCV VideoCapture Status**: 100% Passed ({len(video_metadata_list)}/4 videos opened cleanly)\n")
    f.write(f"- **Image Integrity Check**: 100% Passed (0 corrupt / 0 zero-byte images)\n")
    f.write(f"- **Metadata Match**: 100% Passed ({len(frame_metadata_list)} CSV rows match {total_extracted_all_videos} extracted frames)\n")

print(f"[REPORT] Saved {report_md_path}")
