"""
isolate_unlabeled.py
====================
Isolates raw images that have NOT been labeled in Roboflow yet.

Strategy
--------
Roboflow renames files by appending a hash suffix, e.g.:
  raw:       8482302_hd_1920_1080_25fps_t00-24_f0006.jpg
  roboflow:  8482302_hd_1920_1080_25fps_t00-24_f0006_jpg.rf.LCEuFKEGcH3R2gL47TZu.jpg

The raw stem is always a substring of the roboflow filename stem (before the
first ".rf." token). So we match by checking:
  raw_stem  in  roboflow_filename   (case-insensitive, both lowercased)
"""

import pathlib
import shutil

# Configuration
BASE = pathlib.Path(r"C:\Users\sarim\safety monitoring")

RAW_DIRS = [
    BASE / "data_collection" / "filtered_candidates",
]

ROBOFLOW_DIR  = BASE / "roboflow"
OUTPUT_DIR    = BASE / "unlabeled_images"
IMAGE_EXTS    = {".jpg", ".jpeg", ".png"}

# Step 1: Collect all Roboflow image filenames (lowercased)
print("=" * 60)
print("Step 1 - Scanning Roboflow folder ...")
roboflow_names = set()
for img_path in ROBOFLOW_DIR.rglob("*"):
    if img_path.suffix.lower() in IMAGE_EXTS and img_path.is_file():
        roboflow_names.add(img_path.name.lower())

labeled_count = len(roboflow_names)
print(f"  Roboflow images found : {labeled_count}")

# Step 2: Collect all raw images (de-duplicated by name)
print("\nStep 2 - Scanning raw image folders ...")
raw_images = {}
for raw_dir in RAW_DIRS:
    if not raw_dir.exists():
        print(f"  [WARN] Directory not found, skipping: {raw_dir}")
        continue
    for img_path in raw_dir.rglob("*"):
        if img_path.suffix.lower() in IMAGE_EXTS and img_path.is_file():
            key = img_path.name.lower()
            if key not in raw_images:
                raw_images[key] = img_path

total_raw = len(raw_images)
print(f"  Unique raw images found: {total_raw}")

# Step 3: Match raw images against Roboflow filenames
# NOTE: Roboflow replaces dots with dashes in filenames, e.g.
#   raw:       10810476_hd_1920_1080_30fps_t00.23_f0007.jpg
#   roboflow:  10810476_hd_1920_1080_30fps_t00-23_f0007_jpg.rf.<hash>.jpg
# Strategy: normalise both sides by replacing '.' with '-' before matching.
print("\nStep 3 - Matching ...")
labeled_raw   = []
unlabeled_raw = []

# Build a normalised set of roboflow name-parts (stems, dots→dashes)
roboflow_normalised = {name.replace(".", "-") for name in roboflow_names}
roboflow_concat_norm = "\n".join(roboflow_normalised)

for raw_name_lower, raw_path in raw_images.items():
    raw_stem      = pathlib.Path(raw_name_lower).stem          # e.g. "10810476_hd_1920_1080_30fps_t00.23_f0007"
    raw_stem_norm = raw_stem.replace(".", "-")                  # e.g. "10810476_hd_1920_1080_30fps_t00-23_f0007"

    # Exact filename match (unlikely but safe)
    exact_match = raw_name_lower in roboflow_names

    # Normalised stem is a substring of any normalised roboflow name
    substring_match = raw_stem_norm in roboflow_concat_norm

    if exact_match or substring_match:
        labeled_raw.append(raw_name_lower)
    else:
        unlabeled_raw.append(raw_name_lower)

print(f"  Raw images matched to Roboflow : {len(labeled_raw)}")
print(f"  Raw images NOT in Roboflow     : {len(unlabeled_raw)}")

# Step 4: Copy unlabeled images to output folder
print(f"\nStep 4 - Copying unlabeled images to:\n  {OUTPUT_DIR}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

copied  = 0
skipped = 0
for raw_name_lower in unlabeled_raw:
    src = raw_images[raw_name_lower]
    dst = OUTPUT_DIR / src.name
    if dst.exists():
        skipped += 1
        continue
    shutil.copy2(src, dst)
    copied += 1

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Total labeled images found in roboflow : {labeled_count}")
print(f"  Total raw images scanned               : {total_raw}")
print(f"  Raw images matched (labeled)           : {len(labeled_raw)}")
print(f"  Unannotated images copied to output    : {copied}")
if skipped:
    print(f"  Already existed in output (skipped)    : {skipped}")
print(f"\n  Output directory: {OUTPUT_DIR}")
print("=" * 60)
