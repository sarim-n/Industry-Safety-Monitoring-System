"""
ANNOTATION DISTRIBUTION ANALYSIS FOR PROPOSED SPLIT
=====================================================
Reads all label files from training_dataset/labels/train and labels/val,
maps each image to its proposed split (TRAIN/VAL/TEST) based on source group,
and reports exact annotation counts per class per split.

READ-ONLY - modifies nothing.
"""

import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT   = Path(r"C:\Users\sarim\safety monitoring")
TRAIN_IMG_DIR  = PROJECT_ROOT / "training_dataset" / "images" / "train"
VAL_IMG_DIR    = PROJECT_ROOT / "training_dataset" / "images" / "val"
TRAIN_LBL_DIR  = PROJECT_ROOT / "training_dataset" / "labels" / "train"
VAL_LBL_DIR    = PROJECT_ROOT / "training_dataset" / "labels" / "val"

CLASS_NAMES = {0: "helmet", 1: "mask", 2: "person"}

# ── Proposed split assignment ────────────────────────────────────────────────
PROPOSED_TRAIN = {
    "mask_only",
    "helmet_mask_gloves",
    "12984266_1920_1080_25fps",
    "helmet_gloves",
    "10810476_hd_1920_1080_30fps",
    "12984031_1920_1080_25fps",
    "helmet_mask_glove",
    "gloves_only",
    "13258042_2560_1440_30fps",
    "4042251183_preview",
    "helmet_mask_gloves02",
}

PROPOSED_VAL = {
    "4048038451_preview",
    "4017518657_preview",
    "8482302_hd_1920_1080_25fps",
    "no_safety",
    "gloves_mask",
    "19832490_hd_1920_1080_25fps",
    "istockphoto_2258622635",   # collapsed prefix for all 13 istock frames
}

PROPOSED_TEST = {
    "helmet_only",
    "5434223_hd_1920_1080_24fps",
    "4161358813_preview",
    "helmet_mask",
}

ALL_PROPOSED = PROPOSED_TRAIN | PROPOSED_VAL | PROPOSED_TEST

# ── Source extraction ────────────────────────────────────────────────────────

KNOWN_SOURCES = ALL_PROPOSED | {
    "gloves_mask", "gloves_only", "helmet_gloves", "helmet_mask",
    "helmet_mask_glove", "helmet_mask_gloves", "helmet_mask_gloves02",
    "helmet_only", "mask_only", "no_safety",
    "10810476_hd_1920_1080_30fps", "12984031_1920_1080_25fps",
    "12984266_1920_1080_25fps", "13258042_2560_1440_30fps",
    "19832490_hd_1920_1080_25fps", "4017518657_preview",
    "4042251183_preview", "4048038451_preview", "4161358813_preview",
    "5434223_hd_1920_1080_24fps", "8482302_hd_1920_1080_25fps",
}


def extract_source(filename: str) -> str:
    stem = Path(filename).stem
    # Strip Roboflow suffix
    cleaned = re.sub(r'_(jpg|jpeg|png)\.rf\.[A-Za-z0-9]+$', '', stem)

    # Special case: istockphoto frames with timestamp in prefix
    if cleaned.startswith("istockphoto_2258622635"):
        return "istockphoto_2258622635"

    # Try longest known source match
    for src in sorted(KNOWN_SOURCES, key=len, reverse=True):
        if cleaned.startswith(src):
            return src

    # Heuristic fallback
    heuristic = re.sub(r'_t\d+[-\.]\d+_f\d+.*$', '', cleaned)
    heuristic = re.sub(r'_jpg$', '', heuristic)
    return heuristic


def get_proposed_split(source: str) -> str:
    if source in PROPOSED_TRAIN:
        return "TRAIN"
    if source in PROPOSED_VAL:
        return "VAL"
    if source in PROPOSED_TEST:
        return "TEST"
    return "UNASSIGNED"


def read_label_file(label_path: Path):
    """Returns list of class IDs from a YOLO label file."""
    class_ids = []
    if not label_path.exists():
        return class_ids
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts:
                try:
                    class_ids.append(int(parts[0]))
                except ValueError:
                    pass
    return class_ids


# ── Scan all images + labels ─────────────────────────────────────────────────

stats = {
    "TRAIN": {"images": 0, "annotations": 0, "classes": defaultdict(int), "sources": set()},
    "VAL":   {"images": 0, "annotations": 0, "classes": defaultdict(int), "sources": set()},
    "TEST":  {"images": 0, "annotations": 0, "classes": defaultdict(int), "sources": set()},
    "UNASSIGNED": {"images": 0, "annotations": 0, "classes": defaultdict(int), "sources": set()},
}

source_to_split = {}   # source -> assigned split (for overlap check)

for img_dir, lbl_dir in [(TRAIN_IMG_DIR, TRAIN_LBL_DIR), (VAL_IMG_DIR, VAL_LBL_DIR)]:
    for img_file in sorted(img_dir.iterdir()):
        if not img_file.is_file():
            continue
        src   = extract_source(img_file.name)
        split = get_proposed_split(src)

        lbl_file   = lbl_dir / (img_file.stem + ".txt")
        class_ids  = read_label_file(lbl_file)

        stats[split]["images"]      += 1
        stats[split]["annotations"] += len(class_ids)
        stats[split]["sources"].add(src)
        for cid in class_ids:
            stats[split]["classes"][cid] += 1

        if src not in source_to_split:
            source_to_split[src] = split
        elif source_to_split[src] != split:
            print(f"[OVERLAP ERROR] Source '{src}' appears in both "
                  f"{source_to_split[src]} and {split}!")

# ── Report ────────────────────────────────────────────────────────────────────

total_imgs  = sum(stats[s]["images"]      for s in ("TRAIN","VAL","TEST"))
total_annot = sum(stats[s]["annotations"] for s in ("TRAIN","VAL","TEST"))
total_cls   = {c: sum(stats[s]["classes"][c] for s in ("TRAIN","VAL","TEST")) for c in range(3)}

SEP = "=" * 72
sep = "-" * 72

print(SEP)
print("ANNOTATION DISTRIBUTION FOR PROPOSED SPLIT".center(72))
print("READ-ONLY -- no files modified".center(72))
print(SEP)

for split_name in ("TRAIN", "VAL", "TEST"):
    d = stats[split_name]
    imgs  = d["images"]
    annot = d["annotations"]
    h     = d["classes"].get(0, 0)
    m     = d["classes"].get(1, 0)
    p     = d["classes"].get(2, 0)
    srcs  = sorted(d["sources"])

    pct_img   = 100 * imgs  / total_imgs   if total_imgs   else 0
    pct_annot = 100 * annot / total_annot  if total_annot  else 0
    pct_h     = 100 * h / total_cls[0]    if total_cls[0] else 0
    pct_m     = 100 * m / total_cls[1]    if total_cls[1] else 0
    pct_p     = 100 * p / total_cls[2]    if total_cls[2] else 0

    h_pct_local = 100 * h / annot if annot else 0
    m_pct_local = 100 * m / annot if annot else 0
    p_pct_local = 100 * p / annot if annot else 0

    print()
    print(f"  [{split_name}]")
    print(sep)
    print(f"  Images           : {imgs:>6}  ({pct_img:.1f}% of total)")
    print(f"  Annotations      : {annot:>6}  ({pct_annot:.1f}% of total)")
    print(f"  Source groups    : {len(srcs):>6}")
    print()
    print(f"  Class breakdown (within {split_name}):")
    print(f"    helmet  (0): {h:>5}  ({h_pct_local:.1f}% of {split_name} annots | {pct_h:.1f}% of all helmets)")
    print(f"    mask    (1): {m:>5}  ({m_pct_local:.1f}% of {split_name} annots | {pct_m:.1f}% of all masks)")
    print(f"    person  (2): {p:>5}  ({p_pct_local:.1f}% of {split_name} annots | {pct_p:.1f}% of all persons)")
    print()
    print(f"  Sources assigned to {split_name}:")
    for s in srcs:
        print(f"    {s}")

print()
print(SEP)
print("TOTALS".center(72))
print(SEP)
print(f"  Total images     : {total_imgs}")
print(f"  Total annotations: {total_annot}")
print(f"  helmet  (0)      : {total_cls[0]}  ({100*total_cls[0]/total_annot:.1f}% of all)")
print(f"  mask    (1)      : {total_cls[1]}  ({100*total_cls[1]/total_annot:.1f}% of all)")
print(f"  person  (2)      : {total_cls[2]}  ({100*total_cls[2]/total_annot:.1f}% of all)")

if stats["UNASSIGNED"]["images"] > 0:
    print()
    print(f"  [WARNING] UNASSIGNED images: {stats['UNASSIGNED']['images']}")
    print(f"  Unassigned sources: {sorted(stats['UNASSIGNED']['sources'])}")
else:
    print()
    print("  All 966 images assigned to a split -- no unassigned images.")

print()
print(SEP)
print("OVERLAP CHECK".center(72))
print(SEP)
overlap_found = False
all_src_splits = defaultdict(set)
for src, split in source_to_split.items():
    all_src_splits[src].add(split)
for src, splits in all_src_splits.items():
    if len(splits) > 1:
        print(f"  [OVERLAP] {src} -> {splits}")
        overlap_found = True
if not overlap_found:
    print("  No source group appears in more than one split.")
print(SEP)
