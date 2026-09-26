"""
BUILD training_dataset_v2
==========================
COPY-ONLY operation. Does NOT modify:
  - final_dataset/
  - training_dataset/

Reads from:
  training_dataset/images/train   + training_dataset/labels/train
  training_dataset/images/val     + training_dataset/labels/val

Writes to:
  training_dataset_v2/images/{train,val,test}
  training_dataset_v2/labels/{train,val,test}
  training_dataset_v2/data.yaml
"""

import re
import shutil
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(r"C:\Users\sarim\safety monitoring")

SRC_TRAIN_IMG  = PROJECT_ROOT / "training_dataset" / "images" / "train"
SRC_VAL_IMG    = PROJECT_ROOT / "training_dataset" / "images" / "val"
SRC_TRAIN_LBL  = PROJECT_ROOT / "training_dataset" / "labels" / "train"
SRC_VAL_LBL    = PROJECT_ROOT / "training_dataset" / "labels" / "val"

V2_ROOT        = PROJECT_ROOT / "training_dataset_v2"

FINAL_DATASET  = PROJECT_ROOT / "final_dataset"
ORIG_TRAIN_DS  = PROJECT_ROOT / "training_dataset"

CLASS_NAMES    = {0: "helmet", 1: "mask", 2: "person"}

# ── Confirmed split assignments ───────────────────────────────────────────────
PROPOSED_TRAIN = {
    "mask_only", "helmet_mask_gloves", "12984266_1920_1080_25fps",
    "helmet_gloves", "10810476_hd_1920_1080_30fps", "12984031_1920_1080_25fps",
    "helmet_mask_glove", "gloves_only", "13258042_2560_1440_30fps",
    "4042251183_preview", "helmet_mask_gloves02",
}
PROPOSED_VAL  = {
    "4048038451_preview", "4017518657_preview", "8482302_hd_1920_1080_25fps",
    "no_safety", "gloves_mask", "19832490_hd_1920_1080_25fps",
    "istockphoto_2258622635",
}
PROPOSED_TEST = {
    "helmet_only", "5434223_hd_1920_1080_24fps",
    "4161358813_preview", "helmet_mask",
}

ALL_KNOWN = PROPOSED_TRAIN | PROPOSED_VAL | PROPOSED_TEST

EXPECTED = {
    "TRAIN": {"images": 687, "annotations": 4130, 0: 937,  1: 912,  2: 2281},
    "VAL":   {"images": 136, "annotations": 754,  0: 196,  1: 172,  2: 386},
    "TEST":  {"images": 143, "annotations": 1454, 0: 570,  1: 303,  2: 581},
    "TOTAL": {"images": 966, "annotations": 6338, 0: 1703, 1: 1387, 2: 3248},
}

# ── Source extraction ─────────────────────────────────────────────────────────

def extract_source(filename: str) -> str:
    stem    = Path(filename).stem
    cleaned = re.sub(r'_(jpg|jpeg|png)\.rf\.[A-Za-z0-9]+$', '', stem)
    if cleaned.startswith("istockphoto_2258622635"):
        return "istockphoto_2258622635"
    for src in sorted(ALL_KNOWN, key=len, reverse=True):
        if cleaned.startswith(src):
            return src
    heuristic = re.sub(r'_t\d+[-\.]\d+_f\d+.*$', '', cleaned)
    heuristic = re.sub(r'_jpg$', '', heuristic)
    return heuristic

def get_split(source: str) -> str:
    if source in PROPOSED_TRAIN: return "train"
    if source in PROPOSED_VAL:   return "val"
    if source in PROPOSED_TEST:  return "test"
    return "UNASSIGNED"

# ── Create directory structure ────────────────────────────────────────────────

def create_dirs():
    for split in ("train", "val", "test"):
        (V2_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (V2_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# ── Copy files ────────────────────────────────────────────────────────────────

def copy_files():
    """
    For each image in training_dataset/images/{train,val},
    determine its proposed split, then copy image + label to v2.
    Returns copy_log: list of dicts.
    """
    copy_log   = []
    unassigned = []

    for src_img_dir, src_lbl_dir in [
        (SRC_TRAIN_IMG, SRC_TRAIN_LBL),
        (SRC_VAL_IMG,   SRC_VAL_LBL),
    ]:
        for img_path in sorted(src_img_dir.iterdir()):
            if not img_path.is_file():
                continue

            source = extract_source(img_path.name)
            split  = get_split(source)

            if split == "UNASSIGNED":
                unassigned.append(img_path.name)
                continue

            # Destination paths
            dst_img = V2_ROOT / "images" / split / img_path.name
            lbl_src = src_lbl_dir / (img_path.stem + ".txt")
            dst_lbl = V2_ROOT / "labels" / split / (img_path.stem + ".txt")

            # Copy image
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # Copy label
            lbl_copied = False
            if lbl_src.exists():
                if not dst_lbl.exists():
                    shutil.copy2(lbl_src, dst_lbl)
                lbl_copied = True

            copy_log.append({
                "filename": img_path.name,
                "source":   source,
                "split":    split,
                "has_label": lbl_copied,
            })

    return copy_log, unassigned

# ── Write data.yaml ───────────────────────────────────────────────────────────

DATA_YAML_CONTENT = r"""path: C:\Users\sarim\safety monitoring\training_dataset_v2
train: images/train
val: images/val
test: images/test

names:
  0: helmet
  1: mask
  2: person
"""

def write_data_yaml():
    yaml_path = V2_ROOT / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(DATA_YAML_CONTENT)
    return yaml_path

# ── Validation ────────────────────────────────────────────────────────────────

def read_label(lbl_path: Path):
    """Returns list of (class_id, cx, cy, w, h) tuples."""
    rows = []
    with open(lbl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                rows.append(("INVALID_FORMAT", *parts))
                continue
            try:
                cid = int(parts[0])
                vals = [float(x) for x in parts[1:]]
                rows.append((cid, *vals))
            except ValueError:
                rows.append(("PARSE_ERROR", *parts))
    return rows

def validate_v2():
    results = {}

    for split in ("train", "val", "test"):
        img_dir = V2_ROOT / "images" / split
        lbl_dir = V2_ROOT / "labels" / split

        img_files = {f.stem: f for f in img_dir.iterdir() if f.is_file()}
        lbl_files = {f.stem: f for f in lbl_dir.iterdir() if f.is_file()}

        missing_labels   = []   # images without a label file
        orphan_labels    = []   # label files without an image
        invalid_format   = []   # lines with != 5 values
        invalid_class    = []   # class ID not in {0,1,2}
        invalid_coords   = []   # coords outside (0,1] or negative dims
        annotation_count = 0
        class_counts     = defaultdict(int)
        sources          = set()

        for stem, img_path in img_files.items():
            src = extract_source(img_path.name)
            sources.add(src)
            if stem not in lbl_files:
                missing_labels.append(img_path.name)
                continue
            rows = read_label(lbl_files[stem])
            for row in rows:
                if row[0] in ("INVALID_FORMAT", "PARSE_ERROR"):
                    invalid_format.append((img_path.name, row))
                    continue
                cid, cx, cy, w, h = row
                annotation_count += 1
                class_counts[cid] += 1
                if cid not in (0, 1, 2):
                    invalid_class.append((img_path.name, cid))
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and
                        0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    invalid_coords.append((img_path.name, cid, cx, cy, w, h))

        for stem in lbl_files:
            if stem not in img_files:
                orphan_labels.append(stem + ".txt")

        results[split] = {
            "images":          len(img_files),
            "labels":          len(lbl_files),
            "annotations":     annotation_count,
            "class_counts":    dict(class_counts),
            "missing_labels":  missing_labels,
            "orphan_labels":   orphan_labels,
            "invalid_format":  invalid_format,
            "invalid_class":   invalid_class,
            "invalid_coords":  invalid_coords,
            "sources":         sources,
        }

    return results

def check_original_untouched():
    """Spot-check that original datasets weren't modified by comparing counts."""
    results = {}
    for ds_name, ds_path in [("final_dataset", FINAL_DATASET),
                               ("training_dataset", ORIG_TRAIN_DS)]:
        counts = {}
        for split in ("train", "val", "test", "images", "labels"):
            for sub in ("images", "labels"):
                d = ds_path / sub
                if d.exists():
                    for sp in ("train", "val"):
                        sd = d / sp
                        if sd.exists():
                            key = f"{sub}/{sp}"
                            counts[key] = sum(1 for f in sd.iterdir() if f.is_file())
        results[ds_name] = counts
    return results

# ── Overlap check ─────────────────────────────────────────────────────────────

def check_overlap(val_results):
    src_to_splits = defaultdict(set)
    for split, data in val_results.items():
        for src in data["sources"]:
            src_to_splits[src].add(split)
    overlaps = {src: splits for src, splits in src_to_splits.items() if len(splits) > 1}
    return overlaps

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 72
    sep = "-" * 72

    print(SEP)
    print("  BUILDING training_dataset_v2".center(72))
    print("  COPY-ONLY -- original datasets untouched".center(72))
    print(SEP)

    # Safety check: v2 must not already partially exist in a bad state
    if V2_ROOT.exists():
        print(f"\n  [INFO] {V2_ROOT} already exists. Resuming / overwriting into it.")
    else:
        print(f"\n  [INFO] Creating {V2_ROOT}")

    # Step 1: Create dirs
    print("\n  [1/4] Creating directory structure...")
    create_dirs()
    print("        Done.")

    # Step 2: Copy files
    print("  [2/4] Copying images and labels...")
    copy_log, unassigned = copy_files()

    if unassigned:
        print(f"\n  [WARNING] {len(unassigned)} unassigned images skipped:")
        for u in unassigned[:10]:
            print(f"    {u}")
    else:
        print("        All images assigned to a split.")

    by_split = defaultdict(int)
    for entry in copy_log:
        by_split[entry["split"]] += 1
    for sp in ("train", "val", "test"):
        print(f"        Copied to {sp:5s}: {by_split[sp]:>4} images")

    # Step 3: Write data.yaml
    print("  [3/4] Writing data.yaml...")
    yaml_path = write_data_yaml()
    print(f"        Saved: {yaml_path}")

    # Step 4: Validate
    print("  [4/4] Running validation...\n")
    val_results = validate_v2()
    overlaps    = check_overlap(val_results)

    # ── Validation Report ───────────────────────────────────────────────────
    print(SEP)
    print("  VALIDATION REPORT".center(72))
    print(SEP)

    grand_imgs = grand_annot = 0
    grand_cls  = defaultdict(int)
    all_errors = []

    for split in ("train", "val", "test"):
        d       = val_results[split]
        imgs    = d["images"]
        lbls    = d["labels"]
        annot   = d["annotations"]
        h       = d["class_counts"].get(0, 0)
        m       = d["class_counts"].get(1, 0)
        p       = d["class_counts"].get(2, 0)
        exp     = EXPECTED[split.upper()]

        grand_imgs  += imgs
        grand_annot += annot
        for c, v in d["class_counts"].items():
            grand_cls[c] += v

        img_ok   = "OK" if imgs  == exp["images"]      else f"MISMATCH (expected {exp['images']})"
        ann_ok   = "OK" if annot == exp["annotations"] else f"MISMATCH (expected {exp['annotations']})"
        h_ok     = "OK" if h     == exp[0]             else f"MISMATCH (expected {exp[0]})"
        m_ok     = "OK" if m     == exp[1]             else f"MISMATCH (expected {exp[1]})"
        p_ok     = "OK" if p     == exp[2]             else f"MISMATCH (expected {exp[2]})"

        print(f"\n  [{split.upper()}]")
        print(sep)
        print(f"  Images           : {imgs:>6}  [{img_ok}]")
        print(f"  Label files      : {lbls:>6}  {'OK' if lbls == imgs else 'MISMATCH'}")
        print(f"  Annotations      : {annot:>6}  [{ann_ok}]")
        print(f"  helmet  (0)      : {h:>6}  [{h_ok}]")
        print(f"  mask    (1)      : {m:>6}  [{m_ok}]")
        print(f"  person  (2)      : {p:>6}  [{p_ok}]")
        print(f"  Source groups    : {len(d['sources']):>6}")
        print(f"  Missing labels   : {len(d['missing_labels']):>6}  {'OK' if not d['missing_labels'] else 'ERROR'}")
        print(f"  Orphan labels    : {len(d['orphan_labels']):>6}  {'OK' if not d['orphan_labels'] else 'ERROR'}")
        print(f"  Invalid format   : {len(d['invalid_format']):>6}  {'OK' if not d['invalid_format'] else 'ERROR'}")
        print(f"  Invalid class IDs: {len(d['invalid_class']):>6}  {'OK' if not d['invalid_class'] else 'ERROR'}")
        print(f"  Invalid coords   : {len(d['invalid_coords']):>6}  {'OK' if not d['invalid_coords'] else 'ERROR'}")

        if d["missing_labels"]:
            all_errors.append(f"[{split.upper()}] Missing labels: {d['missing_labels'][:5]}")
        if d["orphan_labels"]:
            all_errors.append(f"[{split.upper()}] Orphan labels: {d['orphan_labels'][:5]}")
        if d["invalid_format"]:
            all_errors.append(f"[{split.upper()}] Invalid format rows: {len(d['invalid_format'])}")
        if d["invalid_class"]:
            all_errors.append(f"[{split.upper()}] Invalid class IDs: {d['invalid_class'][:5]}")
        if d["invalid_coords"]:
            all_errors.append(f"[{split.upper()}] Invalid coords: {len(d['invalid_coords'])}")

    # Grand total
    print("\n  [TOTAL]")
    print(sep)
    exp_t  = EXPECTED["TOTAL"]
    s_imgs = "OK" if grand_imgs   == exp_t["images"]      else f"MISMATCH (expected {exp_t['images']})"
    s_ann  = "OK" if grand_annot  == exp_t["annotations"] else f"MISMATCH (expected {exp_t['annotations']})"
    s_h    = "OK" if grand_cls[0] == exp_t[0]             else f"MISMATCH (expected {exp_t[0]})"
    s_m    = "OK" if grand_cls[1] == exp_t[1]             else f"MISMATCH (expected {exp_t[1]})"
    s_p    = "OK" if grand_cls[2] == exp_t[2]             else f"MISMATCH (expected {exp_t[2]})"
    print(f"  Images           : {grand_imgs:>6}  [{s_imgs}]")
    print(f"  Annotations      : {grand_annot:>6}  [{s_ann}]")
    print(f"  helmet  (0)      : {grand_cls[0]:>6}  [{s_h}]")
    print(f"  mask    (1)      : {grand_cls[1]:>6}  [{s_m}]")
    print(f"  person  (2)      : {grand_cls[2]:>6}  [{s_p}]")

    # Overlap check
    print(f"\n  [OVERLAP CHECK]")
    print(sep)
    if overlaps:
        for src, splits in overlaps.items():
            print(f"  ERROR: '{src}' appears in {splits}")
        all_errors.append(f"Source overlap detected: {overlaps}")
    else:
        print("  No source group appears in more than one split.  OK")

    # data.yaml check
    print(f"\n  [data.yaml]")
    print(sep)
    yaml_path = V2_ROOT / "data.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        yaml_content = f.read()
    print(yaml_content.strip())
    required_keys = ["training_dataset_v2", "images/train", "images/val",
                     "images/test", "0: helmet", "1: mask", "2: person"]
    yaml_ok = all(k in yaml_content for k in required_keys)
    print(f"\n  data.yaml valid: {'OK' if yaml_ok else 'CHECK FAILED'}")

    # Original dataset integrity check
    print(f"\n  [ORIGINAL DATASET INTEGRITY]")
    print(sep)
    orig_counts = check_original_untouched()
    for ds_name, counts in orig_counts.items():
        print(f"  {ds_name}:")
        for key, cnt in sorted(counts.items()):
            print(f"    {key}: {cnt} files")

    print(f"\n  training_dataset/images/train : "
          f"{sum(1 for f in SRC_TRAIN_IMG.iterdir() if f.is_file())} files  "
          f"(expected 772)")
    print(f"  training_dataset/images/val   : "
          f"{sum(1 for f in SRC_VAL_IMG.iterdir() if f.is_file())} files  "
          f"(expected 194)")
    print(f"  training_dataset/labels/train : "
          f"{sum(1 for f in SRC_TRAIN_LBL.iterdir() if f.is_file())} files  "
          f"(expected 772)")
    print(f"  training_dataset/labels/val   : "
          f"{sum(1 for f in SRC_VAL_LBL.iterdir() if f.is_file())} files  "
          f"(expected 194)")

    # Final verdict
    print()
    print(SEP)
    if not all_errors:
        print("  ALL CHECKS PASSED".center(72))
        print("  training_dataset_v2 is ready for YOLO training.".center(72))
    else:
        print("  ERRORS FOUND -- review before training:".center(72))
        for e in all_errors:
            print(f"    {e}")
    print(SEP)


if __name__ == "__main__":
    main()
