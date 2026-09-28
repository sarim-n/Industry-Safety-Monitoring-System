import os
import shutil
import hashlib
import csv
import json
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(r"C:\Users\sarim\safety monitoring")
V2_DIR = PROJECT_ROOT / "training_dataset_v2"
MASK_DATA_DIR = PROJECT_ROOT / "mask data" / "train"

V3_DIR = PROJECT_ROOT / "training_dataset_v3_mask_hard_negative"
V3_IMAGES_TRAIN = V3_DIR / "images" / "train"
V3_IMAGES_VAL = V3_DIR / "images" / "val"
V3_IMAGES_TEST = V3_DIR / "images" / "test"

V3_LABELS_TRAIN = V3_DIR / "labels" / "train"
V3_LABELS_VAL = V3_DIR / "labels" / "val"
V3_LABELS_TEST = V3_DIR / "labels" / "test"

for d in [V3_IMAGES_TRAIN, V3_IMAGES_VAL, V3_IMAGES_TEST, V3_LABELS_TRAIN, V3_LABELS_VAL, V3_LABELS_TEST]:
    d.mkdir(parents=True, exist_ok=True)

print("==================================================================")
print("   PHASE 3 & 4 — BUILDING & ANALYZING RUN 3 TRAINING DATASET")
print("==================================================================")

def get_file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# 1. Copy VAL and TEST splits completely untouched from V2 to V3
print("\n[STEP 1] Copying VAL and TEST splits untouched...")

val_imgs = list((V2_DIR / "images" / "val").glob("*.jpg")) + list((V2_DIR / "images" / "val").glob("*.png"))
test_imgs = list((V2_DIR / "images" / "test").glob("*.jpg")) + list((V2_DIR / "images" / "test").glob("*.png"))

for img_p in val_imgs:
    shutil.copy2(str(img_p), str(V3_IMAGES_VAL / img_p.name))
    lbl_p = V2_DIR / "labels" / "val" / (img_p.stem + ".txt")
    if lbl_p.exists():
        shutil.copy2(str(lbl_p), str(V3_LABELS_VAL / lbl_p.name))

for img_p in test_imgs:
    shutil.copy2(str(img_p), str(V3_IMAGES_TEST / img_p.name))
    lbl_p = V2_DIR / "labels" / "test" / (img_p.stem + ".txt")
    if lbl_p.exists():
        shutil.copy2(str(lbl_p), str(V3_LABELS_TEST / lbl_p.name))

print(f"  VAL  : {len(val_imgs)} images copied.")
print(f"  TEST : {len(test_imgs)} images copied.")

# 2. Duplicate Detection between V2 train and mask data
print("\n[STEP 2] Duplicate Detection between V2 train and mask data...")

v2_train_imgs = list((V2_DIR / "images" / "train").glob("*.jpg")) + list((V2_DIR / "images" / "train").glob("*.png"))
v2_hashes = {get_file_md5(p): p for p in v2_train_imgs}

mask_imgs = list(MASK_DATA_DIR.glob("*.jpg")) + list(MASK_DATA_DIR.glob("*.png"))

duplicate_images = []
unique_mask_imgs = []

for m_p in mask_imgs:
    h = get_file_md5(m_p)
    if h in v2_hashes:
        duplicate_images.append({
            "mask_file": m_p.name,
            "existing_file": v2_hashes[h].name,
            "md5": h
        })
    else:
        unique_mask_imgs.append(m_p)

print(f"  Existing V2 Train Images : {len(v2_train_imgs)}")
print(f"  New Mask Data Images     : {len(mask_imgs)}")
print(f"  Duplicates Detected      : {len(duplicate_images)}")
print(f"  Unique Mask Data Images  : {len(unique_mask_imgs)}")

# 3. Populate TRAIN Split for V3
print("\n[STEP 3] Populating V3 TRAIN Split...")

# Copy existing V2 train
for img_p in v2_train_imgs:
    shutil.copy2(str(img_p), str(V3_IMAGES_TRAIN / img_p.name))
    lbl_p = V2_DIR / "labels" / "train" / (img_p.stem + ".txt")
    if lbl_p.exists():
        shutil.copy2(str(lbl_p), str(V3_LABELS_TRAIN / lbl_p.name))

# Copy unique new mask data train
for img_p in unique_mask_imgs:
    dst_img_name = "mask_hn_" + img_p.name
    dst_img_p = V3_IMAGES_TRAIN / dst_img_name
    shutil.copy2(str(img_p), str(dst_img_p))
    
    src_lbl_p = MASK_DATA_DIR / "labels" / (img_p.stem + ".txt")
    dst_lbl_p = V3_LABELS_TRAIN / (Path(dst_img_name).stem + ".txt")
    if src_lbl_p.exists():
        shutil.copy2(str(src_lbl_p), str(dst_lbl_p))

v3_train_imgs = list(V3_IMAGES_TRAIN.glob("*.jpg")) + list(V3_IMAGES_TRAIN.glob("*.png"))
print(f"  Total V3 TRAIN Images : {len(v3_train_imgs)}")

# 4. Write data.yaml for V3
data_yaml_path = V3_DIR / "data.yaml"
with open(data_yaml_path, "w", encoding="utf-8") as f:
    f.write(f"path: {str(V3_DIR)}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n\n")
    f.write("names:\n")
    f.write("  0: helmet\n")
    f.write("  1: mask\n")
    f.write("  2: person\n")

print(f"  Saved {data_yaml_path}")

# 5. Compute Dataset Statistics Before Training (Phase 4)
print("\n[STEP 5] Computing Statistics (Phase 4)...")

def count_split_boxes(images_dir, labels_dir):
    imgs = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    counts = {0: 0, 1: 0, 2: 0} # 0=helmet, 1=mask, 2=person
    tot_boxes = 0
    for img_p in imgs:
        lbl_p = labels_dir / (img_p.stem + ".txt")
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c = int(parts[0])
                        if c in counts:
                            counts[c] += 1
                            tot_boxes += 1
    return len(imgs), tot_boxes, counts[2], counts[0], counts[1]

v2_imgs, v2_boxes, v2_p, v2_h, v2_m = count_split_boxes(V2_DIR / "images" / "train", V2_DIR / "labels" / "train")
mask_n, mask_b, mask_p, mask_h, mask_m = count_split_boxes(MASK_DATA_DIR, MASK_DATA_DIR / "labels")
v3_imgs, v3_boxes, v3_p, v3_h, v3_m = count_split_boxes(V3_IMAGES_TRAIN, V3_LABELS_TRAIN)

v2_m_p_ratio = v2_m / max(1, v2_p)
mask_m_p_ratio = mask_m / max(1, mask_p)
v3_m_p_ratio = v3_m / max(1, v3_p)

# Save Comparison CSV & Report
reports_dir = PROJECT_ROOT / "reports" / "mask_data_validation_v1"
reports_dir.mkdir(parents=True, exist_ok=True)

csv_out = reports_dir / "dataset_statistics_before_training.csv"
with open(csv_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset_split", "images", "total_boxes", "person_boxes", "helmet_boxes", "mask_boxes", "mask_person_ratio"])
    writer.writerow(["Existing TRAIN (v2)", v2_imgs, v2_boxes, v2_p, v2_h, v2_m, f"{v2_m_p_ratio:.4f}"])
    writer.writerow(["New Mask Data TRAIN", mask_n, mask_b, mask_p, mask_h, mask_m, f"{mask_m_p_ratio:.4f}"])
    writer.writerow(["Combined Run 3 TRAIN (v3)", v3_imgs, v3_boxes, v3_p, v3_h, v3_m, f"{v3_m_p_ratio:.4f}"])

print(f"\nSaved dataset statistics comparison to {csv_out}")
print("==================================================================")
print(f"  Existing V2 TRAIN : {v2_imgs} images | {v2_p} persons | {v2_h} helmets | {v2_m} masks (Ratio: {v2_m_p_ratio:.4f})")
print(f"  New Mask Data     : {mask_n} images | {mask_p} persons | {mask_h} helmets | {mask_m} masks (Ratio: {mask_m_p_ratio:.4f})")
print(f"  Combined V3 TRAIN : {v3_imgs} images | {v3_p} persons | {v3_h} helmets | {v3_m} masks (Ratio: {v3_m_p_ratio:.4f})")
print("==================================================================")
