import os
import sys
import shutil
import hashlib
from pathlib import Path

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def analyze_split(img_dir, lbl_dir):
    img_files = list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.png")) + list(Path(img_dir).glob("*.jpeg"))
    
    total_images = len(img_files)
    person_boxes = 0
    helmet_boxes = 0
    mask_boxes = 0
    images_with_mask = 0
    images_without_mask = 0
    
    for img_p in img_files:
        lbl_p = Path(lbl_dir) / (img_p.stem + ".txt")
        has_mask = False
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        if cls_id == 0:
                            helmet_boxes += 1
                        elif cls_id == 1:
                            mask_boxes += 1
                            has_mask = True
                        elif cls_id == 2:
                            person_boxes += 1
        if has_mask:
            images_with_mask += 1
        else:
            images_without_mask += 1
            
    ratio = (mask_boxes / person_boxes) if person_boxes > 0 else 0.0
    return {
        "images": total_images,
        "person_boxes": person_boxes,
        "helmet_boxes": helmet_boxes,
        "mask_boxes": mask_boxes,
        "mask_person_ratio": ratio,
        "images_with_mask": images_with_mask,
        "images_without_mask": images_without_mask
    }

def main():
    PROJECT_ROOT = Path(r"C:\Users\sarim\safety monitoring")
    
    SRC_V2 = PROJECT_ROOT / "training_dataset_v2"
    SRC_NEW = PROJECT_ROOT / "data_collection" / "roboflow_more_full_export"
    
    DEST_RUN4 = PROJECT_ROOT / "training_dataset_run4"
    
    print("==================================================================")
    print("          BUILDING RUN 4 TRAINING DATASET (training_dataset_run4)")
    print("==================================================================")
    
    # Check source existence
    assert (SRC_V2 / "images" / "train").exists(), f"Source train images missing: {SRC_V2 / 'images' / 'train'}"
    assert (SRC_NEW / "train" / "images").exists(), f"Source new export missing: {SRC_NEW / 'train' / 'images'}"
    
    # 1. Analyze Existing Train
    print("\n1. Analyzing Existing Training Dataset (training_dataset_v2)...")
    stats_existing = analyze_split(SRC_V2 / "images" / "train", SRC_V2 / "labels" / "train")
    
    # Compute MD5 hashes of existing train images
    print("   Computing hashes for exact duplicate checking...")
    existing_hashes = {}
    for img_p in (SRC_V2 / "images" / "train").glob("*.*"):
        if img_p.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            h = compute_md5(img_p)
            existing_hashes[h] = img_p.name
    print(f"   Existing train hashes cataloged: {len(existing_hashes)}")
    
    # 2. Analyze New Exported Data & Validate Labels
    print("\n2. Analyzing New Exported Roboflow Data & Checking Labels...")
    new_img_dir = SRC_NEW / "train" / "images"
    new_lbl_dir = SRC_NEW / "train" / "labels"
    stats_new = analyze_split(new_img_dir, new_lbl_dir)
    
    new_img_files = list(new_img_dir.glob("*.jpg")) + list(new_img_dir.glob("*.png")) + list(new_img_dir.glob("*.jpeg"))
    
    exact_dups = []
    eligible_new = []
    invalid_labels = []
    unexpected_classes = set()
    
    for img_p in new_img_files:
        h = compute_md5(img_p)
        if h in existing_hashes:
            exact_dups.append((img_p.name, existing_hashes[h]))
        else:
            # Check label validity
            lbl_p = new_lbl_dir / (img_p.stem + ".txt")
            valid = True
            if lbl_p.exists():
                with open(lbl_p, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) < 5:
                            invalid_labels.append((lbl_p.name, line_num, "Malformed line (< 5 items)"))
                            valid = False
                            continue
                        cls_id = int(parts[0])
                        if cls_id not in [0, 1, 2]:
                            unexpected_classes.add(cls_id)
                            invalid_labels.append((lbl_p.name, line_num, f"Unexpected class ID {cls_id}"))
                            valid = False
                        coords = [float(x) for x in parts[1:5]]
                        for c in coords:
                            if c < 0.0 or c > 1.0:
                                invalid_labels.append((lbl_p.name, line_num, f"Out of bounds coordinate {c}"))
                                valid = False
            if valid:
                eligible_new.append(img_p)
                
    print(f"   Total new images scanned : {len(new_img_files)}")
    print(f"   Exact duplicates found   : {len(exact_dups)}")
    print(f"   Invalid label issues     : {len(invalid_labels)}")
    print(f"   Unexpected classes found : {list(unexpected_classes)}")
    print(f"   Eligible new images      : {len(eligible_new)}")
    
    # 3. Create training_dataset_run4 Directory Structure
    print("\n3. Creating destination structure at training_dataset_run4...")
    if DEST_RUN4.exists():
        print("   Destination training_dataset_run4 exists, overwriting...")
        shutil.rmtree(DEST_RUN4)
        
    for split in ["train", "val", "test"]:
        (DEST_RUN4 / "images" / split).mkdir(parents=True, exist_ok=True)
        (DEST_RUN4 / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    # 4. Copy Validation and Test splits from training_dataset_v2 (EXACT COPY)
    print("\n4. Copying Validation and Test splits byte-for-byte from training_dataset_v2...")
    for split in ["val", "test"]:
        src_img_split = SRC_V2 / "images" / split
        src_lbl_split = SRC_V2 / "labels" / split
        dest_img_split = DEST_RUN4 / "images" / split
        dest_lbl_split = DEST_RUN4 / "labels" / split
        
        for f in src_img_split.glob("*.*"):
            shutil.copy2(f, dest_img_split / f.name)
        for f in src_lbl_split.glob("*.txt"):
            shutil.copy2(f, dest_lbl_split / f.name)
            
    val_count = len(list((DEST_RUN4 / "images" / "val").glob("*.*")))
    test_count = len(list((DEST_RUN4 / "images" / "test").glob("*.*")))
    print(f"   Val images copied : {val_count}")
    print(f"   Test images copied: {test_count}")
    
    # 5. Populate Train split
    print("\n5. Populating Run 4 Train split (Existing Train + Eligible New)...")
    dest_train_img = DEST_RUN4 / "images" / "train"
    dest_train_lbl = DEST_RUN4 / "labels" / "train"
    
    # Copy existing train
    for f in (SRC_V2 / "images" / "train").glob("*.*"):
        shutil.copy2(f, dest_train_img / f.name)
    for f in (SRC_V2 / "labels" / "train").glob("*.txt"):
        shutil.copy2(f, dest_train_lbl / f.name)
        
    # Copy eligible new train
    for img_p in eligible_new:
        shutil.copy2(img_p, dest_train_img / img_p.name)
        lbl_p = new_lbl_dir / (img_p.stem + ".txt")
        if lbl_p.exists():
            shutil.copy2(lbl_p, dest_train_lbl / (img_p.stem + ".txt"))
            
    # 6. Analyze Final Run 4 Train Split
    stats_run4_train = analyze_split(dest_train_img, dest_train_lbl)
    
    # 7. Write data.yaml
    data_yaml_content = f"""path: {DEST_RUN4.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: helmet
  1: mask
  2: person
"""
    with open(DEST_RUN4 / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)
    print(f"   Created data.yaml at {DEST_RUN4 / 'data.yaml'}")
    
    # 8. Print Dataset Statistics Comparison Table
    print("\n" + "=" * 80)
    print("                     RUN 4 DATASET STATISTICS COMPARISON                     ")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Existing Train':<15} | {'New Data':<15} | {'Run 4 Train':<15}")
    print("-" * 80)
    print(f"{'Images':<25} | {stats_existing['images']:<15} | {stats_new['images']:<15} | {stats_run4_train['images']:<15}")
    print(f"{'Person boxes':<25} | {stats_existing['person_boxes']:<15} | {stats_new['person_boxes']:<15} | {stats_run4_train['person_boxes']:<15}")
    print(f"{'Helmet boxes':<25} | {stats_existing['helmet_boxes']:<15} | {stats_new['helmet_boxes']:<15} | {stats_run4_train['helmet_boxes']:<15}")
    print(f"{'Mask boxes':<25} | {stats_existing['mask_boxes']:<15} | {stats_new['mask_boxes']:<15} | {stats_run4_train['mask_boxes']:<15}")
    print(f"{'Mask/person ratio':<25} | {stats_existing['mask_person_ratio']:<15.4f} | {stats_new['mask_person_ratio']:<15.4f} | {stats_run4_train['mask_person_ratio']:<15.4f}")
    print(f"{'Images with mask':<25} | {stats_existing['images_with_mask']:<15} | {stats_new['images_with_mask']:<15} | {stats_run4_train['images_with_mask']:<15}")
    print(f"{'Images without mask':<25} | {stats_existing['images_without_mask']:<15} | {stats_new['images_without_mask']:<15} | {stats_run4_train['images_without_mask']:<15}")
    print("=" * 80)
    
    # Save statistics report JSON
    import json
    stats_out = {
        "existing_train": stats_existing,
        "new_data": stats_new,
        "run4_train": stats_run4_train,
        "exact_duplicates": len(exact_dups),
        "invalid_labels": len(invalid_labels)
    }
    with open(PROJECT_ROOT / "reports" / "run4_dataset_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"Saved dataset statistics to {PROJECT_ROOT / 'reports' / 'run4_dataset_stats.json'}")

if __name__ == '__main__':
    main()
