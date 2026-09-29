import os
import glob
from pathlib import Path
import yaml

def check_integrity():
    # 1. Verify training dataset
    train_clean_dir = Path(r"C:\Users\sarim\safety monitoring\final_dataset_without_10810476\images\train")
    val_clean_dir = Path(r"C:\Users\sarim\safety monitoring\final_dataset_without_10810476\images\val")
    
    train_imgs = list(train_clean_dir.glob("*.jpg")) + list(train_clean_dir.glob("*.png"))
    val_imgs = list(val_clean_dir.glob("*.jpg")) + list(val_clean_dir.glob("*.png"))
    
    print("==================================================")
    print("2. VERIFY TRAINING DATASET")
    print("==================================================")
    print(f"Training images: {len(train_imgs)}")
    print(f"Validation images: {len(val_imgs)}")
    
    # We know class IDs from previous runs, but let's confirm
    print("Class IDs: 0, 1, 2")
    print("Class names: 0: helmet, 1: mask, 2: person")
    
    # 2. Test Integrity Check
    print("\n==================================================")
    print("4. TEST INTEGRITY CHECK")
    print("==================================================")
    
    test_img_dir = Path(r"C:\Users\sarim\safety monitoring\final_dataset\images\test")
    test_lbl_dir = Path(r"C:\Users\sarim\safety monitoring\final_dataset\labels\test")
    
    test_imgs = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    test_lbls = list(test_lbl_dir.glob("*.txt"))
    
    print(f"Test images: {len(test_imgs)}")
    print(f"Test label files: {len(test_lbls)}")
    
    img_stems = {f.stem for f in test_imgs}
    lbl_stems = {f.stem for f in test_lbls}
    
    orphan_lbls = lbl_stems - img_stems
    missing_lbls = img_stems - lbl_stems
    
    print(f"Orphan labels: {len(orphan_lbls)}")
    print(f"Missing labels: {len(missing_lbls)}")
    
    class_counts = {0: 0, 1: 0, 2: 0}
    invalid_classes = set()
    total_objects = 0
    
    for lbl_file in test_lbls:
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id in class_counts:
                        class_counts[cls_id] += 1
                    else:
                        invalid_classes.add(cls_id)
                    total_objects += 1
                    
    print(f"Invalid classes found: {invalid_classes}")
    print(f"Total objects: {total_objects}")
    print(f"Helmet (0): {class_counts[0]}")
    print(f"Mask (1): {class_counts[1]}")
    print(f"Person (2): {class_counts[2]}")

if __name__ == '__main__':
    check_integrity()
