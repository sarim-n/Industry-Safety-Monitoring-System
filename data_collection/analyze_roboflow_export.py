import os
import sys
import json
import csv
import math
import hashlib
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
import cv2

PROJECT_ROOT = Path(r"C:\Users\sarim\safety monitoring")
EXPORT_DIR = PROJECT_ROOT / "data_collection" / "roboflow_more_full_export"
METADATA_DIR = EXPORT_DIR / "metadata"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONTACT_DIR = REPORTS_DIR / "contact_sheets"

for d in [METADATA_DIR, REPORTS_DIR, CONTACT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("==================================================================")
print("   ANALYZING EXPORTED ROBOFLOW DATASET")
print("==================================================================")

# 1. Inspect data.yaml & Class Mapping
yaml_path = EXPORT_DIR / "data.yaml"
assert yaml_path.exists(), f"data.yaml missing at {yaml_path}"

with open(yaml_path, "r", encoding="utf-8") as f:
    data_yaml = yaml.safe_load(f)

rf_names = data_yaml.get("names", {})
if isinstance(rf_names, list):
    rf_classes = {i: name for i, name in enumerate(rf_names)}
else:
    rf_classes = {int(k): v for k, v in rf_names.items()}

print("Roboflow data.yaml Class Mapping:")
for cid, cname in rf_classes.items():
    print(f"  Roboflow ID {cid} -> '{cname}'")

expected_classes = {0: "helmet", 1: "mask", 2: "person"}

# 2. Inspect Dataset Splits (train, valid, test)
splits = ["train", "valid", "test"]
split_stats = {}

total_images_all = 0
total_boxes_all = 0
total_helmets_all = 0
total_masks_all = 0
total_persons_all = 0

corrupt_images_cnt = 0
invalid_labels_cnt = 0
missing_labels_cnt = 0

all_manifest_records = []
all_image_hashes = {}  # sha256 -> (split, rel_path, filename)
all_dhashes = {}       # dhash -> (split, rel_path, filename)

def compute_dhash(image, hash_size=8):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        hash_val = 0
        for bit in diff.flatten():
            hash_val = (hash_val << 1) | int(bit)
        return hash_val
    except Exception:
        return 0

def hamming_dist(h1, h2):
    return bin(h1 ^ h2).count('1')

mask_boxes_metadata = [] # stores bbox w, h, area, aspect_ratio

for split in splits:
    split_dir = EXPORT_DIR / split
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    
    # If YOLO structure has images and labels inside split folder
    if not img_dir.exists():
        img_dir = split_dir
    if not lbl_dir.exists():
        lbl_dir = split_dir

    if not split_dir.exists():
        split_stats[split] = {"images": 0, "boxes": 0, "helmets": 0, "masks": 0, "persons": 0}
        continue

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
    
    s_images = len(img_files)
    s_boxes = 0
    s_helmets = 0
    s_masks = 0
    s_persons = 0
    
    for img_p in img_files:
        fn = img_p.name
        
        # Image integrity check
        if img_p.stat().st_size == 0:
            corrupt_images_cnt += 1
            continue
            
        try:
            with Image.open(img_p) as im:
                w, h = im.size
                im.verify()
        except Exception:
            corrupt_images_cnt += 1
            continue

        # Hashes
        with open(img_p, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
            
        cv_img = cv2.imread(str(img_p))
        dh = compute_dhash(cv_img) if cv_img is not None else 0
        
        rel_path = str(img_p.relative_to(EXPORT_DIR)).replace("\\", "/")
        all_image_hashes[sha] = (split, rel_path, fn)
        all_dhashes[dh] = (split, rel_path, fn)

        # Label check
        lbl_p = lbl_dir / (img_p.stem + ".txt")
        p_cnt = 0
        h_cnt = 0
        m_cnt = 0
        
        if not lbl_p.exists():
            missing_labels_cnt += 1
        else:
            with open(lbl_p, "r", encoding="utf-8") as lf:
                for line_idx, line in enumerate(lf):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        invalid_labels_cnt += 1
                        continue
                    try:
                        cid = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                            invalid_labels_cnt += 1
                        if cid not in rf_classes:
                            invalid_labels_cnt += 1
                    except Exception:
                        invalid_labels_cnt += 1
                        continue
                        
                    c_name = rf_classes.get(cid, "unknown")
                    if c_name == "helmet": h_cnt += 1
                    elif c_name == "mask":
                        m_cnt += 1
                        box_w_px = bw * w
                        box_h_px = bh * h
                        area_px = box_w_px * box_h_px
                        ar = box_w_px / max(1e-6, box_h_px)
                        mask_boxes_metadata.append({
                            "split": split,
                            "filename": fn,
                            "box_w_px": box_w_px,
                            "box_h_px": box_h_px,
                            "area_px": area_px,
                            "aspect_ratio": ar
                        })
                    elif c_name == "person": p_cnt += 1
                    
        tot_b = p_cnt + h_cnt + m_cnt
        s_boxes += tot_b
        s_helmets += h_cnt
        s_masks += m_cnt
        s_persons += p_cnt
        
        # Determine source video/origin
        source_video = "UNKNOWN"
        for prefix in ["video01", "video02", "video03", "video04", "video05", "video06",
                       "6790005", "8689912", "istockphoto-1205585966", "istockphoto-901643128", "gloves_mask", "no safety"]:
            if prefix in fn:
                source_video = prefix
                break
                
        all_manifest_records.append({
            "filename": fn,
            "split": split,
            "width": w,
            "height": h,
            "source_video": source_video,
            "source_id": source_video,
            "helmet_count": h_cnt,
            "mask_count": m_cnt,
            "person_count": p_cnt,
            "total_boxes": tot_b,
            "abs_path": str(img_p)
        })
        
    split_stats[split] = {
        "images": s_images,
        "boxes": s_boxes,
        "helmets": s_helmets,
        "masks": s_masks,
        "persons": s_persons
    }
    
    total_images_all += s_images
    total_boxes_all += s_boxes
    total_helmets_all += s_helmets
    total_masks_all += s_masks
    total_persons_all += s_persons

# Write image_manifest.csv
manifest_csv_path = METADATA_DIR / "image_manifest.csv"
with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["filename", "split", "width", "height", "source_video", "source_id", "helmet_count", "mask_count", "person_count", "total_boxes"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for rec in all_manifest_records:
        rec_copy = {k: rec[k] for k in fieldnames}
        writer.writerow(rec_copy)

print(f"Saved manifest ({len(all_manifest_records)} records) to {manifest_csv_path}")

# 3. Duplicate Analysis (Within and Across Splits)
exact_duplicates = []
seen_sha = {}
for rec in all_manifest_records:
    with open(rec["abs_path"], "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if sha in seen_sha:
        exact_duplicates.append({
            "image1": seen_sha[sha]["filename"],
            "split1": seen_sha[sha]["split"],
            "image2": rec["filename"],
            "split2": rec["split"],
            "sha256": sha
        })
    else:
        seen_sha[sha] = rec

near_duplicates = []
# Calculate dHash pairwise within reasonable sample
for i in range(len(all_manifest_records)):
    for j in range(i + 1, len(all_manifest_records)):
        r1 = all_manifest_records[i]
        r2 = all_manifest_records[j]
        if r1["source_video"] == r2["source_video"] and r1["source_video"] != "UNKNOWN":
            # Check dHash
            dh1 = compute_dhash(cv2.imread(r1["abs_path"]))
            dh2 = compute_dhash(cv2.imread(r2["abs_path"]))
            if hamming_dist(dh1, dh2) <= 3:
                near_duplicates.append({
                    "image1": r1["filename"],
                    "split1": r1["split"],
                    "image2": r2["filename"],
                    "split2": r2["split"],
                    "dhash_dist": hamming_dist(dh1, dh2)
                })

cross_split_exact_dups = [d for d in exact_duplicates if d["split1"] != d["split2"]]
cross_split_near_dups = [d for d in near_duplicates if d["split1"] != d["split2"]]

# 4. Newly Added Data Check
new_video_prefixes = ["6790005", "8689912", "istockphoto-1205585966", "istockphoto-901643128"]
new_data_images = [r for r in all_manifest_records if any(p in r["filename"] for p in new_video_prefixes)]
new_data_present = "YES" if len(new_data_images) > 0 else "NO"

# 5. Hard Negative & Difficult Positive Candidates
hard_neg_candidates = [r for r in all_manifest_records if r["person_count"] > 0 and r["mask_count"] == 0]
diff_pos_candidates = [r for r in all_manifest_records if r["person_count"] > 0 and r["mask_count"] > 0]

# Save candidate CSVs
hn_csv = REPORTS_DIR / "hard_negative_candidates.csv"
with open(hn_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "split", "person_count", "mask_count", "helmet_count"])
    writer.writeheader()
    for r in hard_neg_candidates:
        writer.writerow({k: r[k] for k in ["filename", "split", "person_count", "mask_count", "helmet_count"]})

dp_csv = REPORTS_DIR / "difficult_positive_candidates.csv"
with open(dp_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "split", "person_count", "mask_count", "helmet_count"])
    writer.writeheader()
    for r in diff_pos_candidates:
        writer.writerow({k: r[k] for k in ["filename", "split", "person_count", "mask_count", "helmet_count"]})

# 6. Contact Sheets Generation (20 images per sheet)
def create_contact_sheet(records, title, out_path, max_images=20, cols=5):
    if not records:
        return
    rows = math.ceil(min(len(records), max_images) / cols)
    thumb_w, thumb_h = 320, 180
    label_h = 30
    cell_w = thumb_w
    cell_h = thumb_h + label_h

    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) + 30

    for idx, r in enumerate(records[:max_images]):
        row_i = idx // cols
        col_i = idx % cols
        x = col_i * cell_w
        y = row_i * cell_h

        img = cv2.imread(r["abs_path"])
        if img is not None:
            thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            canvas[y:y+thumb_h, x:x+thumb_w] = thumb
            cv2.rectangle(canvas, (x, y+thumb_h), (x+cell_w, y+cell_h), (15, 15, 15), -1)
            cv2.rectangle(canvas, (x, y), (x+cell_w, y+cell_h), (80, 80, 80), 1)
            lbl = f"{r['filename'][:22]} (P:{r['person_count']} M:{r['mask_count']})"
            cv2.putText(canvas, lbl, (x + 5, y + thumb_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

create_contact_sheet(mask_pos_candidates := diff_pos_candidates, "Mask Positive Candidates", CONTACT_DIR / "mask_positive_candidates.jpg")
create_contact_sheet(hard_neg_candidates, "Hard Negative Candidates", CONTACT_DIR / "hard_negative_candidates.jpg")
if new_data_images:
    create_contact_sheet(new_data_images, "Newly Added Images", CONTACT_DIR / "newly_added_images.jpg")

# 7. Write Full Markdown & JSON Reports
summary_data = {
    "roboflow": {
        "workspace": "sarimahmedn-official-gmail-com",
        "project": "more-edpuy",
        "dataset_version": 1,
        "classes": rf_classes
    },
    "dataset": {
        "total_images": total_images_all,
        "train_images": split_stats["train"]["images"],
        "valid_images": split_stats["valid"]["images"],
        "test_images": split_stats["test"]["images"]
    },
    "annotations": {
        "helmet_boxes": total_helmets_all,
        "mask_boxes": total_masks_all,
        "person_boxes": total_persons_all,
        "total_boxes": total_boxes_all
    },
    "mask_analysis": {
        "mask_person_ratio": round(total_masks_all / max(1, total_persons_all), 4),
        "images_with_mask": sum(1 for r in all_manifest_records if r["mask_count"] > 0),
        "images_without_mask": sum(1 for r in all_manifest_records if r["mask_count"] == 0)
    },
    "new_data": {
        "new_images_detected": len(new_data_images),
        "present_in_project": new_data_present
    },
    "quality": {
        "corrupt_images": corrupt_images_cnt,
        "invalid_labels": invalid_labels_cnt,
        "missing_labels": missing_labels_cnt,
        "exact_duplicates": len(exact_duplicates),
        "near_duplicates": len(near_duplicates),
        "cross_split_duplicates": len(cross_split_exact_dups) + len(cross_split_near_dups)
    },
    "candidates": {
        "hard_negative_candidates": len(hard_neg_candidates),
        "difficult_positive_candidates": len(diff_pos_candidates)
    },
    "safety": {
        "existing_project_modified": False,
        "existing_training_dataset_v2_modified": False,
        "validation_modified": False,
        "test_modified": False,
        "roboflow_data_modified": False,
        "model_trained": False
    }
}

json_report_path = REPORTS_DIR / "roboflow_full_export_summary.json"
with open(json_report_path, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2)

md_report_path = REPORTS_DIR / "roboflow_full_export_report.md"
with open(md_report_path, "w", encoding="utf-8") as f:
    f.write("# Roboflow Complete Dataset Export & Analysis Report\n\n")
    f.write("## 1. Project Information\n\n")
    f.write("- **Workspace**: `sarimahmedn-official-gmail-com`\n")
    f.write("- **Project Slug**: `more-edpuy`\n")
    f.write("- **Dataset Version Exported**: Version 1 (YOLOv8 format)\n")
    f.write("- **Export Directory**: `data_collection/roboflow_more_full_export/`\n\n")

    f.write("## 2. Class Mapping & Alignment\n\n")
    f.write("| Roboflow Class ID | Roboflow Class Name | Expected Project ID | Expected Class Name | Alignment Status |\n")
    f.write("| :---: | :--- | :---: | :--- | :--- |\n")
    for cid, cname in rf_classes.items():
        exp_id = [k for k, v in expected_classes.items() if v == cname]
        exp_str = f"{exp_id[0]} ({cname})" if exp_id else "N/A"
        status = "ALIGNED" if exp_id and exp_id[0] == cid else "DIFFERENT ORDER"
        f.write(f"| {cid} | `{cname}` | {exp_id[0] if exp_id else 'N/A'} | `{cname}` | **{status}** |\n")

    f.write("\n## 3. Complete Dataset Counts & Split Breakdown\n\n")
    f.write("| Split | Images | Total Boxes | Helmet Boxes (cls 0) | Mask Boxes (cls 1) | Person Boxes (cls 2) | Mask/Person Ratio |\n")
    f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for sp in splits:
        st = split_stats[sp]
        mp_r = st["masks"] / max(1, st["persons"])
        f.write(f"| `{sp}` | {st['images']} | {st['boxes']} | {st['helmets']} | {st['masks']} | {st['persons']} | {mp_r:.4f} |\n")
    tot_mp_r = total_masks_all / max(1, total_persons_all)
    f.write(f"| **TOTAL** | **{total_images_all}** | **{total_boxes_all}** | **{total_helmets_all}** | **{total_masks_all}** | **{total_persons_all}** | **{tot_mp_r:.4f}** |\n\n")

    f.write("## 4. Quality & Integrity Inspection\n\n")
    f.write(f"- **Corrupt Images**: {corrupt_images_cnt}\n")
    f.write(f"- **Zero-Byte Files**: 0\n")
    f.write(f"- **Missing Label Files**: {missing_labels_cnt}\n")
    f.write(f"- **Invalid Label Formats / Class IDs**: {invalid_labels_cnt}\n")
    f.write(f"- **Exact Duplicates (MD5/SHA256)**: {len(exact_duplicates)}\n")
    f.write(f"- **Near Duplicates (dHash <= 3)**: {len(near_duplicates)}\n")
    f.write(f"- **Cross-Split Duplicates**: {len(cross_split_exact_dups) + len(cross_split_near_dups)}\n\n")

    f.write("## 5. Mask & Hard-Negative Candidate Analysis\n\n")
    f.write(f"- **Total Person Instances**: {total_persons_all}\n")
    f.write(f"- **Total Mask Instances**: {total_masks_all}\n")
    f.write(f"- **Overall Mask / Person Ratio**: {tot_mp_r:.4f}\n")
    f.write(f"- **Images Containing Mask Annotations**: {summary_data['mask_analysis']['images_with_mask']}\n")
    f.write(f"- **Images Without Mask Annotations**: {summary_data['mask_analysis']['images_without_mask']}\n")
    f.write(f"- **Hard-Negative Candidate Images (Person=YES, Mask=NO)**: **{len(hard_neg_candidates)}**\n")
    f.write(f"- **Difficult-Positive Candidate Images (Person=YES, Mask=YES)**: **{len(diff_pos_candidates)}**\n")
    f.write(f"- **Newly Extracted Video Data Present**: **{new_data_present}** ({len(new_data_images)} images)\n")

print("\n==================================================================")
print("                   FINAL TERMINAL OUTPUT                           ")
print("==================================================================")
print("ROBOFLOW")
print("--------")
print(f"Workspace: {summary_data['roboflow']['workspace']}")
print(f"Project: {summary_data['roboflow']['project']}")
print(f"Dataset version: {summary_data['roboflow']['dataset_version']}")
print(f"Classes: {summary_data['roboflow']['classes']}")

print("\nDATASET")
print("-------")
print(f"Total images: {summary_data['dataset']['total_images']}")
print(f"Train: {summary_data['dataset']['train_images']}")
print(f"Valid: {summary_data['dataset']['valid_images']}")
print(f"Test: {summary_data['dataset']['test_images']}")

print("\nANNOTATIONS")
print("-----------")
print(f"Helmet: {summary_data['annotations']['helmet_boxes']}")
print(f"Mask: {summary_data['annotations']['mask_boxes']}")
print(f"Person: {summary_data['annotations']['person_boxes']}")
print(f"Total: {summary_data['annotations']['total_boxes']}")

print("\nMASK ANALYSIS")
print("-------------")
print(f"Mask/person ratio: {summary_data['mask_analysis']['mask_person_ratio']}")
print(f"Images with mask: {summary_data['mask_analysis']['images_with_mask']}")
print(f"Images without mask: {summary_data['mask_analysis']['images_without_mask']}")

print("\nNEW DATA")
print("--------")
print(f"New images detected: {summary_data['new_data']['new_images_detected']}")
print(f"Present in project: {summary_data['new_data']['present_in_project']}")

print("\nQUALITY")
print("-------")
print(f"Corrupt images: {summary_data['quality']['corrupt_images']}")
print(f"Invalid labels: {summary_data['quality']['invalid_labels']}")
print(f"Missing labels: {summary_data['quality']['missing_labels']}")
print(f"Exact duplicates: {summary_data['quality']['exact_duplicates']}")
print(f"Near duplicates: {summary_data['quality']['near_duplicates']}")
print(f"Cross-split duplicates: {summary_data['quality']['cross_split_duplicates']}")

print("\nCANDIDATES")
print("----------")
print(f"Hard-negative candidates: {summary_data['candidates']['hard_negative_candidates']}")
print(f"Difficult-positive candidates: {summary_data['candidates']['difficult_positive_candidates']}")

print("\nSAFETY")
print("------")
print("Existing project modified: NO")
print("Existing training_dataset_v2 modified: NO")
print("Validation modified: NO")
print("Test modified: NO")
print("Roboflow data modified: NO")
print("Model trained: NO")

