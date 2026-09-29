import json
from pathlib import Path

coco_path = Path("mask data/train/_annotations.coco.json")
with open(coco_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

print("Categories in file:")
for c in coco_data.get("categories", []):
    print(" ", c)

# Find video05 image IDs
vid05_img_ids = [img["id"] for img in coco_data["images"] if "video05" in img["file_name"]]
print(f"\nNumber of Video 05 images: {len(vid05_img_ids)}")

vid05_anns = [a for a in coco_data["annotations"] if a["image_id"] in vid05_img_ids]
print(f"Number of Video 05 annotations: {len(vid05_anns)}")

for a in vid05_anns:
    print(" ", a)
