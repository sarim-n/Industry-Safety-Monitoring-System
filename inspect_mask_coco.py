import json
from pathlib import Path

coco_path = Path("mask data/train/_annotations.coco.json")
assert coco_path.exists(), f"File not found: {coco_path}"

with open(coco_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

print("Keys in COCO JSON:", list(coco_data.keys()))
print("Categories:", coco_data.get("categories"))
print("Number of images in COCO JSON:", len(coco_data.get("images", [])))
print("Number of annotations in COCO JSON:", len(coco_data.get("annotations", [])))

if coco_data.get("annotations"):
    print("Sample annotation:", coco_data["annotations"][0])
