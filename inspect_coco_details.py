import json
from pathlib import Path
from collections import Counter

coco_path = Path("mask data/train/_annotations.coco.json")
with open(coco_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

cats = {c["id"]: c["name"] for c in coco_data["categories"]}
print("Categories dict:", cats)

cat_counts = Counter(ann["category_id"] for ann in coco_data["annotations"])
for cid, cnt in cat_counts.items():
    print(f"Category ID {cid} ({cats.get(cid)}): {cnt} annotations")

# Check sample annotations per category
for cid in cats:
    sample_anns = [a for a in coco_data["annotations"] if a["category_id"] == cid][:5]
    print(f"\nSample 5 annotations for Category {cid} ({cats.get(cid)}):")
    for sa in sample_anns:
        print(" ", sa)
