import os
import json
from pathlib import Path

mask_dir = Path("mask data")
all_files = list(mask_dir.rglob("*"))

print(f"Total files/folders in 'mask data': {len(all_files)}")
json_files = list(mask_dir.rglob("*.json"))
txt_files = list(mask_dir.rglob("*.txt"))
jpg_files = list(mask_dir.rglob("*.jpg"))

print(f"JSON files: {len(json_files)} -> {[f.name for f in json_files]}")
print(f"TXT files: {len(txt_files)} -> {[f.name for f in txt_files]}")
print(f"JPG files: {len(jpg_files)}")

# Check image sizes
from PIL import Image
if jpg_files:
    img0 = Image.open(jpg_files[0])
    print("Sample image size:", img0.size)
