import os
import glob

source_dirs = [
    r"C:\Users\sarim\safety monitoring\roboflow",
    r"C:\Users\sarim\safety monitoring\roboflow_batch2"
]

all_images = []
for src in source_dirs:
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        all_images.extend(glob.glob(os.path.join(src, "**", ext), recursive=True))

missing_labels = []
duplicate_names = []
seen = set()

for img in all_images:
    stem = os.path.splitext(os.path.basename(img))[0]
    if stem in seen:
        duplicate_names.append(stem)
        continue
    seen.add(stem)

    txt1 = os.path.splitext(img)[0] + ".txt"
    txt2 = img.replace("images", "labels").rsplit(".", 1)[0] + ".txt"

    if not os.path.exists(txt1) and not os.path.exists(txt2):
        missing_labels.append(img)

print(f"Total raw images scanned: {len(all_images)}")
print(f"Duplicate image filenames skipped: {len(duplicate_names)}")
print(f"Images missing .txt label files: {len(missing_labels)}")