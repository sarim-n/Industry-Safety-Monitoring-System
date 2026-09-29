import os
import glob

dataset_dir = r"C:\Users\sarim\safety monitoring\final_dataset_without_10810476"

splits = ['train', 'val', 'test']

for split in splits:
    images_dir = os.path.join(dataset_dir, 'images', split)
    labels_dir = os.path.join(dataset_dir, 'labels', split)
    
    if not os.path.exists(images_dir):
        print(f"Skipping {split} - images dir does not exist")
        continue
        
    images = glob.glob(os.path.join(images_dir, '*.*'))
    labels = glob.glob(os.path.join(labels_dir, '*.txt'))
    
    image_stems = set([os.path.splitext(os.path.basename(img))[0] for img in images])
    label_stems = set([os.path.splitext(os.path.basename(lbl))[0] for lbl in labels])
    
    print(f"[{split}] Images: {len(images)}")
    print(f"[{split}] Labels: {len(labels)}")
    
    orphans = label_stems - image_stems
    missing = image_stems - label_stems
    
    if orphans:
        print(f"[{split}] Orphan labels: {len(orphans)}")
    if missing:
        print(f"[{split}] Missing labels: {len(missing)}")
        
    # Check for 10810476
    bad_files = [img for img in images if '10810476' in img]
    if bad_files:
        print(f"[{split}] FOUND BAD FILES: {len(bad_files)}")
