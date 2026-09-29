import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import glob
from ultralytics import YOLO

def main():
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
            raise RuntimeError("Orphan labels found!")
        if missing:
            print(f"[{split}] Missing labels: {len(missing)}")
            raise RuntimeError("Missing labels found!")
            
        # Check for 10810476 in basename
        bad_files = [img for img in images if '10810476' in os.path.basename(img)]
        if bad_files:
            print(f"[{split}] FOUND BAD FILES: {len(bad_files)}")
            raise RuntimeError("Bad files found!")

    print("VERIFICATION PASSED. STARTING TRAINING...")

    # Train YOLOv8s @ 800
    model = YOLO('yolov8s.pt')
    results = model.train(
        data=os.path.join(dataset_dir, 'data.yaml'),
        imgsz=800,
        epochs=50,
        patience=15,
        batch=4,
        optimizer='auto',
        seed=42,
        amp=True,
        workers=2,
        project='runs/detect',
        name='safety_v1-4_run2b_clean10810476_yolov8s_800',
        exist_ok=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0
    )
    print("TRAINING COMPLETE.")

if __name__ == '__main__':
    main()
