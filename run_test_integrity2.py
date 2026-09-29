import os
from pathlib import Path

test_img_dir = Path(r'C:\Users\sarim\safety monitoring\roboflow_batch2\test\images')
test_lbl_dir = Path(r'C:\Users\sarim\safety monitoring\roboflow_batch2\test\labels')

if not test_img_dir.exists():
    print('No test images dir')
else:
    print(f'Test images: {len(list(test_img_dir.glob("*.jpg")))}')

if not test_lbl_dir.exists():
    print('No test labels dir')
else:
    test_lbls = list(test_lbl_dir.glob("*.txt"))
    print(f'Test label files: {len(test_lbls)}')
    class_counts = {0: 0, 1: 0, 2: 0}
    for lbl_file in test_lbls:
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_counts[int(parts[0])] += 1
    print(class_counts)
