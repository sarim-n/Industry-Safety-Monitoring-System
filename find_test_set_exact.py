import os
from pathlib import Path
import json

base_dir = Path(r'C:\Users\sarim\safety monitoring')

def check_dir(img_dir):
    imgs = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    if len(imgs) == 143:
        lbl_dir = img_dir.parent.parent / 'labels' / img_dir.name
        if not lbl_dir.exists():
            lbl_dir = img_dir.parent / 'labels' # just in case
        if not lbl_dir.exists():
            return None
        
        lbls = list(lbl_dir.glob('*.txt'))
        
        counts = {0: 0, 1: 0, 2: 0}
        total = 0
        for lbl_file in lbls:
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            cid = int(parts[0])
                            counts[cid] = counts.get(cid, 0) + 1
                            total += 1
                        except: pass
        
        return {
            'path': str(img_dir),
            'images': len(imgs),
            'labels': len(lbls),
            'total_objects': total,
            'helmet': counts.get(0, 0),
            'mask': counts.get(1, 0),
            'person': counts.get(2, 0)
        }
    return None

results = []
for p in base_dir.rglob('*'):
    if p.is_dir() and 'site-packages' not in str(p) and '.gemini' not in str(p):
        res = check_dir(p)
        if res:
            results.append(res)

for r in results:
    print(r)
