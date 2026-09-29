import os
from pathlib import Path

base_dir = Path(r'C:\Users\sarim\safety monitoring')

def get_label_dir(img_dir):
    # Try adjacent labels folder
    # e.g., if img_dir is test/images, label is test/labels
    if img_dir.name == 'images':
        lbl_dir = img_dir.parent / 'labels'
        if lbl_dir.exists(): return lbl_dir
    # e.g. if img_dir is test, label is test/labels maybe?
    # or just test itself?
    lbl_dir = img_dir.parent / 'labels' / img_dir.name
    if lbl_dir.exists(): return lbl_dir
    
    # What if labels are in the same dir?
    if list(img_dir.glob('*.txt')):
        return img_dir
        
    # What if img_dir is `roboflow_export/test/images` ?
    # covered by first check.
    
    # What if it's `roboflow_export/test` and labels are in `roboflow_export/test`?
    # covered by third check.
    
    return None

results = []
for p in base_dir.rglob('*'):
    if p.is_dir() and 'site-packages' not in str(p) and '.gemini' not in str(p):
        imgs = list(p.glob('*.jpg')) + list(p.glob('*.png'))
        if len(imgs) == 143:
            lbl_dir = get_label_dir(p)
            if not lbl_dir: continue
            
            lbls = list(lbl_dir.glob('*.txt'))
            # Exclude classes.txt if present
            lbls = [l for l in lbls if l.name != 'classes.txt']
            
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
            
            results.append({
                'path': str(p),
                'images': len(imgs),
                'labels': len(lbls),
                'total_objects': total,
                'helmet': counts.get(0, 0),
                'mask': counts.get(1, 0),
                'person': counts.get(2, 0)
            })

for r in results:
    print(r)
