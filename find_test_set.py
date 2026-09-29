import os
from pathlib import Path

base = Path(r'C:\Users\sarim\safety monitoring')
for p in base.rglob('test/images'):
    imgs = list(p.glob('*.jpg')) + list(p.glob('*.png'))
    if len(imgs) > 0:
        print(f'{p}: {len(imgs)}')
