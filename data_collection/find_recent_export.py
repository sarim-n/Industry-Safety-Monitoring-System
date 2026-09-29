import os
import time
from pathlib import Path

root = Path(r"C:\Users\sarim\safety monitoring")
now = time.time()

print("Files/folders created/modified in the last 10 minutes:")
for p in root.rglob("*"):
    try:
        mtime = p.stat().st_mtime
        if now - mtime < 600:
            if "dataset" not in p.name and ".git" not in str(p) and "__pycache__" not in str(p):
                print(f"  {p.stat().st_mtime} | {'DIR ' if p.is_dir() else 'FILE'} | {p}")
    except Exception:
        pass
