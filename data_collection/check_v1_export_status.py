import os
import sys
import shutil
from pathlib import Path
from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
        api_key, _ = winreg.QueryValueEx(key, "ROBOFLOW_API_KEY")
        winreg.CloseKey(key)
        if api_key:
            os.environ["ROBOFLOW_API_KEY"] = api_key
    except Exception:
        pass

rf = Roboflow(api_key=api_key)
project = rf.workspace("sarimahmedn-official-gmail-com").project("more-edpuy")

v1 = project.version(1)
print(f"Version 1 stats: name={v1.name}, images={v1.images}, splits={v1.splits}")

target_dir = Path(r"C:\Users\sarim\safety monitoring\data_collection\roboflow_more_full_export")
if target_dir.exists():
    shutil.rmtree(str(target_dir))
target_dir.mkdir(parents=True, exist_ok=True)

print("Downloading dataset in YOLOv8 format...")
dataset = v1.download("yolov8", location=str(target_dir), overwrite=True)

print("\nDownload completed! Checking location contents:")
loc_path = Path(dataset.location)
items = list(loc_path.rglob("*"))
print(f"Total items found in {loc_path}: {len(items)}")

for p in loc_path.glob("*"):
    print("  ", p.name, "DIR" if p.is_dir() else "FILE")
