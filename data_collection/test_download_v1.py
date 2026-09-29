import os
import sys
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

PROJECT_ROOT = Path(r"C:\Users\sarim\safety monitoring")
TARGET_DIR = PROJECT_ROOT / "data_collection" / "roboflow_more_full_export"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

rf = Roboflow(api_key=api_key)
project = rf.workspace("sarimahmedn-official-gmail-com").project("more-edpuy")

version_1 = project.version(1)
print(f"Version 1 info: name={version_1.name}, images={version_1.images}")

print("\nDownloading Version 1 dataset in YOLOv8 format...")
dataset = version_1.download("yolov8", location=str(TARGET_DIR))

print("Dataset download complete!")
print("Dataset location:", dataset.location)
