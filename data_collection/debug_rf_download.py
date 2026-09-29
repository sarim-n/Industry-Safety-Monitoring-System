import os
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
print("v1:", v1)
print("v1.download:", v1.download)

# Download into a specific folder and print full result
target_path = Path(r"C:\Users\sarim\safety monitoring\data_collection\roboflow_more_full_export")
ds = v1.download("yolov8", location=str(target_path))
print("ds.location:", ds.location)

# Check all contents in ds.location recursively
loc_path = Path(ds.location)
all_items = list(loc_path.rglob("*"))
print(f"Total items in {loc_path}: {len(all_items)}")
for item in all_items[:10]:
    print(" ", item)
