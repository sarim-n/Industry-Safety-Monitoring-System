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

target_dir = Path(r"C:\Users\sarim\safety monitoring\data_collection\roboflow_more_full_export")
dataset = v1.download("yolov8", location=str(target_dir))
print("Downloaded dataset object:", dataset)
print("dataset.location:", dataset.location)

print("Listing files in target_dir:", list(target_dir.glob("*")))
