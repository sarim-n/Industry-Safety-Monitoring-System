import os
import sys
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

print("Project dict / attributes:")
for attr in dir(project):
    if not attr.startswith("_"):
        print(" ", attr)

print("\nProject stats / metadata:")
print("  name       :", getattr(project, "name", None))
print("  id         :", getattr(project, "id", None))
print("  type       :", getattr(project, "type", None))
print("  classes    :", getattr(project, "classes", None))
print("  images     :", getattr(project, "images", None))
print("  annotation :", getattr(project, "annotation", None))

# Try project.version or project.export or dataset download methods
try:
    print("\nTrying project.export('yolov8'):")
    dataset = project.export("yolov8")
    print("Dataset export result:", dataset)
except Exception as e:
    print("project.export() error:", e)

try:
    print("\nTrying project.version(1):")
    v1 = project.version(1)
    print("Version 1 result:", v1)
except Exception as e:
    print("project.version(1) error:", e)
