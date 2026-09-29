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

print("Project splits:", getattr(project, "splits", None))

# Inspect list_versions
print("List versions:", project.list_versions())

# Let's check if generate_version method exists or what arguments it takes
import inspect
print("\ngenerate_version signature:", inspect.signature(project.generate_version))

# Let's check if there is a version 1 after checking or if we generate version 1
try:
    print("\nAttempting to generate version 1 if needed...")
    # generate_version parameters in Roboflow SDK
    v_info = project.generate_version(settings={"augmentation": {}, "preprocessing": {}})
    print("Generated version result:", v_info)
except Exception as e:
    print("generate_version error:", e)

# Re-check versions
versions = project.versions()
print("\nAvailable versions now:", versions)
for v in versions:
    print(f"  Version {v.version}: {v.name} | Images: {v.images}")
