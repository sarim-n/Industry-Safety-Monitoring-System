import os
import sys

api_key = os.environ.get("ROBOFLOW_API_KEY")

# Fallback: check Windows user environment variable if not in current process env
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

if not api_key:
    print("[ERROR] ROBOFLOW_API_KEY environment variable is not set.")
    sys.exit(1)

try:
    from roboflow import Roboflow
except ImportError:
    print("[INFO] Installing roboflow python package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
    from roboflow import Roboflow

rf = Roboflow(api_key=api_key)

workspace_name = "sarimahmedn-official-gmail-com"
project_slug = "more-edpuy"

print("==================================================================")
print("   CONNECTING TO ROBOFLOW WORKSPACE & PROJECT")
print("==================================================================")

try:
    workspace = rf.workspace(workspace_name)
    print(f"Workspace: {workspace_name}  OK")
    
    project = workspace.project(project_slug)
    print(f"Project  : {project.name} (id: {project.id}, type: {project.type})")
    
    versions = project.versions()
    print(f"\nAvailable versions ({len(versions)}):")
    for v in versions:
        # Inspect version properties safely
        v_num = getattr(v, "version", "unknown")
        v_name = getattr(v, "name", "")
        v_imgs = getattr(v, "images", "unknown")
        v_split = getattr(v, "split", {})
        print(f"  Version {v_num}: {v_name} | Images: {v_imgs} | Split: {v_split}")
        
    latest_v = versions[0] if versions else None
    if latest_v:
        print(f"\nLatest Version: {latest_v.version}")
        
except Exception as e:
    print(f"[ERROR] Failed to query Roboflow API: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
