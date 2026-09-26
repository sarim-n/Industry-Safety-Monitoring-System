import torch
import ultralytics
import cv2
from ultralytics import YOLO
from pathlib import Path

SEP = "=" * 52

print(SEP)
print("  ENVIRONMENT VERIFICATION REPORT")
print(SEP)

# Python
import sys
print(f"  Python        : {sys.version.split()[0]}")

# PyTorch
print(f"  PyTorch       : {torch.__version__}")

# CUDA
cuda_ok = torch.cuda.is_available()
print(f"  CUDA available: {cuda_ok}")
print(f"  CUDA version  : {torch.version.cuda}")

# GPU
if cuda_ok:
    gpu_name = torch.cuda.get_device_name(0)
    props    = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    free_gb  = (props.total_memory - torch.cuda.memory_reserved(0)) / 1024**3
    print(f"  GPU name      : {gpu_name}")
    print(f"  GPU VRAM total: {total_gb:.2f} GB")
    print(f"  GPU VRAM free : {free_gb:.2f} GB")
else:
    print("  GPU name      : NONE")
    print("  GPU VRAM      : N/A")

# Ultralytics
print(f"  Ultralytics   : {ultralytics.__version__}")
print(f"  YOLO import   : OK")
print(f"  OpenCV        : {cv2.__version__}")

# data.yaml
import yaml
yaml_path = Path(r"C:\Users\sarim\safety monitoring\training_dataset_v2\data.yaml")
if yaml_path.exists():
    d = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names_ok = d.get("names") == {0: "helmet", 1: "mask", 2: "person"}
    print(f"  data.yaml     : EXISTS  |  names valid: {names_ok}")
else:
    print(f"  data.yaml     : MISSING")

print(SEP)
if cuda_ok:
    print("  STATUS: READY for YOLOv8n training on GPU")
else:
    print("  STATUS: WARNING -- CUDA not available, CPU-only")
print(SEP)
