"""
YOLO GPU SMOKE TEST
====================
- Loads YOLOv8n (downloads pretrained weights if not cached)
- Forces inference on CUDA device 0 (RTX 2050)
- Runs on one image from training_dataset_v2/images/val
- Reports GPU usage, inference time, detections
- Does NOT train, does NOT modify any dataset files
"""

import time
from pathlib import Path
import torch
from ultralytics import YOLO

SEP = "=" * 60
sep = "-" * 60

VAL_DIR = Path(r"C:\Users\sarim\safety monitoring\training_dataset_v2\images\val")

print(SEP)
print("  YOLO GPU SMOKE TEST".center(60))
print(SEP)

# ── 1. CUDA sanity ──────────────────────────────────────────────
print("\n[1] CUDA CHECK")
print(sep)
cuda_ok = torch.cuda.is_available()
print(f"  CUDA available : {cuda_ok}")
if not cuda_ok:
    print("  [FATAL] CUDA not available. Aborting smoke test.")
    raise SystemExit(1)

device_name = torch.cuda.get_device_name(0)
props       = torch.cuda.get_device_properties(0)
total_vram  = props.total_memory / 1024**3
free_vram   = (props.total_memory - torch.cuda.memory_reserved(0)) / 1024**3
print(f"  Device 0       : {device_name}")
print(f"  VRAM total     : {total_vram:.2f} GB")
print(f"  VRAM free      : {free_vram:.2f} GB")

# ── 2. Pick one val image ────────────────────────────────────────
print("\n[2] SELECTING TEST IMAGE")
print(sep)
val_images = sorted([f for f in VAL_DIR.iterdir()
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
if not val_images:
    print("  [FATAL] No images found in val directory.")
    raise SystemExit(1)
test_image = val_images[0]
print(f"  Image selected : {test_image.name}")
print(f"  Full path      : {test_image}")

# ── 3. Load YOLOv8n ─────────────────────────────────────────────
print("\n[3] LOADING YOLOv8n")
print(sep)
t0 = time.perf_counter()
model = YOLO("yolov8n.pt")
load_time = time.perf_counter() - t0
print(f"  Model          : YOLOv8n (pretrained COCO weights)")
print(f"  Load time      : {load_time:.2f}s")

# Confirm model is on GPU after first forward pass
print(f"  Moving to      : cuda:0")

# ── 4. Warm-up pass (forces CUDA kernel init) ────────────────────
print("\n[4] CUDA WARM-UP (1 forward pass to init kernels)")
print(sep)
_ = model.predict(str(test_image), device=0, verbose=False)
torch.cuda.synchronize()

vram_after_warmup = torch.cuda.memory_reserved(0) / 1024**3
print(f"  Warm-up done. VRAM reserved after init: {vram_after_warmup:.3f} GB")

# ── 5. Timed inference ───────────────────────────────────────────
print("\n[5] TIMED INFERENCE (device=0)")
print(sep)

torch.cuda.synchronize()
t_start = time.perf_counter()
results = model.predict(str(test_image), device=0, verbose=False)
torch.cuda.synchronize()
inf_time_ms = (time.perf_counter() - t_start) * 1000

print(f"  Inference time : {inf_time_ms:.1f} ms")

# ── 6. Parse detections ──────────────────────────────────────────
print("\n[6] DETECTIONS (pretrained COCO classes — not our custom classes)")
print(sep)
r = results[0]
boxes = r.boxes

if boxes is None or len(boxes) == 0:
    print("  No objects detected in this image (expected — pretrained COCO weights)")
else:
    print(f"  Objects found  : {len(boxes)}")
    for box in boxes:
        cls_id    = int(box.cls[0])
        cls_name  = model.names[cls_id]
        conf      = float(box.conf[0])
        xyxy      = box.xyxy[0].tolist()
        print(f"    cls={cls_id} ({cls_name:15s})  conf={conf:.2f}  box={[round(v,1) for v in xyxy]}")

print(f"\n  NOTE: Detections use COCO 80-class weights (not helmet/mask/person).")
print(f"        After training on our dataset, classes will be 0=helmet 1=mask 2=person.")

# ── 7. GPU memory after inference ───────────────────────────────
print("\n[7] GPU MEMORY USAGE")
print(sep)
vram_reserved = torch.cuda.memory_reserved(0)  / 1024**3
vram_allocated = torch.cuda.memory_allocated(0) / 1024**3
print(f"  VRAM reserved  : {vram_reserved:.3f} GB")
print(f"  VRAM allocated : {vram_allocated:.3f} GB")
print(f"  VRAM free      : {total_vram - vram_reserved:.3f} GB")

# ── 8. Confirm CUDA was used ─────────────────────────────────────
print("\n[8] CUDA CONFIRMATION")
print(sep)
# Check the device of model parameters
param_device = next(model.model.parameters()).device
print(f"  Model params on: {param_device}")
cuda_used = param_device.type == "cuda"
print(f"  CUDA used      : {cuda_used}")

# ── Final Report ─────────────────────────────────────────────────
print()
print(SEP)
print("  SMOKE TEST SUMMARY".center(60))
print(SEP)
print(f"  Model loaded        : YES (YOLOv8n)")
print(f"  GPU device          : {device_name}")
print(f"  CUDA used           : {cuda_used}")
print(f"  Inference completed : YES")
print(f"  Inference time      : {inf_time_ms:.1f} ms")
print(f"  Detections          : {len(boxes) if boxes is not None else 0} objects")
print(f"  VRAM used (model)   : {vram_allocated:.3f} GB / {total_vram:.2f} GB")
print(f"  VRAM free           : {total_vram - vram_reserved:.3f} GB")
print()
if cuda_used:
    print("  STATUS: SMOKE TEST PASSED -- GPU READY FOR TRAINING".center(60))
else:
    print("  STATUS: WARNING -- GPU NOT USED".center(60))
print(SEP)
