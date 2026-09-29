"""
Run 4: Controlled Model Experiment with Roboflow Exported Data (Balanced Mask Data)
====================================================================================
Model: YOLOv8s @ imgsz=800
Dataset: training_dataset_run4 (995 train images, 136 val images, 143 test images untouched)
Output: runs/detect/safety_v1-4_run4_yolov8s_800/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    import time
    import json
    import torch
    from pathlib import Path
    from ultralytics import YOLO

    PROJECT_ROOT = Path(r"C:\Users\sarim\safety monitoring")
    DATA_YAML    = PROJECT_ROOT / "training_dataset_run4" / "data.yaml"
    RUNS_DIR     = PROJECT_ROOT / "runs"

    CONFIG = dict(
        data       = str(DATA_YAML),
        imgsz      = 800,
        epochs     = 50,
        batch      = 4,
        device     = 0,
        patience   = 15,
        workers    = 2,
        project    = str(RUNS_DIR / "detect"),
        name       = "safety_v1-4_run4_yolov8s_800",
        exist_ok   = True,
        pretrained = True,
        optimizer  = "auto",
        verbose    = True,
        seed       = 42,
        save       = True,
        save_period= -1,
        # Augmentation (Identical to Run 2B & Run 3)
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        degrees    = 0.0,
        translate  = 0.1,
        scale      = 0.5,
        flipud     = 0.0,
        fliplr     = 0.5,
        mosaic     = 1.0,
        mixup      = 0.0,
    )

    SEP = "=" * 64
    sep = "-" * 64

    print(SEP)
    print("  YOLOv8s — RUN 4 (BALANCED MASK DATA @ imgsz=800)".center(64))
    print(SEP)

    print("\n[PRE-FLIGHT]")
    print(sep)
    assert DATA_YAML.exists(), f"data.yaml not found: {DATA_YAML}"
    print(f"  data.yaml  : {DATA_YAML}  OK")

    cuda_ok = torch.cuda.is_available()
    assert cuda_ok, "CUDA not available — aborting."
    gpu_name   = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU        : {gpu_name}  ({total_vram:.1f} GB)")
    print(f"  Batch      : {CONFIG['batch']}  |  imgsz: {CONFIG['imgsz']}  |  epochs: {CONFIG['epochs']}")
    print(f"  Output     : {CONFIG['project']}/{CONFIG['name']}")
    print(f"  TEST split : untouched")

    print(f"\n[LOADING MODEL]")
    model = YOLO("yolov8s.pt")
    print(f"  yolov8s.pt loaded OK")

    torch.cuda.reset_peak_memory_stats(0)

    print(f"\n[TRAINING — starting now]")
    print(SEP)

    t_start = time.perf_counter()
    try:
        results = model.train(**CONFIG)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
            print("\n[CUDA OOM DETECTED] Reducing batch size from 4 to 2...")
            torch.cuda.empty_cache()
            CONFIG['batch'] = 2
            results = model.train(**CONFIG)
        else:
            raise e

    t_end = time.perf_counter()
    train_duration_sec = t_end - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated(0) / 1024**2

    print(SEP)
    print("  TRAINING COMPLETE".center(64))
    print(SEP)
    print(f"  Duration : {train_duration_sec / 60.0:.2f} minutes ({train_duration_sec:.1f} s)")
    print(f"  Peak VRAM: {peak_vram_mb:.1f} MB")

    # Save run metadata
    run_meta = {
        "duration_sec": train_duration_sec,
        "peak_vram_mb": peak_vram_mb,
        "best_epoch": results.save_dir if hasattr(results, "save_dir") else str(RUNS_DIR / "detect" / "safety_v1-4_run4_yolov8s_800")
    }
    with open(RUNS_DIR / "detect" / "safety_v1-4_run4_yolov8s_800" / "run4_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

if __name__ == '__main__':
    main()
