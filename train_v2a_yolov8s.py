"""
Run 2A: YOLOv8s @ imgsz=640
============================
Model capacity experiment: YOLOv8n -> YOLOv8s
Hardware: RTX 2050 (4 GB VRAM)
All training hyperparameters kept identical to Run 1 (safety_v1-4).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    import time
    import csv
    import torch
    from pathlib import Path
    from ultralytics import YOLO

    # ── Paths ─────────────────────────────────────────────────────────────────
    PROJECT_ROOT = Path(r"C:\Users\sarim\safety monitoring")
    DATA_YAML    = PROJECT_ROOT / "training_dataset_v2" / "data.yaml"
    RUNS_DIR     = PROJECT_ROOT / "runs"

    # ── Config ────────────────────────────────────────────────────────────────
    CONFIG = dict(
        data       = str(DATA_YAML),
        imgsz      = 640,
        epochs     = 50,
        batch      = 4,
        device     = 0,
        patience   = 15,
        workers    = 2,
        project    = str(RUNS_DIR / "detect"),
        name       = "safety_v1-4_run2_yolov8s_640",
        exist_ok   = True,
        pretrained = True,
        optimizer  = "auto",
        verbose    = True,
        seed       = 42,
        save       = True,
        save_period= -1,
        # Augmentation (Identical to Run 1)
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
    print("  YOLOv8s — RUN 2A (imgsz=640)".center(64))
    print(SEP)

    # ── Pre-flight ────────────────────────────────────────────────────────────
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

    # ── Load Model ────────────────────────────────────────────────────────────
    print(f"\n[LOADING MODEL]")
    model = YOLO("yolov8s.pt")
    print(f"  yolov8s.pt loaded OK")

    # Reset peak memory stats before training
    torch.cuda.reset_peak_memory_stats(0)

    print(f"\n[TRAINING — starting now]")
    print(SEP)

    t_start = time.perf_counter()
    results = model.train(**CONFIG)
    t_end   = time.perf_counter()
    elapsed_min = (t_end - t_start) / 60

    print()
    print(SEP)
    print("  TRAINING COMPLETE — RESULTS REPORT".center(64))
    print(SEP)

    save_dir  = Path(results.save_dir)
    best_ckpt = save_dir / "weights" / "best.pt"
    last_ckpt = save_dir / "weights" / "last.pt"
    results_csv = save_dir / "results.csv"

    print(f"\n  Run directory  : {save_dir}")
    print(f"  Training time  : {elapsed_min:.1f} minutes")
    print(f"  best.pt        : {'EXISTS' if best_ckpt.exists() else 'MISSING'}")
    print(f"  last.pt        : {'EXISTS' if last_ckpt.exists() else 'MISSING'}")
    print(f"  results.csv    : {'EXISTS' if results_csv.exists() else 'MISSING'}")

    # ── Metrics from results dict ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("  FINAL EPOCH METRICS")
    print(sep)
    try:
        rd = results.results_dict
        print(f"  Precision  : {rd.get('metrics/precision(B)', 'N/A')}")
        print(f"  Recall     : {rd.get('metrics/recall(B)', 'N/A')}")
        print(f"  mAP@50     : {rd.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP@50-95  : {rd.get('metrics/mAP50-95(B)', 'N/A')}")
    except Exception as e:
        print(f"  [Metrics error: {e}]")

    # ── Best epoch search ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  BEST EPOCH (from results.csv)")
    print(sep)
    try:
        best_epoch = -1
        best_map50 = -1.0
        best_row   = {}

        with open(results_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows   = [{k.strip(): v.strip() for k, v in r.items()} for r in reader]

        for row in rows:
            map50_key = next((k for k in row if "mAP50" in k and "95" not in k), None)
            if map50_key:
                try:
                    val = float(row[map50_key])
                    ep  = int(float(row.get("epoch", -1)))
                    if val > best_map50:
                        best_map50 = val
                        best_epoch = ep
                        best_row   = row
                except ValueError:
                    pass

        print(f"  Best epoch     : {best_epoch + 1}  (0-indexed: {best_epoch})")
        print(f"  Best mAP@50    : {best_map50:.4f}")
    except Exception as e:
        print(f"  [CSV parse error: {e}]")

    # ── Per-class validation on best.pt ───────────────────────────────────────
    print(f"\n{sep}")
    print("  PER-CLASS METRICS (val on best.pt)")
    print(sep)
    try:
        best_model  = YOLO(str(best_ckpt))
        val_results = best_model.val(
            data    = str(DATA_YAML),
            device  = 0,
            split   = "val",
            verbose = False,
        )
        class_names = {0: "helmet", 1: "mask", 2: "person"}
        ap50   = val_results.box.ap50
        ap5095 = val_results.box.ap
        prec   = val_results.box.p
        rec    = val_results.box.r

        print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'AP@50':>10} {'AP@50-95':>10}")
        print(f"  {'-'*54}")
        for i, name in class_names.items():
            try:
                print(f"  {name:<10} {prec[i]:>10.4f} {rec[i]:>10.4f} "
                      f"{ap50[i]:>10.4f} {ap5095[i]:>10.4f}")
            except IndexError:
                print(f"  {name:<10} N/A")

        print(f"\n  Overall:")
        print(f"  Precision  : {val_results.box.mp:.4f}")
        print(f"  Recall     : {val_results.box.mr:.4f}")
        print(f"  mAP@50     : {val_results.box.map50:.4f}")
        print(f"  mAP@50-95  : {val_results.box.map:.4f}")
        
        # Speed details
        speed = val_results.speed
        preprocess_ms = speed.get('preprocess', 0.0)
        inference_ms  = speed.get('inference', 0.0)
        postprocess_ms = speed.get('postprocess', 0.0)
        total_ms_per_img = preprocess_ms + inference_ms + postprocess_ms
        fps = 1000.0 / total_ms_per_img if total_ms_per_img > 0 else 0.0
        
        print(f"\n  Speed & Latency:")
        print(f"  Pre-process  : {preprocess_ms:.2f} ms")
        print(f"  Inference    : {inference_ms:.2f} ms")
        print(f"  Post-process : {postprocess_ms:.2f} ms")
        print(f"  Total latency: {total_ms_per_img:.2f} ms / image")
        print(f"  FPS          : {fps:.1f}")

    except Exception as e:
        print(f"  [Per-class val error: {e}]")

    # ── GPU memory summary ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  GPU / VRAM SUMMARY")
    print(sep)
    peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"  Peak VRAM allocated : {peak_vram:.3f} GB / {total_vram:.2f} GB")
    print(f"  VRAM headroom       : {total_vram - peak_vram:.3f} GB")

    print()
    print(SEP)
    print("  RUN 2A COMPLETE".center(64))
    print(SEP)

if __name__ == "__main__":
    main()
