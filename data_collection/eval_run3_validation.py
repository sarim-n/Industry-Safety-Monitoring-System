import os
import sys
import json
from pathlib import Path
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    PROJECT_ROOT = Path(r"C:\Users\sarim\safety monitoring")
    VAL_YAML = PROJECT_ROOT / "training_dataset_v2" / "data.yaml"

    RUN2B_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "safety_v1-4_run2b_yolov8s_800" / "weights" / "best.pt"
    RUN3_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "safety_v1-4_run3_mask_hard_negative" / "weights" / "best.pt"

    model_2b = YOLO(str(RUN2B_WEIGHTS))
    model_3 = YOLO(str(RUN3_WEIGHTS))

    print("\n[EVALUATING RUN 2B BASELINE ON VALIDATION SET]")
    res_2b = model_2b.val(data=str(VAL_YAML), imgsz=800, split="val", verbose=True)

    print("\n[EVALUATING RUN 3 EXPERIMENT ON VALIDATION SET]")
    res_3 = model_3.val(data=str(VAL_YAML), imgsz=800, split="val", verbose=True)

    def extract_metrics(res):
        # res.results_dict keys:
        # 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
        # per-class array: res.box.p, res.box.r, res.box.ap50, res.box.ap
        p = res.box.p
        r = res.box.r
        map50 = res.box.ap50
        map95 = res.box.ap
        
        names = res.names
        metrics = {
            'overall': {
                'precision': float(res.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(res.results_dict.get('metrics/recall(B)', 0)),
                'map50': float(res.results_dict.get('metrics/mAP50(B)', 0)),
                'map50_95': float(res.results_dict.get('metrics/mAP50-95(B)', 0))
            },
            'per_class': {}
        }
        
        for i, c_name in names.items():
            metrics['per_class'][c_name] = {
                'precision': float(p[i]),
                'recall': float(r[i]),
                'map50': float(map50[i]),
                'map50_95': float(map95[i])
            }
        return metrics

    m2b = extract_metrics(res_2b)
    m3 = extract_metrics(res_3)

    print("\n==================================================================")
    print("                    VALIDATION METRICS COMPARISON                  ")
    print("==================================================================")
    print("OVERALL:")
    print(f"  Run 2B : P={m2b['overall']['precision']:.4f}, R={m2b['overall']['recall']:.4f}, mAP50={m2b['overall']['map50']:.4f}, mAP50-95={m2b['overall']['map50_95']:.4f}")
    print(f"  Run 3  : P={m3['overall']['precision']:.4f}, R={m3['overall']['recall']:.4f}, mAP50={m3['overall']['map50']:.4f}, mAP50-95={m3['overall']['map50_95']:.4f}")

    for c_name in ['helmet', 'mask', 'person']:
        b_c = m2b['per_class'][c_name]
        r3_c = m3['per_class'][c_name]
        print(f"\nCLASS [{c_name.upper()}]:")
        print(f"  Run 2B : P={b_c['precision']:.4f}, R={b_c['recall']:.4f}, mAP50={b_c['map50']:.4f}, mAP50-95={b_c['map50_95']:.4f}")
        print(f"  Run 3  : P={r3_c['precision']:.4f}, R={r3_c['recall']:.4f}, mAP50={r3_c['map50']:.4f}, mAP50-95={r3_c['map50_95']:.4f}")
        print(f"  Change : P={r3_c['precision']-b_c['precision']:+.4f}, R={r3_c['recall']-b_c['recall']:+.4f}, mAP50={r3_c['map50']-b_c['map50']:+.4f}, mAP50-95={r3_c['map50_95']-b_c['map50_95']:+.4f}")

    out_dir = PROJECT_ROOT / "reports" / "run3_mask_hard_negative"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "val_metrics_comparison.json", "w") as f:
        json.dump({'run2b': m2b, 'run3': m3}, f, indent=2)

    print(f"\nSaved metrics comparison to {out_dir / 'val_metrics_comparison.json'}")

if __name__ == '__main__':
    main()
