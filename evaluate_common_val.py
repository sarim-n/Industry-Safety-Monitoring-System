import os
import glob
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def setup_yaml(base_yaml, val_txt_path, out_yaml_path):
    with open(base_yaml, 'r') as f:
        lines = f.readlines()
    with open(out_yaml_path, 'w') as f:
        for line in lines:
            if line.startswith('val:'):
                f.write(f'val: {val_txt_path}\n')
            else:
                f.write(line)

def run_evaluation():
    original_val_dir = r"C:\Users\sarim\safety monitoring\final_dataset\images\val"
    base_yaml = r"C:\Users\sarim\safety monitoring\final_dataset\data.yaml"
    
    all_images = glob.glob(os.path.join(original_val_dir, "*.*"))
    list_10810476 = [img for img in all_images if "10810476" in os.path.basename(img)]
    list_clean = [img for img in all_images if "10810476" not in os.path.basename(img)]
    
    print(f"Total images: {len(all_images)}")
    print(f"Clean images: {len(list_clean)}")
    print(f"10810476 images: {len(list_10810476)}")
    
    # Create txt files
    with open("val_all.txt", "w") as f: f.write("\n".join(all_images))
    with open("val_clean.txt", "w") as f: f.write("\n".join(list_clean))
    with open("val_10810476.txt", "w") as f: f.write("\n".join(list_10810476))
    
    # Create yaml files
    setup_yaml(base_yaml, os.path.abspath("val_all.txt"), "data_val_all.yaml")
    setup_yaml(base_yaml, os.path.abspath("val_clean.txt"), "data_val_clean.yaml")
    setup_yaml(base_yaml, os.path.abspath("val_10810476.txt"), "data_val_10810476.yaml")
    
    model_a_path = r"C:\Users\sarim\safety monitoring\runs\detect\safety_v1-4_run2b_yolov8s_800\weights\best.pt"
    model_b_path = r"C:\Users\sarim\turf_ai\runs\detect\runs\detect\safety_v1-4_run2b_clean10810476_yolov8s_800\weights\best.pt"
    
    models = {
        "Model A (Original)": model_a_path,
        "Model B (Clean)": model_b_path
    }
    
    datasets = {
        "A (All 194)": "data_val_all.yaml",
        "B (Clean 183)": "data_val_clean.yaml",
        "C (10810476 11)": "data_val_10810476.yaml"
    }
    
    for model_name, model_path in models.items():
        print(f"\n======================================")
        print(f"Evaluating {model_name}")
        print(f"======================================")
        model = YOLO(model_path)
        
        for ds_name, ds_yaml in datasets.items():
            print(f"\n--- Dataset: {ds_name} ---")
            results = model.val(data=os.path.abspath(ds_yaml), imgsz=800, batch=4, workers=2, project="runs/eval_comparison", name=f"{model_name.replace(' ', '_')}_{ds_name.replace(' ', '_')}", exist_ok=True)
            print(f"mAP50: {results.box.map50}")
            print(f"mAP50-95: {results.box.map}")
            print(f"Speed: {results.speed}")

if __name__ == '__main__':
    run_evaluation()
