import os
import cv2
import numpy as np
from PIL import Image

def analyze_confusion_matrix():
    cm_path = r"runs/detect/safety_v1-4/confusion_matrix.png"
    cm_norm_path = r"runs/detect/safety_v1-4/confusion_matrix_normalized.png"
    print("Confusion matrix files exist:", os.path.exists(cm_path), os.path.exists(cm_norm_path))

def analyze_val_batches():
    batches = ['val_batch0', 'val_batch1', 'val_batch2']
    base_dir = r"runs/detect/safety_v1-4"
    for b in batches:
        pred_p = os.path.join(base_dir, f"{b}_pred.jpg")
        lbl_p = os.path.join(base_dir, f"{b}_labels.jpg")
        if os.path.exists(pred_p):
            im_pred = Image.open(pred_p)
            im_lbl = Image.open(lbl_p)
            print(f"{b}: Pred image size {im_pred.size}, Label image size {im_lbl.size}")

if __name__ == '__main__':
    analyze_confusion_matrix()
    analyze_val_batches()
