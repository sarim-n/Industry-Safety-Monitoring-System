"""
Rule 1 Person Duplicate Suppression Module
==========================================
Provides deterministic, person-class-only duplicate suppression for the safety pipeline.

Pipeline position:
  YOLO -> PERSON DUPLICATE SUPPRESSION -> PPE Association -> Temporal Confirmation
"""

import math
from typing import List, Dict, Any, Tuple
from src.safety.ppe_association import box_iou


def box_ioa(box_inner: Tuple[float, float, float, float], box_outer: Tuple[float, float, float, float]) -> float:
    x1 = max(box_inner[0], box_outer[0])
    y1 = max(box_inner[1], box_outer[1])
    x2 = min(box_inner[2], box_outer[2])
    y2 = min(box_inner[3], box_outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_inner = max(0.0, box_inner[2] - box_inner[0]) * max(0.0, box_inner[3] - box_inner[1])
    return inter / area_inner if area_inner > 0.0 else 0.0


def compute_pair_features(b1: List[float], b2: List[float], conf1: float, conf2: float, img_w: float, img_h: float) -> Dict[str, Any]:
    if conf1 >= conf2:
        box_a, conf_a = b1, conf1
        box_b, conf_b = b2, conf2
    else:
        box_a, conf_a = b2, conf2
        box_b, conf_b = b1, conf1

    w_a = max(0.0, box_a[2] - box_a[0])
    h_a = max(0.0, box_a[3] - box_a[1])
    area_a = w_a * h_a
    cx_a = (box_a[0] + box_a[2]) / 2.0
    cy_a = (box_a[1] + box_a[3]) / 2.0
    diag_a = math.sqrt(w_a**2 + h_a**2)

    w_b = max(0.0, box_b[2] - box_b[0])
    h_b = max(0.0, box_b[3] - box_b[1])
    area_b = w_b * h_b
    cx_b = (box_b[0] + box_b[2]) / 2.0
    cy_b = (box_b[1] + box_b[3]) / 2.0
    diag_b = math.sqrt(w_b**2 + h_b**2)

    iou = box_iou(box_a, box_b)
    dx = abs(cx_a - cx_b)
    dy = abs(cy_a - cy_b)
    center_dist = math.sqrt(dx**2 + dy**2)

    diag_large = max(diag_a, diag_b)
    area_large = max(area_a, area_b)
    area_small = min(area_a, area_b)

    norm_dist_diag_large = center_dist / max(1e-6, diag_large)
    max_containment = max(box_ioa(box_a, box_b), box_ioa(box_b, box_a))
    area_ratio = area_small / max(1e-6, area_large)

    return {
        'iou': iou,
        'center_dist': center_dist,
        'norm_dist_diag_large': norm_dist_diag_large,
        'max_containment': max_containment,
        'area_ratio': area_ratio
    }


def is_rule1_duplicate(b1: List[float], b2: List[float], conf1: float, conf2: float, img_w: float = 800.0, img_h: float = 800.0) -> bool:
    """
    Evaluates whether two candidate person bounding boxes trigger Rule 1.
    Rule 1 (Ultra Conservative):
      IoU >= 0.65 AND MaxContainment >= 0.95 AND NormCenterDist <= 0.10 AND AreaRatio >= 0.60
    """
    feats = compute_pair_features(b1, b2, conf1, conf2, img_w, img_h)
    return (feats['iou'] >= 0.65) and \
           (feats['max_containment'] >= 0.95) and \
           (feats['norm_dist_diag_large'] <= 0.10) and \
           (feats['area_ratio'] >= 0.60)


def suppress_duplicate_person_detections(person_detections: List[Dict[str, Any]], img_w: float = 800.0, img_h: float = 800.0) -> List[Dict[str, Any]]:
    """
    Suppresses duplicate person detections using Rule 1.
    
    Args:
        person_detections: List of detection dicts {'cls': 2, 'conf': float, 'box': [x1, y1, x2, y2]}
        img_w: Image width
        img_h: Image height
        
    Returns:
        List of retained person detection dicts.
    """
    if not person_detections:
        return []

    # Filter out non-person detections or preserve them
    person_only = [d for d in person_detections if d.get('cls') == 2]
    non_person = [d for d in person_detections if d.get('cls') != 2]

    if not person_only:
        return person_detections

    # Sort descending by confidence (break ties deterministically)
    sorted_boxes = sorted(
        person_only,
        key=lambda x: (x['conf'], x['box'][0], x['box'][1]),
        reverse=True
    )

    kept = []
    for candidate in sorted_boxes:
        should_suppress = False
        for retained in kept:
            if is_rule1_duplicate(candidate['box'], retained['box'], candidate['conf'], retained['conf'], img_w, img_h):
                should_suppress = True
                break
        if not should_suppress:
            kept.append(candidate)

    return kept + non_person
