"""
Unit Tests for Rule 1 Person Duplicate Suppression Module
==========================================================
Verifies deterministic person-class duplicate suppression:
  IoU >= 0.65 AND MaxContainment >= 0.95 AND NormCenterDist <= 0.10 AND AreaRatio >= 0.60
"""

import unittest
from src.safety.person_suppression import (
    is_rule1_duplicate,
    suppress_duplicate_person_detections,
    compute_pair_features
)


class TestPersonSuppression(unittest.TestCase):

    def test_a_exact_duplicate_boxes(self):
        """Test A: Exact duplicate boxes -> suppress lower confidence box."""
        box_a = [100.0, 100.0, 300.0, 500.0]
        box_b = [100.0, 100.0, 300.0, 500.0]
        dets = [
            {'cls': 2, 'conf': 0.85, 'box': box_a},
            {'cls': 2, 'conf': 0.60, 'box': box_b}
        ]
        res = suppress_duplicate_person_detections(dets, 800, 800)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]['conf'], 0.85)

    def test_b_high_overlap_contained_duplicate(self):
        """Test B: High overlap contained duplicate -> suppress lower confidence box."""
        box_a = [100.0, 100.0, 300.0, 500.0]  # area = 80000
        box_b = [105.0, 105.0, 295.0, 495.0]  # area = 74100 (almost 100% contained inside A)
        dets = [
            {'cls': 2, 'conf': 0.90, 'box': box_a},
            {'cls': 2, 'conf': 0.65, 'box': box_b}
        ]
        res = suppress_duplicate_person_detections(dets, 800, 800)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]['conf'], 0.90)

    def test_c_iou_below_threshold(self):
        """Test C: IoU below threshold (0.55 < 0.65) -> preserve both."""
        box_a = [100.0, 100.0, 300.0, 500.0]
        box_b = [150.0, 100.0, 350.0, 500.0]  # IoU ~ 0.60 < 0.65
        dets = [
            {'cls': 2, 'conf': 0.85, 'box': box_a},
            {'cls': 2, 'conf': 0.70, 'box': box_b}
        ]
        res = suppress_duplicate_person_detections(dets, 800, 800)
        self.assertEqual(len(res), 2)

    def test_d_center_distance_above_threshold(self):
        """Test D: Center distance above threshold (NDist > 0.10) -> preserve both."""
        box_a = [100.0, 100.0, 300.0, 500.0]
        box_b = [160.0, 160.0, 360.0, 560.0]  # shifted center
        dets = [
            {'cls': 2, 'conf': 0.85, 'box': box_a},
            {'cls': 2, 'conf': 0.70, 'box': box_b}
        ]
        res = suppress_duplicate_person_detections(dets, 800, 800)
        self.assertEqual(len(res), 2)

    def test_e_area_ratio_below_threshold(self):
        """Test E: Area ratio below threshold (AreaRatio < 0.60) -> preserve both."""
        box_a = [100.0, 100.0, 400.0, 600.0]  # area = 150000
        box_b = [150.0, 150.0, 300.0, 400.0]  # area = 37500 (area ratio = 0.25 < 0.60)
        dets = [
            {'cls': 2, 'conf': 0.85, 'box': box_a},
            {'cls': 2, 'conf': 0.70, 'box': box_b}
        ]
        res = suppress_duplicate_person_detections(dets, 800, 800)
        self.assertEqual(len(res), 2)

    def test_f_legitimate_overlapping_workers(self):
        """Test F: Legitimate overlapping workers -> preserve both."""
        # Two workers standing side by side with partial overlap
        worker_1 = [100.0, 100.0, 250.0, 500.0]
        worker_2 = [180.0, 100.0, 330.0, 500.0]
        dets = [
            {'cls': 2, 'conf': 0.88, 'box': worker_1},
            {'cls': 2, 'conf': 0.82, 'box': worker_2}
        ]
        res = suppress_duplicate_person_detections(dets, 800, 800)
        self.assertEqual(len(res), 2)

    def test_g_different_classes(self):
        """Test G: Non-person classes (helmet=0, mask=1) -> must NOT be suppressed."""
        helmet_box = [150.0, 100.0, 200.0, 150.0]
        mask_box = [150.0, 150.0, 200.0, 180.0]
        dets = [
            {'cls': 0, 'conf': 0.95, 'box': helmet_box},
            {'cls': 1, 'conf': 0.90, 'box': mask_box}
        ]
        res = suppress_duplicate_person_detections(dets, 800, 800)
        self.assertEqual(len(res), 2)

    def test_h_equal_confidence_deterministic_case(self):
        """Test H: Equal confidence boxes -> deterministic tie-breaking."""
        box_a = [100.0, 100.0, 300.0, 500.0]
        box_b = [100.0, 100.0, 300.0, 500.0]
        dets = [
            {'cls': 2, 'conf': 0.80, 'box': box_a},
            {'cls': 2, 'conf': 0.80, 'box': box_b}
        ]
        res = suppress_duplicate_person_detections(dets, 800, 800)
        self.assertEqual(len(res), 1)

    def test_i_empty_detection_list(self):
        """Test I: Empty detection list -> returns empty list."""
        self.assertEqual(suppress_duplicate_person_detections([], 800, 800), [])

    def test_j_single_person(self):
        """Test J: Single person box -> returns single person box intact."""
        dets = [{'cls': 2, 'conf': 0.85, 'box': [100.0, 100.0, 300.0, 500.0]}]
        res = suppress_duplicate_person_detections(dets, 800, 800)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]['conf'], 0.85)


if __name__ == '__main__':
    unittest.main()
