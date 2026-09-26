"""
Regression Unit Tests for PPE Observability (Phase 7.6)
=========================================================
Tests geometry-based PPE observability rules and priority state resolution:
- Case A: Full person + helmet + mask -> SAFE
- Case B: Full person + helmet + no mask -> NO_MASK
- Case C: Full person + no helmet + mask -> NO_HELMET
- Case D: Full person + no helmet + no mask -> NO_HELMET_AND_MASK
- Case E: Face-only / insufficient PPE visibility -> UNCERTAIN
- Case F: Helmet clearly detected + face region uncertain -> helmet=YES, mask=UNKNOWN, overall=UNCERTAIN
- Case G: Helmet clearly detected + mask clearly detected + body area cropped -> SAFE
- Case H: Bottom of person cropped but head/face visible -> normal PPE evaluation
- Case I: Helmet candidate detected outside head ROI -> helmet not associated, evaluates observability
"""

import unittest
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.safety.ppe_association import (
    PPEAssociationConfig,
    PPEAssociator,
    PersonPPEState
)


class TestPPEObservability(unittest.TestCase):

    def setUp(self):
        self.config = PPEAssociationConfig()
        self.associator = PPEAssociator(self.config)
        self.img_w = 1920.0
        self.img_h = 1080.0

    def test_case_a_full_person_safe(self):
        """Case A: Full person visible, helmet + mask visible -> SAFE."""
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 200.0, 400.0, 800.0]},  # Person
            {'cls': 0, 'conf': 0.85, 'box': [220.0, 180.0, 380.0, 280.0]},  # Helmet
            {'cls': 1, 'conf': 0.80, 'box': [260.0, 240.0, 340.0, 320.0]}   # Mask
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].helmet_detected, "YES")
        self.assertEqual(states[0].mask_detected, "YES")
        self.assertEqual(states[0].safety_status, "SAFE")

    def test_case_b_full_person_no_mask(self):
        """Case B: Full person visible, helmet visible, mask absent -> NO_MASK."""
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 200.0, 400.0, 800.0]},  # Person
            {'cls': 0, 'conf': 0.85, 'box': [220.0, 180.0, 380.0, 280.0]}   # Helmet only
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].helmet_detected, "YES")
        self.assertEqual(states[0].mask_detected, "NO")
        self.assertEqual(states[0].safety_status, "NO_MASK")

    def test_case_c_full_person_no_helmet(self):
        """Case C: Full person visible, helmet absent, mask visible -> NO_HELMET."""
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 200.0, 400.0, 800.0]},  # Person
            {'cls': 1, 'conf': 0.80, 'box': [260.0, 240.0, 340.0, 320.0]}   # Mask only
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].helmet_detected, "NO")
        self.assertEqual(states[0].mask_detected, "YES")
        self.assertEqual(states[0].safety_status, "NO_HELMET")

    def test_case_d_full_person_neither_ppe(self):
        """Case D: Full person visible, neither PPE detected -> NO_HELMET_AND_MASK."""
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 200.0, 400.0, 800.0]}   # Person only
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].helmet_detected, "NO")
        self.assertEqual(states[0].mask_detected, "NO")
        self.assertEqual(states[0].safety_status, "NO_HELMET_AND_MASK")

    def test_case_e_face_only_crop(self):
        """Case E: Only face/head visible (aspect ratio < 1.0) -> UNCERTAIN."""
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 150.0, 380.0, 280.0]}   # w=180, h=130 (ar=0.72)
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].helmet_detected, "UNKNOWN")
        self.assertEqual(states[0].safety_status, "UNCERTAIN")

    def test_case_f_helmet_detected_face_uncertain(self):
        """Case F: Helmet clearly detected + face region uncertain -> helmet=YES, mask=UNKNOWN, overall=UNCERTAIN."""
        # Small person box where face ROI height < min_face_height_px (8px)
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 100.0, 215.0, 120.0]}, # Person: w=15, h=20 (face h = 6.4px < 8px)
            {'cls': 0, 'conf': 0.85, 'box': [200.0, 95.0, 215.0, 108.0]}   # Helmet associated
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].helmet_detected, "YES")
        self.assertEqual(states[0].mask_detected, "UNKNOWN")
        self.assertEqual(states[0].safety_status, "UNCERTAIN")
        self.assertEqual(states[0].uncertain_reason, "MASK_REGION_UNOBSERVABLE")

    def test_case_g_helmet_and_mask_body_cropped(self):
        """Case G: Helmet clearly detected + mask clearly detected + body cropped -> SAFE."""
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 100.0, 400.0, 1078.0]}, # Bottom touching image bottom
            {'cls': 0, 'conf': 0.85, 'box': [220.0, 80.0, 380.0, 180.0]},   # Helmet
            {'cls': 1, 'conf': 0.80, 'box': [260.0, 140.0, 340.0, 220.0]}   # Mask
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].helmet_detected, "YES")
        self.assertEqual(states[0].mask_detected, "YES")
        self.assertEqual(states[0].safety_status, "SAFE")

    def test_case_h_bottom_cropped_visible_head_face(self):
        """Case H: Bottom cropped but head/face visible -> normal PPE evaluation."""
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 100.0, 400.0, 1078.0]}, # Bottom touching image bottom
            {'cls': 0, 'conf': 0.85, 'box': [220.0, 80.0, 380.0, 180.0]}    # Helmet present, mask absent
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].helmet_detected, "YES")
        self.assertEqual(states[0].mask_detected, "NO")
        self.assertEqual(states[0].safety_status, "NO_MASK")

    def test_case_i_helmet_candidate_outside_head_roi(self):
        """Case I: Helmet candidate detected far away -> not associated, evaluates observability."""
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 200.0, 400.0, 800.0]},  # Person at x=200..400
            {'cls': 0, 'conf': 0.85, 'box': [1000.0, 200.0, 1150.0, 300.0]} # Helmet far away at x=1000
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].helmet_detected, "NO")
        self.assertEqual(states[0].mask_detected, "NO")
        self.assertEqual(states[0].safety_status, "NO_HELMET_AND_MASK")


if __name__ == '__main__':
    unittest.main()
