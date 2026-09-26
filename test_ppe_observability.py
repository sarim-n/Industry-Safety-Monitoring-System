"""
Regression Unit Tests for PPE Observability (Phase 7.5)
=========================================================
Tests geometry-based PPE observability rules for partial/face-only person detections:
- Case A: Full person visible, helmet + mask visible -> SAFE
- Case B: Full person visible, helmet absent, mask visible -> NO_HELMET
- Case C: Full person visible, helmet visible, mask absent -> NO_MASK
- Case D: Full person visible, neither PPE detected -> NO_HELMET_AND_MASK
- Case E: Face-only crop, helmet/mask unobservable -> UNCERTAIN
- Case F: Top boundary cropped, head region unobservable -> UNCERTAIN
- Case G: Bottom half cropped, head + face clearly visible -> Normal evaluation
- Case H: Face region too small to reliably evaluate mask -> UNCERTAIN
- Case I: Helmet region outside frame (head_y1 < 0) -> UNCERTAIN
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
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 200.0, 400.0, 800.0]},  # Person (w=200, h=600)
            {'cls': 0, 'conf': 0.85, 'box': [220.0, 180.0, 380.0, 280.0]},  # Helmet
            {'cls': 1, 'conf': 0.80, 'box': [260.0, 240.0, 340.0, 320.0]}   # Mask
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].helmet_detected, "YES")
        self.assertEqual(states[0].mask_detected, "YES")
        self.assertEqual(states[0].safety_status, "SAFE")

    def test_case_b_full_person_no_helmet(self):
        """Case B: Full person visible, helmet absent, mask visible -> NO_HELMET."""
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 200.0, 400.0, 800.0]},  # Person
            {'cls': 1, 'conf': 0.80, 'box': [260.0, 240.0, 340.0, 320.0]}   # Mask only
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].head_visibility, "VISIBLE")
        self.assertEqual(states[0].helmet_detected, "NO")
        self.assertEqual(states[0].mask_detected, "YES")
        self.assertEqual(states[0].safety_status, "NO_HELMET")

    def test_case_c_full_person_no_mask(self):
        """Case C: Full person visible, helmet visible, mask absent -> NO_MASK."""
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 200.0, 400.0, 800.0]},  # Person
            {'cls': 0, 'conf': 0.85, 'box': [220.0, 180.0, 380.0, 280.0]}   # Helmet only
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].face_visibility, "VISIBLE")
        self.assertEqual(states[0].helmet_detected, "YES")
        self.assertEqual(states[0].mask_detected, "NO")
        self.assertEqual(states[0].safety_status, "NO_MASK")

    def test_case_d_full_person_neither_ppe(self):
        """Case D: Full person visible, neither PPE detected -> NO_HELMET_AND_MASK."""
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 200.0, 400.0, 800.0]}   # Person only
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].head_visibility, "VISIBLE")
        self.assertEqual(states[0].face_visibility, "VISIBLE")
        self.assertEqual(states[0].helmet_detected, "NO")
        self.assertEqual(states[0].mask_detected, "NO")
        self.assertEqual(states[0].safety_status, "NO_HELMET_AND_MASK")

    def test_case_e_face_only_crop(self):
        """Case E: Only face/head visible (aspect ratio low) -> UNCERTAIN (never NO_HELMET_AND_MASK)."""
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 150.0, 350.0, 280.0]}   # Face crop: w=150, h=130 (ar=0.867)
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].head_visibility, "CROPPED")
        self.assertEqual(states[0].helmet_detected, "UNKNOWN")
        self.assertEqual(states[0].safety_status, "UNCERTAIN")

    def test_case_f_top_boundary_cropped(self):
        """Case F: Person top boundary cropped (y1 <= margin) -> UNCERTAIN."""
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 2.0, 400.0, 600.0]}     # Person touching top edge (y1=2.0)
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].head_visibility, "CROPPED")
        self.assertEqual(states[0].helmet_detected, "UNKNOWN")
        self.assertEqual(states[0].safety_status, "UNCERTAIN")

    def test_case_g_bottom_cropped_visible_head(self):
        """Case G: Bottom half cropped but head + face clearly visible -> Normal evaluation."""
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 150.0, 400.0, 1078.0]}, # Bottom touching image bottom (y2=1078)
            {'cls': 0, 'conf': 0.85, 'box': [220.0, 120.0, 380.0, 220.0]}   # Helmet present
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].head_visibility, "VISIBLE")
        self.assertEqual(states[0].face_visibility, "VISIBLE")
        self.assertEqual(states[0].helmet_detected, "YES")
        self.assertEqual(states[0].mask_detected, "NO")
        self.assertEqual(states[0].safety_status, "NO_MASK")

    def test_case_h_face_too_small(self):
        """Case H: Face region too small to evaluate mask -> mask=UNKNOWN -> UNCERTAIN."""
        # Tiny person box where face ROI height < min_face_height_px (8px)
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 100.0, 215.0, 120.0]}   # Small person: w=15, h=20 (face h = 0.32*20 = 6.4px < 8px)
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].mask_detected, "UNKNOWN")
        self.assertEqual(states[0].safety_status, "UNCERTAIN")

    def test_case_i_helmet_region_outside_frame(self):
        """Case I: Helmet region top extends outside frame (head_y1 < 0) -> helmet=UNKNOWN -> UNCERTAIN."""
        # y1 is 10px from top, height is 240px. top_offset = 25px. y1 - 25 = -15 < 3.0.
        detections = [
            {'cls': 2, 'conf': 0.90, 'box': [200.0, 10.0, 350.0, 250.0]}    # y1=10, h=240, ar=1.6
        ]
        states = self.associator.process_detections(detections, self.img_w, self.img_h)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].head_visibility, "CROPPED")
        self.assertEqual(states[0].helmet_detected, "UNKNOWN")
        self.assertEqual(states[0].safety_status, "UNCERTAIN")


if __name__ == '__main__':
    unittest.main()
