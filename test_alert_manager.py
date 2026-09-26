"""
Unit Tests for AlertManager (Phase 4)
======================================
Tests AlertManager functionality:
1. First confirmed violation -> alert emitted
2. Same violation immediately again -> suppressed by cooldown
3. Same violation after cooldown -> emitted
4. Different violation type / worker track -> emitted independently
5. SAFE state -> suppressed (not a supported violation)
6. UNCERTAIN state -> suppressed (not a supported violation)
"""

import sys
import os
import unittest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.safety.alert_manager import AlertManager, AlertManagerConfig


class TestAlertManager(unittest.TestCase):

    def setUp(self):
        self.config = AlertManagerConfig(cooldown_seconds=5.0)
        self.alert_manager = AlertManager(self.config)

    def test_1_first_confirmed_violation_emits_alert(self):
        """First confirmed violation for a track should emit an alert."""
        res = self.alert_manager.should_emit_alert(track_id=1, violation_type="NO_HELMET", timestamp_sec=10.0)
        self.assertTrue(res, "First confirmed violation MUST emit an alert.")

    def test_2_same_violation_immediately_again_is_suppressed(self):
        """Same violation immediately after (e.g., +0.5s) should be suppressed by cooldown."""
        self.alert_manager.should_emit_alert(track_id=1, violation_type="NO_HELMET", timestamp_sec=10.0)
        res_immediate = self.alert_manager.should_emit_alert(track_id=1, violation_type="NO_HELMET", timestamp_sec=10.5)
        self.assertFalse(res_immediate, "Immediate duplicate violation within cooldown MUST be suppressed.")

    def test_3_same_violation_after_cooldown_emits_alert(self):
        """Same violation after cooldown elapsed (e.g., +5.1s) should emit a new alert."""
        self.alert_manager.should_emit_alert(track_id=1, violation_type="NO_HELMET", timestamp_sec=10.0)
        res_after_cooldown = self.alert_manager.should_emit_alert(track_id=1, violation_type="NO_HELMET", timestamp_sec=15.1)
        self.assertTrue(res_after_cooldown, "Violation after 5.0s cooldown MUST emit an alert.")

    def test_4_different_violation_type_or_worker_handled_correctly(self):
        """Different violation type or worker track ID should be tracked independently."""
        # Worker 1: NO_HELMET
        res_w1 = self.alert_manager.should_emit_alert(track_id=1, violation_type="NO_HELMET", timestamp_sec=10.0)
        self.assertTrue(res_w1)

        # Worker 2: NO_HELMET (different worker, same time)
        res_w2 = self.alert_manager.should_emit_alert(track_id=2, violation_type="NO_HELMET", timestamp_sec=10.0)
        self.assertTrue(res_w2, "Alert for Worker 2 MUST emit independently of Worker 1.")

        # Worker 1: NO_MASK (different violation type, same worker)
        res_w1_mask = self.alert_manager.should_emit_alert(track_id=1, violation_type="NO_MASK", timestamp_sec=10.1)
        self.assertTrue(res_w1_mask, "Different violation type for same worker MUST emit independently.")

    def test_5_safe_state_does_not_generate_alert(self):
        """SAFE state is not a violation and MUST NOT generate an alert."""
        res = self.alert_manager.should_emit_alert(track_id=1, violation_type="SAFE", timestamp_sec=10.0)
        self.assertFalse(res, "SAFE status MUST NOT emit an alert.")

    def test_6_uncertain_state_does_not_generate_alert(self):
        """UNCERTAIN state is not a confirmed violation and MUST NOT generate an alert."""
        res = self.alert_manager.should_emit_alert(track_id=1, violation_type="UNCERTAIN", timestamp_sec=10.0)
        self.assertFalse(res, "UNCERTAIN status MUST NOT emit an alert.")


if __name__ == "__main__":
    unittest.main()
