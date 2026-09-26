"""
Synthetic Unit & Integration Tests for Temporal Confirmation Module
===================================================================
Tests all 11 required temporal confirmation scenarios:
1. Persistent NO_HELMET (5 frames) -> 1 event
2. Intermittent violation (NO_HELMET, SAFE, NO_HELMET) -> 0 events
3. Persistent NO_MASK (5 frames) -> 1 event
4. Persistent combined violation (NO_HELMET_AND_MASK x 5) -> 1 event
5. UNKNOWN/UNCERTAIN interruption -> 0 events
6. Violation transition (NO_HELMET -> NO_MASK) -> Streak resets
7. Person matching (High IoU) -> Same track ID
8. Person separation (Low IoU) -> Different track IDs
9. Temporary disappearance (<= 10 frames) -> Track retained
10. Long disappearance (> 10 frames) -> Track removed
11. Cooldown -> Suppressed repeated events
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.safety.ppe_association import PersonPPEState
from src.safety.temporal_confirmation import (
    TemporalConfirmationConfig,
    ConfirmedViolationEvent,
    TemporalConfirmationEngine
)


def helper_make_state(
    bbox=(100.0, 100.0, 200.0, 300.0),
    safety_status="SAFE",
    helmet_detected="YES",
    mask_detected="YES",
    person_idx=1
) -> PersonPPEState:
    return PersonPPEState(
        person_index=person_idx,
        person_bbox=bbox,
        person_confidence=0.85,
        touches_left_boundary=False,
        touches_right_boundary=False,
        touches_top_boundary=False,
        touches_bottom_boundary=False,
        head_visibility="VISIBLE",
        face_visibility="VISIBLE",
        helmet_detected=helmet_detected,
        mask_detected=mask_detected,
        safety_status=safety_status
    )


class TestTemporalConfirmationEngine(unittest.TestCase):

    def setUp(self):
        self.config = TemporalConfirmationConfig(
            confirmation_frames=5,
            min_track_iou=0.30,
            max_missed_frames=10,
            fps=30.0,
            alert_cooldown_seconds=5.0,
            uncertain_breaks_streak=True
        )
        self.engine = TemporalConfirmationEngine(self.config)

    def test_01_persistent_no_helmet(self):
        """Scenario 1: NO_HELMET x 5 -> Exactly 1 confirmed event."""
        all_events = []
        for frame in range(5):
            state = helper_make_state(safety_status="NO_HELMET", helmet_detected="NO", mask_detected="YES")
            events = self.engine.process_frame([state], frame_index=frame)
            all_events.extend(events)

        self.assertEqual(len(all_events), 1)
        self.assertEqual(all_events[0].violation, "NO_HELMET")
        self.assertTrue(all_events[0].confirmed)
        self.assertEqual(all_events[0].consecutive_streak, 5)

    def test_02_intermittent_violation(self):
        """Scenario 2: NO_HELMET, NO_HELMET, SAFE, NO_HELMET, NO_HELMET -> 0 events."""
        sequence = ["NO_HELMET", "NO_HELMET", "SAFE", "NO_HELMET", "NO_HELMET"]
        all_events = []
        for frame, status in enumerate(sequence):
            state = helper_make_state(safety_status=status)
            events = self.engine.process_frame([state], frame_index=frame)
            all_events.extend(events)

        self.assertEqual(len(all_events), 0)

    def test_03_persistent_no_mask(self):
        """Scenario 3: NO_MASK x 5 -> Exactly 1 confirmed event."""
        all_events = []
        for frame in range(5):
            state = helper_make_state(safety_status="NO_MASK", helmet_detected="YES", mask_detected="NO")
            events = self.engine.process_frame([state], frame_index=frame)
            all_events.extend(events)

        self.assertEqual(len(all_events), 1)
        self.assertEqual(all_events[0].violation, "NO_MASK")

    def test_04_persistent_combined_violation(self):
        """Scenario 4: NO_HELMET_AND_MASK x 5 -> Exactly 1 confirmed event."""
        all_events = []
        for frame in range(5):
            state = helper_make_state(safety_status="NO_HELMET_AND_MASK", helmet_detected="NO", mask_detected="NO")
            events = self.engine.process_frame([state], frame_index=frame)
            all_events.extend(events)

        self.assertEqual(len(all_events), 1)
        self.assertEqual(all_events[0].violation, "NO_HELMET_AND_MASK")

    def test_05_uncertain_interruption(self):
        """Scenario 5: NO_HELMET, NO_HELMET, UNCERTAIN, NO_HELMET, NO_HELMET -> 0 events."""
        sequence = ["NO_HELMET", "NO_HELMET", "UNCERTAIN", "NO_HELMET", "NO_HELMET"]
        all_events = []
        for frame, status in enumerate(sequence):
            state = helper_make_state(safety_status=status)
            events = self.engine.process_frame([state], frame_index=frame)
            all_events.extend(events)

        self.assertEqual(len(all_events), 0)

    def test_06_violation_transition(self):
        """Scenario 6: NO_HELMET x 3, NO_MASK x 2 -> Streak resets on transition, 0 events."""
        sequence = ["NO_HELMET", "NO_HELMET", "NO_HELMET", "NO_MASK", "NO_MASK"]
        all_events = []
        for frame, status in enumerate(sequence):
            state = helper_make_state(safety_status=status)
            events = self.engine.process_frame([state], frame_index=frame)
            all_events.extend(events)

        self.assertEqual(len(all_events), 0)

    def test_07_person_matching(self):
        """Scenario 7: High IoU consecutive boxes -> Same Track ID."""
        state1 = helper_make_state(bbox=(100.0, 100.0, 200.0, 300.0))
        state2 = helper_make_state(bbox=(102.0, 101.0, 202.0, 301.0))

        self.engine.process_frame([state1], frame_index=0)
        self.assertEqual(self.engine.active_tracks[0].track_id, 1)

        self.engine.process_frame([state2], frame_index=1)
        self.assertEqual(len(self.engine.active_tracks), 1)
        self.assertEqual(self.engine.active_tracks[0].track_id, 1)

    def test_08_person_separation(self):
        """Scenario 8: Low/Zero IoU boxes -> Different Track IDs."""
        state1 = helper_make_state(bbox=(100.0, 100.0, 200.0, 300.0))
        state2 = helper_make_state(bbox=(500.0, 500.0, 600.0, 700.0))

        self.engine.process_frame([state1], frame_index=0)
        t1_id = self.engine.active_tracks[0].track_id

        self.engine.process_frame([state2], frame_index=1)
        t2_id = self.engine.active_tracks[-1].track_id

        self.assertNotEqual(t1_id, t2_id)

    def test_09_temporary_disappearance(self):
        """Scenario 9: Person disappears for 5 frames (<= 10) -> Track retained."""
        state1 = helper_make_state(bbox=(100.0, 100.0, 200.0, 300.0))
        self.engine.process_frame([state1], frame_index=0)

        # Disappear for 5 frames
        for frame in range(1, 6):
            self.engine.process_frame([], frame_index=frame)

        self.assertEqual(len(self.engine.active_tracks), 1)
        self.assertEqual(self.engine.active_tracks[0].missed_frames, 5)

        # Reappear
        self.engine.process_frame([state1], frame_index=6)
        self.assertEqual(self.engine.active_tracks[0].missed_frames, 0)

    def test_10_long_disappearance(self):
        """Scenario 10: Person disappears for 12 frames (> 10) -> Track removed."""
        state1 = helper_make_state(bbox=(100.0, 100.0, 200.0, 300.0))
        self.engine.process_frame([state1], frame_index=0)

        # Disappear for 12 frames
        for frame in range(1, 13):
            self.engine.process_frame([], frame_index=frame)

        self.assertEqual(len(self.engine.active_tracks), 0)

    def test_11_cooldown(self):
        """Scenario 11: Repeated NO_HELMET during cooldown -> Second event suppressed."""
        all_events = []
        # Frame 0..4 (5 frames) -> Confirms alert at frame 4 (t = 0.133s)
        for frame in range(5):
            state = helper_make_state(safety_status="NO_HELMET")
            events = self.engine.process_frame([state], frame_index=frame, timestamp_sec=frame / 30.0)
            all_events.extend(events)

        self.assertEqual(len(all_events), 1)

        # Frame 5..15 (within 5 sec cooldown) -> Extra events must be suppressed!
        for frame in range(5, 16):
            state = helper_make_state(safety_status="NO_HELMET")
            events = self.engine.process_frame([state], frame_index=frame, timestamp_sec=frame / 30.0)
            all_events.extend(events)

        self.assertEqual(len(all_events), 1)  # Still only 1 event!


if __name__ == '__main__':
    unittest.main()
