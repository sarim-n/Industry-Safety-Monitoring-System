"""
Temporal Confirmation Module for Industrial Safety Monitoring System
======================================================================
Provides lightweight, deterministic, IoU-based spatial tracking and
temporal violation confirmation with streak logic and cooldown control.

Pipeline position:
  YOLO -> Confidence Thresholds -> PPE Association -> TEMPORAL CONFIRMATION -> Event
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from src.safety.ppe_association import PersonPPEState, BBox, box_iou


@dataclass
class TemporalConfirmationConfig:
    """Configurable parameters for temporal tracking and confirmation."""
    confirmation_frames: int = 5         # Consecutive violation observations required
    min_track_iou: float = 0.30          # Minimum IoU to match person bbox across frames
    max_missed_frames: int = 10          # Max frames to retain track if person is missing
    fps: float = 30.0                    # Assumed frame rate for time calculations
    alert_cooldown_seconds: float = 5.0  # Minimum seconds between repeated alerts for same track
    uncertain_breaks_streak: bool = True # Conservative rule: UNCERTAIN breaks violation streak


@dataclass
class ConfirmedViolationEvent:
    track_id: int
    violation: str                       # "NO_HELMET", "NO_MASK", "NO_HELMET_AND_MASK"
    confirmed: bool
    frame_index: int
    timestamp_sec: float
    person_bbox: Tuple[float, float, float, float]
    consecutive_streak: int


@dataclass
class TemporaryWorkerTrack:
    track_id: int
    last_bbox: Tuple[float, float, float, float]
    last_frame_index: int
    missed_frames: int = 0

    # Temporal violation streak state
    current_violation: Optional[str] = None
    consecutive_violation_count: int = 0

    # Cooldown tracking
    last_alert_frame: Optional[int] = None
    last_alert_timestamp: Optional[float] = None


class TemporalConfirmationEngine:
    """Engine for tracking temporary worker bboxes and confirming violations across time."""

    def __init__(self, config: Optional[TemporalConfirmationConfig] = None):
        self.config = config or TemporalConfirmationConfig()
        self.next_track_id: int = 1
        self.active_tracks: List[TemporaryWorkerTrack] = []

    def reset(self) -> None:
        """Reset all active tracks and reset track ID counter."""
        self.next_track_id = 1
        self.active_tracks.clear()

    def process_frame(
        self,
        person_states: List[PersonPPEState],
        frame_index: int,
        timestamp_sec: Optional[float] = None
    ) -> List[ConfirmedViolationEvent]:
        """
        Process a single frame's PPE association outputs.

        Args:
            person_states: List of PersonPPEState outputs from PPEAssociator.
            frame_index: 0-indexed integer frame counter.
            timestamp_sec: Optional timestamp in seconds (calculated from FPS if None).

        Returns:
            List of ConfirmedViolationEvent objects emitted in this frame.
        """
        if timestamp_sec is None:
            timestamp_sec = frame_index / self.config.fps

        events: List[ConfirmedViolationEvent] = []

        # 1. Match PersonPPEStates against Active Tracks using IoU
        # Candidate matches: (iou, track_idx, state_idx)
        candidates = []
        for t_idx, track in enumerate(self.active_tracks):
            for s_idx, state in enumerate(person_states):
                iou = box_iou(list(track.last_bbox), list(state.person_bbox))
                if iou >= self.config.min_track_iou:
                    candidates.append((iou, t_idx, s_idx))

        # Sort candidate matches by IoU descending for greedy deterministic matching
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_tracks = set()
        matched_states = set()
        track_state_pairs: List[Tuple[TemporaryWorkerTrack, PersonPPEState]] = []

        for iou, t_idx, s_idx in candidates:
            if t_idx not in matched_tracks and s_idx not in matched_states:
                matched_tracks.add(t_idx)
                matched_states.add(s_idx)
                track_state_pairs.append((self.active_tracks[t_idx], person_states[s_idx]))

        # 2. Update Unmatched Active Tracks (Increase missed_frames)
        surviving_tracks = []
        for t_idx, track in enumerate(self.active_tracks):
            if t_idx not in matched_tracks:
                track.missed_frames += 1
                if track.missed_frames <= self.config.max_missed_frames:
                    surviving_tracks.append(track)
            else:
                surviving_tracks.append(track)

        self.active_tracks = surviving_tracks

        # 3. Create New Tracks for Unmatched Detections
        for s_idx, state in enumerate(person_states):
            if s_idx not in matched_states:
                new_track = TemporaryWorkerTrack(
                    track_id=self.next_track_id,
                    last_bbox=state.person_bbox,
                    last_frame_index=frame_index,
                    missed_frames=0
                )
                self.next_track_id += 1
                self.active_tracks.append(new_track)
                track_state_pairs.append((new_track, state))

        # 4. Process Matched Tracks and Evaluate Temporal Confirmation
        for track, state in track_state_pairs:
            track.last_bbox = state.person_bbox
            track.last_frame_index = frame_index
            track.missed_frames = 0

            status = state.safety_status
            eligible_violations = {"NO_HELMET", "NO_MASK", "NO_HELMET_AND_MASK"}

            if status == "SAFE":
                # SAFE breaks active violation streak
                track.current_violation = None
                track.consecutive_violation_count = 0

            elif status == "UNCERTAIN":
                # Conservative rule: UNCERTAIN breaks consecutive streak
                if self.config.uncertain_breaks_streak:
                    track.current_violation = None
                    track.consecutive_violation_count = 0

            elif status in eligible_violations:
                if track.current_violation == status:
                    track.consecutive_violation_count += 1
                else:
                    # Violation type changed or new violation starting
                    track.current_violation = status
                    track.consecutive_violation_count = 1

                # Check if confirmation streak threshold reached
                if track.consecutive_violation_count >= self.config.confirmation_frames:
                    # Check Cooldown
                    in_cooldown = False
                    if track.last_alert_timestamp is not None:
                        elapsed_time = timestamp_sec - track.last_alert_timestamp
                        if elapsed_time < self.config.alert_cooldown_seconds:
                            in_cooldown = True

                    if not in_cooldown:
                        track.last_alert_frame = frame_index
                        track.last_alert_timestamp = timestamp_sec

                        events.append(ConfirmedViolationEvent(
                            track_id=track.track_id,
                            violation=status,
                            confirmed=True,
                            frame_index=frame_index,
                            timestamp_sec=timestamp_sec,
                            person_bbox=state.person_bbox,
                            consecutive_streak=track.consecutive_violation_count
                        ))

        return events
