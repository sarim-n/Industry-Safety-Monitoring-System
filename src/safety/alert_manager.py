"""
Alert Manager Module for Industrial Safety Monitoring System
============================================================
Handles confirmed violation alert evaluation, enforcing cooldowns
and preventing duplicate alerts per track.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Set


@dataclass
class AlertManagerConfig:
    """Configurable parameters for alert evaluation."""
    cooldown_seconds: float = 5.0
    supported_violations: Tuple[str, ...] = ("NO_HELMET", "NO_MASK", "NO_HELMET_AND_MASK")


class AlertManager:
    """Evaluates whether confirmed violations qualify for alert emission."""

    def __init__(self, config: Optional[AlertManagerConfig] = None):
        self.config = config or AlertManagerConfig()
        # Storage: (track_id, violation_type) -> last_alert_timestamp
        self.last_alert_timestamps: Dict[Tuple[int, str], float] = {}

    def should_emit_alert(self, track_id: int, violation_type: str, timestamp_sec: float) -> bool:
        """
        Determine if an alert should be emitted for a confirmed violation event.

        Args:
            track_id: Temporary Track ID of the worker.
            violation_type: "NO_HELMET", "NO_MASK", or "NO_HELMET_AND_MASK".
            timestamp_sec: Current timestamp in seconds.

        Returns:
            True if alert should be emitted, False if suppressed or invalid.
        """
        if violation_type not in self.config.supported_violations:
            return False

        key = (track_id, violation_type)
        if key in self.last_alert_timestamps:
            last_ts = self.last_alert_timestamps[key]
            elapsed = timestamp_sec - last_ts
            if elapsed < self.config.cooldown_seconds:
                return False

        # Record alert emission
        self.last_alert_timestamps[key] = timestamp_sec
        return True

    def reset(self) -> None:
        """Reset all recorded alert timestamps."""
        self.last_alert_timestamps.clear()
