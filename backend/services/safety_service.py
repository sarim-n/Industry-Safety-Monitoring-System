"""
Safety Service Module
======================
Manages event logs, CSV parsing, in-memory live state tracking, aggregate statistics calculation,
and thread-safe live stream frame buffering.
"""

import os
import csv
import time
import threading
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from backend.models import (
    EventModel,
    WorkerStateModel,
    SystemStatusResponse,
    StatisticsResponse,
    TelemetryUpdate
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVENTS_CSV_PATH = PROJECT_ROOT / "reports" / "alerts_v1" / "events.csv"


def generate_offline_placeholder_jpg(width: int = 640, height: int = 360) -> bytes:
    """Generate a clean dark-theme 'VIDEO STREAM OFFLINE' JPEG image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (25, 15, 11)  # Dark slate background (BGR)

    # Draw border
    cv2.rectangle(img, (10, 10), (width - 10, height - 10), (51, 65, 85), 1)

    # Text strings
    text1 = "VIDEO STREAM OFFLINE"
    text2 = "Start: python run_live.py --source 0 --enable-api"

    (w1, _), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    (w2, _), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)

    cx, cy = width // 2, height // 2

    # Draw status dot
    cv2.circle(img, (cx - w1 // 2 - 15, cy - 14), 5, (0, 0, 220), -1)

    cv2.putText(img, text1, (cx - w1 // 2 + 5, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (248, 250, 252), 2, cv2.LINE_AA)
    cv2.putText(img, text2, (cx - w2 // 2, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)

    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return encoded.tobytes()



class SafetyService:
    """Service handling state reading, event log parsing, statistics aggregation, and live frame buffering."""

    def __init__(self, csv_path: Path = EVENTS_CSV_PATH):
        self.csv_path = Path(csv_path)
        self.in_memory_workers: Dict[int, str] = {}
        self.system_running: bool = False
        self.last_event_in_memory: Optional[EventModel] = None

        # Thread-safe Live Stream Frame Buffer
        self._frame_lock = threading.Lock()
        self.latest_frame_bytes: Optional[bytes] = None
        self.last_frame_timestamp: float = 0.0

    def update_frame(self, jpeg_bytes: bytes) -> None:
        """Store the latest JPEG frame received from run_live.py."""
        with self._frame_lock:
            self.latest_frame_bytes = jpeg_bytes
            self.last_frame_timestamp = time.time()
            self.system_running = True

    def get_latest_frame(self) -> Optional[bytes]:
        """Retrieve the latest available JPEG frame."""
        with self._frame_lock:
            return self.latest_frame_bytes

    def is_stream_active(self, timeout_sec: float = 3.0) -> bool:
        """Check if live stream frames have been received recently."""
        with self._frame_lock:
            if self.latest_frame_bytes is None:
                return False
            return (time.time() - self.last_frame_timestamp) < timeout_sec

    def read_all_events(self) -> List[EventModel]:
        """Read all events from events.csv file."""
        events: List[EventModel] = []

        if not self.csv_path.exists():
            return events

        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row:
                        continue
                    try:
                        worker_id = int(row.get("worker_id", 0))
                        events.append(
                            EventModel(
                                timestamp=row.get("timestamp", ""),
                                worker_id=worker_id,
                                violation_type=row.get("violation_type", ""),
                                evidence_path=row.get("evidence_path", "")
                            )
                        )
                    except (ValueError, KeyError):
                        continue
        except Exception as e:
            print(f"[SafetyService] Warning reading CSV events: {e}")

        return events

    def get_events(self, limit: int = 20, offset: int = 0) -> List[EventModel]:
        """Get paginated events (most recent first)."""
        all_events = self.read_all_events()
        # Return most recent first
        all_events.reverse()
        return all_events[offset : offset + limit]

    def get_statistics(self) -> StatisticsResponse:
        """Calculate aggregate safety statistics from event records."""
        events = self.read_all_events()

        total_events = len(events)
        no_helmet = sum(1 for e in events if e.violation_type == "NO_HELMET")
        no_mask = sum(1 for e in events if e.violation_type == "NO_MASK")
        no_helmet_and_mask = sum(1 for e in events if e.violation_type == "NO_HELMET_AND_MASK")

        unique_workers_set = {e.worker_id for e in events}
        unique_workers_set.update(self.in_memory_workers.keys())

        latest_timestamp = events[-1].timestamp if events else None

        return StatisticsResponse(
            total_events=total_events,
            no_helmet_count=no_helmet,
            no_mask_count=no_mask,
            no_helmet_and_mask_count=no_helmet_and_mask,
            unique_workers=len(unique_workers_set),
            latest_event_timestamp=latest_timestamp
        )

    def get_system_status(self) -> SystemStatusResponse:
        """Return system status including active workers count and last event."""
        events = self.read_all_events()
        last_event = self.last_event_in_memory or (events[-1] if events else None)
        active_stream = self.is_stream_active(timeout_sec=3.0)

        return SystemStatusResponse(
            system_running=self.system_running or active_stream or bool(self.in_memory_workers),
            active_workers=len(self.in_memory_workers),
            confirmed_violations=len(events),
            last_event=last_event
        )

    def get_workers(self) -> List[WorkerStateModel]:
        """Return current in-memory worker safety states."""
        return [
            WorkerStateModel(worker_id=wid, status=status)
            for wid, status in self.in_memory_workers.items()
        ]

    def update_telemetry(self, update: TelemetryUpdate) -> None:
        """Update live in-memory telemetry state from processing pipeline."""
        self.system_running = update.system_running
        self.in_memory_workers = {w.worker_id: w.status for w in update.active_workers}
        if update.latest_event:
            self.last_event_in_memory = update.latest_event
