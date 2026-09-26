"""
Safety Service Module
======================
Manages event logs, CSV parsing, in-memory live state tracking, and aggregate statistics calculation.
"""

import os
import csv
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


class SafetyService:
    """Service handling state reading, event log parsing, and statistics aggregation."""

    def __init__(self, csv_path: Path = EVENTS_CSV_PATH):
        self.csv_path = Path(csv_path)
        self.in_memory_workers: Dict[int, str] = {}
        self.system_running: bool = False
        self.last_event_in_memory: Optional[EventModel] = None

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
        # Also include active worker IDs from memory if present
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

        return SystemStatusResponse(
            system_running=self.system_running or bool(self.in_memory_workers),
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
