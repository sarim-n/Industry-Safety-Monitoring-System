"""
Evidence Capture and Event Logging Module
==========================================
Saves annotated evidence snapshot frames and records structured event logs
to CSV for confirmed safety violations.
"""

import os
import csv
import cv2
from datetime import datetime
from pathlib import Path
from typing import Optional


class EvidenceManager:
    """
    Manages saving evidence snapshot images and appending events to CSV log.
    """

    def __init__(
        self,
        evidence_dir: str = "evidence",
        csv_path: str = "reports/alerts_v1/events.csv"
    ):
        self.evidence_dir = Path(evidence_dir)
        self.csv_path = Path(csv_path)

        # Create directories if needed
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize CSV log with header if not existing
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "worker_id", "violation_type", "evidence_path"])

    def capture_evidence(
        self,
        frame: cv2.Mat,
        worker_id: int,
        violation_type: str,
        system_datetime: Optional[datetime] = None
    ) -> str:
        """
        Save evidence frame and log event to CSV.

        Args:
            frame: OpenCV BGR image frame (annotated).
            worker_id: Temporary Track ID of the worker.
            violation_type: Confirmed violation type ("NO_HELMET", "NO_MASK", "NO_HELMET_AND_MASK").
            system_datetime: Optional datetime object (defaults to datetime.now()).

        Returns:
            Relative path to the saved evidence image.
        """
        now = system_datetime or datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%HH%MM%SS")
        iso_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        filename = f"{timestamp_str}_{violation_type}_worker_{worker_id}.jpg"
        img_path = self.evidence_dir / filename

        # Handle duplicate filenames in the same second by appending counter
        counter = 1
        while img_path.exists():
            filename = f"{timestamp_str}_{violation_type}_worker_{worker_id}_{counter}.jpg"
            img_path = self.evidence_dir / filename
            counter += 1

        # Save image frame
        cv2.imwrite(str(img_path), frame)

        # Format relative path for CSV
        rel_evidence_path = str(img_path).replace("\\", "/")

        # Append row to events.csv
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([iso_timestamp, worker_id, violation_type, rel_evidence_path])

        return rel_evidence_path
