"""
API Bridge Module
==================
Non-blocking HTTP bridge pushing live safety monitoring state and annotated video frames
from run_live.py to the FastAPI backend asynchronously without blocking CV pipeline execution.
"""

import threading
import queue
import logging
import cv2
import numpy as np
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class NonBlockingAPIBridge:
    """
    Asynchronously pushes telemetry updates and live video frames to FastAPI backend
    without blocking video processing.
    """

    def __init__(self, api_url: str = "http://127.0.0.1:8000/api/telemetry", enabled: bool = True):
        self.api_url = api_url
        self.frame_url = api_url.rstrip("/") + "/frame"
        self.enabled = enabled

        self.telemetry_queue: queue.Queue = queue.Queue(maxsize=5)
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)

        self.running = False
        self.telemetry_worker: Optional[threading.Thread] = None
        self.frame_worker: Optional[threading.Thread] = None

        if self.enabled:
            self._start_workers()

    def _telemetry_worker_loop(self):
        try:
            import requests
        except ImportError:
            logger.warning("[API Bridge] 'requests' library not installed. Disabling API bridge.")
            return

        session = requests.Session()

        while self.running:
            try:
                payload = self.telemetry_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if payload is None:
                break

            try:
                session.post(self.api_url, json=payload, timeout=0.4)
            except Exception:
                pass

            self.telemetry_queue.task_done()

    def _frame_worker_loop(self):
        try:
            import requests
        except ImportError:
            return

        session = requests.Session()

        while self.running:
            try:
                frame_bgr = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if frame_bgr is None:
                break

            try:
                # Encode JPEG in background thread
                success, encoded_img = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if success:
                    jpeg_bytes = encoded_img.tobytes()
                    session.post(
                        self.frame_url,
                        data=jpeg_bytes,
                        headers={"Content-Type": "image/jpeg"},
                        timeout=0.4
                    )
            except Exception:
                pass

            self.frame_queue.task_done()

    def _start_workers(self):
        self.running = True
        self.telemetry_worker = threading.Thread(target=self._telemetry_worker_loop, daemon=True)
        self.frame_worker = threading.Thread(target=self._frame_worker_loop, daemon=True)

        self.telemetry_worker.start()
        self.frame_worker.start()

    def push_update(self, active_workers: List[Dict[str, Any]], latest_event: Optional[Dict[str, Any]] = None):
        """Enqueue telemetry JSON payload for background transmission."""
        if not self.enabled or not self.running:
            return

        payload = {
            "system_running": True,
            "active_workers": active_workers,
            "latest_event": latest_event
        }

        try:
            self.telemetry_queue.put_nowait(payload)
        except queue.Full:
            pass

    def push_frame(self, frame_bgr: np.ndarray):
        """Enqueue annotated OpenCV BGR frame for background JPEG encoding and transmission."""
        if not self.enabled or not self.running:
            return

        # If queue is full, drop previous un-sent frame so pipeline never blocks
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.frame_queue.put_nowait(frame_bgr)
        except queue.Full:
            pass

    def stop(self):
        """Cleanly stop background worker threads."""
        if self.running:
            self.running = False
            try:
                self.telemetry_queue.put_nowait(None)
                self.frame_queue.put_nowait(None)
            except Exception:
                pass

            if self.telemetry_worker and self.telemetry_worker.is_alive():
                self.telemetry_worker.join(timeout=0.4)
            if self.frame_worker and self.frame_worker.is_alive():
                self.frame_worker.join(timeout=0.4)
