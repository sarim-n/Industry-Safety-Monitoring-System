"""
API Bridge Module
==================
Non-blocking HTTP bridge pushing live safety monitoring state from run_live.py
to the FastAPI backend when available.
"""

import threading
import queue
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class NonBlockingAPIBridge:
    """
    Asynchronously pushes telemetry updates to FastAPI backend without blocking video processing.
    """

    def __init__(self, api_url: str = "http://127.0.0.1:8000/api/telemetry", enabled: bool = True):
        self.api_url = api_url
        self.enabled = enabled
        self.queue: queue.Queue = queue.Queue(maxsize=10)
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None

        if self.enabled:
            self._start_worker()

    def _worker_loop(self):
        try:
            import requests
        except ImportError:
            logger.warning("[API Bridge] 'requests' library not installed. Disabling API bridge.")
            return

        session = requests.Session()

        while self.running:
            try:
                payload = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if payload is None:
                break

            try:
                session.post(self.api_url, json=payload, timeout=0.5)
            except Exception:
                # Silently catch network errors if backend server is offline
                pass

            self.queue.task_done()

    def _start_worker(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def push_update(self, active_workers: List[Dict[str, Any]], latest_event: Optional[Dict[str, Any]] = None):
        """Enqueue telemetry payload for background transmission."""
        if not self.enabled or not self.running:
            return

        payload = {
            "system_running": True,
            "active_workers": active_workers,
            "latest_event": latest_event
        }

        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            # Drop older update if queue is full to prevent latency build-up
            pass

    def stop(self):
        """Cleanly stop worker thread."""
        if self.running:
            self.running = False
            try:
                self.queue.put_nowait(None)
            except Exception:
                pass
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=0.5)
