"""
Voice Alert Module for Industrial Safety Monitoring System
===========================================================
Provides offline, non-blocking text-to-speech (TTS) voice alerts
for confirmed safety violations.
"""

import threading
import queue
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Violation message mapping
VIOLATION_MESSAGES: Dict[str, str] = {
    "NO_HELMET": "Warning! Please wear your helmet.",
    "NO_MASK": "Warning! Please wear your mask.",
    "NO_HELMET_AND_MASK": "Warning! Please wear your helmet and mask.",
}


class VoiceAlertEngine:
    """
    Offline non-blocking text-to-speech voice alert engine.
    Uses a background worker thread and queue so the main frame-processing
    loop is never blocked by audio playback.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.speech_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        self._engine = None

        if self.enabled:
            self._start_worker()

    def _init_tts_engine(self):
        """Initialize local pyttsx3 or SAPI5 engine inside the worker thread."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            return engine
        except Exception as e:
            logger.warning(f"VoiceAlertEngine TTS initialization warning/fallback: {e}")
            return None

    def _worker_loop(self):
        """Background thread loop to consume and speak alert messages."""
        engine = self._init_tts_engine()

        while self.running:
            try:
                message = self.speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if message is None:
                break

            if engine is not None:
                try:
                    engine.say(message)
                    engine.runAndWait()
                except Exception as e:
                    logger.warning(f"Error during TTS playback: {e}")
            else:
                # Fallback print if TTS engine is unavailable (e.g., headless CI environment)
                logger.info(f"[VOICE ALERT fallback]: {message}")

            self.speech_queue.task_done()

        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def _start_worker(self):
        """Start the background speech worker thread."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def speak(self, violation_type: str) -> bool:
        """
        Enqueue a voice alert message for the given violation type.

        Args:
            violation_type: "NO_HELMET", "NO_MASK", or "NO_HELMET_AND_MASK".

        Returns:
            True if message was enqueued, False otherwise.
        """
        if not self.enabled:
            return False

        message = VIOLATION_MESSAGES.get(violation_type)
        if not message:
            return False

        self.speech_queue.put(message)
        return True

    def stop(self):
        """Cleanly terminate the background speech worker thread."""
        if self.running:
            self.running = False
            self.speech_queue.put(None)
            if self.worker_thread is not None and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=1.0)
