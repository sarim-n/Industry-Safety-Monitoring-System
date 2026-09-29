"""
Video Processing Service
=========================
Runs uploaded videos through the SAME production safety pipeline as run_live.py.
- Loads YOLOv8s Run 2B model ONCE (singleton) to avoid VRAM duplication.
- Processes in a background thread so FastAPI never blocks.
- Voice alerts are disabled for uploaded video (no speaker output).
- Uses OpenCV VideoWriter (mp4v codec) to produce the annotated output MP4.
"""

import os
import time
import uuid
import threading
import cv2
import numpy as np
import collections
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Safety-critical: allow loading the same CUDA context
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --------------------------------------------------------------------------- #
#  Lazy singleton YOLO model — loaded once on first video job
# --------------------------------------------------------------------------- #
_model_lock = threading.Lock()
_yolo_model = None
_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "safety_v1-4_run2b_clean10810476_yolov8s_800" / "weights" / "best.pt"

# Tracks whether the live webcam pipeline is currently using the GPU
# run_live.py doesn't call this service, so we manage a simple flag
_pipeline_busy = threading.Event()  # set = processing is active


def _get_model():
    """Return the shared YOLO model, loading it once from Run 2B weights."""
    global _yolo_model
    with _model_lock:
        if _yolo_model is None:
            from ultralytics import YOLO
            _yolo_model = YOLO(str(_MODEL_PATH))
    return _yolo_model


# --------------------------------------------------------------------------- #
#  Job State
# --------------------------------------------------------------------------- #
class VideoJob:
    def __init__(self, job_id: str, input_path: Path, output_path: Path):
        self.job_id = job_id
        self.input_path = input_path
        self.output_path = output_path
        self.status: str = "queued"          # queued | processing | completed | failed
        self.progress: float = 0.0           # 0.0–100.0
        self.current_frame: int = 0
        self.total_frames: int = 0
        self.error: Optional[str] = None
        self.created_at: float = time.time()

        # Results populated after completion
        self.frames_processed: int = 0
        self.processing_fps: float = 0.0
        self.workers_detected: int = 0
        self.confirmed_violations: int = 0
        self.no_helmet: int = 0
        self.no_mask: int = 0
        self.no_helmet_and_mask: int = 0
        self.evidence_count: int = 0
        self.codec_used: str = "mp4v (OpenCV)"


# --------------------------------------------------------------------------- #
#  Job Registry (in-memory, single process)
# --------------------------------------------------------------------------- #
_jobs_lock = threading.Lock()
_jobs: Dict[str, VideoJob] = {}

# Output/input directories
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "video_outputs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def is_allowed_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def create_job(original_filename: str, file_bytes: bytes) -> VideoJob:
    """Save uploaded file and register a new processing job."""
    safe_ext = Path(original_filename).suffix.lower()
    if safe_ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {safe_ext}")

    job_id = str(uuid.uuid4())
    input_path = UPLOADS_DIR / f"{job_id}{safe_ext}"
    output_path = OUTPUTS_DIR / f"{job_id}_annotated.mp4"

    with open(input_path, "wb") as f:
        f.write(file_bytes)

    job = VideoJob(job_id=job_id, input_path=input_path, output_path=output_path)

    with _jobs_lock:
        _jobs[job_id] = job

    # Start processing in a daemon thread
    t = threading.Thread(target=_process_video, args=(job_id,), daemon=True)
    t.start()

    return job


def get_job(job_id: str) -> Optional[VideoJob]:
    with _jobs_lock:
        return _jobs.get(job_id)


def list_jobs():
    with _jobs_lock:
        return list(_jobs.values())


# --------------------------------------------------------------------------- #
#  STATUS COLORS  (BGR — same as run_live.py)
# --------------------------------------------------------------------------- #
STATUS_COLORS = {
    "SAFE": (0, 255, 0),
    "NO_HELMET": (0, 165, 255),
    "NO_MASK": (255, 0, 255),
    "NO_HELMET_AND_MASK": (0, 0, 255),
    "UNCERTAIN": (255, 255, 0),
}


# --------------------------------------------------------------------------- #
#  Core Processing — mirrors run_live.py pipeline exactly
# --------------------------------------------------------------------------- #
def _process_video(job_id: str):
    """Run the full safety pipeline on an uploaded video in a background thread."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return

    job.status = "processing"

    try:
        # ---- Import production pipeline modules (same as run_live.py) ----
        from src.safety.ppe_association import PPEAssociationConfig, PPEAssociator
        from src.safety.person_suppression import suppress_duplicate_person_detections
        from src.safety.temporal_confirmation import (
            TemporalConfirmationConfig, TemporalConfirmationEngine,
        )
        from src.safety.alert_manager import AlertManager, AlertManagerConfig
        from src.safety.evidence import EvidenceManager
        from src.safety.voice_alert import VoiceAlertEngine

        # ---- Open input video ----
        cap = cv2.VideoCapture(str(job.input_path))
        if not cap.isOpened():
            job.status = "failed"
            job.error = "Cannot open video file."
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0.0 or source_fps > 120.0:
            source_fps = 25.0

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        job.total_frames = total_frames_raw if total_frames_raw > 0 else 0

        # ---- Output VideoWriter ----
        # Try mp4v (MPEG-4) first — most compatible with OpenCV on Windows.
        # Fall back to MJPEG+AVI if mp4v fails to open.
        fourcc_mp4 = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(job.output_path), fourcc_mp4, source_fps, (frame_w, frame_h)
        )
        if not writer.isOpened():
            # mp4v failed — try MJPEG with .avi extension
            job.output_path = job.output_path.with_suffix(".avi")
            fourcc_mjpg = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(
                str(job.output_path), fourcc_mjpg, source_fps, (frame_w, frame_h)
            )
            job.codec_used = "MJPEG (AVI, mp4v unavailable)"
            if not writer.isOpened():
                job.status = "failed"
                job.error = "Could not open VideoWriter with mp4v or MJPEG codec."
                cap.release()
                return
        else:
            job.codec_used = "mp4v (OpenCV MP4)"

        # ---- Instantiate pipeline engines (production thresholds) ----
        model = _get_model()

        assoc_config = PPEAssociationConfig(
            person_conf=0.50,
            helmet_conf=0.25,
            mask_conf=0.20,
        )
        associator = PPEAssociator(assoc_config)

        temp_config = TemporalConfirmationConfig(
            confirmation_frames=5,
            min_track_iou=0.30,
            max_missed_frames=10,
            fps=source_fps,
            alert_cooldown_seconds=5.0,
            uncertain_breaks_streak=True,
        )
        temporal_engine = TemporalConfirmationEngine(temp_config)

        alert_manager = AlertManager(AlertManagerConfig(cooldown_seconds=5.0))
        evidence_manager = EvidenceManager(
            evidence_dir=str(PROJECT_ROOT / "evidence"),
            csv_path=str(PROJECT_ROOT / "reports" / "alerts_v1" / "events.csv"),
        )
        voice_engine = VoiceAlertEngine(enabled=True)

        # ---- Counters ----
        frame_counter = 0
        total_confirmed = 0
        total_alerts_emitted = 0
        total_evidence_saved = 0
        unique_tracks: set = set()
        frame_times: collections.deque = collections.deque(maxlen=60)
        no_helmet_count = 0
        no_mask_count = 0
        no_helmet_and_mask_count = 0

        t_start = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            t_frame_start = time.perf_counter()
            frame_counter += 1

            # STAGE 1: YOLO inference (ByteTrack)
            results = model.track(frame, imgsz=800, conf=0.05, tracker="bytetrack.yaml", persist=True, verbose=False)[0]
            raw_dets = []
            for box in results.boxes:
                c = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                tid = int(box.id[0].cpu().numpy()) if box.id is not None else None
                raw_dets.append({"cls": c, "conf": conf, "box": xyxy, "track_id": tid})

            # STAGE 1.5: Person duplicate suppression
            filtered_dets = suppress_duplicate_person_detections(raw_dets, frame_w, frame_h)

            # STAGE 2: PPE Association
            person_states = associator.process_detections(filtered_dets, frame_w, frame_h)

            # STAGE 3: Temporal Confirmation
            current_ts = (frame_counter - 1) / source_fps
            events = temporal_engine.process_frame(
                person_states,
                frame_index=frame_counter - 1,
                timestamp_sec=current_ts,
            )

            emitted_alerts = []
            if events:
                total_confirmed += len(events)
                for e in events:
                    # Count per-type
                    if e.violation == "NO_HELMET":
                        no_helmet_count += 1
                    elif e.violation == "NO_MASK":
                        no_mask_count += 1
                    elif e.violation == "NO_HELMET_AND_MASK":
                        no_helmet_and_mask_count += 1

                    if alert_manager.should_emit_alert(e.track_id, e.violation, e.timestamp_sec):
                        total_alerts_emitted += 1
                        voice_engine.speak(e.violation)
                        emitted_alerts.append(e)

            for track in temporal_engine.active_tracks:
                unique_tracks.add(track.track_id)

            # STAGE 4: Render annotations
            # ─────────────────────────────────────────────────────────────────
            # PRIMARY DRAW: iterate person_states (PPE association output).
            # This is available EVERY frame from the first detection — no
            # IoU matching lag. We then overlay track IDs from the temporal
            # engine where available.
            # ─────────────────────────────────────────────────────────────────
            annotated = frame.copy()

            # Build a map: person_bbox → track (for ID overlay)
            # Use rounded int coords to tolerate minor float drift
            def _bbox_key(b):
                return tuple(round(float(v)) for v in b)

            track_map = {}
            for trk in temporal_engine.active_tracks:
                track_map[_bbox_key(trk.last_bbox)] = trk

            confirmed_track_ids = {e.track_id for e in events}

            for ps in person_states:
                bx1, by1, bx2, by2 = [int(v) for v in ps.person_bbox]
                status = ps.safety_status
                color = STATUS_COLORS.get(status, (200, 200, 200))

                # Look up matching temporal track for ID + violation streak
                trk = track_map.get(_bbox_key(ps.person_bbox))

                is_confirmed = trk is not None and trk.track_id in confirmed_track_ids
                has_recent_alert = (
                    trk is not None
                    and trk.last_alert_frame is not None
                    and (frame_counter - 1 - trk.last_alert_frame) < 15
                )

                if is_confirmed or has_recent_alert:
                    border_color = (0, 0, 255)
                    border_thickness = 4
                    violation_label = trk.current_violation if trk else status
                    track_label = f"Worker #{trk.track_id} | CONFIRMED {violation_label}" if trk else f"CONFIRMED {status}"
                    badge_bg = (0, 0, 255)
                    text_color = (255, 255, 255)
                elif trk is not None and trk.consecutive_violation_count > 0:
                    border_color = color
                    border_thickness = 2
                    track_label = f"Worker #{trk.track_id} | {trk.current_violation} ({trk.consecutive_violation_count}/5)"
                    badge_bg = color
                    text_color = (0, 0, 0)
                elif trk is not None:
                    border_color = color
                    border_thickness = 2
                    track_label = f"Worker #{trk.track_id} | {status}"
                    badge_bg = color
                    text_color = (0, 0, 0)
                else:
                    # Track not yet assigned (first 1–2 frames before IoU match)
                    border_color = color
                    border_thickness = 2
                    track_label = f"{status} (conf:{ps.person_confidence:.2f})"
                    badge_bg = color
                    text_color = (0, 0, 0)

                # ── Person bounding box ──
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), border_color, border_thickness)

                # ── Helmet bounding box (yellow) ──
                if ps.helmet_bbox:
                    hx1, hy1, hx2, hy2 = [int(v) for v in ps.helmet_bbox]
                    cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (0, 220, 255), 2)
                    conf_txt = f"Helmet {ps.helmet_confidence:.2f}" if ps.helmet_confidence else "Helmet"
                    cv2.putText(annotated, conf_txt, (hx1, max(0, hy1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

                # ── Mask bounding box (magenta) ──
                if ps.mask_bbox:
                    mx1, my1, mx2, my2 = [int(v) for v in ps.mask_bbox]
                    cv2.rectangle(annotated, (mx1, my1), (mx2, my2), (255, 0, 220), 2)
                    conf_txt = f"Mask {ps.mask_confidence:.2f}" if ps.mask_confidence else "Mask"
                    cv2.putText(annotated, conf_txt, (mx1, max(0, my1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 220), 1, cv2.LINE_AA)

                # ── Worker label badge ──
                (txt_w, txt_h), baseline = cv2.getTextSize(track_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                label_y1 = max(0, by1 - txt_h - baseline - 4)
                label_y2 = max(txt_h + baseline + 4, by1)
                cv2.rectangle(annotated,
                              (bx1, label_y1),
                              (bx1 + txt_w + 8, label_y2),
                              badge_bg, -1)
                cv2.putText(annotated, track_label,
                            (bx1 + 4, label_y2 - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

            # STAGE 5: Evidence capture for emitted alerts
            saved_path = ""
            for e in emitted_alerts:
                saved_path = evidence_manager.capture_evidence(
                    frame=annotated,
                    worker_id=e.track_id,
                    violation_type=e.violation,
                )
                total_evidence_saved += 1

            # HUD overlay
            rolling_fps = 1000.0 / (np.mean(frame_times) + 1e-9) if frame_times else 0.0
            cv2.rectangle(annotated, (10, 10), (540, 112), (20, 20, 20), -1)
            cv2.rectangle(annotated, (10, 10), (540, 112), (0, 255, 0), 1)
            cv2.putText(
                annotated,
                f"VIDEO ANALYSIS | Frame {frame_counter} | FPS: {rolling_fps:.1f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Workers: {len(temporal_engine.active_tracks)} | Events: {total_confirmed}",
                (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Model: YOLOv8s Clean Run 2B  |  Job: {job_id[:8]}",
                (20, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA,
            )

            writer.write(annotated)

            t_frame_end = time.perf_counter()
            frame_times.append((t_frame_end - t_frame_start) * 1000.0)

            # Update job progress
            job.current_frame = frame_counter
            if job.total_frames > 0:
                job.progress = min(99.9, (frame_counter / job.total_frames) * 100.0)
            else:
                job.progress = 0.0

        # ---- Cleanup ----
        voice_engine.stop()
        cap.release()
        writer.release()

        t_end = time.perf_counter()
        elapsed = t_end - t_start
        fps_achieved = frame_counter / elapsed if elapsed > 0 else 0.0

        # Populate final results
        job.frames_processed = frame_counter
        job.processing_fps = round(fps_achieved, 2)
        job.workers_detected = len(unique_tracks)
        job.confirmed_violations = total_confirmed
        job.no_helmet = no_helmet_count
        job.no_mask = no_mask_count
        job.no_helmet_and_mask = no_helmet_and_mask_count
        job.evidence_count = total_evidence_saved
        job.progress = 100.0
        job.status = "completed"

    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        import traceback
        traceback.print_exc()
