"""
FastAPI Main Application for Industrial Safety Monitoring System
================================================================
"""

import os
import time
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from backend.models import (
    HealthResponse,
    WorkerStateModel,
    EventModel,
    SystemStatusResponse,
    StatisticsResponse,
    TelemetryUpdate
)
from backend.services.safety_service import (
    SafetyService,
    generate_offline_placeholder_jpg
)
from backend.services import video_processing_service as vps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = PROJECT_ROOT / "evidence"

app = FastAPI(
    title="Industrial AI Safety Monitoring API",
    description="FastAPI REST API serving worker safety states, events, statistics, evidence snapshots, and MJPEG live stream.",
    version="1.0.0"
)

# Step 8: CORS Configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared SafetyService Instance
safety_service = SafetyService()


@app.get("/api/health", response_model=HealthResponse)
def get_health():
    """Returns backend service health status."""
    return HealthResponse(status="ok")


@app.get("/api/status", response_model=SystemStatusResponse)
def get_status():
    """Returns system operational status, active worker counts, and last event."""
    return safety_service.get_system_status()


@app.get("/api/workers", response_model=List[WorkerStateModel])
def get_workers():
    """Returns list of active worker safety states."""
    return safety_service.get_workers()


@app.get("/api/events", response_model=List[EventModel])
def get_events(
    limit: int = Query(20, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """Returns recorded safety events from CSV log with pagination."""
    return safety_service.get_events(limit=limit, offset=offset)


@app.get("/api/statistics", response_model=StatisticsResponse)
def get_statistics():
    """Returns aggregate safety metrics and event totals."""
    return safety_service.get_statistics()


@app.post("/api/telemetry")
def update_telemetry(update: TelemetryUpdate):
    """Receives live state/event updates from OpenCV processing pipeline."""
    safety_service.update_telemetry(update)
    return {"status": "updated"}


@app.post("/api/telemetry/frame")
async def update_telemetry_frame(request: Request):
    """Receives raw annotated JPEG frame bytes from run_live.py api_bridge."""
    frame_bytes = await request.body()
    if frame_bytes:
        safety_service.update_frame(frame_bytes)
    return {"status": "frame_updated"}


def generate_mjpeg_stream():
    """MJPEG stream generator yielding multipart/x-mixed-replace JPEG frames."""
    placeholder_bytes = generate_offline_placeholder_jpg()

    while True:
        time.sleep(0.033)  # Rate limit stream generator loop (~30 FPS max)

        if safety_service.is_stream_active(timeout_sec=3.0):
            frame_bytes = safety_service.get_latest_frame()
            if not frame_bytes:
                frame_bytes = placeholder_bytes
        else:
            frame_bytes = placeholder_bytes

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/api/video/stream")
def video_stream():
    """
    Serves live annotated OpenCV video stream as MJPEG.
    Displays live feed when run_live.py is active, or offline banner when inactive.
    """
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO UPLOAD & PROCESSING ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB hard cap


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Accept a video file upload, validate it, and start background safety processing.
    Returns job_id immediately without blocking.
    """
    # 1. Extension validation
    if not vps.is_allowed_extension(file.filename or ""):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed: {', '.join(vps.ALLOWED_EXTENSIONS)}",
        )

    # 2. Read file (size guard)
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 2 GB limit.")

    # 3. Create job
    try:
        job = vps.create_job(file.filename or "upload.mp4", file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": "Processing started in background.",
    }


@app.get("/api/video/status/{job_id}")
def get_video_status(job_id: str):
    """Poll job processing status, progress, and frame counts."""
    job = vps.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": round(job.progress, 1),
        "current_frame": job.current_frame,
        "total_frames": job.total_frames,
        "error": job.error,
    }


@app.get("/api/video/output/{job_id}")
def get_video_output(job_id: str):
    """Stream the annotated output video for browser playback (MP4 or AVI)."""
    job = vps.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job not completed. Status: {job.status}")
    if not job.output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found on disk.")

    suffix = job.output_path.suffix.lower()
    media_type = "video/mp4" if suffix == ".mp4" else "video/x-msvideo"
    return FileResponse(
        path=str(job.output_path),
        media_type=media_type,
        filename=f"annotated_{job_id[:8]}{suffix}",
        headers={"Accept-Ranges": "bytes"},
    )



@app.get("/api/video/results/{job_id}")
def get_video_results(job_id: str):
    """Return final statistics for a completed job."""
    job = vps.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status not in ("completed", "failed"):
        raise HTTPException(status_code=409, detail=f"Job not finished. Status: {job.status}")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "error": job.error,
        "frames_processed": job.frames_processed,
        "processing_fps": job.processing_fps,
        "workers_detected": job.workers_detected,
        "confirmed_violations": job.confirmed_violations,
        "no_helmet": job.no_helmet,
        "no_mask": job.no_mask,
        "no_helmet_and_mask": job.no_helmet_and_mask,
        "evidence_count": job.evidence_count,
        "codec_used": job.codec_used,
        "model": "YOLOv8s Clean Run 2B — runs/detect/runs/detect/safety_v1-4_run2b_clean10810476_yolov8s_800/weights/best.pt",
    }


@app.get("/api/video/upload")
async def get_video_upload():
    """Return upload constraints."""
    return {
        "allowed_extensions": list(vps.ALLOWED_EXTENSIONS),
        "max_size_bytes": MAX_UPLOAD_BYTES,
        "model": "YOLOv8s Clean Run 2B",
    }


@app.get("/api/evidence/{filename:path}")
def get_evidence(filename: str):
    """
    Serves evidence image snapshot files safely.
    Prevents path traversal and returns HTTP 404 if file does not exist.
    """
    # 1. Security check: Reject path traversal sequences
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        raise HTTPException(status_code=400, detail="Path traversal rejected")

    # 2. Resolve target path securely
    target_path = (EVIDENCE_DIR / filename).resolve()
    base_path = EVIDENCE_DIR.resolve()

    try:
        target_path.relative_to(base_path)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal rejected")

    # 3. Verify existence
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Evidence file not found")

    return FileResponse(str(target_path))
