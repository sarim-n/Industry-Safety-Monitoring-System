"""
FastAPI Main Application for Industrial Safety Monitoring System
================================================================
"""

import os
import time
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, Request
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
