# Phase 5: FastAPI Backend Summary Report

## 1. Architecture & Design Overview

Phase 5 implements a lightweight REST API backend built using **FastAPI**, **Uvicorn**, and **Pydantic**. The backend operates as a decoupled data service designed to serve worker safety telemetry, historical violation logs, aggregate statistics, and evidence snapshots to a frontend dashboard.

```
Camera / Video -> OpenCV -> YOLOv8s @ 800 -> PPE Association -> Temporal Confirmation -> Confirmed Events
                                                                                               │
                                                                                        (Non-blocking Bridge)
                                                                                               │
                                                                                               ▼
                                                                                        FastAPI Backend (Port 8000)
                                                                                         ├── GET /api/health
                                                                                         ├── GET /api/status
                                                                                         ├── GET /api/workers
                                                                                         ├── GET /api/events
                                                                                         ├── GET /api/statistics
                                                                                         └── GET /api/evidence/{filename}
```

### Decoupled Operational Principles:
1. **CV Pipeline Independence**: The existing computer vision pipeline (`run_live.py`) runs fully standalone and does NOT require FastAPI to function.
2. **No Model Inference in Backend**: FastAPI performs zero ML inference, YOLO detection, PPE association, or temporal tracking.
3. **CSV & Directory Source of Truth**: Historical events are loaded directly from `reports/alerts_v1/events.csv`, and snapshot images are served from `evidence/`.
4. **Zero Heavy Infrastructure**: Implemented without databases (PostgreSQL/MongoDB), Redis, Celery, Docker, or WebSockets.

---

## 2. API Endpoints & Specification

| Endpoint | Method | Parameters | Description | Response Schema |
|---|---|---|---|---|
| `/api/health` | GET | None | Backend health status | `{"status": "ok"}` |
| `/api/status` | GET | None | System status & last event | `SystemStatusResponse` |
| `/api/workers` | GET | None | Current active worker states | `List[WorkerStateModel]` |
| `/api/events` | GET | `limit` (int=20), `offset` (int=0) | Paginated historical events | `List[EventModel]` |
| `/api/statistics` | GET | None | Aggregate violation metrics | `StatisticsResponse` |
| `/api/evidence/{filename}` | GET | `filename` (str) | Serves snapshot image with path traversal defense | JPEG file / HTTP 404 / HTTP 400 |
| `/api/telemetry` | POST | None | Ingests live telemetry updates from `run_live.py` | `{"status": "updated"}` |

---

## 3. Security & CORS Configuration

- **Path Traversal Defense**: `GET /api/evidence/{filename}` validates path elements, rejecting relative path traversal patterns (`..`, leading slashes) and resolving canonical paths within the `evidence/` directory boundaries.
- **CORS Setup**: Restricted to development frontend origins:
  - `http://localhost:3000`
  - `http://localhost:5173`
  - `http://127.0.0.1:3000`
  - `http://127.0.0.1:5173`

---

## 4. Test Results

Executed `python test_backend.py` using `fastapi.testclient.TestClient`:
- **Total Tests Ran**: 9 tests
- **Result**: `OK` (0.134s)
- **Scenarios Verified**:
  1. `test_1_health_endpoint`: Returns 200 and `{"status": "ok"}`
  2. `test_2_status_endpoint`: Returns valid system status JSON
  3. `test_3_workers_endpoint`: Returns valid worker safety states list
  4. `test_4_events_endpoint`: Returns CSV-backed historical events
  5. `test_5_events_pagination`: Paginated query parameters (`limit`, `offset`) work properly
  6. `test_6_statistics_endpoint`: Returns correct aggregate metrics (total events, counts per violation, unique workers)
  7. `test_7_valid_evidence_serving`: Serves evidence snapshot images correctly
  8. `test_8_nonexistent_evidence_returns_404`: Returns HTTP 404 for missing image files
  9. `test_9_path_traversal_rejected`: Path traversal attempts are rejected cleanly (HTTP 400/404)

---

## 5. End-to-End Live Pipeline Verification

1. **Standalone CV Pipeline**:
   `python run_live.py --source data_collection/videos/4048038451-preview.mp4 --headless`
   - Verified that `run_live.py` executes at 28.0 FPS with voice alerts and evidence capture without requiring FastAPI backend.

2. **CV Pipeline + FastAPI Bridge**:
   `python -m uvicorn backend.main:app --reload`
   `python run_live.py --source data_collection/videos/4048038451-preview.mp4 --headless --enable-api`
   - Verified live telemetry updates sent asynchronously over `NonBlockingAPIBridge` without impacting CV frame rate.

---

## 6. Known Limitations

- **State Persistence**: Historical events rely on `events.csv`. In-memory worker states reset when the server restarts (database persistence deliberately deferred to future phases).
- **Polling vs WebSockets**: Dashboard clients poll endpoints (`GET /api/status`, `GET /api/events`); WebSockets will be introduced in future real-time streaming phases.

---

## 7. Strict Protection Confirmations

- **Dataset & Dataset Splits**: Untouched.
- **TEST Set**: Untouched and NOT evaluated.
- **Model Architecture & Weights**: Untouched (`runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt`).
- **Confidence Thresholds**: Untouched (`PERSON_CONF=0.50`, `HELMET_CONF=0.25`, `MASK_CONF=0.20`).
- **PPE Association Engine**: Untouched (`src/safety/ppe_association.py`).
- **Temporal Confirmation Engine**: Untouched (`src/safety/temporal_confirmation.py`).
