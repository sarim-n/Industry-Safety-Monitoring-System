# Industrial Safety Monitoring System — FastAPI Backend

Lightweight REST API backend built with FastAPI, Uvicorn, and Pydantic for serving real-time worker safety states, historical violation logs, aggregate statistics, and evidence snapshot images.

---

## 1. Quick Start

### Start API Server
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Interactive API documentation available at:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Run Backend Unit Tests
```bash
python test_backend.py
```

---

## 2. API Endpoints Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check (`{"status": "ok"}`) |
| `/api/status` | GET | System status, active worker counts, and last event |
| `/api/workers` | GET | Active worker safety states (`SAFE`, `NO_HELMET`, `NO_MASK`, etc.) |
| `/api/events` | GET | Historical violation events read from `reports/alerts_v1/events.csv` (params: `limit`, `offset`) |
| `/api/statistics` | GET | Aggregate safety metrics and per-violation event counts |
| `/api/evidence/{filename}` | GET | Serves evidence snapshot image securely with path traversal protection |
| `/api/telemetry` | POST | Ingests live telemetry updates from `run_live.py` pipeline |

---

## 3. CORS Configuration

CORS middleware configured for frontend development servers:
- `http://localhost:3000`
- `http://localhost:5173`
- `http://127.0.0.1:3000`
- `http://127.0.0.1:5173`

---

## 4. Architecture

- **Data Sources**: CSV event log (`reports/alerts_v1/events.csv`) and Evidence directory (`evidence/`).
- **Live State**: Managed in-memory via `SafetyService`.
- **CV Pipeline Decoupling**: CV pipeline (`run_live.py`) runs independently and pushes non-blocking telemetry updates to `/api/telemetry` when available.
