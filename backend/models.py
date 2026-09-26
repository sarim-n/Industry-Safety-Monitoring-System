"""
Pydantic Data Schemas for FastAPI Safety Monitoring Backend
============================================================
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class WorkerStateModel(BaseModel):
    worker_id: int
    status: str = Field(..., description="SAFE, NO_HELMET, NO_MASK, NO_HELMET_AND_MASK, UNCERTAIN")


class EventModel(BaseModel):
    timestamp: str
    worker_id: int
    violation_type: str
    evidence_path: str


class SystemStatusResponse(BaseModel):
    system_running: bool
    active_workers: int
    confirmed_violations: int
    last_event: Optional[EventModel] = None


class StatisticsResponse(BaseModel):
    total_events: int
    no_helmet_count: int
    no_mask_count: int
    no_helmet_and_mask_count: int
    unique_workers: int
    latest_event_timestamp: Optional[str] = None


class TelemetryUpdate(BaseModel):
    system_running: bool = True
    active_workers: List[WorkerStateModel] = []
    latest_event: Optional[EventModel] = None
