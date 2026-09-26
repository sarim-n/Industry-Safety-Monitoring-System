# Phase 4: Alerts & Evidence Capture Summary Report

## 1. Executive Summary & Setup
- **Pipeline Version**: Phase 4 Live Safety Monitoring System
- **Script**: `run_live.py`
- **Target Model**: `runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt` (YOLOv8s @ 800)
- **Input Source**: `C:\Users\sarim\safety monitoring\data_collection\videos\4048038451-preview.mp4` (Video File Mode)
- **Resolution**: 898x506 @ 25.0 FPS
- **Operating Thresholds**: `PERSON_CONF=0.50`, `HELMET_CONF=0.25`, `MASK_CONF=0.20`
- **Temporal Config**: `CONFIRMATION_FRAMES=5`, `ALERT_COOLDOWN_SECONDS=5.0`
- **TEST Set Status**: Untouched and NOT evaluated.

## 2. Phase 4 Alert & Evidence Metrics

- **Total Frames Processed**: 301 frames (10.74 seconds)
- **Unique Workers Observed**: 6
- **Confirmed Violation Events**: 7
- **Voice & System Alerts Emitted**: **7**
- **Alerts Suppressed by Cooldown**: **0**
- **Evidence Images Created**: **7**
- **Event Log Path**: `reports/alerts_v1/events.csv`
- **Evidence Directory**: `evidence/`

## 3. Real-Time Pipeline Throughput & Latency

- **Average Measured Throughput**: **28.0 FPS**
- **Average Frame Latency**: **25.15 ms / frame**
- **YOLO Inference Latency**: `21.22 ms`
- **PPE Association Latency**: `0.07 ms`
- **Temporal Confirmation Latency**: `0.02 ms`
- **HUD & Evidence Rendering Latency**: `0.83 ms`

## 4. Strict Protection Confirmations
- **Dataset & Splits**: Untouched.
- **Model Architecture & Weights**: Untouched (`runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt`).
- **Confidence Thresholds**: Untouched (`PERSON_CONF=0.50`, `HELMET_CONF=0.25`, `MASK_CONF=0.20`).
- **PPE Association Algorithm**: Untouched.
- **Temporal Confirmation Engine**: Untouched.
- **TEST Set**: Untouched.
