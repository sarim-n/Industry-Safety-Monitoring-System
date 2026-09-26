"""
Phase 3: Live OpenCV Safety Monitoring Pipeline
================================================
Integrates YOLOv8s @ 800 detection, Phase 1 PPE Association, and Phase 2 Temporal Confirmation
into a real-time OpenCV video/webcam streaming pipeline.

Pipeline Architecture:
Video / Webcam -> OpenCV -> YOLOv8s @ 800 -> Detection Filtering -> PPE Association -> Temporal Confirmation -> Live HUD Visualization

Usage:
  python run_live.py --source data_collection/videos/4048038451-preview.mp4
  python run_live.py --source 0                      # Webcam mode
  python run_live.py --source video.mp4 --headless   # Non-display performance benchmark
  python run_live.py --source video.mp4 --save-video output.mp4
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import csv
import argparse
import collections
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from src.safety.ppe_association import (
    PPEAssociationConfig,
    PersonPPEState,
    PPEAssociator
)
from src.safety.temporal_confirmation import (
    TemporalConfirmationConfig,
    ConfirmedViolationEvent,
    TemporaryWorkerTrack,
    TemporalConfirmationEngine
)
from src.safety.alert_manager import (
    AlertManagerConfig,
    AlertManager
)
from src.safety.voice_alert import (
    VoiceAlertEngine
)
from src.safety.evidence import (
    EvidenceManager
)
from src.safety.api_bridge import (
    NonBlockingAPIBridge
)
from datetime import datetime

# Status color palette (BGR format)
STATUS_COLORS = {
    "SAFE": (0, 255, 0),              # Bright Green
    "NO_HELMET": (0, 165, 255),        # Orange
    "NO_MASK": (255, 0, 255),          # Purple
    "NO_HELMET_AND_MASK": (0, 0, 255), # Red
    "UNCERTAIN": (255, 255, 0)         # Cyan/Yellow
}


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Industrial AI Safety Monitoring Pipeline")
    parser.add_argument("--source", type=str, default=r"data_collection/videos/4048038451-preview.mp4",
                        help="Video file path or webcam index (0 for default webcam)")
    parser.add_argument("--weights", type=str, default=r"runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt",
                        help="Path to trained YOLO best.pt weights")
    parser.add_argument("--imgsz", type=int, default=800, help="Inference resolution (default: 800)")
    parser.add_argument("--headless", action="store_true", help="Run without cv2.imshow GUI display")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = process all)")
    parser.add_argument("--save-video", type=str, default="", help="Optional path to save annotated output video")
    parser.add_argument("--disable-voice", action="store_true", help="Disable TTS voice alerts")
    parser.add_argument("--enable-api", action="store_true", help="Enable live telemetry push to FastAPI backend")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000/api/telemetry", help="FastAPI telemetry endpoint URL")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Validate Weights Path
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = Path(PROJECT_ROOT) / weights_path
    assert weights_path.exists(), f"Model weights file not found: {weights_path}"

    # 2. Determine Video / Webcam Source
    source_input = args.source
    is_webcam = False
    if source_input.isdigit():
        cap_source = int(source_input)
        is_webcam = True
    else:
        cap_source = str(Path(source_input) if Path(source_input).is_absolute() else Path(PROJECT_ROOT) / source_input)
        if not os.path.exists(cap_source):
            print(f"[ERROR] Source video file not found: {cap_source}")
            sys.exit(1)

    print("==================================================================")
    print("   LIVE OPENCV SAFETY MONITORING PIPELINE (PHASE 5: BACKEND INTEGRATED)")
    print("==================================================================")
    print(f"  Model Weights : {weights_path}")
    print(f"  Resolution    : {args.imgsz}x{args.imgsz}")
    print(f"  Input Source  : {cap_source} {'(Webcam)' if is_webcam else '(Video File)'}")
    print(f"  Headless Mode : {args.headless}")
    print(f"  Voice Alerts  : {'Disabled' if args.disable_voice else 'Enabled (Offline TTS)'}")
    print(f"  API Bridge    : {'Enabled -> ' + args.api_url if args.enable_api else 'Disabled (Local Only)'}")

    # 3. Open Video Capture
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video capture source: {cap_source}")
        sys.exit(1)

    # Detect Video FPS and Resolution
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0.0 or source_fps > 120.0 or is_webcam:
        effective_fps = 30.0
    else:
        effective_fps = source_fps

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1

    print(f"  Source Props  : {frame_w}x{frame_h} @ {effective_fps:.1f} FPS (Total frames: {total_source_frames})")

    # 4. Initialize Core Pipeline Engines
    print("\n[LOADING YOLO MODEL & PIPELINE ENGINES]")
    model = YOLO(str(weights_path))

    # Frozen Validation-Selected Confidence Thresholds
    assoc_config = PPEAssociationConfig(
        person_conf=0.50,
        helmet_conf=0.25,
        mask_conf=0.20
    )
    associator = PPEAssociator(assoc_config)

    # Frozen Temporal Confirmation Settings
    temp_config = TemporalConfirmationConfig(
        confirmation_frames=5,
        min_track_iou=0.30,
        max_missed_frames=10,
        fps=effective_fps,
        alert_cooldown_seconds=5.0,
        uncertain_breaks_streak=True
    )
    temporal_engine = TemporalConfirmationEngine(temp_config)

    # Phase 4 & 5 Engines: AlertManager, VoiceAlertEngine, EvidenceManager, API Bridge
    alert_manager = AlertManager(AlertManagerConfig(cooldown_seconds=5.0))
    voice_engine = VoiceAlertEngine(enabled=not args.disable_voice)
    evidence_manager = EvidenceManager(
        evidence_dir=str(Path(PROJECT_ROOT) / "evidence"),
        csv_path=str(Path(PROJECT_ROOT) / "reports" / "alerts_v1" / "events.csv")
    )
    api_bridge = NonBlockingAPIBridge(api_url=args.api_url, enabled=args.enable_api)

    # Video Writer if requested
    video_writer = None
    if args.save_video:
        out_path = Path(args.save_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(out_path), fourcc, effective_fps, (frame_w, frame_h))
        print(f"  Saving output to: {out_path}")

    # Create Performance Logging CSV
    reports_dir = Path(PROJECT_ROOT) / "reports" / "live_pipeline_v1"
    reports_dir.mkdir(parents=True, exist_ok=True)
    perf_csv_path = reports_dir / "live_performance_log.csv"

    # Latency and Performance Tracking
    frame_times = collections.deque(maxlen=30)
    yolo_times = collections.deque(maxlen=30)
    assoc_times = collections.deque(maxlen=30)
    temp_times = collections.deque(maxlen=30)
    render_times = collections.deque(maxlen=30)

    frame_counter = 0
    total_confirmed_events = 0
    total_alerts_emitted = 0
    total_alerts_suppressed = 0
    total_evidence_saved = 0
    unique_tracks_seen = set()

    perf_csv_rows = []

    print("\n[STARTING LIVE PROCESSING LOOP — Press 'q' in window to exit]")
    print("------------------------------------------------------------------")

    t_loop_start = time.perf_counter()

    try:
        while True:
            t_frame_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[INFO] End of video stream reached.")
                break

            frame_counter += 1

            # --- STAGE 1: YOLO Inference ---
            t0 = time.perf_counter()
            results = model.predict(frame, imgsz=args.imgsz, conf=0.05, verbose=False)[0]
            t1 = time.perf_counter()

            raw_dets = []
            for box in results.boxes:
                c = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                raw_dets.append({'cls': c, 'conf': conf, 'box': xyxy})

            # --- STAGE 2: PPE Association ---
            t2_start = time.perf_counter()
            person_states = associator.process_detections(raw_dets, frame_w, frame_h)
            t2_end = time.perf_counter()

            # --- STAGE 3: Temporal Confirmation ---
            t3_start = time.perf_counter()
            current_timestamp = (frame_counter - 1) / effective_fps
            events = temporal_engine.process_frame(
                person_states,
                frame_index=frame_counter - 1,
                timestamp_sec=current_timestamp
            )
            t3_end = time.perf_counter()

            emitted_alerts_this_frame = []
            if events:
                total_confirmed_events += len(events)
                for e in events:
                    print(f"  >>> [FRAME {frame_counter:4d} | t={e.timestamp_sec:.2f}s] CONFIRMED EVENT: Track #{e.track_id} -> {e.violation} (Streak: {e.consecutive_streak})")
                    
                    # Phase 4 AlertManager Check
                    if alert_manager.should_emit_alert(e.track_id, e.violation, e.timestamp_sec):
                        total_alerts_emitted += 1
                        voice_engine.speak(e.violation)
                        emitted_alerts_this_frame.append(e)
                    else:
                        total_alerts_suppressed += 1
                        print(f"      ↳ ALERT SUPPRESSED (Cooldown active for Track #{e.track_id})")

            # Update unique tracks seen
            for track in temporal_engine.active_tracks:
                unique_tracks_seen.add(track.track_id)

            # --- STAGE 4: Visualization & Rendering ---
            t4_start = time.perf_counter()
            annotated_frame = frame.copy()

            # Draw worker bounding boxes & status badges for active tracks
            for track in temporal_engine.active_tracks:
                # Find matching PersonPPEState
                state = next((ps for ps in person_states if ps.person_bbox == track.last_bbox), None)

                bx1, by1, bx2, by2 = [int(v) for v in track.last_bbox]
                status = state.safety_status if state else "UNCERTAIN"

                # Pick color
                color = STATUS_COLORS.get(status, (200, 200, 200))

                # Highlight if confirmed violation or active streak
                is_confirmed_now = any(e.track_id == track.track_id for e in events)
                if is_confirmed_now or (track.last_alert_frame and (frame_counter - 1 - track.last_alert_frame) < 15):
                    # Flashing border for confirmed alert
                    cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 4)
                    badge_title = f"Worker #{track.track_id} | CONFIRMED {track.current_violation}"
                    badge_bg = (0, 0, 255)
                elif track.consecutive_violation_count > 0:
                    cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 2)
                    badge_title = f"Worker #{track.track_id} | {track.current_violation} ({track.consecutive_violation_count}/5)"
                    badge_bg = color
                else:
                    cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 2)
                    badge_title = f"Worker #{track.track_id} | {status}"
                    badge_bg = color

                # Draw PPE item bboxes if associated
                if state:
                    if state.helmet_bbox:
                        hx1, hy1, hx2, hy2 = [int(v) for v in state.helmet_bbox]
                        cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (255, 255, 0), 2)
                    if state.mask_bbox:
                        mx1, my1, mx2, my2 = [int(v) for v in state.mask_bbox]
                        cv2.rectangle(annotated_frame, (mx1, my1), (mx2, my2), (255, 0, 255), 2)

                # Draw badge label text
                (txt_w, txt_h), _ = cv2.getTextSize(badge_title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (bx1, max(0, by1 - 22)), (bx1 + txt_w + 8, max(22, by1)), badge_bg, -1)
                cv2.putText(annotated_frame, badge_title, (bx1 + 4, max(16, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if badge_bg != (0, 0, 255) else (255, 255, 255), 1, cv2.LINE_AA)

            t4_end = time.perf_counter()

            # --- STAGE 5: Evidence Capture ---
            for e in emitted_alerts_this_frame:
                saved_path = evidence_manager.capture_evidence(
                    frame=annotated_frame,
                    worker_id=e.track_id,
                    violation_type=e.violation
                )
                total_evidence_saved += 1
                print(f"      ↳ EVIDENCE CAPTURED: {saved_path}")

            # Record Latencies
            t_frame_end = time.perf_counter()
            total_ms = (t_frame_end - t_frame_start) * 1000.0
            yolo_ms = (t1 - t0) * 1000.0
            assoc_ms = (t2_end - t2_start) * 1000.0
            temp_ms = (t3_end - t3_start) * 1000.0
            render_ms = (t4_end - t4_start) * 1000.0

            frame_times.append(total_ms)
            yolo_times.append(yolo_ms)
            assoc_times.append(assoc_ms)
            temp_times.append(temp_ms)
            render_times.append(render_ms)

            rolling_fps = 1000.0 / (np.mean(frame_times)) if frame_times else 0.0

            # Draw HUD Box
            hud_bg = (20, 20, 20)
            cv2.rectangle(annotated_frame, (10, 10), (540, 135), hud_bg, -1)
            cv2.rectangle(annotated_frame, (10, 10), (540, 135), (0, 255, 0), 1)

            cv2.putText(annotated_frame, f"FPS: {rolling_fps:.1f} (YOLO: {np.mean(yolo_times):.1f}ms | Assoc: {np.mean(assoc_times):.1f}ms | Temp: {np.mean(temp_times):.1f}ms)",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Frame: {frame_counter} | Active Workers: {len(temporal_engine.active_tracks)} | Confirmed Events: {total_confirmed_events}",
                        (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Alerts Emitted: {total_alerts_emitted} | Suppressed: {total_alerts_suppressed} | Evidence Saved: {total_evidence_saved}",
                        (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Res: {args.imgsz}x{args.imgsz} | Model: YOLOv8s @ 800",
                        (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, "Press 'q' to exit live window",
                        (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

            # --- STAGE 6: Optional API Telemetry & Frame Push ---
            if api_bridge and api_bridge.enabled:
                worker_list = []
                for trk in temporal_engine.active_tracks:
                    st = next((ps.safety_status for ps in person_states if ps.person_bbox == trk.last_bbox), "UNCERTAIN")
                    worker_list.append({"worker_id": trk.track_id, "status": st})
                latest_evt = None
                if emitted_alerts_this_frame:
                    e = emitted_alerts_this_frame[-1]
                    latest_evt = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "worker_id": e.track_id,
                        "violation_type": e.violation,
                        "evidence_path": saved_path if 'saved_path' in locals() else ""
                    }
                api_bridge.push_update(active_workers=worker_list, latest_event=latest_evt)
                api_bridge.push_frame(annotated_frame)

            # Record CSV performance row
            perf_csv_rows.append({
                'frame': frame_counter,
                'total_frame_ms': f"{total_ms:.2f}",
                'yolo_ms': f"{yolo_ms:.2f}",
                'assoc_ms': f"{assoc_ms:.2f}",
                'temp_ms': f"{temp_ms:.2f}",
                'render_ms': f"{render_ms:.2f}",
                'rolling_fps': f"{rolling_fps:.1f}",
                'active_workers': len(temporal_engine.active_tracks),
                'confirmed_events_emitted': len(events)
            })

            # Save video frame if requested
            if video_writer is not None:
                video_writer.write(annotated_frame)

            # Display GUI window unless headless
            if not args.headless:
                cv2.imshow("Industrial AI Safety Monitoring — Phase 5 Live Pipeline", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] 'q' pressed. Exiting live loop cleanly...")
                    break

            if args.max_frames > 0 and frame_counter >= args.max_frames:
                print(f"\n[INFO] Reached max requested frames ({args.max_frames}). Exiting loop...")
                break

    finally:
        voice_engine.stop()
        api_bridge.stop()
        t_loop_end = time.perf_counter()
        total_elapsed_sec = t_loop_end - t_loop_start

        # Release Resources
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    # 5. Export Performance CSV Log
    with open(perf_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['frame', 'total_frame_ms', 'yolo_ms', 'assoc_ms', 'temp_ms', 'render_ms', 'rolling_fps', 'active_workers', 'confirmed_events_emitted']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(perf_csv_rows)

    # 6. Generate Alert Summary Markdown Report (Phase 4 Deliverable)
    alerts_dir = Path(PROJECT_ROOT) / "reports" / "alerts_v1"
    alerts_dir.mkdir(parents=True, exist_ok=True)
    alert_summary_path = alerts_dir / "alert_summary.md"
    mean_fps = frame_counter / total_elapsed_sec if total_elapsed_sec > 0 else 0.0
    mean_yolo = np.mean(yolo_times) if yolo_times else 0.0
    mean_assoc = np.mean(assoc_times) if assoc_times else 0.0
    mean_temp = np.mean(temp_times) if temp_times else 0.0
    mean_render = np.mean(render_times) if render_times else 0.0
    mean_total = np.mean(frame_times) if frame_times else 0.0

    with open(alert_summary_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 4: Alerts & Evidence Capture Summary Report\n\n")
        f.write("## 1. Executive Summary & Setup\n")
        f.write("- **Pipeline Version**: Phase 4 Live Safety Monitoring System\n")
        f.write("- **Script**: `run_live.py`\n")
        f.write("- **Target Model**: `runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt` (YOLOv8s @ 800)\n")
        f.write(f"- **Input Source**: `{cap_source}` {'(Webcam Mode)' if is_webcam else '(Video File Mode)'}\n")
        f.write(f"- **Resolution**: {frame_w}x{frame_h} @ {effective_fps:.1f} FPS\n")
        f.write("- **Operating Thresholds**: `PERSON_CONF=0.50`, `HELMET_CONF=0.25`, `MASK_CONF=0.20`\n")
        f.write("- **Temporal Config**: `CONFIRMATION_FRAMES=5`, `ALERT_COOLDOWN_SECONDS=5.0`\n")
        f.write("- **TEST Set Status**: Untouched and NOT evaluated.\n\n")

        f.write("## 2. Phase 4 Alert & Evidence Metrics\n\n")
        f.write(f"- **Total Frames Processed**: {frame_counter} frames ({total_elapsed_sec:.2f} seconds)\n")
        f.write(f"- **Unique Workers Observed**: {len(unique_tracks_seen)}\n")
        f.write(f"- **Confirmed Violation Events**: {total_confirmed_events}\n")
        f.write(f"- **Voice & System Alerts Emitted**: **{total_alerts_emitted}**\n")
        f.write(f"- **Alerts Suppressed by Cooldown**: **{total_alerts_suppressed}**\n")
        f.write(f"- **Evidence Images Created**: **{total_evidence_saved}**\n")
        f.write(f"- **Event Log Path**: `reports/alerts_v1/events.csv`\n")
        f.write(f"- **Evidence Directory**: `evidence/`\n\n")

        f.write("## 3. Real-Time Pipeline Throughput & Latency\n\n")
        f.write(f"- **Average Measured Throughput**: **{mean_fps:.1f} FPS**\n")
        f.write(f"- **Average Frame Latency**: **{mean_total:.2f} ms / frame**\n")
        f.write(f"- **YOLO Inference Latency**: `{mean_yolo:.2f} ms`\n")
        f.write(f"- **PPE Association Latency**: `{mean_assoc:.2f} ms`\n")
        f.write(f"- **Temporal Confirmation Latency**: `{mean_temp:.2f} ms`\n")
        f.write(f"- **HUD & Evidence Rendering Latency**: `{mean_render:.2f} ms`\n\n")

        f.write("## 4. Strict Protection Confirmations\n")
        f.write("- **Dataset & Splits**: Untouched.\n")
        f.write("- **Model Architecture & Weights**: Untouched (`runs/detect/safety_v1-4_run2b_yolov8s_800/weights/best.pt`).\n")
        f.write("- **Confidence Thresholds**: Untouched (`PERSON_CONF=0.50`, `HELMET_CONF=0.25`, `MASK_CONF=0.20`).\n")
        f.write("- **PPE Association Algorithm**: Untouched.\n")
        f.write("- **Temporal Confirmation Engine**: Untouched.\n")
        f.write("- **TEST Set**: Untouched.\n")

    print(f"\n[ALERT SUMMARY REPORT SAVED] {alert_summary_path}")
    print(f"[PERFORMANCE LOG SAVED] {perf_csv_path}")
    print("\n================ PHASE 4 PIPELINE RUN COMPLETE ================")


if __name__ == "__main__":
    main()
