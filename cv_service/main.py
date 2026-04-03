"""
EagleVision CV Microservice  –  main.py
========================================
Entry point. Reads video → runs detection + tracking + optical flow
→ classifies activity → publishes to Kafka → writes to TimescaleDB.
"""

import os
import sys
import time
import logging
import argparse
import cv2
import numpy as np

from detector      import EquipmentDetector
from tracker_state import MachineStateTracker
from flow_analyzer import OpticalFlowAnalyzer
from classifier    import ActivityClassifier
from kafka_pub     import KafkaPublisher
from db_writer     import DBWriter

# ─────────────────── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("cv_service")

# ─────────────────── Config from env ────────────────────────────────────────
CONFIG = {
    "video_source"          : os.getenv("VIDEO_SOURCE", "data/input.mp4"),
    "kafka_servers"         : os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    "kafka_topic"           : os.getenv("KAFKA_TOPIC", "equipment_events"),
    "yolo_model"            : os.getenv("YOLO_MODEL", "yolov8m.pt"),
    "flow_threshold"        : float(os.getenv("FLOW_THRESHOLD", "1.5")),
    "inactive_threshold"    : float(os.getenv("INACTIVE_THRESHOLD", "0.5")),
    "db_host"               : os.getenv("DB_HOST", "localhost"),
    "db_port"               : int(os.getenv("DB_PORT", "5432")),
    "db_name"               : os.getenv("DB_NAME", "eaglevision"),
    "db_user"               : os.getenv("DB_USER", "eagle"),
    "db_password"           : os.getenv("DB_PASSWORD", "eagle_secret"),
    "output_video"          : os.getenv("OUTPUT_VIDEO", "data/output.mp4"),
    "target_fps"            : int(os.getenv("TARGET_FPS", "15")),    # process every N fps
    "display_local"         : os.getenv("DISPLAY_LOCAL", "false").lower() == "true",
}


def build_equipment_id(cls_name: str, track_id: int) -> str:
    prefix = {"excavator": "EX", "dump_truck": "DT", "bulldozer": "BZ"}.get(cls_name, "EQ")
    return f"{prefix}-{track_id:03d}"


def run_pipeline(config: dict) -> None:
    log.info("🦅 EagleVision CV Service starting …")

    # ── Init components ───────────────────────────────────────────────────
    detector   = EquipmentDetector(config["yolo_model"])
    flow_anal  = OpticalFlowAnalyzer(
        flow_threshold     = config["flow_threshold"],
        inactive_threshold = config["inactive_threshold"],
    )
    classifier = ActivityClassifier()
    state_mgr  = MachineStateTracker()
    publisher  = KafkaPublisher(config["kafka_servers"], config["kafka_topic"])
    db_writer  = DBWriter(config)

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(config["video_source"])
    if not cap.isOpened():
        log.error(f"Cannot open video: {config['video_source']}")
        sys.exit(1)

    orig_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info(f"Video: {width}×{height} @ {orig_fps:.1f}fps  total_frames={total_f}")

    # ── Output writer ─────────────────────────────────────────────────────
    out_writer = None
    if config["output_video"]:
        fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(config["output_video"], fourcc, orig_fps, (width, height))

    # ── Frame-skip factor ─────────────────────────────────────────────────
    skip = max(1, int(orig_fps / config["target_fps"]))

    prev_gray = None
    frame_id  = 0
    t_start   = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("End of video stream.")
                break

            frame_id += 1
            if frame_id % skip != 0:
                # still write original frame to output for smooth playback
                if out_writer:
                    out_writer.write(frame)
                continue

            # ── Gray for optical flow ─────────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Detect + Track ────────────────────────────────────────────
            detections = detector.detect(frame)   # list of Detection objects

            # ── Compute dense optical flow (full frame) ───────────────────
            flow = None
            if prev_gray is not None:
                flow = flow_anal.compute_flow(prev_gray, gray)

            annotated = frame.copy()

            for det in detections:
                # ── Regional flow analysis ────────────────────────────────
                flow_upper, flow_lower, motion_source = flow_anal.analyze_region(
                    flow, det.bbox, height
                ) if flow is not None else (0.0, 0.0, "none")

                # ── State: ACTIVE / INACTIVE ──────────────────────────────
                state = "ACTIVE" if motion_source != "none" else "INACTIVE"

                # ── Activity classification ───────────────────────────────
                activity = classifier.classify(
                    track_id      = det.track_id,
                    bbox          = det.bbox,
                    flow_upper    = flow_upper,
                    flow_lower    = flow_lower,
                    motion_source = motion_source,
                )

                # ── Time accumulation ─────────────────────────────────────
                eq_id    = build_equipment_id(det.cls_name, det.track_id)
                dt_secs  = skip / orig_fps
                stats    = state_mgr.update(eq_id, state, dt_secs)

                # ── Build Kafka payload ───────────────────────────────────
                video_ts  = frame_id / orig_fps
                h  = int(video_ts // 3600)
                m  = int((video_ts % 3600) // 60)
                s  = video_ts % 60
                ts = f"{h:02d}:{m:02d}:{s:06.3f}"

                payload = {
                    "frame_id"       : frame_id,
                    "equipment_id"   : eq_id,
                    "equipment_class": det.cls_name,
                    "track_id"       : det.track_id,
                    "timestamp"      : ts,
                    "utilization"    : {
                        "current_state"   : state,
                        "current_activity": activity,
                        "motion_source"   : motion_source,
                    },
                    "flow_metrics"   : {
                        "flow_upper": round(flow_upper, 3),
                        "flow_lower": round(flow_lower, 3),
                    },
                    "bbox"           : {
                        "x1": det.bbox[0], "y1": det.bbox[1],
                        "x2": det.bbox[2], "y2": det.bbox[3],
                    },
                    "time_analytics" : {
                        "total_tracked_seconds" : round(stats["total_tracked"], 2),
                        "total_active_seconds"  : round(stats["total_active"],  2),
                        "total_idle_seconds"    : round(stats["total_idle"],    2),
                        "utilization_percent"   : round(stats["utilization_pct"], 1),
                    },
                }

                # ── Publish to Kafka ──────────────────────────────────────
                publisher.publish(payload)

                # ── Write to DB ───────────────────────────────────────────
                db_writer.insert(payload)

                # ── Annotate frame ────────────────────────────────────────
                annotated = draw_annotations(annotated, det, state, activity,
                                             motion_source, stats)

            # ── Write output frame ────────────────────────────────────────
            if out_writer:
                out_writer.write(annotated)

            if config["display_local"]:
                cv2.imshow("EagleVision", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            prev_gray = gray

            if frame_id % 100 == 0:
                elapsed = time.time() - t_start
                log.info(f"Frame {frame_id}/{total_f} – elapsed {elapsed:.1f}s – "
                         f"active machines: {state_mgr.active_count()}")

    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()
        publisher.close()
        db_writer.close()
        log.info("✅ Pipeline finished.")


# ─────────────────── Annotation helpers ─────────────────────────────────────
COLORS = {
    "ACTIVE"  : (0, 200, 100),
    "INACTIVE": (0, 100, 220),
}
ACTIVITY_ICONS = {
    "DIGGING"        : "⛏",
    "SWINGING/LOADING": "🔄",
    "DUMPING"        : "📤",
    "WAITING"        : "⏸",
}

def draw_annotations(frame, det, state, activity, motion_source, stats):
    x1, y1, x2, y2 = det.bbox
    color = COLORS.get(state, (200, 200, 200))

    # BBox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Header label
    label = f"{det.cls_name.upper()} #{det.track_id} | {state}"
    lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)[0]
    cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

    # Activity + motion source
    sub = f"{activity} [{motion_source}]"
    cv2.putText(frame, sub, (x1 + 4, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Utilization strip at bottom of box
    util = stats["utilization_pct"]
    bar_w = x2 - x1
    filled = int(bar_w * util / 100)
    cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 10), (50, 50, 50), -1)
    cv2.rectangle(frame, (x1, y2 + 2), (x1 + filled, y2 + 10), color, -1)

    util_txt = f"{util:.1f}% util | A:{stats['total_active']:.0f}s I:{stats['total_idle']:.0f}s"
    cv2.putText(frame, util_txt, (x1, y2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    return frame


# ─────────────────── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EagleVision CV Service")
    parser.add_argument("--video",   default=CONFIG["video_source"])
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    CONFIG["video_source"]  = args.video
    CONFIG["display_local"] = args.display
    run_pipeline(CONFIG)
