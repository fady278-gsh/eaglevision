# 🦅 EagleVision — Equipment Utilization & Activity Classification

> Real-time, microservices-based pipeline for construction equipment monitoring.
> Tracks ACTIVE/INACTIVE states, classifies work activities, and streams utilization analytics through Apache Kafka to a live Streamlit dashboard.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Quick Start](#quick-start)
3. [Local Run (No Docker)](#local-run-no-docker)
4. [Component Guide](#component-guide)
5. [Technical Design Decisions](#technical-design-decisions)
6. [Activity Classification Logic](#activity-classification-logic)
7. [Articulated Motion Solution](#articulated-motion-solution)
8. [Kafka Payload Format](#kafka-payload-format)
9. [Database Schema](#database-schema)
10. [Project Structure](#project-structure)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         EagleVision Pipeline                              │
│                                                                           │
│  Video Input (file / RTSP stream)                                        │
│       │                                                                   │
│       ▼                                                                   │
│  ┌──────────────┐   ┌────────────────────┐   ┌──────────────────────┐   │
│  │  YOLOv8m     │──▶│  Farneback Dense   │──▶│  Rule-based Activity │   │
│  │  + ByteTrack │   │  Optical Flow      │   │  Classifier          │   │
│  │  (Re-ID)     │   │  Region Analysis   │   │  (State Machine)     │   │
│  └──────────────┘   └────────────────────┘   └──────────────────────┘   │
│         │                                              │                  │
│         └──────────────────┬─────────────────────────┘                  │
│                             ▼                                             │
│                   ┌──────────────────┐                                   │
│                   │  Time Accumulator │  ← per track_id                  │
│                   │  (active/idle)    │                                   │
│                   └────────┬─────────┘                                   │
│                            │                                              │
│              ┌─────────────┼──────────────┐                              │
│              ▼             ▼              ▼                              │
│       ┌────────────┐ ┌──────────┐ ┌────────────────┐                   │
│       │   Kafka    │ │TimescaleDB│ │ Annotated Video │                  │
│       │  Producer  │ │  Sink    │ │ (output.mp4)    │                   │
│       └─────┬──────┘ └──────────┘ └────────────────┘                   │
│             │                                                             │
│             ▼                                                             │
│       ┌─────────────────────────────────┐                               │
│       │      Kafka Topic                │                               │
│       │      equipment_events           │                               │
│       └──────────────┬──────────────────┘                               │
│                       │                                                   │
│              ┌────────┴────────┐                                        │
│              ▼                 ▼                                         │
│       ┌────────────┐   ┌──────────────┐                                 │
│       │ Streamlit  │   │ Future: more  │                                 │
│       │ Dashboard  │   │ consumers     │                                 │
│       └────────────┘   └──────────────┘                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

**Services:**

| Service | Image / Build | Port | Role |
|---|---|---|---|
| `zookeeper` | confluentinc/cp-zookeeper:7.6 | 2181 | Kafka coordinator |
| `kafka` | confluentinc/cp-kafka:7.6 | 9092 | Message broker |
| `kafka_ui` | provectuslabs/kafka-ui | 8090 | Broker monitoring |
| `timescaledb` | timescale/timescaledb:pg16 | 5432 | Time-series DB |
| `cv_service` | ./cv_service | — | Detection + Kafka producer |
| `streamlit_ui` | ./ui | 8501 | Live dashboard |

---

## Quick Start

### Prerequisites

- Docker ≥ 24.0 + Docker Compose V2
- A video file placed at `data/input.mp4` (see below)

### 1 — Get a test video

```bash
pip install yt-dlp
python scripts/download_test_video.py
# downloads first 60s of the excavator clip to data/input.mp4
```

Or bring your own fixed-camera construction video:
```bash
mkdir -p data && cp /path/to/your/video.mp4 data/input.mp4
```

### 2 — Start all services

```bash
docker compose up --build
```

### 3 — Open the dashboard

```
http://localhost:8501
```

- Click **▶ Connect** to start consuming Kafka events
- Watch machines appear as they are detected
- Kafka UI available at `http://localhost:8090`

### 4 — Stop

```bash
docker compose down -v
```

---

## Local Run (No Docker)

Fastest way to test without setting up Kafka or TimescaleDB:

```bash
# 1. Install dependencies
pip install -r cv_service/requirements.txt

# 2. Run pipeline (Kafka + DB are auto-disabled)
python scripts/run_local.py --video data/input.mp4

# 3. Output annotated video saved to:
#    data/output.mp4
```

### Run tests

```bash
pip install pytest
pytest tests/test_core.py -v
# 14 tests — all pass
```

---

## Component Guide

### `cv_service/`

| File | Responsibility |
|---|---|
| `main.py` | Pipeline orchestrator: read frames → detect → flow → classify → publish |
| `detector.py` | YOLOv8 + ByteTrack; maps COCO/custom classes to equipment names |
| `flow_analyzer.py` | Farneback dense optical flow; region-split analysis |
| `classifier.py` | Temporal state machine; classifies DIGGING/SWINGING/DUMPING/WAITING |
| `tracker_state.py` | Per-machine time accumulation (active/idle seconds, utilization %) |
| `kafka_pub.py` | confluent_kafka Producer; equipment_id as partition key |
| `db_writer.py` | Batched inserts to TimescaleDB via psycopg2 connection pool |

### `ui/`

| File | Responsibility |
|---|---|
| `app.py` | Streamlit dashboard: Kafka consumer thread + live charts + machine cards |

### `db/`

| File | Responsibility |
|---|---|
| `init.sql` | Schema: hypertable, continuous aggregate, retention policy, view |

---

## Technical Design Decisions

### Why Farneback Dense Optical Flow (not Lucas-Kanade)?

Lucas-Kanade is a **sparse** tracker — it needs good feature points (corners).
On construction equipment in daylight, feature points are often occluded by dirt,
motion blur, or uniform surfaces (side of a dump truck).

Farneback gives a **per-pixel** flow vector for the entire frame, so even regions
with weak texture still get reliable motion estimates. On a fixed camera the
background flow is near zero, which gives us a clean baseline for thresholding.

Trade-off: Farneback is ~3–4× slower than L-K sparse. At 720p it runs ~20–25 fps
on CPU (Python), which is sufficient for the 15 fps target. On GPU with CUDA
cv2.cuda.FarnebackOpticalFlow it reaches 60+ fps.

### Why YOLOv8m + ByteTrack for Re-ID?

ByteTrack is the default tracker in Ultralytics and implements:
- Kalman filter for state prediction during brief occlusions
- IoU + low-score detection association (handles partially visible equipment)
- Stable track_id across frames = free Re-ID for fixed-camera scenes

Alternative considered: **DeepSORT** (adds appearance embedding via Re-ID CNN).
Rejected at prototype stage because: (1) requires a second model, (2) ByteTrack
already achieves >90% MOTA on MOT17 without appearance features, (3) construction
equipment in fixed-camera scenes rarely has ambiguous Re-ID since only 2–5 machines
are visible simultaneously.

### Why Rule-based Classifier (not LSTM/C3D)?

| Approach | Pros | Cons |
|---|---|---|
| Rule-based (this) | Zero training data, instant, interpretable | Needs threshold tuning |
| LSTM + YOLO | Learns complex patterns | Needs 1000s of labelled clips |
| C3D | Captures 3D spatio-temporal | Very slow, needs GPU, large model |

At prototype stage, no labelled activity dataset exists. The rule-based state machine
performs well because: flow magnitude + centroid trajectory + motion_source already
encode the activity with high signal-to-noise on fixed-camera footage.

**Upgrade path:** Replace `classifier.py` with an LSTM head trained on the
TimescaleDB-accumulated feature vectors (flow_upper, flow_lower, cx_delta, cy_delta)
once enough labelled data is collected. The interface is unchanged.

### Why TimescaleDB over plain PostgreSQL?

Construction sites run 8-hour shifts → ~1M+ events/day at 30fps × 10 machines.
TimescaleDB hypertables auto-partition by time, giving:
- 10–100× faster range queries on time columns
- Built-in continuous aggregates (per-minute utilization) with no extra ETL
- Retention policies (auto-drop data older than 30 days)
- Compatible with all PostgreSQL tooling (psycopg2, SQLAlchemy, Grafana)

### Kafka Partition Strategy

`equipment_id` is used as the partition key. This guarantees:
- All events for one machine land on the same partition → ordered processing
- Multiple consumer instances can each handle a subset of machines
- No cross-partition joins needed for per-machine time analytics

---

## Articulated Motion Solution

**The Problem:** An excavator arm digs while the tracks remain completely stationary.
A naive "whole-bbox motion" detector would classify this as INACTIVE — wrong.

**The Solution: Vertical Region Split + Regional Flow Thresholding**

```
┌─────────────────────────────┐  ← bbox top (y1)
│                             │
│      UPPER REGION (55%)     │  ← arm / boom / bucket
│      flow_upper computed    │
│                             │
├─────────────────────────────┤  ← split_y = y1 + 0.55*(y2-y1)
│                             │
│      LOWER REGION (45%)     │  ← body / undercarriage / tracks
│      flow_lower computed    │
│                             │
└─────────────────────────────┘  ← bbox bottom (y2)
```

**Decision table:**

| flow_upper | flow_lower | motion_source | State |
|---|---|---|---|
| HIGH (>1.5) | LOW (<0.5) | `arm_only` | ✅ ACTIVE |
| HIGH | HIGH | `full_body` | ✅ ACTIVE |
| LOW | HIGH | `body_only` | ✅ ACTIVE |
| LOW | LOW | `none` | ❌ INACTIVE |

**Tuning note:** The `flow_threshold` (1.5 px/frame) and `inactive_threshold` (0.5)
are environment variables. For higher-resolution video or longer focal lengths,
increase `FLOW_THRESHOLD`. For very slow equipment, decrease it.

---

## Activity Classification Logic

```
INPUT: (flow_upper, flow_lower, motion_source, centroid_dy, lateral_motion)
       averaged over 8-frame sliding window

IF motion_source == "none" AND flow_upper < 0.8:
    → WAITING

ELIF motion_source in ["arm_only", "full_body"] AND flow_upper ≥ 1.5:

    IF centroid_dy > 1.5 px/frame (arm moving DOWN):
        → DIGGING  (bucket entering ground)

    ELIF centroid_dy < -1.5 px/frame AND flow_upper > 2.5 (arm moving UP fast):
        → DUMPING  (lifting + releasing material)

    ELIF lateral_motion AND flow_upper ≥ 2.5 (swinging horizontally):
        → SWINGING/LOADING  (rotating to truck)

    ELSE:
        → DIGGING  (default active state for arm motion)

ELSE:
    → WAITING
```

The 8-frame history window smooths over single-frame noise from camera jitter,
passing vehicles, or wind-blown dust.

---

## Kafka Payload Format

```json
{
  "frame_id": 450,
  "equipment_id": "EX-001",
  "equipment_class": "excavator",
  "track_id": 1,
  "timestamp": "00:00:15.000",
  "utilization": {
    "current_state": "ACTIVE",
    "current_activity": "DIGGING",
    "motion_source": "arm_only"
  },
  "flow_metrics": {
    "flow_upper": 3.241,
    "flow_lower": 0.182
  },
  "bbox": { "x1": 120, "y1": 80, "x2": 420, "y2": 380 },
  "time_analytics": {
    "total_tracked_seconds": 15.0,
    "total_active_seconds": 12.5,
    "total_idle_seconds": 2.5,
    "utilization_percent": 83.3
  }
}
```

---

## Database Schema

**Main table:** `equipment_events` (TimescaleDB hypertable, partitioned by `time`)

**Continuous aggregate:** `equipment_utilization_1min` — pre-computed per-minute
utilization averages, refreshed every 30 seconds.

**Helper view:** `latest_equipment_status` — latest state per machine for dashboard
cold-start queries.

**Retention:** Raw events kept 30 days; aggregates kept indefinitely.

---

## Project Structure

```
eaglevision/
├── cv_service/
│   ├── main.py              # Pipeline entry point
│   ├── detector.py          # YOLOv8 + ByteTrack
│   ├── flow_analyzer.py     # Farneback optical flow + region analysis
│   ├── classifier.py        # Activity state machine
│   ├── tracker_state.py     # Per-machine time accumulation
│   ├── kafka_pub.py         # Kafka producer
│   ├── db_writer.py         # TimescaleDB batch writer
│   ├── requirements.txt
│   └── Dockerfile
├── ui/
│   ├── app.py               # Streamlit dashboard
│   ├── requirements.txt
│   └── Dockerfile
├── db/
│   └── init.sql             # TimescaleDB schema + hypertable setup
├── scripts/
│   ├── download_test_video.py
│   └── run_local.py         # Local run without Docker
├── tests/
│   └── test_core.py         # 14 unit tests
├── data/                    # Place input.mp4 here (gitignored)
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Environment Variables

Copy `.env.example` to `.env` to override defaults:

```bash
VIDEO_SOURCE=data/input.mp4
YOLO_MODEL=yolov8m.pt
FLOW_THRESHOLD=1.5
INACTIVE_THRESHOLD=0.5
TARGET_FPS=15
KAFKA_BOOTSTRAP_SERVERS=kafka:29092
KAFKA_TOPIC=equipment_events
DB_HOST=timescaledb
DB_PORT=5432
DB_NAME=eaglevision
DB_USER=eagle
DB_PASSWORD=eagle_secret
```

---

*Built by Fady Youssif — [Portfolio](https://fady-youssif-portfolio.netlify.app) · [GitHub](https://github.com/fady278-gsh) · [LinkedIn](https://linkedin.com/in/fady-youssif)*
