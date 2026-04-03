#!/usr/bin/env python3
"""
scripts/run_local.py
====================
Runs the full pipeline locally WITHOUT Docker:
  - Skips Kafka (prints payloads to stdout)
  - Skips TimescaleDB
  - Saves annotated output video to data/output.mp4
  - Prints a shift summary at the end

Perfect for quick prototyping and assessment demo.

Usage:
    cd eaglevision
    pip install -r cv_service/requirements.txt
    python scripts/run_local.py --video data/input.mp4
"""

import sys
import os
import argparse
import json
import logging

# Add cv_service to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cv_service"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")

# Patch out Kafka and DB for local run
import kafka_pub
import db_writer

class _NoOpPublisher:
    def publish(self, payload):
        print(json.dumps(payload, indent=2)[:300] + " …")   # print truncated
    def close(self): pass

class _NoOpDB:
    def insert(self, payload): pass
    def close(self): pass

# Monkey-patch before main imports
kafka_pub.KafkaPublisher = lambda *a, **kw: _NoOpPublisher()
db_writer.DBWriter       = lambda *a, **kw: _NoOpDB()

from main import run_pipeline, CONFIG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EagleVision Local Test")
    parser.add_argument("--video",   default="data/input.mp4")
    parser.add_argument("--display", action="store_true",
                        help="Show live window (needs display)")
    parser.add_argument("--fps",     type=int, default=15)
    args = parser.parse_args()

    CONFIG["video_source"]  = args.video
    CONFIG["display_local"] = args.display
    CONFIG["target_fps"]    = args.fps
    CONFIG["output_video"]  = "data/output.mp4"
    # Disable external services
    CONFIG["kafka_servers"] = "DISABLED"
    CONFIG["db_host"]       = "DISABLED"

    run_pipeline(CONFIG)
