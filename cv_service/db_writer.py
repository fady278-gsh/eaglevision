"""
db_writer.py  –  TimescaleDB (PostgreSQL) Sink
================================================
Writes every equipment event to TimescaleDB using a connection pool.
Batch inserts every N events for performance without sacrificing latency.
"""

from __future__ import annotations
import logging
import threading
from datetime import datetime, timezone
from typing import List

log = logging.getLogger("db_writer")

BATCH_SIZE = 20   # insert every N events


class DBWriter:
    """
    Buffers events and batch-inserts to TimescaleDB.
    Thread-safe via internal lock.
    """

    _INSERT_SQL = """
    INSERT INTO equipment_events (
        time, frame_id, equipment_id, equipment_class, track_id,
        current_state, current_activity, motion_source,
        flow_upper, flow_lower,
        bbox_x1, bbox_y1, bbox_x2, bbox_y2,
        total_tracked_secs, total_active_secs, total_idle_secs, utilization_percent
    ) VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s, %s
    )
    """

    def __init__(self, config: dict):
        self._config  = config
        self._buffer: List[tuple] = []
        self._lock    = threading.Lock()
        self._conn    = None
        self._connect()

    def _connect(self):
        try:
            import psycopg2
            from psycopg2 import pool
            self._pool = psycopg2.pool.SimpleConnectionPool(
                minconn = 1,
                maxconn = 3,
                host     = self._config["db_host"],
                port     = self._config["db_port"],
                dbname   = self._config["db_name"],
                user     = self._config["db_user"],
                password = self._config["db_password"],
            )
            log.info("TimescaleDB connection pool created.")
        except ImportError:
            log.warning("psycopg2 not installed. DB writes disabled.")
            self._pool = None
        except Exception as e:
            log.warning(f"DB connection failed: {e}. DB writes disabled.")
            self._pool = None

    def insert(self, payload: dict) -> None:
        if self._pool is None:
            return
        util = payload["utilization"]
        flow = payload.get("flow_metrics", {})
        bbox = payload.get("bbox", {})
        ta   = payload["time_analytics"]

        row = (
            datetime.now(timezone.utc),
            payload["frame_id"],
            payload["equipment_id"],
            payload["equipment_class"],
            payload["track_id"],
            util["current_state"],
            util["current_activity"],
            util["motion_source"],
            flow.get("flow_upper", 0.0),
            flow.get("flow_lower", 0.0),
            bbox.get("x1", 0), bbox.get("y1", 0),
            bbox.get("x2", 0), bbox.get("y2", 0),
            ta["total_tracked_seconds"],
            ta["total_active_seconds"],
            ta["total_idle_seconds"],
            ta["utilization_percent"],
        )

        with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= BATCH_SIZE:
                self._flush()

    def _flush(self):
        """Must be called with self._lock held."""
        if not self._buffer or self._pool is None:
            return
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.executemany(self._INSERT_SQL, self._buffer)
            conn.commit()
            log.debug(f"DB: inserted batch of {len(self._buffer)} rows.")
            self._buffer.clear()
        except Exception as e:
            conn.rollback()
            log.error(f"DB insert failed: {e}")
        finally:
            self._pool.putconn(conn)

    def close(self):
        with self._lock:
            self._flush()
        if self._pool:
            self._pool.closeall()
            log.info("DB pool closed.")
