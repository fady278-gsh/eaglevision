-- ─────────────────────────────────────────────────────────
-- EagleVision  –  TimescaleDB Schema
-- ─────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ── Main events hypertable ────────────────────────────────
CREATE TABLE IF NOT EXISTS equipment_events (
    time                TIMESTAMPTZ     NOT NULL,
    frame_id            INTEGER         NOT NULL,
    equipment_id        TEXT            NOT NULL,
    equipment_class     TEXT            NOT NULL,
    track_id            INTEGER         NOT NULL,
    current_state       TEXT            NOT NULL,   -- ACTIVE | INACTIVE
    current_activity    TEXT            NOT NULL,   -- DIGGING | SWINGING | DUMPING | WAITING
    motion_source       TEXT            NOT NULL,   -- arm_only | full_body | none
    flow_upper          FLOAT,
    flow_lower          FLOAT,
    bbox_x1             INTEGER,
    bbox_y1             INTEGER,
    bbox_x2             INTEGER,
    bbox_y2             INTEGER,
    total_tracked_secs  FLOAT           NOT NULL DEFAULT 0,
    total_active_secs   FLOAT           NOT NULL DEFAULT 0,
    total_idle_secs     FLOAT           NOT NULL DEFAULT 0,
    utilization_percent FLOAT           NOT NULL DEFAULT 0
);

SELECT create_hypertable('equipment_events', 'time', if_not_exists => TRUE);

-- ── Indexes for fast dashboard queries ──────────────────
CREATE INDEX IF NOT EXISTS idx_eq_events_equip_id
    ON equipment_events (equipment_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_eq_events_state
    ON equipment_events (current_state, time DESC);

-- ── Continuous Aggregate: per-minute utilization ─────────
CREATE MATERIALIZED VIEW IF NOT EXISTS equipment_utilization_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time)  AS bucket,
    equipment_id,
    equipment_class,
    AVG(utilization_percent)        AS avg_utilization,
    MAX(total_active_secs)          AS max_active_secs,
    MAX(total_idle_secs)            AS max_idle_secs,
    COUNT(*)                        AS frame_count
FROM equipment_events
GROUP BY bucket, equipment_id, equipment_class
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'equipment_utilization_1min',
    start_offset  => INTERVAL '10 minutes',
    end_offset    => INTERVAL '10 seconds',
    schedule_interval => INTERVAL '30 seconds',
    if_not_exists => TRUE
);

-- ── Retention policy: keep 30 days raw data ─────────────
SELECT add_retention_policy(
    'equipment_events',
    INTERVAL '30 days',
    if_not_exists => TRUE
);

-- ── Helper view: latest state per machine ────────────────
CREATE OR REPLACE VIEW latest_equipment_status AS
SELECT DISTINCT ON (equipment_id)
    equipment_id,
    equipment_class,
    current_state,
    current_activity,
    motion_source,
    total_active_secs,
    total_idle_secs,
    utilization_percent,
    time AS last_seen
FROM equipment_events
ORDER BY equipment_id, time DESC;
