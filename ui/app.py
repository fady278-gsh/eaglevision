"""
app.py  –  EagleVision Streamlit Dashboard
==========================================
Real-time equipment utilization dashboard consuming from Kafka.
Displays:
  1. Annotated video feed (base64 JPEG frames via Kafka)
  2. Per-machine live status cards
  3. Utilization gauges & time counters
  4. Historical utilization chart (from TimescaleDB)
"""

import os
import json
import time
import base64
import logging
import threading
from collections import defaultdict, deque
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

log = logging.getLogger("ui")

# ─────────────────── Streamlit page config ──────────────────────────────────
st.set_page_config(
    page_title = "EagleVision | Equipment Analytics",
    page_icon  = "🦅",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ─────────────────── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;500;700&display=swap');

:root {
    --bg:       #0a0d14;
    --surface:  #111520;
    --border:   #1e2433;
    --active:   #00e5a0;
    --inactive: #3b7bff;
    --warning:  #ffb340;
    --danger:   #ff4d6d;
    --text:     #c8d0e0;
    --muted:    #4a5468;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif;
}

.ev-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 0 8px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.ev-logo { font-size: 2rem; }
.ev-title { font-size: 1.6rem; font-weight: 700; letter-spacing: 0.04em; color: #fff; }
.ev-subtitle { font-size: 0.78rem; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; }

.machine-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.machine-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 3px 0 0 3px;
}
.machine-card.active::before  { background: var(--active); }
.machine-card.inactive::before { background: var(--inactive); }

.mc-id     { font-family: 'IBM Plex Mono', monospace; font-size: 1rem; font-weight: 600; color: #fff; }
.mc-state  { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 4px; margin-left: 8px; }
.mc-state.active   { background: rgba(0,229,160,0.15); color: var(--active); }
.mc-state.inactive { background: rgba(59,123,255,0.15); color: var(--inactive); }
.mc-activity { font-size: 0.85rem; color: var(--muted); margin-top: 4px; }
.mc-util   { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 600; color: var(--active); }

.util-bar-bg   { background: var(--border); border-radius: 4px; height: 6px; margin-top: 8px; }
.util-bar-fill { height: 6px; border-radius: 4px; background: linear-gradient(90deg, var(--active), #00b8d9); transition: width 0.4s ease; }

.stat-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}
.stat-val  { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: #fff; }
.stat-lbl  { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }

.activity-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
}
.badge-digging  { background: rgba(255,179,64,0.15);  color: var(--warning); }
.badge-swinging { background: rgba(0,184,217,0.15);   color: #00b8d9; }
.badge-dumping  { background: rgba(255,77,109,0.15);  color: var(--danger); }
.badge-waiting  { background: rgba(74,84,104,0.15);   color: var(--muted); }

.section-hdr {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 20px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* Streamlit widget overrides */
.stMetric label { color: var(--muted) !important; font-size: 0.75rem !important; }
.stMetric value { color: #fff !important; }
div[data-testid="metric-container"] { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────── Config ──────────────────────────────────────────────────
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC   = os.getenv("KAFKA_TOPIC", "equipment_events")
DB_CONFIG     = {
    "host"    : os.getenv("DB_HOST",     "localhost"),
    "port"    : int(os.getenv("DB_PORT", "5432")),
    "dbname"  : os.getenv("DB_NAME",     "eaglevision"),
    "user"    : os.getenv("DB_USER",     "eagle"),
    "password": os.getenv("DB_PASSWORD", "eagle_secret"),
}
MAX_HISTORY = 120   # events per machine for chart


# ─────────────────── Session state initialisation ────────────────────────────
def _init_state():
    if "machines" not in st.session_state:
        st.session_state.machines    = {}          # equipment_id → latest payload
    if "history"  not in st.session_state:
        st.session_state.history     = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
    if "consumer" not in st.session_state:
        st.session_state.consumer    = None
    if "consuming" not in st.session_state:
        st.session_state.consuming   = False
    if "event_count" not in st.session_state:
        st.session_state.event_count = 0
    if "last_frame" not in st.session_state:
        st.session_state.last_frame  = None

_init_state()


# ─────────────────── Kafka consumer thread ───────────────────────────────────
def kafka_consumer_loop():
    try:
        from confluent_kafka import Consumer, KafkaError
    except ImportError:
        st.session_state.consuming = False
        return

    consumer = Consumer({
        "bootstrap.servers"  : KAFKA_SERVERS,
        "group.id"           : "eaglevision_ui",
        "auto.offset.reset"  : "latest",
        "enable.auto.commit" : True,
    })
    consumer.subscribe([KAFKA_TOPIC])
    st.session_state.consumer  = consumer
    st.session_state.consuming = True

    while st.session_state.consuming:
        msg = consumer.poll(timeout=0.5)
        if msg is None:
            continue
        if msg.error():
            continue
        try:
            payload = json.loads(msg.value().decode("utf-8"))
            eid     = payload.get("equipment_id", "UNK")
            st.session_state.machines[eid]    = payload
            st.session_state.history[eid].append({
                "ts"  : datetime.utcnow(),
                "util": payload["time_analytics"]["utilization_percent"],
                "state": payload["utilization"]["current_state"],
            })
            st.session_state.event_count += 1

            # Store latest annotated frame if present
            if "frame_b64" in payload:
                st.session_state.last_frame = payload["frame_b64"]
        except Exception:
            pass

    consumer.close()


# ─────────────────── DB query helpers ────────────────────────────────────────
@st.cache_data(ttl=30)
def query_history_db():
    try:
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql("""
            SELECT
                time_bucket('30 seconds', time) AS bucket,
                equipment_id,
                AVG(utilization_percent) AS util_pct,
                MAX(total_active_secs)   AS active_secs,
                MAX(total_idle_secs)     AS idle_secs
            FROM equipment_events
            WHERE time > now() - INTERVAL '10 minutes'
            GROUP BY bucket, equipment_id
            ORDER BY bucket
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────── Rendering helpers ───────────────────────────────────────
ACTIVITY_BADGE = {
    "DIGGING"        : ("badge-digging",  "⛏ DIGGING"),
    "SWINGING/LOADING":("badge-swinging", "🔄 SWINGING"),
    "DUMPING"        : ("badge-dumping",  "📤 DUMPING"),
    "WAITING"        : ("badge-waiting",  "⏸ WAITING"),
}

def fmt_seconds(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def render_machine_card(eid: str, payload: dict):
    util    = payload["utilization"]
    ta      = payload["time_analytics"]
    state   = util["current_state"]
    act     = util["current_activity"]
    pct     = ta["utilization_percent"]
    cls_state = "active" if state == "ACTIVE" else "inactive"
    badge_cls, badge_txt = ACTIVITY_BADGE.get(act, ("badge-waiting", act))

    st.markdown(f"""
    <div class="machine-card {cls_state}">
      <div style="display:flex;align-items:center;justify-content:space-between">
        <div>
          <span class="mc-id">{eid}</span>
          <span class="mc-state {cls_state}">{state}</span>
        </div>
        <span class="mc-util">{pct:.1f}%</span>
      </div>
      <div class="mc-activity">
        {payload['equipment_class'].upper()} &nbsp;·&nbsp;
        <span class="activity-badge {badge_cls}">{badge_txt}</span>
        &nbsp;·&nbsp; {util['motion_source'].replace('_',' ')}
      </div>
      <div class="util-bar-bg">
        <div class="util-bar-fill" style="width:{min(pct,100):.1f}%"></div>
      </div>
      <div style="display:flex;gap:20px;margin-top:10px">
        <div class="stat-box" style="flex:1">
          <div class="stat-val" style="color:var(--active)">{fmt_seconds(ta['total_active_seconds'])}</div>
          <div class="stat-lbl">Working</div>
        </div>
        <div class="stat-box" style="flex:1">
          <div class="stat-val" style="color:var(--inactive)">{fmt_seconds(ta['total_idle_seconds'])}</div>
          <div class="stat-lbl">Idle</div>
        </div>
        <div class="stat-box" style="flex:1">
          <div class="stat-val">{fmt_seconds(ta['total_tracked_seconds'])}</div>
          <div class="stat-lbl">Tracked</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_util_gauge(machines: dict) -> go.Figure:
    if not machines:
        return go.Figure()

    eids  = list(machines.keys())
    utils = [m["time_analytics"]["utilization_percent"] for m in machines.values()]

    fig = go.Figure(go.Bar(
        x           = utils,
        y           = eids,
        orientation = "h",
        marker      = dict(
            color = ["#00e5a0" if u >= 70 else "#ffb340" if u >= 40 else "#ff4d6d" for u in utils],
            line  = dict(color="#1e2433", width=1),
        ),
        text        = [f"{u:.1f}%" for u in utils],
        textposition= "outside",
    ))
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(color="#c8d0e0", size=12),
        xaxis         = dict(range=[0, 110], gridcolor="#1e2433", title="Utilization %"),
        yaxis         = dict(gridcolor="#1e2433"),
        margin        = dict(l=10, r=30, t=10, b=10),
        height        = max(200, len(eids) * 60),
        showlegend    = False,
    )
    return fig


def render_history_chart(history: dict) -> go.Figure:
    fig = go.Figure()
    colors = ["#00e5a0", "#3b7bff", "#ffb340", "#ff4d6d", "#00b8d9"]
    for i, (eid, events) in enumerate(history.items()):
        if len(events) < 2:
            continue
        evlist = list(events)
        fig.add_trace(go.Scatter(
            x    = [e["ts"] for e in evlist],
            y    = [e["util"] for e in evlist],
            name = eid,
            mode = "lines",
            line = dict(color=colors[i % len(colors)], width=2),
            fill = "tozeroy",
            fillcolor = f"rgba({','.join(str(int(colors[i%len(colors)].lstrip('#')[j:j+2], 16)) for j in (0,2,4))},0.08)",
        ))
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(color="#c8d0e0", size=11),
        xaxis         = dict(gridcolor="#1e2433"),
        yaxis         = dict(gridcolor="#1e2433", range=[0, 105], title="Utilization %"),
        legend        = dict(bgcolor="rgba(0,0,0,0)"),
        margin        = dict(l=10, r=10, t=10, b=10),
        height        = 220,
    )
    return fig


# ─────────────────── Main layout ─────────────────────────────────────────────
st.markdown("""
<div class="ev-header">
  <div class="ev-logo">🦅</div>
  <div>
    <div class="ev-title">EAGLEVISION</div>
    <div class="ev-subtitle">Equipment Utilization Intelligence · Real-time</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Controls ─────────────────────────────────────────────────────────────────
ctrl_col1, ctrl_col2, ctrl_col3, _ = st.columns([1, 1, 1, 5])

with ctrl_col1:
    if st.button("▶ Connect", use_container_width=True, type="primary"):
        if not st.session_state.consuming:
            t = threading.Thread(target=kafka_consumer_loop, daemon=True)
            t.start()

with ctrl_col2:
    if st.button("⏹ Stop", use_container_width=True):
        st.session_state.consuming = False

with ctrl_col3:
    auto_refresh = st.toggle("Auto Refresh", value=True)

# ── Top KPI strip ─────────────────────────────────────────────────────────────
machines = st.session_state.machines
n_active  = sum(1 for m in machines.values() if m["utilization"]["current_state"] == "ACTIVE")
n_total   = len(machines)
avg_util  = (sum(m["time_analytics"]["utilization_percent"] for m in machines.values()) / n_total
             if n_total else 0)
events_ps = st.session_state.event_count

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("🟢 Active Machines",   f"{n_active} / {n_total}")
kpi2.metric("📊 Fleet Utilization",  f"{avg_util:.1f}%")
kpi3.metric("📡 Events Processed",   f"{events_ps:,}")
kpi4.metric("🕐 Last Update",        datetime.utcnow().strftime("%H:%M:%S") if machines else "–")

st.markdown("---")

# ── Main columns ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    # Video feed
    st.markdown('<div class="section-hdr">📹 Live Feed</div>', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    if st.session_state.last_frame:
        try:
            img_bytes = base64.b64decode(st.session_state.last_frame)
            frame_placeholder.image(img_bytes, use_column_width=True, channels="BGR")
        except Exception:
            frame_placeholder.info("Waiting for video frames …")
    else:
        frame_placeholder.info("Waiting for video frames from CV service …")

    # Utilization bar chart
    st.markdown('<div class="section-hdr">📈 Fleet Utilization</div>', unsafe_allow_html=True)
    if machines:
        st.plotly_chart(render_util_gauge(machines), use_container_width=True, config={"displayModeBar": False})

    # History chart
    if st.session_state.history:
        st.markdown('<div class="section-hdr">📉 Utilization History (last 2 min)</div>', unsafe_allow_html=True)
        st.plotly_chart(render_history_chart(st.session_state.history), use_container_width=True, config={"displayModeBar": False})

with right_col:
    st.markdown('<div class="section-hdr">🚜 Machine Status</div>', unsafe_allow_html=True)
    if machines:
        for eid, payload in sorted(machines.items()):
            render_machine_card(eid, payload)
    else:
        st.markdown("""
        <div style="text-align:center;padding:40px 20px;color:#4a5468;border:1px dashed #1e2433;border-radius:10px">
            <div style="font-size:2rem">🚧</div>
            <div style="margin-top:8px;font-size:0.85rem">Waiting for equipment detections …</div>
            <div style="font-size:0.72rem;margin-top:4px">Click ▶ Connect to start consuming</div>
        </div>
        """, unsafe_allow_html=True)

    # Per-machine detail expanders
    if machines:
        st.markdown('<div class="section-hdr">🔍 Detailed Stats</div>', unsafe_allow_html=True)
        for eid, payload in sorted(machines.items()):
            with st.expander(f"{eid}  –  {payload['equipment_class'].title()}"):
                ta = payload["time_analytics"]
                shift_secs  = 8 * 3600
                used_pct    = min(100, ta["total_tracked_seconds"] / shift_secs * 100)
                st.progress(int(ta["utilization_percent"]),
                            text=f"Utilization: {ta['utilization_percent']:.1f}%")
                st.progress(int(used_pct),
                            text=f"Shift elapsed: {used_pct:.1f}% of 8h")
                c1, c2 = st.columns(2)
                c1.metric("Working", fmt_seconds(ta["total_active_seconds"]))
                c2.metric("Idle",    fmt_seconds(ta["total_idle_seconds"]))
                idle_loss = (ta["total_idle_seconds"] / max(ta["total_tracked_seconds"], 1)) * 100
                if idle_loss > 30:
                    st.warning(f"⚠️ High idle ratio: {idle_loss:.0f}% — review scheduling")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh and st.session_state.consuming:
    time.sleep(1.5)
    st.rerun()
