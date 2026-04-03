"""
classifier.py  –  Activity Classification via Temporal State Machine
=====================================================================
Classifies each tracked machine into one of 4 activities:
    DIGGING  |  SWINGING/LOADING  |  DUMPING  |  WAITING

Approach: Rule-based state machine with a short sliding-window history.
No training data required → works zero-shot with optical flow features.

Why rule-based over learned?
  - Construction activity duration is highly variable → hard to window-size
  - Flow features + bbox trajectory encode activity sufficiently for this scale
  - Zero labelled video data available at prototype stage
  - Easily replaceable with an LSTM head when labelled data becomes available
"""

from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple

log = logging.getLogger("classifier")

# ── Activity constants ────────────────────────────────────────────────────
DIGGING          = "DIGGING"
SWINGING_LOADING = "SWINGING/LOADING"
DUMPING          = "DUMPING"
WAITING          = "WAITING"

# ── Thresholds ────────────────────────────────────────────────────────────
ARM_ACTIVE_THR   = 1.5    # flow_upper px/frame → arm is definitely moving
ARM_SWING_THR    = 2.5    # high lateral motion → swing/loading
BODY_MOVE_THR    = 1.2    # body/tracks moving threshold

HISTORY_LEN      = 8      # frames to smooth decisions


@dataclass
class _MachineHistory:
    bbox_history      : Deque[Tuple[int,int,int,int]] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    upper_flow_hist   : Deque[float]                  = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    lower_flow_hist   : Deque[float]                  = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    motion_src_hist   : Deque[str]                    = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    last_activity     : str                           = WAITING


class ActivityClassifier:
    """
    Stateful per-track activity classifier.

    Temporal smoothing via majority-vote over history window eliminates
    single-frame noise (e.g., camera vibration spike, occlusion).
    """

    def __init__(self):
        self._history: Dict[int, _MachineHistory] = {}

    def classify(
        self,
        track_id    : int,
        bbox        : Tuple[int, int, int, int],
        flow_upper  : float,
        flow_lower  : float,
        motion_source: str,
    ) -> str:

        if track_id not in self._history:
            self._history[track_id] = _MachineHistory()

        h = self._history[track_id]
        h.bbox_history.append(bbox)
        h.upper_flow_hist.append(flow_upper)
        h.lower_flow_hist.append(flow_lower)
        h.motion_src_hist.append(motion_source)

        # Wait for at least 3 frames before deciding
        if len(h.bbox_history) < 3:
            return h.last_activity

        # ── Smoothed features ─────────────────────────────────────────────
        avg_upper  = sum(h.upper_flow_hist) / len(h.upper_flow_hist)
        avg_lower  = sum(h.lower_flow_hist) / len(h.lower_flow_hist)
        src_counts = {s: list(h.motion_src_hist).count(s) for s in set(h.motion_src_hist)}
        dominant_src = max(src_counts, key=src_counts.get)

        # ── BBox motion features ─────────────────────────────────────────
        bboxes = list(h.bbox_history)
        cx_delta, cy_delta = self._centroid_delta(bboxes)
        lateral_motion = abs(cx_delta) > 3     # pixels/frame horizontal
        vertical_motion = cy_delta              # positive = moving down in frame

        # ── Decision tree ────────────────────────────────────────────────
        activity = self._decide(
            avg_upper, avg_lower, dominant_src,
            lateral_motion, vertical_motion
        )

        h.last_activity = activity
        return activity

    # ── Core decision logic ───────────────────────────────────────────────
    @staticmethod
    def _decide(
        avg_upper   : float,
        avg_lower   : float,
        dominant_src: str,
        lateral_motion: bool,
        vertical_motion: float,
    ) -> str:

        if dominant_src == "none" and avg_upper < 0.8 and avg_lower < 0.8:
            return WAITING

        # Machine driving (body moving, arm quiet)
        if dominant_src == "body_only":
            return WAITING   # treat repositioning truck as waiting for now

        # Arm is active (arm_only or full_body → excavator working)
        if dominant_src in ("arm_only", "full_body") and avg_upper >= ARM_ACTIVE_THR:
            # Downward arm motion → digging bucket going into ground
            if vertical_motion > 1.5:
                return DIGGING
            # Upward arm motion → lifting + lateral → dumping
            if vertical_motion < -1.5 and avg_upper > ARM_SWING_THR:
                return DUMPING
            # Lateral swing → swinging to truck or swinging back
            if lateral_motion and avg_upper >= ARM_SWING_THR:
                return SWINGING_LOADING
            # High arm flow but no clear direction → defaulting to digging
            if avg_upper >= ARM_ACTIVE_THR:
                return DIGGING

        return WAITING

    # ── Centroid trajectory ───────────────────────────────────────────────
    @staticmethod
    def _centroid_delta(bboxes) -> Tuple[float, float]:
        """Mean per-frame centroid displacement over window."""
        if len(bboxes) < 2:
            return 0.0, 0.0
        cxs = [(b[0] + b[2]) / 2 for b in bboxes]
        cys = [(b[1] + b[3]) / 2 for b in bboxes]
        dx  = (cxs[-1] - cxs[0]) / max(len(bboxes) - 1, 1)
        dy  = (cys[-1] - cys[0]) / max(len(bboxes) - 1, 1)
        return dx, dy

    def reset(self, track_id: int):
        """Call when a track is lost for > N frames."""
        self._history.pop(track_id, None)
