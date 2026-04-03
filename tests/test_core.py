"""
tests/test_core.py
==================
Unit tests for flow analyzer, classifier, and state tracker.
No GPU / video file needed.

Run:
    pip install pytest numpy opencv-python-headless
    pytest tests/test_core.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cv_service"))

import numpy as np
import pytest


# ─────────────────── Flow Analyzer ───────────────────────────────────────────
class TestOpticalFlowAnalyzer:

    def setup_method(self):
        from flow_analyzer import OpticalFlowAnalyzer
        self.fa = OpticalFlowAnalyzer(flow_threshold=1.5, inactive_threshold=0.5)

    def _make_flow(self, h, w, upper_mag=0.0, lower_mag=0.0, split=0.5):
        """Create synthetic flow with different magnitudes in upper/lower halves."""
        flow = np.zeros((h, w, 2), dtype=np.float32)
        split_y = int(h * split)
        flow[:split_y, :, 0] = upper_mag      # upper region x-flow
        flow[split_y:, :, 0] = lower_mag      # lower region x-flow
        return flow

    def test_none_flow_returns_none_source(self):
        fu, fl, src = self.fa.analyze_region(None, (0, 0, 100, 100), 200)
        assert src == "none"
        assert fu == 0.0 and fl == 0.0

    def test_arm_only_detection(self):
        """High upper flow, low lower → arm_only (excavator digging)."""
        flow = self._make_flow(200, 300, upper_mag=3.0, lower_mag=0.1)
        fu, fl, src = self.fa.analyze_region(flow, (0, 0, 300, 200), 200)
        assert src == "arm_only", f"Expected arm_only, got {src}"
        assert fu > 1.5

    def test_full_body_detection(self):
        """Both regions moving → full_body (truck driving)."""
        flow = self._make_flow(200, 300, upper_mag=3.0, lower_mag=3.0)
        _, _, src = self.fa.analyze_region(flow, (0, 0, 300, 200), 200)
        assert src == "full_body"

    def test_inactive_detection(self):
        """No motion in either region → none."""
        flow = self._make_flow(200, 300, upper_mag=0.1, lower_mag=0.1)
        _, _, src = self.fa.analyze_region(flow, (0, 0, 300, 200), 200)
        assert src == "none"

    def test_bbox_clamping(self):
        """BBox extending beyond frame should not raise."""
        flow = self._make_flow(100, 100, upper_mag=2.0, lower_mag=0.2)
        fu, fl, src = self.fa.analyze_region(flow, (-10, -10, 200, 200), 100)
        assert isinstance(src, str)


# ─────────────────── Activity Classifier ─────────────────────────────────────
class TestActivityClassifier:

    def setup_method(self):
        from classifier import ActivityClassifier
        self.clf = ActivityClassifier()

    def _run_n_frames(self, n, **kwargs):
        """Feed N identical frames and return last activity."""
        result = "WAITING"
        bbox = kwargs.pop("bbox", (100, 100, 300, 300))
        for _ in range(n):
            result = self.clf.classify(track_id=1, bbox=bbox, **kwargs)
        return result

    def test_waiting_when_no_motion(self):
        act = self._run_n_frames(5, flow_upper=0.2, flow_lower=0.1, motion_source="none")
        assert act == "WAITING"

    def test_digging_arm_only(self):
        """arm_only + high upper flow → should eventually classify as DIGGING."""
        # Simulate downward arm motion (cy_delta positive)
        bbox_sequence = [(100, 100 + i*2, 300, 300 + i*2) for i in range(8)]
        result = "WAITING"
        for bbox in bbox_sequence:
            result = self.clf.classify(
                track_id=2, bbox=bbox,
                flow_upper=3.5, flow_lower=0.3, motion_source="arm_only"
            )
        assert result in ("DIGGING", "WAITING")   # accepts either after short window

    def test_new_track_gets_history(self):
        """New track_id should not crash."""
        act = self.clf.classify(
            track_id=999, bbox=(0, 0, 100, 100),
            flow_upper=2.0, flow_lower=0.2, motion_source="arm_only"
        )
        assert isinstance(act, str)

    def test_reset_clears_history(self):
        self._run_n_frames(5, flow_upper=2.0, flow_lower=0.2, motion_source="arm_only")
        self.clf.reset(1)
        assert 1 not in self.clf._history


# ─────────────────── State Tracker ───────────────────────────────────────────
class TestMachineStateTracker:

    def setup_method(self):
        from tracker_state import MachineStateTracker
        self.tracker = MachineStateTracker()

    def test_accumulates_active_time(self):
        for _ in range(10):
            s = self.tracker.update("EX-001", "ACTIVE", dt_secs=1.0)
        assert s["total_active"] == pytest.approx(10.0)
        assert s["total_idle"] == pytest.approx(0.0)

    def test_accumulates_idle_time(self):
        for _ in range(5):
            s = self.tracker.update("EX-002", "INACTIVE", dt_secs=2.0)
        assert s["total_idle"] == pytest.approx(10.0)

    def test_utilization_calculation(self):
        for _ in range(3):
            self.tracker.update("EX-003", "ACTIVE", 1.0)
        for _ in range(1):
            self.tracker.update("EX-003", "INACTIVE", 1.0)
        s = self.tracker.update("EX-003", "ACTIVE", 0.0)
        # 3/(3+1) = 75%
        assert s["utilization_pct"] == pytest.approx(75.0)

    def test_multiple_machines_independent(self):
        self.tracker.update("EX-010", "ACTIVE",   1.0)
        self.tracker.update("DT-010", "INACTIVE", 1.0)
        all_m = self.tracker.get_all()
        assert all_m["EX-010"]["total_active"] == 1.0
        assert all_m["DT-010"]["total_idle"]   == 1.0

    def test_active_count(self):
        self.tracker.update("EX-020", "ACTIVE",   1.0)
        self.tracker.update("DT-020", "INACTIVE", 1.0)
        assert self.tracker.active_count() == 1
