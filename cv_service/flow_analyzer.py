"""
flow_analyzer.py  –  Region-based Optical Flow for Articulated Motion
======================================================================
Solves the core challenge: detecting ACTIVE state when ONLY the arm/boom
of an excavator moves while the tracks remain stationary.

Algorithm
---------
Farneback Dense Optical Flow on grayscale frames.
For each detected equipment bounding box:
  1. Split bbox vertically → upper_region (arm/boom) & lower_region (body/tracks)
  2. Compute mean flow magnitude per region
  3. Classify motion_source:
       - "arm_only"   → upper HIGH,  lower LOW
       - "full_body"  → both HIGH
       - "body_only"  → upper LOW,   lower HIGH
       - "none"       → both LOW

Why Farneback over Lucas-Kanade?
  Dense flow gives per-pixel vectors → reliable even without good feature points.
  On fixed-camera footage, background flow ≈ 0, so threshold is clean.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple
import cv2
import numpy as np

log = logging.getLogger("flow_analyzer")


class OpticalFlowAnalyzer:
    """
    Computes dense optical flow and analyses motion per equipment region.

    Parameters
    ----------
    flow_threshold     : Mean magnitude (px/frame) above which region is "moving"
    inactive_threshold : Same but lower bound for "definitely inactive"
    upper_ratio        : Fraction of bbox height assigned to upper (arm) region
    """

    # Farneback hyperparameters (tuned for 720p construction footage)
    _FB_PARAMS = dict(
        pyr_scale  = 0.5,
        levels     = 3,
        winsize    = 13,     # larger → smoother but slower
        iterations = 3,
        poly_n     = 5,
        poly_sigma = 1.2,
        flags      = 0,
    )

    def __init__(
        self,
        flow_threshold    : float = 1.5,
        inactive_threshold: float = 0.5,
        upper_ratio       : float = 0.55,   # top 55% = arm/boom region
    ):
        self.flow_threshold     = flow_threshold
        self.inactive_threshold = inactive_threshold
        self.upper_ratio        = upper_ratio

    # ── Dense flow computation ────────────────────────────────────────────
    def compute_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> np.ndarray:
        """
        Returns HxWx2 flow array (dx, dy per pixel).
        Applies slight Gaussian blur to reduce sensor noise.
        """
        prev_blur = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        flow = cv2.calcOpticalFlowFarneback(
            prev_blur, curr_blur, None, **self._FB_PARAMS
        )
        return flow

    # ── Regional analysis ────────────────────────────────────────────────
    def analyze_region(
        self,
        flow  : Optional[np.ndarray],
        bbox  : Tuple[int, int, int, int],
        frame_h: int,
    ) -> Tuple[float, float, str]:
        """
        Parameters
        ----------
        flow    : HxWx2 optical flow array (or None for first frame)
        bbox    : (x1, y1, x2, y2) absolute pixel coords
        frame_h : frame height for clamping

        Returns
        -------
        (flow_upper, flow_lower, motion_source)
        """
        if flow is None:
            return 0.0, 0.0, "none"

        x1, y1, x2, y2 = bbox
        # Clamp to frame bounds
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(flow.shape[1], x2);  y2 = min(flow.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0, "none"

        split_y = int(y1 + (y2 - y1) * self.upper_ratio)

        # Upper region (arm / boom)
        flow_u = self._mean_magnitude(flow[y1:split_y, x1:x2])

        # Lower region (body / tracks)
        flow_l = self._mean_magnitude(flow[split_y:y2, x1:x2])

        motion_source = self._classify_source(flow_u, flow_l)
        return float(flow_u), float(flow_l), motion_source

    # ── Internals ────────────────────────────────────────────────────────
    @staticmethod
    def _mean_magnitude(flow_crop: np.ndarray) -> float:
        if flow_crop.size == 0:
            return 0.0
        mag = np.sqrt(flow_crop[..., 0] ** 2 + flow_crop[..., 1] ** 2)
        return float(np.mean(mag))

    def _classify_source(self, flow_u: float, flow_l: float) -> str:
        upper_active = flow_u > self.flow_threshold
        lower_active = flow_l > self.flow_threshold
        upper_def_inactive = flow_u < self.inactive_threshold
        lower_def_inactive = flow_l < self.inactive_threshold

        if upper_active and lower_active:
            return "full_body"
        if upper_active and not lower_active:
            return "arm_only"       # excavator digging with stationary tracks
        if lower_active and not upper_active:
            return "body_only"      # truck moving, arm idle
        return "none"

    # ── Visualization helper (for debugging) ─────────────────────────────
    def flow_to_bgr(self, flow: np.ndarray) -> np.ndarray:
        """Convert flow to HSV-coloured BGR image for visual debugging."""
        h, w = flow.shape[:2]
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
