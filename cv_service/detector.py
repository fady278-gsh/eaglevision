"""
detector.py  –  YOLOv8 + ByteTrack Equipment Detection & Tracking
==================================================================
Wraps Ultralytics YOLOv8 with persistent tracking (ByteTrack).
Returns structured Detection objects per frame.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

log = logging.getLogger("detector")

# ── COCO / custom class mapping ───────────────────────────────────────────
# YOLOv8 COCO classes that map to construction equipment:
COCO_EQUIPMENT_CLASSES = {
    7  : "dump_truck",   # truck
    # Custom fine-tuned classes (add after fine-tuning):
    # 80: "excavator",
    # 81: "bulldozer",
    # 82: "crane",
}

CUSTOM_CLASSES = {
    0: "excavator",
    1: "dump_truck",
    2: "bulldozer",
    3: "crane",
    4: "wheel_loader",
}

# Minimum confidence for detection
MIN_CONFIDENCE = 0.35


@dataclass
class Detection:
    track_id  : int
    cls_id    : int
    cls_name  : str
    confidence: float
    bbox      : Tuple[int, int, int, int]   # x1, y1, x2, y2 (absolute pixels)


class EquipmentDetector:
    """
    Detects and tracks construction equipment in a video frame.

    Strategy
    --------
    1. Run YOLOv8 with ByteTrack (persist=True) so track_ids are stable.
    2. Filter only equipment classes.
    3. Return Detection objects with stable track_id = Re-ID solution.

    Re-ID approach
    --------------
    ByteTrack (built into Ultralytics) handles short occlusions and re-entries
    by using Kalman filter prediction + IoU + appearance cost matrix.
    For long disappearances (>30 frames), a new ID is assigned — acceptable
    for fixed-camera construction sites where equipment rarely leaves frame.
    """

    def __init__(self, model_path: str = "yolov8m.pt"):
        try:
            from ultralytics import YOLO
            self.model      = YOLO(model_path)
            self.is_custom  = self._is_custom_model(model_path)
            self.class_map  = CUSTOM_CLASSES if self.is_custom else COCO_EQUIPMENT_CLASSES
            log.info(f"YOLO model loaded: {model_path}  custom={self.is_custom}")
        except ImportError:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model '{model_path}': {e}")

    @staticmethod
    def _is_custom_model(path: str) -> bool:
        """Heuristic: custom model filenames differ from yolov8*.pt."""
        import os
        base = os.path.basename(path).lower()
        return not (base.startswith("yolov8") and base.endswith(".pt") and len(base) < 14)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection + tracking on a single BGR frame.
        Returns list of Detection objects (may be empty).
        """
        results = self.model.track(
            frame,
            persist   = True,          # keeps track state across frames
            tracker   = "bytetrack.yaml",
            conf      = MIN_CONFIDENCE,
            iou       = 0.45,
            verbose   = False,
        )

        detections: List[Detection] = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id     = int(boxes.cls[i].item())
            conf       = float(boxes.conf[i].item())
            track_id   = int(boxes.id[i].item()) if boxes.id is not None else -1

            # Filter only equipment classes
            if cls_id not in self.class_map:
                continue

            cls_name = self.class_map[cls_id]
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

            # Sanity-check bbox size (skip micro-detections)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            detections.append(Detection(
                track_id   = track_id,
                cls_id     = cls_id,
                cls_name   = cls_name,
                confidence = conf,
                bbox       = (x1, y1, x2, y2),
            ))

        return detections
