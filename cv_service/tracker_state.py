"""
tracker_state.py  –  Per-Machine Time Accumulation
====================================================
Maintains running counters of ACTIVE / IDLE seconds per equipment_id.
Thread-safe for potential multi-stream use.
"""

from __future__ import annotations
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict

log = logging.getLogger("tracker_state")

WORK_SHIFT_HOURS = 8.0   # target shift duration in hours


@dataclass
class _MachineStats:
    total_tracked : float = 0.0
    total_active  : float = 0.0
    total_idle    : float = 0.0
    last_state    : str   = "INACTIVE"

    @property
    def utilization_pct(self) -> float:
        if self.total_tracked < 0.001:
            return 0.0
        return (self.total_active / self.total_tracked) * 100.0

    @property
    def remaining_shift_secs(self) -> float:
        shift_secs = WORK_SHIFT_HOURS * 3600
        return max(0.0, shift_secs - self.total_tracked)

    def to_dict(self) -> dict:
        return {
            "total_tracked"    : round(self.total_tracked, 2),
            "total_active"     : round(self.total_active, 2),
            "total_idle"       : round(self.total_idle, 2),
            "utilization_pct"  : round(self.utilization_pct, 2),
            "remaining_shift_s": round(self.remaining_shift_secs, 2),
        }


class MachineStateTracker:
    """
    Thread-safe per-machine time accumulator.

    Usage
    -----
    stats = tracker.update("EX-001", "ACTIVE", dt_seconds=0.033)
    """

    def __init__(self):
        self._machines: Dict[str, _MachineStats] = {}
        self._lock = threading.Lock()

    def update(self, equipment_id: str, state: str, dt_secs: float) -> dict:
        with self._lock:
            if equipment_id not in self._machines:
                self._machines[equipment_id] = _MachineStats()
                log.info(f"New machine registered: {equipment_id}")

            m = self._machines[equipment_id]
            m.total_tracked += dt_secs
            if state == "ACTIVE":
                m.total_active += dt_secs
            else:
                m.total_idle   += dt_secs
            m.last_state = state
            return m.to_dict()

    def get_all(self) -> dict:
        with self._lock:
            return {eid: m.to_dict() for eid, m in self._machines.items()}

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for m in self._machines.values() if m.last_state == "ACTIVE")

    def reset(self, equipment_id: str):
        with self._lock:
            self._machines.pop(equipment_id, None)

    def summary_log(self):
        """Log a shift summary for all machines."""
        with self._lock:
            log.info("═══════ SHIFT SUMMARY ═══════")
            for eid, m in self._machines.items():
                log.info(
                    f"  {eid}: tracked={m.total_tracked:.0f}s "
                    f"active={m.total_active:.0f}s ({m.utilization_pct:.1f}%) "
                    f"idle={m.total_idle:.0f}s"
                )
