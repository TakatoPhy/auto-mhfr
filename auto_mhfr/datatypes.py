"""Data structures for Auto-MHFR Analyzer & Lock."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


class ScanDirection(Enum):
    UP = auto()
    DOWN = auto()


@dataclass
class ScanPoint:
    """Single measurement during a current scan."""
    current_mA: float
    frequency_THz: float
    timestamp: float
    direction: ScanDirection


@dataclass
class ModeSegment:
    """A contiguous mode-hop-free region."""
    segment_id: int
    direction: ScanDirection
    current_start_mA: float
    current_end_mA: float
    freq_start_THz: float
    freq_end_THz: float
    points: list[ScanPoint] = field(default_factory=list)
    slope_GHz_per_mA: float = 0.0
    is_boundary_segment: bool = False  # True if at scan edge (partial)

    @property
    def current_center_mA(self) -> float:
        return (self.current_start_mA + self.current_end_mA) / 2.0

    @property
    def current_width_mA(self) -> float:
        return abs(self.current_end_mA - self.current_start_mA)

    @property
    def freq_min_THz(self) -> float:
        return min(self.freq_start_THz, self.freq_end_THz)

    @property
    def freq_max_THz(self) -> float:
        return max(self.freq_start_THz, self.freq_end_THz)

    def contains_frequency(self, freq_THz: float) -> bool:
        return self.freq_min_THz <= freq_THz <= self.freq_max_THz

    def current_for_frequency(self, freq_THz: float) -> Optional[float]:
        """Estimate current needed for a given frequency via linear interpolation."""
        if not self.contains_frequency(freq_THz):
            return None
        if abs(self.slope_GHz_per_mA) < 1e-6:
            return self.current_center_mA
        # freq = freq_start + slope * (I - I_start) / 1000  (GHz->THz)
        delta_freq_GHz = (freq_THz - self.freq_start_THz) * 1000.0
        delta_I = delta_freq_GHz / self.slope_GHz_per_mA
        return self.current_start_mA + delta_I

    def margin_at_current(self, current_mA: float) -> float:
        """Distance in mA from nearest mode boundary."""
        lo = min(self.current_start_mA, self.current_end_mA)
        hi = max(self.current_start_mA, self.current_end_mA)
        return min(abs(current_mA - lo), abs(current_mA - hi))

    def margin_for_frequency(self, freq_THz: float) -> float:
        """Margin in mA at the current corresponding to the given frequency."""
        I = self.current_for_frequency(freq_THz)
        if I is None:
            return 0.0
        return self.margin_at_current(I)


@dataclass
class ScanResult:
    """Complete result of a bidirectional scan."""
    up_segments: list[ModeSegment]
    down_segments: list[ModeSegment]
    raw_up: list[ScanPoint]
    raw_down: list[ScanPoint]
    scan_range_mA: tuple[float, float]
    timestamp: float

    @property
    def all_segments(self) -> list[ModeSegment]:
        return self.up_segments + self.down_segments


@dataclass
class ScanSummary:
    """Laser characteristics extracted from a bidirectional scan.

    All values here are MEASURED, not configured.
    """
    sensitivity_GHz_per_mA: float       # median slope across all segments
    mode_hop_size_GHz: float            # median jump size at boundaries
    mhfr_width_up_mA: float             # median MHFR width on up-sweep
    mhfr_width_down_mA: float           # median MHFR width on down-sweep
    hysteresis_GHz: float               # freq difference between branches at same current
    num_up_segments: int
    num_down_segments: int

    def report(self) -> str:
        return (
            f"Sensitivity:     {self.sensitivity_GHz_per_mA:.3f} GHz/mA\n"
            f"Mode hop size:   {self.mode_hop_size_GHz:.2f} GHz\n"
            f"MHFR width (Up): {self.mhfr_width_up_mA:.2f} mA\n"
            f"MHFR width (Dn): {self.mhfr_width_down_mA:.2f} mA\n"
            f"Hysteresis:      {self.hysteresis_GHz:.2f} GHz\n"
            f"Segments:        {self.num_up_segments} up, {self.num_down_segments} down"
        )


@dataclass
class LockCandidate:
    """A current value where target_freq exists on BOTH Up and Down branches.

    For the lock to be robust against hysteresis, the target frequency must
    be inside an MHFR on both the up-sweep and down-sweep at the same current.
    The effective margin is the minimum margin across both branches.
    """
    target_freq_THz: float
    optimal_current_mA: float
    up_segment: ModeSegment
    down_segment: ModeSegment
    up_margin_mA: float   # margin to nearest boundary on up branch
    down_margin_mA: float  # margin to nearest boundary on down branch

    @property
    def min_margin_mA(self) -> float:
        return min(self.up_margin_mA, self.down_margin_mA)

    @property
    def is_safe(self) -> bool:
        return self.up_margin_mA > 0 and self.down_margin_mA > 0

    def summary(self) -> str:
        return (
            f"I_set: {self.optimal_current_mA:.2f} mA, "
            f"Up seg #{self.up_segment.segment_id} "
            f"(margin {self.up_margin_mA:.2f} mA), "
            f"Down seg #{self.down_segment.segment_id} "
            f"(margin {self.down_margin_mA:.2f} mA), "
            f"min_margin: {self.min_margin_mA:.2f} mA"
        )


@dataclass
class LockStatus:
    """Real-time status during locked operation."""
    is_locked: bool
    current_freq_THz: float
    target_freq_THz: float
    error_THz: float
    current_mA: float
    pid_output: float
    mode_hop_detected: bool
    time_locked_s: float


@dataclass
class DriftRecord:
    """Single record of a successful lock's optimal current."""
    timestamp: float
    channel_name: str
    optimal_current_mA: float
    target_freq_THz: float
