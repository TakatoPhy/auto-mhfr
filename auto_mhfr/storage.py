"""Save/load scan results and laser profiles for fast startup."""

import json
import os
import time
from pathlib import Path
from typing import Optional

from .datatypes import (
    ScanDirection,
    ScanPoint,
    ModeSegment,
    ScanResult,
    ScanSummary,
    LockCandidate,
)


def _scan_point_to_dict(p: ScanPoint) -> dict:
    return {
        "current_mA": p.current_mA,
        "frequency_THz": p.frequency_THz,
        "timestamp": p.timestamp,
        "direction": p.direction.name,
    }


def _dict_to_scan_point(d: dict) -> ScanPoint:
    return ScanPoint(
        current_mA=d["current_mA"],
        frequency_THz=d["frequency_THz"],
        timestamp=d["timestamp"],
        direction=ScanDirection[d["direction"]],
    )


def _segment_to_dict(s: ModeSegment) -> dict:
    return {
        "segment_id": s.segment_id,
        "direction": s.direction.name,
        "current_start_mA": s.current_start_mA,
        "current_end_mA": s.current_end_mA,
        "freq_start_THz": s.freq_start_THz,
        "freq_end_THz": s.freq_end_THz,
        "slope_GHz_per_mA": s.slope_GHz_per_mA,
        "is_boundary_segment": s.is_boundary_segment,
        # points omitted to keep file small
    }


def _dict_to_segment(d: dict) -> ModeSegment:
    return ModeSegment(
        segment_id=d["segment_id"],
        direction=ScanDirection[d["direction"]],
        current_start_mA=d["current_start_mA"],
        current_end_mA=d["current_end_mA"],
        freq_start_THz=d["freq_start_THz"],
        freq_end_THz=d["freq_end_THz"],
        slope_GHz_per_mA=d["slope_GHz_per_mA"],
        is_boundary_segment=d.get("is_boundary_segment", False),
    )


def _summary_to_dict(s: ScanSummary) -> dict:
    return {
        "sensitivity_GHz_per_mA": s.sensitivity_GHz_per_mA,
        "mode_hop_size_GHz": s.mode_hop_size_GHz,
        "mhfr_width_up_mA": s.mhfr_width_up_mA,
        "mhfr_width_down_mA": s.mhfr_width_down_mA,
        "hysteresis_GHz": s.hysteresis_GHz,
        "num_up_segments": s.num_up_segments,
        "num_down_segments": s.num_down_segments,
    }


def _dict_to_summary(d: dict) -> ScanSummary:
    return ScanSummary(**d)


def _candidate_to_dict(c: LockCandidate) -> dict:
    return {
        "target_freq_THz": c.target_freq_THz,
        "optimal_current_mA": c.optimal_current_mA,
        "up_margin_mA": c.up_margin_mA,
        "down_margin_mA": c.down_margin_mA,
        "up_segment_id": c.up_segment.segment_id,
        "down_segment_id": c.down_segment.segment_id,
    }


class LaserProfile:
    """Saved profile for a single laser channel.

    Contains:
    - scan_result: full segment data from last scan
    - summary: measured laser characteristics (stable across days)
    - candidates: safe operating points from last scan
    - saved_at: timestamp of last save
    """

    def __init__(
        self,
        name: str,
        scan_result: ScanResult,
        summary: ScanSummary,
        candidates: list[LockCandidate],
    ):
        self.name = name
        self.scan_result = scan_result
        self.summary = summary
        self.candidates = candidates
        self.saved_at = time.time()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "saved_at": self.saved_at,
            "summary": _summary_to_dict(self.summary),
            "up_segments": [_segment_to_dict(s) for s in self.scan_result.up_segments],
            "down_segments": [_segment_to_dict(s) for s in self.scan_result.down_segments],
            "scan_range_mA": list(self.scan_result.scan_range_mA),
            "candidates": [_candidate_to_dict(c) for c in self.candidates],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LaserProfile":
        up_segs = [_dict_to_segment(s) for s in d["up_segments"]]
        down_segs = [_dict_to_segment(s) for s in d["down_segments"]]

        scan_result = ScanResult(
            up_segments=up_segs,
            down_segments=down_segs,
            raw_up=[],
            raw_down=[],
            scan_range_mA=tuple(d["scan_range_mA"]),
            timestamp=d["saved_at"],
        )
        summary = _dict_to_summary(d["summary"])

        # Reconstruct candidates with segment references
        seg_by_id = {s.segment_id: s for s in up_segs + down_segs}
        candidates = []
        for cd in d.get("candidates", []):
            up_seg = seg_by_id.get(cd["up_segment_id"])
            dn_seg = seg_by_id.get(cd["down_segment_id"])
            if up_seg and dn_seg:
                candidates.append(LockCandidate(
                    target_freq_THz=cd["target_freq_THz"],
                    optimal_current_mA=cd["optimal_current_mA"],
                    up_segment=up_seg,
                    down_segment=dn_seg,
                    up_margin_mA=cd["up_margin_mA"],
                    down_margin_mA=cd["down_margin_mA"],
                ))

        profile = cls(d["name"], scan_result, summary, candidates)
        profile.saved_at = d["saved_at"]
        return profile

    @property
    def age_hours(self) -> float:
        return (time.time() - self.saved_at) / 3600


class ProfileStore:
    """Save/load laser profiles to disk."""

    def __init__(self, directory: str = "~/.auto_mhfr"):
        self.directory = Path(os.path.expanduser(directory))
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        safe_name = name.replace("/", "_").replace(" ", "_")
        return self.directory / f"{safe_name}.json"

    def save(self, profile: LaserProfile) -> None:
        path = self._path(profile.name)
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    def load(self, name: str) -> Optional[LaserProfile]:
        path = self._path(name)
        if not path.exists():
            return None
        with open(path) as f:
            return LaserProfile.from_dict(json.load(f))

    def list_profiles(self) -> list[str]:
        return [p.stem for p in self.directory.glob("*.json")]

    def delete(self, name: str) -> None:
        path = self._path(name)
        if path.exists():
            path.unlink()
