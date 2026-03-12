"""Core logic: MHFR Scanner, Sweet Spot Finder, and Laser Locker."""

import time
import logging
import threading
from typing import Callable, Optional, Union

import numpy as np

from .datatypes import (
    ScanDirection,
    ScanPoint,
    ModeSegment,
    ScanResult,
    ScanSummary,
    LockCandidate,
    LockStatus,
)
from .config import ChannelConfig, SystemConfig
from .wavemeter import WavemeterInterface, MockWavemeter
from .dac_backends import DACInterface
from .pid import PIDController

logger = logging.getLogger(__name__)

WavemeterLike = Union[WavemeterInterface, MockWavemeter]


# ---------------------------------------------------------------------------
# 1. MHFRScanner -- Hysteresis-aware bidirectional scan + segment detection
# ---------------------------------------------------------------------------


class MHFRScanner:
    """Performs hysteresis-aware bidirectional current scans
    and detects mode-hop-free regions (MHFRs)."""

    def __init__(
        self,
        wavemeter: WavemeterLike,
        dac: DACInterface,
        config: ChannelConfig,
    ):
        self.wm = wavemeter
        self.dac = dac
        self.cfg = config
        self._segment_counter = 0

    # -- scanning ----------------------------------------------------------

    def scan_unidirectional(
        self,
        start_mA: float,
        end_mA: float,
        direction: ScanDirection,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> list[ScanPoint]:
        """Scan current from start to end, recording (I, f, direction)."""
        sc = self.cfg.scan
        step = sc.current_step_mA if end_mA > start_mA else -sc.current_step_mA
        n_steps = int(abs(end_mA - start_mA) / sc.current_step_mA) + 1
        currents = np.linspace(start_mA, end_mA, n_steps)

        points: list[ScanPoint] = []
        for i, I in enumerate(currents):
            self.dac.set_current_mA(self.cfg.dac_channel, float(I))
            # Inform mock wavemeter of direction
            if isinstance(self.wm, MockWavemeter):
                self.wm.set_state(float(I), direction)
            time.sleep(sc.settle_time_s)

            freq = self.wm.get_frequency_averaged(
                self.cfg.wavemeter_channel, n=sc.readings_per_point
            )
            points.append(
                ScanPoint(
                    current_mA=float(I),
                    frequency_THz=freq,
                    timestamp=time.time(),
                    direction=direction,
                )
            )

            if progress_callback and n_steps > 0:
                progress_callback((i + 1) / n_steps)

        logger.info(
            "Scan %s: %d points, %.2f -> %.2f mA",
            direction.name,
            len(points),
            start_mA,
            end_mA,
        )
        return points

    def scan_bidirectional(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ScanResult:
        """Full up-then-down scan. Returns ScanResult with detected segments."""
        sc = self.cfg.scan
        t0 = time.time()

        # Move to start first
        self.dac.set_current_mA(self.cfg.dac_channel, sc.current_min_mA)
        time.sleep(sc.settle_time_s * 10)

        # Up sweep
        logger.info("Starting UP sweep: %.2f -> %.2f mA", sc.current_min_mA, sc.current_max_mA)
        raw_up = self.scan_unidirectional(
            sc.current_min_mA,
            sc.current_max_mA,
            ScanDirection.UP,
            progress_callback=lambda p: progress_callback("UP", p)
            if progress_callback
            else None,
        )

        # Down sweep (start from where we are)
        logger.info("Starting DOWN sweep: %.2f -> %.2f mA", sc.current_max_mA, sc.current_min_mA)
        raw_down = self.scan_unidirectional(
            sc.current_max_mA,
            sc.current_min_mA,
            ScanDirection.DOWN,
            progress_callback=lambda p: progress_callback("DOWN", p)
            if progress_callback
            else None,
        )

        # Detect segments
        up_segments = self._detect_segments(raw_up, ScanDirection.UP)
        down_segments = self._detect_segments(raw_down, ScanDirection.DOWN)

        result = ScanResult(
            up_segments=up_segments,
            down_segments=down_segments,
            raw_up=raw_up,
            raw_down=raw_down,
            scan_range_mA=(sc.current_min_mA, sc.current_max_mA),
            timestamp=t0,
        )

        logger.info(
            "Scan complete: %d UP segments, %d DOWN segments",
            len(up_segments),
            len(down_segments),
        )
        return result

    # -- segment detection -------------------------------------------------

    def _detect_segments(
        self, points: list[ScanPoint], direction: ScanDirection
    ) -> list[ModeSegment]:
        """Detect MHFRs by finding frequency discontinuities.

        Algorithm:
        1. Compute df between consecutive points.
        2. A mode hop occurs where |df| > threshold (typically 1 GHz).
        3. Contiguous non-hop regions form segments.
        4. Fit linear slope within each segment.
        """
        if len(points) < 2:
            return []

        threshold_THz = self.cfg.scan.mode_hop_threshold_GHz / 1000.0

        # Find hop indices
        hop_indices: list[int] = []
        for i in range(1, len(points)):
            df = abs(points[i].frequency_THz - points[i - 1].frequency_THz)
            if df > threshold_THz:
                hop_indices.append(i)

        # Build segment boundaries
        boundaries = [0] + hop_indices + [len(points)]
        segments: list[ModeSegment] = []

        for j in range(len(boundaries) - 1):
            start_idx = boundaries[j]
            end_idx = boundaries[j + 1]
            seg_points = points[start_idx:end_idx]

            if len(seg_points) < 2:
                continue

            self._segment_counter += 1
            seg = ModeSegment(
                segment_id=self._segment_counter,
                direction=direction,
                current_start_mA=seg_points[0].current_mA,
                current_end_mA=seg_points[-1].current_mA,
                freq_start_THz=seg_points[0].frequency_THz,
                freq_end_THz=seg_points[-1].frequency_THz,
                points=seg_points,
                is_boundary_segment=(start_idx == 0 or end_idx == len(points)),
            )
            seg.slope_GHz_per_mA = self._fit_segment_slope(seg)
            segments.append(seg)

        return segments

    @staticmethod
    def _fit_segment_slope(segment: ModeSegment) -> float:
        """Linear regression of freq vs current within segment."""
        if len(segment.points) < 2:
            return 0.0
        currents = np.array([p.current_mA for p in segment.points])
        freqs = np.array([p.frequency_THz for p in segment.points])
        if currents[-1] == currents[0]:
            return 0.0
        # slope in GHz/mA = (df_THz / dI_mA) * 1000
        coeffs = np.polyfit(currents, freqs, 1)
        return float(coeffs[0] * 1000.0)  # THz/mA -> GHz/mA

    @staticmethod
    def summarize(result: ScanResult) -> ScanSummary:
        """Extract laser characteristics from scan results.

        All values are measured from the data, not assumed.
        """
        # Slopes (sensitivity)
        all_slopes = [
            s.slope_GHz_per_mA
            for s in result.all_segments
            if not s.is_boundary_segment and len(s.points) >= 5
        ]
        sensitivity = float(np.median(all_slopes)) if all_slopes else 0.0

        # MHFR widths
        up_widths = [
            s.current_width_mA
            for s in result.up_segments
            if not s.is_boundary_segment
        ]
        down_widths = [
            s.current_width_mA
            for s in result.down_segments
            if not s.is_boundary_segment
        ]
        mhfr_up = float(np.median(up_widths)) if up_widths else 0.0
        mhfr_down = float(np.median(down_widths)) if down_widths else 0.0

        # Mode hop size: frequency jumps between consecutive segments
        hop_sizes: list[float] = []
        for segments in (result.up_segments, result.down_segments):
            for i in range(1, len(segments)):
                prev = segments[i - 1]
                curr = segments[i]
                jump = abs(curr.freq_start_THz - prev.freq_end_THz) * 1000.0
                hop_sizes.append(jump)
        mode_hop_size = float(np.median(hop_sizes)) if hop_sizes else 0.0

        # Hysteresis: frequency difference between up and down at same current
        hyst_values: list[float] = []
        for up_seg in result.up_segments:
            for dn_seg in result.down_segments:
                # Find overlapping current range
                lo = max(
                    min(up_seg.current_start_mA, up_seg.current_end_mA),
                    min(dn_seg.current_start_mA, dn_seg.current_end_mA),
                )
                hi = min(
                    max(up_seg.current_start_mA, up_seg.current_end_mA),
                    max(dn_seg.current_start_mA, dn_seg.current_end_mA),
                )
                if lo >= hi:
                    continue
                I_mid = (lo + hi) / 2.0
                f_up = up_seg.freq_start_THz + (
                    up_seg.slope_GHz_per_mA
                    * (I_mid - up_seg.current_start_mA)
                    / 1000.0
                )
                f_dn = dn_seg.freq_start_THz + (
                    dn_seg.slope_GHz_per_mA
                    * (I_mid - dn_seg.current_start_mA)
                    / 1000.0
                )
                hyst_values.append(abs(f_up - f_dn) * 1000.0)
        hysteresis = float(np.median(hyst_values)) if hyst_values else 0.0

        return ScanSummary(
            sensitivity_GHz_per_mA=sensitivity,
            mode_hop_size_GHz=mode_hop_size,
            mhfr_width_up_mA=mhfr_up,
            mhfr_width_down_mA=mhfr_down,
            hysteresis_GHz=hysteresis,
            num_up_segments=len(result.up_segments),
            num_down_segments=len(result.down_segments),
        )


# ---------------------------------------------------------------------------
# 2. SweetSpotFinder -- Both branches must be safe
# ---------------------------------------------------------------------------


class SweetSpotFinder:
    """Finds optimal operating points where target_freq is inside an MHFR
    on BOTH the up-sweep and down-sweep branches.

    If the laser hops to the other branch (due to perturbation or hysteresis),
    it must still be safely within an MHFR. Any candidate where only one
    branch is safe is rejected -- the user should adjust temperature instead.
    """

    def __init__(self, config: ChannelConfig):
        self.cfg = config

    def find_candidates(
        self,
        scan_result: ScanResult,
        target_freq_THz: float,
    ) -> list[LockCandidate]:
        """Find current values where target_freq is in an MHFR on both branches.

        Algorithm:
        1. For each up-segment containing target_freq, compute I_up.
        2. For each down-segment containing target_freq, compute I_down.
        3. For each (up_seg, down_seg) pair, check if the current ranges
           overlap. The optimal current is the midpoint of the overlap,
           maximizing the minimum margin across both branches.
        """
        # Collect segments that contain the target frequency
        up_hits: list[tuple[ModeSegment, float]] = []  # (segment, I_for_target)
        down_hits: list[tuple[ModeSegment, float]] = []

        for seg in scan_result.up_segments:
            if seg.contains_frequency(target_freq_THz):
                I = seg.current_for_frequency(target_freq_THz)
                if I is not None:
                    up_hits.append((seg, I))

        for seg in scan_result.down_segments:
            if seg.contains_frequency(target_freq_THz):
                I = seg.current_for_frequency(target_freq_THz)
                if I is not None:
                    down_hits.append((seg, I))

        candidates: list[LockCandidate] = []

        for up_seg, I_up in up_hits:
            for down_seg, I_down in down_hits:
                # The current ranges of both segments must overlap.
                # Within the overlap, any current value keeps us inside
                # both MHFRs (the frequency will differ by hysteresis,
                # but both are within their respective MHFRs).
                up_lo = min(up_seg.current_start_mA, up_seg.current_end_mA)
                up_hi = max(up_seg.current_start_mA, up_seg.current_end_mA)
                dn_lo = min(down_seg.current_start_mA, down_seg.current_end_mA)
                dn_hi = max(down_seg.current_start_mA, down_seg.current_end_mA)

                overlap_lo = max(up_lo, dn_lo)
                overlap_hi = min(up_hi, dn_hi)

                if overlap_lo >= overlap_hi:
                    continue  # no overlap

                # Optimal current: center of the overlap region
                # maximizes min(margin_up, margin_down)
                I_opt = (overlap_lo + overlap_hi) / 2.0

                up_margin = min(I_opt - up_lo, up_hi - I_opt)
                dn_margin = min(I_opt - dn_lo, dn_hi - I_opt)

                candidates.append(
                    LockCandidate(
                        target_freq_THz=target_freq_THz,
                        optimal_current_mA=I_opt,
                        up_segment=up_seg,
                        down_segment=down_seg,
                        up_margin_mA=up_margin,
                        down_margin_mA=dn_margin,
                    )
                )

        # Sort by min margin (most robust first)
        candidates.sort(key=lambda c: c.min_margin_mA, reverse=True)
        return candidates

    def suggest_temperature_adjustment(
        self,
        scan_result: ScanResult,
        target_freq_THz: float,
    ) -> Optional[str]:
        """If no safe candidate exists, estimate thermistor adjustment.

        Cases:
        1. target_freq not in ANY segment -> move temperature a lot
        2. target_freq in one branch only -> move temperature slightly so
           both branches cover it at the same current range
        """
        up_has = any(
            s.contains_frequency(target_freq_THz) for s in scan_result.up_segments
        )
        down_has = any(
            s.contains_frequency(target_freq_THz) for s in scan_result.down_segments
        )
        sensitivity = self.cfg.thermistor_sensitivity_GHz_per_kOhm
        if sensitivity == 0:
            return None

        if not up_has and not down_has:
            # Target not in any segment
            closest_freq = None
            min_gap = float("inf")
            for seg in scan_result.all_segments:
                for bound in (seg.freq_min_THz, seg.freq_max_THz):
                    gap = abs(bound - target_freq_THz)
                    if gap < min_gap:
                        min_gap = gap
                        closest_freq = bound
            if closest_freq is None:
                return "No segments found. Check laser and wavemeter."
            gap_GHz = min_gap * 1000.0
            delta_R = gap_GHz / sensitivity
            direction = "increase" if target_freq_THz > closest_freq else "decrease"
            return (
                f"Target {target_freq_THz:.4f} THz is {gap_GHz:.1f} GHz away "
                f"from nearest MHFR edge. "
                f"Suggest: {direction} thermistor R by ~{abs(delta_R):.2f} kOhm."
            )

        if up_has and not down_has:
            branch_ok, branch_ng = "UP", "DOWN"
        elif down_has and not up_has:
            branch_ok, branch_ng = "DOWN", "UP"
        else:
            # Both branches have it, but no overlapping current range
            return (
                f"Target {target_freq_THz:.4f} THz exists on both branches "
                f"but at different current ranges (no overlap). "
                f"Adjust thermistor R to shift mode structure until "
                f"the current ranges overlap."
            )

        return (
            f"Target {target_freq_THz:.4f} THz is only in {branch_ok}-sweep MHFRs, "
            f"not in {branch_ng}-sweep. This is NOT safe against hysteresis. "
            f"Adjust thermistor R so that both branches cover the target "
            f"at the same current range."
        )


# ---------------------------------------------------------------------------
# 3. LaserLocker -- Approach + PID lock with mode hop detection
# ---------------------------------------------------------------------------


class LaserLocker:
    """Engages and maintains frequency lock with mode-hop awareness."""

    def __init__(
        self,
        wavemeter: WavemeterLike,
        dac: DACInterface,
        config: ChannelConfig,
    ):
        self.wm = wavemeter
        self.dac = dac
        self.cfg = config

        # PID is initialized on engage_lock() with auto-tuned or manual gains
        self._pid: Optional[PIDController] = None

        self._lock_status = LockStatus(
            is_locked=False,
            current_freq_THz=0.0,
            target_freq_THz=0.0,
            error_THz=0.0,
            current_mA=0.0,
            pid_output=0.0,
            mode_hop_detected=False,
            time_locked_s=0.0,
        )

        self._prev_freq: Optional[float] = None
        self._lock_start_time: Optional[float] = None
        self._on_mode_hop: Optional[Callable] = None
        self._on_unlock: Optional[Callable] = None

    # -- lock engagement ---------------------------------------------------

    @staticmethod
    def auto_tune_pid(summary: ScanSummary, margin_mA: float) -> tuple[float, float, float]:
        """Calculate PID gains from measured laser sensitivity.

        kp = 1 / |sensitivity|  (corrects 1 GHz error with exact current change)
        ki = kp / Ti            (Ti = integration time, ~1s)
        kd = kp * Td            (Td = derivative time, ~0.05s)

        Output is clamped to stay within the MHFR margin.

        Returns (kp, ki, kd).
        """
        sens = abs(summary.sensitivity_GHz_per_mA)
        if sens < 1e-6:
            sens = 0.2  # fallback

        kp = 1.0 / sens       # mA/GHz, e.g. 5.0 for sens=0.2
        # Conservative: use 60% of proportional gain to avoid overshoot
        kp *= 0.6
        ki = kp / 1.0         # Ti = 1 second
        kd = kp * 0.05        # Td = 50 ms

        logger.info(
            "Auto-tuned PID: kp=%.2f, ki=%.2f, kd=%.3f mA/GHz "
            "(sensitivity=%.3f GHz/mA, margin=%.2f mA)",
            kp, ki, kd, summary.sensitivity_GHz_per_mA, margin_mA,
        )
        return kp, ki, kd

    def engage_lock(
        self,
        candidate: LockCandidate,
        summary: Optional[ScanSummary] = None,
    ) -> bool:
        """Set current to optimal value and start PID.

        If summary is provided, PID gains are auto-tuned from measured
        sensitivity. Otherwise uses manual PIDConfig values.
        """
        lc = self.cfg.lock
        ch = self.cfg.dac_channel
        target_I = candidate.optimal_current_mA
        margin = candidate.min_margin_mA

        # Auto-tune or use manual PID gains
        manual = self.cfg.pid.kp != 0.0
        if manual:
            kp, ki, kd = self.cfg.pid.kp, self.cfg.pid.ki, self.cfg.pid.kd
            logger.info("Using manual PID: kp=%.2f, ki=%.2f, kd=%.3f", kp, ki, kd)
        elif summary is not None:
            kp, ki, kd = self.auto_tune_pid(summary, margin)
        else:
            raise ValueError(
                "No scan summary provided and PID gains are not manually set. "
                "Run scan first or set PIDConfig.kp/ki/kd."
            )

        logger.info(
            "Setting current to %.2f mA (up margin=%.2f, down margin=%.2f)",
            target_I,
            candidate.up_margin_mA,
            candidate.down_margin_mA,
        )
        self.dac.set_current_mA(ch, target_I)
        if isinstance(self.wm, MockWavemeter):
            self.wm.set_state(target_I, ScanDirection.UP)
        time.sleep(lc.settle_after_set_s * 3)

        # Verify frequency
        freq = self.wm.get_frequency_averaged(self.cfg.wavemeter_channel)
        error = abs(freq - candidate.target_freq_THz)
        logger.info(
            "Current set: f=%.6f THz, error=%.1f MHz",
            freq,
            error * 1e6,
        )

        # Create PID with auto-tuned gains
        # Output limits: stay within the smaller margin of up/down branches
        current_I = self.dac.get_current_mA(ch)
        self._pid = PIDController(
            kp=kp,
            ki=ki,
            kd=kd,
            output_min=current_I - margin,
            output_max=current_I + margin,
            integral_limit=self.cfg.pid.integral_limit,
        )
        self._pid.reset(output_bias=current_I)

        self._lock_status = LockStatus(
            is_locked=True,
            current_freq_THz=freq,
            target_freq_THz=candidate.target_freq_THz,
            error_THz=0.0,
            current_mA=current_I,
            pid_output=current_I,
            mode_hop_detected=False,
            time_locked_s=0.0,
        )
        self._prev_freq = None
        self._lock_start_time = time.time()

        logger.info("Lock engaged: target=%.6f THz", candidate.target_freq_THz)
        return True

    # -- lock loop ---------------------------------------------------------

    def lock_step(self) -> LockStatus:
        """Single iteration of the lock loop."""
        if not self._lock_status.is_locked:
            return self._lock_status

        freq = self.wm.get_frequency_averaged(self.cfg.wavemeter_channel)

        # Mode hop detection
        mode_hop = False
        if self._prev_freq is not None:
            mode_hop = self.detect_mode_hop(self._prev_freq, freq)
        self._prev_freq = freq

        if mode_hop:
            logger.warning(
                "MODE HOP DETECTED: %.6f -> %.6f THz (%.1f GHz jump)",
                self._prev_freq or 0,
                freq,
                abs(freq - (self._prev_freq or freq)) * 1000,
            )
            if self.cfg.lock.unlock_on_mode_hop:
                self.disengage()
                self._lock_status.mode_hop_detected = True
                if self._on_mode_hop:
                    self._on_mode_hop()
                return self._lock_status

        # PID update: error in GHz, output in mA (absolute)
        error_THz = freq - self._lock_status.target_freq_THz
        error_GHz = error_THz * 1000.0
        pid_output = self._pid.update(error_GHz)

        # Apply correction: PID output is absolute current
        self.dac.set_current_mA(self.cfg.dac_channel, pid_output)
        if isinstance(self.wm, MockWavemeter):
            self.wm.set_state(pid_output, ScanDirection.UP)

        self._lock_status = LockStatus(
            is_locked=True,
            current_freq_THz=freq,
            target_freq_THz=self._lock_status.target_freq_THz,
            error_THz=error_THz,
            current_mA=pid_output,
            pid_output=pid_output,
            mode_hop_detected=mode_hop,
            time_locked_s=time.time() - (self._lock_start_time or time.time()),
        )
        return self._lock_status

    def run_lock_loop(
        self,
        callback: Optional[Callable[[LockStatus], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Continuous lock loop until stop_event or mode hop."""
        if stop_event is None:
            stop_event = threading.Event()

        dt = self.cfg.pid.dt_s
        logger.info("Lock loop started (dt=%.3f s)", dt)

        while not stop_event.is_set():
            status = self.lock_step()
            if callback:
                callback(status)
            if not status.is_locked:
                logger.info("Lock disengaged, stopping loop.")
                break
            stop_event.wait(dt)

        logger.info("Lock loop ended.")

    def detect_mode_hop(self, prev_freq: float, curr_freq: float) -> bool:
        """True if frequency jumped by more than alert threshold."""
        threshold_THz = self.cfg.lock.mode_hop_alert_GHz / 1000.0
        return abs(curr_freq - prev_freq) > threshold_THz

    def disengage(self) -> None:
        """Stop PID, hold current DAC value."""
        self._lock_status = LockStatus(
            is_locked=False,
            current_freq_THz=self._lock_status.current_freq_THz,
            target_freq_THz=self._lock_status.target_freq_THz,
            error_THz=self._lock_status.error_THz,
            current_mA=self._lock_status.current_mA,
            pid_output=self._lock_status.pid_output,
            mode_hop_detected=self._lock_status.mode_hop_detected,
            time_locked_s=self._lock_status.time_locked_s,
        )
        logger.info("Lock disengaged. Holding current at %.2f mA", self._lock_status.current_mA)
        if self._on_unlock:
            self._on_unlock()

    def register_mode_hop_callback(self, fn: Callable) -> None:
        self._on_mode_hop = fn

    def register_unlock_callback(self, fn: Callable) -> None:
        self._on_unlock = fn

    @property
    def status(self) -> LockStatus:
        return self._lock_status


# ---------------------------------------------------------------------------
# 4. MultiChannelManager -- Orchestrate multiple lasers
# ---------------------------------------------------------------------------


class MultiChannelManager:
    """Manage scan, analysis, and lock for multiple lasers sharing
    one wavemeter (with switcher) and one or more DAC devices."""

    def __init__(
        self,
        wavemeter: WavemeterLike,
        dac: DACInterface,
        config: SystemConfig,
    ):
        self.wm = wavemeter
        self.dac = dac
        self.config = config

        self._scanners: dict[str, MHFRScanner] = {}
        self._finders: dict[str, SweetSpotFinder] = {}
        self._lockers: dict[str, LaserLocker] = {}
        self._results: dict[str, ScanResult] = {}
        self._summaries: dict[str, ScanSummary] = {}
        self._candidates: dict[str, list[LockCandidate]] = {}

        for ch_cfg in config.channels:
            name = ch_cfg.name
            self._scanners[name] = MHFRScanner(wavemeter, dac, ch_cfg)
            self._finders[name] = SweetSpotFinder(ch_cfg)
            self._lockers[name] = LaserLocker(wavemeter, dac, ch_cfg)

    @property
    def channel_names(self) -> list[str]:
        return [ch.name for ch in self.config.channels]

    def scan_all(self) -> dict[str, ScanResult]:
        """Run bidirectional scan for each channel sequentially."""
        for ch_cfg in self.config.channels:
            name = ch_cfg.name
            logger.info("=== Scanning %s (wm_ch=%d, dac_ch=%d) ===",
                        name, ch_cfg.wavemeter_channel, ch_cfg.dac_channel)
            result = self._scanners[name].scan_bidirectional()
            self._results[name] = result
            self._summaries[name] = MHFRScanner.summarize(result)
        return self._results

    def find_all_candidates(self) -> dict[str, list[LockCandidate]]:
        """Find safe operating points for each channel."""
        for ch_cfg in self.config.channels:
            name = ch_cfg.name
            if name not in self._results:
                logger.warning("No scan result for %s, skipping.", name)
                continue
            candidates = self._finders[name].find_candidates(
                self._results[name], ch_cfg.target_freq_THz
            )
            self._candidates[name] = candidates
        return self._candidates

    def lock_all(self) -> dict[str, bool]:
        """Engage lock on all channels that have a safe candidate."""
        results: dict[str, bool] = {}
        for ch_cfg in self.config.channels:
            name = ch_cfg.name
            candidates = self._candidates.get(name, [])
            if not candidates:
                logger.warning("No safe candidate for %s, skipping lock.", name)
                results[name] = False
                continue
            ok = self._lockers[name].engage_lock(
                candidates[0], summary=self._summaries.get(name)
            )
            results[name] = ok
            logger.info("Lock %s: %s", name, "OK" if ok else "FAILED")
        return results

    def run_all_locks(
        self,
        stop_event: Optional[threading.Event] = None,
        callback: Optional[Callable[[str, LockStatus], None]] = None,
    ) -> None:
        """Run lock loops for all engaged channels in parallel threads."""
        if stop_event is None:
            stop_event = threading.Event()

        threads: list[threading.Thread] = []
        for ch_cfg in self.config.channels:
            name = ch_cfg.name
            locker = self._lockers[name]
            if not locker.status.is_locked:
                continue

            def make_cb(n):
                return lambda s: callback(n, s) if callback else None

            t = threading.Thread(
                target=locker.run_lock_loop,
                kwargs={"callback": make_cb(name), "stop_event": stop_event},
                name=f"lock-{name}",
                daemon=True,
            )
            threads.append(t)
            t.start()
            logger.info("Lock thread started for %s", name)

        for t in threads:
            t.join()

    def quick_check(self, name: str) -> dict:
        """Quick check: set saved current, read frequency, compare to target.

        Returns dict with 'freq_THz', 'error_MHz', 'ok' (bool).
        Much faster than a full scan -- takes <1 second per channel.
        """
        ch_cfg = next(c for c in self.config.channels if c.name == name)
        candidates = self._candidates.get(name, [])
        if not candidates:
            return {"freq_THz": 0, "error_MHz": float("inf"), "ok": False,
                    "reason": "no saved candidate"}

        best = candidates[0]
        # Set current to saved optimal value
        self.dac.set_current_mA(ch_cfg.dac_channel, best.optimal_current_mA)
        if isinstance(self.wm, MockWavemeter):
            self.wm.set_state(best.optimal_current_mA, ScanDirection.UP)
        time.sleep(ch_cfg.lock.settle_after_set_s)

        freq = self.wm.get_frequency_averaged(ch_cfg.wavemeter_channel)
        error_MHz = (freq - best.target_freq_THz) * 1e6
        # If error < half the mode hop size, we're likely in the same mode
        summary = self._summaries.get(name)
        threshold_MHz = 500  # 0.5 GHz default
        if summary and summary.mode_hop_size_GHz > 0:
            threshold_MHz = summary.mode_hop_size_GHz * 1000 / 2

        ok = abs(error_MHz) < threshold_MHz
        logger.info(
            "Quick check %s: f=%.6f THz, error=%.1f MHz, %s",
            name, freq, error_MHz, "OK" if ok else "SHIFTED - rescan needed",
        )
        return {
            "freq_THz": freq,
            "error_MHz": error_MHz,
            "ok": ok,
            "reason": "ok" if ok else "mode structure shifted",
        }

    def quick_check_all(self) -> dict[str, dict]:
        """Quick check all channels. Returns dict of results."""
        results = {}
        for ch_cfg in self.config.channels:
            results[ch_cfg.name] = self.quick_check(ch_cfg.name)
        return results

    def load_profiles(self, store: "ProfileStore") -> list[str]:
        """Load saved profiles for all channels. Returns list of loaded names."""
        from .storage import ProfileStore
        loaded = []
        for ch_cfg in self.config.channels:
            profile = store.load(ch_cfg.name)
            if profile:
                self._results[ch_cfg.name] = profile.scan_result
                self._summaries[ch_cfg.name] = profile.summary
                self._candidates[ch_cfg.name] = profile.candidates
                loaded.append(ch_cfg.name)
                logger.info("Loaded profile for %s (%.1f hours old)",
                            ch_cfg.name, profile.age_hours)
        return loaded

    def save_profiles(self, store: "ProfileStore") -> None:
        """Save current scan results as profiles."""
        from .storage import LaserProfile
        for ch_cfg in self.config.channels:
            name = ch_cfg.name
            if name in self._results and name in self._summaries:
                profile = LaserProfile(
                    name=name,
                    scan_result=self._results[name],
                    summary=self._summaries[name],
                    candidates=self._candidates.get(name, []),
                )
                store.save(profile)
                logger.info("Saved profile for %s", name)

    def get_summary(self, name: str) -> Optional[ScanSummary]:
        return self._summaries.get(name)

    def get_locker(self, name: str) -> Optional[LaserLocker]:
        return self._lockers.get(name)

    def report(self) -> str:
        """Print status of all channels."""
        lines = []
        for ch_cfg in self.config.channels:
            name = ch_cfg.name
            lines.append(f"\n=== {name} (wm_ch={ch_cfg.wavemeter_channel}, "
                         f"dac_ch={ch_cfg.dac_channel}) ===")
            s = self._summaries.get(name)
            if s:
                lines.append(s.report())
            cands = self._candidates.get(name, [])
            if cands:
                lines.append(f"Best candidate: {cands[0].summary()}")
            else:
                advice = self._finders[name].suggest_temperature_adjustment(
                    self._results[name], ch_cfg.target_freq_THz
                ) if name in self._results else None
                lines.append(f"No safe candidate.{' ' + advice if advice else ''}")
            locker = self._lockers[name]
            if locker.status.is_locked:
                ls = locker.status
                lines.append(f"LOCKED: err={ls.error_THz*1e6:+.1f} MHz, "
                             f"I={ls.current_mA:.2f} mA")
        return "\n".join(lines)
