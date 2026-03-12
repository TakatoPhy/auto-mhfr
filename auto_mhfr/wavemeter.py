"""HighFinesse wavemeter interface via wlmData.dll."""

import ctypes
import time
import logging
from typing import Optional

import numpy as np

from .datatypes import ScanDirection

logger = logging.getLogger(__name__)


class WavemeterReadError(Exception):
    """Raised when the wavemeter returns an error code."""

    ERROR_CODES = {
        -3: "NO_VALUE",
        -4: "NO_SIGNAL",
        -5: "BAD_SIGNAL",
        -6: "LOW_SIGNAL",
        -7: "BIG_SIGNAL",
        -8: "WLM_MISSING",
        -9: "OUT_OF_RANGE",
    }

    def __init__(self, code: float):
        self.code = int(code)
        name = self.ERROR_CODES.get(self.code, f"UNKNOWN({self.code})")
        super().__init__(f"Wavemeter error: {name}")


class WavemeterInterface:
    """Interface to HighFinesse wavemeter via wlmData.dll (ctypes)."""

    DLL_PATH = r"C:\Windows\System32\wlmData.dll"

    def __init__(self, dll_path: Optional[str] = None):
        path = dll_path or self.DLL_PATH
        self._dll = ctypes.WinDLL(path)  # type: ignore[attr-defined]
        self._setup_functions()

    def _setup_functions(self) -> None:
        # GetFrequencyNum(channel: c_long, value: c_double) -> c_double
        self._dll.GetFrequencyNum.argtypes = [ctypes.c_long, ctypes.c_double]
        self._dll.GetFrequencyNum.restype = ctypes.c_double

        # GetWavelengthNum(channel: c_long, value: c_double) -> c_double
        self._dll.GetWavelengthNum.argtypes = [ctypes.c_long, ctypes.c_double]
        self._dll.GetWavelengthNum.restype = ctypes.c_double

        # GetExposureNum(channel: c_long, sensor: c_long, value: c_long) -> c_long
        self._dll.GetExposureNum.argtypes = [
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
        ]
        self._dll.GetExposureNum.restype = ctypes.c_long

    def get_frequency_THz(self, channel: int = 1) -> float:
        """Read frequency in THz. Raises WavemeterReadError on error."""
        result = self._dll.GetFrequencyNum(ctypes.c_long(channel), ctypes.c_double(0))
        if result < 0:
            raise WavemeterReadError(result)
        return result

    def get_wavelength_nm(self, channel: int = 1) -> float:
        result = self._dll.GetWavelengthNum(ctypes.c_long(channel), ctypes.c_double(0))
        if result < 0:
            raise WavemeterReadError(result)
        return result

    def get_frequency_averaged(
        self, channel: int, n: int = 3, delay_s: float = 0.01
    ) -> float:
        """Average n readings, rejecting outliers (median filter)."""
        readings = []
        for _ in range(n):
            try:
                readings.append(self.get_frequency_THz(channel))
            except WavemeterReadError:
                pass
            if delay_s > 0:
                time.sleep(delay_s)
        if not readings:
            raise WavemeterReadError(-3)
        return float(np.median(readings))

    def is_signal_good(self, channel: int = 1) -> bool:
        try:
            self.get_frequency_THz(channel)
            return True
        except WavemeterReadError:
            return False


class MockWavemeter:
    """Simulated wavemeter that models IF-ECDL mode structure.

    Models piecewise-linear frequency tuning with mode hops,
    direction-dependent mode boundaries (hysteresis), and measurement noise.
    """

    def __init__(
        self,
        base_freq_THz: float = 441.330,
        sensitivity_GHz_per_mA: float = -0.2,
        mode_spacing_GHz: float = 3.35,
        hysteresis_offset_mA: float = 1.5,
        base_current_mA: float = 100.0,
        noise_MHz: float = 2.0,
    ):
        self.base_freq = base_freq_THz
        self.sensitivity = sensitivity_GHz_per_mA
        self.mode_spacing = mode_spacing_GHz
        self.hysteresis_offset = hysteresis_offset_mA
        self.base_current = base_current_mA
        self.noise_MHz = noise_MHz

        self._current_mA = base_current_mA
        self._direction = ScanDirection.UP
        self._rng = np.random.default_rng(42)

    def set_state(self, current_mA: float, direction: ScanDirection) -> None:
        self._current_mA = current_mA
        self._direction = direction

    def get_frequency_THz(self, channel: int = 1) -> float:
        """Model IF-ECDL frequency with mode hops and hysteresis.

        Physics: Within a cavity mode, freq tunes linearly with current
        at `sensitivity` GHz/mA. At mode boundaries the laser hops to
        the adjacent cavity mode (FSR = mode_spacing GHz apart).
        The MHFR width differs between up/down sweeps (hysteresis).

        Sawtooth pattern:
          - Within mode n, freq = base + n*net_jump + sensitivity*pos
          - net_jump = sensitivity*mhfr + mode_spacing (per mode period)
          - At boundary: pos resets, n increments -> frequency jumps by mode_spacing
        """
        I = self._current_mA

        if self._direction == ScanDirection.UP:
            mhfr_mA = 1.5
            boundary_offset = 0.0
        else:
            mhfr_mA = 3.0
            boundary_offset = self.hysteresis_offset

        # Position relative to the boundary grid
        shifted = I - self.base_current - boundary_offset
        mode_index = int(np.floor(shifted / mhfr_mA))
        pos_in_mode = shifted - mode_index * mhfr_mA  # 0..mhfr_mA

        # Net frequency shift per mode period:
        # continuous tuning over mhfr + jump by mode_spacing
        net_jump_GHz = self.sensitivity * mhfr_mA + self.mode_spacing

        # Frequency: base + accumulated mode jumps + continuous tuning within mode
        f = self.base_freq + (
            mode_index * net_jump_GHz + self.sensitivity * pos_in_mode
        ) / 1000.0

        # Add noise
        noise = self._rng.normal(0, self.noise_MHz * 1e-6)
        return f + noise

    def get_frequency_averaged(
        self, channel: int, n: int = 3, delay_s: float = 0.0
    ) -> float:
        readings = [self.get_frequency_THz(channel) for _ in range(n)]
        return float(np.median(readings))

    def is_signal_good(self, channel: int = 1) -> bool:
        return True
