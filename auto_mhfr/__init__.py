"""Auto-MHFR Analyzer & Lock for IF-ECDL lasers."""

__version__ = "0.1.0"

from .datatypes import (
    ScanDirection,
    ScanPoint,
    ModeSegment,
    ScanResult,
    ScanSummary,
    LockCandidate,
    LockStatus,
)
from .config import SystemConfig, ScanConfig, PIDConfig, LockConfig, ChannelConfig
from .mhfr_analyzer import (
    MHFRScanner,
    SweetSpotFinder,
    LaserLocker,
    MultiChannelManager,
)
from .wavemeter import MockWavemeter
from .dac_backends import DACInterface, MockDAC
from .pid import PIDController
