# Auto-MHFR

Automatic mode-hop-free region (MHFR) analyzer and frequency lock for interference-filter external-cavity diode lasers (IF-ECDLs).

## What it does

1. **Scans** laser current bidirectionally while reading a HighFinesse wavemeter
2. **Maps** mode-hop-free regions on both up-sweep and down-sweep branches
3. **Finds safe operating points** where the target frequency is inside an MHFR on *both* branches (hysteresis-aware)
4. **Auto-tunes PID** gains from measured laser sensitivity
5. **Locks** the laser frequency via current feedback through NI-DAQ
6. **Saves profiles** so subsequent startups skip the full scan

Supports simultaneous multi-channel operation (e.g., 461 nm + 679 nm + 707 nm for Sr cooling).

## Why

Existing wavemeter-lock tools apply PID feedback without knowing where mode hops are. When a perturbation pushes the laser across a mode boundary, the lock fails. Auto-MHFR prevents this by characterizing the full mode structure first and choosing operating points with maximum margin from mode boundaries on both hysteresis branches.

## Hardware requirements

- **Wavemeter**: HighFinesse (WSU, WS7, WS8, etc.) with USB/network connection
- **Current control**: NI-DAQ analog output → Thorlabs LDC202C (or similar current controller with modulation input)
- **OS**: Windows (HighFinesse DLL is Windows-only)

## Installation

```bash
pip install -e ".[all]"
```

Or minimal (no GUI):

```bash
pip install -e .
```

## Quick start

### GUI

```bash
auto-mhfr gui
```

Buttons: **SCAN** → **QUICK CHECK** → **LOCK** → **STOP**

### Simulation (no hardware needed)

```bash
auto-mhfr simulate
```

Runs a 3-channel mock demo to see how the system works.

### Python API

```python
from auto_mhfr import (
    SystemConfig, ChannelConfig, ScanConfig,
    MultiChannelManager,
)
from auto_mhfr.wavemeter import WavemeterInterface
from auto_mhfr.dac_backends import NIDAQmxDAC

config = SystemConfig(channels=[
    ChannelConfig(
        name="461nm",
        wavemeter_channel=1,
        dac_channel=0,
        target_freq_THz=650.504,
        scan=ScanConfig(current_min_mA=80, current_max_mA=120),
    ),
])

wm = WavemeterInterface()  # connects to HighFinesse DLL
dac = NIDAQmxDAC(device="Dev1")
mgr = MultiChannelManager(config, wm, dac)

# Scan, find safe points, lock
mgr.scan_all()
mgr.find_all_candidates()
mgr.lock_all()
mgr.run_all_locks()  # blocking loop
```

## Configuration

All you need to set per channel:

| Parameter | Description |
|-----------|-------------|
| `wavemeter_channel` | HighFinesse switcher channel (1-8) |
| `dac_channel` | NI-DAQ analog output channel |
| `target_freq_THz` | Target frequency |
| `current_min_mA` / `current_max_mA` | Scan range |

Everything else (sensitivity, MHFR width, PID gains) is **measured automatically** from the scan.

## Architecture

```
auto_mhfr/
├── mhfr_analyzer.py   # Core: MHFRScanner, SweetSpotFinder, LaserLocker, MultiChannelManager
├── datatypes.py        # ScanPoint, ModeSegment, ScanResult, LockCandidate, etc.
├── config.py           # ScanConfig, PIDConfig, LockConfig, ChannelConfig, SystemConfig
├── wavemeter.py        # HighFinesse DLL interface + MockWavemeter
├── dac_backends.py     # NI-DAQ + MockDAC
├── pid.py              # PID controller with anti-windup
├── storage.py          # Profile save/load (JSON)
├── plotting.py         # Matplotlib scan visualization
├── gui.py              # PyQt5 GUI
└── cli.py              # Command-line interface
```

## License

MIT
