"""GUI for Auto-MHFR Analyzer & Lock."""

import sys
import threading
import logging
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QGridLayout,
    QDoubleSpinBox,
    QSpinBox,
    QTextEdit,
    QTabWidget,
    QMessageBox,
    QSplitter,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QColor, QFont

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from .config import SystemConfig, ScanConfig, ChannelConfig, PIDConfig, LockConfig
from .mhfr_analyzer import (
    MHFRScanner,
    SweetSpotFinder,
    LaserLocker,
    MultiChannelManager,
)
from .datatypes import ScanResult, ScanSummary, LockCandidate, LockStatus, ScanDirection
from .storage import ProfileStore
from .wavemeter import MockWavemeter
from .dac_backends import MockDAC, DACInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal bridge: worker threads -> GUI
# ---------------------------------------------------------------------------

class WorkerSignals(QObject):
    scan_done = pyqtSignal(str)          # channel name
    scan_progress = pyqtSignal(str, str, float)  # channel, phase, progress
    lock_status = pyqtSignal(str, object)  # channel, LockStatus
    quick_check_done = pyqtSignal(dict)  # {name: {ok, error_MHz, ...}}
    error = pyqtSignal(str)
    log = pyqtSignal(str)


# ---------------------------------------------------------------------------
# Scan plot widget
# ---------------------------------------------------------------------------

class ScanPlotWidget(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 5))
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_scan(
        self,
        result: ScanResult,
        target_freq: Optional[float] = None,
        candidates: Optional[list[LockCandidate]] = None,
    ):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Raw data
        if result.raw_up:
            ax.plot(
                [p.current_mA for p in result.raw_up],
                [p.frequency_THz for p in result.raw_up],
                ".", color="tab:blue", alpha=0.4, markersize=2, label="Up",
            )
        if result.raw_down:
            ax.plot(
                [p.current_mA for p in result.raw_down],
                [p.frequency_THz for p in result.raw_down],
                ".", color="tab:red", alpha=0.4, markersize=2, label="Down",
            )

        # Segments
        for seg in result.up_segments:
            ax.plot(
                [p.current_mA for p in seg.points],
                [p.frequency_THz for p in seg.points],
                "-", color="tab:blue", linewidth=1.5,
            )
        for seg in result.down_segments:
            ax.plot(
                [p.current_mA for p in seg.points],
                [p.frequency_THz for p in seg.points],
                "-", color="tab:red", linewidth=1.5,
            )

        # Target
        if target_freq:
            ax.axhline(target_freq, color="green", linestyle="--",
                       linewidth=1.5, label=f"Target: {target_freq:.4f} THz")

        # Candidates
        if candidates:
            best = candidates[0]
            ax.axvline(best.optimal_current_mA, color="gold", linestyle=":",
                       linewidth=2, label=f"I_set={best.optimal_current_mA:.2f} mA")

        ax.set_xlabel("Current (mA)")
        ax.set_ylabel("Frequency (THz)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()


# ---------------------------------------------------------------------------
# Channel panel: one per laser
# ---------------------------------------------------------------------------

class ChannelPanel(QGroupBox):
    def __init__(self, ch_cfg: ChannelConfig, parent=None):
        super().__init__(ch_cfg.name, parent)
        self.ch_cfg = ch_cfg
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # Status indicator
        self.status_label = QLabel("IDLE")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("monospace", 14, QFont.Bold))
        self._set_status_color("gray")
        layout.addWidget(self.status_label)

        # Info grid
        info = QGridLayout()
        self.lbl_freq = QLabel("-")
        self.lbl_error = QLabel("-")
        self.lbl_current = QLabel("-")
        self.lbl_margin = QLabel("-")
        self.lbl_sensitivity = QLabel("-")
        self.lbl_mhfr = QLabel("-")

        info.addWidget(QLabel("Frequency:"), 0, 0)
        info.addWidget(self.lbl_freq, 0, 1)
        info.addWidget(QLabel("Error:"), 1, 0)
        info.addWidget(self.lbl_error, 1, 1)
        info.addWidget(QLabel("Current:"), 2, 0)
        info.addWidget(self.lbl_current, 2, 1)
        info.addWidget(QLabel("Margin:"), 3, 0)
        info.addWidget(self.lbl_margin, 3, 1)
        info.addWidget(QLabel("Sensitivity:"), 4, 0)
        info.addWidget(self.lbl_sensitivity, 4, 1)
        info.addWidget(QLabel("MHFR (Up/Dn):"), 5, 0)
        info.addWidget(self.lbl_mhfr, 5, 1)
        layout.addLayout(info)

        self.setLayout(layout)

    def _set_status_color(self, color: str):
        colors = {
            "gray": "#888888",
            "green": "#2ecc71",
            "yellow": "#f1c40f",
            "red": "#e74c3c",
            "blue": "#3498db",
        }
        c = colors.get(color, color)
        self.status_label.setStyleSheet(
            f"background-color: {c}; color: white; padding: 8px; border-radius: 4px;"
        )

    def set_scanning(self):
        self.status_label.setText("SCANNING...")
        self._set_status_color("blue")

    def set_scan_result(self, summary: ScanSummary, candidates: list[LockCandidate]):
        self.lbl_sensitivity.setText(f"{summary.sensitivity_GHz_per_mA:.3f} GHz/mA")
        self.lbl_mhfr.setText(
            f"{summary.mhfr_width_up_mA:.2f} / {summary.mhfr_width_down_mA:.2f} mA"
        )
        if candidates:
            best = candidates[0]
            self.lbl_margin.setText(f"{best.min_margin_mA:.2f} mA")
            self.status_label.setText("READY")
            self._set_status_color("green")
        else:
            self.lbl_margin.setText("NO SAFE POINT")
            self.status_label.setText("ADJUST TEMP")
            self._set_status_color("yellow")

    def set_locked(self):
        self.status_label.setText("LOCKED")
        self._set_status_color("green")

    def update_lock_status(self, s: LockStatus):
        self.lbl_freq.setText(f"{s.current_freq_THz:.6f} THz")
        self.lbl_error.setText(f"{s.error_THz * 1e6:+.1f} MHz")
        self.lbl_current.setText(f"{s.current_mA:.3f} mA")
        if s.mode_hop_detected:
            self.status_label.setText("MODE HOP!")
            self._set_status_color("red")

    def set_idle(self):
        self.status_label.setText("IDLE")
        self._set_status_color("gray")

    def set_error(self, msg: str):
        self.status_label.setText("ERROR")
        self._set_status_color("red")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(
        self,
        wavemeter,
        dac: DACInterface,
        config: SystemConfig,
    ):
        super().__init__()
        self.wm = wavemeter
        self.dac = dac
        self.config = config
        self.mgr = MultiChannelManager(wavemeter, dac, config)
        self.store = ProfileStore()

        self.signals = WorkerSignals()
        self.signals.scan_done.connect(self._on_scan_done)
        self.signals.lock_status.connect(self._on_lock_status)
        self.signals.quick_check_done.connect(self._on_quick_check_done)
        self.signals.error.connect(self._on_error)
        self.signals.log.connect(self._on_log)

        self._stop_event = threading.Event()
        self._lock_thread: Optional[threading.Thread] = None

        self._build_ui()
        self.setWindowTitle("Auto-MHFR Analyzer & Lock")
        self.resize(1200, 800)

        # Try loading saved profiles on startup
        self._load_saved_profiles()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # -- Top: buttons --
        btn_layout = QHBoxLayout()

        self.btn_scan = QPushButton("SCAN")
        self.btn_scan.setFont(QFont("sans-serif", 16, QFont.Bold))
        self.btn_scan.setMinimumHeight(60)
        self.btn_scan.setStyleSheet("background-color: #3498db; color: white;")
        self.btn_scan.clicked.connect(self._on_scan_clicked)
        btn_layout.addWidget(self.btn_scan)

        self.btn_quick = QPushButton("QUICK CHECK")
        self.btn_quick.setFont(QFont("sans-serif", 16, QFont.Bold))
        self.btn_quick.setMinimumHeight(60)
        self.btn_quick.setStyleSheet("background-color: #9b59b6; color: white;")
        self.btn_quick.setEnabled(False)
        self.btn_quick.clicked.connect(self._on_quick_check_clicked)
        btn_layout.addWidget(self.btn_quick)

        self.btn_lock = QPushButton("LOCK")
        self.btn_lock.setFont(QFont("sans-serif", 16, QFont.Bold))
        self.btn_lock.setMinimumHeight(60)
        self.btn_lock.setStyleSheet("background-color: #2ecc71; color: white;")
        self.btn_lock.setEnabled(False)
        self.btn_lock.clicked.connect(self._on_lock_clicked)
        btn_layout.addWidget(self.btn_lock)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setFont(QFont("sans-serif", 16, QFont.Bold))
        self.btn_stop.setMinimumHeight(60)
        self.btn_stop.setStyleSheet("background-color: #e74c3c; color: white;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        btn_layout.addWidget(self.btn_stop)

        main_layout.addLayout(btn_layout)

        # -- Middle: splitter with channels + plot --
        splitter = QSplitter(Qt.Horizontal)

        # Channel panels
        ch_widget = QWidget()
        ch_layout = QVBoxLayout(ch_widget)
        self.panels: dict[str, ChannelPanel] = {}
        for ch_cfg in self.config.channels:
            panel = ChannelPanel(ch_cfg)
            self.panels[ch_cfg.name] = panel
            ch_layout.addWidget(panel)
        ch_layout.addStretch()
        splitter.addWidget(ch_widget)

        # Plot tabs (one per channel)
        self.plot_tabs = QTabWidget()
        self.plots: dict[str, ScanPlotWidget] = {}
        for ch_cfg in self.config.channels:
            plot = ScanPlotWidget()
            self.plots[ch_cfg.name] = plot
            self.plot_tabs.addTab(plot, ch_cfg.name)
        splitter.addWidget(self.plot_tabs)

        splitter.setSizes([300, 700])
        main_layout.addWidget(splitter)

        # -- Bottom: log --
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("monospace", 9))
        main_layout.addWidget(self.log_text)

    # -- actions --

    def _load_saved_profiles(self):
        loaded = self.mgr.load_profiles(self.store)
        if loaded:
            self._log(f"Loaded saved profiles: {', '.join(loaded)}")
            self.btn_quick.setEnabled(True)
            # Show saved data in panels
            for name in loaded:
                summary = self.mgr.get_summary(name)
                cands = self.mgr._candidates.get(name, [])
                result = self.mgr._results.get(name)
                if summary:
                    self.panels[name].set_scan_result(summary, cands)
                if result:
                    ch_cfg = next(c for c in self.config.channels if c.name == name)
                    self.plots[name].plot_scan(result, ch_cfg.target_freq_THz, cands)
        else:
            self._log("No saved profiles found. Run SCAN first.")

    def _on_quick_check_clicked(self):
        self.btn_quick.setEnabled(False)
        self._log("Quick checking all channels...")

        def worker():
            try:
                results = self.mgr.quick_check_all()
                self.signals.quick_check_done.emit(results)
            except Exception as e:
                self.signals.error.emit(str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_quick_check_done(self, results: dict):
        all_ok = True
        for name, r in results.items():
            if r["ok"]:
                self._log(f"{name}: OK (error={r['error_MHz']:+.1f} MHz)")
                self.panels[name].status_label.setText("CHECK OK")
                self.panels[name]._set_status_color("green")
            else:
                self._log(f"{name}: SHIFTED ({r['reason']}, error={r['error_MHz']:+.1f} MHz) - rescan needed")
                self.panels[name].status_label.setText("SHIFTED")
                self.panels[name]._set_status_color("yellow")
                all_ok = False

        self.btn_quick.setEnabled(True)
        if all_ok:
            self.btn_lock.setEnabled(True)
            self._log("All channels OK. Ready to lock.")
        else:
            self._log("Some channels shifted. Run SCAN for those channels.")

    def _on_scan_clicked(self):
        self.btn_scan.setEnabled(False)
        self.btn_lock.setEnabled(False)
        for panel in self.panels.values():
            panel.set_scanning()
        self._log("Starting scan for all channels...")

        def worker():
            try:
                self.mgr.scan_all()
                self.mgr.find_all_candidates()
                for ch_cfg in self.config.channels:
                    self.signals.scan_done.emit(ch_cfg.name)
            except Exception as e:
                self.signals.error.emit(str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_scan_done(self, name: str):
        summary = self.mgr.get_summary(name)
        cands = self.mgr._candidates.get(name, [])
        result = self.mgr._results.get(name)
        ch_cfg = next(c for c in self.config.channels if c.name == name)

        if summary and result:
            self.panels[name].set_scan_result(summary, cands)
            self.plots[name].plot_scan(result, ch_cfg.target_freq_THz, cands)
            if cands:
                self._log(f"{name}: OK (margin={cands[0].min_margin_mA:.2f} mA)")
            else:
                self._log(f"{name}: NO SAFE POINT - adjust temperature")

        # Enable lock if all channels scanned
        all_done = all(
            self.mgr.get_summary(c.name) is not None
            for c in self.config.channels
        )
        if all_done:
            self.btn_scan.setEnabled(True)
            self.btn_quick.setEnabled(True)
            any_lockable = any(
                len(self.mgr._candidates.get(c.name, [])) > 0
                for c in self.config.channels
            )
            self.btn_lock.setEnabled(any_lockable)
            # Save profiles for fast startup next time
            try:
                self.mgr.save_profiles(self.store)
                self._log("Profiles saved for next startup.")
            except Exception as e:
                self._log(f"Warning: failed to save profiles: {e}")

    def _on_lock_clicked(self):
        self.btn_lock.setEnabled(False)
        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._stop_event.clear()
        self._log("Locking all channels...")

        def worker():
            try:
                results = self.mgr.lock_all()
                for name, ok in results.items():
                    if ok:
                        self.signals.log.emit(f"{name}: LOCKED")
                    else:
                        self.signals.log.emit(f"{name}: lock failed")

                self.mgr.run_all_locks(
                    stop_event=self._stop_event,
                    callback=lambda name, s: self.signals.lock_status.emit(name, s),
                )
            except Exception as e:
                self.signals.error.emit(str(e))

        self._lock_thread = threading.Thread(target=worker, daemon=True)
        self._lock_thread.start()

    def _on_stop_clicked(self):
        self._stop_event.set()
        self.btn_stop.setEnabled(False)
        self.btn_scan.setEnabled(True)
        self.btn_lock.setEnabled(True)
        for panel in self.panels.values():
            panel.set_idle()
        self._log("Lock stopped.")

    def _on_lock_status(self, name: str, status: LockStatus):
        panel = self.panels.get(name)
        if panel:
            if status.is_locked:
                panel.set_locked()
            panel.update_lock_status(status)

    def _on_error(self, msg: str):
        self._log(f"ERROR: {msg}")
        self.btn_scan.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

    def _on_log(self, msg: str):
        self._log(msg)

    def _log(self, msg: str):
        self.log_text.append(msg)

    def closeEvent(self, event):
        self._stop_event.set()
        event.accept()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_gui(wavemeter, dac: DACInterface, config: SystemConfig):
    """Launch the GUI with given hardware and config."""
    app = QApplication(sys.argv)
    window = MainWindow(wavemeter, dac, config)
    window.show()
    sys.exit(app.exec_())


def run_demo():
    """Launch GUI with mock hardware for testing."""
    config = SystemConfig(channels=[
        ChannelConfig(
            name="461nm",
            wavemeter_channel=1,
            dac_channel=0,
            target_freq_THz=441.3268,
            scan=ScanConfig(
                current_min_mA=95, current_max_mA=105,
                current_step_mA=0.05, settle_time_s=0.0,
            ),
        ),
        ChannelConfig(
            name="679nm",
            wavemeter_channel=2,
            dac_channel=1,
            target_freq_THz=441.3268,
            scan=ScanConfig(
                current_min_mA=95, current_max_mA=105,
                current_step_mA=0.05, settle_time_s=0.0,
            ),
        ),
        ChannelConfig(
            name="707nm",
            wavemeter_channel=3,
            dac_channel=2,
            target_freq_THz=441.3268,
            scan=ScanConfig(
                current_min_mA=95, current_max_mA=105,
                current_step_mA=0.05, settle_time_s=0.0,
            ),
        ),
    ])

    wm = MockWavemeter()
    dac = MockDAC()
    run_gui(wm, dac, config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_demo()
