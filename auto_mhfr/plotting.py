"""Visualization for scan results and lock performance."""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .datatypes import ScanResult, LockCandidate, ModeSegment, ScanDirection


def plot_scan_result(
    result: ScanResult,
    target_freq_THz: Optional[float] = None,
    candidates: Optional[list[LockCandidate]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Two-panel plot: freq vs current (top), df/dI derivative (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # -- Top panel: frequency vs current --
    # Raw data
    if result.raw_up:
        I_up = [p.current_mA for p in result.raw_up]
        f_up = [p.frequency_THz for p in result.raw_up]
        ax1.plot(I_up, f_up, ".", color="tab:blue", alpha=0.3, markersize=1, label="Up raw")

    if result.raw_down:
        I_dn = [p.current_mA for p in result.raw_down]
        f_dn = [p.frequency_THz for p in result.raw_down]
        ax1.plot(I_dn, f_dn, ".", color="tab:red", alpha=0.3, markersize=1, label="Down raw")

    # Highlight segments
    colors_up = plt.cm.Blues(np.linspace(0.4, 0.8, max(len(result.up_segments), 1)))
    colors_dn = plt.cm.Reds(np.linspace(0.4, 0.8, max(len(result.down_segments), 1)))

    for i, seg in enumerate(result.up_segments):
        I_s = [p.current_mA for p in seg.points]
        f_s = [p.frequency_THz for p in seg.points]
        ax1.plot(I_s, f_s, "-", color=colors_up[i], linewidth=1.5,
                 label=f"Up #{seg.segment_id}" if i < 3 else None)

    for i, seg in enumerate(result.down_segments):
        I_s = [p.current_mA for p in seg.points]
        f_s = [p.frequency_THz for p in seg.points]
        ax1.plot(I_s, f_s, "-", color=colors_dn[i], linewidth=1.5,
                 label=f"Down #{seg.segment_id}" if i < 3 else None)

    # Target frequency line
    if target_freq_THz is not None:
        ax1.axhline(target_freq_THz, color="green", linestyle="--", linewidth=1,
                     label=f"Target: {target_freq_THz:.4f} THz")

    # Candidates (safe on both branches)
    if candidates:
        for j, cand in enumerate(candidates):
            marker = "*" if j == 0 else "o"
            ax1.plot(cand.optimal_current_mA, cand.target_freq_THz,
                     marker, color="gold" if j == 0 else "orange",
                     markersize=12, zorder=10,
                     label=f"#{j+1} (min_margin={cand.min_margin_mA:.1f} mA)")

    ax1.set_ylabel("Frequency (THz)")
    ax1.set_title("MHFR Scan Result")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # -- Bottom panel: numerical derivative --
    for points, color, label in [
        (result.raw_up, "tab:blue", "Up df/dI"),
        (result.raw_down, "tab:red", "Down df/dI"),
    ]:
        if len(points) < 2:
            continue
        I_arr = np.array([p.current_mA for p in points])
        f_arr = np.array([p.frequency_THz for p in points])
        dI = np.diff(I_arr)
        df = np.diff(f_arr) * 1000.0  # THz -> GHz
        mask = np.abs(dI) > 1e-6
        derivative = np.where(mask, df / dI, 0)
        ax2.plot(I_arr[1:], derivative, ".", markersize=1, color=color, alpha=0.5, label=label)

    # Mode hop threshold
    threshold = result.up_segments[0].slope_GHz_per_mA if result.up_segments else -0.2
    ax2.axhline(threshold, color="gray", linestyle=":", alpha=0.5, label=f"Expected slope: {threshold:.2f}")
    ax2.set_ylabel("df/dI (GHz/mA)")
    ax2.set_xlabel("Current (mA)")
    ax2.set_ylim(-5, 5)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_hysteresis(
    result: ScanResult,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Overlay up and down scans to visualize hysteresis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if result.raw_up:
        ax.plot(
            [p.current_mA for p in result.raw_up],
            [p.frequency_THz for p in result.raw_up],
            "-", color="tab:blue", linewidth=0.8, label="Up sweep",
        )
    if result.raw_down:
        ax.plot(
            [p.current_mA for p in result.raw_down],
            [p.frequency_THz for p in result.raw_down],
            "-", color="tab:red", linewidth=0.8, label="Down sweep",
        )

    ax.set_xlabel("Current (mA)")
    ax.set_ylabel("Frequency (THz)")
    ax.set_title("Hysteresis: Up vs Down Sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_lock_history(
    timestamps: list[float],
    frequencies: list[float],
    currents: list[float],
    target_freq_THz: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Time series of lock performance."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    t = np.array(timestamps)
    t = t - t[0]  # relative time

    # Frequency error
    errors_MHz = (np.array(frequencies) - target_freq_THz) * 1e6
    ax1.plot(t, errors_MHz, "-", linewidth=0.5, color="tab:blue")
    ax1.axhline(0, color="green", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Frequency error (MHz)")
    ax1.set_title("Lock Performance")
    ax1.grid(True, alpha=0.3)

    # Current
    ax2.plot(t, currents, "-", linewidth=0.5, color="tab:orange")
    ax2.set_ylabel("Current (mA)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig
