"""Command-line interface for Auto-MHFR Analyzer & Lock."""

import argparse
import logging
import sys
import threading
import time

from .config import SystemConfig, ScanConfig, ChannelConfig
from .mhfr_analyzer import MultiChannelManager
from .wavemeter import MockWavemeter
from .dac_backends import MockDAC
from .plotting import plot_scan_result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-mhfr",
        description="Auto-MHFR Analyzer & Lock for IF-ECDL lasers",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    # simulate (multi-channel)
    p_sim = sub.add_parser("simulate", help="Full simulation with mock hardware")
    p_sim.add_argument("--lock-time", type=float, default=5.0, help="Lock duration (s)")
    p_sim.add_argument("--step", type=float, default=0.05)

    return parser


def cmd_simulate(args) -> None:
    """Multi-channel simulation with mock hardware."""
    scan_cfg = ScanConfig(
        current_min_mA=95,
        current_max_mA=105,
        current_step_mA=args.step,
        settle_time_s=0.0,
    )

    config = SystemConfig(channels=[
        ChannelConfig(
            name="461nm",
            wavemeter_channel=1,
            dac_channel=0,
            target_freq_THz=441.3268,  # mock-compatible value
            scan=scan_cfg,
        ),
        ChannelConfig(
            name="679nm",
            wavemeter_channel=2,
            dac_channel=1,
            target_freq_THz=441.3268,
            scan=scan_cfg,
        ),
        ChannelConfig(
            name="707nm",
            wavemeter_channel=3,
            dac_channel=2,
            target_freq_THz=441.3268,
            scan=scan_cfg,
        ),
    ])

    wm = MockWavemeter()
    dac = MockDAC()

    mgr = MultiChannelManager(wm, dac, config)

    # 1. Scan all
    print("=" * 60)
    print("Phase 1: Scanning all channels")
    print("=" * 60)
    mgr.scan_all()

    # Print summaries
    for name in mgr.channel_names:
        s = mgr.get_summary(name)
        if s:
            print(f"\n--- {name} ---")
            print(s.report())

    # 2. Find candidates
    print("\n" + "=" * 60)
    print("Phase 2: Finding safe operating points")
    print("=" * 60)
    all_cands = mgr.find_all_candidates()

    for name, cands in all_cands.items():
        if cands:
            print(f"\n{name}: {len(cands)} candidate(s), best: {cands[0].summary()}")
        else:
            print(f"\n{name}: No safe candidate!")

    # 3. Lock all
    print("\n" + "=" * 60)
    print("Phase 3: Locking")
    print("=" * 60)
    lock_results = mgr.lock_all()
    for name, ok in lock_results.items():
        print(f"  {name}: {'LOCKED' if ok else 'FAILED'}")

    # 4. Run lock loops
    stop = threading.Event()

    def on_status(name, s):
        print(
            f"  [{name:5s}] t={s.time_locked_s:5.1f}s  "
            f"err={s.error_THz * 1e6:+7.1f} MHz  "
            f"I={s.current_mA:.3f} mA",
            flush=True,
        )

    def stop_after(seconds):
        time.sleep(seconds)
        stop.set()

    timer = threading.Thread(target=stop_after, args=(args.lock_time,), daemon=True)
    timer.start()
    mgr.run_all_locks(stop_event=stop, callback=on_status)

    # Final report
    print("\n" + "=" * 60)
    print("Final Report")
    print("=" * 60)
    print(mgr.report())


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "simulate":
        cmd_simulate(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
