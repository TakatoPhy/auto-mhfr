"""Configuration dataclasses for Auto-MHFR system."""

from dataclasses import dataclass, field


@dataclass
class ScanConfig:
    current_min_mA: float = 80.0
    current_max_mA: float = 120.0
    current_step_mA: float = 0.01
    settle_time_s: float = 0.05
    readings_per_point: int = 3
    mode_hop_threshold_GHz: float = 1.0


@dataclass
class PIDConfig:
    """PID config. Gains are auto-tuned from scan results by default.

    Set kp/ki/kd to non-zero to override auto-tuning.
    """
    kp: float = 0.0       # 0 = auto-tune from sensitivity
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 0.5   # GHz*s anti-windup
    dt_s: float = 0.1             # loop interval


@dataclass
class LockConfig:
    settle_after_set_s: float = 0.2
    lock_tolerance_THz: float = 0.0001  # 100 MHz
    mode_hop_alert_GHz: float = 3.0
    relock_attempts: int = 3
    unlock_on_mode_hop: bool = True


@dataclass
class ChannelConfig:
    """Configuration for a single laser channel."""
    name: str = ""
    wavemeter_channel: int = 1
    dac_channel: int = 0
    target_freq_THz: float = 0.0
    scan: ScanConfig = field(default_factory=ScanConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    lock: LockConfig = field(default_factory=LockConfig)
    thermistor_sensitivity_GHz_per_kOhm: float = 0.0


@dataclass
class SystemConfig:
    channels: list[ChannelConfig] = field(default_factory=list)

    @staticmethod
    def single(
        wavemeter_channel: int = 1,
        dac_channel: int = 0,
        **kwargs,
    ) -> "SystemConfig":
        """Convenience: create config for a single laser."""
        ch = ChannelConfig(
            wavemeter_channel=wavemeter_channel,
            dac_channel=dac_channel,
            **kwargs,
        )
        return SystemConfig(channels=[ch])
