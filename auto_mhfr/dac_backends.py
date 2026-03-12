"""DAC interfaces for laser current control."""

from abc import ABC, abstractmethod


class DACInterface(ABC):
    @abstractmethod
    def set_current_mA(self, channel: int, value_mA: float) -> None: ...

    @abstractmethod
    def get_current_mA(self, channel: int) -> float: ...

    @abstractmethod
    def get_range_mA(self, channel: int) -> tuple[float, float]: ...

    @abstractmethod
    def get_resolution_mA(self, channel: int) -> float: ...


class MockDAC(DACInterface):
    """Software DAC for testing."""

    def __init__(
        self,
        min_mA: float = 0.0,
        max_mA: float = 200.0,
        resolution_bits: int = 16,
    ):
        self._min = min_mA
        self._max = max_mA
        self._resolution = (max_mA - min_mA) / (2**resolution_bits)
        self._values: dict[int, float] = {}

    def set_current_mA(self, channel: int, value_mA: float) -> None:
        clamped = max(self._min, min(self._max, value_mA))
        # Quantize to DAC resolution
        steps = round((clamped - self._min) / self._resolution)
        self._values[channel] = self._min + steps * self._resolution

    def get_current_mA(self, channel: int) -> float:
        return self._values.get(channel, self._min)

    def get_range_mA(self, channel: int) -> tuple[float, float]:
        return (self._min, self._max)

    def get_resolution_mA(self, channel: int) -> float:
        return self._resolution


class NIDAQmxDAC(DACInterface):
    """National Instruments DAQmx backend.

    Maps voltage output to laser current controller input.
    Requires: pip install nidaqmx
    """

    def __init__(
        self,
        device: str = "Dev1",
        ao_channel: int = 0,
        voltage_to_mA: float = 20.0,
        voltage_range: tuple[float, float] = (0.0, 10.0),
    ):
        import nidaqmx

        self._device = device
        self._ao_channel = ao_channel
        self._v_to_mA = voltage_to_mA
        self._v_range = voltage_range
        self._task: nidaqmx.Task | None = None
        self._current_voltage: dict[int, float] = {}

    def _get_task(self, channel: int):
        import nidaqmx

        if self._task is None:
            self._task = nidaqmx.Task()
            ch_name = f"{self._device}/ao{channel}"
            self._task.ao_channels.add_ao_voltage_chan(
                ch_name,
                min_val=self._v_range[0],
                max_val=self._v_range[1],
            )
        return self._task

    def set_current_mA(self, channel: int, value_mA: float) -> None:
        voltage = value_mA / self._v_to_mA
        voltage = max(self._v_range[0], min(self._v_range[1], voltage))
        task = self._get_task(channel)
        task.write(voltage)
        self._current_voltage[channel] = voltage

    def get_current_mA(self, channel: int) -> float:
        return self._current_voltage.get(channel, 0.0) * self._v_to_mA

    def get_range_mA(self, channel: int) -> tuple[float, float]:
        return (
            self._v_range[0] * self._v_to_mA,
            self._v_range[1] * self._v_to_mA,
        )

    def get_resolution_mA(self, channel: int) -> float:
        # 16-bit DAC typical
        return (self._v_range[1] - self._v_range[0]) * self._v_to_mA / 65536

    def close(self) -> None:
        if self._task is not None:
            self._task.close()
            self._task = None
