"""PID controller with anti-windup and bumpless transfer."""

import time


class PIDController:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_min: float,
        output_max: float,
        integral_limit: float = float("inf"),
        derivative_filter_alpha: float = 0.1,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit
        self.alpha = derivative_filter_alpha

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0
        self._bias = 0.0
        self._last_time: float | None = None

        self._p_term = 0.0
        self._i_term = 0.0
        self._d_term = 0.0

    def reset(self, output_bias: float = 0.0) -> None:
        """Reset state. output_bias enables bumpless transfer."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0
        self._bias = output_bias
        self._last_time = None

    def update(self, error: float, dt: float | None = None) -> float:
        now = time.monotonic()
        if dt is None:
            if self._last_time is None:
                dt = 0.0
            else:
                dt = now - self._last_time
        self._last_time = now

        # Proportional
        self._p_term = self.kp * error

        # Integral with anti-windup
        if dt > 0:
            self._integral += error * dt
            self._integral = max(
                -self.integral_limit, min(self.integral_limit, self._integral)
            )
        self._i_term = self.ki * self._integral

        # Derivative with low-pass filter
        if dt > 0:
            raw_derivative = (error - self._prev_error) / dt
            self._prev_derivative = (
                self.alpha * raw_derivative
                + (1 - self.alpha) * self._prev_derivative
            )
        self._d_term = self.kd * self._prev_derivative

        self._prev_error = error

        output = self._bias + self._p_term + self._i_term + self._d_term
        return max(self.output_min, min(self.output_max, output))

    @property
    def components(self) -> tuple[float, float, float]:
        return (self._p_term, self._i_term, self._d_term)
