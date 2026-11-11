"""
Soil Moisture Signal Conditioning & Stability Filter
===================================================
Provides robust temporal stabilization for dual soil-moisture sensors (probe 1
and probe 2) in agricultural IoT telemetry (NEST):

1. Median Filter:
   - Eliminates isolated impulse anomalies, EMI glitches, and ADC bit flips.
   - Sliding window over recent historical observations.
   - Operates independently on `soil_moisture_1` and `soil_moisture_2`.

2. Exponentially Weighted Moving Average (EWMA) with Sudden-Change Bypass:
   - Smooths continuous high-frequency noise and thermal drift.
   - Adapts dynamic weighting: when a genuine physical transition occurs
     (e.g., active irrigation infiltration or torrential precipitation with |Δ| > threshold),
     the filter fast-tracks / snaps to the new level rather than lagging behind.
   - Operates independently on both probes.

3. Hysteresis State Machine (Actuator Chatter Prevention):
   - Prevents actuator flapping/oscillation around the Management Allowed Depletion
     Point (MADP) and Field Capacity (FC).
   - Requires moisture to drop cleanly below `(MADP - margin)` to trigger irrigation,
     and only deactivates irrigation when moisture recovers above `(MADP + margin)`.
   - Preserves original raw readings alongside filtered outputs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class StabilizedReading:
    """Output container preserving raw readings while exposing stabilized signals."""
    raw_1: float
    raw_2: float
    filtered_1: float
    filtered_2: float
    stabilized_avg: float
    is_sudden_change_1: bool
    is_sudden_change_2: bool
    irrigation_decision: bool
    irrigation_advisory: bool
    stress_status: str
    state_description: str


class MedianFilter:
    """
    1D causal sliding-window median filter.
    Preserves edge dynamics while discarding single-step impulse noise.
    """
    def __init__(self, window_size: int = 5):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if window_size % 2 == 0:
            raise ValueError("window_size should preferably be odd for symmetric median filtering")
        self.window_size = window_size
        self._buffer: deque[float] = deque(maxlen=window_size)

    def update(self, val: float) -> float:
        self._buffer.append(float(val))
        sorted_window = sorted(self._buffer)
        n = len(sorted_window)
        mid = n // 2
        if n % 2 == 1:
            return sorted_window[mid]
        return (sorted_window[mid - 1] + sorted_window[mid]) / 2.0

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def values(self) -> list[float]:
        return list(self._buffer)


class EWMASmoother:
    """
    Exponentially Weighted Moving Average (EWMA) with sudden change bypass.

    Formula:
        filtered_t = alpha * current + (1 - alpha) * filtered_{t-1}

    Sudden-Change Bypass:
        If |current - filtered_{t-1}| >= sudden_delta_threshold,
        the change is recognized as a genuine physical event (irrigation or rain infiltration),
        and the filter either bypasses smoothing or applies a high fast_alpha so the state
        tracks immediately without lag latency.
    """
    def __init__(
        self,
        alpha: float = 0.3,
        sudden_delta_threshold: float = 3.0,
        fast_alpha: float = 0.95,
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 < fast_alpha <= 1.0:
            raise ValueError("fast_alpha must be in (0, 1]")
        self.alpha = alpha
        self.sudden_delta_threshold = sudden_delta_threshold
        self.fast_alpha = fast_alpha
        self._last_estimate: float | None = None

    def update(self, val: float) -> tuple[float, bool]:
        """
        Updates the EWMA filter with a new observation.

        Returns:
            (filtered_value, is_sudden_change)
        """
        val = float(val)
        if self._last_estimate is None:
            self._last_estimate = val
            return val, False

        delta = abs(val - self._last_estimate)
        is_sudden = delta >= self.sudden_delta_threshold

        effective_alpha = self.fast_alpha if is_sudden else self.alpha
        smoothed = effective_alpha * val + (1.0 - effective_alpha) * self._last_estimate
        self._last_estimate = smoothed
        return smoothed, is_sudden

    def reset(self) -> None:
        self._last_estimate = None

    @property
    def current_estimate(self) -> float | None:
        return self._last_estimate


class HysteresisSwitch:
    """
    Dual-threshold Schmitt Trigger / Hysteresis State Machine.
    Prevents valve/actuator flapping when moisture hovers near the threshold.

    Rules for Irrigation Control:
      - When OFF: Turns ON only when moisture <= (threshold_low = threshold - margin).
      - When ON: Remains ON until moisture >= (threshold_high = threshold + margin).
    """
    def __init__(
        self,
        threshold: float,
        margin: float = 1.0,
        initial_state: bool = False,
    ):
        if margin < 0.0:
            raise ValueError("margin must be non-negative")
        self.threshold = threshold
        self.margin = margin
        self.state: bool = initial_state
        self.threshold_low = threshold - margin
        self.threshold_high = threshold + margin

    def update(self, current_value: float) -> bool:
        """
        Evaluate actuator state against current reading.

        For irrigation:
            Lower moisture -> needs water.
            If current_value <= threshold_low -> activate (True).
            If current_value >= threshold_high -> deactivate (False).
            Otherwise -> maintain previous state.
        """
        if not self.state:
            # Currently OFF: only turn ON if dropping below lower bound
            if current_value <= self.threshold_low:
                self.state = True
        else:
            # Currently ON: only turn OFF if rising above upper bound
            if current_value >= self.threshold_high:
                self.state = False
        return self.state

    def reset(self, initial_state: bool = False) -> None:
        self.state = initial_state


class SoilMoistureStabilizer:
    """
    Comprehensive dual-probe stabilization pipeline:
    1. Independent Median Filters (Impulse noise suppression for Probe 1 & Probe 2)
    2. Independent EWMA Smoothers with Sudden-Change Infiltration Detection
    3. Hysteresis Switch on stabilized average moisture to prevent control chatter
    """
    def __init__(
        self,
        median_window: int = 5,
        ewma_alpha: float = 0.3,
        sudden_change_threshold: float = 3.0,
        madp_threshold: float = 20.0,
        hysteresis_margin: float = 1.0,
    ):
        # Probe 1 pipeline
        self.med1 = MedianFilter(window_size=median_window)
        self.ewma1 = EWMASmoother(
            alpha=ewma_alpha,
            sudden_delta_threshold=sudden_change_threshold,
            fast_alpha=0.95,
        )

        # Probe 2 pipeline
        self.med2 = MedianFilter(window_size=median_window)
        self.ewma2 = EWMASmoother(
            alpha=ewma_alpha,
            sudden_delta_threshold=sudden_change_threshold,
            fast_alpha=0.95,
        )

        # Hysteresis controller (threshold = MADP)
        self.madp_threshold = float(madp_threshold)
        self.hysteresis = HysteresisSwitch(
            threshold=madp_threshold,
            margin=hysteresis_margin,
            initial_state=False,
        )

    def process_packet(
        self,
        soil_moisture_1: float,
        soil_moisture_2: float,
    ) -> StabilizedReading:
        """
        Process a single observation of dual probe readings.
        Preserves original raw values while generating stabilized signals.
        """
        sm1 = float(soil_moisture_1)
        sm2 = float(soil_moisture_2)

        # Step 1: Median filtering
        med1_val = self.med1.update(sm1)
        med2_val = self.med2.update(sm2)

        # Step 2: EWMA with sudden-change bypass
        filt1, sudden1 = self.ewma1.update(med1_val)
        filt2, sudden2 = self.ewma2.update(med2_val)

        # Combined stabilized average
        stabilized_avg = (filt1 + filt2) / 2.0
        stabilized_avg_rounded = round(stabilized_avg, 3)

        # Step 3: Hysteresis evaluation (actuator valve ON/OFF switching)
        irrigate = self.hysteresis.update(stabilized_avg)

        # Step 4: Irrigation advisory separated from hysteresis (strictly stabilized_avg < MADP)
        irrigation_advisory = stabilized_avg_rounded < self.madp_threshold
        stress_status = "IRRIGATION_STRESS" if irrigation_advisory else "NO_STRESS"

        if irrigate:
            state_desc = "IRRIGATING"
        elif irrigation_advisory:
            state_desc = "IRRIGATION_STRESS (STANDBY)"
        else:
            state_desc = "STANDBY"

        if sudden1 or sudden2:
            state_desc += " [SUDDEN_CHANGE_DETECTED]"

        return StabilizedReading(
            raw_1=sm1,
            raw_2=sm2,
            filtered_1=round(filt1, 3),
            filtered_2=round(filt2, 3),
            stabilized_avg=stabilized_avg_rounded,
            is_sudden_change_1=sudden1,
            is_sudden_change_2=sudden2,
            irrigation_decision=irrigate,
            irrigation_advisory=irrigation_advisory,
            stress_status=stress_status,
            state_description=state_desc,
        )

    def process_sequence(
        self,
        sm1_series: Sequence[float],
        sm2_series: Sequence[float],
    ) -> list[StabilizedReading]:
        """Process an entire sequence of dual probe readings chronologically."""
        if len(sm1_series) != len(sm2_series):
            raise ValueError("sm1_series and sm2_series must have identical lengths")
        results = []
        for s1, s2 in zip(sm1_series, sm2_series):
            results.append(self.process_packet(s1, s2))
        return results

    def reset(self) -> None:
        """Reset internal filter states."""
        self.med1.reset()
        self.med2.reset()
        self.ewma1.reset()
        self.ewma2.reset()
        self.hysteresis.reset()
