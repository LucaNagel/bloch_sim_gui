"""Rough gradient load of a sequence relative to the scanner limits.

The estimate compares each gradient axis with the amplitude and slew-rate
limits of the scanner profile. It is meant for display only: it models no
coil heating, amplifier duty cycle, peripheral nerve stimulation, or vendor
safety supervision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRADIENT_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class _AxisLoad:
    """Per-event load summary of one gradient axis, sorted by start time."""

    events: tuple
    starts_s: np.ndarray
    ends_s: np.ndarray
    peak_amplitude: np.ndarray
    peak_slew: np.ndarray
    # Slew into the first and out of the last raster cell of each event.
    leading_slew: np.ndarray
    trailing_slew: np.ndarray
    squared_area: float


class GradientLoadEstimate:
    """Amplitude and slew of a gradient program as fractions of the limits.

    ``max_gradient_hz_per_m`` and ``max_slew_hz_per_m_s`` are the scanner
    limits converted to the canonical frequency units of the events, so the
    estimate follows the gyromagnetic ratio of the imaged nucleus. Gradient
    events hold piecewise constant raster cells. Their slew is the step
    between neighbouring cells over one raster interval; an event that starts
    or ends at zero ramps over half a cell, as a Pulseq trapezoid does.
    """

    def __init__(
        self,
        gradient_events,
        *,
        duration_s: float,
        max_gradient_hz_per_m: float,
        max_slew_hz_per_m_s: float,
    ):
        for name, value in (
            ("max_gradient_hz_per_m", max_gradient_hz_per_m),
            ("max_slew_hz_per_m_s", max_slew_hz_per_m_s),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self.duration_s = max(float(duration_s), 0.0)
        self.max_gradient_hz_per_m = float(max_gradient_hz_per_m)
        self.max_slew_hz_per_m_s = float(max_slew_hz_per_m_s)
        self._axes = {
            axis: self._summarize_axis(
                [event for event in gradient_events if event.axis == axis]
            )
            for axis in GRADIENT_AXES
        }

    def _summarize_axis(self, events) -> _AxisLoad:
        events = sorted(events, key=lambda event: event.start_s)
        count = len(events)
        peak_amplitude = np.zeros(count)
        peak_slew = np.zeros(count)
        leading = np.zeros(count)
        trailing = np.zeros(count)
        squared_area = 0.0
        previous = None
        for index, event in enumerate(events):
            samples = np.asarray(event.samples_hz_per_m, dtype=float)
            raster = float(event.raster_s)
            if samples.size == 0:
                previous = None
                continue
            peak_amplitude[index] = np.max(np.abs(samples))
            squared_area += float(np.sum(samples**2) * raster)
            joined = (
                previous is not None
                and abs(previous.end_s - event.start_s) <= 0.5 * raster
                and np.isclose(previous.raster_s, raster)
            )
            if joined:
                step = samples[0] - previous.samples_hz_per_m[-1]
                leading[index] = abs(step) / raster
                # The joint belongs to both events; it replaces the half-cell
                # ramp to zero assumed for the previous event's end.
                trailing[index - 1] = leading[index]
            else:
                leading[index] = 2.0 * abs(samples[0]) / raster
            trailing[index] = 2.0 * abs(samples[-1]) / raster
            inner = (
                np.max(np.abs(np.diff(samples))) / raster if samples.size > 1 else 0.0
            )
            peak_slew[index] = inner
            previous = event
        peak_slew = np.maximum(peak_slew, np.maximum(leading, trailing))
        scale_g = 1.0 / self.max_gradient_hz_per_m
        scale_s = 1.0 / self.max_slew_hz_per_m_s
        return _AxisLoad(
            events=tuple(events),
            starts_s=np.array([event.start_s for event in events], dtype=float),
            ends_s=np.array([event.end_s for event in events], dtype=float),
            peak_amplitude=peak_amplitude * scale_g,
            peak_slew=peak_slew * scale_s,
            leading_slew=leading * scale_s,
            trailing_slew=trailing * scale_s,
            squared_area=squared_area * scale_g**2,
        )

    def peak_amplitude(self) -> tuple[float, str]:
        """Return the largest amplitude fraction and its axis."""
        return self._peak("peak_amplitude")

    def peak_slew(self) -> tuple[float, str]:
        """Return the largest slew fraction and its axis."""
        return self._peak("peak_slew")

    def _peak(self, attribute) -> tuple[float, str]:
        best = (0.0, "")
        for axis, load in self._axes.items():
            values = getattr(load, attribute)
            if values.size and float(np.max(values)) > best[0]:
                best = (float(np.max(values)), axis)
        return best

    def rms_amplitude(self) -> tuple[float, str]:
        """Return the largest time-averaged RMS amplitude fraction and its axis."""
        if self.duration_s <= 0:
            return 0.0, ""
        best = (0.0, "")
        for axis, load in self._axes.items():
            rms = float(np.sqrt(load.squared_area / self.duration_s))
            if rms > best[0]:
                best = (rms, axis)
        return best

    def binned(self, start_s: float, end_s: float, bins: int):
        """Return bin edges and the peak amplitude and slew fraction per bin.

        Each bin holds the largest value over all axes inside it. Events
        shorter than a bin contribute their peaks without being resampled, so
        an overview of a long sequence stays cheap.
        """
        bins = max(1, int(bins))
        start_s = float(start_s)
        end_s = max(float(end_s), start_s + 1e-12)
        edges = np.linspace(start_s, end_s, bins + 1)
        width = (end_s - start_s) / bins
        amplitude = np.zeros(bins)
        slew = np.zeros(bins)
        for load in self._axes.values():
            if not load.events:
                continue
            visible = np.flatnonzero((load.ends_s > start_s) & (load.starts_s < end_s))
            if visible.size == 0:
                continue
            first_bin = np.clip(
                ((load.starts_s[visible] - start_s) / width).astype(int), 0, bins - 1
            )
            last_bin = np.clip(
                ((load.ends_s[visible] - start_s) / width).astype(int), 0, bins - 1
            )
            # An event inside one bin contributes its peaks directly; events
            # cut by the window edge are resampled so that only their visible
            # part counts.
            short = (
                (first_bin == last_bin)
                & (load.starts_s[visible] >= start_s)
                & (load.ends_s[visible] <= end_s)
            )
            np.maximum.at(
                amplitude, first_bin[short], load.peak_amplitude[visible][short]
            )
            np.maximum.at(slew, first_bin[short], load.peak_slew[visible][short])
            for index in visible[~short]:
                self._bin_event(load, int(index), start_s, width, amplitude, slew)
        return edges, amplitude, slew

    def _bin_event(self, load, index, start_s, width, amplitude, slew):
        event = load.events[index]
        samples = np.abs(np.asarray(event.samples_hz_per_m, dtype=float))
        raster = float(event.raster_s)
        bins = amplitude.size
        centres = event.start_s + (np.arange(samples.size) + 0.5) * raster
        cell_bins = np.clip(((centres - start_s) / width).astype(int), 0, bins - 1)
        inside = (centres >= start_s) & (centres < start_s + bins * width)
        np.maximum.at(
            amplitude, cell_bins[inside], samples[inside] / self.max_gradient_hz_per_m
        )
        raw = np.asarray(event.samples_hz_per_m, dtype=float)
        steps = np.concatenate(
            (
                [load.leading_slew[index]],
                np.abs(np.diff(raw)) / raster / self.max_slew_hz_per_m_s,
                [load.trailing_slew[index]],
            )
        )
        boundaries = event.start_s + np.arange(samples.size + 1) * raster
        boundary_bins = np.clip(
            ((boundaries - start_s) / width).astype(int), 0, bins - 1
        )
        inside = (boundaries >= start_s) & (boundaries <= start_s + bins * width)
        np.maximum.at(slew, boundary_bins[inside], steps[inside])
