"""Sparse compiler for canonical sequence programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

from .model import ADCEvent, GradientEvent, RFEvent, SequenceProgram


@dataclass(frozen=True)
class CompiledSequence:
    """Interval representation consumed by the streaming Bloch kernel."""

    dt_s: np.ndarray
    interval_end_s: np.ndarray
    rf_hz: np.ndarray
    gradient_hz_per_m: np.ndarray
    adc_times_s: np.ndarray
    adc_state_indices: np.ndarray
    adc_demodulation: np.ndarray
    checkpoint_times_s: np.ndarray
    checkpoint_state_indices: np.ndarray
    duration_s: float
    metadata: dict

    def __post_init__(self) -> None:
        for name in (
            "dt_s",
            "interval_end_s",
            "rf_hz",
            "gradient_hz_per_m",
            "adc_times_s",
            "adc_state_indices",
            "adc_demodulation",
            "checkpoint_times_s",
            "checkpoint_state_indices",
        ):
            value = np.ascontiguousarray(getattr(self, name))
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def n_intervals(self) -> int:
        return int(self.dt_s.size)


class SequenceCompiler:
    """Compile events while collapsing RF-free gradient evolution."""

    def compile(
        self,
        program: SequenceProgram,
        *,
        checkpoints_s: Sequence[float] = (),
    ) -> CompiledSequence:
        checkpoints = self._validate_checkpoints(checkpoints_s, program.duration_s)
        self._validate_gradient_overlaps(program.gradient_events)

        adc_times, adc_demod = self._adc_samples(program.adc_events)
        boundaries = self._build_boundaries(program, adc_times, checkpoints)
        if boundaries.size == 1:
            dt = np.zeros(0, dtype=float)
            interval_end = np.zeros(0, dtype=float)
            rf = np.zeros(0, dtype=np.complex128)
            gradients = np.zeros((0, 3), dtype=float)
        else:
            starts = boundaries[:-1]
            interval_end = boundaries[1:]
            dt = interval_end - starts
            rf_integrals = np.zeros(dt.size, dtype=np.complex128)
            gradient_integrals = np.zeros((dt.size, 3), dtype=float)

            # Accumulate each event only into intervals it overlaps. Iterating over
            # every event for every interval made full GRE/EPI programs quadratic
            # in practice, even though the event collection is sparse.
            for event in program.rf_events:
                first, last = self._overlapping_interval_range(
                    interval_end, starts, event.start_s, event.end_s
                )
                for index in range(first, last):
                    rf_integrals[index] += self._rf_event_integral(
                        event, starts[index], interval_end[index]
                    )
            for event in program.gradient_events:
                axis_index = "xyz".index(event.axis)
                first, last = self._overlapping_interval_range(
                    interval_end, starts, event.start_s, event.end_s
                )
                for index in range(first, last):
                    gradient_integrals[index, axis_index] += self._event_integral(
                        event.start_s,
                        event.raster_s,
                        event.samples_hz_per_m,
                        starts[index],
                        interval_end[index],
                    )
            rf = rf_integrals / dt
            gradients = gradient_integrals / dt[:, None]

        adc_states = self._times_to_state_indices(boundaries, adc_times)
        checkpoint_states = self._times_to_state_indices(boundaries, checkpoints)
        return CompiledSequence(
            dt_s=np.asarray(dt, dtype=np.float64),
            interval_end_s=np.asarray(interval_end, dtype=np.float64),
            rf_hz=np.asarray(rf, dtype=np.complex128),
            gradient_hz_per_m=np.asarray(gradients, dtype=np.float64),
            adc_times_s=adc_times,
            adc_state_indices=adc_states,
            adc_demodulation=adc_demod,
            checkpoint_times_s=checkpoints,
            checkpoint_state_indices=checkpoint_states,
            duration_s=program.duration_s,
            metadata={
                "source": program.source,
                "version": program.version,
                "program_metadata": dict(program.metadata),
            },
        )

    @staticmethod
    def _validate_checkpoints(values: Sequence[float], duration: float) -> np.ndarray:
        checkpoints = np.asarray(tuple(values), dtype=float)
        if checkpoints.size == 0:
            return np.zeros(0, dtype=float)
        if checkpoints.ndim != 1 or not np.all(np.isfinite(checkpoints)):
            raise ValueError("checkpoints_s must be a finite one-dimensional sequence")
        if np.any(checkpoints < 0) or np.any(checkpoints > duration):
            raise ValueError("checkpoint times must lie within the sequence")
        return np.unique(checkpoints)

    @staticmethod
    def _validate_gradient_overlaps(events: Iterable[GradientEvent]) -> None:
        for axis in "xyz":
            axis_events = sorted(
                (event for event in events if event.axis == axis),
                key=lambda event: event.start_s,
            )
            previous = None
            for event in axis_events:
                if previous is not None:
                    tolerance = max(1e-15, max(previous.end_s, event.end_s) * 1e-12)
                    if event.start_s < previous.end_s - tolerance:
                        raise ValueError(
                            f"overlapping gradient events on axis {axis}: "
                            f"{previous.start_s}..{previous.end_s} and "
                            f"{event.start_s}..{event.end_s}"
                        )
                previous = event

    def _build_boundaries(
        self,
        program: SequenceProgram,
        adc_times: np.ndarray,
        checkpoints: np.ndarray,
    ) -> np.ndarray:
        values = [0.0, program.duration_s]
        values.extend(adc_times.tolist())
        values.extend(checkpoints.tolist())
        for event in program.events:
            values.extend((event.start_s, event.end_s))
        rf_ranges = [(event.start_s, event.end_s) for event in program.rf_events]
        for event in program.rf_events:
            values.extend(
                (
                    event.start_s + np.arange(1, event.samples_hz.size) * event.raster_s
                ).tolist()
            )
        for event in program.gradient_events:
            for rf_start, rf_end in rf_ranges:
                overlap_start = max(event.start_s, rf_start)
                overlap_end = min(event.end_s, rf_end)
                if overlap_start >= overlap_end:
                    continue
                first = max(
                    1,
                    int(np.floor((overlap_start - event.start_s) / event.raster_s)) + 1,
                )
                last = min(
                    event.samples_hz_per_m.size - 1,
                    int(np.ceil((overlap_end - event.start_s) / event.raster_s)) - 1,
                )
                if last >= first:
                    values.extend(
                        (
                            event.start_s
                            + np.arange(first, last + 1, dtype=float) * event.raster_s
                        ).tolist()
                    )
        boundaries = np.unique(np.asarray(values, dtype=float))
        tolerance = max(1e-15, max(1.0, program.duration_s) * 1e-13)
        boundaries[np.abs(boundaries) < tolerance] = 0.0
        boundaries[np.abs(boundaries - program.duration_s) < tolerance] = (
            program.duration_s
        )
        boundaries = boundaries[(boundaries >= 0) & (boundaries <= program.duration_s)]
        return np.unique(boundaries)

    @staticmethod
    def _rf_event_integral(event: RFEvent, start: float, end: float):
        overlap_start = max(start, event.start_s)
        overlap_end = min(end, event.end_s)
        if overlap_start >= overlap_end:
            return 0j
        first = max(0, int(np.floor((overlap_start - event.start_s) / event.raster_s)))
        last = min(
            event.samples_hz.size - 1,
            int(np.ceil((overlap_end - event.start_s) / event.raster_s)) - 1,
        )
        total = 0j
        for index in range(first, last + 1):
            cell_start = event.start_s + index * event.raster_s
            cell_end = cell_start + event.raster_s
            width = max(
                0.0, min(overlap_end, cell_end) - max(overlap_start, cell_start)
            )
            phase = event.phase_offset_rad + 2 * np.pi * event.frequency_offset_hz * (
                cell_start - event.start_s
            )
            total += event.samples_hz[index] * np.exp(1j * phase) * width
        return total

    @staticmethod
    def _overlapping_interval_range(
        interval_end: np.ndarray,
        interval_start: np.ndarray,
        event_start: float,
        event_end: float,
    ) -> Tuple[int, int]:
        """Return the half-open interval index range overlapping one event."""
        first = int(np.searchsorted(interval_end, event_start, side="right"))
        last = int(np.searchsorted(interval_start, event_end, side="left"))
        return first, last

    @staticmethod
    def _event_integral(
        event_start: float,
        raster: float,
        samples: np.ndarray,
        start: float,
        end: float,
    ):
        overlap_start = max(start, event_start)
        event_end = event_start + samples.size * raster
        overlap_end = min(end, event_end)
        if overlap_start >= overlap_end:
            return samples.dtype.type(0)
        first = max(0, int(np.floor((overlap_start - event_start) / raster)))
        last = min(
            samples.size - 1, int(np.ceil((overlap_end - event_start) / raster)) - 1
        )
        total = samples.dtype.type(0)
        for index in range(first, last + 1):
            cell_start = event_start + index * raster
            cell_end = cell_start + raster
            width = max(
                0.0, min(overlap_end, cell_end) - max(overlap_start, cell_start)
            )
            total += samples[index] * width
        return total

    @staticmethod
    def _adc_samples(events: Tuple[ADCEvent, ...]) -> Tuple[np.ndarray, np.ndarray]:
        samples = []
        order = 0
        for event in events:
            for time_value in event.sample_times_s:
                relative_time = time_value - event.start_s
                receiver_phase = (
                    event.phase_offset_rad
                    + 2 * np.pi * event.frequency_offset_hz * relative_time
                )
                samples.append((float(time_value), order, np.exp(-1j * receiver_phase)))
                order += 1
        samples.sort(key=lambda item: (item[0], item[1]))
        return (
            np.asarray([item[0] for item in samples], dtype=float),
            np.asarray([item[2] for item in samples], dtype=np.complex128),
        )

    @staticmethod
    def _times_to_state_indices(
        boundaries: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        if times.size == 0:
            return np.zeros(0, dtype=np.int32)
        indices = np.searchsorted(boundaries, times, side="left")
        if np.any(indices >= boundaries.size) or not np.allclose(
            boundaries[indices], times, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("compiler failed to place an observation on a boundary")
        if np.any(indices > np.iinfo(np.int32).max):
            raise ValueError("compiled sequence has too many state boundaries")
        return indices.astype(np.int32)
