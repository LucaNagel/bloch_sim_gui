"""Sparse compiler for canonical sequence programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence, Tuple

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
    adc_gradient_moment_cyc_per_m: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float)
    )

    def __post_init__(self) -> None:
        for name in (
            "dt_s",
            "interval_end_s",
            "rf_hz",
            "gradient_hz_per_m",
            "adc_times_s",
            "adc_state_indices",
            "adc_demodulation",
            "adc_gradient_moment_cyc_per_m",
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
        simulation_timestep_s: float | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> CompiledSequence:
        def status(message: str) -> None:
            if status_callback is not None:
                status_callback(message)

        if simulation_timestep_s is not None and (
            not np.isfinite(simulation_timestep_s) or simulation_timestep_s <= 0
        ):
            raise ValueError("simulation_timestep_s must be finite and positive")

        status(f"Validating {len(program.events):,} sequence events…")
        checkpoints = self._validate_checkpoints(checkpoints_s, program.duration_s)
        self._validate_gradient_overlaps(program.gradient_events)

        status(f"Expanding {len(program.adc_events):,} ADC events…")
        adc_times, adc_demod = self._adc_samples(program.adc_events)
        resolution = (
            "native event rasters"
            if simulation_timestep_s is None
            else f"{simulation_timestep_s * 1e6:g} µs RF-active time step"
        )
        status(f"Building the sparse sequence timeline ({resolution})…")
        boundaries = self._build_boundaries(
            program,
            adc_times,
            checkpoints,
            simulation_timestep_s,
        )
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
            status(f"Rasterizing {len(program.rf_events):,} RF events…")
            for event in program.rf_events:
                first, last = self._overlapping_interval_range(
                    interval_end, starts, event.start_s, event.end_s
                )
                for index in range(first, last):
                    rf_integrals[index] += self._rf_event_integral(
                        event, starts[index], interval_end[index]
                    )
            status(f"Rasterizing {len(program.gradient_events):,} gradient events…")
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

        status(f"Finalizing {dt.size:,} compiled intervals…")
        adc_states = self._times_to_state_indices(boundaries, adc_times)
        checkpoint_states = self._times_to_state_indices(boundaries, checkpoints)
        state_gradient_moments = np.vstack(
            (
                np.zeros((1, 3), dtype=float),
                np.cumsum(gradients * dt[:, None], axis=0),
            )
        )
        return CompiledSequence(
            dt_s=np.asarray(dt, dtype=np.float64),
            interval_end_s=np.asarray(interval_end, dtype=np.float64),
            rf_hz=np.asarray(rf, dtype=np.complex128),
            gradient_hz_per_m=np.asarray(gradients, dtype=np.float64),
            adc_times_s=adc_times,
            adc_state_indices=adc_states,
            adc_demodulation=adc_demod,
            adc_gradient_moment_cyc_per_m=state_gradient_moments[adc_states],
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
        simulation_timestep_s: float | None,
    ) -> np.ndarray:
        values = [0.0, program.duration_s]
        values.extend(adc_times.tolist())
        values.extend(checkpoints.tolist())
        for event in program.events:
            values.extend((event.start_s, event.end_s))
        rf_ranges = sorted(
            ((event.start_s, event.end_s) for event in program.rf_events),
            key=lambda value: value[0],
        )
        rf_starts = np.asarray([value[0] for value in rf_ranges], dtype=float)
        rf_ends = np.asarray([value[1] for value in rf_ranges], dtype=float)
        rf_max_ends = np.maximum.accumulate(rf_ends)
        for event in program.rf_events:
            step = (
                event.raster_s
                if simulation_timestep_s is None
                else max(event.raster_s, simulation_timestep_s)
            )
            internal_count = max(
                0,
                int(np.ceil((event.end_s - event.start_s) / step)) - 1,
            )
            values.extend(
                (event.start_s + np.arange(1, internal_count + 1) * step).tolist()
            )
        if simulation_timestep_s is not None:
            return self._coalesce_boundaries(values, program.duration_s)
        for event in program.gradient_events:
            # Only RF events that can overlap this gradient need inspection.
            # The former all-gradient x all-RF loop dominated compilation of
            # long 3D acquisitions even though most pairs are far apart.
            first_rf = int(np.searchsorted(rf_max_ends, event.start_s, side="right"))
            for rf_index in range(first_rf, rf_starts.size):
                rf_start = rf_starts[rf_index]
                if rf_start >= event.end_s:
                    break
                rf_end = rf_ends[rf_index]
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
        return self._coalesce_boundaries(values, program.duration_s)

    @staticmethod
    def _coalesce_boundaries(values, duration: float) -> np.ndarray:
        boundaries = np.sort(np.asarray(values, dtype=float))
        tolerance = max(
            1e-15,
            32 * np.spacing(max(1.0, abs(duration))),
        )
        boundaries[np.abs(boundaries) < tolerance] = 0.0
        boundaries[np.abs(boundaries - duration) < tolerance] = duration
        boundaries = boundaries[(boundaries >= 0) & (boundaries <= duration)]
        if boundaries.size == 0:
            return boundaries
        coalesced = [float(boundaries[0])]
        group_anchor = float(boundaries[0])
        for value in boundaries[1:]:
            value = float(value)
            if value - group_anchor <= tolerance:
                if value == duration:
                    coalesced[-1] = value
                continue
            coalesced.append(value)
            group_anchor = value
        return np.asarray(coalesced, dtype=float)

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
        insertions = np.searchsorted(boundaries, times, side="left")
        right = np.clip(insertions, 0, boundaries.size - 1)
        left = np.clip(insertions - 1, 0, boundaries.size - 1)
        right_distance = np.abs(boundaries[right] - times)
        left_distance = np.abs(boundaries[left] - times)
        indices = np.where(left_distance <= right_distance, left, right)
        tolerance = max(
            1e-12,
            32 * np.spacing(max(1.0, abs(float(boundaries[-1])))),
        )
        if np.any(np.abs(boundaries[indices] - times) > tolerance):
            raise RuntimeError("compiler failed to place an observation on a boundary")
        if np.any(indices > np.iinfo(np.int32).max):
            raise ValueError("compiled sequence has too many state boundaries")
        return indices.astype(np.int32)
