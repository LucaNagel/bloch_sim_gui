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
    transverse_crush_times_s: np.ndarray
    transverse_crush_state_indices: np.ndarray
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
            "transverse_crush_times_s",
            "transverse_crush_state_indices",
        ):
            value = np.ascontiguousarray(getattr(self, name))
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def n_intervals(self) -> int:
        return int(self.dt_s.size)


class SequenceCompiler:
    """Compile events while collapsing RF-free gradient evolution."""

    def simulation_state_times(
        self,
        program: SequenceProgram,
        *,
        simulation_timestep_s: float | None = None,
    ) -> np.ndarray:
        """Return the initial time and every compiled interval end.

        Event boundaries and ADC sample times are retained even when no RF or
        ADC is active at neighbouring intervals.
        """
        if simulation_timestep_s is not None and (
            not np.isfinite(simulation_timestep_s) or simulation_timestep_s <= 0
        ):
            raise ValueError("simulation_timestep_s must be finite and positive")
        self._validate_gradient_overlaps(program.gradient_events)
        adc_times, _ = self._adc_samples(program.adc_events)
        boundaries = self._build_boundaries(
            program,
            adc_times,
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            simulation_timestep_s,
            False,
        )
        boundaries = np.ascontiguousarray(boundaries, dtype=np.float64)
        boundaries.setflags(write=False)
        return boundaries

    def compile(
        self,
        program: SequenceProgram,
        *,
        checkpoints_s: Sequence[float] = (),
        extra_boundaries_s: Sequence[float] = (),
        simulation_timestep_s: float | None = None,
        acquisition_only: bool = False,
        status_callback: Callable[[str], None] | None = None,
    ) -> CompiledSequence:
        def status(message: str) -> None:
            if status_callback is not None:
                status_callback(message)

        if simulation_timestep_s is not None and (
            not np.isfinite(simulation_timestep_s) or simulation_timestep_s <= 0
        ):
            raise ValueError("simulation_timestep_s must be finite and positive")
        if acquisition_only and simulation_timestep_s is not None:
            raise ValueError(
                "acquisition-only compilation does not use an RF simulation time step"
            )

        status(f"Validating {len(program.events):,} sequence events…")
        checkpoints = self._validate_checkpoints(checkpoints_s, program.duration_s)
        extra_boundaries = self._validate_extra_boundaries(
            extra_boundaries_s, program.duration_s
        )
        transverse_crush_times = self._ideal_spoiler_times(program, program.duration_s)
        timeline_boundaries = np.unique(
            np.concatenate((extra_boundaries, transverse_crush_times))
        )
        self._validate_gradient_overlaps(program.gradient_events)

        status(f"Expanding {len(program.adc_events):,} ADC events…")
        adc_times, adc_demod = self._adc_samples(program.adc_events)
        if acquisition_only:
            resolution = "acquisition boundaries without RF raster expansion"
        else:
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
            timeline_boundaries,
            simulation_timestep_s,
            acquisition_only,
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
            if acquisition_only:
                status("Skipping RF rasterization for acquisition inference…")
            else:
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
        transverse_crush_states = self._times_to_state_indices(
            boundaries, transverse_crush_times
        )
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
            transverse_crush_times_s=transverse_crush_times,
            transverse_crush_state_indices=transverse_crush_states,
            duration_s=program.duration_s,
            metadata={
                "source": program.source,
                "version": program.version,
                "program_metadata": dict(program.metadata),
                "extra_boundary_count": int(extra_boundaries.size),
                "transverse_crush_count": int(transverse_crush_times.size),
                "acquisition_only": bool(acquisition_only),
            },
        )

    def compile_acquisition(
        self,
        program: SequenceProgram,
        *,
        status_callback: Callable[[str], None] | None = None,
    ) -> CompiledSequence:
        """Compile exact ADC times and gradient moments without expanding RF rasters.

        The returned timeline retains every event boundary and ADC observation, so
        Cartesian/CSI inference and plotting see the same gradient moments as a full
        compilation. RF amplitudes are intentionally zero because no Bloch evolution
        is performed with this lightweight representation.
        """
        return self.compile(
            program,
            acquisition_only=True,
            status_callback=status_callback,
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
    def _validate_extra_boundaries(
        values: Sequence[float], duration: float
    ) -> np.ndarray:
        boundaries = np.asarray(tuple(values), dtype=float)
        if boundaries.size == 0:
            return np.zeros(0, dtype=float)
        if boundaries.ndim != 1 or not np.all(np.isfinite(boundaries)):
            raise ValueError(
                "extra_boundaries_s must be a finite one-dimensional sequence"
            )
        if np.any(boundaries < 0) or np.any(boundaries > duration):
            raise ValueError("extra sequence boundaries must lie within the sequence")
        return np.unique(boundaries)

    @classmethod
    def _ideal_spoiler_times(
        cls, program: SequenceProgram, duration: float
    ) -> np.ndarray:
        """Return explicitly declared ideal transverse-crusher times.

        Generated sequences use ``IdealSpoilerEndTimes``.  The two older keys
        remain accepted so previously exported generated Pulseq files acquire
        the same semantics when they are loaded again.  No gradient waveform
        is inspected or classified heuristically.
        """
        definitions = program.metadata.get("definitions", {})
        if not isinstance(definitions, dict):
            return np.zeros(0, dtype=float)

        if "IdealSpoilerEndTimes" in definitions:
            raw_values = definitions.get("IdealSpoilerEndTimes")
        else:
            legacy_values = []
            for key in ("SpoilerEndTimes", "EndImageSpoilerEndTimes"):
                value = definitions.get(key)
                if value is None:
                    continue
                array = np.asarray(value, dtype=float)
                if array.ndim == 0:
                    array = array.reshape(1)
                legacy_values.extend(array.reshape(-1).tolist())
            raw_values = legacy_values

        if raw_values is None:
            return np.zeros(0, dtype=float)
        values = np.asarray(raw_values, dtype=float)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.size == 0:
            return np.zeros(0, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(
                "IdealSpoilerEndTimes must be a finite one-dimensional sequence"
            )
        tolerance = max(1e-15, abs(duration) * 1e-12)
        if np.any(values < -tolerance) or np.any(values > duration + tolerance):
            raise ValueError("ideal spoiler end times must lie within the sequence")
        # Pulseq definitions are text while the reloaded event duration is a
        # floating-point sum.  Snap sub-raster round-off at either endpoint so
        # the crusher maps to an existing simulation state exactly.
        return np.unique(np.clip(values, 0.0, duration))

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
        extra_boundaries: np.ndarray,
        simulation_timestep_s: float | None,
        acquisition_only: bool,
    ) -> np.ndarray:
        values = [0.0, program.duration_s]
        values.extend(adc_times.tolist())
        values.extend(checkpoints.tolist())
        values.extend(extra_boundaries.tolist())
        for event in program.events:
            values.extend((event.start_s, event.end_s))
        if acquisition_only:
            return self._coalesce_boundaries(values, program.duration_s)
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
