"""Sparse compiler for canonical sequence programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence, Tuple

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
    # Absolute interval midpoints and the half-step durations (mid - start,
    # end - mid). Inside an RF pulse the compiler evaluates both on the
    # pulse-relative grid, so they need not equal differences of absolute
    # boundary times. Derived from the boundaries when omitted.
    interval_mid_s: np.ndarray | None = None
    interval_half_s: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.interval_mid_s is None or self.interval_half_s is None:
            dt = np.asarray(self.dt_s, dtype=np.float64)
            end = np.asarray(self.interval_end_s, dtype=np.float64)
            start = np.concatenate(([0.0], end[:-1])) if end.size else end
            middle = start + dt / 2.0
            if self.interval_mid_s is None:
                object.__setattr__(self, "interval_mid_s", middle)
            if self.interval_half_s is None:
                object.__setattr__(
                    self,
                    "interval_half_s",
                    np.column_stack((middle - start, end - middle)).reshape(-1, 2),
                )
        for name in (
            "dt_s",
            "interval_end_s",
            "interval_mid_s",
            "interval_half_s",
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


IDEAL_SPOILER_DEFINITION = "IdealSpoilerEndTimes"
LEGACY_IDEAL_SPOILER_DEFINITIONS = ("SpoilerEndTimes", "EndImageSpoilerEndTimes")


def declares_ideal_spoilers(program: SequenceProgram) -> bool:
    """Whether a program states where its ideal transverse crushers are.

    Sequences exported by this simulator carry these definitions, even when
    the list is empty. Ideal-crusher simulation is then exact for them.
    """
    definitions = (getattr(program, "metadata", None) or {}).get("definitions")
    if not isinstance(definitions, dict):
        return False
    return any(
        key in definitions
        for key in (IDEAL_SPOILER_DEFINITION, *LEGACY_IDEAL_SPOILER_DEFINITIONS)
    )


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
        definitions = program.metadata.get("definitions", {})
        if not isinstance(definitions, dict):
            definitions = {}
        checkpoints = self._validate_checkpoints(checkpoints_s, program.duration_s)
        declared = np.asarray(
            definitions.get("MagnetizationCheckpointTimes", ()), dtype=float
        )
        if declared.ndim == 0:
            declared = declared.reshape(1)
        # Definitions undergo text rounding during Pulseq export/import.
        tolerance = max(1e-15, program.duration_s * 1e-12)
        if np.any(declared < -tolerance) or np.any(
            declared > program.duration_s + tolerance
        ):
            raise ValueError(
                "declared magnetization checkpoints must lie within the sequence"
            )
        declared = self._validate_checkpoints(
            np.clip(declared, 0.0, program.duration_s), program.duration_s
        )
        checkpoints = np.unique(np.concatenate((checkpoints, declared)))
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
        rf_events = program.rf_events
        frames = {}
        if boundaries.size == 1:
            dt = np.zeros(0, dtype=float)
            interval_end = np.zeros(0, dtype=float)
            interval_mid = np.zeros(0, dtype=float)
            interval_half = np.zeros((0, 2), dtype=float)
            rf = np.zeros(0, dtype=np.complex128)
            gradients = np.zeros((0, 3), dtype=float)
        else:
            starts = boundaries[:-1]
            interval_end = boundaries[1:]
            dt = interval_end - starts
            interval_mid = starts + dt / 2.0
            interval_half = np.column_stack(
                (interval_mid - starts, interval_end - interval_mid)
            )
            # RF pulses are compiled on their own pulse-relative time grid:
            # absolute boundaries carry the rounding of the pulse start, so
            # identical pulses at different times would otherwise get
            # slightly different dt, half steps, RF and gradient integrals.
            # The boundaries themselves stay absolute.
            if not acquisition_only:
                frames = self._rf_pulse_frames(
                    rf_events,
                    boundaries,
                    simulation_timestep_s,
                    self._time_tolerance(program.duration_s),
                )
            frame_of_interval = np.full(dt.size, -1, dtype=np.int64)
            local_start = np.zeros(dt.size, dtype=float)
            local_end = np.zeros(dt.size, dtype=float)
            for event_index, (first, last, local) in frames.items():
                lower = local[:-1]
                upper = local[1:]
                width = upper - lower
                middle = lower + width / 2.0
                frame_of_interval[first:last] = event_index
                local_start[first:last] = lower
                local_end[first:last] = upper
                dt[first:last] = width
                interval_mid[first:last] = rf_events[event_index].start_s + middle
                interval_half[first:last, 0] = middle - lower
                interval_half[first:last, 1] = upper - middle
            rf_integrals = np.zeros(dt.size, dtype=np.complex128)

            # Accumulate each event only into intervals it overlaps. Iterating over
            # every event for every interval made full GRE/EPI programs quadratic
            # in practice, even though the event collection is sparse.
            if acquisition_only:
                status("Skipping RF rasterization for acquisition inference…")
            else:
                status(f"Rasterizing {len(rf_events):,} RF events…")
                rf_ranges = self._overlapping_interval_ranges(
                    interval_end, starts, rf_events
                )
                frame_cache = {}
                for event_index, event in enumerate(rf_events):
                    frame = frames.get(event_index)
                    if frame is not None:
                        rf_integrals[frame[0] : frame[1]] += self._rf_frame_integrals(
                            event, frame[2], frame_cache
                        )
                        continue
                    first, last = rf_ranges[event_index]
                    for index in range(first, last):
                        rf_integrals[index] += self._rf_event_integral(
                            event, starts[index], interval_end[index]
                        )
            status(f"Rasterizing {len(program.gradient_events):,} gradient events…")
            tolerance = self._time_tolerance(program.duration_s)
            gradient_ranges = self._overlapping_interval_ranges(
                interval_end, starts, program.gradient_events
            )
            gradient_integrals = self._gradient_integrals(
                program.gradient_events,
                gradient_ranges,
                starts,
                interval_end,
                frame_of_interval,
                local_start,
                local_end,
                frames,
                rf_events,
                simulation_timestep_s,
                tolerance,
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
            interval_mid_s=np.asarray(interval_mid, dtype=np.float64),
            interval_half_s=np.asarray(interval_half, dtype=np.float64),
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
                "pulse_relative_rf_events": len(frames),
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

        if IDEAL_SPOILER_DEFINITION in definitions:
            raw_values = definitions.get("IdealSpoilerEndTimes")
        else:
            legacy_values = []
            for key in LEGACY_IDEAL_SPOILER_DEFINITIONS:
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
        tolerance = self._time_tolerance(program.duration_s)
        for event in program.rf_events:
            grid = self._rf_local_grid(event, simulation_timestep_s, tolerance)
            values.extend((event.start_s + grid[1:-1]).tolist())
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
    def _time_tolerance(duration: float) -> float:
        """Return the distance below which two sequence times coincide."""
        return float(max(1e-15, 32 * np.spacing(max(1.0, abs(duration)))))

    @staticmethod
    def _rf_step(event: RFEvent, simulation_timestep_s: float | None) -> float:
        """Return the boundary step inside one RF event.

        A step at or above the RF raster is used as is. A finer step divides
        every RF sample into equal parts no longer than the step, so sample
        boundaries stay interval boundaries and the time step also bounds the
        RF-active interval length of coarsely rastered pulses.
        """
        if simulation_timestep_s is None:
            return event.raster_s
        if simulation_timestep_s >= event.raster_s:
            return simulation_timestep_s
        parts = int(np.ceil(event.raster_s / simulation_timestep_s * (1.0 - 1e-9)))
        return event.raster_s / parts

    @staticmethod
    def _rf_local_grid(
        event: RFEvent, simulation_timestep_s: float | None, tolerance: float
    ) -> np.ndarray:
        """Return the pulse-relative RF boundaries from 0 to the pulse duration.

        The grid depends only on the pulse itself, never on its start time.
        Steps closer than ``tolerance`` to the end coincide with it.
        """
        step = SequenceCompiler._rf_step(event, simulation_timestep_s)
        duration = event.samples_hz.size * event.raster_s
        internal = np.arange(1, max(1, int(np.ceil(duration / step))), dtype=float)
        internal = internal * step
        internal = internal[internal < duration - tolerance]
        return np.concatenate(([0.0], internal, [duration]))

    @classmethod
    def _rf_pulse_frames(
        cls,
        rf_events: Sequence[RFEvent],
        boundaries: np.ndarray,
        simulation_timestep_s: float | None,
        tolerance: float,
    ) -> dict:
        """Map RF event indices to (first interval, end interval, local times).

        The local times are the pulse's boundaries relative to its start,
        snapped to the pulse grid or the RF raster when they lie within
        ``tolerance`` of it, so the absolute rounding of the start cancels.
        Overlapping RF events and pulses whose boundaries do not tile the
        pulse exactly keep absolute compilation.
        """
        order = sorted(range(len(rf_events)), key=lambda i: rf_events[i].start_s)
        overlapping = set()
        cluster = []
        cluster_end = -np.inf
        for event_index in order:
            event = rf_events[event_index]
            if cluster and event.start_s < cluster_end - tolerance:
                cluster.append(event_index)
                cluster_end = max(cluster_end, event.end_s)
                continue
            if len(cluster) > 1:
                overlapping.update(cluster)
            cluster = [event_index]
            cluster_end = event.end_s
        if len(cluster) > 1:
            overlapping.update(cluster)

        frames = {}
        for event_index, event in enumerate(rf_events):
            if event_index in overlapping:
                continue
            grid = cls._rf_local_grid(event, simulation_timestep_s, tolerance)
            duration = grid[-1]
            if duration <= 2 * tolerance:
                continue
            first = cls._nearest_boundary(boundaries, event.start_s, tolerance)
            last = cls._nearest_boundary(boundaries, event.end_s, tolerance)
            if first is None or last is None or last <= first:
                continue
            local = cls._snap_local_times(
                boundaries[first : last + 1] - event.start_s,
                grid,
                event.raster_s,
                tolerance,
            )
            if (
                local[0] != 0.0
                or local[-1] != duration
                or np.any(local[1:] <= local[:-1])
            ):
                continue
            frames[event_index] = (first, last, local)
        return frames

    @staticmethod
    def _nearest_boundary(
        boundaries: np.ndarray, time_s: float, tolerance: float
    ) -> int | None:
        index = int(np.searchsorted(boundaries, time_s))
        candidates = [
            candidate
            for candidate in (index - 1, index)
            if 0 <= candidate < boundaries.size
            and abs(boundaries[candidate] - time_s) <= tolerance
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda value: abs(boundaries[value] - time_s))

    @staticmethod
    def _snap_local_times(
        offsets: np.ndarray, grid: np.ndarray, raster: float, tolerance: float
    ) -> np.ndarray:
        """Snap pulse-relative times to the pulse grid, then the RF raster."""
        offsets = np.asarray(offsets, dtype=float)
        index = np.clip(np.searchsorted(grid, offsets), 1, grid.size - 1)
        left = grid[index - 1]
        right = grid[index]
        nearest = np.where(offsets - left <= right - offsets, left, right)
        on_raster = np.round(offsets / raster) * raster
        return np.where(
            np.abs(offsets - nearest) <= tolerance,
            nearest,
            np.where(np.abs(offsets - on_raster) <= tolerance, on_raster, offsets),
        )

    @classmethod
    def _pulse_relative_offset(
        cls,
        event: RFEvent,
        local: np.ndarray,
        time_s: float,
        simulation_timestep_s: float | None,
        raster: float,
        tolerance: float,
    ) -> float:
        """Return ``time_s`` relative to the pulse start, snapped like its grid.

        Times off the pulse's local boundaries snap to the pulse step, the RF
        raster, or ``raster`` (the other event's raster), in that order.
        """
        offset = time_s - event.start_s
        nearest = local[np.argmin(np.abs(local - offset))]
        if abs(offset - nearest) <= tolerance:
            return float(nearest)
        steps = (cls._rf_step(event, simulation_timestep_s), event.raster_s, raster)
        for step in steps:
            snapped = np.round(offset / step) * step
            if abs(offset - snapped) <= tolerance:
                return float(snapped)
        return float(offset)

    @classmethod
    def _coalesce_boundaries(cls, values, duration: float) -> np.ndarray:
        boundaries = np.sort(np.asarray(values, dtype=float))
        tolerance = cls._time_tolerance(duration)
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

    @classmethod
    def _rf_frame_integrals(
        cls, event: RFEvent, local: np.ndarray, cache: dict
    ) -> np.ndarray:
        """Return one pulse's RF integral for each of its pulse-relative intervals.

        Pulses that differ only in their constant phase offset (RF spoiling,
        phase cycling) share one rasterization: ``cache`` keeps the integrals
        without that phase per pulse shape and local grid.
        """
        key = (
            event.samples_hz.tobytes(),
            event.samples_hz.dtype.str,
            event.raster_s,
            event.frequency_offset_hz,
            local.tobytes(),
        )
        unphased = cache.get(key)
        if unphased is None:
            unphased = np.zeros(local.size - 1, dtype=np.complex128)
            for index in range(local.size - 1):
                total = cls._rf_event_integral_unphased(
                    event, local[index], local[index + 1], event_start_s=0.0
                )
                if total is not None:
                    unphased[index] = total
            cache[key] = unphased
        if event.phase_offset_rad != 0.0:
            return cls._rotate_phase(unphased, event.phase_offset_rad)
        return unphased

    @staticmethod
    def _rotate_phase(values, phase_rad: float):
        """Multiply ``values`` by ``exp(1j * phase_rad)`` the same way for any shape.

        NumPy's complex multiply takes SIMD loops for arrays and a scalar path
        for single values, which may round differently on some CPUs (seen on
        x86-64 Linux). Rotating with real multiplies and adds keeps a pulse's
        frame integrals bit-identical to its per-interval integrals everywhere.
        """
        values = np.asarray(values, dtype=np.complex128)
        cos = np.cos(phase_rad)
        sin = np.sin(phase_rad)
        real = values.real
        imag = values.imag
        rotated = np.empty_like(values)
        rotated.real = real * cos - imag * sin
        rotated.imag = real * sin + imag * cos
        return rotated

    @classmethod
    def _rf_event_integral(
        cls,
        event: RFEvent,
        start: float,
        end: float,
        event_start_s: float | None = None,
    ):
        """Integrate one RF event over [start, end].

        ``event_start_s`` places the event on another time axis, e.g. 0.0 for
        pulse-relative ``start`` and ``end``.
        """
        total = cls._rf_event_integral_unphased(event, start, end, event_start_s)
        if total is None:
            return 0j
        # The constant phase is applied once, so pulses that differ only in
        # their phase offset (e.g. RF spoiling) differ by one exact rotation
        # instead of by the rounding of large phase arguments.
        if event.phase_offset_rad != 0.0:
            total = cls._rotate_phase(total, event.phase_offset_rad)[()]
        return total

    @staticmethod
    def _rf_event_integral_unphased(
        event: RFEvent, start: float, end: float, event_start_s: float | None = None
    ):
        """Integrate one RF event over [start, end] without its phase offset.

        Returns ``None`` when the event does not overlap the interval.
        """
        event_start = event.start_s if event_start_s is None else event_start_s
        overlap_start = max(start, event_start)
        overlap_end = min(end, event_start + event.samples_hz.size * event.raster_s)
        if overlap_start >= overlap_end:
            return None
        first = max(0, int(np.floor((overlap_start - event_start) / event.raster_s)))
        last = min(
            event.samples_hz.size - 1,
            int(np.ceil((overlap_end - event_start) / event.raster_s)) - 1,
        )
        total = 0j
        for index in range(first, last + 1):
            cell_start = event_start + index * event.raster_s
            cell_end = cell_start + event.raster_s
            width = max(
                0.0, min(overlap_end, cell_end) - max(overlap_start, cell_start)
            )
            phase = 2 * np.pi * event.frequency_offset_hz * (cell_start - event_start)
            total += event.samples_hz[index] * np.exp(1j * phase) * width
        return total

    @staticmethod
    def _overlapping_interval_ranges(
        interval_end: np.ndarray, interval_start: np.ndarray, events
    ) -> List[Tuple[int, int]]:
        """Return the half-open interval index range overlapping each event.

        One search for all events keeps the loop over events in Python: a
        search per event releases and retakes the GIL each time, which
        starves the GUI thread while a worker compiles.
        """
        event_starts = np.fromiter(
            (event.start_s for event in events), dtype=float, count=len(events)
        )
        event_ends = np.fromiter(
            (event.end_s for event in events), dtype=float, count=len(events)
        )
        first = np.searchsorted(interval_end, event_starts, side="right")
        last = np.searchsorted(interval_start, event_ends, side="left")
        return list(zip(first.tolist(), last.tolist()))

    @classmethod
    def _gradient_integrals(
        cls,
        events,
        ranges,
        starts,
        interval_end,
        frame_of_interval,
        local_start,
        local_end,
        frames,
        rf_events,
        simulation_timestep_s,
        tolerance,
    ) -> np.ndarray:
        """Integrate every gradient event over the intervals it overlaps.

        All (event, interval) pairs are handled together in NumPy. Each sum
        adds the raster cells in the same order as a scalar loop over cells
        and the pairs reach each interval in event order, so the result is
        bit-identical to rasterizing one interval at a time.
        """
        integrals = np.zeros((starts.size, 3), dtype=float)
        if not events:
            return integrals
        firsts = np.fromiter((first for first, _ in ranges), dtype=np.int64)
        lasts = np.fromiter((last for _, last in ranges), dtype=np.int64)
        lengths = np.maximum(lasts - firsts, 0)
        pair_count = int(lengths.sum())
        if pair_count == 0:
            return integrals
        pair_event = np.repeat(np.arange(len(events), dtype=np.int64), lengths)
        pair_interval = (
            np.arange(pair_count, dtype=np.int64)
            - np.repeat(np.cumsum(lengths) - lengths, lengths)
            + np.repeat(firsts, lengths)
        )
        event_start = np.fromiter((event.start_s for event in events), dtype=float)
        raster = np.fromiter((event.raster_s for event in events), dtype=float)
        sample_count = np.fromiter(
            (event.samples_hz_per_m.size for event in events), dtype=np.int64
        )
        sample_offset = np.cumsum(sample_count) - sample_count
        samples = np.concatenate([event.samples_hz_per_m for event in events])
        axis = np.fromiter(("xyz".index(event.axis) for event in events), np.int64)

        pair_event_start = event_start[pair_event]
        pair_start = starts[pair_interval]
        pair_stop = interval_end[pair_interval]
        # Intervals inside an RF pulse frame use pulse-relative times.
        frame_ids = frame_of_interval[pair_interval]
        in_frame = np.flatnonzero(frame_ids >= 0)
        if in_frame.size:
            pair_start[in_frame] = local_start[pair_interval[in_frame]]
            pair_stop[in_frame] = local_end[pair_interval[in_frame]]
            keys, inverse = np.unique(
                np.column_stack((pair_event[in_frame], frame_ids[in_frame])),
                axis=0,
                return_inverse=True,
            )
            offsets = np.fromiter(
                (
                    cls._pulse_relative_offset(
                        rf_events[frame_index],
                        frames[frame_index][2],
                        events[event_index].start_s,
                        simulation_timestep_s,
                        events[event_index].raster_s,
                        tolerance,
                    )
                    for event_index, frame_index in keys.tolist()
                ),
                dtype=float,
                count=keys.shape[0],
            )
            pair_event_start[in_frame] = offsets[inverse.reshape(-1)]

        totals = cls._piecewise_integrals(
            pair_event_start,
            raster[pair_event],
            samples,
            sample_offset[pair_event],
            sample_count[pair_event],
            pair_start,
            pair_stop,
        )
        np.add.at(integrals, (pair_interval, axis[pair_event]), totals)
        return integrals

    @staticmethod
    def _piecewise_integrals(
        event_start: np.ndarray,
        raster: np.ndarray,
        samples: np.ndarray,
        sample_offset: np.ndarray,
        sample_count: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> np.ndarray:
        """Integrate piecewise-constant events over [start, end], one per row.

        Row ``i`` integrates the event whose samples are
        ``samples[sample_offset[i]:sample_offset[i] + sample_count[i]]``.
        Rows with the same number of raster cells share one sequential
        ``cumsum``, so every row adds its cells left to right like a scalar
        loop would.
        """
        event_end = event_start + sample_count * raster
        overlap_start = np.maximum(start, event_start)
        overlap_end = np.minimum(end, event_end)
        totals = np.zeros(start.shape, dtype=float)
        valid = overlap_start < overlap_end
        if not np.any(valid):
            return totals
        with np.errstate(invalid="ignore"):
            first = np.maximum(
                0, np.floor((overlap_start - event_start) / raster).astype(np.int64)
            )
            last = np.minimum(
                sample_count - 1,
                np.ceil((overlap_end - event_start) / raster).astype(np.int64) - 1,
            )
        cells = np.where(valid, last - first + 1, 0)
        rows_with_cells = np.flatnonzero(cells > 0)
        row_cells = cells[rows_with_cells]
        for count in np.unique(row_cells).tolist():
            rows = rows_with_cells[row_cells == count]
            index = first[rows, None] + np.arange(count, dtype=np.int64)
            row_raster = raster[rows, None]
            cell_start = event_start[rows, None] + index * row_raster
            cell_end = cell_start + row_raster
            width = np.maximum(
                0.0,
                np.minimum(overlap_end[rows, None], cell_end)
                - np.maximum(overlap_start[rows, None], cell_start),
            )
            terms = samples[sample_offset[rows, None] + index] * width
            totals[rows] = np.cumsum(terms, axis=1)[:, -1]
        return totals

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
                # The simulator reports transverse magnetization as Mx+iMy.
                # With its Bloch sign convention a positive off-resonance
                # evolves with negative complex phase, so a positive receiver
                # frequency/phase is removed by multiplication with +i*phase.
                # Using the opposite sign doubles the phase ramp and turns RF
                # phase cycling or receiver offsets into Cartesian shifts.
                samples.append((float(time_value), order, np.exp(1j * receiver_phase)))
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
