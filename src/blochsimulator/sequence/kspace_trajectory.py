"""k-space trajectories calculated from the gradient waveforms of a sequence.

The gradient moment ``M(t) = ∫ G dt`` of a :class:`SequenceProgram` is a
property of the whole timeline. The k-space position of the transverse
magnetization is that moment referenced to the last excitation:

* an excitation pulse resets ``k`` to zero at its centre (isodelay),
* a refocusing pulse negates ``k`` at its centre and conjugates the object;
  later gradients keep adding to ``k`` (spin echo at ``k = 0``),
* inversion, saturation and preparation pulses leave ``k`` unchanged.

Gradients in Hz/m are nucleus specific in this package, so ``k`` in cycles/m
follows from the waveforms alone; the gyromagnetic ratio only converts to
physical gradient units. Optional :class:`GradientImperfections` model the
scanner's actual gradient (per-axis delay, amplitude scaling, and an
exponential eddy-current response). All lengths are in metres, frequencies
in Hz, and times in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .model import GradientEvent, RFEvent, SequenceProgram

AXES = ("x", "y", "z")
RF_ROLES = ("excitation", "refocusing", "inversion", "saturation", "preparation")
# Pulses at or above this flip angle without an explicit Pulseq ``use`` are
# refocusing pulses when an acquisition follows before the next excitation.
REFOCUSING_THRESHOLD_DEG = 135.0
# Pulses below this flip angle neither excite nor refocus (e.g. empty RF).
MINIMUM_FLIP_DEG = 1e-3
# Eddy currents are simulated until the slowest component decayed this far.
EDDY_TAIL_TIME_CONSTANTS = 6.0


@dataclass(frozen=True)
class GradientImperfections:
    """Deviation of the played gradient from the programmed waveform.

    ``delay_s`` delays the gradient of each axis relative to RF and ADC.
    ``scale`` multiplies each axis (gradient calibration). The eddy-current
    model subtracts ``a_i (h_i * dG/dt)`` with ``h_i(t) = exp(-t / tau_i)``
    from the gradient of every axis; it does not change a gradient's final
    moment once the response decayed, but it bends the trajectory during
    and shortly after gradient ramps.
    """

    delay_s: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    eddy_amplitudes: Tuple[float, ...] = ()
    eddy_time_constants_s: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        delay = tuple(float(value) for value in self.delay_s)
        scale = tuple(float(value) for value in self.scale)
        amplitudes = tuple(float(value) for value in self.eddy_amplitudes)
        constants = tuple(float(value) for value in self.eddy_time_constants_s)
        if len(delay) != 3 or not np.all(np.isfinite(delay)):
            raise ValueError("delay_s must contain three finite values")
        if len(scale) != 3 or not np.all(np.isfinite(scale)):
            raise ValueError("scale must contain three finite values")
        if len(amplitudes) != len(constants):
            raise ValueError("eddy amplitudes and time constants must pair up")
        if not np.all(np.isfinite(amplitudes)):
            raise ValueError("eddy amplitudes must be finite")
        if any(not np.isfinite(tau) or tau <= 0 for tau in constants):
            raise ValueError("eddy time constants must be positive and finite")
        object.__setattr__(self, "delay_s", delay)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "eddy_amplitudes", amplitudes)
        object.__setattr__(self, "eddy_time_constants_s", constants)

    @property
    def is_ideal(self) -> bool:
        return (
            not any(self.delay_s)
            and all(value == 1.0 for value in self.scale)
            and not any(self.eddy_amplitudes)
        )

    def to_dict(self) -> dict:
        return {
            "delay_s": list(self.delay_s),
            "scale": list(self.scale),
            "eddy_amplitudes": list(self.eddy_amplitudes),
            "eddy_time_constants_s": list(self.eddy_time_constants_s),
        }

    @classmethod
    def from_mapping(cls, values: Optional[Mapping]) -> "GradientImperfections":
        if values is None:
            return cls()
        if isinstance(values, cls):
            return values
        return cls(
            delay_s=tuple(values.get("delay_s", (0.0, 0.0, 0.0))),
            scale=tuple(values.get("scale", (1.0, 1.0, 1.0))),
            eddy_amplitudes=tuple(values.get("eddy_amplitudes", ())),
            eddy_time_constants_s=tuple(values.get("eddy_time_constants_s", ())),
        )


@dataclass(frozen=True)
class RFPulseRole:
    """Timing and k-space role of one RF event."""

    index: int
    start_s: float
    center_s: float
    end_s: float
    flip_angle_deg: float
    role: str
    frequency_offset_hz: float = 0.0


def _rf_center_s(event: RFEvent, timing: Optional[Mapping]) -> float:
    if timing is not None and "center_s" in timing:
        return float(timing["center_s"])
    magnitude = np.abs(event.samples_hz)
    peak = np.flatnonzero(magnitude >= magnitude.max() * (1.0 - 1e-9))
    # A flat top (hard pulse) has its centre in the middle of the plateau.
    index = 0.5 * (peak[0] + peak[-1]) if peak.size else 0.0
    return float(event.start_s + (index + 0.5) * event.raster_s)


def rf_flip_angle_deg(event: RFEvent) -> float:
    """Small-tip flip angle ``360° |∫ B1 dt|`` of a piecewise-constant pulse."""
    return float(abs(np.sum(event.samples_hz)) * event.raster_s * 360.0)


def classify_rf_events(
    program: SequenceProgram,
    *,
    refocusing_threshold_deg: float = REFOCUSING_THRESHOLD_DEG,
) -> Tuple[RFPulseRole, ...]:
    """Assign every RF event its k-space role.

    An explicit Pulseq ``use`` wins. Otherwise pulses below
    ``refocusing_threshold_deg`` excite; stronger pulses refocus when an
    excitation precedes them and an ADC follows before the next excitation,
    and invert otherwise.
    """
    timings = {
        round(float(item["start_s"]), 12): item
        for item in program.metadata.get("rf_timing", ())
        if isinstance(item, Mapping) and "start_s" in item
    }
    events = program.rf_events
    adc_starts = np.asarray([event.start_s for event in program.adc_events])
    provisional = []
    for index, event in enumerate(events):
        timing = timings.get(round(float(event.start_s), 12))
        flip = rf_flip_angle_deg(event)
        use = str((timing or {}).get("use", "undefined")).strip().lower()
        if use in RF_ROLES:
            role = use
        elif flip < MINIMUM_FLIP_DEG:
            role = "none"
        elif flip < float(refocusing_threshold_deg):
            role = "excitation"
        else:
            role = "strong"
        provisional.append(
            RFPulseRole(
                index=index,
                start_s=float(event.start_s),
                center_s=_rf_center_s(event, timing),
                end_s=float(event.end_s),
                flip_angle_deg=flip,
                role=role,
                frequency_offset_hz=float(event.frequency_offset_hz),
            )
        )
    excitation_centres = np.asarray(
        [item.center_s for item in provisional if item.role == "excitation"]
    )
    resolved = []
    for item in provisional:
        role = item.role
        if role == "strong":
            previous = excitation_centres[excitation_centres < item.center_s]
            following = excitation_centres[excitation_centres > item.center_s]
            horizon = following[0] if following.size else np.inf
            acquires = np.any((adc_starts > item.center_s) & (adc_starts < horizon))
            role = "refocusing" if previous.size and acquires else "inversion"
        resolved.append(
            RFPulseRole(
                index=item.index,
                start_s=item.start_s,
                center_s=item.center_s,
                end_s=item.end_s,
                flip_angle_deg=item.flip_angle_deg,
                role=role,
                frequency_offset_hz=item.frequency_offset_hz,
            )
        )
    return tuple(resolved)


@dataclass(frozen=True)
class _AxisMoment:
    """Exact cumulative moment of one axis at its raster edges."""

    edges_s: np.ndarray
    moment: np.ndarray
    raster_starts_s: np.ndarray
    raster_s: np.ndarray
    raster_values: np.ndarray

    def __call__(self, times_s) -> np.ndarray:
        if self.edges_s.size == 0:
            return np.zeros(np.shape(times_s), dtype=float)
        return np.interp(times_s, self.edges_s, self.moment, left=0.0)

    def gradient(self, times_s) -> np.ndarray:
        """Piecewise-constant gradient in Hz/m at ``times_s``."""
        times = np.asarray(times_s, dtype=float)
        values = np.zeros(times.shape, dtype=float)
        if self.raster_starts_s.size == 0:
            return values
        cell = np.searchsorted(self.raster_starts_s, times, side="right") - 1
        valid = cell >= 0
        cell = np.clip(cell, 0, self.raster_starts_s.size - 1)
        inside = valid & (times < self.raster_starts_s[cell] + self.raster_s[cell])
        values[inside] = self.raster_values[cell[inside]]
        return values


def _apply_eddy_currents(samples, raster_s, amplitudes, constants):
    """Subtract exponential eddy currents from a rastered gradient.

    The response to each gradient step decays exactly exponentially, so a
    first-order recursion with the step sizes as input is exact on the raster.
    Returns the corrected samples and the moment of the response tail that
    remains after the last sample (summed analytically).
    """
    steps = np.diff(samples, prepend=0.0)
    actual = np.asarray(samples, dtype=float).copy()
    remainder = 0.0
    from scipy.signal import lfilter

    for amplitude, tau in zip(amplitudes, constants):
        if amplitude == 0:
            continue
        decay = np.exp(-raster_s / tau)
        response = amplitude * lfilter([1.0], [1.0, -decay], steps)
        actual -= response
        if response.size:
            remainder -= response[-1] * decay / (1.0 - decay) * raster_s
    return actual, remainder


def _axis_moment(
    events: Sequence[GradientEvent],
    *,
    delay_s: float,
    scale: float,
    eddy_amplitudes: Sequence[float],
    eddy_time_constants_s: Sequence[float],
) -> _AxisMoment:
    events = sorted(events, key=lambda event: event.start_s)
    if not events:
        empty = np.zeros(0, dtype=float)
        return _AxisMoment(empty, empty, empty, empty, empty)
    blocks = []
    if eddy_amplitudes and any(eddy_amplitudes):
        tail = EDDY_TAIL_TIME_CONSTANTS * max(eddy_time_constants_s)
        # Rasterize clusters of events whose gaps are shorter than the eddy
        # tail onto one regular grid so their responses overlap correctly.
        cluster = [events[0]]
        clusters = [cluster]
        for event in events[1:]:
            previous = cluster[-1]
            same_raster = np.isclose(event.raster_s, previous.raster_s)
            if same_raster and event.start_s - previous.end_s < tail:
                cluster.append(event)
            else:
                cluster = [event]
                clusters.append(cluster)
        for members in clusters:
            raster = members[0].raster_s
            start = members[0].start_s
            stop = members[-1].end_s + tail
            count = int(np.ceil((stop - start) / raster - 1e-9))
            samples = np.zeros(count, dtype=float)
            for event in members:
                offset = int(round((event.start_s - start) / raster))
                samples[offset : offset + event.samples_hz_per_m.size] = (
                    event.samples_hz_per_m
                )
            samples, remainder = _apply_eddy_currents(
                samples, raster, eddy_amplitudes, eddy_time_constants_s
            )
            blocks.append((start, raster, samples, remainder))
    else:
        blocks = [
            (event.start_s, event.raster_s, np.asarray(event.samples_hz_per_m), 0.0)
            for event in events
        ]
    edges = []
    moments = []
    starts = []
    rasters = []
    values = []
    total = 0.0
    for start, raster, samples, remainder in blocks:
        samples = float(scale) * np.asarray(samples, dtype=float)
        block_edges = start + float(delay_s) + np.arange(samples.size + 1) * raster
        cumulative = total + np.concatenate(([0.0], np.cumsum(samples) * raster))
        if edges and block_edges[0] <= edges[-1][-1]:
            # Abutting events share their boundary; drop the duplicate edge.
            block_edges = block_edges[1:]
            cumulative = cumulative[1:]
        edges.append(block_edges)
        moments.append(cumulative)
        starts.append(start + float(delay_s) + np.arange(samples.size) * raster)
        rasters.append(np.full(samples.size, raster))
        values.append(samples)
        # The truncated eddy tail is added before the next block starts.
        total = float(cumulative[-1]) + float(scale) * remainder
    if total != moments[-1][-1]:
        edges.append(np.array([edges[-1][-1] + rasters[-1][-1]]))
        moments.append(np.array([total]))
    return _AxisMoment(
        np.concatenate(edges),
        np.concatenate(moments),
        np.concatenate(starts),
        np.concatenate(rasters),
        np.concatenate(values),
    )


@dataclass(frozen=True)
class KSpaceTrajectory:
    """k-space trajectory of a sequence, evaluable at any time.

    ``adc_k_cyc_per_m`` holds one scanner-xyz k vector per ADC sample in the
    chronological ADC order of the simulation result. Samples acquired before
    the first excitation have NaN k-space positions.
    ``adc_time_since_excitation_s`` is the time from the last excitation
    centre and drives off-resonance and T2* weighting of a sample.
    """

    rf_pulses: Tuple[RFPulseRole, ...]
    adc_times_s: np.ndarray
    adc_k_cyc_per_m: np.ndarray
    adc_time_since_excitation_s: np.ndarray
    adc_event_index: np.ndarray
    adc_event_starts: np.ndarray
    duration_s: float
    imperfections: GradientImperfections = field(default_factory=GradientImperfections)
    _axes: Tuple[_AxisMoment, ...] = field(default=(), repr=False, compare=False)
    _segment_starts_s: np.ndarray = field(
        default_factory=lambda: np.zeros(0), repr=False, compare=False
    )
    _segment_conjugated: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool), repr=False, compare=False
    )
    _segment_offsets: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3)), repr=False, compare=False
    )

    @property
    def num_adc_events(self) -> int:
        return int(self.adc_event_starts.size)

    @property
    def excitation_times_s(self) -> np.ndarray:
        return np.asarray(
            [pulse.center_s for pulse in self.rf_pulses if pulse.role == "excitation"]
        )

    def gradient_moment(self, times_s) -> np.ndarray:
        """Absolute gradient moment ``∫_0^t G dt`` in cycles/m, shape (..., 3)."""
        times = np.asarray(times_s, dtype=float)
        return np.stack([axis(times) for axis in self._axes], axis=-1)

    def gradient_hz_per_m(self, times_s) -> np.ndarray:
        """Played (imperfect) gradient in Hz/m at ``times_s``, shape (..., 3)."""
        times = np.asarray(times_s, dtype=float)
        return np.stack([axis.gradient(times) for axis in self._axes], axis=-1)

    def k_at(self, times_s) -> np.ndarray:
        """k-space position in cycles/m at arbitrary times, shape (..., 3)."""
        times = np.asarray(times_s, dtype=float)
        moment = self.gradient_moment(times)
        segment = np.searchsorted(self._segment_starts_s, times, side="right") - 1
        result = np.full(moment.shape, np.nan, dtype=float)
        valid = segment >= 0
        index = segment[valid]
        result[valid] = moment[valid] + self._segment_offsets[index]
        return result

    def conjugation_at(self, times_s) -> np.ndarray:
        """Whether the signal at ``times_s`` encodes the conjugated object.

        An odd number of refocusing pulses since the excitation conjugates
        the transverse magnetization, so ``s(t) = sum conj(m) e^{-2 pi i k x}``.
        Returns 1.0 (conjugated), 0.0 (not) or NaN before the first excitation.
        """
        times = np.asarray(times_s, dtype=float)
        segment = np.searchsorted(self._segment_starts_s, times, side="right") - 1
        conjugated = np.full(times.shape, np.nan)
        valid = segment >= 0
        conjugated[valid] = self._segment_conjugated[segment[valid]]
        return conjugated

    def adc_event_slice(self, event: int) -> slice:
        starts = self.adc_event_starts
        stop = starts[event + 1] if event + 1 < starts.size else self.adc_times_s.size
        return slice(int(starts[event]), int(stop))

    def event_k(self, event: int) -> np.ndarray:
        return self.adc_k_cyc_per_m[self.adc_event_slice(event)]

    def timeline(
        self,
        start_s: float = 0.0,
        stop_s: Optional[float] = None,
        *,
        max_points: int = 200_000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return times and k positions that resolve every gradient raster edge
        and RF centre in ``[start_s, stop_s]`` (decimated to ``max_points``)."""
        stop = self.duration_s if stop_s is None else float(stop_s)
        pieces = [np.asarray([start_s, stop])]
        for axis in self._axes:
            edges = axis.edges_s
            pieces.append(edges[(edges >= start_s) & (edges <= stop)])
        centres = self._segment_starts_s
        inside = centres[(centres >= start_s) & (centres <= stop)]
        # Evaluate just before and at each reset so the jump stays visible.
        pieces.append(inside)
        pieces.append(np.maximum(inside - 1e-12, start_s))
        times = np.unique(np.concatenate(pieces))
        if times.size > max_points:
            keep = np.unique(
                np.concatenate(
                    (
                        np.linspace(0, times.size - 1, max_points).astype(int),
                        np.searchsorted(times, inside),
                    )
                ).clip(0, times.size - 1)
            )
            times = times[keep]
        return times, self.k_at(times)

    def nominal_kmax_cyc_per_m(self, fov_m: Sequence[float], matrix: Sequence[int]):
        """Nyquist-limited maximum |k| per axis of a matrix/FOV pair."""
        return np.asarray(matrix, dtype=float) / (2.0 * np.asarray(fov_m, dtype=float))

    def event_summary(self, event: int) -> dict:
        """Readout diagnostics of one ADC event."""
        k = self.event_k(event)
        times = self.adc_time_since_excitation_s[self.adc_event_slice(event)]
        radius = np.linalg.norm(k, axis=1)
        centre = int(np.nanargmin(radius)) if np.isfinite(radius).any() else 0
        return {
            "samples": int(k.shape[0]),
            "k_start_cyc_per_m": k[0].tolist(),
            "k_end_cyc_per_m": k[-1].tolist(),
            "k_extent_cyc_per_m": np.nanmax(np.abs(k), axis=0).tolist(),
            "k_centre_sample": centre,
            "echo_time_s": float(times[centre]),
            "readout_start_s": float(times[0]),
            "readout_duration_s": float(times[-1] - times[0]),
        }


def calculate_kspace(
    program: SequenceProgram,
    *,
    imperfections: Optional[GradientImperfections] = None,
    refocusing_threshold_deg: float = REFOCUSING_THRESHOLD_DEG,
) -> KSpaceTrajectory:
    """Calculate the k-space trajectory of ``program``.

    The gradient moment is integrated exactly from the piecewise-constant
    waveforms and referenced to excitation and refocusing pulse centres.
    """
    imperfections = GradientImperfections.from_mapping(imperfections)
    gradients = program.gradient_events
    axes = tuple(
        _axis_moment(
            [event for event in gradients if event.axis == axis],
            delay_s=imperfections.delay_s[index],
            scale=imperfections.scale[index],
            eddy_amplitudes=imperfections.eddy_amplitudes,
            eddy_time_constants_s=imperfections.eddy_time_constants_s,
        )
        for index, axis in enumerate(AXES)
    )

    def moment(times):
        return np.stack([axis(np.asarray(times, dtype=float)) for axis in axes], -1)

    pulses = classify_rf_events(
        program, refocusing_threshold_deg=refocusing_threshold_deg
    )
    acting = sorted(
        (pulse for pulse in pulses if pulse.role in ("excitation", "refocusing")),
        key=lambda pulse: pulse.center_s,
    )
    starts, conjugations, offsets = [], [], []
    conjugated, offset, excited = False, np.zeros(3), False
    last_excitation = []
    for pulse in acting:
        at = moment(pulse.center_s)
        if pulse.role == "excitation":
            conjugated, offset, excited = False, -at, True
        elif excited:
            # k(t_r+) = -k(t_r-) with k = M + offset; gradients keep adding
            # to k afterwards, only the object is conjugated.
            conjugated, offset = not conjugated, -2.0 * at - offset
        else:
            continue
        starts.append(pulse.center_s)
        conjugations.append(conjugated)
        offsets.append(np.array(offset, dtype=float))
        last_excitation.append(
            pulse.center_s if pulse.role == "excitation" else last_excitation[-1]
        )
    segment_starts = np.asarray(starts, dtype=float)
    segment_conjugated = np.asarray(conjugations, dtype=bool)
    segment_offsets = np.asarray(offsets, dtype=float).reshape(-1, 3)
    excitation_of_segment = np.asarray(last_excitation, dtype=float)

    adc_events = program.adc_events
    if adc_events:
        adc_times = np.concatenate([event.sample_times_s for event in adc_events])
        counts = np.asarray([event.num_samples for event in adc_events])
        event_starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        event_index = np.repeat(np.arange(len(adc_events)), counts)
    else:
        adc_times = np.zeros(0, dtype=float)
        event_starts = np.zeros(0, dtype=int)
        event_index = np.zeros(0, dtype=int)

    trajectory = KSpaceTrajectory(
        rf_pulses=pulses,
        adc_times_s=adc_times,
        adc_k_cyc_per_m=np.zeros((adc_times.size, 3)),
        adc_time_since_excitation_s=np.zeros(adc_times.size),
        adc_event_index=event_index,
        adc_event_starts=event_starts.astype(int),
        duration_s=float(program.duration_s),
        imperfections=imperfections,
        _axes=axes,
        _segment_starts_s=segment_starts,
        _segment_conjugated=segment_conjugated,
        _segment_offsets=segment_offsets,
    )
    adc_k = trajectory.k_at(adc_times)
    segment = np.searchsorted(segment_starts, adc_times, side="right") - 1
    since = np.full(adc_times.size, np.nan)
    valid = segment >= 0
    since[valid] = adc_times[valid] - excitation_of_segment[segment[valid]]
    object.__setattr__(trajectory, "adc_k_cyc_per_m", adc_k)
    object.__setattr__(trajectory, "adc_time_since_excitation_s", since)
    return trajectory


def gradient_error(
    program: SequenceProgram,
    imperfections,
    *,
    refocusing_threshold_deg: float = REFOCUSING_THRESHOLD_DEG,
) -> np.ndarray:
    """Return ``k_actual - k_nominal`` per ADC sample, shape ``(adc, 3)``.

    The deviation is in cycles/m and scanner xyz, in the chronological ADC
    order of a simulation result; samples before the first excitation are
    zero. It includes the errors of prephasers and rephasers, i.e. of every
    gradient since the excitation.
    """
    actual = calculate_kspace(
        program,
        imperfections=imperfections,
        refocusing_threshold_deg=refocusing_threshold_deg,
    ).adc_k_cyc_per_m
    nominal = calculate_kspace(
        program, refocusing_threshold_deg=refocusing_threshold_deg
    ).adc_k_cyc_per_m
    return np.nan_to_num(actual - nominal, nan=0.0)


def classify_readout(k_cyc_per_m, *, tolerance: float = 0.02) -> str:
    """Name the shape of one readout's k-space path.

    Returns ``"cartesian_line"``, ``"radial"``, ``"spiral"``, ``"point"`` or
    ``"non_cartesian"``. Only the two axes with the largest extent are
    considered, so the classification is orientation independent.
    """
    k = np.asarray(k_cyc_per_m, dtype=float)
    k = k[np.all(np.isfinite(k), axis=1)]
    if k.shape[0] < 3:
        return "point"
    extent = np.ptp(k, axis=0)
    scale = float(np.max(np.abs(k))) if k.size else 0.0
    if scale <= 0 or float(extent.max()) <= tolerance * scale:
        return "point"
    order = np.argsort(extent)[::-1]
    a, b = k[:, order[0]], k[:, order[1]]
    if extent[order[1]] <= tolerance * extent[order[0]]:
        return "cartesian_line"
    centred = np.column_stack((a, b))
    direction = centred[-1] - centred[0]
    norm = np.linalg.norm(direction)
    if norm > 0:
        normal = np.array([-direction[1], direction[0]]) / norm
        distance = np.abs((centred - centred[0]) @ normal)
        if float(distance.max()) <= tolerance * norm:
            return "radial"
    angle = np.unwrap(np.arctan2(b, a))
    radius = np.hypot(a, b)
    start_near_centre = radius[0] <= 0.1 * radius.max()
    if abs(angle[-1] - angle[0]) >= 2 * np.pi and start_near_centre:
        return "spiral"
    if abs(angle[-1] - angle[0]) >= 2 * np.pi and radius[-1] <= 0.1 * radius.max():
        return "spiral"
    return "non_cartesian"
