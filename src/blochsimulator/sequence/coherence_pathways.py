"""Weighted coherence pathways of an arbitrary sequence program.

Gradient-waveform spoiling replaces a uniform voxel by a few spins. Whether
that is accurate depends on the gradient moments that the *relevant*
coherence pathways have accumulated when the ADC samples them. This module
tracks those pathways with an extended phase graph (EPG) that uses continuous
3D moments instead of integer dephasing orders:

* a transverse state has a complex amplitude and a reference moment ``r``;
  at time ``t`` its dephasing across the voxel is ``m(t) - r``, where ``m``
  is the cumulative gradient moment of the program;
* an RF pulse with flip angle ``a`` and phase ``p`` at moment ``m_p`` keeps
  ``cos^2(a/2)`` of every transverse state, mirrors ``sin^2(a/2)`` of it to
  ``2 m_p - r`` (a refocused echo), stores ``sin(a)/2`` as longitudinal
  modulation, and turns longitudinal states back into transverse ones;
* transverse states decay with ``T2`` and modulated longitudinal states with
  ``T1``.

States whose moments agree within a tolerance are merged with their complex
amplitudes, so RF spoiling and echo interference are captured. Off-resonance
and diffusion are ignored, which keeps old echoes at least as strong as in
the real object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import SequenceProgram


@dataclass(frozen=True)
class CoherencePathwayTrain:
    """Dephasing and relative amplitude of pathways seen by the ADCs.

    ``phase_cycles_per_voxel`` holds one row per observation: the X/Y/Z
    dephasing of a pathway across one phantom voxel at an ADC sample.
    ``weights`` is the pathway amplitude relative to the strongest pathway at
    the same ADC. The strongest pathway itself is the reference signal and is
    not listed, because its readout dephasing is ordinary k-space encoding.
    """

    phase_cycles_per_voxel: np.ndarray
    weights: np.ndarray
    adc_indices: np.ndarray
    evaluated_adc_count: int
    total_adc_count: int
    rf_pulse_count: int
    refocusing_pulse_count: int
    maximum_state_count: int
    states_were_capped: bool

    @property
    def n_observations(self) -> int:
        return int(self.weights.size)

    @property
    def maximum_weight(self) -> float:
        return float(np.max(self.weights, initial=0.0))


def coherence_pathways_from_program(
    program: SequenceProgram,
    voxel_basis_m,
    *,
    t1_s: float = np.inf,
    t2_s: float = np.inf,
    longitudinal_equilibrium: float = 1.0,
    weight_threshold: float = 0.01,
    maximum_adcs: int = 1024,
    maximum_states: int = 4096,
    merge_tolerance_cycles: float = 1e-3,
    prune_amplitude: float = 1e-4,
) -> CoherencePathwayTrain:
    """Track weighted coherence pathways through ``program``.

    ``voxel_basis_m`` holds the three physical voxel edge vectors as columns,
    as returned by :func:`phantom_voxel_basis_m`. ``t1_s`` and ``t2_s`` should
    be the longest relaxation times in the object, which keeps old echoes
    alive longest. ``longitudinal_equilibrium`` is the unmodulated
    longitudinal magnetization that ``T1`` recovers towards, relative to the
    initial value: 1 for thermal polarization, 0 for hyperpolarized samples.

    Only observations with a weight above ``weight_threshold`` are returned,
    since weaker pathways cannot exceed a sampling-error target of that size.
    """
    if not isinstance(program, SequenceProgram):
        raise TypeError("program must be a SequenceProgram")
    basis = np.asarray(voxel_basis_m, dtype=np.float64)
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
        raise ValueError("voxel_basis_m must be a finite 3x3 matrix")
    t1_s = float(t1_s)
    t2_s = float(t2_s)
    if not (t1_s > 0.0 and t2_s > 0.0):
        raise ValueError("relaxation times must be positive")
    if not np.isfinite(longitudinal_equilibrium) or longitudinal_equilibrium < 0:
        raise ValueError("longitudinal_equilibrium must be non-negative")
    if not np.isfinite(weight_threshold) or weight_threshold < 0.0:
        raise ValueError("weight_threshold must be non-negative")
    if merge_tolerance_cycles <= 0.0:
        raise ValueError("merge_tolerance_cycles must be positive")

    moment = _MomentFunction(program, basis)
    pulses = sorted(
        (
            pulse
            for pulse in (_rf_center_flip_phase(event) for event in program.rf_events)
            if pulse[1] > 1e-9
        ),
        key=lambda item: item[0],
    )
    adcs = sorted(program.adc_events, key=lambda event: event.start_s)
    total_adc_count = len(adcs)
    adcs = adcs[: max(0, int(maximum_adcs))]

    # Events in time order; a pulse acts before an ADC that starts with it.
    # Pulses after the last evaluated ADC cannot change any observation.
    last_adc_s = adcs[-1].start_s if adcs else -np.inf
    timeline = [
        (time_s, 0, (flip, phase))
        for time_s, flip, phase in pulses
        if time_s <= last_adc_s
    ]
    timeline += [(event.start_s, 1, index) for index, event in enumerate(adcs)]
    timeline.sort(key=lambda item: (item[0], item[1]))

    f_ref = np.zeros((0, 3), dtype=np.float64)
    f_amp = np.zeros(0, dtype=np.complex128)
    z_k = np.zeros((0, 3), dtype=np.float64)
    z_amp = np.zeros(0, dtype=np.complex128)
    z0 = 1.0
    current_s = 0.0
    maximum_state_count = 0
    states_were_capped = False
    refocusing_pulse_count = 0
    phase_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    adc_rows: list[np.ndarray] = []

    for time_s, kind, payload in timeline:
        delta_s = max(0.0, float(time_s) - current_s)
        if delta_s > 0.0:
            f_amp = f_amp * np.exp(-delta_s / t2_s)
            longitudinal_decay = np.exp(-delta_s / t1_s)
            z_amp = z_amp * longitudinal_decay
            z0 = longitudinal_equilibrium + (z0 - longitudinal_equilibrium) * (
                longitudinal_decay
            )
            current_s = float(time_s)
        if kind == 0:
            flip, phase = payload
            if flip > 0.75 * np.pi:
                refocusing_pulse_count += 1
            pulse_moment = moment(np.array([time_s]))[0]
            f_ref, f_amp, z_k, z_amp, z0 = _apply_pulse(
                f_ref, f_amp, z_k, z_amp, z0, flip, phase, pulse_moment
            )
            f_ref, f_amp = _merge(f_ref, f_amp, merge_tolerance_cycles)
            z_k, z_amp = _merge_longitudinal(z_k, z_amp, merge_tolerance_cycles)
            # b E(k) + conj(b) E(-k) at k = 0 is unmodulated magnetization.
            unmodulated = np.all(np.abs(z_k) < 0.5 * merge_tolerance_cycles, axis=1)
            if np.any(unmodulated):
                z0 += float(2.0 * np.sum(z_amp[unmodulated].real))
                z_k, z_amp = z_k[~unmodulated], z_amp[~unmodulated]
            f_ref, f_amp, capped_f = _prune(
                f_ref, f_amp, prune_amplitude, maximum_states
            )
            z_k, z_amp, capped_z = _prune(z_k, z_amp, prune_amplitude, maximum_states)
            states_were_capped |= capped_f or capped_z
            maximum_state_count = max(maximum_state_count, f_amp.size + z_amp.size)
            continue

        event = adcs[int(payload)]
        if f_amp.size == 0:
            continue
        magnitudes = np.abs(f_amp)
        strongest = int(np.argmax(magnitudes))
        if magnitudes[strongest] <= 0.0:
            continue
        weights = magnitudes / magnitudes[strongest]
        separated = np.any(
            np.abs(f_ref - f_ref[strongest]) > merge_tolerance_cycles, axis=1
        )
        selected = separated & (weights > weight_threshold)
        if not np.any(selected):
            continue
        sample_times = np.unique(
            [event.start_s, 0.5 * (event.start_s + event.end_s), event.end_s]
        )
        sample_moments = moment(sample_times)
        phases = sample_moments[:, None, :] - f_ref[None, selected, :]
        phase_rows.append(phases.reshape(-1, 3))
        weight_rows.append(np.tile(weights[selected], sample_times.size))
        adc_rows.append(np.full(phases.shape[0] * phases.shape[1], int(payload)))

    if phase_rows:
        phases = np.concatenate(phase_rows)
        weights = np.concatenate(weight_rows)
        adc_indices = np.concatenate(adc_rows)
    else:
        phases = np.zeros((0, 3), dtype=np.float64)
        weights = np.zeros(0, dtype=np.float64)
        adc_indices = np.zeros(0, dtype=np.int64)
    return CoherencePathwayTrain(
        phase_cycles_per_voxel=phases,
        weights=weights,
        adc_indices=adc_indices,
        evaluated_adc_count=len(adcs),
        total_adc_count=total_adc_count,
        rf_pulse_count=len(pulses),
        refocusing_pulse_count=refocusing_pulse_count,
        maximum_state_count=maximum_state_count,
        states_were_capped=bool(states_were_capped),
    )


def deduplicate_observations(
    phase_cycles_per_voxel, weights, tolerance_cycles: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Keep each distinct dephasing once, with its largest weight."""
    phases = np.asarray(phase_cycles_per_voxel, dtype=np.float64).reshape(-1, 3)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if phases.shape[0] != weights.size:
        raise ValueError("phases and weights must have the same length")
    if weights.size == 0:
        return phases, weights
    keys = np.round(phases / tolerance_cycles).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    maximum = np.zeros(unique_keys.shape[0], dtype=np.float64)
    np.maximum.at(maximum, inverse, weights)
    first = np.full(unique_keys.shape[0], weights.size, dtype=np.int64)
    np.minimum.at(first, inverse, np.arange(weights.size))
    return phases[first], maximum


class _MomentFunction:
    """Cumulative gradient moment in cycles per voxel, piecewise linear."""

    def __init__(self, program: SequenceProgram, basis: np.ndarray) -> None:
        self._knots = []
        for axis in "xyz":
            events = sorted(
                (event for event in program.gradient_events if event.axis == axis),
                key=lambda event: event.start_s,
            )
            times = [0.0]
            values = [0.0]
            for event in events:
                samples = np.asarray(event.samples_hz_per_m, dtype=np.float64)
                start = float(event.start_s)
                knot_times = start + event.raster_s * np.arange(samples.size + 1)
                knot_values = values[-1] + np.concatenate(
                    ([0.0], np.cumsum(samples * event.raster_s))
                )
                times.extend(knot_times.tolist())
                values.extend(knot_values.tolist())
            self._knots.append((np.asarray(times), np.asarray(values)))
        self._basis = basis

    def __call__(self, times_s: np.ndarray) -> np.ndarray:
        times_s = np.asarray(times_s, dtype=np.float64)
        moments = np.column_stack(
            [np.interp(times_s, knots, values) for knots, values in self._knots]
        )
        return moments @ self._basis


def _rf_center_flip_phase(event) -> tuple[float, float, float]:
    """Return the magnitude-weighted centre, flip angle and phase of a pulse."""
    samples = np.asarray(event.samples_hz, dtype=np.complex128)
    magnitude = np.abs(samples)
    total_magnitude = float(np.sum(magnitude))
    if total_magnitude <= 0.0:
        return float(event.start_s), 0.0, 0.0
    times = event.start_s + (np.arange(samples.size) + 0.5) * event.raster_s
    center_s = float(np.sum(magnitude * times) / total_magnitude)
    net = complex(np.sum(samples))
    # Strongly phase-modulated pulses, such as adiabatic ones, rotate by much
    # more than their net area. Their total area is the safer flip estimate.
    area = abs(net) if abs(net) >= 0.5 * total_magnitude else total_magnitude
    flip = float(np.clip(2.0 * np.pi * area * event.raster_s, 0.0, np.pi))
    phase = float(event.phase_offset_rad + np.angle(net))
    return center_s, flip, phase


def _apply_pulse(f_ref, f_amp, z_k, z_amp, z0, flip, phase, pulse_moment):
    """Rotate all states; see the module docstring for the transitions."""
    keep = np.cos(0.5 * flip) ** 2
    mirror = np.sin(0.5 * flip) ** 2
    sine = np.sin(flip)
    cosine = np.cos(flip)
    rotation = np.exp(1j * phase)
    m_p = pulse_moment[None, :]
    # Transverse output: M+' = keep M+ + e^{2ip} mirror M- - i e^{ip} sin Mz.
    new_ref = [f_ref, 2.0 * m_p - f_ref]
    new_amp = [keep * f_amp, rotation**2 * mirror * np.conj(f_amp)]
    excite = -1j * rotation * sine
    if z_amp.size:
        new_ref += [m_p + z_k, m_p - z_k]
        new_amp += [excite * z_amp, excite * np.conj(z_amp)]
    if z0 != 0.0:
        new_ref.append(m_p)
        new_amp.append(np.array([excite * z0]))
    # Longitudinal output keeps the k = r - m_p half of the real modulation.
    stored_k = f_ref - m_p
    stored_amp = -0.5j * np.conj(rotation) * sine * f_amp
    return (
        np.concatenate(new_ref),
        np.concatenate(new_amp),
        np.concatenate([z_k, stored_k]),
        np.concatenate([cosine * z_amp, stored_amp]),
        cosine * z0,
    )


def _merge(states, amplitudes, tolerance):
    if amplitudes.size == 0:
        return states, amplitudes
    keys = np.round(states / tolerance).astype(np.int64)
    unique_keys, first, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    merged = np.zeros(unique_keys.shape[0], dtype=np.complex128)
    np.add.at(merged, inverse.reshape(-1), amplitudes)
    return states[first], merged


def _merge_longitudinal(states, amplitudes, tolerance):
    """Merge Z(k) with Z(-k), which describe one real modulation."""
    if amplitudes.size == 0:
        return states, amplitudes
    keys = np.round(states / tolerance).astype(np.int64)
    first_nonzero = np.argmax(keys != 0, axis=1)
    flip = keys[np.arange(keys.shape[0]), first_nonzero] < 0
    states = np.where(flip[:, None], -states, states)
    amplitudes = np.where(flip, np.conj(amplitudes), amplitudes)
    return _merge(states, amplitudes, tolerance)


def _prune(states, amplitudes, minimum, maximum_count):
    magnitudes = np.abs(amplitudes)
    keep = magnitudes > minimum
    states, amplitudes, magnitudes = states[keep], amplitudes[keep], magnitudes[keep]
    if amplitudes.size <= maximum_count:
        return states, amplitudes, False
    order = np.argsort(magnitudes)[::-1][:maximum_count]
    return states[order], amplitudes[order], True
