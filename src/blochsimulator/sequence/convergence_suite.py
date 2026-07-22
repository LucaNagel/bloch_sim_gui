"""Representative sequence-class cases for time-step convergence studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .convergence import (
    DEFAULT_TIMESTEPS_S,
    ConvergenceCriteria,
    SpinProbeEnsemble,
    TimestepConvergenceResult,
    run_timestep_convergence,
)
from .model import GradientEvent, RFEvent, SequenceProgram


@dataclass(frozen=True)
class SequenceConvergenceCase:
    """One representative sequence motif and its deliberately difficult probes."""

    name: str
    family: str
    description: str
    program: SequenceProgram
    probes: SpinProbeEnsemble

    def __post_init__(self) -> None:
        for field_name in ("name", "family", "description"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, value)
        if not isinstance(self.program, SequenceProgram):
            raise TypeError("program must be SequenceProgram")
        if not isinstance(self.probes, SpinProbeEnsemble):
            raise TypeError("probes must be SpinProbeEnsemble")


@dataclass(frozen=True)
class SequenceConvergenceCaseResult:
    """Convergence output associated with its sequence-class definition."""

    case: SequenceConvergenceCase
    convergence: TimestepConvergenceResult


@dataclass(frozen=True)
class SequenceConvergenceSuiteResult:
    """Combined convergence results for all representative sequence classes."""

    case_results: tuple[SequenceConvergenceCaseResult, ...]

    def __post_init__(self) -> None:
        values = tuple(self.case_results)
        if not values:
            raise ValueError("case_results must not be empty")
        object.__setattr__(self, "case_results", values)

    @property
    def coarsest_passing_timestep_s(self) -> float | None:
        passing = [
            record["simulation_timestep_s"]
            for record in self.summary_records()
            if record["all_passed"] and record["simulation_timestep_s"] is not None
        ]
        return max(passing, default=None)

    def summary_records(self) -> list[dict]:
        """Summarize the limiting sequence class for every candidate step."""

        point_count = len(self.case_results[0].convergence.points)
        if any(
            len(case_result.convergence.points) != point_count
            for case_result in self.case_results
        ):
            raise ValueError("all convergence cases must use the same time-step grid")

        records = []
        for point_index in range(point_count):
            points = [
                case_result.convergence.points[point_index]
                for case_result in self.case_results
            ]
            timesteps = {point.simulation_timestep_s for point in points}
            if len(timesteps) != 1:
                raise ValueError(
                    "all convergence cases must use the same ordered time-step grid"
                )
            limiting_index = int(
                np.argmax([point.max_vector_error for point in points])
            )
            limiting_case = self.case_results[limiting_index]
            limiting_point = points[limiting_index]
            timestep_s = limiting_point.simulation_timestep_s
            records.append(
                {
                    "timestep": (
                        "native" if timestep_s is None else f"{timestep_s * 1e6:g} us"
                    ),
                    "simulation_timestep_s": timestep_s,
                    "simulation_timestep_us": (
                        None if timestep_s is None else timestep_s * 1e6
                    ),
                    "all_passed": all(point.passed for point in points),
                    "limiting_case": limiting_case.case.name,
                    "limiting_family": limiting_case.case.family,
                    "max_vector_error": limiting_point.max_vector_error,
                    "largest_rms_vector_error": max(
                        point.rms_vector_error for point in points
                    ),
                    "total_interval_count": sum(
                        point.interval_count for point in points
                    ),
                    "total_runtime_s": sum(point.runtime_s for point in points),
                }
            )
        return records

    def to_records(self) -> list[dict]:
        """Flatten all per-case measurements for CSV/dataframe export."""

        records = []
        for case_result in self.case_results:
            for record in case_result.convergence.to_records():
                records.append(
                    {
                        "case": case_result.case.name,
                        "family": case_result.case.family,
                        "description": case_result.case.description,
                        "probe_count": case_result.case.probes.n_spins,
                        **record,
                    }
                )
        return records


def run_sequence_convergence_suite(
    cases: Iterable[SequenceConvergenceCase],
    *,
    timesteps_s=DEFAULT_TIMESTEPS_S,
    criteria: ConvergenceCriteria | None = None,
    max_rf_checkpoints: int = 64,
    simulator=None,
) -> SequenceConvergenceSuiteResult:
    """Run the same time-step grid over several representative sequence cases."""

    cases = tuple(cases)
    if not cases:
        raise ValueError("cases must not be empty")
    if any(not isinstance(case, SequenceConvergenceCase) for case in cases):
        raise TypeError("cases must contain SequenceConvergenceCase values")
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("sequence convergence case names must be unique")
    timesteps_s = tuple(timesteps_s)
    criteria = ConvergenceCriteria() if criteria is None else criteria

    if simulator is None:
        from ..simulator import BlochSimulator

        simulator = BlochSimulator(use_parallel=False)
    results = tuple(
        SequenceConvergenceCaseResult(
            case=case,
            convergence=run_timestep_convergence(
                case.program,
                case.probes,
                timesteps_s=timesteps_s,
                criteria=criteria,
                max_rf_checkpoints=max_rf_checkpoints,
                simulator=simulator,
            ),
        )
        for case in cases
    )
    return SequenceConvergenceSuiteResult(results)


def make_default_sequence_convergence_cases(
    *, rf_raster_s: float = 10e-6
) -> tuple[SequenceConvergenceCase, ...]:
    """Build fast motifs covering the major supported sequence classes."""

    raster = float(rf_raster_s)
    if not np.isfinite(raster) or raster <= 0:
        raise ValueError("rf_raster_s must be finite and positive")
    return (
        make_hard_pulse_case(raster),
        make_slice_selective_case(raster),
        make_spin_echo_case(raster),
        make_bssfp_case(raster),
        make_mprage_case(raster),
        make_spectral_selective_case(raster),
        make_adiabatic_inversion_case(raster),
    )


def make_hard_pulse_case(raster_s: float = 10e-6) -> SequenceConvergenceCase:
    """Non-selective block excitation used by FID and simple GRE."""

    duration = _aligned_duration(1e-3, raster_s)
    samples = _real_flip_pulse("block", 90.0, duration, raster_s)
    program = _program(
        (RFEvent(0.0, samples, raster_s),),
        duration,
        "hard_pulse_fid_gre",
    )
    probes = SpinProbeEnsemble.from_axes(
        [[0.0, 0.0, 0.0]],
        frequency_offsets_hz=[-500.0, -200.0, 0.0, 200.0, 500.0],
        b1_scales=[0.8, 1.0, 1.2],
        relaxation_times_s=[(1.0, 0.08), (2.0, 0.2)],
    )
    return SequenceConvergenceCase(
        name="hard_pulse_fid_gre",
        family="FID/GRE hard pulse",
        description="Non-selective 90 degree block pulse over B0 and B1 variation",
        program=program,
        probes=probes,
    )


def make_slice_selective_case(raster_s: float = 10e-6) -> SequenceConvergenceCase:
    """Shared slice-selective excitation motif for EPI, UTE, CSI and 2D scans."""

    duration = _aligned_duration(1.5e-3, raster_s)
    samples = _real_flip_pulse("sinc", 90.0, duration, raster_s)
    time_bandwidth = 4.0
    slice_thickness_m = 5e-3
    gradient_hz_per_m = time_bandwidth / duration / slice_thickness_m
    gradient = np.full(samples.size, gradient_hz_per_m, dtype=np.float64)
    program = _program(
        (
            RFEvent(0.0, samples, raster_s),
            GradientEvent("z", 0.0, gradient, raster_s),
        ),
        duration,
        "slice_selective_epi_ute_csi",
    )
    probes = SpinProbeEnsemble.from_axes(
        np.column_stack(
            (
                np.zeros(7),
                np.zeros(7),
                np.linspace(-1.5, 1.5, 7) * slice_thickness_m,
            )
        ),
        frequency_offsets_hz=[-250.0, 0.0, 250.0],
        b1_scales=[0.8, 1.0, 1.2],
        relaxation_times_s=[(1.0, 0.08)],
    )
    return SequenceConvergenceCase(
        name="slice_selective_epi_ute_csi",
        family="Slice-selective imaging",
        description="Sinc excitation with simultaneous slice gradient",
        program=program,
        probes=probes,
    )


def make_spin_echo_case(raster_s: float = 10e-6) -> SequenceConvergenceCase:
    """Shaped 90/180 degree pair with off-resonant refocusing."""

    excitation_duration = _aligned_duration(1e-3, raster_s)
    refocusing_duration = _aligned_duration(1.5e-3, raster_s)
    echo_time = _aligned_duration(20e-3, raster_s)
    excitation = _real_flip_pulse("gaussian", 90.0, excitation_duration, raster_s)
    refocusing = _real_flip_pulse("gaussian", 180.0, refocusing_duration, raster_s)
    refocusing_start = echo_time / 2.0 - refocusing_duration / 2.0
    program = _program(
        (
            RFEvent(0.0, excitation, raster_s),
            RFEvent(refocusing_start, refocusing, raster_s, phase_offset_rad=np.pi / 2),
        ),
        echo_time,
        "spin_echo_refocusing",
    )
    probes = SpinProbeEnsemble.from_axes(
        [[0.0, 0.0, 0.0]],
        frequency_offsets_hz=[-300.0, -100.0, 0.0, 100.0, 300.0],
        b1_scales=[0.8, 1.0, 1.2],
        relaxation_times_s=[(0.8, 0.05), (1.5, 0.12)],
    )
    return SequenceConvergenceCase(
        name="spin_echo_refocusing",
        family="Spin echo",
        description="Gaussian 90/180 degree pulse pair over B0, B1 and T2",
        program=program,
        probes=probes,
    )


def make_bssfp_case(raster_s: float = 10e-6) -> SequenceConvergenceCase:
    """Alternating-phase repeated RF train with bSSFP off-resonance sensitivity."""

    repetitions = 24
    tr_s = _aligned_duration(4e-3, raster_s)
    pulse_duration = _aligned_duration(0.3e-3, raster_s)
    pulse = _real_flip_pulse("gaussian", 30.0, pulse_duration, raster_s)
    events = tuple(
        RFEvent(
            repetition * tr_s,
            pulse,
            raster_s,
            phase_offset_rad=np.pi * (repetition % 2),
        )
        for repetition in range(repetitions)
    )
    program = _program(events, repetitions * tr_s, "bssfp_repeated_train")
    probes = SpinProbeEnsemble.from_axes(
        [[0.0, 0.0, 0.0]],
        frequency_offsets_hz=np.linspace(-250.0, 250.0, 9),
        b1_scales=[0.8, 1.0, 1.2],
        relaxation_times_s=[(1.0, 0.06), (2.0, 0.2)],
    )
    return SequenceConvergenceCase(
        name="bssfp_repeated_train",
        family="bSSFP",
        description="24 alternating-phase shaped pulses across one bSSFP band period",
        program=program,
        probes=probes,
    )


def make_mprage_case(raster_s: float = 10e-6) -> SequenceConvergenceCase:
    """Inversion followed by a low-flip readout train with T1 accumulation."""

    inversion_duration = _aligned_duration(1e-3, raster_s)
    readout_duration = _aligned_duration(0.2e-3, raster_s)
    inversion_time = _aligned_duration(80e-3, raster_s)
    readout_tr = _aligned_duration(6e-3, raster_s)
    readout_count = 16
    inversion = _real_flip_pulse("gaussian", 180.0, inversion_duration, raster_s)
    readout = _real_flip_pulse("gaussian", 12.0, readout_duration, raster_s)
    events = [RFEvent(0.0, inversion, raster_s)]
    events.extend(
        RFEvent(inversion_time + index * readout_tr, readout, raster_s)
        for index in range(readout_count)
    )
    duration = inversion_time + readout_count * readout_tr
    program = _program(tuple(events), duration, "mprage_inversion_readout_train")
    probes = SpinProbeEnsemble.from_axes(
        [[0.0, 0.0, 0.0]],
        frequency_offsets_hz=[-150.0, 0.0, 150.0],
        b1_scales=[0.8, 1.0, 1.2],
        relaxation_times_s=[(0.6, 0.05), (1.2, 0.09), (2.0, 0.15)],
    )
    return SequenceConvergenceCase(
        name="mprage_inversion_readout_train",
        family="MPRAGE/inversion recovery",
        description="Gaussian inversion and 16 low-flip readout pulses",
        program=program,
        probes=probes,
    )


def make_spectral_selective_case(
    raster_s: float = 10e-6,
) -> SequenceConvergenceCase:
    """Complex phase-modulated RF waveform over a dense frequency axis."""

    duration = _aligned_duration(2.5e-3, raster_s)
    count = int(round(duration / raster_s))
    normalized_time = (np.arange(count, dtype=float) + 0.5) / count - 0.5
    envelope = np.exp(-0.5 * (normalized_time / 0.2) ** 2)
    phase = 0.9 * np.sin(4.0 * np.pi * normalized_time)
    base = envelope * np.exp(1j * phase)
    samples = _scale_complex_area(base, 35.0, raster_s)
    program = _program(
        (RFEvent(0.0, samples, raster_s),),
        duration,
        "spectral_selective_phase_modulated",
    )
    probes = SpinProbeEnsemble.from_axes(
        [[0.0, 0.0, 0.0]],
        frequency_offsets_hz=np.linspace(-1800.0, 1800.0, 13),
        b1_scales=[0.7, 1.0, 1.3],
        relaxation_times_s=[(1.0, 0.08), (2.0, 0.2)],
    )
    return SequenceConvergenceCase(
        name="spectral_selective_phase_modulated",
        family="Spectral-selective RF",
        description="Complex Gaussian pulse with rapid phase modulation",
        program=program,
        probes=probes,
    )


def make_adiabatic_inversion_case(
    raster_s: float = 10e-6,
) -> SequenceConvergenceCase:
    """Hyperbolic-secant-like amplitude and frequency sweep."""

    duration = _aligned_duration(4e-3, raster_s)
    count = int(round(duration / raster_s))
    normalized_time = (np.arange(count, dtype=float) + 0.5) / count * 2.0 - 1.0
    amplitude = 900.0 / np.cosh(4.0 * normalized_time)
    instantaneous_frequency = 1800.0 * np.tanh(4.0 * normalized_time)
    phase = 2.0 * np.pi * np.cumsum(instantaneous_frequency) * raster_s
    phase -= phase[count // 2]
    samples = amplitude * np.exp(1j * phase)
    program = _program(
        (RFEvent(0.0, samples, raster_s),),
        duration,
        "adiabatic_frequency_sweep",
    )
    probes = SpinProbeEnsemble.from_axes(
        [[0.0, 0.0, 0.0]],
        frequency_offsets_hz=np.linspace(-1800.0, 1800.0, 9),
        b1_scales=[0.6, 1.0, 1.4],
        relaxation_times_s=[(1.0, 0.08), (2.0, 0.2)],
    )
    return SequenceConvergenceCase(
        name="adiabatic_frequency_sweep",
        family="Adiabatic inversion",
        description="Sech-like amplitude with tanh frequency sweep",
        program=program,
        probes=probes,
    )


def _program(events, duration_s: float, case_name: str) -> SequenceProgram:
    return SequenceProgram(
        events=tuple(events),
        duration_s=float(duration_s),
        source="time-step convergence suite",
        metadata={"convergence_case": case_name},
    )


def _aligned_duration(duration_s: float, raster_s: float) -> float:
    count = max(1, int(round(float(duration_s) / float(raster_s))))
    return count * float(raster_s)


def _real_flip_pulse(
    shape: str, flip_angle_deg: float, duration_s: float, raster_s: float
) -> np.ndarray:
    count = int(round(duration_s / raster_s))
    normalized_time = (np.arange(count, dtype=float) + 0.5) / count - 0.5
    if shape == "block":
        waveform = np.ones(count, dtype=np.float64)
    elif shape == "gaussian":
        waveform = np.exp(-0.5 * (normalized_time / 0.18) ** 2)
    elif shape == "sinc":
        waveform = np.sinc(4.0 * normalized_time) * np.hamming(count)
    else:
        raise ValueError(f"unsupported pulse shape {shape!r}")
    area = float(np.sum(waveform) * raster_s)
    return waveform * (flip_angle_deg / 360.0) / area


def _scale_complex_area(
    waveform: np.ndarray, flip_angle_deg: float, raster_s: float
) -> np.ndarray:
    area = np.sum(waveform) * raster_s
    if abs(area) <= np.finfo(float).eps:
        raise ValueError("complex pulse has zero integrated area")
    return waveform * (flip_angle_deg / 360.0) / area
