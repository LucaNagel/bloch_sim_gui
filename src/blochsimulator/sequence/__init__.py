"""Event-based, spatially resolved MRI sequence simulation interfaces."""

from .acquisition import (
    AcquisitionDimensions,
    CartesianAcquisition,
    CartesianAcquisitionFrames,
    CartesianAcquisitionVolumes,
    SpectroscopicAcquisition,
    infer_cartesian_acquisition,
    infer_cartesian_acquisition_frames,
    infer_cartesian_acquisition_volumes,
    infer_spectroscopic_acquisition,
    make_cartesian_epi,
)
from .compiler import CompiledSequence, SequenceCompiler
from .convergence import (
    DEFAULT_TIMESTEPS_S,
    ConvergenceCriteria,
    SpinProbeEnsemble,
    TimestepConvergencePoint,
    TimestepConvergenceResult,
    default_probe_checkpoints,
    run_timestep_convergence,
)
from .convergence_suite import (
    SequenceConvergenceCase,
    SequenceConvergenceCaseResult,
    SequenceConvergenceSuiteResult,
    make_adiabatic_inversion_case,
    make_bssfp_case,
    make_default_sequence_convergence_cases,
    make_hard_pulse_case,
    make_mprage_case,
    make_slice_selective_case,
    make_spectral_selective_case,
    make_spin_echo_case,
    run_sequence_convergence_suite,
)
from .model import ADCEvent, GradientEvent, RFEvent, SequenceProgram
from .probe import SequenceProbeResult
from .reference import ReferenceSimulationResult, simulate_reference_sequence
from .result import SequenceSimulationResult
from .bruker_export import BrukerExportOptions, export_bruker_raw

from .pulseq import (
    PulseqImportError,
    UnsupportedPulseqVersionError,
    load_pulseq,
)

__all__ = [
    "ADCEvent",
    "AcquisitionDimensions",
    "CartesianAcquisition",
    "CartesianAcquisitionFrames",
    "CartesianAcquisitionVolumes",
    "SpectroscopicAcquisition",
    "CompiledSequence",
    "ConvergenceCriteria",
    "DEFAULT_TIMESTEPS_S",
    "GradientEvent",
    "RFEvent",
    "SequenceCompiler",
    "SequenceConvergenceCase",
    "SequenceConvergenceCaseResult",
    "SequenceConvergenceSuiteResult",
    "SequenceProbeResult",
    "SequenceProgram",
    "ReferenceSimulationResult",
    "SequenceSimulationResult",
    "SpinProbeEnsemble",
    "TimestepConvergencePoint",
    "TimestepConvergenceResult",
    "default_probe_checkpoints",
    "make_adiabatic_inversion_case",
    "make_bssfp_case",
    "make_default_sequence_convergence_cases",
    "make_hard_pulse_case",
    "make_mprage_case",
    "make_slice_selective_case",
    "make_spectral_selective_case",
    "make_spin_echo_case",
    "run_sequence_convergence_suite",
    "run_timestep_convergence",
    "simulate_reference_sequence",
    "BrukerExportOptions",
    "export_bruker_raw",
    "PulseqImportError",
    "UnsupportedPulseqVersionError",
    "infer_cartesian_acquisition",
    "infer_cartesian_acquisition_frames",
    "infer_cartesian_acquisition_volumes",
    "infer_spectroscopic_acquisition",
    "load_pulseq",
    "make_cartesian_epi",
]
