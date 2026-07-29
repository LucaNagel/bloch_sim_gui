__version__ = "2.0.0"

from .simulator import (
    BlochSimulator,
    TissueParameters,
    PulseSequence,
    SpinEcho,
    SpinEchoTipAxis,
    InversionRecovery,
    GradientEcho,
    SliceSelectRephase,
    CustomPulse,
    design_rf_pulse,
    apply_rf_carrier,
)  # noqa: F401

try:
    from . import notebook_exporter
except ImportError:
    pass

# visualization is available but not imported by default to avoid PyQt5 dependencies
# from . import visualization
from . import kspace  # noqa: F401
from . import phantom  # noqa: F401
from .phantom_design import (  # noqa: F401
    PhantomDesign,
    ShapeDefinition,
    SpectralPeakDefinition,
)
from .spectral_phantom import ChemicalSpecies, SpectralPhantom  # noqa: F401
from .dynamic_phantom import (  # noqa: F401
    DynamicB0,
    DynamicSpectralPhantom,
    KineticRegionDefinition,
    PyruvateInflow,
    TimeCurve,
    rasterize_kpl_regions,
    simulate_two_pool_kinetics,
)
from . import pulse_loader  # noqa: F401
from .sequence import (  # noqa: F401
    ADCEvent,
    AcquisitionDimensions,
    CartesianAcquisition,
    CartesianAcquisitionFrames,
    CartesianAcquisitionVolumes,
    CompiledSequence,
    GradientEvent,
    RFEvent,
    SequenceCompiler,
    SequenceProgram,
    SequenceSimulationResult,
    UnsupportedPulseqVersionError,
    infer_cartesian_acquisition,
    infer_cartesian_acquisition_frames,
    infer_cartesian_acquisition_volumes,
    load_pulseq,
    make_cartesian_epi,
)
