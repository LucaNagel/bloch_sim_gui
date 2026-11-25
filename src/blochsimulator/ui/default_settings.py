"""Persistent defaults shared by sequence and phantom creation UIs."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..simulation_defaults import DEFAULT_SIMULATION_TIMESTEP_S
from ..kernel_capabilities import GUI_KERNEL_NAMES, resolve_kernel_request
from ..units import NUCLEUS_GAMMA_HZ_PER_T

# Revision 2 made Native automatic the dynamic default. Older settings stored
# the previous default, Optimized NumPy, even when it was never chosen.
KERNEL_SETTINGS_REVISION = 2
DEFAULT_SEQUENCE_REFERENCE_PPM = {
    nucleus: 171.0 if nucleus == "C13" else 0.0 for nucleus in NUCLEUS_GAMMA_HZ_PER_T
}

# RF-active sequence time step presets (label, key, µs); see "Choosing the RF
# Time Step and Phantom Grid" in docs/DEVELOPER_GUIDE.md for the benchmarks.
SEQUENCE_TIMESTEP_PRESETS = (
    ("Accurate — 1 µs", "accurate", 1.0),
    ("Balanced — 5 µs", "balanced", 5.0),
    ("Fast — 10 µs", "fast", 10.0),
    ("Very fast — 20 µs", "very_fast", 20.0),
    ("Custom", "custom", None),
)
DEFAULT_SEQUENCE_TIMESTEP_PRESET = "very_fast"
DEFAULT_SEQUENCE_TIMESTEP_US = DEFAULT_SIMULATION_TIMESTEP_S * 1e6
SEQUENCE_TIMESTEP_TOOLTIP = (
    "Time step while RF is active. Larger steps average RF and simultaneous "
    "gradients over fewer intervals; ADC times, event boundaries and free "
    "precession stay exact.\n\n"
    "Benchmarked signal error against 1 µs (speed-up):\n"
    "• Block, sinc and CSI excitation: < 0.02% at 10 µs, < 0.1% at 20 µs "
    "(×3–10 faster at 20 µs)\n"
    "• Spectrally selective Gaussian (SS-bSSFP): 0.3% at 5 µs, 1% at 10 µs, "
    "4% at 20 µs (×11)\n"
    "• Spectral-spatial (SPSP): 1% at 2 µs, 5% at 5 µs, ≥ 10% from 10 µs — "
    "use 1 µs\n\n"
    "Otherwise the error grows roughly with the square of the step. After "
    "each run a few probe spins (dynamic phantoms: a few voxels in the "
    "dynamic kernel) repeat the sequence at a quarter of the step, adding about "
    "a tenth of the run time (half for dynamic phantoms); a largest "
    "magnetization difference above 1% is reported with the result. Pools "
    "far off the selected frequency (e.g. lactate in SS-bSSFP) can be much "
    "worse than the signal benchmarks above."
)
# Magnetization error above which a finished run notes that the RF time step
# is too coarse: a fraction of M0 for probe spins, and of each pool's peak
# magnetization in its voxel for dynamic phantoms.
TIMESTEP_ERROR_NOTE_THRESHOLD = 0.01
# The check gets about this share of the main run's wall time, but at least
# TIMESTEP_ERROR_MIN_INTERVALS quarter-step RF intervals; longer sequences are
# checked from their start. One quarter-step interval of the step and
# reference runs costs 1.2-3.1 us on an M3 (SS-bSSFP with 6 probes, FLASH with
# 54), plus about 0.01 s per check.
TIMESTEP_ERROR_RUNTIME_FRACTION = 0.1
TIMESTEP_ERROR_SECONDS_PER_INTERVAL = 3e-6
TIMESTEP_ERROR_MIN_INTERVALS = 1_000
# Dynamic phantoms are checked in the dynamic kernel itself, on at most 256
# voxels. One quarter-step interval of that costs about 3.7 us on an M3, plus
# about 0.02 s per check, almost independent of the voxel count.
# The error of hyperpolarized bSSFP builds up over the pulse train, so a check
# cut to a tenth of the run sees only its start (SS-bSSFP 8x8x4: 6-8x too
# low). Half the run brings the estimate within a factor of 2.
TIMESTEP_ERROR_DYNAMIC_SECONDS_PER_INTERVAL = 4e-6
TIMESTEP_ERROR_DYNAMIC_RUNTIME_FRACTION = 0.5


@dataclass(frozen=True)
class WorkspaceDefaults:
    """Values used when a new sequence workspace or phantom is created."""

    sequence_fov_mm: tuple[float, float, float] = (220.0, 220.0, 160.0)
    phantom_fov_mm: tuple[float, float, float] = (220.0, 220.0, 220.0)
    # Keep the historical attribute name for project/settings compatibility.
    # The value is now the concrete nucleus shared by every workspace.
    phantom_nucleus: str = "H1"
    field_strength_t: float = 3.0
    sequence_reference_ppm_by_nucleus: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SEQUENCE_REFERENCE_PPM)
    )

    def __post_init__(self):
        references = dict(DEFAULT_SEQUENCE_REFERENCE_PPM)
        references.update(self.sequence_reference_ppm_by_nucleus)
        object.__setattr__(self, "sequence_reference_ppm_by_nucleus", references)

    def sequence_reference_ppm(self, nucleus: str) -> float:
        """Return the configured absolute sequence reference for one nucleus."""
        return float(self.sequence_reference_ppm_by_nucleus[nucleus])

    @classmethod
    def from_settings(cls, settings) -> "WorkspaceDefaults":
        defaults = cls()
        if settings is None:
            return defaults

        def positive_value(key: str, fallback: float) -> float:
            try:
                value = float(settings.value(key, fallback))
            except (TypeError, ValueError):
                return fallback
            return value if np.isfinite(value) and value > 0.0 else fallback

        sequence_fov = tuple(
            positive_value(f"defaults/sequence_fov_{axis}_mm", fallback)
            for axis, fallback in zip("xyz", defaults.sequence_fov_mm)
        )
        phantom_fov = tuple(
            positive_value(f"defaults/phantom_fov_{axis}_mm", fallback)
            for axis, fallback in zip("xyz", defaults.phantom_fov_mm)
        )
        raw_nucleus = settings.value(
            "defaults/phantom_nucleus", defaults.phantom_nucleus
        )
        # Older settings used ``auto`` (H1 for static, C13 for dynamic). A
        # single shared workspace default must be concrete, so migrate it to
        # the backwards-compatible static default H1.
        nucleus = "H1" if raw_nucleus in (None, "", "auto") else str(raw_nucleus)
        if nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
            nucleus = defaults.phantom_nucleus
        field_strength_t = positive_value(
            "defaults/field_strength_t", defaults.field_strength_t
        )
        references = dict(defaults.sequence_reference_ppm_by_nucleus)
        for reference_nucleus, fallback in references.items():
            try:
                value = float(
                    settings.value(
                        f"defaults/sequence_reference_ppm/{reference_nucleus}", fallback
                    )
                )
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and -10000.0 <= value <= 10000.0:
                references[reference_nucleus] = value
        return cls(sequence_fov, phantom_fov, nucleus, field_strength_t, references)

    def save(self, settings) -> None:
        for axis, value in zip("xyz", self.sequence_fov_mm):
            settings.setValue(f"defaults/sequence_fov_{axis}_mm", float(value))
        for axis, value in zip("xyz", self.phantom_fov_mm):
            settings.setValue(f"defaults/phantom_fov_{axis}_mm", float(value))
        settings.setValue("defaults/phantom_nucleus", self.phantom_nucleus)
        settings.setValue("defaults/field_strength_t", float(self.field_strength_t))
        for nucleus in NUCLEUS_GAMMA_HZ_PER_T:
            settings.setValue(
                f"defaults/sequence_reference_ppm/{nucleus}",
                self.sequence_reference_ppm(nucleus),
            )


def load_kernel_settings(settings) -> tuple[str, str]:
    """Return the persisted sequence and dynamic kernel choices for the GUI."""
    if settings is None:
        return GUI_KERNEL_NAMES["sequence"][0], GUI_KERNEL_NAMES["dynamic"][0]
    sequence_kernel = str(
        settings.value("sequence/kernel", GUI_KERNEL_NAMES["sequence"][0])
    )
    dynamic_kernel = str(
        settings.value("sequence/dynamic_kernel", GUI_KERNEL_NAMES["dynamic"][0])
    )
    # Old names resolve to their public names. The dynamic NumPy path
    # ("optimized", now "numpy") is no longer offered in the GUI, so every
    # saved NumPy choice, of any settings revision, resolves to the default.
    return (
        resolve_kernel_request("sequence", sequence_kernel),
        resolve_kernel_request("dynamic", dynamic_kernel),
    )


def save_kernel_settings(settings, sequence_kernel: str, dynamic_kernel: str) -> None:
    """Persist explicit kernel choices together with the settings revision."""
    settings.setValue("sequence/kernel", str(sequence_kernel))
    settings.setValue("sequence/dynamic_kernel", str(dynamic_kernel))
    settings.setValue("sequence/kernel_settings_revision", KERNEL_SETTINGS_REVISION)
