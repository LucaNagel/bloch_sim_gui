"""Persistent defaults shared by sequence and phantom creation UIs."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

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
DEFAULT_SEQUENCE_TIMESTEP_US = 20.0
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
    "Otherwise the error grows roughly with the square of the step."
)


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
    try:
        revision = int(settings.value("sequence/kernel_settings_revision", 1))
    except (TypeError, ValueError):
        revision = 1
    if revision < KERNEL_SETTINGS_REVISION and dynamic_kernel == "optimized":
        dynamic_kernel = GUI_KERNEL_NAMES["dynamic"][0]
    return (
        resolve_kernel_request("sequence", sequence_kernel),
        resolve_kernel_request("dynamic", dynamic_kernel),
    )


def save_kernel_settings(settings, sequence_kernel: str, dynamic_kernel: str) -> None:
    """Persist explicit kernel choices together with the settings revision."""
    settings.setValue("sequence/kernel", str(sequence_kernel))
    settings.setValue("sequence/dynamic_kernel", str(dynamic_kernel))
    settings.setValue("sequence/kernel_settings_revision", KERNEL_SETTINGS_REVISION)
