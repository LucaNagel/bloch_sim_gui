"""Persistent defaults shared by sequence and phantom creation UIs."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..kernel_capabilities import GUI_KERNEL_NAMES, resolve_kernel_request
from ..units import NUCLEUS_GAMMA_HZ_PER_T

# Revision 2 made Native automatic the dynamic default. Older settings stored
# the previous default, Optimized NumPy, even when it was never chosen.
KERNEL_SETTINGS_REVISION = 2


@dataclass(frozen=True)
class WorkspaceDefaults:
    """Values used when a new sequence workspace or phantom is created."""

    sequence_fov_mm: tuple[float, float, float] = (220.0, 220.0, 160.0)
    phantom_fov_mm: tuple[float, float, float] = (220.0, 220.0, 220.0)
    # Keep the historical attribute name for project/settings compatibility.
    # The value is now the concrete nucleus shared by every workspace.
    phantom_nucleus: str = "H1"
    field_strength_t: float = 3.0

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
        return cls(sequence_fov, phantom_fov, nucleus, field_strength_t)

    def save(self, settings) -> None:
        for axis, value in zip("xyz", self.sequence_fov_mm):
            settings.setValue(f"defaults/sequence_fov_{axis}_mm", float(value))
        for axis, value in zip("xyz", self.phantom_fov_mm):
            settings.setValue(f"defaults/phantom_fov_{axis}_mm", float(value))
        settings.setValue("defaults/phantom_nucleus", self.phantom_nucleus)
        settings.setValue("defaults/field_strength_t", float(self.field_strength_t))


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
