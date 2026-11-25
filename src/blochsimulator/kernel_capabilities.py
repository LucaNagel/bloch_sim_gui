"""Shared kernel names and runtime capability detection.

The public API accepts stable kernel names even when an optional accelerator is
not present.  ``fallback_supported`` distinguishes dynamic requests that can
return an exact CPU result from validation kernels that require their compiled
extension.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Mapping


# Older names stay readable (projects, settings, scripts, benchmarks). They
# run exactly the path they ran before; results carry the public name. The
# static "optimized" kernel is the fast-math C kernel, now called "legacy".
KERNEL_ALIASES = {
    "sequence": {"optimized": "legacy"},
    "dynamic": {
        "native_parallel": "production",
        "native_serial": "production",
        "optimized": "numpy",
    },
}


class _KernelNames(tuple):
    """Public kernel names; ``in`` also accepts the family's old aliases.

    Iteration and indexing yield only the public names. Membership tests
    accept the old names too, so scripts that check ``name in ...`` before
    passing an old name keep working.
    """

    def __new__(cls, names, aliases):
        self = super().__new__(cls, names)
        self._aliases = frozenset(aliases)
        return self

    def __contains__(self, name) -> bool:
        return tuple.__contains__(self, name) or name in self._aliases


# Public kernel names: one production kernel and one reference per family.
# The static production kernel runs on the spin-major executor and falls back
# to the fast-math C kernel ("legacy", the previous default) for objects it
# cannot run. strict_reference is a no-fast-math check of the legacy kernel;
# the physical static reference is sequence.reference (no kernel name).
SEQUENCE_KERNEL_NAMES = _KernelNames(
    ("production", "legacy", "strict_reference"), KERNEL_ALIASES["sequence"]
)
DYNAMIC_KERNEL_NAMES = _KernelNames(
    ("production", "numpy", "reference"), KERNEL_ALIASES["dynamic"]
)
# The dynamic production kernel with one thread is the former native_serial.
_DYNAMIC_INTERNAL = {
    "production": "native_parallel",
    "native_parallel": "native_parallel",
    "native_serial": "native_serial",
    "numpy": "optimized",
    "optimized": "optimized",
    "reference": "reference",
}
_DYNAMIC_PUBLIC = {
    "native_parallel": "production",
    "native_serial": "production",
    "optimized": "numpy",
    "reference": "reference",
}
# The settings dialog offers only these choices; the first one is the default.
# The remaining kernels stay available from Python for benchmarks and validation.
GUI_SEQUENCE_KERNEL_NAMES = ("production", "legacy", "strict_reference")
GUI_DYNAMIC_KERNEL_NAMES = ("production",)
GUI_KERNEL_NAMES = {
    "sequence": GUI_SEQUENCE_KERNEL_NAMES,
    "dynamic": GUI_DYNAMIC_KERNEL_NAMES,
}
_FAMILY_NAMES = {"sequence": SEQUENCE_KERNEL_NAMES, "dynamic": DYNAMIC_KERNEL_NAMES}


def canonical_kernel_name(family: str, name: str) -> str:
    """Return the public kernel name for a name or one of its old aliases.

    Raises ``ValueError`` for an unknown family or name.
    """
    if family not in _FAMILY_NAMES:
        raise ValueError("kernel family must be 'sequence' or 'dynamic'")
    name = str(name)
    name = KERNEL_ALIASES[family].get(name, name)
    if name not in _FAMILY_NAMES[family]:
        accepted = (*_FAMILY_NAMES[family], *KERNEL_ALIASES[family])
        raise ValueError(
            f"{family} kernel must be one of "
            + ", ".join(repr(item) for item in accepted)
        )
    return name


def internal_dynamic_kernel(name: str) -> str:
    """Return the internal dynamic path name for a public name or alias."""
    canonical_kernel_name("dynamic", name)
    return _DYNAMIC_INTERNAL[str(name)]


def public_dynamic_kernel(internal: str | None) -> str | None:
    """Return the public name of an internal dynamic path name."""
    return None if internal is None else _DYNAMIC_PUBLIC.get(internal, internal)


# Reason recorded when the spin-major executor is switched off on purpose.
SPIN_MAJOR_DISABLED_REASON = "disabled by request"


class KernelFallbackWarning(RuntimeWarning):
    """A dynamic simulation ran on another path than the requested kernel.

    The result is still valid, but usually much slower.
    ``warnings.simplefilter("error", KernelFallbackWarning)`` turns the
    fallback into an exception.
    """


def dynamic_kernel_fallback(metadata: Mapping | None) -> str | None:
    """Return why a dynamic run left its requested kernel, or None.

    ``metadata`` is the metadata of a dynamic simulation result. A spin-major
    executor that was switched off on purpose is not a fallback; the run then
    used the NumPy kernel.
    """
    metadata = dict(metadata or {})
    # Results written before the renaming carry the internal path names.
    actual = public_dynamic_kernel(metadata.get("sequence_kernel"))
    contrast_from = metadata.get("contrast_kernel_fallback_from")
    if contrast_from:
        contrast_from = public_dynamic_kernel(contrast_from)
        reason = metadata.get("contrast_kernel_fallback_reason")
        return (
            f"requested the {contrast_from} kernel, ran {actual}: contrast "
            "relaxation needs the spin-major executor"
            + (f" ({reason})" if reason else "")
        )
    requested = metadata.get("requested_sequence_kernel") or actual
    try:
        requested_public = canonical_kernel_name("dynamic", requested)
    except ValueError:
        requested_public = requested
    if requested_public != actual:
        reason = metadata.get("native_fallback_reason") or "no reason recorded"
        if reason == f"spin-major executor: {SPIN_MAJOR_DISABLED_REASON}":
            # Switching the executor off on purpose runs the NumPy kernel.
            return None
        return f"requested the {requested} kernel, ran {actual}: {reason}"
    reason = metadata.get("native_spin_major_fallback_reason")
    if (
        actual == "production"
        and not metadata.get("native_spin_major_enabled")
        and reason
        and reason != SPIN_MAJOR_DISABLED_REASON
    ):
        return f"the {actual} kernel ran without its spin-major executor: {reason}"
    return None


@dataclass(frozen=True)
class KernelCapability:
    """Availability and dispatch contract for one stable kernel name."""

    family: str
    name: str
    available: bool
    experimental: bool = False
    fallback_supported: bool = False
    reason: str | None = None
    numerical_contract: str | None = None
    details: Mapping[str, object] | None = None

    def to_metadata(self) -> dict:
        """Return a serialization-friendly capability record."""
        record = asdict(self)
        record["details"] = dict(self.details or {})
        return record


def _extension_status(module_name: str, required_attributes=()):
    try:
        module = import_module(module_name)
    except (ImportError, OSError) as error:
        return False, f"compiled extension is unavailable: {error}", None
    missing = tuple(name for name in required_attributes if not hasattr(module, name))
    if missing:
        return (
            False,
            "compiled extension is missing " + ", ".join(missing),
            module,
        )
    return True, None, module


def detect_kernel_capabilities() -> dict[str, dict[str, KernelCapability]]:
    """Detect ordinary and dynamic-native kernel capabilities."""
    legacy_available, legacy_reason, _ = _extension_status(
        "blochsimulator.blochsimulator_cy", ("simulate_sequence_chunk",)
    )
    strict_available, strict_reason, _ = _extension_status(
        "blochsimulator.blochsimulator_strict_cy", ("simulate_sequence_chunk",)
    )
    spin_available, spin_reason, spin_module = _extension_status(
        "blochsimulator.dynamic_spin_major_cy",
        ("run_spin_major_chunk", "propagator_values", "propagator_columns"),
    )
    if spin_available and getattr(spin_module, "API_VERSION", None) != 9:
        spin_available = False
        spin_reason = "compiled extension has an incompatible API version"
    dynamic_details = {
        "spin_major": spin_available,
        "spin_major_spatial_tx": bool(
            spin_available and getattr(spin_module, "SPATIAL_TX_SUPPORTED", False)
        ),
        "spin_major_multi_pool": bool(
            spin_available and getattr(spin_module, "MULTI_POOL_SUPPORTED", False)
        ),
        "spin_major_spatial_inflow": bool(
            spin_available and getattr(spin_module, "SPATIAL_INFLOW_SUPPORTED", False)
        ),
        "spin_major_rf_cache_multi_pool": bool(
            spin_available
            and getattr(spin_module, "RF_CACHE_MULTI_POOL_SUPPORTED", False)
        ),
        "spin_major_rf_cache_phase": bool(
            spin_available and getattr(spin_module, "RF_CACHE_PHASE_SUPPORTED", False)
        ),
        "spin_major_contrast_relaxation": bool(
            spin_available
            and getattr(spin_module, "CONTRAST_RELAXATION_SUPPORTED", False)
        ),
    }

    sequence = {
        # Production needs the legacy extension for its fallback.
        "production": KernelCapability(
            "sequence",
            "production",
            legacy_available,
            fallback_supported=True,
            reason=legacy_reason,
            numerical_contract="float64-close",
            details={"spin_major": spin_available},
        ),
        "legacy": KernelCapability(
            "sequence",
            "legacy",
            legacy_available,
            reason=legacy_reason,
            numerical_contract="legacy-fast-math",
        ),
        "strict_reference": KernelCapability(
            "sequence",
            "strict_reference",
            strict_available,
            reason=strict_reason,
            numerical_contract="strict-float64",
        ),
    }
    dynamic = {
        "production": KernelCapability(
            "dynamic",
            "production",
            spin_available,
            fallback_supported=True,
            reason=spin_reason,
            numerical_contract="float64-close",
            details=dynamic_details,
        ),
        "numpy": KernelCapability(
            "dynamic",
            "numpy",
            True,
            numerical_contract="bitwise",
        ),
        "reference": KernelCapability(
            "dynamic",
            "reference",
            True,
            numerical_contract="bitwise",
        ),
    }
    return {"sequence": sequence, "dynamic": dynamic}


def kernel_capability_metadata() -> dict[str, dict[str, dict]]:
    """Return all capability records in a JSON-serializable mapping."""
    return {
        family: {name: value.to_metadata() for name, value in values.items()}
        for family, values in detect_kernel_capabilities().items()
    }


def resolve_kernel_request(family: str, requested: str) -> str:
    """Resolve a persisted GUI choice without hiding supported CPU fallback.

    Old names resolve to their public name. Choices that the settings dialog
    no longer offers, unknown names, and unavailable kernels without a CPU
    fallback resolve to the GUI default.
    """
    family = str(family)
    capabilities = detect_kernel_capabilities()
    if family not in capabilities:
        raise ValueError("kernel family must be 'sequence' or 'dynamic'")
    # "optimized" was the static GUI default before the spin-major switch;
    # a stored default follows the new default.
    if family == "sequence" and str(requested) == "optimized":
        return GUI_KERNEL_NAMES[family][0]
    try:
        requested = canonical_kernel_name(family, requested)
    except ValueError:
        return GUI_KERNEL_NAMES[family][0]
    capability = capabilities[family].get(requested)
    if (
        requested in GUI_KERNEL_NAMES[family]
        and capability is not None
        and (capability.available or capability.fallback_supported)
    ):
        return requested
    return GUI_KERNEL_NAMES[family][0]
