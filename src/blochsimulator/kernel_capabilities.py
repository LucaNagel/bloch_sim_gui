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


SEQUENCE_KERNEL_NAMES = ("optimized", "reference", "strict_reference")
DYNAMIC_KERNEL_NAMES = (
    "optimized",
    "native_parallel",
    "native_serial",
    "reference",
)
# The settings dialog offers only these choices; the first one is the default.
# The remaining kernels stay available from Python for benchmarks and validation.
GUI_SEQUENCE_KERNEL_NAMES = ("optimized", "strict_reference")
GUI_DYNAMIC_KERNEL_NAMES = ("native_parallel", "optimized")
GUI_KERNEL_NAMES = {
    "sequence": GUI_SEQUENCE_KERNEL_NAMES,
    "dynamic": GUI_DYNAMIC_KERNEL_NAMES,
}


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
    dynamic_available, dynamic_reason, dynamic_module = _extension_status(
        "blochsimulator.dynamic_bloch_cy",
        (
            "apply_rf_rotation_transverse_block",
            "run_dynamic_plan_block_no_inflow",
        ),
    )
    dynamic_details = {
        "complete_two_pool_plan": bool(
            dynamic_module is not None
            and hasattr(
                dynamic_module, "apply_dynamic_rf_block_with_concentration_inflow"
            )
        ),
        "complete_multi_pool_plan": bool(
            dynamic_module is not None
            and hasattr(dynamic_module, "run_dynamic_plan_block_multi_pool")
        ),
    }

    sequence = {
        "optimized": KernelCapability(
            "sequence",
            "optimized",
            legacy_available,
            reason=legacy_reason,
            numerical_contract="legacy-fast-math",
        ),
        "reference": KernelCapability(
            "sequence",
            "reference",
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
        "optimized": KernelCapability(
            "dynamic",
            "optimized",
            True,
            numerical_contract="bitwise",
        ),
        "native_parallel": KernelCapability(
            "dynamic",
            "native_parallel",
            dynamic_available,
            fallback_supported=True,
            reason=dynamic_reason,
            numerical_contract="float64-close",
            details=dynamic_details,
        ),
        "native_serial": KernelCapability(
            "dynamic",
            "native_serial",
            dynamic_available,
            fallback_supported=True,
            reason=dynamic_reason,
            numerical_contract="float64-close",
            details=dynamic_details,
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

    Choices that the settings dialog no longer offers, unknown names, and
    unavailable kernels without a CPU fallback resolve to the GUI default.
    """
    family = str(family)
    requested = str(requested)
    capabilities = detect_kernel_capabilities()
    if family not in capabilities:
        raise ValueError("kernel family must be 'sequence' or 'dynamic'")
    capability = capabilities[family].get(requested)
    if (
        requested in GUI_KERNEL_NAMES[family]
        and capability is not None
        and (capability.available or capability.fallback_supported)
    ):
        return requested
    return GUI_KERNEL_NAMES[family][0]
