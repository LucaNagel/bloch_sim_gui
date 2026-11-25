import json
from dataclasses import replace

import pytest

import blochsimulator
import blochsimulator.kernel_capabilities as capability_module
from blochsimulator.kernel_capabilities import (
    DYNAMIC_KERNEL_NAMES,
    GUI_KERNEL_NAMES,
    KERNEL_ALIASES,
    SEQUENCE_KERNEL_NAMES,
    canonical_kernel_name,
    detect_kernel_capabilities,
    dynamic_kernel_fallback,
    kernel_capability_metadata,
    resolve_kernel_request,
)


def test_shared_kernel_registry_covers_api_families():
    capabilities = detect_kernel_capabilities()

    assert tuple(capabilities) == ("sequence", "dynamic")
    assert set(capabilities["sequence"]) == set(SEQUENCE_KERNEL_NAMES)
    assert set(capabilities["dynamic"]) == set(DYNAMIC_KERNEL_NAMES)
    assert "metal_hybrid" not in DYNAMIC_KERNEL_NAMES
    assert blochsimulator.SEQUENCE_KERNEL_NAMES is SEQUENCE_KERNEL_NAMES
    assert blochsimulator.DYNAMIC_KERNEL_NAMES is DYNAMIC_KERNEL_NAMES
    assert blochsimulator.detect_kernel_capabilities is detect_kernel_capabilities


def test_kernel_capabilities_report_contract_fallback_and_plan_coverage():
    capabilities = detect_kernel_capabilities()

    strict = capabilities["sequence"]["strict_reference"]
    assert strict.available
    assert strict.numerical_contract == "strict-float64"
    assert not strict.fallback_supported

    production = capabilities["sequence"]["production"]
    assert production.available
    assert production.fallback_supported
    assert production.numerical_contract == "float64-close"
    assert production.details["spin_major"]
    legacy = capabilities["sequence"]["legacy"]
    assert legacy.available
    assert legacy.numerical_contract == "legacy-fast-math"
    assert not legacy.fallback_supported

    native = capabilities["dynamic"]["production"]
    assert native.available
    assert native.fallback_supported
    assert native.numerical_contract == "float64-close"
    assert native.details["spin_major"]
    assert native.details["spin_major_spatial_tx"]


def test_kernel_capability_metadata_is_json_serializable():
    metadata = kernel_capability_metadata()

    encoded = json.dumps(metadata, sort_keys=True)
    assert '"strict_reference"' in encoded
    assert '"production"' in encoded


def test_persisted_kernel_resolution_keeps_safe_fallback_only(monkeypatch):
    capabilities = detect_kernel_capabilities()
    capabilities["sequence"]["strict_reference"] = replace(
        capabilities["sequence"]["strict_reference"],
        available=False,
        reason="missing test extension",
    )
    capabilities["dynamic"]["production"] = replace(
        capabilities["dynamic"]["production"],
        available=False,
        reason="missing test extension",
    )
    monkeypatch.setattr(
        capability_module, "detect_kernel_capabilities", lambda: capabilities
    )

    assert resolve_kernel_request("sequence", "strict_reference") == "production"
    assert resolve_kernel_request("dynamic", "production") == "production"
    assert resolve_kernel_request("dynamic", "native_parallel") == "production"
    assert resolve_kernel_request("sequence", "invalid") == "production"
    with pytest.raises(ValueError, match="kernel family"):
        resolve_kernel_request("invalid", "production")


def test_gui_kernel_lists_are_api_subsets_with_cpu_defaults():
    assert GUI_KERNEL_NAMES == {
        "sequence": ("production", "legacy", "strict_reference"),
        "dynamic": ("production",),
    }
    assert set(GUI_KERNEL_NAMES["sequence"]) <= set(SEQUENCE_KERNEL_NAMES)
    assert set(GUI_KERNEL_NAMES["dynamic"]) <= set(DYNAMIC_KERNEL_NAMES)


def test_persisted_kernels_outside_gui_lists_resolve_to_gui_defaults():
    assert resolve_kernel_request("sequence", "reference") == "production"
    # "optimized" was the static GUI default and follows the new default;
    # an explicit legacy choice stays.
    assert resolve_kernel_request("sequence", "optimized") == "production"
    assert resolve_kernel_request("sequence", "legacy") == "legacy"
    assert resolve_kernel_request("dynamic", "invalid") == "production"
    # The NumPy path is no longer offered in the GUI; old settings migrate.
    for kernel in ("native_serial", "reference", "metal_hybrid", "optimized", "numpy"):
        assert resolve_kernel_request("dynamic", kernel) == "production"


def test_old_kernel_names_are_aliases_of_the_public_names():
    assert KERNEL_ALIASES == {
        "sequence": {"optimized": "legacy"},
        "dynamic": {
            "native_parallel": "production",
            "native_serial": "production",
            "optimized": "numpy",
        },
    }
    assert canonical_kernel_name("sequence", "optimized") == "legacy"
    assert canonical_kernel_name("sequence", "legacy") == "legacy"
    assert canonical_kernel_name("sequence", "strict_reference") == "strict_reference"
    assert canonical_kernel_name("dynamic", "optimized") == "numpy"
    assert canonical_kernel_name("dynamic", "native_serial") == "production"
    assert canonical_kernel_name("dynamic", "reference") == "reference"
    for family, name in (("sequence", "reference"), ("dynamic", "metal_hybrid")):
        with pytest.raises(ValueError):
            canonical_kernel_name(family, name)
    with pytest.raises(ValueError, match="kernel family"):
        canonical_kernel_name("invalid", "production")
    assert blochsimulator.canonical_kernel_name is canonical_kernel_name


def test_dynamic_fallback_reads_old_and_new_metadata():
    assert (
        dynamic_kernel_fallback(
            {"sequence_kernel": "production", "requested_sequence_kernel": "production"}
        )
        is None
    )
    # An old name in the request is not a fallback to the public name.
    assert (
        dynamic_kernel_fallback(
            {
                "sequence_kernel": "production",
                "requested_sequence_kernel": "native_serial",
            }
        )
        is None
    )
    # Results saved before the renaming carry the internal path names.
    assert (
        dynamic_kernel_fallback(
            {
                "sequence_kernel": "native_parallel",
                "requested_sequence_kernel": "native_parallel",
            }
        )
        is None
    )
    reason = dynamic_kernel_fallback(
        {
            "sequence_kernel": "optimized",
            "requested_sequence_kernel": "native_parallel",
            "native_fallback_reason": "test reason",
        }
    )
    assert reason == "requested the native_parallel kernel, ran numpy: test reason"


def test_kernel_name_lists_accept_old_names_in_membership_tests():
    # Scripts such as benchmarks/benchmark_kernels.py filter old names with
    # ``in``; iteration still yields only the public names.
    assert tuple(SEQUENCE_KERNEL_NAMES) == ("production", "legacy", "strict_reference")
    assert tuple(DYNAMIC_KERNEL_NAMES) == ("production", "numpy", "reference")
    assert "optimized" in SEQUENCE_KERNEL_NAMES
    for name in ("optimized", "native_serial", "native_parallel"):
        assert name in DYNAMIC_KERNEL_NAMES
    assert "reference" not in SEQUENCE_KERNEL_NAMES
    assert "metal_hybrid" not in DYNAMIC_KERNEL_NAMES
