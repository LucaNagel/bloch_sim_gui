import json
from dataclasses import replace

import pytest

import blochsimulator
import blochsimulator.kernel_capabilities as capability_module
from blochsimulator.kernel_capabilities import (
    DYNAMIC_KERNEL_NAMES,
    GUI_KERNEL_NAMES,
    SEQUENCE_KERNEL_NAMES,
    detect_kernel_capabilities,
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

    native = capabilities["dynamic"]["native_parallel"]
    assert native.available
    assert native.fallback_supported
    assert native.numerical_contract == "float64-close"
    assert native.details["complete_two_pool_plan"]
    assert native.details["complete_multi_pool_plan"]
    assert native.details["spin_major"]
    assert native.details["spin_major_spatial_tx"]


def test_kernel_capability_metadata_is_json_serializable():
    metadata = kernel_capability_metadata()

    encoded = json.dumps(metadata, sort_keys=True)
    assert '"strict_reference"' in encoded
    assert '"native_parallel"' in encoded


def test_persisted_kernel_resolution_keeps_safe_fallback_only(monkeypatch):
    capabilities = detect_kernel_capabilities()
    capabilities["sequence"]["strict_reference"] = replace(
        capabilities["sequence"]["strict_reference"],
        available=False,
        reason="missing test extension",
    )
    capabilities["dynamic"]["native_parallel"] = replace(
        capabilities["dynamic"]["native_parallel"],
        available=False,
        reason="missing test extension",
    )
    monkeypatch.setattr(
        capability_module, "detect_kernel_capabilities", lambda: capabilities
    )

    assert resolve_kernel_request("sequence", "strict_reference") == "optimized"
    assert resolve_kernel_request("dynamic", "native_parallel") == "native_parallel"
    assert resolve_kernel_request("sequence", "invalid") == "optimized"
    with pytest.raises(ValueError, match="kernel family"):
        resolve_kernel_request("invalid", "optimized")


def test_gui_kernel_lists_are_api_subsets_with_cpu_defaults():
    assert GUI_KERNEL_NAMES == {
        "sequence": ("optimized", "strict_reference"),
        "dynamic": ("native_parallel", "optimized"),
    }
    assert set(GUI_KERNEL_NAMES["sequence"]) <= set(SEQUENCE_KERNEL_NAMES)
    assert set(GUI_KERNEL_NAMES["dynamic"]) <= set(DYNAMIC_KERNEL_NAMES)


def test_persisted_kernels_outside_gui_lists_resolve_to_gui_defaults():
    assert resolve_kernel_request("sequence", "reference") == "optimized"
    assert resolve_kernel_request("dynamic", "invalid") == "native_parallel"
    for kernel in ("native_serial", "reference", "metal_hybrid"):
        assert resolve_kernel_request("dynamic", kernel) == "native_parallel"
    assert resolve_kernel_request("dynamic", "optimized") == "optimized"
