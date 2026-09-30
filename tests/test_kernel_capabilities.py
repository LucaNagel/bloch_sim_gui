import json
from dataclasses import replace

import pytest

import blochsimulator
import blochsimulator.kernel_capabilities as capability_module
from blochsimulator.kernel_capabilities import (
    DYNAMIC_CPU_KERNEL_NAMES,
    DYNAMIC_KERNEL_NAMES,
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
    assert set(DYNAMIC_CPU_KERNEL_NAMES) == set(DYNAMIC_KERNEL_NAMES) - {"metal_hybrid"}
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

    metal = capabilities["dynamic"]["metal_hybrid"]
    assert metal.experimental
    assert metal.fallback_supported
    assert metal.numerical_contract == "experimental-mixed"
    if not metal.available:
        assert metal.reason


def test_kernel_capability_metadata_is_json_serializable():
    metadata = kernel_capability_metadata()

    encoded = json.dumps(metadata, sort_keys=True)
    assert '"strict_reference"' in encoded
    assert '"metal_hybrid"' in encoded


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
