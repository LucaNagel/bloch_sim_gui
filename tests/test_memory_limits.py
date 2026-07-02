import pytest

from blochsimulator.memory import (
    GIB,
    MemoryBudget,
    MemoryPolicy,
    SimulationMemoryError,
    enforce_memory_budget,
    enforce_sequence_memory,
    resolve_memory_budget,
)
from blochsimulator.simulator import BlochSimulator


def test_automatic_budget_keeps_minimum_reserve():
    budget = resolve_memory_budget(
        available_bytes=12 * GIB,
        total_bytes=16 * GIB,
    )

    assert budget.limit_bytes == 10 * GIB
    assert budget.reserve_bytes == 2 * GIB


def test_automatic_budget_scales_reserve_on_large_systems():
    budget = resolve_memory_budget(
        available_bytes=56 * GIB,
        total_bytes=64 * GIB,
    )

    assert budget.reserve_bytes == int(6.4 * GIB)
    assert budget.limit_bytes == 56 * GIB - int(6.4 * GIB)


def test_custom_reserve_is_subtracted_from_available_memory():
    policy = MemoryPolicy(mode="custom_reserve", reserve_bytes=4 * GIB)
    budget = resolve_memory_budget(
        policy=policy,
        available_bytes=20 * GIB,
        total_bytes=32 * GIB,
    )

    assert budget.limit_bytes == 16 * GIB


def test_explicit_budget_overrides_hardware_detection():
    budget = resolve_memory_budget(
        explicit_limit_bytes=256 * 1024**2,
        available_bytes=16 * GIB,
    )

    assert budget.limit_bytes == 256 * 1024**2
    assert budget.mode == "fixed_limit"


def test_missing_hardware_metrics_use_conservative_fallback(monkeypatch):
    monkeypatch.setattr(
        "blochsimulator.memory.system_memory_bytes", lambda: (None, None)
    )

    budget = resolve_memory_budget()

    assert budget.limit_bytes == 512 * 1024**2


def test_error_reports_estimate_and_reduction_options():
    with pytest.raises(SimulationMemoryError) as exc_info:
        enforce_memory_budget(
            2 * GIB,
            MemoryBudget(1 * GIB, 2 * GIB, 4 * GIB, 1 * GIB, "custom_reserve"),
            description="Requested test data",
            suggestions="Use Endpoint mode",
        )

    message = str(exc_info.value)
    assert message.startswith("Memory limit exceeded:")
    assert "needs approximately 2.00 GiB RAM" in message
    assert "safe budget is 1.00 GiB" in message
    assert "use endpoint mode" in message


def test_standard_simulation_is_rejected_before_large_allocation():
    simulator = BlochSimulator(memory_limit_bytes=64 * 1024**2)

    with pytest.raises(SimulationMemoryError, match="positions"):
        simulator._check_standard_simulation_memory(
            ntime=100_000,
            npos=100,
            nfreq=10,
            mode=2,
        )


def test_endpoint_simulation_stays_within_same_budget():
    simulator = BlochSimulator(memory_limit_bytes=64 * 1024**2)

    simulator._check_standard_simulation_memory(
        ntime=100_000,
        npos=100,
        nfreq=10,
        mode=0,
    )


def test_sequence_guard_runs_before_array_creation(monkeypatch):
    monkeypatch.setattr(
        "blochsimulator.memory.system_memory_bytes",
        lambda: (128 * 1024**2, 8 * GIB),
    )

    with pytest.raises(SimulationMemoryError, match="compiled sequence"):
        enforce_sequence_memory(2_000_000)
