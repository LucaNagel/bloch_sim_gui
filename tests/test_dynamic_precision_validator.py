from dataclasses import replace

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from scripts import validate_dynamic_precision as validator
from test_dynamic_native_plan import _native_plan_fixture


def test_validator_executes_requested_subvoxel_sampling():
    phantom, program, raster = _native_plan_fixture()
    report = validator.compare_kernels(
        program,
        phantom,
        simulation_timestep_s=raster,
        candidates=("native_serial",),
        spin_sampling=(2, 2, 2),
        native_variants=True,
        refined_timestep_s=raster / 2,
    )
    assert report["spoiler_mode"] == "gradient"
    assert report["spin_sampling"]["spins_per_voxel"] == 8
    for candidate in report["candidates"].values():
        assert candidate["metadata"]["n_simulated_spins"] == 32
        assert candidate["gate_passed"]
        assert candidate["metrics"]["checkpoint_pool_state"]["within_tolerance"]
    assert report["timestep_refinement"]["compiled_grid_changed"] is False


@pytest.mark.parametrize("fault", ["pools", "final", "checkpoint", "nonfinite"])
def test_validator_rejects_errors_hidden_by_combined_signal(monkeypatch, fault):
    phantom, program, raster = _native_plan_fixture()
    reference = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program,
        phantom,
        simulation_timestep_s=raster,
        checkpoints_s=(program.duration_s / 2, program.duration_s),
    )
    if fault == "pools":
        values = reference.species_signal.copy()
        values[0] += 0.1
        values[1] -= 0.1
        candidate = replace(reference, species_signal=values)
    elif fault == "nonfinite":
        values = reference.final_pool_magnetization.copy()
        values.flat[0] = np.nan
        candidate = replace(reference, final_pool_magnetization=values)
    else:
        attribute = (
            "final_pool_magnetization"
            if fault == "final"
            else "checkpoint_pool_magnetization"
        )
        values = getattr(reference, attribute).copy()
        values.flat[0] += 0.5
        candidate = replace(reference, **{attribute: values})
    candidate = replace(
        candidate, metadata={**candidate.metadata, "sequence_kernel": "native_serial"}
    )

    class ControlledSimulator:
        def __init__(self, *, dynamic_sequence_kernel, **kwargs):
            self.kernel = dynamic_sequence_kernel

        def simulate_dynamic_sequence(self, *args, **kwargs):
            return reference if self.kernel == "optimized" else candidate

    monkeypatch.setattr(validator, "BlochSimulator", ControlledSimulator)
    report = validator.compare_kernels(
        program,
        phantom,
        simulation_timestep_s=raster,
        candidates=("native_serial",),
    )
    result = report["candidates"]["native_serial"]
    assert result["metrics"]["total_signal"]["nrmse"] == 0
    assert result["failed_metrics"]
    assert not result["gate_passed"]


def test_validator_rejects_ideal_spoiling_with_subvoxel_request():
    phantom, program, raster = _native_plan_fixture()
    with pytest.raises(ValueError, match="subvoxel sampling requires"):
        validator.compare_kernels(
            program,
            phantom,
            simulation_timestep_s=raster,
            spin_sampling=(2, 1, 1),
            spoiler_mode="ideal",
        )


def test_requested_spin_major_variant_cannot_pass_on_previous_kernel(monkeypatch):
    from blochsimulator import dynamic_phantom

    monkeypatch.setattr(dynamic_phantom, "_spin_major_kernel", lambda: None)
    phantom, program, raster = _native_plan_fixture()
    report = validator.compare_kernels(
        program,
        phantom,
        simulation_timestep_s=raster,
        candidates=("native_serial",),
        native_variants=True,
    )
    assert report["candidates"]["native_serial_previous"]["gate_passed"]
    stepwise = report["candidates"]["native_serial_stepwise"]
    assert stepwise["accuracy_gate_passed"]
    assert not stepwise["requested_backend_executed"]
    assert not stepwise["gate_passed"]
    assert stepwise["fallback_used"]


def test_missing_checkpoint_cannot_pass_array_gate():
    assert not validator._array_metric(np.ones((2, 3)), None)["within_tolerance"]


@pytest.mark.parametrize("scale", [1.0, 1e6])
def test_absolute_floor_tracks_reference_units_at_signal_cancellation(scale):
    reference = np.array([0.0, 1.0]) * scale
    roundoff = np.array([5e-13, 5e-11]) * scale
    metric = validator._array_metric(reference, reference + roundoff)
    assert metric["within_tolerance"]
    assert metric["absolute_tolerance"] == pytest.approx(1e-12 * scale)
    assert not validator._array_metric(
        reference, reference + np.array([5e-11, 5e-11]) * scale
    )["within_tolerance"]


@pytest.mark.parametrize("name", ["rtol", "atol", "signal_nrmse_gate"])
def test_validator_rejects_invalid_tolerances(name):
    phantom, program, raster = _native_plan_fixture()
    with pytest.raises(ValueError, match=name):
        validator.compare_kernels(
            program,
            phantom,
            simulation_timestep_s=raster,
            **{name: np.nan},
        )
