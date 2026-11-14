"""The dynamic time-step check runs the dynamic kernel on a voxel subset.

Voxels of a dynamic phantom do not couple, so a subset run reproduces the
full-phantom result of each selected voxel. The check compares the GUI step
with a quarter step on such a subset, including conversion, inflow and
washout, and must track the error against the converged (1 us) solution.
"""

import json
import warnings

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import (
    DynamicSpectralPhantom,
    PyruvateInflow,
    TimeCurve,
    VoxelWashout,
)
from blochsimulator.kernel_capabilities import (
    GUI_DYNAMIC_KERNEL_NAMES,
    KernelFallbackWarning,
)
from blochsimulator.sequence.convergence import (
    DYNAMIC_TIMESTEP_ERROR_PEAK_FLOOR,
    _timestep_error_program,
    default_probe_checkpoints,
    dynamic_timestep_error_voxels,
    estimate_dynamic_timestep_error,
    timestep_error_frequencies,
)
from blochsimulator.sequence.model import (
    ADCEvent,
    GradientEvent,
    RFEvent,
    SequenceProgram,
)
from blochsimulator.spectral_phantom import ChemicalSpecies
from blochsimulator.ui import sequence_simulation_widget as widget_module
from blochsimulator.ui.default_settings import (
    TIMESTEP_ERROR_DYNAMIC_RUNTIME_FRACTION,
    TIMESTEP_ERROR_DYNAMIC_SECONDS_PER_INTERVAL,
    TIMESTEP_ERROR_MIN_INTERVALS,
    TIMESTEP_ERROR_RUNTIME_FRACTION,
    TIMESTEP_ERROR_SECONDS_PER_INTERVAL,
)
from blochsimulator.ui.sequence_simulation_widget import (
    SequenceSimulationThread,
    SequenceSimulationWidget,
)
from blochsimulator.ui.theme import tone_of

FIELD_T = 3.0
NUCLEUS = "C13"
RASTER_S = 1e-6
OPTIONS = dict(field_strength_t=FIELD_T, nucleus=NUCLEUS, sequence_reference_ppm=0.0)


def _slice_train(repetitions=4, repetition_s=4e-3):
    """Phase-cycled 30 degree, 1 ms sinc pulses under a slice gradient."""
    lobes = np.linspace(-2.0, 2.0, int(round(1e-3 / RASTER_S)))
    shape = np.sinc(lobes)
    pulse_hz = 30.0 / 360.0 / 1e-3 / np.mean(shape) * shape
    events = []
    for index in range(repetitions):
        start = index * repetition_s
        phase = np.pi * index
        events.append(RFEvent(start, pulse_hz * np.exp(1j * phase), RASTER_S))
        events.append(GradientEvent("z", start, np.full(100, 1.0e5), 10e-6))
        events.append(GradientEvent("z", start + 1.0e-3, np.full(50, -1.0e5), 10e-6))
        events.append(ADCEvent(start + 1.6e-3, 32, 50e-6, phase_offset_rad=phase))
    return SequenceProgram(tuple(events), repetitions * repetition_s)


def _phantom(shape=(1, 1, 6), fov=(0.01, 0.01, 0.03), seed=None):
    """Pyruvate, lactate and a co-labelled pool with exchange and washout.

    The last voxel is a washout source compartment. Rates are far above in
    vivo values so that conversion, inflow and washout act within a few ms.
    With ``seed`` the maps vary randomly instead of along the voxel order.
    """
    count = int(np.prod(shape))
    if seed is None:
        grids = [np.linspace(0.0, 1.0, count).reshape(shape)] * 6
    else:
        rng = np.random.default_rng(seed)
        grids = [rng.random(shape) for _ in range(6)]
    density = {
        "Pyruvate": 1.0 + 0.2 * grids[0],
        "Lactate": 0.1 + 0.05 * grids[0],
        "C1": 0.5 + 0.1 * grids[0],
    }
    source = np.zeros(shape, dtype=bool)
    source.reshape(-1)[-1] = True
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=fov,
        pools=tuple(
            ChemicalSpecies(name, shift, t1, t2)
            for name, shift, t1, t2 in zip(
                density, (0.0, 12.0, 9.0), (30.0, 25.0, 20.0), (1.0, 0.8, 0.6)
            )
        ),
        initial_concentration_maps=density,
        initial_spin_density_maps=density,
        kpl_map_s_inv=np.where(source, 0.0, 2.0 + 18.0 * grids[1]),
        b0_map=-150.0 + 300.0 * grids[2],
        field_strength=FIELD_T,
        nucleus=NUCLEUS,
        spectral_reference_ppm=0.0,
    )
    phantom.tx_sensitivity_map = (0.8 + 0.3 * grids[3]).astype(np.complex128)
    phantom.pyruvate_inflow = PyruvateInflow(
        TimeCurve((0.0, 8e-3, 20e-3, 32e-3), (20.0, 300.0, 100.0, 200.0)),
        0.8 + 0.4 * grids[4],
        TimeCurve((0.0, 16e-3, 32e-3), (0.5, 0.3, 0.45)),
        colabelled_pools=("C1",),
    )
    rates = np.broadcast_to(np.asarray((30.0, 300.0, 120.0)), (1, 3, 3)).copy()
    phantom.voxel_washout = VoxelWashout(
        loss_map_s_inv=5.0 + 40.0 * grids[5],
        compartment_map=np.where(source, 0, -1).astype(np.int32),
        times_s=np.asarray((2e-3, 12e-3, 30e-3)),
        concentration_rates=rates,
        magnetization_rates=0.4 * rates,
    )
    phantom.__post_init__()
    return phantom


def _simulator(kernel):
    return BlochSimulator(
        use_parallel=kernel == "native_parallel", dynamic_sequence_kernel=kernel
    )


@pytest.mark.parametrize("kernel", GUI_DYNAMIC_KERNEL_NAMES)
def test_voxel_subset_reproduces_the_full_phantom(kernel):
    program = _slice_train()
    phantom = _phantom()
    checkpoints = np.union1d(
        default_probe_checkpoints(program, max_rf_checkpoints=8), (0.0,)
    )
    subset = np.asarray((1, 3, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("error", KernelFallbackWarning)
        full, part = (
            _simulator(kernel).simulate_dynamic_sequence(
                program,
                phantom,
                simulation_timestep_s=20e-6,
                checkpoints_s=checkpoints,
                **OPTIONS,
                **extra,
            )
            for extra in ({}, {"_voxel_indices": subset})
        )

    full_final = np.asarray(full.final_pool_magnetization)
    full_final = full_final.reshape(full_final.shape[0], -1, 3)
    np.testing.assert_allclose(
        part.final_pool_magnetization, full_final[:, subset], rtol=0, atol=1e-12
    )
    full_checkpoints = np.asarray(full.checkpoint_pool_magnetization)
    full_checkpoints = full_checkpoints.reshape(full_checkpoints.shape[:2] + (-1, 3))
    np.testing.assert_allclose(
        part.checkpoint_pool_magnetization,
        full_checkpoints[:, :, subset],
        rtol=0,
        atol=1e-12,
    )
    # The co-labelled pool and the washout source take part in the subset.
    assert part.metadata["colabelled_pools"] == ["C1"]
    assert np.abs(part.final_pool_magnetization[2]).max() > 0


@pytest.mark.parametrize("indices", ([3, 1], [0, 6], [-1, 2], [], [[1, 2]]))
def test_voxel_subset_must_be_increasing_flat_indices(indices):
    with pytest.raises(ValueError, match="_voxel_indices"):
        BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
            _slice_train(1),
            _phantom(),
            simulation_timestep_s=20e-6,
            _voxel_indices=np.asarray(indices, dtype=np.int64),
            **OPTIONS,
        )


def test_small_phantom_is_checked_whole():
    phantom = _phantom()
    voxels = dynamic_timestep_error_voxels(phantom, max_voxels=6, **OPTIONS)
    np.testing.assert_array_equal(voxels, np.flatnonzero(phantom.mask.ravel()))


def test_voxel_selection_keeps_the_extremes_of_every_map():
    phantom = _phantom(shape=(6, 6, 6), fov=(0.03, 0.03, 0.03), seed=7)
    voxels = dynamic_timestep_error_voxels(phantom, max_voxels=40, **OPTIONS)

    active = np.flatnonzero(phantom.mask.ravel())
    assert voxels.size == 40
    assert np.all(np.diff(voxels) > 0)
    assert np.all(np.isin(voxels, active))
    frequencies = timestep_error_frequencies(phantom, **OPTIONS)
    maps = [frequencies[index] for index in range(len(phantom.pools))]
    maps += [
        np.abs(phantom.tx_sensitivity_map),
        phantom.kpl_map_s_inv,
        phantom.pyruvate_inflow.delivery_map,
        phantom.voxel_washout.loss_map_s_inv,
    ]
    for values in maps:
        values = np.asarray(values).ravel()[active]
        for extreme in (np.argmin(values), np.argmax(values)):
            assert active[extreme] in voxels
    # The washout source compartment is checked.
    assert np.flatnonzero(phantom.voxel_washout.source_mask.ravel())[0] in voxels


def test_skipped_statuses_name_the_dynamic_source():
    program = _slice_train(1)
    phantom = _phantom()
    native = estimate_dynamic_timestep_error(program, phantom, None, **OPTIONS)
    identical = estimate_dynamic_timestep_error(program, phantom, 1e-6, **OPTIONS)
    coarse = estimate_dynamic_timestep_error(
        program, phantom, 20e-6, reference_timestep_s=40e-6, **OPTIONS
    )

    assert native.status == "native_raster"
    assert identical.status == "identical_discretization"
    assert identical.max_vector_error == 0.0
    assert coarse.status == "finer_than_reference"
    for estimate in (native, identical, coarse):
        metadata = json.loads(json.dumps(estimate.to_metadata(), allow_nan=False))
        assert metadata["source"] == "dynamic_kernel"
        assert not estimate.exceeds(0.0)


def _true_error(simulator, program, phantom, step_s, voxels):
    """Per-pool relative error at ``step_s`` against 1 us, as the check measures."""

    checked, _ = _timestep_error_program(program, 1e-6, 10**9)
    checkpoints = np.union1d(
        default_probe_checkpoints(checked, max_rf_checkpoints=256), (0.0,)
    )
    candidate, gold = (
        np.asarray(
            simulator.simulate_dynamic_sequence(
                checked,
                phantom,
                simulation_timestep_s=timestep_s,
                checkpoints_s=checkpoints,
                _voxel_indices=voxels,
                **OPTIONS,
            ).checkpoint_pool_magnetization
        )
        for timestep_s in (step_s, 1e-6)
    )
    peaks = np.linalg.norm(gold, axis=-1).max(axis=0)
    scales = np.maximum(
        peaks, DYNAMIC_TIMESTEP_ERROR_PEAK_FLOOR * peaks.max(axis=1, keepdims=True)
    )
    return float((np.linalg.norm(candidate - gold, axis=-1) / scales).max())


@pytest.mark.parametrize("kernel", GUI_DYNAMIC_KERNEL_NAMES)
def test_estimate_tracks_the_error_against_1_us(kernel):
    program = _slice_train()
    phantom = _phantom()
    simulator = _simulator(kernel)
    with warnings.catch_warnings():
        warnings.simplefilter("error", KernelFallbackWarning)
        estimate = estimate_dynamic_timestep_error(
            program, phantom, 20e-6, simulator=simulator, **OPTIONS
        )
        voxels = np.flatnonzero(phantom.mask.ravel())
        true_error = _true_error(simulator, program, phantom, 20e-6, voxels)

    assert estimate.status == "checked"
    assert estimate.source == "dynamic_kernel"
    assert estimate.reference_timestep_s == pytest.approx(5e-6)
    assert estimate.probe_count == voxels.size
    assert estimate.checked_duration_s == pytest.approx(program.duration_s)
    # Second order: the quarter-step reference sees 1 - 1/16 of the error.
    assert estimate.max_vector_error == pytest.approx(
        (1.0 - 1.0 / 16.0) * true_error, rel=0.05
    )
    probe = estimate.to_metadata()["worst_probe"]
    assert probe["pool"] in {pool.name for pool in phantom.pools}
    assert probe["voxel_index"] in voxels
    assert probe["peak_magnetization"] > 0
    assert 0.0 <= estimate.worst_time_s <= program.duration_s
    json.dumps(estimate.to_metadata(), allow_nan=False)


def test_budget_truncates_the_dynamic_check():
    program = _slice_train()
    estimate = estimate_dynamic_timestep_error(
        program, _phantom(), 20e-6, max_reference_intervals=500, **OPTIONS
    )

    # One 1 ms pulse needs 200 quarter-step intervals: two fit, the third not.
    assert estimate.status == "truncated"
    assert estimate.checked_duration_s == pytest.approx(8e-3)
    assert 0.0 < estimate.worst_time_s <= estimate.checked_duration_s
    assert estimate.max_vector_error > 0


def _run_worker(program, phantom, kernel="optimized"):
    """Run one GUI simulation of a dynamic phantom at the 20 us GUI step."""

    outputs, errors = [], []
    worker = SequenceSimulationThread(
        BlochSimulator(use_parallel=False, dynamic_sequence_kernel=kernel),
        program,
        phantom,
        (),
        live_preview=False,
        simulation_timestep_s=20e-6,
        **OPTIONS,
    )
    worker.result_ready.connect(outputs.append)
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors
    return outputs[0]


@pytest.mark.parametrize("kernel", GUI_DYNAMIC_KERNEL_NAMES)
def test_gui_checks_dynamic_phantoms_in_the_dynamic_kernel(monkeypatch, kernel):
    seen = []
    original = widget_module.estimate_dynamic_timestep_error

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    def static_check(*args, **kwargs):
        raise AssertionError("the static probe check must not run")

    monkeypatch.setattr(widget_module, "estimate_dynamic_timestep_error", spy)
    monkeypatch.setattr(widget_module, "estimate_timestep_error", static_check)
    program = _slice_train()
    phantom = _phantom()

    check = _run_worker(program, phantom, kernel).metadata["timestep_error"]

    assert check == json.loads(json.dumps(check, allow_nan=False))
    assert check["source"] == "dynamic_kernel"
    assert check["status"] == "checked"
    assert len(seen) == 1
    assert seen[0]["simulator"].dynamic_sequence_kernel == kernel
    # 800 quarter-step intervals fit the minimum budget of a short run.
    assert seen[0]["max_reference_intervals"] >= TIMESTEP_ERROR_MIN_INTERVALS
    direct = estimate_dynamic_timestep_error(
        program, phantom, 20e-6, simulator=_simulator(kernel), **OPTIONS
    )
    assert check["max_vector_error"] == pytest.approx(direct.max_vector_error)
    pool = check["worst_probe"]["pool"]
    assert pool in {pool.name for pool in phantom.pools}

    note = SequenceSimulationWidget._timestep_error_note(check, threshold=0.0)
    percent = f"{100 * check['max_vector_error']:.3g}%"
    assert (
        f"RF time step 20 µs: pool magnetization differs by up to {percent} "
        "of the pool's peak from a 5 µs run" in note
    )
    assert f"({pool} at " in note
    monkeypatch.setattr(widget_module, "TIMESTEP_ERROR_NOTE_THRESHOLD", 0.0)
    widget = SequenceSimulationWidget()
    widget._show_timestep_check(check)
    text = widget.timestep_check_label.text()
    assert "pool magnetization within" in text
    assert "of each pool's peak of a 5 µs run" in text
    assert "consider a smaller step" in text
    assert tone_of(widget.timestep_check_label) == "warning"


def test_gui_dynamic_check_budget_and_truncation(monkeypatch):
    budget = SequenceSimulationThread._timestep_error_budget
    static = budget(5.0)
    assert static == int(
        5.0 * TIMESTEP_ERROR_RUNTIME_FRACTION / TIMESTEP_ERROR_SECONDS_PER_INTERVAL
    )
    dynamic = budget(
        5.0,
        TIMESTEP_ERROR_DYNAMIC_SECONDS_PER_INTERVAL,
        TIMESTEP_ERROR_DYNAMIC_RUNTIME_FRACTION,
    )
    assert dynamic == int(
        5.0
        * TIMESTEP_ERROR_DYNAMIC_RUNTIME_FRACTION
        / TIMESTEP_ERROR_DYNAMIC_SECONDS_PER_INTERVAL
    )
    # The dynamic error builds up over long pulse trains, so its check gets a
    # larger share of the run than the probe check.
    assert TIMESTEP_ERROR_DYNAMIC_RUNTIME_FRACTION > TIMESTEP_ERROR_RUNTIME_FRACTION

    # The worker uses the dynamic share, not the static one.
    monkeypatch.setattr(widget_module, "TIMESTEP_ERROR_RUNTIME_FRACTION", 0.0)
    monkeypatch.setattr(widget_module, "TIMESTEP_ERROR_DYNAMIC_RUNTIME_FRACTION", 1e6)
    program = _slice_train(8)
    check = _run_worker(program, _phantom()).metadata["timestep_error"]
    assert check["status"] == "checked"
    assert check["checked_duration_s"] == pytest.approx(program.duration_s)

    # However long the main run takes here, the check gets the minimum budget.
    monkeypatch.setattr(widget_module, "TIMESTEP_ERROR_DYNAMIC_RUNTIME_FRACTION", 0.0)
    check = _run_worker(program, _phantom()).metadata["timestep_error"]

    # 8 pulses need 1600 quarter-step intervals; the first 1000 are checked.
    assert check["source"] == "dynamic_kernel"
    assert check["status"] == "truncated"
    assert 0.0 < check["checked_duration_s"] < program.duration_s
    note = SequenceSimulationWidget._timestep_error_note(check, threshold=0.0)
    assert f"first {check['checked_duration_s']:.4g} s checked" in note


def test_gui_reports_a_failed_dynamic_check(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("dynamic check broke")

    monkeypatch.setattr(widget_module, "estimate_dynamic_timestep_error", broken)
    result = _run_worker(_slice_train(1), _phantom())

    assert result.metadata["timestep_error"] == {
        "status": "failed",
        "source": "dynamic_kernel",
        "error": "RuntimeError: dynamic check broke",
    }
    message = SequenceSimulationWidget._execution_completion_message(result.metadata)
    assert "RF time-step check failed: RuntimeError: dynamic check broke" in message
