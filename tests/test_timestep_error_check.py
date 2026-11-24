"""The RF time-step check estimates its error from a few probe spins."""

import json

import numpy as np
import pytest

import blochsimulator.sequence as seq
from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import (
    GradientEvent,
    RFEvent,
    SequenceProgram,
    estimate_timestep_error,
    load_pulseq,
    timestep_error_inputs,
)
from blochsimulator.sequence.compiler import IDEAL_SPOILER_DEFINITION
from blochsimulator.sequence.convergence import _timestep_error_program
from blochsimulator.spectral_phantom import ChemicalSpecies, SpectralPhantom
from blochsimulator.ui.default_settings import TIMESTEP_ERROR_NOTE_THRESHOLD
from blochsimulator.ui.sequence_simulation_widget import (
    SequenceSimulationThread,
    SequenceSimulationWidget,
)
from blochsimulator.ui.theme import tone_of
from blochsimulator.units import ppm_to_hz

RF_RASTER_S = 1e-6


def _block_program(gradient=False):
    """A 100 µs block pulse, optionally under a constant z gradient."""

    events = [RFEvent(0.0, np.full(100, 500.0), RF_RASTER_S)]
    if gradient:
        events.append(GradientEvent("z", 0.0, np.full(10, 4.0e4), 10e-6))
    return SequenceProgram(tuple(events), duration_s=2e-4)


def _rf_train_program(count=6):
    """Sinc pulses under a z gradient, each followed by a declared spoiler."""

    samples = 500.0 * np.sinc(np.linspace(-3.0, 3.0, 100))
    events = []
    spoilers = []
    for index in range(count):
        start = index * 1e-3
        events.append(RFEvent(start, samples, RF_RASTER_S))
        events.append(GradientEvent("z", start, np.full(10, 4.0e4), 10e-6))
        spoilers.append(start + 5e-4)
    return SequenceProgram(
        tuple(events),
        duration_s=count * 1e-3,
        metadata={
            "definitions": {
                IDEAL_SPOILER_DEFINITION: spoilers,
                "MagnetizationCheckpointTimes": [time + 1e-4 for time in spoilers],
            }
        },
    )


@pytest.fixture(scope="module")
def spsp_program(tmp_path_factory):
    pulse = seq.design_spectral_spatial_pulse(
        bands=(
            seq.SpectralBand(0.0, 100.0, name="Ref"),
            seq.SpectralBand(150.0, 60.0, amplitude=0.4, name="P2"),
        ),
        flip_angle_deg=30.0,
        slice_thickness_m=30e-3,
        spatial_time_bandwidth_product=2.0,
        subpulse_duration_s=0.2e-3,
        subpulse_count=3,
        nucleus="H1",
        max_gradient_t_per_m=0.032,
        max_slew_t_per_m_s=130.0,
        name="SPSP",
    ).to_dict()
    path = tmp_path_factory.mktemp("spsp") / "spsp.seq"
    seq.make_pulseq_slice_selection(
        rf_pulse_type="spectral_spatial",
        rf_spectral_spatial_pulse=pulse,
        slice_thickness_m=30e-3,
        nucleus="H1",
        flip_angle_deg=30.0,
    ).write(str(path), v141_compat=True)
    return load_pulseq(path)


def _slab_phantom():
    """Five voxels across 4 cm along the slice axis."""

    shape = (1, 1, 5)
    return Phantom(shape, (0.01, 0.01, 0.04), np.ones(shape), np.full(shape, 0.1))


LEGACY = BlochSimulator(use_parallel=False, sequence_kernel="legacy")


def test_block_pulse_has_no_time_step_error():
    estimate = estimate_timestep_error(
        _block_program(), 20e-6, frequency_offsets_hz=(0.0, 150.0)
    )
    legacy = estimate_timestep_error(
        _block_program(),
        20e-6,
        frequency_offsets_hz=(0.0, 150.0),
        simulator=LEGACY,
    )

    assert estimate.status == "checked"
    assert estimate.reference_timestep_s == pytest.approx(5e-6)
    # Production splits free precession around each merged RF interval; the
    # legacy kernel rotates a constant RF sample exactly.
    assert estimate.max_vector_error < 1e-5
    assert legacy.max_vector_error < 1e-9
    assert not estimate.exceeds(TIMESTEP_ERROR_NOTE_THRESHOLD)
    assert SequenceSimulationWidget._timestep_error_note(estimate.to_metadata()) == ""


def test_spsp_pulse_at_20_us_exceeds_one_percent(spsp_program):
    estimate = estimate_timestep_error(
        spsp_program,
        20e-6,
        simulator=LEGACY,
        **timestep_error_inputs(_slab_phantom()),
    )

    assert estimate.status == "checked"
    # Legacy kernel: measured 1.55 % (20 µs against 5 µs) at the slab edge.
    assert estimate.max_vector_error > TIMESTEP_ERROR_NOTE_THRESHOLD
    assert estimate.probe_count == 9
    assert abs(estimate.worst_probe["position_m"][2]) > 0.0
    note = SequenceSimulationWidget._timestep_error_note(estimate.to_metadata())
    assert note.startswith("RF time step 20 µs: probe magnetization differs")
    assert f"{100 * estimate.max_vector_error:.3g}%" in note
    assert "5 µs run" in note


def test_spsp_error_shrinks_with_the_step(spsp_program):
    inputs = timestep_error_inputs(_slab_phantom())
    errors = [
        estimate_timestep_error(
            spsp_program, step, simulator=LEGACY, **inputs
        ).max_vector_error
        for step in (20e-6, 10e-6)
    ]
    production = [
        estimate_timestep_error(spsp_program, step, **inputs).max_vector_error
        for step in (20e-6, 10e-6)
    ]

    # Second order in the RF step: halving it divides the error by about four.
    assert errors[1] < errors[0] / 3
    # Production (measured 0.78 % and 0.32 %) is below legacy and shrinks too.
    assert production[0] < errors[0]
    assert production[1] < production[0] / 2


def test_probe_lines_follow_rf_active_gradients():
    inputs = timestep_error_inputs(_slab_phantom())

    without = estimate_timestep_error(_block_program(), 20e-6, **inputs)
    with_gradient = estimate_timestep_error(
        _block_program(gradient=True), 20e-6, line_samples=5, **inputs
    )

    assert without.probe_count == 1
    assert with_gradient.probe_count == 5


def test_budget_truncates_at_an_rf_boundary_and_trims_definitions():
    program = _rf_train_program()

    estimate = estimate_timestep_error(
        program, 20e-6, max_reference_intervals=50, line_samples=3
    )
    checked, truncated = _timestep_error_program(program, 5e-6, 50)

    # Each pulse needs 20 reference intervals; the third would exceed 50.
    assert truncated
    assert estimate.status == "truncated"
    assert estimate.reference_interval_count >= 40
    # The checked part ends where the first dropped pulse would start.
    assert estimate.checked_duration_s == pytest.approx(2e-3)
    assert checked.duration_s == pytest.approx(2e-3)
    assert len(checked.rf_events) == 2
    definitions = checked.metadata["definitions"]
    assert definitions[IDEAL_SPOILER_DEFINITION] == pytest.approx([5e-4, 1.5e-3])
    assert definitions["MagnetizationCheckpointTimes"] == pytest.approx([6e-4, 1.6e-3])
    note = SequenceSimulationWidget._timestep_error_note(
        {**estimate.to_metadata(), "max_vector_error": 0.5}
    )
    assert f"first {checked.duration_s:.4g} s checked" in note


def test_untruncated_program_keeps_everything_but_adc():
    program = _rf_train_program()

    checked, truncated = _timestep_error_program(program, 5e-6, 10**6)

    assert not truncated
    assert checked.duration_s == program.duration_s
    assert checked.metadata["definitions"] == program.metadata["definitions"]


def test_skipped_statuses_are_json_safe():
    native = estimate_timestep_error(_block_program(), None)
    # Without RF both steps compile alike; with RF a quarter step now always
    # subdivides the samples.
    no_rf = SequenceProgram(
        (GradientEvent("z", 0.0, np.full(10, 4.0e4), 10e-6),), duration_s=2e-4
    )
    identical = estimate_timestep_error(no_rf, 1e-6)
    coarse = estimate_timestep_error(
        _block_program(), 20e-6, reference_timestep_s=40e-6
    )

    assert native.status == "native_raster"
    assert identical.status == "identical_discretization"
    assert identical.max_vector_error == 0.0
    assert coarse.status == "finer_than_reference"
    for estimate in (native, identical, coarse):
        metadata = json.loads(json.dumps(estimate.to_metadata(), allow_nan=False))
        assert metadata["status"] == estimate.status
        assert not estimate.exceeds(0.0)
    assert native.to_metadata()["max_vector_error"] is None


def test_checked_metadata_is_json_safe(spsp_program):
    metadata = estimate_timestep_error(spsp_program, 20e-6).to_metadata()

    restored = json.loads(json.dumps(metadata, allow_nan=False))
    assert "b1_scale" not in restored["worst_probe"]
    assert restored["max_vector_error"] == pytest.approx(metadata["max_vector_error"])


def test_inputs_take_median_then_extremes_of_static_phantom():
    shape = (4, 1, 1)
    phantom = Phantom(
        shape,
        (0.04, 0.01, 0.01),
        np.array([1.0, 2.0, 3.0, 4.0]).reshape(shape),
        np.array([0.1, 0.2, 0.3, 0.4]).reshape(shape),
        b0_map=np.array([-30.0, 0.0, 0.3, 50.0]).reshape(shape),
    )

    inputs = timestep_error_inputs(phantom)

    # 0.3 Hz is within 0.5 Hz of the median and is skipped.
    np.testing.assert_allclose(inputs["frequency_offsets_hz"], [0.0, -30.0, 50.0])
    np.testing.assert_allclose(inputs["t1_s"], [2.0, 1.0, 4.0])
    np.testing.assert_allclose(inputs["t2_s"], [0.2, 0.1, 0.4])
    assert inputs["positions_m"].shape == (4, 3)


def test_inputs_cover_every_spectral_component_and_dynamic_pool():
    shape = (1, 1, 2)
    spectral = SpectralPhantom(
        shape=shape,
        fov=(0.01, 0.01, 0.02),
        species=[
            ChemicalSpecies("a", 0.0, 1.0, 0.1),
            ChemicalSpecies("b", 2.0, 2.0, 0.2),
        ],
        concentration_maps={"a": np.ones(shape), "b": np.ones(shape)},
        b0_map=np.array([-5.0, 5.0]).reshape(shape),
        field_strength=3.0,
        nucleus="H1",
    )
    dynamic = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.01, 0.01, 0.02),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 0.5),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.zeros(shape),
        b0_map=np.array([-5.0, 5.0]).reshape(shape),
        field_strength=7.0,
        nucleus="C13",
    )

    spectral_inputs = timestep_error_inputs(spectral)
    dynamic_inputs = timestep_error_inputs(dynamic)

    b_hz = spectral.get_frequency_offset("b", 3.0, "H1")
    # Medians of both groups first (lower middle of two voxels), then extremes.
    np.testing.assert_allclose(
        spectral_inputs["frequency_offsets_hz"],
        [-5.0, b_hz - 5.0, 5.0, b_hz + 5.0],
    )
    np.testing.assert_allclose(spectral_inputs["t1_s"], [1.0, 2.0, 1.0, 2.0])
    lactate_hz = dynamic.get_frequency_offset("Lactate", 7.0, "C13")
    np.testing.assert_allclose(
        dynamic_inputs["frequency_offsets_hz"],
        [-5.0, lactate_hz - 5.0, 5.0, lactate_hz + 5.0],
    )
    np.testing.assert_allclose(dynamic_inputs["t2_s"], [1.0, 0.5, 1.0, 0.5])
    assert dynamic_inputs["positions_m"].shape == (2, 3)

    capped = timestep_error_inputs(dynamic, max_frequencies=3)
    np.testing.assert_allclose(
        capped["frequency_offsets_hz"], [-5.0, lactate_hz - 5.0, 5.0]
    )


def test_dynamic_inputs_follow_the_sequence_reference():
    shape = (1, 1, 1)
    dynamic = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.01,) * 3,
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 0.5),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.zeros(shape),
        field_strength=7.0,
        nucleus="C13",
        spectral_reference_ppm=171.0,
    )

    inputs = timestep_error_inputs(
        dynamic, field_strength_t=7.0, nucleus="C13", sequence_reference_ppm=170.0
    )

    shift_hz = ppm_to_hz(1.0, 7.0, "C13")
    np.testing.assert_allclose(
        inputs["frequency_offsets_hz"],
        [shift_hz, dynamic.get_frequency_offset("Lactate", 7.0, "C13") + shift_hz],
    )


def _run_worker(program, phantom, step_s, kernel="production"):
    outputs, errors, stages = [], [], []
    worker = SequenceSimulationThread(
        BlochSimulator(use_parallel=False, sequence_kernel=kernel),
        program,
        phantom,
        (),
        live_preview=False,
        nucleus="H1",
        field_strength_t=3.0,
        simulation_timestep_s=step_s,
    )
    worker.result_ready.connect(outputs.append)
    worker.failed.connect(errors.append)
    worker.stage.connect(stages.append)
    worker.run()
    assert not errors
    return outputs[0], stages


def test_worker_records_and_widget_reports_spsp_error(spsp_program):
    result, stages = _run_worker(spsp_program, _slab_phantom(), 20e-6, "legacy")

    check = result.metadata["timestep_error"]
    assert check["status"] == "checked"
    assert check["max_vector_error"] > TIMESTEP_ERROR_NOTE_THRESHOLD
    assert "Checking the RF time step against a quarter step…" in stages
    percent = f"{100 * check['max_vector_error']:.3g}%"
    message = SequenceSimulationWidget._execution_completion_message(result.metadata)
    assert f"RF time step 20 µs: probe magnetization differs by up to {percent}" in (
        message
    )

    widget = SequenceSimulationWidget()
    widget._show_timestep_check(check)
    assert not widget.timestep_check_label.isHidden()
    assert f"{100 * check['max_vector_error']:.2g}%" in (
        widget.timestep_check_label.text()
    )
    assert "consider a smaller step" in widget.timestep_check_label.text()
    assert tone_of(widget.timestep_check_label) == "warning"

    widget.simulation_timestep_us.setValue(5.0)
    assert widget.timestep_check_label.isHidden()


def test_worker_block_pulse_adds_no_note():
    phantom = _slab_phantom()
    result, _ = _run_worker(_block_program(gradient=True), phantom, 20e-6)

    check = result.metadata["timestep_error"]
    assert check["status"] == "checked"
    # The splitting error of a constant pulse under a gradient: 2.3e-5 measured.
    assert check["max_vector_error"] < 1e-3
    message = SequenceSimulationWidget._execution_completion_message(result.metadata)
    assert "RF time step" not in message

    widget = SequenceSimulationWidget()
    widget._show_timestep_check(check)
    assert "< 0.01%" in widget.timestep_check_label.text()
    assert tone_of(widget.timestep_check_label) == "muted"


def test_failed_check_keeps_the_result_and_is_reported(monkeypatch):
    import blochsimulator.ui.sequence_simulation_widget as module

    def broken(*args, **kwargs):
        raise RuntimeError("probe run broke")

    monkeypatch.setattr(module, "estimate_timestep_error", broken)
    result, _ = _run_worker(_block_program(), _slab_phantom(), 20e-6)

    assert result.metadata["timestep_error"] == {
        "status": "failed",
        "source": "static_probes",
        "error": "RuntimeError: probe run broke",
    }
    message = SequenceSimulationWidget._execution_completion_message(result.metadata)
    assert "RF time-step check failed: RuntimeError: probe run broke" in message


def test_check_budget_follows_the_main_runtime(monkeypatch):
    import blochsimulator.ui.sequence_simulation_widget as module
    from blochsimulator.sequence.convergence import (
        DEFAULT_TIMESTEP_ERROR_INTERVAL_BUDGET,
    )
    from blochsimulator.ui.default_settings import (
        TIMESTEP_ERROR_MIN_INTERVALS,
        TIMESTEP_ERROR_RUNTIME_FRACTION,
        TIMESTEP_ERROR_SECONDS_PER_INTERVAL,
    )

    budget = SequenceSimulationThread._timestep_error_budget
    assert budget(0.0) == TIMESTEP_ERROR_MIN_INTERVALS
    assert budget(5.0) == int(
        5.0 * TIMESTEP_ERROR_RUNTIME_FRACTION / TIMESTEP_ERROR_SECONDS_PER_INTERVAL
    )
    assert budget(1e4) == DEFAULT_TIMESTEP_ERROR_INTERVAL_BUDGET

    seen = []
    original = module.estimate_timestep_error

    def spy(*args, **kwargs):
        seen.append(kwargs["max_reference_intervals"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "estimate_timestep_error", spy)
    # However long the main run takes here, the check gets the minimum budget.
    monkeypatch.setattr(module, "TIMESTEP_ERROR_RUNTIME_FRACTION", 0.0)
    # 60 pulses of 20 quarter-step intervals exceed the 1000-interval minimum.
    result, _ = _run_worker(_rf_train_program(60), _slab_phantom(), 20e-6)

    assert seen == [TIMESTEP_ERROR_MIN_INTERVALS]
    check = result.metadata["timestep_error"]
    assert check["status"] == "truncated"
    assert 0.0 < check["checked_duration_s"] < 60e-3
    # Declared checkpoints join the RF-end observations; the time stays valid.
    assert 0.0 < check["worst_time_s"] <= check["checked_duration_s"]
