"""Validate excitation volumes independently of any ADC acquisition."""

import numpy as np
import nbformat
import pytest

pytest.importorskip("pypulseq")
pytest.importorskip("PyQt5")

from blochsimulator import BlochSimulator
from blochsimulator.notebook_exporter import export_pulseq_generation_notebook
from blochsimulator.phantom import Phantom
from blochsimulator.project_io import load_project, save_project
from blochsimulator.spectral_phantom import ChemicalSpecies, SpectralPhantom
from blochsimulator.sequence import (
    ScannerParameters,
    SequenceCompiler,
    load_pulseq,
    make_pulseq_slice_selection,
)
from blochsimulator.ui.main_window import _capture_widget_state, _restore_widget_state
from blochsimulator.ui.sequence_simulation_widget import (
    SequenceSimulationThread,
    SequenceSimulationWidget,
)
from blochsimulator.units import NUCLEUS_GAMMA_HZ_PER_T


def _program(tmp_path, **parameters):
    sequence = make_pulseq_slice_selection(**parameters)
    assert sequence.check_timing()[0]
    path = tmp_path / "excitation.seq"
    sequence.write(str(path), v141_compat=True)
    return load_pulseq(path)


def _phantom(shape=(3, 3, 101), fov=(0.012, 0.012, 0.020), **kwargs):
    return Phantom(shape, fov, np.full(shape, 1e6), np.full(shape, 1e6), **kwargs)


@pytest.mark.parametrize("pulse_type", ["sinc", "slr", "gaussian", "block", "designer"])
def test_shared_rf_shapes_preserve_nominal_flip_and_phase(tmp_path, pulse_type):
    parameters = {}
    if pulse_type == "designer":
        parameters.update(
            rf_custom_waveform_hz=np.full(5000, 30 / (360 * 5e-3) * np.exp(0.3j)),
            rf_custom_raster_s=1e-6,
            rf_custom_flip_angle_deg=30,
        )
    program = _program(
        tmp_path,
        rf_pulse_type=pulse_type,
        flip_angle_deg=30,
        rf_phase_deg=45,
        **parameters,
    )
    result = BlochSimulator(use_parallel=False).simulate_sequence(
        program, _phantom(shape=(1, 1, 1)), simulation_timestep_s=2e-6
    )
    magnetization = result.final_magnetization[0, 0, 0]
    assert np.hypot(*magnetization[:2]) == pytest.approx(0.5, abs=3e-6)
    assert magnetization[2] == pytest.approx(np.cos(np.pi / 6), abs=3e-6)
    # The Bloch kernel excites +My with real RF and +Mx with imaginary RF.
    rf_phase = np.pi / 4 + (0.3 if pulse_type == "designer" else 0.0)
    expected_phase = np.pi / 2 - rf_phase
    assert np.angle(magnetization[0] + 1j * magnetization[1]) == pytest.approx(
        expected_phase, abs=3e-6
    )


@pytest.mark.parametrize("rephase", [False, True])
@pytest.mark.parametrize("axis", ["+x", "-y", "+z"])
def test_selection_round_trip_has_no_adc_spoiler_or_tail(tmp_path, rephase, axis):
    program = _program(
        tmp_path,
        rephase=rephase,
        slice_axis=axis,
        scanner_parameters=ScannerParameters(),
    )
    compiled = SequenceCompiler().compile(program)
    assert len(program.rf_events) == 1
    assert len(program.gradient_events) == 1 + int(rephase)
    assert {e.axis for e in program.gradient_events} == {axis[-1]}
    assert program.duration_s == pytest.approx(max(e.end_s for e in program.events))
    assert not program.adc_events
    assert compiled.adc_times_s.size == 0
    assert compiled.transverse_crush_times_s.size == 0
    definitions = program.metadata["definitions"]
    assert definitions["SliceGradientAxis"] == axis
    if rephase:
        assert compiled.checkpoint_times_s == pytest.approx(
            [definitions["SliceSelectionEndTime"]]
        )
        assert definitions["RephasingEndTime"] == pytest.approx(program.duration_s)
    else:
        assert compiled.checkpoint_times_s.size == 0
        assert definitions["SliceSelectionEndTime"] == pytest.approx(program.duration_s)


def test_3d_profile_uses_local_tx_and_keeps_receiver_out_of_magnetization(tmp_path):
    program = _program(tmp_path, flip_angle_deg=30.0)
    shape = (3, 3, 101)
    tx = np.broadcast_to(np.array([0.5, 1.0, 1.5])[:, None, None], shape).copy()
    phantom = _phantom(tx_sensitivity_map=tx)
    simulator = BlochSimulator(use_parallel=False)
    result = simulator.simulate_sequence(program, phantom, simulation_timestep_s=2e-6)
    assert result.final_magnetization.shape == shape + (3,)
    assert result.signal.size == 0
    expected_angles = np.deg2rad([15.0, 30.0, 45.0])
    mxy = np.hypot(result.mx, result.my)
    np.testing.assert_allclose(mxy[:, 1, 50], np.sin(expected_angles), atol=3e-6)
    np.testing.assert_allclose(result.mz[:, 1, 50], np.cos(expected_angles), atol=3e-6)
    assert mxy[:, :, :20].max() < 2e-4
    np.testing.assert_allclose(mxy[:, 0, :], mxy[:, 2, :], atol=1e-12)
    changed_receiver = _phantom(
        tx_sensitivity_map=tx, rx_sensitivity_maps=np.full((1, *shape), 100.0j)
    )
    rx_result = simulator.simulate_sequence(
        program, changed_receiver, simulation_timestep_s=2e-6
    )
    np.testing.assert_array_equal(
        result.final_magnetization, rx_result.final_magnetization
    )


def test_rephasing_retains_original_state_and_refocuses_slice_phase(tmp_path):
    simulator = BlochSimulator(use_parallel=False)
    phantom = _phantom()
    before = simulator.simulate_sequence(
        _program(tmp_path, flip_angle_deg=30.0), phantom, simulation_timestep_s=2e-6
    )
    after = simulator.simulate_sequence(
        _program(tmp_path, flip_angle_deg=30.0, rephase=True),
        phantom,
        simulation_timestep_s=2e-6,
    )
    np.testing.assert_array_equal(
        after.checkpoint_magnetization[0], before.final_magnetization
    )
    a = before.mx[1, 1] + 1j * before.my[1, 1]
    b = after.mx[1, 1] + 1j * after.my[1, 1]
    selected = np.abs(a) > 0.35
    assert abs(b[selected].mean()) > 5 * abs(a[selected].mean())
    assert np.std(np.unwrap(np.angle(b[selected]))) < 0.03
    np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-8)


@pytest.mark.parametrize("axis,sign", [("+x", 1), ("-x", -1)])
def test_signed_axis_and_slice_offset_place_excitation_correctly(tmp_path, axis, sign):
    program = _program(
        tmp_path, slice_axis=axis, slice_offset_m=0.003, flip_angle_deg=30.0
    )
    phantom = _phantom(shape=(101, 1, 1), fov=(0.020, 0.012, 0.012))
    result = BlochSimulator(use_parallel=False).simulate_sequence(
        program, phantom, simulation_timestep_s=2e-6
    )
    x = ((np.arange(101) + 0.5) / 101 - 0.5) * 0.020
    profile = np.hypot(result.mx[:, 0, 0], result.my[:, 0, 0])
    center = np.sum(x * profile**2) / np.sum(profile**2)
    assert center == pytest.approx(sign * 0.003, abs=1e-5)


def test_gui_worker_volume_and_project_round_trip(tmp_path):
    widget = SequenceSimulationWidget()
    # Appending the source preserves indices in existing project files.
    assert widget.sequence_source.itemText(9) == "Pulseq .seq file"
    widget.sequence_source.setCurrentIndex(widget.SLICE_SELECTION_SOURCE)
    widget.slice_selection_axis.setCurrentText("-Y")
    widget.slice_selection_rf_phase_deg.setValue(37.0)
    widget.slice_selection_rephase.setChecked(True)
    assert widget.slice_selection_rephase_duration_ms.isEnabled()
    assert widget._ensure_current_generated_sequence(), widget._generation_error
    phantom = _phantom(shape=(5, 5, 5), fov=(0.008, 0.008, 0.008))
    widget.phantom = phantom
    worker = SequenceSimulationThread(
        widget.simulator,
        widget.program,
        phantom,
        widget._checkpoint_seconds(),
        "voxel",
        7.0,
        "C13",
        False,
        65536,
        simulation_timestep_s=10e-6,
    )
    results, failures = [], []
    worker.result_ready.connect(results.append)
    worker.failed.connect(failures.append)
    worker.run()
    assert not failures
    assert len(results) == 1
    result = results[0]
    widget._finished(result, record_run=False)
    assert widget.views.currentWidget() is widget.result_volume_viewer
    viewer = widget.result_volume_viewer
    assert viewer.map_combo.currentText() == "|Mxy|"
    assert "After slice rephasing" in viewer.state_combo.itemText(0)
    assert "After slice selection" in viewer.state_combo.itemText(1)
    viewer.state_combo.setCurrentIndex(1)
    np.testing.assert_allclose(
        viewer.volume.data,
        np.hypot(
            result.checkpoint_magnetization[0, ..., 0],
            result.checkpoint_magnetization[0, ..., 1],
        ),
    )
    viewer.map_combo.setCurrentText("Mz")
    np.testing.assert_array_equal(
        viewer.volume.data, result.checkpoint_magnetization[0, ..., 2]
    )
    path = tmp_path / "excitation.blochproj"
    save_project(
        path,
        {"sequence_controls": _capture_widget_state(widget)},
        phantom=phantom,
        program=widget.program,
        sequence_result=result,
    )
    loaded = load_project(path)
    np.testing.assert_array_equal(
        loaded["sequence_result"].final_magnetization, result.final_magnetization
    )
    np.testing.assert_array_equal(
        loaded["sequence_result"].checkpoint_magnetization,
        result.checkpoint_magnetization,
    )
    restored = SequenceSimulationWidget()
    _restore_widget_state(restored, loaded["state"]["sequence_controls"])
    assert restored.sequence_source.currentIndex() == restored.SLICE_SELECTION_SOURCE
    assert restored.slice_selection_axis.currentText() == "-Y"
    assert restored.slice_selection_rephase.isChecked()
    restored.program, restored.phantom = loaded["program"], loaded["phantom"]
    restored._finished(loaded["sequence_result"], record_run=False)
    assert (
        "After slice selection" in restored.result_volume_viewer.state_combo.itemText(1)
    )


def test_c13_physical_waveform_units_and_notebook_export(tmp_path):
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.SLICE_SELECTION_SOURCE)
    widget.nucleus.setCurrentText("C13")
    assert widget._ensure_current_generated_sequence(), widget._generation_error
    kind, parameters, filename = widget._pulseq_export_spec()
    notebook = export_pulseq_generation_notebook(
        tmp_path / "excitation.ipynb", kind, parameters, seq_filename=filename
    )
    cells = nbformat.read(notebook, as_version=4).cells
    assert "make_pulseq_slice_selection" in cells[1].source
    result = widget.simulator.simulate_sequence(widget.program, _phantom())
    assert result.metadata["physical_waveform_nucleus"] == "C13"
    np.testing.assert_allclose(
        result.sequence_waveforms["gradient_t_per_m"],
        result.sequence_waveforms["gradient_hz_per_m"] / NUCLEUS_GAMMA_HZ_PER_T["C13"],
    )


def test_spectral_phantom_keeps_pool_excitation_and_automatic_checkpoints(tmp_path):
    program = _program(tmp_path, rephase=True, nucleus="C13", flip_angle_deg=30)
    shape = (3, 3, 5)
    phantom = SpectralPhantom(
        shape=shape,
        fov=(0.008, 0.008, 0.010),
        species=[ChemicalSpecies("Water", 0.0, t1=0.8, t2=0.09, t2_star=0.09)],
        concentration_maps={"Water": np.ones(shape)},
        nucleus="C13",
        field_strength=7.0,
        spectral_reference_ppm=163.5,
    )
    phantom.tx_sensitivity_map = np.full(shape, 0.5 + 0.1j)
    simulator = BlochSimulator(use_parallel=False)
    result = simulator.simulate_spectral_sequence(
        program, phantom, sequence_reference_ppm=163.5, simulation_timestep_s=5e-6
    )
    component = phantom.to_component_phantoms(sequence_reference_ppm=163.5)[0][1]
    reference = simulator.simulate_sequence(
        program, component, simulation_timestep_s=5e-6
    )
    assert result.pool_names == ("Water",)
    assert result.signal.size == 0
    np.testing.assert_array_equal(
        result.final_magnetization, reference.final_magnetization
    )
    np.testing.assert_array_equal(
        result.checkpoint_magnetization, reference.checkpoint_magnetization
    )
    np.testing.assert_array_equal(
        result.final_pool_magnetization[0], result.final_magnetization
    )
    assert np.max(np.hypot(result.mx, result.my)) > 0.1
