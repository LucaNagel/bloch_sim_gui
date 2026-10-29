import numpy as np
import pytest

from blochsimulator.sequence import (
    PRESS_REFOCUSING_PULSE_TYPES,
    PressTimingPlan,
    infer_spectroscopic_acquisition,
    load_pulseq,
    make_pulseq_press,
)
from blochsimulator.simulator import BlochSimulator


def _press(**overrides):
    parameters = dict(
        voxel_size_m=(10e-3, 10e-3, 10e-3),
        voxel_offset_m=(4e-3, -3e-3, 2e-3),
        spectral_points=64,
        repetition_time_s=0.2,
        echo_time_s=30e-3,
    )
    parameters.update(overrides)
    return make_pulseq_press(**parameters)


def _load(sequence, tmp_path, name="press.seq"):
    path = tmp_path / name
    sequence.write(path)
    return load_pulseq(path)


def _centre(event):
    return 0.5 * (event.start_s + event.end_s)


def _echo_mxy(program, positions):
    probes = BlochSimulator(use_parallel=False).simulate_sequence_probes(
        program,
        np.asarray(positions, dtype=float),
        np.zeros(1),
        checkpoints_s=(program.adc_events[0].start_s,),
        t1_s=10.0,
        t2_s=10.0,
        simulation_timestep_s=10e-6,
    )
    magnetization = probes.magnetization[0, :, 0]
    return magnetization[:, 0] + 1j * magnetization[:, 1]


@pytest.mark.parametrize("first_echo_time_s", [None, 12e-3])
def test_press_refocusing_centres_and_first_sample_follow_the_echo_times(
    tmp_path, first_echo_time_s
):
    sequence = _press(first_echo_time_s=first_echo_time_s)
    assert sequence.check_timing()[0]
    program = _load(sequence, tmp_path)
    excitation, first, second = (_centre(event) for event in program.rf_events)
    first_echo = 15e-3 if first_echo_time_s is None else first_echo_time_s
    definitions = sequence.definitions

    assert first - excitation == pytest.approx(first_echo / 2, abs=1e-6)
    assert 2 * (second - first) == pytest.approx(30e-3, abs=1e-6)
    assert program.adc_events[0].start_s - excitation == pytest.approx(30e-3, abs=1e-6)
    assert definitions["TE"] == pytest.approx(30e-3, abs=1e-6)
    assert definitions["PRESSFirstEchoTime"] == pytest.approx(first_echo, abs=1e-6)
    assert definitions["PRESSSecondEchoTime"] == pytest.approx(
        30e-3 - first_echo, abs=1e-6
    )
    assert definitions["RefocusingPhaseDeg"] == 90.0


def _assert_localised(program, centre):
    centre = np.asarray(centre, dtype=float)
    assert abs(_echo_mxy(program, centre[None, :])[0]) > 0.98
    for axis in range(3):
        for side in (-1.0, 1.0):
            # A 4 mm wide column several crusher cycles long stands in for a
            # voxel of isochromats; it lies fully outside one slab.
            positions = np.tile(centre, (161, 1))
            positions[:, axis] += side * (11e-3 + np.linspace(-2e-3, 2e-3, 161))
            outside = _echo_mxy(program, positions)
            assert abs(outside.mean()) < 0.05, (axis, side)

        positions = np.tile(centre, (41, 1))
        positions[:, axis] += np.linspace(-0.5e-3, 0.5e-3, 41)
        assert abs(_echo_mxy(program, positions).mean()) > 0.95, axis


def test_press_localises_the_offset_voxel_and_crushes_outside_signal(tmp_path):
    _assert_localised(_load(_press(), tmp_path), [4e-3, -3e-3, 2e-3])


def test_press_voxel_follows_the_encoding_frame(tmp_path):
    program = _load(_press(encoding_axes=("+y", "+z", "+x")), tmp_path)
    # Logical (read, phase, partition) maps onto physical (y, z, x).
    _assert_localised(program, [2e-3, 4e-3, -3e-3])


def test_press_timing_plan_and_minimums():
    plan = _press(timing_only=True)
    assert isinstance(plan, PressTimingPlan)
    assert plan.minimum_echo_time_s == pytest.approx(
        2 * plan.minimum_first_echo_time_s, rel=1e-9
    )
    assert plan.spectral_bandwidth_hz == pytest.approx(1 / plan.dwell_s)

    shortest = _press(echo_time_s=plan.minimum_echo_time_s)
    assert shortest.check_timing()[0]
    assert shortest.definitions["MinimumEchoTime"] == pytest.approx(
        plan.minimum_echo_time_s, abs=1e-6
    )
    with pytest.raises(ValueError, match="echo_time_s is too short"):
        _press(echo_time_s=0.8 * plan.minimum_echo_time_s)
    with pytest.raises(ValueError, match="first_echo_time_s is too short"):
        _press(first_echo_time_s=0.5 * plan.minimum_first_echo_time_s)
    with pytest.raises(ValueError, match="repetition_time_s is too short"):
        _press(repetition_time_s=0.5 * plan.minimum_repetition_time_s)
    with pytest.raises(ValueError, match="first_echo_time_s must lie"):
        _press(first_echo_time_s=40e-3)


def test_press_is_inferred_as_a_single_voxel_spectroscopic_grid(tmp_path):
    sequence = _press(repetitions=3, spectral_points=32)
    program = _load(sequence, tmp_path)
    acquisition = infer_spectroscopic_acquisition(program)

    assert acquisition.matrix == (1, 1)
    assert acquisition.spectral_points == 32
    assert acquisition.num_repetitions == 3
    definitions = sequence.definitions
    assert definitions["TrajectoryType"] == "single_voxel_press"
    assert definitions["VoxelSize"] == pytest.approx([10e-3] * 3)
    assert definitions["VoxelOffset"] == pytest.approx([4e-3, -3e-3, 2e-3])
    assert len(definitions["IdealSpoilerEndTimes"]) == 3


@pytest.mark.parametrize("pulse_type", PRESS_REFOCUSING_PULSE_TYPES)
def test_every_press_refocusing_pulse_type_builds(pulse_type):
    sequence = _press(refocusing_pulse_type=pulse_type)
    assert sequence.check_timing()[0]
    assert sequence.definitions["RefocusingRFPulseType"] == pulse_type


def test_press_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="refocusing_pulse_type"):
        _press(refocusing_pulse_type="designer")
    with pytest.raises(ValueError, match="voxel"):
        _press(voxel_size_m=(10e-3, 10e-3))
    with pytest.raises(ValueError, match="unknown nucleus"):
        _press(nucleus="X1")


def test_gui_press_form_builds_places_exports_and_reconstructs(
    qt_application, tmp_path
):
    import nbformat

    from blochsimulator.notebook_exporter import export_pulseq_generation_notebook
    from blochsimulator.phantom import Phantom
    from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget

    widget = SequenceSimulationWidget()
    try:
        widget.sequence_live_preview.setChecked(True)
        widget.sequence_source.setCurrentIndex(widget.PRESS_SOURCE)
        widget.press_spectral_points.setValue(64)
        widget.press_repetition_time_ms.setValue(200.0)
        widget._geometry_offset_requested(4e-3, -3e-3, 2e-3)
        qt_application.processEvents()

        assert widget.press_slice_offset_mm.value() == pytest.approx(2.0)
        assert widget._ensure_current_generated_sequence(), widget._generation_error
        definitions = widget.program.metadata["definitions"]
        assert definitions["TrajectoryType"] == "single_voxel_press"
        assert definitions["VoxelOffset"] == pytest.approx([4e-3, -3e-3, 2e-3])
        assert widget.spectroscopic_acquisition.matrix == (1, 1)
        assert widget.spectroscopic_acquisition.spectral_points == 64
        geometry = widget._sequence_geometry()
        assert geometry.fov_m == pytest.approx((10e-3, 10e-3))
        assert geometry.slice_thickness_m == pytest.approx(10e-3)
        assert geometry.slice_positions_m == pytest.approx((2e-3,))
        assert geometry.in_plane_center_m == pytest.approx((4e-3, -3e-3))
        # The default sphere is sized around the voxel, so all of it is inside.
        default = widget._default_sequence_phantom()
        corner = np.linalg.norm([9e-3, 8e-3, 7e-3])
        assert 0.3 * default.fov[0] > corner

        # TE1 = 0 means TE / 2 and is never raised to a minimum.
        minimums = widget._timing_minimums_ms()
        assert widget.press_first_echo_time_ms not in minimums
        assert minimums[widget.press_echo_time_ms] > 0
        assert "TE1" in widget.press_timing_info.text()
        widget.press_first_echo_time_ms.setValue(12.0)
        assert widget._press_pulseq_parameters()["first_echo_time_s"] == 12e-3
        widget.press_first_echo_time_ms.setValue(0.0)

        widget.press_refocusing_pulse_type.setCurrentIndex(
            widget.press_refocusing_pulse_type.findData("block")
        )
        qt_application.processEvents()
        assert not widget.press_refocusing_tbw.isEnabled()
        assert widget.program.metadata["definitions"]["RefocusingRFPulseType"] == (
            "block"
        )
        widget.press_refocusing_pulse_type.setCurrentIndex(
            widget.press_refocusing_pulse_type.findData("sinc")
        )
        qt_application.processEvents()

        kind, parameters, filename = widget._pulseq_export_spec()
        assert (kind, filename) == ("press", "press_svs.seq")
        assert widget._write_pulseq_path(tmp_path / "press").name == "press.seq"
        notebook = export_pulseq_generation_notebook(
            tmp_path / "press.ipynb", kind, parameters, seq_filename=filename
        )
        code = "\n".join(
            cell.source for cell in nbformat.read(notebook, as_version=4).cells
        )
        assert "make_pulseq_press" in code

        shape = (21, 21, 21)
        phantom = Phantom(
            shape, (0.03, 0.03, 0.03), np.full(shape, 1.0), np.full(shape, 0.1)
        )
        result = widget.simulator.simulate_sequence(
            widget.program, phantom, simulation_timestep_s=10e-6
        )
        assert result.signal.shape == (64,)
        widget.phantom = phantom
        widget._finished(result, record_run=False)
        assert "CSI" in widget.reconstruction_info.text()
        # One voxel is not an image: the explorer shows where it lies in the
        # phantom above its spectrum instead of 1 × 1 k-space and image.
        explorer = widget.reconstruction_explorer
        assert explorer.kspace_panel.isHidden()
        assert explorer.image_panel.isHidden()
        assert explorer.phantom_panel.isHidden()
        assert not explorer.voxel_location_view.isHidden()
        assert "Single-voxel spectroscopy" in explorer.summary.text()
    finally:
        widget.close()
        widget.deleteLater()
        qt_application.processEvents()
