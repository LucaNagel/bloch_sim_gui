import numpy as np
import pytest

from blochsimulator.sequence import (
    SliceFidTimingPlan,
    infer_spectroscopic_acquisition,
    load_pulseq,
    make_pulseq_slice_fid,
)
from blochsimulator.simulator import BlochSimulator


def _slice_fid(**overrides):
    parameters = dict(
        slice_thickness_m=6e-3,
        slice_offset_m=4e-3,
        spectral_points=64,
        repetition_time_s=0.2,
    )
    parameters.update(overrides)
    return make_pulseq_slice_fid(**parameters)


def _load(sequence, tmp_path, name="slice_fid.seq"):
    path = tmp_path / name
    sequence.write(path)
    return load_pulseq(path)


def _first_sample_mxy(program, positions):
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


def test_slice_fid_timing_plan_and_acquisition_delay(tmp_path):
    plan = _slice_fid(timing_only=True)
    assert isinstance(plan, SliceFidTimingPlan)
    assert plan.spectral_bandwidth_hz == pytest.approx(1 / plan.dwell_s)

    shortest = _slice_fid()
    assert shortest.check_timing()[0]
    assert shortest.definitions["AcquisitionDelay"] == pytest.approx(
        plan.minimum_acquisition_delay_s, abs=1e-6
    )
    delayed = _slice_fid(acquisition_delay_s=8e-3)
    program = _load(delayed, tmp_path)
    rf = program.rf_events[0]
    rf_centre = 0.5 * (rf.start_s + rf.end_s)
    # ADC events start at their first sample, as in PRESS.
    assert program.adc_events[0].start_s - rf_centre == pytest.approx(8e-3, abs=2e-6)
    assert delayed.definitions["TR"] == pytest.approx(0.2, abs=1e-5)
    with pytest.raises(ValueError, match="acquisition_delay_s is too short"):
        _slice_fid(acquisition_delay_s=0.5 * plan.minimum_acquisition_delay_s)
    with pytest.raises(ValueError, match="repetition_time_s is too short"):
        _slice_fid(repetition_time_s=0.5 * plan.minimum_repetition_time_s)
    with pytest.raises(ValueError, match="slice_axis"):
        _slice_fid(slice_axis="z")


@pytest.mark.parametrize(
    "slice_axis, axis, sign", [("+z", 2, 1.0), ("-y", 1, -1.0), ("+x", 0, 1.0)]
)
def test_slice_fid_excites_and_rephases_only_the_offset_slice(
    tmp_path, slice_axis, axis, sign
):
    program = _load(_slice_fid(slice_axis=slice_axis), tmp_path)
    centre = np.zeros(3)
    centre[axis] = sign * 4e-3
    # A column across the slice keeps its phase after the rephaser.
    positions = np.tile(centre, (41, 1))
    positions[:, axis] += np.linspace(-1e-3, 1e-3, 41)
    assert abs(_first_sample_mxy(program, positions).mean()) > 0.95
    # In-plane positions are not encoded and all contribute.
    in_plane = np.tile(centre, (5, 1))
    other = [index for index in range(3) if index != axis]
    in_plane[:, other[0]] = np.linspace(-20e-3, 20e-3, 5)
    assert np.all(np.abs(_first_sample_mxy(program, in_plane)) > 0.98)
    outside = centre.copy()
    outside[axis] += sign * 12e-3
    assert abs(_first_sample_mxy(program, outside[None, :])[0]) < 0.05


def test_slice_fid_is_inferred_as_a_1x1_spectroscopic_grid(tmp_path):
    sequence = _slice_fid(repetitions=3, spectral_points=32)
    program = _load(sequence, tmp_path)
    acquisition = infer_spectroscopic_acquisition(program)

    assert acquisition.matrix == (1, 1)
    assert acquisition.spectral_points == 32
    assert acquisition.num_repetitions == 3
    definitions = sequence.definitions
    assert definitions["TrajectoryType"] == "slice_selective_fid"
    assert definitions["SliceThickness"] == pytest.approx(6e-3)
    assert definitions["SliceOffset"] == pytest.approx(4e-3)
    assert len(definitions["IdealSpoilerEndTimes"]) == 3
    unspoiled = _slice_fid(spoil_after_readout=False)
    assert unspoiled.definitions["IdealSpoilerEndTimes"] == []


def test_gui_slice_fid_form_builds_places_exports_and_reconstructs(
    qt_application, tmp_path
):
    import nbformat

    from blochsimulator.notebook_exporter import export_pulseq_generation_notebook
    from blochsimulator.phantom import Phantom
    from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget

    widget = SequenceSimulationWidget()
    try:
        widget.sequence_live_preview.setChecked(True)
        widget.sequence_source.setCurrentIndex(widget.SLICE_FID_SOURCE)
        widget.slice_fid_spectral_points.setValue(64)
        widget.slice_fid_repetition_time_ms.setValue(200.0)
        widget.slice_fid_axis.setCurrentText("-Y")
        widget._geometry_offset_requested(0.0, 0.0, 3e-3)
        qt_application.processEvents()

        assert widget.slice_fid_offset_mm.value() == pytest.approx(3.0)
        assert widget._ensure_current_generated_sequence(), widget._generation_error
        definitions = widget.program.metadata["definitions"]
        assert definitions["TrajectoryType"] == "slice_selective_fid"
        assert widget.spectroscopic_acquisition.matrix == (1, 1)
        assert widget.spectroscopic_acquisition.spectral_points == 64
        geometry = widget._sequence_geometry()
        assert geometry.slice_thickness_m == pytest.approx(5e-3)
        assert geometry.slice_positions_m == pytest.approx((3e-3,))
        assert "TR" in widget.slice_fid_timing_info.text()

        # A zero delay means the shortest delay and is never raised.
        minimums = widget._timing_minimums_ms()
        assert widget.slice_fid_acquisition_delay_ms not in minimums
        assert widget._slice_fid_pulseq_parameters()["acquisition_delay_s"] is None

        kind, parameters, filename = widget._pulseq_export_spec()
        assert (kind, filename) == ("slice_fid", "slice_fid.seq")
        assert widget._write_pulseq_path(tmp_path / "fid").name == "fid.seq"
        notebook = export_pulseq_generation_notebook(
            tmp_path / "fid.ipynb", kind, parameters, seq_filename=filename
        )
        code = "\n".join(
            cell.source for cell in nbformat.read(notebook, as_version=4).cells
        )
        assert "make_pulseq_slice_fid" in code

        shape = (9, 9, 9)
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
    finally:
        widget.close()
        widget.deleteLater()
        qt_application.processEvents()
