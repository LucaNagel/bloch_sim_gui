import runpy
from pathlib import Path

import numpy as np
import pytest

from blochsimulator.sequence import infer_spectroscopic_acquisition, load_pulseq


pypulseq = pytest.importorskip("pypulseq")

CSI_SCRIPT = runpy.run_path(
    str(Path(__file__).parents[1] / "sequences" / "scripts" / "generate_csi.py")
)
main = CSI_SCRIPT["main"]
phase_encoding_indices = CSI_SCRIPT["phase_encoding_indices"]
MULTIREP_MAIN = runpy.run_path(
    str(
        Path(__file__).parents[1] / "sequences" / "scripts" / "generate_csi_mutlirep.py"
    )
)["main"]


@pytest.mark.parametrize("ordering", ["linear", "spiral", "centric"])
def test_csi_phase_encoding_orders_cover_grid_once(ordering):
    order = phase_encoding_indices(5, 4, ordering, fov=(0.20, 0.12))
    assert len(order) == 20
    assert len(set(order)) == 20
    assert set(order) == {(x, y) for y in range(4) for x in range(5)}


def test_csi_linear_and_center_out_ordering():
    assert phase_encoding_indices(3, 2, "linear") == [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
    ]
    assert phase_encoding_indices(5, 5, "spiral")[0] == (2, 2)
    assert phase_encoding_indices(5, 5, "centric")[0] == (2, 2)


def test_csi_centric_radius_never_decreases():
    order = phase_encoding_indices(6, 5, "centric", fov=(0.24, 0.15))
    radii = [np.hypot((x - 3) / 0.24, (y - 2) / 0.15) for x, y in order]
    assert np.all(np.diff(radii) >= -1e-12)


def test_csi_spectral_parameters_and_timing():
    sequence = main(
        n_x=2,
        n_y=3,
        spectral_bandwidth_hz=2000.0,
        n_spectral_points=16,
        phase_encoding_order="centric",
        te=6e-3,
        tr=30e-3,
    )
    assert sequence.check_timing()[0]
    assert sequence.definitions["MatrixSize"] == [2, 3, 16]
    assert sequence.definitions["SpectralBandwidth"] == pytest.approx(2000.0)
    assert sequence.definitions["SpectralResolution"] == pytest.approx(125.0)
    assert sequence.definitions["PhaseEncodingOrder"] == "centric"
    assert sequence.definitions["TE"] >= 6e-3
    assert sequence.definitions["TR"] >= 30e-3
    adc_times, _ = sequence.adc_times()
    assert adc_times.size == 2 * 3 * 16


def test_csi_resolution_can_determine_number_of_points():
    sequence = main(
        n_x=1,
        n_y=1,
        spectral_bandwidth_hz=1000.0,
        spectral_resolution_hz=30.0,
        te=6e-3,
        tr=50e-3,
    )
    assert sequence.definitions["SpectralPoints"] == 34
    assert sequence.definitions["SpectralResolution"] <= 30.0


def test_csi_multirep_repeats_complete_grid_with_rep_labels(tmp_path):
    sequence = MULTIREP_MAIN(
        n_x=2,
        n_y=3,
        n_spectral_points=8,
        spectral_bandwidth_hz=2000.0,
        phase_encoding_order="centric",
        te=6e-3,
        tr=30e-3,
        n_repetitions=3,
    )
    path = tmp_path / "csi_multirep.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)
    acquisition = infer_spectroscopic_acquisition(program)

    assert sequence.check_timing()[0]
    assert sequence.definitions["Repetitions"] == 3
    assert sequence.adc_times()[0].size == 2 * 3 * 8 * 3
    assert acquisition.num_repetitions == 3
    assert acquisition.num_samples == 2 * 3 * 8 * 3
    assert program.metadata["adc_label_values"]["REP"] == (
        (0,) * 6 + (1,) * 6 + (2,) * 6
    )


def test_csi_adds_configurable_readout_spoilers(tmp_path):
    fov = (0.02, 0.04)
    slice_thickness = 0.01
    sequence = main(
        fov=fov,
        n_x=1,
        n_y=1,
        slice_thickness=slice_thickness,
        n_spectral_points=8,
        spectral_bandwidth_hz=2000.0,
        te=6e-3,
        tr=30e-3,
        spoiler_duration=2e-3,
        spoiler_cycles=3.0,
        spoiler_cycles_per_voxel=0.5,
    )
    path = tmp_path / "csi_spoilers.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)

    definitions = program.metadata["definitions"]
    assert definitions["SpoilAfterReadout"]
    assert definitions["SpoilerAxes"] == "xyz"
    assert definitions["SpoilerDuration"] == pytest.approx(2e-3)
    spoiler_end = float(np.asarray(definitions["SpoilerEndTimes"]).reshape(-1)[0])
    events = [
        event
        for event in program.gradient_events
        if np.isclose(event.end_s, spoiler_end, rtol=0.0, atol=1e-9)
    ]
    assert {event.axis for event in events} == {"x", "y", "z"}
    expected_cycles = {"x": 0.5, "y": 0.5, "z": 3.0}
    extents = {"x": fov[0], "y": fov[1], "z": slice_thickness}
    for event in events:
        moment = np.sum(event.samples_hz_per_m) * event.raster_s
        assert moment * extents[event.axis] == pytest.approx(
            expected_cycles[event.axis], abs=2e-5
        )


def test_csi_spoiler_can_be_disabled():
    sequence = main(
        n_x=1,
        n_y=1,
        n_spectral_points=8,
        spectral_bandwidth_hz=2000.0,
        te=6e-3,
        tr=20e-3,
        spoil_after_readout=False,
    )

    assert not sequence.definitions["SpoilAfterReadout"]
    assert sequence.definitions["SpoilerAxes"] == "none"
    assert sequence.definitions["SpoilerEndTimes"] == []
