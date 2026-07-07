import runpy
from pathlib import Path

import numpy as np
import pytest


pypulseq = pytest.importorskip("pypulseq")

CSI_SCRIPT = runpy.run_path(
    str(Path(__file__).parents[1] / "sequences" / "scripts" / "generate_csi.py")
)
main = CSI_SCRIPT["main"]
phase_encoding_indices = CSI_SCRIPT["phase_encoding_indices"]


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
