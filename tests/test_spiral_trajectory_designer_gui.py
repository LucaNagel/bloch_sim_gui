import sys

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator.sequence.kspace_trajectory import calculate_kspace
from blochsimulator.sequence.scanner import ScannerParameters
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget
from blochsimulator.ui.spiral_trajectory_designer import SpiralTrajectoryDesigner


def test_designer_warns_about_limits_and_undersampling_and_exports(tmp_path):
    app = QApplication.instance() or QApplication(sys.argv)
    designer = SpiralTrajectoryDesigner(
        fov_m=(0.12, 0.12),
        matrix=(32, 32),
        sampling_bandwidth_hz=100e3,
        turns=8,
        nucleus="C13",
        scanner_parameters={"max_grad_mtm": 32.0, "max_slew_tms": 130.0},
    )
    app.processEvents()

    assert designer.design.bandwidth_limited
    text = designer.message.text()
    assert "lower the bandwidth" in text
    assert "undersample" in text
    assert "1H" not in text
    rows = [
        designer.info_table.item(row, 0).text()
        for row in range(designer.info_table.rowCount())
    ]
    assert rows[:3] == ["Bandwidth", "Readout", "Peak gradient"]

    designer.nyquist_button.click()
    designer.recalculate()
    assert designer.turns.value() == pytest.approx(16.0)
    assert "undersample" not in designer.message.text()

    path = tmp_path / "spiral.npz"
    designer.export_design(path)
    with np.load(path) as exported:
        assert exported["k_cyc_per_m"].shape == (1024, 2)
    designer.close()
    designer.deleteLater()
    app.processEvents()


def test_applied_carbon_design_is_played_by_the_spiral_sequence():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    # A clinical gradient system, on which the 13C spiral is slew limited.
    widget.scanner_parameters = ScannerParameters(max_grad_mtm=32.0, max_slew_tms=130.0)
    widget.nucleus.setCurrentText("C13")
    widget.sequence_live_preview.setChecked(True)
    widget.sequence_source.setCurrentIndex(1)
    widget.read_matrix.setValue(16)
    widget.phase_matrix.setValue(16)
    widget.epi_read_fov_mm.setValue(120.0)
    widget.epi_phase_fov_mm.setValue(120.0)
    widget.epi_slice_thickness_mm.setValue(10.0)
    widget.sampling_bandwidth_khz.setValue(50.0)
    widget.epi_readout_trajectory.setCurrentText("Spiral")
    app.processEvents()
    assert widget.epi_spiral_design_button.isEnabled()

    designer = widget.open_spiral_designer()
    designer.turns.setValue(8.0)
    designer.recalculate()
    design = designer.design
    assert design.nucleus == "C13"
    assert design.bandwidth_limited
    designer.apply_button.click()
    app.processEvents()

    assert widget.epi_spiral_turns.value() == pytest.approx(8.0)
    assert widget.sampling_bandwidth_khz.value() == pytest.approx(
        design.sampling_bandwidth_hz / 1e3, abs=2e-3
    )
    assert widget.program.source == "internal-spiral"
    k = calculate_kspace(widget.program).adc_k_cyc_per_m[:256, :2]
    k_max = 16 / (2 * 0.12)
    assert np.abs(k - design.k_cyc_per_m).max() < 2e-3 * k_max

    designer.close()
    widget.close()
    widget.deleteLater()
    app.processEvents()
