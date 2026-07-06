import sys
import runpy
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator.ui.main_window import BlochSimulatorGUI
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget
from blochsimulator.sequence import SequenceCompiler


EXAMPLE_MAIN = runpy.run_path(
    str(Path(__file__).parents[1] / "examples" / "generate_epi.py")
)["main"]


def test_sequence_workspace_is_lazy_and_initializes_on_selection():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    assert window.sequence_simulation_widget is None
    window.tab_widget.setCurrentIndex(window.sequence_simulation_tab_index)
    app.processEvents()
    assert isinstance(window.sequence_simulation_widget, SequenceSimulationWidget)
    assert window.sequence_simulation_widget.program.source == "internal-fid"
    window.close()
    window.deleteLater()
    app.processEvents()


def test_phantom_workspace_is_visible():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    tab_names = [
        window.tab_widget.tabText(index) for index in range(window.tab_widget.count())
    ]

    assert "🔬 Phantom" in tab_names
    assert window.phantom_widget is None
    window.tab_widget.setCurrentIndex(window.phantom_tab_index)
    app.processEvents()
    assert window.phantom_widget is not None

    window.close()
    window.deleteLater()
    app.processEvents()


def test_sequence_workspace_builds_cartesian_epi_from_controls():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    assert not widget.acquisition_group.isHidden()
    assert not widget.acquisition_group.isEnabled()
    widget.sequence_source.setCurrentIndex(1)
    assert widget.acquisition_group.isEnabled()
    widget.read_matrix.setValue(6)
    widget.phase_matrix.setValue(4)
    widget.sampling_bandwidth_khz.setValue(25.0)
    app.processEvents()

    assert widget.acquisition is not None
    assert widget.program.source == "internal-cartesian-epi"
    assert widget.acquisition.read_matrix == 6
    assert widget.acquisition.phase_matrix == 4
    assert widget.acquisition.dwell_s == 40e-6
    compiled = SequenceCompiler().compile(widget.program)
    assert compiled.adc_times_s.size == 24
    widget.acquisition.validate_gradient_moments(compiled.adc_gradient_moment_cyc_per_m)
    assert "25.000 kHz" in widget.sequence_info.text()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_scopes_object_controls_to_the_selected_source():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()

    assert widget.object_source.currentText() == "Phantom tab / designer"
    assert widget.object_type.currentIndex() == 0
    assert not widget.object_type.isEnabled()
    assert not widget.t1_ms.isEnabled()
    assert "No phantom selected" in widget.phantom_summary.text()

    widget.object_source.setCurrentIndex(1)
    assert widget.object_type.currentText() == "Uniform cube"
    assert widget.object_type.isEnabled()
    assert widget.t1_ms.isEnabled()
    assert widget.phantom_summary.isHidden()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_displays_cartesian_kspace_and_reconstruction():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.sequence_source.setCurrentIndex(1)
    widget.read_matrix.setValue(4)
    widget.phase_matrix.setValue(3)
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(2)
    widget._build_phantom()
    result = widget.simulator.simulate_sequence(widget.program, widget.phantom)
    widget._finished(result)
    app.processEvents()

    assert widget.kspace_view.image.shape == (4, 3)
    assert widget.reconstruction_view.image.shape == (4, 3)
    assert np.all(np.isfinite(widget.reconstruction_view.image))
    assert "grid=3×4" in widget.kspace_info.text()
    assert "|IFFT2|" in widget.reconstruction_info.text()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_infers_imported_epi_and_syncs_fov(tmp_path):
    path = tmp_path / "generated_epi.seq"
    EXAMPLE_MAIN(
        write_seq=True,
        seq_filename=str(path),
        fov=0.22,
        n_x=4,
        n_y=4,
        slice_thickness=4e-3,
        n_slices=1,
    )
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget._load_pulseq_path(path)
    widget.matrix_size.setValue(4)
    widget.z_matrix_size.setValue(4)

    assert widget.acquisition is not None
    assert widget.acquisition.read_matrix == 4
    assert widget.acquisition.phase_matrix == 4
    assert widget.fov_cm.value() == pytest.approx(22.0)
    assert widget.fov_z_cm.value() == pytest.approx(0.4)

    widget._build_phantom()
    result = widget.simulator.simulate_sequence(widget.program, widget.phantom)
    widget._finished(result)
    app.processEvents()
    assert widget.kspace_view.image.shape == (4, 4)
    assert widget.reconstruction_view.image.shape == (4, 4)
    assert np.ptp(widget.state_view.image) < 1e-10

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_keeps_run_button_outside_scroll_area():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.resize(900, 420)
    widget.show()
    app.processEvents()

    scroll_bar = widget.controls_scroll.verticalScrollBar()
    assert scroll_bar.maximum() > 0
    assert widget.run_button.isVisible()
    assert not widget.controls_scroll.viewport().isAncestorOf(widget.run_button)

    widget.close()
    widget.deleteLater()
    app.processEvents()
