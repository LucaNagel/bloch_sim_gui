import sys
import runpy
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from blochsimulator.ui.main_window import BlochSimulatorGUI
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget
from blochsimulator.sequence import (
    SequenceCompiler,
    SequenceSimulationResult,
    SpectroscopicAcquisition,
)


EXAMPLE_MAIN = runpy.run_path(
    str(Path(__file__).parents[1] / "sequences" / "scripts" / "generate_epi.py")
)["main"]


def test_sequence_workspace_is_lazy_and_initializes_on_selection():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    assert window.sequence_simulation_widget is None
    window.tab_widget.setCurrentIndex(window.sequence_simulation_tab_index)
    app.processEvents()

    assert window.sequence_simulation_widget.simulation_timestep_us.value() == 1.0
    assert isinstance(window.sequence_simulation_widget, SequenceSimulationWidget)
    assert window.sequence_simulation_widget.program.source == "internal-fid"
    assert window.tab_widget.cornerWidget(Qt.TopRightCorner) is window.workspace_switch

    window.set_workspace_mode("sequence")
    assert window.workspace_mode == "sequence"
    assert window.free_mode_left_container.isHidden()
    assert window.free_mode_playback_header.isHidden()
    assert window.toolbar_run_bar.isHidden()
    assert window.status_run_bar.isHidden()
    assert window.tab_widget.isTabVisible(window.sequence_simulation_tab_index)
    assert window.tab_widget.isTabVisible(window.phantom_tab_index)
    assert not window.tab_widget.isTabVisible(0)
    assert window.workspace_mode_selector.currentData() == "sequence"

    window.set_workspace_mode("free")
    assert window.workspace_mode == "free"
    assert not window.free_mode_left_container.isHidden()
    assert not window.free_mode_playback_header.isHidden()
    assert window.tab_widget.isTabVisible(0)
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
    assert widget.sequence_source.itemText(1) == "EPI"
    assert widget.acquisition_group.isHidden()
    widget.sequence_source.setCurrentIndex(1)
    assert not widget.acquisition_group.isHidden()
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

    widget.spectroscopic_acquisition = SpectroscopicAcquisition(
        matrix=(4, 3),
        fov_m=(0.2, 0.15),
        spectral_points=8,
        dwell_s=1e-3,
        encoding_indices=tuple((x, y) for y in range(3) for x in range(4)),
    )
    widget._configure_spectroscopy_selectors()
    assert widget.spectral_point_slider.maximum() == 7
    assert widget.spectrum_x_slider.maximum() == 3
    assert widget.spectrum_y_slider.maximum() == 2
    widget.spectral_point_slider.setValue(5)
    widget.spectrum_x_slider.setValue(2)
    widget.spectrum_y_selector.setValue(1)
    assert widget.spectral_point_selector.value() == 5
    assert widget.spectrum_x_selector.value() == 2
    assert widget.spectrum_y_slider.value() == 1

    csi = widget.spectroscopic_acquisition
    adc_times = np.concatenate(
        [
            event * 0.02 + np.arange(csi.spectral_points) * csi.dwell_s
            for event in range(csi.num_encodings)
        ]
    )
    widget.result = SequenceSimulationResult(
        signal=np.ones(csi.num_samples, dtype=np.complex128),
        adc_times_s=adc_times,
        final_magnetization=np.zeros((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        metadata={"spectroscopic_acquisition": csi.to_metadata()},
    )
    widget._show_spectroscopic_result(widget.result)
    widget.split_view_checkbox.setChecked(True)
    assert widget.view_stack.currentIndex() == 1
    assert widget.split_image_item.image.shape == (4, 3)
    widget.split_signal_source.setCurrentText("FID")
    assert "FID" in widget.split_signal_plot.getPlotItem().titleLabel.text
    widget._set_csi_voxel(3, 2)
    assert widget.spectrum_x_selector.value() == 3
    assert widget.spectrum_y_selector.value() == 2
    assert widget.split_voxel_marker.getData()[0][0] == pytest.approx(3)

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
    assert widget.built_in_properties_group.isHidden()
    assert "No phantom selected" in widget.phantom_summary.text()

    widget.object_source.setCurrentIndex(1)
    assert widget.object_type.currentText() == "Uniform cube"
    assert widget.object_type.isEnabled()
    assert widget.t1_ms.isEnabled()
    assert not widget.built_in_properties_group.isHidden()
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


def test_sequence_workspace_selects_multislice_cartesian_frames(tmp_path):
    path = tmp_path / "multislice_epi.seq"
    EXAMPLE_MAIN(
        write_seq=True,
        seq_filename=str(path),
        fov=0.22,
        n_x=4,
        n_y=4,
        slice_thickness=4e-3,
        n_slices=3,
    )
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget._load_pulseq_path(path)
    widget.matrix_size.setValue(4)
    widget.z_matrix_size.setValue(3)
    widget._build_phantom()
    result = widget.simulator.simulate_sequence(widget.program, widget.phantom)
    widget._finished(result)
    app.processEvents()

    assert widget.acquisition_frames.num_frames == 3
    assert widget.frame_selector.count() == 4
    assert widget.frame_selector.itemText(0) == "All 3 frames (montage)"
    assert widget.frame_selector.itemText(2) == "slice=1"
    assert widget.frame_slider.minimum() == -1
    assert widget.frame_slider.maximum() == 2
    assert widget.frame_slider.value() == -1
    assert widget.kspace_view.image.shape == (14, 4)
    assert widget.reconstruction_view.image.shape == (14, 4)
    assert "montage of 3 frames" in widget.reconstruction_info.text()
    widget.frame_selector.setCurrentIndex(3)
    assert widget.reconstruction_view.image.shape == (4, 4)
    assert "slice=2" in widget.reconstruction_info.text()
    assert widget.frame_slider.value() == 2
    widget.frame_slider.setValue(1)
    assert widget.frame_selector.currentData() == 1
    assert "slice=1" in widget.reconstruction_info.text()
    assert widget.kspace_zoom_info.text().startswith("Zoom: ")
    assert widget.kspace_view.ui.histogram.axis.tickStrings([1.234], 1.0, 0.1) == [
        "1.23"
    ]
    assert np.array_equal(np.unique(result.to_xarray().slice_index), [0, 1, 2])

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
