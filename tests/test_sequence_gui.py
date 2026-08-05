import sys
import runpy
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QMenu, QMessageBox, QScrollArea, QToolBar

from blochsimulator.ui.main_window import BlochSimulatorGUI
from blochsimulator.ui.sequence_simulation_widget import (
    SequenceProbeThread,
    SequenceSimulationWidget,
    _event_step_plot_data,
)
from blochsimulator.ui.default_settings import WorkspaceDefaults
from blochsimulator.sequence import (
    ADCEvent,
    AcquisitionDimensions,
    GradientEvent,
    RFEvent,
    SequenceCompiler,
    SequenceProbeResult,
    SequenceProgram,
    SequenceSimulationResult,
    SpectroscopicAcquisition,
    load_pulseq,
)
from blochsimulator.units import NUCLEUS_GAMMA_HZ_PER_T, rf_hz_to_gauss


def _select_workspace(window, mode, timeout_ms=2_000):
    """Select a workspace and wait until its deferred transition completes."""
    index = window.workspace_mode_selector.findData(mode)
    assert index >= 0
    window.workspace_mode_selector.setCurrentIndex(index)

    deadline = time.monotonic() + timeout_ms / 1_000
    while window.workspace_mode != mode and time.monotonic() < deadline:
        QTest.qWait(10)

    assert window.workspace_mode == mode


EXAMPLE_MAIN = runpy.run_path(
    str(Path(__file__).parents[1] / "sequences" / "scripts" / "generate_epi.py")
)["main"]
SPECTRAL_3D_MAIN = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "sequences"
        / "scripts"
        / "generate_3d_bssfp_spectral_selective.py"
    )
)["main"]


def test_sequence_workspace_is_lazy_and_initializes_on_selection(tmp_path):
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    window.app_settings.setValue("sequence/kernel", "reference")
    window.app_settings.setValue("sequence/dynamic_kernel", "native_parallel")
    window.app_settings.setValue("sequence/timestep_preset", "fast")
    window.app_settings.setValue("sequence/timestep_us", 10.0)
    window.app_settings.setValue("simulation/thread_mode", "manual")
    window.app_settings.setValue("simulation/manual_threads", 2)
    window.app_settings.setValue("defaults/sequence_fov_x_mm", 180.0)
    window.app_settings.setValue("defaults/sequence_fov_y_mm", 170.0)
    window.app_settings.setValue("defaults/sequence_fov_z_mm", 80.0)
    window.app_settings.setValue("defaults/field_strength_t", 7.0)
    assert window.sequence_simulation_widget is None
    assert not window.tab_widget.isTabVisible(window.sequence_simulation_tab_index)
    assert not window.tab_widget.isTabVisible(window.phantom_tab_index)
    assert not window.time_control.compact
    assert window.tab_widget.tabBar().usesScrollButtons()
    assert not window.tab_widget.tabBar().expanding()
    assert window.tab_widget.tabBar().elideMode() == Qt.ElideRight
    window.tab_widget.setCurrentIndex(window.sequence_simulation_tab_index)
    app.processEvents()

    assert window.sequence_simulation_widget.simulation_timestep_us.value() == 10.0
    assert window.sequence_simulation_widget.simulator.sequence_kernel == "reference"
    assert (
        window.sequence_simulation_widget.simulator.dynamic_sequence_kernel
        == "native_parallel"
    )
    assert window.sequence_simulation_widget.simulator.num_threads == 2
    assert window.sequence_simulation_widget.epi_read_fov_mm.value() == 180.0
    assert window.sequence_simulation_widget.epi_phase_fov_mm.value() == 170.0
    assert window.sequence_simulation_widget.bssfp_partition_fov_mm.value() == 80.0
    assert window.sequence_simulation_widget.field_strength_t.value() == 7.0
    updated_defaults = WorkspaceDefaults(
        sequence_fov_mm=(90.0, 85.0, 40.0), field_strength_t=9.4
    )
    window.sequence_simulation_widget.set_workspace_defaults(updated_defaults)
    assert window.sequence_simulation_widget.epi_read_fov_mm.value() == 90.0
    assert window.sequence_simulation_widget.epi_phase_fov_mm.value() == 85.0
    assert window.sequence_simulation_widget.bssfp_partition_fov_mm.value() == 40.0
    assert window.sequence_simulation_widget.field_strength_t.value() == 9.4
    assert isinstance(window.sequence_simulation_widget, SequenceSimulationWidget)
    assert window.sequence_simulation_widget.program.source == "internal-fid"
    assert window.sequence_simulation_widget._rf_designer is window.rf_designer
    assert window.sequence_simulation_widget._rf_designer_pulse_data is not None
    assert window.tab_widget.cornerWidget(Qt.TopRightCorner) is None
    assert window.workspace_switch.parentWidget() is window.free_mode_colormap_controls
    assert window.statusBar().isAncestorOf(window.status_export_button)
    assert window.findChild(QToolBar, "main_toolbar") is None
    assert window.mag_3d.export_3d_btn.isHidden()
    assert window.mag_3d.view_layout.indexOf(window.mag_3d.track_checkbox) >= 0
    assert window.mag_3d.view_layout.indexOf(window.mag_3d.mean_checkbox) >= 0
    ancestor = window.mag_3d.parentWidget()
    while ancestor is not None:
        assert not isinstance(ancestor, QScrollArea)
        ancestor = ancestor.parentWidget()
    assert window.centralWidget().layout().contentsMargins().left() == 6
    assert window.sequence_simulation_placeholder.layout().contentsMargins().left() == 0
    window._set_tooltips_enabled(True)
    assert all(
        window.tab_widget.tabToolTip(index).strip()
        for index in range(window.tab_widget.count())
    )

    file_menu = window.findChild(QMenu, "menu_file")
    tools_menu = window.findChild(QMenu, "menu_tools")
    assert file_menu is not None
    assert tools_menu is not None
    assert [
        action.text()
        for action in file_menu.actions()
        if action.objectName() == "action_export_results"
    ] == ["Export Results..."]
    assert all(
        action.objectName() != "action_export_results_tools"
        for action in tools_menu.actions()
    )

    sequence_widget = window.sequence_simulation_widget
    sequence_widget.sequence_live_preview.setChecked(True)
    sequence_widget.sequence_source.setCurrentIndex(1)
    sequence_widget.read_matrix.setValue(4)
    sequence_widget.phase_matrix.setValue(3)
    sequence_widget.epi_repetition_time_ms.setValue(100.0)
    sequence_widget.epi_rf_pulse_type.setCurrentText("RF Pulse Designer")
    window.rf_designer.duration.setValue(2.0)
    app.processEvents()
    assert sequence_widget.program.metadata["definitions"]["RFPulseType"] == (
        "designer"
    )
    assert sequence_widget.program.metadata["definitions"]["RFDuration"] == (
        pytest.approx(2e-3)
    )

    window.mag_3d.refresh_viewport = MagicMock()
    window.anim_timer.start(10_000)
    window.set_workspace_mode("sequence")
    assert window.workspace_mode == "sequence"
    assert not window.anim_timer.isActive()
    assert window.free_mode_left_container.isHidden()
    assert window.free_mode_playback_header.isHidden()
    assert window.status_run_bar.isHidden()
    assert window.tab_widget.isTabVisible(window.sequence_simulation_tab_index)
    assert window.tab_widget.isTabVisible(window.phantom_tab_index)
    assert not window.tab_widget.isTabVisible(0)
    assert window.tab_widget.tabToolTip(window.sequence_simulation_tab_index)
    assert window.tab_widget.tabToolTip(window.phantom_tab_index)
    assert window.workspace_mode_selector.currentData() == "sequence"

    window.set_workspace_mode("free")
    assert window.workspace_mode == "free"
    assert not window.free_mode_left_container.isHidden()
    assert not window.free_mode_playback_header.isHidden()
    assert window.tab_widget.isTabVisible(0)
    assert not window.tab_widget.isTabVisible(window.sequence_simulation_tab_index)
    assert not window.tab_widget.isTabVisible(window.phantom_tab_index)
    expected_free_tab = (
        window.magnetization_tab_index
        if sys.platform == "darwin"
        else window.mag_3d_tab_index
    )
    assert window.tab_widget.currentIndex() == expected_free_tab
    app.processEvents()
    if sys.platform != "darwin":
        window.mag_3d.refresh_viewport.assert_called()

    tab_changes = []
    window.tab_widget.currentChanged.connect(tab_changes.append)
    _select_workspace(window, "sequence")
    _select_workspace(window, "free")
    _select_workspace(window, "sequence")

    assert window.workspace_mode == "sequence"
    assert window.tab_widget.currentIndex() == window.sequence_simulation_tab_index
    expected_tab_changes = [
        window.sequence_simulation_tab_index,
        expected_free_tab,
        window.sequence_simulation_tab_index,
    ]
    assert tab_changes == expected_tab_changes
    window.close()
    window.deleteLater()
    app.processEvents()


def test_phantom_workspace_is_visible():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    tab_names = [
        window.tab_widget.tabText(index) for index in range(window.tab_widget.count())
    ]

    assert "Phantom" in tab_names
    assert "Parameter Sweep" in tab_names
    assert all(not name.startswith(("🔬", "📊")) for name in tab_names)
    assert window.phantom_widget is None
    assert not window.tab_widget.isTabVisible(window.phantom_tab_index)
    window.set_workspace_mode("sequence")
    assert window.tab_widget.isTabVisible(window.phantom_tab_index)
    window.tab_widget.setCurrentIndex(window.phantom_tab_index)
    app.processEvents()
    assert window.phantom_widget is not None

    window.close()
    window.deleteLater()
    app.processEvents()


def test_workspace_roundtrip_remains_interactive():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.show()
    app.processEvents()
    initial_window_size = window.size()
    initial_frame_geometry = window.frameGeometry()

    _select_workspace(window, "sequence")
    assert window.workspace_mode == "sequence"
    assert not window.workspace_header.isVisible()
    assert window.tab_widget.cornerWidget(Qt.TopRightCorner) is window.workspace_switch

    _select_workspace(window, "free")
    assert window.workspace_mode == "free"
    assert not window.workspace_header.isVisible()
    assert window.tab_widget.cornerWidget(Qt.TopRightCorner) is None
    assert window.workspace_switch.parentWidget() is window.free_mode_colormap_controls
    assert window.size() == initial_window_size
    assert window.frameGeometry() == initial_frame_geometry

    preview_checked = window.status_preview_checkbox.isChecked()
    QTest.mouseClick(window.status_preview_checkbox, Qt.LeftButton)
    assert window.status_preview_checkbox.isChecked() is not preview_checked

    window.close()
    window.deleteLater()
    app.processEvents()


def test_playback_keeps_the_visible_heatmap_canvas_enabled():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.anim_timer.start(10_000)

    window.tab_widget.setCurrentIndex(window.magnetization_tab_index)
    app.processEvents()
    assert window.mxy_heatmap_layout.updatesEnabled()
    assert window.mz_heatmap_layout.updatesEnabled()

    window.tab_widget.setCurrentIndex(window.signal_tab_index)
    app.processEvents()
    assert window.signal_heatmap_layout.updatesEnabled()
    assert not window.mxy_heatmap_layout.updatesEnabled()

    window.anim_timer.stop()
    window.close()
    window.deleteLater()
    app.processEvents()


def test_sequence_workspace_builds_cartesian_epi_from_controls():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    assert widget.sequence_source.itemText(1) == "EPI"
    assert widget.acquisition_group.isHidden()
    widget.sequence_live_preview.setChecked(True)
    widget.sequence_source.setCurrentIndex(1)
    assert not widget.acquisition_group.isHidden()
    assert widget.acquisition_group.isEnabled()
    widget.read_matrix.setValue(6)
    widget.phase_matrix.setValue(4)
    widget.sampling_bandwidth_khz.setValue(25.0)
    widget.epi_spoiler_cycles_per_slice.setValue(6.0)
    widget.epi_spoiler_cycles_per_voxel.setValue(0.25)
    widget.epi_spoiler_duration_ms.setValue(2.0)
    app.processEvents()

    assert widget.acquisition is not None
    assert widget.program.source == "internal-cartesian-epi"
    assert widget.acquisition.read_matrix == 6
    assert widget.acquisition.phase_matrix == 4
    assert widget.acquisition.dwell_s == 40e-6
    definitions = widget.program.metadata["definitions"]
    assert definitions["SpoilAfterSlice"]
    assert definitions["SpoilerCyclesPerSlice"] == pytest.approx(6.0)
    assert definitions["SpoilerCyclesPerVoxel"] == pytest.approx(0.25)
    assert definitions["SpoilerDuration"] == pytest.approx(2e-3)
    assert definitions["SpoilerAxes"] == "xyz"
    assert len(definitions["SpoilerEndTimes"]) == 1
    compiled = SequenceCompiler().compile(widget.program)
    assert compiled.adc_times_s.size == 24
    widget.acquisition.validate_gradient_moments(compiled.adc_gradient_moment_cyc_per_m)
    assert "25.000 kHz" in widget.sequence_info.text()
    assert "Spoilers: 1" in widget.sequence_info.text()

    widget.epi_spoil_after_slice.setChecked(False)
    app.processEvents()
    assert not widget.epi_spoiler_cycles_per_slice.isEnabled()
    assert widget.program.metadata["definitions"]["SpoilerAxes"] == "none"
    widget.epi_spoil_after_slice.setChecked(True)

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
    assert widget.spectrum_x_slider.isHidden()
    assert widget.spectrum_y_slider.isHidden()
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
    assert widget.split_view_checkbox.isChecked()
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


def test_generated_sequence_fov_warning_lists_undersized_in_plane_axes(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(1)
    widget.epi_read_fov_mm.setValue(100.0)
    widget.epi_phase_fov_mm.setValue(150.0)
    widget.epi_slice_thickness_mm.setValue(5.0)
    widget.epi_slice_count.setValue(2)
    widget.phantom = type("PhantomExtent", (), {"fov": (0.2, 0.15, 0.03)})()
    warning = {}

    def capture_warning(_parent, title, message, buttons, default):
        warning.update(
            title=title,
            message=message,
            buttons=buttons,
            default=default,
        )
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "warning", capture_warning)

    assert not widget._confirm_generated_sequence_fov()
    assert warning["title"] == "Sequence FOV is smaller than the phantom"
    assert "Read / x: sequence 100 mm < phantom 200 mm" in warning["message"]
    assert "Phase / y" not in warning["message"]
    assert "Slice / partition / z" not in warning["message"]
    assert warning["buttons"] == QMessageBox.Yes | QMessageBox.No
    assert warning["default"] == QMessageBox.No

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_generated_sequence_fov_warning_ignores_slice_extent(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(1)
    widget.epi_read_fov_mm.setValue(200.0)
    widget.epi_phase_fov_mm.setValue(150.0)
    widget.epi_slice_thickness_mm.setValue(5.0)
    widget.epi_slice_count.setValue(2)
    widget.phantom = type("PhantomExtent", (), {"fov": (0.2, 0.15, 0.03)})()
    warning = MagicMock()
    monkeypatch.setattr(QMessageBox, "warning", warning)

    assert widget._confirm_generated_sequence_fov()
    warning.assert_not_called()

    widget.close()
    widget.deleteLater()
    app.processEvents()


@pytest.mark.parametrize(
    "source_index,expected_fov_m",
    [
        (1, (0.12, 0.13, 0.014)),
        (2, (0.14, 0.15, 0.016)),
        (3, (0.17, 0.18, 0.19)),
    ],
)
def test_generated_sequence_fov_uses_current_controls(source_index, expected_fov_m):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.epi_read_fov_mm.setValue(120.0)
    widget.epi_phase_fov_mm.setValue(130.0)
    widget.epi_slice_thickness_mm.setValue(7.0)
    widget.epi_slice_count.setValue(2)
    widget.csi_read_fov_mm.setValue(140.0)
    widget.csi_phase_fov_mm.setValue(150.0)
    widget.csi_slice_thickness_mm.setValue(16.0)
    widget.bssfp_read_fov_mm.setValue(170.0)
    widget.bssfp_phase_fov_mm.setValue(180.0)
    widget.bssfp_partition_fov_mm.setValue(190.0)
    widget.sequence_source.setCurrentIndex(source_index)

    assert widget._generated_sequence_fov_m() == pytest.approx(expected_fov_m)

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
    assert widget.probe_group.isHidden()

    widget.object_source.setCurrentIndex(1)
    assert widget.object_type.currentText() == "Uniform cube"
    assert widget.object_type.isEnabled()
    assert widget.t1_ms.isEnabled()
    assert not widget.built_in_properties_group.isHidden()
    assert widget.phantom_summary.isHidden()
    assert widget.probe_group.isHidden()

    widget.object_source.setCurrentIndex(2)
    assert widget.object_source.currentText() == "Spin probe"
    assert widget.built_in_properties_group.isHidden()
    assert widget.phantom_summary.isHidden()
    assert not widget.probe_group.isHidden()
    assert not widget.probe_controls.isHidden()
    assert widget.run_button.isHidden()
    assert widget.progress.isHidden()

    widget.object_source.setCurrentIndex(0)
    assert widget.probe_group.isHidden()
    assert not widget.run_button.isHidden()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_builds_multislice_repeated_epi_from_controls():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_live_preview.setChecked(True)
    widget.sequence_source.setCurrentIndex(1)
    widget.read_matrix.setValue(4)
    widget.phase_matrix.setValue(4)
    widget.epi_flip_angle_deg.setValue(30.0)
    widget.epi_slice_count.setValue(2)
    widget.epi_repetitions.setValue(3)
    widget.epi_repetition_time_ms.setValue(100.0)
    widget.epi_slice_thickness_mm.setValue(4.0)
    widget.epi_slice_gap_mm.setValue(2.0)
    app.processEvents()

    compiled = SequenceCompiler().compile(widget.program)
    dimensions = AcquisitionDimensions.from_program(widget.program)
    definitions = widget.program.metadata["definitions"]

    assert widget.program.duration_s == pytest.approx(300e-3)
    assert compiled.adc_times_s.size == 4 * 4 * 2 * 3
    assert dimensions.varying_axes == ("slice", "repetition")
    assert widget.acquisition_frames.num_frames == 6
    assert widget.acquisition_frames.varying_axes == ("slice", "repetition")
    assert widget.frame_selector.count() == 7
    assert widget.frame_selector.itemText(6) == "slice=1, repetition=2"
    assert definitions["SliceThickness"] == pytest.approx(4e-3)
    assert definitions["SliceGap"] == pytest.approx(2e-3)
    assert definitions["SliceSpacing"] == pytest.approx(6e-3)
    assert definitions["SlicePositions"] == pytest.approx((-3e-3, 3e-3))
    assert definitions["FOV"] == pytest.approx((0.22, 0.22, 10e-3))
    assert definitions["FlipAngleDeg"] == pytest.approx(30.0)
    assert definitions["Repetitions"] == 3
    assert definitions["RepetitionTime"] == pytest.approx(100e-3)
    assert len({event.frequency_offset_hz for event in widget.program.rf_events}) == 2
    rf = widget.program.rf_events[0]
    assert 360.0 * abs(np.sum(rf.samples_hz) * rf.raster_s) == pytest.approx(30.0)
    assert "frames=6 (slice, repetition)" in widget.sequence_info.text()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_configures_rf_pulse_for_epi_and_spiral(tmp_path):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_live_preview.setChecked(True)
    widget.sequence_source.setCurrentIndex(1)
    widget.read_matrix.setValue(4)
    widget.phase_matrix.setValue(3)
    widget.epi_repetition_time_ms.setValue(100.0)
    widget.epi_flip_angle_deg.setValue(35.0)
    widget.epi_rf_pulse_type.setCurrentText("SLR")
    widget.epi_rf_duration_ms.setValue(2.5)
    widget.epi_rf_time_bandwidth_product.setValue(3.5)
    widget.epi_rf_slr_sharpness.setCurrentText("5")
    app.processEvents()

    definitions = widget.program.metadata["definitions"]
    rf = widget.program.rf_events[0]
    assert definitions["RFPulseType"] == "slr"
    assert definitions["RFDuration"] == pytest.approx(2.5e-3)
    assert definitions["RFTimeBandwidthProduct"] == pytest.approx(3.5)
    assert definitions["RFSLRSharpness"] == pytest.approx(5.0)
    assert rf.samples_hz.size * rf.raster_s == pytest.approx(2.5e-3)
    assert 360.0 * abs(np.sum(rf.samples_hz) * rf.raster_s) == pytest.approx(35.0)
    assert widget.epi_rf_time_bandwidth_product.isEnabled()
    assert not widget.epi_rf_apodization.isEnabled()
    assert widget.epi_rf_slr_sharpness.isEnabled()

    widget.epi_readout_trajectory.setCurrentText("Spiral")
    app.processEvents()
    assert widget.program.metadata["definitions"]["RFPulseType"] == "slr"
    assert widget.program.metadata["definitions"]["RFDuration"] == pytest.approx(2.5e-3)
    output = widget._write_pulseq_path(tmp_path / "spiral_slr.seq")
    exported = load_pulseq(output).metadata["definitions"]
    assert exported["RFPulseType"] == "slr"
    assert exported["RFSLRSharpness"] == pytest.approx(5.0)

    widget.epi_rf_pulse_type.setCurrentText("Block")
    app.processEvents()
    assert not widget.epi_rf_time_bandwidth_product.isEnabled()
    assert not widget.epi_rf_apodization.isEnabled()
    assert not widget.epi_rf_slr_sharpness.isEnabled()
    assert widget.program.metadata["definitions"]["RFPulseType"] == "block"
    assert widget.program.metadata["definitions"][
        "RFTimeBandwidthProduct"
    ] == pytest.approx(1.0)

    designer_raster_s = 10e-6
    designer_samples = 100
    designer_flip_angle_deg = 30.0
    designer_waveform_hz = np.full(
        designer_samples,
        designer_flip_angle_deg
        / (360.0 * designer_samples * designer_raster_s)
        * np.exp(1j * np.pi / 4.0),
        dtype=np.complex128,
    )
    widget.set_rf_designer_pulse(
        (
            rf_hz_to_gauss(designer_waveform_hz),
            np.arange(designer_samples) * designer_raster_s,
        ),
        {
            "duration": designer_samples * designer_raster_s * 1000.0,
            "flip_angle": designer_flip_angle_deg,
            "pulse_type": "Gaussian",
            "freq_offset": 75.0,
        },
    )
    widget.epi_rf_pulse_type.setCurrentText("RF Pulse Designer")
    app.processEvents()
    definitions = widget.program.metadata["definitions"]
    rf_integral = (
        np.sum(widget.program.rf_events[0].samples_hz)
        * widget.program.rf_events[0].raster_s
    )
    assert definitions["RFPulseType"] == "designer"
    assert definitions["RFDesignerPulseName"] == "Gaussian"
    assert definitions["RFDesignerFlipAngleDeg"] == pytest.approx(30.0)
    assert definitions["RFFrequencyOffset"] == pytest.approx(75.0)
    assert not widget.epi_rf_duration_ms.isEnabled()
    assert widget.epi_rf_duration_ms.value() == pytest.approx(1.0)
    assert 360.0 * abs(rf_integral) == pytest.approx(35.0)
    assert np.angle(rf_integral) == pytest.approx(np.pi / 4.0)

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_builds_spiral_readout_from_controls(tmp_path):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_live_preview.setChecked(True)
    widget.sequence_source.setCurrentIndex(1)
    widget.read_matrix.setValue(8)
    widget.phase_matrix.setValue(8)
    widget.epi_slice_count.setValue(2)
    widget.epi_slice_thickness_mm.setValue(4.0)
    widget.epi_slice_gap_mm.setValue(2.0)
    widget.epi_repetitions.setValue(2)
    widget.epi_repetition_time_ms.setValue(100.0)
    widget.epi_spiral_turns.setValue(4.0)
    widget.epi_readout_trajectory.setCurrentText("Spiral")
    app.processEvents()

    assert widget.program.source == "internal-spiral"
    assert widget.acquisition is None
    assert widget.spiral_acquisition is not None
    assert widget.spiral_acquisition.matrix == (8, 8)
    assert widget.spiral_acquisition.num_frames == 4
    assert widget.spiral_acquisition.varying_axes == ("slice", "repetition")
    assert widget.frame_selector.count() == 5
    definitions = widget.program.metadata["definitions"]
    assert definitions["Trajectory"] == "spiral"
    assert definitions["SpiralTurns"] == pytest.approx(4.0)
    assert definitions["SliceGap"] == pytest.approx(2e-3)
    assert definitions["FOV"] == pytest.approx((0.22, 0.22, 10e-3))
    assert "Spiral:" in widget.sequence_info.text()
    output = widget._write_pulseq_path(tmp_path / "interactive_spiral.seq")
    exported = SequenceCompiler().compile(load_pulseq(output))
    assert exported.adc_times_s.size == 8 * 8 * 2 * 2

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_displays_cartesian_kspace_and_reconstruction():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.sequence_live_preview.setChecked(True)
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


def test_sequence_workspace_infers_imported_epi_and_syncs_fov(tmp_path, monkeypatch):
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
    original_compile_acquisition = SequenceCompiler.compile_acquisition
    compile_calls = []

    def counting_compile_acquisition(compiler, *args, **kwargs):
        compile_calls.append(1)
        return original_compile_acquisition(compiler, *args, **kwargs)

    monkeypatch.setattr(
        SequenceCompiler, "compile_acquisition", counting_compile_acquisition
    )
    widget._load_pulseq_path(path)
    widget.matrix_size.setValue(4)
    widget.z_matrix_size.setValue(4)

    assert widget.acquisition is not None
    assert widget.acquisition.read_matrix == 4
    assert widget.acquisition.phase_matrix == 4
    assert len(compile_calls) == 1
    assert widget.epi_read_fov_mm.suffix() == " mm"
    assert widget.fov_mm.suffix() == " mm"
    assert widget.fov_z_mm.suffix() == " mm"
    assert widget.fov_mm.value() == pytest.approx(220.0)
    assert widget.fov_z_mm.value() == pytest.approx(4.0)

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


def test_sequence_workspace_selects_position_sorted_3d_volumes(tmp_path):
    path = tmp_path / "cartesian_3d.seq"
    SPECTRAL_3D_MAIN(
        write_seq=True,
        seq_filename=str(path),
        n_read=4,
        n_phase=2,
        n_partition=2,
        n_repetition=2,
        dummy_repetitions=0,
        use_alpha_half=False,
        target_tr=8e-3,
    )
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget._load_pulseq_path(path)
    compiled = SequenceCompiler().compile(widget.program)
    result = SequenceSimulationResult(
        signal=np.ones(compiled.adc_times_s.size, dtype=np.complex128),
        adc_times_s=compiled.adc_times_s,
        final_magnetization=np.zeros((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        adc_gradient_moment_cyc_per_m=compiled.adc_gradient_moment_cyc_per_m,
        metadata={
            "acquisition_dimensions": widget.acquisition_frames.dimensions.to_metadata(),
            "cartesian_acquisition_frames": widget.acquisition_frames.to_metadata(),
            "cartesian_acquisition_volumes": widget.acquisition_volumes.to_metadata(),
        },
    )
    widget.result = result
    widget._configure_frame_selector()
    widget._show_cartesian_result(result)
    app.processEvents()

    assert widget.acquisition_volumes.matrix == (4, 2, 2)
    assert widget.frame_selector.count() == 3
    assert widget.frame_selector.itemText(0) == "All 2 volumes (montage)"
    assert widget.frame_selector.itemText(2) == "repetition=1"
    assert widget.kspace_view.image.shape == (9, 2)
    assert widget.reconstruction_view.image.shape == (9, 2)
    assert "3D |IFFT3|" in widget.reconstruction_info.text()
    widget.frame_selector.setCurrentIndex(2)
    assert widget.kspace_view.image.shape == (4, 2)
    assert widget.reconstruction_view.image.shape == (4, 2)
    assert "repetition=1" in widget.reconstruction_info.text()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_keeps_run_button_outside_scroll_area():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.resize(900, 420)
    widget.show()
    app.processEvents()

    assert widget.run_button.isVisible()
    assert not widget.controls_scroll.viewport().isAncestorOf(widget.run_button)
    assert widget.output_group.isHidden()
    assert widget.probe_group.isHidden()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_focused_sequence_workspace_uses_wider_control_panel():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.resize(1800, 900)
    widget.show()
    app.processEvents()

    widget.activate_focused_workspace_layout()
    app.processEvents()

    control_width, viewer_width = widget.workspace_splitter.sizes()
    expected_control_width = min(
        widget.FOCUSED_CONTROL_WIDTH,
        max(
            widget.MINIMUM_FOCUSED_CONTROL_WIDTH,
            round(widget.workspace_splitter.width() * 0.30),
        ),
    )
    assert control_width == expected_control_width
    assert widget.FOCUSED_CONTROL_WIDTH == 420
    assert viewer_width >= widget.MINIMUM_FOCUSED_VIEWER_WIDTH
    assert widget.layout().contentsMargins().left() == 0
    assert widget.split_view_checkbox.parentWidget() is widget.signal_page
    assert widget.views.tabBar().font().bold()
    assert widget.sequence_title.font().bold()
    assert widget.sequence_title.font().pointSize() >= 12
    assert widget.object_form.labelAlignment() & Qt.AlignLeft
    for image_view in (
        widget.kspace_view,
        widget.reconstruction_view,
        widget.state_view,
    ):
        assert image_view.ui.histogram.width() == 48

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_starting_sequence_opens_signal_tab(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.views.setCurrentIndex(0)
    widget.view_stack.setCurrentIndex(1)
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.SequenceSimulationThread.start",
        lambda _worker: None,
    )

    widget._run()

    assert widget.view_stack.currentWidget() is widget.normal_signal_page
    assert widget.views.currentIndex() == widget.signal_tab_index
    assert widget.views.tabText(widget.views.currentIndex()) == "Signal / CSI spectrum"

    widget.worker.deleteLater()
    widget.worker = None
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_builds_geometry_probe_positions():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.matrix_size.setValue(4)
    widget.z_matrix_size.setValue(3)
    widget._build_phantom()
    widget.probe_max_positions.setValue(5)

    positions = widget._probe_geometry_positions_m()
    ppm_axis, hz_axis = widget._probe_single_frequency_axis_hz()

    assert positions.shape == (5, 3)
    assert np.all(np.isfinite(positions))
    assert ppm_axis.shape == (1,)
    assert hz_axis.shape == (1,)
    assert widget.probe_frequency_units.currentText() == "Hz"
    assert widget.probe_ppm_min.value() == pytest.approx(-2500.0)
    assert widget.probe_ppm_max.value() == pytest.approx(2500.0)
    cancel_position = widget.probe_button_layout.getItemPosition(
        widget.probe_button_layout.indexOf(widget.cancel_probe_button)
    )
    spectral_position = widget.probe_button_layout.getItemPosition(
        widget.probe_button_layout.indexOf(widget.run_probe_button)
    )
    assert cancel_position[:2] == (1, 0)
    assert spectral_position[:2] == (0, 0)
    assert widget.probe_initial_mz.maximum() == pytest.approx(1e7)
    widget.probe_initial_mz.setValue(2.5e6)
    assert widget.probe_initial_mz.value() == pytest.approx(2.5e6)

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_passes_large_initial_mz_to_probe_worker(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.SequenceProbeThread.start",
        lambda _worker: None,
    )
    widget.probe_initial_mz.setValue(3e6)

    widget._start_probe(
        positions=np.array([[0.0, 0.0, 0.0]]),
        hz_axis=np.array([0.0]),
        display_axis=np.array([0.0]),
        checkpoints=np.array([0.0]),
        label="spectral",
    )

    assert widget.probe_worker.initial_magnetization == pytest.approx((0.0, 0.0, 3e6))
    widget.probe_worker.deleteLater()
    widget.probe_worker = None
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_probe_defaults_to_individual_rf_event_ends():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.program = widget.program.__class__(
        events=(
            RFEvent(0.001, np.array([100.0, 200.0]), 1e-3),
            RFEvent(0.010, np.array([300.0]), 2e-3),
        ),
        duration_s=0.020,
        metadata={"definitions": {"EndImageSpoilerEndTimes": [0.018]}},
    )

    assert widget._probe_checkpoints_s() == pytest.approx(
        [0.0, 0.003, 0.012, 0.018, 0.020]
    )
    widget.probe_time_sampling.setCurrentText("Uniform timeline")
    widget.probe_time_points.setValue(5)
    assert widget._probe_checkpoints_s() == pytest.approx(
        [0.0, 0.005, 0.010, 0.015, 0.020]
    )

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_plot_data_retains_event_extrema_and_resolves_zoom():
    samples = np.linspace(-10.0, 10.0, 1001)
    event = GradientEvent("x", 0.0, samples, 1e-6)

    overview_x, overview_y = _event_step_plot_data(
        (event,),
        samples_attribute="samples_hz_per_m",
        start_s=0.0,
        end_s=event.end_s,
        max_vertices=60,
    )
    zoom_x, zoom_y = _event_step_plot_data(
        (event,),
        samples_attribute="samples_hz_per_m",
        start_s=0.0004,
        end_s=0.0006,
        max_vertices=6000,
    )

    assert np.nanmin(overview_y) == pytest.approx(-10.0)
    assert np.nanmax(overview_y) == pytest.approx(10.0)
    assert np.count_nonzero(np.isfinite(zoom_y)) > np.count_nonzero(
        np.isfinite(overview_y)
    )
    assert np.all(np.diff(overview_x[np.isfinite(overview_x)]) >= 0)


def test_sequence_plot_data_caps_vertices_when_many_events_are_visible():
    events = tuple(
        GradientEvent("x", index * 2e-6, np.asarray([float(index)]), 1e-6)
        for index in range(2000)
    )

    x, y = _event_step_plot_data(
        events,
        samples_attribute="samples_hz_per_m",
        start_s=0.0,
        end_s=events[-1].end_s,
        max_vertices=900,
    )

    assert x.size <= 900
    assert y.size == x.size


def test_sequence_workspace_caps_overview_adc_markers():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.program = SequenceProgram((ADCEvent(0.0, 10_000, 1e-6),), duration_s=10e-3)
    widget._acquisition_compiled = None

    widget._show_program()

    adc_items = [
        item
        for item in widget.gradient_plot.listDataItems()
        if item.opts.get("symbol") == "o"
    ]
    assert len(adc_items) == 1
    assert adc_items[0].xData.size <= 5000
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_can_display_physical_b1_and_gradient_units():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    gamma = NUCLEUS_GAMMA_HZ_PER_T["H1"]
    widget.nucleus.setCurrentText("H1")
    widget.program = SequenceProgram(
        (
            RFEvent(0.0, np.asarray([gamma / 1e4]), 1e-3),
            GradientEvent("x", 0.0, np.asarray([gamma * 0.02]), 1e-3),
        ),
        duration_s=1e-3,
    )
    widget._acquisition_compiled = None

    widget._show_program()

    assert widget.waveform_units.currentData() == "physical"
    assert "nominal peak B1 1 G" in widget.waveform_value_summary.text()
    assert "Gx 0.02" in widget.waveform_value_summary.text()
    assert np.nanmax(widget._rf_waveform_item.yData) == pytest.approx(1.0)
    assert np.nanmax(widget._gradient_waveform_items["x"].yData) == pytest.approx(0.02)

    widget.waveform_units.setCurrentIndex(widget.waveform_units.findData("simulation"))
    assert np.nanmax(widget._rf_waveform_item.yData) == pytest.approx(gamma / 1e4)
    assert np.nanmax(widget._gradient_waveform_items["x"].yData) == pytest.approx(
        gamma * 0.02 * 1e-3
    )
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_displays_geometry_probe_result(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    playback_clock = [100.0]
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.time.monotonic",
        lambda: playback_clock[0],
    )
    positions = np.array(
        [
            [-0.01, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
        ],
        dtype=float,
    )
    result = SequenceProbeResult(
        time_s=np.array([0.0, 0.01]),
        positions_m=positions,
        frequency_offsets_hz=np.array([0.0]),
        magnetization=np.ones((2, 3, 1, 3), dtype=float),
        metadata={"probe_type": "geometry", "frequency_offsets_ppm": np.array([0.0])},
    )

    widget.probe_result = result
    widget._show_probe_result()

    assert "Geometry probe" in widget.probe_info.text()
    assert widget.probe_spatial_viewer.result is result
    assert widget.probe_spectrum_viewer.result is result
    assert widget.probe_spatial_viewer.mxy_plot.listDataItems()
    assert widget.probe_time_control.isEnabled()
    assert widget.probe_time_control.time_slider.maximum() == 1
    assert widget.probe_time_control.time_slider.value() == 1

    widget.probe_time_control.time_slider.setValue(0)
    assert widget.probe_spectrum_viewer.time_index == 0
    assert widget.probe_spatial_viewer.time_index == 0
    assert widget.probe_magnetization_viewer.time_slider.value() == 0

    widget.probe_time_control.speed_spin.setValue(2.5)
    widget.probe_time_control.play_pause_button.setChecked(True)
    assert widget.probe_playback_timer.isActive()
    assert widget.probe_time_control.play_pause_button.text() == "Pause"

    playback_clock[0] += 4.0
    widget._advance_probe_playback()
    assert widget.probe_time_control.time_slider.value() == 1

    widget.probe_time_control.reset_button.click()
    assert not widget.probe_playback_timer.isActive()
    assert widget.probe_time_control.time_slider.value() == 0
    assert widget.probe_time_control.play_pause_button.text() == "Play"

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_probe_playback_modes_map_complete_timeline_and_adc_status():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    times_s = np.array([0.0, 0.01, 0.02, 0.021, 0.05, 0.08, 0.081, 0.1])
    result = SequenceProbeResult(
        time_s=times_s,
        positions_m=np.array([[0.0, 0.0, 0.0]]),
        frequency_offsets_hz=np.array([0.0]),
        magnetization=np.ones((times_s.size, 1, 1, 3), dtype=float),
        metadata={
            "probe_type": "geometry",
            "configured_playback_times_s": np.array([0.0, 0.01, 0.1]),
            "adc_times_s": np.array([0.02, 0.021, 0.08, 0.081]),
            "adc_event_indices": np.array([0, 0, 1, 1]),
            "adc_sample_dwell_s": np.full(4, 0.001),
            "adc_windows_s": np.array([[0.02, 0.022], [0.08, 0.082]]),
        },
    )

    widget.probe_result = result
    widget._show_probe_result()

    assert widget.probe_playback_mode.currentText() == "Configured checkpoints"
    assert np.array_equal(widget._probe_playback_indices, [0, 1, 7])
    assert widget.probe_time_control.time_slider.maximum() == 2

    widget.probe_playback_mode.setCurrentText("ADC only")
    assert np.array_equal(widget._probe_playback_indices, [2, 3, 5, 6])
    assert widget.probe_time_control.time_slider.maximum() == 3
    assert widget._probe_playback_clock_ms == pytest.approx([0.0, 1.0, 2.0, 3.0])
    widget.probe_time_control.time_slider.setValue(2)
    assert widget.probe_spectrum_viewer.time_index == 5

    widget.probe_playback_mode.setCurrentText("All simulation steps")
    assert np.array_equal(widget._probe_playback_indices, np.arange(times_s.size))
    assert widget.probe_time_control.time_slider.maximum() == times_s.size - 1
    assert not widget.probe_adc_status.isHidden()
    widget.probe_time_control.time_slider.setValue(4)
    assert widget.probe_adc_status.text() == "ADC: off"
    widget.probe_time_control.time_slider.setValue(5)
    assert widget.probe_adc_status.text() == "ADC: on"

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_probe_thread_stores_every_compiled_simulation_state():
    app = QApplication.instance() or QApplication(sys.argv)
    program = SequenceProgram(
        events=(
            RFEvent(0.0, np.array([100.0, 100.0]), 0.001),
            ADCEvent(0.005, 1, 0.001),
        ),
        duration_s=0.01,
    )

    class ProbeSimulator:
        received_checkpoints = None

        def simulate_sequence_probes(self, _program, positions, frequencies, **kwargs):
            self.received_checkpoints = np.asarray(kwargs["checkpoints_s"])
            return SequenceProbeResult(
                time_s=self.received_checkpoints,
                positions_m=np.asarray(positions),
                frequency_offsets_hz=np.asarray(frequencies),
                magnetization=np.ones(
                    (self.received_checkpoints.size, 1, 1, 3), dtype=float
                ),
            )

    simulator = ProbeSimulator()
    worker = SequenceProbeThread(
        simulator,
        program,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([0.0]),
        np.array([0.0, 0.002, 0.01]),
        1.0,
        0.1,
        simulation_timestep_s=0.001,
    )
    completed = []
    worker.result_ready.connect(completed.append)

    worker.run()

    assert simulator.received_checkpoints == pytest.approx(
        [0.0, 0.001, 0.002, 0.005, 0.01]
    )
    assert completed[0].metadata["stored_timeline"] == "all_simulation_steps"
    assert completed[0].metadata["configured_playback_times_s"] == pytest.approx(
        [0.0, 0.002, 0.01]
    )
    assert completed[0].metadata["adc_times_s"] == pytest.approx([0.005])

    worker.deleteLater()
    app.processEvents()
