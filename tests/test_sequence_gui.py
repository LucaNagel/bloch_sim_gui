import sys
import runpy
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import QApplication

from blochsimulator.ui.main_window import BlochSimulatorGUI
from blochsimulator.ui.sequence_simulation_widget import (
    SequenceSimulationWidget,
    _event_step_plot_data,
)
from blochsimulator.sequence import (
    AcquisitionDimensions,
    GradientEvent,
    RFEvent,
    SequenceCompiler,
    SequenceProbeResult,
    SequenceSimulationResult,
    SpectroscopicAcquisition,
)


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
    window.app_settings.setValue("sequence/timestep_preset", "fast")
    window.app_settings.setValue("sequence/timestep_us", 10.0)
    window.app_settings.setValue("simulation/thread_mode", "manual")
    window.app_settings.setValue("simulation/manual_threads", 2)
    assert window.sequence_simulation_widget is None
    window.tab_widget.setCurrentIndex(window.sequence_simulation_tab_index)
    app.processEvents()

    assert window.sequence_simulation_widget.simulation_timestep_us.value() == 10.0
    assert window.sequence_simulation_widget.simulator.sequence_kernel == "reference"
    assert window.sequence_simulation_widget.simulator.num_threads == 2
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


def test_sequence_workspace_builds_multislice_repeated_epi_from_controls():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(1)
    widget.read_matrix.setValue(4)
    widget.phase_matrix.setValue(4)
    widget.epi_flip_angle_deg.setValue(30.0)
    widget.epi_slice_count.setValue(2)
    widget.epi_repetitions.setValue(3)
    widget.epi_repetition_time_ms.setValue(100.0)
    widget.epi_slice_thickness_mm.setValue(4.0)
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
    assert not widget.probe_group.isChecked()
    assert widget.probe_controls.isHidden()

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
