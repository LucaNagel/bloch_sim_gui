import sys
import runpy
import time
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PyQt5.QtCore import QSettings, Qt, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QGroupBox,
    QLabel,
    QMenu,
    QMessageBox,
    QScrollArea,
    QSpinBox,
    QToolBar,
    QWidget,
)

from blochsimulator.ui.main_window import BlochSimulatorGUI
from blochsimulator.ui.magnetization_viewer import MagnetizationViewer
from blochsimulator.ui.sequence_simulation_widget import (
    SequenceProbeThread,
    SequenceSimulationThread,
    SequenceSimulationWidget,
    _SequenceSimulationPayload,
    _animation_checkpoint_times,
    _event_step_plot_data,
    _rf_phase_plot_data,
    _sequence_preview_range_ms,
    _sequence_preview_timing_text,
    _split_animation_result,
)
from blochsimulator.ui.volume_viewer import SequenceMagnetizationAnimationViewer
from blochsimulator.ui.default_settings import WorkspaceDefaults
from blochsimulator.phantom import Phantom
from blochsimulator.project_io import load_project
from blochsimulator.simulator import BlochSimulator
from blochsimulator.spectral_phantom import ChemicalSpecies, SpectralPhantom
from blochsimulator.sequence import (
    ADCEvent,
    AcquisitionDimensions,
    CartesianAcquisition,
    EncodingFrame,
    GradientEvent,
    RFEvent,
    SequenceCompiler,
    SequenceProbeResult,
    SequenceProgram,
    SequenceSimulationResult,
    SpectroscopicAcquisition,
    load_pulseq,
    make_pulseq_bssfp,
    make_pulseq_flash,
)
from blochsimulator.sequence.scanner import ScannerParameters
from blochsimulator.sequence.rf_pulses import (
    slr_time_bandwidth_product,
)
from blochsimulator.units import NUCLEUS_GAMMA_HZ_PER_T, rf_hz_to_gauss


def test_3d_vector_viewer_shows_equilibrium_before_simulation(qt_application):
    viewer = MagnetizationViewer()

    np.testing.assert_allclose(
        viewer.vector_plot.pos, [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(
        viewer.mean_vector.pos, [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    assert viewer._has_vector_data
    assert viewer._has_mean_vector
    assert not viewer.path_points


def _select_workspace(window, mode, timeout_ms=2_000, *, use_popup=False):
    """Select a workspace and wait until its deferred transition completes."""
    index = window.workspace_mode_selector.findData(mode)
    assert index >= 0
    if use_popup:
        selector = window.workspace_mode_selector
        QTest.mouseClick(selector, Qt.LeftButton)
        view = selector.view()
        assert view.isVisible()
        item = selector.model().index(index, 0)
        # On X11, Qt can suppress the synthetic mouse release following the
        # click that opens the combo box. Select the visible popup item through
        # its keyboard path, which exercises the same popup-acceptance logic
        # without relying on that platform-specific pointer timing.
        view.setCurrentIndex(item)
        QTest.keyClick(view, Qt.Key_Return)
    else:
        window.workspace_mode_selector.setCurrentIndex(index)

    deadline = time.monotonic() + timeout_ms / 1_000
    while window.workspace_mode != mode and time.monotonic() < deadline:
        QTest.qWait(10)

    assert window.workspace_mode == mode


def test_sequence_widget_accepts_strict_reference_kernel():
    widget = SimpleNamespace(simulator=SimpleNamespace(sequence_kernel="optimized"))

    SequenceSimulationWidget.set_sequence_kernel(widget, "strict_reference")

    assert widget.simulator.sequence_kernel == "strict_reference"
    with pytest.raises(ValueError, match="strict_reference"):
        SequenceSimulationWidget.set_sequence_kernel(widget, "invalid")


@pytest.mark.parametrize("source", ["epi", "flash", "epsi"])
def test_timing_feedback_moves_to_run_status_and_survives_in_log(source, monkeypatch):
    widget = SequenceSimulationWidget()
    show_timing_status = getattr(widget, f"_show_{source}_timing_status")
    plan = SimpleNamespace(
        minimum_echo_time_s=0.01236,
        minimum_repetition_time_s=0.03677,
        minimum_echo_spacing_s=0.002,
    )
    notes = ["Bandwidth 150 → 142.858 kHz"]
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.time.strftime",
        lambda _format: "2026-10-20 12:34:56",
    )

    show_timing_status(plan, notes)
    warning = widget.status.text()
    assert "Adjusted automatically: Bandwidth 150 → 142.858 kHz." in warning
    assert "Shortest possible:" in warning
    assert not widget.findChildren(QLabel, f"{source}_timing_info")
    log = widget.log_widget.toPlainText()
    assert "[2026-10-20 12:34:56]" in log
    assert "timing [Warning]: " + warning in log

    # Settle timers and live previews can report the same correction repeatedly.
    show_timing_status(plan, notes)
    assert widget.log_widget.toPlainText() == log

    widget._status_update("Preparing and compiling sequence…")
    assert widget.status.text() == "Preparing and compiling sequence…"
    show_timing_status(plan, notes)
    assert widget.status.text() == "Preparing and compiling sequence…"
    assert widget.log_widget.toPlainText().count(warning) == 1
    assert "Sequence simulation: Preparing and compiling sequence…" in (
        widget.log_widget.toPlainText()
    )

    show_timing_status(None, [], error="ADC duration <limit>")
    assert widget.status.text() == "Timing cannot be resolved: ADC duration <limit>"
    assert "timing [Error]: Timing cannot be resolved: ADC duration <limit>" in (
        widget.log_widget.toPlainText()
    )


def test_sequence_tab_order_and_log_shares_application_history(qt_application):
    window = BlochSimulatorGUI()
    window.show()
    window.log_message("Message before opening Sequence Mode")
    window.set_workspace_mode("sequence")
    widget = window.sequence_simulation_widget

    visible_tabs = [
        widget.views.tabText(index)
        for index in range(widget.views.count())
        if widget.views.isTabVisible(index)
    ]
    assert visible_tabs == [
        "Sequence",
        "Geometry",
        "Reconstruction Explorer",
        "Spin Probe",
        "Signal",
        "Spatial Magnetization",
        "3D Magnetization Animation",
        "Log",
    ]
    assert widget.views.tabText(widget.log_tab_index) == "Log"
    assert widget.log_widget.isReadOnly()
    assert widget.log_widget.document() is window.log_widget.document()
    assert "Message before opening Sequence Mode" in widget.log_widget.toPlainText()

    widget._show_epi_timing_status(None, [], error="ADC timing exceeds limit")
    assert "ADC timing exceeds limit" in window.log_widget.toPlainText()
    assert window.log_widget.toPlainText().count("ADC timing exceeds limit") == 1
    widget.views.setCurrentIndex(widget.log_tab_index)
    qt_application.processEvents()
    assert widget.log_widget.isVisible()

    window.set_workspace_mode("free")
    window.log_message("Message after switching to Free Mode")
    window.set_workspace_mode("sequence")
    assert "Message after switching to Free Mode" in widget.log_widget.toPlainText()
    assert widget.log_widget.toPlainText() == window.log_widget.toPlainText()


def test_execution_completion_message_reports_backend_contract_and_fallback():
    complete = SequenceSimulationWidget._execution_completion_message(
        {
            "requested_sequence_kernel": "native_parallel",
            "sequence_kernel": "native_parallel",
            "effective_threads": 8,
            "numerical_contract": "float64-close",
            "native_full_plan_enabled": True,
            "native_full_plan_model": "multi-pool-coupled-concentration-inflow",
        }
    )
    assert "used Native automatic" in complete
    assert "8 CPU threads" in complete
    assert "tight Float64 contract" in complete
    assert "complete native plan (multi pool coupled concentration inflow)" in complete

    fallback = SequenceSimulationWidget._execution_completion_message(
        {
            "requested_sequence_kernel": "native_parallel",
            "sequence_kernel": "optimized",
            "effective_threads": 1,
            "numerical_contract": "bitwise",
            "native_fallback_reason": "initial-time ADC requires the CPU path",
        }
    )
    assert "requested Native automatic; used Optimized CPU" in fallback
    assert "1 CPU thread" in fallback
    assert "bitwise contract" in fallback
    assert "fallback: initial-time ADC requires the CPU path" in fallback

    standard = SequenceSimulationWidget._execution_completion_message(
        {"sequence_kernel": "optimized", "numerical_contract": "legacy-fast-math"}
    )
    assert "legacy" not in standard
    assert "Float64 with fast-math (standard C build)" in standard


def test_rf_phase_plot_includes_programmed_phase_and_carrier_evolution():
    event = RFEvent(
        1e-3,
        np.ones(3, dtype=np.complex128),
        1e-3,
        frequency_offset_hz=250.0,
        phase_offset_rad=np.deg2rad(30.0),
    )

    x_ms, phase_deg = _rf_phase_plot_data((event,), start_s=0.0, end_s=5e-3)
    finite = np.isfinite(x_ms) & np.isfinite(phase_deg)

    assert x_ms[finite] == pytest.approx((1.5, 2.5, 3.5))
    assert phase_deg[finite] == pytest.approx((30.0, 120.0, -150.0))


def test_post_run_animation_uses_time_resolution_and_preserves_result():
    app = QApplication.instance() or QApplication(sys.argv)
    program = SequenceProgram(
        events=(RFEvent(0.0, np.full(1, 50.0 + 0.0j), 1e-3),),
        duration_s=6e-3,
    )
    phantom = Phantom(
        shape=(2, 1, 1),
        fov=(0.02, 0.01, 0.01),
        t1_map=np.ones((2, 1, 1)),
        t2_map=np.ones((2, 1, 1)),
        pd_map=np.ones((2, 1, 1)),
    )
    simulator = BlochSimulator(use_parallel=False)
    baseline = simulator.simulate_sequence(
        program,
        phantom,
        checkpoints_s=(2e-3,),
        simulation_timestep_s=1e-3,
    )
    animation_times = _animation_checkpoint_times(
        program,
        time_resolution_s=1e-3,
        maximum_frames=100,
        checkpoints_s=(2e-3,),
        simulation_timestep_s=1e-3,
    )
    assert animation_times == pytest.approx(np.arange(7) * 1e-3)

    worker = SequenceSimulationThread(
        simulator,
        program,
        phantom,
        (2e-3,),
        live_preview=False,
        simulation_timestep_s=1e-3,
        animation_time_resolution_s=1e-3,
        animation_maximum_frames=100,
        animation_storage_dtype="float16",
    )
    completed = []
    progress_updates = []
    worker.result_ready.connect(completed.append)
    worker.progress.connect(lambda done, total: progress_updates.append((done, total)))
    worker.run()

    assert len(completed) == 1
    payload = completed[0]
    assert isinstance(payload, _SequenceSimulationPayload)
    assert np.array_equal(
        payload.result.final_magnetization, baseline.final_magnetization
    )
    assert np.array_equal(payload.result.signal, baseline.signal)
    assert payload.result.checkpoint_times_s == pytest.approx([2e-3])
    assert payload.result.checkpoint_magnetization.dtype == np.float64
    assert payload.animation.magnetization.dtype == np.float16
    assert payload.animation.time_s == pytest.approx(animation_times)
    assert payload.animation.magnetization[-1].astype(np.float64) == pytest.approx(
        baseline.final_magnetization, abs=5e-4
    )
    assert progress_updates[-1][0] == progress_updates[-1][1]
    assert [done for done, _ in progress_updates] == sorted(
        done for done, _ in progress_updates
    )

    worker.deleteLater()
    app.processEvents()


def test_post_run_animation_snaps_rf_active_targets_to_existing_boundaries():
    program = SequenceProgram(
        events=(RFEvent(0.0, np.full(4, 50.0 + 0.0j), 1e-3),),
        duration_s=4e-3,
    )
    compiled = SequenceCompiler().compile(
        program,
        simulation_timestep_s=1e-3,
    )
    animation_times = _animation_checkpoint_times(
        program,
        time_resolution_s=1.3e-3,
        maximum_frames=100,
        simulation_timestep_s=1e-3,
    )
    state_boundaries = np.concatenate(([0.0], compiled.interval_end_s))
    assert all(np.any(state_boundaries == value) for value in animation_times)


def test_post_run_animation_viewer_selects_time_map_and_pool():
    app = QApplication.instance() or QApplication(sys.argv)
    phantom = Phantom(
        shape=(2, 1, 1),
        fov=(0.02, 0.01, 0.01),
        t1_map=np.ones((2, 1, 1)),
        t2_map=np.ones((2, 1, 1)),
        pd_map=np.ones((2, 1, 1)),
    )
    states = np.zeros((3, 2, 1, 1, 3), dtype=np.float16)
    states[:, ..., 2] = np.asarray([1.0, 0.5, 0.0])[:, None, None, None]
    pools = np.zeros((3, 2, 2, 1, 1, 3), dtype=np.float16)
    pools[:, 1, ..., 0] = np.asarray([0.0, 0.25, 0.75])[:, None, None, None]
    viewer = SequenceMagnetizationAnimationViewer()
    assert not viewer.capture_enabled.isChecked()
    assert viewer.time_resolution_ms.value() == pytest.approx(1.0)
    assert viewer.storage_dtype_combo.currentText() == "float32"
    viewer.set_animation(
        np.asarray([0.0, 0.001, 0.002]),
        states,
        phantom=phantom,
        pool_magnetization=pools,
        pool_names=("Pyruvate", "Lactate"),
        storage_dtype="float16",
    )

    assert viewer.time_slider.maximum() == 2
    assert viewer.pool_combo.count() == 3
    assert "float16" in viewer.storage_info.text()
    viewer.pool_combo.setCurrentText("Lactate")
    viewer.map_combo.setCurrentText("Mx / real(Mxy)")
    viewer.time_slider.setValue(2)
    assert viewer.volume.data[:, 0, 0] == pytest.approx([0.75, 0.75])
    assert "2.000 ms" in viewer.time_label.text()
    viewer.play_button.setChecked(True)
    assert viewer.playback_timer.isActive()
    viewer.play_button.setChecked(False)
    assert not viewer.playback_timer.isActive()

    viewer.close()
    viewer.deleteLater()
    app.processEvents()


def test_animation_frame_limit_uses_time_resolution_and_storage_dtype():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.program = SequenceProgram(events=(), duration_s=1.0)
    widget.animation_enabled.setChecked(True)

    class LargeAnimationObject:
        nvoxels = 1_000_000

    widget.phantom = LargeAnimationObject()
    widget.animation_time_resolution_ms.setValue(1.0)
    widget.animation_storage_dtype.setCurrentText("float32")
    _, float32_frames, float32_note = widget._animation_request()
    widget.animation_storage_dtype.setCurrentText("float16")
    _, float16_frames, float16_note = widget._animation_request()

    assert float16_frames > float32_frames
    assert "requested" in float32_note
    assert "requested" in float16_note

    widget.animation_time_resolution_ms.setValue(100.0)
    _, coarse_frames, coarse_note = widget._animation_request()
    assert coarse_frames == 11
    assert coarse_note == ""

    widget.animation_time_resolution_ms.setValue(1.0)
    widget.set_animation_memory_budget_bytes(1024 * 1024**2)
    _, larger_budget_frames, larger_budget_note = widget._animation_request()
    assert larger_budget_frames > float16_frames
    assert "1024 MiB limit" in larger_budget_note

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_animation_float16_fallback_reports_storage_reason():
    times = np.asarray([0.0, 0.001])
    checkpoints = np.zeros((2, 1, 1, 1, 3), dtype=np.float32)
    checkpoints[1, ..., 2] = 70_000.0
    result = SequenceSimulationResult(
        signal=np.zeros(0, dtype=np.complex128),
        adc_times_s=np.zeros(0),
        final_magnetization=checkpoints[-1],
        checkpoint_magnetization=checkpoints,
        checkpoint_times_s=times,
    )

    _, animation = _split_animation_result(
        result,
        user_checkpoints_s=(),
        animation_times_s=times,
        dtype="float16",
    )

    assert animation.storage_dtype == "float32"
    assert "float16 was requested" in animation.storage_note
    assert "7e+04" in animation.storage_note


def test_post_run_animation_separates_pool_frames_from_manual_checkpoints():
    times = np.asarray([0.0, 0.001, 0.002])
    checkpoints = np.arange(3 * 2 * 1 * 1 * 3, dtype=np.float64).reshape(3, 2, 1, 1, 3)
    pool_checkpoints = np.stack((checkpoints, checkpoints + 100.0), axis=1)
    result = SequenceSimulationResult(
        signal=np.zeros(0, dtype=np.complex128),
        adc_times_s=np.zeros(0),
        final_magnetization=checkpoints[-1],
        checkpoint_magnetization=checkpoints,
        checkpoint_times_s=times,
        pool_names=("Pyruvate", "Lactate"),
        checkpoint_pool_magnetization=pool_checkpoints,
    )

    clean, animation = _split_animation_result(
        result,
        user_checkpoints_s=(0.001,),
        animation_times_s=(0.0, 0.002),
        dtype="float32",
    )

    assert clean.checkpoint_times_s == pytest.approx([0.001])
    assert np.array_equal(clean.checkpoint_magnetization, checkpoints[1:2])
    assert np.array_equal(clean.checkpoint_pool_magnetization, pool_checkpoints[1:2])
    assert animation.pool_names == ("Pyruvate", "Lactate")
    assert animation.magnetization.dtype == np.float32
    assert animation.pool_magnetization.dtype == np.float32
    assert np.array_equal(animation.pool_magnetization, pool_checkpoints[[0, 2]])


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
    window.app_settings.setValue("sequence/kernel", "strict_reference")
    window.app_settings.setValue("sequence/dynamic_kernel", "native_parallel")
    window.app_settings.setValue("sequence/timestep_preset", "fast")
    window.app_settings.setValue("sequence/timestep_us", 10.0)
    window.app_settings.setValue("sequence/spoiler_mode", "gradient")
    window.app_settings.setValue("sequence/subvoxel_spins_x", 3)
    window.app_settings.setValue("sequence/subvoxel_spins_y", 5)
    window.app_settings.setValue("sequence/subvoxel_spins_z", 11)
    window.app_settings.setValue("sequence/subvoxel_sampling_method", "stratified")
    # Keep the stored grid instead of the recommended one.
    window.app_settings.setValue("sequence/subvoxel_auto_adjust", False)
    window.app_settings.setValue("simulation/thread_mode", "manual")
    window.app_settings.setValue("simulation/manual_threads", 2)
    window.app_settings.setValue("defaults/sequence_fov_x_mm", 180.0)
    window.app_settings.setValue("defaults/sequence_fov_y_mm", 170.0)
    window.app_settings.setValue("defaults/sequence_fov_z_mm", 80.0)
    window.app_settings.setValue("defaults/field_strength_t", 7.0)
    window.app_settings.setValue("defaults/phantom_nucleus", "C13")
    window.app_settings.setValue("defaults/sequence_reference_ppm/C13", 175.5)
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
    assert (
        window.sequence_simulation_widget.simulator.sequence_kernel
        == "strict_reference"
    )
    assert (
        window.sequence_simulation_widget.simulator.dynamic_sequence_kernel
        == "native_parallel"
    )
    assert window.sequence_simulation_widget.simulator.num_threads == 2
    assert window.sequence_simulation_widget.spoiler_mode == "gradient"
    assert window.sequence_simulation_widget.subvoxel_spin_counts == (3, 5, 11)
    assert window.sequence_simulation_widget.subvoxel_sampling_method == "stratified"
    assert window.sequence_simulation_widget._configured_spin_sampling().counts_xyz == (
        3,
        5,
        11,
    )
    assert window.sequence_simulation_widget.epi_read_fov_mm.value() == 180.0
    assert window.sequence_simulation_widget.epi_phase_fov_mm.value() == 170.0
    assert window.sequence_simulation_widget.bssfp_partition_fov_mm.value() == 80.0
    assert window.sequence_simulation_widget.field_strength_t.value() == 7.0
    assert window.sequence_simulation_widget.nucleus.currentText() == "C13"
    assert window.sequence_simulation_widget.sequence_reference_ppm.value() == 175.5
    updated_defaults = WorkspaceDefaults(
        sequence_fov_mm=(90.0, 85.0, 40.0),
        phantom_nucleus="P31",
        field_strength_t=9.4,
        sequence_reference_ppm_by_nucleus={"P31": -10.0},
    )
    window.sequence_simulation_widget.set_workspace_defaults(updated_defaults)
    assert window.sequence_simulation_widget.epi_read_fov_mm.value() == 90.0
    assert window.sequence_simulation_widget.epi_phase_fov_mm.value() == 85.0
    assert window.sequence_simulation_widget.bssfp_partition_fov_mm.value() == 40.0
    assert window.sequence_simulation_widget.field_strength_t.value() == 9.4
    assert window.sequence_simulation_widget.nucleus.currentText() == "P31"
    assert window.sequence_simulation_widget.sequence_reference_ppm.value() == -10.0
    window.sequence_simulation_widget.sequence_reference_ppm.setValue(5.0)
    window.sequence_simulation_widget.set_workspace_defaults(updated_defaults)
    assert window.sequence_simulation_widget.sequence_reference_ppm.value() == -10.0
    assert isinstance(window.sequence_simulation_widget, SequenceSimulationWidget)
    assert window.sequence_simulation_widget.program.source == "internal-flash-2d"
    assert window.sequence_simulation_widget._rf_designer is window.rf_designer
    assert window.sequence_simulation_widget._rf_designer_pulse_data is not None
    assert window.tab_widget.cornerWidget(Qt.TopRightCorner) is None
    assert window.workspace_switch.parentWidget() is window.free_mode_colormap_controls
    assert window.statusBar().isAncestorOf(window.status_export_button)
    assert window.findChild(QToolBar, "main_toolbar") is None
    assert window.mag_3d.export_3d_btn.isHidden()
    assert window.mag_3d.view_layout.indexOf(window.mag_3d.track_checkbox) >= 0
    assert window.mag_3d.view_layout.indexOf(window.mag_3d.spin_display_combo) >= 0
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
    assert window.findChild(QAction, "action_load_params") is None
    assert window.findChild(QAction, "action_save_params") is None
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
    # A rectangle keeps the P31 slice rephase lobe within the EPI timing.
    window.rf_designer.pulse_type.setCurrentText("Rectangle")
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


def test_parameter_loop_opens_on_demand_beside_simulations():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()

    assert (
        window.sequence_parameter_loop_tab_index
        == window.simulation_explorer_tab_index + 1
    )
    assert (
        window.tab_widget.tabText(window.sequence_parameter_loop_tab_index)
        == "Parameter Loop"
    )
    assert not window.tab_widget.isTabVisible(window.sequence_parameter_loop_tab_index)

    window.set_workspace_mode("sequence")
    sequence_widget = window.sequence_simulation_widget
    assert sequence_widget is not None
    assert not window.tab_widget.isTabVisible(window.sequence_parameter_loop_tab_index)

    # Opening the on-demand editor must refresh a stale button state left by a
    # preceding sequence/source transition.
    sequence_widget.parameter_loop_button.setEnabled(False)
    sequence_widget.open_parameter_loop_button.click()
    app.processEvents()

    assert window.tab_widget.isTabVisible(window.sequence_parameter_loop_tab_index)
    assert window.tab_widget.currentIndex() == window.sequence_parameter_loop_tab_index
    assert sequence_widget.parameter_loop_button.isEnabled()
    assert (
        window.sequence_parameter_loop_panel_layout.indexOf(
            sequence_widget.parameter_loop_group
        )
        == 0
    )
    assert (
        window.sequence_parameter_loop_plot_layout.indexOf(
            sequence_widget.parameter_loop_plot
        )
        == 0
    )
    assert not sequence_widget.parameter_loop_plot.isHidden()

    window.close()
    window.deleteLater()
    app.processEvents()


def test_session_simulations_tab_is_next_to_phantom_b1_and_accepts_runs():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()

    assert window.receiver_noise_tab_index == window.b1_combo_tab_index + 1
    assert window.spoiling_analysis_tab_index == window.receiver_noise_tab_index + 1
    assert window.psf_analysis_tab_index == window.spoiling_analysis_tab_index + 1
    assert window.kspace_tab_index == window.psf_analysis_tab_index + 1
    assert window.simulation_explorer_tab_index == window.kspace_tab_index + 1
    assert window.tab_widget.tabText(window.simulation_explorer_tab_index) == (
        "Simulations"
    )
    assert not window.tab_widget.isTabVisible(window.simulation_explorer_tab_index)

    window.set_workspace_mode("sequence")
    assert window.tab_widget.isTabVisible(window.simulation_explorer_tab_index)
    sequence_widget = window.sequence_simulation_widget
    sequence_widget.object_source.setCurrentIndex(1)
    sequence_widget.sequence_live_preview.setChecked(False)
    sequence_widget.sequence_source.setCurrentIndex(sequence_widget.EPI_SOURCE)
    sequence_widget.read_matrix.setValue(4)
    sequence_widget.phase_matrix.setValue(3)
    sequence_widget.matrix_size.setValue(5)
    sequence_widget._build_phantom()
    result = SimpleNamespace(
        metadata={
            "simulation_finished_at_utc": "2026-08-28T10:20:30+00:00",
            "simulation_wall_time_s": 0.5,
            "sequence_kernel": "optimized",
        },
        adc_times_s=np.arange(4, dtype=float),
        signal=np.zeros(4, dtype=np.complex128),
    )
    run = sequence_widget._register_session_simulation_run(result)

    assert run.display_name == "Run 1"
    assert (
        sequence_widget.result_volume_viewer.simulation_combo.currentData()
        == run.run_id
    )
    assert window.simulation_explorer.run_tree.topLevelItemCount() == 1
    assert window.simulation_explorer.run_tree.topLevelItem(0).text(1) == (
        sequence_widget.program.source
    )

    sequence_widget.read_matrix.setValue(9)
    sequence_widget.phase_matrix.setValue(8)
    sequence_widget.matrix_size.setValue(7)
    restored = []

    def restore_run(loaded_run):
        restored.append(loaded_run)
        assert sequence_widget.sequence_source.currentIndex() == (
            sequence_widget.EPI_SOURCE
        )
        assert sequence_widget.read_matrix.value() == 4
        assert sequence_widget.phase_matrix.value() == 3
        assert sequence_widget.matrix_size.value() == 5
        return True

    sequence_widget.restore_session_simulation_run = restore_run
    window._open_session_simulation_run(run)
    assert restored == [run]

    run.display_name = "Renamed run"
    window._rename_session_simulation_run(run)
    simulation_combo = sequence_widget.result_volume_viewer.simulation_combo
    assert (
        simulation_combo.itemText(simulation_combo.findData(run.run_id))
        == "Renamed run"
    )
    window._delete_session_simulation_run(run)
    assert sequence_widget.result_volume_viewer.simulation_combo.count() == 0

    window.close()
    window.deleteLater()
    app.processEvents()


def test_session_simulations_export_single_and_multiple_projects(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    widget = window.sequence_simulation_widget
    widget.object_source.setCurrentIndex(1)
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(2)
    widget._build_phantom()

    def make_result(value):
        return SequenceSimulationResult(
            signal=np.full(4, value, dtype=np.complex128),
            adc_times_s=np.arange(4, dtype=float) * 1e-4,
            final_magnetization=np.zeros(widget.phantom.shape + (3,)),
            checkpoint_magnetization=None,
            checkpoint_times_s=np.empty(0),
            metadata={"sequence_kernel": "optimized"},
        )

    widget.matrix_size.setValue(2)
    first = widget._register_session_simulation_run(make_result(1.0))
    first.created_at_utc = "2026-08-28T10:20:30"
    widget.matrix_size.setValue(3)
    widget._build_phantom()
    second = widget._register_session_simulation_run(make_result(2.0))
    second.created_at_utc = "2026-08-28T10:20:31"

    requested_defaults = []

    def save_filename(_parent, _title, default, _filter):
        requested_defaults.append(Path(default).name)
        return str(tmp_path / Path(default).name), "Bloch projects (*.blochproj)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", save_filename)
    window._export_session_simulation_runs((first,))
    assert requested_defaults[0] == "bloch_project_20260828_102030.blochproj"
    single = load_project(tmp_path / requested_defaults[0])
    assert single["state"]["sequence_controls"]["matrix_size"]["value"] == 2
    np.testing.assert_allclose(single["sequence_result"].signal, 1.0)

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    from blochsimulator.project_naming import ProjectExportNaming

    monkeypatch.setattr(
        window,
        "_prompt_session_project_export",
        lambda _runs: (batch_dir, ProjectExportNaming()),
    )
    second.display_name = "Renal baseline"
    second.custom_name = True
    window._export_session_simulation_runs((first, second))
    batch_paths = sorted(batch_dir.glob("*.blochproj"))
    assert {path.name for path in batch_paths} == {
        "bloch_project_20260828_102030.blochproj",
        "Renal baseline.blochproj",
    }
    loaded_sizes = sorted(
        load_project(path)["state"]["sequence_controls"]["matrix_size"]["value"]
        for path in batch_paths
    )
    assert loaded_sizes == [2, 3]

    window.close()
    window.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("regenerate_sequence", [False, True])
def test_project_export_keeps_simulated_parameters_after_edits(
    qt_application, tmp_path, monkeypatch, regenerate_sequence
):
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    window.set_workspace_mode("sequence")
    widget = window.sequence_simulation_widget
    widget.object_source.setCurrentIndex(1)
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(1)
    widget.nucleus.setCurrentText("C13")
    widget.field_strength_t.setValue(3.0)
    widget.sequence_reference_ppm.setValue(171.0)
    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)
    widget.read_matrix.setValue(2)
    widget.phase_matrix.setValue(2)
    widget.epi_flip_angle_deg.setValue(30.0)
    assert widget._ensure_current_generated_sequence()
    widget._build_phantom()
    widget.phantom.metadata["provenance"] = {"state": "simulated"}
    expected_rf = widget.program.rf_events[0].samples_hz.copy()
    result = widget.simulator.simulate_sequence(widget.program, widget.phantom)
    widget._finished(result)
    run = window.simulation_explorer.runs[-1]

    widget.read_matrix.setValue(3)
    widget.epi_flip_angle_deg.setValue(70.0)
    widget.sequence_reference_ppm.setValue(183.35)
    widget.field_strength_t.setValue(9.4)
    widget.phantom.metadata["provenance"]["state"] = "edited"
    widget.matrix_size.setValue(3)
    widget._build_phantom()
    if regenerate_sequence:
        assert widget._ensure_current_generated_sequence()

    path = tmp_path / "completed.blochproj"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args[-1]))
    window.save_project()
    assert not errors
    assert widget.read_matrix.value() == 3
    assert widget.sequence_reference_ppm.value() == 183.35

    def assert_simulated_project(filename):
        saved = load_project(filename)
        controls = saved["state"]["sequence_controls"]
        assert controls["read_matrix"]["value"] == 2
        assert controls["epi_flip_angle_deg"]["value"] == 30.0
        assert controls["matrix_size"]["value"] == 2
        assert controls["sequence_reference_ppm"]["value"] == 171.0
        assert controls["field_strength_t"]["value"] == 3.0
        assert saved["phantom"].shape == (2, 2, 1)
        assert saved["phantom"].metadata["provenance"]["state"] == "simulated"
        definitions = saved["program"].metadata["definitions"]
        assert definitions["SequenceReferencePpm"] == 171.0
        assert definitions["FieldStrengthT"] == 3.0
        np.testing.assert_array_equal(
            saved["program"].rf_events[0].samples_hz, expected_rf
        )
        np.testing.assert_array_equal(saved["sequence_result"].signal, result.signal)

    assert_simulated_project(path)
    session_path = tmp_path / "session.blochproj"
    window._open_session_simulation_run(run)
    widget.sequence_reference_ppm.setValue(180.0)
    window._save_session_run_project(run, session_path)
    assert_simulated_project(session_path)

    window.open_project(str(path))
    assert not errors
    widget.read_matrix.setValue(4)
    widget.sequence_reference_ppm.setValue(190.0)
    reopened_path = tmp_path / "reopened.blochproj"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args: (str(reopened_path), "")
    )
    window.save_project()
    assert not errors
    assert_simulated_project(reopened_path)


def test_project_export_without_result_saves_current_parameters(
    qt_application, tmp_path, monkeypatch
):
    window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    widget = window.sequence_simulation_widget
    widget.read_matrix.setValue(7)
    widget.sequence_reference_ppm.setValue(183.35)
    path = tmp_path / "unrun.blochproj"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    window.save_project()
    saved = load_project(path)
    assert saved["sequence_result"] is None
    assert saved["state"]["sequence_controls"]["read_matrix"]["value"] == 7
    assert (
        saved["state"]["sequence_controls"]["sequence_reference_ppm"]["value"] == 183.35
    )


def test_completed_spin_probe_is_retained_and_can_be_reopened(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    widget = window.sequence_simulation_widget
    widget.object_source.setCurrentIndex(2)
    result = SequenceProbeResult(
        time_s=np.array([0.0, 0.01]),
        positions_m=np.array([[0.0, 0.0, 0.0]]),
        frequency_offsets_hz=np.array([-100.0, 100.0]),
        magnetization=np.ones((2, 1, 2, 3)),
        metadata={
            "probe_type": "spectral",
            "simulation_finished_at_utc": "2026-08-28T10:20:30+00:00",
            "simulation_wall_time_s": 0.25,
        },
    )

    widget._probe_finished(result, np.array([-100.0, 100.0]), "Hz", "spectral")

    retained = window.simulation_explorer.runs
    assert len(retained) == 1
    run = retained[0]
    assert run.run_type == "spin_probe"
    assert run.result is result
    assert run.runtime_s == pytest.approx(0.25)
    initial_mz = widget.probe_initial_mz.value()
    widget.probe_initial_mz.setValue(initial_mz + 5.0)
    widget.views.setCurrentIndex(
        next(
            index
            for index in range(widget.views.count())
            if widget.views.tabText(index) == "Spin Probe"
        )
    )
    path = tmp_path / "probe.blochproj"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    window.save_project()
    saved = load_project(path)
    assert isinstance(saved["sequence_result"], SequenceProbeResult)
    assert (
        saved["state"]["sequence_controls"]["probe_initial_mz"]["value"] == initial_mz
    )
    np.testing.assert_array_equal(
        saved["sequence_result"].magnetization, result.magnetization
    )
    widget.probe_result = None

    window._open_session_simulation_run(run)

    assert widget.probe_result is result
    assert widget.views.tabText(widget.views.currentIndex()) == "Spin Probe"
    assert "Showing Run 1" in widget.probe_status.text()

    window.close()
    window.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("use_popup", [False, True])
def test_workspace_roundtrip_remains_interactive(use_popup):
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.show()
    app.processEvents()
    initial_window_size = window.size()
    initial_frame_geometry = window.frameGeometry()
    assert window.workspace_switch.isVisible()
    assert window.workspace_mode_selector.isVisible()

    for _ in range(3):
        _select_workspace(window, "sequence", use_popup=use_popup)
        app.processEvents()
        assert window.workspace_mode == "sequence"
        assert not window.workspace_header.isVisible()
        assert (
            window.tab_widget.cornerWidget(Qt.TopRightCorner) is window.workspace_switch
        )
        assert window.workspace_switch.isVisible()
        assert window.workspace_mode_selector.isVisible()

        _select_workspace(window, "free", use_popup=use_popup)
        app.processEvents()
        assert window.workspace_mode == "free"
        assert not window.workspace_header.isVisible()
        assert window.tab_widget.cornerWidget(Qt.TopRightCorner) is None
        assert (
            window.workspace_switch.parentWidget() is window.free_mode_colormap_controls
        )
        assert window.workspace_switch.isVisible()
        assert window.workspace_mode_selector.isVisible()
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
    widget.sequence_reference_ppm.setValue(183.35)
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
    assert definitions["SequenceReferencePpm"] == pytest.approx(183.35)
    assert definitions["SpoilAfterSlice"]
    assert definitions["SpoilerCyclesPerSlice"] == pytest.approx(6.0)
    assert definitions["SpoilerCyclesPerVoxel"] == pytest.approx(0.25)
    assert definitions["SpoilerDuration"] == pytest.approx(2e-3)
    assert definitions["SpoilerAxes"] == "xyz"
    assert np.asarray(definitions["SpoilerEndTimes"]).size == 1
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


def test_sequence_workspace_generates_oriented_flash_from_controls():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.FLASH_SOURCE)
    assert not widget.flash_group.isHidden()
    widget.flash_read_matrix.setValue(8)
    widget.flash_phase_matrix.setValue(4)
    widget.flash_slice_orientation.setCurrentText("Coronal (XZ)")
    assert widget.flash_read_gradient_axis.currentText() == "+X"
    assert widget.flash_phase_gradient_axis.currentText() == "+Z"
    assert widget.flash_slice_gradient_axis.text().startswith("-Y")
    widget.flash_read_gradient_axis.setCurrentText("+Z")
    widget.flash_phase_gradient_axis.setCurrentText("-X")
    assert widget.flash_slice_orientation.currentText() == "Custom"
    assert widget.flash_slice_gradient_axis.text().startswith("-Y")
    widget.flash_slice_offset_mm.setValue(6.0)
    widget.flash_echo_time_ms.setValue(5.0)
    widget.flash_repetition_time_ms.setValue(15.0)
    widget.flash_repetitions.setValue(2)
    widget.flash_acquisition_interval_ms.setValue(100.0)
    widget.generate_sequence_button.click()
    app.processEvents()

    assert widget.program.source == "internal-flash-2d"
    assert widget.acquisition.read_matrix == 8
    assert widget.acquisition.phase_matrix == 4
    definitions = widget.program.metadata["definitions"]
    assert definitions["Name"] == "flash_2d"
    assert definitions["ReadoutAxis"] == "+z"
    assert definitions["PhaseEncodingAxis"] == "-x"
    assert definitions["PartitionEncodingAxis"] == "-y"
    assert definitions["SliceOffset"] == pytest.approx(6e-3)
    assert definitions["TE"] == pytest.approx(5e-3)
    assert definitions["TR"] == pytest.approx(15e-3)
    assert definitions["AcquisitionInterval"] == pytest.approx(100e-3)
    assert np.asarray(definitions["AcquisitionStartTimes"]) == pytest.approx(
        (0.0, 100e-3)
    )
    widget.phantom = Phantom(
        shape=(1, 1, 1),
        fov=(0.01, 0.01, 0.01),
        t1_map=np.ones((1, 1, 1)),
        t2_map=np.ones((1, 1, 1)),
        pd_map=np.ones((1, 1, 1)),
    )
    widget.animation_enabled.setChecked(True)
    widget.animation_time_resolution_ms.setValue(2.0)
    time_resolution_s, maximum_frames, animation_note = widget._animation_request()
    assert time_resolution_s == pytest.approx(2e-3)
    assert animation_note == ""
    animation_times = _animation_checkpoint_times(
        widget.program,
        time_resolution_s=time_resolution_s,
        maximum_frames=maximum_frames,
        simulation_timestep_s=widget.simulation_timestep_us.value() * 1e-6,
    )
    assert animation_times.size > 6

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_epi_csi_and_flash_expose_independent_read_and_phase_directions():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()

    for prefix, parameters in (
        ("epi", widget._epi_pulseq_parameters),
        ("csi", widget._csi_pulseq_parameters),
        ("flash", widget._flash_pulseq_parameters),
    ):
        getattr(widget, f"{prefix}_read_gradient_axis").setCurrentText("-Z")
        getattr(widget, f"{prefix}_phase_gradient_axis").setCurrentText("+X")
        assert getattr(widget, f"{prefix}_slice_orientation").currentText() == "Custom"
        assert getattr(widget, f"{prefix}_slice_gradient_axis").text().startswith("-Y")
        assert parameters()["encoding_axes"] == ("-z", "+x", "-y")

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_three_dimensional_sequences_expose_independent_signed_encoding_axes():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()

    for prefix, parameters in (
        ("bssfp", widget._bssfp_pulseq_parameters),
        ("ss_bssfp", widget._ss_bssfp_pulseq_parameters),
        ("radial_me", widget._radial_me_bssfp_pulseq_parameters),
        ("me_bssfp", widget._me_bssfp_pulseq_parameters),
    ):
        getattr(widget, f"{prefix}_read_gradient_axis").setCurrentText("-Z")
        getattr(widget, f"{prefix}_phase_gradient_axis").setCurrentText("+X")
        assert (
            getattr(widget, f"{prefix}_partition_gradient_axis").text().startswith("-Y")
        )
        assert parameters()["encoding_axes"] == ("-z", "+x", "-y")

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_ss_and_me_bssfp_default_to_scanner_z_readout():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()

    for prefix, parameters in (
        ("ss_bssfp", widget._ss_bssfp_pulseq_parameters),
        ("me_bssfp", widget._me_bssfp_pulseq_parameters),
    ):
        assert getattr(widget, f"{prefix}_read_gradient_axis").currentText() == "+Z"
        assert getattr(widget, f"{prefix}_phase_gradient_axis").currentText() == "+Y"
        assert (
            getattr(widget, f"{prefix}_partition_gradient_axis").text().startswith("-X")
        )
        assert parameters()["encoding_axes"] == ("+z", "+y", "-x")

    assert widget.bssfp_read_gradient_axis.currentText() == "+X"
    assert widget.radial_me_read_gradient_axis.currentText() == "+X"

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_dynamic_sequence_controls_export_full_acquisition_intervals():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()

    cases = (
        (widget.csi_acquisition_interval_ms, widget._csi_pulseq_parameters),
        (widget.flash_acquisition_interval_ms, widget._flash_pulseq_parameters),
        (widget.bssfp_acquisition_interval_ms, widget._bssfp_pulseq_parameters),
        (
            widget.ss_bssfp_acquisition_interval_ms,
            widget._ss_bssfp_pulseq_parameters,
        ),
        (
            widget.radial_me_acquisition_interval_ms,
            widget._radial_me_bssfp_pulseq_parameters,
        ),
        (
            widget.me_bssfp_acquisition_interval_ms,
            widget._me_bssfp_pulseq_parameters,
        ),
    )
    for control, parameters in cases:
        assert parameters()["acquisition_interval_s"] is None
        assert control.specialValueText() == "Back-to-back"
        control.setValue(1234.0)
        assert parameters()["acquisition_interval_s"] == pytest.approx(1.234)

    widget.epi_repetition_time_ms.setValue(1234.0)
    assert widget._epi_pulseq_parameters()["repetition_time_s"] == pytest.approx(1.234)

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


def test_phantom_resolution_warning_identifies_flash_read_and_phase_axes(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    shape = (45, 32, 32)
    widget.phantom = Phantom(
        shape=shape,
        fov=(0.045, 0.032, 0.013),
        t1_map=np.ones(shape),
        t2_map=np.ones(shape),
    )
    widget.program = SequenceProgram(events=(), duration_s=0.0)
    widget.acquisition = CartesianAcquisition(
        read_matrix=64,
        phase_matrix=64,
        fov_m=(0.045, 0.032),
        dwell_s=1e-5,
    )
    warning = {}

    def capture_warning(_parent, title, message, buttons, default):
        warning.update(title=title, message=message, buttons=buttons, default=default)
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "warning", capture_warning)

    assert not widget._confirm_phantom_resolution_for_run()
    assert warning["title"] == "Phantom resolution is lower than sequence resolution"
    assert (
        "Read / x: phantom voxel 1 mm > sequence pixel 0.7031 mm" in warning["message"]
    )
    assert "Phase / y: phantom voxel 1 mm > sequence pixel 0.5 mm" in warning["message"]
    assert "Partition" not in warning["message"]
    assert warning["buttons"] == QMessageBox.Yes | QMessageBox.No
    assert warning["default"] == QMessageBox.No

    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: QMessageBox.Yes)
    assert widget._confirm_phantom_resolution_for_run()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_phantom_resolution_warning_respects_encoding_direction_and_2d_slice(
    monkeypatch,
):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    shape = (128, 64, 32)
    widget.phantom = Phantom(
        shape=shape,
        fov=(0.045, 0.032, 0.013),
        t1_map=np.ones(shape),
        t2_map=np.ones(shape),
    )
    widget.program = SequenceProgram(events=(), duration_s=0.0)
    widget.acquisition = CartesianAcquisition(
        read_matrix=64,
        phase_matrix=64,
        fov_m=(0.013, 0.045),
        dwell_s=1e-5,
        encoding_frame=EncodingFrame.from_read_phase_axes("+z", "+x"),
    )
    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message, *_args: messages.append(message)
        or QMessageBox.No,
    )

    assert not widget._confirm_phantom_resolution_for_run()
    assert "Read / z: phantom voxel 0.4062 mm > sequence pixel 0.2031 mm" in messages[0]
    assert "Phase / x" not in messages[0]
    assert "Partition / y" not in messages[0]

    widget.acquisition = CartesianAcquisition(
        read_matrix=32,
        phase_matrix=64,
        fov_m=(0.013, 0.045),
        dwell_s=1e-5,
        encoding_frame=EncodingFrame.from_read_phase_axes("+z", "+x"),
    )
    assert widget._confirm_phantom_resolution_for_run()
    assert len(messages) == 1

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_phantom_resolution_warning_cancels_run_before_worker_starts(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    shape = (32, 32, 1)
    widget.phantom = Phantom(
        shape=shape,
        fov=(0.032, 0.032, 0.003),
        t1_map=np.ones(shape),
        t2_map=np.ones(shape),
    )
    widget.program = SequenceProgram(events=(), duration_s=0.0)
    widget.acquisition = CartesianAcquisition(64, 64, (0.032, 0.032), 1e-5)
    monkeypatch.setattr(widget, "_build_phantom", lambda: None)
    monkeypatch.setattr(widget, "_checkpoint_seconds", lambda: ())
    monkeypatch.setattr(widget, "_animation_request", lambda: (None, None, ""))
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: QMessageBox.No)

    widget._run()

    assert widget.worker is None
    assert widget.run_button.isEnabled()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_phantom_resolution_warning_checks_partition_for_declared_3d_grid(
    monkeypatch,
):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.INTERNAL_SOURCE)
    shape = (64, 64, 16)
    widget.phantom = Phantom(
        shape=shape,
        fov=(0.045, 0.032, 0.013),
        t1_map=np.ones(shape),
        t2_map=np.ones(shape),
    )
    widget.program = SequenceProgram(
        events=(),
        duration_s=0.0,
        metadata={
            "definitions": {
                "EncodingMatrixSize": (64, 64, 64),
                "EncodingFOV": (0.045, 0.032, 0.013),
            }
        },
    )
    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message, *_args: messages.append(message)
        or QMessageBox.No,
    )

    assert not widget._confirm_phantom_resolution_for_run()
    assert "Partition / z: phantom voxel 0.8125 mm" in messages[0]
    assert "Read / x" not in messages[0]
    assert "Phase / y" not in messages[0]

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
    assert widget.phantom_summary.text().startswith("Default sphere (H1 water)")
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
    assert not widget.run_button.isHidden()
    assert not widget.run_button.isEnabled()
    assert widget.run_probe_button.isEnabled()
    assert widget.run_geometry_probe_button.isEnabled()
    assert not widget.progress.isHidden()

    widget.object_source.setCurrentIndex(0)
    assert widget.probe_group.isHidden()
    assert widget.run_button.isEnabled()
    assert not widget.run_probe_button.isEnabled()
    assert not widget.run_geometry_probe_button.isEnabled()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_simulation_object_summary_uses_structured_table(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    assert "QGroupBox { font-weight: bold; }" in widget.styleSheet()
    phantom = Phantom(
        shape=(2, 3, 4),
        fov=(2e-3, 6e-3, 8e-3),
        t1_map=np.full((2, 3, 4), 0.8),
        t2_map=np.full((2, 3, 4), 0.09),
        pd_map=np.ones((2, 3, 4)),
    )
    monkeypatch.setattr(widget, "_loaded_designed_phantom", lambda: phantom)

    widget.refresh_object_summary()
    rows = {
        widget.simulation_object_table.item(row, 0)
        .text(): widget.simulation_object_table.item(row, 1)
        .text()
        for row in range(widget.simulation_object_table.rowCount())
    }

    assert "Frequency model" in rows
    assert rows["Selected phantom"] == phantom.name
    assert "matrix (2, 3, 4)" in rows["Geometry"]
    assert "T1 800–800 ms" in rows["Relaxation"]
    assert widget.frequency_reference_info.isHidden()
    assert widget.phantom_summary.isHidden()

    widget.object_source.setCurrentIndex(1)
    rows = {
        widget.simulation_object_table.item(row, 0)
        .text(): widget.simulation_object_table.item(row, 1)
        .text()
        for row in range(widget.simulation_object_table.rowCount())
    }
    assert rows["Simulation object"] == "Uniform cube"
    assert "density" in rows["Relaxation / density"]

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_epi_generation_uses_nearest_supported_bandwidth():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.read_matrix.setValue(8)
    widget.phase_matrix.setValue(4)
    widget.sampling_bandwidth_khz.setValue(150.0)
    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)

    widget.generate_sequence_button.click()
    app.processEvents()

    assert widget.program.source == "internal-cartesian-epi"
    assert widget._generated_pulseq_sequence.check_timing()[0]
    # The field is rounded up (not to nearest) so regenerating keeps the same dwell.
    assert widget.sampling_bandwidth_khz.value() == pytest.approx(142.857, abs=1.5e-3)
    assert widget.acquisition.sampling_bandwidth_hz == pytest.approx(
        142_857.14285714287
    )

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_explicit_sequence_generation_error_has_guidance_and_details(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)
    raw_error = (
        "Invalid Cartesian acquisition: EPI sequence timing check failed:\n"
        "namespace(block=4, event='adc', field='delay', "
        "error_type='RASTER')"
    )
    widget._generation_error = raw_error
    monkeypatch.setattr(widget, "_reload_selected_generated_sequence", lambda: False)
    captured = {}

    def inspect_dialog(dialog):
        captured["title"] = dialog.windowTitle()
        captured["text"] = dialog.text()
        captured["info"] = dialog.informativeText()
        captured["details"] = dialog.detailedText()
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "exec_", inspect_dialog)

    widget.generate_sequence_button.click()

    assert "cannot produce a valid sequence" in captured["text"]
    assert "timing-raster" in captured["info"]
    assert "last valid sequence" in captured["info"]
    assert captured["details"] == raw_error

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_invalid_epi_bandwidth_restores_last_valid_value(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    # A clinical gradient system, on which 500 kHz needs too much gradient.
    widget.scanner_parameters = ScannerParameters(max_grad_mtm=32.0, max_slew_tms=130.0)
    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)
    widget.sampling_bandwidth_khz.setValue(500.0)
    captured = {}

    def inspect_dialog(dialog):
        captured["info"] = dialog.informativeText()
        captured["details"] = dialog.detailedText()
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "exec_", inspect_dialog)

    widget.generate_sequence_button.click()

    assert widget.sampling_bandwidth_khz.value() == pytest.approx(50.0)
    assert "Reduce bandwidth" in captured["info"]
    assert "restored to the last valid value, 50.000 kHz" in captured["info"]
    assert "Invalid Cartesian acquisition" in captured["details"]
    assert "the read gradient needs" in captured["details"]
    assert "above the 32 mT/m limit" in captured["details"]

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_spoiling_analysis_tab_is_shown_only_for_gradient_spoiling(tmp_path):
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    index = window.spoiling_analysis_tab_index
    tabs = window.tab_widget
    assert tabs.tabText(index) == "Spoiling Analysis"
    assert not tabs.isTabVisible(index)

    window.set_workspace_mode("sequence")
    sequence = window.sequence_simulation_widget
    # Manual grids stay as set; auto adjust has its own test.
    sequence.spoiling_auto_adjust.setChecked(False)
    sequence.set_spoiler_configuration("ideal", (1, 1, 9))
    assert not tabs.isTabVisible(index)
    sequence.set_spoiler_configuration("gradient", (1, 4, 1), "midpoint")
    assert tabs.isTabVisible(index)

    sequence.sequence_source.setCurrentIndex(sequence.FLASH_SOURCE)
    tabs.setCurrentIndex(index)
    view = sequence.spoiling_analysis_view
    assert window.spoiling_analysis_page.isAncestorOf(view)
    snapshot = view.snapshot
    assert snapshot is not None and snapshot.title.startswith("FLASH (2D)")
    assert snapshot.phases.shape == (snapshot.weights.size, 3)
    assert snapshot.recommendation == sequence._spoiling_spin_grid_recommendation
    # The hidden window defers drawing until the view is shown.
    assert view._dirty
    view.render()
    assert view.stack.currentIndex() == 1
    assert view.apply_button.isEnabled()
    view.apply_button.click()
    assert sequence._configured_spin_sampling().counts_xyz == (
        snapshot.recommendation.counts_xyz
    )
    assert view.snapshot.sampling.counts_xyz == snapshot.recommendation.counts_xyz
    view.render()
    assert not view.apply_button.isEnabled()

    # Switching to the ideal crusher leaves the tab before hiding it.
    sequence.set_spoiler_configuration("ideal", (1, 1, 9))
    assert not tabs.isTabVisible(index)
    assert tabs.currentIndex() == window.sequence_simulation_tab_index
    sequence.set_spoiler_configuration("gradient", (1, 4, 1), "stratified")
    assert tabs.isTabVisible(index)
    window.set_workspace_mode("free")
    assert not tabs.isTabVisible(index)
    window.set_workspace_mode("sequence")
    assert tabs.isTabVisible(index)

    window.close()
    window.deleteLater()
    app.processEvents()


def test_spoiling_auto_adjust_selects_recommended_grid_and_links_to_tab(tmp_path):
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    window.set_workspace_mode("sequence")
    sequence = window.sequence_simulation_widget
    tabs = window.tab_widget
    assert sequence.spoiling_auto_adjust.isChecked()
    assert sequence.spoiling_analysis_link.text().startswith("<a ")

    sequence.sequence_source.setCurrentIndex(sequence.FLASH_SOURCE)
    sequence.set_spoiler_configuration("gradient", (1, 1, 1), "midpoint")
    app.processEvents()
    recommendation = sequence._spoiling_spin_grid_recommendation
    assert recommendation is not None
    assert recommendation.counts_xyz != (1, 1, 1)
    assert sequence._configured_spin_sampling().counts_xyz == (
        recommendation.counts_xyz
    )
    assert window.app_settings.value("sequence/subvoxel_spins_z", type=int) == (
        recommendation.counts_xyz[2]
    )
    assert not sequence.spoiling_apply_recommended_grid.isEnabled()

    # Unchecked, a manual grid stays; checking again applies at once.
    sequence.spoiling_auto_adjust.setChecked(False)
    assert not window.app_settings.value("sequence/subvoxel_auto_adjust", type=bool)
    sequence.set_spoiler_configuration("gradient", (1, 1, 1), "midpoint")
    app.processEvents()
    assert sequence._configured_spin_sampling().counts_xyz == (1, 1, 1)
    sequence.spoiling_auto_adjust.setChecked(True)
    assert window.app_settings.value("sequence/subvoxel_auto_adjust", type=bool)
    assert sequence._configured_spin_sampling().counts_xyz == (
        recommendation.counts_xyz
    )

    # Project state keeps the choice; states without it keep their grid.
    state = window._complete_project_state()["spoiler_configuration"]
    assert state["auto_adjust"] is True
    sequence.set_spoiler_configuration(
        "gradient", (1, 1, 1), "midpoint", auto_adjust=False
    )
    app.processEvents()
    assert not sequence.spoiling_auto_adjust.isChecked()
    assert window.app_settings.value("sequence/subvoxel_auto_adjust", type=bool)
    assert sequence._configured_spin_sampling().counts_xyz == (1, 1, 1)

    sequence.spoiling_analysis_link.linkActivated.emit("spoiling-analysis")
    assert tabs.currentIndex() == window.spoiling_analysis_tab_index
    assert window.spoiling_analysis_page.isAncestorOf(sequence.spoiling_analysis_view)

    window.close()
    window.deleteLater()
    app.processEvents()


def test_shared_spoiling_quality_unit_covers_every_sequence_source():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_live_preview.setChecked(False)
    assert widget.spoiler_mode == "ideal"
    assert widget.spoiling_quality_group.isHidden()
    widget.set_spoiler_configuration("gradient", (1, 1, 9))
    assert not widget.spoiling_quality_group.isHidden()

    expected_text = {
        widget.INTERNAL_SOURCE: "Internal FID has no gradient spoiler",
        widget.EPI_SOURCE: "EPI / spiral end-of-slice spoiler",
        widget.CSI_SOURCE: "CSI end-of-FID spoiler",
        widget.BSSFP_SOURCE: "fully balanced",
        widget.SS_BSSFP_SOURCE: "Spectral-spatial bSSFP end-of-volume spoiler",
        widget.RADIAL_ME_BSSFP_SOURCE: "fully balanced",
        widget.ME_BSSFP_SOURCE: "fully balanced",
        widget.FLASH_SOURCE: "FLASH (2D)",
        widget.EPSI_MGE_SOURCE: "EPSI / MGE end-of-echo-train spoiler",
        widget.PULSEQ_SOURCE: "Imported Pulseq sequence",
    }
    for source_index, text in expected_text.items():
        widget.sequence_source.setCurrentIndex(source_index)
        assert text in widget.spoiling_quality_info.text()

    phantom = Phantom(
        shape=(2, 2, 2),
        fov=(2e-3, 2e-3, 2e-3),
        t1_map=np.ones((2, 2, 2)),
        t2_map=np.ones((2, 2, 2)),
        pd_map=np.ones((2, 2, 2)),
    )
    widget._loaded_designed_phantom = lambda: phantom
    raster = 10e-6
    events = []
    for index in range(4):
        start = index * 10e-3
        events.extend(
            (
                RFEvent(
                    start,
                    np.full(100, np.deg2rad(30.0) / (2 * np.pi * 100 * raster)),
                    raster,
                ),
                ADCEvent(start + 2e-3, 16, 20e-6),
                GradientEvent("z", start + 5e-3, np.full(100, 3 / 2e-3 / 1e-3), raster),
            )
        )
    widget.program = SequenceProgram(tuple(events), 40e-3, source="custom.seq")
    widget._update_spoiling_quality()
    text = widget.spoiling_quality_info.text()
    assert "Imported Pulseq sequence: coherence pathways" in text
    assert "Weighted subvoxel sampling error" in text
    assert not widget.spoiling_apply_recommended_grid.isHidden()
    recommendation = widget._spoiling_spin_grid_recommendation
    assert recommendation is not None and recommendation.counts_xyz[2] >= 2
    view = widget.spoiling_analysis_view
    snapshot = view.snapshot
    assert snapshot.weighted and snapshot.recommendation == recommendation
    assert snapshot.observation_label == "ADC number"
    assert snapshot.phases.shape == (snapshot.weights.size, 3)
    assert np.all(snapshot.observation >= 1)
    assert snapshot.summary == text
    view.show()
    assert view.stack.currentIndex() == 1
    assert view.axis_selector.currentText() == "Z"
    view.hide()

    widget.program = replace(
        widget.program,
        metadata={"definitions": {"IdealSpoilerEndTimes": [9e-3]}},
    )
    widget._update_spoiling_quality()
    assert "declares ideal spoilers" in widget.spoiling_quality_info.text()
    assert widget.spoiling_apply_recommended_grid.isHidden()
    assert widget.spoiling_analysis_view.snapshot is None
    assert "declares ideal spoilers" in widget.spoiling_analysis_view._message

    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)
    assert "Remaining coherent signal" in widget.spoiling_quality_info.text()
    assert not widget.spoiling_apply_recommended_grid.isHidden()
    snapshot = widget.spoiling_analysis_view.snapshot
    assert snapshot.observation_label == "Spoiler order"
    assert not snapshot.weighted and np.all(snapshot.weights == 1.0)
    widget.epi_spoil_after_slice.setChecked(False)
    assert "remaining coherent signal is 100%" in (widget.spoiling_quality_info.text())
    assert widget.spoiling_apply_recommended_grid.isHidden()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_ideal_spoiler_diagnostics_are_hidden_for_every_sequence_source():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()

    assert widget.ss_bssfp_spoiler_info.parent() is None
    for source_index in range(widget.sequence_source.count()):
        widget.sequence_source.setCurrentIndex(source_index)
        assert widget.spoiling_quality_group.isHidden()

    widget.set_spoiler_configuration("gradient", (1, 1, 9))
    for source_index in range(widget.sequence_source.count()):
        widget.sequence_source.setCurrentIndex(source_index)
        assert not widget.spoiling_quality_group.isHidden()

    widget.set_spoiler_configuration("ideal", (1, 1, 9))
    for source_index in range(widget.sequence_source.count()):
        widget.sequence_source.setCurrentIndex(source_index)
        assert widget.spoiling_quality_group.isHidden()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_ideal_spoiling_always_uses_one_spin_and_hides_quality_group():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.set_dynamic_sequence_kernel("optimized")
    widget.set_spoiler_configuration("gradient", (3, 5, 11), "stratified")

    assert widget._configured_spin_sampling().counts_xyz == (3, 5, 11)
    assert not widget.spoiling_quality_group.isHidden()

    widget.set_spoiler_configuration("ideal", (3, 5, 11), "stratified")

    assert widget.subvoxel_spin_counts == (1, 1, 1)
    assert widget._configured_spin_sampling().counts_xyz == (1, 1, 1)
    assert widget.spoiling_quality_group.isHidden()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_spoiling_quality_is_a_hover_table_instead_of_a_visible_text_block():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)

    tooltip = widget.spoiling_quality_button.toolTip()
    assert widget.spoiling_quality_info.isHidden()
    assert "<table" in tooltip
    assert "Cycles / phantom voxel XYZ" in tooltip
    assert "Recommended grid" in tooltip
    assert widget.spoiling_quality_button.text() == "ⓘ"

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_missing_phantom_dialog_links_to_the_phantom_tab(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    opened = []
    monkeypatch.setattr(widget, "_open_phantom_tab", lambda: opened.append(True))

    def inspect_dialog(dialog):
        assert "href='open-phantom'" in dialog.text()
        link_label = next(
            label
            for label in dialog.findChildren(QLabel)
            if "Open the Phantom tab" in label.text()
        )
        link_label.linkActivated.emit("open-phantom")
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "exec_", inspect_dialog)
    widget._show_no_phantom_dialog()

    assert opened == [True]
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_loaded_pulseq_warning_links_to_prefilled_simulation_settings(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    opened = []
    monkeypatch.setattr(
        widget,
        "loaded_pulseq_spoiler_recommendation",
        lambda: {
            "sampling_method": "midpoint",
            "counts_xyz": (2, 3, 11),
        },
    )
    monkeypatch.setattr(
        widget, "_show_simulation_settings", lambda: opened.append(True)
    )
    monkeypatch.setattr(QMessageBox, "open", lambda _dialog: None)

    widget._display_pulseq_spoiler_warning()

    dialog = widget._pulseq_spoiler_warning_dialog
    assert "Gradient waveform" in dialog.text()
    assert "2&times;3&times;11" in dialog.text()
    link_label = next(
        label
        for label in dialog.findChildren(QLabel)
        if "Open Simulation Settings" in label.text()
    )
    link_label.linkActivated.emit("open-settings")
    assert opened == [True]

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_loaded_pulseq_settings_use_multispin_fallback_without_train_analysis():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    # Start without a generated Cartesian acquisition, like an unanalysed file.
    widget.sequence_source.setCurrentIndex(widget.INTERNAL_SOURCE)
    widget.sequence_source.setCurrentIndex(widget.PULSEQ_SOURCE)
    widget.program = SequenceProgram(events=(), duration_s=0.0, source="custom.seq")

    recommendation = widget.loaded_pulseq_spoiler_recommendation()

    assert recommendation["spoiler_mode"] == "gradient"
    assert recommendation["sampling_method"] == "midpoint"
    assert recommendation["counts_xyz"] == (4, 4, 4)
    assert recommendation["counts_are_fallback"]
    assert "doubling the spins per axis" in recommendation["detail"]

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_successful_background_pulseq_load_displays_spoiler_warning():
    widget = MagicMock()
    widget._loaded_pulseq_declares_ideal_spoilers.return_value = False
    payload = object()

    SequenceSimulationWidget._pulseq_load_finished(widget, payload, "custom.seq")

    widget._apply_loaded_pulseq.assert_called_once_with(payload, "custom.seq")
    widget._reset_pulseq_load_controls.assert_called_once_with("Pulseq loaded")
    widget._display_pulseq_spoiler_warning.assert_called_once_with()


def test_flash_auto_spoiler_tracks_phantom_geometry_and_subvoxel_grid(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    phantom = Phantom(
        shape=(2, 2, 2),
        fov=(1e-3, 2e-3, 4e-3),
        t1_map=np.ones((2, 2, 2)),
        t2_map=np.ones((2, 2, 2)),
        pd_map=np.ones((2, 2, 2)),
    )
    monkeypatch.setattr(widget, "_loaded_designed_phantom", lambda: phantom)

    assert widget.flash_auto_spoiler.isChecked()
    assert not widget.flash_spoiler_cycles_per_slice.isEnabled()
    assert not widget.flash_spoiler_cycles_per_voxel.isEnabled()

    widget.set_spoiler_configuration("gradient", (1, 4, 1))
    widget.refresh_object_summary()

    # Default FLASH orientation maps read/phase/slice to X/Y/Z. The phase axis
    # is preferred because it is the sampled in-plane axis.
    assert widget.flash_spoiler_cycles_per_slice.value() == pytest.approx(1.5)
    assert widget.flash_spoiler_cycles_per_voxel.value() == pytest.approx(3.4375)
    assert "Effective cycles/phantom voxel XYZ: 0.5, 1, 1." in (
        widget.flash_spoiler_info.text()
    )
    assert "#b45309" in widget.flash_spoiler_info.styleSheet()
    assert "artificial subvoxel rephasing" in widget.flash_spoiler_info.text()
    assert "Recommended:" in widget.flash_spoiler_info.text()
    assert widget.flash_apply_recommended_grid.isEnabled()

    widget.flash_auto_spoiler.setChecked(False)
    assert widget.flash_spoiler_cycles_per_slice.isEnabled()
    assert widget.flash_spoiler_cycles_per_voxel.isEnabled()
    widget.flash_spoiler_cycles_per_slice.setValue(2.25)
    widget.flash_slice_thickness_mm.setValue(4.0)
    assert widget.flash_spoiler_cycles_per_slice.value() == pytest.approx(2.25)

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_derives_shaped_rf_bandwidth_and_shares_reference():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.SS_BSSFP_SOURCE)

    assert not widget.ss_bssfp_rf_bandwidth_hz.isEnabled()
    assert widget.ss_bssfp_spoiler_cycles.value() == pytest.approx(0.0)
    assert widget.ss_bssfp_spoiler_cycles_per_voxel.value() == pytest.approx(1.0)
    assert "Remaining coherent signal" in widget.ss_bssfp_spoiler_info.text()
    assert widget.ss_bssfp_rf_bandwidth_hz.value() == pytest.approx(
        widget.ss_bssfp_rf_time_bandwidth_product.value() / 2.33 * 1000.0,
        rel=2e-5,
    )
    widget.ss_bssfp_rf_pulse_type.setCurrentText("SLR")
    assert not widget.ss_bssfp_rf_sinc_lobes.isEnabled()
    assert widget.ss_bssfp_rf_sinc_lobes.isHidden()
    assert widget.ss_bssfp_rf_apodization.isHidden()
    assert not widget.ss_bssfp_rf_time_bandwidth_product.isEnabled()
    assert widget.ss_bssfp_rf_slr_sharpness.isEnabled()
    assert not widget.ss_bssfp_rf_slr_sharpness.isHidden()
    assert isinstance(widget.ss_bssfp_rf_slr_sharpness, QSpinBox)
    widget.ss_bssfp_rf_time_bandwidth_product.setValue(4.0)
    widget.ss_bssfp_rf_slr_sharpness.setValue(5)
    parameters = widget._ss_bssfp_pulseq_parameters()
    assert widget.ss_bssfp_rf_bandwidth_hz.value() == pytest.approx(
        widget.ss_bssfp_rf_time_bandwidth_product.value() / 2.33 * 1000.0,
        rel=2e-5,
    )
    assert parameters["spectral_rf_bandwidth_factor_hz_ms"] == pytest.approx(4000.0)
    assert parameters["spectral_rf_slr_sharpness"] == pytest.approx(5.0)

    widget.ss_bssfp_rf_pulse_type.setCurrentText("Sinc")
    assert widget.ss_bssfp_rf_sinc_lobes.isEnabled()
    assert not widget.ss_bssfp_rf_sinc_lobes.isHidden()
    assert not widget.ss_bssfp_rf_apodization.isHidden()
    assert not widget.ss_bssfp_rf_time_bandwidth_product.isEnabled()
    assert not widget.ss_bssfp_rf_slr_sharpness.isEnabled()
    widget.ss_bssfp_rf_duration_ms.setValue(2.0)
    widget.ss_bssfp_rf_sinc_lobes.setValue(5)
    assert widget.ss_bssfp_rf_bandwidth_hz.value() == pytest.approx(
        widget.ss_bssfp_rf_time_bandwidth_product.value() / 2.0 * 1000.0,
        rel=2e-5,
    )
    parameters = widget._ss_bssfp_pulseq_parameters()
    assert parameters["spectral_rf_bandwidth_hz"] is None
    assert parameters["spectral_rf_bandwidth_factor_hz_ms"] == pytest.approx(6000.0)
    assert parameters["spectral_rf_sinc_lobes"] == 5

    widget.ss_bssfp_rf_pulse_type.setCurrentText("Gaussian")
    assert not widget.ss_bssfp_rf_sinc_lobes.isEnabled()
    assert widget.ss_bssfp_rf_sinc_lobes.isHidden()
    assert widget.ss_bssfp_rf_apodization.isHidden()
    assert not widget.ss_bssfp_rf_time_bandwidth_product.isEnabled()
    assert not widget.ss_bssfp_rf_slr_sharpness.isEnabled()
    assert widget.ss_bssfp_rf_slr_sharpness.isHidden()

    widget.epi_rf_sinc_lobes.setValue(6)
    assert not widget.epi_rf_time_bandwidth_product.isEnabled()
    assert widget.epi_rf_bandwidth_hz.value() == pytest.approx(
        widget.epi_rf_time_bandwidth_product.value()
        / widget.epi_rf_duration_ms.value()
        * 1000.0,
        rel=2e-5,
    )

    widget.me_bssfp_rf_duration_ms.setValue(1.0)
    assert not widget.me_bssfp_rf_bandwidth_hz.isEnabled()
    assert widget.me_bssfp_rf_bandwidth_hz.value() == pytest.approx(
        widget.me_bssfp_rf_time_bandwidth_product.value() * 1000.0,
        rel=2e-5,
    )

    widget.field_strength_t.setValue(9.4)
    widget.nucleus.setCurrentText("C13")
    for advanced_parameters in (
        widget._ss_bssfp_pulseq_parameters(),
        widget._radial_me_bssfp_pulseq_parameters(),
        widget._me_bssfp_pulseq_parameters(),
    ):
        assert advanced_parameters["field_strength_t"] == pytest.approx(9.4)
        assert advanced_parameters["nucleus"] == "C13"

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_all_generated_sequences_share_rf_controls_and_loaded_pulse_parameters():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    sources = {
        "epi": widget.EPI_SOURCE,
        "epsi": widget.EPSI_MGE_SOURCE,
        "csi": widget.CSI_SOURCE,
        "flash": widget.FLASH_SOURCE,
        "bssfp": widget.BSSFP_SOURCE,
        "ss_bssfp": widget.SS_BSSFP_SOURCE,
        "radial_me": widget.RADIAL_ME_BSSFP_SOURCE,
        "me_bssfp": widget.ME_BSSFP_SOURCE,
    }

    for prefix, source in sources.items():
        # Only the selected sequence form is editable.
        widget.sequence_source.setCurrentIndex(source)
        pulse_type = getattr(widget, f"{prefix}_rf_pulse_type")
        assert [pulse_type.itemText(index) for index in range(pulse_type.count())] == [
            "Sinc",
            "SLR",
            "Gaussian",
            "Block",
            "RF Pulse Designer",
            "Spectral-spatial Designer",
            "Load RF pulse…",
        ]
        pulse_type.setCurrentText("SLR")
        getattr(widget, f"{prefix}_rf_time_bandwidth_product").setValue(3.5)
        getattr(widget, f"{prefix}_rf_slr_sharpness").setValue(5)
        getattr(widget, f"{prefix}_rf_offset_hz").setValue(125.0)
        parameters = widget._shared_rf_parameters(prefix)
        assert parameters["rf_pulse_type"] == "slr"
        assert parameters["rf_time_bandwidth_product"] == pytest.approx(4.0)
        assert parameters["rf_slr_sharpness"] == pytest.approx(5.0)
        assert parameters["rf_frequency_offset_hz"] == pytest.approx(125.0)
        assert getattr(widget, f"{prefix}_rf_slr_sharpness").isEnabled()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_bssfp_starter_settings_are_hidden_when_preparation_is_disabled():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()

    for prefix in ("bssfp", "ss_bssfp", "radial_me", "me_bssfp"):
        enabled = getattr(widget, f"{prefix}_alpha_half")
        mode = getattr(widget, f"{prefix}_alpha_half_use_ratios")
        ratio_values = getattr(widget, f"{prefix}_alpha_half_ratio_container")
        absolute_values = getattr(widget, f"{prefix}_alpha_half_absolute_container")

        enabled.setChecked(False)
        assert mode.isHidden()
        assert ratio_values.isHidden()
        assert absolute_values.isHidden()

        enabled.setChecked(True)
        mode.setChecked(True)
        assert not mode.isHidden()
        assert not ratio_values.isHidden()
        assert absolute_values.isHidden()

        mode.setChecked(False)
        assert ratio_values.isHidden()
        assert not absolute_values.isHidden()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_mode_loads_a_free_mode_rf_pulse_file(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    pulse_path = (
        Path(__file__).parents[1]
        / "rfpulses"
        / "bruker"
        / "13C_Ultimate_SPSP_Pulse_QuEMRT.exc"
    )
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(pulse_path), ""),
    )

    widget.bssfp_rf_pulse_type.setCurrentText("Load RF pulse…")
    parameter_factories = {
        "epi": widget._epi_pulseq_parameters,
        "csi": widget._csi_pulseq_parameters,
        "flash": widget._flash_pulseq_parameters,
        "bssfp": widget._bssfp_pulseq_parameters,
        "ss_bssfp": widget._ss_bssfp_pulseq_parameters,
        "radial_me": widget._radial_me_bssfp_pulseq_parameters,
        "me_bssfp": widget._me_bssfp_pulseq_parameters,
    }
    for prefix, parameter_factory in parameter_factories.items():
        getattr(widget, f"{prefix}_rf_pulse_type").setCurrentText("RF Pulse Designer")
        parameters = parameter_factory()
        key_prefix = "spectral_" if prefix == "ss_bssfp" else ""

        assert not getattr(widget, f"{prefix}_rf_duration_ms").isEnabled()
        assert not getattr(widget, f"{prefix}_rf_offset_hz").isEnabled()
        assert parameters[f"{key_prefix}rf_pulse_type"] == "designer"
        assert parameters[f"{key_prefix}rf_custom_name"] == pulse_path.name
        assert parameters[f"{key_prefix}rf_custom_waveform_hz"]
        assert parameters[f"{key_prefix}rf_custom_flip_angle_deg"] == pytest.approx(
            90.0
        )

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_ss_bssfp_reloads_and_calibrates_consecutive_headerless_rf_files(
    monkeypatch,
):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    pulse_directory = Path(__file__).parents[1] / "rfpulses"
    pulse_paths = iter(
        (
            pulse_directory / "SLR_sharpness_1.txt",
            pulse_directory / "SLR_sharpness_5.txt",
        )
    )
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(next(pulse_paths)), ""),
    )
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.PulseImportDialog.exec_",
        lambda self: 1,
    )
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.PulseImportDialog.get_options",
        lambda self: {
            "duration_s": 2.333e-3,
            "amp_unit": "percent",
            "phase_unit": "deg",
            "layout": "amp_phase_interleaved",
        },
    )

    widget.ss_bssfp_read_matrix.setValue(2)
    widget.ss_bssfp_phase_matrix.setValue(1)
    widget.ss_bssfp_partition_matrix.setValue(1)
    widget.ss_bssfp_repetitions.setValue(1)
    widget.sequence_source.setCurrentIndex(widget.SS_BSSFP_SOURCE)

    def generated_rotations_deg():
        return sorted(
            360.0 * abs(np.sum(event.samples_hz) * event.raster_s)
            for event in widget.program.rf_events
        )

    widget.ss_bssfp_rf_pulse_type.setCurrentText("Load RF pulse…")
    assert widget._ensure_current_generated_sequence()
    assert (
        widget._generated_pulseq_sequence.definitions["SpectralRFDesignerPulseName"]
        == "SLR_sharpness_1.txt"
    )
    assert generated_rotations_deg() == pytest.approx((45.0, 90.0), abs=0.02)

    widget.ss_bssfp_rf_pulse_type.setCurrentText("Load RF pulse…")
    assert widget._sequence_generation_pending
    assert (
        widget._generated_pulseq_sequence.definitions["SpectralRFDesignerPulseName"]
        == "SLR_sharpness_1.txt"
    )
    assert widget._ensure_current_generated_sequence()
    assert (
        widget._generated_pulseq_sequence.definitions["SpectralRFDesignerPulseName"]
        == "SLR_sharpness_5.txt"
    )
    assert generated_rotations_deg() == pytest.approx((45.0, 90.0), abs=0.02)

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_ss_bssfp_matches_named_phantom_peaks_and_uses_phantom_voxel_size(
    monkeypatch,
):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.SS_BSSFP_SOURCE)
    shape = (2, 4, 8)
    species = [
        ChemicalSpecies("Shape 1: Pyruvate", 0.0, 25.0, 0.3, frequency_offset_hz=0.0),
        ChemicalSpecies(
            "Shape 2: Lactate", 0.0, 25.0, 0.3, frequency_offset_hz=925.44725
        ),
    ]
    # Peak offsets are relative to pyruvate at the default C13 sequence
    # reference, so pyruvate sits at 0 Hz in the sequence frame.
    phantom = SpectralPhantom(
        shape=shape,
        fov=(1e-3, 2e-3, 4e-3),
        species=species,
        concentration_maps={item.name: np.ones(shape) for item in species},
        field_strength=7.0,
        nucleus="C13",
        spectral_reference_ppm=171.0,
    )
    monkeypatch.setattr(widget, "_loaded_designed_phantom", lambda: phantom)

    widget.flash_auto_spoiler.setChecked(False)
    widget.flash_spoiler_cycles_per_slice.setValue(4.0)
    widget.flash_spoiler_cycles_per_voxel.setValue(0.0)

    widget.ss_bssfp_target_names.setText("Lac, Py")
    widget.refresh_object_summary()
    parameters = widget._ss_bssfp_pulseq_parameters()

    assert widget.sequence_reference_ppm.value() == pytest.approx(171.0)
    assert parameters["target_frequency_offsets_hz"] == pytest.approx((1655, -245))
    assert parameters["receiver_frequency_offsets_hz"] == pytest.approx(
        (925.44725, 0.0)
    )
    assert parameters["end_image_spoiler_voxel_size_m"] == pytest.approx(
        (0.5e-3, 0.5e-3, 0.5e-3)
    )
    assert "Receiver offsets match" in widget.ss_bssfp_spoiler_info.text()

    widget.set_spoiler_configuration("gradient", (4, 4, 4))
    assert "0, 0, 0.6667" in widget.flash_spoiler_info.text()
    assert "grid 43.3%" in widget.flash_spoiler_info.text()
    widget.flash_spoiler_cycles_per_slice.setValue(6.0)
    assert "0, 0, 1" in widget.flash_spoiler_info.text()
    assert "#b45309" in widget.flash_spoiler_info.styleSheet()
    assert "artificial subvoxel rephasing" in widget.flash_spoiler_info.text()

    widget.ss_bssfp_spoiler_cycles_per_voxel.setValue(4.0)
    assert "aliases" in widget.ss_bssfp_spoiler_info.text()

    assert widget.ss_bssfp_target_offsets_hz.isEnabled()
    assert widget.ss_bssfp_receiver_offsets_hz.isEnabled()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_ss_bssfp_peak_matching_uses_the_simulated_frequency_frame(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.SS_BSSFP_SOURCE)
    shape = (2, 2, 2)
    species = [
        ChemicalSpecies("Pyruvate", 0.0, 25.0, 0.3, frequency_offset_hz=0.0),
        ChemicalSpecies("Lactate", 0.0, 25.0, 0.3, frequency_offset_hz=925.44725),
    ]
    # Without a spectral reference the peaks sit at absolute 0 ppm, far below
    # the 171 ppm C13 sequence reference.
    phantom = SpectralPhantom(
        shape=shape,
        fov=(2e-3, 2e-3, 2e-3),
        species=species,
        concentration_maps={item.name: np.ones(shape) for item in species},
        field_strength=7.0,
        nucleus="C13",
    )
    monkeypatch.setattr(widget, "_loaded_designed_phantom", lambda: phantom)
    widget.ss_bssfp_target_names.setText("Lac, Py")
    widget.ss_bssfp_receiver_offsets_hz.setText("925.44725, 0")
    widget.refresh_object_summary()

    assert widget.nucleus.currentText() == "C13"
    assert widget.sequence_reference_ppm.value() == pytest.approx(171.0)
    simulated = {
        name: float(np.unique(component.chemical_shift_map)[0])
        for name, component in phantom.to_component_phantoms(
            7.0, "C13", sequence_reference_ppm=171.0
        )
    }
    assert widget._ss_bssfp_phantom_peak_offsets() == pytest.approx(
        (simulated["Lactate"], simulated["Pyruvate"])
    )
    assert simulated["Pyruvate"] == pytest.approx(-12813.885)
    assert (
        "Receiver/phantom mismatch: +1.281e+04 Hz, +1.281e+04 Hz."
        in widget.ss_bssfp_spoiler_info.text()
    )

    widget.sequence_reference_ppm.setValue(0.0)
    assert widget._ss_bssfp_phantom_peak_offsets() == pytest.approx((925.44725, 0.0))
    assert "Receiver offsets match" in widget.ss_bssfp_spoiler_info.text()

    widget.sequence_reference_ppm.setValue(171.0)
    assert "Receiver/phantom mismatch" in widget.ss_bssfp_spoiler_info.text()

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
    assert definitions["AcquisitionInterval"] == pytest.approx(100e-3)
    assert np.asarray(definitions["AcquisitionStartTimes"]) == pytest.approx(
        (0.0, 100e-3, 200e-3)
    )
    assert len({event.frequency_offset_hz for event in widget.program.rf_events}) == 2
    rf = widget.program.rf_events[0]
    assert 360.0 * abs(np.sum(rf.samples_hz) * rf.raster_s) == pytest.approx(
        30.0, abs=2e-3
    )
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
    widget.epi_slice_thickness_mm.setValue(20.0)
    widget.epi_rf_time_bandwidth_product.setValue(3.5)
    widget.epi_rf_slr_sharpness.setValue(5)
    app.processEvents()

    definitions = widget.program.metadata["definitions"]
    rf = widget.program.rf_events[0]
    assert definitions["RFPulseType"] == "slr"
    assert definitions["RFDuration"] == pytest.approx(2.5e-3)
    # SLR sharpness sets the designed TBW; the TBW field is not used.
    assert definitions["RFTimeBandwidthProduct"] == pytest.approx(
        slr_time_bandwidth_product(5), rel=1e-6
    )
    assert widget.epi_rf_time_bandwidth_product.value() == pytest.approx(6.0)
    assert definitions["RFSLRSharpness"] == pytest.approx(5.0)
    assert rf.samples_hz.size * rf.raster_s == pytest.approx(2.5e-3)
    assert 360.0 * abs(np.sum(rf.samples_hz) * rf.raster_s) == pytest.approx(
        35.0, abs=2e-3
    )
    assert not widget.epi_rf_time_bandwidth_product.isEnabled()
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

    widget.epi_rf_pulse_type.setCurrentText("Gaussian")
    app.processEvents()
    assert widget.program.metadata["definitions"]["RFPulseType"] == "gaussian"
    assert not widget.epi_rf_time_bandwidth_product.isEnabled()
    assert not widget.epi_rf_apodization.isEnabled()
    assert not widget.epi_rf_slr_sharpness.isEnabled()

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

    saved_designer_state = widget.rf_designer_pulse_state()
    restored_widget = SequenceSimulationWidget()
    assert restored_widget.restore_rf_designer_pulse_state(saved_designer_state)
    restored_widget.flash_rf_pulse_type.setCurrentText("RF Pulse Designer")
    assert restored_widget.flash_rf_duration_ms.value() == pytest.approx(1.0)
    np.testing.assert_allclose(
        restored_widget._rf_designer_pulse_data["waveform_hz"],
        designer_waveform_hz,
    )
    restored_widget.close()
    restored_widget.deleteLater()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_legacy_project_recovers_two_ms_sequence_mode_rf_designer_pulse():
    app = QApplication.instance() or QApplication(sys.argv)
    raster_s = 1e-6
    duration_s = 2e-3
    flip_angle_deg = 90.0
    waveform_hz = np.full(
        int(duration_s / raster_s),
        flip_angle_deg / (360.0 * duration_s),
        dtype=np.complex128,
    )
    program = SequenceProgram(
        events=(RFEvent(0.0, waveform_hz, raster_s),),
        duration_s=duration_s,
        source="internal-flash-2d",
        metadata={
            "definitions": {
                "Name": "flash_2d",
                "FlipAngleDeg": flip_angle_deg,
                "RFPulseType": "designer",
                "RFDuration": duration_s,
                "RFTimeBandwidthProduct": 8.738,
                "RFDesignerPulseName": "SLR_sharpness_5.txt",
            }
        },
    )

    restored_widget = SequenceSimulationWidget()
    assert restored_widget.restore_rf_designer_pulse_state(None, program=program)
    restored_widget.flash_rf_pulse_type.setCurrentText("RF Pulse Designer")

    assert restored_widget.flash_rf_duration_ms.value() == pytest.approx(2.0)
    assert restored_widget._rf_designer_pulse_data["duration_s"] == pytest.approx(
        duration_s
    )
    np.testing.assert_array_equal(
        restored_widget._rf_designer_pulse_data["waveform_hz"], waveform_hz
    )
    restored_widget.close()
    restored_widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_builds_spiral_readout_from_controls(tmp_path):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.field_strength_t.setValue(9.4)
    widget.nucleus.setCurrentText("C13")
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
    assert definitions["FieldStrengthT"] == pytest.approx(9.4)
    assert definitions["Nucleus"] == "C13"
    assert "Spiral:" in widget.sequence_info.text()
    output = widget._write_pulseq_path(tmp_path / "interactive_spiral.seq")
    exported_program = load_pulseq(output)
    exported = SequenceCompiler().compile(exported_program)
    assert exported.adc_times_s.size == 8 * 8 * 2 * 2
    assert exported_program.metadata["definitions"]["FieldStrengthT"] == pytest.approx(
        9.4
    )
    assert exported_program.metadata["definitions"]["Nucleus"] == "C13"

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
    tab_labels = [widget.views.tabText(index) for index in range(widget.views.count())]
    assert "2D k-space / Reconstruction" in tab_labels
    assert "2D k-space" not in tab_labels
    assert "2D Reconstruction" not in tab_labels
    assert (
        widget.kspace_view.parentWidget().parentWidget()
        is widget.reconstruction_view.parentWidget().parentWidget()
    )

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
    # The generator writes IdealSpoilerEndTimes, so the ideal spoiler applies
    # and no gradient-waveform spin grid is recommended.
    assert widget.loaded_pulseq_spoiler_recommendation() is None

    widget._build_phantom()
    result = widget.simulator.simulate_sequence(widget.program, widget.phantom)
    widget._finished(result)
    app.processEvents()
    assert widget.kspace_view.image.shape == (4, 4)
    assert widget.reconstruction_view.image.shape == (4, 4)
    assert widget.result_volume_viewer.result is result
    assert all(
        widget.views.tabText(index) != "Final Mz"
        for index in range(widget.views.count())
    )

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
    assert widget.frame_selector.isHidden()
    assert widget.frame_slider.minimum() == -1
    assert widget.frame_slider.maximum() == 2
    assert widget.frame_slider.value() == -1
    assert widget.frame_value_label.text() == "All 3 frames (montage)"
    assert widget.kspace_view.image.shape == (14, 4)
    assert widget.reconstruction_view.image.shape == (14, 4)
    assert "montage of 3 frames" in widget.reconstruction_info.text()
    widget.frame_selector.setCurrentIndex(3)
    assert widget.reconstruction_view.image.shape == (4, 4)
    assert "slice=2" in widget.reconstruction_info.text()
    assert widget.frame_slider.value() == 2
    widget.frame_slider.setValue(1)
    assert widget.frame_selector.currentData() == 1
    assert widget.frame_value_label.text() == "slice=1"
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

    widget.sequence_source.setCurrentText("EPI")
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
    # QSplitter sizes are advisory. Platform-specific minimum-size hints from
    # the viewer may reduce the requested control width (notably under Xvfb).
    assert widget.MINIMUM_FOCUSED_CONTROL_WIDTH <= control_width
    assert control_width <= expected_control_width
    assert widget.FOCUSED_CONTROL_WIDTH == 600
    assert viewer_width >= widget.MINIMUM_FOCUSED_VIEWER_WIDTH
    assert widget.controls_scroll.horizontalScrollBar().maximum() == 0
    assert widget.layout().contentsMargins().left() == 0
    assert widget.split_view_checkbox.parentWidget() is widget.signal_page
    assert widget.views.tabBar().font().bold()
    assert widget.sequence_reference_ppm.toolTip()
    assert all(
        widget.views.tabToolTip(index).strip() for index in range(widget.views.count())
    )
    assert widget.sequence_title.font().bold()
    assert widget.sequence_title.font().pointSize() >= 12
    titled_groups = {
        group.title(): group
        for group in widget.findChildren(QGroupBox)
        if group.title()
    }
    assert titled_groups["Spoiling"].fontInfo().bold()
    assert titled_groups["Simulation object"].fontInfo().bold()
    assert widget.object_form.labelAlignment() & Qt.AlignLeft
    for image_view in (
        widget.kspace_view,
        widget.reconstruction_view,
    ):
        assert image_view.ui.histogram.width() == 48

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_starting_sequence_preserves_selected_result_tab(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.INTERNAL_SOURCE)
    widget.object_source.setCurrentIndex(1)
    widget.views.setCurrentIndex(widget.sequence_tab_index)
    widget.view_stack.setCurrentIndex(1)
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.SequenceSimulationThread.start",
        lambda _worker: None,
    )

    widget._run()

    assert widget.view_stack.currentIndex() == 1
    assert widget.views.currentIndex() == widget.sequence_tab_index
    assert widget.views.tabText(widget.signal_tab_index) == "Signal"
    assert "ADC signal" in widget.views.tabToolTip(widget.signal_tab_index)

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
    spectral_position = widget.probe_button_layout.getItemPosition(
        widget.probe_button_layout.indexOf(widget.run_probe_button)
    )
    geometry_position = widget.probe_button_layout.getItemPosition(
        widget.probe_button_layout.indexOf(widget.run_geometry_probe_button)
    )
    parameter_loop_position = widget.probe_button_layout.getItemPosition(
        widget.probe_button_layout.indexOf(widget.open_parameter_loop_button)
    )
    simulation_position = widget.probe_button_layout.getItemPosition(
        widget.probe_button_layout.indexOf(widget.run_button)
    )
    assert spectral_position[:2] == (0, 0)
    assert geometry_position[:2] == (0, 1)
    assert parameter_loop_position[:2] == (1, 0)
    assert simulation_position[:2] == (1, 1)
    assert widget.run_probe_button.text() == "Spectral Probe"
    assert widget.run_geometry_probe_button.text() == "Geometry Probe"
    assert widget.open_parameter_loop_button.text() == "Param. Loop"
    assert widget.run_button.text() == "Seq. Simulation"
    assert widget.cancel_probe_button is widget.cancel_button
    assert widget.export_cancel_layout.indexOf(widget.export_button) == 0
    assert widget.export_cancel_layout.indexOf(widget.open_result_button) == 1
    assert widget.export_cancel_layout.indexOf(widget.cancel_button) == 2
    assert widget.probe_initial_mz.maximum() == pytest.approx(1e7)
    widget.probe_initial_mz.setValue(2.5e6)
    assert widget.probe_initial_mz.value() == pytest.approx(2.5e6)

    widget.close()
    widget.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("saved_reference_ppm", [None, 0.0, 183.35])
@pytest.mark.parametrize("configured_reference_ppm", [171.0, 175.5])
def test_sequence_reference_defaults_follow_nucleus_and_respect_loaded_values(
    qt_application, saved_reference_ppm, configured_reference_ppm
):
    widget = SequenceSimulationWidget()
    widget.set_workspace_defaults(
        WorkspaceDefaults(
            sequence_reference_ppm_by_nucleus={"C13": configured_reference_ppm}
        )
    )
    assert widget.sequence_reference_ppm.value() == 0.0
    widget.nucleus.setCurrentText("C13")
    assert widget.sequence_reference_ppm.value() == configured_reference_ppm

    widget.sequence_reference_ppm.setValue(180.0)
    widget.field_strength_t.setValue(7.0)
    widget.epi_rf_pulse_type.setCurrentText("Gaussian")
    assert widget.sequence_reference_ppm.value() == 180.0

    widget.nucleus.setCurrentText("H1")
    assert widget.sequence_reference_ppm.value() == 0.0
    definitions = {"Nucleus": "C13"}
    if saved_reference_ppm is not None:
        definitions["SequenceReferencePpm"] = saved_reference_ppm
    widget.program = SequenceProgram(
        events=(), duration_s=1e-3, metadata={"definitions": definitions}
    )
    widget._apply_pulseq_frequency_reference()
    expected = (
        configured_reference_ppm if saved_reference_ppm is None else saved_reference_ppm
    )
    assert widget.nucleus.currentText() == "C13"
    assert widget.sequence_reference_ppm.value() == pytest.approx(expected)


def test_spectral_phantom_probe_defaults_follow_window_relative_to_reference():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    shape = (1, 1, 1)
    phantom = SpectralPhantom(
        shape=shape,
        fov=(0.01, 0.01, 0.01),
        species=[ChemicalSpecies("Peak", 5.0, 1.0, 0.02)],
        concentration_maps={"Peak": np.ones(shape)},
        nucleus="C13",
        spectral_reference_ppm=175.0,
        spectral_window_center_ppm=177.5,
        spectral_bandwidth_ppm=15.0,
        spectral_points=257,
    )
    widget.probe_frequency_units.setCurrentText("ppm")
    widget.sequence_reference_ppm.setValue(175.0)

    widget._apply_probe_defaults_from_phantom(phantom)

    assert widget.probe_ppm_min.value() == pytest.approx(-5.0)
    assert widget.probe_ppm_max.value() == pytest.approx(10.0)
    assert widget.probe_points.value() == 257

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_reference_confirmation_can_continue_or_cancel(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    shape = (1, 1, 1)
    widget.phantom = SpectralPhantom(
        shape=shape,
        fov=(0.01, 0.01, 0.01),
        species=[ChemicalSpecies("Peak", 0.0, 1.0, 0.02)],
        concentration_maps={"Peak": np.ones(shape)},
        spectral_reference_ppm=175.0,
        spectral_window_center_ppm=180.0,
        spectral_bandwidth_ppm=10.0,
    )
    widget.sequence_reference_ppm.setValue(185.0)
    assert widget._confirm_sequence_reference_for_run()

    widget.sequence_reference_ppm.setValue(185.1)
    observed = []

    def choose(result):
        def close_dialog():
            dialog = QApplication.activeModalWidget()
            assert isinstance(dialog, QMessageBox)
            observed.append(
                {
                    "text": dialog.text(),
                    "continue": dialog.button(QMessageBox.Yes).text(),
                    "cancel": dialog.button(QMessageBox.Cancel).text(),
                    "default": dialog.defaultButton().text(),
                }
            )
            dialog.done(result)

        QTimer.singleShot(0, close_dialog)

    choose(QMessageBox.Yes)
    assert widget._confirm_sequence_reference_for_run()
    assert observed[-1]["continue"] == "Continue"
    assert observed[-1]["cancel"] == "Cancel"
    assert observed[-1]["default"] == "Cancel"
    assert "outside the simulated spectral window" in observed[-1]["text"]

    fov_confirmation = MagicMock(return_value=True)
    monkeypatch.setattr(widget, "_build_phantom", lambda: None)
    monkeypatch.setattr(widget, "_confirm_generated_sequence_fov", fov_confirmation)
    choose(QMessageBox.Cancel)
    widget._run()

    assert widget.worker is None
    fov_confirmation.assert_not_called()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_bssfp_parameter_refresh_preserves_probe_frequency_limits():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.probe_ppm_min.setValue(-250.0)
    widget.probe_ppm_max.setValue(250.0)
    widget.bssfp_read_matrix.setValue(2)
    widget.bssfp_phase_matrix.setValue(1)
    widget.bssfp_partition_matrix.setValue(1)
    widget.sequence_source.setCurrentIndex(widget.BSSFP_SOURCE)

    widget.generate_sequence_button.click()
    widget.bssfp_flip_angle_deg.setValue(30.0)
    widget.generate_sequence_button.click()

    assert widget.probe_frequency_units.currentText() == "Hz"
    assert widget.probe_ppm_min.value() == pytest.approx(-250.0)
    assert widget.probe_ppm_max.value() == pytest.approx(250.0)

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


def test_sequence_probe_equilibrium_and_spoiling_controls(monkeypatch):
    from blochsimulator.ui.main_window import (
        _capture_widget_state,
        _restore_widget_state,
    )

    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    monkeypatch.setattr(
        "blochsimulator.ui.sequence_simulation_widget.SequenceProbeThread.start",
        lambda _worker: None,
    )
    probe_kwargs = dict(
        positions=np.array([[0.0, 0.0, 0.0]]),
        hz_axis=np.array([0.0]),
        display_axis=np.array([0.0]),
        checkpoints=np.array([0.0]),
        label="spectral",
    )
    assert widget.probe_polarization.currentIndex() == 0
    assert widget.probe_ideal_spoiling.isChecked()
    assert "above one" not in widget.probe_initial_mz.toolTip()

    widget._start_probe(**probe_kwargs)
    assert widget.probe_worker.equilibrium_mz == 1.0
    assert widget.probe_worker.ideal_spoiling is True
    widget.probe_worker.deleteLater()

    widget.probe_polarization.setCurrentIndex(1)
    widget.probe_hp_enhancement.setValue(2000.0)
    widget.probe_ideal_spoiling.setChecked(False)
    widget._start_probe(**probe_kwargs)
    assert widget.probe_worker.equilibrium_mz == pytest.approx(1.0 / 2000.0)
    assert widget.probe_worker.ideal_spoiling is False
    widget.probe_worker.deleteLater()
    widget.probe_worker = None

    widget.object_source.setCurrentIndex(2)
    rows = dict(widget._simulation_object_summary_rows())
    assert rows["Polarization"].startswith("Hyperpolarized, ε = 2000")
    assert "from above toward thermal 0.0005" in rows["Polarization"]
    assert rows["Ideal spoilers"].startswith("Off")

    state = _capture_widget_state(widget)
    assert state["probe_polarization"]["index"] == 1
    assert state["probe_hp_enhancement"]["value"] == 2000.0
    assert state["probe_ideal_spoiling"]["value"] is False
    restored = SequenceSimulationWidget()
    _restore_widget_state(restored, state)
    assert restored.probe_polarization.currentIndex() == 1
    assert restored._probe_equilibrium_mz() == pytest.approx(1.0 / 2000.0)
    assert not restored.probe_ideal_spoiling.isChecked()

    note = SequenceSimulationWidget._probe_ideal_spoiler_note
    assert "set to zero at 2 declared" in note(
        {
            "ideal_spoiling_applied": True,
            "declared_ideal_spoiler_end_times_s": [0.1, 0.2],
        }
    )
    assert "1 declared ideal spoiler time" in note(
        {
            "ideal_spoiling_applied": False,
            "declared_ideal_spoiler_end_times_s": [0.1],
        }
    )
    assert "declares no ideal spoilers" in note(
        {"ideal_spoiling_applied": False, "declared_ideal_spoiler_end_times_s": []}
    )
    for owner in (widget, restored):
        owner.close()
        owner.deleteLater()
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


def test_sequence_workspace_displays_zero_reference_lines():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.program = SequenceProgram((), duration_s=10e-3)
    widget._acquisition_compiled = None

    widget._show_program()

    assert widget.rf_zero_line in widget.rf_plot.plotItem.items
    assert widget.gradient_zero_line in widget.gradient_plot.plotItem.items
    assert widget.rf_zero_line.value() == pytest.approx(0.0)
    assert widget.gradient_zero_line.value() == pytest.approx(0.0)
    assert widget.rf_zero_line.angle == 0
    assert widget.gradient_zero_line.angle == 0
    widget.close()
    widget.deleteLater()
    app.processEvents()


@pytest.mark.parametrize(
    "definitions, echo, repetition",
    [
        ({"TE": 0.02, "TR": 0.06}, "20 ms", "60 ms"),
        (
            {"EchoTimes": [0.002, 0.004, 0.006], "TE": 0.002, "TR": 0.01},
            "2, 4, 6 ms",
            "10 ms",
        ),
        ({"echotime": "0.005", "RepetitionTime": [0.1]}, "5 ms", "100 ms"),
        ({"TR": [0.01, 0.02]}, "not specified", "10, 20 ms"),
        ({}, "not specified", "not specified"),
        ({"TE": float("nan"), "TR": -1}, "not specified", "not specified"),
        ({"TE": "unknown", "TR": []}, "not specified", "not specified"),
        (
            {"EchoTimes": [float("inf")], "TE": 0.005, "TR": 0.1},
            "5 ms",
            "100 ms",
        ),
        ({"EchoTimes": np.arange(1, 9) * 0.001}, "1–8 ms (8 values)", "not specified"),
    ],
)
def test_sequence_preview_displays_declared_timings(definitions, echo, repetition):
    program = SequenceProgram((), duration_s=0.1, metadata={"definitions": definitions})
    assert _sequence_preview_timing_text(program) == (
        f"Echo time (TE): {echo}    |    Repetition time (TR): {repetition}"
    )


@pytest.mark.parametrize(
    "definitions, image_time",
    [
        ({"TR": 0.01, "AcquisitionInterval": 0.64}, "640 ms"),
        (
            {
                "TR": 0.1,
                "AcquisitionInterval": 2.5,
                "MinimumAcquisitionInterval": 0.3,
                "SlicePositions": [-0.005, 0.0, 0.005],
            },
            "2.5 s (3 slices, minimum 300 ms)",
        ),
        ({"TR": 0.01, "VolumeInterval": [1.2]}, "1.2 s"),
    ],
)
def test_sequence_preview_displays_time_per_image(definitions, image_time):
    program = SequenceProgram((), duration_s=0.1, metadata={"definitions": definitions})
    assert _sequence_preview_timing_text(program).endswith(
        f"    |    Time per image: {image_time}"
    )


def test_default_flash_sequence_reports_time_per_image(qt_application):
    widget = SequenceSimulationWidget()
    assert widget.sequence_source.currentIndex() == widget.FLASH_SOURCE
    assert widget._ensure_current_generated_sequence(), widget._generation_error
    widget._show_program()
    assert "Time per image:" in widget.sequence_timing_summary.text()


def test_sequence_preview_timing_refresh_and_saved_visibility(tmp_path, qt_application):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    parent = QWidget()
    parent.app_settings = settings
    widget = SequenceSimulationWidget(parent)
    widget.program = SequenceProgram(
        (), duration_s=0.06, metadata={"definitions": {"TE": 0.02, "TR": 0.06}}
    )
    widget._acquisition_compiled = None
    widget._show_program()
    assert not widget.sequence_timing_summary.isHidden()
    assert "TE): 20 ms" in widget.sequence_timing_summary.text()
    assert "TR): 60 ms" in widget.sequence_timing_summary.text()
    widget.rf_plot.setXRange(5.0, 15.0, padding=0)
    widget.set_sequence_timing_visible(False)
    assert widget.sequence_timing_summary.isHidden()
    assert widget.rf_plot.viewRange()[0] == pytest.approx([5.0, 15.0])
    widget.program = SequenceProgram((), duration_s=0.02)
    widget._acquisition_compiled = None
    widget._show_program()
    assert widget.sequence_timing_summary.isHidden()
    widget.set_sequence_timing_visible(True)
    assert not widget.sequence_timing_summary.isHidden()
    assert widget.sequence_timing_summary.text().count("not specified") == 2

    settings.setValue("interface/show_sequence_timing", False)
    settings.sync()
    reopened = SequenceSimulationWidget(parent)
    assert reopened.sequence_timing_summary.isHidden()
    parent.deleteLater()


@pytest.mark.parametrize(
    "definitions, end_ms",
    [
        ({"TR": 0.01}, 10.0),
        ({"RepetitionTime": [0.012]}, 12.0),
        ({"tr": "0.008"}, 8.0),
        ({"TR": 0.1}, 30.0),
        ({}, 30.0),
        ({"TR": -1.0, "RepetitionTime": 0.01}, 10.0),
        ({"TR": [0.01, 0.02]}, 30.0),
        ({"TR": float("nan")}, 30.0),
        ({"TR": "unknown"}, 30.0),
    ],
)
def test_sequence_preview_uses_one_declared_repetition(definitions, end_ms):
    program = SequenceProgram(
        (), duration_s=0.03, metadata={"definitions": definitions}
    )

    assert _sequence_preview_range_ms(program) == pytest.approx((0.0, end_ms))


def test_sequence_workspace_overviews_track_and_navigate_both_waveform_plots(
    qt_application,
):
    widget = SequenceSimulationWidget()
    widget.program = SequenceProgram(
        tuple(
            event
            for start_s in (0.0, 0.01, 0.02)
            for event in (
                RFEvent(start_s, np.asarray([100.0]), 1e-3),
                GradientEvent("x", start_s, np.asarray([1000.0]), 1e-3),
            )
        ),
        duration_s=0.03,
        metadata={"definitions": {"TR": 0.01}},
    )
    widget._acquisition_compiled = None
    widget.views.setCurrentIndex(widget.sequence_tab_index)
    widget.resize(1600, 1000)
    widget.show()
    widget._show_program()
    qt_application.processEvents()
    widget.views.currentWidget().grab()

    for plot, overview in (
        (widget.rf_plot, widget.rf_overview),
        (widget.gradient_plot, widget.gradient_overview),
    ):
        assert plot.getViewBox().viewRange()[0] == pytest.approx([0.0, 10.0])
        assert overview.isVisible()
        assert overview.viewRange()[0] == pytest.approx([0.0, 30.0])
        assert overview.region.getRegion() == pytest.approx((0.0, 10.0))
        assert np.nanmax(overview.listDataItems()[0].xData) > 20.0
        curve = overview.listDataItems()[0].curve
        curve_bounds = curve.mapRectToScene(curve.boundingRect())
        assert overview.vb.sceneBoundingRect().intersects(curve_bounds)
        assert curve_bounds.height() > 1.0
        bounds = plot.getViewBox().mapRectToItem(
            plot.plotItem, plot.getViewBox().rect()
        )
        assert overview.geometry().right() == pytest.approx(bounds.right() - 8)
        assert overview.geometry().bottom() == pytest.approx(bounds.bottom() - 8)

    widget.rf_plot.setXRange(5.0, 15.0, padding=0)
    for overview in (widget.rf_overview, widget.gradient_overview):
        assert overview.region.getRegion() == pytest.approx((5.0, 15.0))

    widget.gradient_overview.region.setRegion((10.0, 20.0))
    for plot in (widget.rf_plot, widget.gradient_plot):
        assert plot.getViewBox().viewRange()[0] == pytest.approx([10.0, 20.0])
    assert widget.rf_overview.region.getRegion() == pytest.approx((10.0, 20.0))

    widget.rf_overview.region.setRegion((0.0, 5.0))
    widget.waveform_units.setCurrentIndex(widget.waveform_units.findData("simulation"))
    assert np.nanmax(widget.rf_overview.listDataItems()[0].yData) == pytest.approx(100)
    assert np.nanmax(
        widget.gradient_overview.listDataItems()[0].yData
    ) == pytest.approx(1)
    widget._preserve_sequence_plot_range_on_next_show = True
    widget._show_program()
    for overview in (widget.rf_overview, widget.gradient_overview):
        assert overview.region.getRegion() == pytest.approx((0.0, 5.0))
        assert (
            overview.scene() is widget.rf_plot.scene()
            or overview.scene() is widget.gradient_plot.scene()
        )

    widget.resize(1800, 1100)
    qt_application.processEvents()
    for plot, overview in (
        (widget.rf_plot, widget.rf_overview),
        (widget.gradient_plot, widget.gradient_overview),
    ):
        bounds = plot.getViewBox().mapRectToItem(
            plot.plotItem, plot.getViewBox().rect()
        )
        assert overview.geometry().right() == pytest.approx(bounds.right() - 8)
        assert overview.geometry().bottom() == pytest.approx(bounds.bottom() - 8)


def test_sequence_workspace_default_preview_follows_regenerated_tr():
    widget = SequenceSimulationWidget()
    widget.program = SequenceProgram(
        (), duration_s=0.03, metadata={"definitions": {"TR": 0.01}}
    )
    widget._acquisition_compiled = None
    widget._show_program()
    widget.program = SequenceProgram(
        (), duration_s=0.02, metadata={"definitions": {"TR": 0.005}}
    )
    widget._acquisition_compiled = None
    widget._preserve_sequence_plot_range_on_next_show = True

    widget._show_program()

    for plot, overview in (
        (widget.rf_plot, widget.rf_overview),
        (widget.gradient_plot, widget.gradient_overview),
    ):
        assert plot.getViewBox().viewRange()[0] == pytest.approx([0.0, 5.0])
        assert overview.region.getRegion() == pytest.approx((0.0, 5.0))
        assert overview.viewRange()[0] == pytest.approx([0.0, 20.0])


def test_sequence_workspace_preserves_timeline_zoom_when_regenerating():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.rf_plot.setXRange(5.0, 10.0, padding=0)
    widget.program = SequenceProgram((ADCEvent(0.0, 20, 1e-3),), duration_s=20e-3)
    widget._acquisition_compiled = None
    widget._preserve_sequence_plot_range_on_next_show = True

    widget._show_program()

    assert widget.rf_plot.getViewBox().viewRange()[0] == pytest.approx([5.0, 10.0])
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_workspace_timeline_cannot_leave_the_sequence():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.program = SequenceProgram((ADCEvent(0.0, 20, 1e-3),), duration_s=20e-3)
    widget._acquisition_compiled = None
    widget._show_program()

    # A zoom window that starts before the sequence keeps its width.
    widget.rf_plot.setXRange(-0.12, 1.9, padding=0)
    for plot in (widget.rf_plot, widget.gradient_plot):
        assert plot.getViewBox().viewRange()[0] == pytest.approx([0.0, 2.02])

    widget.gradient_plot.setXRange(-5.0, 30.0, padding=0)
    for plot in (widget.rf_plot, widget.gradient_plot):
        assert plot.getViewBox().viewRange()[0] == pytest.approx([0.0, 20.0])
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
            RFEvent(
                0.0,
                np.asarray([gamma / 1e4]),
                1e-3,
                phase_offset_rad=np.deg2rad(30.0),
            ),
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
    assert np.nanmax(widget._rf_phase_item.yData) == pytest.approx(30.0)
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


def test_new_probe_result_preserves_spin_probe_view_configuration():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    positions = np.array([[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0]])
    frequencies = np.array([-100.0, 0.0, 100.0])

    def result(times, scale):
        times = np.asarray(times, dtype=float)
        return SequenceProbeResult(
            time_s=times,
            positions_m=positions,
            frequency_offsets_hz=frequencies,
            magnetization=np.full(
                (times.size, positions.shape[0], frequencies.size, 3),
                scale,
                dtype=float,
            ),
            metadata={
                "probe_type": "grid",
                "stored_timeline": "configured_checkpoints",
                "configured_playback_times_s": times,
            },
        )

    widget.probe_result = result([0.0, 0.01, 0.02], 1.0)
    widget._show_probe_result()
    spectrum = widget.probe_spectrum_viewer
    spatial = widget.probe_spatial_viewer
    spectrum.position_slider.setValue(1)
    spectrum.selection_center.setValue(100.0)
    spectrum.add_frequency_selection()
    spatial.freq_slider.setValue(2)
    widget.probe_views.setCurrentIndex(1)
    widget.probe_time_control.time_slider.setValue(2)

    widget._probe_finished(
        result([0.0, 0.012, 0.024], 2.0), frequencies, "Hz", mode="grid"
    )

    assert widget.probe_views.currentIndex() == 1
    assert spectrum.position_slider.value() == 1
    assert spectrum.frequency_selections == [
        {
            "mode": "Single frequency",
            "center_hz": 100.0,
            "width": 100.0,
            "width_kind": "FWHM",
        }
    ]
    assert spectrum.selection_list.currentRow() == 0
    assert spatial.freq_slider.value() == 2
    assert widget.probe_time_control.time_slider.value() == 2
    assert spectrum.time_index == 2
    assert np.all(widget.probe_result.magnetization == 2.0)

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


def test_sequence_probe_thread_stores_only_configured_checkpoints():
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

    assert simulator.received_checkpoints == pytest.approx([0.0, 0.002, 0.01])
    assert completed[0].metadata["stored_timeline"] == "configured_checkpoints"
    assert completed[0].metadata["configured_playback_times_s"] == pytest.approx(
        [0.0, 0.002, 0.01]
    )
    assert completed[0].metadata["adc_times_s"] == pytest.approx([0.005])

    worker.deleteLater()
    app.processEvents()


def test_sequence_workspace_routes_probe_memory_warning_to_settings_dialog():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget._show_memory_limit_warning = MagicMock()
    message = "Memory limit exceeded: probe test details"

    widget._probe_failed(message)

    widget._show_memory_limit_warning.assert_called_once_with(message)
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_reconstruction_explorer_hides_2d_tab_and_sequence_settings(qt_application):
    window = BlochSimulatorGUI()
    window.show()
    window.set_workspace_mode("sequence")
    widget = window.sequence_simulation_widget
    explorer = widget.reconstruction_explorer

    assert not widget.views.isTabVisible(widget.two_d_result_tab_index)
    assert widget.csi_slice_thickness_mm.value() == pytest.approx(3.0)

    widget.views.setCurrentIndex(widget.reconstruction_explorer_tab_index)
    qt_application.processEvents()
    assert not widget.control_column.isVisible()
    assert explorer.controls_scroll.isVisible()

    explorer.sequence_settings_button.click()
    qt_application.processEvents()
    assert widget.control_column.isVisible()

    explorer.sequence_settings_button.click()
    widget.views.setCurrentIndex(widget.log_tab_index)
    qt_application.processEvents()
    assert widget.control_column.isVisible()


def test_phantom_designer_starts_from_default_sphere(qt_application):
    window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    window.tab_widget.setCurrentIndex(window.phantom_tab_index)
    qt_application.processEvents()
    phantom_widget = window.phantom_widget.phantom_creator
    assert phantom_widget.current_phantom is None

    phantom_widget._open_spectral_designer()
    dialog = phantom_widget._retained_spectral_designer_dialogs[-1]
    try:
        assert dialog.design.shapes
        assert dialog.design.nucleus == phantom_widget.get_nucleus()
    finally:
        dialog.reject()


def test_phantom_designer_from_sequence_buttons_follow_the_sequence(qt_application):
    window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    sequence_widget = window.sequence_simulation_widget
    sequence_widget.sequence_source.setCurrentIndex(sequence_widget.FLASH_SOURCE)
    qt_application.processEvents()
    geometry = sequence_widget.current_sequence_geometry()
    assert geometry.is_2d
    window.tab_widget.setCurrentIndex(window.phantom_tab_index)
    qt_application.processEvents()
    phantom_widget = window.phantom_widget.phantom_creator

    phantom_widget._open_spectral_designer()
    dialog = phantom_widget._retained_spectral_designer_dialogs[-1]
    try:
        x_button, y_button, z_button = dialog.sequence_grid_buttons
        assert x_button.isEnabled() and y_button.isEnabled()
        # The slice direction of the 2D FLASH is not encoded.
        assert not z_button.isEnabled()
        x_button.click()
        assert dialog.matrix_spins[0].value() == 2 * geometry.matrix[0]
        assert dialog.fov_spins[0].value() == pytest.approx(geometry.fov_m[0] * 1000.0)
    finally:
        dialog.reject()


def _type_zero(app, spin):
    spin.setFocus()
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin, "0")
    QTest.keyClick(spin, Qt.Key_Return)
    app.processEvents()


def test_typing_zero_picks_the_shortest_flash_timing_without_touching_others():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.FLASH_SOURCE)
    widget.show()
    app.processEvents()
    widget.flash_echo_time_ms.setValue(5.0)
    widget.flash_repetition_time_ms.setValue(15.0)

    _type_zero(app, widget.flash_repetition_time_ms)
    plan = make_pulseq_flash(**widget._flash_pulseq_parameters(), timing_only=True)
    minimum_tr_ms = widget._ceil_spin_value(
        plan.minimum_repetition_time_s * 1e3,
        widget.flash_repetition_time_ms.decimals(),
    )
    assert widget.flash_repetition_time_ms.value() == pytest.approx(minimum_tr_ms)
    assert minimum_tr_ms < 15.0
    assert widget.flash_echo_time_ms.value() == pytest.approx(5.0)

    _type_zero(app, widget.flash_echo_time_ms)
    assert widget.flash_echo_time_ms.value() == pytest.approx(
        widget._ceil_spin_value(
            plan.minimum_echo_time_s * 1e3, widget.flash_echo_time_ms.decimals()
        )
    )
    assert widget.flash_repetition_time_ms.value() == pytest.approx(minimum_tr_ms)
    widget.close()


def test_typing_zero_bisects_to_the_shortest_buildable_bssfp_tr():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.BSSFP_SOURCE)
    widget.show()
    app.processEvents()
    spin = widget.bssfp_repetition_time_ms
    original = spin.value()

    _type_zero(app, spin)
    minimum = spin.value()
    assert spin.minimum() < minimum < original
    make_pulseq_bssfp(**widget._bssfp_pulseq_parameters())
    spin.setValue(minimum - 10.0 ** -spin.decimals())
    with pytest.raises(Exception):
        make_pulseq_bssfp(**widget._bssfp_pulseq_parameters())
    widget.close()
