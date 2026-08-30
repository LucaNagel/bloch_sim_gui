import sys
import runpy
import time
from pathlib import Path
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
    QToolBar,
)

from blochsimulator.ui.main_window import BlochSimulatorGUI
from blochsimulator.ui.sequence_simulation_widget import (
    SequenceProbeThread,
    SequenceSimulationThread,
    SequenceSimulationWidget,
    _SequenceSimulationPayload,
    _animation_checkpoint_times,
    _event_step_plot_data,
    _rf_phase_plot_data,
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
    GradientEvent,
    RFEvent,
    SequenceCompiler,
    SequenceProbeResult,
    SequenceProgram,
    SequenceSimulationResult,
    SpectroscopicAcquisition,
    load_pulseq,
)
from blochsimulator.sequence.rf_pulses import (
    rf_time_bandwidth_product_from_envelope,
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
    window.app_settings.setValue("sequence/kernel", "reference")
    window.app_settings.setValue("sequence/dynamic_kernel", "native_parallel")
    window.app_settings.setValue("sequence/timestep_preset", "fast")
    window.app_settings.setValue("sequence/timestep_us", 10.0)
    window.app_settings.setValue("sequence/spoiler_mode", "gradient")
    window.app_settings.setValue("sequence/subvoxel_spins_x", 3)
    window.app_settings.setValue("sequence/subvoxel_spins_y", 5)
    window.app_settings.setValue("sequence/subvoxel_spins_z", 11)
    window.app_settings.setValue("sequence/subvoxel_sampling_method", "stratified")
    window.app_settings.setValue("simulation/thread_mode", "manual")
    window.app_settings.setValue("simulation/manual_threads", 2)
    window.app_settings.setValue("defaults/sequence_fov_x_mm", 180.0)
    window.app_settings.setValue("defaults/sequence_fov_y_mm", 170.0)
    window.app_settings.setValue("defaults/sequence_fov_z_mm", 80.0)
    window.app_settings.setValue("defaults/field_strength_t", 7.0)
    window.app_settings.setValue("defaults/phantom_nucleus", "C13")
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
    updated_defaults = WorkspaceDefaults(
        sequence_fov_mm=(90.0, 85.0, 40.0),
        phantom_nucleus="P31",
        field_strength_t=9.4,
    )
    window.sequence_simulation_widget.set_workspace_defaults(updated_defaults)
    assert window.sequence_simulation_widget.epi_read_fov_mm.value() == 90.0
    assert window.sequence_simulation_widget.epi_phase_fov_mm.value() == 85.0
    assert window.sequence_simulation_widget.bssfp_partition_fov_mm.value() == 40.0
    assert window.sequence_simulation_widget.field_strength_t.value() == 9.4
    assert window.sequence_simulation_widget.nucleus.currentText() == "P31"
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


def test_session_simulations_tab_is_next_to_phantom_b1_and_accepts_runs():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()

    assert window.simulation_explorer_tab_index == window.b1_combo_tab_index + 1
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
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(batch_dir),
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


def test_completed_spin_probe_is_retained_and_can_be_reopened():
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
    widget.probe_result = None

    window._open_session_simulation_run(run)

    assert widget.probe_result is result
    assert widget.views.tabText(widget.views.currentIndex()) == "Spin Probe"
    assert "Showing Run 1" in widget.probe_status.text()

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
    monkeypatch.setattr(widget, "_selected_designed_phantom", lambda: phantom)

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
    widget._selected_designed_phantom = lambda: phantom
    widget.acquisition = MagicMock(
        moment_origins_cyc_per_m=((0.0, 0.0, 0.0), (1000.0, 0.0, 0.0))
    )
    widget._update_spoiling_quality()
    assert "Imported Pulseq ADC moment train" in widget.spoiling_quality_info.text()
    assert "maximum sampling error" in widget.spoiling_quality_info.text()

    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)
    assert "Remaining coherent signal" in widget.spoiling_quality_info.text()
    assert not widget.spoiling_apply_recommended_grid.isHidden()
    widget.epi_spoil_after_slice.setChecked(False)
    assert "remaining coherent signal is 100%" in (widget.spoiling_quality_info.text())
    assert widget.spoiling_apply_recommended_grid.isHidden()

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_ideal_spoiling_always_uses_one_spin_and_hides_quality_group():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.set_dynamic_sequence_kernel("metal_hybrid")
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
    monkeypatch.setattr(widget, "_selected_designed_phantom", lambda: phantom)

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
    assert not widget.ss_bssfp_rf_time_bandwidth_product.isEnabled()
    assert widget.ss_bssfp_rf_slr_sharpness.isEnabled()
    widget.ss_bssfp_rf_time_bandwidth_product.setValue(4.0)
    widget.ss_bssfp_rf_slr_sharpness.setValue(5.0)
    parameters = widget._ss_bssfp_pulseq_parameters()
    assert widget.ss_bssfp_rf_bandwidth_hz.value() == pytest.approx(
        widget.ss_bssfp_rf_time_bandwidth_product.value() / 2.33 * 1000.0,
        rel=2e-5,
    )
    assert parameters["spectral_rf_bandwidth_factor_hz_ms"] == pytest.approx(4000.0)
    assert parameters["spectral_rf_slr_sharpness"] == pytest.approx(5.0)

    widget.ss_bssfp_rf_pulse_type.setCurrentText("Sinc")
    assert widget.ss_bssfp_rf_sinc_lobes.isEnabled()
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
    assert not widget.ss_bssfp_rf_time_bandwidth_product.isEnabled()
    assert not widget.ss_bssfp_rf_slr_sharpness.isEnabled()

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
    prefixes = (
        "epi",
        "csi",
        "flash",
        "bssfp",
        "ss_bssfp",
        "radial_me",
        "me_bssfp",
    )

    for prefix in prefixes:
        pulse_type = getattr(widget, f"{prefix}_rf_pulse_type")
        assert [pulse_type.itemText(index) for index in range(pulse_type.count())] == [
            "Sinc",
            "SLR",
            "Gaussian",
            "Block",
            "RF Pulse Designer",
        ]
        pulse_type.setCurrentText("SLR")
        getattr(widget, f"{prefix}_rf_time_bandwidth_product").setValue(3.5)
        getattr(widget, f"{prefix}_rf_slr_sharpness").setValue(5.0)
        getattr(widget, f"{prefix}_rf_offset_hz").setValue(125.0)
        parameters = widget._shared_rf_parameters(prefix)
        assert parameters["rf_pulse_type"] == "slr"
        assert parameters["rf_time_bandwidth_product"] == pytest.approx(4.0)
        assert parameters["rf_slr_sharpness"] == pytest.approx(5.0)
        assert parameters["rf_frequency_offset_hz"] == pytest.approx(125.0)
        assert getattr(widget, f"{prefix}_rf_slr_sharpness").isEnabled()
        assert hasattr(widget, f"{prefix}_rf_load_button")

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

    widget._load_sequence_rf_pulse("bssfp")
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


def test_ss_bssfp_matches_named_phantom_peaks_and_uses_phantom_voxel_size(
    monkeypatch,
):
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    shape = (2, 4, 8)
    species = [
        ChemicalSpecies("Shape 1: Pyruvate", 0.0, 25.0, 0.3, frequency_offset_hz=0.0),
        ChemicalSpecies(
            "Shape 2: Lactate", 0.0, 25.0, 0.3, frequency_offset_hz=925.44725
        ),
    ]
    phantom = SpectralPhantom(
        shape=shape,
        fov=(1e-3, 2e-3, 4e-3),
        species=species,
        concentration_maps={item.name: np.ones(shape) for item in species},
        field_strength=7.0,
        nucleus="C13",
    )
    monkeypatch.setattr(widget, "_selected_designed_phantom", lambda: phantom)

    widget.flash_auto_spoiler.setChecked(False)
    widget.flash_spoiler_cycles_per_slice.setValue(4.0)
    widget.flash_spoiler_cycles_per_voxel.setValue(0.0)

    widget.ss_bssfp_target_names.setText("Lac, Py")
    widget.refresh_object_summary()
    parameters = widget._ss_bssfp_pulseq_parameters()

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
    widget.epi_rf_slr_sharpness.setValue(5.0)
    app.processEvents()

    definitions = widget.program.metadata["definitions"]
    rf = widget.program.rf_events[0]
    assert definitions["RFPulseType"] == "slr"
    assert definitions["RFDuration"] == pytest.approx(2.5e-3)
    expected_tbw = rf_time_bandwidth_product_from_envelope(rf.samples_hz)
    assert definitions["RFTimeBandwidthProduct"] == pytest.approx(
        expected_tbw, rel=1e-6
    )
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

    widget.close()
    widget.deleteLater()
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
    simulation_position = widget.probe_button_layout.getItemPosition(
        widget.probe_button_layout.indexOf(widget.run_button)
    )
    assert spectral_position[:2] == (0, 0)
    assert geometry_position[:2] == (0, 1)
    assert simulation_position[:2] == (0, 2)
    assert widget.run_probe_button.text() == "Spectral Probe"
    assert widget.run_geometry_probe_button.text() == "Geometry Probe"
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
