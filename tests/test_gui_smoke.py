import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QLineEdit,
    QRadioButton,
    QSlider,
)
from blochsimulator.memory import GIB, MemoryPolicy
from blochsimulator.sequence import ScannerParameters
from blochsimulator.ui.dialogs import PulseImportDialog, SettingsDialog
from blochsimulator.ui.default_settings import WorkspaceDefaults
from blochsimulator.ui.magnetization_viewer import MagnetizationViewer
from blochsimulator.ui.main_window import BlochSimulatorGUI, _apply_platform_style
from blochsimulator.ui.rf_pulse_designer import RFPulseDesigner


def test_application_style_uses_native_macos_controls():
    app = MagicMock()

    _apply_platform_style(app, platform="darwin")

    app.setStyle.assert_not_called()


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_application_style_retains_fusion_elsewhere(platform):
    app = MagicMock()

    _apply_platform_style(app, platform=platform)

    app.setStyle.assert_called_once_with("Fusion")


def test_log_messages_include_local_timestamp():
    window = MagicMock()
    cursor = window.log_widget.textCursor.return_value

    with patch(
        "blochsimulator.ui.main_window.time.strftime",
        return_value="2026-07-07 14:05:09",
    ):
        BlochSimulatorGUI.log_message(window, "Compiling sequence…")

    window.log_widget.append.assert_called_once_with(
        "[2026-07-07 14:05:09] Compiling sequence…"
    )
    window.log_widget.moveCursor.assert_called_once_with(cursor.End)


@pytest.mark.parametrize(
    ("component", "expected"),
    [
        ("Magnitude", [5.0, 5.0]),
        ("Real", [3.0, -3.0]),
        ("Imaginary", [4.0, 4.0]),
        ("Phase", [np.arctan2(4.0, 3.0), np.arctan2(4.0, -3.0)]),
    ],
)
def test_signal_line_plot_uses_selected_component(component, expected):
    window = MagicMock()
    window.last_result = {"signal": np.array([3.0 + 4.0j, -3.0 + 4.0j])}
    window.last_time = np.array([0.0, 0.001])
    window.signal_component.currentText.return_value = component

    BlochSimulatorGUI._render_signal_lines(window)

    plotted_values = window.signal_plot.plot.call_args.args[1]
    assert np.allclose(plotted_values, expected)
    assert window.signal_plot.plot.call_args.kwargs["name"] == component


def test_gui_instantiation():
    """Smoke test to ensure the main window can be instantiated without crashing."""
    # Ensure a QApplication exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    try:
        window = BlochSimulatorGUI()
        assert window is not None
        assert window.windowTitle() == "Bloch Equation Simulator"

        # Check if tutorial manager is initialized
        assert hasattr(window, "tutorial_manager")
        assert window.tutorial_manager is not None
        assert window.findChild(QAction, "action_settings") is not None
        assert window.findChild(QAction, "action_project_explorer") is not None
        assert window.time_step_spin.minimum() == pytest.approx(0.01)

    finally:
        # We don't want to show the window or start the event loop
        pass


@pytest.mark.parametrize(
    "pulse_type", ["Adiabatic Half Passage", "Adiabatic Full Passage"]
)
def test_adiabatic_passage_uses_direct_b1_and_ten_ms_default(pulse_type):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    designer = RFPulseDesigner(compact=True)
    designer.duration.setValue(1.0)
    designer.pulse_type.setCurrentText(pulse_type)

    assert designer.duration.value() == pytest.approx(10.0)
    assert not designer.flip_angle.isEnabled()
    assert designer.adiabatic_parameter_hint.isVisibleTo(designer)
    assert "directly" in designer.adiabatic_parameter_hint.text()
    assert np.max(np.abs(designer.get_pulse()[0])) == pytest.approx(0.0)

    designer.b1_amplitude.setValue(0.08)
    pulse_before = designer.get_pulse()[0].copy()
    assert np.max(np.abs(pulse_before)) == pytest.approx(0.08)

    designer.flip_angle.setValue(321.0)
    assert np.allclose(designer.get_pulse()[0], pulse_before)


def test_adiabatic_free_mode_prompt_confirms_time_step_only_once():
    window = MagicMock()
    window.workspace_mode = "free"
    window._adiabatic_timestep_warning_shown = False
    window.rf_designer.is_adiabatic_passage.return_value = True
    window.rf_designer.pulse_type.currentText.return_value = "Adiabatic Full Passage"
    window.rf_designer.b1_amplitude.value.return_value = 0.08
    window.time_step_spin.value.return_value = 1.0

    confirm_button = object()
    cancel_button = object()
    dialog = MagicMock()
    dialog.addButton.side_effect = [confirm_button, cancel_button]
    dialog.clickedButton.return_value = confirm_button
    with patch("blochsimulator.ui.main_window.QMessageBox", return_value=dialog) as box:
        proceed = BlochSimulatorGUI._confirm_adiabatic_free_mode_run(window)
        proceed_again = BlochSimulatorGUI._confirm_adiabatic_free_mode_run(window)

    assert proceed
    assert proceed_again
    box.assert_called_once_with(window)
    assert window._adiabatic_timestep_warning_shown
    window.time_step_spin.setValue.assert_called_once_with(0.01)


def test_adiabatic_free_mode_skips_prompt_at_recommended_time_step():
    window = MagicMock()
    window.workspace_mode = "free"
    window._adiabatic_timestep_warning_shown = False
    window.rf_designer.is_adiabatic_passage.return_value = True
    window.rf_designer.pulse_type.currentText.return_value = "Adiabatic Half Passage"
    window.rf_designer.b1_amplitude.value.return_value = 0.08
    window.time_step_spin.value.return_value = 0.01

    with patch("blochsimulator.ui.main_window.QMessageBox") as box:
        proceed = BlochSimulatorGUI._confirm_adiabatic_free_mode_run(window)

    assert proceed
    box.assert_not_called()


def test_adiabatic_free_mode_run_requires_direct_b1():
    window = MagicMock()
    window.workspace_mode = "free"
    window.rf_designer.is_adiabatic_passage.return_value = True
    window.rf_designer.pulse_type.currentText.return_value = "Adiabatic Half Passage"
    window.rf_designer.b1_amplitude.value.return_value = 0.0

    with patch("blochsimulator.ui.main_window.QMessageBox.warning") as warning:
        proceed = BlochSimulatorGUI._confirm_adiabatic_free_mode_run(window)

    assert not proceed
    warning.assert_called_once()
    window.rf_designer.b1_amplitude.setFocus.assert_called_once()


def test_settings_dialog_returns_selected_values(tmp_path):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = SettingsDialog(
        MemoryPolicy(), tmp_path, tooltips_enabled=True, detected_thread_count=8
    )
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData("custom_reserve"))
    dialog.reserve_spin.setValue(3.5)

    policy = dialog.get_policy()
    assert policy.mode == "custom_reserve"
    assert policy.reserve_bytes == int(3.5 * GIB)
    assert dialog.reserve_spin.isEnabled()
    assert not dialog.limit_spin.isEnabled()
    dialog.animation_memory_budget_spin.setValue(2048.0)
    assert dialog.animation_memory_budget_bytes() == 2048 * 1024**2
    assert dialog.get_export_directory() == tmp_path
    assert dialog.tooltips_enabled()
    assert dialog.sequence_live_progress_enabled()
    assert dialog.sequence_kernel() == "optimized"
    dialog.sequence_kernel_combo.setCurrentIndex(
        dialog.sequence_kernel_combo.findData("reference")
    )
    assert dialog.sequence_kernel() == "reference"
    assert dialog.dynamic_sequence_kernel() == "optimized"
    assert dialog.dynamic_sequence_kernel_combo.findData("metal_hybrid") >= 0
    dialog.dynamic_sequence_kernel_combo.setCurrentIndex(
        dialog.dynamic_sequence_kernel_combo.findData("metal_hybrid")
    )
    assert dialog.dynamic_sequence_kernel() == "metal_hybrid"
    assert not any(spin.isEnabled() for spin in dialog.subvoxel_spin_count_spins)
    dialog.dynamic_sequence_kernel_combo.setCurrentIndex(
        dialog.dynamic_sequence_kernel_combo.findData("native_parallel")
    )
    assert dialog.dynamic_sequence_kernel() == "native_parallel"
    assert dialog.sequence_timestep_preset() == "balanced"
    assert dialog.sequence_timestep_us() == pytest.approx(5.0)
    assert not dialog.sequence_timestep_us_spin.isEnabled()
    dialog.sequence_timestep_preset_combo.setCurrentIndex(
        dialog.sequence_timestep_preset_combo.findData("custom")
    )
    dialog.sequence_timestep_us_spin.setValue(7.5)
    assert dialog.sequence_timestep_us() == pytest.approx(7.5)
    assert dialog.sequence_timestep_us_spin.isEnabled()
    assert dialog.sequence_spoiler_mode() == "ideal"
    assert dialog.subvoxel_spin_counts() == (1, 1, 1)
    assert dialog.subvoxel_sampling_method() == "midpoint"
    assert not any(spin.isEnabled() for spin in dialog.subvoxel_spin_count_spins)
    assert not dialog.subvoxel_sampling_method_combo.isEnabled()
    assert not any(label.isEnabled() for label in dialog.subvoxel_control_labels)
    dialog.sequence_spoiler_mode_combo.setCurrentIndex(
        dialog.sequence_spoiler_mode_combo.findData("gradient")
    )
    for spin, value in zip(dialog.subvoxel_spin_count_spins, (3, 5, 11)):
        spin.setValue(value)
    assert dialog.sequence_spoiler_mode() == "gradient"
    assert dialog.subvoxel_spin_counts() == (3, 5, 11)
    dialog.subvoxel_sampling_method_combo.setCurrentIndex(
        dialog.subvoxel_sampling_method_combo.findData("stratified")
    )
    assert dialog.subvoxel_sampling_method() == "stratified"
    assert all(spin.isEnabled() for spin in dialog.subvoxel_spin_count_spins)
    assert dialog.subvoxel_sampling_method_combo.isEnabled()
    assert all(label.isEnabled() for label in dialog.subvoxel_control_labels)
    assert dialog.thread_mode() == "automatic"
    assert not dialog.manual_thread_count_spin.isEnabled()
    dialog.thread_mode_combo.setCurrentIndex(
        dialog.thread_mode_combo.findData("manual")
    )
    dialog.manual_thread_count_spin.setValue(3)
    assert dialog.thread_mode() == "manual"
    assert dialog.manual_thread_count() == 3
    assert dialog.manual_thread_count_spin.isEnabled()
    dialog.scanner_max_grad_spin.setValue(40.0)
    dialog.scanner_max_slew_spin.setValue(180.0)
    scanner = dialog.scanner_parameters()
    assert scanner.max_grad_mtm == pytest.approx(40.0)
    assert scanner.max_slew_tms == pytest.approx(180.0)
    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
        "General",
        "Defaults",
        "Simulation",
        "Scanner",
        "Memory",
        "Interface",
    ]


def test_settings_dialog_groups_kernel_extras_and_has_english_tooltips(tmp_path):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = SettingsDialog(MemoryPolicy(), tmp_path, tooltips_enabled=True)

    sequence = dialog.sequence_kernel_combo
    sequence_optimized = sequence.findData("optimized")
    sequence_reference = sequence.findData("reference")
    assert sequence.itemText(sequence_optimized) == "Optimized (recommended)"
    assert sequence.itemText(sequence_reference) == "Reference (advanced validation)"
    assert sequence_reference > sequence_optimized + 1
    assert "ordinary Bloch simulations" in sequence.itemData(
        sequence_optimized, Qt.ToolTipRole
    )
    assert "numerical comparisons" in sequence.itemData(
        sequence_reference, Qt.ToolTipRole
    )

    dynamic = dialog.dynamic_sequence_kernel_combo
    expected_labels = {
        "optimized": "Optimized NumPy (recommended)",
        "native_parallel": "Native automatic (recommended for large objects)",
        "native_serial": "Native serial (advanced benchmark)",
        "reference": "Reference (advanced validation)",
        "metal_hybrid": "CPU + Apple GPU (experimental)",
    }
    indices = {kernel: dynamic.findData(kernel) for kernel in expected_labels}
    assert all(index >= 0 for index in indices.values())
    assert indices["native_serial"] > indices["native_parallel"] + 1
    for kernel, label in expected_labels.items():
        index = indices[kernel]
        assert dynamic.itemText(index) == label
        assert dynamic.itemData(index, Qt.ToolTipRole)

    assert "safe fallback" in dynamic.itemData(indices["optimized"], Qt.ToolTipRole)
    assert "multiple CPU workers" in dynamic.itemData(
        indices["native_parallel"], Qt.ToolTipRole
    )
    assert "one CPU worker" in dynamic.itemData(
        indices["native_serial"], Qt.ToolTipRole
    )
    assert "correctness reference" in dynamic.itemData(
        indices["reference"], Qt.ToolTipRole
    )
    assert "Apple Silicon" in dynamic.itemData(indices["metal_hybrid"], Qt.ToolTipRole)


def test_memory_limit_uses_warning_instead_of_generic_error():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    window._show_memory_limit_warning = MagicMock()
    message = "Memory limit exceeded: test details"

    with patch("blochsimulator.ui.main_window.QMessageBox.critical") as critical:
        window.on_simulation_error(message)

    window._show_memory_limit_warning.assert_called_once_with(message)
    critical.assert_not_called()


def test_simulation_controls_use_non_overlapping_grid_rows():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    layout = window.simulation_controls_layout
    assert isinstance(layout, QGridLayout)

    def position(widget):
        return layout.getItemPosition(layout.indexOf(widget))

    assert position(window.pos_range) == (2, 1, 1, 1)
    assert position(window.freq_spin) == (3, 1, 1, 1)
    assert position(window.freq_center) == (4, 1, 1, 1)
    assert position(window.freq_range) == (5, 1, 1, 1)
    assert position(window.freq_axis_mode) == (6, 1, 1, 1)


def test_free_mode_and_slice_explorer_spatial_controls_use_mm():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    window.pos_spin.setValue(3)
    window.pos_range.setValue(20.0)
    window._synchronize_workspace_nucleus("C13")

    assert window.pos_range.suffix() == " mm"
    assert window._collect_simulation_parameters(internal_format=True)[
        "pos_range_mm"
    ] == pytest.approx(20.0)
    assert (
        window._collect_simulation_parameters(internal_format=True)["nucleus"] == "C13"
    )
    parameters = window._collect_simulation_parameters()
    assert parameters["nucleus"] == "C13"
    assert parameters["position_range_mm"] == pytest.approx(20.0)
    assert parameters["position_range_cm"] == pytest.approx(2.0)
    assert parameters["position_axis"][:, 2] == pytest.approx([-0.01, 0.0, 0.01])

    explorer = window.slice_explorer
    assert explorer.pos_range.suffix() == " mm"
    assert explorer.pos_range.value() == pytest.approx(40.0)
    assert explorer.plot_profile.getAxis("bottom").labelText == "Position (mm)"
    explorer.num_points.setValue(51)
    explorer.run_simulation()
    profile_x, _ = explorer.plot_profile.listDataItems()[0].getData()
    assert profile_x[[0, -1]] == pytest.approx([-20.0, 20.0])


def test_free_mode_export_uses_parameters_captured_for_completed_run(tmp_path):
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    captured = {
        "tissue": {"field_strength_t": 7.0, "t1_ms": 1234.0},
        "sequence": {"sequence_type": "Spin Echo", "te_s": 0.02},
        "simulation": {
            "mode": "endpoint",
            "time_step_us": 4.0,
            "num_positions": 3,
            "field_strength_t": 7.0,
        },
    }
    window._last_export_parameters = captured
    window.last_result = {"time": np.array([0.0]), "mx": np.array([[0.0]])}
    window.last_time = np.array([0.0])
    window.pos_spin.setValue(99)
    window.time_step_spin.setValue(25.0)
    window.tissue_widget.set_field_strength(1.5)

    dialog = MagicMock()
    dialog.exec_.return_value = QDialog.Accepted
    dialog.get_export_options.return_value = {
        "base_path": str(tmp_path / "run"),
        "image": False,
        "image_format": "png",
        "animation": False,
        "animation_format": "gif",
        "animation_fps": 30,
        "include_sequence": False,
        "hdf5": True,
        "notebook_analysis": False,
        "notebook_repro": False,
        "csv": False,
        "csv_format": "csv",
    }
    with (
        patch("blochsimulator.ui.main_window.ExportDataDialog", return_value=dialog),
        patch.object(window.simulator, "save_results") as save_results,
        patch("blochsimulator.ui.main_window.QMessageBox.information"),
    ):
        window.export_results()

    assert save_results.call_args.args[1] is captured["sequence"]
    assert save_results.call_args.args[2] is captured["simulation"]
    assert save_results.call_args.args[2]["num_positions"] == 3
    assert save_results.call_args.args[2]["field_strength_t"] == 7.0


def test_simulation_controls_have_explanatory_tooltips():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    controls = (
        window.mode_combo,
        window.pos_spin,
        window.pos_range,
        window.freq_spin,
        window.freq_center,
        window.freq_range,
        window.freq_axis_mode,
        window.time_step_spin,
        window.extra_tail_spin,
        window.max_traces_spin,
    )

    registered = dict(window._tooltip_registry)
    assert all(registered.get(control, "").strip() for control in controls)


def test_all_main_window_fields_have_tooltips():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    field_types = (
        QAbstractSpinBox,
        QCheckBox,
        QComboBox,
        QRadioButton,
        QSlider,
    )
    registered = {field for field, _ in window._tooltip_registry}
    missing = [
        field for field in window.findChildren(field_types) if field not in registered
    ]

    assert not missing


def test_free_mode_result_controls_use_consistent_headers_and_rows():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()

    assert window.tab_widget.tabBar().font().bold()
    assert window.workspace_header.isHidden()
    assert window.workspace_switch.parentWidget() is window.free_mode_colormap_controls

    vector_layout = window.mag_3d.layout()
    assert vector_layout.indexOf(
        window.mag_3d.control_container
    ) < vector_layout.indexOf(window.mag_3d.gl_widget)
    assert window.mag_3d.vector_palette_combo.findText("Viridis") >= 0
    assert [
        window.mag_3d.spin_display_combo.itemText(index)
        for index in range(window.mag_3d.spin_display_combo.count())
    ] == ["Show all spins", "Show all spins + mean", "Show only mean"]
    rainbow = window.mag_3d.color_for_index(0, 5).name()
    window.mag_3d.vector_palette_combo.setCurrentText("Viridis")
    assert window.mag_3d.color_for_index(0, 5).name() != rainbow

    assert window.spectrum_controls_layout.indexOf(window.spectrum_component_combo) >= 0
    for control in (
        window.mean_only_checkbox,
        window.spatial_markers_checkbox,
        window.spatial_component_combo,
    ):
        assert window.spatial_options_layout.indexOf(control) >= 0

    assert window.mag_3d.page_title.isHidden()
    assert window.mag_3d.header_container.isHidden()
    assert window.mag_component.width() == window.signal_component.width() == 220
    assert window.rf_designer_tab.page_title.isHidden()
    assert window.slice_explorer.page_title.isHidden()
    assert window.param_sweep_widget.page_title.isHidden()
    assert window.rf_designer_tab.control_panel.minimumWidth() == 400
    assert window.slice_explorer.control_panel.minimumWidth() == 400
    assert window.rf_designer_tab.control_panel.maximumWidth() == 400
    assert window.slice_explorer.control_panel.maximumWidth() == 400


@pytest.mark.parametrize(
    ("display_mode", "show_spins", "show_mean"),
    [
        ("Show all spins", True, False),
        ("Show all spins + mean", True, True),
        ("Show only mean", False, True),
    ],
)
def test_3d_spin_display_mode_does_not_change_tip_path_tracking(
    display_mode, show_spins, show_mean
):
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = MagnetizationViewer()
    viewer.vector_plot = MagicMock()
    viewer.mean_vector = MagicMock()
    viewer.spin_display_combo.setCurrentText(display_mode)
    viewer.vector_plot.reset_mock()
    viewer.mean_vector.reset_mock()
    viewer._clear_path()

    viewer.update_magnetization(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    )

    viewer.vector_plot.setVisible.assert_called_with(show_spins)
    viewer.mean_vector.setVisible.assert_called_with(show_mean)
    assert viewer.track_checkbox.isChecked()
    assert len(viewer.path_points) == 1


def test_core_dialog_fields_have_tooltips(tmp_path):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialogs = (
        SettingsDialog(MemoryPolicy(), tmp_path, tooltips_enabled=True),
        PulseImportDialog(),
    )
    field_types = (QAbstractSpinBox, QCheckBox, QComboBox, QLineEdit)
    missing = [
        field
        for dialog in dialogs
        for field in dialog.findChildren(field_types)
        if not (
            isinstance(field, QLineEdit)
            and isinstance(field.parentWidget(), QAbstractSpinBox)
        )
        if not field.toolTip().strip()
    ]

    assert not missing


def test_tooltips_can_be_disabled_and_restored():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    window._set_tooltips_enabled(False)
    assert all(not widget.toolTip() for widget, _ in window._tooltip_registry)
    assert all(
        not window.tab_widget.tabToolTip(index)
        for index, _ in window._tab_tooltip_registry
    )

    window._set_tooltips_enabled(True)
    assert all(widget.toolTip() == text for widget, text in window._tooltip_registry)
    assert all(
        window.tab_widget.tabToolTip(index) == text
        for index, text in window._tab_tooltip_registry
    )


def test_configured_export_directory_is_used(tmp_path, monkeypatch):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    monkeypatch.delenv("BLOCH_EXPORT_DIR", raising=False)
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    export_directory = tmp_path / "custom-exports"
    window.app_settings.setValue("general/export_directory", str(export_directory))
    window.app_settings.setValue("sequence/kernel", "reference")
    window.app_settings.setValue("sequence/dynamic_kernel", "native_parallel")
    window.app_settings.setValue("sequence/timestep_preset", "fast")
    window.app_settings.setValue("sequence/timestep_us", 10.0)
    window.app_settings.setValue("sequence/spoiler_mode", "gradient")
    window.app_settings.setValue("sequence/subvoxel_spins_x", 3)
    window.app_settings.setValue("sequence/subvoxel_spins_y", 5)
    window.app_settings.setValue("sequence/subvoxel_spins_z", 11)
    window.app_settings.setValue("simulation/thread_mode", "manual")
    window.app_settings.setValue("simulation/manual_threads", 3)

    assert window._get_export_directory() == export_directory
    assert export_directory.is_dir()
    assert window._load_sequence_kernel() == "reference"
    assert window._load_dynamic_sequence_kernel() == "native_parallel"
    assert window._load_sequence_timestep_preset() == "fast"
    assert window._load_sequence_timestep_us() == pytest.approx(10.0)
    assert window._load_sequence_spoiler_mode() == "gradient"
    assert window._load_subvoxel_spin_counts() == (3, 5, 11)
    assert window._load_configured_num_threads() == 3

    window._ensure_sequence_simulation_workspace(window.sequence_simulation_tab_index)
    assert window.sequence_simulation_widget._export_directory() == export_directory
    window.sequence_simulation_widget.result = object()
    with patch(
        "blochsimulator.ui.sequence_simulation_widget.QFileDialog.getSaveFileName",
        return_value=("", ""),
    ) as sequence_export_dialog:
        window.sequence_simulation_widget._export_results()
    assert Path(sequence_export_dialog.call_args.args[2]).parent == export_directory

    window.app_settings.setValue("sequence/kernel", "invalid")
    assert window._load_sequence_kernel() == "optimized"
    window.app_settings.setValue("sequence/dynamic_kernel", "invalid")
    assert window._load_dynamic_sequence_kernel() == "optimized"
    window.app_settings.setValue("sequence/dynamic_kernel", "metal_hybrid")
    assert window._load_dynamic_sequence_kernel() == "metal_hybrid"
    window.app_settings.setValue("sequence/spoiler_mode", "invalid")
    assert window._load_sequence_spoiler_mode() == "ideal"


def test_simulation_settings_are_persisted_and_applied(tmp_path):
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    window.sequence_simulation_widget = MagicMock()
    dialog = MagicMock()
    dialog.exec_.return_value = QDialog.Accepted
    dialog.get_export_directory.return_value = tmp_path
    dialog.get_policy.return_value = MemoryPolicy()
    dialog.tooltips_enabled.return_value = True
    dialog.sequence_live_progress_enabled.return_value = False
    dialog.sequence_kernel.return_value = "reference"
    dialog.dynamic_sequence_kernel.return_value = "native_parallel"
    dialog.sequence_timestep_preset.return_value = "fast"
    dialog.sequence_timestep_us.return_value = 10.0
    dialog.sequence_spoiler_mode.return_value = "gradient"
    dialog.subvoxel_spin_counts.return_value = (3, 5, 11)
    dialog.subvoxel_sampling_method.return_value = "stratified"
    dialog.thread_mode.return_value = "manual"
    dialog.manual_thread_count.return_value = 2
    dialog.animation_memory_budget_bytes.return_value = 2048 * 1024**2
    scanner_parameters = ScannerParameters(max_grad_mtm=40.0, max_slew_tms=180.0)
    dialog.scanner_parameters.return_value = scanner_parameters
    workspace_defaults = WorkspaceDefaults(
        sequence_fov_mm=(180.0, 170.0, 80.0),
        phantom_fov_mm=(60.0, 50.0, 40.0),
        phantom_nucleus="C13",
        field_strength_t=7.0,
    )
    dialog.workspace_defaults.return_value = workspace_defaults

    with patch("blochsimulator.ui.main_window.SettingsDialog", return_value=dialog):
        window.show_settings(initial_tab="simulation")

    assert window.app_settings.value("sequence/timestep_preset") == "fast"
    assert float(window.app_settings.value("sequence/timestep_us")) == 10.0
    assert window.app_settings.value("sequence/spoiler_mode") == "gradient"
    assert int(window.app_settings.value("sequence/subvoxel_spins_x")) == 3
    assert int(window.app_settings.value("sequence/subvoxel_spins_y")) == 5
    assert int(window.app_settings.value("sequence/subvoxel_spins_z")) == 11
    assert (
        window.app_settings.value("sequence/subvoxel_sampling_method") == "stratified"
    )
    assert window.app_settings.value("simulation/thread_mode") == "manual"
    assert int(window.app_settings.value("simulation/manual_threads")) == 2
    assert int(window.app_settings.value("memory/animation_replay_mib")) == 2048
    assert float(window.app_settings.value("scanner/max_grad_mtm")) == 40.0
    assert float(window.app_settings.value("scanner/max_slew_tms")) == 180.0
    assert float(window.app_settings.value("defaults/sequence_fov_x_mm")) == 180.0
    assert window.app_settings.value("defaults/phantom_nucleus") == "C13"
    assert float(window.app_settings.value("defaults/field_strength_t")) == 7.0
    assert window.simulator.sequence_kernel == "reference"
    assert window.simulator.dynamic_sequence_kernel == "native_parallel"
    assert window.simulator.num_threads == 2
    window.sequence_simulation_widget.set_sequence_timestep_us.assert_called_once_with(
        10.0
    )
    window.sequence_simulation_widget.set_dynamic_sequence_kernel.assert_called_once_with(
        "native_parallel"
    )
    window.sequence_simulation_widget.set_spoiler_configuration.assert_called_once_with(
        "gradient", (3, 5, 11), "stratified"
    )
    window.sequence_simulation_widget.set_thread_configuration.assert_called_once_with(
        "manual", 2
    )
    window.sequence_simulation_widget.set_animation_memory_budget_bytes.assert_called_once_with(
        2048 * 1024**2
    )
    window.sequence_simulation_widget.set_scanner_parameters.assert_called_once_with(
        scanner_parameters
    )
    window.sequence_simulation_widget.set_workspace_defaults.assert_called_once_with(
        workspace_defaults
    )


def test_changing_only_spoiler_settings_preserves_active_sequence_parameters(
    tmp_path,
):
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    window.sequence_simulation_widget = MagicMock()
    scanner_parameters = window._load_scanner_parameters()
    workspace_defaults = WorkspaceDefaults.from_settings(window.app_settings)
    dialog = MagicMock()
    dialog.exec_.return_value = QDialog.Accepted
    dialog.get_export_directory.return_value = tmp_path
    dialog.get_policy.return_value = MemoryPolicy()
    dialog.tooltips_enabled.return_value = True
    dialog.sequence_live_progress_enabled.return_value = True
    dialog.sequence_kernel.return_value = "optimized"
    dialog.dynamic_sequence_kernel.return_value = "optimized"
    dialog.sequence_timestep_preset.return_value = "balanced"
    dialog.sequence_timestep_us.return_value = 5.0
    dialog.sequence_spoiler_mode.return_value = "gradient"
    dialog.subvoxel_spin_counts.return_value = (1, 1, 17)
    dialog.subvoxel_sampling_method.return_value = "midpoint"
    dialog.thread_mode.return_value = "automatic"
    dialog.manual_thread_count.return_value = 4
    dialog.animation_memory_budget_bytes.return_value = 512 * 1024**2
    dialog.scanner_parameters.return_value = scanner_parameters
    dialog.workspace_defaults.return_value = workspace_defaults

    with patch("blochsimulator.ui.main_window.SettingsDialog", return_value=dialog):
        window.show_settings(initial_tab="simulation")

    window.sequence_simulation_widget.set_spoiler_configuration.assert_called_once_with(
        "gradient", (1, 1, 17), "midpoint"
    )
    window.sequence_simulation_widget.set_scanner_parameters.assert_not_called()
    window.sequence_simulation_widget.set_workspace_defaults.assert_not_called()


def test_workspace_defaults_round_trip_through_settings(tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    expected = WorkspaceDefaults(
        sequence_fov_mm=(210.0, 190.0, 75.0),
        phantom_fov_mm=(80.0, 70.0, 60.0),
        phantom_nucleus="P31",
        field_strength_t=9.4,
    )
    expected.save(settings)
    settings.sync()

    assert WorkspaceDefaults.from_settings(settings) == expected


def test_legacy_automatic_nucleus_default_migrates_to_shared_h1(tmp_path):
    settings = QSettings(str(tmp_path / "legacy-settings.ini"), QSettings.IniFormat)
    settings.setValue("defaults/phantom_nucleus", "auto")

    assert WorkspaceDefaults.from_settings(settings).phantom_nucleus == "H1"


if __name__ == "__main__":
    pytest.main([__file__])
