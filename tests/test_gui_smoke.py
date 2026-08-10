import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from PyQt5.QtCore import QSettings
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
from blochsimulator.ui.main_window import BlochSimulatorGUI, _apply_platform_style


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

    finally:
        # We don't want to show the window or start the event loop
        pass


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
    assert dialog.get_export_directory() == tmp_path
    assert dialog.tooltips_enabled()
    assert dialog.sequence_live_progress_enabled()
    assert dialog.sequence_kernel() == "optimized"
    dialog.sequence_kernel_combo.setCurrentIndex(
        dialog.sequence_kernel_combo.findData("reference")
    )
    assert dialog.sequence_kernel() == "reference"
    assert dialog.dynamic_sequence_kernel() == "optimized"
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
    assert dialog.subvoxel_spin_counts() == (1, 1, 9)
    assert not any(spin.isEnabled() for spin in dialog.subvoxel_spin_count_spins)
    dialog.sequence_spoiler_mode_combo.setCurrentIndex(
        dialog.sequence_spoiler_mode_combo.findData("gradient")
    )
    for spin, value in zip(dialog.subvoxel_spin_count_spins, (3, 5, 11)):
        spin.setValue(value)
    assert dialog.sequence_spoiler_mode() == "gradient"
    assert dialog.subvoxel_spin_counts() == (3, 5, 11)
    assert all(spin.isEnabled() for spin in dialog.subvoxel_spin_count_spins)
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

    assert window.pos_range.suffix() == " mm"
    assert window._collect_simulation_parameters(internal_format=True)[
        "pos_range_mm"
    ] == pytest.approx(20.0)
    parameters = window._collect_simulation_parameters()
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


def test_parameter_loader_migrates_legacy_cm_position_range(tmp_path):
    window = BlochSimulatorGUI()
    legacy_path = tmp_path / "legacy_cm.json"
    legacy_path.write_text(
        json.dumps({"version": "1.1", "simulation": {"pos_range": 2.0}})
    )

    with patch(
        "blochsimulator.ui.main_window.QFileDialog.getOpenFileName",
        return_value=(str(legacy_path), "JSON Files (*.json)"),
    ):
        window.load_parameters()

    assert window.pos_range.value() == pytest.approx(20.0)


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

    window.app_settings.setValue("sequence/kernel", "invalid")
    assert window._load_sequence_kernel() == "optimized"
    window.app_settings.setValue("sequence/dynamic_kernel", "invalid")
    assert window._load_dynamic_sequence_kernel() == "optimized"
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
    dialog.thread_mode.return_value = "manual"
    dialog.manual_thread_count.return_value = 2
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
    assert window.app_settings.value("simulation/thread_mode") == "manual"
    assert int(window.app_settings.value("simulation/manual_threads")) == 2
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
        "gradient", (3, 5, 11)
    )
    window.sequence_simulation_widget.set_thread_configuration.assert_called_once_with(
        "manual", 2
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
    dialog.thread_mode.return_value = "automatic"
    dialog.manual_thread_count.return_value = 4
    dialog.scanner_parameters.return_value = scanner_parameters
    dialog.workspace_defaults.return_value = workspace_defaults

    with patch("blochsimulator.ui.main_window.SettingsDialog", return_value=dialog):
        window.show_settings(initial_tab="simulation")

    window.sequence_simulation_widget.set_spoiler_configuration.assert_called_once_with(
        "gradient", (1, 1, 17)
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


if __name__ == "__main__":
    pytest.main([__file__])
