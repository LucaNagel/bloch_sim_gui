import pytest
import sys
from unittest.mock import MagicMock, patch
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
from blochsimulator.ui.dialogs import PulseImportDialog, SettingsDialog
from blochsimulator.ui.main_window import BlochSimulatorGUI


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
    assert dialog.sequence_timestep_preset() == "balanced"
    assert dialog.sequence_timestep_us() == pytest.approx(5.0)
    assert not dialog.sequence_timestep_us_spin.isEnabled()
    dialog.sequence_timestep_preset_combo.setCurrentIndex(
        dialog.sequence_timestep_preset_combo.findData("custom")
    )
    dialog.sequence_timestep_us_spin.setValue(7.5)
    assert dialog.sequence_timestep_us() == pytest.approx(7.5)
    assert dialog.sequence_timestep_us_spin.isEnabled()
    assert dialog.thread_mode() == "automatic"
    assert not dialog.manual_thread_count_spin.isEnabled()
    dialog.thread_mode_combo.setCurrentIndex(
        dialog.thread_mode_combo.findData("manual")
    )
    dialog.manual_thread_count_spin.setValue(3)
    assert dialog.thread_mode() == "manual"
    assert dialog.manual_thread_count() == 3
    assert dialog.manual_thread_count_spin.isEnabled()
    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
        "General",
        "Simulation",
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

    window._set_tooltips_enabled(True)
    assert all(widget.toolTip() == text for widget, text in window._tooltip_registry)


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
    window.app_settings.setValue("sequence/timestep_preset", "fast")
    window.app_settings.setValue("sequence/timestep_us", 10.0)
    window.app_settings.setValue("simulation/thread_mode", "manual")
    window.app_settings.setValue("simulation/manual_threads", 3)

    assert window._get_export_directory() == export_directory
    assert export_directory.is_dir()
    assert window._load_sequence_kernel() == "reference"
    assert window._load_sequence_timestep_preset() == "fast"
    assert window._load_sequence_timestep_us() == pytest.approx(10.0)
    assert window._load_configured_num_threads() == 3

    window.app_settings.setValue("sequence/kernel", "invalid")
    assert window._load_sequence_kernel() == "optimized"


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
    dialog.sequence_timestep_preset.return_value = "fast"
    dialog.sequence_timestep_us.return_value = 10.0
    dialog.thread_mode.return_value = "manual"
    dialog.manual_thread_count.return_value = 2

    with patch("blochsimulator.ui.main_window.SettingsDialog", return_value=dialog):
        window.show_settings(initial_tab="simulation")

    assert window.app_settings.value("sequence/timestep_preset") == "fast"
    assert float(window.app_settings.value("sequence/timestep_us")) == 10.0
    assert window.app_settings.value("simulation/thread_mode") == "manual"
    assert int(window.app_settings.value("simulation/manual_threads")) == 2
    assert window.simulator.sequence_kernel == "reference"
    assert window.simulator.num_threads == 2
    window.sequence_simulation_widget.set_sequence_timestep_us.assert_called_once_with(
        10.0
    )
    window.sequence_simulation_widget.set_thread_configuration.assert_called_once_with(
        "manual", 2
    )


if __name__ == "__main__":
    pytest.main([__file__])
