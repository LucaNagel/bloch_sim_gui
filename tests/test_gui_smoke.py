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
    QGridLayout,
    QLineEdit,
    QRadioButton,
    QSlider,
)
from blochsimulator.memory import GIB, MemoryPolicy
from blochsimulator.ui.dialogs import PulseImportDialog, SettingsDialog
from blochsimulator.ui.main_window import BlochSimulatorGUI


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

    dialog = SettingsDialog(MemoryPolicy(), tmp_path, tooltips_enabled=True)
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData("custom_reserve"))
    dialog.reserve_spin.setValue(3.5)

    policy = dialog.get_policy()
    assert policy.mode == "custom_reserve"
    assert policy.reserve_bytes == int(3.5 * GIB)
    assert dialog.reserve_spin.isEnabled()
    assert not dialog.limit_spin.isEnabled()
    assert dialog.get_export_directory() == tmp_path
    assert dialog.tooltips_enabled()
    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
        "General",
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

    assert window._get_export_directory() == export_directory
    assert export_directory.is_dir()


if __name__ == "__main__":
    pytest.main([__file__])
