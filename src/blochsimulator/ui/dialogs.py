from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QDialogButtonBox,
    QFileDialog,
    QCheckBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QWidget,
)
from typing import Optional
from pathlib import Path

from ..memory import MemoryPolicy, format_bytes, resolve_memory_budget


class SettingsDialog(QDialog):
    """Configure persistent application, memory and interface settings."""

    MODES = (
        ("Automatic reserve (recommended)", "automatic"),
        ("Custom free-memory reserve", "custom_reserve"),
        ("Fixed simulation limit", "fixed_limit"),
    )

    def __init__(
        self,
        policy: MemoryPolicy,
        export_directory: Path,
        tooltips_enabled: bool,
        parent=None,
        initial_tab: str = "general",
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(600)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("settings_tabs")
        layout.addWidget(self.tabs)

        general_tab = QWidget()
        general_form = QFormLayout(general_tab)
        export_layout = QHBoxLayout()
        self.export_directory_edit = QLineEdit(str(export_directory))
        self.export_directory_edit.setObjectName("default_export_directory")
        self.export_directory_edit.setToolTip(
            "Default folder offered by image, data, animation and notebook export dialogs."
        )
        export_layout.addWidget(self.export_directory_edit, 1)
        self.export_browse_button = QPushButton("Browse...")
        self.export_browse_button.setToolTip(
            "Choose an existing folder as the default export location."
        )
        self.export_browse_button.clicked.connect(self._browse_export_directory)
        export_layout.addWidget(self.export_browse_button)
        general_form.addRow("Default export directory:", export_layout)
        self.tabs.addTab(general_tab, "General")

        memory_tab = QWidget()
        memory_layout = QVBoxLayout(memory_tab)
        form = QFormLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("memory_policy_mode")
        for label, mode in self.MODES:
            self.mode_combo.addItem(label, mode)
        mode_index = self.mode_combo.findData(policy.mode)
        self.mode_combo.setCurrentIndex(max(0, mode_index))
        self.mode_combo.setToolTip(
            "Choose automatic RAM reservation, a custom amount of memory kept "
            "free, or a fixed maximum allocation per simulation."
        )
        form.addRow("Policy:", self.mode_combo)

        self.reserve_spin = QDoubleSpinBox()
        self.reserve_spin.setObjectName("memory_reserve_gib")
        self.reserve_spin.setRange(0.25, 256.0)
        self.reserve_spin.setDecimals(2)
        self.reserve_spin.setSingleStep(0.5)
        self.reserve_spin.setSuffix(" GiB")
        self.reserve_spin.setValue(policy.reserve_bytes / 1024**3)
        self.reserve_spin.setToolTip(
            "Amount of currently available RAM that must remain unused by the simulation."
        )
        form.addRow("Keep free:", self.reserve_spin)

        self.limit_spin = QDoubleSpinBox()
        self.limit_spin.setObjectName("memory_limit_gib")
        self.limit_spin.setRange(0.25, 1024.0)
        self.limit_spin.setDecimals(2)
        self.limit_spin.setSingleStep(1.0)
        self.limit_spin.setSuffix(" GiB")
        self.limit_spin.setValue(policy.limit_bytes / 1024**3)
        self.limit_spin.setToolTip(
            "Maximum estimated RAM allocation permitted for one simulation."
        )
        form.addRow("Maximum per simulation:", self.limit_spin)

        memory_layout.addLayout(form)

        explanation = QLabel(
            "Automatic mode keeps at least 2 GiB or 10% of total system RAM "
            "free, whichever is larger. The fixed limit still preserves a "
            "512 MiB emergency reserve."
        )
        explanation.setWordWrap(True)
        memory_layout.addWidget(explanation)

        self.status_label = QLabel()
        self.status_label.setObjectName("memory_budget_summary")
        self.status_label.setWordWrap(True)
        memory_layout.addWidget(self.status_label)
        memory_layout.addStretch()
        self.tabs.addTab(memory_tab, "Memory")

        interface_tab = QWidget()
        interface_layout = QVBoxLayout(interface_tab)
        self.tooltips_checkbox = QCheckBox("Show explanatory tooltips")
        self.tooltips_checkbox.setObjectName("tooltips_enabled")
        self.tooltips_checkbox.setChecked(bool(tooltips_enabled))
        self.tooltips_checkbox.setToolTip(
            "Show short explanations when the pointer rests over controls and input fields."
        )
        interface_layout.addWidget(self.tooltips_checkbox)
        interface_layout.addStretch()
        self.tabs.addTab(interface_tab, "Interface")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.mode_combo.currentIndexChanged.connect(self._update_summary)
        self.reserve_spin.valueChanged.connect(self._update_summary)
        self.limit_spin.valueChanged.connect(self._update_summary)
        self._update_summary()

        tab_names = {"general": 0, "memory": 1, "interface": 2}
        self.tabs.setCurrentIndex(tab_names.get(initial_tab, 0))

    def get_policy(self) -> MemoryPolicy:
        return MemoryPolicy(
            mode=str(self.mode_combo.currentData()),
            reserve_bytes=int(self.reserve_spin.value() * 1024**3),
            limit_bytes=int(self.limit_spin.value() * 1024**3),
        )

    def get_export_directory(self) -> Path:
        return Path(self.export_directory_edit.text()).expanduser()

    def tooltips_enabled(self) -> bool:
        return self.tooltips_checkbox.isChecked()

    def _browse_export_directory(self):
        current = str(self.get_export_directory())
        selected = QFileDialog.getExistingDirectory(
            self, "Select Default Export Directory", current
        )
        if selected:
            self.export_directory_edit.setText(selected)

    def _update_summary(self):
        mode = str(self.mode_combo.currentData())
        self.reserve_spin.setEnabled(mode == "custom_reserve")
        self.limit_spin.setEnabled(mode == "fixed_limit")

        budget = resolve_memory_budget(policy=self.get_policy())
        if budget.available_bytes is None:
            summary = (
                f"Hardware memory could not be detected. Effective budget: "
                f"{format_bytes(budget.limit_bytes)}."
            )
        else:
            summary = (
                f"System RAM: {format_bytes(budget.total_bytes or 0)} total, "
                f"{format_bytes(budget.available_bytes)} currently available. "
                f"Effective budget now: {format_bytes(budget.limit_bytes)}."
            )
        self.status_label.setText(summary)


class PulseImportDialog(QDialog):
    """Dialog to configure loading of custom amp/phase pulse files."""

    def __init__(self, parent=None, filename: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Import RF Pulse Options")
        layout = QVBoxLayout()
        form = QFormLayout()

        if filename:
            form.addRow(QLabel(f"File: {Path(filename).name}"))

        self.layout_mode = QComboBox()
        self.layout_mode.setObjectName("import_layout_mode")
        self.layout_mode.addItems(
            [
                "Interleaved: amp, phase, amp, phase",
                "Interleaved: phase, amp, phase, amp",
                "Columns: amp | phase per row",
            ]
        )
        self.layout_mode.setCurrentIndex(0)
        self.layout_mode.setToolTip(
            "Describe how amplitude and phase values are arranged in the imported text file."
        )
        form.addRow("Data layout:", self.layout_mode)

        self.amp_unit = QComboBox()
        self.amp_unit.setObjectName("import_amp_unit")
        self.amp_unit.addItems(
            [
                "Percent (0-100)",
                "Fraction (0-1)",
                "Gauss",
                "mT",
                "uT",
            ]
        )
        self.amp_unit.setCurrentIndex(0)
        self.amp_unit.setToolTip(
            "Physical or relative unit used by the imported RF amplitude values."
        )
        form.addRow("Amplitude unit:", self.amp_unit)

        self.phase_unit = QComboBox()
        self.phase_unit.setObjectName("import_phase_unit")
        self.phase_unit.addItems(["Degrees", "Radians"])
        self.phase_unit.setCurrentIndex(0)
        self.phase_unit.setToolTip("Angular unit used by imported RF phase values.")
        form.addRow("Phase unit:", self.phase_unit)

        self.duration_ms = QDoubleSpinBox()
        self.duration_ms.setObjectName("import_duration_ms")
        self.duration_ms.setRange(0.001, 100000.0)
        self.duration_ms.setDecimals(3)
        self.duration_ms.setSingleStep(0.1)
        self.duration_ms.setValue(1.0)
        self.duration_ms.setToolTip(
            "Total duration assigned to the imported RF waveform, in milliseconds."
        )
        form.addRow("Duration (ms):", self.duration_ms)

        layout.addLayout(form)
        layout.addWidget(
            QLabel(
                "Tip: Percent/fraction amplitudes are treated as relative and rescaled from flip angle."
            )
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_options(self) -> dict:
        layout_choice = self.layout_mode.currentText()
        if layout_choice.startswith("Interleaved: amp"):
            layout = "amp_phase_interleaved"
        elif layout_choice.startswith("Interleaved: phase"):
            layout = "phase_amp_interleaved"
        else:
            layout = "columns"

        amp_unit_text = self.amp_unit.currentText().lower()
        if "percent" in amp_unit_text:
            amp_unit = "percent"
        elif "fraction" in amp_unit_text:
            amp_unit = "fraction"
        elif amp_unit_text.startswith("mt"):
            amp_unit = "mt"
        elif amp_unit_text.startswith("ut"):
            amp_unit = "ut"
        else:
            amp_unit = "gauss"

        phase_unit = (
            "deg" if self.phase_unit.currentText().lower().startswith("deg") else "rad"
        )

        return {
            "layout": layout,
            "amp_unit": amp_unit,
            "phase_unit": phase_unit,
            "duration_s": float(self.duration_ms.value()) / 1000.0,
        }
