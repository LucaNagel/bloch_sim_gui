from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QDialogButtonBox,
)
from typing import Optional
from pathlib import Path

from ..memory import MemoryPolicy, format_bytes, resolve_memory_budget


class MemorySettingsDialog(QDialog):
    """Configure the RAM budget used by simulations."""

    MODES = (
        ("Automatic reserve (recommended)", "automatic"),
        ("Custom free-memory reserve", "custom_reserve"),
        ("Fixed simulation limit", "fixed_limit"),
    )

    def __init__(self, policy: MemoryPolicy, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Memory Settings")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("memory_policy_mode")
        for label, mode in self.MODES:
            self.mode_combo.addItem(label, mode)
        mode_index = self.mode_combo.findData(policy.mode)
        self.mode_combo.setCurrentIndex(max(0, mode_index))
        form.addRow("Policy:", self.mode_combo)

        self.reserve_spin = QDoubleSpinBox()
        self.reserve_spin.setObjectName("memory_reserve_gib")
        self.reserve_spin.setRange(0.25, 256.0)
        self.reserve_spin.setDecimals(2)
        self.reserve_spin.setSingleStep(0.5)
        self.reserve_spin.setSuffix(" GiB")
        self.reserve_spin.setValue(policy.reserve_bytes / 1024**3)
        form.addRow("Keep free:", self.reserve_spin)

        self.limit_spin = QDoubleSpinBox()
        self.limit_spin.setObjectName("memory_limit_gib")
        self.limit_spin.setRange(0.25, 1024.0)
        self.limit_spin.setDecimals(2)
        self.limit_spin.setSingleStep(1.0)
        self.limit_spin.setSuffix(" GiB")
        self.limit_spin.setValue(policy.limit_bytes / 1024**3)
        form.addRow("Maximum per simulation:", self.limit_spin)

        layout.addLayout(form)

        explanation = QLabel(
            "Automatic mode keeps at least 2 GiB or 10% of total system RAM "
            "free, whichever is larger. The fixed limit still preserves a "
            "512 MiB emergency reserve."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.status_label = QLabel()
        self.status_label.setObjectName("memory_budget_summary")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.mode_combo.currentIndexChanged.connect(self._update_summary)
        self.reserve_spin.valueChanged.connect(self._update_summary)
        self.limit_spin.valueChanged.connect(self._update_summary)
        self._update_summary()

    def get_policy(self) -> MemoryPolicy:
        return MemoryPolicy(
            mode=str(self.mode_combo.currentData()),
            reserve_bytes=int(self.reserve_spin.value() * 1024**3),
            limit_bytes=int(self.limit_spin.value() * 1024**3),
        )

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
        form.addRow("Amplitude unit:", self.amp_unit)

        self.phase_unit = QComboBox()
        self.phase_unit.setObjectName("import_phase_unit")
        self.phase_unit.addItems(["Degrees", "Radians"])
        self.phase_unit.setCurrentIndex(0)
        form.addRow("Phase unit:", self.phase_unit)

        self.duration_ms = QDoubleSpinBox()
        self.duration_ms.setObjectName("import_duration_ms")
        self.duration_ms.setRange(0.001, 100000.0)
        self.duration_ms.setDecimals(3)
        self.duration_ms.setSingleStep(0.1)
        self.duration_ms.setValue(1.0)
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
