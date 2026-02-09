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
