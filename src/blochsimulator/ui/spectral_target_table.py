"""Per-metabolite target editor for spectrally selective sequences.

Each row holds one spectral target: its name, the RF carrier offset, the
receiver demodulation offset, the nominal flip angle and the startup flip
angle. Every number has its own spin box, so the values can be edited one
metabolite at a time and varied by a parameter loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# (field, column header, loop label, suffix, minimum, maximum, decimals)
TARGET_FIELDS = (
    ("rf_offset_hz", "RF offset", "RF target offset", " Hz", -1e6, 1e6, 3),
    ("receiver_offset_hz", "Receiver offset", "Receiver offset", " Hz", -1e6, 1e6, 5),
    ("flip_angle_deg", "Flip angle", "Target flip angle", "°", 0.0, 3600.0, 2),
    (
        "startup_flip_angle_deg",
        "Startup flip",
        "Startup flip angle",
        "°",
        0.0,
        3600.0,
        2,
    ),
)
STARTUP_FIELD = "startup_flip_angle_deg"


@dataclass
class TargetRow:
    name: QLineEdit
    spins: Dict[str, QDoubleSpinBox]
    remove: QToolButton


class SpectralTargetTable(QWidget):
    """Edit spectral targets as one row of named numeric fields per metabolite."""

    changed = pyqtSignal()

    def __init__(self, parent=None, *, prefix="spectral_target"):
        super().__init__(parent)
        self._prefix = prefix
        self.rows: List[TargetRow] = []
        self._startup_visible = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(6)
        self._headers = [QLabel("Metabolite")]
        self._headers.extend(QLabel(header) for _, header, *_ in TARGET_FIELDS)
        for column, header in enumerate(self._headers):
            self._grid.addWidget(header, 0, column)
        layout.addLayout(self._grid)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add metabolite")
        self.add_button.setObjectName(f"{prefix}_add")
        self.add_button.clicked.connect(self._add_clicked)
        buttons.addWidget(self.add_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    def _make_spin(self, field, value):
        _, _, label, suffix, minimum, maximum, decimals = next(
            item for item in TARGET_FIELDS if item[0] == field
        )
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        spin.setValue(float(value))
        spin.setToolTip(label)
        spin.valueChanged.connect(self._value_changed)
        return spin

    def _append_row(self, name, values):
        row_index = len(self.rows) + 1
        name_edit = QLineEdit(str(name))
        name_edit.setMinimumWidth(70)
        name_edit.setObjectName(f"{self._prefix}_{row_index}_name")
        name_edit.editingFinished.connect(self._value_changed)
        spins = {}
        for field, *_ in TARGET_FIELDS:
            spin = self._make_spin(field, values.get(field, 0.0))
            spin.setObjectName(f"{self._prefix}_{row_index}_{field}")
            spins[field] = spin
        remove = QToolButton()
        remove.setText("×")
        remove.setToolTip("Remove this metabolite")
        row = TargetRow(name_edit, spins, remove)
        remove.clicked.connect(lambda _checked=False, row=row: self._remove(row))
        self._grid.addWidget(name_edit, row_index, 0)
        for column, (field, *_) in enumerate(TARGET_FIELDS, start=1):
            self._grid.addWidget(spins[field], row_index, column)
        self._grid.addWidget(remove, row_index, len(TARGET_FIELDS) + 1)
        self.rows.append(row)
        spins[STARTUP_FIELD].setVisible(self._startup_visible)

    def _clear_rows(self):
        for row in self.rows:
            for widget in (row.name, *row.spins.values(), row.remove):
                self._grid.removeWidget(widget)
                widget.deleteLater()
        self.rows = []

    def set_targets(self, names: Sequence[str], values: Dict[str, Sequence[float]]):
        """Show ``names`` with per-field ``values``; keeps rows when the count matches.

        Keeping the existing spin boxes matters while a parameter loop holds a
        reference to one of them.
        """
        names = list(names)
        if len(names) != len(self.rows):
            self._clear_rows()
            for index, name in enumerate(names):
                self._append_row(
                    name,
                    {
                        field: self._value_at(values.get(field, ()), index)
                        for field, *_ in TARGET_FIELDS
                    },
                )
        else:
            for index, (row, name) in enumerate(zip(self.rows, names)):
                if row.name.text() != name:
                    row.name.setText(name)
                for field, spin in row.spins.items():
                    value = self._value_at(values.get(field, ()), index)
                    if abs(spin.value() - value) > 0.5 * 10.0 ** -spin.decimals():
                        blocked = spin.blockSignals(True)
                        spin.setValue(value)
                        spin.blockSignals(blocked)
        self._update_remove_buttons()

    @staticmethod
    def _value_at(values, index):
        values = list(values)
        if index < len(values):
            return float(values[index])
        return float(values[-1]) if values else 0.0

    def targets(self):
        """Return ``(names, {field: values})`` in row order."""
        names = [row.name.text().strip() for row in self.rows]
        values = {
            field: [row.spins[field].value() for row in self.rows]
            for field, *_ in TARGET_FIELDS
        }
        return names, values

    def set_startup_visible(self, visible: bool):
        self._startup_visible = bool(visible)
        self._headers[-1].setVisible(self._startup_visible)
        for row in self.rows:
            row.spins[STARTUP_FIELD].setVisible(self._startup_visible)

    def loop_controls(self):
        """Return ``(key, label, spin)`` for every numeric field of every row."""
        controls = []
        for index, row in enumerate(self.rows):
            name = row.name.text().strip() or f"Target {index + 1}"
            for field, _, label, *_ in TARGET_FIELDS:
                spin = row.spins[field]
                if not spin.isVisibleTo(self) or not spin.isEnabled():
                    continue
                controls.append(
                    (f"{self._prefix}[{index}].{field}", f"{name}: {label}", spin)
                )
        return controls

    def _update_remove_buttons(self):
        for row in self.rows:
            row.remove.setEnabled(len(self.rows) > 1)

    def _add_clicked(self):
        names, values = self.targets()
        last = {
            field: (values[field][-1] if values[field] else 0.0) for field in values
        }
        self._append_row(f"Target {len(self.rows) + 1}", last)
        self._update_remove_buttons()
        self.changed.emit()

    def _remove(self, row):
        if len(self.rows) <= 1 or row not in self.rows:
            return
        names, values = self.targets()
        index = self.rows.index(row)
        del names[index]
        for field in values:
            del values[field][index]
        self._clear_rows()
        self.set_targets(names, values)
        self.changed.emit()

    def _value_changed(self, *_):
        self.changed.emit()
