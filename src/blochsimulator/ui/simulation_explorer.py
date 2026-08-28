"""Session-only browser for completed sequence simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass
class SessionSimulationRun:
    """References and display metadata for one in-memory simulation run."""

    run_id: str
    created_at_utc: str
    sequence_name: str
    phantom_name: str
    phantom_shape: tuple
    sequence_duration_s: float
    adc_samples: int
    runtime_s: Optional[float]
    kernel: str
    result: object = field(repr=False)
    state: dict = field(repr=False, default_factory=dict)
    display_name: str = ""
    custom_name: bool = False
    run_type: str = "sequence"


def _date_text(value):
    try:
        return (
            datetime.fromisoformat(str(value))
            .astimezone()
            .strftime("%Y-%m-%d %H:%M:%S")
        )
    except (TypeError, ValueError):
        return str(value or "—")


def _duration_text(seconds):
    if seconds is None:
        return "—"
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return f"{seconds * 1000.0:.3g} ms"
    if seconds < 60.0:
        return f"{seconds:.3g} s"
    minutes, remaining = divmod(seconds, 60.0)
    return f"{int(minutes)} min {remaining:04.1f} s"


class SessionSimulationExplorer(QWidget):
    """Browse, reopen, and delete simulations retained for this app session."""

    run_open_requested = pyqtSignal(object)
    run_deleted = pyqtSignal(object)
    runs_export_requested = pyqtSignal(object)
    run_renamed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("session_simulation_explorer")
        self._runs = {}
        self._next_run_number = 1

        root = QVBoxLayout(self)
        heading = QLabel("Session simulations")
        heading_font = heading.font()
        heading_font.setBold(True)
        heading_font.setPointSize(max(heading_font.pointSize() + 2, 12))
        heading.setFont(heading_font)
        root.addWidget(heading)
        note = QLabel(
            "Completed sequence simulations are kept temporarily in memory. "
            "They disappear when the application closes."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        splitter = QSplitter(Qt.Vertical)
        self.run_tree = QTreeWidget()
        self.run_tree.setObjectName("session_simulation_tree")
        self.run_tree.setHeaderLabels(
            ["Run", "Sequence", "Phantom", "Completed", "Runtime"]
        )
        self.run_tree.setRootIsDecorated(False)
        self.run_tree.setAlternatingRowColors(True)
        self.run_tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.run_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        header = self.run_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.run_tree.itemSelectionChanged.connect(self._selection_changed)
        self.run_tree.itemDoubleClicked.connect(lambda *_: self._open_selected())
        splitter.addWidget(self.run_tree)

        self.details_table = QTableWidget(0, 2)
        self.details_table.setObjectName("session_simulation_details")
        self.details_table.setHorizontalHeaderLabels(["Metadata", "Value"])
        self.details_table.verticalHeader().setVisible(False)
        self.details_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.details_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.details_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.details_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.details_table.setAlternatingRowColors(True)
        self.details_table.setWordWrap(True)
        splitter.addWidget(self.details_table)
        splitter.setSizes([420, 240])
        root.addWidget(splitter, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel("No simulations in this session")
        footer.addWidget(self.status_label, 1)
        self.open_button = QPushButton("Show simulation")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_selected)
        footer.addWidget(self.open_button)
        self.export_button = QPushButton("Export selected…")
        self.export_button.setEnabled(False)
        self.export_button.setToolTip(
            "Save one or more selected session simulations as Bloch projects"
        )
        self.export_button.clicked.connect(self._export_selected)
        footer.addWidget(self.export_button)
        self.rename_button = QPushButton("Rename…")
        self.rename_button.setEnabled(False)
        self.rename_button.setToolTip(
            "Rename one simulation and use that name for future project exports"
        )
        self.rename_button.clicked.connect(self._rename_selected)
        footer.addWidget(self.rename_button)
        self.delete_button = QPushButton("Delete run")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self._delete_selected)
        footer.addWidget(self.delete_button)
        root.addLayout(footer)

    @property
    def runs(self):
        return tuple(self._runs.values())

    def add_run(self, run):
        """Add a completed run and select its metadata row."""
        if run.run_id in self._runs:
            return run
        if not run.display_name:
            run.display_name = f"Run {self._next_run_number}"
        self._next_run_number += 1
        self._runs[run.run_id] = run
        item = QTreeWidgetItem(
            [
                run.display_name,
                run.sequence_name,
                run.phantom_name,
                _date_text(run.created_at_utc),
                _duration_text(run.runtime_s),
            ]
        )
        item.setData(0, Qt.UserRole, run.run_id)
        self.run_tree.insertTopLevelItem(0, item)
        self.run_tree.setCurrentItem(item)
        self._selection_changed()
        return run

    def selected_run(self):
        item = self.run_tree.currentItem()
        if item is None:
            return None
        return self._runs.get(item.data(0, Qt.UserRole))

    def selected_runs(self):
        selected_ids = {
            item.data(0, Qt.UserRole) for item in self.run_tree.selectedItems()
        }
        return tuple(
            run
            for row in range(self.run_tree.topLevelItemCount())
            for item in (self.run_tree.topLevelItem(row),)
            for run in (self._runs.get(item.data(0, Qt.UserRole)),)
            if run is not None and run.run_id in selected_ids
        )

    def _selection_changed(self):
        run = self.selected_run()
        selected = self.selected_runs()
        self.open_button.setEnabled(run is not None)
        self.export_button.setEnabled(bool(selected))
        self.rename_button.setEnabled(len(selected) == 1)
        self.delete_button.setEnabled(bool(selected))
        self.delete_button.setText(
            "Delete selected" if len(selected) > 1 else "Delete run"
        )
        self._show_details(run)
        count = len(self._runs)
        self.status_label.setText(
            "No simulations in this session"
            if count == 0
            else f"{count} simulation{'s' if count != 1 else ''} retained in memory"
        )

    def _show_details(self, run):
        if run is None:
            self.details_table.setRowCount(0)
            return
        shape = " × ".join(str(value) for value in run.phantom_shape) or "—"
        result = run.result
        signal_shape = " × ".join(
            str(value)
            for value in getattr(getattr(result, "signal", None), "shape", ())
        )
        rows = [
            ("Run", run.display_name),
            (
                "Type",
                "Spin probe" if run.run_type == "spin_probe" else "Sequence simulation",
            ),
            ("Sequence", run.sequence_name),
            ("Sequence duration", _duration_text(run.sequence_duration_s)),
            (
                "Probe context" if run.run_type == "spin_probe" else "Phantom",
                run.phantom_name,
            ),
            (
                "Probe grid" if run.run_type == "spin_probe" else "Phantom matrix",
                shape,
            ),
            ("ADC samples", str(run.adc_samples)),
            ("Signal shape", signal_shape or "—"),
            ("Simulation runtime", _duration_text(run.runtime_s)),
            ("Kernel", run.kernel or "—"),
            ("Completed", _date_text(run.created_at_utc)),
        ]
        self.details_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.details_table.setItem(row, 0, QTableWidgetItem(name))
            self.details_table.setItem(row, 1, QTableWidgetItem(value))
        self.details_table.resizeRowsToContents()

    def _open_selected(self):
        run = self.selected_run()
        if run is not None:
            self.run_open_requested.emit(run)

    def _export_selected(self):
        runs = self.selected_runs()
        if runs:
            self.runs_export_requested.emit(runs)

    def _rename_selected(self):
        runs = self.selected_runs()
        if len(runs) != 1:
            return
        run = runs[0]
        name, accepted = QInputDialog.getText(
            self,
            "Rename simulation",
            "Simulation name:",
            text=run.display_name,
        )
        name = str(name).strip()
        if not accepted or not name:
            return
        run.display_name = name
        run.custom_name = True
        for row in range(self.run_tree.topLevelItemCount()):
            item = self.run_tree.topLevelItem(row)
            if item.data(0, Qt.UserRole) == run.run_id:
                item.setText(0, name)
                break
        self._show_details(run)
        self.run_renamed.emit(run)

    def _delete_selected(self):
        runs = self.selected_runs()
        if not runs:
            return
        description = (
            runs[0].display_name
            if len(runs) == 1
            else f"{len(runs)} selected simulation runs"
        )
        answer = QMessageBox.question(
            self,
            "Delete simulation run" if len(runs) == 1 else "Delete simulation runs",
            f"Delete {description} from this session?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        run_ids = {run.run_id for run in runs}
        for row in reversed(range(self.run_tree.topLevelItemCount())):
            item = self.run_tree.topLevelItem(row)
            if item.data(0, Qt.UserRole) in run_ids:
                self.run_tree.takeTopLevelItem(row)
        for run in runs:
            self._runs.pop(run.run_id, None)
            self.run_deleted.emit(run)
        if self.run_tree.topLevelItemCount():
            self.run_tree.setCurrentItem(self.run_tree.topLevelItem(0))
        self._selection_changed()
