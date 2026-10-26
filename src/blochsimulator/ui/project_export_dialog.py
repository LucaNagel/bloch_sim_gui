"""Project filename controls for batch exports and direct loop saves."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..project_naming import ProjectExportNamer, ProjectExportNaming


class ProjectExportNamingWidget(QWidget):
    """Configure project names and preview the same allocation used by export."""

    def __init__(self, parent=None, *, object_prefix="project_export", naming=None):
        super().__init__(parent)
        naming = naming or ProjectExportNaming()
        self._directory = Path(".")
        self._examples = [("bloch_project_DATE_TIME", "DATE_TIME")]
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.mode = QComboBox()
        self.mode.setObjectName(f"{object_prefix}_naming_mode")
        self.mode.addItem("Current names", "existing")
        self.mode.addItem("Prefix + date and time", "timestamp")
        self.mode.addItem("Prefix + number", "numbered")
        self.mode.setCurrentIndex(self.mode.findData(naming.mode))
        form.addRow("File naming", self.mode)
        self.prefix = QLineEdit(naming.prefix)
        self.prefix.setObjectName(f"{object_prefix}_filename_prefix")
        self.prefix.setToolTip(
            "Filename prefix, for example simulation or bloch_project"
        )
        form.addRow("Prefix", self.prefix)
        numbering = QWidget()
        numbering_layout = QHBoxLayout(numbering)
        numbering_layout.setContentsMargins(0, 0, 0, 0)
        self.start_number = QSpinBox()
        self.start_number.setObjectName(f"{object_prefix}_start_number")
        self.start_number.setRange(0, 2**31 - 1)
        self.start_number.setValue(naming.start_number)
        numbering_layout.addWidget(QLabel("Start"))
        numbering_layout.addWidget(self.start_number)
        self.digits = QSpinBox()
        self.digits.setObjectName(f"{object_prefix}_number_digits")
        self.digits.setRange(1, 9)
        self.digits.setValue(naming.digits)
        self.digits.setToolTip("Minimum number of digits; 2 produces 01, 02, …")
        numbering_layout.addWidget(QLabel("Digits"))
        numbering_layout.addWidget(self.digits)
        numbering_layout.addStretch(1)
        form.addRow("Numbering", numbering)
        self.preview = QLabel()
        self.preview.setTextFormat(Qt.PlainText)
        self.preview.setWordWrap(True)
        self.preview.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Preview", self.preview)
        self.mode.currentIndexChanged.connect(self._update_controls)
        self.prefix.textChanged.connect(self._update_preview)
        self.start_number.valueChanged.connect(self._update_preview)
        self.digits.valueChanged.connect(self._update_preview)
        self._update_controls()

    def options(self):
        naming = ProjectExportNaming(
            mode=self.mode.currentData(),
            prefix=self.prefix.text(),
            start_number=self.start_number.value(),
            digits=self.digits.value(),
        )
        naming.validated_prefix()
        return naming

    def set_preview_context(self, directory, examples):
        self._directory = Path(directory or ".").expanduser()
        self._examples = list(examples)[:3]
        self._update_preview()

    def _update_controls(self, *_):
        mode = self.mode.currentData()
        self.prefix.setEnabled(mode != "existing")
        self.start_number.setEnabled(mode == "numbered")
        self.digits.setEnabled(mode == "numbered")
        self._update_preview()

    def _update_preview(self, *_):
        try:
            namer = ProjectExportNamer(self._directory, self.options())
            names = [
                namer.next_path(stem, timestamp).name
                for stem, timestamp in self._examples
            ]
            self.preview.setText("\n".join(names))
        except (ValueError, OSError) as exc:
            self.preview.setText(str(exc))


class SimulationProjectExportDialog(QDialog):
    """Choose a batch destination and filename policy in one dialog."""

    def __init__(
        self, directory, examples, parent=None, *, naming=None, write_notebook=False
    ):
        super().__init__(parent)
        self.setWindowTitle("Export simulations as projects")
        self.resize(620, 360)
        self._examples = list(examples)
        layout = QVBoxLayout(self)
        note = QLabel(
            f"Export {len(self._examples)} simulations. Numbering follows the "
            "Simulations list from top to bottom. Existing files are skipped."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Output folder"))
        self.directory = QLineEdit(str(directory))
        folder_layout.addWidget(self.directory, 1)
        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self._browse)
        folder_layout.addWidget(self.browse_button)
        layout.addLayout(folder_layout)
        self.naming = ProjectExportNamingWidget(self, naming=naming)
        layout.addWidget(self.naming)
        self.write_notebook = QCheckBox(
            "Write example notebook (load_simulations.ipynb)"
        )
        self.write_notebook.setToolTip(
            "Save a Jupyter notebook next to the projects that shows how to load "
            "them and access the sequence, phantom, results and loop parameters."
        )
        self.write_notebook.setChecked(bool(write_notebook))
        layout.addWidget(self.write_notebook)
        self.directory.textChanged.connect(self._update_preview)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update_preview()

    def _browse(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Export simulations as projects", self.directory.text()
        )
        if directory:
            self.directory.setText(directory)

    def _update_preview(self, *_):
        self.naming.set_preview_context(self.directory.text(), self._examples)

    def export_options(self):
        directory = self.directory.text().strip()
        if not directory:
            raise ValueError("Choose an output folder.")
        return Path(directory).expanduser(), self.naming.options()

    def accept(self):
        try:
            self.export_options()
        except ValueError as exc:
            QMessageBox.warning(self, "Export simulations", str(exc))
            return
        super().accept()
