"""Export dialog of the sequence workspace: one dialog for every format.

The dialog only collects a :class:`SequenceExportRequest`; writing is done
by :func:`blochsimulator.sequence.export_bundle.perform_sequence_export`.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..sequence.bruker_export import BrukerExportOptions
from ..sequence.bruker_jcamp import BrukerTemplateMetadata, load_bruker_template
from ..sequence.dicom_export import DICOM_PATIENT_POSITIONS
from ..sequence.export_bundle import FORMAT_LABELS, SequenceExportRequest
from ..sequence.paravision import (
    DEFAULT_PARAVISION_VERSION,
    PARAVISION_PROFILES,
    get_paravision_profile,
)

BRUKER_METHODS = (
    "Bruker:FLASH",
    "Bruker:MGE",
    "Bruker:EPI",
    "Bruker:RARE",
    "Bruker:CSI",
    "Bruker:SPIRAL",
    "Bruker:NSPECT",
    "User:lucaCSI4",
    "User:pulseq",
)
OPTIONAL_FORMAT_MODULES = {"dicom": "pydicom", "nifti": "nibabel"}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def template_summary(
    template: Optional[BrukerTemplateMetadata], *, anonymize=False
) -> str:
    """Short multi-line description of a template for the dialog preview."""

    if template is None:
        return (
            "No dataset loaded: subject, study and scanner fields use "
            "BlochSimulator defaults."
        )
    subject = template.subject_name or template.subject_id or "?"
    if template.subject_id and template.subject_id != subject:
        subject = f"{subject} ({template.subject_id})"
    if anonymize:
        subject = "anonymized (new subject and study UIDs)"
    lines = [
        f"Source: {template.source_path or 'edited'}",
        f"ParaVision {template.paravision_version or '?'} · user {template.owner or '?'}"
        f" · station {template.station or '?'}",
        f"Institution: {template.institution or '-'}",
        f"Subject: {subject}",
        f"Study: {template.study_name or '-'} #{template.study_number}",
        f"Coils: {template.coil_receive_name or '-'} / {template.coil_transmit_name or '-'}",
    ]
    if template.reference_frequencies_mhz:
        frequencies = ", ".join(
            f"{name} {value:.6f} MHz"
            for name, value in template.reference_frequencies_mhz
        )
        lines.append(f"Reference frequencies: {frequencies}")
    return "\n".join(lines)


class TemplateMetadataDialog(QDialog):
    """Edit the subject, study and user fields of a loaded template."""

    TEXT_FIELDS = (
        ("owner", "User (owner)"),
        ("operator", "Operator"),
        ("institution", "Institution"),
        ("station", "Station"),
        ("subject_id", "Subject ID"),
        ("subject_name", "Subject name"),
        ("subject_type", "Subject type"),
        ("subject_birth_date", "Birth date"),
        ("study_name", "Study name"),
        ("referring_physician", "Referring physician"),
        ("coil_receive_name", "Receive coil"),
        ("coil_transmit_name", "Transmit coil"),
    )
    SEXES = ("unknown", "male", "female", "other")

    def __init__(self, template: BrukerTemplateMetadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit subject and study metadata")
        self._template = template
        form = QFormLayout(self)
        self.fields: dict[str, QLineEdit] = {}
        for name, label in self.TEXT_FIELDS:
            edit = QLineEdit(str(getattr(template, name) or ""))
            edit.setObjectName(f"template_{name}")
            self.fields[name] = edit
            form.addRow(label, edit)
        self.subject_sex = QComboBox()
        self.subject_sex.setObjectName("template_subject_sex")
        self.subject_sex.addItems(self.SEXES)
        if template.subject_sex and self.subject_sex.findText(template.subject_sex) < 0:
            self.subject_sex.addItem(template.subject_sex)
        self.subject_sex.setCurrentText(template.subject_sex or "unknown")
        form.addRow("Sex", self.subject_sex)
        self.subject_weight = QDoubleSpinBox()
        self.subject_weight.setObjectName("template_subject_weight")
        self.subject_weight.setRange(0.0, 1000.0)
        self.subject_weight.setDecimals(4)
        self.subject_weight.setSuffix(" kg")
        self.subject_weight.setSpecialValueText("unknown")
        self.subject_weight.setValue(float(template.subject_weight_kg or 0.0))
        form.addRow("Weight", self.subject_weight)
        self.study_number = QSpinBox()
        self.study_number.setObjectName("template_study_number")
        self.study_number.setRange(1, 99999)
        self.study_number.setValue(int(template.study_number or 1))
        self.study_number.setToolTip(
            "Study number of the template study; a new study gets the next number"
        )
        form.addRow("Study number", self.study_number)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def template(self) -> BrukerTemplateMetadata:
        values = {name: edit.text().strip() for name, edit in self.fields.items()}
        values["subject_sex"] = self.subject_sex.currentText().strip()
        weight = self.subject_weight.value()
        values["subject_weight_kg"] = weight if weight > 0 else None
        values["study_number"] = self.study_number.value()
        old_owner = self._template.owner
        new_owner = values["owner"]
        if new_owner and old_owner and new_owner != old_owner:
            user_host = self._template.user_host
            if user_host.startswith(f"{old_owner}@"):
                values["user_host"] = f"{new_owner}@{user_host.partition('@')[2]}"
            data_root = self._template.data_root
            if data_root.rstrip("/").endswith(f"/{old_owner}"):
                values["data_root"] = (
                    f"{data_root.rstrip('/')[: -len(old_owner)]}{new_owner}"
                )
        return self._template.with_updates(**values)


class SequenceExportDialog(QDialog):
    """Choose folder, formats and Bruker/DICOM/NIfTI metadata of an export."""

    def __init__(
        self,
        parent=None,
        *,
        directory,
        base_name: str = "sequence_result",
        formats=("netcdf", "notebook"),
        suggested_method: str = "Bruker:FLASH",
        scan_name: str = "",
        matrix=(1, 1),
        fov_m=(0.1, 0.1),
        slice_thickness_mm: float = 1.0,
        raw_data_files: Optional[str] = None,
        template: Optional[BrukerTemplateMetadata] = None,
        anonymize: bool = False,
        template_directory=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export sequence simulation result")
        self.setObjectName("sequence_export_dialog")
        self._template = template
        self._template_directory = Path(template_directory or directory)
        self._preferred_raw_files = raw_data_files

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(self._build_output_group(directory, base_name))
        layout.addWidget(self._build_format_group(formats))
        layout.addWidget(self._build_image_group())
        layout.addWidget(self._build_metadata_group(anonymize))
        layout.addWidget(
            self._build_bruker_group(
                suggested_method, scan_name, matrix, fov_m, slice_thickness_mm
            )
        )
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.addWidget(scroll)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Export")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        outer.addWidget(self.buttons)
        # Show every group without scrolling where the screen is tall enough.
        width = max(640, content.sizeHint().width() + 40)
        height = content.sizeHint().height() + self.buttons.sizeHint().height() + 40
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            height = min(height, int(0.9 * screen.availableGeometry().height()))
        self.resize(width, max(height, 480))

        if template is not None and template.paravision_version in PARAVISION_PROFILES:
            self._select_version(template.paravision_version)
        self._update_raw_data_choices()
        self._update_template_preview()
        self._update_enabled_state()

    # -- construction -----------------------------------------------------

    def _build_output_group(self, directory, base_name) -> QGroupBox:
        group = QGroupBox("Output")
        form = QFormLayout(group)
        folder = QWidget()
        row = QHBoxLayout(folder)
        row.setContentsMargins(0, 0, 0, 0)
        self.directory_edit = QLineEdit(str(directory))
        self.directory_edit.setObjectName("sequence_export_directory")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_directory)
        row.addWidget(self.directory_edit, 1)
        row.addWidget(browse)
        form.addRow("Folder", folder)
        self.base_name_edit = QLineEdit(str(base_name))
        self.base_name_edit.setObjectName("sequence_export_base_name")
        self.base_name_edit.setToolTip(
            "File name without extension: <name>.nc, <name>.ipynb, <name>_dicom/, "
            "<name>_nifti/ ..."
        )
        form.addRow("Base name", self.base_name_edit)
        return group

    def _build_format_group(self, formats) -> QGroupBox:
        group = QGroupBox("Formats")
        grid = QGridLayout(group)
        self.format_checks: dict[str, QCheckBox] = {}
        for index, (name, label) in enumerate(FORMAT_LABELS.items()):
            check = QCheckBox(label)
            check.setObjectName(f"sequence_export_format_{name}")
            check.setChecked(name in formats)
            module = OPTIONAL_FORMAT_MODULES.get(name)
            if module is not None and not module_available(module):
                check.setChecked(False)
                check.setEnabled(False)
                check.setToolTip(
                    f"Needs {module}: pip install 'blochsimulator[export]'"
                )
            check.toggled.connect(self._update_enabled_state)
            self.format_checks[name] = check
            grid.addWidget(check, index // 2, index % 2)
        return group

    def _build_image_group(self) -> QGroupBox:
        group = QGroupBox("Images (Bruker, DICOM, NIfTI)")
        row = QHBoxLayout(group)
        self.include_pools = QCheckBox("Separate image per pool / metabolite")
        self.include_pools.setObjectName("sequence_export_include_pools")
        self.include_pools.setChecked(True)
        self.include_phase = QCheckBox("Phase images (DICOM, NIfTI)")
        self.include_phase.setObjectName("sequence_export_include_phase")
        self.include_phase.setChecked(True)
        row.addWidget(self.include_pools)
        row.addWidget(self.include_phase)
        row.addStretch(1)
        self.image_group = group
        return group

    def _build_metadata_group(self, anonymize) -> QGroupBox:
        group = QGroupBox("Subject, study and scanner (Bruker, DICOM, NIfTI)")
        layout = QVBoxLayout(group)
        self.template_preview = QLabel()
        self.template_preview.setObjectName("sequence_export_template_preview")
        self.template_preview.setWordWrap(True)
        self.template_preview.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.template_preview)
        buttons = QHBoxLayout()
        self.load_template_button = QPushButton("Load from ParaVision dataset...")
        self.load_template_button.setObjectName("sequence_export_load_template")
        self.load_template_button.setToolTip(
            "Copy user, scanner, subject and study fields from a measured study or scan"
        )
        self.load_template_button.clicked.connect(self._load_template)
        self.edit_template_button = QPushButton("Edit...")
        self.edit_template_button.setObjectName("sequence_export_edit_template")
        self.edit_template_button.clicked.connect(self._edit_template)
        self.clear_template_button = QPushButton("Clear")
        self.clear_template_button.clicked.connect(self._clear_template)
        buttons.addWidget(self.load_template_button)
        buttons.addWidget(self.edit_template_button)
        buttons.addWidget(self.clear_template_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        form = QFormLayout()
        self.anonymize = QCheckBox("Anonymize subject (keep scanner and user)")
        self.anonymize.setObjectName("sequence_export_anonymize")
        self.anonymize.setChecked(bool(anonymize))
        self.anonymize.toggled.connect(self._update_template_preview)
        form.addRow(self.anonymize)
        self.patient_position = QComboBox()
        self.patient_position.setObjectName("sequence_export_patient_position")
        self.patient_position.addItem("From dataset (default Head_Supine)", None)
        for name, code in DICOM_PATIENT_POSITIONS.items():
            self.patient_position.addItem(f"{name} ({code})", name)
        form.addRow("Patient position", self.patient_position)
        layout.addLayout(form)
        return group

    def _build_bruker_group(
        self, suggested_method, scan_name, matrix, fov_m, slice_thickness_mm
    ) -> QGroupBox:
        group = QGroupBox("Bruker ParaVision")
        form = QFormLayout(group)
        self.paravision_version = QComboBox()
        self.paravision_version.setObjectName("sequence_export_paravision_version")
        for key, profile in PARAVISION_PROFILES.items():
            self.paravision_version.addItem(profile.label, key)
        self._select_version(DEFAULT_PARAVISION_VERSION)
        self.paravision_version.currentIndexChanged.connect(
            self._update_raw_data_choices
        )
        form.addRow("Version", self.paravision_version)

        layout_widget = QWidget()
        layout_grid = QGridLayout(layout_widget)
        layout_grid.setContentsMargins(0, 0, 0, 0)
        self.new_study = QRadioButton("New study in the output folder")
        self.new_study.setObjectName("sequence_export_new_study")
        self.new_study.setChecked(True)
        self.append_study = QRadioButton("Append scan to study")
        self.append_study.setObjectName("sequence_export_append_study")
        self._layout_buttons = QButtonGroup(self)
        self._layout_buttons.addButton(self.new_study)
        self._layout_buttons.addButton(self.append_study)
        self.append_study.toggled.connect(self._update_enabled_state)
        self.study_dir_edit = QLineEdit()
        self.study_dir_edit.setObjectName("sequence_export_study_directory")
        self.study_dir_edit.setPlaceholderText("existing ParaVision study folder")
        self.study_dir_browse = QPushButton("Browse...")
        self.study_dir_browse.clicked.connect(self._browse_study)
        layout_grid.addWidget(self.new_study, 0, 0, 1, 3)
        layout_grid.addWidget(self.append_study, 1, 0)
        layout_grid.addWidget(self.study_dir_edit, 1, 1)
        layout_grid.addWidget(self.study_dir_browse, 1, 2)
        form.addRow("Study", layout_widget)

        self.method_name = QComboBox()
        self.method_name.setObjectName("sequence_export_method")
        self.method_name.setEditable(True)
        methods = list(BRUKER_METHODS)
        if suggested_method and suggested_method not in methods:
            methods.insert(0, suggested_method)
        self.method_name.addItems(methods)
        self.method_name.setCurrentText(suggested_method or BRUKER_METHODS[0])
        self.method_name.setToolTip(
            f"Suggested from the acquisition: {suggested_method}"
        )
        form.addRow("Method", self.method_name)

        self.scan_name = QLineEdit(scan_name)
        self.scan_name.setObjectName("sequence_export_scan_name")
        form.addRow("Scan name", self.scan_name)

        self.raw_data_files = QComboBox()
        self.raw_data_files.setObjectName("sequence_export_raw_data_files")
        form.addRow("Raw data", self.raw_data_files)

        self.conjugate = QCheckBox(
            "Store complex conjugate (like measured ParaVision data)"
        )
        self.conjugate.setObjectName("sequence_export_conjugate")
        self.conjugate.setChecked(True)
        self.conjugate.setToolTip(
            "Measured ParaVision raw data are the complex conjugate of the MR "
            "signal and are reconstructed with RECO_qopts = COMPLEX_CONJUGATE.\n"
            "Checked: the simulated signal is stored conjugated and marked that "
            "way, so ParaVision reconstructs it like a measured scan.\n"
            "Unchecked: the simulated signal is stored unchanged."
        )
        form.addRow(self.conjugate)

        self.geometry = QGroupBox("Override geometry")
        self.geometry.setObjectName("sequence_export_geometry")
        self.geometry.setCheckable(True)
        self.geometry.setChecked(False)
        self.geometry.setToolTip(
            "Unchecked: matrix, FOV and slice come from the simulation"
        )
        geometry = QFormLayout(self.geometry)
        self.matrix_read, self.matrix_phase = QSpinBox(), QSpinBox()
        for box, value in zip((self.matrix_read, self.matrix_phase), matrix):
            box.setRange(1, 8192)
            box.setValue(int(value))
        geometry.addRow("Matrix", _pair(self.matrix_read, self.matrix_phase))
        self.fov_read, self.fov_phase = QDoubleSpinBox(), QDoubleSpinBox()
        for box, value in zip((self.fov_read, self.fov_phase), fov_m):
            box.setRange(0.001, 10000.0)
            box.setDecimals(4)
            box.setSuffix(" mm")
            box.setValue(float(value) * 1000.0)
        geometry.addRow("FOV", _pair(self.fov_read, self.fov_phase))
        self.slice_thickness = QDoubleSpinBox()
        self.slice_thickness.setRange(0.001, 10000.0)
        self.slice_thickness.setDecimals(4)
        self.slice_thickness.setSuffix(" mm")
        self.slice_thickness.setValue(float(slice_thickness_mm))
        geometry.addRow("Slice thickness", self.slice_thickness)
        form.addRow(self.geometry)
        self.bruker_group = group
        return group

    # -- state ------------------------------------------------------------

    def _select_version(self, key: str):
        index = self.paravision_version.findData(key)
        if index >= 0:
            self.paravision_version.setCurrentIndex(index)

    def _update_raw_data_choices(self):
        profile = get_paravision_profile(self.paravision_version.currentData())
        current = self.raw_data_files.currentText() or self._preferred_raw_files
        self.raw_data_files.blockSignals(True)
        self.raw_data_files.clear()
        self.raw_data_files.addItems(list(profile.raw_data_files))
        if current in profile.raw_data_files:
            self.raw_data_files.setCurrentText(current)
        else:
            self.raw_data_files.setCurrentText(profile.default_raw_data_files)
        self.raw_data_files.blockSignals(False)
        self.raw_data_files.setEnabled(len(profile.raw_data_files) > 1)

    def _update_enabled_state(self, *_):
        selected = set(self.selected_formats())
        self.bruker_group.setEnabled("bruker" in selected)
        self.image_group.setEnabled(bool(selected & {"bruker", "dicom", "nifti"}))
        self.include_phase.setEnabled(bool(selected & {"dicom", "nifti"}))
        appending = self.append_study.isChecked()
        self.study_dir_edit.setEnabled(appending)
        self.study_dir_browse.setEnabled(appending)

    def _update_template_preview(self, *_):
        self.template_preview.setText(
            template_summary(self._template, anonymize=self.anonymize.isChecked())
        )
        has_template = self._template is not None
        self.edit_template_button.setEnabled(has_template)
        self.clear_template_button.setEnabled(has_template)

    def set_template(self, template: Optional[BrukerTemplateMetadata]):
        self._template = template
        if template is not None and template.paravision_version in PARAVISION_PROFILES:
            self._select_version(template.paravision_version)
        self._update_template_preview()

    # -- actions ----------------------------------------------------------

    def _browse_directory(self):
        path = QFileDialog.getExistingDirectory(
            self, "Export folder", self.directory_edit.text()
        )
        if path:
            self.directory_edit.setText(path)

    def _browse_study(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "ParaVision study to append to",
            self.study_dir_edit.text() or self.directory_edit.text(),
        )
        if path:
            self.study_dir_edit.setText(path)

    def _load_template(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "ParaVision study or scan folder",
            str(self._template_directory),
        )
        if not path:
            return
        try:
            template = load_bruker_template(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot read ParaVision dataset", str(error))
            return
        self._template_directory = Path(path).parent
        self.set_template(template)

    def _edit_template(self):
        if self._template is None:
            return
        dialog = TemplateMetadataDialog(self._template, self)
        if dialog.exec_() == QDialog.Accepted:
            self.set_template(dialog.template())

    def _clear_template(self):
        self.set_template(None)

    # -- result -----------------------------------------------------------

    def selected_formats(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, check in self.format_checks.items()
            if check.isChecked() and check.isEnabled()
        )

    def template(self) -> Optional[BrukerTemplateMetadata]:
        return self._template

    def validation_error(self) -> Optional[str]:
        formats = self.selected_formats()
        if not formats:
            return "Select at least one export format."
        if not self.directory_edit.text().strip():
            return "Choose an export folder."
        if not self.base_name_edit.text().strip():
            return "Enter a base name."
        if "bruker" in formats:
            if not self.method_name.currentText().strip():
                return "Enter a ParaVision method."
            if self.append_study.isChecked():
                study = Path(self.study_dir_edit.text().strip()).expanduser()
                if not self.study_dir_edit.text().strip() or not study.is_dir():
                    return "Choose an existing ParaVision study folder to append to."
                if not (study / "subject").is_file():
                    return (
                        f"{study} is not a ParaVision study folder (no subject file)."
                    )
        return None

    def accept(self):
        error = self.validation_error()
        if error is not None:
            QMessageBox.warning(self, "Export settings", error)
            return
        super().accept()

    def request(self) -> SequenceExportRequest:
        appending = self.append_study.isChecked()
        bruker = BrukerExportOptions(
            method_name=self.method_name.currentText().strip() or None,
            scan_name=self.scan_name.text().strip() or None,
            raw_data_files=self.raw_data_files.currentText() or None,
            paravision_version=self.paravision_version.currentData(),
            study_layout="append" if appending else "new",
            conjugate_raw_data=self.conjugate.isChecked(),
        )
        if self.geometry.isChecked():
            bruker = replace(
                bruker,
                matrix=(self.matrix_read.value(), self.matrix_phase.value()),
                fov_m=(self.fov_read.value() / 1000.0, self.fov_phase.value() / 1000.0),
                slice_thickness_mm=self.slice_thickness.value(),
            )
        study_text = self.study_dir_edit.text().strip()
        return SequenceExportRequest(
            directory=Path(self.directory_edit.text().strip()).expanduser(),
            base_name=self.base_name_edit.text().strip(),
            formats=self.selected_formats(),
            bruker=bruker,
            bruker_study_dir=(
                Path(study_text).expanduser() if appending and study_text else None
            ),
            template=self._template,
            anonymize=self.anonymize.isChecked(),
            patient_position=self.patient_position.currentData(),
            include_pools=self.include_pools.isChecked(),
            include_phase=self.include_phase.isChecked(),
        )


def _pair(first: QWidget, second: QWidget) -> QWidget:
    widget = QWidget()
    row = QHBoxLayout(widget)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(first)
    row.addWidget(QLabel("read x phase"))
    row.addWidget(second)
    return widget
