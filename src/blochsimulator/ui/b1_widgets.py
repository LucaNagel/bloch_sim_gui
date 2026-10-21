"""Editors and slice views for spatial transmit/receive B1 fields."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QSignalBlocker, QTimer, pyqtSignal
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..b1_fields import (
    B1Field,
    b1_preset_options,
    create_b1_preset,
    load_b1_field,
)
from ..paths import workspace_directory
from .volume_viewer import VolumeViewerWidget
from .styles import BOLD_GROUP_TITLES_STYLE


def _complex_label(kind: str) -> str:
    return "Transmit B1+" if kind == "transmit" else "Receive B1−"


def _magnitude_label(kind: str) -> str:
    return "B1+" if kind == "transmit" else "B1−"


class B1FieldEditor(QGroupBox):
    """Load/create one B1 field and edit its object-space geometry."""

    field_changed = pyqtSignal(object)

    def __init__(self, kind: str, parent=None):
        self.kind = "receive" if str(kind).lower() == "receive" else "transmit"
        # The enclosing tab already identifies the field type.
        super().__init__("", parent)
        self.field: Optional[B1Field] = None
        self.reference_phantom = None
        self._updating_controls = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Load a complex NumPy, HDF5, MATLAB, or NetCDF map, or generate a "
            "2D/3D default field on the current phantom geometry."
        )
        intro.setWordWrap(True)
        intro.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout.addWidget(intro)

        source_group = QGroupBox("Field source")
        source_layout = QGridLayout(source_group)
        source_layout.setColumnStretch(1, 1)
        source_layout.setColumnStretch(3, 1)
        self.load_button = QPushButton("Load field…")
        self.load_button.clicked.connect(self._load_field)
        source_layout.addWidget(self.load_button, 0, 0, 1, 4)
        self.dimension_combo = QComboBox()
        self.dimension_combo.addItems(["2D", "3D"])
        self.dimension_combo.currentTextChanged.connect(self._preset_changed)
        source_layout.addWidget(QLabel("Dimension"), 1, 0)
        source_layout.addWidget(self.dimension_combo, 1, 1)
        self.preset_combo = QComboBox()
        for identifier, label in b1_preset_options(self.kind):
            self.preset_combo.addItem(label, identifier)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        source_layout.addWidget(QLabel("Preset"), 1, 2)
        source_layout.addWidget(self.preset_combo, 1, 3)
        self.uniform_magnitude = QDoubleSpinBox()
        self.uniform_magnitude.setRange(0.0, 1e6)
        self.uniform_magnitude.setDecimals(4)
        self.uniform_magnitude.setValue(1.0)
        self.uniform_magnitude.setMinimumWidth(105)
        magnitude_label = (
            "Maximum |B1+|" if self.kind == "transmit" else "Maximum RSS |B1−|"
        )
        source_layout.addWidget(QLabel(magnitude_label), 2, 0)
        source_layout.addWidget(self.uniform_magnitude, 2, 1)
        self.uniform_phase = QDoubleSpinBox()
        self.uniform_phase.setRange(-360.0, 360.0)
        self.uniform_phase.setDecimals(2)
        self.uniform_phase.setSuffix("°")
        self.uniform_phase.setMinimumWidth(105)
        source_layout.addWidget(QLabel("Global phase"), 2, 2)
        source_layout.addWidget(self.uniform_phase, 2, 3)
        self.ramp_mode_label = QLabel("Ramp type")
        self.ramp_mode_combo = QComboBox()
        self.ramp_mode_combo.addItem("Magnitude", "magnitude")
        self.ramp_mode_combo.addItem("Phase", "phase")
        self.ramp_axis_label = QLabel("Ramp axis")
        self.ramp_axis_combo = QComboBox()
        for axis in "XYZ":
            self.ramp_axis_combo.addItem(axis, axis.lower())
        source_layout.addWidget(self.ramp_mode_label, 3, 0)
        source_layout.addWidget(self.ramp_mode_combo, 3, 1)
        source_layout.addWidget(self.ramp_axis_label, 3, 2)
        source_layout.addWidget(self.ramp_axis_combo, 3, 3)
        self.create_button = QPushButton("Create preset")
        self.create_button.clicked.connect(self._create_preset)
        source_layout.addWidget(self.create_button, 4, 0, 1, 4)
        source_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout.addWidget(source_group)

        self.source_info = QLabel(
            "Unity fallback is active (no explicit field loaded)."
        )
        self.source_info.setWordWrap(True)
        self.source_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout.addWidget(self.source_info)

        geometry = QGroupBox("Spatial geometry")
        geometry_layout = QGridLayout(geometry)
        geometry_layout.addWidget(QLabel("Axis"), 0, 0)
        geometry_layout.addWidget(QLabel("Native extent"), 0, 1)
        geometry_layout.addWidget(QLabel("Stretch"), 0, 2)
        geometry_layout.addWidget(QLabel("Rotation"), 0, 3)
        geometry_layout.setColumnStretch(1, 3)
        geometry_layout.setColumnStretch(2, 2)
        geometry_layout.setColumnStretch(3, 2)
        self.fov_spins = []
        self.scale_spins = []
        self.rotation_spins = []
        for row, axis in enumerate("XYZ", start=1):
            geometry_layout.addWidget(QLabel(axis), row, 0)
            fov = QDoubleSpinBox()
            fov.setRange(0.001, 1e6)
            fov.setDecimals(4)
            fov.setValue(240.0)
            fov.setSuffix(" mm")
            fov.setMinimumWidth(125)
            fov.valueChanged.connect(self._geometry_changed)
            geometry_layout.addWidget(fov, row, 1)
            self.fov_spins.append(fov)

            scale = QDoubleSpinBox()
            scale.setRange(0.01, 100.0)
            scale.setDecimals(4)
            scale.setValue(1.0)
            scale.setSuffix("×")
            scale.setMinimumWidth(90)
            scale.valueChanged.connect(self._geometry_changed)
            geometry_layout.addWidget(scale, row, 2)
            self.scale_spins.append(scale)

            rotation = QDoubleSpinBox()
            rotation.setRange(-360.0, 360.0)
            rotation.setDecimals(2)
            rotation.setSuffix("°")
            rotation.setMinimumWidth(90)
            rotation.valueChanged.connect(self._geometry_changed)
            geometry_layout.addWidget(rotation, row, 3)
            self.rotation_spins.append(rotation)
        geometry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout.addWidget(geometry)

        actions = QHBoxLayout()
        self.reset_button = QPushButton("Reset transform")
        self.reset_button.clicked.connect(self._reset_transform)
        actions.addWidget(self.reset_button)
        self.clear_button = QPushButton("Use unity fallback")
        self.clear_button.clicked.connect(self.clear_field)
        actions.addWidget(self.clear_button)
        actions.addStretch()
        layout.addLayout(actions)
        layout.addStretch()
        self._set_geometry_enabled(False)
        self._preset_changed()

    def set_reference_phantom(self, phantom) -> None:
        self.reference_phantom = phantom
        if phantom is not None:
            self.dimension_combo.setCurrentText(f"{min(3, max(2, phantom.ndim))}D")
        self._preset_changed()

    def _default_geometry(self, spatial_ndim: int):
        phantom = self.reference_phantom
        if phantom is not None:
            shape = tuple(int(value) for value in phantom.shape[:spatial_ndim])
            fov = tuple(float(value) for value in phantom.fov[:spatial_ndim])
            if len(shape) < spatial_ndim:
                shape += (32,) * (spatial_ndim - len(shape))
                in_plane = min(fov) if fov else 0.24
                fov += (in_plane,) * (spatial_ndim - len(fov))
            return shape, fov
        return (
            ((64, 64), (0.24, 0.24))
            if spatial_ndim == 2
            else (
                (64, 64, 64),
                (0.24, 0.24, 0.24),
            )
        )

    def _load_field(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            f"Load {_complex_label(self.kind)} field",
            str(workspace_directory("b1_fields")),
            "B1 fields (*.npy *.npz *.h5 *.hdf5 *.mat *.nc);;All files (*)",
        )
        if not filename:
            return
        phantom_fov = (
            None if self.reference_phantom is None else self.reference_phantom.fov
        )
        try:
            field = load_b1_field(
                filename,
                kind=self.kind,
                default_fov_m=phantom_fov,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Could not load B1 field", str(exc))
            return
        self.set_field(field)

    def _preset_changed(self, _value=None) -> None:
        is_ramp = self.preset_combo.currentData() == "linear_ramp"
        for widget in (
            self.ramp_mode_label,
            self.ramp_mode_combo,
            self.ramp_axis_label,
            self.ramp_axis_combo,
        ):
            widget.setEnabled(is_ramp)
        is_3d = self.dimension_combo.currentText() == "3D"
        z_item = self.ramp_axis_combo.model().item(2)
        if z_item is not None:
            z_item.setEnabled(is_3d)
        if not is_3d and self.ramp_axis_combo.currentData() == "z":
            self.ramp_axis_combo.setCurrentIndex(0)

    def _create_preset(self) -> None:
        ndim = 3 if self.dimension_combo.currentText() == "3D" else 2
        shape, fov = self._default_geometry(ndim)
        preset = str(self.preset_combo.currentData())
        if ndim == 3:
            shape = (64, 64, 64)
        elif preset != "uniform":
            maximum_size = 128
            shape = tuple(min(count, maximum_size) for count in shape)
        try:
            field = create_b1_preset(
                preset,
                shape,
                fov,
                kind=self.kind,
                magnitude=self.uniform_magnitude.value(),
                phase_deg=self.uniform_phase.value(),
                ramp_axis=str(self.ramp_axis_combo.currentData()),
                ramp_mode=str(self.ramp_mode_combo.currentData()),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Could not create B1 preset", str(exc))
            return
        self.set_field(field)

    def _create_uniform(self) -> None:
        """Compatibility helper for callers of the previous uniform action."""
        index = self.preset_combo.findData("uniform")
        self.preset_combo.setCurrentIndex(index)
        self._create_preset()

    def set_field(self, field: Optional[B1Field]) -> None:
        if field is not None and field.kind != self.kind:
            raise ValueError(f"expected a {self.kind} B1 field")
        self.field = field
        self._updating_controls = True
        try:
            if field is None:
                self.source_info.setText(
                    "Unity fallback is active (no explicit field loaded)."
                )
                self._set_geometry_enabled(False)
            else:
                self.dimension_combo.setCurrentText(f"{field.spatial_ndim}D")
                for axis in range(3):
                    self.fov_spins[axis].setValue(
                        field.fov_m[axis] * 1000.0 if axis < field.spatial_ndim else 1.0
                    )
                    self.scale_spins[axis].setValue(field.scale_xyz[axis])
                    self.rotation_spins[axis].setValue(field.rotation_deg_xyz[axis])
                source = (
                    Path(field.source_path).name if field.source_path else "generated"
                )
                channels = (
                    f", {field.n_channels} receive channel(s)"
                    if self.kind == "receive"
                    else ""
                )
                self.source_info.setText(
                    f"{field.name}: {field.spatial_ndim}D {field.spatial_shape}{channels}; "
                    f"source {source}."
                )
                self._set_geometry_enabled(True)
        finally:
            self._updating_controls = False
        self.field_changed.emit(field)

    def clear_field(self) -> None:
        self.set_field(None)

    def _set_geometry_enabled(self, enabled: bool) -> None:
        ndim = self.field.spatial_ndim if self.field is not None else 0
        for axis in range(3):
            self.fov_spins[axis].setEnabled(enabled and axis < ndim)
            self.scale_spins[axis].setEnabled(enabled and axis < ndim)
            # Rotating a 2D plane around any object-space axis is meaningful.
            self.rotation_spins[axis].setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)

    def _geometry_changed(self, _value=None) -> None:
        if self._updating_controls or self.field is None:
            return
        fov_m = tuple(
            self.fov_spins[axis].value() / 1000.0
            for axis in range(self.field.spatial_ndim)
        )
        scale = tuple(spin.value() for spin in self.scale_spins)
        rotation = tuple(spin.value() for spin in self.rotation_spins)
        self.field.set_fov_m(fov_m)
        self.field.set_transform(scale, rotation)
        self.field_changed.emit(self.field)

    def _reset_transform(self) -> None:
        if self.field is None:
            return
        self._updating_controls = True
        try:
            for spin in self.scale_spins:
                spin.setValue(1.0)
            for spin in self.rotation_spins:
                spin.setValue(0.0)
        finally:
            self._updating_controls = False
        self.field.set_transform((1, 1, 1), (0, 0, 0))
        self.field_changed.emit(self.field)


class B1FieldPreview(QWidget):
    """Preview native relative B1 or sequence-scaled transmit B1 on a phantom."""

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = "receive" if str(kind).lower() == "receive" else "transmit"
        self.field: Optional[B1Field] = None
        self.phantom = None
        self.sequence_context = {}
        self._physical_display_data = None
        self._physical_display_mask = None
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Display"))
        self.component_combo = QComboBox()
        for label, identifier in (
            ("Magnitude", "magnitude"),
            ("Phase", "phase"),
            ("Real", "real"),
            ("Imaginary", "imaginary"),
        ):
            self.component_combo.addItem(label, identifier)
        if self.kind == "transmit":
            self.component_combo.addItem(
                "Max B1 on phantom (selected sequence)", "max_b1_gauss"
            )
            self.component_combo.setToolTip(
                "Max B1 resamples the transformed transmit field onto the phantom "
                "and multiplies |B1+| by the loaded sequence's nominal RF peak."
            )
        self.component_combo.currentTextChanged.connect(self._update)
        controls.addWidget(self.component_combo)
        self.channel_label = QLabel("Receive channel")
        controls.addWidget(self.channel_label)
        self.channel_combo = QComboBox()
        self.channel_combo.currentIndexChanged.connect(self._update)
        controls.addWidget(self.channel_combo)
        receive_controls = self.kind == "receive"
        self.channel_label.setVisible(receive_controls)
        self.channel_combo.setVisible(receive_controls)
        controls.addStretch()
        layout.addLayout(controls)
        self.volume = VolumeViewerWidget()
        self.volume.indices_changed.connect(self._update_selected_physical_value)
        layout.addWidget(self.volume, 1)
        self.info = QLabel("No explicit field")
        self.info.setWordWrap(True)
        self.info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout.addWidget(self.info)
        self._update()

    def set_phantom(self, phantom) -> None:
        self.phantom = phantom
        self._update()

    def set_sequence_context(self, context) -> None:
        self.sequence_context = dict(context or {})
        self._update()

    def set_field(self, field: Optional[B1Field]) -> None:
        self.field = field
        previous = self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        if field is not None:
            for channel in range(field.n_channels):
                self.channel_combo.addItem(str(channel + 1), channel)
        self.channel_combo.blockSignals(previous)
        self.channel_combo.setEnabled(field is not None and field.n_channels > 1)
        self._update()

    def _update(self, *_):
        component = self.component_combo.currentData()
        if component == "max_b1_gauss":
            self._update_physical_transmit_b1()
            return
        self._physical_display_data = None
        self._physical_display_mask = None
        if self.field is None:
            self.volume.set_volume(
                np.ones((1, 1), dtype=float),
                fov_m=(0.24, 0.24),
                name="Unity fallback",
                unit="relative",
            )
            self.info.setText(
                "No explicit field; the simulator uses a uniform value of 1."
            )
            return
        channel = max(0, self.channel_combo.currentIndex())
        values = self.field.data[channel]
        component_label = self.component_combo.currentText()
        if component == "magnitude":
            data, unit = np.abs(values), "relative"
        elif component == "phase":
            data, unit = np.angle(values, deg=True), "°"
        elif component == "real":
            data, unit = values.real, "relative"
        else:
            data, unit = values.imag, "relative"
        effective_fov = tuple(
            self.field.fov_m[axis] * self.field.scale_xyz[axis]
            for axis in range(self.field.spatial_ndim)
        )
        self.volume.set_volume(
            data,
            fov_m=effective_fov,
            name=f"{self.field.name} · {component_label}",
            unit=unit,
        )
        rotation = " / ".join(f"{value:g}°" for value in self.field.rotation_deg_xyz)
        self.info.setText(
            "Native field values; stretching is reflected in the axes. The full "
            f"rotation X/Y/Z ({rotation}) is shown in the Combination tab."
        )

    def _update_physical_transmit_b1(self) -> None:
        peak = self.sequence_context.get("nominal_peak_b1_gauss")
        try:
            peak = float(peak)
        except (TypeError, ValueError):
            peak = None
        if peak is not None and (not np.isfinite(peak) or peak < 0.0):
            peak = None

        phantom = self.phantom
        if phantom is not None:
            tx_source = getattr(phantom, "tx_sensitivity_map", None)
            if tx_source is None:
                tx_source = np.ones(tuple(phantom.shape), dtype=float)
            relative = np.abs(np.asarray(tx_source))
            mask = np.asarray(phantom.mask, dtype=bool)
            fov_m = tuple(phantom.fov)
            grid_name = f"{phantom.name} phantom voxels"
        else:
            if self.field is None:
                relative = np.ones((1, 1), dtype=float)
                fov_m = (0.24, 0.24)
                grid_name = "unity fallback"
            else:
                relative = np.abs(self.field.data[0])
                fov_m = tuple(
                    self.field.fov_m[axis] * self.field.scale_xyz[axis]
                    for axis in range(self.field.spatial_ndim)
                )
                grid_name = f"{self.field.name} native grid"
            mask = np.ones(relative.shape, dtype=bool)

        data = np.asarray(relative, dtype=float)
        if peak is None:
            data.fill(0.0)
        else:
            data *= peak
        self._physical_display_data = data
        self._physical_display_mask = mask
        self.volume.set_volume(
            data,
            mask=mask,
            fov_m=fov_m,
            name=f"Maximum transmit B1 · {grid_name}",
            unit="G",
        )
        self._update_selected_physical_value()

    def _update_selected_physical_value(self, _indices=None) -> None:
        data = self._physical_display_data
        if data is None:
            return
        index = tuple(self.volume.indices[: data.ndim])
        mask = self._physical_display_mask
        peak = self.sequence_context.get("nominal_peak_b1_gauss")
        try:
            peak = float(peak)
        except (TypeError, ValueError):
            peak = None
        source = str(
            self.sequence_context.get("sequence_source") or "selected sequence"
        )
        nucleus = str(self.sequence_context.get("nucleus") or "")
        if peak is None or not np.isfinite(peak):
            self.info.setText(
                "No valid sequence RF waveform is selected, so physical B1 in "
                "gauss is not available yet."
            )
            return
        active = mask is None or bool(mask[index])
        selected = float(data[index])
        pending = bool(self.sequence_context.get("parameters_pending"))
        pending_text = (
            " Parameters have changed; this uses the currently loaded sequence "
            "until you generate the updated one."
            if pending
            else ""
        )
        active_text = "active" if active else "inactive"
        self.info.setText(
            f"Voxel {index} ({active_text}): max B1 {selected:.5g} G. "
            f"Nominal peak {peak:.5g} G from {source} ({nucleus}); the displayed "
            f"map is the applied phantom |B1+ scale| × nominal peak.{pending_text}"
        )


class B1WorkspaceWidget(QWidget):
    """Focused workspace for Tx/Rx fields and their application to a phantom."""

    fields_changed = pyqtSignal(object, object)
    phantom_updated = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.phantom = None
        self.sequence_context = {}
        self._fields_pending = False
        self._editor_tab_index = 0
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(120)
        self._apply_timer.timeout.connect(self.apply_fields)
        self._build_ui()

    @property
    def tx_field(self) -> Optional[B1Field]:
        return self.tx_editor.field

    @property
    def rx_field(self) -> Optional[B1Field]:
        return self.rx_editor.field

    def _build_ui(self) -> None:
        # Style only the frame titles; child controls keep their normal font.
        self.setStyleSheet(BOLD_GROUP_TITLES_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        header = QLabel(
            "Define independent complex transmit (B1+) and receive (B1−) fields. "
            "The transformed fields are interpolated onto the current Phantom grid."
        )
        header.setWordWrap(True)
        header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header.setMaximumHeight(52)
        self.header = header
        layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.splitter = splitter
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        self.editor_tabs = QTabWidget()
        self.tx_editor = B1FieldEditor("transmit")
        self.rx_editor = B1FieldEditor("receive")
        self.editor_tabs.addTab(self.tx_editor, "Transmit B1+")
        self.editor_tabs.addTab(self.rx_editor, "Receive B1−")
        controls_layout.addWidget(self.editor_tabs, 1)

        application_group = QGroupBox("Phantom coupling")
        application_layout = QVBoxLayout(application_group)
        self.auto_apply = QCheckBox(
            "Apply changes automatically to the current phantom"
        )
        self.auto_apply.setChecked(True)
        self.auto_apply.toggled.connect(self._auto_apply_changed)
        application_layout.addWidget(self.auto_apply)
        self.apply_button = QPushButton("Apply B1 fields now")
        self.apply_button.clicked.connect(self.apply_fields)
        application_layout.addWidget(self.apply_button)
        self.apply_status = QLabel("No phantom is selected yet.")
        self.apply_status.setWordWrap(True)
        application_layout.addWidget(self.apply_status)
        controls_layout.addWidget(application_group)
        controls_container.setMinimumWidth(450)
        controls_container.setMaximumWidth(620)
        splitter.addWidget(controls_container)

        self.preview_tabs = QTabWidget()
        self.tx_preview = B1FieldPreview("transmit")
        self.rx_preview = B1FieldPreview("receive")
        self.preview_tabs.addTab(self.tx_preview, "Transmit preview")
        self.preview_tabs.addTab(self.rx_preview, "Receive preview")
        # Editing a field opens its matching preview. Preview selection itself
        # stays independent so users can still compare either field manually.
        self.editor_tabs.currentChanged.connect(self._editor_tab_changed)
        splitter.addWidget(self.preview_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 1180])
        layout.addWidget(splitter, 1)

        self.tx_editor.field_changed.connect(self._field_changed)
        self.rx_editor.field_changed.connect(self._field_changed)

    def _show_matching_preview(self, index: int) -> None:
        if 0 <= index < self.preview_tabs.count():
            self.preview_tabs.setCurrentIndex(index)

    def _editor_tab_changed(self, index: int) -> None:
        if not self.confirm_pending_changes():
            with QSignalBlocker(self.editor_tabs):
                self.editor_tabs.setCurrentIndex(self._editor_tab_index)
            return
        self._editor_tab_index = index
        self._show_matching_preview(index)

    def confirm_pending_changes(self) -> bool:
        """Resolve unapplied edits before leaving an editor or the B1 tab."""
        if not self._fields_pending or self.phantom is None:
            return True
        if self.auto_apply.isChecked() and self._apply_timer.isActive():
            return self.apply_fields()
        answer = QMessageBox.question(
            self,
            "Apply B1 field changes?",
            "The transmit or receive B1 fields have unapplied changes.\n\n"
            "Apply them to the current phantom before switching tabs?\n"
            "No keeps the edits without applying them. Cancel stays on this tab.",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            if self.apply_fields():
                return True
            QMessageBox.warning(
                self, "Could not apply B1 fields", self.apply_status.text()
            )
            return False
        return answer == QMessageBox.No

    def set_phantom(self, phantom) -> None:
        previous_phantom = self.phantom
        self.phantom = phantom
        if phantom is not previous_phantom:
            self._fields_pending = (
                self.tx_field is not None or self.rx_field is not None
            )
        self.tx_preview.set_phantom(phantom)
        self.rx_preview.set_phantom(phantom)
        self.tx_editor.set_reference_phantom(phantom)
        self.rx_editor.set_reference_phantom(phantom)
        self.apply_button.setEnabled(phantom is not None)
        if phantom is None:
            self.apply_status.setText("No phantom is selected yet.")
        elif (
            previous_phantom is None and self.tx_field is None and self.rx_field is None
        ):
            # Preserve B1 maps embedded in a loaded Phantom file. Unity defaults
            # are intentionally shown as ordinary uniform fields so the user can
            # immediately edit their geometry.
            tx = getattr(phantom, "tx_sensitivity_map", None)
            rx = getattr(phantom, "rx_sensitivity_maps", None)
            if tx is not None:
                self.tx_editor.set_field(
                    B1Field(
                        data=np.asarray(tx),
                        fov_m=tuple(phantom.fov),
                        kind="transmit",
                        spatial_ndim=phantom.ndim,
                        name="Phantom transmit B1+",
                    )
                )
            if rx is not None:
                self.rx_editor.set_field(
                    B1Field(
                        data=np.asarray(rx),
                        fov_m=tuple(phantom.fov),
                        kind="receive",
                        spatial_ndim=phantom.ndim,
                        name="Phantom receive B1−",
                    )
                )
            # These fields were read from the phantom's already applied maps.
            self._fields_pending = False
            if tx is None and rx is None and self.auto_apply.isChecked():
                self._apply_timer.start()
        elif self.auto_apply.isChecked():
            self._apply_timer.start()
        else:
            self.apply_status.setText(
                f"Ready to apply fields to {phantom.name} ({phantom.shape})."
            )

    def set_sequence_context(self, context) -> None:
        self.sequence_context = dict(context or {})
        self.tx_preview.set_sequence_context(self.sequence_context)

    def _field_changed(self, _field=None) -> None:
        self._fields_pending = True
        self.tx_preview.set_field(self.tx_field)
        self.rx_preview.set_field(self.rx_field)
        self.fields_changed.emit(self.tx_field, self.rx_field)
        if self.auto_apply.isChecked() and self.phantom is not None:
            self._apply_timer.start()
        elif self.phantom is not None:
            self.apply_status.setText(
                "Field geometry changed; click Apply B1 fields now."
            )

    def _auto_apply_changed(self, enabled: bool) -> None:
        if enabled and self.phantom is not None:
            self._apply_timer.start()
        elif not enabled:
            self._apply_timer.stop()

    def apply_fields(self) -> bool:
        self._apply_timer.stop()
        phantom = self.phantom
        if phantom is None:
            self.apply_status.setText(
                "Create or load a phantom before applying B1 fields."
            )
            return False
        try:
            if self.tx_field is None:
                tx = np.ones(tuple(phantom.shape), dtype=np.complex128)
            else:
                tx = self.tx_field.resample_to_phantom(phantom)[0]
            if self.rx_field is None:
                rx = np.ones((1, *tuple(phantom.shape)), dtype=np.complex128)
            else:
                rx = self.rx_field.resample_to_phantom(phantom)
            phantom.tx_sensitivity_map = tx
            phantom.rx_sensitivity_maps = rx
        except Exception as exc:
            self.apply_status.setText(f"B1 application failed: {exc}")
            return False

        self._fields_pending = False
        self.tx_preview.set_phantom(phantom)

        active = np.asarray(phantom.mask, dtype=bool)
        covered = active & (np.abs(tx) > np.finfo(float).eps)
        covered &= np.any(np.abs(rx) > np.finfo(float).eps, axis=0)
        coverage = 100.0 * np.count_nonzero(covered) / max(1, np.count_nonzero(active))
        self.apply_status.setText(
            f"Applied to {phantom.name}: {coverage:.1f}% of active voxels are "
            f"inside both fields; {rx.shape[0]} receive channel(s)."
        )
        self.phantom_updated.emit(phantom)
        return True


# Samples along the longest axis when the view covers the whole B1 field.
_FIELD_GRID_POINTS = 96


def _phantom_support_on_grid(phantom, coordinates) -> np.ndarray:
    """Return nearest-voxel phantom support at the centres of a 3D grid.

    ``coordinates`` holds the X, Y and Z grid-centre vectors in metres. Grid
    axes beyond the phantom's dimensionality only meet the phantom in the
    plane through the origin.
    """
    shape = tuple(int(count) for count in phantom.shape)
    positions = np.asarray(phantom.positions, dtype=float).reshape(*shape, 3)
    support = np.asarray(phantom.mask, dtype=bool).reshape(shape)
    support = support.reshape(shape + (1,) * (3 - len(shape)))
    indices = []
    inside = []
    for axis, grid in enumerate(coordinates):
        grid = np.asarray(grid, dtype=float)
        if axis < len(shape):
            line = positions[
                tuple(
                    slice(None) if other == axis else 0 for other in range(len(shape))
                )
            ][:, axis]
            count = shape[axis]
            step = (
                (line[-1] - line[0]) / (count - 1)
                if count > 1
                else float(phantom.fov[axis])
            )
            index = np.rint((grid - line[0]) / step).astype(int)
            inside.append((index >= 0) & (index < count))
            indices.append(np.clip(index, 0, count - 1))
        else:
            step = grid[1] - grid[0] if grid.size > 1 else np.inf
            inside.append(np.abs(grid) <= 0.5 * step + 1e-12)
            indices.append(np.zeros(grid.size, dtype=int))
    return (
        support[np.ix_(*indices)]
        & inside[0][:, None, None]
        & inside[1][None, :, None]
        & inside[2][None, None, :]
    )


class B1PhantomCombinationWidget(QWidget):
    """Show B1 magnitude maps in orthogonal slices with the phantom outline."""

    PLANES = ("xy", "xz", "yz")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.phantom = None
        self.tx_field: Optional[B1Field] = None
        self.rx_field: Optional[B1Field] = None
        self.sequence_context = {}
        # Display grid shared by both views: (shape, fov_m, phantom support).
        self._grid = None
        self._relative = {}
        self._synchronizing_slices = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("View"))
        self.extent_combo = QComboBox()
        self.extent_combo.addItem("Phantom grid", "phantom")
        self.extent_combo.addItem("Whole B1 field", "field")
        self.extent_combo.setToolTip(
            "Phantom grid: B1 exactly as each phantom voxel receives it in the "
            "simulation.\nWhole B1 field: the full transformed field around the "
            "phantom, to check alignment."
        )
        self.extent_combo.currentIndexChanged.connect(self._rebuild)
        controls.addWidget(self.extent_combo)
        self.show_phantom = QCheckBox("Phantom outline")
        self.show_phantom.setChecked(True)
        self.show_phantom.toggled.connect(self._refresh_outlines)
        controls.addWidget(self.show_phantom)
        controls.addWidget(QLabel("Transmit values"))
        self.transmit_display_combo = QComboBox()
        self.transmit_display_combo.addItem("Relative |B1+|", "relative")
        self.transmit_display_combo.addItem(
            "Max B1 (selected sequence)", "max_b1_gauss"
        )
        self.transmit_display_combo.setToolTip(
            "Scale transmit-field samples by the loaded sequence's nominal RF "
            "peak and report the resulting B1 amplitude in gauss."
        )
        self.transmit_display_combo.currentIndexChanged.connect(self._refresh)
        controls.addWidget(self.transmit_display_combo)
        controls.addStretch()
        layout.addLayout(controls)

        self.channel_combo = QComboBox()
        self.channel_combo.currentIndexChanged.connect(self._rebuild)
        self.viewers = {}
        self.outline_items = {}
        self.view_infos = {}

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        self.view_splitter = splitter
        for kind, title in (("transmit", "Transmit B1+"), ("receive", "Receive B1−")):
            group = QGroupBox(title)
            group_layout = QVBoxLayout(group)
            if kind == "receive":
                channel_row = QHBoxLayout()
                channel_row.addWidget(QLabel("Receive channel"))
                channel_row.addWidget(self.channel_combo)
                channel_row.addStretch()
                group_layout.addLayout(channel_row)
            viewer = VolumeViewerWidget(show_3d=False)
            viewer.info.hide()
            viewer.indices_changed.connect(
                lambda indices, selected=kind: self._slices_changed(selected, indices)
            )
            outlines = {}
            for plane, view in zip(
                self.PLANES, (viewer.xy_view, viewer.xz_view, viewer.yz_view)
            ):
                outline = pg.IsocurveItem(
                    level=0.5, pen=pg.mkPen((255, 165, 20), width=2)
                )
                outline.setZValue(50)
                view.getView().addItem(outline, ignoreBounds=True)
                outlines[plane] = outline
            group_layout.addWidget(viewer, 1)
            info = QLabel("Create a phantom and define this B1 field.")
            info.setWordWrap(True)
            group_layout.addWidget(info)
            self.viewers[kind] = viewer
            self.outline_items[kind] = outlines
            self.view_infos[kind] = info
            if kind == "transmit":
                self.tx_panel = group
                self.tx_info = info
            else:
                self.rx_panel = group
                self.rx_info = info
            splitter.addWidget(group)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, 1)
        self._update_receive_channels()

    def set_phantom(self, phantom) -> None:
        self.phantom = phantom
        self._rebuild()

    def set_fields(
        self,
        tx_field: Optional[B1Field],
        rx_field: Optional[B1Field],
    ) -> None:
        self.tx_field = tx_field
        self.rx_field = rx_field
        self._update_receive_channels()

    def set_sequence_context(self, context) -> None:
        self.sequence_context = dict(context or {})
        self._refresh()

    def _update_receive_channels(self) -> None:
        previous = self.channel_combo.blockSignals(True)
        current = self.channel_combo.currentIndex()
        self.channel_combo.clear()
        if self.rx_field is not None:
            for channel in range(self.rx_field.n_channels):
                self.channel_combo.addItem(str(channel + 1), channel)
        self.channel_combo.setCurrentIndex(
            min(max(current, 0), self.channel_combo.count() - 1)
        )
        self.channel_combo.blockSignals(previous)
        self.channel_combo.setEnabled(
            self.rx_field is not None and self.rx_field.n_channels > 1
        )
        self._rebuild()

    def _display_grid(self):
        """Return grid shape, FOV, sample positions and phantom support."""
        phantom = self.phantom
        if phantom is not None and self.extent_combo.currentData() == "phantom":
            shape = tuple(int(count) for count in phantom.shape)
            support = np.asarray(phantom.mask, dtype=bool).reshape(shape)
            shape3 = shape + (1,) * (3 - len(shape))
            return (
                shape,
                tuple(float(value) for value in phantom.fov),
                np.asarray(phantom.positions, dtype=float),
                support.reshape(shape3),
            )

        half = np.zeros(3, dtype=float)
        for field in (self.tx_field, self.rx_field):
            if field is not None:
                half = np.maximum(half, np.abs(field.transformed_corners()).max(axis=0))
        if phantom is not None:
            shape = tuple(int(count) for count in phantom.shape)
            voxel = np.zeros(3, dtype=float)
            voxel[: len(shape)] = np.asarray(phantom.fov, dtype=float) / shape
            positions = np.asarray(phantom.positions, dtype=float)
            half = np.maximum(half, np.abs(positions).max(axis=0) + 0.5 * voxel)
        if not np.any(half > 0):
            return None
        step = 2.0 * float(half.max()) / _FIELD_GRID_POINTS
        counts = np.maximum(1, np.ceil(2.0 * half / step - 1e-9).astype(int))
        fov = counts * step
        coordinates = [
            ((np.arange(count) + 0.5) / count - 0.5) * extent
            for count, extent in zip(counts, fov)
        ]
        grid = np.meshgrid(*coordinates, indexing="ij")
        positions = np.column_stack([axis.ravel() for axis in grid])
        support = (
            None if phantom is None else _phantom_support_on_grid(phantom, coordinates)
        )
        return tuple(int(count) for count in counts), tuple(fov), positions, support

    def _rebuild(self, *_):
        """Resample the selected fields onto the shared display grid."""
        self._relative = {}
        self._grid = None
        grid = (
            self._display_grid()
            if any(
                item is not None
                for item in (self.phantom, self.tx_field, self.rx_field)
            )
            else None
        )
        if grid is not None:
            shape, fov, positions, support = grid
            self._grid = (shape, fov, support)
            selections = (
                ("transmit", self.tx_field, 0),
                ("receive", self.rx_field, max(0, self.channel_combo.currentIndex())),
            )
            for kind, field, channel in selections:
                if field is None:
                    # Simulations use a uniform unit field when none is defined.
                    self._relative[kind] = np.ones(shape, dtype=float)
                else:
                    channel = min(channel, field.n_channels - 1)
                    sampled = field.sample_world(positions, channels=[channel])[0]
                    self._relative[kind] = np.abs(sampled).reshape(shape)
        self._refresh()

    def _transmit_peak_gauss(self):
        if self.transmit_display_combo.currentData() != "max_b1_gauss":
            return None
        try:
            peak = float(self.sequence_context.get("nominal_peak_b1_gauss"))
        except (TypeError, ValueError):
            return None
        return peak if np.isfinite(peak) and peak >= 0.0 else None

    def _refresh(self, *_):
        phantom = self.phantom
        peak = self._transmit_peak_gauss()
        color_map = pg.colormap.get("viridis")
        for kind, field in (("transmit", self.tx_field), ("receive", self.rx_field)):
            viewer = self.viewers[kind]
            relative = self._relative.get(kind)
            viewer.setVisible(relative is not None)
            if relative is not None:
                values, name, unit = (
                    relative,
                    f"Relative |{_magnitude_label(kind)}|",
                    "",
                )
                if kind == "transmit" and peak is not None:
                    values, name, unit = peak * relative, "Max B1", "G"
                shape, fov, _support = self._grid
                levels = None
                if np.ptp(values) <= 1e-12 * max(1.0, float(np.max(values))):
                    # A uniform field would otherwise get an arbitrary scale.
                    level = float(np.max(values))
                    levels = (min(0.0, level - 1.0), max(level, 1e-12))
                promoted = viewer._promote_to_volume(np.asarray(values))
                if promoted.shape == viewer.data.shape and tuple(
                    fov + (viewer.fov_m[2],) * (3 - len(fov))
                ) == tuple(viewer.fov_m):
                    viewer.update_volume_data(
                        values, name=name, unit=unit, levels=levels, color_map=color_map
                    )
                else:
                    viewer.set_volume(
                        values,
                        fov_m=fov,
                        name=name,
                        unit=unit,
                        levels=levels,
                        color_map=color_map,
                    )
            self.view_infos[kind].setText(self._info_text(kind, field, phantom))

    def _info_text(self, kind, field, phantom) -> str:
        if field is None and phantom is None:
            return "Create a phantom and define this B1 field."
        if field is None:
            text = f"{phantom.name} is shown with the uniform fallback field."
            peak = self._transmit_peak_gauss()
            if kind == "transmit" and peak is not None:
                text += f" Max B1 is uniformly {peak:.5g} G."
            return text
        channel = max(0, self.channel_combo.currentIndex())
        scale = " × ".join(f"{value:g}" for value in field.scale_xyz)
        rotation = " / ".join(f"{value:g}°" for value in field.rotation_deg_xyz)
        channel_text = f" Channel {channel + 1}." if field.n_channels > 1 else ""
        geometry = (
            f"{field.name}.{channel_text} Stretch X/Y/Z {scale}; rotation X/Y/Z "
            f"{rotation}."
        )
        if phantom is None:
            return (
                f"{geometry} Create or load a phantom in the Phantom tab to see "
                "its outline."
            )
        value_text = self._field_value_text(kind, field, phantom, channel)
        return f"Orange outline: {phantom.name}. Colour: {geometry}{value_text}"

    def _field_value_text(self, kind, field, phantom, channel) -> str:
        if kind != "transmit":
            values = np.abs(field.data[channel])
            return f" Relative |B1−| {values.min():.5g}–{values.max():.5g}."
        if self.transmit_display_combo.currentData() != "max_b1_gauss":
            values = np.abs(field.data[0])
            return f" Relative |B1+| {values.min():.5g}–{values.max():.5g}."

        peak = self.sequence_context.get("nominal_peak_b1_gauss")
        try:
            peak = float(peak)
        except (TypeError, ValueError):
            peak = None
        if peak is None or not np.isfinite(peak):
            return " Select a valid sequence RF waveform to calculate max B1 in G."
        if phantom is None:
            relative = np.abs(field.data[0])
            location = "native field"
        else:
            tx_source = getattr(phantom, "tx_sensitivity_map", None)
            if tx_source is None:
                tx_source = np.ones(tuple(phantom.shape), dtype=float)
            relative = np.abs(np.asarray(tx_source))
            active = np.asarray(phantom.mask, dtype=bool)
            relative = relative[active]
            location = "active phantom voxels"
        if relative.size == 0:
            return f" No {location} are available for max-B1 display."
        values = peak * relative
        pending_text = (
            " Parameters are pending regeneration; values use the currently "
            "loaded sequence."
            if self.sequence_context.get("parameters_pending")
            else ""
        )
        return (
            f" Max B1 in {location}: {values.min():.5g}–{values.max():.5g} G "
            f"(nominal {peak:.5g} G).{pending_text}"
        )

    def _slices_changed(self, kind, indices) -> None:
        self._update_outlines(kind)
        if self._synchronizing_slices:
            return
        other = self.viewers["receive" if kind == "transmit" else "transmit"]
        if other.data.shape != self.viewers[kind].data.shape:
            return
        if tuple(other.indices) == tuple(indices):
            return
        # Keep transmit and receive planes at the same position.
        self._synchronizing_slices = True
        try:
            other.set_indices(indices)
        finally:
            self._synchronizing_slices = False

    def _refresh_outlines(self, *_):
        for kind in self.viewers:
            self._update_outlines(kind)

    def _update_outlines(self, kind) -> None:
        viewer = self.viewers[kind]
        support = None if self._grid is None else self._grid[2]
        visible = (
            support is not None
            and self.show_phantom.isChecked()
            and support.shape == viewer.data.shape
        )
        ix, iy, iz = viewer.indices
        for plane, outline in self.outline_items[kind].items():
            if not visible:
                outline.setData(None)
                outline.hide()
                continue
            axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
            values = {
                "xy": support[:, :, iz],
                "xz": support[:, iy, :],
                "yz": support[ix, :, :],
            }[plane]
            steps = [
                viewer.fov_m[axis] * 1000.0 / viewer.data.shape[axis] for axis in axes
            ]
            # Isocurves place sample i at i + 0.5 like ImageItem pixels; the
            # one-sample zero padding closes contours at the grid border.
            transform = QTransform()
            transform.translate(
                -viewer.fov_m[axes[0]] * 500.0 - steps[0],
                -viewer.fov_m[axes[1]] * 500.0 - steps[1],
            )
            transform.scale(*steps)
            outline.setTransform(transform)
            outline.setData(np.pad(values.astype(float), 1), level=0.5)
            outline.show()
