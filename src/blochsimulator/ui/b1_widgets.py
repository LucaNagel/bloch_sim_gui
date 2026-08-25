"""Editors and 3D alignment views for spatial transmit/receive B1 fields."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
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

try:
    import pyqtgraph.opengl as gl

    HAS_OPENGL = True
except Exception:
    gl = None
    HAS_OPENGL = False


def _complex_label(kind: str) -> str:
    return "Transmit B1+" if kind == "transmit" else "Receive B1−"


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
        source_layout.addWidget(QLabel("Reference |B1|"), 2, 0)
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
        splitter.addWidget(self.preview_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 1180])
        layout.addWidget(splitter, 1)

        self.tx_editor.field_changed.connect(self._field_changed)
        self.rx_editor.field_changed.connect(self._field_changed)

    def set_phantom(self, phantom) -> None:
        previous_phantom = self.phantom
        self.phantom = phantom
        self.tx_preview.set_phantom(phantom)
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

    def apply_fields(self) -> None:
        phantom = self.phantom
        if phantom is None:
            self.apply_status.setText(
                "Create or load a phantom before applying B1 fields."
            )
            return
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
            return

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


def _line_segments(corners: np.ndarray) -> np.ndarray:
    if len(corners) == 4:
        edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    else:
        edges = (
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
    return np.vstack([corners[list(edge)] for edge in edges])


class B1PhantomCombinationWidget(QWidget):
    """Show an active phantom inside a transformed B1 field in 3D."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.phantom = None
        self.tx_field: Optional[B1Field] = None
        self.rx_field: Optional[B1Field] = None
        self.sequence_context = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.show_phantom = QCheckBox("Show phantom")
        self.show_phantom.setChecked(True)
        self.show_phantom.toggled.connect(self._refresh)
        controls.addWidget(self.show_phantom)
        self.show_field = QCheckBox("Show B1 samples")
        self.show_field.setChecked(True)
        self.show_field.toggled.connect(self._refresh)
        controls.addWidget(self.show_field)
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
        self.channel_combo.currentIndexChanged.connect(self._refresh)
        self.gl_views = {}
        self.field_scatters = {}
        self.phantom_scatters = {}
        self.field_bounds_items = {}
        self.phantom_bounds_items = {}
        self.grids = {}
        self.view_infos = {}

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        self.view_splitter = splitter
        open_gl_enabled = (
            HAS_OPENGL and os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen"
        )
        for kind, title in (("transmit", "Transmit B1+"), ("receive", "Receive B1−")):
            group = QGroupBox(title)
            group_layout = QVBoxLayout(group)
            if kind == "receive":
                channel_row = QHBoxLayout()
                channel_row.addWidget(QLabel("Receive channel"))
                channel_row.addWidget(self.channel_combo)
                channel_row.addStretch()
                group_layout.addLayout(channel_row)
            if open_gl_enabled:
                gl_view = gl.GLViewWidget()
                gl_view.setCameraPosition(distance=400, elevation=25, azimuth=35)
                grid = gl.GLGridItem()
                gl_view.addItem(grid)
                field_scatter = gl.GLScatterPlotItem(
                    pos=np.zeros((0, 3)), color=(0.1, 0.7, 1.0, 0.45), size=5
                )
                gl_view.addItem(field_scatter)
                phantom_scatter = gl.GLScatterPlotItem(
                    pos=np.zeros((0, 3)), color=(1.0, 0.72, 0.1, 0.75), size=6
                )
                gl_view.addItem(phantom_scatter)
                field_bounds = gl.GLLinePlotItem(
                    pos=np.zeros((0, 3)),
                    color=(0.2, 0.8, 1.0, 0.95),
                    width=2,
                    mode="lines",
                    antialias=True,
                )
                gl_view.addItem(field_bounds)
                phantom_bounds = gl.GLLinePlotItem(
                    pos=np.zeros((0, 3)),
                    color=(1.0, 0.65, 0.05, 0.95),
                    width=2,
                    mode="lines",
                    antialias=True,
                )
                gl_view.addItem(phantom_bounds)
                group_layout.addWidget(gl_view, 1)
            else:
                gl_view = grid = field_scatter = phantom_scatter = None
                field_bounds = phantom_bounds = None
                unavailable = QLabel(
                    "The interactive 3D combination view requires OpenGL. Field "
                    "alignment is still calculated and applied to simulations."
                )
                unavailable.setWordWrap(True)
                unavailable.setAlignment(Qt.AlignCenter)
                unavailable.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                group_layout.addWidget(unavailable, 1)
            info = QLabel("Create a phantom and define this B1 field.")
            info.setWordWrap(True)
            group_layout.addWidget(info)
            self.gl_views[kind] = gl_view
            self.grids[kind] = grid
            self.field_scatters[kind] = field_scatter
            self.phantom_scatters[kind] = phantom_scatter
            self.field_bounds_items[kind] = field_bounds
            self.phantom_bounds_items[kind] = phantom_bounds
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
        self._refresh()

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
        self._refresh()

    def _refresh(self, *_):
        phantom = self.phantom
        selections = (
            ("transmit", self.tx_field, 0),
            ("receive", self.rx_field, max(0, self.channel_combo.currentIndex())),
        )
        for kind, field, channel in selections:
            if self.gl_views[kind] is not None:
                self._refresh_gl(kind, field, phantom, channel)
            info = self.view_infos[kind]
            if field is None and phantom is None:
                info.setText("Create a phantom and define this B1 field.")
            elif field is None:
                fallback_text = ""
                if (
                    kind == "transmit"
                    and self.transmit_display_combo.currentData() == "max_b1_gauss"
                ):
                    try:
                        peak = float(self.sequence_context.get("nominal_peak_b1_gauss"))
                    except (TypeError, ValueError):
                        peak = np.nan
                    if np.isfinite(peak) and peak >= 0.0:
                        fallback_text = f" Max B1 is uniformly {peak:.5g} G."
                info.setText(
                    f"{phantom.name} is shown with the uniform fallback field."
                    f"{fallback_text}"
                )
            elif phantom is None:
                info.setText(
                    f"{field.name} is ready. Create or load a phantom in the "
                    "Phantom tab."
                )
            else:
                scale = " × ".join(f"{value:g}" for value in field.scale_xyz)
                rotation = " / ".join(f"{value:g}°" for value in field.rotation_deg_xyz)
                channel_text = (
                    f" Channel {channel + 1}." if field.n_channels > 1 else ""
                )
                value_text = self._field_value_text(kind, field, phantom, channel)
                info.setText(
                    f"Orange: {phantom.name}. Blue: {field.name}.{channel_text} "
                    f"Stretch X/Y/Z {scale}; rotation X/Y/Z {rotation}."
                    f"{value_text}"
                )

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

    def _refresh_gl(self, kind, field, phantom, channel) -> None:
        gl_view = self.gl_views[kind]
        grid = self.grids[kind]
        field_scatter = self.field_scatters[kind]
        phantom_scatter = self.phantom_scatters[kind]
        field_bounds = self.field_bounds_items[kind]
        phantom_bounds = self.phantom_bounds_items[kind]
        extents = [100.0]
        if field is None or not self.show_field.isChecked():
            field_scatter.setData(pos=np.zeros((0, 3)))
            field_bounds.setData(pos=np.zeros((0, 3)))
        else:
            positions, indices = field.transformed_voxel_positions(max_points=20000)
            positions_mm = positions * 1000.0
            channel = min(max(0, channel), field.n_channels - 1)
            values = np.abs(field.data[channel].ravel()[indices])
            if (
                kind == "transmit"
                and self.transmit_display_combo.currentData() == "max_b1_gauss"
            ):
                try:
                    peak = float(self.sequence_context.get("nominal_peak_b1_gauss"))
                except (TypeError, ValueError):
                    peak = np.nan
                if np.isfinite(peak) and peak >= 0.0:
                    values = peak * values
            low, high = float(values.min()), float(values.max())
            normalized = (
                np.ones_like(values)
                if np.isclose(low, high)
                else np.clip((values - low) / (high - low), 0.0, 1.0)
            )
            colors = pg.colormap.get("viridis").map(normalized, mode="float")
            colors[:, 3] = 0.18 + 0.52 * normalized
            size = max(4.5, 18.0 / np.cbrt(max(1, len(positions_mm))))
            field_scatter.setData(
                pos=np.asarray(positions_mm, dtype=np.float32),
                color=np.asarray(colors, dtype=np.float32),
                size=size,
            )
            corners_mm = field.transformed_corners() * 1000.0
            field_bounds.setData(
                pos=np.asarray(_line_segments(corners_mm), dtype=np.float32)
            )
            extents.append(float(np.max(np.ptp(corners_mm, axis=0))))

        if phantom is None or not self.show_phantom.isChecked():
            phantom_scatter.setData(pos=np.zeros((0, 3)))
            phantom_bounds.setData(pos=np.zeros((0, 3)))
        else:
            active = np.asarray(phantom.mask, dtype=bool).ravel()
            positions = np.asarray(phantom.positions, dtype=float)[active]
            stride = max(1, int(np.ceil(len(positions) / 20000)))
            positions_mm = positions[::stride] * 1000.0
            phantom_scatter.setData(
                pos=np.asarray(positions_mm, dtype=np.float32),
                color=(1.0, 0.72, 0.1, 0.72),
                size=max(5.0, 20.0 / np.cbrt(max(1, len(positions_mm)))),
            )
            ndim = len(phantom.shape)
            half = np.zeros(3, dtype=float)
            half[:ndim] = np.asarray(phantom.fov) * 500.0
            if ndim == 2:
                corners = np.asarray(
                    [
                        (-half[0], -half[1], 0),
                        (half[0], -half[1], 0),
                        (half[0], half[1], 0),
                        (-half[0], half[1], 0),
                    ]
                )
            else:
                corners = np.asarray(
                    [
                        (x, y, z)
                        for z in (-half[2], half[2])
                        for y in (-half[1], half[1])
                        for x in (-half[0], half[0])
                    ]
                )
            phantom_bounds.setData(
                pos=np.asarray(_line_segments(corners), dtype=np.float32)
            )
            extents.append(float(np.max(np.asarray(phantom.fov) * 1000.0)))

        extent = max(extents)
        grid.setSize(x=extent * 1.2, y=extent * 1.2, z=0)
        spacing = max(extent / 10.0, 1e-6)
        grid.setSpacing(x=spacing, y=spacing, z=1.0)
        gl_view.setCameraPosition(distance=extent * 1.8)
