"""Workspace for the static B0 field of the current phantom."""

from __future__ import annotations

import weakref
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QSignalBlocker, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..b0_field import B0_PATTERN_MODES, B0FieldSettings, PhantomB0Field
from ..dynamic_phantom import TimeCurve
from .styles import BOLD_GROUP_TITLES_STYLE
from .volume_viewer import VolumeViewerWidget


def _number_spin(minimum, maximum, value, suffix, decimals=3):
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    spin.setSuffix(suffix)
    spin.setKeyboardTracking(False)
    return spin


class B0WorkspaceWidget(QWidget):
    """Edit the static B0 field and apply it to the current phantom.

    The field is an analytic spatial pattern plus a constant offset per
    region (designer shapes or anatomy labels). A uniform drift over time is
    not edited here; one stored with the phantom (scripting API or an older
    project) is passed through unchanged.
    """

    phantom_updated = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.phantom = None
        self.field: Optional[PhantomB0Field] = None
        self._updating = False
        self._settings_pending = False
        self._session_settings: Optional[B0FieldSettings] = None
        # Drift curve stored with the phantom; the GUI does not edit it.
        self._drift_curve_hz: Optional[TimeCurve] = None
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(120)
        self._apply_timer.timeout.connect(self.apply_field)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(60)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_shape = None
        self._build_ui()
        self._update_pattern_info()

    # -- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setStyleSheet(BOLD_GROUP_TITLES_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        header = QLabel(
            "The B0 offset of each voxel is the constant offset of the region it "
            "belongs to plus a spatial inhomogeneity pattern. It shifts every "
            "peak in that voxel and applies to any phantom, including the "
            "anatomical mouse."
        )
        header.setWordWrap(True)
        header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header.setMaximumHeight(52)
        self.header = header
        layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.splitter = splitter
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        pattern_group = QGroupBox("Spatial B0 inhomogeneity")
        pattern_layout = QVBoxLayout(pattern_group)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.mode_combo = QComboBox()
        for key, label in B0_PATTERN_MODES:
            self.mode_combo.addItem(label, key)
        self.mode_combo.setToolTip(
            "Analytic spatial B0 variation added to each region's constant offset"
        )
        form.addRow("Pattern", self.mode_combo)
        self.strength_spin = _number_spin(-1000.0, 1000.0, 0.0, " ppm")
        self.strength_spin.setToolTip(
            "Strength of the pattern; the text below states the resulting range"
        )
        form.addRow("Strength", self.strength_spin)
        center_row = QHBoxLayout()
        self.center_spins = []
        for axis in "XYZ":
            spin = _number_spin(-50.0, 150.0, 50.0, " %", decimals=1)
            spin.setPrefix(f"{axis} ")
            spin.setToolTip(
                f"Pattern centre along {axis} in percent of the phantom FOV "
                "(50 % = FOV centre)"
            )
            self.center_spins.append(spin)
            center_row.addWidget(spin)
        self.center_reset = QPushButton("Reset")
        self.center_reset.setToolTip("Move the pattern centre to the FOV centre")
        self.center_reset.clicked.connect(self._reset_center)
        center_row.addWidget(self.center_reset)
        form.addRow("Centre", center_row)
        pattern_layout.addLayout(form)
        self.pattern_info = QLabel()
        self.pattern_info.setWordWrap(True)
        pattern_layout.addWidget(self.pattern_info)
        controls_layout.addWidget(pattern_group)
        self.mode_combo.currentIndexChanged.connect(self._pattern_changed)
        self.strength_spin.valueChanged.connect(self._pattern_changed)
        for spin in self.center_spins:
            spin.valueChanged.connect(self._pattern_changed)

        offsets_group = QGroupBox("Constant B0 offset per region")
        offsets_layout = QVBoxLayout(offsets_group)
        self.offsets_help = QLabel()
        self.offsets_help.setWordWrap(True)
        offsets_layout.addWidget(self.offsets_help)
        self.offset_table = QTableWidget(0, 3)
        self.offset_table.setHorizontalHeaderLabels(
            ["Region", "Voxels", "B0 offset (ppm)"]
        )
        header_view = self.offset_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.Stretch)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.offset_table.verticalHeader().setVisible(False)
        self.offset_table.setMinimumHeight(140)
        self.offset_table.cellChanged.connect(self._offset_changed)
        offsets_layout.addWidget(self.offset_table)
        self.reset_offsets_button = QPushButton("Reset all offsets to 0 ppm")
        self.reset_offsets_button.clicked.connect(self._reset_offsets)
        offsets_layout.addWidget(self.reset_offsets_button)
        controls_layout.addWidget(offsets_group, 1)

        application_group = QGroupBox("Phantom coupling")
        application_layout = QVBoxLayout(application_group)
        self.auto_apply = QCheckBox(
            "Apply changes automatically to the current phantom"
        )
        self.auto_apply.setChecked(True)
        self.auto_apply.toggled.connect(self._auto_apply_changed)
        application_layout.addWidget(self.auto_apply)
        self.apply_button = QPushButton("Apply B0 field now")
        self.apply_button.clicked.connect(self.apply_field)
        application_layout.addWidget(self.apply_button)
        self.apply_status = QLabel("No phantom is selected yet.")
        self.apply_status.setWordWrap(True)
        application_layout.addWidget(self.apply_status)
        controls_layout.addWidget(application_group)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setWidget(controls)
        scrollbar = controls_scroll.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        controls_scroll.setMinimumWidth(
            max(450, controls.minimumSizeHint().width() + scrollbar)
        )
        controls_scroll.setMaximumWidth(620 + scrollbar)
        splitter.addWidget(controls_scroll)

        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_help = QLabel(
            "Resulting B0 map of the phantom in ppm, shown inside the phantom "
            "support. Drag in a slice to move the pattern centre (crosshair). "
            "The colour scale is symmetric about 0 ppm."
        )
        preview_help.setWordWrap(True)
        preview_layout.addWidget(preview_help)
        self.volume = VolumeViewerWidget(self)
        # Neutral (0 ppm) background, so voxels outside the phantom do not
        # read as the most negative offset of the diverging colour scale.
        self.volume.outside_value = 0.0
        self._build_center_targets()
        preview_layout.addWidget(self.volume, 1)
        self.preview_info = QLabel()
        self.preview_info.setWordWrap(True)
        preview_layout.addWidget(self.preview_info)
        splitter.addWidget(preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 1180])
        layout.addWidget(splitter, 1)

    def _build_center_targets(self) -> None:
        """Let a left-drag in a slice move the pattern centre, not the view."""
        self.center_targets = {}
        widget_ref = weakref.ref(self)
        for plane, view in (
            ("xy", self.volume.xy_view),
            ("xz", self.volume.xz_view),
            ("yz", self.volume.yz_view),
        ):
            box = view.getView()
            target = pg.TargetItem(
                pos=(0.0, 0.0),
                size=16,
                symbol="crosshair",
                pen=pg.mkPen((80, 220, 255), width=2),
                movable=True,
            )
            target.setZValue(2000)
            target.setToolTip("Drag to move the B0 pattern centre")
            target.setVisible(False)
            box.addItem(target, ignoreBounds=True)
            self.center_targets[plane] = target

            def target_moved(item, selected_plane=plane, ref=widget_ref):
                widget = ref()
                if widget is not None:
                    widget._center_target_moved(selected_plane, item.pos())

            def drag_center(event, axis=None, selected_plane=plane, ref=widget_ref):
                # Replaces the view-box pan: dragging anywhere in the slice
                # moves the centre to the cursor; wheel zoom is unchanged.
                widget = ref()
                if widget is None or not widget._center_enabled():
                    event.ignore()
                    return
                if event.button() != Qt.LeftButton:
                    event.ignore()
                    return
                event.accept()
                view_box = widget.center_targets[selected_plane].getViewBox()
                widget._center_target_moved(
                    selected_plane, view_box.mapSceneToView(event.scenePos())
                )

            target.sigPositionChanged.connect(target_moved)
            box.mouseDragEvent = drag_center

    def _center_enabled(self) -> bool:
        return self.field is not None and self.mode_combo.currentData() != "none"

    def _center_target_moved(self, plane, position) -> None:
        if self._updating or self.phantom is None:
            return
        fov_mm = np.asarray(self.phantom.fov, dtype=float) * 1000.0
        axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
        center = [spin.value() / 100.0 for spin in self.center_spins]
        for axis, value in zip(axes, (position.x(), position.y())):
            center[axis] = min(max(0.5 + value / max(fov_mm[axis], 1e-12), -0.5), 1.5)
        self._updating = True
        try:
            for spin, value in zip(self.center_spins, center):
                spin.setValue(value * 100.0)
        finally:
            self._updating = False
        self._pattern_changed()

    def _update_center_targets(self) -> None:
        visible = self._center_enabled() and self.phantom is not None
        fov_mm = (
            np.asarray(self.phantom.fov, dtype=float) * 1000.0
            if self.phantom is not None
            else np.ones(3)
        )
        center = [spin.value() / 100.0 for spin in self.center_spins]
        self._updating = True
        try:
            for plane, axes in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
                target = self.center_targets[plane]
                target.setPos(
                    (center[axes[0]] - 0.5) * fov_mm[axes[0]],
                    (center[axes[1]] - 0.5) * fov_mm[axes[1]],
                )
                target.setVisible(visible)
        finally:
            self._updating = False

    # -- settings <-> UI --------------------------------------------------
    def settings(self) -> B0FieldSettings:
        """Field settings currently shown in the workspace."""
        offsets = {}
        for row in range(self.offset_table.rowCount()):
            name_item = self.offset_table.item(row, 0)
            value_item = self.offset_table.item(row, 2)
            if name_item is None or value_item is None:
                continue
            try:
                offsets[name_item.text()] = float(value_item.text())
            except ValueError:
                offsets[name_item.text()] = 0.0
        return B0FieldSettings(
            mode=str(self.mode_combo.currentData()),
            strength_ppm=self.strength_spin.value(),
            center=tuple(spin.value() / 100.0 for spin in self.center_spins),
            region_offsets_ppm=offsets,
            drift_curve_hz=self._drift_curve_hz,
        )

    def _show_settings(self, settings: B0FieldSettings) -> None:
        self._updating = True
        try:
            self.mode_combo.setCurrentIndex(
                max(0, self.mode_combo.findData(settings.mode))
            )
            self.strength_spin.setValue(settings.strength_ppm)
            for spin, value in zip(self.center_spins, settings.center):
                spin.setValue(float(value) * 100.0)
            self._populate_offsets(settings.region_offsets_ppm)
            self._drift_curve_hz = settings.drift_curve_hz
        finally:
            self._updating = False
        self._update_pattern_info()

    def _populate_offsets(self, offsets) -> None:
        field = self.field
        names = [] if field is None else field.region_names
        with QSignalBlocker(self.offset_table):
            self.offset_table.setRowCount(len(names))
            for row, name in enumerate(names):
                name_item = QTableWidgetItem(name)
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                self.offset_table.setItem(row, 0, name_item)
                mask = field.region_mask(name)
                count_item = QTableWidgetItem(
                    "" if mask is None else str(int(np.count_nonzero(mask)))
                )
                count_item.setFlags(count_item.flags() & ~Qt.ItemIsEditable)
                count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.offset_table.setItem(row, 1, count_item)
                value = float(offsets.get(name, 0.0))
                value_item = QTableWidgetItem(f"{value:g}")
                if not field.editable:
                    value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
                self.offset_table.setItem(row, 2, value_item)
        kind = None if field is None else field.region_kind
        if kind == "shape":
            text = (
                "One offset per Phantom Designer shape, e.g. for susceptibility "
                "differences between compartments. Later shapes overwrite earlier "
                "ones where they overlap; the values are kept in the design, so "
                "Edit Current keeps them."
            )
        elif kind == "tissue":
            text = (
                "One offset per anatomy label, e.g. for susceptibility differences "
                "between tissues (air in the lung, iron in the liver)."
            )
        elif field is None:
            text = "Create or load a phantom to list its regions."
        else:
            text = (
                "This phantom has no shapes or anatomy labels; only the spatial "
                "pattern applies."
            )
        self.offsets_help.setText(text)
        self.reset_offsets_button.setEnabled(bool(names))

    def _reset_center(self) -> None:
        self._updating = True
        try:
            for spin in self.center_spins:
                spin.setValue(50.0)
        finally:
            self._updating = False
        self._pattern_changed()

    def _pattern_changed(self, *_):
        if self._updating:
            return
        if (
            self.sender() is self.mode_combo
            and self.mode_combo.currentData() != "none"
            and self.strength_spin.value() == 0.0
        ):
            # A pattern without strength has no effect; start from 1 ppm.
            with QSignalBlocker(self.strength_spin):
                self.strength_spin.setValue(1.0)
        self._update_pattern_info()
        self._settings_changed()

    def _offset_changed(self, row, column):
        if self._updating or column != 2:
            return
        item = self.offset_table.item(row, column)
        try:
            value = float(item.text())
            if not np.isfinite(value):
                raise ValueError
        except (AttributeError, TypeError, ValueError):
            with QSignalBlocker(self.offset_table):
                item.setText("0")
        self._settings_changed()

    def _reset_offsets(self):
        with QSignalBlocker(self.offset_table):
            for row in range(self.offset_table.rowCount()):
                item = self.offset_table.item(row, 2)
                if item is not None:
                    item.setText("0")
        self._settings_changed()

    def _update_pattern_info(self):
        mode = str(self.mode_combo.currentData())
        amplitude = self.strength_spin.value()
        enabled = mode != "none"
        for spin in self.center_spins:
            spin.setEnabled(enabled)
        self.center_reset.setEnabled(enabled)
        if hasattr(self, "center_targets"):
            self._update_center_targets()
        self.strength_spin.setEnabled(enabled)
        if mode == "none":
            text = "No spatial variation: only the per-region offsets apply."
        elif mode.startswith("linear"):
            axis = mode[-1].upper()
            text = (
                f"Linear gradient along {axis}: 0 ppm at the centre, changing by "
                f"{amplitude:g} ppm per half FOV ({2.0 * amplitude:g} ppm across "
                "the FOV). Moving the centre shifts the zero crossing."
            )
        else:
            plane = "XY plane" if mode == "radial_xy" else "3D"
            text = (
                f"Radial ({plane}): {-amplitude / 2.0:g} ppm at the centre, rising "
                f"to {amplitude / 2.0:g} ppm at the distance of the FOV corner from "
                "the FOV centre. Moving the centre moves the extremum."
            )
        self.pattern_info.setText(text)

    def _settings_changed(self):
        if self._updating:
            return
        self._settings_pending = True
        try:
            self._session_settings = self.settings()
        except ValueError:
            pass
        self._preview_timer.start()
        if self.phantom is None:
            return
        if self.auto_apply.isChecked():
            self._apply_timer.start()
        else:
            self.apply_status.setText("The B0 field changed; click Apply B0 field now.")

    def _auto_apply_changed(self, enabled: bool) -> None:
        if enabled and self.phantom is not None and self._settings_pending:
            self._apply_timer.start()
        elif not enabled:
            self._apply_timer.stop()

    # -- phantom ------------------------------------------------------------
    def set_phantom(self, phantom) -> None:
        if phantom is self.phantom and phantom is not None:
            # The same object may carry a new design (Edit Current keeps it);
            # rebuild the region bookkeeping but keep pending edits.
            self.field = PhantomB0Field(phantom)
            self._preview_timer.start()
            return
        self._apply_timer.stop()
        self.phantom = phantom
        self.field = None if phantom is None else PhantomB0Field(phantom)
        self._settings_pending = False
        self.apply_button.setEnabled(phantom is not None)
        if phantom is None:
            self._show_settings(self._session_settings or B0FieldSettings())
            self.apply_status.setText("No phantom is selected yet.")
            self._update_preview()
            return
        stored = self.field.stored_settings()
        session = self._session_settings
        if stored is not None:
            self._show_settings(stored)
            status = f"Loaded the B0 field stored with {phantom.name}."
            if stored.drift_curve_hz is not None:
                status += " Its stored B0 drift is kept unchanged (not editable here)."
        elif session is not None and not session.is_empty():
            # A new phantom without its own field takes the workspace field,
            # as the B1 workspace does; offsets carry over by region name.
            carried = B0FieldSettings(
                mode=session.mode,
                strength_ppm=session.strength_ppm,
                center=session.center,
                region_offsets_ppm={
                    name: value
                    for name, value in session.region_offsets_ppm.items()
                    if name in self.field.region_names
                },
                # A drift is not edited here, so it never carries over.
                drift_curve_hz=None,
            )
            self._show_settings(carried)
            self._settings_pending = not carried.is_empty()
            status = f"Ready to apply the workspace B0 field to {phantom.name}."
        else:
            self._show_settings(B0FieldSettings())
            status = f"{phantom.name} has no additional B0 field."
        if not self.field.editable:
            status = (
                f"{phantom.name} stores its B0 offsets in Hz (legacy design); "
                "re-create its shapes in the Phantom Designer to edit B0 here."
            )
            self._settings_pending = False
        self.apply_status.setText(status)
        self._update_preview()
        if self._settings_pending and self.auto_apply.isChecked():
            self._apply_timer.start()

    def confirm_pending_changes(self) -> bool:
        """Apply auto-applied edits that are still waiting on the timer."""
        if self._apply_timer.isActive():
            return self.apply_field()
        return True

    def apply_field(self) -> bool:
        self._apply_timer.stop()
        if self.phantom is None or self.field is None:
            self.apply_status.setText(
                "Create or load a phantom before applying a B0 field."
            )
            return False
        try:
            settings = self.settings()
            b0_map = self.field.apply(settings)
        except Exception as exc:
            self.apply_status.setText(f"B0 application failed: {exc}")
            return False
        self._settings_pending = False
        self._session_settings = settings
        mask = np.asarray(self.phantom.mask, dtype=bool).reshape(b0_map.shape)
        values = b0_map[mask] if np.any(mask) else b0_map.ravel()
        drift = (
            ""
            if settings.drift_curve_hz is None
            else f"; drift {min(settings.drift_curve_hz.values):g} … "
            f"{max(settings.drift_curve_hz.values):g} Hz"
        )
        self.apply_status.setText(
            f"Applied to {self.phantom.name}: B0 {np.min(values):.3g} … "
            f"{np.max(values):.3g} ppm inside the phantom{drift}."
        )
        self._update_preview()
        self.phantom_updated.emit(self.phantom)
        return True

    # -- preview --------------------------------------------------------------
    def _update_preview(self):
        self._preview_timer.stop()
        field = self.field
        if field is None:
            self.volume.set_volume(np.zeros((1, 1, 1)), fov_m=(1.0, 1.0, 1.0))
            self._preview_shape = None
            self.preview_info.setText("No phantom is selected yet.")
            return
        try:
            b0_map = field.preview_map(self.settings())
        except ValueError as exc:
            self.preview_info.setText(f"Preview unavailable: {exc}")
            return
        mask = np.asarray(self.phantom.mask, dtype=bool).reshape(b0_map.shape)
        values = b0_map[mask] if np.any(mask) else b0_map.ravel()
        limit = float(np.max(np.abs(values))) if values.size else 0.0
        limit = limit if limit > 1e-9 else 1.0
        display = np.where(mask, b0_map, np.nan) if np.any(mask) else b0_map
        display = np.nan_to_num(display, nan=0.0)
        color_map = pg.colormap.get("CET-D1")
        key = (id(self.phantom), b0_map.shape)
        if key == self._preview_shape:
            self.volume.update_volume_data(
                display,
                name="B0 offset",
                unit="ppm",
                levels=(-limit, limit),
                color_map=color_map,
            )
        else:
            self.volume.set_volume(
                display,
                mask=mask,
                fov_m=tuple(self.phantom.fov),
                name="B0 offset",
                unit="ppm",
                levels=(-limit, limit),
                color_map=color_map,
            )
            self._preview_shape = key
        self._update_center_targets()
        pending = " (not applied yet)" if self._settings_pending else ""
        self.preview_info.setText(
            f"B0 inside the phantom{pending}: mean {np.mean(values):.3g} ppm, range "
            f"{np.min(values):.3g} … {np.max(values):.3g} ppm, standard deviation "
            f"{np.std(values):.3g} ppm on the "
            f"{'×'.join(str(n) for n in b0_map.shape)} phantom grid."
        )
