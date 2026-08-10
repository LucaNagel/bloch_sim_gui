"""Interactive orthogonal-slice and 3D viewers for phantom volumes."""

from __future__ import annotations

import os
import weakref
from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..units import NUCLEUS_GAMMA_HZ_PER_T, hz_to_ppm, ppm_to_hz
from .widgets import compact_image_histogram

try:
    import pyqtgraph.opengl as gl

    HAS_OPENGL = True
except Exception:
    gl = None
    HAS_OPENGL = False


class _SliceScrollViewBox(pg.ViewBox):
    """Use an unmodified wheel/trackpad gesture to step through slices."""

    def __init__(self, scroll_callback):
        super().__init__(enableMenu=False)
        self._scroll_callback = scroll_callback

    def wheelEvent(self, event, axis=None):
        modifier_getter = getattr(event, "modifiers", None)
        modifiers = (
            modifier_getter()
            if callable(modifier_getter)
            else QApplication.keyboardModifiers()
        )
        if modifiers == Qt.NoModifier:
            delta_getter = getattr(event, "delta", None)
            if callable(delta_getter):
                delta = float(delta_getter())
            else:
                angle_delta = getattr(event, "angleDelta", lambda: None)()
                delta = 0.0 if angle_delta is None else float(angle_delta.y())
            if delta:
                self._scroll_callback(1 if delta > 0 else -1)
                event.accept()
                return
        super().wheelEvent(event, axis=axis)


class VolumeViewerWidget(QWidget):
    """Show a scalar 3D volume as linked orthogonal planes and a point cloud."""

    indices_changed = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = np.zeros((1, 1, 1), dtype=float)
        self.mask = np.ones((1, 1, 1), dtype=bool)
        self.fov_m = (1.0, 1.0, 1.0)
        self.display_levels = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        slices = QWidget()
        slices_layout = QGridLayout(slices)
        self.slices_layout = slices_layout
        slices_layout.setColumnStretch(0, 1)
        slices_layout.setColumnStretch(1, 1)
        slices_layout.setColumnStretch(2, 1)
        self.xy_view = self._image_view("xy", "Axial (XY)", "x", "y")
        self.xz_view = self._image_view("xz", "Coronal (XZ)", "x", "z")
        self.yz_view = self._image_view("yz", "Sagittal (YZ)", "y", "z")
        self.slice_markers = {}
        viewer_ref = weakref.ref(self)
        for plane, view in (
            ("xy", self.xy_view),
            ("xz", self.xz_view),
            ("yz", self.yz_view),
        ):
            marker = pg.ScatterPlotItem(
                size=16,
                symbol="+",
                pen=pg.mkPen("y", width=2),
                brush=None,
            )
            marker.setZValue(100)
            view.getView().addItem(marker, ignoreBounds=True)
            self.slice_markers[plane] = marker

            def select_from_click(event, selected_plane=plane, ref=viewer_ref):
                viewer = ref()
                if viewer is not None:
                    viewer._slice_clicked(selected_plane, event)

            view.getView().scene().sigMouseClicked.connect(select_from_click)
        slices_layout.addWidget(self.xy_view, 0, 0)
        slices_layout.addWidget(self.xz_view, 0, 1)
        slices_layout.addWidget(self.yz_view, 0, 2)
        self.index_labels = []
        self.sliders = []
        self.slider_controls = []
        for axis in "XYZ":
            controls = QWidget()
            controls_layout = QHBoxLayout(controls)
            controls_layout.setContentsMargins(4, 0, 4, 0)
            controls_layout.setSpacing(6)
            controls_layout.addWidget(QLabel(f"{axis} index"))
            slider = QSlider(Qt.Horizontal)
            slider.setMinimumWidth(80)
            slider.setToolTip(
                f"Select the {axis} slice. You can also scroll on the matching image."
            )
            slider.valueChanged.connect(self._indices_updated)
            label = QLabel("0")
            label.setMinimumWidth(82)
            controls_layout.addWidget(slider, 1)
            controls_layout.addWidget(label)
            self.sliders.append(slider)
            self.index_labels.append(label)
            self.slider_controls.append(controls)
        # Each plane is moved along its orthogonal axis.
        slices_layout.addWidget(self.slider_controls[2], 1, 0)
        slices_layout.addWidget(self.slider_controls[1], 1, 1)
        slices_layout.addWidget(self.slider_controls[0], 1, 2)
        self.tabs.addTab(slices, "Orthogonal slices")

        self.gl_view = None
        self.scatter = None
        self.bounds = None
        self.gl_grid = None
        if HAS_OPENGL and os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen":
            self.gl_view = gl.GLViewWidget()
            self.gl_view.setCameraPosition(distance=300, elevation=25, azimuth=35)
            self.gl_grid = gl.GLGridItem()
            self.gl_view.addItem(self.gl_grid)
            self.bounds = gl.GLLinePlotItem(
                pos=np.zeros((0, 3)),
                color=(0.85, 0.85, 0.85, 0.8),
                width=1.5,
                mode="lines",
                antialias=True,
            )
            self.gl_view.addItem(self.bounds)
            self.scatter = gl.GLScatterPlotItem(
                pos=np.zeros((0, 3)), color=(1, 1, 1, 1), size=5
            )
            self.gl_view.addItem(self.scatter)
            self.tabs.addTab(self.gl_view, "3D")
        else:
            self.tabs.addTab(QLabel("OpenGL volume view unavailable"), "3D")

        self.info = QLabel("No volume")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

    def _image_view(
        self,
        plane: str,
        title: str,
        horizontal_axis: str,
        vertical_axis: str,
    ) -> pg.ImageView:
        viewer_ref = weakref.ref(self)

        def scroll_slice(step, selected_plane=plane, ref=viewer_ref):
            viewer = ref()
            if viewer is not None:
                viewer._slice_scrolled(selected_plane, step)

        view = pg.ImageView(view=_SliceScrollViewBox(scroll_slice))
        view.ui.roiBtn.hide()
        view.ui.menuBtn.hide()
        compact_image_histogram(view)
        view.ui.histogram.axis.tickStrings = lambda values, scale, spacing: [
            f"{value * scale:.2f}" for value in values
        ]
        view.setObjectName(title)
        view.setToolTip(
            f"{title}; horizontal {horizontal_axis} [mm], "
            f"vertical {vertical_axis} [mm]. Click to select a voxel; scroll or "
            "swipe vertically to change the slice."
        )
        return view

    @property
    def indices(self) -> Tuple[int, int, int]:
        return tuple(slider.value() for slider in self.sliders)

    def set_volume(
        self,
        data,
        *,
        mask=None,
        fov_m=None,
        name: str = "Volume",
        unit: str = "",
        levels=None,
    ) -> None:
        native_values = np.asarray(data, dtype=float)
        values = self._promote_to_volume(native_values)
        if values is None:
            raise ValueError("volume data must be 1D, 2D, or 3D")

        if mask is None:
            volume_mask = np.ones(values.shape, dtype=bool)
        else:
            native_mask = np.asarray(mask, dtype=bool)
            if native_mask.shape == native_values.shape:
                # Apply exactly the same singleton axes as for the data.  A
                # plain (X, Y) mask cannot be broadcast directly to (X, Y, 1).
                mask_candidate = self._promote_to_volume(native_mask)
            else:
                mask_candidate = native_mask
            try:
                volume_mask = np.broadcast_to(mask_candidate, values.shape).copy()
            except ValueError:
                promoted_mask = self._promote_to_volume(native_mask)
                if promoted_mask is None:
                    raise ValueError(
                        f"mask shape {native_mask.shape} is incompatible with "
                        f"volume shape {native_values.shape}"
                    ) from None
                try:
                    volume_mask = np.broadcast_to(promoted_mask, values.shape).copy()
                except ValueError:
                    raise ValueError(
                        f"mask shape {native_mask.shape} is incompatible with "
                        f"volume shape {native_values.shape}"
                    ) from None

        self.data = values
        self.mask = volume_mask
        if levels is None:
            self.display_levels = None
        else:
            low, high = (float(value) for value in levels)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise ValueError(
                    "display levels must be finite and strictly increasing"
                )
            self.display_levels = (low, high)
        if fov_m is not None:
            fov = tuple(float(value) for value in fov_m)
            if len(fov) < 3:
                in_plane_voxel = min(
                    fov[index] / values.shape[index] for index in range(len(fov))
                )
                fov = fov + (in_plane_voxel,) * (3 - len(fov))
            self.fov_m = fov[:3]
        previous_signal_states = [slider.blockSignals(True) for slider in self.sliders]
        try:
            for slider, count in zip(self.sliders, values.shape):
                slider.setRange(0, count - 1)
                slider.setValue(count // 2)
        finally:
            for slider, was_blocked in zip(self.sliders, previous_signal_states):
                slider.blockSignals(was_blocked)
        self._indices_updated()
        self._update_3d()
        active = values[self.mask & np.isfinite(values)]
        if active.size:
            self.info.setText(
                f"{name}: shape={values.shape}; range={active.min():.5g}…"
                f"{active.max():.5g} {unit}"
            )
        else:
            self.info.setText(f"{name}: no active voxels")

    @staticmethod
    def _promote_to_volume(values):
        """Return a canonical ``(X, Y, Z)`` view without reordering axes."""
        if values.ndim == 1:
            return values[:, None, None]
        if values.ndim == 2:
            return values[:, :, None]
        if values.ndim == 3:
            return values
        return None

    def _indices_updated(self):
        # Clamp defensively as Qt may still deliver a queued valueChanged event
        # after a volume with different dimensions has been installed.
        indices = tuple(
            min(max(value, 0), count - 1)
            for value, count in zip(self.indices, self.data.shape)
        )
        if indices != self.indices:
            for slider, value in zip(self.sliders, indices):
                previous = slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(previous)
        ix, iy, iz = indices
        for axis, (label, value, count, fov) in enumerate(
            zip(self.index_labels, indices, self.data.shape, self.fov_m)
        ):
            position_mm = ((value + 0.5) / count - 0.5) * fov * 1000.0
            label.setText(f"{value} ({position_mm:.4g} mm)")
        levels = self._levels()
        xy = np.where(self.mask[:, :, iz], self.data[:, :, iz], np.nan)
        xz = np.where(self.mask[:, iy, :], self.data[:, iy, :], np.nan)
        yz = np.where(self.mask[ix, :, :], self.data[ix, :, :], np.nan)
        self._set_slice_image(self.xy_view, xy, levels, (self.fov_m[0], self.fov_m[1]))
        self._set_slice_image(self.xz_view, xz, levels, (self.fov_m[0], self.fov_m[2]))
        self._set_slice_image(self.yz_view, yz, levels, (self.fov_m[1], self.fov_m[2]))
        positions_mm = tuple(
            ((value + 0.5) / count - 0.5) * fov * 1000.0
            for value, count, fov in zip(indices, self.data.shape, self.fov_m)
        )
        self.slice_markers["xy"].setData([positions_mm[0]], [positions_mm[1]])
        self.slice_markers["xz"].setData([positions_mm[0]], [positions_mm[2]])
        self.slice_markers["yz"].setData([positions_mm[1]], [positions_mm[2]])
        self.indices_changed.emit(indices)

    def _slice_clicked(self, plane: str, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        view = {"xy": self.xy_view, "xz": self.xz_view, "yz": self.yz_view}[plane]
        view_box = view.getView()
        if not view_box.sceneBoundingRect().contains(event.scenePos()):
            return
        point = view_box.mapSceneToView(event.scenePos())
        self._select_plane_coordinates(plane, point.x(), point.y())

    def _slice_scrolled(self, plane: str, step: int) -> None:
        axis = {"xy": 2, "xz": 1, "yz": 0}.get(plane)
        if axis is None:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")
        slider = self.sliders[axis]
        slider.setValue(
            int(np.clip(slider.value() + int(np.sign(step)), 0, slider.maximum()))
        )

    def _select_plane_coordinates(
        self, plane: str, horizontal_mm: float, vertical_mm: float
    ) -> None:
        """Select a voxel from physical coordinates in an orthogonal plane."""
        axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        if plane not in axes:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")
        selected = list(self.indices)
        for axis, coordinate_mm in zip(
            axes[plane], (float(horizontal_mm), float(vertical_mm))
        ):
            count = self.data.shape[axis]
            extent_mm = self.fov_m[axis] * 1000.0
            index = int(np.floor((coordinate_mm / extent_mm + 0.5) * count))
            selected[axis] = int(np.clip(index, 0, count - 1))
        previous_states = [slider.blockSignals(True) for slider in self.sliders]
        try:
            for slider, value in zip(self.sliders, selected):
                slider.setValue(value)
        finally:
            for slider, was_blocked in zip(self.sliders, previous_states):
                slider.blockSignals(was_blocked)
        self._indices_updated()

    @staticmethod
    def _set_slice_image(view: pg.ImageView, values, levels, fov_m) -> None:
        """Display a slice without passing NaN/Inf histogram ranges to Qt."""
        low, high = levels
        display = np.nan_to_num(
            np.asarray(values, dtype=float),
            copy=True,
            nan=low,
            posinf=high,
            neginf=low,
        )
        view.setImage(
            display,
            autoLevels=False,
            levels=levels,
            autoHistogramRange=False,
            pos=(-fov_m[0] * 500.0, -fov_m[1] * 500.0),
            scale=(
                fov_m[0] * 1000.0 / display.shape[0],
                fov_m[1] * 1000.0 / display.shape[1],
            ),
        )
        view.ui.histogram.setHistogramRange(low, high)

    def _levels(self):
        if self.display_levels is not None:
            return self.display_levels
        valid = self.data[self.mask & np.isfinite(self.data)]
        if valid.size == 0:
            return (0.0, 1.0)
        low, high = float(valid.min()), float(valid.max())
        if np.isclose(low, high):
            delta = max(1e-6, abs(low) * 1e-6)
            return low - delta, high + delta
        return low, high

    def _update_3d(self):
        if self.scatter is None:
            return
        indices = np.argwhere(self.mask & np.isfinite(self.data))
        if indices.size == 0:
            self.scatter.setData(pos=np.zeros((0, 3)))
            return
        maximum_points = 20000
        stride = max(1, int(np.ceil(len(indices) / maximum_points)))
        indices = indices[::stride]
        shape = np.asarray(self.data.shape, dtype=float)
        fov_mm = np.asarray(self.fov_m, dtype=float) * 1000.0
        positions = ((indices + 0.5) / shape - 0.5) * fov_mm
        values = self.data[tuple(indices.T)]
        low, high = self._levels()
        normalized = np.clip((values - low) / max(high - low, 1e-15), 0, 1)
        colors = pg.colormap.get("viridis").map(normalized, mode="float")
        size = max(4.5, 18.0 / np.cbrt(max(1, len(indices))))
        self.scatter.setData(
            pos=np.asarray(positions, dtype=np.float32),
            color=np.asarray(colors, dtype=np.float32),
            size=size,
        )
        half = fov_mm / 2.0
        corners = np.asarray(
            [
                (x, y, z)
                for z in (-half[2], half[2])
                for y in (-half[1], half[1])
                for x in (-half[0], half[0])
            ],
            dtype=np.float32,
        )
        edge_indices = (
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
        self.bounds.setData(
            pos=np.vstack([corners[list(edge)] for edge in edge_indices])
        )
        self.gl_grid.setSize(x=fov_mm[0], y=fov_mm[1], z=0)
        self.gl_grid.setSpacing(
            x=max(fov_mm[0] / 10.0, 1e-6),
            y=max(fov_mm[1] / 10.0, 1e-6),
            z=1.0,
        )
        self.gl_grid.resetTransform()
        self.gl_grid.translate(0, 0, -half[2])
        self.gl_view.setCameraPosition(distance=float(max(fov_mm) * 1.8))


class PhantomInspectorWidget(QWidget):
    """Interactive spatial maps plus per-voxel frequency distribution."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.phantom = None
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Map"))
        self.map_combo = QComboBox()
        self.map_combo.currentTextChanged.connect(self._map_changed)
        row.addWidget(self.map_combo)
        row.addWidget(QLabel("Frequency display"))
        self.frequency_unit_combo = QComboBox()
        self.frequency_unit_combo.addItems(["ppm", "Hz", "kHz"])
        self.frequency_unit_combo.currentTextChanged.connect(
            self._frequency_display_changed
        )
        row.addWidget(self.frequency_unit_combo)
        self.preview_field_strength = QDoubleSpinBox()
        self.preview_field_strength.setRange(0.001, 1000.0)
        self.preview_field_strength.setDecimals(4)
        self.preview_field_strength.setValue(7.0)
        self.preview_field_strength.setSuffix(" T")
        self.preview_field_strength.valueChanged.connect(
            self._frequency_display_changed
        )
        row.addWidget(self.preview_field_strength)
        self.preview_nucleus = QComboBox()
        self.preview_nucleus.addItems(sorted(NUCLEUS_GAMMA_HZ_PER_T))
        self.preview_nucleus.currentTextChanged.connect(self._frequency_display_changed)
        row.addWidget(self.preview_nucleus)
        row.addStretch()
        layout.addLayout(row)
        self.volume = VolumeViewerWidget()
        self.volume.indices_changed.connect(self._update_spectrum)
        layout.addWidget(self.volume, 3)
        self.spectrum_plot = pg.PlotWidget(title="Frequency distribution at voxel")
        self.spectrum_plot.setLabel("bottom", "Frequency", "ppm")
        self.spectrum_plot.setLabel("left", "Amplitude", "a.u.")
        layout.addWidget(self.spectrum_plot, 1)
        self.spectrum_info = QLabel("No spectral distribution")
        layout.addWidget(self.spectrum_info)

    def set_phantom(self, phantom) -> None:
        self.phantom = phantom
        self.preview_field_strength.blockSignals(True)
        self.preview_nucleus.blockSignals(True)
        self.frequency_unit_combo.blockSignals(True)
        self.preview_field_strength.setValue(
            float(getattr(phantom, "field_strength", 7.0))
        )
        nucleus = str(getattr(phantom, "nucleus", "C13"))
        nucleus_index = self.preview_nucleus.findText(nucleus)
        self.preview_nucleus.setCurrentIndex(max(0, nucleus_index))
        self.frequency_unit_combo.setCurrentText("ppm")
        self.frequency_unit_combo.blockSignals(False)
        self.preview_nucleus.blockSignals(False)
        self.preview_field_strength.blockSignals(False)
        self.map_combo.blockSignals(True)
        self.map_combo.clear()
        self.map_combo.addItems(
            ["Proton density", "T1", "T2/T2*", "B0", "Mean frequency", "Mask"]
        )
        if hasattr(phantom, "species"):
            self.map_combo.addItems(
                [f"Peak: {species.name}" for species in phantom.species]
            )
        if hasattr(phantom, "kpl_map_s_inv"):
            self.map_combo.addItem("kPL")
        self.map_combo.blockSignals(False)
        self._map_changed()
        # Updating the volume normally emits an index change, but an already
        # selected centre voxel may not emit one on every Qt backend. Refresh
        # explicitly so a just-loaded phantom always displays its spectrum.
        self._update_spectrum()

    def _frequency_display_changed(self, *_):
        if self.map_combo.currentText() in {"B0", "Mean frequency"}:
            self._map_changed()
        else:
            self._update_spectrum()

    def _frequency_display(self):
        unit = self.frequency_unit_combo.currentText()
        field_strength = self.preview_field_strength.value()
        nucleus = self.preview_nucleus.currentText()
        scale = 1.0
        if unit == "kHz":
            scale = 1e-3
        return unit, field_strength, nucleus, scale

    def _b0_map_for_display(self):
        unit, field_strength, nucleus, scale = self._frequency_display()
        if unit == "ppm":
            if hasattr(self.phantom, "get_b0_offset_map_ppm"):
                return (
                    self.phantom.get_b0_offset_map_ppm(field_strength, nucleus),
                    "ppm",
                )
            b0 = (
                np.zeros(self.phantom.shape)
                if self.phantom.b0_map is None
                else self.phantom.b0_map
            )
            return hz_to_ppm(b0, field_strength, nucleus), "ppm"
        if hasattr(self.phantom, "get_b0_offset_map_hz"):
            data = self.phantom.get_b0_offset_map_hz(field_strength, nucleus)
        elif hasattr(self.phantom, "b0_offset_hz"):
            data = self.phantom.b0_offset_hz(field_strength, nucleus)
        else:
            data = (
                np.zeros(self.phantom.shape)
                if self.phantom.b0_map is None
                else self.phantom.b0_map
            )
        return data * scale, unit

    def _mean_frequency_for_display(self):
        unit, field_strength, nucleus, scale = self._frequency_display()
        if unit == "ppm":
            if hasattr(self.phantom, "df_map_ppm"):
                return self.phantom.df_map_ppm, "ppm"
            return (
                hz_to_ppm(self.phantom.effective_df_map, field_strength, nucleus),
                "ppm",
            )
        if hasattr(self.phantom, "df_map_ppm"):
            return (
                ppm_to_hz(self.phantom.df_map_ppm, field_strength, nucleus) * scale,
                unit,
            )
        return self.phantom.effective_df_map * scale, unit

    def _map_changed(self):
        if self.phantom is None:
            return
        choice = self.map_combo.currentText()
        unit = ""
        if choice == "Proton density":
            data = self.phantom.pd_map
        elif choice == "T1":
            data = self.phantom.t1_map * 1000
            unit = "ms"
        elif choice == "T2/T2*":
            data = self.phantom.t2_map * 1000
            unit = "ms"
        elif choice == "B0":
            data, unit = self._b0_map_for_display()
        elif choice == "Mean frequency":
            data, unit = self._mean_frequency_for_display()
        elif choice == "Mask":
            data = self.phantom.mask.astype(float)
        elif choice == "kPL":
            data = self.phantom.kpl_map_s_inv
            unit = "s⁻¹"
        elif choice.startswith("Peak: "):
            name = choice[len("Peak: ") :]
            data = self.phantom.concentration_maps[name]
        else:
            return
        self.volume.set_volume(
            data,
            mask=self.phantom.mask,
            fov_m=self.phantom.fov,
            name=choice,
            unit=unit,
        )

    def _update_spectrum(self, index=None):
        self.spectrum_plot.clear()
        if self.phantom is None or not hasattr(self.phantom, "spectrum_at"):
            self.spectrum_info.setText("Phantom has one frequency per voxel")
            return
        index = self.volume.indices if index is None else tuple(index)
        native_index = index[: len(self.phantom.shape)]
        unit, field_strength, nucleus, scale = self._frequency_display()
        if unit == "ppm" and hasattr(self.phantom, "spectrum_at_ppm"):
            frequency, spectrum = self.phantom.spectrum_at_ppm(
                native_index,
                absolute=True,
                linewidth_field_strength=field_strength,
                nucleus=nucleus,
            )
            axis_label = "Frequency"
            axis_unit = "ppm"
            reference = getattr(self.phantom, "spectral_reference_ppm", None)
            reference_suffix = (
                f"; reference {reference:g} ppm" if reference is not None else ""
            )
        else:
            frequency, spectrum = self.phantom.spectrum_at(
                native_index,
                field_strength=field_strength,
                nucleus=nucleus,
            )
            frequency = frequency * scale
            axis_label = "Frequency offset"
            axis_unit = unit
            reference_suffix = f"; {field_strength:g} T {nucleus}"
        self.spectrum_plot.setLabel("bottom", axis_label, axis_unit)
        self.spectrum_plot.plot(frequency, spectrum, pen=pg.mkPen("c", width=2))
        bandwidth = getattr(self.phantom, "spectral_bandwidth_ppm", None)
        points = getattr(self.phantom, "spectral_points", None)
        spectral_suffix = (
            f"; BW {bandwidth:g} ppm, {points} points"
            if bandwidth is not None and points is not None
            else ""
        )
        self.spectrum_info.setText(
            f"Voxel {native_index}; {self.phantom.n_species} Lorentzian components"
            f"{spectral_suffix}{reference_suffix}"
        )


class SequenceResultVolumeViewer(QWidget):
    """Spatial magnetization viewer with explicit acquisition semantics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.phantom = None
        layout = QVBoxLayout(self)
        semantics = QLabel(
            "This view shows magnetization in object coordinates, not a 3D image "
            "reconstruction. No z-FFT is applied. A z-IFFT is only valid for a "
            "measured Cartesian kz acquisition."
        )
        semantics.setWordWrap(True)
        layout.addWidget(semantics)
        row = QHBoxLayout()
        self.map_combo = QComboBox()
        self.map_combo.addItems(["Mz", "|Mxy|", "Mx", "My", "Mxy phase"])
        self.map_combo.currentTextChanged.connect(self._update)
        row.addWidget(QLabel("Map"))
        row.addWidget(self.map_combo)
        self.state_combo = QComboBox()
        self.state_combo.currentIndexChanged.connect(self._update)
        row.addWidget(QLabel("State"))
        row.addWidget(self.state_combo)
        self.pool_combo = QComboBox()
        self.pool_combo.addItem("Sum")
        self.pool_combo.currentIndexChanged.connect(self._update)
        row.addWidget(QLabel("Pool"))
        row.addWidget(self.pool_combo)
        row.addStretch()
        layout.addLayout(row)
        self.volume = VolumeViewerWidget()
        layout.addWidget(self.volume)

    def set_result(self, result, phantom) -> None:
        self.result = result
        self.phantom = phantom
        self.state_combo.blockSignals(True)
        self.state_combo.clear()
        self.state_combo.addItem("Final")
        for value in np.asarray(result.checkpoint_times_s):
            self.state_combo.addItem(f"Checkpoint {value * 1000:.3f} ms")
        self.state_combo.blockSignals(False)
        self.pool_combo.blockSignals(True)
        self.pool_combo.clear()
        self.pool_combo.addItem("Sum")
        for name in getattr(result, "pool_names", ()):
            self.pool_combo.addItem(str(name))
        self.pool_combo.setEnabled(bool(getattr(result, "pool_names", ())))
        self.pool_combo.blockSignals(False)
        self._update()

    def _magnetization(self):
        pool_index = self.pool_combo.currentIndex() - 1
        if self.state_combo.currentIndex() == 0:
            if pool_index >= 0 and self.result.final_pool_magnetization is not None:
                return self.result.final_pool_magnetization[pool_index]
            return self.result.final_magnetization
        checkpoint = self.state_combo.currentIndex() - 1
        if pool_index >= 0 and self.result.checkpoint_pool_magnetization is not None:
            return self.result.checkpoint_pool_magnetization[checkpoint, pool_index]
        return self.result.checkpoint_magnetization[checkpoint]

    def _update(self):
        if self.result is None:
            return
        magnetization = self._magnetization()
        mx, my, mz = (magnetization[..., index] for index in range(3))
        choice = self.map_combo.currentText()
        if choice == "Mz":
            data, unit = mz, ""
        elif choice == "|Mxy|":
            data, unit = np.hypot(mx, my), ""
        elif choice == "Mx":
            data, unit = mx, ""
        elif choice == "My":
            data, unit = my, ""
        else:
            data, unit = np.arctan2(my, mx), "rad"
        self.volume.set_volume(
            data,
            mask=self.phantom.mask,
            fov_m=self.phantom.fov,
            name=choice,
            unit=unit,
        )
