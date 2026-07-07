"""Interactive orthogonal-slice and 3D viewers for phantom volumes."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph.opengl as gl

    HAS_OPENGL = True
except Exception:
    gl = None
    HAS_OPENGL = False


class VolumeViewerWidget(QWidget):
    """Show a scalar 3D volume as linked orthogonal planes and a point cloud."""

    indices_changed = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = np.zeros((1, 1, 1), dtype=float)
        self.mask = np.ones((1, 1, 1), dtype=bool)
        self.fov_m = (1.0, 1.0, 1.0)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        slices = QWidget()
        slices_layout = QGridLayout(slices)
        self.xy_view = self._image_view("Axial (XY)")
        self.xz_view = self._image_view("Coronal (XZ)")
        self.yz_view = self._image_view("Sagittal (YZ)")
        slices_layout.addWidget(self.xy_view, 0, 0)
        slices_layout.addWidget(self.xz_view, 0, 1)
        slices_layout.addWidget(self.yz_view, 0, 2)
        controls = QWidget()
        controls_layout = QGridLayout(controls)
        self.index_labels = []
        self.sliders = []
        for row, axis in enumerate("XYZ"):
            controls_layout.addWidget(QLabel(f"{axis} index"), row, 0)
            slider = QSlider(Qt.Horizontal)
            slider.valueChanged.connect(self._indices_updated)
            label = QLabel("0")
            controls_layout.addWidget(slider, row, 1)
            controls_layout.addWidget(label, row, 2)
            self.sliders.append(slider)
            self.index_labels.append(label)
        slices_layout.addWidget(controls, 1, 0, 1, 3)
        self.tabs.addTab(slices, "Orthogonal slices")

        self.gl_view = None
        self.scatter = None
        if HAS_OPENGL and os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen":
            self.gl_view = gl.GLViewWidget()
            self.gl_view.setCameraPosition(distance=2.5)
            axis = gl.GLAxisItem()
            axis.setSize(1, 1, 1)
            self.gl_view.addItem(axis)
            self.scatter = gl.GLScatterPlotItem(
                pos=np.zeros((0, 3)), color=(1, 1, 1, 1), size=3
            )
            self.gl_view.addItem(self.scatter)
            self.tabs.addTab(self.gl_view, "3D")
        else:
            self.tabs.addTab(QLabel("OpenGL volume view unavailable"), "3D")

        self.info = QLabel("No volume")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

    @staticmethod
    def _image_view(title: str) -> pg.ImageView:
        view = pg.ImageView()
        view.ui.roiBtn.hide()
        view.ui.menuBtn.hide()
        view.ui.histogram.axis.tickStrings = lambda values, scale, spacing: [
            f"{value * scale:.2f}" for value in values
        ]
        view.setObjectName(title)
        view.setToolTip(title)
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
        if fov_m is not None:
            fov = tuple(float(value) for value in fov_m)
            self.fov_m = fov + (1.0,) * (3 - len(fov))
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
        for label, value in zip(self.index_labels, indices):
            label.setText(str(value))
        levels = self._levels()
        xy = np.where(self.mask[:, :, iz], self.data[:, :, iz], np.nan)
        xz = np.where(self.mask[:, iy, :], self.data[:, iy, :], np.nan)
        yz = np.where(self.mask[ix, :, :], self.data[ix, :, :], np.nan)
        self._set_slice_image(self.xy_view, xy, levels)
        self._set_slice_image(self.xz_view, xz, levels)
        self._set_slice_image(self.yz_view, yz, levels)
        self.indices_changed.emit(indices)

    @staticmethod
    def _set_slice_image(view: pg.ImageView, values, levels) -> None:
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
            display.T,
            autoLevels=False,
            levels=levels,
            autoHistogramRange=False,
        )
        view.ui.histogram.setHistogramRange(low, high)

    def _levels(self):
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
        positions = (indices + 0.5) / shape - 0.5
        values = self.data[tuple(indices.T)]
        low, high = self._levels()
        normalized = np.clip((values - low) / max(high - low, 1e-15), 0, 1)
        colors = pg.colormap.get("viridis").map(normalized, mode="float")
        size = max(2.0, 12.0 / np.cbrt(max(1, len(indices))))
        self.scatter.setData(
            pos=np.asarray(positions, dtype=np.float32),
            color=np.asarray(colors, dtype=np.float32),
            size=size,
        )


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
        row.addStretch()
        layout.addLayout(row)
        self.volume = VolumeViewerWidget()
        self.volume.indices_changed.connect(self._update_spectrum)
        layout.addWidget(self.volume, 3)
        self.spectrum_plot = pg.PlotWidget(title="Frequency distribution at voxel")
        self.spectrum_plot.setLabel("bottom", "Frequency", "Hz")
        self.spectrum_plot.setLabel("left", "Amplitude", "a.u.")
        layout.addWidget(self.spectrum_plot, 1)
        self.spectrum_info = QLabel("No spectral distribution")
        layout.addWidget(self.spectrum_info)

    def set_phantom(self, phantom) -> None:
        self.phantom = phantom
        self.map_combo.blockSignals(True)
        self.map_combo.clear()
        self.map_combo.addItems(
            ["Proton density", "T1", "T2/T2*", "B0", "Mean frequency", "Mask"]
        )
        if hasattr(phantom, "species"):
            self.map_combo.addItems(
                [f"Peak: {species.name}" for species in phantom.species]
            )
        self.map_combo.blockSignals(False)
        self._map_changed()

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
            if getattr(self.phantom, "b0_map_ppm", None) is not None:
                data = self.phantom.b0_map_ppm
                unit = "ppm"
            else:
                data = (
                    np.zeros(self.phantom.shape)
                    if self.phantom.b0_map is None
                    else self.phantom.b0_map
                )
                unit = "Hz"
        elif choice == "Mean frequency":
            data = self.phantom.effective_df_map
            unit = "Hz"
        elif choice == "Mask":
            data = self.phantom.mask.astype(float)
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
        frequency, spectrum = self.phantom.spectrum_at(native_index)
        self.spectrum_plot.plot(frequency, spectrum, pen=pg.mkPen("c", width=2))
        self.spectrum_info.setText(
            f"Voxel {native_index}; {self.phantom.n_species} Lorentzian components"
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
        self._update()

    def _magnetization(self):
        if self.state_combo.currentIndex() == 0:
            return self.result.final_magnetization
        return self.result.checkpoint_magnetization[self.state_combo.currentIndex() - 1]

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
