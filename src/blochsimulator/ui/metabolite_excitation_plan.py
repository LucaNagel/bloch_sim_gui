"""Metabolite-first SPSP planning: one automatically designed pulse per target."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..sequence.spectral_spatial import (
    SpectralSpatialPeak,
    SpectralSpatialPulse,
    SpectralSpatialPulseLibrary,
    SpectralSpatialSchedule,
    SpectralSpatialScheduleEntry,
    design_metabolite_selective_pulse,
    simulate_spectral_spatial_profile,
)
from ..units import NUCLEUS_GAMMA_HZ_PER_T, ppm_to_hz
from .spectral_spatial_designer import (
    SpectralSpatialPulseDesignerDialog,
    cached_spectral_spatial_response,
    render_spectral_spatial_waveforms,
    spectral_spatial_peak_sources,
    spectral_spatial_response_key,
    store_spectral_spatial_response,
)

_SUBSTRATE_FLIP_DEG = 10.0
_PRODUCT_FLIP_DEG = 30.0
_PEAK_COLORS = ("c", "m", "y", "g", "r", "w", (255, 140, 0), (120, 160, 255))


@dataclass
class _TargetPulse:
    """Pulse state of one target metabolite inside the plan."""

    pulse: SpectralSpatialPulse | None = None
    manual: bool = False
    design_key: tuple | None = None
    pulse_id: str | None = None
    error: str = ""


def _pulse_target(pulse: SpectralSpatialPulse, fallback: str = "") -> str:
    selective = pulse.design_metadata.get("metabolite_selective", {})
    if isinstance(selective, dict) and selective.get("target"):
        return str(selective["target"])
    if fallback:
        return fallback
    if pulse.bands and pulse.bands[0].name:
        return str(pulse.bands[0].name)
    return str(pulse.name)


def _response(pulse, positions_m, frequencies_hz, *, flip_angle_deg):
    """Return a cached Bloch response of ``pulse`` scaled to one flip angle."""
    key = spectral_spatial_response_key(pulse, positions_m, frequencies_hz) + (
        float(flip_angle_deg),
    )
    response = cached_spectral_spatial_response(key)
    if response is None:
        response = simulate_spectral_spatial_profile(
            pulse,
            positions_m,
            frequencies_hz,
            flip_angle_deg=flip_angle_deg,
            spatial_offset_m=pulse.reference_spatial_center_m,
        )
        store_spectral_spatial_response(key, response)
    return response


class MetaboliteExcitationPlanDialog(QDialog):
    """Choose which metabolite each acquisition step excites.

    The user lists the peaks present in the sample and orders the targets;
    every distinct target receives its own spectral-spatial pulse that excites
    it and suppresses all other listed peaks. Steps with the same target share
    one pulse and differ only in flip angle and repeat count.
    """

    def __init__(
        self,
        parent=None,
        *,
        scanner_parameters,
        nucleus="C13",
        field_strength_t=3.0,
        sequence_reference_ppm=0.0,
        carrier_offset_hz=0.0,
        target_thickness_m=10e-3,
        phantom=None,
        library=None,
        schedule=None,
        selected_pulse_id=None,
        allow_multiple_steps=True,
    ):
        super().__init__(parent)
        self.setWindowTitle("Metabolite excitation plan")
        self.resize(1400, 860)
        self.scanner_parameters = scanner_parameters
        self.nucleus = str(nucleus)
        self.field_strength_t = float(field_strength_t)
        self.sequence_reference_ppm = float(sequence_reference_ppm)
        self.carrier_offset_hz = float(carrier_offset_hz)
        self.phantom = phantom
        self.allow_multiple_steps = bool(allow_multiple_steps)
        self._input_library = SpectralSpatialPulseLibrary.from_mapping(library)
        self._input_schedule = SpectralSpatialSchedule.from_mapping(schedule)
        self._targets: dict[str, _TargetPulse] = {}
        self._result = None
        self._updating = False
        self._build_ui(float(target_thickness_m))
        restored = self._restore_state(selected_pulse_id)
        if not restored:
            self._load_peak_source()
            self._one_step_per_metabolite()
        self._refresh_all()

    # ------------------------------------------------------------------ UI
    @staticmethod
    def _double(minimum, maximum, value, suffix="", decimals=2):
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(decimals)
        control.setValue(value)
        control.setSuffix(suffix)
        control.setKeyboardTracking(False)
        return control

    def _build_ui(self, target_thickness_m):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(left)

        # 1. Peaks present in the sample.
        peaks_group = QGroupBox("1. Metabolites in the sample")
        peaks_layout = QVBoxLayout(peaks_group)
        peaks_hint = QLabel(
            "Every checked peak is either excited (when a step targets it) or "
            "suppressed by the other steps' pulses."
        )
        peaks_hint.setWordWrap(True)
        peaks_layout.addWidget(peaks_hint)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Load from"))
        self.peak_source = QComboBox()
        for label, specification in spectral_spatial_peak_sources(
            nucleus=self.nucleus,
            field_strength_t=self.field_strength_t,
            sequence_reference_ppm=self.sequence_reference_ppm,
            carrier_offset_hz=self.carrier_offset_hz,
            phantom=self.phantom,
        ):
            self.peak_source.addItem(label, specification)
        source_row.addWidget(self.peak_source, 1)
        peaks_layout.addLayout(source_row)
        self.peak_table = QTableWidget(0, 4)
        self.peak_table.setHorizontalHeaderLabels(
            ["Use", "Metabolite", "Position (ppm)", "Offset from carrier (Hz)"]
        )
        self.peak_table.verticalHeader().setVisible(False)
        self.peak_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.peak_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        peaks_layout.addWidget(self.peak_table)
        peak_buttons = QHBoxLayout()
        self.add_peak_button = QPushButton("Add peak")
        self.remove_peak_button = QPushButton("Remove peak")
        peak_buttons.addWidget(self.add_peak_button)
        peak_buttons.addWidget(self.remove_peak_button)
        peak_buttons.addStretch(1)
        peaks_layout.addLayout(peak_buttons)
        left_layout.addWidget(peaks_group, 3)

        # 2. Acquisition order.
        steps_group = QGroupBox(
            "2. Acquisition plan" if self.allow_multiple_steps else "2. Target"
        )
        steps_layout = QVBoxLayout(steps_group)
        self.steps_table = QTableWidget(0, 4)
        self.steps_table.setHorizontalHeaderLabels(
            ["Excite", "Flip (°)", "Repeats", "Pulse"]
        )
        self.steps_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.steps_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.steps_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.steps_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        steps_layout.addWidget(self.steps_table)
        self.step_buttons = QWidget()
        step_buttons = QHBoxLayout(self.step_buttons)
        step_buttons.setContentsMargins(0, 0, 0, 0)
        self.add_step_button = QPushButton("Add step")
        self.remove_step_button = QPushButton("Remove")
        self.step_up_button = QPushButton("Up")
        self.step_down_button = QPushButton("Down")
        self.per_metabolite_button = QPushButton("One step per metabolite")
        self.per_metabolite_button.setToolTip(
            "Replace the plan with one step for every checked peak, in table order"
        )
        for button in (
            self.add_step_button,
            self.remove_step_button,
            self.step_up_button,
            self.step_down_button,
        ):
            step_buttons.addWidget(button)
        step_buttons.addStretch(1)
        step_buttons.addWidget(self.per_metabolite_button)
        steps_layout.addWidget(self.step_buttons)
        self.order_options = QWidget()
        order_form = QFormLayout(self.order_options)
        order_form.setContentsMargins(0, 0, 0, 0)
        self.selection_level = QComboBox()
        self.selection_level.addItem("Image frame (all TRs of one image)", "frame")
        self.selection_level.addItem("TR (interleave every excitation)", "tr")
        self.cyclic = QCheckBox("Repeat the plan until the scan ends")
        self.cyclic.setChecked(True)
        order_form.addRow("Next step after each", self.selection_level)
        order_form.addRow("", self.cyclic)
        steps_layout.addWidget(self.order_options)
        self.single_step_note = QLabel(
            "This sequence plays one excitation pulse; the flip-angle setting "
            "of the sequence scales it. Plans with several metabolites in a row "
            "are available for FLASH."
        )
        self.single_step_note.setWordWrap(True)
        steps_layout.addWidget(self.single_step_note)
        for widget in (self.step_buttons, self.order_options):
            widget.setVisible(self.allow_multiple_steps)
        self.single_step_note.setVisible(not self.allow_multiple_steps)
        left_layout.addWidget(steps_group, 3)

        # 3. Shared pulse settings.
        settings_group = QGroupBox("3. Pulse requirements (all targets)")
        settings_form = QFormLayout(settings_group)
        self.thickness_mm = self._double(0.1, 500.0, target_thickness_m * 1e3, " mm")
        self.passband_hz = self._double(1.0, 5000.0, 20.0, " Hz", 1)
        self.passband_hz.setPrefix("± ")
        self.passband_hz.setToolTip(
            "Frequency range around the target that keeps at least 90 % of the "
            "requested signal (B0 inhomogeneity tolerance)"
        )
        self.suppression_hz = self._double(1.0, 5000.0, 20.0, " Hz", 1)
        self.suppression_hz.setPrefix("± ")
        self.suppression_hz.setToolTip(
            "Frequency range around every other peak that must stay suppressed"
        )
        self.leak_pct = self._double(0.1, 50.0, 5.0, " %", 1)
        self.leak_pct.setToolTip(
            "Largest allowed signal of a suppressed peak, relative to the "
            "target's signal at the same flip angle"
        )
        self.max_duration_ms = self._double(2.0, 200.0, 50.0, " ms", 1)
        self.max_duration_ms.setToolTip(
            "Longer pulses separate close peaks better but lose more signal to T2*"
        )
        settings_form.addRow("Slice / slab thickness", self.thickness_mm)
        settings_form.addRow("Excite target within", self.passband_hz)
        settings_form.addRow("Suppress others within", self.suppression_hz)
        settings_form.addRow("Allowed leak", self.leak_pct)
        settings_form.addRow("Maximum pulse duration", self.max_duration_ms)

        self.advanced_toggle = QCheckBox("Show advanced pulse construction")
        settings_form.addRow(self.advanced_toggle)
        self.advanced = QWidget()
        advanced_form = QFormLayout(self.advanced)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        self.spatial_tbw = self._double(1.0, 20.0, 3.0, "", 2)
        minimum_subpulse_s = 3.0 / (
            (float(self.scanner_parameters.max_grad_mtm) / 1000.0)
            * NUCLEUS_GAMMA_HZ_PER_T[self.nucleus]
            * target_thickness_m
        )
        self.subpulse_duration_ms = self._double(
            0.01, 100.0, max(0.7, minimum_subpulse_s * 1.05 * 1000.0), " ms", 3
        )
        self.spectral_phase_type = QComboBox()
        self.spectral_phase_type.addItem("Linear phase", "linear")
        self.spectral_phase_type.addItem("Minimum phase", "minimum")
        self.max_gradient = self._double(
            0.1, 1000.0, float(self.scanner_parameters.max_grad_mtm), " mT/m"
        )
        self.max_slew = self._double(
            1.0, 10000.0, float(self.scanner_parameters.max_slew_tms), " T/m/s"
        )
        self.max_b1_ut = self._double(0.01, 1000.0, 200.0, " µT")
        advanced_form.addRow("Spatial time-bandwidth product", self.spatial_tbw)
        advanced_form.addRow("Subpulse duration", self.subpulse_duration_ms)
        advanced_form.addRow("Spectral phase", self.spectral_phase_type)
        advanced_form.addRow("Maximum gradient", self.max_gradient)
        advanced_form.addRow("Maximum slew rate", self.max_slew)
        advanced_form.addRow("Maximum B1", self.max_b1_ut)
        self.advanced.setVisible(False)
        settings_form.addRow(self.advanced)
        manual_row = QHBoxLayout()
        self.edit_manual_button = QPushButton(
            "Edit selected pulse in detailed designer…"
        )
        self.edit_manual_button.setToolTip(
            "Hand-tune the selected target's pulse; it is then kept as is and "
            "no longer redesigned automatically"
        )
        self.reset_automatic_button = QPushButton("Back to automatic design")
        manual_row.addWidget(self.edit_manual_button)
        manual_row.addWidget(self.reset_automatic_button)
        manual_row.addStretch(1)
        settings_form.addRow(manual_row)
        left_layout.addWidget(settings_group, 0)

        # Right: results.
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setSizes([620, 780])
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        overview_hint = QLabel(
            "Flip angle that each step gives every peak at the slice centre. "
            "Green: intended target. Red: leak above the allowed limit."
        )
        overview_hint.setWordWrap(True)
        overview_layout.addWidget(overview_hint)
        self.overview_table = QTableWidget(0, 0)
        self.overview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        overview_layout.addWidget(self.overview_table)
        self.tabs.addTab(overview, "Overview")
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel("bottom", "Offset from carrier (Hz)")
        self.spectrum_plot.setLabel("left", "Signal relative to target")
        self.spectrum_plot.addLegend()
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.25)
        self.tabs.addTab(self.spectrum_plot, "Spectrum (selected step)")
        self.profile_plot = pg.PlotWidget()
        self.profile_plot.setLabel("bottom", "Position (mm)")
        self.profile_plot.setLabel("left", "|Mxy| / M0")
        self.profile_plot.addLegend()
        self.profile_plot.showGrid(x=True, y=True, alpha=0.25)
        self.tabs.addTab(self.profile_plot, "Slice profiles (selected step)")
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setLabel("bottom", "Time (ms)")
        self.waveform_plot.setLabel("left", "Normalized amplitude")
        self.waveform_plot.addLegend()
        self.tabs.addTab(self.waveform_plot, "Waveforms (selected step)")

        bottom = QHBoxLayout()
        self.status = QLabel()
        self.status.setWordWrap(True)
        bottom.addWidget(self.status, 1)
        self.design_button = QPushButton("Design pulses")
        self.design_button.setToolTip(
            "Design every pulse that is missing or out of date and check it "
            "with a Bloch simulation"
        )
        bottom.addWidget(self.design_button)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Apply to sequence")
        bottom.addWidget(self.buttons)
        root.addLayout(bottom)

        self.peak_source.activated.connect(lambda *_: self._load_peak_source())
        self.peak_table.itemChanged.connect(self._peak_table_changed)
        self.add_peak_button.clicked.connect(self._add_peak)
        self.remove_peak_button.clicked.connect(self._remove_peaks)
        self.add_step_button.clicked.connect(self._add_step)
        self.remove_step_button.clicked.connect(self._remove_step)
        self.step_up_button.clicked.connect(lambda: self._move_step(-1))
        self.step_down_button.clicked.connect(lambda: self._move_step(1))
        self.per_metabolite_button.clicked.connect(self._per_metabolite_clicked)
        self.steps_table.itemSelectionChanged.connect(self._refresh_selected_plots)
        self.advanced_toggle.toggled.connect(self.advanced.setVisible)
        for control in (
            self.thickness_mm,
            self.passband_hz,
            self.suppression_hz,
            self.leak_pct,
            self.max_duration_ms,
            self.spatial_tbw,
            self.subpulse_duration_ms,
            self.max_gradient,
            self.max_slew,
            self.max_b1_ut,
        ):
            control.valueChanged.connect(lambda *_: self._refresh_all())
        self.spectral_phase_type.currentIndexChanged.connect(
            lambda *_: self._refresh_all()
        )
        self.edit_manual_button.clicked.connect(self._edit_selected_manually)
        self.reset_automatic_button.clicked.connect(self._reset_selected_to_automatic)
        self.design_button.clicked.connect(lambda: self.design_pending())
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)

    # --------------------------------------------------------------- peaks
    def _append_peak_row(self, name, absolute_ppm, *, enabled=True, metadata=None):
        row = self.peak_table.rowCount()
        self.peak_table.insertRow(row)
        use_item = QTableWidgetItem()
        use_item.setFlags(
            Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
        )
        use_item.setCheckState(Qt.Checked if enabled else Qt.Unchecked)
        self.peak_table.setItem(row, 0, use_item)
        name_item = QTableWidgetItem(str(name))
        name_item.setData(Qt.UserRole, dict(metadata or {}))
        self.peak_table.setItem(row, 1, name_item)
        self.peak_table.setItem(row, 2, QTableWidgetItem(f"{float(absolute_ppm):.6g}"))
        offset_item = QTableWidgetItem()
        offset_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.peak_table.setItem(row, 3, offset_item)

    def _set_peak_rows(self, components, enabled_keys=None):
        self._updating = True
        self.peak_table.blockSignals(True)
        self.peak_table.setRowCount(0)
        for component in components:
            if isinstance(component, SpectralSpatialPeak):
                absolute_ppm = component.absolute_ppm
                metadata = {
                    "t1_s": component.t1_s,
                    "t2_star_s": component.t2_star_s,
                    "source": component.source,
                }
            else:
                absolute_ppm = component.chemical_shift_ppm
                metadata = {
                    "t1_s": float(getattr(component, "t1_s", np.inf)),
                    "t2_star_s": float(getattr(component, "t2_star_s", np.inf)),
                    "source": "preset",
                }
            key = getattr(component, "key", None)
            self._append_peak_row(
                component.name,
                absolute_ppm,
                enabled=enabled_keys is None or key in enabled_keys,
                metadata=metadata,
            )
        self.peak_table.blockSignals(False)
        self._updating = False
        self._refresh_peak_offsets()

    def _load_peak_source(self):
        specification = self.peak_source.currentData()
        if not isinstance(specification, dict):
            return
        components = specification.get("components", ())
        self._set_peak_rows(components, specification.get("enabled_keys"))
        if specification.get("key") == "custom" and not components:
            self._add_peak()
        self._refresh_all()

    def _add_peak(self):
        self.peak_table.blockSignals(True)
        self._append_peak_row(
            f"Peak {self.peak_table.rowCount() + 1}",
            self.sequence_reference_ppm,
            metadata={"source": "custom"},
        )
        self.peak_table.blockSignals(False)
        self._refresh_all()
        self.peak_table.selectRow(self.peak_table.rowCount() - 1)

    def _remove_peaks(self):
        rows = sorted(
            {index.row() for index in self.peak_table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.peak_table.removeRow(row)
        self._refresh_all()

    def _peak_table_changed(self, *_):
        if not self._updating:
            self._refresh_all()

    def _refresh_peak_offsets(self):
        self.peak_table.blockSignals(True)
        for row in range(self.peak_table.rowCount()):
            try:
                absolute_ppm = float(self.peak_table.item(row, 2).text())
                text = f"{self._carrier_offset_hz(absolute_ppm):+.1f}"
            except (AttributeError, TypeError, ValueError):
                text = "invalid"
            self.peak_table.item(row, 3).setText(text)
        self.peak_table.blockSignals(False)

    def _carrier_offset_hz(self, absolute_ppm):
        sequence_hz = float(
            ppm_to_hz(
                float(absolute_ppm) - self.sequence_reference_ppm,
                self.field_strength_t,
                self.nucleus,
            )
        )
        return sequence_hz - self.carrier_offset_hz

    def peaks(self, *, strict=False) -> tuple[SpectralSpatialPeak, ...]:
        """Return the checked peaks relative to the current carrier."""
        peaks = []
        names = set()
        for row in range(self.peak_table.rowCount()):
            use_item = self.peak_table.item(row, 0)
            if use_item is None or use_item.checkState() != Qt.Checked:
                continue
            try:
                name = self.peak_table.item(row, 1).text().strip()
                absolute_ppm = float(self.peak_table.item(row, 2).text())
                if not name:
                    raise ValueError("name must not be empty")
                if name in names:
                    raise ValueError(f"duplicate metabolite name {name!r}")
                if not np.isfinite(absolute_ppm):
                    raise ValueError("position must be finite")
            except (AttributeError, TypeError, ValueError) as exc:
                if strict:
                    raise ValueError(f"invalid peak in row {row + 1}: {exc}") from exc
                continue
            names.add(name)
            metadata = self.peak_table.item(row, 1).data(Qt.UserRole) or {}
            offset = self._carrier_offset_hz(absolute_ppm)

            def lifetime(key):
                value = metadata.get(key, np.inf)
                return np.inf if value is None else float(value)

            peaks.append(
                SpectralSpatialPeak(
                    name=name,
                    absolute_ppm=absolute_ppm,
                    sequence_offset_hz=offset + self.carrier_offset_hz,
                    carrier_offset_hz=offset,
                    t1_s=lifetime("t1_s"),
                    t2_star_s=lifetime("t2_star_s"),
                    source=str(metadata.get("source", "custom")),
                )
            )
        return tuple(peaks)

    # --------------------------------------------------------------- steps
    def _append_step(self, target, flip_deg, repeats=1, *, entry_id=None, enabled=True):
        row = self.steps_table.rowCount()
        self.steps_table.insertRow(row)
        target_combo = QComboBox()
        target_combo.setProperty("target", str(target))
        target_combo.currentIndexChanged.connect(
            lambda *_, combo=target_combo: self._step_target_changed(combo)
        )
        flip = self._double(0.1, 180.0, float(flip_deg), "", 1)
        flip.valueChanged.connect(lambda *_: self._refresh_all())
        repeat = QSpinBox()
        repeat.setRange(1, 10_000)
        repeat.setValue(int(repeats))
        repeat.valueChanged.connect(lambda *_: self._refresh_summary())
        self.steps_table.setCellWidget(row, 0, target_combo)
        self.steps_table.setCellWidget(row, 1, flip)
        self.steps_table.setCellWidget(row, 2, repeat)
        status = QTableWidgetItem()
        status.setData(Qt.UserRole, {"entry_id": entry_id, "enabled": enabled})
        self.steps_table.setItem(row, 3, status)
        self._populate_target_combo(target_combo)
        return row

    def _populate_target_combo(self, combo):
        target = combo.property("target")
        names = [peak.name for peak in self.peaks()]
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        if target in names:
            combo.setCurrentIndex(names.index(target))
        else:
            combo.addItem(f"{target} (not in list)", target)
            combo.setCurrentIndex(combo.count() - 1)
        combo.blockSignals(False)

    def _step_target_changed(self, combo):
        data = combo.currentData()
        combo.setProperty("target", data if data is not None else combo.currentText())
        self._refresh_all()

    def steps(self) -> list[dict]:
        rows = []
        for row in range(self.steps_table.rowCount()):
            widgets = [self.steps_table.cellWidget(row, column) for column in range(3)]
            status = self.steps_table.item(row, 3)
            if status is None or None in widgets:
                continue  # row is still being built
            state = status.data(Qt.UserRole) or {}
            rows.append(
                {
                    "target": str(widgets[0].property("target")),
                    "flip_angle_deg": widgets[1].value(),
                    "repeat_count": widgets[2].value(),
                    "entry_id": state.get("entry_id"),
                    "enabled": bool(state.get("enabled", True)),
                }
            )
        return rows

    def _default_flip(self, target):
        peaks = self.peaks()
        if self.allow_multiple_steps and peaks and peaks[0].name == target:
            return _SUBSTRATE_FLIP_DEG
        return _PRODUCT_FLIP_DEG

    def _one_step_per_metabolite(self):
        self.steps_table.setRowCount(0)
        peaks = self.peaks()
        if not self.allow_multiple_steps:
            peaks = peaks[:1]
        for peak in peaks:
            self._append_step(peak.name, self._default_flip(peak.name))
        if self.steps_table.rowCount():
            self.steps_table.selectRow(0)

    def _per_metabolite_clicked(self):
        self._one_step_per_metabolite()
        self._refresh_all()

    def _add_step(self):
        steps = self.steps()
        used = {step["target"] for step in steps}
        peaks = self.peaks()
        if not peaks:
            return
        target = next((peak.name for peak in peaks if peak.name not in used), None)
        target = target or peaks[0].name
        row = self._append_step(target, self._default_flip(target))
        self.steps_table.selectRow(row)
        self._refresh_all()

    def _remove_step(self):
        row = self.steps_table.currentRow()
        if row >= 0 and self.steps_table.rowCount() > 1:
            self.steps_table.removeRow(row)
            self.steps_table.selectRow(min(row, self.steps_table.rowCount() - 1))
            self._refresh_all()

    def _move_step(self, direction):
        row = self.steps_table.currentRow()
        new_row = row + int(direction)
        if row < 0 or not 0 <= new_row < self.steps_table.rowCount():
            return
        steps = self.steps()
        steps.insert(new_row, steps.pop(row))
        self._set_steps(steps)
        self.steps_table.selectRow(new_row)
        self._refresh_all()

    def _set_steps(self, steps):
        self.steps_table.setRowCount(0)
        for step in steps:
            self._append_step(
                step["target"],
                step["flip_angle_deg"],
                step["repeat_count"],
                entry_id=step.get("entry_id"),
                enabled=step.get("enabled", True),
            )

    def _selected_step(self):
        row = self.steps_table.currentRow()
        steps = self.steps()
        if not steps:
            return None
        return steps[row if 0 <= row < len(steps) else 0]

    # -------------------------------------------------------------- design
    def design_settings(self) -> dict:
        """Keyword arguments shared by every automatic design of this plan."""
        return {
            "slice_thickness_m": self.thickness_mm.value() / 1000.0,
            "passband_half_width_hz": self.passband_hz.value(),
            "suppression_half_width_hz": self.suppression_hz.value(),
            "leak_tolerance": self.leak_pct.value() / 100.0,
            "max_duration_s": self.max_duration_ms.value() / 1000.0,
            "spatial_time_bandwidth_product": self.spatial_tbw.value(),
            "subpulse_duration_s": self.subpulse_duration_ms.value() / 1000.0,
            "spectral_phase_type": self.spectral_phase_type.currentData(),
            "rf_raster_s": float(self.scanner_parameters.rf_raster_time_s),
            "gradient_raster_s": float(self.scanner_parameters.grad_raster_time_s),
            "max_gradient_t_per_m": self.max_gradient.value() / 1000.0,
            "max_slew_t_per_m_s": self.max_slew.value(),
            "max_b1_t": self.max_b1_ut.value() * 1e-6,
            "nucleus": self.nucleus,
        }

    def _design_flip(self, target):
        flips = [s["flip_angle_deg"] for s in self.steps() if s["target"] == target]
        return max(flips) if flips else _PRODUCT_FLIP_DEG

    def _design_key(self, target, peaks):
        return (
            target,
            round(self._design_flip(target), 6),
            tuple((peak.name, round(peak.absolute_ppm, 9)) for peak in peaks),
            self.nucleus,
            self.field_strength_t,
            self.sequence_reference_ppm,
            self.carrier_offset_hz,
            tuple(sorted(self.design_settings().items())),
        )

    def _target_state(self, target, peaks=None) -> str:
        """Return ``ok``, ``leak``, ``manual``, ``stale``, ``missing`` or ``error``."""
        peaks = self.peaks() if peaks is None else peaks
        if target not in {peak.name for peak in peaks}:
            return "missing"
        state = self._targets.get(target)
        if state is None or state.pulse is None:
            failed = (
                state is not None
                and state.error
                and state.design_key == self._design_key(target, peaks)
            )
            return "error" if failed else "stale"
        if state.manual:
            return "manual"
        if state.design_key != self._design_key(target, peaks):
            return "stale"
        selective = state.pulse.design_metadata.get("metabolite_selective", {})
        return "ok" if selective.get("meets_tolerance") else "leak"

    def stale_targets(self) -> list[str]:
        peaks = self.peaks()
        targets = dict.fromkeys(step["target"] for step in self.steps())
        return [
            target for target in targets if self._target_state(target, peaks) == "stale"
        ]

    def design_pending(self) -> bool:
        """Design every stale target; returns whether all targets have a pulse."""
        try:
            peaks = self.peaks(strict=True)
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid metabolite list", str(exc))
            return False
        targets = self.stale_targets()
        if not targets:
            return self._all_targets_designed()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for index, target in enumerate(targets):
                self.status.setText(
                    f"Designing pulse {index + 1} of {len(targets)}: {target}…"
                )
                QApplication.processEvents()
                state = self._targets.setdefault(target, _TargetPulse())
                key = self._design_key(target, peaks)
                try:
                    design = design_metabolite_selective_pulse(
                        target,
                        peaks,
                        flip_angle_deg=self._design_flip(target),
                        name=f"{target} SPSP",
                        **self.design_settings(),
                    )
                except Exception as exc:
                    state.pulse = None
                    state.error = str(exc)
                    state.design_key = key
                    continue
                metadata = dict(design.pulse.design_metadata)
                selective = dict(metadata["metabolite_selective"])
                selective["reference"] = {
                    "nucleus": self.nucleus,
                    "field_strength_t": self.field_strength_t,
                    "sequence_reference_ppm": self.sequence_reference_ppm,
                    "rf_carrier_offset_hz": self.carrier_offset_hz,
                }
                metadata["metabolite_selective"] = selective
                metadata["frequency_prescription"] = self._frequency_prescription()
                state.pulse = replace(design.pulse, design_metadata=metadata)
                state.error = ""
                state.manual = False
                state.design_key = key
        finally:
            QApplication.restoreOverrideCursor()
        self._refresh_all()
        return self._all_targets_designed()

    def _frequency_prescription(self):
        """Peak table snapshot so the detailed designer opens with these peaks."""
        rows = []
        for peak in self.peaks():
            rows.append(
                {
                    "enabled": True,
                    "name": peak.name,
                    "absolute_ppm": peak.absolute_ppm,
                    "relative_weight": 1.0,
                    "t1_s": None if not np.isfinite(peak.t1_s) else peak.t1_s,
                    "t2_star_s": (
                        None if not np.isfinite(peak.t2_star_s) else peak.t2_star_s
                    ),
                    "source": peak.source,
                }
            )
        return {
            "schema_version": 1,
            "preset_key": "custom",
            "bands_linked_to_peak_table": False,
            "reference": {
                "nucleus": self.nucleus,
                "field_strength_t": self.field_strength_t,
                "sequence_reference_ppm": self.sequence_reference_ppm,
                "rf_carrier_offset_hz": self.carrier_offset_hz,
            },
            "peak_rows": rows,
        }

    def _all_targets_designed(self):
        peaks = self.peaks()
        return bool(self.steps()) and all(
            self._target_state(step["target"], peaks) in {"ok", "leak", "manual"}
            for step in self.steps()
        )

    def _edit_selected_manually(self):
        step = self._selected_step()
        if step is None:
            return
        target = step["target"]
        state = self._targets.setdefault(target, _TargetPulse())
        if state.pulse is None:
            self.design_pending()
        initial = self._targets[target].pulse
        dialog = SpectralSpatialPulseDesignerDialog(
            self,
            scanner_parameters=self.scanner_parameters,
            nucleus=self.nucleus,
            field_strength_t=self.field_strength_t,
            sequence_reference_ppm=self.sequence_reference_ppm,
            carrier_offset_hz=self.carrier_offset_hz,
            target_thickness_m=self.thickness_mm.value() / 1000.0,
            initial_pulse=initial,
            phantom=self.phantom,
        )
        if dialog.exec_() != QDialog.Accepted or dialog.pulse() is None:
            return
        pulse = dialog.pulse()
        metadata = dict(pulse.design_metadata)
        metadata.pop("metabolite_selective", None)
        metadata["manual_target"] = target
        state.pulse = replace(pulse, design_metadata=metadata)
        state.manual = True
        state.error = ""
        self._refresh_all()

    def _reset_selected_to_automatic(self):
        step = self._selected_step()
        if step is None:
            return
        state = self._targets.get(step["target"])
        if state is not None and state.manual:
            state.manual = False
            state.pulse = None
            state.design_key = None
        self._refresh_all()

    # ------------------------------------------------------------- restore
    def _restore_state(self, selected_pulse_id) -> bool:
        library = self._input_library
        if not self.allow_multiple_steps:
            entries = []
            if selected_pulse_id is not None and selected_pulse_id in library.pulses:
                entries = [
                    SpectralSpatialScheduleEntry(
                        pulse_id=selected_pulse_id,
                        flip_angle_deg=library.get(
                            selected_pulse_id
                        ).reference_flip_angle_deg,
                    )
                ]
            entries = entries or self._input_schedule.entries[:1]
        else:
            entries = list(self._input_schedule.entries)
            index = self.selection_level.findData(self._input_schedule.selection_level)
            self.selection_level.setCurrentIndex(max(0, index))
            self.cyclic.setChecked(self._input_schedule.cyclic)
        steps = []
        for entry in entries:
            if entry.pulse_id not in library.pulses:
                continue
            pulse = library.get(entry.pulse_id)
            target = _pulse_target(
                pulse,
                entry.target_metabolite
                or str(pulse.design_metadata.get("manual_target", "")),
            )
            if target not in self._targets:
                selective = "metabolite_selective" in pulse.design_metadata
                self._targets[target] = _TargetPulse(
                    pulse=pulse, manual=not selective, pulse_id=entry.pulse_id
                )
            steps.append(
                {
                    "target": target,
                    "flip_angle_deg": entry.effective_flip_angle_deg(pulse),
                    "repeat_count": entry.repeat_count,
                    "entry_id": entry.entry_id,
                    "enabled": entry.enabled,
                }
            )
        if not steps:
            return False
        self._restore_peaks_and_settings()
        self._set_steps(steps)
        self.steps_table.selectRow(0)
        # Mark restored automatic pulses current when nothing they depend on
        # changed; a changed reference or carrier makes them stale.
        peaks = self.peaks()
        for target, state in self._targets.items():
            if state.manual or state.pulse is None:
                continue
            selective = state.pulse.design_metadata["metabolite_selective"]
            reference = selective.get("reference", {})
            if (
                reference.get("nucleus") == self.nucleus
                and np.isclose(
                    reference.get("field_strength_t", np.nan), self.field_strength_t
                )
                and np.isclose(
                    reference.get("sequence_reference_ppm", np.nan),
                    self.sequence_reference_ppm,
                )
                and np.isclose(
                    reference.get("rf_carrier_offset_hz", np.nan),
                    self.carrier_offset_hz,
                )
                and set(_selective_peak_names(selective)) == {p.name for p in peaks}
            ):
                state.design_key = self._design_key(target, peaks)
        return True

    def _restore_peaks_and_settings(self):
        automatic = [
            state.pulse
            for state in self._targets.values()
            if state.pulse is not None and not state.manual
        ]
        reference = automatic[0] if automatic else None
        if reference is None:
            for state in self._targets.values():
                if state.pulse is not None:
                    reference = state.pulse
                    break
        peaks = ()
        if reference is not None:
            prescription = reference.design_metadata.get("frequency_prescription", {})
            peaks = tuple(
                SpectralSpatialPeak(
                    name=str(row["name"]),
                    absolute_ppm=float(row["absolute_ppm"]),
                    sequence_offset_hz=0.0,
                    carrier_offset_hz=0.0,
                    t1_s=np.inf if row.get("t1_s") is None else float(row["t1_s"]),
                    t2_star_s=(
                        np.inf
                        if row.get("t2_star_s") is None
                        else float(row["t2_star_s"])
                    ),
                    source=str(row.get("source", "custom")),
                )
                for row in prescription.get("peak_rows", ())
                if row.get("enabled", True)
            )
        if peaks:
            self.peak_source.insertItem(
                0,
                "Peaks of the current plan",
                {"key": "plan", "components": peaks, "enabled_keys": None},
            )
            self.peak_source.setCurrentIndex(0)
            self._set_peak_rows(peaks)
        else:
            self._load_peak_source()
        if reference is None:
            return
        selective = reference.design_metadata.get("metabolite_selective", {})
        # The slice thickness always follows the sequence form.
        values = [
            (self.spatial_tbw, reference.spatial_time_bandwidth_product),
            (self.subpulse_duration_ms, reference.subpulse_duration_s * 1000.0),
            (self.max_gradient, reference.max_gradient_t_per_m * 1000.0),
            (self.max_slew, reference.max_slew_t_per_m_s),
        ]
        if reference.max_b1_t is not None:
            values.append((self.max_b1_ut, reference.max_b1_t * 1e6))
        for control, key, scale in (
            (self.passband_hz, "passband_half_width_hz", 1.0),
            (self.suppression_hz, "suppression_half_width_hz", 1.0),
            (self.leak_pct, "leak_tolerance", 100.0),
            (self.max_duration_ms, "max_duration_s", 1000.0),
        ):
            if key in selective:
                values.append((control, float(selective[key]) * scale))
        for control, value in values:
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        index = self.spectral_phase_type.findData(reference.spectral_phase_type)
        self.spectral_phase_type.blockSignals(True)
        self.spectral_phase_type.setCurrentIndex(max(0, index))
        self.spectral_phase_type.blockSignals(False)

    # ------------------------------------------------------------- display
    def _refresh_all(self):
        if self._updating:
            return
        self._refresh_peak_offsets()
        for row in range(self.steps_table.rowCount()):
            self._populate_target_combo(self.steps_table.cellWidget(row, 0))
        self._refresh_step_status()
        self._refresh_overview()
        self._refresh_selected_plots()
        self._refresh_summary()

    def _refresh_step_status(self):
        peaks = self.peaks()
        texts = {
            "stale": ("needs design", None),
            "missing": ("metabolite is not in the list", "#c62828"),
            "manual": ("hand-edited pulse", None),
        }
        for row, step in enumerate(self.steps()):
            item = self.steps_table.item(row, 3)
            target = step["target"]
            state = self._target_state(target, peaks)
            pulse_state = self._targets.get(target)
            if state in texts:
                text, color = texts[state]
                if state == "manual":
                    text += f" · {pulse_state.pulse.duration_s * 1e3:.1f} ms"
            elif state == "error":
                text, color = "✗ no pulse possible (hover for why)", "#c62828"
            else:
                selective = pulse_state.pulse.design_metadata["metabolite_selective"]
                leak = 100.0 * selective["worst_leak_fraction"]
                text = (
                    f"{pulse_state.pulse.duration_s * 1e3:.1f} ms · leak {leak:.1f} %"
                )
                if selective["worst_leak_peak"]:
                    text += f" ({selective['worst_leak_peak']})"
                if state == "leak":
                    text = "⚠ " + text + " above limit"
                    color = "#c62828"
                else:
                    text = "✓ " + text
                    color = "#2e7d32"
                if selective["passband_min_fraction"] < 0.9:
                    text += (
                        f" · target only {100 * selective['passband_min_fraction']:.0f} %"
                        " at band edge"
                    )
            tooltip = text
            if state == "error":
                tooltip = f"Design failed: {pulse_state.error}"
            if state == "leak":
                tooltip += "\n\n" + self._leak_advice(selective)
            item.setText(text)
            item.setToolTip(tooltip)
            item.setForeground(QBrush(QColor(color)) if color else QBrush())

    def _leak_advice(self, selective):
        rate = float(selective.get("spectral_sampling_rate_hz", 0.0))
        guard = float(selective.get("guard_hz", 0.0))
        return (
            f"The fastest spectral sampling this slice and these gradient limits "
            f"allow folds another peak to within {guard:.0f} Hz of the protected "
            f"bands (sampling {rate:.0f} Hz). Narrow the ± ranges, use a thicker "
            "slice or a shorter subpulse, or raise the gradient limits under "
            "advanced settings."
        )

    def _evaluated_pulses(self):
        """Return ``[(step, pulse)]`` for steps whose pulse can be shown."""
        peaks = self.peaks()
        result = []
        for step in self.steps():
            state = self._targets.get(step["target"])
            usable = self._target_state(step["target"], peaks) in {
                "ok",
                "leak",
                "manual",
                "stale",
            }
            result.append((step, state.pulse if state and usable else None))
        return result

    def _refresh_overview(self):
        peaks = self.peaks()
        evaluated = self._evaluated_pulses()
        table = self.overview_table
        table.clear()
        table.setRowCount(len(evaluated))
        table.setColumnCount(len(peaks))
        table.setHorizontalHeaderLabels([peak.name for peak in peaks])
        table.setVerticalHeaderLabels(
            [
                f"{index + 1}. {step['target']}"
                for index, (step, _) in enumerate(evaluated)
            ]
        )
        tolerance = self.leak_pct.value() / 100.0
        frequencies = np.array([peak.carrier_offset_hz for peak in peaks])
        for row, (step, pulse) in enumerate(evaluated):
            if pulse is None or not peaks:
                continue
            response = _response(
                pulse,
                [pulse.reference_spatial_center_m],
                frequencies,
                flip_angle_deg=step["flip_angle_deg"],
            )
            ideal = np.sin(np.deg2rad(min(step["flip_angle_deg"], 90.0)))
            for column, peak in enumerate(peaks):
                flip = float(response["flip_angle_deg"][0, column])
                relative = float(np.abs(response["mxy"][0, column])) / ideal
                item = QTableWidgetItem(f"{flip:.1f}°")
                item.setTextAlignment(Qt.AlignCenter)
                item.setToolTip(f"{100 * relative:.1f} % of the ideal target signal")
                if peak.name == step["target"]:
                    item.setBackground(QBrush(QColor(46, 125, 50, 90)))
                elif relative > tolerance:
                    item.setBackground(QBrush(QColor(198, 40, 40, 110)))
                table.setItem(row, column, item)
        table.resizeColumnsToContents()

    def _refresh_selected_plots(self):
        for plot in (self.spectrum_plot, self.profile_plot, self.waveform_plot):
            plot.clear()
        step = self._selected_step()
        if step is None:
            return
        state = self._targets.get(step["target"])
        if state is None or state.pulse is None:
            return
        pulse = state.pulse
        flip = step["flip_angle_deg"]
        peaks = self.peaks()
        ideal = np.sin(np.deg2rad(min(flip, 90.0)))
        center = [pulse.reference_spatial_center_m]

        offsets = [peak.carrier_offset_hz for peak in peaks] or [0.0]
        span = max(offsets) - min(offsets)
        margin = max(100.0, 0.25 * span, 2.0 * self.suppression_hz.value())
        frequencies = np.linspace(min(offsets) - margin, max(offsets) + margin, 801)
        response = _response(pulse, center, frequencies, flip_angle_deg=flip)
        self.spectrum_plot.plot(
            frequencies,
            np.abs(response["mxy"][0]) / ideal,
            pen=pg.mkPen("c", width=2),
            name="Excited signal",
        )
        tolerance = self.leak_pct.value() / 100.0
        self.spectrum_plot.addItem(
            pg.InfiniteLine(
                tolerance,
                angle=0,
                pen=pg.mkPen((198, 40, 40), style=Qt.DashLine),
                label=f"allowed leak {100 * tolerance:g} %",
                labelOpts={"position": 0.95},
            )
        )
        for index, peak in enumerate(peaks):
            is_target = peak.name == step["target"]
            half_width = (
                self.passband_hz.value() if is_target else self.suppression_hz.value()
            )
            region = pg.LinearRegionItem(
                (
                    peak.carrier_offset_hz - half_width,
                    peak.carrier_offset_hz + half_width,
                ),
                movable=False,
                brush=(
                    pg.mkBrush(46, 125, 50, 50)
                    if is_target
                    else pg.mkBrush(198, 40, 40, 35)
                ),
            )
            self.spectrum_plot.addItem(region)
            self.spectrum_plot.addItem(
                pg.InfiniteLine(
                    peak.carrier_offset_hz,
                    angle=90,
                    pen=pg.mkPen(_PEAK_COLORS[index % len(_PEAK_COLORS)]),
                    label=peak.name,
                    labelOpts={"position": 0.9, "rotateAxis": (1, 0)},
                )
            )

        thickness = pulse.reference_thickness_m
        positions = pulse.reference_spatial_center_m + np.linspace(
            -1.5 * thickness, 1.5 * thickness, 201
        )
        if peaks:
            profile = _response(
                pulse,
                positions,
                np.array([peak.carrier_offset_hz for peak in peaks]),
                flip_angle_deg=flip,
            )
            for index, peak in enumerate(peaks):
                self.profile_plot.plot(
                    positions * 1000.0,
                    np.abs(profile["mxy"][:, index]),
                    pen=pg.mkPen(
                        _PEAK_COLORS[index % len(_PEAK_COLORS)],
                        width=3 if peak.name == step["target"] else 1,
                    ),
                    name=peak.name,
                )
        render_spectral_spatial_waveforms(
            self.waveform_plot, pulse, flip_angle_deg=flip
        )

    def _refresh_summary(self):
        steps = self.steps()
        if not steps:
            self.status.setText("Add at least one step.")
            return
        stale = self.stale_targets()
        if self.allow_multiple_steps:
            level = self.selection_level.currentData()
            plan = " → ".join(
                f"{step['target']} {step['flip_angle_deg']:g}°"
                + (f" ×{step['repeat_count']}" if step["repeat_count"] > 1 else "")
                for step in steps
            )
            text = f"{plan} (next step per {'frame' if level == 'frame' else 'TR'})"
        else:
            text = f"Excite {steps[0]['target']}"
        if stale:
            text += f" · {len(stale)} pulse(s) still need a design"
        peaks = self.peaks()
        leaking = [
            target
            for target in dict.fromkeys(step["target"] for step in steps)
            if self._target_state(target, peaks) == "leak"
        ]
        if leaking:
            text += (
                f" · ⚠ {', '.join(leaking)}: other peaks leak above the limit "
                "(hover the pulse status for advice)"
            )
        self.status.setText(text)

    # -------------------------------------------------------------- output
    def _accept(self):
        if not self.design_pending():
            peaks = self.peaks()
            reasons = []
            for target in dict.fromkeys(step["target"] for step in self.steps()):
                state = self._target_state(target, peaks)
                if state == "missing":
                    reasons.append(f"{target}: not in the metabolite list")
                elif state == "error":
                    reasons.append(f"{target}: {self._targets[target].error}")
            QMessageBox.warning(
                self,
                "Plan incomplete",
                "Every step needs a designed pulse for a metabolite in the list."
                + "".join(f"\n\n{reason}" for reason in reasons),
            )
            return
        self._result = self._build_output()
        self.accept()

    def _build_output(self):
        library = SpectralSpatialPulseLibrary.from_mapping(self._input_library)
        users: dict[str, set] = {}
        for entry in self._input_schedule.entries:
            if entry.pulse_id in library.pulses:
                users.setdefault(entry.pulse_id, set()).add(
                    _pulse_target(library.get(entry.pulse_id), entry.target_metabolite)
                )
        steps = self.steps()
        pulse_ids = {}
        for target in dict.fromkeys(step["target"] for step in steps):
            state = self._targets[target]
            pulse_id = state.pulse_id
            if pulse_id in library.pulses and users.get(pulse_id, {target}) <= {target}:
                if library.get(pulse_id) is not state.pulse:
                    library.replace(pulse_id, state.pulse)
            else:
                pulse_id = library.add(state.pulse)
            pulse_ids[target] = pulse_id
        entries = []
        for step in steps:
            arguments = {
                "pulse_id": pulse_ids[step["target"]],
                "display_name": step["target"],
                "target_metabolite": step["target"],
                "flip_angle_deg": step["flip_angle_deg"],
                "repeat_count": step["repeat_count"],
                "enabled": step["enabled"],
            }
            if step["entry_id"]:
                arguments["entry_id"] = step["entry_id"]
            entries.append(SpectralSpatialScheduleEntry(**arguments))
        selected_id = entries[0].pulse_id
        if self.allow_multiple_steps:
            schedule = SpectralSpatialSchedule(
                entries,
                cyclic=self.cyclic.isChecked(),
                selection_level=self.selection_level.currentData(),
            )
            referenced = set(pulse_ids.values())
            for pulse_id, pulse in list(library.pulses.items()):
                if (
                    pulse_id not in referenced
                    and "metabolite_selective" in pulse.design_metadata
                ):
                    library.remove(pulse_id)
        elif self._input_schedule.entries:
            # A single-pulse sequence must not discard a FLASH plan.
            schedule = SpectralSpatialSchedule.from_mapping(self._input_schedule)
        else:
            schedule = SpectralSpatialSchedule(entries)
        return library, schedule, selected_id

    def pulse_library(self) -> SpectralSpatialPulseLibrary:
        return None if self._result is None else self._result[0]

    def pulse_schedule(self) -> SpectralSpatialSchedule:
        return None if self._result is None else self._result[1]

    def selected_pulse_id(self) -> str | None:
        return None if self._result is None else self._result[2]


def _selective_peak_names(selective):
    """Peak names stored in ``metabolite_selective`` design metadata."""
    peaks = [selective.get("target_peak") or {"name": selective.get("target", "")}]
    peaks.extend(selective.get("suppressed", ()))
    return [str(peak["name"]) for peak in peaks]
