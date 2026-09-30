"""Target-driven spectral-spatial pulse designer and Bloch validation UI."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, QUrl, Qt
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QCheckBox,
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..metabolite_presets import metabolite_presets
from ..sequence.spectral_spatial import (
    LARSON_TOOLBOX_URL,
    SpectralBand,
    SpectralSpatialPulseLibrary,
    SpectralSpatialPulse,
    SpectralSpatialPeak,
    SpectralSpatialSchedule,
    SpectralSpatialScheduleEntry,
    design_spectral_spatial_pulse,
    fold_frequency_offset_hz,
    simulate_spectral_spatial_profile,
    simulate_spectral_spatial_schedule_preview,
    spectral_spatial_peaks,
    spectral_spatial_schedule_warnings,
    validate_spectral_spatial_schedule,
)
from ..units import NUCLEUS_GAMMA_HZ_PER_T, ppm_to_hz


def render_spectral_spatial_waveforms(plot, pulse, *, flip_angle_deg=None):
    """Render the shared normalized SPSP RF/gradient waveform view."""
    pulse = SpectralSpatialPulse.from_mapping(pulse)
    flip = (
        pulse.reference_flip_angle_deg
        if flip_angle_deg is None
        else float(flip_angle_deg)
    )
    rf, gradient = pulse.scaled_waveforms(flip_angle_deg=flip)
    plot.clear()
    rf_time = (np.arange(rf.size) + 0.5) * pulse.rf_raster_s * 1000.0
    grad_time = (np.arange(gradient.size) + 0.5) * pulse.gradient_raster_s * 1000.0
    rf_scale = np.max(np.abs(rf)) or 1.0
    physical_gradient = gradient / NUCLEUS_GAMMA_HZ_PER_T[pulse.nucleus]
    gradient_scale = np.max(np.abs(physical_gradient)) or 1.0
    plot.plot(
        rf_time,
        np.abs(rf) / rf_scale,
        pen=pg.mkPen("c", width=2),
        name="|B1| / max",
    )
    plot.plot(
        rf_time,
        np.real(rf) / rf_scale,
        pen=pg.mkPen("m"),
        name="Re(B1) / max",
    )
    plot.plot(
        grad_time,
        physical_gradient / gradient_scale,
        pen=pg.mkPen("y"),
        name="G / max",
    )


class SpectralSpatialPulseDesignerDialog(QDialog):
    """Design an SPSP waveform from explicit spatial and spectral targets."""

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
        initial_pulse=None,
        phantom=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Spectral-spatial excitation designer")
        self.resize(1400, 900)
        self.scanner_parameters = scanner_parameters
        self.nucleus = str(nucleus)
        self.field_strength_t = float(field_strength_t)
        self.sequence_reference_ppm = float(sequence_reference_ppm)
        self.carrier_offset_hz = float(carrier_offset_hz)
        self.phantom = phantom
        if not np.isfinite(self.carrier_offset_hz) or not np.isfinite(
            self.sequence_reference_ppm
        ):
            raise ValueError("sequence reference and carrier offset must be finite")
        self._pulse = (
            None
            if initial_pulse is None
            else SpectralSpatialPulse.from_mapping(initial_pulse)
        )
        self._design_dirty = self._pulse is None
        self._last_response = None
        self._target_lines = []
        self._bands_linked_to_peak_table = False
        self._updating_peak_table = False
        self._build_ui(float(target_thickness_m))
        self._restore_initial_pulse()
        if self._pulse is None:
            self._load_default_band()
            self._set_default_simulation_range(float(target_thickness_m), None)
        else:
            self._set_default_simulation_range(
                self._pulse.reference_thickness_m,
                self._pulse.spectral_alias_bandwidth_hz,
            )
            if not self._design_dirty:
                self._show_pulse()
                self._simulate_current_range()

    @staticmethod
    def _double(minimum, maximum, value, suffix="", decimals=3):
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setValue(value)
        widget.setSuffix(suffix)
        return widget

    def _populate_peak_preset_choices(self):
        """Expose nucleus-specific pathways plus an editable common list."""
        presets = tuple(metabolite_presets(self.nucleus))
        by_key = {preset.key: preset for preset in presets}

        def available(keys):
            return tuple(by_key[key] for key in keys if key in by_key)

        specifications = []
        if self.nucleus == "C13":
            specifications.extend(
                (
                    (
                        "Pyruvate metabolism (13C)",
                        "c13_pyruvate",
                        available(
                            (
                                "pyruvate",
                                "lactate",
                                "alanine",
                                "pyruvatehydrate",
                                "bicarbonate",
                            )
                        ),
                        None,
                    ),
                    (
                        "Fumarate / malate (13C)",
                        "c13_fumarate",
                        available(("fumarate", "malate1", "malate4")),
                        None,
                    ),
                    (
                        "alpha-Ketoglutarate / 2HG (13C)",
                        "c13_ketoglutarate",
                        available(
                            (
                                "alpha_ketoglutarate_c1",
                                "alpha_ketoglutarate_hydrate",
                                "2hg_c1",
                            )
                        ),
                        None,
                    ),
                    (
                        "Redox: DHA / vitamin C (13C)",
                        "c13_redox",
                        available(("dehydroascorbate_c1", "vitamin_c_c1")),
                        None,
                    ),
                )
            )
            common_enabled = {
                "pyruvate",
                "lactate",
                "alanine",
                "pyruvatehydrate",
            }
        elif self.nucleus == "H1":
            specifications.extend(
                (
                    (
                        "Brain MRS metabolites (1H)",
                        "h1_brain",
                        available(
                            (
                                "naa",
                                "creatine",
                                "choline",
                                "glutamate",
                                "glutamine",
                                "myoinositol",
                                "lactate",
                            )
                        ),
                        None,
                    ),
                    (
                        "Edited metabolites (1H)",
                        "h1_edited",
                        available(("gaba", "glutathione", "2hg", "lactate")),
                        None,
                    ),
                )
            )
            common_enabled = {"naa", "creatine", "choline", "lactate"}
        elif self.nucleus == "P31":
            specifications.extend(
                (
                    (
                        "Bioenergetics (31P)",
                        "p31_bioenergetics",
                        available(("pcr", "pi", "gamma_atp", "alpha_atp", "beta_atp")),
                        None,
                    ),
                    (
                        "Phospholipid metabolism (31P)",
                        "p31_phospholipids",
                        available(("pe", "pc", "gpe", "gpc", "membrane_phospholipids")),
                        None,
                    ),
                )
            )
            common_enabled = {"pcr", "pi", "gamma_atp", "alpha_atp", "beta_atp"}
        else:
            common_enabled = set()

        if presets:
            specifications.append(
                (
                    f"Common {self.nucleus} metabolites (choose rows)",
                    "common",
                    presets,
                    common_enabled,
                )
            )

        phantom_peaks = ()
        if (
            self.phantom is not None
            and str(getattr(self.phantom, "nucleus", self.nucleus)) == self.nucleus
        ):
            try:
                phantom_peaks = spectral_spatial_peaks(
                    phantom=self.phantom,
                    field_strength_t=self.field_strength_t,
                    nucleus=self.nucleus,
                    sequence_reference_ppm=self.sequence_reference_ppm,
                    rf_carrier_offset_hz=self.carrier_offset_hz,
                )
            except (AttributeError, TypeError, ValueError):
                phantom_peaks = ()
        if phantom_peaks:
            specifications.append(
                ("Current phantom peaks", "phantom", phantom_peaks, None)
            )
        specifications.append(("Custom peak list", "custom", (), None))

        self.object_preset.clear()
        for label, key, components, enabled_keys in specifications:
            self.object_preset.addItem(
                label,
                {
                    "key": key,
                    "components": tuple(components),
                    "enabled_keys": enabled_keys,
                },
            )

    def _append_peak_row(
        self,
        name,
        absolute_ppm,
        relative_weight=1.0,
        *,
        enabled=True,
        metadata=None,
    ):
        row = self.peak_table.rowCount()
        self.peak_table.insertRow(row)
        use_item = QTableWidgetItem()
        use_item.setFlags(
            Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
        )
        use_item.setCheckState(Qt.Checked if enabled else Qt.Unchecked)
        self.peak_table.setItem(row, 0, use_item)

        name_item = QTableWidgetItem(str(name))
        name_item.setToolTip(str(name))
        name_item.setData(Qt.UserRole, dict(metadata or {}))
        self.peak_table.setItem(row, 1, name_item)
        self.peak_table.setItem(row, 2, QTableWidgetItem(f"{float(absolute_ppm):.9g}"))
        for column in (3, 4):
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.peak_table.setItem(row, column, item)
        self.peak_table.setItem(
            row, 5, QTableWidgetItem(f"{float(relative_weight):.9g}")
        )

    def _load_peak_preset(self, *_):
        specification = self.object_preset.currentData()
        if not isinstance(specification, dict):
            return
        self._updating_peak_table = True
        self.peak_table.blockSignals(True)
        self.peak_table.setRowCount(0)
        components = specification.get("components", ())
        enabled_keys = specification.get("enabled_keys")
        weights = (1.0, 0.8, 0.6, 0.45, 0.3, 0.2)
        for index, component in enumerate(components):
            key = getattr(component, "key", None)
            enabled = enabled_keys is None or key in enabled_keys
            metadata = {
                "key": key,
                "t1_s": float(getattr(component, "t1_s", np.inf)),
                "t2_star_s": float(getattr(component, "t2_star_s", np.inf)),
                "source": str(getattr(component, "source", "preset")),
            }
            self._append_peak_row(
                component.name,
                (
                    component.absolute_ppm
                    if isinstance(component, SpectralSpatialPeak)
                    else component.chemical_shift_ppm
                ),
                getattr(
                    component,
                    "relative_weight",
                    weights[min(index, len(weights) - 1)],
                ),
                enabled=enabled,
                metadata=metadata,
            )
        if specification.get("key") == "custom" and not components:
            self._append_peak_row(
                "Peak 1",
                self.sequence_reference_ppm,
                1.0,
                metadata={"source": "custom"},
            )
        self.peak_table.blockSignals(False)
        self._updating_peak_table = False
        self._bands_linked_to_peak_table = False
        self._refresh_peak_offsets()
        self._update_object_description()

    def _add_custom_peak(self):
        self.peak_table.blockSignals(True)
        self._append_peak_row(
            f"Peak {self.peak_table.rowCount() + 1}",
            self.sequence_reference_ppm,
            1.0,
            metadata={"source": "custom"},
        )
        self.peak_table.blockSignals(False)
        self._bands_linked_to_peak_table = False
        self._refresh_peak_offsets()
        self.peak_table.selectRow(self.peak_table.rowCount() - 1)

    def _remove_selected_peaks(self):
        rows = sorted(
            {index.row() for index in self.peak_table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.peak_table.removeRow(row)
        self._bands_linked_to_peak_table = False
        self._refresh_peak_offsets()

    def _peak_table_changed(self, *_):
        if not self._updating_peak_table:
            self._bands_linked_to_peak_table = False
            self._refresh_peak_offsets()

    def _refresh_peak_offsets(self):
        self._updating_peak_table = True
        self.peak_table.blockSignals(True)
        for row in range(self.peak_table.rowCount()):
            try:
                absolute_ppm = float(self.peak_table.item(row, 2).text())
                sequence_hz = float(
                    ppm_to_hz(
                        absolute_ppm - self.sequence_reference_ppm,
                        self.field_strength_t,
                        self.nucleus,
                    )
                )
                carrier_hz = sequence_hz - self.carrier_offset_hz
                values = (f"{sequence_hz:+.3f}", f"{carrier_hz:+.3f}")
            except (AttributeError, TypeError, ValueError):
                values = ("invalid", "invalid")
            for column, value in zip((3, 4), values):
                item = self.peak_table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    self.peak_table.setItem(row, column, item)
                item.setText(value)
        self.peak_table.blockSignals(False)
        self._updating_peak_table = False
        self._update_object_description()

    def _peak_table_state(self, *, strict=True):
        rows = []
        for row in range(self.peak_table.rowCount()):
            use_item = self.peak_table.item(row, 0)
            name_item = self.peak_table.item(row, 1)
            ppm_item = self.peak_table.item(row, 2)
            weight_item = self.peak_table.item(row, 5)
            try:
                name = name_item.text().strip()
                absolute_ppm = float(ppm_item.text())
                relative_weight = float(weight_item.text())
                if not name:
                    raise ValueError("peak name must not be empty")
                if not np.isfinite(absolute_ppm):
                    raise ValueError("position must be finite")
                if not np.isfinite(relative_weight) or relative_weight < 0:
                    raise ValueError("relative flip must be finite and non-negative")
            except (AttributeError, TypeError, ValueError) as exc:
                if strict:
                    raise ValueError(
                        f"invalid metabolite peak in row {row + 1}: {exc}"
                    ) from exc
                continue
            metadata = name_item.data(Qt.UserRole)
            metadata = metadata if isinstance(metadata, dict) else {}

            def lifetime(key):
                value = metadata.get(key, np.inf)
                return np.inf if value is None else float(value)

            rows.append(
                {
                    "enabled": use_item is not None
                    and use_item.checkState() == Qt.Checked,
                    "name": name,
                    "absolute_ppm": absolute_ppm,
                    "relative_weight": relative_weight,
                    "t1_s": lifetime("t1_s"),
                    "t2_star_s": lifetime("t2_star_s"),
                    "source": str(metadata.get("source", "custom")),
                }
            )
        return rows

    def _build_ui(self, target_thickness_m):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QScrollArea.NoFrame)
        controls_scroll.setMinimumWidth(610)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_scroll.setWidget(controls)

        target_group = QGroupBox("1. Desired excitation")
        target_layout = QVBoxLayout(target_group)
        target_form = QFormLayout()
        self.name_edit = QLineEdit("Sequence SPSP")
        self.flip_angle = self._double(0.1, 360.0, 30.0, "°")
        self.spatial_center_mm = self._double(-1000.0, 1000.0, 0.0, " mm")
        self.thickness_mm = self._double(
            0.1, 1000.0, target_thickness_m * 1000.0, " mm"
        )
        self.spatial_transition_mm = self._double(
            0.0, 1000.0, target_thickness_m * 1000.0 / 3.0, " mm"
        )
        self.spatial_center_mm.setToolTip(
            "Reference center used by this preview. Sequence builders override it "
            "with the actual slice or slab position."
        )
        target_form.addRow("Name", self.name_edit)
        target_form.addRow("Reference flip angle", self.flip_angle)
        target_form.addRow("Spatial target center", self.spatial_center_mm)
        target_form.addRow("Spatial target width", self.thickness_mm)
        target_form.addRow("Spatial transition width", self.spatial_transition_mm)
        target_layout.addLayout(target_form)

        reference = QLabel(
            f"Spectral frequencies are offsets from the RF carrier · {self.nucleus} "
            f"at {self.field_strength_t:g} T · absolute sequence reference "
            f"{self.sequence_reference_ppm:g} ppm = 0 Hz · current RF carrier "
            f"{self.carrier_offset_hz:+g} Hz relative to that reference. Pass "
            "bands repeat at the alias spacing."
        )
        reference.setWordWrap(True)
        target_layout.addWidget(reference)

        self.bands = QTableWidget(0, 6)
        self.bands.setHorizontalHeaderLabels(
            [
                "Name",
                "From (Hz)",
                "To (Hz)",
                "Transition (Hz)",
                "Rel. flip",
                "Phase (°)",
            ]
        )
        band_header = self.bands.horizontalHeader()
        band_header.setSectionResizeMode(QHeaderView.Fixed)
        for column, width in enumerate((110, 80, 80, 110, 75, 75)):
            self.bands.setColumnWidth(column, width)
        for column, tip in enumerate(
            (
                "Name stored with the pulse.",
                "Lower pass-band edge relative to the RF carrier.",
                "Upper pass-band edge relative to the RF carrier.",
                "Unconstrained transition width on either side of the pass band.",
                "Flip angle relative to the strongest target band.",
                "Target transverse phase.",
            )
        ):
            self.bands.horizontalHeaderItem(column).setToolTip(tip)
        self.bands.setMinimumHeight(165)
        target_layout.addWidget(self.bands)
        band_buttons = QHBoxLayout()
        add_band = QPushButton("Add pass band")
        remove_band = QPushButton("Remove selected")
        add_band.clicked.connect(lambda: self._append_band(SpectralBand(0.0, 80.0)))
        remove_band.clicked.connect(self._remove_selected_band)
        band_buttons.addWidget(add_band)
        band_buttons.addWidget(remove_band)
        target_layout.addLayout(band_buttons)
        controls_layout.addWidget(target_group)

        preset_group = QGroupBox("Optional metabolite presets")
        preset_layout = QVBoxLayout(preset_group)
        preset_form = QFormLayout()
        self.object_preset = QComboBox()
        self._populate_peak_preset_choices()
        self.object_preset.currentIndexChanged.connect(self._load_peak_preset)
        self.preset_band_width_hz = self._double(0.1, 100000.0, 40.0, " Hz", 1)
        preset_form.addRow("Preset", self.object_preset)
        preset_form.addRow("Band width per peak", self.preset_band_width_hz)
        preset_layout.addLayout(preset_form)
        self.peak_table = QTableWidget(0, 6)
        self.peak_table.setHorizontalHeaderLabels(
            [
                "Use",
                "Peak",
                "ppm",
                "Seq. (Hz)",
                "Carrier (Hz)",
                "Rel. flip",
            ]
        )
        peak_header = self.peak_table.horizontalHeader()
        peak_header.setSectionResizeMode(QHeaderView.Fixed)
        for column, width in enumerate((38, 120, 64, 100, 105, 65)):
            self.peak_table.setColumnWidth(column, width)
        for column, tip in enumerate(
            (
                "Include this peak when creating target bands and profiles.",
                "Editable peak or metabolite name.",
                "Editable absolute chemical-shift position in ppm.",
                "Calculated offset from the sequence reference.",
                "Calculated offset from the RF carrier.",
                "Relative target flip angle and profile weighting.",
            )
        ):
            self.peak_table.horizontalHeaderItem(column).setToolTip(tip)
        self.peak_table.setMinimumHeight(190)
        self.peak_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.peak_table.setToolTip(
            "Select and edit absolute peak positions. Sequence- and carrier-relative "
            "frequencies are recalculated from the current nucleus, B0, sequence "
            "reference and RF carrier."
        )
        self.peak_table.cellChanged.connect(self._peak_table_changed)
        preset_layout.addWidget(self.peak_table)
        peak_buttons = QHBoxLayout()
        add_peak = QPushButton("Add custom peak")
        remove_peak = QPushButton("Remove selected peaks")
        add_peak.clicked.connect(self._add_custom_peak)
        remove_peak.clicked.connect(self._remove_selected_peaks)
        peak_buttons.addWidget(add_peak)
        peak_buttons.addWidget(remove_peak)
        preset_layout.addLayout(peak_buttons)
        self.object_description = QLabel()
        self.object_description.setWordWrap(True)
        preset_layout.addWidget(self.object_description)
        preset_buttons = QHBoxLayout()
        add_preset = QPushButton("Replace target bands with preset")
        add_preset.clicked.connect(self._replace_bands_with_preset)
        self.simulate_button = QPushButton("Plot metabolite spatial profiles")
        self.simulate_button.clicked.connect(self._simulate_metabolite_profiles)
        preset_buttons.addWidget(add_preset)
        preset_buttons.addWidget(self.simulate_button)
        preset_layout.addLayout(preset_buttons)
        controls_layout.addWidget(preset_group)

        simulation_group = QGroupBox("2. Simulation and validation range")
        simulation_form = QFormLayout(simulation_group)
        self.auto_simulation_range = QCheckBox(
            "Automatic from spatial target and one alias period"
        )
        self.auto_simulation_range.setChecked(True)
        self.sim_position_min_mm = self._double(-1000.0, 1000.0, -15.0, " mm")
        self.sim_position_max_mm = self._double(-1000.0, 1000.0, 15.0, " mm")
        self.sim_position_points = QSpinBox()
        self.sim_position_points.setRange(5, 2001)
        self.sim_position_points.setValue(101)
        self.sim_frequency_min_hz = self._double(
            -1_000_000.0, 1_000_000.0, -1500.0, " Hz", 1
        )
        self.sim_frequency_max_hz = self._double(
            -1_000_000.0, 1_000_000.0, 1500.0, " Hz", 1
        )
        self.sim_frequency_points = QSpinBox()
        self.sim_frequency_points.setRange(9, 8193)
        self.sim_frequency_points.setValue(513)
        simulation_form.addRow(self.auto_simulation_range)
        simulation_form.addRow("Position from", self.sim_position_min_mm)
        simulation_form.addRow("Position to", self.sim_position_max_mm)
        simulation_form.addRow("Position samples", self.sim_position_points)
        simulation_form.addRow("Frequency from", self.sim_frequency_min_hz)
        simulation_form.addRow("Frequency to", self.sim_frequency_max_hz)
        simulation_form.addRow("Frequency samples", self.sim_frequency_points)
        controls_layout.addWidget(simulation_group)

        method_group = QGroupBox("Method and attribution")
        method_layout = QVBoxLayout(method_group)
        method_note = QLabel(
            "Design method based on Larson/Kerr spectral-spatial RF pulse "
            "design. This is an independent native-Python flyback "
            "implementation, not a claim of MATLAB-reference equivalence or "
            "endorsement by LarsonLab, Stanford, or UCSF."
        )
        method_note.setWordWrap(True)
        method_note.setToolTip(
            "Reference toolbox: BSD-3-Clause. Methods: Kerr et al., ISMRM "
            "2008; Larson et al., J Magn Reson 194 (2008), 121-127. "
            "See the project SPSP documentation and third-party notices."
        )
        method_layout.addWidget(method_note)
        self.open_larson_reference_button = QPushButton(
            "Open LarsonLab reference toolbox"
        )
        self.open_larson_reference_button.setToolTip(LARSON_TOOLBOX_URL)
        self.open_larson_reference_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(LARSON_TOOLBOX_URL))
        )
        method_layout.addWidget(self.open_larson_reference_button)
        controls_layout.addWidget(method_group)

        self.advanced_group = QGroupBox("3. Advanced pulse construction")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_layout = QVBoxLayout(self.advanced_group)
        advanced_content = QWidget()
        advanced_form = QFormLayout(advanced_content)
        self.spatial_tbw = self._double(1.0, 20.0, 3.0, "", 2)
        minimum_subpulse_s = 3.0 / (
            (float(self.scanner_parameters.max_grad_mtm) / 1000.0)
            * NUCLEUS_GAMMA_HZ_PER_T[self.nucleus]
            * target_thickness_m
        )
        default_subpulse_ms = max(0.7, minimum_subpulse_s * 1.05 * 1000.0)
        self.subpulse_duration_ms = self._double(
            0.01, 100.0, default_subpulse_ms, " ms"
        )
        self.subpulse_count = QSpinBox()
        self.subpulse_count.setRange(3, 101)
        self.subpulse_count.setSingleStep(2)
        self.subpulse_count.setValue(11)
        self.spectral_phase_type = QComboBox()
        self.spectral_phase_type.addItem("Linear phase", "linear")
        self.spectral_phase_type.addItem("Minimum phase", "minimum")
        self.spectral_passband_ripple_pct = self._double(0.001, 99.0, 1.0, " %", 3)
        self.spectral_stopband_ripple_pct = self._double(0.001, 99.0, 1.0, " %", 3)
        self.spatial_passband_ripple_pct = self._double(0.001, 99.0, 1.0, " %", 3)
        self.spatial_stopband_ripple_pct = self._double(0.001, 99.0, 1.0, " %", 3)
        scanner_max_gradient_mtm = float(self.scanner_parameters.max_grad_mtm)
        self.max_gradient = self._double(
            min(0.1, scanner_max_gradient_mtm),
            scanner_max_gradient_mtm,
            scanner_max_gradient_mtm,
            " mT/m",
            1,
        )
        scanner_max_slew_tms = float(self.scanner_parameters.max_slew_tms)
        self.max_slew = self._double(
            min(1.0, scanner_max_slew_tms),
            scanner_max_slew_tms,
            scanner_max_slew_tms,
            " T/m/s",
            1,
        )
        self.max_b1_ut = self._double(0.01, 1000.0, 200.0, " µT", 2)
        self.max_duration_ms = self._double(0.01, 10000.0, 100.0, " ms", 2)
        advanced_form.addRow("Spatial time-bandwidth product", self.spatial_tbw)
        advanced_form.addRow("RF subpulse duration", self.subpulse_duration_ms)
        advanced_form.addRow("Spectral subpulses", self.subpulse_count)
        advanced_form.addRow("Spectral FIR phase", self.spectral_phase_type)
        advanced_form.addRow(
            "Spectral pass-band ripple", self.spectral_passband_ripple_pct
        )
        advanced_form.addRow(
            "Spectral stop-band ripple", self.spectral_stopband_ripple_pct
        )
        advanced_form.addRow(
            "Spatial pass-band ripple", self.spatial_passband_ripple_pct
        )
        advanced_form.addRow(
            "Spatial stop-band ripple", self.spatial_stopband_ripple_pct
        )
        advanced_form.addRow("Maximum gradient", self.max_gradient)
        advanced_form.addRow("Maximum slew", self.max_slew)
        advanced_form.addRow("Maximum B1", self.max_b1_ut)
        advanced_form.addRow("Maximum pulse duration", self.max_duration_ms)
        advanced_layout.addWidget(advanced_content)
        self.advanced_group.toggled.connect(advanced_content.setVisible)
        advanced_content.setVisible(False)
        controls_layout.addWidget(self.advanced_group)

        self.status = QLabel("Define the target, then design and simulate the pulse.")
        self.status.setWordWrap(True)
        controls_layout.addWidget(self.status)
        action_buttons = QHBoxLayout()
        self.design_button = QPushButton("Design pulse and simulate")
        self.design_button.clicked.connect(self._design)
        self.resimulate_button = QPushButton("Simulate selected range")
        self.resimulate_button.clicked.connect(self._simulate_requested_range)
        action_buttons.addWidget(self.design_button)
        action_buttons.addWidget(self.resimulate_button)
        controls_layout.addLayout(action_buttons)
        controls_layout.addStretch()
        splitter.addWidget(controls_scroll)

        self.tabs = QTabWidget()
        self._build_response_tab()
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setLabel("bottom", "Time", units="ms")
        self.waveform_plot.setLabel("left", "Normalized amplitude")
        self.waveform_plot.addLegend()
        self.tabs.addTab(self.waveform_plot, "Waveforms")
        self.object_plot = pg.PlotWidget()
        self.object_plot.setLabel("bottom", "Position", units="mm")
        self.object_plot.setLabel("left", "Relative transverse signal")
        self.object_plot.addLegend()
        self.tabs.addTab(self.object_plot, "Metabolite profiles")
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes((620, 780))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._connect_controls()
        self._load_peak_preset()
        self._update_object_description()
        self._simulation_range_mode_changed()

    def _build_response_tab(self):
        self.response_widget = QWidget()
        layout = QVBoxLayout(self.response_widget)
        self.response_description = QLabel(
            "The heatmap is a Bloch simulation over the selected position and "
            "frequency ranges. Dashed lines mark the requested pass-band edges."
        )
        self.response_description.setWordWrap(True)
        layout.addWidget(self.response_description)

        self.response_graphics = pg.GraphicsLayoutWidget()
        self.response_plot = self.response_graphics.addPlot(row=0, col=0)
        self.response_plot.setLabel(
            "bottom", "Frequency offset from RF carrier", units="Hz"
        )
        self.response_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.response_plot.setLabel("left", "Position", units="mm")
        self.response_plot.showGrid(x=True, y=True, alpha=0.2)
        self.response_image = pg.ImageItem(axisOrder="row-major")
        self.response_plot.addItem(self.response_image)
        self.response_colorbar = pg.ColorBarItem(values=(0.0, 1.0), interactive=False)
        self.response_colorbar.setLabel("right", "Flip angle", units="deg")
        self.response_graphics.addItem(self.response_colorbar, row=0, col=1)
        self.response_colorbar.setImageItem(self.response_image)
        layout.addWidget(self.response_graphics, 3)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Displayed spatial cut at frequency"))
        self.selected_frequency_hz = self._double(
            -1_000_000.0, 1_000_000.0, 0.0, " Hz", 1
        )
        selector_row.addWidget(self.selected_frequency_hz)
        selector_row.addWidget(QLabel("Displayed spectral cut at position"))
        self.selected_position_mm = self._double(-1000.0, 1000.0, 0.0, " mm")
        selector_row.addWidget(self.selected_position_mm)
        layout.addLayout(selector_row)

        cut_splitter = QSplitter(Qt.Horizontal)
        self.spatial_cut_plot = pg.PlotWidget()
        self.spatial_cut_plot.setLabel("bottom", "Position", units="mm")
        self.spatial_cut_plot.setLabel("left", "Flip angle", units="deg")
        self.spectral_cut_plot = pg.PlotWidget()
        self.spectral_cut_plot.setLabel("bottom", "Frequency offset", units="Hz")
        self.spectral_cut_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.spectral_cut_plot.setLabel("left", "Flip angle", units="deg")
        cut_splitter.addWidget(self.spatial_cut_plot)
        cut_splitter.addWidget(self.spectral_cut_plot)
        cut_splitter.setMinimumHeight(220)
        layout.addWidget(cut_splitter, 2)

        self.validation_metrics = QLabel("No response simulated yet.")
        self.validation_metrics.setWordWrap(True)
        layout.addWidget(self.validation_metrics)
        self.tabs.addTab(self.response_widget, "Excitation map")

    def _connect_controls(self):
        for widget in (
            self.flip_angle,
            self.spatial_center_mm,
            self.thickness_mm,
            self.spatial_transition_mm,
            self.spatial_tbw,
            self.subpulse_duration_ms,
            self.subpulse_count,
            self.spectral_passband_ripple_pct,
            self.spectral_stopband_ripple_pct,
            self.spatial_passband_ripple_pct,
            self.spatial_stopband_ripple_pct,
            self.max_gradient,
            self.max_slew,
            self.max_b1_ut,
            self.max_duration_ms,
        ):
            widget.valueChanged.connect(self._mark_design_dirty)
        self.spectral_phase_type.currentIndexChanged.connect(self._mark_design_dirty)
        self.name_edit.textChanged.connect(self._mark_design_dirty)
        self.bands.cellChanged.connect(self._bands_changed)
        self.auto_simulation_range.toggled.connect(self._simulation_range_mode_changed)
        self.selected_frequency_hz.valueChanged.connect(self._update_response_cuts)
        self.selected_position_mm.valueChanged.connect(self._update_response_cuts)

    def _mark_design_dirty(self, *_):
        self._design_dirty = True
        if self._pulse is not None:
            self.status.setText(
                "Target or construction settings changed; redesign before accepting."
            )

    def _bands_changed(self, *_):
        self._bands_linked_to_peak_table = False
        self._mark_design_dirty()

    def _simulation_range_mode_changed(self, *_):
        editable = not self.auto_simulation_range.isChecked()
        for widget in (
            self.sim_position_min_mm,
            self.sim_position_max_mm,
            self.sim_frequency_min_hz,
            self.sim_frequency_max_hz,
        ):
            widget.setEnabled(editable)
        if not editable:
            bandwidth = (
                None if self._pulse is None else self._pulse.spectral_alias_bandwidth_hz
            )
            self._set_default_simulation_range(
                self.thickness_mm.value() / 1000.0, bandwidth
            )

    def _set_default_simulation_range(self, thickness_m, alias_bandwidth_hz):
        center_mm = self.spatial_center_mm.value()
        half_position_mm = 1.5 * float(thickness_m) * 1000.0
        values = (
            (self.sim_position_min_mm, center_mm - half_position_mm),
            (self.sim_position_max_mm, center_mm + half_position_mm),
        )
        if alias_bandwidth_hz is not None:
            values += (
                (self.sim_frequency_min_hz, -float(alias_bandwidth_hz) / 2.0),
                (self.sim_frequency_max_hz, float(alias_bandwidth_hz) / 2.0),
            )
        for widget, value in values:
            previous = widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(previous)

    def _load_default_band(self):
        self.bands.blockSignals(True)
        self.bands.setRowCount(0)
        self._append_band(SpectralBand(0.0, 80.0, 1.0, 0.0, "Reference"))
        self.bands.blockSignals(False)
        self._bands_linked_to_peak_table = False
        self._design_dirty = True

    def _restore_initial_pulse(self):
        if self._pulse is None:
            return
        pulse = self._pulse
        prescription = pulse.design_metadata.get("frequency_prescription", {})
        if not isinstance(prescription, dict):
            prescription = {}
        peak_rows = prescription.get("peak_rows", ())
        if peak_rows:
            preset_key = str(prescription.get("preset_key", "custom"))
            for index in range(self.object_preset.count()):
                data = self.object_preset.itemData(index)
                if isinstance(data, dict) and data.get("key") == preset_key:
                    self.object_preset.blockSignals(True)
                    self.object_preset.setCurrentIndex(index)
                    self.object_preset.blockSignals(False)
                    break
            self._restore_peak_table_state(peak_rows)
        widgets_and_values = (
            (self.name_edit, pulse.name),
            (self.flip_angle, pulse.reference_flip_angle_deg),
            (self.spatial_center_mm, pulse.reference_spatial_center_m * 1000.0),
            (self.thickness_mm, pulse.reference_thickness_m * 1000.0),
            (
                self.spatial_transition_mm,
                pulse.reference_spatial_transition_m * 1000.0,
            ),
            (self.spatial_tbw, pulse.spatial_time_bandwidth_product),
            (self.subpulse_duration_ms, pulse.subpulse_duration_s * 1000.0),
            (
                self.subpulse_count,
                max(3, int(round(pulse.duration_s / pulse.subpulse_period_s))),
            ),
            (self.max_gradient, max(0.1, pulse.max_gradient_t_per_m * 1000.0)),
            (self.max_slew, max(1.0, pulse.max_slew_t_per_m_s)),
            (
                self.spectral_passband_ripple_pct,
                pulse.bands[0].passband_ripple * 100.0,
            ),
            (
                self.spectral_stopband_ripple_pct,
                pulse.bands[0].stopband_ripple * 100.0,
            ),
            (
                self.spatial_passband_ripple_pct,
                pulse.spatial_passband_ripple * 100.0,
            ),
            (
                self.spatial_stopband_ripple_pct,
                pulse.spatial_stopband_ripple * 100.0,
            ),
            (
                self.max_b1_ut,
                200.0 if pulse.max_b1_t is None else pulse.max_b1_t * 1e6,
            ),
            (
                self.max_duration_ms,
                (
                    100.0
                    if pulse.max_duration_s is None
                    else pulse.max_duration_s * 1000.0
                ),
            ),
        )
        for widget, value in widgets_and_values:
            widget.blockSignals(True)
            if isinstance(widget, QLineEdit):
                widget.setText(str(value))
            else:
                widget.setValue(value)
            widget.blockSignals(False)
        phase_index = self.spectral_phase_type.findData(pulse.spectral_phase_type)
        self.spectral_phase_type.setCurrentIndex(max(0, phase_index))
        linked_to_peaks = bool(prescription.get("bands_linked_to_peak_table", False))
        context = prescription.get("reference", {})
        context_changed = False
        if linked_to_peaks and isinstance(context, dict):
            try:
                context_changed = (
                    str(context.get("nucleus")) != self.nucleus
                    or not np.isclose(
                        float(context["field_strength_t"]),
                        self.field_strength_t,
                        atol=1e-12,
                        rtol=0.0,
                    )
                    or not np.isclose(
                        float(context["sequence_reference_ppm"]),
                        self.sequence_reference_ppm,
                        atol=1e-12,
                        rtol=0.0,
                    )
                    or not np.isclose(
                        float(context["rf_carrier_offset_hz"]),
                        self.carrier_offset_hz,
                        atol=1e-9,
                        rtol=0.0,
                    )
                )
            except (KeyError, TypeError, ValueError):
                context_changed = False

        restored_bands = tuple(pulse.bands)
        if context_changed and peak_rows:
            peaks_by_name = {
                peak.name.strip().casefold(): peak for peak in self._resolved_peaks()
            }
            restored_bands = tuple(
                (
                    replace(
                        band,
                        center_hz=peaks_by_name[
                            band.name.strip().casefold()
                        ].carrier_offset_hz,
                    )
                    if band.name.strip().casefold() in peaks_by_name
                    else band
                )
                for band in pulse.bands
            )
        self.bands.blockSignals(True)
        self.bands.setRowCount(0)
        for band in restored_bands:
            self._append_band(band)
        self.bands.blockSignals(False)
        self._bands_linked_to_peak_table = linked_to_peaks
        self._design_dirty = context_changed
        if context_changed:
            self.status.setText(
                "Sequence reference, B0 or RF carrier changed. Peak offsets and "
                "target bands were recalculated; redesign is required before "
                "simulation or acceptance."
            )

    def _restore_peak_table_state(self, rows):
        self._updating_peak_table = True
        self.peak_table.blockSignals(True)
        self.peak_table.setRowCount(0)
        for row in rows:
            if not isinstance(row, dict):
                continue
            self._append_peak_row(
                row.get("name", "Peak"),
                row.get("absolute_ppm", self.sequence_reference_ppm),
                row.get("relative_weight", 1.0),
                enabled=bool(row.get("enabled", True)),
                metadata={
                    "t1_s": row.get("t1_s", np.inf),
                    "t2_star_s": row.get("t2_star_s", np.inf),
                    "source": row.get("source", "custom"),
                },
            )
        self.peak_table.blockSignals(False)
        self._updating_peak_table = False
        self._refresh_peak_offsets()

    def _append_band(self, band: SpectralBand):
        row = self.bands.rowCount()
        self.bands.insertRow(row)
        low = band.center_hz - band.width_hz / 2.0
        high = band.center_hz + band.width_hz / 2.0
        values = (
            band.name,
            f"{low:.9g}",
            f"{high:.9g}",
            f"{band.transition_width_hz:.9g}",
            f"{band.amplitude:.9g}",
            f"{band.phase_deg:.9g}",
        )
        for column, value in enumerate(values):
            self.bands.setItem(row, column, QTableWidgetItem(value))
        self._bands_linked_to_peak_table = False
        self._design_dirty = True

    def _remove_selected_band(self):
        rows = sorted(
            {index.row() for index in self.bands.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.bands.removeRow(row)
        self._bands_linked_to_peak_table = False
        self._mark_design_dirty()

    def _band_values(self):
        result = []
        for row in range(self.bands.rowCount()):
            texts = [
                (
                    ""
                    if self.bands.item(row, column) is None
                    else self.bands.item(row, column).text().strip()
                )
                for column in range(6)
            ]
            try:
                low_hz = float(texts[1])
                high_hz = float(texts[2])
                if high_hz <= low_hz:
                    raise ValueError("upper edge must be greater than lower edge")
                result.append(
                    SpectralBand(
                        center_hz=(low_hz + high_hz) / 2.0,
                        width_hz=high_hz - low_hz,
                        transition_width_hz=float(texts[3]),
                        amplitude=float(texts[4]),
                        phase_deg=float(texts[5]),
                        name=texts[0] or f"Band {row + 1}",
                        passband_ripple=(
                            self.spectral_passband_ripple_pct.value() / 100.0
                        ),
                        stopband_ripple=(
                            self.spectral_stopband_ripple_pct.value() / 100.0
                        ),
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"invalid spectral band in row {row + 1}: {exc}"
                ) from exc
        if not result:
            raise ValueError("add at least one spectral pass band")
        return tuple(result)

    def _design(self):
        try:
            candidate = design_spectral_spatial_pulse(
                bands=self._band_values(),
                flip_angle_deg=self.flip_angle.value(),
                slice_thickness_m=self.thickness_mm.value() / 1000.0,
                spatial_center_m=self.spatial_center_mm.value() / 1000.0,
                spatial_transition_m=self.spatial_transition_mm.value() / 1000.0,
                spatial_time_bandwidth_product=self.spatial_tbw.value(),
                subpulse_duration_s=self.subpulse_duration_ms.value() / 1000.0,
                subpulse_count=self.subpulse_count.value(),
                spectral_phase_type=self.spectral_phase_type.currentData(),
                spatial_passband_ripple=(
                    self.spatial_passband_ripple_pct.value() / 100.0
                ),
                spatial_stopband_ripple=(
                    self.spatial_stopband_ripple_pct.value() / 100.0
                ),
                rf_raster_s=self.scanner_parameters.rf_raster_time_s,
                gradient_raster_s=self.scanner_parameters.grad_raster_time_s,
                max_gradient_t_per_m=self.max_gradient.value() / 1000.0,
                max_slew_t_per_m_s=self.max_slew.value(),
                max_b1_t=self.max_b1_ut.value() * 1e-6,
                max_duration_s=self.max_duration_ms.value() / 1000.0,
                nucleus=self.nucleus,
                name=self.name_edit.text().strip() or "Sequence SPSP",
            )
            metadata = dict(candidate.design_metadata)
            specification = self.object_preset.currentData()
            peak_rows = self._peak_table_state()
            for peak_row in peak_rows:
                for key in ("t1_s", "t2_star_s"):
                    if not np.isfinite(peak_row[key]):
                        peak_row[key] = None
            metadata["frequency_prescription"] = {
                "schema_version": 1,
                "preset_key": (
                    specification.get("key", "custom")
                    if isinstance(specification, dict)
                    else "custom"
                ),
                "bands_linked_to_peak_table": bool(self._bands_linked_to_peak_table),
                "reference": {
                    "nucleus": self.nucleus,
                    "field_strength_t": float(self.field_strength_t),
                    "sequence_reference_ppm": float(self.sequence_reference_ppm),
                    "rf_carrier_offset_hz": float(self.carrier_offset_hz),
                },
                "peak_rows": peak_rows,
            }
            candidate = replace(candidate, design_metadata=metadata)
            self._pulse = candidate
            self._design_dirty = False
            if self.auto_simulation_range.isChecked():
                self._set_default_simulation_range(
                    candidate.reference_thickness_m,
                    candidate.spectral_alias_bandwidth_hz,
                )
            self._show_pulse()
            self._simulate_current_range()
        except Exception as exc:
            self._design_dirty = True
            self.status.setText(f"Design failed: {exc}")
            QMessageBox.critical(self, "SPSP design failed", str(exc))

    def _show_pulse(self):
        pulse = self._pulse
        if pulse is None:
            return
        render_spectral_spatial_waveforms(self.waveform_plot, pulse)
        gamma_hz_per_t = NUCLEUS_GAMMA_HZ_PER_T[pulse.nucleus]
        b1_max_ut = np.max(np.abs(pulse.rf_hz)) / gamma_hz_per_t * 1e6
        specifications_met = pulse.design_metadata.get("metrics", {}).get(
            "all_filter_specifications_met"
        )
        specification_text = (
            ""
            if specifications_met is None
            else " · sampled FIR specs " + ("met" if specifications_met else "not met")
        )
        self.status.setText(
            f"Designed: {pulse.duration_s * 1000:.3f} ms · alias spacing "
            f"{pulse.spectral_alias_bandwidth_hz:.1f} Hz · "
            f"{pulse.spectral_phase_type} spectral FIR · "
            f"B1max {b1_max_ut:.2f} µT · "
            f"Gmax {pulse.max_gradient_t_per_m * 1000:.2f} mT/m · "
            f"slew {pulse.max_slew_t_per_m_s:.1f} T/m/s"
            f"{specification_text}"
        )
        self.simulate_button.setEnabled(True)
        self._update_object_description()

    def _requested_grid(self):
        position_min = self.sim_position_min_mm.value()
        position_max = self.sim_position_max_mm.value()
        frequency_min = self.sim_frequency_min_hz.value()
        frequency_max = self.sim_frequency_max_hz.value()
        if position_max <= position_min:
            raise ValueError("simulation position 'to' must exceed 'from'")
        if frequency_max <= frequency_min:
            raise ValueError("simulation frequency 'to' must exceed 'from'")
        positions = np.linspace(
            position_min / 1000.0,
            position_max / 1000.0,
            self.sim_position_points.value(),
        )
        frequencies = np.linspace(
            frequency_min,
            frequency_max,
            self.sim_frequency_points.value(),
        )
        return positions, frequencies

    def _simulate_requested_range(self):
        if self._pulse is None or self._design_dirty:
            self._design()
            return
        try:
            self._simulate_current_range()
        except Exception as exc:
            QMessageBox.critical(self, "SPSP simulation failed", str(exc))

    def _simulate_current_range(self):
        pulse = self._pulse
        if pulse is None:
            return
        positions, frequencies = self._requested_grid()
        response = simulate_spectral_spatial_profile(
            pulse,
            positions,
            frequencies,
            spatial_offset_m=pulse.reference_spatial_center_m,
        )
        self._last_response = response
        image = response["flip_angle_deg"]
        frequency_step_hz = frequencies[1] - frequencies[0]
        position_step_mm = (positions[1] - positions[0]) * 1000.0
        upper_level = max(
            1.0,
            pulse.reference_flip_angle_deg,
            float(np.nanpercentile(image, 99.5)),
        )
        self.response_image.setImage(
            image,
            autoLevels=False,
            levels=(0.0, upper_level),
        )
        self.response_colorbar.setLevels((0.0, upper_level))
        self.response_image.setRect(
            frequencies[0] - frequency_step_hz / 2.0,
            positions[0] * 1000.0 - position_step_mm / 2.0,
            frequencies.size * frequency_step_hz,
            positions.size * position_step_mm,
        )
        self.response_plot.setXRange(
            frequencies[0] - frequency_step_hz / 2.0,
            frequencies[-1] + frequency_step_hz / 2.0,
            padding=0.0,
        )
        self.response_plot.setYRange(
            positions[0] * 1000.0 - position_step_mm / 2.0,
            positions[-1] * 1000.0 + position_step_mm / 2.0,
            padding=0.0,
        )
        self._show_target_boundaries()
        self._set_cut_selectors(positions, frequencies)
        self._update_response_cuts()
        self._update_validation_metrics()
        self.response_description.setText(
            f"Bloch simulation: {positions.size} positions from "
            f"{positions[0] * 1000:.1f} to {positions[-1] * 1000:.1f} mm and "
            f"{frequencies.size} frequency offsets from {frequencies[0]:.1f} to "
            f"{frequencies[-1]:.1f} Hz. Brightness and the color scale show "
            "flip angle in degrees; dashed lines mark the requested target edges."
        )
        self.tabs.setCurrentWidget(self.response_widget)

    def _show_target_boundaries(self):
        for item in self._target_lines:
            self.response_plot.removeItem(item)
        self._target_lines = []
        pen = pg.mkPen((255, 255, 255, 190), width=1.5, style=Qt.DashLine)
        center_mm = self._pulse.reference_spatial_center_m * 1000.0
        half_width_mm = self._pulse.reference_thickness_m * 500.0
        for position_mm in (center_mm - half_width_mm, center_mm + half_width_mm):
            line = pg.InfiniteLine(pos=position_mm, angle=0, pen=pen, movable=False)
            self.response_plot.addItem(line)
            self._target_lines.append(line)
        bandwidth = self._pulse.spectral_alias_bandwidth_hz
        x_min = self.sim_frequency_min_hz.value()
        x_max = self.sim_frequency_max_hz.value()
        for band in self._pulse.bands:
            first_alias = int(np.floor((x_min - band.center_hz) / bandwidth)) - 1
            last_alias = int(np.ceil((x_max - band.center_hz) / bandwidth)) + 1
            for alias in range(first_alias, last_alias + 1):
                center = band.center_hz + alias * bandwidth
                for edge in (
                    center - band.width_hz / 2.0,
                    center + band.width_hz / 2.0,
                ):
                    if x_min <= edge <= x_max:
                        line = pg.InfiniteLine(
                            pos=edge, angle=90, pen=pen, movable=False
                        )
                        self.response_plot.addItem(line)
                        self._target_lines.append(line)

    def _set_cut_selectors(self, positions, frequencies):
        for selector, minimum, maximum, target in (
            (
                self.selected_position_mm,
                positions[0] * 1000.0,
                positions[-1] * 1000.0,
                self._pulse.reference_spatial_center_m * 1000.0,
            ),
            (
                self.selected_frequency_hz,
                frequencies[0],
                frequencies[-1],
                max(self._pulse.bands, key=lambda band: band.amplitude).center_hz,
            ),
        ):
            previous = selector.blockSignals(True)
            selector.setRange(float(minimum), float(maximum))
            selector.setValue(float(np.clip(target, minimum, maximum)))
            selector.blockSignals(previous)

    def _update_response_cuts(self, *_):
        if self._last_response is None:
            return
        response = self._last_response
        positions_mm = response["positions_m"] * 1000.0
        frequencies = response["frequency_offsets_hz"]
        position_index = int(
            np.argmin(np.abs(positions_mm - self.selected_position_mm.value()))
        )
        frequency_index = int(
            np.argmin(np.abs(frequencies - self.selected_frequency_hz.value()))
        )
        flip = response["flip_angle_deg"]
        self.spatial_cut_plot.clear()
        self.spatial_cut_plot.plot(
            positions_mm,
            flip[:, frequency_index],
            pen=pg.mkPen("c", width=2),
        )
        self.spatial_cut_plot.setTitle(
            f"Spatial profile at {frequencies[frequency_index]:.1f} Hz"
        )
        center_mm = self._pulse.reference_spatial_center_m * 1000.0
        half_width_mm = self._pulse.reference_thickness_m * 500.0
        self.spatial_cut_plot.addItem(
            pg.LinearRegionItem(
                values=(center_mm - half_width_mm, center_mm + half_width_mm),
                brush=pg.mkBrush(80, 160, 255, 35),
                movable=False,
            )
        )
        self.spectral_cut_plot.clear()
        self.spectral_cut_plot.plot(
            frequencies,
            flip[position_index, :],
            pen=pg.mkPen("m", width=2),
        )
        self.spectral_cut_plot.setTitle(
            f"Spectrum at {positions_mm[position_index]:.1f} mm"
        )
        for band in self._pulse.bands:
            self.spectral_cut_plot.addItem(
                pg.LinearRegionItem(
                    values=(
                        band.center_hz - band.width_hz / 2.0,
                        band.center_hz + band.width_hz / 2.0,
                    ),
                    brush=pg.mkBrush(255, 180, 60, 30),
                    movable=False,
                )
            )

    def _update_validation_metrics(self):
        response = self._last_response
        pulse = self._pulse
        positions = response["positions_m"]
        frequencies = response["frequency_offsets_hz"]
        flip = response["flip_angle_deg"]
        spatial_distance = np.abs(positions - pulse.reference_spatial_center_m)
        spatial_pass = spatial_distance <= pulse.reference_thickness_m / 2.0
        spatial_transition = spatial_distance <= (
            pulse.reference_thickness_m / 2.0 + pulse.reference_spatial_transition_m
        )
        spectral_pass = np.zeros(frequencies.size, dtype=bool)
        spectral_transition = np.zeros(frequencies.size, dtype=bool)
        target_flip = np.zeros(frequencies.size, dtype=float)
        strongest = max(band.amplitude for band in pulse.bands) or 1.0
        for band in pulse.bands:
            half_alias = pulse.spectral_alias_bandwidth_hz / 2.0
            distance = np.abs(
                (frequencies - band.center_hz + half_alias)
                % pulse.spectral_alias_bandwidth_hz
                - half_alias
            )
            inside = distance <= band.width_hz / 2.0
            transition = distance <= band.width_hz / 2.0 + band.transition_width_hz
            spectral_pass |= inside
            spectral_transition |= transition
            target_flip[inside] = (
                pulse.reference_flip_angle_deg * band.amplitude / strongest
            )
        pass_mask = spatial_pass[:, None] & spectral_pass[None, :]
        target = np.broadcast_to(target_flip[None, :], flip.shape)
        if np.any(pass_mask):
            pass_error = np.abs(flip[pass_mask] - target[pass_mask])
            rms_error = float(np.sqrt(np.mean(pass_error**2)))
            max_error = float(np.max(pass_error))
        else:
            rms_error = float("nan")
            max_error = float("nan")
        stop_mask = ~(spatial_transition[:, None] & spectral_transition[None, :])
        max_stop = float(np.max(flip[stop_mask])) if np.any(stop_mask) else float("nan")
        gamma_hz_per_t = NUCLEUS_GAMMA_HZ_PER_T[pulse.nucleus]
        b1_max_ut = float(np.max(np.abs(pulse.rf_hz)) / gamma_hz_per_t * 1e6)
        self.validation_metrics.setText(
            f"Target comparison on sampled grid · pass-band RMS error: {rms_error:.2f}° · "
            f"maximum pass-band error: {max_error:.2f}° · maximum stop-band flip: "
            f"{max_stop:.2f}° · duration: {pulse.duration_s * 1000:.3f} ms · "
            f"B1max: {b1_max_ut:.2f} µT · Gmax: "
            f"{pulse.max_gradient_t_per_m * 1000:.2f} mT/m"
        )

    def _update_object_description(self, *_):
        peaks = self._resolved_peaks(strict=False)
        alias_period_hz = (
            None if self._pulse is None else self._pulse.spectral_alias_bandwidth_hz
        )
        names = ", ".join(peak.name for peak in peaks) or "none"
        folded = ""
        if alias_period_hz is not None and peaks:
            folded_values = ", ".join(
                f"{peak.name} {fold_frequency_offset_hz(peak.carrier_offset_hz, alias_period_hz):+.1f} Hz"
                for peak in peaks
            )
            folded = f" Current folded positions: {folded_values}."
        self.object_description.setText(
            f"{len(peaks)} active peak(s) at {self.field_strength_t:g} T "
            f"{self.nucleus}: {names}. Absolute ppm positions are converted using "
            f"sequence reference {self.sequence_reference_ppm:g} ppm and RF carrier "
            f"{self.carrier_offset_hz:+g} Hz.{folded} These are discrete reference "
            "frequencies; use the Excitation map for the continuous frequency sweep."
        )

    def _resolved_peaks(self, *, strict=True):
        rows = self._peak_table_state(strict=strict)
        peaks = []
        for row in rows:
            if not row["enabled"]:
                continue
            sequence_offset_hz = float(
                ppm_to_hz(
                    row["absolute_ppm"] - self.sequence_reference_ppm,
                    self.field_strength_t,
                    self.nucleus,
                )
            )
            peaks.append(
                SpectralSpatialPeak(
                    name=row["name"],
                    absolute_ppm=row["absolute_ppm"],
                    sequence_offset_hz=sequence_offset_hz,
                    carrier_offset_hz=sequence_offset_hz - self.carrier_offset_hz,
                    t1_s=row["t1_s"],
                    t2_star_s=row["t2_star_s"],
                    relative_weight=row["relative_weight"],
                    source=row["source"],
                )
            )
        if strict and not peaks:
            raise ValueError("select at least one metabolite peak")
        return tuple(peaks)

    def _preset_peaks(self):
        """Compatibility view used by existing profile plotting code."""
        peaks = self._resolved_peaks()
        return tuple(
            (
                peak.name,
                peak.carrier_offset_hz,
                peak.relative_weight,
            )
            for peak in peaks
        )

    def _replace_bands_with_preset(self):
        try:
            peaks = self._preset_peaks()
            self.bands.blockSignals(True)
            self.bands.setRowCount(0)
            width_hz = self.preset_band_width_hz.value()
            for name, offset, weight in peaks:
                self._append_band(
                    SpectralBand(
                        center_hz=offset,
                        width_hz=width_hz,
                        amplitude=weight,
                        name=name,
                    )
                )
            self.bands.blockSignals(False)
            self._bands_linked_to_peak_table = True
            self._mark_design_dirty()
        except Exception as exc:
            self.bands.blockSignals(False)
            QMessageBox.critical(self, "Invalid metabolite peaks", str(exc))

    def _simulate_metabolite_profiles(self):
        if self._pulse is None or self._design_dirty:
            self._design()
            if self._pulse is None or self._design_dirty:
                return
        try:
            peaks = self._preset_peaks()
            positions, _ = self._requested_grid()
            frequencies = [peak[1] for peak in peaks]
            response = simulate_spectral_spatial_profile(
                self._pulse,
                positions,
                frequencies,
                spatial_offset_m=self._pulse.reference_spatial_center_m,
            )
            self.object_plot.clear()
            colors = ("c", "m", "y", "g", "r", "w")
            for index, (name, offset, concentration) in enumerate(peaks):
                self.object_plot.plot(
                    positions * 1000.0,
                    concentration * np.abs(response["mxy"][:, index]),
                    pen=pg.mkPen(colors[index % len(colors)], width=2),
                    name=f"{name} ({offset:+.1f} Hz)",
                )
            self.tabs.setCurrentWidget(self.object_plot)
            self.status.setText(
                self.status.text()
                + f" · metabolite profiles: {len(peaks)} discrete peaks at "
                f"{self.field_strength_t:g} T"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Metabolite profile simulation failed", str(exc))

    # Compatibility aliases for callers from the first implementation.
    def _simulate_quick_object(self):
        self._simulate_metabolite_profiles()

    def _show_response_preview(self):
        self._simulate_current_range()

    def _accept_if_valid(self):
        if self._pulse is None or self._design_dirty:
            self._design()
        if self._pulse is not None and not self._design_dirty:
            self.accept()

    def pulse(self):
        return self._pulse


class SpectralSpatialSchedulePreviewDialog(QDialog):
    """Reduced-grid preview of an entire SPSP pulse schedule."""

    def __init__(
        self,
        parent=None,
        *,
        library,
        schedule,
        phantom=None,
        nucleus="C13",
        field_strength_t=3.0,
        sequence_reference_ppm=0.0,
        carrier_offset_hz=0.0,
        target_thickness_m=None,
        pulse_spacing_s=0.0,
        spoiling="ideal",
    ):
        super().__init__(parent)
        self.setWindowTitle("Preview spectral-spatial pulse schedule")
        self.resize(1320, 880)
        self.library = SpectralSpatialPulseLibrary.from_mapping(library)
        self.schedule = SpectralSpatialSchedule.from_mapping(schedule)
        self.nucleus = str(nucleus)
        self.field_strength_t = float(field_strength_t)
        self.sequence_reference_ppm = float(sequence_reference_ppm)
        self.carrier_offset_hz = float(carrier_offset_hz)
        self.phantom = phantom
        expanded_entries = self.schedule.expanded_entries()
        if not expanded_entries:
            raise ValueError(
                "The spectral-spatial schedule has no enabled pulse steps to preview."
            )
        first_pulse = self.library.get(expanded_entries[0].pulse_id)
        self.target_thickness_m = (
            first_pulse.reference_thickness_m
            if target_thickness_m is None
            else float(target_thickness_m)
        )
        half_extent = self.target_thickness_m
        self.positions_m = np.linspace(-half_extent, half_extent, 17)
        self.peaks = spectral_spatial_peaks(
            phantom=phantom,
            presets=metabolite_presets(self.nucleus),
            field_strength_t=self.field_strength_t,
            nucleus=self.nucleus,
            sequence_reference_ppm=self.sequence_reference_ppm,
            rf_carrier_offset_hz=self.carrier_offset_hz,
        )
        self.preview = simulate_spectral_spatial_schedule_preview(
            self.library,
            self.schedule,
            self.peaks,
            self.positions_m,
            thickness_m=self.target_thickness_m,
            pulse_spacing_s=pulse_spacing_s,
            include_relaxation=True,
            spoiling=spoiling,
        )
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(800)
        self.play_timer.timeout.connect(self._advance)
        self._build_ui()
        self._show_step(0)

    def _build_ui(self):
        root = QVBoxLayout(self)
        navigation = QHBoxLayout()
        previous = QPushButton("Previous")
        previous.clicked.connect(lambda: self._show_step(self.step.value() - 2))
        navigation.addWidget(previous)
        self.step = QSpinBox()
        self.step.setRange(1, len(self.preview["steps"]))
        self.step.valueChanged.connect(lambda value: self._show_step(value - 1))
        navigation.addWidget(QLabel("Schedule step"))
        navigation.addWidget(self.step)
        next_button = QPushButton("Next")
        next_button.clicked.connect(lambda: self._show_step(self.step.value()))
        navigation.addWidget(next_button)
        self.auto_play = QCheckBox("Automatic playback")
        self.auto_play.toggled.connect(
            lambda enabled: (
                self.play_timer.start() if enabled else self.play_timer.stop()
            )
        )
        navigation.addWidget(self.auto_play)
        self.fixed_scales = QCheckBox("Use common fixed scales")
        self.fixed_scales.setChecked(True)
        self.fixed_scales.toggled.connect(
            lambda *_: self._show_step(self.step.value() - 1)
        )
        navigation.addWidget(self.fixed_scales)
        navigation.addStretch()
        root.addLayout(navigation)

        self.step_description = QLabel()
        self.step_description.setWordWrap(True)
        root.addWidget(self.step_description)
        self.tabs = QTabWidget()
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setLabel("bottom", "Time", units="ms")
        self.waveform_plot.setLabel("left", "Normalized amplitude")
        self.waveform_plot.addLegend()
        self.tabs.addTab(self.waveform_plot, "Waveforms")

        self.map_plot = pg.PlotWidget()
        self.map_plot.setLabel("bottom", "Frequency offset from RF carrier", units="Hz")
        self.map_plot.setLabel("left", "Position", units="mm")
        self.map_image = pg.ImageItem(axisOrder="row-major")
        self.map_plot.addItem(self.map_image)
        self.tabs.addTab(self.map_plot, "Excitation map")

        self.profile_plot = pg.PlotWidget()
        self.profile_plot.setLabel("bottom", "Position", units="mm")
        self.profile_plot.setLabel("left", "|Mxy|")
        self.profile_plot.addLegend()
        self.tabs.addTab(self.profile_plot, "Metabolite profiles")

        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        self.overview_table = QTableWidget(len(self.preview["steps"]), len(self.peaks))
        self.overview_table.setHorizontalHeaderLabels(
            [peak.name for peak in self.peaks]
        )
        self.overview_table.setVerticalHeaderLabels(
            [str(index + 1) for index in range(len(self.preview["steps"]))]
        )
        center = int(np.argmin(np.abs(self.positions_m)))
        isolated = np.asarray(self.preview["isolated_mxy"])
        for row in range(isolated.shape[0]):
            for column in range(isolated.shape[2]):
                self.overview_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(f"{abs(isolated[row, center, column]):.5f}"),
                )
        overview_layout.addWidget(
            QLabel("Schedule step × metabolite: isolated |Mxy| at the slice centre")
        )
        overview_layout.addWidget(self.overview_table)
        self.tabs.addTab(overview, "Schedule overview")

        self.cumulative_plot = pg.PlotWidget()
        self.cumulative_plot.setLabel("bottom", "Expanded schedule step")
        self.cumulative_plot.setLabel("left", "Magnetization")
        self.cumulative_plot.addLegend()
        steps = np.arange(1, len(self.preview["steps"]) + 1)
        cumulative_mxy = np.asarray(self.preview["cumulative_mxy"])
        cumulative_mz = np.asarray(self.preview["cumulative_mz"])
        colors = ("c", "m", "y", "g", "r", "w")
        for index, peak in enumerate(self.peaks):
            color = colors[index % len(colors)]
            self.cumulative_plot.plot(
                steps,
                np.abs(cumulative_mxy[:, center, index]),
                pen=pg.mkPen(color, width=2),
                symbol="o",
                name=f"{peak.name} |Mxy|",
            )
            self.cumulative_plot.plot(
                steps,
                cumulative_mz[:, center, index],
                pen=pg.mkPen(color, style=Qt.DashLine),
                name=f"{peak.name} Mz",
            )
        self.tabs.addTab(self.cumulative_plot, "Cumulative magnetization")
        root.addWidget(self.tabs, 1)

        source = (
            "current phantom" if self.peaks[0].source == "phantom" else "preset peaks"
        )
        warnings = spectral_spatial_schedule_warnings(
            self.library,
            self.schedule,
            self.peaks,
            phantom_sequence_reference_ppm=(
                None
                if self.phantom is None
                else getattr(self.phantom, "metadata", {}).get("sequence_reference_ppm")
            ),
            sequence_reference_ppm=self.sequence_reference_ppm,
        )
        self.warning_label = QLabel(
            f"Peak source: {source}. "
            + (
                " Warnings: " + " ".join(warnings)
                if warnings
                else "No target/reference warning."
            )
        )
        self.warning_label.setWordWrap(True)
        root.addWidget(self.warning_label)
        assumptions = QLabel(
            "Fast-preview assumptions: " + self.preview["approximation"]
        )
        assumptions.setWordWrap(True)
        root.addWidget(assumptions)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _advance(self):
        self._show_step(self.step.value())

    def _show_step(self, index):
        count = len(self.preview["steps"])
        index = int(index) % count
        previous = self.step.blockSignals(True)
        self.step.setValue(index + 1)
        self.step.blockSignals(previous)
        metadata = self.preview["steps"][index]
        pulse = self.library.get(metadata["pulse_id"])
        render_spectral_spatial_waveforms(
            self.waveform_plot, pulse, flip_angle_deg=metadata["flip_angle_deg"]
        )
        alias = pulse.spectral_alias_bandwidth_hz
        frequencies = np.linspace(-alias / 2.0, alias / 2.0, 81)
        response = simulate_spectral_spatial_profile(
            pulse,
            self.positions_m,
            frequencies,
            flip_angle_deg=metadata["flip_angle_deg"],
            thickness_m=self.target_thickness_m,
        )
        image = response["flip_angle_deg"]
        upper = (
            max(float(step["flip_angle_deg"]) for step in self.preview["steps"])
            if self.fixed_scales.isChecked()
            else max(1.0, float(np.nanmax(image)))
        )
        self.map_image.setImage(image, autoLevels=False, levels=(0.0, upper))
        self.map_image.setRect(
            frequencies[0],
            self.positions_m[0] * 1000.0,
            frequencies[-1] - frequencies[0],
            (self.positions_m[-1] - self.positions_m[0]) * 1000.0,
        )
        self.profile_plot.clear()
        isolated = np.asarray(self.preview["isolated_mxy"])[index]
        colors = ("c", "m", "y", "g", "r", "w")
        folded = np.asarray(self.preview["folded_offsets_hz"])[index]
        peak_descriptions = []
        for peak_index, peak in enumerate(self.peaks):
            self.profile_plot.plot(
                self.positions_m * 1000.0,
                np.abs(isolated[:, peak_index]),
                pen=pg.mkPen(colors[peak_index % len(colors)], width=2),
                name=peak.name,
            )
            peak_descriptions.append(
                f"{peak.name}: {peak.absolute_ppm:g} ppm; "
                f"{peak.sequence_offset_hz:+.2f} Hz sequence; "
                f"{peak.carrier_offset_hz:+.2f} Hz carrier; "
                f"{folded[peak_index]:+.2f} Hz folded"
            )
        if self.fixed_scales.isChecked():
            self.profile_plot.setYRange(0.0, 1.0, padding=0.02)
        self.step_description.setText(
            f"Step {index + 1}/{count}: {metadata['display_name']} · pulse ID "
            f"{metadata['pulse_id']} · target {metadata['target_metabolite'] or 'unspecified'} · "
            f"{metadata['flip_angle_deg']:.4g}° · amplitude scale "
            f"{metadata['amplitude_scale']:.4g}. " + "; ".join(peak_descriptions)
        )


class SpectralSpatialScheduleDialog(QDialog):
    """Manage the SPSP pulse library and ordered schedule independently."""

    def __init__(
        self,
        parent=None,
        *,
        library=None,
        schedule=None,
        scanner_parameters,
        nucleus="C13",
        field_strength_t=3.0,
        sequence_reference_ppm=0.0,
        carrier_offset_hz=0.0,
        target_thickness_m=10e-3,
        phantom=None,
        pulse_spacing_s=0.0,
        spoiling="ideal",
    ):
        super().__init__(parent)
        self.setWindowTitle("Spectral-spatial pulse library and schedule")
        self.resize(1180, 760)
        self._library = SpectralSpatialPulseLibrary.from_mapping(library)
        self._schedule = SpectralSpatialSchedule.from_mapping(schedule)
        self.scanner_parameters = scanner_parameters
        self.nucleus = str(nucleus)
        self.field_strength_t = float(field_strength_t)
        self.sequence_reference_ppm = float(sequence_reference_ppm)
        self.carrier_offset_hz = float(carrier_offset_hz)
        self.target_thickness_m = float(target_thickness_m)
        self.phantom = phantom
        self.pulse_spacing_s = float(pulse_spacing_s)
        self.spoiling = str(spoiling)
        self._build_ui()
        self._refresh_library()
        self._refresh_schedule()

    def _build_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter, 1)

        library_widget = QWidget()
        library_layout = QVBoxLayout(library_widget)
        library_layout.addWidget(
            QLabel(
                "Pulse library (waveforms are designed once and referenced by stable ID)"
            )
        )
        self.library_table = QTableWidget(0, 4)
        self.library_table.setHorizontalHeaderLabels(
            ["Pulse ID", "Name", "Nucleus", "Duration (ms)"]
        )
        self.library_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.library_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.library_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        library_layout.addWidget(self.library_table)
        buttons = QHBoxLayout()
        for text, callback in (
            ("New", self._new_pulse),
            ("Duplicate", self._duplicate_pulse),
            ("Edit", self._edit_pulse),
            ("Delete", self._delete_pulse),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        buttons.addStretch()
        library_layout.addLayout(buttons)
        splitter.addWidget(library_widget)

        schedule_widget = QWidget()
        schedule_layout = QVBoxLayout(schedule_widget)
        schedule_layout.addWidget(
            QLabel(
                "Ordered schedule. Frame mode applies one expanded step to a complete FLASH image; "
                "TR mode is advanced use and mixes contrasts within k-space."
            )
        )
        mode_row = QHBoxLayout()
        self.selection_level = QComboBox()
        self.selection_level.addItem("Complete image frame", "frame")
        self.selection_level.addItem("Every TR/RF event (contrast mixing)", "tr")
        self.selection_level.setCurrentIndex(
            max(0, self.selection_level.findData(self._schedule.selection_level))
        )
        self.cyclic = QCheckBox("Repeat schedule cyclically")
        self.cyclic.setChecked(self._schedule.cyclic)
        mode_row.addWidget(QLabel("Advance schedule per"))
        mode_row.addWidget(self.selection_level)
        mode_row.addWidget(self.cyclic)
        mode_row.addStretch()
        schedule_layout.addLayout(mode_row)
        self.schedule_table = QTableWidget(0, 7)
        self.schedule_table.setHorizontalHeaderLabels(
            [
                "On",
                "Display name",
                "Target metabolite",
                "Pulse",
                "Scale by",
                "Value",
                "Repeats",
            ]
        )
        self.schedule_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.schedule_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.schedule_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.schedule_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        schedule_layout.addWidget(self.schedule_table)
        schedule_buttons = QHBoxLayout()
        for text, callback in (
            ("Add selected pulse", self._add_schedule_entry),
            ("Duplicate entry", self._duplicate_schedule_entry),
            ("Delete entry", self._delete_schedule_entry),
            ("Move up", lambda: self._move_schedule_entry(-1)),
            ("Move down", lambda: self._move_schedule_entry(1)),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            schedule_buttons.addWidget(button)
        preview = QPushButton("Preview pulse schedule")
        preview.clicked.connect(self._preview)
        schedule_buttons.addStretch()
        schedule_buttons.addWidget(preview)
        schedule_layout.addLayout(schedule_buttons)
        splitter.addWidget(schedule_widget)
        splitter.setSizes((280, 440))

        self.status = QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dialog_buttons.accepted.connect(self._accept)
        dialog_buttons.rejected.connect(self.reject)
        root.addWidget(dialog_buttons)

    def _selected_library_id(self):
        row = self.library_table.currentRow()
        if row < 0:
            return None
        return self.library_table.item(row, 0).text()

    def _refresh_library(self, select_id=None):
        self.library_table.setRowCount(0)
        selected_row = -1
        for row, (pulse_id, pulse) in enumerate(self._library.pulses.items()):
            self.library_table.insertRow(row)
            for column, text in enumerate(
                (
                    pulse_id,
                    pulse.name,
                    pulse.nucleus,
                    f"{pulse.duration_s * 1000.0:.6g}",
                )
            ):
                item = QTableWidgetItem(str(text))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.library_table.setItem(row, column, item)
            if pulse_id == select_id:
                selected_row = row
        if selected_row < 0 and self.library_table.rowCount():
            selected_row = 0
        if selected_row >= 0:
            self.library_table.selectRow(selected_row)

    def _refresh_schedule(self, select_row=None):
        self.schedule_table.setRowCount(0)
        for row, entry in enumerate(self._schedule.entries):
            self.schedule_table.insertRow(row)
            enabled = QTableWidgetItem()
            enabled.setFlags(enabled.flags() | Qt.ItemIsUserCheckable)
            enabled.setCheckState(Qt.Checked if entry.enabled else Qt.Unchecked)
            enabled.setData(Qt.UserRole, entry.entry_id)
            self.schedule_table.setItem(row, 0, enabled)
            self.schedule_table.setItem(row, 1, QTableWidgetItem(entry.display_name))
            self.schedule_table.setItem(
                row, 2, QTableWidgetItem(entry.target_metabolite)
            )
            pulse = QComboBox()
            for pulse_id, value in self._library.pulses.items():
                pulse.addItem(value.name, pulse_id)
            pulse.setCurrentIndex(max(0, pulse.findData(entry.pulse_id)))
            self.schedule_table.setCellWidget(row, 3, pulse)
            mode = QComboBox()
            mode.addItem("Flip angle (deg)", "flip_angle")
            mode.addItem("RF amplitude scale", "amplitude")
            mode.setCurrentIndex(max(0, mode.findData(entry.scale_mode)))
            self.schedule_table.setCellWidget(row, 4, mode)
            value = (
                entry.flip_angle_deg
                if entry.flip_angle_deg is not None
                else entry.amplitude_scale
            )
            self.schedule_table.setItem(row, 5, QTableWidgetItem(f"{value:.9g}"))
            self.schedule_table.setItem(
                row, 6, QTableWidgetItem(str(entry.repeat_count))
            )
        if self.schedule_table.rowCount():
            self.schedule_table.selectRow(
                min(self.schedule_table.rowCount() - 1, max(0, select_row or 0))
            )

    def _commit_schedule_table(self):
        entries = []
        for row in range(self.schedule_table.rowCount()):
            pulse = self.schedule_table.cellWidget(row, 3)
            mode = self.schedule_table.cellWidget(row, 4).currentData()
            value = float(self.schedule_table.item(row, 5).text())
            entries.append(
                SpectralSpatialScheduleEntry(
                    entry_id=str(self.schedule_table.item(row, 0).data(Qt.UserRole)),
                    pulse_id=str(pulse.currentData()),
                    display_name=self.schedule_table.item(row, 1).text().strip(),
                    target_metabolite=self.schedule_table.item(row, 2).text().strip(),
                    amplitude_scale=value if mode == "amplitude" else 1.0,
                    flip_angle_deg=value if mode == "flip_angle" else None,
                    enabled=self.schedule_table.item(row, 0).checkState() == Qt.Checked,
                    repeat_count=int(self.schedule_table.item(row, 6).text()),
                )
            )
        self._schedule = SpectralSpatialSchedule(
            entries,
            cyclic=self.cyclic.isChecked(),
            selection_level=str(self.selection_level.currentData()),
        )

    def _open_designer(self, pulse=None):
        dialog = SpectralSpatialPulseDesignerDialog(
            self,
            scanner_parameters=self.scanner_parameters,
            nucleus=self.nucleus,
            field_strength_t=self.field_strength_t,
            sequence_reference_ppm=self.sequence_reference_ppm,
            carrier_offset_hz=self.carrier_offset_hz,
            target_thickness_m=self.target_thickness_m,
            initial_pulse=pulse,
            phantom=self.phantom,
        )
        return dialog.pulse() if dialog.exec_() == QDialog.Accepted else None

    def _new_pulse(self):
        pulse = self._open_designer()
        if pulse is not None:
            pulse_id = self._library.add(pulse)
            self._refresh_library(pulse_id)
            self._refresh_schedule(self.schedule_table.currentRow())

    def _duplicate_pulse(self):
        pulse_id = self._selected_library_id()
        if pulse_id is not None:
            duplicate_id = self._library.duplicate(pulse_id)
            self._refresh_library(duplicate_id)
            self._refresh_schedule(self.schedule_table.currentRow())

    def _edit_pulse(self):
        pulse_id = self._selected_library_id()
        if pulse_id is None:
            return
        pulse = self._open_designer(self._library.get(pulse_id))
        if pulse is not None:
            self._library.replace(pulse_id, pulse)
            self._refresh_library(pulse_id)
            self._refresh_schedule(self.schedule_table.currentRow())

    def _delete_pulse(self):
        pulse_id = self._selected_library_id()
        if pulse_id is None:
            return
        self._commit_schedule_table()
        self._schedule.entries = [
            entry for entry in self._schedule.entries if entry.pulse_id != pulse_id
        ]
        self._library.remove(pulse_id)
        self._refresh_library()
        self._refresh_schedule()

    def _add_schedule_entry(self):
        self._commit_schedule_table()
        pulse_id = self._selected_library_id()
        if pulse_id is None:
            return
        pulse = self._library.get(pulse_id)
        self._schedule.add(
            SpectralSpatialScheduleEntry(
                pulse_id=pulse_id,
                display_name=pulse.name,
                target_metabolite=pulse.bands[0].name if pulse.bands else "",
                flip_angle_deg=pulse.reference_flip_angle_deg,
            )
        )
        self._refresh_schedule(len(self._schedule.entries) - 1)

    def _duplicate_schedule_entry(self):
        self._commit_schedule_table()
        row = self.schedule_table.currentRow()
        if row >= 0:
            self._schedule.duplicate(row)
            self._refresh_schedule(row + 1)

    def _delete_schedule_entry(self):
        self._commit_schedule_table()
        row = self.schedule_table.currentRow()
        if row >= 0:
            self._schedule.remove(row)
            self._refresh_schedule(max(0, row - 1))

    def _move_schedule_entry(self, direction):
        self._commit_schedule_table()
        row = self.schedule_table.currentRow()
        target = row + int(direction)
        if 0 <= row < len(self._schedule.entries) and 0 <= target < len(
            self._schedule.entries
        ):
            self._schedule.move(row, target)
            self._refresh_schedule(target)

    def _preview(self):
        try:
            self._commit_schedule_table()
            validate_spectral_spatial_schedule(
                self._library,
                self._schedule,
                nucleus=self.nucleus,
                thickness_m=self.target_thickness_m,
                scanner_parameters=self.scanner_parameters,
            )
            dialog = SpectralSpatialSchedulePreviewDialog(
                self,
                library=self._library,
                schedule=self._schedule,
                phantom=self.phantom,
                nucleus=self.nucleus,
                field_strength_t=self.field_strength_t,
                sequence_reference_ppm=self.sequence_reference_ppm,
                carrier_offset_hz=self.carrier_offset_hz,
                target_thickness_m=self.target_thickness_m,
                pulse_spacing_s=self.pulse_spacing_s,
                spoiling=self.spoiling,
            )
            dialog.exec_()
        except Exception as exc:
            QMessageBox.critical(self, "Schedule preview failed", str(exc))

    def _accept(self):
        try:
            self._commit_schedule_table()
            warnings = validate_spectral_spatial_schedule(
                self._library,
                self._schedule,
                nucleus=self.nucleus,
                thickness_m=self.target_thickness_m,
                scanner_parameters=self.scanner_parameters,
            )
            self.status.setText(" ".join(warnings))
            if warnings:
                QMessageBox.warning(
                    self,
                    "SPSP design warnings",
                    "\n\n".join(warnings)
                    + "\n\nThe schedule can be saved, but the requested stop-band "
                    "suppression is not guaranteed.",
                )
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid SPSP schedule", str(exc))

    def pulse_library(self):
        return SpectralSpatialPulseLibrary.from_mapping(self._library)

    def pulse_schedule(self):
        return SpectralSpatialSchedule.from_mapping(self._schedule)
