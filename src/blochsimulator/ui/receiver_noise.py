"""Single-channel receiver noise settings for the sequence workspace."""

from decimal import Decimal
import math
from pathlib import Path
import sys

from PyQt5.QtCore import QLocale, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ..sequence.measured_fid import (
    check_noise_scan,
    load_measured_scan,
    measured_calibration_inputs,
)
from ..sequence.receiver import ReceiverCalibration, ReceiverNoise
from ..sequence.reference_scan import ReferenceScan, reference_signal_efficiency
from ..sequence.snr_budget import format_amount

KELVIN_AT_0_CELSIUS = 273.15
# Scanners quote e.g. 7.05 T for a nominal 7 T magnet.
MEASURED_FIELD_TOLERANCE = 0.02
MEASURED_DATA_FILTER = (
    "Measured data (fid rawdata.job0 acqp method *.npy *.txt *.csv *.dat);;"
    "ParaVision raw data (fid rawdata.job0 acqp method);;"
    "Arrays (*.npy *.txt *.csv *.dat);;All files (*)"
)


class ScientificSpinBox(QDoubleSpinBox):
    """Non-negative value with native arrows, scientific notation and decimal steps."""

    def __init__(self, value, parent=None):
        super().__init__(parent)
        # Fixed display decimals would otherwise round physical signal noise
        # (often much smaller than 1e-6) to zero. Format the value separately.
        self.setDecimals(323)
        self.setRange(0.0, sys.float_info.max)
        self.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        locale = QLocale.c()
        locale.setNumberOptions(QLocale.RejectGroupSeparator)
        self.setLocale(locale)
        self._number_validator = QDoubleValidator(0.0, sys.float_info.max, 323, self)
        self._number_validator.setLocale(locale)
        self._number_validator.setNotation(QDoubleValidator.ScientificNotation)
        self.setValue(value)

    def textFromValue(self, value):
        text = repr(float(value))
        # Shortest round-trip text; whole numbers such as an SNR of 100 drop ".0".
        return text[:-2] if text.endswith(".0") else text

    def valueFromText(self, text):
        try:
            value = float(text)
        except ValueError:
            return self.value()
        return value if math.isfinite(value) and value >= 0 else self.value()

    def validate(self, text, position):
        validator = getattr(self, "_number_validator", None)
        if validator is None:
            return super().validate(text, position)
        return validator.validate(text, position)

    def setText(self, text):
        """Restore legacy text-based noise inputs without changing their value."""
        self.lineEdit().setText(str(text))
        if self.hasAcceptableInput():
            self.interpretText()

    def stepBy(self, steps):
        self.interpretText()
        current = Decimal(str(self.value()))
        if current == 0:
            step = Decimal("0.001")
        else:
            power = current.adjusted()
            # Match Qt's adaptive step at a decade boundary, so Down then Up
            # returns to the initial value (e.g. 0.01 -> 0.0099 -> 0.01).
            if steps < 0 and current == Decimal(10) ** power:
                power -= 1
            step = Decimal(10) ** (power - 1)
        updated = current + steps * step
        updated = min(max(updated, Decimal(0)), Decimal(str(self.maximum())))
        self.setValue(float(updated))


def _spin(minimum, maximum, value, decimals, suffix, tooltip):
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setValue(value)
    box.setSuffix(suffix)
    box.setToolTip(tooltip)
    return box


class ReceiverNoiseWidget(QGroupBox):
    calibration_requested = pyqtSignal()
    apply_requested = pyqtSignal()
    mode_changed = pyqtSignal()
    preview_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Receiver / Noise", parent)
        self.calibration = None
        self._result_available = False
        self._measured_fid = None
        self._measured_noise = None
        # Returns the (transmit, receive) B1 fields the simulated object uses;
        # the main window points it at the B1 workspace.
        self.coil_fields = None
        self._reference_error = False
        form = QFormLayout(self)
        self.mode = QComboBox()
        self.mode.addItem("Off", "off")
        self.mode.addItem("Relative SNR", "relative")
        self.mode.addItem("Calibrated from phantom FID", "calibrated")
        form.addRow("Noise", self.mode)
        self.snr = ScientificSpinBox(100.0)
        self.snr.setToolTip(
            "Single-scan SNR of the ideal 90° FID of the whole simulated object "
            "(initial magnetization plus any delivered bolus dose) at the "
            "reference bandwidth: FID amplitude at t = 0 / noise std per real or "
            "imaginary channel. Scientific notation is accepted; arrows and "
            "Up / Down keys step relative to the current value."
        )
        self.amplitude = QLineEdit("1")
        self.reference_std = ScientificSpinBox(0.01)
        self.averages = QSpinBox()
        self.averages.setRange(1, 2**20)
        self.averages.setToolTip(
            "Number of averages of the reference scans; noise must come from "
            "a 0° scan with the same averages, gain and bandwidth."
        )
        self.bandwidth_khz = _spin(
            0.001,
            1e4,
            10.0,
            3,
            " kHz",
            "Relative SNR: bandwidth at which the SNR is defined. Calibration: "
            "receiver bandwidth of the reference scans (1 / dwell, e.g. Bruker "
            "SW_h). Noise std scales with √(bandwidth) for other ADC dwells.",
        )
        self.flip_angle = _spin(
            0.1, 179.9, 90.0, 1, " °", "Flip angle of the reference FID scan."
        )
        self.concentration = QLineEdit()
        self.volume = QLineEdit()
        self.concentration.setPlaceholderText("mM, e.g. 2000")
        self.volume.setPlaceholderText("ml, e.g. 50")
        self.volume.setToolTip("Phantom volume inside the sensitive coil region.")
        self.enrichment = _spin(
            0.01,
            100.0,
            100.0,
            2,
            " %",
            "Isotope fraction of the observed nucleus, e.g. 1.1 % natural 13C.",
        )
        self.nuclei_per_molecule = QSpinBox()
        self.nuclei_per_molecule.setRange(1, 64)
        self.nuclei_per_molecule.setToolTip(
            "Labelled positions per molecule that contribute to the measured peak."
        )
        self.reference_temperature = _spin(
            -273.0,
            200.0,
            20.0,
            1,
            " °C",
            "Phantom temperature; sets its thermal polarization.",
        )
        self.target_temperature = _spin(
            -273.0,
            200.0,
            37.0,
            1,
            " °C",
            "Fallback object temperature for phantoms without temperature metadata. "
            "A temperature saved by the Phantom Designer takes precedence for "
            "receiver noise and physical signal units.",
        )
        self.noise_factor = _spin(
            0.01,
            100.0,
            1.0,
            3,
            "",
            "Noise std multiplier for the simulated object, e.g. extra coil "
            "loading in vivo compared with the phantom.",
        )
        self.amplitude.setToolTip(
            "Measured FID amplitude immediately after excitation, before decay. "
            "Use the first valid FID point only if acquisition delay is negligible. "
            "Amplitude and noise must have the same scanner scaling. 'Load FID…' "
            "extrapolates it to the acquisition start."
        )
        sigma_help = (
            "Standard deviation of the real or imaginary noise channel. "
            "Do not use magnitude standard deviation or complex RMS. "
            "Scientific notation is accepted."
            " Use the arrows or Up / Down keys for steps relative to the current value."
        )
        self.reference_std.setToolTip(sigma_help + " In measured scanner units.")
        form.addRow("FID SNR (90°, single scan)", self.snr)
        form.addRow("Reference bandwidth", self.bandwidth_khz)
        # Qt 5 keeps the spacing of hidden form rows, so the calibration rows
        # live in their own form that is hidden as a whole.
        self.calibration_box = QWidget()
        calibration_form = QFormLayout(self.calibration_box)
        calibration_form.setContentsMargins(0, 0, 0, 0)
        calibration_form.addRow("Measured data", self._build_measured_loader())
        self.measured_info = QLabel()
        self.measured_info.setWordWrap(True)
        calibration_form.addRow(self.measured_info)
        calibration_form.addRow("Measured FID amplitude", self.amplitude)
        calibration_form.addRow("Measured noise std", self.reference_std)
        calibration_form.addRow("Averages", self.averages)
        calibration_form.addRow("Reference flip angle", self.flip_angle)
        calibration_form.addRow("Reference concentration (mM)", self.concentration)
        calibration_form.addRow("Reference volume (ml)", self.volume)
        calibration_form.addRow("Isotope enrichment", self.enrichment)
        calibration_form.addRow(
            "Labelled nuclei per molecule", self.nuclei_per_molecule
        )
        calibration_form.addRow("Phantom temperature", self.reference_temperature)
        calibration_form.addRow("Simulated object temperature", self.target_temperature)
        calibration_form.addRow("Noise factor (loading)", self.noise_factor)
        self._build_reference_scan()
        calibration_form.addRow(self.reference_scan_box)
        form.addRow(self.calibration_box)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setToolTip("Same seed and clean ADC data reproduce the same noise")
        form.addRow("Random seed", self.seed)
        self._align_label_columns(form, calibration_form)
        self.calibration_button = QPushButton("Set calibration")
        self.calibration_button.clicked.connect(self.calibration_requested)
        form.addRow(self.calibration_button)
        self.info = QLabel()
        self.info.setWordWrap(True)
        form.addRow(self.info)
        self.assumptions = QLabel(
            "Single receive channel; noise std is per real / imaginary channel "
            "and scales with √(bandwidth) for each ADC event. Simulated runs are "
            "single scans. Calibration assumes a fully relaxed reference at "
            "t = 0 in a homogeneous coil unless the reference scan model says "
            "otherwise, physical 3D voxel volume, the same nucleus, field and "
            "receive coil. Sequence streaming previews remain noiseless."
        )
        self.assumptions.setWordWrap(True)
        form.addRow(self.assumptions)
        buttons = QHBoxLayout()
        self.apply_button = QPushButton("Apply to result")
        self.apply_button.clicked.connect(self.apply_requested)
        self.new_button = QPushButton("New realization")
        self.new_button.clicked.connect(self._new_realization)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.new_button)
        form.addRow(buttons)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        for control in (
            self.amplitude,
            self.reference_std,
            self.concentration,
            self.volume,
        ):
            editor = (
                control.lineEdit() if isinstance(control, QDoubleSpinBox) else control
            )
            editor.textChanged.connect(self._refresh)
        self.snr.lineEdit().textChanged.connect(self._refresh)
        # Qt may block the internal line editor's signals while its arrows
        # update the value; retain live preview updates for both input paths.
        self.snr.valueChanged.connect(self._refresh)
        self.reference_std.valueChanged.connect(self._refresh)
        for control in (*self._calibration_spin_boxes(), self.bandwidth_khz):
            control.valueChanged.connect(self._refresh)
        for control in self._reference_spin_boxes():
            control.valueChanged.connect(self._refresh)
        self.reference_scan_box.toggled.connect(self._refresh)
        self.reference_use_coil.toggled.connect(self._refresh)
        self.reference_geometry.currentIndexChanged.connect(self._refresh)
        self.seed.valueChanged.connect(self._refresh)
        self._refresh()

    def _build_measured_loader(self):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        self.load_fid_button = QPushButton("Load FID…")
        self.load_fid_button.setToolTip(
            "Read a ParaVision fid / rawdata.job0 (with acqp and method) or a "
            "complex .npy / text array. Fills the amplitude extrapolated to the "
            "acquisition start, the noise std of the FID tail unless a noise "
            "scan is loaded, averages (NA × NR), bandwidth, flip angle and the "
            "reference scan timing."
        )
        self.load_noise_button = QPushButton("Load noise scan…")
        self.load_noise_button.setToolTip(
            "Read a scan without excitation (0° pulse) with the FID's averages, "
            "receiver gain and bandwidth; its noise std replaces the FID tail "
            "estimate."
        )
        self.load_fid_button.clicked.connect(
            lambda: self._choose_measured_file("Load measured FID", self.load_fid)
        )
        self.load_noise_button.clicked.connect(
            lambda: self._choose_measured_file("Load noise scan", self.load_noise_scan)
        )
        layout.addWidget(self.load_fid_button)
        layout.addWidget(self.load_noise_button)
        return row

    def _choose_measured_file(self, title, load):
        path, _ = QFileDialog.getOpenFileName(self, title, "", MEASURED_DATA_FILTER)
        if path:
            load(path)

    def load_fid(self, path):
        """Read a measured FID and fill the calibration inputs from it."""
        try:
            scan = load_measured_scan(path)
        except (OSError, ValueError) as exc:
            self.measured_info.setText(f"Could not read the FID: {exc}.")
            return False
        self._measured_fid = scan
        return self._apply_measured_data()

    def load_noise_scan(self, path):
        """Read a scan without excitation as the calibration noise."""
        try:
            scan = load_measured_scan(path)
            if self._measured_fid is not None:
                check_noise_scan(self._measured_fid, scan)
        except (OSError, ValueError) as exc:
            self.measured_info.setText(f"Could not use the noise scan: {exc}.")
            return False
        self._measured_noise = scan
        if self._measured_fid is None:
            self.measured_info.setText("Noise scan loaded; load the FID next.")
            return True
        return self._apply_measured_data()

    def _apply_measured_data(self):
        fid, noise, note = self._measured_fid, self._measured_noise, ""
        if noise is not None:
            try:
                check_noise_scan(fid, noise)
            except ValueError as exc:
                noise, self._measured_noise = None, None
                note = f"Noise scan dropped ({exc}); using the FID tail.\n"
        try:
            inputs = measured_calibration_inputs(fid, noise)
        except ValueError as exc:
            self.measured_info.setText(f"{note}Could not analyse the FID: {exc}.")
            return False
        self.amplitude.setText(f"{inputs.amplitude.amplitude:.6g}")
        self.reference_std.setValue(float(f"{inputs.noise.std:.6g}"))
        self.averages.setValue(inputs.averages)
        if inputs.dwell_s is not None:
            self.bandwidth_khz.setValue(1e-3 / inputs.dwell_s)
        if inputs.flip_angle_deg is not None:
            self.flip_angle.setValue(inputs.flip_angle_deg)
        if inputs.repetition_time_s is not None:
            self.reference_tr.setValue(inputs.repetition_time_s)
        if inputs.dummy_scans is not None:
            self.reference_dummy_scans.setValue(inputs.dummy_scans)
        if inputs.amplitude.t2star_s is not None:
            self.reference_t2star.setValue(inputs.amplitude.t2star_s * 1e3)
        if inputs.amplitude_delay_s is not None:
            self.reference_delay.setValue(inputs.amplitude_delay_s * 1e3)
        scanner = ""
        if inputs.nucleus is not None and inputs.field_strength_t is not None:
            scanner = f" ({inputs.nucleus} at {inputs.field_strength_t:.4g} T)"
        source = Path(fid.source)
        self.measured_info.setText(
            f"{note}{source.parent.name}/{source.name}{scanner}: " + inputs.summary()
        )
        return True

    def _build_reference_scan(self):
        """Optional signal model of the measured reference FID."""
        box = self.reference_scan_box = QGroupBox("Reference scan model")
        box.setCheckable(True)
        box.setChecked(False)
        box.setToolTip(
            "Correct the measured amplitude for T1 saturation by dummy scans and "
            "averages, T2* decay until the measured point and the B1 profile "
            "over the phantom. Off: the reference is fully relaxed, measured at "
            "t = 0 and in a homogeneous coil."
        )
        self.reference_t1 = _spin(
            0.0,
            1e4,
            0.0,
            3,
            " s",
            "Phantom T1. ∞: full recovery before every repetition.",
        )
        self.reference_t1.setSpecialValueText("∞ (full recovery)")
        self.reference_tr = _spin(
            0.0,
            1e4,
            0.0,
            3,
            " s",
            "Repetition time of the reference scan; needed with T1 when dummy "
            "scans or averages repeat the excitation.",
        )
        self.reference_tr.setSpecialValueText("not set")
        self.reference_dummy_scans = QSpinBox()
        self.reference_dummy_scans.setRange(0, 2**16)
        self.reference_dummy_scans.setToolTip(
            "Excitations before the averaged repetitions; they saturate the "
            "phantom with T1. The sample is fully relaxed before the first one."
        )
        self.reference_t2star = _spin(
            0.0,
            1e5,
            0.0,
            2,
            " ms",
            "Phantom T2* (linewidth 1 / (π T2*)). ∞: no decay before the "
            "measured point.",
        )
        self.reference_t2star.setSpecialValueText("∞ (no decay)")
        self.reference_delay = _spin(
            0.0,
            1e4,
            0.0,
            3,
            " ms",
            "Time from the excitation centre to the measured amplitude: half "
            "the pulse, the ADC delay and the sample offset. 0 if the amplitude "
            "was extrapolated to t = 0.",
        )
        self.reference_geometry = QComboBox()
        self.reference_geometry.addItem("Sphere", "sphere")
        self.reference_geometry.addItem("Cylinder along z", "cylinder")
        self.reference_geometry.setToolTip(
            "Shape of the reference volume inside the coil fields."
        )
        self.reference_diameter = _spin(
            0.1,
            1e4,
            20.0,
            1,
            " mm",
            "Inner diameter of the cylinder; its length follows from the volume.",
        )
        self.reference_position = tuple(
            _spin(
                -1e4,
                1e4,
                0.0,
                1,
                " mm",
                f"Phantom centre {axis} in the object coordinates of the B1 fields.",
            )
            for axis in "xyz"
        )
        position = QWidget()
        position_layout = QHBoxLayout(position)
        position_layout.setContentsMargins(0, 0, 0, 0)
        for control in self.reference_position:
            position_layout.addWidget(control)
        self.reference_use_coil = QCheckBox("Use B1 workspace coil fields")
        self.reference_use_coil.setToolTip(
            "Sample the B1+ and B1− fields of the B1 workspace over the phantom. "
            "Simulations must apply the same fields to their object, because "
            "noise is calibrated per unit B1− weight."
        )
        self.reference_info = QLabel()
        self.reference_info.setWordWrap(True)
        scan_form = QFormLayout(box)
        scan_form.addRow("Phantom T1", self.reference_t1)
        scan_form.addRow("Repetition time", self.reference_tr)
        scan_form.addRow("Dummy scans", self.reference_dummy_scans)
        scan_form.addRow("Phantom T2*", self.reference_t2star)
        scan_form.addRow("Amplitude delay", self.reference_delay)
        scan_form.addRow(self.reference_use_coil)
        scan_form.addRow("Geometry", self.reference_geometry)
        scan_form.addRow("Cylinder diameter", self.reference_diameter)
        scan_form.addRow("Centre x / y / z", position)
        scan_form.addRow(self.reference_info)

    def _reference_spin_boxes(self):
        return (
            self.reference_t1,
            self.reference_tr,
            self.reference_dummy_scans,
            self.reference_t2star,
            self.reference_delay,
            self.reference_diameter,
            *self.reference_position,
        )

    def coil_fields_changed(self):
        """Follow edits of the coil fields that the reference model samples."""
        if self.reference_scan_box.isChecked() and self.reference_use_coil.isChecked():
            self._refresh()

    def reference_t2star_s(self):
        """T2* of the reference scan model, or None without one."""
        if not self.reference_scan_box.isChecked():
            return None
        return self.reference_t2star.value() * 1e-3 or None

    def _reference_scan(self, volume_ml):
        """Entered reference scan, or None when the model is off."""
        if not self.reference_scan_box.isChecked():
            return None
        geometry = self.reference_geometry.currentData()
        return ReferenceScan(
            volume_ml=volume_ml,
            flip_angle_deg=self.flip_angle.value(),
            averages=self.averages.value(),
            dummy_scans=self.reference_dummy_scans.value(),
            repetition_time_s=self.reference_tr.value() or None,
            t1_s=self.reference_t1.value() or None,
            t2star_s=self.reference_t2star.value() * 1e-3 or None,
            amplitude_delay_s=self.reference_delay.value() * 1e-3,
            geometry=geometry,
            diameter_m=(
                self.reference_diameter.value() * 1e-3
                if geometry == "cylinder"
                else None
            ),
            position_m=tuple(
                control.value() * 1e-3 for control in self.reference_position
            ),
        )

    def _reference_efficiency(self, scan):
        tx_field = rx_field = None
        if self.reference_use_coil.isChecked():
            if self.coil_fields is not None:
                tx_field, rx_field = self.coil_fields()
            if tx_field is None and rx_field is None:
                raise ValueError(
                    "Define a B1 field in the B1 workspace or clear "
                    "'Use B1 workspace coil fields'"
                )
        return reference_signal_efficiency(scan, tx_field, rx_field)

    def _reference_inputs(self, volume_ml):
        """Calibration fields of the reference model and its efficiency."""
        scan = self._reference_scan(volume_ml)
        if scan is None:
            return {"signal_efficiency": None, "reference_scan": None}, None
        efficiency = self._reference_efficiency(scan)
        metadata = scan.to_metadata()
        metadata["coil_model"] = efficiency.coil_model
        return {
            "signal_efficiency": efficiency.efficiency,
            "reference_scan": metadata,
        }, efficiency

    def _reference_text(self, efficiency):
        flip = self.flip_angle.value()
        parts = [
            f"Measured amplitude per ideal 90° FID: η = {efficiency.efficiency:.4g}"
        ]
        transverse = math.sin(math.radians(flip))
        parts.append(
            f"sin {flip:g}° = {transverse:.4g}, with saturation and T2* decay "
            f"{efficiency.homogeneous:.4g}"
        )
        if efficiency.coil_model:
            coverage = (
                ""
                if efficiency.coverage == 1.0
                else f"; {100 * (1 - efficiency.coverage):.3g} % of the phantom "
                "lies outside the fields"
            )
            parts.append(
                f"{efficiency.coil_model} over {efficiency.sample_points} points "
                f"× {efficiency.efficiency / efficiency.homogeneous:.4g}{coverage}"
            )
        return "; ".join(parts) + "."

    @staticmethod
    def _align_label_columns(*forms):
        labels = [
            form.itemAt(row, QFormLayout.LabelRole).widget()
            for form in forms
            for row in range(form.rowCount())
            if form.itemAt(row, QFormLayout.LabelRole) is not None
        ]
        width = max(label.sizeHint().width() for label in labels)
        for label in labels:
            label.setMinimumWidth(width)

    def _calibration_spin_boxes(self):
        return (
            self.averages,
            self.flip_angle,
            self.enrichment,
            self.nuclei_per_molecule,
            self.reference_temperature,
            self.target_temperature,
            self.noise_factor,
        )

    def _mode_changed(self):
        self._refresh()
        self.mode_changed.emit()

    def _calibration_inputs(self):
        """ReceiverCalibration fields except the scanner context."""

        def required(field, name):
            text = field.text().strip()
            if not text:
                raise ValueError(f"Enter the {name}")
            return float(text)

        volume_ml = required(self.volume, "reference volume")
        inputs = {
            "measured_amplitude": float(self.amplitude.text()),
            "measured_noise_std": float(self.reference_std.text()),
            "concentration_mm": required(self.concentration, "reference concentration"),
            "volume_ml": volume_ml,
            "dwell_s": self._reference_dwell_s(),
            "averages": self.averages.value(),
            "flip_angle_deg": self.flip_angle.value(),
            "temperature_k": self.reference_temperature.value() + KELVIN_AT_0_CELSIUS,
            "nuclei_per_molecule": self.nuclei_per_molecule.value(),
            "enrichment": self.enrichment.value() / 100.0,
            "target_temperature_k": self.target_temperature.value()
            + KELVIN_AT_0_CELSIUS,
            "noise_factor": self.noise_factor.value(),
        }
        inputs.update(self._reference_inputs(volume_ml)[0])
        return inputs

    def _reference_dwell_s(self):
        return 1.0 / (self.bandwidth_khz.value() * 1e3)

    def _relative_snr(self):
        try:
            snr = float(self.snr.text())
        except ValueError:
            snr = math.nan
        if not math.isfinite(snr) or snr <= 0:
            raise ValueError("Relative SNR must be positive and finite")
        return snr

    def _calibration_matches_inputs(self):
        if self.calibration is None:
            return False
        try:
            inputs = self._calibration_inputs()
        except ValueError:
            return False
        efficiency = inputs.pop("signal_efficiency")
        stored = self.calibration.signal_efficiency
        # Coil fields are sampled again from the B1 workspace on every check.
        same_efficiency = (
            stored == efficiency
            if stored is None or efficiency is None
            else math.isclose(stored, efficiency, rel_tol=1e-9)
        )
        return (
            self.calibration.polarization is None
            and same_efficiency
            and all(
                getattr(self.calibration, key) == value for key, value in inputs.items()
            )
        )

    def _calibration_text(self):
        calibration = self.calibration
        return (
            f"Calibrated for {calibration.nucleus} at "
            f"{calibration.field_strength_t:g} T: "
            f"{format_amount(calibration.reference_moles)} nuclei, thermal "
            f"polarization {calibration.reference_polarization:.4g}; single-scan "
            f"SNR {calibration.single_scan_snr:.6g}. Noise std at the reference "
            f"bandwidth: {calibration.noise_std(calibration.dwell_s):.6g} "
            "simulation units. Simulated concentrations are read as mM and Mz "
            "relative to thermal at "
            f"{calibration.target_temperature_k - KELVIN_AT_0_CELSIUS:g} °C "
            f"(polarization {calibration.target_thermal_polarization:.4g})."
            + (
                ""
                if calibration.signal_efficiency is None
                else " Reference amplitude per ideal 90° FID: "
                f"{calibration.signal_efficiency:.4g}."
            )
        )

    def _refresh(self):
        mode = self.mode.currentData()
        calibrated = mode == "calibrated"
        self.snr.setEnabled(mode == "relative")
        self.bandwidth_khz.setEnabled(mode != "off")
        calibration_fields = (
            self.amplitude,
            self.reference_std,
            self.concentration,
            self.volume,
            *self._calibration_spin_boxes(),
        )
        for control, visible in (
            (self.snr, mode == "relative"),
            (self.bandwidth_khz, mode != "off"),
            (self.seed, mode != "off"),
        ):
            control.setVisible(visible)
            self.layout().labelForField(control).setVisible(visible)
        self.calibration_box.setVisible(calibrated)
        self._refresh_reference(calibrated)
        self.calibration_button.setVisible(calibrated)
        self.assumptions.setVisible(mode != "off")
        for control in (
            *calibration_fields,
            self.calibration_button,
            self.load_fid_button,
            self.load_noise_button,
        ):
            control.setEnabled(calibrated)
        self.seed.setEnabled(mode != "off")
        self.apply_button.setEnabled(self._result_available)
        self.new_button.setEnabled(mode != "off")
        if mode == "off":
            self.info.setText("Receiver noise is off.")
        elif mode == "relative":
            self.info.setText(self._relative_text())
        elif self._calibration_matches_inputs():
            self.info.setText(self._calibration_text())
        else:
            try:
                inputs = self._calibration_inputs()
                snr = inputs["measured_amplitude"] / inputs["measured_noise_std"]
                prefix = f"Reference FID SNR: {snr:.6g}. " if snr > 0 else ""
                message = "Set the calibration to use it in simulations."
            except (ValueError, ZeroDivisionError) as exc:
                prefix = ""
                if self._reference_error:
                    message = "Complete the reference scan model."
                else:
                    message = (
                        (f"{exc}. " if "Enter the" in str(exc) else "")
                        + "Enter positive FID amplitude, noise std, concentration and volume."
                    )
            self.info.setText(prefix + message)
        self.preview_changed.emit()

    def _refresh_reference(self, calibrated):
        """Show the reference model's efficiency or why it cannot be computed."""
        self._reference_error = False
        if not (calibrated and self.reference_scan_box.isChecked()):
            self.reference_info.setText("")
            return
        coil = self.reference_use_coil.isChecked()
        cylinder = self.reference_geometry.currentData() == "cylinder"
        # Placement only matters inside non-uniform coil fields.
        self.reference_geometry.setEnabled(coil)
        self.reference_diameter.setEnabled(coil and cylinder)
        for control in self.reference_position:
            control.setEnabled(coil)
        try:
            volume_ml = float(self.volume.text())
        except ValueError:
            self.reference_info.setText("Enter the reference volume.")
            return
        try:
            _, efficiency = self._reference_inputs(volume_ml)
        except ValueError as exc:
            self._reference_error = True
            self.reference_info.setText(f"{exc}.")
            return
        self.reference_info.setText(self._reference_text(efficiency))

    def _relative_text(self):
        try:
            snr = self._relative_snr()
        except ValueError as exc:
            return f"{exc}."
        return (
            f"Noise std = ideal FID amplitude / {snr:.6g} at "
            f"{self.bandwidth_khz.value():g} kHz, scaled with √(bandwidth) for "
            "each ADC event. The amplitude is the whole object's initial plus "
            "delivered magnetization after an ideal 90° pulse, before relaxation; "
            "each run records it, so SNR does not depend on the phantom grid or "
            "signal weighting. Noise is added once to the summed complex ADC "
            "signal."
        )

    def _check_measured_context(self, nucleus, field_strength_t):
        """Refuse a loaded FID of another nucleus or field strength."""
        fid = self._measured_fid
        if fid is None:
            return
        problems = []
        if fid.nucleus is not None and fid.nucleus != nucleus:
            problems.append(fid.nucleus)
        field = fid.field_strength_t
        if field is not None and not math.isclose(
            field, field_strength_t, rel_tol=MEASURED_FIELD_TOLERANCE
        ):
            problems.append(f"{field:.4g} T")
        if problems:
            raise ValueError(
                f"The loaded FID was measured with {' at '.join(problems)}, but "
                f"the workspace uses {nucleus} at {field_strength_t:g} T"
            )

    def capture_calibration(self, *, nucleus, field_strength_t):
        self._check_measured_context(nucleus, field_strength_t)
        self.calibration = ReceiverCalibration(
            **self._calibration_inputs(),
            nucleus=nucleus,
            field_strength_t=field_strength_t,
        )
        self._refresh()

    def settings(self, *, nucleus=None, field_strength_t=None):
        mode = self.mode.currentData()
        if mode == "calibrated":
            if not self._calibration_matches_inputs():
                raise ValueError(
                    "Set the calibration after entering or changing its values"
                )
            if nucleus is not None:
                self.calibration.check_context(
                    nucleus=nucleus, field_strength_t=field_strength_t
                )
        return ReceiverNoise(
            mode=mode,
            snr=self._relative_snr() if mode == "relative" else 100.0,
            seed=self.seed.value(),
            reference_dwell_s=self._reference_dwell_s(),
            calibration=self.calibration if mode == "calibrated" else None,
        )

    def budget_settings(self, *, nucleus=None, field_strength_t=None):
        """Settings for the SNR budget and whether the calibration is a preview.

        Calibration values that are not set yet are previewed for the given
        scanner context; invalid inputs raise ValueError.
        """
        if self.mode.currentData() != "calibrated" or (
            self._calibration_matches_inputs()
        ):
            return self.settings(), False
        if nucleus is None or field_strength_t is None:
            raise ValueError("The calibration preview needs the nucleus and field")
        calibration = ReceiverCalibration(
            **self._calibration_inputs(),
            nucleus=nucleus,
            field_strength_t=field_strength_t,
        )
        noise = ReceiverNoise(
            mode="calibrated",
            seed=self.seed.value(),
            reference_dwell_s=self._reference_dwell_s(),
            calibration=calibration,
        )
        return noise, True

    def set_result_available(self, available):
        self._result_available = bool(available)
        self._refresh()

    def _new_realization(self):
        self.seed.setValue((self.seed.value() + 1) % (2**31))
        self.apply_requested.emit()

    _SPIN_STATE = (
        ("averages", 1),
        ("bandwidth_khz", 10.0),
        ("flip_angle", 90.0),
        ("enrichment", 100.0),
        ("nuclei_per_molecule", 1),
        ("reference_temperature", 20.0),
        ("target_temperature", 37.0),
        ("noise_factor", 1.0),
        ("reference_t1", 0.0),
        ("reference_tr", 0.0),
        ("reference_dummy_scans", 0),
        ("reference_t2star", 0.0),
        ("reference_delay", 0.0),
        ("reference_diameter", 20.0),
    )

    def get_state(self):
        state = {
            "mode": self.mode.currentData(),
            "snr": self.snr.text(),
            "amplitude": self.amplitude.text(),
            "reference_std": self.reference_std.text(),
            "concentration": self.concentration.text(),
            "volume": self.volume.text(),
            "measured_info": self.measured_info.text(),
            "seed": self.seed.value(),
            "calibration": (
                None if self.calibration is None else self.calibration.to_metadata()
            ),
        }
        state.update(
            {name: getattr(self, name).value() for name, _ in self._SPIN_STATE}
        )
        state.update(
            {
                "reference_model": self.reference_scan_box.isChecked(),
                "reference_use_coil": self.reference_use_coil.isChecked(),
                "reference_geometry": self.reference_geometry.currentData(),
                "reference_position_mm": [
                    control.value() for control in self.reference_position
                ],
            }
        )
        return state

    def restore_state(self, state):
        state = state or {}
        self.calibration = (
            ReceiverCalibration(**state["calibration"])
            if state.get("calibration")
            else None
        )
        for name, default in (
            ("snr", "100.0"),
            ("amplitude", "1"),
            ("reference_std", "0.01"),
            ("concentration", ""),
            ("volume", ""),
            ("measured_info", ""),
        ):
            getattr(self, name).setText(str(state.get(name, default)))
        self._measured_fid = self._measured_noise = None
        for name, default in self._SPIN_STATE:
            getattr(self, name).setValue(state.get(name, default))
        self.reference_geometry.setCurrentIndex(
            max(0, self.reference_geometry.findData(state.get("reference_geometry")))
        )
        for control, value in zip(
            self.reference_position, state.get("reference_position_mm", (0, 0, 0))
        ):
            control.setValue(value)
        self.reference_use_coil.setChecked(bool(state.get("reference_use_coil")))
        self.reference_scan_box.setChecked(bool(state.get("reference_model")))
        self.seed.setValue(int(state.get("seed", 0)))
        self.mode.setCurrentIndex(max(0, self.mode.findData(state.get("mode", "off"))))
        self._refresh()
        self.mode_changed.emit()
