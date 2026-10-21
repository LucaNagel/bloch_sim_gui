"""Single-channel receiver noise settings for the sequence workspace."""

from decimal import Decimal
import math
import sys

from PyQt5.QtCore import QLocale, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ..sequence.receiver import ReceiverCalibration, ReceiverNoise
from ..sequence.snr_budget import format_amount

KELVIN_AT_0_CELSIUS = 273.15


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
            "Amplitude and noise must have the same scanner scaling."
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
            "single scans. Calibration assumes a fully relaxed reference, "
            "physical 3D voxel volume, the same nucleus, field and receive coil. "
            "Sequence streaming previews remain noiseless."
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
        self.seed.valueChanged.connect(self._refresh)
        self._refresh()

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

        return {
            "measured_amplitude": float(self.amplitude.text()),
            "measured_noise_std": float(self.reference_std.text()),
            "concentration_mm": required(self.concentration, "reference concentration"),
            "volume_ml": required(self.volume, "reference volume"),
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
        return self.calibration.polarization is None and all(
            getattr(self.calibration, key) == value for key, value in inputs.items()
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
        self.calibration_button.setVisible(calibrated)
        self.assumptions.setVisible(mode != "off")
        for control in (*calibration_fields, self.calibration_button):
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
                message = (
                    (f"{exc}. " if "Enter the" in str(exc) else "")
                    + "Enter positive FID amplitude, noise std, concentration and volume."
                )
            self.info.setText(prefix + message)
        self.preview_changed.emit()

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

    def capture_calibration(self, *, nucleus, field_strength_t):
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
    )

    def get_state(self):
        state = {
            "mode": self.mode.currentData(),
            "snr": self.snr.text(),
            "amplitude": self.amplitude.text(),
            "reference_std": self.reference_std.text(),
            "concentration": self.concentration.text(),
            "volume": self.volume.text(),
            "seed": self.seed.value(),
            "calibration": (
                None if self.calibration is None else self.calibration.to_metadata()
            ),
        }
        state.update(
            {name: getattr(self, name).value() for name, _ in self._SPIN_STATE}
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
        ):
            getattr(self, name).setText(str(state.get(name, default)))
        for name, default in self._SPIN_STATE:
            getattr(self, name).setValue(state.get(name, default))
        self.seed.setValue(int(state.get("seed", 0)))
        self.mode.setCurrentIndex(max(0, self.mode.findData(state.get("mode", "off"))))
        self._refresh()
        self.mode_changed.emit()
