from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QWidget,
    QSizePolicy,
    QTextEdit,
    QMessageBox,
    QFileDialog,
    QDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
import pyqtgraph as pg
from pathlib import Path
from ..simulator import RF_PULSE_TYPE_OPTIONS, design_rf_pulse
from ..sequence.rf_pulses import (
    analytic_rf_shape_parameter,
    rf_envelope_integration_factor,
)
from .dialogs import PulseImportDialog


class RFPulseDesigner(QGroupBox):
    """Widget for designing RF pulses."""

    ADIABATIC_PASSAGE_TYPES = frozenset(
        {"Adiabatic Half Passage", "Adiabatic Full Passage"}
    )
    ADIABATIC_DEFAULT_DURATION_MS = 10.0
    MAX_DESIGN_POINTS = 1_000_000

    pulse_changed = pyqtSignal(object)
    parameters_changed = pyqtSignal(dict)

    def __init__(self, compact=False):
        super().__init__("RF Pulse Design")
        self.compact = compact
        self.target_dt = 5e-6  # default 5 us
        self.last_integration_factor = 1.0
        self.current_pulse = None
        self._syncing = False
        self.init_ui()

    def init_ui(self):
        # Main layout
        if self.compact:
            # Vertical layout for side panel
            main_layout = QVBoxLayout()
            content_layout = main_layout
            control_layout = main_layout
            control_panel = None  # No separate panel container
        else:
            # Full-page title followed by the same control/plot split used by
            # the Slice Explorer. Avoid the small native group-box caption.
            self.setTitle("")
            main_layout = QVBoxLayout()
            title = QLabel("RF Pulse Design")
            title_font = title.font()
            title_font.setBold(True)
            title_font.setPointSize(max(title_font.pointSize() + 2, 12))
            title.setFont(title_font)
            title.setVisible(False)
            self.page_title = title
            main_layout.addWidget(title)
            content_layout = QHBoxLayout()
            main_layout.addLayout(content_layout, 1)
            control_panel = QWidget()
            control_layout = QVBoxLayout()
            control_panel.setLayout(control_layout)
            control_panel.setMinimumWidth(400)
            control_panel.setMaximumWidth(400)
            self.control_panel = control_panel
            content_layout.addWidget(control_panel)

        # Pulse type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Pulse Type:"))
        self.pulse_type = QComboBox()
        prefix = "rf_compact_" if self.compact else "rf_tab_"
        self.pulse_type.setObjectName(f"{prefix}pulse_type")
        self.pulse_type.addItems(list(RF_PULSE_TYPE_OPTIONS))
        self.pulse_type.currentTextChanged.connect(self._on_pulse_type_changed)
        type_layout.addWidget(self.pulse_type)
        control_layout.addLayout(type_layout)

        # Flip angle
        flip_layout = QHBoxLayout()
        self.flip_angle_label = QLabel("Flip Angle (°):")
        flip_layout.addWidget(self.flip_angle_label)
        self.flip_angle = QDoubleSpinBox()
        self.flip_angle.setObjectName(f"{prefix}flip_angle")
        self.flip_angle.setRange(0, 1e4)
        self.flip_angle.setValue(90)
        self.flip_angle.setToolTip(
            "Nominal RF flip angle in degrees. Not used for AHP or AFP."
        )
        self.flip_angle.valueChanged.connect(self.update_pulse)
        flip_layout.addWidget(self.flip_angle)
        control_layout.addLayout(flip_layout)

        # Duration
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Duration (ms):"))
        self.duration = QDoubleSpinBox()
        self.duration.setObjectName(f"{prefix}duration")
        self.duration.setRange(0.001, 1000.0)  # Extended range for custom pulses
        self.duration.setValue(1.0)
        self.duration.setSingleStep(0.1)
        self.duration.setDecimals(3)
        self.duration.valueChanged.connect(self.update_pulse)
        duration_layout.addWidget(self.duration)
        control_layout.addLayout(duration_layout)

        # B1 Amplitude (G)
        b1_layout = QHBoxLayout()
        b1_layout.addWidget(QLabel("B1 Amplitude (G):"))
        self.b1_amplitude = QDoubleSpinBox()
        self.b1_amplitude.setObjectName(f"{prefix}b1_amplitude")
        self.b1_amplitude.setRange(0.0, 1e4)
        self.b1_amplitude.setValue(0.0)
        self.b1_amplitude.setSingleStep(0.01)
        self.b1_amplitude.setDecimals(4)
        self.b1_amplitude.setSpecialValueText("Auto")
        self.b1_amplitude.setToolTip(
            "Set the peak B1 amplitude directly in Gauss. A value above 0 is "
            "required for AHP/AFP; for other pulses, 0 = Auto from Flip Angle."
        )
        self.b1_amplitude.valueChanged.connect(self.update_pulse)
        b1_layout.addWidget(self.b1_amplitude)
        control_layout.addLayout(b1_layout)

        self.adiabatic_parameter_hint = QLabel(
            "AHP/AFP do not use a nominal flip angle. Set B1 Amplitude "
            "directly to a value above 0 G."
        )
        self.adiabatic_parameter_hint.setObjectName(f"{prefix}adiabatic_parameter_hint")
        self.adiabatic_parameter_hint.setWordWrap(True)
        self.adiabatic_parameter_hint.setStyleSheet("color: #b06a00;")
        self.adiabatic_parameter_hint.setVisible(False)
        control_layout.addWidget(self.adiabatic_parameter_hint)

        # Time-bandwidth product (computed from pulse shape; not user-set)
        # tbw_layout = QHBoxLayout()
        # tbw_layout.addWidget(QLabel("Time-BW Product (auto, kHz*ms):"))
        self.tbw = QDoubleSpinBox()
        self.tbw.setObjectName(f"{prefix}tbw")
        self.tbw.setRange(0.001, 1000)
        self.tbw.setValue(1)
        self.tbw.setSingleStep(0.5)
        self.tbw.setReadOnly(True)
        self.tbw.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.tbw.hide()
        # tbw_layout.addWidget(self.tbw)
        # control_layout.addLayout(tbw_layout)
        self.tbw_auto_label = QLabel("Time-bandwidth product (auto): —")
        self.tbw_auto_label.setStyleSheet("color: gray;")
        control_layout.addWidget(self.tbw_auto_label)

        design_tbw_layout = QHBoxLayout()
        design_tbw_layout.addWidget(QLabel("Legacy shape parameter:"))
        self.design_tbw = QDoubleSpinBox()
        self.design_tbw.setObjectName(f"{prefix}design_tbw")
        self.design_tbw.setRange(0.1, 100.0)
        self.design_tbw.setDecimals(2)
        self.design_tbw.setSingleStep(0.5)
        self.design_tbw.setValue(4.0)
        self.design_tbw.setReadOnly(True)
        self.design_tbw.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.design_tbw.setToolTip(
            "Compatibility value; TBW is calculated automatically"
        )
        design_tbw_layout.addWidget(self.design_tbw)
        self.design_tbw_container = QWidget()
        self.design_tbw_container.setLayout(design_tbw_layout)
        self.design_tbw_container.setVisible(False)
        control_layout.addWidget(self.design_tbw_container)

        # RF Pulse bandwidth (computed from pulse shape; not user-set)
        # rf_bandwidth_layout = QHBoxLayout()
        # rf_bandwidth_layout.addWidget(QLabel("RF Bandwidth (auto, kHz):"))
        self.rf_bandwidth = QDoubleSpinBox()
        self.rf_bandwidth.setObjectName(f"{prefix}rf_bandwidth")
        self.rf_bandwidth.setRange(0.001, 100000)
        self.rf_bandwidth.setValue(1)
        self.rf_bandwidth.setSingleStep(0.1)
        self.rf_bandwidth.setReadOnly(True)
        self.rf_bandwidth.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.rf_bandwidth.hide()
        # rf_bandwidth_layout.addWidget(self.rf_bandwidth)
        # control_layout.addLayout(rf_bandwidth_layout)
        self.rf_bandwidth_auto_label = QLabel(
            "RF bandwidth (auto = TBW / duration): — Hz"
        )
        self.rf_bandwidth_auto_label.setStyleSheet("color: gray;")
        control_layout.addWidget(self.rf_bandwidth_auto_label)

        # Lobes control for Sinc pulses
        lobes_layout = QHBoxLayout()
        lobes_layout.addWidget(QLabel("Lobes (Sinc):"))
        self.sinc_lobes = QSpinBox()
        self.sinc_lobes.setObjectName(f"{prefix}sinc_lobes")
        self.sinc_lobes.setRange(1, 100)
        self.sinc_lobes.setValue(3)
        self.sinc_lobes.valueChanged.connect(self.update_pulse)
        lobes_layout.addWidget(self.sinc_lobes)
        self.lobes_container = QWidget()
        self.lobes_container.setLayout(lobes_layout)
        control_layout.addWidget(self.lobes_container)

        slr_layout = QHBoxLayout()
        slr_layout.addWidget(QLabel("SLR sharpness:"))
        self.slr_sharpness = QDoubleSpinBox()
        self.slr_sharpness.setObjectName(f"{prefix}slr_sharpness")
        self.slr_sharpness.setRange(0.1, 20.0)
        self.slr_sharpness.setDecimals(2)
        self.slr_sharpness.setSingleStep(0.5)
        self.slr_sharpness.setValue(1.0)
        self.slr_sharpness.setToolTip(
            "Higher sharpness produces a narrower SLR transition and more "
            "temporal lobes"
        )
        self.slr_sharpness.valueChanged.connect(self.update_pulse)
        slr_layout.addWidget(self.slr_sharpness)
        self.slr_sharpness_container = QWidget()
        self.slr_sharpness_container.setLayout(slr_layout)
        control_layout.addWidget(self.slr_sharpness_container)

        # Apodization
        apod_layout = QHBoxLayout()
        apod_layout.addWidget(QLabel("Apodization:"))
        self.apodization_combo = QComboBox()
        self.apodization_combo.setObjectName(f"{prefix}apodization_combo")
        self.apodization_combo.addItems(["None", "Hamming", "Hanning", "Blackman"])
        self.apodization_combo.currentTextChanged.connect(self.update_pulse)
        apod_layout.addWidget(self.apodization_combo)
        control_layout.addLayout(apod_layout)

        # Phase
        phase_layout = QHBoxLayout()
        phase_layout.addWidget(QLabel("Phase (°):"))
        self.phase = QDoubleSpinBox()
        self.phase.setObjectName(f"{prefix}phase")
        self.phase.setRange(-360, 360)
        self.phase.setValue(0)
        self.phase.valueChanged.connect(self.update_pulse)
        phase_layout.addWidget(self.phase)
        control_layout.addLayout(phase_layout)

        # RF Frequency Offset
        freq_offset_layout = QHBoxLayout()
        freq_offset_layout.addWidget(QLabel("Sequence RF Carrier Offset (Hz):"))
        self.freq_offset = QDoubleSpinBox()
        self.freq_offset.setObjectName(f"{prefix}freq_offset")
        self.freq_offset.setRange(-10000, 10000)
        self.freq_offset.setValue(0.0)
        self.freq_offset.setSingleStep(10)
        self.freq_offset.setDecimals(1)
        self.freq_offset.valueChanged.connect(self.update_pulse)
        freq_offset_layout.addWidget(self.freq_offset)
        control_layout.addLayout(freq_offset_layout)

        # Info label for Custom Pulse
        self.custom_info_label = QLabel("")
        self.custom_info_label.setObjectName(f"{prefix}custom_info_label")
        self.custom_info_label.setStyleSheet("color: gray; font-size: 9pt;")
        self.custom_info_label.setVisible(False)
        control_layout.addWidget(self.custom_info_label)

        # Pulse Explanation (Only in full mode)
        self.explanation_box = QTextEdit()
        self.explanation_box.setObjectName(f"{prefix}explanation_box")
        self.explanation_box.setReadOnly(True)
        self.explanation_box.setMaximumHeight(150)

        if not self.compact:
            control_layout.addWidget(QLabel("Pulse Description:"))
            control_layout.addWidget(self.explanation_box)

        # Buttons
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load from File")
        self.load_button.setObjectName(f"{prefix}load_button")
        self.load_button.setToolTip("Load a custom RF pulse waveform")
        self.load_button.clicked.connect(self.load_pulse_from_file)
        self.save_button = QPushButton("Save to File")
        self.save_button.setObjectName(f"{prefix}save_button")
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        control_layout.addLayout(button_layout)

        control_layout.addStretch()

        # Plot Widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "B1 Amplitude", "G")
        self.plot_widget.setLabel("bottom", "Time", "ms")

        if self.compact:
            self.plot_widget.setMinimumHeight(150)
            main_layout.addWidget(self.plot_widget)
        else:
            # Right column in full mode
            plot_layout = QVBoxLayout()
            self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            plot_layout.addWidget(self.plot_widget)
            content_layout.addLayout(plot_layout, stretch=1)

        self.setLayout(main_layout)

        # Storage for loaded pulse data
        self.loaded_pulse_b1 = None
        self.loaded_pulse_time = None
        self.loaded_pulse_metadata = None

        # Initial pulse
        self.update_pulse()

    def is_adiabatic_passage(self) -> bool:
        """Return whether AHP or AFP is selected."""
        return self.pulse_type.currentText() in self.ADIABATIC_PASSAGE_TYPES

    def _update_parameter_availability(self):
        """Expose the physically meaningful controls for the pulse family."""
        adiabatic_passage = self.is_adiabatic_passage()
        self.flip_angle.setEnabled(not adiabatic_passage)
        self.flip_angle_label.setText(
            "Flip Angle (not applicable):" if adiabatic_passage else "Flip Angle (°):"
        )
        self.adiabatic_parameter_hint.setVisible(adiabatic_passage)
        self.b1_amplitude.setSpecialValueText(
            "Required" if adiabatic_passage else "Auto"
        )

    def _on_pulse_type_changed(self, _pulse_type: str):
        """Apply pulse-family defaults once when the user changes the type."""
        if self.is_adiabatic_passage():
            was_blocked = self.duration.blockSignals(True)
            self.duration.setValue(self.ADIABATIC_DEFAULT_DURATION_MS)
            self.duration.blockSignals(was_blocked)
        self.update_pulse()

    def _update_tbw_auto(
        self, integration_factor: float, time_bandwidth_product: float = None
    ):
        """Set the readout from an explicit bandwidth factor or integration factor."""
        if not hasattr(self, "tbw") or not hasattr(self, "tbw_auto_label"):
            return
        explicit_tbw = (
            time_bandwidth_product is not None
            and np.isfinite(time_bandwidth_product)
            and time_bandwidth_product > 0
        )
        valid_integration_factor = (
            integration_factor is not None
            and np.isfinite(integration_factor)
            and integration_factor > 0
        )
        if not explicit_tbw and not valid_integration_factor:
            self.tbw_auto_label.setText("Time-bandwidth product (auto): —")
            self.last_integration_factor = 1.0
            return
        tbw_auto = (
            float(time_bandwidth_product)
            if explicit_tbw
            else 1.0 / float(integration_factor)
        )
        self.tbw_auto_label.setText(f"Time-bandwidth product (auto): {tbw_auto:.3f}")
        if valid_integration_factor:
            self.last_integration_factor = float(integration_factor)
        # Keep the control in sync without retriggering pulse design
        self.tbw.blockSignals(True)
        self.tbw.setValue(tbw_auto)
        self.tbw.blockSignals(False)

    def _update_rf_bandwidth_auto(
        self,
        integration_factor: float,
        duration: float,
        time_bandwidth_product: float = None,
    ):
        """Set RF bandwidth in Hz from the shape-intrinsic TBW and duration."""
        if not hasattr(self, "rf_bandwidth") or not hasattr(
            self, "rf_bandwidth_auto_label"
        ):
            return
        explicit_tbw = (
            time_bandwidth_product is not None
            and np.isfinite(time_bandwidth_product)
            and time_bandwidth_product > 0
        )
        valid_integration_factor = (
            integration_factor is not None
            and np.isfinite(integration_factor)
            and integration_factor > 0
        )
        if (not explicit_tbw and not valid_integration_factor) or duration <= 0:
            self.rf_bandwidth_auto_label.setText(
                "RF bandwidth (auto = TBW / duration): — Hz"
            )
            self.last_integration_factor = 1.0
            return
        tbw_auto = (
            float(time_bandwidth_product)
            if explicit_tbw
            else 1.0 / float(integration_factor)
        )
        rf_bandwidth_auto = tbw_auto / duration
        self.rf_bandwidth_auto_label.setText(
            f"RF bandwidth (auto = TBW / duration): {rf_bandwidth_auto:.3f} Hz"
        )
        if valid_integration_factor:
            self.last_integration_factor = float(integration_factor)
        # Keep the control in sync without retriggering pulse design
        self.rf_bandwidth.blockSignals(True)
        self.rf_bandwidth.setValue(rf_bandwidth_auto)
        self.rf_bandwidth.blockSignals(False)

    def _design_tbw_for_type(self, pulse_type: str) -> float:
        """Return the fixed construction parameter for an analytic pulse family."""
        pt = pulse_type.lower()
        if pt.startswith("adiabatic") or pt in ("bir-4", "bir4"):
            return 4.0  # modulation parameter for adiabatic-style pulses
        return analytic_rf_shape_parameter(pt, self.sinc_lobes.value())

    def _compute_integration_factor_from_wave(self, b1_wave, t_wave):
        """Compute the shape-only integration factor of a complex waveform."""
        try:
            return rf_envelope_integration_factor(b1_wave)
        except Exception:
            return 1.0

    def _scale_pulse_to_flip(
        self, b1_wave, t_wave, flip_deg: float, integfac: float = 1.0
    ):
        """Scale a complex waveform to achieve a target flip angle (degrees)."""
        b1_wave = np.asarray(b1_wave, dtype=complex)
        t_wave = np.asarray(t_wave, dtype=float)
        if b1_wave.size == 0 or t_wave.size == 0:
            return b1_wave
        flip_rad = np.deg2rad(flip_deg)
        peak = np.max(np.abs(b1_wave)) if np.any(np.abs(b1_wave)) else 1.0
        shape = b1_wave / peak if peak != 0 else b1_wave
        dt = float(np.median(np.diff(t_wave))) if len(t_wave) > 1 else 1e-6
        area = np.trapezoid(shape, dx=dt)
        opt_phase = -np.angle(area) if np.isfinite(area) and area != 0 else 0.0
        aligned_area = np.real(area * np.exp(1j * opt_phase))
        if not np.isfinite(aligned_area) or abs(aligned_area) < 1e-12:
            aligned_area = 1e-12
        aligned_area *= max(integfac, 1e-9)
        gmr_1h_rad_Ts = 267522187.43999997
        pulse_amp_T = flip_rad / (gmr_1h_rad_Ts * aligned_area)
        pulse_amp_G = pulse_amp_T * 1e4
        return shape * pulse_amp_G * np.exp(1j * opt_phase)

    def _apply_phase(self, b1_wave):
        """Apply the event phase while keeping the stored waveform at baseband."""
        b1_wave = np.asarray(b1_wave, dtype=complex)
        phase_rad = np.deg2rad(self.phase.value())
        return b1_wave * np.exp(1j * phase_rad)

    def _carrier_preview(self, b1_wave, t_wave):
        """Return a pulse-local carrier preview without changing baseband storage."""
        b1_wave = np.asarray(b1_wave, dtype=complex)
        t_wave = np.asarray(t_wave, dtype=float)
        t_rel = t_wave - t_wave[0] if t_wave.size else t_wave
        return b1_wave * np.exp(2j * np.pi * self.freq_offset.value() * t_rel)

    def get_integration_factor(self) -> float:
        """Return best-known integration factor (cached or recomputed from current pulse)."""
        if self.current_pulse is not None and len(self.current_pulse) == 2:
            b1_wave, t_wave = self.current_pulse
            computed = self._compute_integration_factor_from_wave(b1_wave, t_wave)
            self.last_integration_factor = computed
            return computed
        return self.last_integration_factor or 1.0

    def update_pulse(self):
        """Update the RF pulse based on current parameters."""
        pulse_type_text = self.pulse_type.currentText().lower()
        self._update_parameter_availability()

        # Update explanation
        desc_map = {
            "rectangle": "<b>Rectangular Pulse</b><br>Constant amplitude hard pulse. Broad excitation bandwidth.",
            "sinc": "<b>Sinc Pulse</b><br>Selective excitation. Fourier transform of a rectangular slice profile. Use 'Lobes' to control bandwidth/sharpness.",
            "slr": "<b>SLR Pulse</b><br>Small-tip linear-phase beta-polynomial design shared with Sequence Mode. Sharpness changes the pulse shape; the time-bandwidth product is calculated from the resulting waveform.",
            "gaussian": "<b>Gaussian Pulse</b><br>Selective pulse with no side lobes in time domain. Smooth excitation profile.",
            "hermite": "<b>Hermite Pulse</b><br>Short selective pulse derived from Hermite polynomials. Good for short TR sequences.",
            "adiabatic half passage": "<b>Adiabatic Half Passage (AHP)</b><br>Frequency sweep from off-resonance to resonance (or vice versa). Generates robust 90° excitation insensitive to B1 inhomogeneity (above a threshold).",
            "adiabatic full passage": "<b>Adiabatic Full Passage (AFP)</b><br>Frequency sweep from far off-resonance to far off-resonance. Generates robust 180° inversion insensitive to B1 inhomogeneity.",
            "bir-4": "<b>BIR-4</b><br>B1-Insensitive Rotation. Composite adiabatic pulse capable of arbitrary flip angles (defined by phase jumps).",
            "custom": "<b>Custom Pulse</b><br>User-loaded waveform. Use 'Load from File' to import.",
        }
        self.explanation_box.setHtml(desc_map.get(pulse_type_text, ""))

        pulse_type = pulse_type_text
        if pulse_type == "rectangle":
            pulse_type = "rect"
        elif pulse_type == "adiabatic half passage":
            pulse_type = "adiabatic_half"
        elif pulse_type == "adiabatic full passage":
            pulse_type = "adiabatic_full"
        elif pulse_type == "bir-4":
            pulse_type = "bir4"

        # Show/hide controls based on type
        self.lobes_container.setVisible(pulse_type == "sinc")
        self.design_tbw_container.setVisible(False)
        self.slr_sharpness_container.setVisible(pulse_type == "slr")
        self.custom_info_label.setVisible(pulse_type == "custom")

        duration = self.duration.value() / 1000  # Convert to seconds
        flip = self.flip_angle.value()
        b1_override = self.b1_amplitude.value()
        phase_rad = np.deg2rad(self.phase.value())

        # Handle Custom Pulse
        if pulse_type == "custom":
            if self.loaded_pulse_b1 is None or self.loaded_pulse_time is None:
                # Fallback if no pulse loaded
                self.plot_widget.clear()
                self.current_pulse = None
                return

            original_b1 = self.loaded_pulse_b1
            original_time = self.loaded_pulse_time
            original_duration = (
                original_time[-1] - original_time[0] if len(original_time) > 1 else 1e-6
            )

            # Resample to new duration
            if duration > 0 and original_duration > 0:
                time_scale = duration / original_duration
                new_time = original_time * time_scale
                # Simple resampling (linear interp) if points are sparse, or just use scaled time
                # Ideally we want to preserve shape. Just scaling time vector is enough if we don't change point count.
                b1 = original_b1.copy()
                time = new_time
            else:
                b1 = original_b1.copy()
                time = original_time.copy()

            # Apply Apodization
            window_type = self.apodization_combo.currentText()
            if window_type != "None" and len(b1) > 1:
                if window_type == "Hamming":
                    win = np.hamming(len(b1))
                elif window_type == "Hanning":
                    win = np.hanning(len(b1))
                elif window_type == "Blackman":
                    win = np.blackman(len(b1))
                else:
                    win = np.ones(len(b1))
                b1 = b1 * win

            # Calculate amplitude scaling
            peak = np.max(np.abs(b1)) if np.any(np.abs(b1)) else 1.0
            shape = b1 / peak if peak != 0 else b1

            # Prefer metadata for the unmodified source pulse. Apodization
            # creates a new shape, so its factors must be recomputed.
            integfac = 1.0
            explicit_tbw = None
            if (
                window_type == "None"
                and self.loaded_pulse_metadata
                and hasattr(self.loaded_pulse_metadata, "bwfac")
                and np.isfinite(self.loaded_pulse_metadata.bwfac)
                and self.loaded_pulse_metadata.bwfac > 0
            ):
                explicit_tbw = float(self.loaded_pulse_metadata.bwfac)
            if (
                window_type == "None"
                and self.loaded_pulse_metadata
                and hasattr(self.loaded_pulse_metadata, "integfac")
                and self.loaded_pulse_metadata.integfac > 0
            ):
                integfac = float(self.loaded_pulse_metadata.integfac)
            else:
                integfac = self._compute_integration_factor_from_wave(b1, time)

            self._update_tbw_auto(integfac, explicit_tbw)
            self._update_rf_bandwidth_auto(integfac, duration, explicit_tbw)
            self.last_integration_factor = float(integfac)

            # Amplitude scaling: B1 override vs Flip Angle
            if b1_override > 0:
                # Manual B1 override
                # Scale shape so peak matches b1_override
                b1 = shape * b1_override
            else:
                # Auto (Flip Angle)
                b1 = self._scale_pulse_to_flip(b1, time, flip, integfac=integfac)

            # Store a baseband waveform. The sequence rasterizer applies the
            # RF carrier later using absolute sequence time.
            b1 = self._apply_phase(b1)

            self.current_pulse = (b1, time)
            self.pulse_changed.emit(self.current_pulse)

            if not self._syncing:
                self.parameters_changed.emit(self.get_state())

            self._update_plot(self._carrier_preview(b1, time), time)
            return

        # Handle Standard Pulses
        design_tbw = self._design_tbw_for_type(pulse_type)

        # Target point count
        if self.target_dt and self.target_dt > 0:
            npoints = max(32, int(np.ceil(duration / self.target_dt)))
            npoints = min(npoints, self.MAX_DESIGN_POINTS)
        else:
            npoints = 100

        # 1. Generate base pulse
        b1_base, time = design_rf_pulse(
            pulse_type,
            duration,
            flip,
            design_tbw,
            npoints,
            freq_offset=0.0,
            slr_sharpness=self.slr_sharpness.value(),
        )

        dt = duration / len(b1_base) if len(b1_base) > 0 else 1e-6
        peak = np.max(np.abs(b1_base)) if np.any(np.abs(b1_base)) else 1.0
        shape = b1_base / peak if peak != 0 else b1_base

        # Apodization
        window_type = self.apodization_combo.currentText()
        if window_type != "None" and len(shape) > 1:
            if window_type == "Hamming":
                win = np.hamming(len(shape))
            elif window_type == "Hanning":
                win = np.hanning(len(shape))
            elif window_type == "Blackman":
                win = np.blackman(len(shape))
            else:
                win = np.ones(len(shape))
            shape = shape * win

        # Compute integration factor
        area = np.trapezoid(shape, dx=dt)
        opt_phase = -np.angle(area) if np.isfinite(area) else 0.0
        aligned_area = np.real(area * np.exp(1j * opt_phase))
        if not np.isfinite(aligned_area) or abs(aligned_area) < 1e-12:
            aligned_area = 1e-12
        integration_factor = self._compute_integration_factor_from_wave(shape, time)

        self._update_tbw_auto(integration_factor)
        self.last_integration_factor = float(integration_factor)

        self._update_rf_bandwidth_auto(
            integration_factor=integration_factor, duration=duration
        )

        # Amplitude scaling
        if self.is_adiabatic_passage():
            # AHP/AFP rotations follow the effective field and therefore do not
            # have a meaningful pulse-area flip angle. A value of zero remains
            # visibly invalid until the user supplies the peak B1 directly.
            pulse_amp_G = b1_override
        elif b1_override > 0:
            # Manual B1 override
            pulse_amp_G = b1_override
        else:
            # Auto (Flip Angle)
            flip_rad = np.deg2rad(flip)
            gmr_1h_rad_Ts = 267522187.43999997
            pulse_amp_T = flip_rad / (gmr_1h_rad_Ts * aligned_area)
            pulse_amp_G = pulse_amp_T * 1e4

        # Combine
        total_phase = opt_phase + phase_rad
        b1 = shape * pulse_amp_G * np.exp(1j * total_phase)

        self.current_pulse = (b1, time)
        self.pulse_changed.emit(self.current_pulse)

        if not self._syncing:
            self.parameters_changed.emit(self.get_state())

        self._update_plot(self._carrier_preview(b1, time), time)

    def _update_plot(self, b1, time):
        """Helper to update the plot widget."""
        self.plot_widget.clear()
        self.plot_widget.plot(time * 1000, np.abs(b1), pen="b", name="Magnitude")
        self.plot_widget.plot(time * 1000, np.real(b1), pen="r", name="Real")
        self.plot_widget.plot(time * 1000, np.imag(b1), pen="g", name="Imaginary")
        if len(time):
            t_max = time[-1] * 1000
            self.plot_widget.setLimits(xMin=0, xMax=max(t_max, 0.1))
            self.plot_widget.setXRange(0, max(t_max, 0.1), padding=0)

    def get_pulse(self):
        """Get the current RF pulse."""
        return self.current_pulse

    def set_time_step(self, dt_s: float):
        """Set desired temporal resolution for designed pulses."""
        if dt_s and dt_s > 0:
            self.target_dt = dt_s
            # Regenerate with new resolution to keep designer in sync
            self.update_pulse()

    def load_pulse_from_file(self):
        """Load RF pulse from a file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load RF Pulse",
            "",
            "Pulse Files (*.exc *.dat *.txt *.csv);;All Files (*)",
        )
        if filename:
            try:
                suffix = Path(filename).suffix.lower()
                if suffix == ".exc":
                    from ..pulse_loader import load_pulse_from_file as load_exc_file

                    b1, time, metadata = load_exc_file(filename)
                else:
                    # Let user describe how to interpret amp/phase text files
                    dlg = PulseImportDialog(self, filename)
                    if dlg.exec_() != QDialog.Accepted:
                        return
                    opts = dlg.get_options()
                    from ..pulse_loader import load_amp_phase_dat

                    b1, time, metadata = load_amp_phase_dat(
                        filename,
                        duration_s=opts["duration_s"],
                        amplitude_unit=opts["amp_unit"],
                        phase_unit=opts["phase_unit"],
                        layout=opts["layout"],
                    )

                # Store loaded data
                self.loaded_pulse_b1 = b1.copy()
                self.loaded_pulse_time = time.copy()
                self.loaded_pulse_metadata = metadata

                # Get basic info
                duration_ms = (
                    metadata.duration * 1000.0
                    if metadata.duration > 0
                    else time[-1] * 1000.0
                )
                max_b1 = metadata.max_b1 if metadata.max_b1 > 0 else np.max(np.abs(b1))

                # Update UI
                self._syncing = True  # Prevent intermediate updates
                try:
                    self.pulse_type.setCurrentText("Custom")
                    self.duration.setValue(duration_ms)
                    self.b1_amplitude.setValue(0.0)  # Reset to Auto
                    self.flip_angle.setValue(
                        metadata.flip_angle if metadata.flip_angle > 0 else 90.0
                    )
                finally:
                    self._syncing = False

                # Update info label
                tbw_hint = None
                try:
                    if (
                        hasattr(metadata, "bwfac")
                        and metadata.bwfac is not None
                        and np.isfinite(metadata.bwfac)
                        and metadata.bwfac > 0
                    ):
                        tbw_hint = float(metadata.bwfac)
                    elif hasattr(metadata, "integfac") and metadata.integfac not in (
                        None,
                        0,
                    ):
                        if np.isfinite(metadata.integfac) and metadata.integfac > 0:
                            integfac = float(metadata.integfac)
                            tbw_hint = 1.0 / integfac
                except Exception:
                    pass

                tbw_text = f", TBW≈{tbw_hint:.3f}" if tbw_hint else ""
                self.custom_info_label.setText(
                    f"Original: {duration_ms:.3f} ms, {max_b1:.6f} G{tbw_text}"
                )
                self.custom_info_label.setVisible(True)

                # Force update to process the pulse (resample/scale)
                self.update_pulse()

                # Show info message
                QMessageBox.information(
                    self,
                    "Pulse Loaded",
                    f"Successfully loaded pulse from:\n{filename}\n\n"
                    f"Flip angle: {metadata.flip_angle}°\n"
                    f"Duration: {duration_ms:.3f} ms\n"
                    f"Points: {len(b1)}\n"
                    f"Max B1: {max_b1:.6f} Gauss",
                )

            except Exception as e:
                QMessageBox.critical(
                    self, "Error Loading Pulse", f"Failed to load pulse file:\n{str(e)}"
                )

    def get_state(self) -> dict:
        """Get the current UI state of the pulse designer."""
        state = {
            "pulse_type": self.pulse_type.currentText(),
            "flip_angle": self.flip_angle.value(),
            "duration": self.duration.value(),
            "b1_amplitude": self.b1_amplitude.value(),
            "phase": self.phase.value(),
            "freq_offset": self.freq_offset.value(),
            "sinc_lobes": self.sinc_lobes.value(),
            "time_bandwidth_product": self.tbw.value(),
            "slr_sharpness": self.slr_sharpness.value(),
            "apodization": self.apodization_combo.currentText(),
        }
        # Include loaded pulse data
        state["loaded_pulse_b1"] = self.loaded_pulse_b1
        state["loaded_pulse_time"] = self.loaded_pulse_time
        state["loaded_pulse_metadata"] = getattr(self, "loaded_pulse_metadata", None)
        return state

    def set_state(self, state: dict):
        """Restore the UI state."""
        if not state or self._syncing:
            return

        self._syncing = True
        try:
            # Block signals to prevent intermediate updates
            self.pulse_type.blockSignals(True)
            self.flip_angle.blockSignals(True)
            self.duration.blockSignals(True)
            self.b1_amplitude.blockSignals(True)
            self.phase.blockSignals(True)
            self.freq_offset.blockSignals(True)
            self.sinc_lobes.blockSignals(True)
            self.design_tbw.blockSignals(True)
            self.slr_sharpness.blockSignals(True)
            self.apodization_combo.blockSignals(True)

            try:
                if "pulse_type" in state:
                    self.pulse_type.setCurrentText(state["pulse_type"])
                if "flip_angle" in state:
                    self.flip_angle.setValue(state["flip_angle"])
                if "duration" in state:
                    self.duration.setValue(state["duration"])
                if "b1_amplitude" in state:
                    self.b1_amplitude.setValue(state["b1_amplitude"])
                if "phase" in state:
                    self.phase.setValue(state["phase"])
                if "freq_offset" in state:
                    self.freq_offset.setValue(state["freq_offset"])
                if "sinc_lobes" in state:
                    self.sinc_lobes.setValue(state["sinc_lobes"])
                if "slr_sharpness" in state:
                    self.slr_sharpness.setValue(state["slr_sharpness"])
                if "apodization" in state:
                    self.apodization_combo.setCurrentText(state["apodization"])

                # Restore loaded data
                if "loaded_pulse_b1" in state:
                    self.loaded_pulse_b1 = state["loaded_pulse_b1"]
                if "loaded_pulse_time" in state:
                    self.loaded_pulse_time = state["loaded_pulse_time"]
                if "loaded_pulse_metadata" in state:
                    self.loaded_pulse_metadata = state["loaded_pulse_metadata"]
            finally:
                self.pulse_type.blockSignals(False)
                self.flip_angle.blockSignals(False)
                self.duration.blockSignals(False)
                self.b1_amplitude.blockSignals(False)
                self.phase.blockSignals(False)
                self.freq_offset.blockSignals(False)
                self.sinc_lobes.blockSignals(False)
                self.design_tbw.blockSignals(False)
                self.slr_sharpness.blockSignals(False)
                self.apodization_combo.blockSignals(False)

            # Trigger update once
            self.update_pulse()
        finally:
            self._syncing = False
