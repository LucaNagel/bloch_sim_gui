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
)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
import pyqtgraph as pg
from pathlib import Path
from ..simulator import design_rf_pulse
from .dialogs import PulseImportDialog


class RFPulseDesigner(QGroupBox):
    """Widget for designing RF pulses."""

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
            control_layout = main_layout
            control_panel = None  # No separate panel container
        else:
            # Horizontal split for main tab
            main_layout = QHBoxLayout()
            control_panel = QWidget()
            control_layout = QVBoxLayout()
            control_panel.setLayout(control_layout)
            control_panel.setMaximumWidth(400)
            main_layout.addWidget(control_panel)

        # Pulse type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Pulse Type:"))
        self.pulse_type = QComboBox()
        prefix = "rf_compact_" if self.compact else "rf_tab_"
        self.pulse_type.setObjectName(f"{prefix}pulse_type")
        self.pulse_type.addItems(
            [
                "Rectangle",
                "Sinc",
                "Gaussian",
                "Hermite",
                "Adiabatic Half Passage",
                "Adiabatic Full Passage",
                "BIR-4",
                "Custom",
            ]
        )
        self.pulse_type.currentTextChanged.connect(self.update_pulse)
        type_layout.addWidget(self.pulse_type)
        control_layout.addLayout(type_layout)

        # Flip angle
        flip_layout = QHBoxLayout()
        flip_layout.addWidget(QLabel("Flip Angle (°):"))
        self.flip_angle = QDoubleSpinBox()
        self.flip_angle.setObjectName(f"{prefix}flip_angle")
        self.flip_angle.setRange(0, 1e4)
        self.flip_angle.setValue(90)
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
            "Set > 0 to override B1 amplitude. 0 = Auto (derive from Flip Angle)."
        )
        self.b1_amplitude.valueChanged.connect(self.update_pulse)
        b1_layout.addWidget(self.b1_amplitude)
        control_layout.addLayout(b1_layout)

        # Time-bandwidth product (computed from pulse shape; not user-set)
        tbw_layout = QHBoxLayout()
        tbw_layout.addWidget(QLabel("Time-BW Product (auto):"))
        self.tbw = QDoubleSpinBox()
        self.tbw.setObjectName(f"{prefix}tbw")
        self.tbw.setRange(0.001, 1000)
        self.tbw.setValue(1)
        self.tbw.setSingleStep(0.5)
        self.tbw.setReadOnly(True)
        self.tbw.setButtonSymbols(QDoubleSpinBox.NoButtons)
        tbw_layout.addWidget(self.tbw)
        control_layout.addLayout(tbw_layout)
        self.tbw_auto_label = QLabel("Auto TBW (≈1/integfac): —")
        self.tbw_auto_label.setStyleSheet("color: gray;")
        control_layout.addWidget(self.tbw_auto_label)

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
        self.phase.setRange(0, 360)
        self.phase.setValue(0)
        self.phase.valueChanged.connect(self.update_pulse)
        phase_layout.addWidget(self.phase)
        control_layout.addLayout(phase_layout)

        # RF Frequency Offset
        freq_offset_layout = QHBoxLayout()
        freq_offset_layout.addWidget(QLabel("RF Frequency Offset (Hz):"))
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
            main_layout.addLayout(plot_layout, stretch=1)

        self.setLayout(main_layout)

        # Storage for loaded pulse data
        self.loaded_pulse_b1 = None
        self.loaded_pulse_time = None
        self.loaded_pulse_metadata = None

        # Initial pulse
        self.update_pulse()

    def _update_tbw_auto(self, integration_factor: float):
        """Set TBW readout from an integration factor (heuristic: TBW ≈ 1/integfac)."""
        if not hasattr(self, "tbw") or not hasattr(self, "tbw_auto_label"):
            return
        if (
            integration_factor is None
            or not np.isfinite(integration_factor)
            or integration_factor <= 0
        ):
            self.tbw_auto_label.setText("Auto TBW (≈1/integfac): —")
            self.last_integration_factor = 1.0
            return
        tbw_auto = 1.0 / integration_factor
        self.tbw_auto_label.setText(f"Auto TBW (≈1/integfac): {tbw_auto:.3f}")
        self.last_integration_factor = float(integration_factor)
        # Keep the control in sync without retriggering pulse design
        self.tbw.blockSignals(True)
        self.tbw.setValue(tbw_auto)
        self.tbw.blockSignals(False)

    def _design_tbw_for_type(self, pulse_type: str) -> float:
        """Return a canonical TBW parameter for the designer (not user-controlled)."""
        pt = pulse_type.lower()
        if pt in ("sinc", "gaussian"):
            return 4.0  # typical shaping parameter
        if pt.startswith("adiabatic") or pt in ("bir-4", "bir4"):
            return 4.0  # modulation parameter for adiabatic-style pulses
        return 1.0  # rectangular and default

    def _compute_integration_factor_from_wave(self, b1_wave, t_wave):
        """Compute integration factor |∫shape dt| / duration for a given complex waveform."""
        try:
            b1_wave = np.asarray(b1_wave, dtype=complex)
            t_wave = np.asarray(t_wave, dtype=float)
            if b1_wave.size < 2 or t_wave.size < 2:
                return 1.0
            duration = float(t_wave[-1] - t_wave[0])
            dt = float(np.median(np.diff(t_wave)))
            peak = np.max(np.abs(b1_wave)) if np.any(np.abs(b1_wave)) else 1.0
            shape = b1_wave / peak if peak != 0 else b1_wave
            area = np.trapezoid(shape, dx=dt)
            aligned = np.real(area * np.exp(-1j * np.angle(area)))
            if not np.isfinite(aligned) or abs(aligned) < 1e-12:
                return 1.0
            return abs(aligned) / max(duration, 1e-12)
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

    def _apply_phase_and_offset(self, b1_wave, t_wave):
        """Apply user-selected phase and frequency offset to a waveform."""
        b1_wave = np.asarray(b1_wave, dtype=complex)
        t_wave = np.asarray(t_wave, dtype=float)
        if b1_wave.shape != t_wave.shape:
            # Allow time to be length N while b1 is length N
            pass
        phase_rad = np.deg2rad(self.phase.value())
        freq_hz = self.freq_offset.value()
        if t_wave.size > 0:
            t_rel = t_wave - t_wave[0]
        else:
            t_rel = t_wave
        # Apply global phase and complex modulation for frequency offset
        return b1_wave * np.exp(1j * (phase_rad + 2 * np.pi * freq_hz * t_rel))

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

        # Update explanation
        desc_map = {
            "rectangle": "<b>Rectangular Pulse</b><br>Constant amplitude hard pulse. Broad excitation bandwidth.",
            "sinc": "<b>Sinc Pulse</b><br>Selective excitation. Fourier transform of a rectangular slice profile. Use 'Lobes' to control bandwidth/sharpness.",
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
        self.custom_info_label.setVisible(pulse_type == "custom")

        duration = self.duration.value() / 1000  # Convert to seconds
        flip = self.flip_angle.value()
        b1_override = self.b1_amplitude.value()
        freq_offset_hz = self.freq_offset.value()
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

            # Get integration factor for TBW display
            integfac = 1.0
            if (
                self.loaded_pulse_metadata
                and hasattr(self.loaded_pulse_metadata, "integfac")
                and self.loaded_pulse_metadata.integfac > 0
            ):
                integfac = float(self.loaded_pulse_metadata.integfac)
            else:
                # Recompute
                integfac = self._compute_integration_factor_from_wave(b1, time)

            self._update_tbw_auto(integfac)
            self.last_integration_factor = float(integfac)

            # Amplitude scaling: B1 override vs Flip Angle
            if b1_override > 0:
                # Manual B1 override
                # Scale shape so peak matches b1_override
                b1 = shape * b1_override
            else:
                # Auto (Flip Angle)
                b1 = self._scale_pulse_to_flip(b1, time, flip, integfac=integfac)

            # Apply Phase and Frequency Offset
            # Note: _apply_phase_and_offset handles self.phase and self.freq_offset internally
            # but we extracted them above. Let's use the helper or manual.
            # Helper uses self.phase/freq_offset.value() directly.
            b1 = self._apply_phase_and_offset(b1, time)

            self.current_pulse = (b1, time)
            self.pulse_changed.emit(self.current_pulse)

            if not self._syncing:
                self.parameters_changed.emit(self.get_state())

            self._update_plot(b1, time)
            return

        # Handle Standard Pulses
        # Calculate TBW based on pulse type
        if pulse_type == "sinc":
            design_tbw = float(self.sinc_lobes.value()) + 1.0
        else:
            design_tbw = self._design_tbw_for_type(pulse_type)

        # Target point count
        if self.target_dt and self.target_dt > 0:
            npoints = max(32, int(np.ceil(duration / self.target_dt)))
            npoints = min(npoints, 50000)
        else:
            npoints = 100

        # 1. Generate base pulse
        b1_base, time = design_rf_pulse(
            pulse_type, duration, flip, design_tbw, npoints, freq_offset=0.0
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
        integration_factor = abs(aligned_area) / max(duration, 1e-12)

        self._update_tbw_auto(integration_factor)
        self.last_integration_factor = float(integration_factor)

        # Amplitude scaling
        if b1_override > 0:
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

        # Frequency Offset
        if freq_offset_hz != 0.0:
            mod = np.exp(2j * np.pi * freq_offset_hz * time)
            b1 = b1 * mod

        self.current_pulse = (b1, time)
        self.pulse_changed.emit(self.current_pulse)

        if not self._syncing:
            self.parameters_changed.emit(self.get_state())

        self._update_plot(b1, time)

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
                    if hasattr(metadata, "integfac") and metadata.integfac not in (
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
                self.apodization_combo.blockSignals(False)

            # Trigger update once
            self.update_pulse()
        finally:
            self._syncing = False
