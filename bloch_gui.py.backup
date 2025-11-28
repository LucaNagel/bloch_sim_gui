"""
bloch_gui.py - Interactive GUI for Bloch equation simulator

This module provides a graphical user interface for designing pulse sequences,
setting parameters, running simulations, and visualizing results.

Author: Your Name
Date: 2024
"""

import sys
import math
import numpy as np
from typing import Optional
from pathlib import Path
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel, QLineEdit, QComboBox, QSlider,
    QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QTabWidget, QTextEdit, QSplitter,
    QProgressBar, QCheckBox, QRadioButton, QButtonGroup, QScrollArea,
    QSizePolicy, QMenu, QDialog, QProgressDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QImage

import pyqtgraph as pg
import pyqtgraph.opengl as gl

# Import the simulator
from bloch_simulator import (
    BlochSimulator, TissueParameters,
    SpinEcho, SpinEchoTipAxis, GradientEcho, SliceSelectRephase, CustomPulse, PulseSequence, design_rf_pulse
)

# Import visualization export tools
from visualization_export import (
    ImageExporter,
    ExportImageDialog,
    AnimationExporter,
    ExportAnimationDialog,
    DatasetExporter,
    imageio as vz_imageio,
)


class SimulationThread(QThread):
    """Thread for running simulations without blocking the GUI."""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    cancelled = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, simulator, sequence, tissue, positions, frequencies, mode, dt=1e-6, m_init=None):
        super().__init__()
        self.simulator = simulator
        self.sequence = sequence
        self.tissue = tissue
        self.positions = positions
        self.frequencies = frequencies
        self.mode = mode
        self.dt = dt
        self.m_init = m_init
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True
        
    def run(self):
        """Run the simulation."""
        try:
            if self._cancel_requested:
                self.cancelled.emit()
                return
            result = self.simulator.simulate(
                self.sequence,
                self.tissue,
                self.positions,
                self.frequencies,
                initial_magnetization=self.m_init,
                mode=self.mode,
                dt=self.dt
            )
            if self._cancel_requested:
                self.cancelled.emit()
                return
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class TissueParameterWidget(QGroupBox):
    """Widget for setting tissue parameters."""

    def __init__(self):
        super().__init__("Tissue Parameters")
        self.sequence_presets_enabled = True  # Default: auto-load presets
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Preset selector
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom", "Gray Matter", "White Matter", "CSF",
            "Muscle", "Fat", "Blood", "Liver",
            "Hyperpolarized 13C Pyruvate"
        ])
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        preset_layout.addWidget(self.preset_combo)

        # Field strength
        preset_layout.addWidget(QLabel("Field:"))
        self.field_combo = QComboBox()
        self.field_combo.addItems(["1.5T", "3.0T", "7.0T"])
        self.field_combo.setCurrentText("3.0T")
        self.field_combo.currentTextChanged.connect(self.load_preset)
        preset_layout.addWidget(self.field_combo)
        layout.addLayout(preset_layout)

        # Sequence-specific presets toggle
        seq_preset_layout = QHBoxLayout()
        self.seq_preset_checkbox = QCheckBox("Auto-load sequence presets")
        self.seq_preset_checkbox.setChecked(True)
        self.seq_preset_checkbox.setToolTip("Automatically load TE/TR/TI presets when sequence changes")
        self.seq_preset_checkbox.toggled.connect(self._toggle_sequence_presets)
        seq_preset_layout.addWidget(self.seq_preset_checkbox)
        layout.addLayout(seq_preset_layout)
        
        # T1 parameter
        t1_layout = QHBoxLayout()
        t1_layout.addWidget(QLabel("T1 (ms):"))
        self.t1_spin = QDoubleSpinBox()
        self.t1_spin.setRange(1, 5000)
        self.t1_spin.setValue(1000)
        self.t1_spin.setSuffix(" ms")
        t1_layout.addWidget(self.t1_spin)
        
        self.t1_slider = QSlider(Qt.Horizontal)
        self.t1_slider.setRange(1, 5000)
        self.t1_slider.setValue(1000)
        self.t1_slider.valueChanged.connect(lambda v: self.t1_spin.setValue(v))
        self.t1_spin.valueChanged.connect(lambda v: self.t1_slider.setValue(int(v)))
        t1_layout.addWidget(self.t1_slider)
        layout.addLayout(t1_layout)
        
        # T2 parameter
        t2_layout = QHBoxLayout()
        t2_layout.addWidget(QLabel("T2 (ms):"))
        self.t2_spin = QDoubleSpinBox()
        self.t2_spin.setRange(1, 2000)
        self.t2_spin.setValue(100)
        self.t2_spin.setSuffix(" ms")
        t2_layout.addWidget(self.t2_spin)
        
        self.t2_slider = QSlider(Qt.Horizontal)
        self.t2_slider.setRange(1, 2000)
        self.t2_slider.setValue(100)
        self.t2_slider.valueChanged.connect(lambda v: self.t2_spin.setValue(v))
        self.t2_spin.valueChanged.connect(lambda v: self.t2_slider.setValue(int(v)))
        t2_layout.addWidget(self.t2_slider)
        layout.addLayout(t2_layout)
        
        # T2* parameter
        t2s_layout = QHBoxLayout()
        t2s_layout.addWidget(QLabel("T2* (ms):"))
        self.t2s_spin = QDoubleSpinBox()
        self.t2s_spin.setRange(1, 200)
        self.t2s_spin.setValue(50)
        self.t2s_spin.setSuffix(" ms")
        t2s_layout.addWidget(self.t2s_spin)
        layout.addLayout(t2s_layout)
        
        # Proton density
        pd_layout = QHBoxLayout()
        pd_layout.addWidget(QLabel("Proton Density:"))
        self.pd_spin = QDoubleSpinBox()
        self.pd_spin.setRange(0, 1)
        self.pd_spin.setSingleStep(0.1)
        self.pd_spin.setValue(1.0)
        pd_layout.addWidget(self.pd_spin)
        layout.addLayout(pd_layout)

        # Initial magnetization (Mz)
        m0_layout = QHBoxLayout()
        m0_layout.addWidget(QLabel("Initial Mz:"))
        self.m0_spin = QDoubleSpinBox()
        self.m0_spin.setRange(-1e9, 1e9)
        self.m0_spin.setDecimals(3)
        self.m0_spin.setValue(1.0)
        m0_layout.addWidget(self.m0_spin)
        layout.addLayout(m0_layout)
        
        self.setLayout(layout)
        
    def load_preset(self):
        """Load tissue parameter preset."""
        preset = self.preset_combo.currentText()
        field_str = self.field_combo.currentText()
        field = float(field_str[:-1])  # Remove 'T'
        
        if preset == "Gray Matter":
            tissue = TissueParameters.gray_matter(field)
        elif preset == "White Matter":
            tissue = TissueParameters.white_matter(field)
        elif preset == "CSF":
            tissue = TissueParameters.csf(field)
        elif preset == "Hyperpolarized 13C Pyruvate":
            # Typical HP 13C pyruvate values (approx.): long T1, slower decay
            self.t1_spin.setValue(60000)  # 60 s
            self.t2_spin.setValue(1000)   # 1 s
            self.t2s_spin.setValue(1000)
            self.pd_spin.setValue(1.0)
            self.m0_spin.setValue(100000)
            return
        else:
            return  # Keep custom values
            
        self.t1_spin.setValue(tissue.t1 * 1000)  # Convert to ms
        self.t2_spin.setValue(tissue.t2 * 1000)  # Convert to ms
        self.m0_spin.setValue(1.0)
        
    def get_parameters(self) -> TissueParameters:
        """Get current tissue parameters."""
        return TissueParameters(
            name=self.preset_combo.currentText(),
            t1=self.t1_spin.value() / 1000,  # Convert to seconds
            t2=self.t2_spin.value() / 1000,  # Convert to seconds
            t2_star=self.t2s_spin.value() / 1000,
            density=self.pd_spin.value()
        )

    def get_initial_mz(self) -> float:
        """Return the initial longitudinal magnetization."""
        return float(self.m0_spin.value())

    def _toggle_sequence_presets(self, enabled: bool):
        """Toggle automatic loading of sequence presets."""
        self.sequence_presets_enabled = enabled


class RFPulseDesigner(QGroupBox):
    """Widget for designing RF pulses."""

    pulse_changed = pyqtSignal(object)
    
    def __init__(self):
        super().__init__("RF Pulse Design")
        self.target_dt = 5e-6  # default 5 us
        self.init_ui()
        self.current_pulse = None
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Pulse type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Pulse Type:"))
        self.pulse_type = QComboBox()
        self.pulse_type.addItems(["Rectangle", "Sinc", "Gaussian", "Hermite", "Custom"])
        self.pulse_type.currentTextChanged.connect(self.update_pulse)
        type_layout.addWidget(self.pulse_type)
        layout.addLayout(type_layout)
        
        # Flip angle
        flip_layout = QHBoxLayout()
        flip_layout.addWidget(QLabel("Flip Angle (°):"))
        self.flip_angle = QDoubleSpinBox()
        self.flip_angle.setRange(0, 180)
        self.flip_angle.setValue(90)
        self.flip_angle.valueChanged.connect(self.update_pulse)
        flip_layout.addWidget(self.flip_angle)
        layout.addLayout(flip_layout)
        
        # Duration
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Duration (ms):"))
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.1, 10)
        self.duration.setValue(1.0)
        self.duration.setSingleStep(0.1)
        self.duration.valueChanged.connect(self.update_pulse)
        duration_layout.addWidget(self.duration)
        layout.addLayout(duration_layout)
        
        # Time-bandwidth product
        tbw_layout = QHBoxLayout()
        tbw_layout.addWidget(QLabel("Time-BW Product:"))
        self.tbw = QDoubleSpinBox()
        self.tbw.setRange(1, 10)
        self.tbw.setValue(4)
        self.tbw.setSingleStep(0.5)
        self.tbw.valueChanged.connect(self.update_pulse)
        tbw_layout.addWidget(self.tbw)
        layout.addLayout(tbw_layout)
        
        # Phase
        phase_layout = QHBoxLayout()
        phase_layout.addWidget(QLabel("Phase (°):"))
        self.phase = QDoubleSpinBox()
        self.phase.setRange(0, 360)
        self.phase.setValue(0)
        self.phase.valueChanged.connect(self.update_pulse)
        phase_layout.addWidget(self.phase)
        layout.addLayout(phase_layout)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'B1 Amplitude', 'G')
        self.plot_widget.setLabel('bottom', 'Time', 'ms')
        self.plot_widget.setMinimumHeight(200)
        layout.addWidget(self.plot_widget)
        
        # Custom pulse settings (initially hidden)
        self.custom_settings_group = QGroupBox("Custom Pulse Settings")
        custom_settings_layout = QVBoxLayout()

        # Duration override
        duration_custom_layout = QHBoxLayout()
        duration_custom_layout.addWidget(QLabel("Duration (ms):"))
        self.custom_duration = QDoubleSpinBox()
        self.custom_duration.setRange(0.01, 100.0)
        self.custom_duration.setValue(1.0)
        self.custom_duration.setSingleStep(0.1)
        self.custom_duration.setDecimals(3)
        self.custom_duration.valueChanged.connect(self.reprocess_custom_pulse)
        duration_custom_layout.addWidget(self.custom_duration)
        custom_settings_layout.addLayout(duration_custom_layout)

        # B1 amplitude override (0 = use flip angle calibration)
        b1_custom_layout = QHBoxLayout()
        b1_custom_layout.addWidget(QLabel("B1 Amplitude (G):"))
        self.custom_b1_amplitude = QDoubleSpinBox()
        self.custom_b1_amplitude.setRange(0.0, 1e3)
        self.custom_b1_amplitude.setValue(0.0)
        self.custom_b1_amplitude.setSingleStep(0.001)
        self.custom_b1_amplitude.setDecimals(6)
        self.custom_b1_amplitude.setSpecialValueText("Auto (from flip angle)")
        self.custom_b1_amplitude.valueChanged.connect(self.reprocess_custom_pulse)
        b1_custom_layout.addWidget(self.custom_b1_amplitude)
        custom_settings_layout.addLayout(b1_custom_layout)

        # Info label
        self.custom_info_label = QLabel("Original: 1.000 ms, 0.000100 G")
        self.custom_info_label.setStyleSheet("color: gray; font-size: 9pt;")
        custom_settings_layout.addWidget(self.custom_info_label)

        self.custom_settings_group.setLayout(custom_settings_layout)
        self.custom_settings_group.setVisible(False)
        layout.addWidget(self.custom_settings_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load from File")
        self.load_button.clicked.connect(self.load_pulse_from_file)
        self.save_button = QPushButton("Save to File")
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Storage for loaded pulse data
        self.loaded_pulse_b1 = None
        self.loaded_pulse_time = None
        self.loaded_pulse_metadata = None

        # Initial pulse
        self.update_pulse()
        
    def update_pulse(self):
        """Update the RF pulse based on current parameters."""
        pulse_type = self.pulse_type.currentText().lower()
        if pulse_type == 'rectangle':
            pulse_type = 'rect'
        elif pulse_type == 'custom':
            # For custom pulses, trigger reprocessing if data is loaded
            if self.loaded_pulse_b1 is not None:
                self.reprocess_custom_pulse()
            return

        # Hide custom settings when not using custom pulse
        self.custom_settings_group.setVisible(False)
            
        duration = self.duration.value() / 1000  # Convert to seconds
        flip = self.flip_angle.value()
        tbw = self.tbw.value()
        phase_rad = np.deg2rad(self.phase.value())
        # Increase time resolution using desired dt
        if self.target_dt and self.target_dt > 0:
            npoints = max(32, int(np.ceil(duration / self.target_dt)))
            npoints = min(npoints, 50000)  # avoid runaway
        else:
            npoints = 100
        
        # Design pulse
        b1, time = design_rf_pulse(pulse_type, duration, flip, tbw, npoints)
        # Sampling time [s]
        dt = duration / len(b1)

        # Normalize pulse shape to unit peak for amplitude scaling
        peak = np.max(np.abs(b1)) if np.any(np.abs(b1)) else 1.0
        shape = b1 / peak

        # Find a global phase that maximizes the real-valued area of the waveform.
        # This keeps the flip-angle calibration stable even for complex pulses.
        area = np.trapezoid(shape, dx=dt)
        opt_phase = -np.angle(area) if np.isfinite(area) else 0.0
        aligned_area = np.real(area * np.exp(1j * opt_phase))
        if not np.isfinite(aligned_area) or abs(aligned_area) < 1e-12:
            aligned_area = 1e-12

        # flip angle in radians:
        flip_rad = np.deg2rad(flip)

        # Gyromagnetic ratio for 1H (rad / (T s)):
        gmr_1h_rad_Ts = 267522187.43999997

        # Calculate required pulse amplitude in Tesla:
        pulse_amp_T = flip_rad / (gmr_1h_rad_Ts * aligned_area)
        # Convert to Gauss:
        pulse_amp_G = pulse_amp_T * 1e4  # Gauss

        # Apply the optimal calibration phase plus any user-selected phase
        total_phase = opt_phase + phase_rad
        b1 = shape * pulse_amp_G * np.exp(1j * total_phase)
        
        # print(f"b1 = {b1}")
        
        # # Re-normalize area to requested flip angle (protects against backend differences)
        # if len(time) > 1 and np.any(np.abs(b1) > 0):
        #     target_area = np.deg2rad(flip) / (4257.0 * 2 * np.pi)
        #     area = np.trapezoid(np.abs(b1), time)
        #     if area > 0:
        #         b1 = b1 * (target_area / area)
                
        # print(f"b1 = {b1}")
                
                    

        self.current_pulse = (b1, time)
        self.pulse_changed.emit(self.current_pulse)
        
        # Update plot
        self.plot_widget.clear()
        # design_rf_pulse returns B1 in Gauss already; plot directly
        self.plot_widget.plot(time * 1000, np.abs(b1), 
                            pen='b', name='Magnitude')
        self.plot_widget.plot(time * 1000, np.real(b1), 
                            pen='r', name='Real')
        self.plot_widget.plot(time * 1000, np.imag(b1), 
                            pen='g', name='Imaginary')
        if len(time):
            t_min, t_max = 0, time[-1] * 1000
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
            "Bruker Pulse Files (*.exc);;All Files (*)"
        )
        if filename:
            try:
                from pulse_loader import load_pulse_from_file as load_exc_file
                b1, time, metadata = load_exc_file(filename)

                # Store the original loaded pulse data
                self.loaded_pulse_b1 = b1.copy()
                self.loaded_pulse_time = time.copy()
                self.loaded_pulse_metadata = metadata

                # Initialize custom settings from loaded pulse
                duration_ms = metadata.duration * 1000.0 if metadata.duration > 0 else time[-1] * 1000.0
                max_b1 = metadata.max_b1 if metadata.max_b1 > 0 else np.max(np.abs(b1))

                # Set custom settings to match loaded pulse
                self.custom_duration.blockSignals(True)
                self.custom_b1_amplitude.blockSignals(True)
                self.custom_duration.setValue(duration_ms)
                self.custom_b1_amplitude.setValue(0.0)  # Start with auto
                self.custom_duration.blockSignals(False)
                self.custom_b1_amplitude.blockSignals(False)

                # Update info label
                self.custom_info_label.setText(
                    f"Original: {metadata.duration*1000:.3f} ms, {max_b1:.6f} G"
                )

                # Show custom settings panel
                self.custom_settings_group.setVisible(True)

                # Store the loaded pulse as current
                self.current_pulse = (b1, time)

                # Update plot
                self.plot_widget.clear()
                self.plot_widget.plot(time * 1000, np.abs(b1),
                                    pen='b', name='Magnitude')
                self.plot_widget.plot(time * 1000, np.real(b1),
                                    pen='r', name='Real')
                self.plot_widget.plot(time * 1000, np.imag(b1),
                                    pen='g', name='Imaginary')
                if len(time):
                    t_min, t_max = 0, time[-1] * 1000
                    self.plot_widget.setLimits(xMin=0, xMax=max(t_max, 0.1))
                    self.plot_widget.setXRange(0, max(t_max, 0.1), padding=0)

                # Update UI to show loaded pulse info
                self.pulse_type.blockSignals(True)
                self.pulse_type.setCurrentText("Custom")
                self.pulse_type.blockSignals(False)

                # Show info message
                QMessageBox.information(
                    self,
                    "Pulse Loaded",
                    f"Successfully loaded pulse from:\n{filename}\n\n"
                    f"Flip angle: {metadata.flip_angle}°\n"
                    f"Duration: {metadata.duration*1000:.3f} ms\n"
                    f"Points: {len(b1)}\n"
                    f"Max B1: {metadata.max_b1:.6f} Gauss\n\n"
                    f"You can now adjust duration and B1 amplitude in the Custom Pulse Settings panel."
                )

                # Emit signal that pulse changed
                self.pulse_changed.emit(self.current_pulse)

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Pulse",
                    f"Failed to load pulse file:\n{str(e)}"
                )

    def reprocess_custom_pulse(self):
        """Reprocess the loaded custom pulse with new duration/amplitude settings."""
        if self.loaded_pulse_b1 is None or self.loaded_pulse_time is None:
            return

        # Get new settings
        new_duration_ms = self.custom_duration.value()
        new_b1_amplitude = self.custom_b1_amplitude.value()

        # Original pulse data
        original_b1 = self.loaded_pulse_b1
        original_time = self.loaded_pulse_time
        original_duration = original_time[-1] - original_time[0]

        # Calculate time scaling factor
        new_duration_s = new_duration_ms / 1000.0
        time_scale = new_duration_s / original_duration if original_duration > 0 else 1.0

        # Rescale time array
        new_time = original_time * time_scale

        # Resample B1 if needed (for now, just use the original samples with scaled time)
        new_b1 = original_b1.copy()

        # Apply B1 amplitude override
        if new_b1_amplitude > 0:
            # Use specified amplitude - scale to match target
            current_max_b1 = np.max(np.abs(original_b1))
            if current_max_b1 > 0:
                b1_scale = new_b1_amplitude / current_max_b1
                new_b1 = original_b1 * b1_scale
        else:
            # Use flip angle calibration (auto mode)
            # Recalculate B1 amplitude to achieve target flip angle
            flip = self.flip_angle.value()  # degrees
            flip_rad = np.deg2rad(flip)

            # Normalize pulse shape to unit peak
            peak = np.max(np.abs(original_b1)) if np.any(np.abs(original_b1)) else 1.0
            shape = original_b1 / peak

            # Calculate sampling time for new duration
            dt = new_duration_s / len(shape) if len(shape) > 0 else 1e-6

            # Find optimal phase that maximizes real-valued area
            area = np.trapezoid (shape, dx=dt)
            opt_phase = -np.angle(area) if np.isfinite(area) and area != 0 else 0.0
            aligned_area = np.real(area * np.exp(1j * opt_phase))
            if not np.isfinite(aligned_area) or abs(aligned_area) < 1e-12:
                aligned_area = 1e-12

            # Gyromagnetic ratio for 1H (rad / (T s))
            gmr_1h_rad_Ts = 267522187.43999997

            # Calculate required pulse amplitude in Tesla
            pulse_amp_T = flip_rad / (gmr_1h_rad_Ts * aligned_area)
            # Convert to Gauss
            pulse_amp_G = pulse_amp_T * 1e4

            # Apply amplitude and phase to achieve target flip angle
            new_b1 = shape * pulse_amp_G * np.exp(1j * opt_phase)

        # Update current pulse
        self.current_pulse = (new_b1, new_time)

        # Update plot
        self.plot_widget.clear()
        self.plot_widget.plot(new_time * 1000, np.abs(new_b1),
                            pen='b', name='Magnitude')
        self.plot_widget.plot(new_time * 1000, np.real(new_b1),
                            pen='r', name='Real')
        self.plot_widget.plot(new_time * 1000, np.imag(new_b1),
                            pen='g', name='Imaginary')
        if len(new_time):
            t_min, t_max = 0, new_time[-1] * 1000
            self.plot_widget.setLimits(xMin=0, xMax=max(t_max, 0.1))
            self.plot_widget.setXRange(0, max(t_max, 0.1), padding=0)

        # Emit signal that pulse changed
        self.pulse_changed.emit(self.current_pulse)


class SequenceDesigner(QGroupBox):
    """Widget for designing pulse sequences."""

    def __init__(self):
        super().__init__("Sequence Design")
        self.default_dt = 1e-6  # 1 us
        self.custom_pulse = None
        self.playhead_line = None
        self.diagram_labels = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Sequence type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Sequence:"))
        self.sequence_type = QComboBox()
        self.sequence_type.addItems([
            "Free Induction Decay",
            "Spin Echo", 
            "Spin Echo (Tip-axis 180)",
            "Gradient Echo",
            "Slice Select + Rephase",
            "SSFP (Loop)",
            "Inversion Recovery",
            "FLASH",
            "EPI",
            "Custom"
        ])
        type_layout.addWidget(self.sequence_type)
        layout.addLayout(type_layout)
        self.sequence_type.currentTextChanged.connect(self.update_diagram)
        self.sequence_type.currentTextChanged.connect(self._update_sequence_options)

        # Sequence-specific options (shown/hidden per selection)
        self.options_container = QVBoxLayout()
        self.options_container.setContentsMargins(0, 0, 0, 0)
        self.spin_echo_opts = QWidget()
        se_layout = QHBoxLayout()
        se_layout.addWidget(QLabel("Echoes:"))
        self.spin_echo_echoes = QSpinBox()
        self.spin_echo_echoes.setRange(1, 128)
        self.spin_echo_echoes.setValue(1)
        self.spin_echo_echoes.valueChanged.connect(lambda _: self.update_diagram())
        se_layout.addWidget(self.spin_echo_echoes)
        self.spin_echo_opts.setLayout(se_layout)
        self.options_container.addWidget(self.spin_echo_opts)
        self.ssfp_opts = QWidget()
        ssfp_layout = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("SSFP repeats:"))
        self.ssfp_repeats = QSpinBox()
        self.ssfp_repeats.setRange(1, 10000)
        self.ssfp_repeats.setValue(16)
        self.ssfp_repeats.valueChanged.connect(lambda _: self.update_diagram())
        row1.addWidget(self.ssfp_repeats)
        ssfp_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Pulse amp (G):"))
        self.ssfp_amp = QDoubleSpinBox()
        self.ssfp_amp.setRange(0.0, 1e3)
        self.ssfp_amp.setDecimals(6)
        self.ssfp_amp.setValue(0.05)
        self.ssfp_amp.valueChanged.connect(lambda _: self.update_diagram())
        row2.addWidget(self.ssfp_amp)
        row2.addWidget(QLabel("Phase (deg):"))
        self.ssfp_phase = QDoubleSpinBox()
        self.ssfp_phase.setRange(-3600, 3600)
        self.ssfp_phase.setDecimals(2)
        self.ssfp_phase.setValue(0.0)
        self.ssfp_phase.valueChanged.connect(lambda _: self.update_diagram())
        row2.addWidget(self.ssfp_phase)
        ssfp_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Pulse dur (ms):"))
        self.ssfp_dur = QDoubleSpinBox()
        self.ssfp_dur.setRange(0.01, 1000.0)
        self.ssfp_dur.setDecimals(3)
        self.ssfp_dur.setValue(1.0)
        self.ssfp_dur.valueChanged.connect(lambda _: self.update_diagram())
        row3.addWidget(self.ssfp_dur)
        row3.addWidget(QLabel("Start delay (ms):"))
        self.ssfp_start_delay = QDoubleSpinBox()
        self.ssfp_start_delay.setRange(0.0, 10000.0)
        self.ssfp_start_delay.setDecimals(3)
        self.ssfp_start_delay.setValue(0.0)
        self.ssfp_start_delay.valueChanged.connect(lambda _: self.update_diagram())
        row3.addWidget(self.ssfp_start_delay)
        ssfp_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Start amp (G):"))
        self.ssfp_start_amp = QDoubleSpinBox()
        self.ssfp_start_amp.setRange(0.0, 1e3)
        self.ssfp_start_amp.setDecimals(6)
        self.ssfp_start_amp.setValue(0.05)
        self.ssfp_start_amp.valueChanged.connect(lambda _: self.update_diagram())
        row4.addWidget(self.ssfp_start_amp)
        row4.addWidget(QLabel("Start phase (deg):"))
        self.ssfp_start_phase = QDoubleSpinBox()
        self.ssfp_start_phase.setRange(-3600, 3600)
        self.ssfp_start_phase.setDecimals(2)
        self.ssfp_start_phase.setValue(180.0)
        self.ssfp_start_phase.valueChanged.connect(lambda _: self.update_diagram())
        row4.addWidget(self.ssfp_start_phase)
        ssfp_layout.addLayout(row4)

        # Alternating phase option (common bSSFP scheme: 0/180/0/180 ...)
        self.ssfp_alternate_phase = QCheckBox("Alternate phase each TR (0/180°)")
        self.ssfp_alternate_phase.setChecked(True)
        self.ssfp_alternate_phase.toggled.connect(lambda _: self.update_diagram())
        ssfp_layout.addWidget(self.ssfp_alternate_phase)

        self.ssfp_opts.setLayout(ssfp_layout)
        self.options_container.addWidget(self.ssfp_opts)
        layout.addLayout(self.options_container)
        self.spin_echo_opts.setVisible(False)
        self.ssfp_opts.setVisible(False)
        self._update_sequence_options()
        
        # TE parameter
        te_layout = QHBoxLayout()
        te_layout.addWidget(QLabel("TE (ms):"))
        self.te_spin = QDoubleSpinBox()
        self.te_spin.setRange(0.1, 200)
        self.te_spin.setValue(20)
        te_layout.addWidget(self.te_spin)
        layout.addLayout(te_layout)
        self.te_spin.valueChanged.connect(lambda _: self.update_diagram())

        # Slice thickness
        thick_layout = QHBoxLayout()
        thick_layout.addWidget(QLabel("Slice thickness (mm):"))
        self.slice_thickness_spin = QDoubleSpinBox()
        self.slice_thickness_spin.setRange(0.05, 50.0)
        self.slice_thickness_spin.setValue(5.0)
        self.slice_thickness_spin.setDecimals(2)
        self.slice_thickness_spin.setSingleStep(0.1)
        self.slice_thickness_spin.valueChanged.connect(lambda _: self.update_diagram())
        thick_layout.addWidget(self.slice_thickness_spin)
        layout.addLayout(thick_layout)

        # Manual slice gradient override
        g_layout = QHBoxLayout()
        g_layout.addWidget(QLabel("Slice G override (G/cm, 0=auto):"))
        self.slice_gradient_spin = QDoubleSpinBox()
        self.slice_gradient_spin.setRange(0.0, 99999.0)
        self.slice_gradient_spin.setDecimals(3)
        self.slice_gradient_spin.setSingleStep(0.1)
        self.slice_gradient_spin.setValue(0.0)
        self.slice_gradient_spin.valueChanged.connect(lambda _: self.update_diagram())
        g_layout.addWidget(self.slice_gradient_spin)
        layout.addLayout(g_layout)
        
        # TR parameter
        tr_layout = QHBoxLayout()
        tr_layout.addWidget(QLabel("TR (ms):"))
        self.tr_spin = QDoubleSpinBox()
        self.tr_spin.setRange(1, 10000)
        self.tr_spin.setValue(100)
        tr_layout.addWidget(self.tr_spin)
        layout.addLayout(tr_layout)
        self.tr_spin.valueChanged.connect(lambda _: self.update_diagram())
        
        # TI parameter (for IR)
        ti_layout = QHBoxLayout()
        ti_layout.addWidget(QLabel("TI (ms):"))
        self.ti_spin = QDoubleSpinBox()
        self.ti_spin.setRange(1, 5000)
        self.ti_spin.setValue(400)
        ti_layout.addWidget(self.ti_spin)
        layout.addLayout(ti_layout)
        self.ti_spin.valueChanged.connect(lambda _: self.update_diagram())
        
        # Sequence diagram
        self.diagram_widget = pg.PlotWidget()
        self.diagram_widget.setLabel('left', '')
        self.diagram_widget.setLabel('bottom', 'Time', 'ms')
        self.diagram_widget.setMinimumHeight(250)
        layout.addWidget(self.diagram_widget)
        self.diagram_arrows = []
        self.playhead_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('y', width=2))
        self.diagram_widget.addItem(self.playhead_line)
        self.playhead_line.hide()
        
        self.setLayout(layout)
        # Draw initial diagram even before simulation
        self.update_diagram()
        
    def _slice_thickness_m(self) -> float:
        """Current slice thickness in meters."""
        return max(self.slice_thickness_spin.value() / 1000.0, 1e-4)

    def _slice_gradient_override(self) -> Optional[float]:
        """Manual slice gradient override in G/cm, or None for auto."""
        val = self.slice_gradient_spin.value()
        return val if val > 0 else None

    def _update_sequence_options(self):
        """Show/hide sequence-specific option widgets."""
        seq_type = self.sequence_type.currentText()
        self.spin_echo_opts.setVisible(seq_type in ("Spin Echo", "Spin Echo (Tip-axis 180)"))
        self.ssfp_opts.setVisible(seq_type == "SSFP (Loop)")

    def get_sequence_preset_params(self, seq_type: str) -> dict:
        """
        Get preset parameters for a specific sequence type.

        When the "Auto-load sequence presets" checkbox is enabled (default), changing
        the sequence type will automatically update relevant parameters to typical
        values for that sequence. This helps users quickly configure standard sequences
        without manual parameter adjustment.

        For example, switching to "Spin Echo" sets TE=20ms and TR=500ms, while
        "Gradient Echo" sets TE=5ms, TR=30ms, and flip_angle=30°. SSFP sequences
        additionally configure pulse durations, phases, and repetition counts.

        When disabled, parameter values are preserved across sequence changes, allowing
        users to maintain custom settings while exploring different sequence types.

        Returns
        -------
        dict
            Dictionary with sequence-specific parameters. Possible keys include:
            - te_ms: Echo time in milliseconds
            - tr_ms: Repetition time in milliseconds
            - ti_ms: Inversion time in milliseconds (for IR sequences)
            - flip_angle: Flip angle in degrees
            - ssfp_repeats: Number of SSFP repetitions
            - ssfp_amp: SSFP pulse amplitude in Gauss
            - ssfp_phase: SSFP pulse phase in degrees
            - ssfp_dur: SSFP pulse duration in milliseconds
            - ssfp_start_delay: Initial delay in milliseconds
            - ssfp_start_amp: Starting pulse amplitude in Gauss
            - ssfp_start_phase: Starting pulse phase in degrees
            - ssfp_alternate_phase: Boolean, alternate phase 0/180° each TR
        """
        presets = {
            "Free Induction Decay": {
                "te_ms": 10,
                "tr_ms": 100
            },
            "Spin Echo": {
                "te_ms": 20,
                "tr_ms": 500
            },
            "Spin Echo (Tip-axis 180)": {
                "te_ms": 20,
                "tr_ms": 500
            },
            "Gradient Echo": {
                "te_ms": 5,
                "tr_ms": 30,
                "flip_angle": 30
            },
            "Slice Select + Rephase": {
                "te_ms": 10,
                "tr_ms": 100
            },
            "SSFP (Loop)": {
                "te_ms": 2,
                "tr_ms": 5,
                "flip_angle": 30,
                "ssfp_repeats": 16,
                "ssfp_amp": 0.05,
                "ssfp_phase": 0.0,
                "ssfp_dur": 1.0,
                "ssfp_start_delay": 0.0,
                "ssfp_start_amp": 0.05,
                "ssfp_start_phase": 180.0,
                "ssfp_alternate_phase": True,
                "pulse_type": "gaussian"
            },
            "Inversion Recovery": {
                "te_ms": 20,
                "tr_ms": 2000,
                "ti_ms": 400
            },
            "FLASH": {
                "te_ms": 3,
                "tr_ms": 10,
                "flip_angle": 15
            },
            "EPI": {
                "te_ms": 30,
                "tr_ms": 3000
            },
            "Custom": {
                "te_ms": 10,
                "tr_ms": 100
            },
        }
        return presets.get(seq_type, {})

    def get_sequence(self, custom_pulse=None):
        """
        Get the current sequence parameters.
        
        If a custom pulse (b1, time) tuple is provided, it will be used for the
        "Custom" sequence option. Gradients are set to zero in that case.
        """
        seq_type = self.sequence_type.currentText()
        te = self.te_spin.value() / 1000  # Convert to seconds
        tr = self.tr_spin.value() / 1000
        ti = self.ti_spin.value() / 1000
        
        # Use explicit B1/gradient arrays when we can so RF designer changes take effect
        if (seq_type == "Free Induction Decay" or seq_type == "Custom") and custom_pulse is not None:
            b1, time = custom_pulse
            b1 = np.asarray(b1, dtype=complex)
            time = np.asarray(time, dtype=float)
            if b1.shape[0] != time.shape[0]:
                raise ValueError("Custom pulse B1 and time arrays must have the same length.")
            gradients = np.zeros((len(time), 3))
            return (b1, gradients, time)
        
        if seq_type == "Spin Echo":
            return SpinEcho(
                te=te,
                tr=tr,
                custom_excitation=custom_pulse,
                slice_thickness=self._slice_thickness_m(),
                slice_gradient_override=self._slice_gradient_override(),
                echo_count=self.spin_echo_echoes.value(),
            )
        elif seq_type == "Spin Echo (Tip-axis 180)":
            return SpinEchoTipAxis(
                te=te,
                tr=tr,
                custom_excitation=custom_pulse,
                slice_thickness=self._slice_thickness_m(),
                slice_gradient_override=self._slice_gradient_override(),
                echo_count=self.spin_echo_echoes.value(),
            )
        elif seq_type == "Gradient Echo":
            return GradientEcho(
                te=te,
                tr=tr,
                flip_angle=30,
                custom_excitation=custom_pulse,
                slice_thickness=self._slice_thickness_m(),
                slice_gradient_override=self._slice_gradient_override(),
            )
        elif seq_type == "Slice Select + Rephase":
            # Use a shorter rephase duration but preserve half-area rewind
            rephase_dur = max(0.2e-3, min(1.0e-3, te / 2))
            return SliceSelectRephase(
                flip_angle=90,
                pulse_duration=3e-3,
                time_bw_product=4.0,
                rephase_duration=rephase_dur,
                slice_thickness=self._slice_thickness_m(),
                slice_gradient_override=self._slice_gradient_override(),
                custom_pulse=custom_pulse,
            )
        else:
            # Return simple FID for now using configured time step and TE for duration
            dt = max(self.default_dt, 1e-6)
            total_duration = max(te, 0.01)  # cover at least 10 ms or TE
            ntime = int(np.ceil(total_duration / dt))
            ntime = min(max(ntime, 1000), 20000)  # keep reasonable bounds
            time = np.linspace(0, total_duration, ntime, endpoint=False)
            b1 = np.zeros(ntime, dtype=complex)
            b1[0] = 0.01  # Hard pulse
            gradients = np.zeros((ntime, 3))
            return (b1, gradients, time)

    def compile_sequence(self, custom_pulse=None, dt: float = None):
        """Return explicit (b1, gradients, time) arrays for the current sequence."""
        dt = dt or self.default_dt
        seq_type = self.sequence_type.currentText()
        if seq_type == "EPI":
            return self._build_epi(custom_pulse, dt)
        if seq_type == "Inversion Recovery":
            return self._build_ir(custom_pulse, dt)
        if seq_type == "SSFP (Loop)":
            return self._build_ssfp(custom_pulse, dt)
        seq = self.get_sequence(custom_pulse=custom_pulse)
        if isinstance(seq, PulseSequence):
            return seq.compile(dt=dt)
        b1, gradients, time = seq
        return np.asarray(b1, dtype=complex), np.asarray(gradients, dtype=float), np.asarray(time, dtype=float)

    def _build_epi(self, custom_pulse, dt):
        """
        Create a basic single-shot EPI echo train with slice-select,
        prephasing, alternating readouts, and phase-encode blips.
        """
        te = self.te_spin.value() / 1000
        tr = self.tr_spin.value() / 1000
        dt = max(dt, 1e-6)

        # Excitation (use provided custom pulse if available)
        if custom_pulse is not None:
            exc_b1, _ = custom_pulse
            exc_b1 = np.asarray(exc_b1, dtype=complex)
        else:
            exc_duration = 1e-3
            n_exc = max(int(np.ceil(exc_duration / dt)), 16)
            exc_b1, _ = design_rf_pulse("sinc", duration=n_exc * dt, flip_angle=90, npoints=n_exc)
        exc_pts = len(exc_b1)
        exc_duration = exc_pts * dt

        # Timing constants (all in points)
        slice_gap_pts = max(int(np.ceil(0.05e-3 / dt)), 1)
        rephase_pts = max(int(np.ceil(0.5e-3 / dt)), 4)
        prephase_pts = max(int(np.ceil(0.4e-3 / dt)), 4)
        settle_pts = max(int(np.ceil(0.05e-3 / dt)), 1)

        readout_dur = max(0.6e-3, min(1.2e-3, te / 4 if te > 0 else 0.8e-3))
        ro_pts = max(int(np.ceil(readout_dur / dt)), 8)
        blip_pts = max(int(np.ceil(0.12e-3 / dt)), 1)
        gap_pts = max(int(np.ceil(0.05e-3 / dt)), 1)
        n_lines = 16  # phase-encode lines in the echo train

        esp = (ro_pts + blip_pts + gap_pts) * dt
        mid_echo_time = (ro_pts * dt) / 2.0 + (n_lines // 2) * esp
        pre_time = (exc_pts + slice_gap_pts + rephase_pts + prephase_pts + settle_pts) * dt
        train_start_time = max(pre_time, te - mid_echo_time)
        train_start_pts = int(np.round(train_start_time / dt))

        # Use a start index that respects prephasing blocks
        actual_train_start = max(train_start_pts, exc_pts + slice_gap_pts + rephase_pts + prephase_pts + settle_pts)
        train_end_pts = actual_train_start + n_lines * ro_pts + (n_lines - 1) * (blip_pts + gap_pts)
        spoil_pts = max(int(np.ceil(0.6e-3 / dt)), 2)
        required_pts = train_end_pts + spoil_pts + 1
        npoints = int(max(np.ceil(tr / dt), required_pts, exc_pts + 1))

        b1 = np.zeros(npoints, dtype=complex)
        gradients = np.zeros((npoints, 3))

        # Apply excitation + slice-select gradient
        n_exc = min(exc_pts, npoints)
        b1[:n_exc] = exc_b1[:n_exc]
        thickness_cm = self._slice_thickness_m() * 100.0
        gamma_hz_per_g = 4258.0
        bw_hz = 4.0 / max(exc_duration, dt)
        slice_g = self._slice_gradient_override() or (bw_hz / (gamma_hz_per_g * thickness_cm))  # G/cm
        gradients[:n_exc, 2] = slice_g

        # Slice rephasing (half area rewind)
        rephase_start = n_exc + slice_gap_pts
        if rephase_start < npoints:
            area_exc = slice_g * n_exc * dt
            rephase_amp = -(0.5 * area_exc) / (rephase_pts * dt)
            gradients[rephase_start:rephase_start + rephase_pts, 2] = rephase_amp

        # Readout prephaser to move to -kmax
        prephase_start = rephase_start + rephase_pts
        read_amp = 8e-3  # readout gradient amplitude
        if prephase_start < npoints:
            prephase_amp = -0.5 * read_amp * (ro_pts / max(prephase_pts, 1))
            gradients[prephase_start:prephase_start + prephase_pts, 0] = prephase_amp

        # Echo train with alternating readouts and Gy blips
        pos = max(actual_train_start, prephase_start + prephase_pts + settle_pts)
        phase_blip_amp = 2.5e-3
        for line in range(n_lines):
            if pos >= npoints:
                break
            ro_end = min(pos + ro_pts, npoints)
            direction = 1 if line % 2 == 0 else -1
            gradients[pos:ro_end, 0] = direction * read_amp
            pos = ro_end
            if line < n_lines - 1:
                # small gap then phase-encode blip
                gap_end = min(pos + gap_pts, npoints)
                pos = gap_end
                blip_end = min(pos + blip_pts, npoints)
                gradients[pos:blip_end, 1] = phase_blip_amp
                pos = blip_end

        # Spoiler after the train
        spoil_start = min(pos + gap_pts, npoints)
        spoil_end = min(spoil_start + spoil_pts, npoints)
        gradients[spoil_start:spoil_end, 0] = 4e-3

        time = np.arange(npoints) * dt
        return b1, gradients, time

    def _build_ir(self, custom_pulse, dt):
        """Basic inversion recovery: 180 inversion, wait TI, then 90 + readout."""
        ti = self.ti_spin.value() / 1000
        te = self.te_spin.value() / 1000
        tr = self.tr_spin.value() / 1000
        dt = max(dt, 1e-6)
        npoints = int(np.ceil(tr / dt))
        b1 = np.zeros(npoints, dtype=complex)
        gradients = np.zeros((npoints, 3))
        inv_b1, _ = design_rf_pulse('sinc', duration=2e-3, flip_angle=180, npoints=max(32, int(2e-3/dt)))
        n_inv = min(len(inv_b1), npoints)
        b1[:n_inv] = inv_b1[:n_inv]
        thickness_cm = self._slice_thickness_m() * 100.0
        gamma_hz_per_g = 4258.0
        bw_hz = 4.0 / max(len(inv_b1) * dt, dt)
        slice_g = self._slice_gradient_override() or (bw_hz / (gamma_hz_per_g * thickness_cm))  # G/cm
        gradients[:n_inv, 2] = slice_g
        start_exc = int(max(ti, (n_inv * dt)) / dt)
        if custom_pulse is not None:
            exc_b1, _ = custom_pulse
            exc_b1 = np.asarray(exc_b1, dtype=complex)
        else:
            exc_b1, _ = design_rf_pulse('sinc', duration=1e-3, flip_angle=90, npoints=max(16, int(1e-3/dt)))
        n_exc = min(len(exc_b1), max(0, npoints - start_exc))
        b1[start_exc:start_exc + n_exc] = exc_b1[:n_exc]
        bw_hz_exc = 4.0 / max(n_exc * dt, dt)
        slice_g_exc = self._slice_gradient_override() or (bw_hz_exc / (gamma_hz_per_g * thickness_cm))
        gradients[start_exc:start_exc + n_exc, 2] = slice_g_exc
        ro_start = start_exc + int(max(0.2e-3, te/2) / dt)
        ro_dur = max(1, int(0.8e-3 / dt))
        if ro_start + ro_dur < npoints:
            gradients[ro_start:ro_start + ro_dur, 0] = 5e-3
        time = np.arange(npoints) * dt
        return b1, gradients, time

    def _build_ssfp(self, custom_pulse, dt):
        """
        Build a simple balanced-SSFP-style pulse train: identical RF pulses every TR,
        with an optional distinct first pulse (amplitude/phase/delay).
        """
        dt = max(dt, 1e-6)
        tr = self.tr_spin.value() / 1000.0
        n_reps = max(1, self.ssfp_repeats.value())
        pulse_amp = self.ssfp_amp.value()
        pulse_phase = np.deg2rad(self.ssfp_phase.value())
        pulse_dur = self.ssfp_dur.value() / 1000.0
        start_amp = self.ssfp_start_amp.value()
        start_phase = np.deg2rad(self.ssfp_start_phase.value())
        start_delay = self.ssfp_start_delay.value() / 1000.0
        alternate = self.ssfp_alternate_phase.isChecked()

        # If a custom pulse is provided, resample it onto dt and override the pulse shape.
        custom_b1 = None
        if custom_pulse is not None:
            b1_wave, t_wave = custom_pulse
            b1_wave = np.asarray(b1_wave, dtype=complex)
            t_wave = np.asarray(t_wave, dtype=float)
            if b1_wave.shape[0] != t_wave.shape[0]:
                raise ValueError("Custom pulse B1 and time arrays must have the same length.")
            if len(t_wave) > 1:
                wave_dt = np.median(np.diff(t_wave))
                wave_duration = (len(t_wave) - 1) * wave_dt
            else:
                wave_dt = dt
                wave_duration = dt
            n_wave = max(1, int(np.round(wave_duration / dt)))
            t_resample = np.arange(0, n_wave) * dt
            # Resample real/imag separately to avoid dropping complex parts
            real_part = np.interp(t_resample, t_wave - t_wave[0], np.real(b1_wave))
            imag_part = np.interp(t_resample, t_wave - t_wave[0], np.imag(b1_wave))
            custom_b1 = real_part + 1j * imag_part
            pulse_dur = wave_duration

        # Determine timeline length
        total_duration = start_delay + pulse_dur + tr * (n_reps - 1) + 0.5 * tr
        npoints = int(np.ceil(total_duration / dt)) + 1
        b1 = np.zeros(npoints, dtype=complex)
        gradients = np.zeros((npoints, 3), dtype=float)
        time = np.arange(npoints) * dt

        base_peak = None
        if custom_b1 is not None:
            base_peak = float(np.max(np.abs(custom_b1))) if np.any(np.abs(custom_b1)) else 1.0

        def _place_pulse(start_s, amp, phase):
            start_idx = int(np.round(start_s / dt))
            n_dur = max(1, int(np.round(pulse_dur / dt)))
            end_idx = min(start_idx + n_dur, npoints)
            if custom_b1 is not None:
                seg = custom_b1
                seg_len = min(end_idx - start_idx, len(seg))
                # Scale/rotate custom waveform by amp/phase controls
                scale = amp / base_peak if base_peak else 1.0
                b1[start_idx:start_idx + seg_len] = seg[:seg_len] * scale * np.exp(1j * phase)
            else:
                b1[start_idx:end_idx] = amp * np.exp(1j * phase)

        # Optional distinct first pulse
        _place_pulse(start_delay, start_amp, start_phase)

        # Remaining pulses evenly spaced by TR
        for k in range(1, n_reps):
            t0 = start_delay + k * tr
            phase = pulse_phase
            if alternate:
                phase = pulse_phase + (math.pi if (k % 2 == 1) else 0.0)
            _place_pulse(t0, pulse_amp, phase)

        return b1, gradients, time

    def set_time_step(self, dt_s: float):
        """Set default time step for fallback/simple sequences."""
        if dt_s and dt_s > 0:
            self.default_dt = dt_s
            self.update_diagram()

    def set_custom_pulse(self, pulse):
        """Store custom pulse for preview (used when sequence type is Custom)."""
        self.custom_pulse = pulse
        # If a custom pulse exists, sync SSFP parameter widgets to its basic stats
        if pulse is not None:
            b1_wave, t_wave = pulse
            b1_wave = np.asarray(b1_wave, dtype=complex)
            if b1_wave.size:
                max_amp = float(np.max(np.abs(b1_wave)))
                self.ssfp_amp.setValue(max_amp)
                self.ssfp_start_amp.setValue(max_amp/2.0)
                phase = float(np.angle(b1_wave[0]))
                self.ssfp_phase.setValue(np.rad2deg(phase))
                self.ssfp_start_phase.setValue(np.rad2deg(phase))
            if t_wave is not None and len(t_wave) > 1:
                duration_s = float(t_wave[-1] - t_wave[0])
                self.ssfp_dur.setValue(max(duration_s * 1000.0, self.ssfp_dur.singleStep()))
        self.update_diagram()
        self._update_sequence_options()

    def update_diagram(self, custom_pulse=None):
        """Render the sequence diagram so users can see the selected waveform."""
        custom = custom_pulse if custom_pulse is not None else self.custom_pulse
        try:
            b1, gradients, time = self.compile_sequence(custom_pulse=custom, dt=self.default_dt)
        except Exception:
            self.diagram_widget.clear()
            return
        self._render_sequence_diagram(b1, gradients, time)

    def _render_sequence_diagram(self, b1, gradients, time):
        """Plot a lane-based sequence diagram (RF, Gradients)."""
        self.diagram_widget.clear()
        if self.playhead_line is not None:
            self.diagram_widget.addItem(self.playhead_line)
            self.playhead_line.hide()
        for arr in self.diagram_arrows:
            try:
                self.diagram_widget.removeItem(arr)
            except Exception:
                pass
        self.diagram_arrows = []
        for lbl in getattr(self, "diagram_labels", []):
            try:
                self.diagram_widget.removeItem(lbl)
            except Exception:
                pass
        self.diagram_labels = []
        if time is None or len(time) == 0:
            return
        max_points = 4000
        if len(time) > max_points:
            idx = np.linspace(0, len(time) - 1, max_points).astype(int)
            time = time[idx]
            b1 = b1[idx]
            gradients = gradients[idx]
        time_ms = (time - time[0]) * 1000.0
        b1_mag = np.abs(b1)
        b1_phase = np.angle(b1)  # Phase in radians

        # Lane positions and labels
        lanes = [
            ("RF Mag", 4.0, 'b'),
            ("RF Phase", 3.0, 'c'),
            ("Gz (slice/readout)", 2.0, 'm'),
            ("Gy (phase1)", 1.0, 'g'),
            ("Gx (phase2)", 0.0, 'r'),
        ]

        # Draw horizontal grid lines for clarity
        for _, y, _ in lanes:
            line = pg.InfiniteLine(pos=y, angle=0, pen=pg.mkPen((180, 180, 180, 120), width=1))
            self.diagram_widget.addItem(line)
        # Add lane labels near time zero
        for label, y, color in lanes:
            txt = pg.TextItem(text=label, color=color, anchor=(0, 0.5))
            txt.setPos(time_ms[0] if len(time_ms) else 0, y)
            self.diagram_widget.addItem(txt)
            self.diagram_labels.append(txt)

        # RF Magnitude lane
        rf_mag_y = lanes[0][1]
        rf_scale = 0.8 if b1_mag.max() == 0 else 0.8 / b1_mag.max()
        self.diagram_widget.plot(time_ms, rf_mag_y + b1_mag * rf_scale,
                                 pen=pg.mkPen('b', width=2), name="RF Mag")

        # RF Phase lane (convert radians to normalized display: -π to π → -0.8 to 0.8)
        rf_phase_y = lanes[1][1]
        # Only plot phase where there's significant RF (avoid noise)
        phase_mask = b1_mag > (b1_mag.max() * 0.01) if b1_mag.max() > 0 else np.zeros_like(b1_mag, dtype=bool)
        if np.any(phase_mask):
            phase_display = b1_phase / np.pi * 0.8  # Normalize to ±0.8 for display
            # Create connected segments only where RF is active
            self.diagram_widget.plot(time_ms, rf_phase_y + phase_display,
                                     pen=pg.mkPen('c', width=2), name="RF Phase",
                                     connect='finite')

            # Add phase reference markers at -π, 0, +π
            phase_ref_pen = pg.mkPen((150, 150, 150, 100), width=1, style=Qt.DashLine)
            for phase_val, label_text in [(-np.pi, '-π'), (0, '0'), (np.pi, '+π')]:
                y_pos = rf_phase_y + (phase_val / np.pi * 0.8)
                ref_line = pg.InfiniteLine(pos=y_pos, angle=0, pen=phase_ref_pen)
                self.diagram_widget.addItem(ref_line)
                # Add small label at the right edge
                if len(time_ms) > 0:
                    phase_label = pg.TextItem(text=label_text, color=(150, 150, 150),
                                             anchor=(1, 0.5), angle=0)
                    phase_label.setPos(time_ms[-1] * 1.02, y_pos)
                    self.diagram_widget.addItem(phase_label)
                    self.diagram_labels.append(phase_label)

        # Gradient lanes
        grad_scales = []
        for i, (label, y, color) in enumerate(lanes[2:]):
            if i >= gradients.shape[1]:
                continue
            g = gradients[:, i]
            scale = 0.8 / (np.max(np.abs(g)) + 1e-9)
            grad_scales.append(scale)
            self.diagram_widget.plot(
                time_ms,
                y + g * scale,
                pen=pg.mkPen(color, width=2, style=Qt.SolidLine),
                name=label
            )
            nonzero = np.where(np.abs(g) > 0)[0]
            if nonzero.size:
                mid = nonzero[nonzero.size // 2]
                x = time_ms[mid]
                angle = 90 if g[mid] > 0 else -90
                arr = pg.ArrowItem(pos=(x, y + g[mid] * scale), angle=angle, brush=color)
                self.diagram_widget.addItem(arr)
                self.diagram_arrows.append(arr)

        # TE / TR markers
        te_ms = self.te_spin.value()
        tr_ms = self.tr_spin.value()
        if tr_ms > 0:
            tr_line = pg.InfiniteLine(pos=tr_ms, angle=90, pen=pg.mkPen((120, 120, 120), style=Qt.DashLine))
            self.diagram_widget.addItem(tr_line)
        if te_ms > 0:
            te_line = pg.InfiniteLine(pos=te_ms, angle=90, pen=pg.mkPen((120, 120, 120), style=Qt.DotLine))
            self.diagram_widget.addItem(te_line)

        self.diagram_widget.setLimits(xMin=0)
        if len(time_ms):
            self.diagram_widget.setXRange(0, time_ms[-1], padding=0)
            if self.playhead_line is not None:
                self.playhead_line.setValue(time_ms[0])
                self.playhead_line.show()

        # Store time array for cursor positioning
        self.preview_time = time_ms if len(time_ms) > 0 else None
        # Ensure playhead is initialized at start
        self.set_cursor_index(0)

    def set_cursor_index(self, idx: int):
        """Move cursor/playhead to a specific time index."""
        if self.playhead_line is None or self.preview_time is None:
            return
        if len(self.preview_time) == 0:
            return
        idx = int(max(0, min(idx, len(self.preview_time) - 1)))
        time_ms = self.preview_time[idx]
        self.playhead_line.setValue(time_ms)
        self.playhead_line.show()
        try:
            self.playhead_line.setVisible(True)
        except Exception:
            pass


class UniversalTimeControl(QWidget):
    """Universal time control widget that synchronizes all time-resolved views."""
    time_changed = pyqtSignal(int)  # Emits time index

    def __init__(self):
        super().__init__()
        self._updating = False  # Prevent circular updates
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Time slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Time:"))
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.time_slider, 1)
        self.time_label = QLabel("0.0 ms")
        self.time_label.setFixedWidth(80)
        slider_layout.addWidget(self.time_label)
        layout.addLayout(slider_layout)

        # Playback controls
        control_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.reset_button = QPushButton("Reset")
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.reset_button)

        # Speed control
        control_layout.addWidget(QLabel("Speed (ms/s):"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.001, 1000.0)
        self.speed_spin.setValue(50.0)  # Default to 50 ms of sim per real second
        self.speed_spin.setSuffix(" ms/s")
        self.speed_spin.setSingleStep(0.01)
        control_layout.addWidget(self.speed_spin)

        layout.addLayout(control_layout)
        self.setLayout(layout)

        self.time_array = None  # Will store time array in seconds

    def set_time_range(self, time_array):
        """Set the time range from a time array (in seconds)."""
        if time_array is None or len(time_array) == 0:
            self.time_array = None
            self.time_slider.setMaximum(0)
            self.time_label.setText("0.0 ms")
            return

        self.time_array = np.asarray(time_array)
        max_idx = len(self.time_array) - 1
        self.time_slider.blockSignals(True)
        self.time_slider.setMaximum(max_idx)
        self.time_slider.setValue(0)
        self.time_slider.blockSignals(False)
        self._update_time_label(0)

    def set_time_index(self, idx: int):
        """Set time index without emitting signal (for external updates)."""
        if self._updating:
            return
        self._updating = True
        idx = int(max(0, min(idx, self.time_slider.maximum())))
        self.time_slider.setValue(idx)
        self._update_time_label(idx)
        self._updating = False

    def _on_slider_changed(self, value):
        """Handle slider value change."""
        if self._updating:
            return
        self._update_time_label(value)
        self._updating = True
        self.time_changed.emit(value)
        self._updating = False

    def _update_time_label(self, idx):
        """Update the time label display."""
        if self.time_array is not None and 0 <= idx < len(self.time_array):
            time_ms = self.time_array[idx] * 1000
            self.time_label.setText(f"{time_ms:.3f} ms")
        else:
            self.time_label.setText("0.0 ms")


class MagnetizationViewer(QWidget):
    """3D visualization of magnetization vector."""
    position_changed = pyqtSignal(int)
    view_filter_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        self.playhead_line = None

        # Add export header for 3D view
        header_3d = QHBoxLayout()
        header_3d.addWidget(QLabel("3D Magnetization Vector"))
        header_3d.addStretch()

        self.export_3d_btn = QPushButton("Export ▼")
        export_3d_menu = QMenu()
        export_3d_menu.addAction("Image (PNG)...", lambda: self._export_3d_screenshot('png'))
        export_3d_menu.addAction("Image (SVG)...", lambda: self._export_3d_screenshot('svg'))
        export_3d_menu.addSeparator()
        export_3d_menu.addAction("Animation (GIF/MP4)...", self._export_3d_animation)
        self.export_3d_btn.setMenu(export_3d_menu)
        header_3d.addWidget(self.export_3d_btn)
        layout.addLayout(header_3d)

        # 3D view
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=5)
        self.gl_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.gl_widget.setMinimumHeight(300)
        
        # Add coordinate axes
        axis = gl.GLAxisItem()
        axis.setSize(2, 2, 2)
        self.gl_widget.addItem(axis)
        
        # Add grid
        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(1, 1)
        self.gl_widget.addItem(grid)
        
        # Initialize magnetization vectors (one per frequency)
        self.vectors = []
        self.vector_colors = []
        self._ensure_vectors(1)
        
        layout.addWidget(self.gl_widget, stretch=5)

        # Preview plot for time cursor
        self.preview_plot = pg.PlotWidget()
        self.preview_plot.setLabel('left', 'M')
        self.preview_plot.setLabel('bottom', 'Time', 'ms')
        self.preview_plot.enableAutoRange(x=False, y=False)
        self.preview_plot.setMaximumHeight(180)
        self.preview_mx = self.preview_plot.plot(pen='r')
        self.preview_my = self.preview_plot.plot(pen='g')
        self.preview_mz = self.preview_plot.plot(pen='b')
        self.preview_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y'))
        self.preview_plot.addItem(self.preview_line)
        layout.addWidget(self.preview_plot, stretch=1)

        # Time slider for scrubbing animation
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.valueChanged.connect(self._slider_moved)
        self.time_slider.setVisible(False)  # Hide in favor of the universal time control
        layout.addWidget(self.time_slider)

        # B1 indicator arrow (optional overlay)
        self.b1_arrow = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 0]]),
            color=(0, 1, 1, 0.9),
            width=3,
            mode='lines'
        )
        self.b1_arrow.setVisible(False)
        self.gl_widget.addItem(self.b1_arrow)
        self.b1_scale = 1.0
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.reset_button = QPushButton("Reset")
        # Hide local play/pause/reset in favor of universal control below
        self.play_button.setVisible(False)
        self.pause_button.setVisible(False)
        self.reset_button.setVisible(False)
        # Local speed control disabled; universal control is authoritative
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.001, 1000.0)
        self.speed_spin.setValue(50.0)  # Default to 50 ms of sim per real second
        self.speed_spin.setSuffix(" ms/s")
        self.speed_spin.setSingleStep(0.01)
        self.speed_spin.setEnabled(False)
        control_layout.addWidget(QLabel("Speed (ms/s):"))
        control_layout.addWidget(self.speed_spin)
        control_container = QWidget()
        controls_v = QVBoxLayout()
        controls_v.setContentsMargins(0, 0, 0, 0)
        controls_v.addLayout(control_layout)
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View mode:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems([
            "All positions x freqs",
            "Positions @ freq",
            "Freqs @ position"
        ])
        self.view_mode_combo.currentTextChanged.connect(self._on_view_mode_changed)
        view_layout.addWidget(self.view_mode_combo)
        self.selector_label = QLabel("All spins")
        view_layout.addWidget(self.selector_label)
        self.selector_slider = QSlider(Qt.Horizontal)
        self.selector_slider.setRange(0, 0)
        self.selector_slider.valueChanged.connect(self._on_selector_changed)
        view_layout.addWidget(self.selector_slider)
        controls_v.addLayout(view_layout)

        # Initialize tracking state and path storage BEFORE checkbox initialization
        self.length_scale = 1.0  # Reference magnitude used to normalize vectors in view
        self.preview_time = None
        self.track_path = False
        self.path_points = []
        self.path_item = gl.GLLinePlotItem(pos=np.zeros((0, 3)), color=(1, 1, 0, 0.8), width=2, mode='line_strip')
        self.gl_widget.addItem(self.path_item)
        self.mean_vector = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1, 1, 0, 1), width=5, mode='lines')
        self.gl_widget.addItem(self.mean_vector)

        # Now create checkboxes that depend on these variables
        self.track_checkbox = QCheckBox("Track tip path")
        self.track_checkbox.setChecked(True)
        self.track_checkbox.toggled.connect(self._toggle_track_path)
        # Sync internal flag to the initial checkbox state so tracking is active on first playback
        self._toggle_track_path(self.track_checkbox.isChecked())
        controls_v.addWidget(self.track_checkbox)
        self.mean_checkbox = QCheckBox("Show mean magnetization")
        self.mean_checkbox.setChecked(False)
        self.mean_checkbox.toggled.connect(self._toggle_mean_vector)
        controls_v.addWidget(self.mean_checkbox)
        control_container.setLayout(controls_v)
        self.control_container = control_container
        layout.addWidget(control_container)

        # Track available position/frequency counts for selector range updates
        self._npos = 1
        self._nfreq = 1
        self._update_selector_range()
        
        self.setLayout(layout)
        
    def _ensure_vectors(self, count: int, colors=None):
        """Create/update GL line items to match the requested count."""
        # Remove excess
        while len(self.vectors) > count:
            vec = self.vectors.pop()
            try:
                self.gl_widget.removeItem(vec)
            except Exception:
                pass
        # Add missing
        while len(self.vectors) < count:
            color = (1, 0, 0, 1)
            if colors and len(colors) > len(self.vectors):
                color = colors[len(self.vectors)]
            vec = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [0, 0, 1]]),
                color=color,
                width=3
            )
            self.gl_widget.addItem(vec)
            self.vectors.append(vec)
        # Update colors if provided
        if colors:
            for vec, col in zip(self.vectors, colors):
                vec.setData(color=col)
        self.vector_colors = colors or self.vector_colors

    def set_length_scale(self, scale: float):
        """Set reference magnitude to normalize displayed vectors."""
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        self.length_scale = float(scale)

    def update_magnetization(self, mx, my=None, mz=None, colors=None):
        """
        Update the magnetization vector display.
        Accepts either separate mx/my/mz scalars/arrays or an array of shape (nfreq, 3).
        """
        if my is None and mz is None:
            arr = np.asarray(mx)
            if arr.ndim == 1:
                arr = arr.reshape(1, 3)
            elif arr.ndim == 2 and arr.shape[1] == 3:
                pass
            else:
                return
            vecs = arr
        else:
            mx_arr = np.atleast_1d(mx)
            my_arr = np.atleast_1d(my)
            mz_arr = np.atleast_1d(mz)
            if mx_arr.shape != my_arr.shape or mx_arr.shape != mz_arr.shape:
                return
            if mx_arr.ndim == 0:
                vecs = np.array([[float(mx_arr), float(my_arr), float(mz_arr)]])
            else:
                vecs = np.stack([mx_arr, my_arr, mz_arr], axis=-1)
        count = vecs.shape[0]
        self._ensure_vectors(count, colors=colors)
        norm = 1.0 / max(self.length_scale, 1e-9)
        vecs_scaled = vecs * norm
        for vec_item, comp in zip(self.vectors, vecs):
            pos = np.array([[0, 0, 0], comp * norm])
            vec_item.setData(pos=pos)
        mean_vec = np.mean(vecs_scaled, axis=0) if vecs.size else None
        if self.track_path and mean_vec is not None:
            self._append_path_point(mean_vec)
        # Mean vector (over all components)
        if mean_vec is not None and (self.mean_checkbox.isChecked() or self.track_path):
            self.mean_vector.setData(pos=np.array([[0, 0, 0], mean_vec]))
            self.mean_vector.setVisible(True)
        else:
            self.mean_vector.setVisible(False)

    def set_preview_data(self, time_ms, mx, my, mz):
        """Update preview plot and slider for scrubbing."""
        if time_ms is None or len(time_ms) == 0:
            self.preview_time = None
            self.preview_mx.clear()
            self.preview_my.clear()
            self.preview_mz.clear()
            self.time_slider.setRange(0, 0)
            return
        self.preview_time = np.asarray(time_ms)
        self.preview_mx.setData(time_ms, mx)
        self.preview_my.setData(time_ms, my)
        self.preview_mz.setData(time_ms, mz)
        x_min, x_max = float(time_ms[0]), float(time_ms[-1])
        # Clamp preview to expected magnetization range
        max_abs = 1.0
        for arr in (mx, my, mz):
            if arr is None:
                continue
            arr_np = np.asarray(arr)
            if arr_np.size:
                with np.errstate(invalid='ignore'):
                    current = np.nanmax(np.abs(arr_np))
                if np.isfinite(current):
                    max_abs = max(max_abs, float(current))
        y_min, y_max = -1.1 * max_abs, 1.1 * max_abs
        self.preview_plot.setLimits(xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max)
        self.preview_plot.setXRange(x_min, x_max, padding=0)
        self.preview_plot.setYRange(y_min, y_max, padding=0)
        max_idx = max(len(time_ms) - 1, 0)
        self.time_slider.blockSignals(True)
        self.time_slider.setRange(0, max_idx)
        self.time_slider.setValue(0)
        self.time_slider.blockSignals(False)
        self._update_cursor_line(0)

    def set_cursor_index(self, idx: int):
        """Move cursor/slider without emitting position change."""
        if self.preview_time is None:
            return
        idx = int(max(0, min(idx, len(self.preview_time) - 1)))
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(idx)
        self.time_slider.blockSignals(False)
        self._update_cursor_line(idx)

    def _slider_moved(self, idx: int):
        self._update_cursor_line(idx)
        self.position_changed.emit(idx)

    def _update_cursor_line(self, idx: int):
        if self.preview_time is None or len(self.preview_time) == 0:
            return
        idx = int(max(0, min(idx, len(self.preview_time) - 1)))
        self.preview_line.setValue(self.preview_time[idx])
        if not self.track_path:
            self._clear_path()
        # Move sequence playhead if visible
        if self.playhead_line is not None:
            try:
                self.playhead_line.setValue(self.preview_time[idx])
            except Exception:
                pass
        # Move time cursors on plots
        try:
            tval = float(self.preview_time[idx])
            for line in (self.mxy_time_line, self.mz_time_line, self.signal_time_line):
                if line is not None:
                    line.show()
                    line.setValue(tval)
            # Also update spatial time lines if they exist
            for line in (getattr(self, 'spatial_mxy_time_line', None), getattr(self, 'spatial_mz_time_line', None)):
                if line is not None:
                    line.show()
                    line.setValue(tval)
        except Exception:
            pass

    def _on_view_mode_changed(self, *args):
        self._update_selector_range()
        self.view_filter_changed.emit()

    def _on_selector_changed(self, value: int):
        # Update label to reflect the new selection, then notify listeners
        self._update_selector_range()
        self.view_filter_changed.emit()

    def _update_selector_range(self):
        """Update selector slider/label based on current mode and data availability."""
        mode = self.view_mode_combo.currentText() if hasattr(self, "view_mode_combo") else "All positions x freqs"
        if mode == "Positions @ freq":
            max_idx = max(0, self._nfreq - 1)
            prefix = "Freq"
        elif mode == "Freqs @ position":
            max_idx = max(0, self._npos - 1)
            prefix = "Pos"
        else:
            max_idx = 0
            prefix = "All"
        self.selector_slider.blockSignals(True)
        self.selector_slider.setMaximum(max_idx)
        self.selector_slider.setValue(min(self.selector_slider.value(), max_idx) if max_idx > 0 else 0)
        self.selector_slider.setVisible(max_idx > 0)
        self.selector_slider.blockSignals(False)
        if max_idx > 0:
            self.selector_label.setText(f"{prefix} idx: {self.selector_slider.value()}")
        else:
            self.selector_label.setText("All spins")

    def set_selector_limits(self, npos: int, nfreq: int, disable: bool = False):
        """Set available position/frequency counts for the selector control."""
        self._npos = max(1, int(npos)) if np.isfinite(npos) else 1
        self._nfreq = max(1, int(nfreq)) if np.isfinite(nfreq) else 1
        enabled = not disable
        self.view_mode_combo.setEnabled(enabled)
        self.selector_slider.setEnabled(enabled)
        self._update_selector_range()

    def get_view_mode(self) -> str:
        """Return the current 3D view mode selection."""
        return self.view_mode_combo.currentText() if hasattr(self, "view_mode_combo") else "All positions x freqs"

    def get_selector_index(self) -> int:
        """Return the currently selected index for the active view mode."""
        return int(self.selector_slider.value()) if hasattr(self, "selector_slider") else 0

    def _toggle_track_path(self, enabled: bool):
        self.track_path = enabled
        if not enabled:
            self._clear_path()

    def _toggle_mean_vector(self, enabled: bool):
        if not enabled:
            self.mean_vector.setVisible(False)

    def _clear_path(self):
        self.path_points = []
        self.path_item.setData(pos=np.zeros((0, 3)))

    def _append_path_point(self, vec):
        """Append a point to the tracked tip path."""
        if not self.track_path:
            return
        vec = np.asarray(vec, dtype=float).ravel()
        if vec.shape[0] != 3:
            return
        self.path_points.append(vec)
        if len(self.path_points) > 5000:
            self.path_points = self.path_points[-5000:]
        self.path_item.setData(pos=np.asarray(self.path_points))

    def _export_3d_screenshot(self, format='png'):
        """Export 3D view as screenshot."""
        exporter = ImageExporter()

        # Get export directory (or use current directory as fallback)
        export_dir = Path.cwd() / "exports"
        export_dir.mkdir(exist_ok=True)
        default_path = export_dir / f"3d_view.{format}"

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export 3D View",
            str(default_path),
            f"{format.upper()} Images (*.{format})"
        )

        if filename:
            try:
                result = exporter.export_widget_screenshot(
                    self.gl_widget,
                    filename,
                    format=format
                )

                if result:
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"3D view exported to:\n{Path(result).name}"
                    )
                else:
                    QMessageBox.warning(self, "Export Failed", "Could not export 3D view.")

            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")

    def _show_not_implemented_3d(self, feature_name):
        """Show a message for features not yet implemented in 3D viewer."""
        QMessageBox.information(
            self,
            "Coming Soon",
            f"{feature_name} export will be available in a future update."
        )

    def _export_3d_animation(self):
        """
        Delegate animation export to parent window (BlochSimulatorGUI) if available.
        """
        win = self.window()
        if win and hasattr(win, "_export_3d_animation"):
            try:
                win._export_3d_animation()
                return
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", str(exc))
                return
        # Fallback message if parent handler not found
        self._show_not_implemented_3d("Animation")


class BlochSimulatorGUI(QMainWindow):
    """Main GUI window for the Bloch simulator."""
    
    def __init__(self):
        super().__init__()
        self.simulator = BlochSimulator(use_parallel=True, num_threads=4)
        self.simulation_thread = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Bloch Equation Simulator")
        self.setGeometry(100, 100, 1400, 900)
        self.last_pulse_range = None
        self.mxy_region = None
        self.mz_region = None
        self.signal_region = None
        self.spatial_plot = None
        self.last_result = None
        self.last_positions = None
        self.last_frequencies = None
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._animate_vector)
        self.anim_interval_ms = 30
        self.anim_data = None
        self.anim_time = None
        self.playback_indices = None  # Mapping from playback frame to full-resolution time index
        self.playback_time = None     # Time array aligned to playback indices (seconds)
        self.playback_time_ms = None  # Same as playback_time but in ms for plot previews
        self.anim_index = 0
        self._frame_step = 1
        self._min_anim_interval_ms = 1000.0 / 120.0  # cap display rate to avoid event loop overload
        self.anim_b1 = None
        self.anim_b1_scale = 1.0
        self.anim_vectors_full = None  # (ntime, npos, nfreq, 3) before flattening
        self.mxy_legend = None
        self.mz_legend = None
        self.signal_legend = None
        self.initial_mz = 1.0  # Track initial Mz to scale plot limits
        self._last_spatial_export = None
        self._last_spectrum_export = None
        self.dataset_exporter = DatasetExporter()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Parameters
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)
        
        # Tissue parameters
        self.tissue_widget = TissueParameterWidget()
        left_layout.addWidget(self.tissue_widget)
        
        # RF Pulse designer
        self.rf_designer = RFPulseDesigner()
        left_layout.addWidget(self.rf_designer)
        
        # Sequence designer
        self.sequence_designer = SequenceDesigner()
        left_layout.addWidget(self.sequence_designer)
        self.rf_designer.pulse_changed.connect(self.sequence_designer.set_custom_pulse)
        self.sequence_designer.set_custom_pulse(self.rf_designer.get_pulse())
        # Connect sequence type changes to preset loader
        self.sequence_designer.sequence_type.currentTextChanged.connect(self._load_sequence_presets)
        # Link 3D viewer playhead to sequence diagram
        self.sequence_designer.update_diagram()
        
        # Simulation controls
        control_group = QGroupBox("Simulation Controls")
        control_layout = QVBoxLayout()
        
        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Endpoint", "Time-resolved"])
        # Default to time-resolved so users see waveforms/animation without changing anything
        self.mode_combo.setCurrentText("Time-resolved")
        mode_layout.addWidget(self.mode_combo)
        control_layout.addLayout(mode_layout)
        
        # Positions
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("Positions:"))
        self.pos_spin = QSpinBox()
        self.pos_spin.setRange(1, 1100)
        self.pos_spin.setValue(1)
        pos_layout.addWidget(self.pos_spin)
        pos_layout.addWidget(QLabel("Range (cm):"))
        self.pos_range = QDoubleSpinBox()
        self.pos_range.setRange(0.01, 9999.0)
        self.pos_range.setValue(2.0)
        self.pos_range.setSingleStep(1.0)
        pos_layout.addWidget(self.pos_range)
        control_layout.addLayout(pos_layout)
        
        # Frequencies
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Frequencies:"))
        self.freq_spin = QSpinBox()
        self.freq_spin.setRange(1, 1100)
        self.freq_spin.setValue(31)
        freq_layout.addWidget(self.freq_spin)
        freq_layout.addWidget(QLabel("Range (Hz):"))
        self.freq_range = QDoubleSpinBox()
        # Avoid zero-span (forces unique frequencies)
        self.freq_range.setRange(0.01, 1e4)
        self.freq_range.setValue(100.0)
        freq_layout.addWidget(self.freq_range)
        control_layout.addLayout(freq_layout)
        # Frequency helper text
        self.freq_label = QLabel("Frequencies: [0]")
        self.freq_label.setWordWrap(True)
        control_layout.addWidget(self.freq_label)

        # Time resolution control
        time_res_layout = QHBoxLayout()
        time_res_layout.addWidget(QLabel("Time step (us):"))
        self.time_step_spin = QDoubleSpinBox()
        self.time_step_spin.setRange(0.1, 5000)
        self.time_step_spin.setValue(1.0)
        self.time_step_spin.setDecimals(2)
        self.time_step_spin.setSingleStep(0.1)
        self.time_step_spin.valueChanged.connect(self._update_time_step)
        time_res_layout.addWidget(self.time_step_spin)
        control_layout.addLayout(time_res_layout)

        # Extra post-sequence simulation time
        tail_layout = QHBoxLayout()
        tail_layout.addWidget(QLabel("Extra tail (ms):"))
        self.extra_tail_spin = QDoubleSpinBox()
        self.extra_tail_spin.setRange(0.0, 1e6)
        self.extra_tail_spin.setValue(1.0)
        self.extra_tail_spin.setDecimals(3)
        self.extra_tail_spin.setSingleStep(1.0)
        tail_layout.addWidget(self.extra_tail_spin)
        control_layout.addLayout(tail_layout)

        # Spectrum range control
        spec_layout = QHBoxLayout()
        spec_layout.addWidget(QLabel("Spectrum range (Hz):"))
        self.spectrum_range = QDoubleSpinBox()
        self.spectrum_range.setRange(10, 1e6)
        self.spectrum_range.setValue(2000)
        self.spectrum_range.setSingleStep(500)
        spec_layout.addWidget(self.spectrum_range)
        control_layout.addLayout(spec_layout)
        
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        left_layout.addStretch()

        # Make the left panel scrollable so controls remain reachable on smaller screens
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)

        # Footer with sticky run controls so the button stays visible
        self.simulate_button = QPushButton("Run Simulation")
        self.simulate_button.clicked.connect(self.run_simulation)
        self.progress_bar = QProgressBar()
        footer = QWidget()
        footer_layout = QVBoxLayout()
        footer_layout.setContentsMargins(0, 4, 0, 0)
        footer_layout.addWidget(self.simulate_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_simulation)
        self.preview_checkbox = QCheckBox("Preview (fast subsample)")
        footer_layout.addWidget(self.cancel_button)
        footer_layout.addWidget(self.preview_checkbox)
        footer_layout.addWidget(self.progress_bar)
        footer.setLayout(footer_layout)

        left_container = QWidget()
        left_container_layout = QVBoxLayout()
        left_container_layout.setContentsMargins(0, 0, 0, 0)
        left_container_layout.addWidget(left_scroll)
        left_container_layout.addWidget(footer)
        left_container_layout.setStretch(0, 1)
        left_container.setLayout(left_container_layout)
        
        # Right panel - Visualization + log
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Magnetization plots
        mag_widget = QWidget()
        mag_layout = QVBoxLayout()
        mag_widget.setLayout(mag_layout)

        # Add export header
        mag_header = QHBoxLayout()
        mag_header.addWidget(QLabel("Magnetization Evolution"))
        mag_header.addStretch()

        mag_export_btn = QPushButton("Export ▼")
        mag_export_menu = QMenu()
        mag_export_menu.addAction("Image (PNG)...", lambda: self._export_magnetization_image('png'))
        mag_export_menu.addAction("Image (SVG)...", lambda: self._export_magnetization_image('svg'))
        mag_export_menu.addSeparator()
        mag_export_menu.addAction("Animation (GIF/MP4)...", self._export_magnetization_animation)
        mag_export_menu.addAction("Data (CSV/NPY)...", self._export_magnetization_data)
        mag_export_btn.setMenu(mag_export_menu)
        mag_header.addWidget(mag_export_btn)
        mag_layout.addLayout(mag_header)

        # Magnetization view filter controls (align with 3D view options)
        mag_view_layout = QHBoxLayout()
        mag_view_layout.addWidget(QLabel("View mode:"))
        self.mag_view_mode = QComboBox()
        self.mag_view_mode.addItems([
            "All positions x freqs",
            "Positions @ freq",
            "Freqs @ position"
        ])
        self.mag_view_mode.currentTextChanged.connect(lambda _: self._refresh_mag_plots())
        mag_view_layout.addWidget(self.mag_view_mode)
        self.mag_view_selector_label = QLabel("All spins")
        mag_view_layout.addWidget(self.mag_view_selector_label)
        self.mag_view_selector = QSlider(Qt.Horizontal)
        self.mag_view_selector.setRange(0, 0)
        self.mag_view_selector.setValue(0)
        self.mag_view_selector.valueChanged.connect(lambda _: self._refresh_mag_plots())
        mag_view_layout.addWidget(self.mag_view_selector)
        mag_layout.addLayout(mag_view_layout)

        self.mxy_plot = pg.PlotWidget()
        self.mxy_plot.setLabel('left', 'Mx / My')
        self.mxy_plot.setLabel('bottom', 'Time', 'ms')
        self.mxy_time_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('y', width=3))
        self.mxy_time_line.hide()
        self.mxy_plot.addItem(self.mxy_time_line)
        
        self.mz_plot = pg.PlotWidget()
        self.mz_plot.setLabel('left', 'Mz')
        self.mz_plot.setLabel('bottom', 'Time', 'ms')
        self.mz_time_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('y', width=3))
        self.mz_time_line.hide()
        self.mz_plot.addItem(self.mz_time_line)

        # Allow resizing between stacked plots so lower plots stay visible
        mag_splitter = QSplitter(Qt.Vertical)
        mag_splitter.addWidget(self.mxy_plot)
        mag_splitter.addWidget(self.mz_plot)
        mag_splitter.setStretchFactor(0, 1)
        mag_splitter.setStretchFactor(1, 1)
        mag_layout.addWidget(mag_splitter)

        # Keep consistent ranges; disable autorange so manual ranges stick
        # (signal_plot is created below; set its autorange off immediately after creation)
        for plt in (self.mxy_plot, self.mz_plot):
            plt.enableAutoRange(x=False, y=False)
        
        self.tab_widget.addTab(mag_widget, "Magnetization")

        # 3D visualization
        self.mag_3d = MagnetizationViewer()
        self.mag_3d.playhead_line = self.sequence_designer.playhead_line
        self.mag_3d.play_button.clicked.connect(self._resume_vector_animation)
        self.mag_3d.pause_button.clicked.connect(self._pause_vector_animation)
        self.mag_3d.reset_button.clicked.connect(self._reset_vector_animation)
        self.mag_3d.position_changed.connect(self._set_animation_index_from_slider)
        # Note: mag_3d.position_changed is also connected to universal control in _setup_time_synchronization()
        # Keep speed in sync but disable local control
        self.mag_3d.speed_spin.valueChanged.connect(self._update_playback_speed)
        self.mag_3d.speed_spin.setEnabled(False)
        self.mag_3d.view_filter_changed.connect(lambda: self._refresh_vector_view())
        # Disable selector until data is available
        self.mag_3d.set_selector_limits(1, 1, disable=True)
        # Show controls so track/mean toggles are available
        if hasattr(self.mag_3d, "control_container"):
            self.mag_3d.control_container.setVisible(True)
        self.tab_widget.addTab(self.mag_3d, "3D Vector")
        
        # Signal plot
        signal_widget = QWidget()
        signal_layout = QVBoxLayout()
        signal_widget.setLayout(signal_layout)

        # Add export header
        signal_header = QHBoxLayout()
        signal_header.addWidget(QLabel("Signal Evolution"))
        signal_header.addStretch()

        signal_export_btn = QPushButton("Export ▼")
        signal_export_menu = QMenu()
        signal_export_menu.addAction("Image (PNG)...", lambda: self._export_signal_image('png'))
        signal_export_menu.addAction("Image (SVG)...", lambda: self._export_signal_image('svg'))
        signal_export_menu.addSeparator()
        signal_export_menu.addAction("Animation (GIF/MP4)...", self._export_signal_animation)
        signal_export_menu.addAction("Data (CSV/NPY)...", self._export_signal_data)
        signal_export_btn.setMenu(signal_export_menu)
        signal_header.addWidget(signal_export_btn)
        signal_layout.addLayout(signal_header)

        self.signal_plot = pg.PlotWidget()
        self.signal_plot.setLabel('left', 'Signal')
        self.signal_plot.setLabel('bottom', 'Time', 'ms')
        self.signal_plot.enableAutoRange(x=False, y=False)
        self.signal_time_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('y', width=3))
        self.signal_time_line.hide()
        self.signal_plot.addItem(self.signal_time_line)
        signal_layout.addWidget(self.signal_plot)

        self.tab_widget.addTab(signal_widget, "Signal")

        # Share time cursor lines with the 3D viewer for synchronized scrubbing
        self.mag_3d.mxy_time_line = self.mxy_time_line
        self.mag_3d.mz_time_line = self.mz_time_line
        self.mag_3d.signal_time_line = self.signal_time_line
        
        # Frequency spectrum
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel('left', 'Magnitude')
        self.spectrum_plot.setLabel('bottom', 'Frequency', 'Hz')
        spectrum_container = QWidget()
        spectrum_layout = QVBoxLayout()

        # Add export header for spectrum
        spectrum_header = QHBoxLayout()
        spectrum_header.addWidget(QLabel("Frequency Spectrum"))
        spectrum_header.addStretch()

        spectrum_export_btn = QPushButton("Export ▼")
        spectrum_export_menu = QMenu()
        spectrum_export_menu.addAction("Image (PNG)...", lambda: self._export_spectrum_image('png'))
        spectrum_export_menu.addAction("Image (SVG)...", lambda: self._export_spectrum_image('svg'))
        spectrum_export_menu.addSeparator()
        spectrum_export_menu.addAction("Animation (GIF/MP4)...", self._export_spectrum_animation)
        spectrum_export_menu.addAction("Data (CSV/NPY)...", self._export_spectrum_data)
        spectrum_export_btn.setMenu(spectrum_export_menu)
        spectrum_header.addWidget(spectrum_export_btn)
        spectrum_layout.addLayout(spectrum_header)

        spectrum_controls = QHBoxLayout()
        spectrum_controls.addWidget(QLabel("Spectrum view:"))
        self.spectrum_mode = QComboBox()
        self.spectrum_mode.addItems(["Mean only", "Mean + individuals", "Individual (select pos)"])
        self.spectrum_mode.currentIndexChanged.connect(lambda _: self.update_plots(self.last_result) if self.last_result else None)
        spectrum_controls.addWidget(self.spectrum_mode)
        self.spectrum_pos_slider = QSlider(Qt.Horizontal)
        self.spectrum_pos_slider.setMinimum(0)
        self.spectrum_pos_slider.setMaximum(0)
        self.spectrum_pos_slider.setValue(0)
        self.spectrum_pos_slider.valueChanged.connect(lambda _: self.update_plots(self.last_result) if self.last_result else None)
        self.spectrum_pos_label = QLabel("Pos idx: 0")
        spectrum_controls.addWidget(self.spectrum_pos_label)
        spectrum_controls.addWidget(self.spectrum_pos_slider)
        spectrum_layout.addLayout(spectrum_controls)

        # Toggle for colored frequency/position markers
        self.spectrum_markers_checkbox = QCheckBox("Show colored frequency markers")
        self.spectrum_markers_checkbox.setChecked(False)
        self.spectrum_markers_checkbox.setToolTip("Display vertical lines at each frequency with 3D-view colors")
        self.spectrum_markers_checkbox.toggled.connect(lambda _: self.update_plots(self.last_result) if self.last_result else None)
        spectrum_layout.addWidget(self.spectrum_markers_checkbox)

        spectrum_layout.addWidget(self.spectrum_plot)
        spectrum_container.setLayout(spectrum_layout)
        self.tab_widget.addTab(spectrum_container, "Spectrum")

        # Spatial profile plot (Mxy and Mz vs position at selected time)
        # Note: Time control is now unified in the universal control below the tabs
        spatial_container = QWidget()
        spatial_layout = QVBoxLayout()

        # Add export header for spatial
        spatial_header = QHBoxLayout()
        spatial_header.addWidget(QLabel("Spatial Profile"))
        spatial_header.addStretch()

        spatial_export_btn = QPushButton("Export ▼")
        spatial_export_menu = QMenu()
        spatial_export_menu.addAction("Image (PNG)...", lambda: self._export_spatial_image('png'))
        spatial_export_menu.addAction("Image (SVG)...", lambda: self._export_spatial_image('svg'))
        spatial_export_menu.addSeparator()
        spatial_export_menu.addAction("Animation (GIF/MP4)...", self._export_spatial_animation)
        spatial_export_menu.addAction("Data (CSV/NPY)...", self._export_spatial_data)
        spatial_export_btn.setMenu(spatial_export_menu)
        spatial_header.addWidget(spatial_export_btn)
        spatial_layout.addLayout(spatial_header)

        # Display controls
        self.mean_only_checkbox = QCheckBox("Mean only (Mag/Signal/3D)")
        self.mean_only_checkbox.stateChanged.connect(lambda _: self.update_plots(self.last_result) if self.last_result else None)
        spatial_layout.addWidget(self.mean_only_checkbox)

        spatial_controls = QHBoxLayout()
        spatial_controls.addWidget(QLabel("Spatial view:"))
        self.spatial_mode = QComboBox()
        self.spatial_mode.addItems(["Mean only", "Mean + individuals", "Individual (select freq)"])
        self.spatial_mode.currentIndexChanged.connect(lambda _: self.update_spatial_plot_from_last_result())
        spatial_controls.addWidget(self.spatial_mode)
        self.spatial_freq_slider = QSlider(Qt.Horizontal)
        self.spatial_freq_slider.setMinimum(0)
        self.spatial_freq_slider.setMaximum(0)
        self.spatial_freq_slider.setValue(0)
        self.spatial_freq_slider.valueChanged.connect(lambda _: self.update_spatial_plot_from_last_result())
        self.spatial_freq_label = QLabel("Freq idx: 0")
        spatial_controls.addWidget(self.spatial_freq_label)
        spatial_controls.addWidget(self.spatial_freq_slider)
        spatial_layout.addLayout(spatial_controls)

        # Toggle for colored position/frequency markers
        self.spatial_markers_checkbox = QCheckBox("Show colored position/frequency markers")
        self.spatial_markers_checkbox.setChecked(False)
        self.spatial_markers_checkbox.setToolTip("Display vertical lines at each position/frequency with 3D-view colors")
        self.spatial_markers_checkbox.toggled.connect(lambda _: self.update_spatial_plot_from_last_result())
        spatial_layout.addWidget(self.spatial_markers_checkbox)

        # Mxy and Mz plots side by side
        spatial_plots_layout = QHBoxLayout()

        # Mxy vs position plot
        self.spatial_mxy_plot = pg.PlotWidget()
        self.spatial_mxy_plot.setLabel('left', 'Mxy (transverse)')
        self.spatial_mxy_plot.setLabel('bottom', 'Position', 'm')
        self.spatial_mxy_plot.enableAutoRange(x=False, y=False)
        # Slice thickness guides (added once and reused)
        slice_pen = pg.mkPen((180, 180, 180), style=Qt.DashLine)
        self.spatial_slice_lines = {
            "mxy": [pg.InfiniteLine(angle=90, pen=slice_pen), pg.InfiniteLine(angle=90, pen=slice_pen)],
            "mz": [pg.InfiniteLine(angle=90, pen=slice_pen), pg.InfiniteLine(angle=90, pen=slice_pen)],
        }
        for ln in self.spatial_slice_lines["mxy"]:
            self.spatial_mxy_plot.addItem(ln)
        spatial_plots_layout.addWidget(self.spatial_mxy_plot)

        # Mz vs position plot
        self.spatial_mz_plot = pg.PlotWidget()
        self.spatial_mz_plot.setLabel('left', 'Mz (longitudinal)')
        self.spatial_mz_plot.setLabel('bottom', 'Position', 'm')
        self.spatial_mz_plot.enableAutoRange(x=False, y=False)
        for ln in self.spatial_slice_lines["mz"]:
            self.spatial_mz_plot.addItem(ln)
        spatial_plots_layout.addWidget(self.spatial_mz_plot)

        spatial_layout.addLayout(spatial_plots_layout)

        spatial_container.setLayout(spatial_layout)
        self.tab_widget.addTab(spatial_container, "Spatial")

        # Add time cursor lines to spatial plots for synchronization
        self.spatial_mxy_time_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('y', width=2))
        self.spatial_mxy_time_line.hide()
        self.spatial_mxy_plot.addItem(self.spatial_mxy_time_line)

        self.spatial_mz_time_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('y', width=2))
        self.spatial_mz_time_line.hide()
        self.spatial_mz_plot.addItem(self.spatial_mz_time_line)

        # Ensure slice guides persist after clear()
        for ln in self.spatial_slice_lines["mxy"]:
            self.spatial_mxy_plot.addItem(ln)
        for ln in self.spatial_slice_lines["mz"]:
            self.spatial_mz_plot.addItem(ln)

        # Share spatial time lines with the 3D viewer for synchronized scrubbing
        self.mag_3d.spatial_mxy_time_line = self.spatial_mxy_time_line
        self.mag_3d.spatial_mz_time_line = self.spatial_mz_time_line

        right_layout.addWidget(self.tab_widget)

        # Universal time control - controls all time-resolved views
        self.time_control = UniversalTimeControl()
        right_layout.addWidget(self.time_control)
        self.time_control.setVisible(False)  # Hidden until time-resolved simulation runs
        # Keep 3D viewer speed control in sync with universal control and disable local knob
        if hasattr(self.mag_3d, "speed_spin"):
            self.mag_3d.speed_spin.setValue(self.time_control.speed_spin.value())
            self.mag_3d.speed_spin.setEnabled(False)

        # Log console
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFixedHeight(140)
        right_layout.addWidget(self.log_widget)
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_container)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)
        
        # Push initial time-step into designers
        self._update_time_step(self.time_step_spin.value())
        
        # Menu bar
        self.create_menu()

        # Status bar
        self.statusBar().showMessage("Ready")

        # Connect universal time control to all views
        self._setup_time_synchronization()

    def _color_for_index(self, idx: int, total: int):
        """Consistent color cycling for multiple frequencies."""
        return pg.intColor(idx, hues=max(total, 1), values=1.0, maxValue=255)

    def _safe_clear_plot(self, plot_widget, persistent_items=None):
        """
        Safely clear a plot widget and re-add persistent items.

        This prevents Qt warnings about items being removed from the wrong scene.

        Parameters
        ----------
        plot_widget : pg.PlotWidget
            The plot widget to clear
        persistent_items : list, optional
            List of items to re-add after clearing (e.g., cursor lines, guide lines)
        """
        plot_widget.clear()
        if persistent_items:
            for item in persistent_items:
                if item is not None:
                    # Check if item is actually in the scene before adding
                    # After clear(), items should not be in the plot
                    try:
                        if item.scene() is None:
                            plot_widget.addItem(item)
                    except (AttributeError, RuntimeError):
                        # Item doesn't have scene() method or was deleted, skip it
                        pass

    def _reshape_to_tpf(self, arr: np.ndarray, pos_len: int, freq_len: int):
        """
        Ensure array shape is (ntime, npos, nfreq).
        Tries to infer axis order using known pos/freq lengths.
        """
        if arr is None:
            return arr
        if arr.ndim == 2:
            # Heuristics for 2D: assume either (time, freq) or (time, pos)
            if arr.shape[1] == freq_len and pos_len == 1:
                return arr[:, None, :]
            if arr.shape[1] == pos_len and freq_len == 1:
                return arr[:, :, None]
            if arr.shape[0] == freq_len and pos_len == 1:
                return arr.T[:, None, :]
            if arr.shape[0] == pos_len and freq_len == 1:
                return arr[:, :, None]
            return arr
        if arr.ndim != 3:
            return arr
        shape = arr.shape
        # Already correct
        if shape[1] == pos_len and shape[2] == freq_len:
            return arr
        # Try permutations
        for perm in ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            if shape[perm[1]] == pos_len and shape[perm[2]] == freq_len:
                return np.transpose(arr, perm)
        # Heuristic: if first axis equals freq_len, assume (freq, time, pos) or (freq, pos, time)
        if shape[0] == freq_len:
            if shape[1] == pos_len:
                return np.transpose(arr, (2, 1, 0))  # (time, pos, freq)
            elif shape[2] == pos_len:
                return np.transpose(arr, (1, 2, 0))  # (time, pos, freq)
        # Heuristic: if first axis equals pos_len, assume (pos, time, freq) or (pos, freq, time)
        if shape[0] == pos_len:
            if shape[1] == freq_len:
                return np.transpose(arr, (2, 0, 1))
            elif shape[2] == freq_len:
                return np.transpose(arr, (1, 0, 2))
        return arr  # fallback

    def _normalize_time_length(self, time: np.ndarray, ntime: int):
        """Match time array length to ntime if simulator returns a different length."""
        if time is None:
            return np.arange(ntime)
        if len(time) == ntime:
            return time
        if len(time) > 1:
            dt = (time[-1] - time[0]) / (len(time) - 1)
        else:
            dt = 1.0
        return np.arange(ntime) * dt

    def _playback_to_full_index(self, playback_idx: int) -> int:
        """Map a playback index to the corresponding full-resolution time index."""
        if self.playback_indices is None or len(self.playback_indices) == 0:
            return int(playback_idx)
        playback_idx = int(max(0, min(playback_idx, len(self.playback_indices) - 1)))
        return int(self.playback_indices[playback_idx])

    def _color_tuple(self, pg_color):
        """Convert a pyqtgraph color to an RGBA tuple for OpenGL items."""
        try:
            return pg_color.getRgbF()
        except Exception:
            try:
                return pg.mkColor(pg_color).getRgbF()
            except Exception:
                return (1, 0, 0, 1)

    def _build_playback_indices(self, total_frames: int) -> np.ndarray:
        """Construct (optionally downsampled) playback indices for animation."""
        indices = np.arange(total_frames, dtype=int)
        max_frames = 0  # set >0 to limit playback frames if needed
        if max_frames and total_frames > max_frames:
            step = max(1, math.ceil(total_frames / max_frames))
            indices = np.arange(0, total_frames, step, dtype=int)
            if indices[-1] != total_frames - 1:
                indices = np.append(indices, total_frames - 1)
        return indices

    def _refresh_vector_view(self, mean_only: bool = None, restart: bool = True):
        """Apply the current vector filter (all/pos/freq) to the 3D view."""
        if self.anim_vectors_full is None or self.playback_indices is None or self.playback_time is None:
            return
        if mean_only is None:
            mean_only = self.mean_only_checkbox.isChecked()

        base_vectors = self.anim_vectors_full
        # Downsample to playback timeline if needed
        if base_vectors.shape[0] != len(self.playback_indices):
            base_vectors = base_vectors[self.playback_indices]

        nframes, npos, nfreq, _ = base_vectors.shape
        mode = self.mag_3d.get_view_mode()
        selector = self.mag_3d.get_selector_index()

        if mean_only or (npos == 1 and nfreq == 1):
            anim = np.mean(base_vectors, axis=(1, 2), keepdims=True)
            colors = [self._color_tuple(pg.mkColor('c'))]
        elif mode == "Positions @ freq":
            fi = min(max(selector, 0), nfreq - 1)
            anim = base_vectors[:, :, fi, :]
            colors = [self._color_tuple(self._color_for_index(i, npos)) for i in range(npos)]
        elif mode == "Freqs @ position":
            pi = min(max(selector, 0), npos - 1)
            anim = base_vectors[:, pi, :, :]
            colors = [self._color_tuple(self._color_for_index(i, nfreq)) for i in range(nfreq)]
        else:
            anim = base_vectors.reshape(nframes, npos * nfreq, 3)
            total = npos * nfreq
            colors = [self._color_tuple(self._color_for_index(i, total)) for i in range(total)]

        self.anim_data = anim
        self.anim_colors = colors
        self.anim_time = self.playback_time

        # Update preview plot with mean vectors
        mean_vectors = np.mean(anim, axis=1)
        self.mag_3d.set_preview_data(
            self.playback_time_ms if self.playback_time_ms is not None else self.playback_time,
            mean_vectors[:, 0],
            mean_vectors[:, 1],
            mean_vectors[:, 2],
        )
        # Ensure vector count matches filter
        self.mag_3d._ensure_vectors(anim.shape[1], colors=colors)
        if restart:
            self._start_vector_animation()
        else:
            if self.anim_index >= len(anim):
                self.anim_index = 0
            self._set_animation_index_from_slider(self.anim_index)

    def _extend_sequence_with_tail(self, sequence_tuple, tail_ms: float, dt: float):
        """Append a zero-B1/gradient tail after the sequence to continue acquisition."""
        tail_s = max(0.0, tail_ms) * 1e-3
        if tail_s <= 0:
            return sequence_tuple
        b1, gradients, time = sequence_tuple
        b1 = np.asarray(b1, dtype=complex)
        gradients = np.asarray(gradients, dtype=float)
        time = np.asarray(time, dtype=float)
        if time.size == 0:
            return (b1, gradients, time)
        if gradients.ndim == 1:
            gradients = gradients.reshape(-1, 1)
        if gradients.shape[1] < 3:
            gradients = np.pad(gradients, ((0, 0), (0, 3 - gradients.shape[1])), mode='constant')
        elif gradients.shape[1] > 3:
            gradients = gradients[:, :3]

        dt_use = max(dt, 1e-9)
        if len(time) > 1:
            diffs = np.diff(time)
            with np.errstate(invalid='ignore'):
                mean_dt = float(np.nanmean(diffs))
            if np.isfinite(mean_dt) and mean_dt > 0:
                dt_use = mean_dt

        n_tail = int(math.ceil(tail_s / dt_use))
        if n_tail <= 0:
            return (b1, gradients, time)

        tail_time = time[-1] + np.arange(1, n_tail + 1) * dt_use
        b1_tail = np.zeros(n_tail, dtype=complex)
        grad_tail = np.zeros((n_tail, gradients.shape[1]), dtype=float)

        b1_ext = np.concatenate([b1, b1_tail])
        gradients_ext = np.vstack([gradients, grad_tail])
        time_ext = np.concatenate([time, tail_time])
        return (b1_ext, gradients_ext, time_ext)

    def log_message(self, message: str):
        """Append a message to the log console."""
        self.log_widget.append(message)
        self.log_widget.moveCursor(self.log_widget.textCursor().End)

    def _update_time_step(self, us_value: float):
        """Propagate desired time resolution (microseconds) to designers."""
        dt_s = max(us_value, 0.1) * 1e-6
        self.rf_designer.set_time_step(dt_s)
        self.sequence_designer.set_time_step(dt_s)
        self.sequence_designer.update_diagram(self.rf_designer.get_pulse())

    def _setup_time_synchronization(self):
        """Setup connections for universal time control synchronization."""
        # Connect universal time control to update all views
        self.time_control.time_changed.connect(self._on_universal_time_changed)

        # Connect play/pause/reset buttons
        self.time_control.play_button.clicked.connect(self._on_universal_play)
        self.time_control.pause_button.clicked.connect(self._on_universal_pause)
        self.time_control.reset_button.clicked.connect(self._on_universal_reset)
        self.time_control.speed_spin.valueChanged.connect(self._update_playback_speed)
        # Mirror speed setting into the 3D viewer knob (kept disabled for clarity)
        if hasattr(self.mag_3d, "speed_spin"):
            self.time_control.speed_spin.valueChanged.connect(self.mag_3d.speed_spin.setValue)

        # Connect 3D vector position changes to universal control
        self.mag_3d.position_changed.connect(self._on_3d_vector_position_changed)

    def _on_universal_time_changed(self, time_idx: int):
        """Central handler for universal time control changes - updates all views."""
        if not hasattr(self, 'last_time') or self.last_time is None:
            return
        actual_idx = self._playback_to_full_index(time_idx)
        # Convert to ms for sequence diagram alignment
        if hasattr(self.sequence_designer, 'set_cursor_index'):
            self.sequence_designer.set_cursor_index(actual_idx)

        # Update sequence diagram cursor
        # Update 3D vector view
        self.mag_3d.set_cursor_index(time_idx)
        self._set_animation_index_from_slider(time_idx)

        # Update spatial view with current time index
        self.update_spatial_plot_from_last_result(time_idx=actual_idx)

        # Update time cursors on plots
        if self.last_time is not None and 0 <= actual_idx < len(self.last_time):
            time_ms = self.last_time[actual_idx] * 1000
            self.mxy_time_line.setValue(time_ms)
            self.mz_time_line.setValue(time_ms)
            self.signal_time_line.setValue(time_ms)
        # Refresh spectrum for this time
        self._refresh_spectrum(time_idx=actual_idx)

    def _on_3d_vector_position_changed(self, time_idx: int):
        """Handle 3D vector view position changes."""
        if not self.time_control._updating:
            self.time_control.set_time_index(time_idx)
        # Propagate the change to all synchronized views
        self._on_universal_time_changed(time_idx)

    def _on_universal_play(self):
        """Handle universal play button."""
        # Use latest speed setting
        self._recompute_anim_interval(self.time_control.speed_spin.value())
        self._resume_vector_animation()

    def _on_universal_pause(self):
        """Handle universal pause button."""
        self._pause_vector_animation()

    def _on_universal_reset(self):
        """Handle universal reset button."""
        self._reset_vector_animation()

    def _set_plot_ranges(self, plot_widget, x_min, x_max, y_min=None, y_max=None):
        """Apply consistent axis ranges."""
        x_min = max(0, x_min)
        # Avoid zero span which can break setRange/limits
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        plot_widget.enableAutoRange(x=False)
        plot_widget.setXRange(x_min, x_max, padding=0)
        limits = {'xMin': 0, 'xMax': x_max}
        if y_min is not None and y_max is not None:
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
            plot_widget.enableAutoRange(y=False)
            plot_widget.setYRange(y_min, y_max, padding=0)
            limits.update({'yMin': y_min, 'yMax': y_max})
        plot_widget.setLimits(**limits)

    def _refresh_mag_plots(self):
        """Re-render magnetization plots using the current filter selection."""
        if self.last_result is not None:
            self.update_plots(self.last_result)

    def _calc_symmetric_limits(self, *arrays, base=1.0, pad=1.1):
        """Compute symmetric y-limits with padding based on provided arrays."""
        max_abs = 0.0
        for arr in arrays:
            if arr is None:
                continue
            arr_np = np.asarray(arr)
            if arr_np.size == 0:
                continue
            with np.errstate(invalid='ignore'):
                current = np.nanmax(np.abs(arr_np))
            if np.isfinite(current):
                max_abs = max(max_abs, float(current))
        max_abs = max(base, max_abs)
        return -pad * max_abs, pad * max_abs

    def _reset_legend(self, plot_widget, attr_name: str, enable: bool):
        """Reset/add a legend on the given plot."""
        existing = getattr(self, attr_name, None)
        if existing is not None:
            try:
                plot_widget.scene().removeItem(existing)
            except Exception:
                pass
            setattr(self, attr_name, None)
        if enable:
            legend = plot_widget.addLegend(offset=(6, 6))
            legend.layout.setSpacing(4)
            setattr(self, attr_name, legend)
            return legend
        return None

    def _update_mag_selector_limits(self, npos: int, nfreq: int, disable: bool = False):
        """Sync magnetization view selector with available pos/freq counts."""
        if not hasattr(self, "mag_view_mode"):
            return
        mode = self.mag_view_mode.currentText()
        if disable:
            max_idx = 0
            prefix = "All"
        elif mode == "Positions @ freq":
            max_idx = max(0, nfreq - 1)
            prefix = "Freq"
        elif mode == "Freqs @ position":
            max_idx = max(0, npos - 1)
            prefix = "Pos"
        else:
            max_idx = 0
            prefix = "All"
        slider = self.mag_view_selector
        slider.blockSignals(True)
        slider.setMaximum(max_idx)
        slider.setValue(min(slider.value(), max_idx) if max_idx > 0 else 0)
        slider.setEnabled(not disable and max_idx > 0)
        slider.setVisible(max_idx > 0)
        slider.blockSignals(False)
        if max_idx > 0:
            self.mag_view_selector_label.setText(f"{prefix} idx: {slider.value()}")
        else:
            self.mag_view_selector_label.setText("All spins")

    def _current_mag_filter(self, npos: int, nfreq: int):
        """Return the active magnetization view filter selection."""
        mode = self.mag_view_mode.currentText() if hasattr(self, "mag_view_mode") else "All positions x freqs"
        if mode == "Positions @ freq":
            idx = min(max(self.mag_view_selector.value(), 0), max(0, nfreq - 1))
        elif mode == "Freqs @ position":
            idx = min(max(self.mag_view_selector.value(), 0), max(0, npos - 1))
        else:
            idx = 0
        return mode, idx

    def _apply_pulse_region(self, plot_widget, attr_name):
        """Highlight pulse duration on a plot."""
        existing = getattr(self, attr_name, None)
        if existing is not None:
            plot_widget.removeItem(existing)
        if self.last_pulse_range is None:
            setattr(self, attr_name, None)
            return
        start_ms = self.last_pulse_range[0] * 1000
        end_ms = self.last_pulse_range[1] * 1000
        region = pg.LinearRegionItem(values=[start_ms, end_ms],
                                     brush=pg.mkBrush(100, 100, 255, 40),
                                     movable=False)
        region.setZValue(-10)
        plot_widget.addItem(region)
        setattr(self, attr_name, region)

    def update_spatial_plot_from_last_result(self, time_idx=None):
        """Update spatial plot with Mxy and Mz profiles across positions at selected time.

        For a given time point (or endpoint), plots:
        - Mxy (transverse magnetization) vs position
        - Mz (longitudinal magnetization) vs position

        Parameters
        ----------
        time_idx : int, optional
            Time index to display. If None, uses last time point for time-resolved data.
        """
        if self.last_result is None or self.last_positions is None:
            self.log_message("Spatial plot: missing result or positions")
            return

        result = self.last_result
        mx = result.get('mx')
        my = result.get('my')
        mz = result.get('mz')

        # Validate data
        if mx is None or my is None or mz is None:
            self.log_message("Spatial plot: missing mx, my, or mz data")
            return

        self.log_message(f"Spatial plot: mx shape = {mx.shape}, my shape = {my.shape}, mz shape = {mz.shape}")

        # Handle time-resolved vs endpoint data
        is_time_resolved = len(mz.shape) == 3  # (ntime, npos, nfreq)

        if is_time_resolved:
            # Store the full time-resolved data
            self.spatial_mx_time_series = mx
            self.spatial_my_time_series = my
            self.spatial_mz_time_series = mz
            ntime = mz.shape[0]

            # Use provided time index or default to last time point
            if time_idx is None:
                time_idx = ntime - 1
            time_idx = min(max(0, time_idx), ntime - 1)

            mx_display = mx[time_idx, :, :]
            my_display = my[time_idx, :, :]
            mz_display = mz[time_idx, :, :]
        else:
            # Endpoint mode: (npos, nfreq)
            mx_display = mx
            my_display = my
            mz_display = mz
            self.spatial_mx_time_series = None
            self.spatial_my_time_series = None
            self.spatial_mz_time_series = None
            time_idx = 0

        # Frequency selection/averaging for spatial view
        freq_count = mx_display.shape[1]
        self.spatial_freq_slider.setMaximum(max(0, freq_count - 1))
        freq_sel = min(self.spatial_freq_slider.value(), freq_count - 1)
        self.spatial_freq_label.setText(f"Freq idx: {freq_sel}")

        spatial_mode = self.spatial_mode.currentText()
        if spatial_mode == "Mean only":
            mxy_pos = np.sqrt(mx_display**2 + my_display**2).mean(axis=1)
            mz_pos = mz_display.mean(axis=1)
        elif spatial_mode == "Mean + individuals":
            mxy_pos = np.sqrt(mx_display**2 + my_display**2).mean(axis=1)
            mz_pos = mz_display.mean(axis=1)
        else:  # Individual (select freq)
            mxy_pos = np.sqrt(mx_display[:, freq_sel]**2 + my_display[:, freq_sel]**2)
            mz_pos = mz_display[:, freq_sel]

        # Choose a signed position axis (prefer the axis with largest span)
        pos_axis = self.last_positions
        spans = np.ptp(pos_axis, axis=0)  # max - min per axis
        axis_idx = int(np.argmax(spans))
        pos_distance = pos_axis[:, axis_idx]

        self.log_message(f"Spatial plot: mxy_pos shape = {mxy_pos.shape}, mz_pos shape = {mz_pos.shape}, pos_distance shape = {pos_distance.shape}")

        # Cache data for export
        self._last_spatial_export = {
            "position_m": pos_distance,
            "mxy": mxy_pos,
            "mz": mz_pos,
            "freq_index": freq_sel,
            "time_idx": time_idx,
            "time_s": self.last_time[time_idx] if self.last_time is not None and len(self.last_time) > time_idx else None,
            "mxy_per_freq": np.sqrt(mx_display**2 + my_display**2),
            "mz_per_freq": mz_display,
        }

        # Update plots
        self._update_spatial_line_plots(pos_distance, mxy_pos, mz_pos, mx_display, my_display, mz_display, freq_sel, spatial_mode)

        # Show time lines if in time-resolved mode
        if is_time_resolved and time_idx < len(self.last_time):
            current_time = self.last_time[time_idx]
            if hasattr(self, 'spatial_mxy_time_line'):
                self.spatial_mxy_time_line.setValue(current_time)
                self.spatial_mxy_time_line.show()
            if hasattr(self, 'spatial_mz_time_line'):
                self.spatial_mz_time_line.setValue(current_time)
                self.spatial_mz_time_line.show()
            # Synchronize sequence diagram playhead
            if hasattr(self, 'sequence_designer') and hasattr(self.sequence_designer, 'playhead_line'):
                if self.sequence_designer.playhead_line is not None:
                    self.sequence_designer.playhead_line.setValue(current_time * 1000.0)
                    if not self.sequence_designer.playhead_line.isVisible():
                        self.sequence_designer.playhead_line.show()

    def _update_spatial_line_plots(self, position, mxy, mz, mx_display=None, my_display=None, mz_display=None, freq_sel=0, spatial_mode="Mean only"):
        """Update the Mxy and Mz line plots."""
        try:
            # Safely clear plots while preserving persistent items
            persistent_mxy = [self.spatial_mxy_time_line] + self.spatial_slice_lines["mxy"]
            persistent_mz = [self.spatial_mz_time_line] + self.spatial_slice_lines["mz"]
            self._safe_clear_plot(self.spatial_mxy_plot, persistent_mxy)
            self._safe_clear_plot(self.spatial_mz_plot, persistent_mz)

            # Plot Mxy vs position
            if spatial_mode == "Mean + individuals" and mx_display is not None and my_display is not None and mz_display is not None:
                total_series = mx_display.shape[1]
                self._reset_legend(self.spatial_mxy_plot, "spatial_mxy_legend", total_series > 1)
                self._reset_legend(self.spatial_mz_plot, "spatial_mz_legend", total_series > 1)
                for fi in range(total_series):
                    color = self._color_for_index(fi, total_series)
                    mxy_ind = np.sqrt(mx_display[:, fi]**2 + my_display[:, fi]**2)
                    self.spatial_mxy_plot.plot(position, mxy_ind, pen=pg.mkPen(color, width=1), name=f"f{fi}")
                    self.spatial_mz_plot.plot(position, mz_display[:, fi], pen=pg.mkPen(color, width=1), name=f"f{fi}")
                self.spatial_mxy_plot.plot(position, mxy, pen=pg.mkPen('b', width=3), name='|Mxy| mean')
                mx_mean = np.mean(mx_display, axis=1)
                my_mean = np.mean(my_display, axis=1)
                self.spatial_mxy_plot.plot(position, mx_mean, pen=pg.mkPen('r', style=Qt.DashLine, width=2), name='Mx mean')
                self.spatial_mxy_plot.plot(position, my_mean, pen=pg.mkPen('g', style=Qt.DotLine, width=2), name='My mean')
                self.spatial_mz_plot.plot(position, mz, pen=pg.mkPen('m', width=3), name='Mz mean')
            else:
                self._reset_legend(self.spatial_mxy_plot, "spatial_mxy_legend", True)
                self._reset_legend(self.spatial_mz_plot, "spatial_mz_legend", False)
                # Plot magnitude plus real/imag (Mx/My)
                self.spatial_mxy_plot.plot(position, mxy, pen=pg.mkPen('b', width=2), name='|Mxy|')
                mx_line = None
                my_line = None
                if mx_display is not None and my_display is not None:
                    if spatial_mode == "Individual (select freq)" and mx_display.ndim == 2:
                        mx_line = mx_display[:, freq_sel]
                        my_line = my_display[:, freq_sel]
                    elif mx_display.ndim == 2:
                        mx_line = np.mean(mx_display, axis=1)
                        my_line = np.mean(my_display, axis=1)
                    else:
                        mx_line = mx_display
                        my_line = my_display
                if mx_line is not None:
                    self.spatial_mxy_plot.plot(position, mx_line, pen=pg.mkPen('r', style=Qt.DashLine, width=2), name='Mx')
                if my_line is not None:
                    self.spatial_mxy_plot.plot(position, my_line, pen=pg.mkPen('g', style=Qt.DotLine, width=2), name='My')
                self.spatial_mz_plot.plot(position, mz, pen=pg.mkPen('r', width=2))
            self.spatial_mxy_plot.setTitle("Transverse Magnetization")
            self.spatial_mz_plot.setTitle("Longitudinal Magnetization")

            # Set consistent axis ranges based on full series if available
            pos_min, pos_max = position.min(), position.max()
            pos_pad = (pos_max - pos_min) * 0.1 if pos_max > pos_min else 0.1

            if self.spatial_mx_time_series is not None and self.spatial_my_time_series is not None:
                mxy_series = np.sqrt(self.spatial_mx_time_series**2 + self.spatial_my_time_series**2)
                mxy_min_all = float(np.nanmin(mxy_series))
                mxy_max_all = float(np.nanmax(mxy_series))
                mx_all = self.spatial_mx_time_series
                my_all = self.spatial_my_time_series
            else:
                mxy_min_all = float(np.nanmin(mxy))
                mxy_max_all = float(np.nanmax(mxy))
                mx_all = mx_display if mx_display is not None else None
                my_all = my_display if my_display is not None else None
            if self.spatial_mz_time_series is not None:
                mz_min_all = float(np.nanmin(self.spatial_mz_time_series))
                mz_max_all = float(np.nanmax(self.spatial_mz_time_series))
            else:
                mz_min_all = float(np.nanmin(mz))
                mz_max_all = float(np.nanmax(mz))

            # Expand transverse range to include real/imag components
            if mx_all is not None:
                mxy_min_all = min(mxy_min_all, float(np.nanmin(mx_all)))
                mxy_max_all = max(mxy_max_all, float(np.nanmax(mx_all)))
            if my_all is not None:
                mxy_min_all = min(mxy_min_all, float(np.nanmin(my_all)))
                mxy_max_all = max(mxy_max_all, float(np.nanmax(my_all)))

            def padded_range(vmin, vmax, scale=1.1):
                if np.isclose(vmin, vmax):
                    pad = max(abs(vmin) * 0.1, 0.1)
                    return vmin - pad, vmax + pad
                span = vmax - vmin
                mid = (vmax + vmin) / 2.0
                half = (span * scale) / 2.0
                return mid - half, mid + half

            mxy_ymin, mxy_ymax = padded_range(mxy_min_all, mxy_max_all, scale=1.1)
            mz_ymin, mz_ymax = padded_range(mz_min_all, mz_max_all, scale=1.1)

            self.spatial_mxy_plot.setXRange(pos_min - pos_pad, pos_max + pos_pad)
            self.spatial_mxy_plot.setYRange(mxy_ymin, mxy_ymax)

            self.spatial_mz_plot.setXRange(pos_min - pos_pad, pos_max + pos_pad)
            self.spatial_mz_plot.setYRange(mz_ymin, mz_ymax)

            # Update slice thickness guides
            slice_thk = None
            try:
                slice_thk = self.sequence_designer._slice_thickness_m()
            except Exception:
                slice_thk = None
            if slice_thk is not None and slice_thk > 0 and np.isfinite(slice_thk):
                center = float(np.median(position))
                half = slice_thk / 2.0
                positions = [center - half, center + half]
                for line, pos in zip(self.spatial_slice_lines["mxy"], positions):
                    line.setValue(pos)
                    line.setVisible(True)
                for line, pos in zip(self.spatial_slice_lines["mz"], positions):
                    line.setValue(pos)
                    line.setVisible(True)
            else:
                for line in self.spatial_slice_lines["mxy"] + self.spatial_slice_lines["mz"]:
                    line.setVisible(False)

            # Add colored vertical markers if enabled
            if self.spatial_markers_checkbox.isChecked() and mx_display is not None and my_display is not None:
                # Determine what we're marking: frequencies or positions
                nfreq = mx_display.shape[1] if mx_display.ndim == 2 else 1
                npos = len(position)

                # Show markers based on current mode
                if spatial_mode in ("Mean + individuals", "Individual (select freq)") and nfreq > 1:
                    # Mark each frequency with colored stem plots
                    for fi in range(nfreq):
                        color = self._color_for_index(fi, nfreq)
                        mxy_val = np.sqrt(mx_display[:, fi]**2 + my_display[:, fi]**2) if mx_display.ndim == 2 else mxy
                        mz_val = mz_display[:, fi] if mz_display.ndim == 2 else mz_pos
                        # Draw stem lines from 0 to current value at each position
                        for pi, pos_val in enumerate(position):
                            # Mxy markers
                            line_mxy = pg.PlotCurveItem([pos_val, pos_val], [0, mxy_val[pi]],
                                                         pen=pg.mkPen(color, width=1.5))
                            self.spatial_mxy_plot.addItem(line_mxy)
                            # Mz markers
                            line_mz = pg.PlotCurveItem([pos_val, pos_val], [0, mz_val[pi]],
                                                        pen=pg.mkPen(color, width=1.5))
                            self.spatial_mz_plot.addItem(line_mz)

            self.log_message("Spatial plot: updated successfully")
        except Exception as e:
            self.log_message(f"Spatial plot: error updating plots: {e}")
            import traceback
            self.log_message(f"Spatial plot: traceback: {traceback.format_exc()}")

    def _load_sequence_presets(self, seq_type: str):
        """Load sequence-specific parameter presets if enabled."""
        if not self.tissue_widget.sequence_presets_enabled:
            return

        presets = self.sequence_designer.get_sequence_preset_params(seq_type)
        if not presets:
            return

        # List of all widgets that might be updated
        widgets_to_block = [
            self.sequence_designer.te_spin,
            self.sequence_designer.tr_spin,
            self.sequence_designer.ti_spin,
            self.rf_designer.flip_angle,  # flip_angle is in RFPulseDesigner, not SequenceDesigner
            self.sequence_designer.ssfp_repeats,
            self.sequence_designer.ssfp_amp,
            self.sequence_designer.ssfp_phase,
            self.sequence_designer.ssfp_dur,
            self.sequence_designer.ssfp_start_delay,
            self.sequence_designer.ssfp_start_amp,
            self.sequence_designer.ssfp_start_phase,
            self.sequence_designer.ssfp_alternate_phase,
            self.rf_designer.pulse_type,
        ]

        # Block signals temporarily to avoid triggering diagram updates multiple times
        for widget in widgets_to_block:
            widget.blockSignals(True)

        # Apply presets
        if "pulse_type" in presets:
            self.rf_designer.pulse_type.setCurrentText(presets["pulse_type"])
        if "te_ms" in presets:
            self.sequence_designer.te_spin.setValue(presets["te_ms"])
        if "tr_ms" in presets:
            self.sequence_designer.tr_spin.setValue(presets["tr_ms"])
        if "ti_ms" in presets:
            self.sequence_designer.ti_spin.setValue(presets["ti_ms"])
        if "flip_angle" in presets:
            self.rf_designer.flip_angle.setValue(presets["flip_angle"])

        # SSFP-specific parameters
        if "ssfp_repeats" in presets:
            self.sequence_designer.ssfp_repeats.setValue(presets["ssfp_repeats"])
        if "ssfp_amp" in presets:
            self.sequence_designer.ssfp_amp.setValue(presets["ssfp_amp"])
        if "ssfp_phase" in presets:
            self.sequence_designer.ssfp_phase.setValue(presets["ssfp_phase"])
        if "ssfp_dur" in presets:
            self.sequence_designer.ssfp_dur.setValue(presets["ssfp_dur"])
        if "ssfp_start_delay" in presets:
            self.sequence_designer.ssfp_start_delay.setValue(presets["ssfp_start_delay"])
        if "ssfp_start_amp" in presets:
            self.sequence_designer.ssfp_start_amp.setValue(presets["ssfp_start_amp"])
        if "ssfp_start_phase" in presets:
            self.sequence_designer.ssfp_start_phase.setValue(presets["ssfp_start_phase"])
        if "ssfp_alternate_phase" in presets:
            self.sequence_designer.ssfp_alternate_phase.setChecked(presets["ssfp_alternate_phase"])

        # Re-enable signals
        for widget in widgets_to_block:
            widget.blockSignals(False)

        # Update diagram once with all new values
        self.sequence_designer.update_diagram()

        self.log_message(f"Loaded presets for {seq_type}: {presets}")

    def create_menu(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = file_menu.addAction("Load Parameters")
        load_action.triggered.connect(self.load_parameters)
        
        save_action = file_menu.addAction("Save Parameters")
        save_action.triggered.connect(self.save_parameters)
        
        file_menu.addSeparator()
        
        export_action = file_menu.addAction("Export Results")
        export_action.triggered.connect(self.export_results)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)
        
    def run_simulation(self):
        """Run the Bloch simulation."""
        self.statusBar().showMessage("Running simulation...")
        self.simulate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)

        self.log_message("Starting simulation...")
        
        # Get parameters
        tissue = self.tissue_widget.get_parameters()
        pulse = self.rf_designer.get_pulse()
        dt_s = max(self.time_step_spin.value(), 0.1) * 1e-6
        sequence_tuple = self.sequence_designer.compile_sequence(custom_pulse=pulse, dt=dt_s)
        # Optionally extend sequence with a zero tail to keep sampling after gradients/RF
        tail_ms = self.extra_tail_spin.value()
        b1_orig_len = len(sequence_tuple[0])
        sequence_tuple = self._extend_sequence_with_tail(sequence_tuple, tail_ms, dt_s)
        if len(sequence_tuple[0]) > b1_orig_len:
            added = len(sequence_tuple[0]) - b1_orig_len
            self.log_message(f"Appended {tail_ms:.3f} ms tail ({added} pts) after sequence.")

        # Always work with explicit arrays so pulse visualization matches the run
        b1_arr, gradients_arr, time_arr = sequence_tuple
        self.sequence_designer._render_sequence_diagram(b1_arr, gradients_arr, time_arr)
        self.last_b1 = np.asarray(b1_arr)
        self.last_time = np.asarray(time_arr)
        # Compute pulse window (where B1 is non-zero above a small threshold)
        b1_abs = np.abs(self.last_b1)
        thr = b1_abs.max() * 1e-3 if b1_abs.size else 0.0
        mask = b1_abs > thr
        if mask.any():
            idx = np.where(mask)[0]
            self.last_pulse_range = (self.last_time[idx[0]], self.last_time[idx[-1]])
        else:
            self.last_pulse_range = None
        # Log pulse characteristics for transparency
        if b1_abs.size:
            peak_b1 = float(b1_abs.max())
            dt = float(np.diff(self.last_time).mean()) if len(self.last_time) > 1 else 0.0
            dt_us = dt * 1e6
            dur_ms = (self.last_time[-1] - self.last_time[0]) * 1e3 if len(self.last_time) else 0.0
            self.log_message(
                f"Pulse stats: N={len(self.last_b1)}, dt≈{dt_us:.3f} µs, duration≈{dur_ms:.3f} ms, peak B1={peak_b1:.5f} G"
            )
        # Log gradient magnitudes (Gauss/cm)
        if gradients_arr.size:
            g_abs = np.max(np.abs(gradients_arr), axis=0)
            gx, gy, gz = (float(g_abs[i]) if i < len(g_abs) else 0.0 for i in range(3))
            self.log_message(f"Gradient peaks (|G|, G/cm): Gx={gx:.4f}, Gy={gy:.4f}, Gz={gz:.4f}")
        # Slice gradient sanity check (estimate vs. compiled gradients)
        try:
            pulse_duration_s = None
            if pulse is not None and len(pulse) == 2 and pulse[1] is not None and len(pulse[1]) > 1:
                pulse_duration_s = float(pulse[1][-1] - pulse[1][0] + (pulse[1][1] - pulse[1][0]))
            if pulse_duration_s is None or not np.isfinite(pulse_duration_s) or pulse_duration_s <= 0:
                pulse_duration_s = float(self.rf_designer.duration.value()) / 1000.0
            tbw_val = float(self.rf_designer.tbw.value())
            bw_hz = tbw_val / max(pulse_duration_s, 1e-9)
            slice_thk_m = self.sequence_designer._slice_thickness_m()
            thickness_cm = max(slice_thk_m, 1e-6) * 100.0
            gamma_hz_per_g = 4258.0
            expected_gz = bw_hz / (gamma_hz_per_g * thickness_cm)
            gz_peak = float(np.max(np.abs(gradients_arr[:, 2]))) if gradients_arr.ndim == 2 and gradients_arr.shape[1] >= 3 else 0.0
            ratio = gz_peak / expected_gz if expected_gz > 0 else 0.0
            override_val = self.sequence_designer._slice_gradient_override()
            self.log_message(
                f"Slice gradient check: target≈{expected_gz:.4f} G/cm (TBW={tbw_val:.2f}, BW≈{bw_hz:.1f} Hz, thickness={slice_thk_m*1000:.2f} mm); "
                f"compiled Gz peak={gz_peak:.4f} G/cm ({ratio:.2f}×, override={override_val})"
            )
            if expected_gz > 0 and abs(ratio - 1.0) > 0.15:
                self.log_message("Slice gradient warning: compiled amplitude differs >15% from estimated requirement.")
        except Exception as exc:
            self.log_message(f"Slice gradient check skipped: {exc}")
        
        # Set up positions and frequencies
        npos = self.pos_spin.value()
        pos_span_cm = self.pos_range.value()
        span_m = pos_span_cm / 100.0
        half_span = span_m / 2.0
        positions = np.zeros((npos, 3))
        if npos > 1:
            # Sample positions along the slice-selection axis (z) so slice profiles are visible
            positions[:, 2] = np.linspace(-half_span, half_span, npos)
        self.log_message(f"positions = {positions}")
        nfreq = self.freq_spin.value()
        freq_range = self.freq_range.value()
        # If multiple frequencies are requested but span is zero/non-positive, auto-expand
        if nfreq > 1 and freq_range <= 0:
            freq_range = max(1.0, nfreq - 1)  # simple 1 Hz spacing baseline
            self.freq_range.setValue(freq_range)
            self.log_message(f"Frequency span was 0; auto-set span to {freq_range:.2f} Hz for {nfreq} freqs.")
        if nfreq > 1:
            frequencies = np.linspace(-freq_range/2, freq_range/2, nfreq)
        else:
            frequencies = np.array([0.0])
        # Determine mode
        mode = 2 if self.mode_combo.currentText() == "Time-resolved" else 0

        # Optional preview mode for faster turnaround
        if self.preview_checkbox.isChecked():
            prev_stride = max(1, int(np.ceil(npos / 64)))  # cap preview positions
            freq_stride = max(1, int(np.ceil(nfreq / 16)))
            if prev_stride > 1:
                positions = positions[::prev_stride]
                npos = positions.shape[0]
            if freq_stride > 1:
                frequencies = frequencies[::freq_stride]
                nfreq = frequencies.shape[0]
            mode = 0  # preview: endpoint only
            dt_s *= 4  # coarser step for speed
            self.log_message(f"Preview mode: subsampled positions (stride {prev_stride}), frequencies (stride {freq_stride}), dt scaled x4, endpoint only.")

        # Initial magnetization (Mz along z) after any preview subsampling
        m0 = self.tissue_widget.get_initial_mz()
        self.initial_mz = abs(m0) if np.isfinite(m0) else 1.0
        nfnpos = nfreq * npos
        m_init = np.zeros((3, nfnpos))
        m_init[2, :] = m0
        if m0 != 1.0:
            self.log_message(f"Initial magnetization set to Mz={m0:.3f}")
        # Normalize 3D view to the chosen initial magnetization
        self.mag_3d.set_length_scale(self.initial_mz)

        self.log_message(
            f"Mode: {'Time-resolved' if mode == 2 else 'Endpoint'}, "
            f"positions: {positions.shape}, frequencies: {frequencies.shape}"
        )
        self.log_message(
            f"B1 len: {len(sequence_tuple[0])}, grad shape: {sequence_tuple[1].shape}"
        )
        self.last_positions = positions
        self.last_frequencies = frequencies
        freq_str = ", ".join(f"{f:.1f}" for f in frequencies[:5])
        if len(frequencies) > 5:
            freq_str += ", ..."
        self.freq_label.setText(f"Frequencies (Hz): [{freq_str}] (centered at 0, span={freq_range:.1f})")
        self.log_message(f"Using frequencies (Hz): {frequencies}")
        
        # Create and start simulation thread
        self.simulation_thread = SimulationThread(
            self.simulator, sequence_tuple, tissue,
            positions, frequencies, mode, dt=dt_s, m_init=m_init
        )
        self.simulation_thread.finished.connect(self.on_simulation_finished)
        self.simulation_thread.cancelled.connect(self.on_simulation_cancelled)
        self.simulation_thread.error.connect(self.on_simulation_error)
        self.simulation_thread.progress.connect(self.progress_bar.setValue)
        self.simulation_thread.start()

    def cancel_simulation(self):
        """Request cancellation of the current simulation."""
        if hasattr(self, "simulation_thread") and self.simulation_thread.isRunning():
            self.simulation_thread.request_cancel()
            self.statusBar().showMessage("Cancellation requested...")
            self.log_message("User requested simulation cancel.")
            
    def on_simulation_finished(self, result):
        """Handle simulation completion."""
        self.statusBar().showMessage("Simulation complete")
        self.simulate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(100)
        self.log_message("Simulation completed successfully.")
        
        # Update plots
        self.update_plots(result)
        self.log_message("Magnetization plots show Mx/My/Mz over time; Signal shows received complex signal (per frequency).")
        
    def on_simulation_error(self, error_msg):
        """Handle simulation error."""
        self.statusBar().showMessage("Simulation failed")
        self.simulate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_message(f"Simulation error: {error_msg}")
        
        QMessageBox.critical(self, "Simulation Error", error_msg)

    def on_simulation_cancelled(self):
        """Handle user cancellation."""
        self.statusBar().showMessage("Simulation cancelled")
        self.simulate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_message("Simulation cancelled.")
        
    def update_plots(self, result):
        """Update all visualization plots."""
        raw_time = result['time']
        # Align arrays to (ntime, npos, nfreq)
        pos_len = self.last_positions.shape[0] if self.last_positions is not None else result['mx'].shape[-2] if result['mx'].ndim == 3 else 1
        freq_len = self.last_frequencies.shape[0] if self.last_frequencies is not None else result['mx'].shape[-1] if result['mx'].ndim == 3 else 1

        mx_arr = self._reshape_to_tpf(result['mx'], pos_len, freq_len)
        my_arr = self._reshape_to_tpf(result['my'], pos_len, freq_len)
        mz_arr = self._reshape_to_tpf(result['mz'], pos_len, freq_len)
        signal_arr = result['signal']
        if signal_arr.ndim == 3:
            signal_arr = self._reshape_to_tpf(signal_arr, pos_len, freq_len)

        # Store reshaped result for spatial plot updates
        self.last_result = result.copy()
        self.last_result['mx'] = mx_arr
        self.last_result['my'] = my_arr
        self.last_result['mz'] = mz_arr
        self.last_result['signal'] = signal_arr

        self.log_message(
            f"Shapes -> mx:{np.shape(mx_arr)}, my:{np.shape(my_arr)}, mz:{np.shape(mz_arr)}, signal:{np.shape(signal_arr)}"
        )
        if freq_len > 1 and (self.last_frequencies is not None) and np.allclose(self.last_frequencies, self.last_frequencies[0]):
            self.log_message("Warning: all frequencies are identical; increase span to visualize separate traces.")

        ntime = mx_arr.shape[0] if mx_arr.ndim == 3 else mx_arr.shape[0]
        time = self._normalize_time_length(raw_time, ntime) * 1000  # ms
        self.last_time = self._normalize_time_length(raw_time, ntime)  # Store in seconds for universal control
        if len(time) == 0:
            return
        x_min, x_max = time[0], time[-1]
        
        # Handle different result shapes
        if mx_arr.ndim == 2:
            # Endpoint mode: show last values as flat lines
            self.log_message("Endpoint mode: showing final magnetization values.")
            mx_all = mx_arr  # shape (npos, nfreq)
            my_all = my_arr
            mz_all = mz_arr
            t_plot = np.array([time[0], time[-1]]) if len(time) > 1 else np.array([0, 1])

            self._safe_clear_plot(self.mxy_plot, [self.mxy_time_line])
            total_series = mx_all.shape[0] * mx_all.shape[1]
            self._reset_legend(self.mxy_plot, "mxy_legend", total_series > 1)
            idx = 0
            for pi in range(mx_all.shape[0]):
                for fi in range(mx_all.shape[1]):
                    color = self._color_for_index(idx, total_series)
                    self.mxy_plot.plot(t_plot, [mx_all[pi, fi], mx_all[pi, fi]], pen=pg.mkPen(color, width=4), name=f"p{pi} f{fi} Mx")
                    self.mxy_plot.plot(t_plot, [my_all[pi, fi], my_all[pi, fi]], pen=pg.mkPen(color, style=Qt.DashLine, width=2), name=f"p{pi} f{fi} My")
                    idx += 1
            mean_mx = float(np.mean(mx_all))
            mean_my = float(np.mean(my_all))
            self.mxy_plot.plot(t_plot, [mean_mx, mean_mx], pen=pg.mkPen('c', width=8), name="Mean Mx")
            self.mxy_plot.plot(t_plot, [mean_my, mean_my], pen=pg.mkPen('c', width=8, style=Qt.DashLine), name="Mean My")
            self._apply_pulse_region(self.mxy_plot, "mxy_region")
            mxy_ymin, mxy_ymax = self._calc_symmetric_limits(mx_all, my_all, base=self.initial_mz)
            self._set_plot_ranges(self.mxy_plot, x_min, x_max, mxy_ymin, mxy_ymax)

            self._safe_clear_plot(self.mz_plot, [self.mz_time_line])
            self._reset_legend(self.mz_plot, "mz_legend", total_series > 1)
            idx = 0
            for pi in range(mz_all.shape[0]):
                for fi in range(mz_all.shape[1]):
                    color = self._color_for_index(idx, total_series)
                    self.mz_plot.plot(t_plot, [mz_all[pi, fi], mz_all[pi, fi]], pen=pg.mkPen(color, width=2), name=f"p{pi} f{fi} Mz")
                    idx += 1
            mean_mz = float(np.mean(mz_all))
            self.mz_plot.plot(t_plot, [mean_mz, mean_mz], pen=pg.mkPen('c', width=8), name="Mean Mz")
            self._apply_pulse_region(self.mz_plot, "mz_region")
            mz_ymin, mz_ymax = self._calc_symmetric_limits(mz_all, base=self.initial_mz)
            self._set_plot_ranges(self.mz_plot, x_min, x_max, mz_ymin, mz_ymax)
            
            # Update 3D view
            # Use first position/frequency vector for static endpoint preview
            self.mag_3d.update_magnetization(mx_all.flatten()[0], my_all.flatten()[0], mz_all.flatten()[0])
            self.anim_timer.stop()
            self.anim_data = None
            self.anim_time = None
            self.mag_3d.set_preview_data(None, None, None, None)
            
            # Signal as single point
            signal_vals = result['signal']
            # Expect (npos, nfreq); reshape if needed
            if signal_vals.ndim == 2:
                sig = signal_vals
            elif signal_vals.ndim == 1:
                sig = signal_vals[:, None]
            else:
                sig = signal_vals.reshape(pos_len, freq_len)
            self._safe_clear_plot(self.signal_plot, [self.signal_time_line])
            total_series = sig.shape[0] * sig.shape[1]
            self._reset_legend(self.signal_plot, "signal_legend", total_series > 1)
            idx = 0
            for pi in range(sig.shape[0]):
                for fi in range(sig.shape[1]):
                    val = sig[pi, fi]
                    color = self._color_for_index(idx, total_series)
                    self.signal_plot.plot(t_plot, [np.abs(val), np.abs(val)], pen=pg.mkPen(color, width=2), name=f'|S| p{pi} f{fi}')
                    self.signal_plot.plot(t_plot, [np.real(val), np.real(val)], pen=pg.mkPen(color, style=Qt.DashLine, width=2), name=f'Re p{pi} f{fi}')
                    self.signal_plot.plot(t_plot, [np.imag(val), np.imag(val)], pen=pg.mkPen(color, style=Qt.DotLine, width=2), name=f'Im p{pi} f{fi}')
                    idx += 1
            mean_sig = np.mean(sig)
            self.signal_plot.plot(t_plot, [np.abs(mean_sig), np.abs(mean_sig)], pen=pg.mkPen('c', width=4), name='|S| mean')
            self._apply_pulse_region(self.signal_plot, "signal_region")
            self._set_plot_ranges(self.signal_plot, x_min, x_max, -1.1, 1.1)
            for line in (self.mxy_time_line, self.mz_time_line, self.signal_time_line):
                if line is not None:
                    line.show()
                    line.setValue(time[0])

            # Spatial plot for endpoint
            self.update_spatial_plot_from_last_result()

            # Hide universal time control in endpoint mode
            self.playback_indices = None
            self.playback_time = None
            self.playback_time_ms = None
            self.anim_vectors_full = None
            self.time_control.setVisible(False)
            # Disable mag filter since only endpoints are shown
            try:
                self._update_mag_selector_limits(pos_len, freq_len, disable=True)
            except Exception:
                pass

            self.spectrum_plot.clear()
            return
        else:
            # Time-resolved mode
            self.log_message("Time-resolved mode: plotting time-series data.")
            mx_all = mx_arr  # (ntime, npos, nfreq) expected
            my_all = my_arr
            mz_all = mz_arr
            ntime, npos, nfreq = mx_all.shape
            mean_only = self.mean_only_checkbox.isChecked()
            self.anim_time = time

            # Update magnetization plots
            self._update_mag_selector_limits(npos, nfreq, disable=mean_only)
            view_mode, selector = self._current_mag_filter(npos, nfreq)

            self._safe_clear_plot(self.mxy_plot, [self.mxy_time_line])
            if mean_only:
                self._reset_legend(self.mxy_plot, "mxy_legend", False)
                mean_mx = np.mean(mx_all, axis=(1, 2))
                mean_my = np.mean(my_all, axis=(1, 2))
                self.mxy_plot.plot(time, mean_mx, pen=pg.mkPen('c', width=4), name="Mean Mx")
                self.mxy_plot.plot(time, mean_my, pen=pg.mkPen('c', width=4, style=Qt.DashLine), name="Mean My")
            else:
                if view_mode == "Positions @ freq":
                    fi = min(selector, nfreq - 1)
                    total_series = npos
                    self._reset_legend(self.mxy_plot, "mxy_legend", total_series > 1)
                    for pi in range(npos):
                        color = self._color_for_index(pi, total_series)
                        self.mxy_plot.plot(time, mx_all[:, pi, fi], pen=pg.mkPen(color, width=2), name=f"p{pi} Mx @ f{fi}")
                        self.mxy_plot.plot(time, my_all[:, pi, fi], pen=pg.mkPen(color, style=Qt.DashLine, width=2), name=f"p{pi} My @ f{fi}")
                    mean_mx = np.mean(mx_all[:, :, fi], axis=1)
                    mean_my = np.mean(my_all[:, :, fi], axis=1)
                elif view_mode == "Freqs @ position":
                    pi = min(selector, npos - 1)
                    total_series = nfreq
                    self._reset_legend(self.mxy_plot, "mxy_legend", total_series > 1)
                    for fi in range(nfreq):
                        color = self._color_for_index(fi, total_series)
                        self.mxy_plot.plot(time, mx_all[:, pi, fi], pen=pg.mkPen(color, width=2), name=f"f{fi} Mx @ p{pi}")
                        self.mxy_plot.plot(time, my_all[:, pi, fi], pen=pg.mkPen(color, style=Qt.DashLine, width=2), name=f"f{fi} My @ p{pi}")
                    mean_mx = np.mean(mx_all[:, pi, :], axis=1)
                    mean_my = np.mean(my_all[:, pi, :], axis=1)
                else:
                    total_series = npos * nfreq
                    self._reset_legend(self.mxy_plot, "mxy_legend", total_series > 1)
                    idx = 0
                    for pi in range(npos):
                        for fi in range(nfreq):
                            color = self._color_for_index(idx, total_series)
                            self.mxy_plot.plot(time, mx_all[:, pi, fi], pen=pg.mkPen(color, width=2), name=f"p{pi} f{fi} Mx")
                            self.mxy_plot.plot(time, my_all[:, pi, fi], pen=pg.mkPen(color, style=Qt.DashLine, width=2), name=f"p{pi} f{fi} My")
                            idx += 1
                    mean_mx = np.mean(mx_all, axis=(1, 2))
                    mean_my = np.mean(my_all, axis=(1, 2))
                self.mxy_plot.plot(time, mean_mx, pen=pg.mkPen('c', width=4), name="Mean Mx")
                self.mxy_plot.plot(time, mean_my, pen=pg.mkPen('c', width=4, style=Qt.DashLine), name="Mean My")
            self._apply_pulse_region(self.mxy_plot, "mxy_region")
            # Use displayed series to set limits
            if mean_only:
                mxy_ymin, mxy_ymax = self._calc_symmetric_limits(mean_mx, mean_my, base=self.initial_mz)
            else:
                if view_mode == "Positions @ freq":
                    mxy_ymin, mxy_ymax = self._calc_symmetric_limits(mx_all[:, :, fi], my_all[:, :, fi], mean_mx, mean_my, base=self.initial_mz)
                elif view_mode == "Freqs @ position":
                    mxy_ymin, mxy_ymax = self._calc_symmetric_limits(mx_all[:, pi, :], my_all[:, pi, :], mean_mx, mean_my, base=self.initial_mz)
                else:
                    mxy_ymin, mxy_ymax = self._calc_symmetric_limits(mx_all, my_all, mean_mx, mean_my, base=self.initial_mz)
            self._set_plot_ranges(self.mxy_plot, x_min, x_max, mxy_ymin, mxy_ymax)
            if self.mxy_time_line is not None:
                self.mxy_time_line.show()
                self.mxy_time_line.setValue(time[0])

            self._safe_clear_plot(self.mz_plot, [self.mz_time_line])
            if mean_only:
                self._reset_legend(self.mz_plot, "mz_legend", False)
                mean_mz = np.mean(mz_all, axis=(1, 2))
                self.mz_plot.plot(time, mean_mz, pen=pg.mkPen('c', width=4), name="Mean Mz")
            else:
                if view_mode == "Positions @ freq":
                    self._reset_legend(self.mz_plot, "mz_legend", total_series > 1)
                    for pi in range(npos):
                        color = self._color_for_index(pi, total_series)
                        self.mz_plot.plot(time, mz_all[:, pi, fi], pen=pg.mkPen(color, width=2), name=f"p{pi} Mz @ f{fi}")
                    mean_mz = np.mean(mz_all[:, :, fi], axis=1)
                elif view_mode == "Freqs @ position":
                    self._reset_legend(self.mz_plot, "mz_legend", total_series > 1)
                    for fi in range(nfreq):
                        color = self._color_for_index(fi, total_series)
                        self.mz_plot.plot(time, mz_all[:, pi, fi], pen=pg.mkPen(color, width=2), name=f"f{fi} Mz @ p{pi}")
                    mean_mz = np.mean(mz_all[:, pi, :], axis=1)
                else:
                    self._reset_legend(self.mz_plot, "mz_legend", total_series > 1)
                    idx = 0
                    for pi in range(npos):
                        for fi in range(nfreq):
                            color = self._color_for_index(idx, total_series)
                            self.mz_plot.plot(time, mz_all[:, pi, fi], pen=pg.mkPen(color, width=2), name=f"p{pi} f{fi} Mz")
                            idx += 1
                    mean_mz = np.mean(mz_all, axis=(1, 2))
                self.mz_plot.plot(time, mean_mz, pen=pg.mkPen('c', width=4), name="Mean Mz")
            self._apply_pulse_region(self.mz_plot, "mz_region")
            if mean_only:
                mz_ymin, mz_ymax = self._calc_symmetric_limits(mean_mz, base=self.initial_mz)
            else:
                if view_mode == "Positions @ freq":
                    mz_ymin, mz_ymax = self._calc_symmetric_limits(mz_all[:, :, fi], mean_mz, base=self.initial_mz)
                elif view_mode == "Freqs @ position":
                    mz_ymin, mz_ymax = self._calc_symmetric_limits(mz_all[:, pi, :], mean_mz, base=self.initial_mz)
                else:
                    mz_ymin, mz_ymax = self._calc_symmetric_limits(mz_all, mean_mz, base=self.initial_mz)
            self._set_plot_ranges(self.mz_plot, x_min, x_max, mz_ymin, mz_ymax)
            if self.mz_time_line is not None:
                self.mz_time_line.show()
                self.mz_time_line.setValue(time[0])
            
            # Update signal plot
            signal_all = result['signal']  # (ntime, npos, nfreq)
            if signal_all.ndim == 3:
                signal = signal_arr  # (ntime, npos, nfreq)
            elif signal_all.ndim == 2:
                # Could be (ntime, nfreq) or (ntime, npos); align to (ntime, npos, nfreq)
                if signal_arr.shape[1] == freq_len:
                    signal = signal_arr[:, None, :]
                else:
                    signal = signal_arr[:, :, None]
            else:
                signal = signal_arr
            if signal.ndim == 1:
                signal = signal[:, None, None]
            self._safe_clear_plot(self.signal_plot, [self.signal_time_line])
            if mean_only:
                self._reset_legend(self.signal_plot, "signal_legend", False)
                mean_sig = np.mean(signal, axis=(1, 2))
                self.signal_plot.plot(time, np.abs(mean_sig), pen=pg.mkPen('c', width=4), name='|S| mean')
            else:
                total_series_sig = signal.shape[1] * signal.shape[2]
                self._reset_legend(self.signal_plot, "signal_legend", total_series_sig > 1)
                idx = 0
                for pi in range(signal.shape[1]):
                    for fi in range(signal.shape[2]):
                        color = self._color_for_index(idx, total_series_sig)
                        self.signal_plot.plot(time, np.abs(signal[:, pi, fi]), pen=color, name=f'|S| p{pi} f{fi}')
                        self.signal_plot.plot(time, np.real(signal[:, pi, fi]), pen=pg.mkPen(color, style=Qt.DashLine), name=f'Re p{pi} f{fi}')
                        self.signal_plot.plot(time, np.imag(signal[:, pi, fi]), pen=pg.mkPen(color, style=Qt.DotLine), name=f'Im p{pi} f{fi}')
                        idx += 1
                mean_sig = np.mean(signal, axis=(1, 2))
                self.signal_plot.plot(time, np.abs(mean_sig), pen=pg.mkPen('c', width=4), name='|S| mean')
            self._apply_pulse_region(self.signal_plot, "signal_region")
            sig_ymin, sig_ymax = self._calc_symmetric_limits(
                np.abs(signal), np.real(signal), np.imag(signal), np.abs(mean_sig),
                base=self.initial_mz
            )
            self._set_plot_ranges(self.signal_plot, x_min, x_max, sig_ymin, sig_ymax)
            if self.signal_time_line is not None:
                self.signal_time_line.show()
                self.signal_time_line.setValue(time[0])
            
        # Update spectrum
        from scipy.fft import fft, fftfreq
        self.spectrum_plot.clear()
        # Spectrum selection (mean vs selected position)
        pos_count = signal.shape[1] if signal.ndim >= 2 else 1
        self.spectrum_pos_slider.setMaximum(max(0, pos_count - 1))
        pos_sel = min(self.spectrum_pos_slider.value(), pos_count - 1)
        self.spectrum_pos_label.setText(f"Pos idx: {pos_sel}")

        spectrum_mode = self.spectrum_mode.currentText()
        if spectrum_mode == "Mean only":
            sig_for_fft = np.mean(signal, axis=tuple(range(1, signal.ndim))) if signal.ndim > 1 else signal
        else:
            # Mean + individuals or Individual: use selected position averaged over freq
            if signal.ndim == 1:
                sig_for_fft = signal
            elif signal.ndim == 2:
                sig_for_fft = signal[:, pos_sel]
            else:  # (ntime, npos, nfreq)
                sig_for_fft = np.mean(signal[:, pos_sel, :], axis=1)
        n = len(sig_for_fft)
        dt = time[1] - time[0] if len(time) > 1 else 1
        spectrum_mode = self.spectrum_mode.currentText()
        self.spectrum_plot.clear()
        spectrum = fft(sig_for_fft)
        freq = fftfreq(n, dt/1000)  # Hz
        spectrum_shift = np.fft.fftshift(spectrum)
        freq_shift = np.fft.fftshift(freq)

        # Compute global spectrum magnitude range for fixed y-limits
        spec_mag_max = float(np.nanmax(np.abs(spectrum_shift)))
        if spectrum_mode == "Mean + individuals":
            sig_mean = np.mean(signal, axis=tuple(range(1, signal.ndim))) if signal.ndim > 1 else sig_for_fft
            spec_mean = np.fft.fftshift(fft(sig_mean))
            spec_mag_max = max(spec_mag_max, float(np.nanmax(np.abs(spec_mean))))
            self.spectrum_plot.plot(freq_shift, np.abs(spec_mean), pen=pg.mkPen('c', width=3), name="Mean")

        # Store global spectrum range (min is always 0 for magnitude)
        self.spectrum_mag_range = (0.0, spec_mag_max * 1.1)

        self.spectrum_plot.plot(freq_shift, np.abs(spectrum_shift), pen='w', name="Selected")
        span = self.spectrum_range.value()
        if span > 0:
            half = span / 2.0
            self.spectrum_plot.setXRange(-half, half, padding=0)
        else:
            self.spectrum_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)

        # Set fixed Y range with 1.1 padding
        self.spectrum_plot.setYRange(self.spectrum_mag_range[0], self.spectrum_mag_range[1], padding=0)

        # Spatial excitation plot (final Mz across positions, per frequency)
        self.update_spatial_plot_from_last_result()

        # Cache full vector timeline (ntime, npos, nfreq, 3) for 3D view
        self.anim_vectors_full = np.stack([mx_all, my_all, mz_all], axis=3)
        total_frames = self.anim_vectors_full.shape[0]
        self.playback_indices = self._build_playback_indices(total_frames)
        self.playback_time = self.last_time[self.playback_indices]
        self.playback_time_ms = self.playback_time * 1000.0
        self.anim_index = 0

        # Prepare B1 for playback (use same downsampling)
        b1_full = self.last_b1 if hasattr(self, 'last_b1') else None
        if b1_full is not None and len(b1_full) >= total_frames:
            self.anim_b1 = np.asarray(b1_full)[self.playback_indices]
            max_b1 = float(np.nanmax(np.abs(b1_full))) if len(b1_full) else 0.0
            self.anim_b1_scale = 1.0 / max(max_b1, 1e-6)
        else:
            self.anim_b1 = None
            self.anim_b1_scale = 1.0

        # Configure 3D selector and rebuild the view
        self.mag_3d.set_selector_limits(npos, nfreq, disable=mean_only)
        self._refresh_vector_view(mean_only=mean_only)
        self.mag_3d.set_cursor_index(0)

        # Initialize universal time control with the time array
        self.time_control.set_time_range(self.playback_time)  # Use playback_time in seconds
        self.time_control.setVisible(True)

        # Ensure signal tab x-range set even if spectrum-only
        self.signal_plot.setXRange(x_min, x_max, padding=0)
        
    def load_parameters(self):
        """Load simulation parameters from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Parameters", "", "JSON Files (*.json)"
        )
        if filename:
            with open(filename, 'r') as f:
                params = json.load(f)
                # Apply parameters to widgets
                self.tissue_widget.t1_spin.setValue(params.get('t1', 1000))
                self.tissue_widget.t2_spin.setValue(params.get('t2', 20))

    def _start_vector_animation(self):
        """Start or restart the 3D vector animation if data exists."""
        if self.anim_data is None or len(self.anim_data) == 0:
            self.anim_timer.stop()
            return
        if self.mag_3d.track_path:
            self.mag_3d._clear_path()
        if self.anim_index >= len(self.anim_data):
            self.anim_index = 0
        # Always recompute interval using current speed control
        self._recompute_anim_interval(self.time_control.speed_spin.value() if hasattr(self, 'time_control') else None)
        self.anim_timer.start(self.anim_interval_ms)

    def _resume_vector_animation(self):
        """Resume playback of the 3D vector."""
        self._start_vector_animation()

    def _pause_vector_animation(self):
        """Pause the 3D vector animation."""
        self.anim_timer.stop()

    def _reset_vector_animation(self):
        """Reset the 3D vector to the first available frame."""
        self.anim_timer.stop()
        self.anim_index = 0
        self.mag_3d._clear_path()
        if self.anim_data is not None and len(self.anim_data) > 0:
            vectors = self.anim_data[0]
            colors = [self._color_tuple(self._color_for_index(i, vectors.shape[0])) for i in range(vectors.shape[0])]
            self.mag_3d.update_magnetization(vectors, colors=colors)
            self.mag_3d.set_cursor_index(0)
        if self.playback_time is not None:
            self.time_control.set_time_index(0)
            self._on_universal_time_changed(0)
        if self.mag_3d.b1_arrow is not None:
            self.mag_3d.b1_arrow.setVisible(False)

    def _recompute_anim_interval(self, sim_ms_per_s: float = None):
        """Compute animation interval so that wall time matches simulation time scaling.

        The speed control sets how many milliseconds of simulation should play per second of wall time.
        For example, sim_ms_per_s=50 means 50 ms of simulation plays in 1 second of real time.
        """
        if sim_ms_per_s is None:
            sim_ms_per_s = self.time_control.speed_spin.value()
        if sim_ms_per_s <= 0:
            sim_ms_per_s = 50.0  # fallback to reasonable speed
        total_frames = len(self.playback_time) if self.playback_time is not None else 0
        if total_frames < 2:
            self.anim_interval_ms = 30
            self._frame_step = 1
            return

        # Calculate time per frame in the simulation data
        duration_ms = max(float(self.playback_time[-1] - self.playback_time[0]) * 1000.0, 1e-6)
        time_per_frame_ms = duration_ms / (total_frames - 1)

        # Desired wall clock time per frame (ms) to achieve target playback speed
        # If we want sim_ms_per_s milliseconds of sim to play in 1000 ms of wall time:
        # wall_time_per_frame = time_per_frame_ms / (sim_ms_per_s / 1000)
        desired_interval_ms = time_per_frame_ms / sim_ms_per_s * 1000.0

        min_interval = getattr(self, "_min_anim_interval_ms", 2.0)
        max_step = 10  # avoid skipping too many frames even at high speed

        if desired_interval_ms < min_interval:
            # Need to skip frames to achieve desired speed
            # If we fire timer at min_interval and want to maintain speed:
            # frame_step = (min_interval / desired_interval_ms)
            frame_step = max(1, int(round(min_interval / max(desired_interval_ms, 1e-6))))
            frame_step = min(frame_step, max_step, total_frames)
            interval_ms = min_interval
        else:
            interval_ms = desired_interval_ms
            frame_step = 1

        self.anim_interval_ms = max(1, int(round(interval_ms)))
        self._frame_step = frame_step

    def _update_playback_speed(self, sim_ms_per_s: float):
        """Adjust playback speed (simulation ms per real second)."""
        self._recompute_anim_interval(sim_ms_per_s)
        if self.anim_timer.isActive():
            self.anim_timer.start(self.anim_interval_ms)

    def _set_animation_index_from_slider(self, idx: int):
        """Scrub animation position from the preview slider."""
        if self.anim_data is None or len(self.anim_data) == 0:
            return
        idx = int(max(0, min(idx, len(self.anim_data) - 1)))

        # Clear path if jumping back to the beginning (e.g., scrubbing backward)
        # This prevents a line being drawn from the current position to index 0
        if idx == 0 and self.mag_3d.track_path and len(self.mag_3d.path_points) > 0:
            self.mag_3d._clear_path()

        self.anim_index = idx
        vectors = self.anim_data[idx]
        self.mag_3d.update_magnetization(vectors, colors=self.anim_colors)
        self.mag_3d.set_cursor_index(idx)
        self._update_b1_arrow(idx)

    def _animate_vector(self):
        """Advance the 3D vector animation if data is available."""
        if self.anim_data is None:
            return
        if self.anim_index >= len(self.anim_data):
            self.anim_index = 0
            # Clear the tip path when looping to avoid drawing a line from end to start
            if self.mag_3d.track_path:
                self.mag_3d._clear_path()
        step = max(1, getattr(self, "_frame_step", 1))
        # Move universal time control (label/slider) then propagate to all views
        self.time_control.set_time_index(self.anim_index)
        self._on_universal_time_changed(self.anim_index)
        self.anim_index += step

    def _update_b1_arrow(self, playback_idx: int):
        """Update the B1 direction/length indicator in the 3D view."""
        if self.anim_b1 is None or self.mag_3d is None:
            return
        idx = int(max(0, min(playback_idx, len(self.anim_b1) - 1)))
        b1_val = self.anim_b1[idx]
        mag = abs(b1_val)
        if not np.isfinite(mag) or mag < 1e-9:
            self.mag_3d.b1_arrow.setVisible(False)
            return
        phase = np.angle(b1_val)
        tip = np.array([
            self.anim_b1_scale * mag * np.cos(phase),
            self.anim_b1_scale * mag * np.sin(phase),
            0.0
        ])
        pos = np.array([[0.0, 0.0, 0.0], tip])
        self.mag_3d.b1_arrow.setData(pos=pos)
        self.mag_3d.b1_arrow.setVisible(True)
                
    def save_parameters(self):
        """Save simulation parameters to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "", "JSON Files (*.json)"
        )
        if filename:
            params = {
                't1': self.tissue_widget.t1_spin.value(),
                't2': self.tissue_widget.t2_spin.value(),
                'te': self.sequence_designer.te_spin.value(),
                'tr': self.sequence_designer.tr_spin.value(),
            }
            with open(filename, 'w') as f:
                json.dump(params, f, indent=2)
                
    def export_results(self):
        """Export simulation results with complete parameters."""
        if self.simulator.last_result is None:
            QMessageBox.warning(self, "No Results",
                              "No simulation results to export")
            return

        # Ask user which format to export
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Export Options")
        layout = QVBoxLayout()

        # Export format selection
        format_group = QGroupBox("Export Format")
        format_layout = QVBoxLayout()
        format_buttons = QButtonGroup(dialog)

        hdf5_radio = QRadioButton("HDF5 (.h5) - Full data + parameters")
        json_radio = QRadioButton("JSON (.json) - Parameters only")
        both_radio = QRadioButton("Both HDF5 + JSON")
        notebook_load_radio = QRadioButton("Jupyter Notebook (.ipynb) - Load data from HDF5")
        notebook_resim_radio = QRadioButton("Jupyter Notebook (.ipynb) - Re-run simulation")
        notebook_both_radio = QRadioButton("HDF5 + Notebook (load data)")

        hdf5_radio.setChecked(True)
        format_buttons.addButton(hdf5_radio, 0)
        format_buttons.addButton(json_radio, 1)
        format_buttons.addButton(both_radio, 2)
        format_buttons.addButton(notebook_load_radio, 3)
        format_buttons.addButton(notebook_resim_radio, 4)
        format_buttons.addButton(notebook_both_radio, 5)

        format_layout.addWidget(hdf5_radio)
        format_layout.addWidget(json_radio)
        format_layout.addWidget(both_radio)
        format_layout.addWidget(QLabel(""))  # Spacer
        format_layout.addWidget(notebook_load_radio)
        format_layout.addWidget(notebook_resim_radio)
        format_layout.addWidget(notebook_both_radio)
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec_() != QDialog.Accepted:
            return

        export_choice = format_buttons.checkedId()

        # Collect all parameters
        sequence_params = self._collect_sequence_parameters()
        simulation_params = self._collect_simulation_parameters()

        # Export based on choice
        if export_choice == 0 or export_choice == 2:  # HDF5 or Both
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Results (HDF5)", "", "HDF5 Files (*.h5)"
            )
            if filename:
                try:
                    self.simulator.save_results(filename, sequence_params, simulation_params)
                    self.statusBar().showMessage(f"Results exported to {filename}")
                except Exception as e:
                    QMessageBox.critical(self, "Export Error", f"Failed to export HDF5:\n{str(e)}")
                    return

        if export_choice == 1 or export_choice == 2:  # JSON or Both
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Parameters (JSON)", "", "JSON Files (*.json)"
            )
            if filename:
                try:
                    self.simulator.save_parameters_json(filename, sequence_params, simulation_params)
                    self.statusBar().showMessage(f"Parameters exported to {filename}")
                except Exception as e:
                    QMessageBox.critical(self, "Export Error", f"Failed to export JSON:\n{str(e)}")

        # Handle notebook exports (options 3, 4, 5)
        if export_choice in [3, 4, 5]:
            self._export_notebook(export_choice, sequence_params, simulation_params)

    def _export_notebook(self, export_choice, sequence_params, simulation_params):
        """
        Export Jupyter notebook.

        Parameters
        ----------
        export_choice : int
            3 = Notebook (load data), 4 = Notebook (resimulate), 5 = HDF5 + Notebook
        sequence_params : dict
            Sequence parameters
        simulation_params : dict
            Simulation parameters
        """
        try:
            from notebook_exporter import export_notebook
        except ImportError:
            QMessageBox.critical(
                self,
                "Import Error",
                "Notebook export requires 'nbformat' package.\n\n"
                "Install with: pip install nbformat"
            )
            return

        # Get tissue parameters
        tissue_params = {
            'name': self.tissue_widget.preset_combo.currentText(),
            't1': self.tissue_widget.t1_spin.value() / 1000,  # ms to s
            't2': self.tissue_widget.t2_spin.value() / 1000,
            't2_star': self.tissue_widget.t2s_spin.value() / 1000,
            'density': self.tissue_widget.pd_spin.value()
        }

        # RF waveform (if available)
        rf_waveform = None
        if hasattr(self, 'last_b1') and self.last_b1 is not None:
            rf_waveform = (self.last_b1, self.last_time)

        # Handle Mode A (load data) or Mode B (resimulate)
        if export_choice == 3:  # Notebook (load data)
            # Need HDF5 file first
            h5_filename, _ = QFileDialog.getSaveFileName(
                self, "Save HDF5 Data File", "", "HDF5 Files (*.h5)"
            )
            if not h5_filename:
                return

            # Save HDF5 file
            try:
                self.simulator.save_results(h5_filename, sequence_params, simulation_params)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save HDF5:\n{str(e)}")
                return

            # Now save notebook
            nb_filename, _ = QFileDialog.getSaveFileName(
                self, "Save Jupyter Notebook", "", "Jupyter Notebooks (*.ipynb)"
            )
            if not nb_filename:
                return

            try:
                export_notebook(
                    mode='load_data',
                    filename=nb_filename,
                    sequence_params=sequence_params,
                    simulation_params=simulation_params,
                    tissue_params=tissue_params,
                    h5_filename=h5_filename
                )
                self.statusBar().showMessage(f"Notebook exported: {nb_filename}")
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Exported:\n  Data: {h5_filename}\n  Notebook: {nb_filename}\n\n"
                    f"Open the notebook in Jupyter to analyze the data."
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export notebook:\n{str(e)}")

        elif export_choice == 4:  # Notebook (resimulate)
            nb_filename, _ = QFileDialog.getSaveFileName(
                self, "Save Jupyter Notebook", "", "Jupyter Notebooks (*.ipynb)"
            )
            if not nb_filename:
                return

            try:
                export_notebook(
                    mode='resimulate',
                    filename=nb_filename,
                    sequence_params=sequence_params,
                    simulation_params=simulation_params,
                    tissue_params=tissue_params,
                    rf_waveform=rf_waveform
                )
                self.statusBar().showMessage(f"Notebook exported: {nb_filename}")
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Notebook exported: {nb_filename}\n\n"
                    f"Open in Jupyter to re-run the simulation."
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export notebook:\n{str(e)}")

        elif export_choice == 5:  # HDF5 + Notebook (load)
            # Save HDF5 first
            h5_filename, _ = QFileDialog.getSaveFileName(
                self, "Save HDF5 Data File", "", "HDF5 Files (*.h5)"
            )
            if not h5_filename:
                return

            try:
                self.simulator.save_results(h5_filename, sequence_params, simulation_params)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save HDF5:\n{str(e)}")
                return

            # Auto-generate notebook filename
            from pathlib import Path
            nb_filename = str(Path(h5_filename).with_suffix('.ipynb'))

            try:
                export_notebook(
                    mode='load_data',
                    filename=nb_filename,
                    sequence_params=sequence_params,
                    simulation_params=simulation_params,
                    tissue_params=tissue_params,
                    h5_filename=h5_filename
                )
                self.statusBar().showMessage(f"Exported HDF5 and notebook")
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Exported:\n  Data: {h5_filename}\n  Notebook: {nb_filename}\n\n"
                    f"Both files saved successfully!"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export notebook:\n{str(e)}")

    def _collect_sequence_parameters(self):
        """Collect all pulse sequence parameters from GUI."""
        seq_type = self.sequence_designer.sequence_type.currentText()

        params = {
            'sequence_type': seq_type,
            'te': self.sequence_designer.te_spin.value() / 1000,  # ms to s
            'tr': self.sequence_designer.tr_spin.value() / 1000,  # ms to s
        }

        # Add sequence-specific parameters
        if 'Echo' in seq_type or 'Gradient' in seq_type:
            if hasattr(self.sequence_designer, 'flip_spin'):
                params['flip_angle'] = self.sequence_designer.flip_spin.value()

        if hasattr(self.sequence_designer, 'echo_count_spin'):
            params['echo_count'] = self.sequence_designer.echo_count_spin.value()

        # RF pulse parameters
        params['rf_pulse_type'] = self.rf_designer.pulse_type.currentText()
        params['rf_flip_angle'] = self.rf_designer.flip_angle.value()
        params['rf_duration'] = self.rf_designer.duration.value() / 1000  # ms to s
        params['rf_time_bw_product'] = self.rf_designer.tbw.value()
        params['rf_phase'] = self.rf_designer.phase.value()

        # Store RF waveform if available
        if hasattr(self, 'last_b1') and self.last_b1 is not None:
            params['b1_waveform'] = self.last_b1
            params['time_waveform'] = self.last_time

        return params

    def _collect_simulation_parameters(self):
        """Collect all simulation parameters from GUI."""
        params = {
            'mode': 'time-resolved' if self.mode_combo.currentText() == "Time-resolved" else 'endpoint',
            'time_step_us': self.time_step_spin.value(),
            'num_positions': self.pos_spin.value(),
            'position_range_cm': self.pos_range.value(),
            'num_frequencies': self.freq_spin.value(),
            'frequency_range_hz': self.freq_range.value(),
            'extra_tail_ms': self.extra_tail_spin.value(),
            'use_parallel': self.simulator.use_parallel,
            'num_threads': self.simulator.num_threads,
            'preview_mode': self.preview_checkbox.isChecked() if hasattr(self, 'preview_checkbox') else False
        }

        # Store initial magnetization
        params['initial_mz'] = self.tissue_widget.get_initial_mz()

        return params

    # ========== Export Methods ==========

    def _get_export_directory(self):
        """Get or create the default export directory."""
        export_dir = Path.cwd() / "exports"
        export_dir.mkdir(exist_ok=True)
        return export_dir

    def _show_not_implemented(self, feature_name):
        """Show a message for features not yet implemented."""
        QMessageBox.information(
            self,
            "Coming Soon",
            f"{feature_name} export will be available in a future update.\n\n"
            "Current available exports:\n"
            "- Static images (PNG, SVG)"
        )

    def _prompt_data_export_path(self, default_name: str):
        """Open a save dialog and return the chosen path and format."""
        export_dir = self._get_export_directory()
        default_path = export_dir / f"{default_name}.csv"
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            str(default_path),
            "CSV (*.csv);;NumPy (*.npy);;DAT/TSV (*.dat *.tsv)"
        )
        if not filename:
            return None, None
        fmt = 'csv'
        sel = (selected_filter or '').lower()
        path = Path(filename)
        suffix = path.suffix.lower()
        if 'npy' in sel or suffix == '.npy':
            fmt = 'npy'
            path = path.with_suffix('.npy')
        elif 'dat' in sel or 'tsv' in sel or suffix in ('.dat', '.tsv'):
            fmt = 'dat'
            path = path.with_suffix('.dat')
        else:
            fmt = 'csv'
            path = path.with_suffix('.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path, fmt

    def _current_playback_index(self) -> int:
        """Return the current full-resolution time index based on the universal slider."""
        if hasattr(self, 'time_control') and self.time_control.time_array is not None:
            playback_idx = int(self.time_control.time_slider.value())
            return int(self._playback_to_full_index(playback_idx))
        if self.last_time is not None:
            return len(self.last_time) - 1
        return 0

    def _calculate_spectrum_data(self, time_idx=None):
        """Compute spectrum arrays used for plotting or export."""
        if self.last_result is None:
            return None
        signal = self.last_result.get('signal')
        if signal is None:
            return None
        time = self.last_time if self.last_time is not None else self.last_result.get('time', None)
        if time is None or len(time) < 2:
            return None
        time = np.asarray(time)
        sig_arr = np.asarray(signal)
        if sig_arr.ndim == 1:
            sig_arr = sig_arr[:, None, None]
        elif sig_arr.ndim == 2:
            sig_arr = sig_arr[:, :, None]
        if time_idx is None:
            time_idx = len(time) - 1
        time_idx = int(max(1, min(time_idx, len(time) - 1)))
        sig_slice = sig_arr[:time_idx + 1]
        time_slice = time[:time_idx + 1]
        n = len(time_slice)
        dt = (time_slice[1] - time_slice[0]) * 1000.0  # seconds -> milliseconds
        spectrum_mode = self.spectrum_mode.currentText() if hasattr(self, 'spectrum_mode') else "Mean only"
        pos_count = sig_slice.shape[1]
        pos_sel = min(self.spectrum_pos_slider.value(), pos_count - 1) if pos_count > 0 else 0

        if spectrum_mode == "Mean only":
            sig_for_fft = np.mean(sig_slice, axis=tuple(range(1, sig_slice.ndim)))
            spec_mean = None
        elif spectrum_mode == "Mean + individuals":
            sig_for_fft = np.mean(sig_slice, axis=tuple(range(1, sig_slice.ndim)))
            spec_mean = np.fft.fftshift(np.fft.fft(sig_for_fft))
        else:
            sig_for_fft = np.mean(sig_slice[:, pos_sel, :], axis=1)
            spec_mean = None

        spectrum = np.fft.fftshift(np.fft.fft(sig_for_fft))
        freq = np.fft.fftshift(np.fft.fftfreq(n, dt / 1000.0))
        return {
            "freq": freq,
            "spectrum": spectrum,
            "spec_mean": spec_mean,
            "mode": spectrum_mode,
            "pos_count": pos_count,
            "pos_sel": pos_sel,
            "time_idx": time_idx,
        }

    def _refresh_spectrum(self, time_idx=None):
        """Update spectrum plot using data up to the specified time index."""
        spec_data = self._calculate_spectrum_data(time_idx)
        if spec_data is None:
            return

        freq = spec_data["freq"]
        spectrum = spec_data["spectrum"]
        spec_mean = spec_data["spec_mean"]
        spectrum_mode = spec_data["mode"]
        pos_count = spec_data["pos_count"]
        pos_sel = spec_data["pos_sel"]

        self.spectrum_pos_slider.setMaximum(max(0, pos_count - 1))
        self.spectrum_pos_label.setText(f"Pos idx: {pos_sel}")

        self.spectrum_plot.clear()
        if spectrum_mode == "Mean + individuals":
            if spec_mean is not None:
                self.spectrum_plot.plot(freq, np.abs(spec_mean), pen=pg.mkPen('c', width=3), name="Mean")
        self.spectrum_plot.plot(freq, np.abs(spectrum), pen='w', name="Selected")

        # Add colored vertical markers if enabled
        if self.spectrum_markers_checkbox.isChecked() and len(freq) > 1:
            # Draw vertical lines at each frequency with colors matching 3D view
            mag = np.abs(spectrum)
            for i, (f, m) in enumerate(zip(freq, mag)):
                color = self._color_for_index(i, len(freq))
                # Vertical line from 0 to magnitude
                line = pg.PlotCurveItem([f, f], [0, m], pen=pg.mkPen(color, width=2))
                self.spectrum_plot.addItem(line)

        span = self.spectrum_range.value()
        if span > 0:
            half = span / 2.0
            self.spectrum_plot.setXRange(-half, half, padding=0)
        else:
            self.spectrum_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)

        # Use fixed Y range if available (computed in update_plots)
        if hasattr(self, 'spectrum_mag_range') and self.spectrum_mag_range is not None:
            self.spectrum_plot.setYRange(self.spectrum_mag_range[0], self.spectrum_mag_range[1], padding=0)
        else:
            self.spectrum_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        # Cache for export
        self._last_spectrum_export = {
            "frequency": freq,
            "selected_magnitude": np.abs(spectrum),
            "selected_phase_rad": np.angle(spectrum),
            "mode": spectrum_mode,
            "time_idx": spec_data["time_idx"],
        }
        if spec_mean is not None:
            self._last_spectrum_export["mean_magnitude"] = np.abs(spec_mean)
            self._last_spectrum_export["mean_phase_rad"] = np.angle(spec_mean)

    def _grab_widget_array(self, widget: QWidget, target_height: int = None) -> np.ndarray:
        """Grab a Qt widget as an RGB numpy array, optionally scaling height."""
        pixmap = widget.grab()
        image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        if target_height and target_height > 0:
            image = image.scaledToHeight(target_height, Qt.SmoothTransformation)
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((image.height(), image.width(), 4))[:, :, :3]
        return arr

    def _ensure_even_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame to even width/height for video encoders."""
        h, w = frame.shape[:2]
        if h % 2 != 0:
            frame = frame[:-1, :, :]
        if w % 2 != 0:
            frame = frame[:, :-1, :]
        return frame

    def _export_widget_animation(self, widgets: list, default_filename: str, before_grab=None):
        """Generic widget-grab animation exporter (GIF/MP4) for plot tabs."""
        if self.last_time is None:
            QMessageBox.warning(self, "No Data", "Please run a time-resolved simulation first.")
            return
        total_frames = len(self.playback_time) if self.playback_time is not None else len(self.last_time)
        if total_frames < 2:
            QMessageBox.warning(self, "No Time Series", "Need at least two time points to export animation.")
            return

        dialog = ExportAnimationDialog(
            self,
            total_frames=total_frames,
            default_filename=default_filename,
            default_directory=self._get_export_directory()
        )
        dialog.mean_only_checkbox.setVisible(False)

        if dialog.exec_() != QDialog.Accepted:
            return
        params = dialog.get_export_params()
        exporter = AnimationExporter()
        indices = exporter._compute_indices(
            total_frames,
            max_frames=params['max_frames'],
            start_idx=params['start_idx'],
            end_idx=params['end_idx']
        )

        fmt = params['format']
        filepath = Path(params['filename'])
        if filepath.suffix.lower() != f".{fmt}":
            filepath = filepath.with_suffix(f".{fmt}")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if vz_imageio is None:
            QMessageBox.warning(self, "Missing Dependency", "Animation export requires the 'imageio' package.")
            return

        if fmt == 'gif':
            writer = vz_imageio.get_writer(str(filepath), mode='I', fps=params['fps'], format='GIF')
        else:
            writer = vz_imageio.get_writer(
                str(filepath),
                fps=params['fps'],
                format='FFMPEG',
                codec='libx264',
                bitrate=exporter.default_bitrate,
                quality=8,
                macro_block_size=None
            )

        progress = QProgressDialog("Exporting animation...", "Cancel", 0, len(indices), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        was_running = self.anim_timer.isActive()
        current_idx = self.anim_index
        self.anim_timer.stop()

        try:
            for i, idx in enumerate(indices):
                if progress.wasCanceled():
                    raise RuntimeError("Animation export cancelled")
                # Sync all plots/time lines
                self._set_animation_index_from_slider(int(idx))
                if before_grab:
                    try:
                        before_grab(int(idx))
                    except Exception:
                        pass
                QApplication.processEvents()
                frames = []
                target_h = params['height'] if params['height'] else None
                for w in widgets:
                    frames.append(self._grab_widget_array(w, target_height=target_h))
                # Normalize heights to smallest to stack horizontally
                min_h = min(f.shape[0] for f in frames)
                frames = [f if f.shape[0] == min_h else f[:min_h, :, :] for f in frames]
                combined = np.hstack(frames)
                target_w = params['width'] if params['width'] else combined.shape[1]
                target_h_final = params['height'] if params['height'] else combined.shape[0]
                if target_w != combined.shape[1] or target_h_final != combined.shape[0]:
                    qimg = QImage(combined.data, combined.shape[1], combined.shape[0], combined.strides[0], QImage.Format_RGB888)
                    # If both width/height are provided, honor exact resolution; otherwise keep aspect
                    aspect_mode = Qt.IgnoreAspectRatio if (params['width'] and params['height']) else Qt.KeepAspectRatio
                    qimg = qimg.copy().scaled(target_w, target_h_final, aspect_mode, Qt.SmoothTransformation)
                    ptr = qimg.bits()
                    ptr.setsize(qimg.byteCount())
                    combined = np.frombuffer(ptr, dtype=np.uint8).reshape((qimg.height(), qimg.width(), 3))
                combined = self._ensure_even_frame(combined)
                writer.append_data(combined)
                progress.setValue(i + 1)
                QApplication.processEvents()
        finally:
            writer.close()
            self._set_animation_index_from_slider(current_idx)
            if was_running:
                self._resume_vector_animation()
            progress.setValue(progress.maximum())

        QMessageBox.information(self, "Export Successful", f"Animation exported successfully:\n{filepath.name}")
        self.log_message(f"Animation exported to {filepath}")

    def _export_magnetization_image(self, default_format='png'):
        """Export magnetization plots as an image."""
        if not self.last_result:
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return

        # Create export dialog with default directory
        export_dir = self._get_export_directory()
        dialog = ExportImageDialog(self, default_filename="magnetization", default_directory=export_dir)
        dialog.format_combo.setCurrentIndex(['png', 'svg', 'pdf'].index(default_format))

        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_export_params()
            exporter = ImageExporter()

            try:
                # Export both plots
                # For now, export them separately (multi-plot export is future work)
                base_path = Path(params['filename'])

                # Export Mxy plot
                mxy_path = base_path.parent / f"{base_path.stem}_mxy{base_path.suffix}"
                result_mxy = exporter.export_pyqtgraph_plot(
                    self.mxy_plot,
                    str(mxy_path),
                    format=params['format'],
                    width=params['width']
                )

                # Export Mz plot
                mz_path = base_path.parent / f"{base_path.stem}_mz{base_path.suffix}"
                result_mz = exporter.export_pyqtgraph_plot(
                    self.mz_plot,
                    str(mz_path),
                    format=params['format'],
                    width=params['width']
                )

                if result_mxy and result_mz:
                    self.log_message(f"Exported magnetization plots to:\n  {result_mxy}\n  {result_mz}")
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Magnetization plots exported successfully:\n\n"
                        f"Mxy: {Path(result_mxy).name}\n"
                        f"Mz: {Path(result_mz).name}"
                    )
                else:
                    self.log_message("Export failed")
                    QMessageBox.warning(self, "Export Failed", "Could not export plots. Check the log for details.")

            except Exception as e:
                self.log_message(f"Export error: {e}")
                QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")

    def _export_magnetization_animation(self):
        """Export magnetization time-series as GIF/MP4."""
        if not self.last_result or 'mx' not in self.last_result or self.last_result['mx'] is None:
            QMessageBox.warning(self, "No Data", "Please run a time-resolved simulation first.")
            return

        mx = self.last_result['mx']
        my = self.last_result['my']
        mz = self.last_result['mz']
        if mx is None or mx.ndim != 3:
            QMessageBox.warning(self, "No Time Series", "Animation export requires time-resolved data.")
            return

        time_s = np.asarray(self.last_time) if self.last_time is not None else np.asarray(self.last_result.get('time', []))
        if time_s is None or len(time_s) == 0 or len(time_s) != mx.shape[0]:
            QMessageBox.warning(self, "Missing Time", "Could not determine time axis for animation export.")
            return

        total_frames = len(time_s)
        dialog = ExportAnimationDialog(
            self,
            total_frames=total_frames,
            default_filename="magnetization",
            default_directory=self._get_export_directory()
        )
        dialog.mean_only_checkbox.setChecked(self.mean_only_checkbox.isChecked())

        if dialog.exec_() != QDialog.Accepted:
            return

        params = dialog.get_export_params()
        start_idx = min(params['start_idx'], total_frames - 1)
        end_idx = max(start_idx, min(params['end_idx'], total_frames - 1))
        mean_only = params['mean_only']

        def _select_component(arr):
            if arr.ndim == 3:
                if mean_only:
                    return np.mean(arr, axis=(1, 2))
                return arr[:, 0, 0]
            elif arr.ndim == 2:
                return arr[:, 0]
            return np.asarray(arr)

        mx_trace = _select_component(mx)
        my_trace = _select_component(my)
        mz_trace = _select_component(mz)

        # Preserve current sequence playhead to restore later
        playhead = getattr(self.sequence_designer, 'playhead_line', None)
        playhead_visible = playhead.isVisible() if playhead is not None else False
        playhead_value = playhead.value() if playhead is not None else 0

        def frame_hook(frame: np.ndarray, actual_idx: int) -> np.ndarray:
            """Append sequence diagram snapshot beside the plot frame."""
            try:
                time_ms = None
                if self.last_time is not None and 0 <= actual_idx < len(self.last_time):
                    time_ms = float(self.last_time[actual_idx] * 1000.0)
                if playhead is not None and time_ms is not None:
                    playhead.show()
                    playhead.setValue(time_ms)
                QApplication.processEvents()
                pixmap = self.sequence_designer.diagram_widget.grab()
                image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
                target_h = frame.shape[0]
                image = image.scaledToHeight(target_h, Qt.SmoothTransformation)
                ptr = image.bits()
                ptr.setsize(image.byteCount())
                arr = np.frombuffer(ptr, dtype=np.uint8).reshape((image.height(), image.width(), 4))[:, :, :3]
                # Combine horizontally
                h = min(frame.shape[0], arr.shape[0])
                frame_resized = frame if frame.shape[0] == h else frame[:h, :, :]
                arr_resized = arr if arr.shape[0] == h else arr[:h, :, :]
                combined = np.hstack([frame_resized, arr_resized])
                # Ensure even dimensions for encoders
                if combined.shape[0] % 2 != 0:
                    combined = combined[:-1, :, :]
                if combined.shape[1] % 2 != 0:
                    combined = combined[:, :-1, :]
                return combined
            except Exception:
                return frame

        groups = [
            {
                'title': 'Transverse Magnetization (Mx/My)',
                'ylabel': 'Magnetization',
                'series': [
                    {'data': mx_trace, 'label': 'Mx', 'color': 'r'},
                    {'data': my_trace, 'label': 'My', 'color': 'g', 'style': '--'},
                ]
            },
            {
                'title': 'Longitudinal Magnetization (Mz)',
                'ylabel': 'Magnetization',
                'series': [
                    {'data': mz_trace, 'label': 'Mz', 'color': 'b'}
                ]
            }
        ]

        exporter = AnimationExporter()
        progress = QProgressDialog("Exporting animation...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        def progress_cb(done, total):
            val = int(done / total * 100) if total else 0
            progress.setValue(val)
            QApplication.processEvents()

        def cancel_cb():
            return progress.wasCanceled()

        try:
            result = exporter.export_time_series_animation(
                time_s,
                groups,
                params['filename'],
                fps=params['fps'],
                max_frames=params['max_frames'],
                start_idx=start_idx,
                end_idx=end_idx,
                width=params['width'],
                height=params['height'],
                format=params['format'],
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                indices=indices,
                frame_hook=frame_hook
            )
            progress.setValue(100)
            if result:
                self.log_message(f"Animation exported to {result}")
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Animation exported successfully:\n{Path(result).name}"
                )
        except Exception as e:
            progress.close()
            if isinstance(e, RuntimeError) and "cancelled" in str(e).lower():
                self.log_message("Animation export cancelled by user.")
            else:
                self.log_message(f"Animation export error: {e}")
                QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")
        finally:
            if playhead is not None:
                playhead.setValue(playhead_value)
                playhead.setVisible(playhead_visible)

    def _export_magnetization_data(self):
        """Export magnetization time series as CSV/NPY/DAT."""
        if not self.last_result or self.last_result.get('mx') is None:
            QMessageBox.warning(self, "No Data", "Please run a time-resolved simulation first.")
            return
        mx = self.last_result.get('mx')
        my = self.last_result.get('my')
        mz = self.last_result.get('mz')
        if mx is None or my is None or mz is None or mx.ndim != 3:
            QMessageBox.warning(self, "No Time Series", "Data export requires time-resolved magnetization data.")
            return
        time_s = np.asarray(self.last_time) if self.last_time is not None else np.asarray(self.last_result.get('time', []))
        if time_s is None or len(time_s) != mx.shape[0]:
            QMessageBox.warning(self, "Missing Time", "Could not determine time axis for data export.")
            return

        path, fmt = self._prompt_data_export_path("magnetization")
        if not path:
            return
        try:
            result_path = self.dataset_exporter.export_magnetization(
                time_s,
                mx,
                my,
                mz,
                self.last_positions,
                self.last_frequencies,
                str(path),
                format=fmt
            )
            self.log_message(f"Magnetization data exported to {result_path}")
            QMessageBox.information(self, "Export Successful", f"Magnetization data saved:\n{Path(result_path).name}")
        except Exception as e:
            self.log_message(f"Magnetization data export failed: {e}")
            QMessageBox.critical(self, "Export Error", f"Could not export magnetization data:\n{e}")

    def _export_3d_animation(self):
        """Export the 3D vector view as a GIF/MP4."""
        if self.anim_data is None or self.playback_time is None or len(self.anim_data) < 2:
            QMessageBox.warning(self, "No Data", "3D animation export requires a time-resolved simulation.")
            return
        if vz_imageio is None:
            QMessageBox.critical(self, "Missing Dependency", "Animation export requires 'imageio'. Install with: pip install imageio imageio-ffmpeg")
            return

        total_frames = len(self.anim_data)
        dialog = ExportAnimationDialog(
            self,
            total_frames=total_frames,
            default_filename="vector3d",
            default_directory=self._get_export_directory()
        )
        # Mean-only not applicable for 3D view (already uses colored vectors); hide toggle
        dialog.mean_only_checkbox.setVisible(False)

        if dialog.exec_() != QDialog.Accepted:
            return

        params = dialog.get_export_params()
        exporter = AnimationExporter()
        indices = exporter._compute_indices(
            total_frames,
            max_frames=params['max_frames'],
            start_idx=params['start_idx'],
            end_idx=params['end_idx']
        )

        # Prepare writer
        fmt = params['format']
        filepath = Path(params['filename'])
        if filepath.suffix.lower() != f".{fmt}":
            filepath = filepath.with_suffix(f".{fmt}")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if fmt == 'gif':
            writer = vz_imageio.get_writer(str(filepath), mode='I', fps=params['fps'], format='GIF')
        else:
            writer = vz_imageio.get_writer(
                str(filepath),
                fps=params['fps'],
                format='FFMPEG',
                codec='libx264',
                bitrate=exporter.default_bitrate,
                quality=8,
                macro_block_size=None
            )

        progress = QProgressDialog("Exporting 3D animation...", "Cancel", 0, len(indices), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        # Save state and pause playback to avoid interference
        was_running = self.anim_timer.isActive()
        current_idx = self.anim_index
        self.anim_timer.stop()

        def _even(val: int) -> int:
            return val if val % 2 == 0 else val + 1

        def grab_frame():
            # Grab main view
            main_pixmap = self.mag_3d.gl_widget.grab()
            main_image = main_pixmap.toImage()

            # Optionally grab sequence diagram
            if params.get('include_sequence', False):
                seq_pixmap = self.sequence_designer.diagram_widget.grab()
                seq_image = seq_pixmap.toImage()

                # Stack vertically: main view on top, sequence diagram below (20% height)
                main_h = params['height'] if params['height'] else main_image.height()
                seq_h = int(main_h * 0.25)  # 25% of main height for diagram
                total_h = main_h + seq_h
                total_w = params['width'] if params['width'] else main_image.width()

                # Scale main image
                main_image = main_image.scaled(total_w, main_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                # Scale sequence diagram
                seq_image = seq_image.scaled(total_w, seq_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

                # Create composite image
                from PyQt5.QtGui import QPainter, QImage as QImg
                composite = QImg(total_w, total_h, QImg.Format_RGBA8888)
                composite.fill(Qt.white)
                painter = QPainter(composite)
                painter.drawImage(0, 0, main_image)
                painter.drawImage(0, main_h, seq_image)
                painter.end()
                image = composite
            else:
                # Just the main view
                target_w = params['width'] if params['width'] else main_image.width()
                target_h = params['height'] if params['height'] else main_image.height()
                aspect_mode = Qt.IgnoreAspectRatio if (params['width'] and params['height']) else Qt.KeepAspectRatio
                image = main_image.scaled(target_w, target_h, aspect_mode, Qt.SmoothTransformation)

            # Enforce even dimensions for codecs
            if image.width() % 2 or image.height() % 2:
                image = image.scaled(_even(image.width()), _even(image.height()), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            image = image.convertToFormat(QImage.Format_RGBA8888)
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            frame = np.frombuffer(ptr, dtype=np.uint8).reshape((image.height(), image.width(), 4))[:, :, :3]
            return frame

        try:
            for i, idx in enumerate(indices):
                if progress.wasCanceled():
                    raise RuntimeError("Animation export cancelled")
                self._set_animation_index_from_slider(int(idx))
                QApplication.processEvents()
                frame = grab_frame()
                writer.append_data(frame)
                progress.setValue(i + 1)
                QApplication.processEvents()
        finally:
            writer.close()
            # Restore playback state
            self._set_animation_index_from_slider(current_idx)
            if was_running:
                self._resume_vector_animation()
            progress.setValue(progress.maximum())

        QMessageBox.information(
            self,
            "Export Successful",
            f"3D animation exported successfully:\n{filepath.name}"
        )
        self.log_message(f"3D animation exported to {filepath}")

    def _export_signal_image(self, default_format='png'):
        """Export signal plot as an image."""
        if not self.last_result:
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return

        export_dir = self._get_export_directory()
        dialog = ExportImageDialog(self, default_filename="signal", default_directory=export_dir)
        dialog.format_combo.setCurrentIndex(['png', 'svg', 'pdf'].index(default_format))

        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_export_params()
            exporter = ImageExporter()

            try:
                result = exporter.export_pyqtgraph_plot(
                    self.signal_plot,
                    params['filename'],
                    format=params['format'],
                    width=params['width']
                )

                if result:
                    self.log_message(f"Exported signal plot to: {result}")
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Signal plot exported successfully:\n{Path(result).name}"
                    )
                else:
                    self.log_message("Export failed")
                    QMessageBox.warning(self, "Export Failed", "Could not export plot. Check the log for details.")

            except Exception as e:
                self.log_message(f"Export error: {e}")
                QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")

    def _export_signal_animation(self):
        """Export received signal as animation."""
        if not self.last_result or 'signal' not in self.last_result:
            QMessageBox.warning(self, "No Data", "Please run a time-resolved simulation first.")
            return

        signal_arr = self.last_result['signal']
        if signal_arr is None or signal_arr.ndim < 2:
            QMessageBox.warning(self, "No Time Series", "Animation export requires time-resolved data.")
            return

        time_s = np.asarray(self.last_time) if self.last_time is not None else np.asarray(self.last_result.get('time', []))
        if time_s is None or len(time_s) == 0:
            QMessageBox.warning(self, "Missing Time", "Could not determine time axis for animation export.")
            return

        # Ensure alignment between time and signal length
        nframes = min(len(time_s), signal_arr.shape[0])
        if nframes < 2:
            QMessageBox.warning(self, "Insufficient Data", "Need at least two time points to export animation.")
            return
        time_s = time_s[:nframes]

        dialog = ExportAnimationDialog(
            self,
            total_frames=nframes,
            default_filename="signal",
            default_directory=self._get_export_directory()
        )
        dialog.mean_only_checkbox.setChecked(self.mean_only_checkbox.isChecked())
        if dialog.exec_() != QDialog.Accepted:
            return

        params = dialog.get_export_params()
        start_idx = min(params['start_idx'], nframes - 1)
        end_idx = max(start_idx, min(params['end_idx'], nframes - 1))
        mean_only = params['mean_only']
        exporter = AnimationExporter()
        indices = exporter._compute_indices(nframes, max_frames=params['max_frames'], start_idx=start_idx, end_idx=end_idx)

        def _select_signal(arr):
            if arr.ndim == 3:
                if mean_only:
                    return np.mean(arr, axis=(1, 2))
                return arr[:, 0, 0]
            if arr.ndim == 2:
                if mean_only:
                    return np.mean(arr, axis=1)
                return arr[:, 0]
            return np.asarray(arr)

        sig_trace = _select_signal(signal_arr)[:nframes]
        groups = [
            {
                'title': 'Signal Magnitude',
                'ylabel': '|S|',
                'series': [{'data': np.abs(sig_trace), 'label': '|S|', 'color': 'c'}]
            },
            {
                'title': 'Signal Components',
                'ylabel': 'Amplitude',
                'series': [
                    {'data': np.real(sig_trace), 'label': 'Real', 'color': 'm'},
                    {'data': np.imag(sig_trace), 'label': 'Imag', 'color': 'y', 'style': '--'},
                ]
            }
        ]

        progress = QProgressDialog("Exporting animation...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        def progress_cb(done, total):
            val = int(done / total * 100) if total else 0
            progress.setValue(val)
            QApplication.processEvents()

        def cancel_cb():
            return progress.wasCanceled()

        try:
            result = exporter.export_time_series_animation(
                time_s,
                groups,
                params['filename'],
                fps=params['fps'],
                max_frames=params['max_frames'],
                start_idx=start_idx,
                end_idx=end_idx,
                width=params['width'],
                height=params['height'],
                format=params['format'],
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                indices=indices
            )
            progress.setValue(100)
            if result:
                self.log_message(f"Signal animation exported to {result}")
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Signal animation exported successfully:\n{Path(result).name}"
                )
        except Exception as e:
            progress.close()
            if isinstance(e, RuntimeError) and "cancelled" in str(e).lower():
                self.log_message("Signal animation export cancelled by user.")
            else:
                self.log_message(f"Signal animation export error: {e}")
                QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")

    def _export_signal_data(self):
        """Export received signal traces as CSV/NPY/DAT."""
        if not self.last_result or self.last_result.get('signal') is None:
            QMessageBox.warning(self, "No Data", "Please run a time-resolved simulation first.")
            return
        signal_arr = self.last_result.get('signal')
        if signal_arr is None or signal_arr.ndim < 2:
            QMessageBox.warning(self, "No Time Series", "Data export requires time-resolved signal data.")
            return

        time_s = np.asarray(self.last_time) if self.last_time is not None else np.asarray(self.last_result.get('time', []))
        nframes = min(len(time_s), signal_arr.shape[0])
        if nframes < 1:
            QMessageBox.warning(self, "Missing Time", "Could not determine time axis for data export.")
            return
        time_s = time_s[:nframes]
        signal_arr = signal_arr[:nframes]

        path, fmt = self._prompt_data_export_path("signal")
        if not path:
            return
        try:
            result_path = self.dataset_exporter.export_signal(time_s, signal_arr, str(path), format=fmt)
            self.log_message(f"Signal data exported to {result_path}")
            QMessageBox.information(self, "Export Successful", f"Signal data saved:\n{Path(result_path).name}")
        except Exception as e:
            self.log_message(f"Signal data export failed: {e}")
            QMessageBox.critical(self, "Export Error", f"Could not export signal data:\n{e}")

    def _export_spectrum_animation(self):
        """Export spectrum plot animation via widget grab."""
        def updater(idx):
            actual_idx = self._playback_to_full_index(idx)
            self._refresh_spectrum(time_idx=actual_idx)
        self._export_widget_animation([self.spectrum_plot], default_filename="spectrum", before_grab=updater)

    def _export_spatial_animation(self):
        """Export spatial plots animation via widget grab."""
        self._export_widget_animation([self.spatial_mxy_plot, self.spatial_mz_plot], default_filename="spatial")

    def _export_spectrum_image(self, default_format='png'):
        """Export spectrum plot as an image."""
        if not self.last_result:
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return

        export_dir = self._get_export_directory()
        dialog = ExportImageDialog(self, default_filename="spectrum", default_directory=export_dir)
        dialog.format_combo.setCurrentIndex(['png', 'svg', 'pdf'].index(default_format))

        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_export_params()
            exporter = ImageExporter()

            try:
                result = exporter.export_pyqtgraph_plot(
                    self.spectrum_plot,
                    params['filename'],
                    format=params['format'],
                    width=params['width']
                )

                if result:
                    self.log_message(f"Exported spectrum plot to: {result}")
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Spectrum plot exported successfully:\n{Path(result).name}"
                    )
                else:
                    self.log_message("Export failed")
                    QMessageBox.warning(self, "Export Failed", "Could not export plot. Check the log for details.")

            except Exception as e:
                self.log_message(f"Export error: {e}")
                QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")

    def _export_spectrum_data(self):
        """Export spectrum data as CSV/NPY/DAT."""
        if self.last_result is None:
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return
        actual_idx = self._current_playback_index()
        self._refresh_spectrum(time_idx=actual_idx)
        export_cache = getattr(self, "_last_spectrum_export", None)
        if not export_cache:
            QMessageBox.warning(self, "No Spectrum", "Spectrum data is not available for export.")
            return

        freq = export_cache.get("frequency")
        if freq is None or len(freq) == 0:
            QMessageBox.warning(self, "No Spectrum", "Spectrum data is not available for export.")
            return
        series = {
            "selected_magnitude": export_cache.get("selected_magnitude"),
            "selected_phase_rad": export_cache.get("selected_phase_rad"),
        }
        if export_cache.get("mean_magnitude") is not None:
            series["mean_magnitude"] = export_cache.get("mean_magnitude")
        if export_cache.get("mean_phase_rad") is not None:
            series["mean_phase_rad"] = export_cache.get("mean_phase_rad")

        path, fmt = self._prompt_data_export_path("spectrum")
        if not path:
            return
        try:
            result_path = self.dataset_exporter.export_spectrum(freq, series, str(path), format=fmt)
            self.log_message(f"Spectrum data exported to {result_path}")
            QMessageBox.information(self, "Export Successful", f"Spectrum data saved:\n{Path(result_path).name}")
        except Exception as e:
            self.log_message(f"Spectrum data export failed: {e}")
            QMessageBox.critical(self, "Export Error", f"Could not export spectrum data:\n{e}")

    def _export_spatial_image(self, default_format='png'):
        """Export spatial plots as images."""
        if not self.last_result:
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return

        export_dir = self._get_export_directory()
        dialog = ExportImageDialog(self, default_filename="spatial", default_directory=export_dir)
        dialog.format_combo.setCurrentIndex(['png', 'svg', 'pdf'].index(default_format))

        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_export_params()
            exporter = ImageExporter()

            try:
                # Export both spatial plots
                base_path = Path(params['filename'])

                # Export Mxy spatial plot
                mxy_path = base_path.parent / f"{base_path.stem}_mxy{base_path.suffix}"
                result_mxy = exporter.export_pyqtgraph_plot(
                    self.spatial_mxy_plot,
                    str(mxy_path),
                    format=params['format'],
                    width=params['width']
                )

                # Export Mz spatial plot
                mz_path = base_path.parent / f"{base_path.stem}_mz{base_path.suffix}"
                result_mz = exporter.export_pyqtgraph_plot(
                    self.spatial_mz_plot,
                    str(mz_path),
                    format=params['format'],
                    width=params['width']
                )

                if result_mxy and result_mz:
                    self.log_message(f"Exported spatial plots to:\n  {result_mxy}\n  {result_mz}")
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Spatial plots exported successfully:\n\n"
                        f"Mxy: {Path(result_mxy).name}\n"
                        f"Mz: {Path(result_mz).name}"
                    )
                else:
                    self.log_message("Export failed")
                    QMessageBox.warning(self, "Export Failed", "Could not export plots. Check the log for details.")

            except Exception as e:
                self.log_message(f"Export error: {e}")
                QMessageBox.critical(self, "Export Error", f"An error occurred:\n{e}")

    def _export_spatial_data(self):
        """Export spatial profiles as CSV/NPY/DAT."""
        if self.last_result is None or self.last_positions is None:
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return
        # Ensure cache reflects current frame
        self.update_spatial_plot_from_last_result(time_idx=self._current_playback_index())
        cache = getattr(self, "_last_spatial_export", None)
        if not cache:
            QMessageBox.warning(self, "No Spatial Data", "Spatial data is not available for export.")
            return

        position = cache.get("position_m")
        mxy = cache.get("mxy")
        mz = cache.get("mz")
        if position is None or mxy is None or mz is None:
            QMessageBox.warning(self, "No Spatial Data", "Spatial data is not available for export.")
            return

        path, fmt = self._prompt_data_export_path("spatial")
        if not path:
            return
        try:
            result_path = self.dataset_exporter.export_spatial(
                position,
                mxy,
                mz,
                str(path),
                format=fmt,
                mxy_per_freq=cache.get("mxy_per_freq"),
                mz_per_freq=cache.get("mz_per_freq")
            )
            self.log_message(f"Spatial data exported to {result_path}")
            QMessageBox.information(self, "Export Successful", f"Spatial data saved:\n{Path(result_path).name}")
        except Exception as e:
            self.log_message(f"Spatial data export failed: {e}")
            QMessageBox.critical(self, "Export Error", f"Could not export spatial data:\n{e}")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About Bloch Simulator",
            "Bloch Equation Simulator\n\n"
            "A Python implementation of the Bloch equation solver\n"
            "originally developed by Brian Hargreaves.\n\n"
            "This GUI provides interactive visualization and\n"
            "parameter control for MRI pulse sequence simulation.\n\n"
            "Version 1.0.0")


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = BlochSimulatorGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
