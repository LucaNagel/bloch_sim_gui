"""
bloch_gui.py - Interactive GUI for Bloch equation simulator

This module provides a graphical user interface for designing pulse sequences,
setting parameters, running simulations, and visualizing results.

Author: Your Name
Date: 2024
"""

import os
import sys
import math
import time
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
    QSizePolicy, QMenu, QDialog, QProgressDialog, QToolBar, QFormLayout,
    QDialogButtonBox, QListWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QImage

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from . import __version__

# Import the simulator
from .simulator import (
    BlochSimulator, TissueParameters,
    SpinEcho, SpinEchoTipAxis, GradientEcho, SliceSelectRephase, CustomPulse, PulseSequence, design_rf_pulse
)

# Import visualization export tools
from .visualization import (
    ImageExporter,
    ExportImageDialog,
    AnimationExporter,
    ExportAnimationDialog,
    ExportDataDialog,
    DatasetExporter,
    imageio as vz_imageio,
)

# Import phantom module for 2D/3D phantom simulation
try:
    from .phantom import Phantom, PhantomFactory
    from .phantom_widget import PhantomWidget
    PHANTOM_AVAILABLE = False  # Disabled per user request
except ImportError:
    PHANTOM_AVAILABLE = False
    print("Phantom module not available - phantom tab will be disabled")

# Import k-space simulation widget
try:
    from .kspace_widget import KSpaceWidget
    from .kspace import KSpaceSimulator, EddyCurrentModel, EPIParameters, CSIParameters
    KSPACE_AVAILABLE = False  # Disabled per user request
except ImportError:
    KSPACE_AVAILABLE = False
    print("K-Space module not available - k-space tab will be disabled")


def get_app_data_dir() -> Path:
    """Return a writable per-user application directory."""
    override = os.environ.get("BLOCH_APP_DIR")
    if override:
        return Path(override).expanduser()

    system = sys.platform
    if system.startswith("win"):
        root = Path(os.environ.get("APPDATA", Path.home()))
    elif system == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    return root / "BlochSimulator"


class CheckableComboBox(QComboBox):
    """A combo box with checkable items for multi-selection."""
    
    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.closeOnLineEditClick = False
        self.lineEdit().installEventFilter(self)
        self.model().dataChanged.connect(self._on_model_data_changed)
        
    def _on_model_data_changed(self, top_left, bottom_right, roles):
        if Qt.CheckStateRole in roles:
            self.update_display_text()
            self.selection_changed.emit()

    def eventFilter(self, obj, event):
        if obj == self.lineEdit() and event.type() == event.MouseButtonRelease:
            if self.closeOnLineEditClick:
                self.hidePopup()
            else:
                self.showPopup()
            return True
        return super().eventFilter(obj, event)

    def showPopup(self):
        super().showPopup()
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        self.closeOnLineEditClick = False

    def add_items(self, items):
        for text in items:
            self.addItem(text)
            item = self.model().item(self.count() - 1)
            item.setCheckState(Qt.Unchecked)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)

    def get_selected_items(self):
        selected = []
        for i in range(self.count()):
            item = self.model().item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected

    def set_selected_items(self, items):
        self.model().blockSignals(True)
        for i in range(self.count()):
            item = self.model().item(i)
            if item.text() in items:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
        self.model().blockSignals(False)
        self.update_display_text()

    def update_display_text(self):
        selected = self.get_selected_items()
        text = ", ".join(selected) if selected else "None"
        self.lineEdit().setText(text)
        
    def currentText(self):
        return self.lineEdit().text()


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


class PulseImportDialog(QDialog):
    """Dialog to configure loading of custom amp/phase pulse files."""

    def __init__(self, parent=None, filename: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Import RF Pulse Options")
        layout = QVBoxLayout()
        form = QFormLayout()

        if filename:
            form.addRow(QLabel(f"File: {Path(filename).name}"))

        self.layout_mode = QComboBox()
        self.layout_mode.addItems([
            "Interleaved: amp, phase, amp, phase",
            "Interleaved: phase, amp, phase, amp",
            "Columns: amp | phase per row",
        ])
        self.layout_mode.setCurrentIndex(0)
        form.addRow("Data layout:", self.layout_mode)

        self.amp_unit = QComboBox()
        self.amp_unit.addItems([
            "Percent (0-100)",
            "Fraction (0-1)",
            "Gauss",
            "mT",
            "uT",
        ])
        self.amp_unit.setCurrentIndex(0)
        form.addRow("Amplitude unit:", self.amp_unit)

        self.phase_unit = QComboBox()
        self.phase_unit.addItems(["Degrees", "Radians"])
        self.phase_unit.setCurrentIndex(0)
        form.addRow("Phase unit:", self.phase_unit)

        self.duration_ms = QDoubleSpinBox()
        self.duration_ms.setRange(0.001, 100000.0)
        self.duration_ms.setDecimals(3)
        self.duration_ms.setSingleStep(0.1)
        self.duration_ms.setValue(1.0)
        form.addRow("Duration (ms):", self.duration_ms)

        layout.addLayout(form)
        layout.addWidget(QLabel("Tip: Percent/fraction amplitudes are treated as relative and rescaled from flip angle."))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_options(self) -> dict:
        layout_choice = self.layout_mode.currentText()
        if layout_choice.startswith("Interleaved: amp"):
            layout = "amp_phase_interleaved"
        elif layout_choice.startswith("Interleaved: phase"):
            layout = "phase_amp_interleaved"
        else:
            layout = "columns"

        amp_unit_text = self.amp_unit.currentText().lower()
        if "percent" in amp_unit_text:
            amp_unit = "percent"
        elif "fraction" in amp_unit_text:
            amp_unit = "fraction"
        elif amp_unit_text.startswith("mt"):
            amp_unit = "mt"
        elif amp_unit_text.startswith("ut"):
            amp_unit = "ut"
        else:
            amp_unit = "gauss"

        phase_unit = "deg" if self.phase_unit.currentText().lower().startswith("deg") else "rad"

        return {
            "layout": layout,
            "amp_unit": amp_unit,
            "phase_unit": phase_unit,
            "duration_s": float(self.duration_ms.value()) / 1000.0,
        }


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
        self.last_integration_factor = 1.0
        self.init_ui()
        self.current_pulse = None
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Pulse type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Pulse Type:"))
        self.pulse_type = QComboBox()
        self.pulse_type.addItems(["Rectangle", "Sinc", "Gaussian", "Hermite", "Adiabatic Half Passage", "Adiabatic Full Passage", "BIR-4", "Custom"])
        self.pulse_type.currentTextChanged.connect(self.update_pulse)
        type_layout.addWidget(self.pulse_type)
        layout.addLayout(type_layout)
        
        # Flip angle
        flip_layout = QHBoxLayout()
        flip_layout.addWidget(QLabel("Flip Angle (°):"))
        self.flip_angle = QDoubleSpinBox()
        self.flip_angle.setRange(0, 1e4)
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
        
        # Time-bandwidth product (computed from pulse shape; not user-set)
        tbw_layout = QHBoxLayout()
        tbw_layout.addWidget(QLabel("Time-BW Product (auto):"))
        self.tbw = QDoubleSpinBox()
        self.tbw.setRange(0.001, 1000)
        self.tbw.setValue(1)
        self.tbw.setSingleStep(0.5)
        self.tbw.setReadOnly(True)
        self.tbw.setButtonSymbols(QDoubleSpinBox.NoButtons)
        tbw_layout.addWidget(self.tbw)
        layout.addLayout(tbw_layout)
        self.tbw_auto_label = QLabel("Auto TBW (≈1/integfac): —")
        self.tbw_auto_label.setStyleSheet("color: gray;")
        layout.addWidget(self.tbw_auto_label)
        
        # Lobes control for Sinc pulses
        lobes_layout = QHBoxLayout()
        lobes_layout.addWidget(QLabel("Lobes (Sinc):"))
        self.sinc_lobes = QSpinBox()
        self.sinc_lobes.setRange(1, 100)
        self.sinc_lobes.setValue(3)
        self.sinc_lobes.valueChanged.connect(self.update_pulse)
        lobes_layout.addWidget(self.sinc_lobes)
        self.lobes_container = QWidget()
        self.lobes_container.setLayout(lobes_layout)
        layout.addWidget(self.lobes_container)
        
        # Apodization
        apod_layout = QHBoxLayout()
        apod_layout.addWidget(QLabel("Apodization:"))
        self.apodization_combo = QComboBox()
        self.apodization_combo.addItems(["None", "Hamming", "Hanning", "Blackman"])
        self.apodization_combo.currentTextChanged.connect(self.update_pulse)
        apod_layout.addWidget(self.apodization_combo)
        layout.addLayout(apod_layout)

        # Phase
        phase_layout = QHBoxLayout()

        phase_layout.addWidget(QLabel("Phase (°):"))
        self.phase = QDoubleSpinBox()
        self.phase.setRange(0, 360)
        self.phase.setValue(0)
        self.phase.valueChanged.connect(self.update_pulse)
        phase_layout.addWidget(self.phase)
        layout.addLayout(phase_layout)

        # RF Frequency Offset
        freq_offset_layout = QHBoxLayout()
        freq_offset_layout.addWidget(QLabel("RF Frequency Offset (Hz):"))
        self.freq_offset = QDoubleSpinBox()
        self.freq_offset.setRange(-10000, 10000)
        self.freq_offset.setValue(0.0)
        self.freq_offset.setSingleStep(10)
        self.freq_offset.setDecimals(1)
        self.freq_offset.valueChanged.connect(self.update_pulse)
        freq_offset_layout.addWidget(self.freq_offset)
        layout.addLayout(freq_offset_layout)

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

    def _update_tbw_auto(self, integration_factor: float):
        """Set TBW readout from an integration factor (heuristic: TBW ≈ 1/integfac)."""
        if not hasattr(self, "tbw") or not hasattr(self, "tbw_auto_label"):
            return
        if integration_factor is None or not np.isfinite(integration_factor) or integration_factor <= 0:
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
        if pt in ("sinc", "gaussian", "hermite"):
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

    def _scale_pulse_to_flip(self, b1_wave, t_wave, flip_deg: float, integfac: float = 1.0):
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
        pulse_type = self.pulse_type.currentText().lower()
        if pulse_type == 'rectangle':
            pulse_type = 'rect'
        elif pulse_type == 'adiabatic half passage':
            pulse_type = 'adiabatic_half'
        elif pulse_type == 'adiabatic full passage':
            pulse_type = 'adiabatic_full'
        elif pulse_type == 'bir-4':
            pulse_type = 'bir4'
        elif pulse_type == 'custom':
            # For custom pulses, trigger reprocessing if data is loaded
            if self.loaded_pulse_b1 is not None:
                self.reprocess_custom_pulse()
            return

        # Hide custom settings when not using custom pulse
        self.custom_settings_group.setVisible(False)
        self.lobes_container.setVisible(pulse_type == 'sinc')
            
        duration = self.duration.value() / 1000  # Convert to seconds
        flip = self.flip_angle.value()
        
        # Calculate TBW based on pulse type
        if pulse_type == 'sinc':
            design_tbw = float(self.sinc_lobes.value()) + 1.0 # Approximation: lobes ~ TBW - 1
        else:
            design_tbw = self._design_tbw_for_type(pulse_type)
            
        phase_rad = np.deg2rad(self.phase.value())
        # Increase time resolution using desired dt
        if self.target_dt and self.target_dt > 0:
            npoints = max(32, int(np.ceil(duration / self.target_dt)))
            npoints = min(npoints, 50000)  # avoid runaway
        else:
            npoints = 100
        
        # Design pulse (without frequency offset first, to get base shape statistics)
        # We apply freq offset manually later to ensure TBW calculation is invariant to offset.
        freq_offset_hz = self.freq_offset.value()
        
        # 1. Generate base pulse (centered at 0 Hz)
        b1_base, time = design_rf_pulse(pulse_type, duration, flip, design_tbw, npoints, freq_offset=0.0)
        
        # Sampling time [s]
        dt = duration / len(b1_base)

        # Normalize pulse shape to unit peak for amplitude scaling
        peak = np.max(np.abs(b1_base)) if np.any(np.abs(b1_base)) else 1.0
        shape = b1_base / peak

        # Apply Apodization
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

        # Find a global phase that maximizes the real-valued area of the waveform.
        # This keeps the flip-angle calibration stable even for complex pulses.
        area = np.trapezoid(shape, dx=dt)
        opt_phase = -np.angle(area) if np.isfinite(area) else 0.0
        aligned_area = np.real(area * np.exp(1j * opt_phase))
        if not np.isfinite(aligned_area) or abs(aligned_area) < 1e-12:
            aligned_area = 1e-12

        # Estimate integration factor relative to a unit-amplitude block
        # This is now independent of frequency offset because we use b1_base
        try:
            integration_factor = abs(aligned_area) / max(duration, 1e-12)
        except Exception:
            integration_factor = None
        self._update_tbw_auto(integration_factor)
        if integration_factor is not None and np.isfinite(integration_factor):
            self.last_integration_factor = float(integration_factor)

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
        
        # 2. Apply frequency offset modulation
        if freq_offset_hz != 0.0:
            # design_rf_pulse uses: b1 * np.exp(2j * np.pi * freq_offset * time)
            # We replicate that here.
            # Note: time array starts at 0
            mod = np.exp(2j * np.pi * freq_offset_hz * time)
            b1 = b1 * mod
        
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
            "Pulse Files (*.exc *.dat *.txt *.csv);;All Files (*)"
        )
        if filename:
            try:
                suffix = Path(filename).suffix.lower()
                if suffix == ".exc":
                    from pulse_loader import load_pulse_from_file as load_exc_file
                    b1, time, metadata = load_exc_file(filename)
                else:
                    # Let user describe how to interpret amp/phase text files
                    dlg = PulseImportDialog(self, filename)
                    if dlg.exec_() != QDialog.Accepted:
                        return
                    opts = dlg.get_options()
                    from pulse_loader import load_amp_phase_dat
                    b1, time, metadata = load_amp_phase_dat(
                        filename,
                        duration_s=opts["duration_s"],
                        amplitude_unit=opts["amp_unit"],
                        phase_unit=opts["phase_unit"],
                        layout=opts["layout"],
                    )

                # Apply user phase/frequency and calibrate amplitude to current flip angle
                b1 = self._apply_phase_and_offset(b1, time)
                integfac = None
                try:
                    if hasattr(metadata, "integfac") and metadata.integfac not in (None, 0):
                        if np.isfinite(metadata.integfac) and metadata.integfac > 0:
                            integfac = float(metadata.integfac)
                except Exception:
                    integfac = None
                if integfac is None:
                    integfac = self._compute_integration_factor_from_wave(b1, time)
                b1 = self._scale_pulse_to_flip(b1, time, self.flip_angle.value(), integfac=integfac)
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
                tbw_hint = None
                integfac = None
                try:
                    if hasattr(metadata, "integfac") and metadata.integfac not in (None, 0):
                        if np.isfinite(metadata.integfac) and metadata.integfac > 0:
                            integfac = float(metadata.integfac)
                            tbw_hint = 1.0 / integfac
                except Exception:
                    tbw_hint = None
                    integfac = None
                tbw_text = f", TBW≈{tbw_hint:.3f} (1/integfac={integfac:.3f})" if tbw_hint else ""
                self.custom_info_label.setText(
                    f"Original: {metadata.duration*1000:.3f} ms, {max_b1:.6f} G{tbw_text}"
                )

                # If we have an estimated TBW from integfac, surface it in the designer control
                if tbw_hint:
                    self.tbw.blockSignals(True)
                    self.tbw.setValue(tbw_hint)
                    self.tbw.blockSignals(False)
                self._update_tbw_auto(integfac)
                if integfac is not None and np.isfinite(integfac):
                    self.last_integration_factor = float(integfac)
                else:
                    self.last_integration_factor = self._compute_integration_factor_from_wave(b1, time)

                # Show custom settings panel
                self.custom_settings_group.setVisible(True)

                # Store the loaded pulse as current
                self.current_pulse = (b1, time)
                # Recompute integration factor/TBW from the loaded waveform
                computed_integ = self._compute_integration_factor_from_wave(b1, time)
                self._update_tbw_auto(computed_integ)
                self.last_integration_factor = float(computed_integ)

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

    def get_state(self) -> dict:
        """Get the current UI state of the pulse designer."""
        state = {
            "pulse_type": self.pulse_type.currentText(),
            "flip_angle": self.flip_angle.value(),
            "duration": self.duration.value(),
            "phase": self.phase.value(),
            "freq_offset": self.freq_offset.value(),
            "sinc_lobes": self.sinc_lobes.value(),
            "apodization": self.apodization_combo.currentText(),
            # Custom settings
            "custom_duration": self.custom_duration.value(),
            "custom_b1_amplitude": self.custom_b1_amplitude.value(),
        }
        # Include loaded pulse data
        state["loaded_pulse_b1"] = self.loaded_pulse_b1
        state["loaded_pulse_time"] = self.loaded_pulse_time
        state["loaded_pulse_metadata"] = getattr(self, "loaded_pulse_metadata", None)
        return state

    def set_state(self, state: dict):
        """Restore the UI state."""
        if not state:
            return
        
        # Block signals to prevent intermediate updates
        self.pulse_type.blockSignals(True)
        self.flip_angle.blockSignals(True)
        self.duration.blockSignals(True)
        self.phase.blockSignals(True)
        self.freq_offset.blockSignals(True)
        self.sinc_lobes.blockSignals(True)
        self.apodization_combo.blockSignals(True)
        self.custom_duration.blockSignals(True)
        self.custom_b1_amplitude.blockSignals(True)
        
        try:
            if "pulse_type" in state:
                self.pulse_type.setCurrentText(state["pulse_type"])
            if "flip_angle" in state:
                self.flip_angle.setValue(state["flip_angle"])
            if "duration" in state:
                self.duration.setValue(state["duration"])
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
                
            if "custom_duration" in state:
                self.custom_duration.setValue(state["custom_duration"])
            if "custom_b1_amplitude" in state:
                self.custom_b1_amplitude.setValue(state["custom_b1_amplitude"])
        finally:
            self.pulse_type.blockSignals(False)
            self.flip_angle.blockSignals(False)
            self.duration.blockSignals(False)
            self.phase.blockSignals(False)
            self.freq_offset.blockSignals(False)
            self.sinc_lobes.blockSignals(False)
            self.apodization_combo.blockSignals(False)
            self.custom_duration.blockSignals(False)
            self.custom_b1_amplitude.blockSignals(False)
        
        # Trigger update once
        self.update_pulse()

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
        metadata = getattr(self, "loaded_pulse_metadata", None)
        integfac = 1.0
        try:
            if metadata is not None and hasattr(metadata, "integfac"):
                if metadata.integfac not in (None, 0) and np.isfinite(metadata.integfac):
                    integfac = float(metadata.integfac)
        except Exception:
            integfac = 1.0

        # Calculate time scaling factor
        new_duration_s = new_duration_ms / 1000.0
        time_scale = new_duration_s / original_duration if original_duration > 0 else 1.0

        # Rescale time array
        new_time = original_time * time_scale

        # Resample B1 if needed (for now, just use the original samples with scaled time)
        new_b1 = original_b1.copy()

        # Apply Apodization
        window_type = self.apodization_combo.currentText()
        if window_type != "None" and len(new_b1) > 1:
            if window_type == "Hamming":
                win = np.hamming(len(new_b1))
            elif window_type == "Hanning":
                win = np.hanning(len(new_b1))
            elif window_type == "Blackman":
                win = np.blackman(len(new_b1))
            else:
                win = np.ones(len(new_b1))
            new_b1 = new_b1 * win

        # Apply B1 amplitude override
        if new_b1_amplitude > 0:
            # Use specified amplitude - scale to match target
            current_max_b1 = np.max(np.abs(new_b1))
            if current_max_b1 > 0:
                b1_scale = new_b1_amplitude / current_max_b1
                new_b1 = new_b1 * b1_scale
        else:
            # Use flip angle calibration (auto mode)
            # Recalculate B1 amplitude to achieve target flip angle
            flip = self.flip_angle.value()  # degrees
            flip_rad = np.deg2rad(flip)

            # Normalize pulse shape to unit peak (using windowed version)
            peak = np.max(np.abs(new_b1)) if np.any(np.abs(new_b1)) else 1.0
            shape = new_b1 / peak

            # Calculate sampling time for new duration
            dt = new_duration_s / len(shape) if len(shape) > 0 else 1e-6

            # Find optimal phase that maximizes real-valued area
            area = np.trapezoid (shape, dx=dt)
            opt_phase = -np.angle(area) if np.isfinite(area) and area != 0 else 0.0
            aligned_area = np.real(area * np.exp(1j * opt_phase))
            if not np.isfinite(aligned_area) or abs(aligned_area) < 1e-12:
                aligned_area = 1e-12
            aligned_area *= integfac  # vendor integration factor scales effective area

            # Gyromagnetic ratio for 1H (rad / (T s))
            gmr_1h_rad_Ts = 267522187.43999997

            # Calculate required pulse amplitude in Tesla
            pulse_amp_T = flip_rad / (gmr_1h_rad_Ts * aligned_area)
            # Convert to Gauss
            pulse_amp_G = pulse_amp_T * 1e4

            # Apply amplitude and phase to achieve target flip angle
            new_b1 = shape * pulse_amp_G * np.exp(1j * opt_phase)
        # Apply user-selected phase and frequency offset
        new_b1 = self._apply_phase_and_offset(new_b1, new_time)

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

        # Update heuristic TBW from integration factor (recomputed)
        computed_integ = self._compute_integration_factor_from_wave(new_b1, new_time)
        self._update_tbw_auto(computed_integ)
        if np.isfinite(computed_integ):
            self.last_integration_factor = float(computed_integ)

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
        self.pulse_states = {}  # Store UI state for each pulse role
        self.pulse_waveforms = {}  # Store (b1, time) for each pulse role
        self.current_role = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Pulse selector
        pulse_layout = QHBoxLayout()
        pulse_layout.addWidget(QLabel("Pulses:"))
        self.pulse_list = QListWidget()
        self.pulse_list.setFixedHeight(60)
        self.pulse_list.currentItemChanged.connect(self._on_pulse_selection_changed)
        pulse_layout.addWidget(self.pulse_list)
        layout.addLayout(pulse_layout)
        
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
        self.ssfp_start_amp.setValue(0.025)
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
        
        # Slice Rephase options
        self.slice_rephase_opts = QWidget()
        sr_layout = QHBoxLayout()
        sr_layout.addWidget(QLabel("Rephase Area (%):"))
        self.rephase_percentage = QDoubleSpinBox()
        self.rephase_percentage.setRange(-200.0, 200.0)
        self.rephase_percentage.setValue(50.0)
        self.rephase_percentage.setSingleStep(1.0)
        self.rephase_percentage.setToolTip("Percentage of slice select gradient area to rewind (50% = half area)")
        self.rephase_percentage.valueChanged.connect(lambda _: self.update_diagram())
        sr_layout.addWidget(self.rephase_percentage)
        self.slice_rephase_opts.setLayout(sr_layout)
        self.options_container.addWidget(self.slice_rephase_opts)

        layout.addLayout(self.options_container)
        self.spin_echo_opts.setVisible(False)
        self.ssfp_opts.setVisible(False)
        self.slice_rephase_opts.setVisible(False)
        
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
        self.ti_widget = QWidget()
        self.ti_widget.setLayout(ti_layout)
        layout.addWidget(self.ti_widget)
        self.ti_spin.valueChanged.connect(lambda _: self.update_diagram())
        # Initialize option visibility after all widgets are created
        self._update_sequence_options()
        
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

    def _effective_tbw(self) -> float:
        """Return best-effort time-bandwidth product from RF designer integration factor."""
        try:
            if hasattr(self, "parent_gui") and hasattr(self.parent_gui, "rf_designer"):
                integ = float(self.parent_gui.rf_designer.get_integration_factor())
                if np.isfinite(integ) and integ > 0:
                    return 1.0 / integ
        except Exception:
            pass
        return 4.0

    def _update_sequence_options(self):
        """Show/hide sequence-specific option widgets and update pulse list."""
        seq_type = self.sequence_type.currentText()
        self.spin_echo_opts.setVisible(seq_type in ("Spin Echo", "Spin Echo (Tip-axis 180)"))
        self.ssfp_opts.setVisible(seq_type == "SSFP (Loop)")
        self.slice_rephase_opts.setVisible(seq_type == "Slice Select + Rephase")
        self.ti_widget.setVisible(seq_type == "Inversion Recovery")
        
        # Update pulse list based on sequence type
        self.pulse_list.blockSignals(True)
        self.pulse_list.clear()
        
        roles = []
        if seq_type in ("Spin Echo", "Spin Echo (Tip-axis 180)"):
            roles = ["Excitation", "Refocusing"]
            
            # Pre-populate states so they share the same pulse type/duration
            if hasattr(self, "parent_gui") and hasattr(self.parent_gui, "rf_designer"):
                current_state = self.parent_gui.rf_designer.get_state()
                
                # Excitation: 90 degrees, same type/duration
                exc_state = current_state.copy()
                exc_state["flip_angle"] = 90.0
                self.pulse_states["Excitation"] = exc_state
                
                # Refocusing: 180 degrees, same type/duration
                ref_state = current_state.copy()
                ref_state["flip_angle"] = 180.0
                self.pulse_states["Refocusing"] = ref_state

                # Pre-generate waveforms for both roles so they are available immediately
                # Temporarily switch roles to force the designer to generate and save each pulse
                for role in roles:
                    self.current_role = role
                    self.parent_gui.rf_designer.set_state(self.pulse_states[role])

        elif seq_type == "Inversion Recovery":
            roles = ["Inversion", "Excitation"]
        elif seq_type in ("Gradient Echo", "Free Induction Decay", "FLASH", "EPI"):
            roles = ["Excitation"]
        elif seq_type == "Custom":
            roles = ["Custom Pulse"]
        else:
            roles = ["Pulse"]

        for role in roles:
            self.pulse_list.addItem(role)
            
        # Select first item by default
        if self.pulse_list.count() > 0:
            self.pulse_list.setCurrentRow(0)
            self.current_role = roles[0]
            
        self.pulse_list.blockSignals(False)
        
        # Trigger state load for the new selection
        # We manually call the handler because we blocked signals to avoid partial updates
        self._on_pulse_selection_changed(self.pulse_list.currentItem(), None)


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
                "te_ms": 3,
                "tr_ms": 10,
                "num_positions": 1,
                "num_frequencies": 201,
                "frequency_range_hz": 100,
                "pulse_type": "gaussian",
                "duration": 2.0,
            },
            "Spin Echo": {
                "te_ms": 5,
                "tr_ms": 20,
                "num_positions": 1,
                "num_frequencies": 201,
                "frequency_range_hz": 100,
                "duration": 1.0, # ms
            },
            "Spin Echo (Tip-axis 180)": {
                "te_ms": 5,
                "tr_ms": 20,
                "num_positions": 1,
                "num_frequencies": 201,
                "frequency_range_hz": 100,
                "duration": 1.0, # ms
            },
            "Gradient Echo": {
                "te_ms": 5,
                "tr_ms": 30,
                "flip_angle": 30,
                "duration": 1.0, # ms
            },
            "Slice Select + Rephase": {
                "te_ms": 10,
                "tr_ms": 100,
                "num_positions": 99,
                "num_frequencies": 3,
                "duration": 1.0, # ms
                "flip_angle": 90,
            },
            "SSFP (Loop)": {
                "te_ms": 2,
                "tr_ms": 5,
                "flip_angle": 30,
                "ssfp_repeats": 100,
                "ssfp_amp": 0.05,
                "ssfp_phase": 0.0,
                "ssfp_dur": 1.0,
                "ssfp_start_delay": 0.0,
                "ssfp_start_amp": 0.025,
                "ssfp_start_phase": 0.0,
                "ssfp_alternate_phase": True,
                "pulse_type": "gaussian",
                "num_positions": 1,
                "num_frequencies": 101,
                "frequency_range_hz": 1000,
                "duration": 1.0, # ms
                "time_step": 10.0,
            },
            "Inversion Recovery": {
                "te_ms": 20,
                "tr_ms": 100,
                "ti_ms": 40,
                "num_positions": 1,
                "num_frequencies": 51,
                "duration": 1.0, # ms
                "flip_angle": 90,
            },
            "FLASH": {
                "te_ms": 3,
                "tr_ms": 10,
                "flip_angle": 15,
                "duration": 1.0, # ms
            },
            "EPI": {
                "te_ms": 30,
                "tr_ms": 100,
                "num_positions": 51,
                "num_frequencies": 3,
                "duration": 1.0, # ms
            },
            "Custom": {
                "te_ms": 10,
                "tr_ms": 100,
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
            # Get RF frequency offset from RF designer
            rf_freq_offset = self.parent_gui.rf_designer.freq_offset.value() if hasattr(self, 'parent_gui') and hasattr(self.parent_gui, 'rf_designer') else 0.0
            
            # Retrieve pulses from waveforms
            exc = self.pulse_waveforms.get("Excitation", custom_pulse)
            ref = self.pulse_waveforms.get("Refocusing")
            
            return SpinEcho(
                te=te,
                tr=tr,
                custom_excitation=exc,
                custom_refocusing=ref,
                slice_thickness=self._slice_thickness_m(),
                slice_gradient_override=self._slice_gradient_override(),
                echo_count=self.spin_echo_echoes.value(),
                rf_freq_offset=rf_freq_offset,
            )
        elif seq_type == "Spin Echo (Tip-axis 180)":
            exc = self.pulse_waveforms.get("Excitation", custom_pulse)
            ref = self.pulse_waveforms.get("Refocusing")
            return SpinEchoTipAxis(
                te=te,
                tr=tr,
                custom_excitation=exc,
                custom_refocusing=ref,
                slice_thickness=self._slice_thickness_m(),
                slice_gradient_override=self._slice_gradient_override(),
                echo_count=self.spin_echo_echoes.value(),
            )
        elif seq_type == "Gradient Echo":
            # Get RF frequency offset from RF designer
            rf_freq_offset = self.parent_gui.rf_designer.freq_offset.value() if hasattr(self, 'parent_gui') and hasattr(self.parent_gui, 'rf_designer') else 0.0
            return GradientEcho(
                te=te,
                tr=tr,
                flip_angle=30,
                custom_excitation=custom_pulse,
                slice_thickness=self._slice_thickness_m(),
                slice_gradient_override=self._slice_gradient_override(),
                rf_freq_offset=rf_freq_offset,
            )
        elif seq_type == "Slice Select + Rephase":
            # Use a shorter rephase duration but preserve half-area rewind
            rephase_dur = max(0.2e-3, min(1.0e-3, te / 2))
            return SliceSelectRephase(
                flip_angle=90,
                pulse_duration=3e-3,
                time_bw_product=self._effective_tbw(),
                rephase_duration=rephase_dur,
                slice_thickness=self._slice_thickness_m(),
                slice_gradient_override=self._slice_gradient_override(),
                custom_pulse=custom_pulse,
            )
        else:
            # Return a simple FID using the current RF designer pulse (resampled to dt)
            dt = max(self.default_dt, 1e-6)
            total_duration = max(te, 0.01)  # cover at least 10 ms or TE
            # Use designer pulse if available; otherwise synthesize a calibrated rect
            pulse = None
            if hasattr(self, "parent_gui") and self.parent_gui is not None:
                pulse = self.parent_gui.rf_designer.get_pulse()
            if pulse is not None and len(pulse) == 2 and pulse[0] is not None:
                b1_wave, t_wave = pulse
                b1_wave = np.asarray(b1_wave, dtype=complex)
                t_wave = np.asarray(t_wave, dtype=float)
                if b1_wave.size < 2 or t_wave.size < 2:
                    b1_wave = np.array([0.0], dtype=complex)
                    t_wave = np.array([0.0], dtype=float)
                wave_duration = float(t_wave[-1] - t_wave[0] + (t_wave[1] - t_wave[0]) if len(t_wave) > 1 else dt)
                n_wave = max(1, int(np.ceil(wave_duration / dt)))
                t_resample = np.arange(0, n_wave) * dt
                real_part = np.interp(t_resample, t_wave - t_wave[0], np.real(b1_wave))
                imag_part = np.interp(t_resample, t_wave - t_wave[0], np.imag(b1_wave))
                b1_exc = real_part + 1j * imag_part
            else:
                exc_duration = 1e-3
                n_exc = max(int(np.ceil(exc_duration / dt)), 16)
                flip = self.parent_gui.rf_designer.flip_angle.value() if hasattr(self, "parent_gui") and self.parent_gui is not None else 90.0
                b1_exc, _ = design_rf_pulse("rect", duration=n_exc * dt, flip_angle=flip, npoints=n_exc)

            ntime = max(len(b1_exc), int(np.ceil(total_duration / dt)))
            ntime = min(max(ntime, 1000), 20000)  # keep reasonable bounds
            b1 = np.zeros(ntime, dtype=complex)
            gradients = np.zeros((ntime, 3))
            b1[:min(len(b1_exc), ntime)] = b1_exc[:min(len(b1_exc), ntime)]
            time = np.arange(ntime) * dt
            return (b1, gradients, time)

    def compile_sequence(self, custom_pulse=None, dt: float = None, log_info: bool = False):
        """Return explicit (b1, gradients, time) arrays for the current sequence."""
        dt = dt or self.default_dt
        seq_type = self.sequence_type.currentText()
        if seq_type == "EPI":
            return self._build_epi(custom_pulse, dt)
        if seq_type == "Inversion Recovery":
            return self._build_ir(custom_pulse, dt)
        if seq_type == "SSFP (Loop)":
            return self._build_ssfp(custom_pulse, dt)
        if seq_type == "Slice Select + Rephase":
            return self._build_slice_select_rephase(custom_pulse, dt, log_info=log_info)
        seq = self.get_sequence(custom_pulse=custom_pulse)
        if isinstance(seq, PulseSequence):
            b1, gradients, time = seq.compile(dt=dt)
            # Scale slice gradients using effective TBW if user has not overridden Gz
            if self._slice_gradient_override() is None and seq_type in ("Spin Echo", "Spin Echo (Tip-axis 180)", "Gradient Echo"):
                scale = self._effective_tbw() / 4.0
                gradients = np.array(gradients, copy=True)
                gradients[:, 2] *= scale
            return b1, gradients, time
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
        tbw = self._effective_tbw()
        bw_hz = tbw / max(exc_duration, dt)
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
        tbw = self._effective_tbw()
        bw_hz = tbw / max(len(inv_b1) * dt, dt)
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
        bw_hz_exc = tbw / max(n_exc * dt, dt)
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

    def _build_slice_select_rephase(self, custom_pulse, dt, log_info=False):
        dt = max(dt, 1e-6)
        te = self.te_spin.value() / 1000.0
        
        # Excitation (use provided custom pulse if available)
        if custom_pulse is not None:
            exc_b1, _ = custom_pulse
            exc_b1 = np.asarray(exc_b1, dtype=complex)
        else:
            exc_duration = 3e-3
            n_exc = max(int(np.ceil(exc_duration / dt)), 16)
            exc_b1, _ = design_rf_pulse("sinc", duration=n_exc * dt, flip_angle=90, npoints=n_exc)
        
        exc_pts = len(exc_b1)
        exc_duration = exc_pts * dt
        
        # Slice gradient
        thickness_cm = self._slice_thickness_m() * 100.0
        gamma_hz_per_g = 4258.0
        tbw = self._effective_tbw()
        bw_hz = tbw / max(exc_duration, dt)
        slice_g = self._slice_gradient_override() or (bw_hz / (gamma_hz_per_g * thickness_cm))
        
        # Rephase parameters
        rephase_pct = self.rephase_percentage.value() / 100.0
        slice_area = slice_g * exc_duration
        rephase_area = -slice_area * rephase_pct
        
        if log_info and hasattr(self, "parent_gui") and self.parent_gui:
            self.parent_gui.log_message(f"Slice Select + Rephase Info:")
            self.parent_gui.log_message(f"  Slice Gradient: {slice_g:.4f} G/cm")
            self.parent_gui.log_message(f"  Pulse Duration: {exc_duration*1000:.3f} ms")
            self.parent_gui.log_message(f"  Slice Area: {slice_area:.6e} G*s")
            self.parent_gui.log_message(f"  Rephase Target: {rephase_pct*100:.1f}%")
            self.parent_gui.log_message(f"  Rephase Area: {rephase_area:.6e} G*s")

        # Timing
        slice_gap_pts = max(int(np.ceil(0.05e-3 / dt)), 1)
        rephase_dur = max(0.2e-3, min(1.0e-3, te / 2))
        rephase_pts = max(int(np.ceil(rephase_dur / dt)), 2)
        rephase_amp = rephase_area / (rephase_pts * dt)
        
        # Total duration
        total_dur = max(te, (exc_pts + slice_gap_pts + rephase_pts) * dt + 0.001)
        npoints = int(np.ceil(total_dur / dt))
        
        b1 = np.zeros(npoints, dtype=complex)
        gradients = np.zeros((npoints, 3))
        time = np.arange(npoints) * dt
        
        # Pulse + Slice Gradient
        n_exc_safe = min(exc_pts, npoints)
        b1[:n_exc_safe] = exc_b1[:n_exc_safe]
        gradients[:n_exc_safe, 2] = slice_g
        
        # Rephase Gradient
        start_rephase = exc_pts + slice_gap_pts
        end_rephase = start_rephase + rephase_pts
        if end_rephase <= npoints:
            gradients[start_rephase:end_rephase, 2] = rephase_amp
            
        return b1, gradients, time

    def set_time_step(self, dt_s: float):
        """Set default time step for fallback/simple sequences."""
        if dt_s and dt_s > 0:
            self.default_dt = dt_s
            self.update_diagram()

    def _on_pulse_selection_changed(self, current, previous):
        """Handle switching between pulses in the list."""
        if not hasattr(self, "parent_gui") or not hasattr(self.parent_gui, "rf_designer"):
            return

        # Save previous state
        if previous:
            prev_role = previous.text()
            self.pulse_states[prev_role] = self.parent_gui.rf_designer.get_state()
        
        # Load new state
        if current:
            curr_role = current.text()
            self.current_role = curr_role
            
            if curr_role in self.pulse_states:
                self.parent_gui.rf_designer.set_state(self.pulse_states[curr_role])
            else:
                # Apply defaults for new roles
                defaults = {}
                if curr_role in ("Refocusing", "Inversion"):
                    defaults = {"flip_angle": 180.0, "duration": 2.0}
                elif curr_role == "Excitation":
                    defaults = {"flip_angle": 90.0, "duration": 1.0}
                
                if defaults:
                    self.parent_gui.rf_designer.set_state(defaults)

    def set_custom_pulse(self, pulse):
        """Store custom pulse for preview (used when sequence type is Custom)."""
        if self.current_role:
            self.pulse_waveforms[self.current_role] = pulse
            
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


class UniversalTimeControl(QGroupBox):
    """Universal time control widget that synchronizes all time-resolved views."""
    time_changed = pyqtSignal(int)  # Emits time index

    def __init__(self):
        super().__init__("Playback Control")
        self._updating = False  # Prevent circular updates
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Time:"))
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.time_slider, 1)

        self.time_label = QLabel("0.0 ms")
        self.time_label.setFixedWidth(90)
        layout.addWidget(self.time_label)

        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.setCheckable(True)
        self.play_pause_button.toggled.connect(self._update_play_pause_label)
        layout.addWidget(self.play_pause_button)

        self.reset_button = QPushButton("Reset")
        layout.addWidget(self.reset_button)

        layout.addWidget(QLabel("Speed (ms/s):"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.001, 1000.0)
        self.speed_spin.setValue(1.0)  # Default to 50 ms of sim per real second
        self.speed_spin.setSuffix(" ms/s")
        self.speed_spin.setSingleStep(0.1)
        layout.addWidget(self.speed_spin)

        # Backwards compatibility for existing signal connections
        self.play_button = self.play_pause_button
        self.pause_button = self.play_pause_button

        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

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

    def _update_play_pause_label(self, is_playing: bool):
        """Keep play/pause button text in sync with state."""
        self.play_pause_button.setText("Pause" if is_playing else "Play")

    def sync_play_state(self, is_playing: bool):
        """Update play toggle without emitting signals."""
        blocked = self.play_pause_button.blockSignals(True)
        self.play_pause_button.setChecked(is_playing)
        self._update_play_pause_label(is_playing)
        self.play_pause_button.blockSignals(blocked)


class MagnetizationViewer(QWidget):
    """3D visualization of magnetization vector."""
    position_changed = pyqtSignal(int)
    view_filter_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._export_dir_provider = None
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
        export_3d_menu.addAction("Sequence diagram (GIF/MP4)...", self._export_sequence_animation)
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
        self.vector_plot = gl.GLLinePlotItem(pos=np.zeros((0, 3)), color=(1, 0, 0, 1), width=3, mode='lines')
        self.gl_widget.addItem(self.vector_plot)
        self.vector_colors = None
        
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

        # Control buttons and view mode selectors
        control_container = QWidget()
        controls_v = QVBoxLayout()
        controls_v.setContentsMargins(0, 0, 0, 0)
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

        self.last_positions = None
        self.last_frequencies = None
        # Track available position/frequency counts for selector range updates
        self._npos = 1
        self._nfreq = 1
        self._update_selector_range()
        
        self.setLayout(layout)
        
    def _ensure_vectors(self, count: int, colors=None):
        """Cache colors for vector updates."""
        if colors is not None:
            self.vector_colors = colors

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
        
        # Construct interleaved array for single draw call: (Origin, Tip, Origin, Tip...)
        pos = np.zeros((count * 2, 3), dtype=np.float32)
        pos[1::2] = vecs_scaled
        
        # Handle colors
        use_colors = self.vector_colors if self.vector_colors is not None and len(self.vector_colors) >= count else None
        if use_colors is not None:
            # Repeat colors for each vertex (2 per line)
            c_arr = np.asarray(use_colors)
            if c_arr.ndim == 1 and c_arr.shape[0] == 4:
                self.vector_plot.setData(pos=pos, color=c_arr, mode='lines')
            else:
                c_expanded = np.repeat(c_arr[:count], 2, axis=0)
                self.vector_plot.setData(pos=pos, color=c_expanded, mode='lines')
        else:
            self.vector_plot.setData(pos=pos, color=(1, 0, 0, 1), mode='lines')
            
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
        # Time cursor lines removed for performance
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
            idx = min(self.selector_slider.value(), max_idx)
            freq_val = self.last_frequencies[idx] if self.last_frequencies is not None and idx < len(self.last_frequencies) else idx
            label_text = f"Freq: {freq_val:.1f} Hz"
        elif mode == "Freqs @ position":
            max_idx = max(0, self._npos - 1)
            idx = min(self.selector_slider.value(), max_idx)
            pos_val = self.last_positions[idx, 2] * 100 if self.last_positions is not None and idx < len(self.last_positions) else idx
            label_text = f"Pos: {pos_val:.2f} cm"
        else:
            max_idx = 0
            label_text = "All spins"

        self.selector_slider.blockSignals(True)
        self.selector_slider.setMaximum(max_idx)
        self.selector_slider.setValue(min(self.selector_slider.value(), max_idx) if max_idx > 0 else 0)
        self.selector_slider.setVisible(max_idx > 0)
        self.selector_slider.blockSignals(False)
        self.selector_label.setText(label_text)

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

        export_dir = self._resolve_export_directory()
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
    def _resolve_export_directory(self) -> Path:
        """Resolve an export directory via provider, window hook, or cwd."""
        # External provider takes precedence
        if callable(self._export_dir_provider):
            try:
                path = Path(self._export_dir_provider())
                path.mkdir(parents=True, exist_ok=True)
                return path
            except Exception:
                pass
        # Ask the top-level window if it exposes an export directory helper
        win = self.window()
        if win and hasattr(win, "_get_export_directory"):
            try:
                path = Path(win._get_export_directory())
                path.mkdir(parents=True, exist_ok=True)
                return path
            except Exception:
                pass
        path = Path.cwd()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def set_export_directory_provider(self, provider):
        """Provide a callable returning a directory for exports."""
        self._export_dir_provider = provider

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

    def _export_sequence_animation(self):
        """Delegate sequence diagram animation export to parent window."""
        win = self.window()
        if win and hasattr(win, "_export_sequence_diagram_animation"):
            try:
                win._export_sequence_diagram_animation()
                return
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", str(exc))
                return
        self._show_not_implemented_3d("Sequence animation")


class ParameterSweepWidget(QWidget):
    """Widget for running parameter sweeps (multiple simulations with varying parameters)."""

    sweep_finished = pyqtSignal(dict)  # Emits results when sweep completes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.sweep_running = False
        self.sweep_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Parameter Sweep")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title_label)

        # Info label
        info_label = QLabel("Run multiple simulations by sweeping a parameter over a range.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Parameter selection
        param_group = QGroupBox("Sweep Parameter")
        param_layout = QVBoxLayout()

        param_sel_layout = QHBoxLayout()
        param_sel_layout.addWidget(QLabel("Parameter:"))
        self.param_combo = QComboBox()
        self.param_combo.addItems([
            "Flip Angle (deg)",
            "TE (ms)",
            "TR (ms)",
            "TI (ms)",
            "B1 Scale Factor",
            "T1 (ms)",
            "T2 (ms)",
            "Frequency Offset (Hz)"
        ])
        self.param_combo.currentTextChanged.connect(self._update_range_limits)
        param_sel_layout.addWidget(self.param_combo)
        param_layout.addLayout(param_sel_layout)

        # Range controls
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Start:"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(-1e6, 1e6)
        self.start_spin.setValue(30)
        self.start_spin.setDecimals(2)
        range_layout.addWidget(self.start_spin)

        range_layout.addWidget(QLabel("End:"))
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(-1e6, 1e6)
        self.end_spin.setValue(90)
        self.end_spin.setDecimals(2)
        range_layout.addWidget(self.end_spin)

        range_layout.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(2, 500)
        self.steps_spin.setValue(13)
        range_layout.addWidget(self.steps_spin)
        param_layout.addLayout(range_layout)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Output metric selection
        metric_group = QGroupBox("Output Metrics to Track")
        metric_layout = QVBoxLayout()
        self.metric_checkboxes = {}
        for metric in [
            "Final Mz (mean)",
            "Final Mxy (mean)",
            "Peak |Mxy|",
            "Signal Magnitude",
            "Signal Phase",
            "Final Mxy map (complex) (per pos/freq)",
            "Final Mz map (per pos/freq)",
        ]:
            cb = QCheckBox(metric)
            cb.setChecked(metric in ["Final Mz (mean)", "Signal Magnitude"])
            self.metric_checkboxes[metric] = cb
            metric_layout.addWidget(cb)
        metric_group.setLayout(metric_layout)
        layout.addWidget(metric_group)

        # Control buttons
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Sweep")
        self.run_button.clicked.connect(self.run_sweep)
        button_layout.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_sweep)
        button_layout.addWidget(self.stop_button)

        self.export_button = QPushButton("Export Results")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results)
        button_layout.addWidget(self.export_button)

        layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Results plot
        self.results_plot = pg.PlotWidget()
        self.results_plot.setLabel('left', 'Metric Value')
        self.results_plot.setLabel('bottom', 'Parameter Value')
        self.results_plot.setMinimumHeight(300)
        self.results_plot.addLegend()
        layout.addWidget(self.results_plot)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setMaximumHeight(200)
        layout.addWidget(self.results_table)

        layout.addStretch()
        self.setLayout(layout)

        # Initialize range limits
        self._update_range_limits()

        # Storage for results
        self.last_sweep_results = None

    def _update_range_limits(self):
        """Update spin box ranges based on selected parameter."""
        param = self.param_combo.currentText()

        if "Flip Angle" in param:
            self.start_spin.setRange(0, 9998)
            self.start_spin.setValue(30)
            self.end_spin.setRange(0, 9999)
            self.end_spin.setValue(90)
        elif "TE" in param or "TR" in param or "TI" in param:
            self.start_spin.setRange(0.1, 10000)
            self.start_spin.setValue(10)
            self.end_spin.setRange(0.1, 10000)
            self.end_spin.setValue(100)
        elif "B1 Scale" in param:
            self.start_spin.setRange(0, 5)
            self.start_spin.setValue(0.5)
            self.start_spin.setDecimals(3)
            self.end_spin.setRange(0, 5)
            self.end_spin.setValue(1.5)
            self.end_spin.setDecimals(3)
        elif "T1" in param:
            self.start_spin.setRange(1, 5000)
            self.start_spin.setValue(500)
            self.end_spin.setRange(1, 5000)
            self.end_spin.setValue(2000)
        elif "T2" in param:
            self.start_spin.setRange(1, 2000)
            self.start_spin.setValue(20)
            self.end_spin.setRange(1, 2000)
            self.end_spin.setValue(100)
        elif "Frequency" in param:
            self.start_spin.setRange(-10000, 10000)
            self.start_spin.setValue(-500)
            self.end_spin.setRange(-10000, 10000)
            self.end_spin.setValue(500)

    def run_sweep(self):
        """Run parameter sweep."""
        if not self.parent_gui:
            QMessageBox.warning(self, "Error", "No parent GUI available.")
            return

        if hasattr(self.parent_gui, "set_sweep_mode"):
            self.parent_gui.set_sweep_mode(True)

        self.sweep_running = True
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)

        # Get sweep parameters
        param_name = self.param_combo.currentText()
        start_val = self.start_spin.value()
        end_val = self.end_spin.value()
        n_steps = self.steps_spin.value()

        # Create parameter array
        param_values = np.linspace(start_val, end_val, n_steps)

        # Get selected metrics
        selected_metrics = [name for name, cb in self.metric_checkboxes.items() if cb.isChecked()]
        if not selected_metrics:
            QMessageBox.warning(self, "No Metrics", "Please select at least one output metric.")
            if hasattr(self.parent_gui, "set_sweep_mode"):
                self.parent_gui.set_sweep_mode(False)
            self.sweep_running = False
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            return

        # Run sweep
        results = {
            'parameter_name': param_name,
            'parameter_values': param_values,
            'metrics': {metric: [] for metric in selected_metrics}
        }

        # Store initial values for parameters that need them (like B1 scale)
        initial_flip_angle = self.parent_gui.rf_designer.flip_angle.value()

        try:
            for i, param_val in enumerate(param_values):
                if not self.sweep_running:
                    break

                # Update parameter
                self._apply_parameter_value(param_name, param_val, initial_flip_angle)

                # Run simulation
                try:
                    self.parent_gui.run_simulation()

                    # Wait for simulation to complete (simple polling)
                    while self.parent_gui.simulation_thread and self.parent_gui.simulation_thread.isRunning():
                        QApplication.processEvents()
                        time.sleep(0.1)

                    # Extract metrics from results
                    if self.parent_gui.last_result:
                        for metric in selected_metrics:
                            value = self._extract_metric(metric, self.parent_gui.last_result)
                            results['metrics'][metric].append(value)
                    else:
                        # No result - append NaN to maintain array length
                        for metric in selected_metrics:
                            results['metrics'][metric].append(float('nan'))
                        self.parent_gui.log_message(f"Warning: No result for {param_name}={param_val:.2f}")
                except Exception as e:
                    import traceback
                    error_msg = f"Error at {param_name}={param_val:.2f}: {str(e)}\n{traceback.format_exc()}"
                    self.parent_gui.log_message(error_msg)
                    QMessageBox.warning(self, "Simulation Error", f"Error at {param_name}={param_val:.2f}: {str(e)}")
                    break

                # Update progress
                self.progress_bar.setValue(int((i + 1) / n_steps * 100))

            # Store and display results
            self.last_sweep_results = results
            self._display_results(results)
        finally:
            self.sweep_running = False
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.export_button.setEnabled(True)
            if hasattr(self.parent_gui, "set_sweep_mode"):
                self.parent_gui.set_sweep_mode(False)

    def stop_sweep(self):
        """Stop the parameter sweep."""
        self.sweep_running = False
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if hasattr(self.parent_gui, "set_sweep_mode"):
            self.parent_gui.set_sweep_mode(False)

    def _apply_parameter_value(self, param_name, value, initial_flip_angle=None):
        """Apply parameter value to the GUI controls."""
        if "Flip Angle" in param_name:
            self.parent_gui.rf_designer.flip_angle.setValue(value)
        elif "TE" in param_name:
            self.parent_gui.sequence_designer.te_spin.setValue(value)
        elif "TR" in param_name:
            self.parent_gui.sequence_designer.tr_spin.setValue(value)
        elif "TI" in param_name:
            self.parent_gui.sequence_designer.ti_spin.setValue(value)
        elif "B1 Scale" in param_name:
            # Scale the initial flip angle (not current, to avoid cumulative scaling)
            if initial_flip_angle is not None:
                self.parent_gui.rf_designer.flip_angle.setValue(initial_flip_angle * value)
            else:
                # Fallback if not provided
                self.parent_gui.rf_designer.flip_angle.setValue(90 * value)
        elif "T1" in param_name:
            self.parent_gui.tissue_widget.t1_spin.setValue(value)
        elif "T2" in param_name:
            self.parent_gui.tissue_widget.t2_spin.setValue(value)
        elif "Frequency" in param_name:
            # Set single frequency offset
            self.parent_gui.freq_range.setValue(0)
            self.parent_gui.freq_center.setValue(value)

    def _extract_metric(self, metric_name, result):
        """Extract metric value from simulation result."""
        try:
            if "Final Mz (mean)" in metric_name:
                mz = result.get('mz')
                if mz is not None:
                    # Handle different array shapes
                    if mz.ndim == 3:  # (ntime, npos, nfreq)
                        return float(np.mean(mz[-1, :, :]))
                    elif mz.ndim == 2:  # (npos, nfreq) - endpoint mode
                        return float(np.mean(mz))
                    else:  # 1D or scalar
                        return float(np.mean(mz))
            elif "Final Mxy (mean)" in metric_name:
                mx = result.get('mx')
                my = result.get('my')
                if mx is not None and my is not None:
                    if mx.ndim == 3:  # Time-resolved
                        mx_final = mx[-1, :, :]
                        my_final = my[-1, :, :]
                    else:  # Endpoint mode
                        mx_final = mx
                        my_final = my
                    mxy = np.sqrt(mx_final**2 + my_final**2)
                    return float(np.mean(mxy))
            elif "Peak |Mxy|" in metric_name:
                mx = result.get('mx')
                my = result.get('my')
                if mx is not None and my is not None:
                    mxy = np.sqrt(mx**2 + my**2)
                    return float(np.max(mxy))
            elif "Signal Magnitude" in metric_name:
                signal = result.get('signal')
                if signal is not None:
                    return float(np.mean(np.abs(signal)))
            elif "Signal Phase" in metric_name:
                signal = result.get('signal')
                if signal is not None:
                    # Use angle in degrees for better readability
                    return float(np.mean(np.angle(signal, deg=True)))
            elif "Final Mxy map" in metric_name:
                mx = result.get('mx')
                my = result.get('my')
                if mx is not None and my is not None:
                    if mx.ndim == 3:
                        mx_final = mx[-1, :, :]
                        my_final = my[-1, :, :]
                    else:
                        mx_final = mx
                        my_final = my
                    return mx_final + 1j * my_final
            elif "Final Mz map" in metric_name:
                mz = result.get('mz')
                if mz is not None:
                    if mz.ndim == 3:
                        return mz[-1, :, :]
                    return mz
        except Exception as e:
            if self.parent_gui:
                self.parent_gui.log_message(f"Error extracting metric '{metric_name}': {str(e)}")
            return float('nan')
        return 0.0

    def _display_results(self, results):
        """Display sweep results in plot and table."""
        self.results_plot.clear()

        param_values = results['parameter_values']
        param_name = results['parameter_name']

        # Plot each metric
        colors = ['r', 'g', 'b', 'y', 'm']
        for i, (metric, values) in enumerate(results['metrics'].items()):
            if len(values) != len(param_values):
                continue
            # Only plot scalar metrics
            scalar_vals = []
            all_scalar = True
            for v in values:
                try:
                    scalar_vals.append(float(v))
                except Exception:
                    all_scalar = False
                    break
            if not all_scalar:
                continue
            color = colors[i % len(colors)]
            self.results_plot.plot(param_values, scalar_vals, pen=pg.mkPen(color, width=2),
                                    symbol='o', symbolBrush=color, name=metric)

        self.results_plot.setLabel('bottom', param_name)

        # Update table
        metrics = list(results['metrics'].keys())
        self.results_table.setRowCount(len(param_values))
        self.results_table.setColumnCount(1 + len(metrics))
        self.results_table.setHorizontalHeaderLabels([param_name] + metrics)

        for row, param_val in enumerate(param_values):
            self.results_table.setItem(row, 0, QTableWidgetItem(f"{param_val:.3f}"))
            for col, metric in enumerate(metrics):
                if row < len(results['metrics'][metric]):
                    value = results['metrics'][metric][row]
                    if isinstance(value, np.ndarray):
                        display_text = f"array{value.shape}"
                    else:
                        try:
                            display_text = f"{float(value):.6f}"
                        except Exception:
                            display_text = str(value)
                    self.results_table.setItem(row, col + 1, QTableWidgetItem(display_text))

        self.results_table.resizeColumnsToContents()

    def export_results(self):
        """Export sweep results to CSV/NPZ/NPY."""
        if not self.last_sweep_results:
            QMessageBox.warning(self, "No Results", "No sweep results to export.")
            return

        export_dir = self._default_export_dir()
        default_path = export_dir / "sweep_results.csv"

        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Sweep Results",
            str(default_path),
            "CSV (*.csv);;NumPy Archive (*.npz);;NumPy Binary (*.npy);;All Files (*)"
        )
        if not filename:
            return

        path = Path(filename)
        ext = path.suffix.lower()
        # Infer from selected filter if no extension was provided
        if ext == "":
            if selected_filter and "npz" in selected_filter.lower():
                path = path.with_suffix(".npz")
                ext = ".npz"
            elif selected_filter and "npy" in selected_filter.lower():
                path = path.with_suffix(".npy")
                ext = ".npy"
            else:
                path = path.with_suffix(".csv")
                ext = ".csv"

        if ext == ".csv":
            array_path = self._save_sweep_results_csv(path)
            extra = f"\nArray metrics exported to:\n{array_path}" if array_path else ""
            QMessageBox.information(self, "Export Complete", f"Results exported to:\n{path}{extra}")
        elif ext == ".npz":
            self._save_sweep_results_npz(path)
            QMessageBox.information(self, "Export Complete", f"Results exported to:\n{path}")
        elif ext == ".npy":
            self._save_sweep_results_npy(path)
            QMessageBox.information(self, "Export Complete", f"Results exported to:\n{path}")
        else:
            QMessageBox.warning(self, "Unsupported format", f"Extension '{ext}' is not supported.")

    def _default_export_dir(self) -> Path:
        """Resolve a writable export directory for sweep results."""
        if getattr(self, "parent_gui", None) and hasattr(self.parent_gui, "_get_export_directory"):
            try:
                return Path(self.parent_gui._get_export_directory())
            except Exception:
                pass
        return Path.cwd()

    def _save_sweep_results_csv(self, path: Path) -> Optional[Path]:
        """Save sweep results to CSV and return any auxiliary array path."""
        results = self.last_sweep_results
        param_name = results['parameter_name']
        metrics = list(results['metrics'].keys())
        array_metrics = {m: [] for m in metrics}

        with open(path, 'w') as f:
            f.write(f"{param_name}," + ",".join(metrics) + "\n")

            for i, param_val in enumerate(results['parameter_values']):
                row = [f"{param_val:.6f}"]
                for metric in metrics:
                    if i < len(results['metrics'][metric]):
                        value = results['metrics'][metric][i]
                        if isinstance(value, np.ndarray):
                            row.append(f"array{value.shape}")
                            array_metrics[metric].append(value)
                        else:
                            try:
                                row.append(f"{float(value):.6f}")
                            except Exception:
                                row.append(str(value))
                    else:
                        row.append("")
                f.write(",".join(row) + "\n")

        stacked_arrays = {}
        for metric, vals in array_metrics.items():
            if not vals:
                continue
            try:
                stacked_arrays[metric] = np.stack(vals)
            except Exception:
                stacked_arrays[metric] = np.array(vals, dtype=object)

        if stacked_arrays:
            array_path = path.with_name(path.stem + "_arrays.npz")
            np.savez(
                array_path,
                parameter_name=results['parameter_name'],
                parameter_values=np.asarray(results['parameter_values']),
                **stacked_arrays,
            )
            return array_path
        return None

    def _save_sweep_results_npz(self, path: Path):
        """Save sweep results into a single NPZ archive."""
        results = self.last_sweep_results
        payload = {
            "parameter_name": results['parameter_name'],
            "parameter_values": np.asarray(results['parameter_values']),
        }
        for metric, values in results['metrics'].items():
            payload[metric] = self._stack_metric_values(values)
        np.savez(path, **payload)

    def _save_sweep_results_npy(self, path: Path):
        """Save sweep results as a NumPy binary with a dictionary payload."""
        results = self.last_sweep_results
        payload = {
            "parameter_name": results['parameter_name'],
            "parameter_values": np.asarray(results['parameter_values']),
            "metrics": {metric: self._stack_metric_values(vals) for metric, vals in results['metrics'].items()},
        }
        np.save(path, payload, allow_pickle=True)

    def _stack_metric_values(self, values):
        """Best-effort stacking for metric values with mixed scalar/array content."""
        try:
            return np.stack([np.asarray(v) for v in values])
        except Exception:
            try:
                return np.asarray(values, dtype=float)
            except Exception:
                return np.asarray(values, dtype=object)


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
        self._playback_anchor_wall = None  # monotonic timestamp for playback pacing
        self._playback_anchor_time_ms = None  # simulation time (ms) at anchor
        self._frame_step = 1
        self._min_anim_interval_ms = 1000.0 / 120.0  # cap display rate to avoid event loop overload
        self._target_render_fps = 60.0
        self._min_render_interval_ms = 1000.0 / self._target_render_fps
        self._last_render_wall = None
        self._suppress_heavy_updates = False
        self._heavy_update_every = 3  # update heavy plots every N frames during playback
        self._playback_frame_counter = 0
        self.anim_b1 = None
        self.anim_b1_scale = 1.0
        self.anim_vectors_full = None  # (ntime, npos, nfreq, 3) before flattening
        self.mxy_legend = None
        self.mz_legend = None
        self.signal_legend = None
        self.initial_mz = 1.0  # Track initial Mz to scale plot limits
        self._last_spatial_export = None
        self._last_spectrum_export = None
        self._spectrum_final_range = None
        self.dataset_exporter = DatasetExporter()
        self._sweep_mode = False
        self.spectrum_y_max = 1.1 # Constant maximum for spectrum Y-axis


        # Pre-calculated plot ranges for stability during animation
        self.spatial_mxy_yrange = None
        self.spatial_mz_yrange = None
        self.spectrum_yrange = None
        # Dirty flags for expensive computations during animation
        self._spectrum_needs_update = False
        self._spatial_needs_update = False

        # Cache for persistent plot items (avoids clear+replot cycle)
        self._mxy_plot_items = {}  # key: (pi, fi, component) -> PlotDataItem
        self._mz_plot_items = {}   # key: (pi, fi) -> PlotDataItem
        self._signal_plot_items = {} # key: (pi, fi, component) -> PlotDataItem
        self._plot_items_initialized = False

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Parameters
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(6, 6, 6, 18)
        left_layout.setSpacing(8)
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(480)
        left_panel.setMinimumWidth(340)
        left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        
        # Tissue parameters
        self.tissue_widget = TissueParameterWidget()
        left_layout.addWidget(self.tissue_widget)
        
        # RF Pulse designer
        self.rf_designer = RFPulseDesigner()
        left_layout.addWidget(self.rf_designer)
        
        # Sequence designer
        self.sequence_designer = SequenceDesigner()
        left_layout.addWidget(self.sequence_designer)
        self.sequence_designer.parent_gui = self
        self.rf_designer.pulse_changed.connect(self.sequence_designer.set_custom_pulse)
        self.rf_designer.pulse_changed.connect(lambda _: self._auto_update_ssfp_amplitude())
        self.sequence_designer.set_custom_pulse(self.rf_designer.get_pulse())
        # Connect sequence type changes to preset loader
        self.sequence_designer.sequence_type.currentTextChanged.connect(self._load_sequence_presets)
        self.sequence_designer.sequence_type.currentTextChanged.connect(lambda _: self._auto_update_ssfp_amplitude())
        self.sequence_designer.ssfp_dur.valueChanged.connect(lambda _: self._auto_update_ssfp_amplitude())
        self.rf_designer.flip_angle.valueChanged.connect(lambda _: self._auto_update_ssfp_amplitude())
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
        self.extra_tail_spin.setValue(5.0)
        self.extra_tail_spin.setDecimals(3)
        self.extra_tail_spin.setSingleStep(1.0)
        tail_layout.addWidget(self.extra_tail_spin)
        control_layout.addLayout(tail_layout)

        # Max traces control for performance
        max_traces_layout = QHBoxLayout()
        max_traces_layout.addWidget(QLabel("Max plot traces:"))
        self.max_traces_spin = QSpinBox()
        self.max_traces_spin.setRange(1, 500)
        self.max_traces_spin.setValue(50)
        self.max_traces_spin.setSingleStep(5)
        self.max_traces_spin.setToolTip("Maximum number of individual traces to plot (for performance)")
        max_traces_layout.addWidget(self.max_traces_spin)
        control_layout.addLayout(max_traces_layout)

        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        left_layout.addStretch()

        # Make the left panel scrollable so controls remain reachable on smaller screens
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_scroll.setMinimumHeight(0)
        left_scroll.setWidget(left_panel)
        # Ensure the scroll area knows the full height so bottom controls are reachable
        left_panel.adjustSize()
        left_panel.setMinimumHeight(left_panel.sizeHint().height() + 32)

        # Footer with sticky run controls so the button stays visible
        self.simulate_button = QPushButton("Run Simulation")
        self.simulate_button.clicked.connect(self.run_simulation)
        self.progress_bar = QProgressBar()
        footer = QWidget()
        footer_layout = QVBoxLayout()
        footer_layout.setContentsMargins(0, 4, 0, 0)
        footer_layout.setSpacing(6)
        footer_layout.addWidget(self.simulate_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_simulation)
        self.preview_checkbox = QCheckBox("Preview (fast subsample)")
        self.preview_checkbox.toggled.connect(lambda val: self._sync_preview_checkboxes(val))
        footer_layout.addWidget(self.cancel_button)
        footer_layout.addWidget(self.preview_checkbox)
        footer_layout.addWidget(self.progress_bar)
        footer.setLayout(footer_layout)
        footer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        footer.setMinimumHeight(footer.sizeHint().height())
        self.simulate_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cancel_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        footer.setMinimumHeight(
            self.simulate_button.sizeHint().height()
            + self.cancel_button.sizeHint().height()
            + self.preview_checkbox.sizeHint().height()
            + self.progress_bar.sizeHint().height()
            + 12
        )

        left_container = QWidget()
        left_container_layout = QVBoxLayout()
        left_container_layout.setContentsMargins(0, 0, 0, 0)
        left_container_layout.setSpacing(6)
        left_container_layout.addWidget(left_scroll)
        left_container_layout.addWidget(footer)
        left_container_layout.setStretch(0, 1)
        left_container.setLayout(left_container_layout)
        left_container.setMinimumWidth(left_panel.minimumWidth())
        left_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        # Right panel - Visualization + log
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # Shared heatmap colormap selector for all tabs
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Heatmap colormap:"))
        self.heatmap_colormap = QComboBox()
        self.heatmap_colormap.addItems(["viridis", "plasma", "magma", "cividis", "inferno", "gray"])
        self.heatmap_colormap.setCurrentText("viridis")
        self.heatmap_colormap.currentTextChanged.connect(self._apply_heatmap_colormap)
        colormap_layout.addWidget(self.heatmap_colormap)

        # Universal time control - controls all time-resolved views
        self.time_control = UniversalTimeControl()
        self.time_control.setEnabled(False)
        colormap_layout.addWidget(self.time_control, 1)
        right_layout.addLayout(colormap_layout)
        
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
        mag_export_menu.addAction("Export Traces (CSV/NPY)...", self._export_magnetization_data)
        mag_export_menu.addAction("Export Full Results...", self.export_results)
        mag_export_btn.setMenu(mag_export_menu)
        mag_header.addWidget(mag_export_btn)
        mag_layout.addLayout(mag_header)

        # Magnetization view filter controls (align with 3D view options)
        mag_view_layout = QHBoxLayout()

        # Add plot type selector (Line vs Heatmap) - default to Heatmap
        mag_view_layout.addWidget(QLabel("Plot type:"))
        self.mag_plot_type = QComboBox()
        self.mag_plot_type.addItems(["Heatmap", "Line"])
        self.mag_plot_type.currentTextChanged.connect(lambda _: self._refresh_mag_plots())
        mag_view_layout.addWidget(self.mag_plot_type)

        # Add component selector for heatmap
        mag_view_layout.addWidget(QLabel("Component:"))
        self.mag_component = QComboBox()
        self.mag_component.addItems(["Magnitude", "Real (Mx/Re)", "Imaginary (My/Im)", "Phase", "Mz"])
        self.mag_component.currentTextChanged.connect(lambda _: self._refresh_mag_plots())
        mag_view_layout.addWidget(self.mag_component)

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

        # Create both line plot and heatmap widgets, but only show one at a time
        self.mxy_plot = pg.PlotWidget()
        self.mxy_plot.setLabel('left', 'Mx / My')
        self.mxy_plot.setLabel('bottom', 'Time', 'ms')
        self.mxy_plot.setDownsampling(mode='peak')
        self.mxy_plot.setClipToView(True)
        self.mxy_plot.hide()  # Hide by default (heatmap is default)

        self.mz_plot = pg.PlotWidget()
        self.mz_plot.setLabel('left', 'Mz')
        self.mz_plot.setLabel('bottom', 'Time', 'ms')
        self.mz_plot.setDownsampling(mode='peak')
        self.mz_plot.setClipToView(True)
        self.mz_plot.hide()  # Hide by default (heatmap is default)

        # Create heatmap widgets using GraphicsLayoutWidget for proper colorbar alignment
        self.mxy_heatmap_layout = pg.GraphicsLayoutWidget()
        self.mxy_heatmap = self.mxy_heatmap_layout.addPlot(row=0, col=0)
        self.mxy_heatmap.setLabel('left', 'Spin Index')
        self.mxy_heatmap.setLabel('bottom', 'Time', 'ms')
        self.mxy_heatmap_item = pg.ImageItem()
        self.mxy_heatmap.addItem(self.mxy_heatmap_item)
        self.mxy_heatmap_colorbar = pg.ColorBarItem(values=(0, 1), interactive=False)
        self.mxy_heatmap_layout.addItem(self.mxy_heatmap_colorbar, row=0, col=1)
        self.mxy_heatmap_colorbar.setImageItem(self.mxy_heatmap_item)
        # Show by default (heatmap is default view)

        self.mz_heatmap_layout = pg.GraphicsLayoutWidget()
        self.mz_heatmap = self.mz_heatmap_layout.addPlot(row=0, col=0)
        self.mz_heatmap.setLabel('left', 'Spin Index')
        self.mz_heatmap.setLabel('bottom', 'Time', 'ms')
        self.mz_heatmap_item = pg.ImageItem()
        self.mz_heatmap.addItem(self.mz_heatmap_item)
        self.mz_heatmap_colorbar = pg.ColorBarItem(values=(0, 1), interactive=False)
        self.mz_heatmap_layout.addItem(self.mz_heatmap_colorbar, row=0, col=1)
        self.mz_heatmap_colorbar.setImageItem(self.mz_heatmap_item)
        # Show by default (heatmap is default view)

        # Allow resizing between stacked plots so lower plots stay visible
        mag_splitter = QSplitter(Qt.Vertical)
        mag_splitter.addWidget(self.mxy_plot)
        mag_splitter.addWidget(self.mz_plot)
        mag_splitter.addWidget(self.mxy_heatmap_layout)
        mag_splitter.addWidget(self.mz_heatmap_layout)
        mag_splitter.setStretchFactor(0, 1)
        mag_splitter.setStretchFactor(1, 1)
        mag_splitter.setStretchFactor(2, 1)
        mag_splitter.setStretchFactor(3, 1)
        mag_layout.addWidget(mag_splitter)

        # Disable autorange so manual ranges stick
        for plt in (self.mxy_plot, self.mz_plot, self.mxy_heatmap, self.mz_heatmap):
            plt.getViewBox().disableAutoRange()
        
        self.tab_widget.addTab(mag_widget, "Magnetization")

        # 3D visualization
        self.mag_3d = MagnetizationViewer()
        self.mag_3d.playhead_line = self.sequence_designer.playhead_line
        self.mag_3d.set_export_directory_provider(self._get_export_directory)
        self.mag_3d.position_changed.connect(lambda idx: self._set_animation_index_from_slider(idx, reset_anchor=True))
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

        # Add signal view controls - default to Heatmap
        signal_view_layout = QHBoxLayout()
        signal_view_layout.addWidget(QLabel("Plot type:"))
        self.signal_plot_type = QComboBox()
        self.signal_plot_type.addItems(["Heatmap", "Line"])
        self.signal_plot_type.currentTextChanged.connect(lambda _: self._refresh_signal_plots())
        signal_view_layout.addWidget(self.signal_plot_type)

        # Add component selector for signal heatmap
        signal_view_layout.addWidget(QLabel("Component:"))
        self.signal_component = QComboBox()
        self.signal_component.addItems(["Magnitude", "Real", "Imaginary", "Phase"])
        self.signal_component.currentTextChanged.connect(lambda _: self._refresh_signal_plots())
        signal_view_layout.addWidget(self.signal_component)

        signal_view_layout.addWidget(QLabel("View mode:"))
        self.signal_view_mode = QComboBox()
        self.signal_view_mode.addItems([
            "All positions x freqs",
            "Positions @ freq",
            "Freqs @ position"
        ])
        self.signal_view_mode.currentTextChanged.connect(lambda _: self._refresh_signal_plots())
        signal_view_layout.addWidget(self.signal_view_mode)
        self.signal_view_selector_label = QLabel("All spins")
        signal_view_layout.addWidget(self.signal_view_selector_label)
        self.signal_view_selector = QSlider(Qt.Horizontal)
        self.signal_view_selector.setRange(0, 0)
        self.signal_view_selector.setValue(0)
        self.signal_view_selector.valueChanged.connect(lambda _: self._refresh_signal_plots())
        signal_view_layout.addWidget(self.signal_view_selector)
        signal_layout.addLayout(signal_view_layout)

        # Create line plot
        self.signal_plot = pg.PlotWidget()
        self.signal_plot.setLabel('left', 'Signal')
        self.signal_plot.setLabel('bottom', 'Time', 'ms')
        self.signal_plot.setDownsampling(mode='peak')
        self.signal_plot.setClipToView(True)
        self.signal_plot.enableAutoRange(x=False, y=False)
        self.signal_plot.hide()  # Hide by default (heatmap is default)

        # Create heatmap using GraphicsLayoutWidget for proper colorbar alignment
        self.signal_heatmap_layout = pg.GraphicsLayoutWidget()
        self.signal_heatmap = self.signal_heatmap_layout.addPlot(row=0, col=0)
        self.signal_heatmap.setLabel('left', 'Spin Index')
        self.signal_heatmap.setLabel('bottom', 'Time', 'ms')
        self.signal_heatmap_item = pg.ImageItem()
        self.signal_heatmap.addItem(self.signal_heatmap_item)
        self.signal_heatmap_colorbar = pg.ColorBarItem(values=(0, 1), interactive=False)
        self.signal_heatmap_layout.addItem(self.signal_heatmap_colorbar, row=0, col=1)
        self.signal_heatmap_colorbar.setImageItem(self.signal_heatmap_item)
        # Show by default (heatmap is default view)
        self.signal_heatmap.getViewBox().disableAutoRange()

        signal_splitter = QSplitter(Qt.Vertical)
        signal_splitter.addWidget(self.signal_plot)
        signal_splitter.addWidget(self.signal_heatmap_layout)
        signal_layout.addWidget(signal_splitter)

        self.tab_widget.addTab(signal_widget, "Signal")

        # Time cursor lines removed for performance
        
        # Frequency spectrum
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel('left', 'Magnitude')
        self.spectrum_plot.setLabel('bottom', 'Frequency', 'Hz')
        self.spectrum_plot.setDownsampling(mode='peak')
        self.spectrum_plot.setClipToView(True)
        
        # 3D Spectrum Plot (Hidden by default)
        self.spectrum_plot_3d = gl.GLViewWidget()
        self.spectrum_plot_3d.opts['distance'] = 40
        self.spectrum_plot_3d.hide()
        
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
        
        # 3D View Toggle
        self.spectrum_3d_toggle = QCheckBox("3D View")
        self.spectrum_3d_toggle.toggled.connect(self._toggle_spectrum_3d_mode)
        spectrum_controls.addWidget(self.spectrum_3d_toggle)
        
        spectrum_controls.addWidget(QLabel("Plot type:"))
        self.spectrum_plot_type = QComboBox()
        self.spectrum_plot_type.addItems(["Line", "Heatmap"])
        self.spectrum_plot_type.setCurrentText("Line")
        self.spectrum_plot_type.currentTextChanged.connect(
            lambda _: self._refresh_spectrum(time_idx=self._current_playback_index() if hasattr(self, '_current_playback_index') else None)
        )
        spectrum_controls.addWidget(self.spectrum_plot_type)

        spectrum_controls.addWidget(QLabel("Spectrum view:"))
        self.spectrum_view_mode = QComboBox()
        self.spectrum_view_mode.addItems(["Mean over positions", "Individual position"])
        self.spectrum_view_mode.currentIndexChanged.connect(lambda _: self._refresh_spectrum(time_idx=self._current_playback_index()) if hasattr(self, '_current_playback_index') else None)
        spectrum_controls.addWidget(self.spectrum_view_mode)
        self.spectrum_pos_slider = QSlider(Qt.Horizontal)
        self.spectrum_pos_slider.setRange(0, 0)
        self.spectrum_pos_slider.valueChanged.connect(lambda _: self._refresh_spectrum(time_idx=self._current_playback_index()) if hasattr(self, '_current_playback_index') else None)
        self.spectrum_pos_label = QLabel("Pos: 0.00 cm")
        spectrum_controls.addWidget(self.spectrum_pos_label)
        spectrum_controls.addWidget(self.spectrum_pos_slider)
        spectrum_layout.addLayout(spectrum_controls)

        # Component selector for spectrum
        spectrum_comp_layout = QHBoxLayout()
        self.spectrum_component_label = QLabel("Component:")
        spectrum_comp_layout.addWidget(self.spectrum_component_label)
        self.spectrum_component_combo = CheckableComboBox()
        self.spectrum_component_combo.add_items(["Magnitude", "Phase", "Phase (unwrapped)", "Real", "Imaginary"])
        self.spectrum_component_combo.set_selected_items(["Magnitude"])
        self.spectrum_component_combo.selection_changed.connect(lambda: self._refresh_spectrum(time_idx=self._current_playback_index()) if hasattr(self, '_current_playback_index') else None)
        spectrum_comp_layout.addWidget(self.spectrum_component_combo)

        # Heatmap specific mode selector
        self.spectrum_heatmap_mode_label = QLabel("Heatmap mode:")
        self.spectrum_heatmap_mode = QComboBox()
        self.spectrum_heatmap_mode.addItems(["Spin vs Frequency (FFT)", "Spin vs Time (Evolution)"])
        self.spectrum_heatmap_mode.currentTextChanged.connect(
            lambda _: self._refresh_spectrum(time_idx=self._current_playback_index()) if hasattr(self, '_current_playback_index') else None
        )
        self.spectrum_heatmap_mode_label.setVisible(False)
        self.spectrum_heatmap_mode.setVisible(False)
        spectrum_comp_layout.addWidget(self.spectrum_heatmap_mode_label)
        spectrum_comp_layout.addWidget(self.spectrum_heatmap_mode)
        
        spectrum_layout.addLayout(spectrum_comp_layout)

        spectrum_layout.addWidget(self.spectrum_plot)
        spectrum_layout.addWidget(self.spectrum_plot_3d)
        # Spectrum heatmap using GraphicsLayoutWidget
        self.spectrum_heatmap_layout = pg.GraphicsLayoutWidget()
        self.spectrum_heatmap = self.spectrum_heatmap_layout.addPlot(row=0, col=0)
        self.spectrum_heatmap.setLabel('left', 'Spin Index')
        self.spectrum_heatmap.setLabel('bottom', 'Frequency', 'Hz')
        self.spectrum_heatmap_item = pg.ImageItem()
        self.spectrum_heatmap.addItem(self.spectrum_heatmap_item)        
        self.spectrum_heatmap_colorbar = pg.ColorBarItem(values=(0, 1), interactive=False)
        self.spectrum_heatmap_layout.addItem(self.spectrum_heatmap_colorbar, row=0, col=1)
        self.spectrum_heatmap_colorbar.setImageItem(self.spectrum_heatmap_item)
        spectrum_layout.addWidget(self.spectrum_heatmap_layout)
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
        spatial_controls.addWidget(QLabel("Plot type:"))
        self.spatial_plot_type = QComboBox()
        self.spatial_plot_type.addItems(["Line", "Heatmap"])
        
        spatial_controls.addWidget(self.spatial_plot_type)
        spatial_controls.addWidget(QLabel("Heatmap mode:"))
        self.spatial_heatmap_mode = QComboBox()
        self.spatial_heatmap_mode.addItems(["Position vs Frequency", "Position vs Time"])
        self.spatial_heatmap_mode.currentTextChanged.connect(lambda _: self.update_spatial_plot_from_last_result())
        spatial_controls.addWidget(self.spatial_heatmap_mode)
        
        self.spatial_plot_type.currentTextChanged.connect(lambda text: self._update_spatial_controls_visibility(text))
        self.spatial_plot_type.currentTextChanged.connect(lambda _: self.update_spatial_plot_from_last_result())

        spatial_controls.addWidget(QLabel("View:"))
        self.spatial_mode = QComboBox() # Renamed from spatial_mode to avoid confusion
        self.spatial_mode.addItems(["Mean over freqs", "Individual freq"])
        self.spatial_mode.currentIndexChanged.connect(self.update_spatial_plot_from_last_result)
        spatial_controls.addWidget(self.spatial_mode)
        self.spatial_freq_slider = QSlider(Qt.Horizontal)
        self.spatial_freq_slider.setRange(0, 0)
        self.spatial_freq_slider.valueChanged.connect(lambda: self.update_spatial_plot_from_last_result())
        self.spatial_freq_label = QLabel("Freq: 0.0 Hz")
        spatial_controls.addWidget(self.spatial_freq_label)
        spatial_controls.addWidget(self.spatial_freq_slider)
        spatial_layout.addLayout(spatial_controls)

        # Toggle for colored position/frequency markers
        self.spatial_markers_checkbox = QCheckBox("Show colored position/frequency markers")
        self.spatial_markers_checkbox.setChecked(False)
        self.spatial_markers_checkbox.setToolTip("Display vertical lines at each position/frequency with 3D-view colors")
        self.spatial_markers_checkbox.toggled.connect(lambda _: self.update_spatial_plot_from_last_result())
        spatial_layout.addWidget(self.spatial_markers_checkbox)

        # Component selector for spatial plot
        spatial_comp_layout = QHBoxLayout()
        self.spatial_component_label = QLabel("Component:")
        spatial_comp_layout.addWidget(self.spatial_component_label)
        self.spatial_component_combo = CheckableComboBox()
        self.spatial_component_combo.add_items(["Magnitude", "Phase", "Phase (unwrapped)", "Real", "Imaginary"])
        self.spatial_component_combo.set_selected_items(["Magnitude", "Real", "Imaginary"])
        self.spatial_component_combo.selection_changed.connect(lambda: self.update_spatial_plot_from_last_result())
        spatial_comp_layout.addWidget(self.spatial_component_combo)
        spatial_layout.addLayout(spatial_comp_layout)

        # Mxy and Mz plots side by side
        spatial_plots_layout = QHBoxLayout()

        # Mxy vs position plot
        self.spatial_mxy_plot = pg.PlotWidget()
        self.spatial_mxy_plot.setLabel('left', 'Mxy (transverse)')
        self.spatial_mxy_plot.setLabel('bottom', 'Position', 'm')
        self.spatial_mxy_plot.enableAutoRange(x=False, y=False)
        self.spatial_mxy_plot.setDownsampling(mode='peak')
        self.spatial_mxy_plot.setClipToView(True)
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
        self.spatial_mz_plot.setDownsampling(mode='peak')
        self.spatial_mz_plot.setClipToView(True)
        for ln in self.spatial_slice_lines["mz"]:
            self.spatial_mz_plot.addItem(ln)
        spatial_plots_layout.addWidget(self.spatial_mz_plot)

        spatial_layout.addLayout(spatial_plots_layout)

        # Heatmap container (hidden by default)
        self.spatial_heatmap_container = QWidget()
        spatial_heatmap_splitter = QSplitter(Qt.Vertical)
        spatial_heatmap_splitter.setContentsMargins(0, 0, 0, 0)

        self.spatial_heatmap_mxy_layout = pg.GraphicsLayoutWidget()
        self.spatial_heatmap_mxy = self.spatial_heatmap_mxy_layout.addPlot(row=0, col=0)
        self.spatial_heatmap_mxy.setLabel('bottom', 'Position', 'm')
        self.spatial_heatmap_mxy.setLabel('left', 'Frequency', 'Hz')
        self.spatial_heatmap_mxy.setTitle("Mxy magnitude (|Mxy|)")
        self.spatial_heatmap_mxy_item = pg.ImageItem()
        self.spatial_heatmap_mxy.addItem(self.spatial_heatmap_mxy_item)        
        self.spatial_heatmap_mxy_colorbar = pg.ColorBarItem(values=(0, 1), interactive=False)
        self.spatial_heatmap_mxy_layout.addItem(self.spatial_heatmap_mxy_colorbar, row=0, col=1)
        self.spatial_heatmap_mxy_colorbar.setImageItem(self.spatial_heatmap_mxy_item)
        spatial_heatmap_splitter.addWidget(self.spatial_heatmap_mxy_layout)

        self.spatial_heatmap_mz_layout = pg.GraphicsLayoutWidget()
        self.spatial_heatmap_mz = self.spatial_heatmap_mz_layout.addPlot(row=0, col=0)
        self.spatial_heatmap_mz.setLabel('bottom', 'Position', 'm')
        self.spatial_heatmap_mz.setLabel('left', 'Frequency', 'Hz')
        self.spatial_heatmap_mz.setTitle("Mz")        
        self.spatial_heatmap_mz_item = pg.ImageItem()
        self.spatial_heatmap_mz.addItem(self.spatial_heatmap_mz_item)        
        self.spatial_heatmap_mz_colorbar = pg.ColorBarItem(values=(0, 1), interactive=False)
        self.spatial_heatmap_mz_layout.addItem(self.spatial_heatmap_mz_colorbar, row=0, col=1)
        self.spatial_heatmap_mz_colorbar.setImageItem(self.spatial_heatmap_mz_item)
        spatial_heatmap_splitter.addWidget(self.spatial_heatmap_mz_layout)

        self.spatial_heatmap_container.setLayout(QVBoxLayout())
        self.spatial_heatmap_container.layout().addWidget(spatial_heatmap_splitter)
        self.spatial_heatmap_container.hide()
        spatial_layout.addWidget(self.spatial_heatmap_container)

        spatial_container.setLayout(spatial_layout)
        self.tab_widget.addTab(spatial_container, "Spatial")

        # === PHANTOM TAB (2D/3D Imaging) ===
        if PHANTOM_AVAILABLE:
            self.phantom_widget = PhantomWidget(self)
            self.tab_widget.addTab(self.phantom_widget, "🔬 Phantom")
        else:
            self.phantom_widget = None

        # === K-SPACE TAB (Signal-based simulation) ===
        if KSPACE_AVAILABLE:
            def get_phantom_for_kspace():
                """Get current phantom from PhantomWidget."""
                if self.phantom_widget is not None:
                    if hasattr(self.phantom_widget, 'creator'):
                        return self.phantom_widget.creator.current_phantom
                return None
            
            def get_magnetization_for_kspace():
                """Get magnetization from last Bloch simulation."""
                if self.last_result is not None:
                    return {
                        'mx': self.last_result.get('mx'),
                        'my': self.last_result.get('my'),
                        'mz': self.last_result.get('mz'),
                    }
                return None
            
            self.kspace_widget = KSpaceWidget(
                self,
                get_phantom_callback=get_phantom_for_kspace,
                get_magnetization_callback=get_magnetization_for_kspace
            )
            self.tab_widget.addTab(self.kspace_widget, "📡 K-Space")
        else:
            self.kspace_widget = None

        # === PARAMETER SWEEP TAB ===
        self.param_sweep_widget = ParameterSweepWidget(self)
        self.tab_widget.addTab(self.param_sweep_widget, "📊 Parameter Sweep")

        # Log console lives in its own tab to save vertical space
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(6, 6, 6, 6)
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        log_layout.addWidget(self.log_widget)
        log_tab.setLayout(log_layout)
        self.tab_widget.addTab(log_tab, "Log")

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

        right_layout.addWidget(self.tab_widget, 1)

        # Apply initial colormap selection now that heatmaps are constructed
        default_cmap = "viridis"
        if hasattr(self, "heatmap_colormap") and self.heatmap_colormap is not None:
            default_cmap = self.heatmap_colormap.currentText()
        self._apply_heatmap_colormap(default_cmap)

        # Default to the 3D Vector tab so users immediately see the vector view
        if hasattr(self, "mag_3d"):
            self.tab_widget.setCurrentWidget(self.mag_3d)

        # Connect tab change to optimize rendering
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_container)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([420, 1000])
        main_layout.addWidget(splitter)
        
        # Push initial time-step into designers
        self._update_time_step(self.time_step_spin.value())
        
        # Menu bar
        self.create_menu()

        # Status bar
        self.statusBar().showMessage("Ready")

        # Connect universal time control to all views
        self._setup_time_synchronization()

        # Connect view mode controls for synchronization
        self.mag_3d.view_mode_combo.currentTextChanged.connect(self._sync_view_modes)
        self.mag_view_mode.currentTextChanged.connect(self._sync_view_modes)
        self.mag_3d.selector_slider.valueChanged.connect(self._sync_selectors)
        self.mag_view_selector.valueChanged.connect(self._sync_selectors)

        # Initialize spatial controls visibility
        self._update_spatial_controls_visibility(self.spatial_plot_type.currentText())

    def _update_spatial_controls_visibility(self, plot_type: str):
        """Enable/disable spatial controls based on plot type."""
        is_heatmap = (plot_type == "Heatmap")
        # Hide/Disable heatmap mode selector if not in heatmap mode
        if hasattr(self, 'spatial_heatmap_mode'):
            self.spatial_heatmap_mode.setEnabled(is_heatmap)
        
        # Hide markers checkbox in heatmap mode (as it applies to line plots)
        if hasattr(self, 'spatial_markers_checkbox'):
            self.spatial_markers_checkbox.setVisible(not is_heatmap)

        # Show/hide component selector (only for line plots)
        if hasattr(self, "spatial_component_combo"):
            self.spatial_component_combo.setVisible(not is_heatmap)
            if hasattr(self, "spatial_component_label"):
                self.spatial_component_label.setVisible(not is_heatmap)

    def _color_for_index(self, idx: int, total: int):
        """Consistent color cycling for multiple frequencies."""
        return pg.intColor(idx, hues=max(total, 1), values=1.0, maxValue=255)

    def _get_trace_indices_to_plot(self, total_traces: int) -> list:
        """
        Get indices of traces to plot, respecting max_traces limit.

        Returns evenly-spaced subset if total exceeds limit, otherwise all indices.
        Always includes first and last trace for boundary visualization.

        Parameters
        ----------
        total_traces : int
            Total number of available traces

        Returns
        -------
        list of int
            Indices to plot
        """
        max_traces = self.max_traces_spin.value()

        if total_traces <= max_traces:
            return list(range(total_traces))

        # Evenly space the indices
        indices = np.linspace(0, total_traces - 1, max_traces, dtype=int)
        return sorted(list(set(indices)))  # Remove duplicates and sort

    def _sync_view_modes(self, text: str):
        """Synchronize the view mode across all relevant tabs."""
        # Prevent recursive signals
        if getattr(self, '_syncing_views', False):
            return
        self._syncing_views = True

        try:
            # Update 3D Viewer
            if self.mag_3d.view_mode_combo.currentText() != text:
                self.mag_3d.view_mode_combo.setCurrentText(text)

            # Update Magnetization Plot
            if self.mag_view_mode.currentText() != text:
                self.mag_view_mode.setCurrentText(text)

            # Update Signal Plot
            if self.signal_view_mode.currentText() != text:
                self.signal_view_mode.setCurrentText(text)

            # Trigger a refresh of the plots with the new mode
            if self.last_result:
                self.update_plots(self.last_result)
                self._refresh_vector_view()

        finally:
            self._syncing_views = False

    def _sync_selectors(self, value: int):
        """Synchronize the view selector sliders across all relevant tabs."""
        if getattr(self, '_syncing_views', False):
            return
        self._syncing_views = True

        try:
            # Update 3D Viewer
            if self.mag_3d.selector_slider.value() != value:
                self.mag_3d.selector_slider.setValue(value)

            # Update Magnetization Plot
            if self.mag_view_selector.value() != value:
                self.mag_view_selector.setValue(value)

            # Update Signal Plot
            if self.signal_view_selector.value() != value:
                self.signal_view_selector.setValue(value)

        finally:
            self._syncing_views = False

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

    def _update_or_create_plot_item(self, plot_widget, cache_dict, key, x_data, y_data, pen, name=None):
        """Update existing plot item or create new one if needed.

        This avoids the expensive clear+replot cycle by reusing plot items.

        Parameters
        ----------
        plot_widget : pg.PlotWidget
            The plot widget containing the items
        cache_dict : dict
            Dictionary mapping keys to PlotDataItem objects
        key : tuple
            Unique identifier for this plot item
        x_data, y_data : np.ndarray
            Data to plot
        pen : QPen or color spec
            Pen style for the line
        name : str, optional
            Legend name for the item

        Returns
        -------
        PlotDataItem
            The updated or newly created plot item
        """
        if key in cache_dict and cache_dict[key] is not None:
            # Update existing item
            try:
                cache_dict[key].setData(x_data, y_data)
                cache_dict[key].setPen(pen)
                return cache_dict[key]
            except (AttributeError, RuntimeError):
                # Item was deleted or invalid, recreate
                pass

        # Create new item
        item = plot_widget.plot(x_data, y_data, pen=pen, name=name)
        cache_dict[key] = item
        return item

    def _invalidate_plot_caches(self):
        """Clear all cached plot items (e.g., when loading new simulation)."""
        # Remove items from plots
        for cache, plot in [(self._mxy_plot_items, self.mxy_plot),
                            (self._mz_plot_items, self.mz_plot),
                            (self._signal_plot_items, self.signal_plot)]:
            for item in cache.values():
                if item is not None:
                    try:
                        plot.removeItem(item)
                    except (AttributeError, RuntimeError):
                        pass
            cache.clear()
        self._plot_items_initialized = False

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

    def _spectrum_fft_len(self, n: int) -> int:
        """Choose an FFT length for smoother spectra."""
        # Next power of two with a minimum to avoid too few bins
        n = max(n, 8)
        n_fft = 1 << (n - 1).bit_length()
        # Apply a mild zero-padding to improve resolution without overkill
        n_fft = min(max(n_fft * 2, 512), 262144)
        return int(n_fft)

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
        # Return all indices to ensure every simulated point is accessible.
        # The animation loop handles frame skipping automatically to maintain speed.
        return np.arange(total_frames, dtype=int)

    def _reset_playback_anchor(self, idx: Optional[int] = None):
        """Record the wall-clock anchor for the current playback position."""
        if self.playback_time_ms is None or len(self.playback_time_ms) == 0:
            self._playback_anchor_wall = None
            self._playback_anchor_time_ms = None
            return
        if idx is None:
            idx = getattr(self, "anim_index", 0)
        idx = int(max(0, min(idx, len(self.playback_time_ms) - 1)))
        self._playback_anchor_wall = time.monotonic()
        self._playback_anchor_time_ms = float(self.playback_time_ms[idx])
        self.anim_index = idx
        self._last_render_wall = None
        self._playback_frame_counter = 0

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

    def _auto_update_ssfp_amplitude(self):
        """Auto-calculate SSFP pulse amplitude from flip angle, duration, and integration factor."""
        try:
            if self.sequence_designer.sequence_type.currentText() != "SSFP (Loop)":
                return
            duration_s = max(self.sequence_designer.ssfp_dur.value() / 1000.0, 1e-9)
            flip_rad = np.deg2rad(self.rf_designer.flip_angle.value())
            integfac = max(self.rf_designer.get_integration_factor(), 1e-6)
            gmr_1h_rad_Ts = 267522187.43999997
            # Required amplitude (Tesla); convert to Gauss
            amp_gauss = float(flip_rad / (gmr_1h_rad_Ts * integfac * duration_s)) * 1e4
            if not np.isfinite(amp_gauss) or amp_gauss <= 0:
                return
            # Update SSFP amplitude controls without triggering extra diagram redraws
            self.sequence_designer.ssfp_amp.blockSignals(True)
            self.sequence_designer.ssfp_amp.setValue(amp_gauss)
            self.sequence_designer.ssfp_amp.blockSignals(False)

            start_amp_val = amp_gauss * 0.5
            self.sequence_designer.ssfp_start_amp.blockSignals(True)
            self.sequence_designer.ssfp_start_amp.setValue(start_amp_val)
            self.sequence_designer.ssfp_start_amp.blockSignals(False)
            # Keep duration/phase-driven diagram in sync
            self.sequence_designer.update_diagram(self.rf_designer.get_pulse())
        except Exception:
            # Fail silently to avoid interrupting UI flow
            return

    def set_sweep_mode(self, enabled: bool):
        """Enable/disable sweep mode (skip heavy plotting during sweeps)."""
        self._sweep_mode = bool(enabled)
        if enabled:
            # Stop any running animation to save resources
            if hasattr(self, 'anim_timer') and self.anim_timer.isActive():
                self.anim_timer.stop()
                self._sync_play_toggle(False)

    def _setup_time_synchronization(self):
        """Setup connections for universal time control synchronization."""
        # Connect universal time control to update all views
        self.time_control.time_changed.connect(self._on_universal_time_changed)

        # Connect play/pause/reset buttons
        self.time_control.play_pause_button.toggled.connect(self._handle_play_pause_toggle)
        self.time_control.reset_button.clicked.connect(self._handle_reset_clicked)
        self.time_control.speed_spin.valueChanged.connect(self._update_playback_speed)

        # Connect 3D vector position changes to universal control
        self.mag_3d.position_changed.connect(self._on_3d_vector_position_changed)

    def _on_universal_time_changed(self, time_idx: int, skip_expensive_updates=False, reset_anchor=True):
        """Central handler for universal time control changes - updates all views.

        Parameters
        ----------
        time_idx : int
            The time index to display
        skip_expensive_updates : bool
            If True, only update time cursors and skip expensive plot redraws.
            Use this during animation playback for better performance.
        reset_anchor : bool
            If True, reset the playback timing anchor (for manual scrubbing).
        """
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

        # Get current visible tab to optimize updates (define early for all code paths)
        current_tab_index = self.tab_widget.currentIndex()

        # Check which plots are actually visible and need updates
        # Tab indices: 0=Magnetization, 1=3D Vector, 2=Signal, 3=Spectrum, 4=Spatial, ...
        mag_tab_visible = (current_tab_index == 0)
        signal_tab_visible = (current_tab_index == 2)
        spectrum_tab_visible = (current_tab_index == 3)
        spatial_tab_visible = (current_tab_index == 4)

        # Update time cursors on plots
        # PyQt/pyqtgraph still processes updates even for hidden tabs, causing lag.
        # Disable updates for plots that aren't visible to improve animation performance.
        if self.last_time is not None and 0 <= actual_idx < len(self.last_time):
            time_ms = self.last_time[actual_idx] * 1000

            # Only update time cursors when NOT animating (during scrubbing/pause)
            # During animation, skip time cursor updates for Magnetization and Signal plots
            # to improve performance - only 3D vector animates
            # Time lines removed for performance

        # Always update visible spectrum/spatial views, even during playback
        if spatial_tab_visible:
            self.update_spatial_plot_from_last_result(time_idx=actual_idx)
            self._spatial_needs_update = False
        else:
            self._spatial_needs_update = True

        if spectrum_tab_visible:
            self._refresh_spectrum(time_idx=actual_idx, skip_fft=skip_expensive_updates)
            self._spectrum_needs_update = False
        else:
            self._spectrum_needs_update = True

    def _on_tab_changed(self, index: int):
        """Handle tab changes to optimize rendering.

        Disable updates on plots that aren't visible to speed up tab switching.
        """
        # Enable updates on all plot widgets first
        all_plot_widgets = [
            self.mxy_plot, self.mz_plot, self.mxy_heatmap_layout, self.mz_heatmap_layout,
            self.signal_plot, self.signal_heatmap_layout, self.spectrum_plot, self.spectrum_heatmap_layout,
            self.spatial_mxy_plot, self.spatial_mz_plot, self.spatial_heatmap_container
        ]
        for widget in all_plot_widgets:
            if widget is not None:
                widget.setUpdatesEnabled(True)

        # Now disable updates on plots not in the current tab
        # Tab indices: 0=Magnetization, 1=3D Vector, 2=Signal, 3=Spectrum, 4=Spatial
        if index != 0:  # Not Magnetization tab
            self.mxy_plot.setUpdatesEnabled(False)
            self.mz_plot.setUpdatesEnabled(False)
        if index != 2:  # Not Signal tab
            self.signal_plot.setUpdatesEnabled(False)
            self.signal_heatmap_layout.setUpdatesEnabled(False)
        if index != 3:  # Not Spectrum tab
            self.spectrum_plot.setUpdatesEnabled(False)
            if hasattr(self, "spectrum_heatmap_layout") and self.spectrum_heatmap_layout is not None:
                self.spectrum_heatmap_layout.setUpdatesEnabled(False)
        else:  # Switching TO Spectrum tab
            # Update spectrum if it's dirty
            if self._spectrum_needs_update and hasattr(self, 'last_result') and self.last_result is not None:
                current_idx = self.time_control.time_slider.value() if hasattr(self, 'time_control') else 0
                actual_idx = self._playback_to_full_index(current_idx)
                self._refresh_spectrum(time_idx=actual_idx, skip_fft=False)
                self._spectrum_needs_update = False
        if index != 4:  # Not Spatial tab
            self.spatial_mxy_plot.setUpdatesEnabled(False)
            self.spatial_mz_plot.setUpdatesEnabled(False)
            if hasattr(self, "spatial_heatmap_container"):
                self.spatial_heatmap_container.setUpdatesEnabled(False)
        else:  # Switching TO Spatial tab
            # Update spatial if it's dirty
            if self._spatial_needs_update and hasattr(self, 'last_result') and self.last_result is not None:
                current_idx = self.time_control.time_slider.value() if hasattr(self, 'time_control') else 0
                actual_idx = self._playback_to_full_index(current_idx)
                self.update_spatial_plot_from_last_result(time_idx=actual_idx)
                self._spatial_needs_update = False

    def _on_3d_vector_position_changed(self, time_idx: int):
        """Handle 3D vector view position changes."""
        if not self.time_control._updating:
            self.time_control.set_time_index(time_idx)
        # Propagate the change to all synchronized views
        # When user is manually scrubbing, do full update (skip_expensive_updates=False)
        is_playing = self.anim_timer.isActive() if hasattr(self, 'anim_timer') else False
        self._on_universal_time_changed(time_idx, skip_expensive_updates=is_playing)

    def _handle_play_pause_toggle(self, playing: bool):
        """Unified handler for play/pause toggle."""
        if playing:
            self._on_universal_play()
        else:
            self._on_universal_pause()

    def _handle_reset_clicked(self):
        """Reset playback and return to paused state."""
        self._on_universal_reset()
        self._sync_play_toggle(False)

    def _sync_play_toggle(self, playing: bool):
        """Keep the play/pause toggle in sync with actual playback."""
        if hasattr(self, "time_control"):
            self.time_control.sync_play_state(playing)

    def _on_universal_play(self):
        """Handle universal play button."""
        # Use latest speed setting
        self._recompute_anim_interval(self.time_control.speed_spin.value())
        self._resume_vector_animation()

    def _on_universal_pause(self):
        """Handle universal pause button."""
        self._pause_vector_animation()
        self._sync_play_toggle(False)

    def _on_universal_reset(self):
        """Handle universal reset button."""
        self._reset_vector_animation()
        self._sync_play_toggle(False)

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
            # Check plot type and switch visibility
            plot_type = self.mag_plot_type.currentText() if hasattr(self, 'mag_plot_type') else "Line"
            if plot_type == "Heatmap":
                self.mxy_plot.hide()
                self.mz_plot.hide()
                self.mxy_heatmap_layout.show()
                self.mz_heatmap_layout.show()
                self._update_mag_heatmaps()
            else:
                self.mxy_plot.show()
                self.mz_plot.show()
                self.mxy_heatmap.hide()
                self.mz_heatmap.hide()
                self.update_plots(self.last_result)

    def _refresh_signal_plots(self):
        """Re-render signal plots using the current filter selection."""
        if self.last_result is None:
            return
        # Check plot type and switch visibility
        plot_type = self.signal_plot_type.currentText() if hasattr(self, 'signal_plot_type') else "Heatmap"
        if plot_type == "Heatmap":
            self.signal_plot.hide()
            self.signal_heatmap_layout.show()
            self._update_signal_heatmaps()
        else:
            self.signal_plot.show()
            self.signal_heatmap_layout.hide()
            # Don't call update_plots to avoid infinite loop
            # Just refresh the specific plot content

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
                # Check if the item is actually in this scene before removing
                if existing.scene() == plot_widget.scene():
                    plot_widget.scene().removeItem(existing)
            except (RuntimeError, AttributeError):
                # Item already deleted or scene mismatch
                pass
            setattr(self, attr_name, None)
        if enable:
            legend = plot_widget.addLegend(offset=(6, 6))
            legend.layout.setSpacing(4)
            setattr(self, attr_name, legend)
            return legend
        return None

    def _apply_heatmap_colormap(self, cmap_name: Optional[str] = None):
        """Apply a shared colormap to all heatmaps/colorbars."""
        if cmap_name is None and hasattr(self, "heatmap_colormap"):
            cmap_name = self.heatmap_colormap.currentText()
        cmap_name = cmap_name or "viridis"
        try:
            cmap = pg.colormap.get(cmap_name)
        except Exception:
            cmap = cmap_name  # fall back to string name if lookup fails

        def _set_cb(cb_item):
            if cb_item is None:
                return
            setter = getattr(cb_item, "setColorMap", None)
            if callable(setter):
                try:
                    setter(cmap)
                except Exception:
                    pass

        for cb in [
            getattr(self, "mxy_heatmap_colorbar", None),
            getattr(self, "mz_heatmap_colorbar", None),
            getattr(self, "signal_heatmap_colorbar", None),
            getattr(self, "spectrum_heatmap_colorbar", None),
            getattr(self, "spatial_heatmap_mxy_colorbar", None),
            getattr(self, "spatial_heatmap_mz_colorbar", None),
        ]:
            _set_cb(cb)

        lut = None
        if hasattr(cmap, "getLookupTable"):
            try:
                lut = cmap.getLookupTable()
            except Exception:
                lut = None
        if lut is not None:
            for img in [
                getattr(self, "mxy_heatmap_item", None),
                getattr(self, "mz_heatmap_item", None),
                getattr(self, "signal_heatmap_item", None),
                getattr(self, "spectrum_heatmap_item", None),
                getattr(self, "spatial_heatmap_mxy_item", None),
                getattr(self, "spatial_heatmap_mz_item", None),
            ]:
                if img is not None and hasattr(img, "setLookupTable"):
                    try:
                        img.setLookupTable(lut)
                    except Exception:
                        pass

        # Keep a consistent palette for the status bar progress as well
        if hasattr(self, "status_progress"):
            self.status_progress.setStyleSheet("")

    def _update_mag_selector_limits(self, npos: int, nfreq: int, disable: bool = False):
        """Sync magnetization view selector with available pos/freq counts."""
        if not hasattr(self, "mag_view_mode"):
            return
        mode = self.mag_view_mode.currentText()
        slider = self.mag_view_selector

        # Helper to find index closest to 0
        def _get_zero_idx(arr):
            if arr is None or len(arr) == 0: return 0
            return int(np.argmin(np.abs(arr)))

        if disable:
            slider.setRange(0, 0)
            slider.setEnabled(False)
            self.mag_view_selector_label.setText("All spins")
        elif mode == "Positions @ freq":
            max_idx = max(0, nfreq - 1)
            slider.setRange(0, max_idx)
            slider.setEnabled(nfreq > 1)
            
            # If current value is out of range or we just switched, try to set to 0 Hz
            if slider.value() > max_idx:
                target = _get_zero_idx(self.last_frequencies) if self.last_frequencies is not None else 0
                slider.setValue(target)
            
            idx = slider.value()
            freq_val = self.last_frequencies[idx] if self.last_frequencies is not None and idx < len(self.last_frequencies) else idx
            self.mag_view_selector_label.setText(f"Freq: {freq_val:.1f} Hz")
        elif mode == "Freqs @ position":
            max_idx = max(0, npos - 1)
            slider.setRange(0, max_idx)
            slider.setEnabled(npos > 1)

            # If current value is out of range or we just switched, try to set to 0 cm
            if slider.value() > max_idx:
                target = _get_zero_idx(self.last_positions[:, 2] * 100) if self.last_positions is not None else 0
                slider.setValue(target)
                
            idx = slider.value()
            pos_val = self.last_positions[idx, 2] * 100 if self.last_positions is not None and idx < len(self.last_positions) else idx
            self.mag_view_selector_label.setText(f"Pos: {pos_val:.2f} cm")
        else:
            slider.setRange(0, 0)
            slider.setEnabled(False)
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

    def _update_signal_selector_limits(self, npos: int, nfreq: int, disable: bool = False):
        """Sync signal view selector with available pos/freq counts."""
        if not hasattr(self, "signal_view_mode"):
            return
        mode = self.signal_view_mode.currentText()
        slider = self.signal_view_selector

        if disable:
            max_idx = 0
            prefix = "All"
            label_text = "All spins"
        elif mode == "Positions @ freq":
            max_idx = max(0, nfreq - 1)
            prefix = "Freq"
            idx = min(slider.value(), max_idx)
            freq_val = self.last_frequencies[idx] if self.last_frequencies is not None and idx < len(self.last_frequencies) else idx
            label_text = f"{prefix}: {freq_val:.1f} Hz"
        elif mode == "Freqs @ position":
            max_idx = max(0, npos - 1)
            prefix = "Pos"
            idx = min(slider.value(), max_idx)
            pos_val = self.last_positions[idx, 2] * 100 if self.last_positions is not None and idx < len(self.last_positions) else idx
            label_text = f"{prefix}: {pos_val:.2f} cm"
        else:
            max_idx = 0
            prefix = "All"
            label_text = "All spins"

        slider.blockSignals(True)
        slider.setMaximum(max_idx)
        slider.setValue(min(slider.value(), max_idx) if max_idx > 0 else 0)
        slider.setEnabled(not disable and max_idx > 0)
        slider.setVisible(max_idx > 0)
        slider.blockSignals(False)

        self.signal_view_selector_label.setText(label_text)

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
            time_idx = int(min(max(0, time_idx), ntime - 1))

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
        max_freq_idx = max(0, freq_count - 1)

        # Update slider range
        self.spatial_freq_slider.blockSignals(True)
        self.spatial_freq_slider.setMaximum(max_freq_idx)
        if self.spatial_freq_slider.value() > max_freq_idx:
            self.spatial_freq_slider.setValue(max_freq_idx)
        self.spatial_freq_slider.blockSignals(False)
        
        freq_sel = min(self.spatial_freq_slider.value(), max_freq_idx)
        
        # Update label
        actual_freq = 0.0
        if self.last_frequencies is not None and freq_sel < len(self.last_frequencies):
             actual_freq = self.last_frequencies[freq_sel]
             self.spatial_freq_label.setText(f"Freq: {actual_freq:.1f} Hz")
        else:
             self.spatial_freq_label.setText(f"Freq idx: {freq_sel}")

        spatial_view_mode = self.spatial_mode.currentText()
        if spatial_view_mode == "Mean over freqs":
            mxy_pos = np.sqrt(mx_display**2 + my_display**2).mean(axis=1)
            mz_pos = mz_display.mean(axis=1)
        elif spatial_view_mode == "Mean + individuals": # This mode is no longer in the dropdown, but keeping for safety
            mxy_pos = np.sqrt(mx_display**2 + my_display**2).mean(axis=1)
            mz_pos = mz_display.mean(axis=1)
        else:  # Individual freq
            mxy_pos = np.sqrt(mx_display[:, freq_sel]**2 + my_display[:, freq_sel]**2)
            mz_pos = mz_display[:, freq_sel]

        # Choose a signed position axis (prefer the axis with largest span)
        pos_axis = self.last_positions
        spans = np.ptp(pos_axis, axis=0)  # max - min per axis
        axis_idx = int(np.argmax(spans))
        pos_distance = pos_axis[:, axis_idx]

        self.log_message(f"Spatial plot: mxy_pos shape = {mxy_pos.shape}, mz_pos shape = {mz_pos.shape}, pos_distance shape = {pos_distance.shape}")
        
        freq_axis = np.asarray(self.last_frequencies) if self.last_frequencies is not None else np.arange(freq_count)
        if freq_axis.shape[0] != freq_count:
            freq_axis = np.linspace(freq_axis.min() if freq_axis.size else 0.0,
                                    freq_axis.max() if freq_axis.size else float(freq_count - 1),
                                    freq_count)

        # Cache data for export and heatmap updates
        self._last_spatial_export = {
            "position_m": pos_distance,
            "mxy": mxy_pos,
            "mz": mz_pos,
            "freq_index": freq_sel,
            "time_idx": time_idx,
            "time_s": self.last_time[time_idx] if self.last_time is not None and len(self.last_time) > time_idx else None,
            "mxy_per_freq": np.sqrt(mx_display**2 + my_display**2),
            "mz_per_freq": mz_display,
            "frequency_axis": freq_axis,
            "heatmap_mode": None,
        }        

        # Update plots
        plot_type = self.spatial_plot_type.currentText() if hasattr(self, "spatial_plot_type") else "Line"
        heatmap_mode = self.spatial_heatmap_mode.currentText() if hasattr(self, "spatial_heatmap_mode") else "Position vs Frequency"
        show_heatmap = (plot_type == "Heatmap")
        self._set_spatial_plot_visibility(show_heatmap)
        if show_heatmap:
            # Heatmap mode
            if heatmap_mode == "Position vs Time" and is_time_resolved:
                mxy_time = np.sqrt(self.spatial_mx_time_series[:, :, freq_sel]**2 + self.spatial_my_time_series[:, :, freq_sel]**2)
                mz_time = self.spatial_mz_time_series[:, :, freq_sel]
                self._last_spatial_export.update({
                    "heatmap_mode": "time",
                    "mxy_time": mxy_time,
                    "mz_time": mz_time,
                })
                self._update_spatial_time_heatmaps(pos_distance, self.last_time, mxy_time, mz_time, freq_sel)
            elif heatmap_mode == "Position vs Time" and not is_time_resolved:
                self.log_message("Spatial heatmap time view requires time-resolved simulation; showing frequency view instead.")
                self._last_spatial_export["heatmap_mode"] = "frequency"
                self._update_spatial_heatmaps(pos_distance, self._last_spatial_export["mxy_per_freq"], mz_display, freq_axis)
            else:
                self._last_spatial_export["heatmap_mode"] = "frequency"
                self._update_spatial_heatmaps(pos_distance, self._last_spatial_export["mxy_per_freq"], mz_display, freq_axis)
        else:
            # Line plot mode
            self._update_spatial_line_plots(pos_distance, mxy_pos, mz_pos, mx_display, my_display, mz_display, freq_sel, spatial_view_mode)

        # Keep sequence diagram in sync but avoid time cursors on spatial (position) axes
        if hasattr(self, 'spatial_mxy_time_line'):
            self.spatial_mxy_time_line.hide()
        if hasattr(self, 'spatial_mz_time_line'):
            self.spatial_mz_time_line.hide()
        if is_time_resolved and time_idx < len(self.last_time):
            current_time = self.last_time[time_idx]
            # Synchronize sequence diagram playhead
            if hasattr(self, 'sequence_designer') and hasattr(self.sequence_designer, 'playhead_line'):
                if self.sequence_designer.playhead_line is not None:
                    self.sequence_designer.playhead_line.setValue(current_time * 1000.0)
                    if not self.sequence_designer.playhead_line.isVisible():
                        self.sequence_designer.playhead_line.show()

    def _set_spatial_plot_visibility(self, show_heatmap: bool):
        """Toggle between line plots and heatmaps in the Spatial view."""
        if hasattr(self, "spatial_heatmap_container"):
            self.spatial_heatmap_container.setVisible(show_heatmap)
        if hasattr(self, "spatial_mxy_plot"):
            self.spatial_mxy_plot.setVisible(not show_heatmap)
        if hasattr(self, "spatial_mz_plot"):
            self.spatial_mz_plot.setVisible(not show_heatmap)
        # Hide Mz plot when showing ONLY phase (wrapped or unwrapped)
        selected = self.spatial_component_combo.get_selected_items()
        if len(selected) > 0 and all(c in ["Phase", "Phase (unwrapped)"] for c in selected):
            self.spatial_mz_plot.setVisible(False)

    def _update_spatial_line_plots(self, position, mxy, mz, mx_display=None, my_display=None, mz_display=None, freq_sel=0, spatial_mode="Mean only"):
        """Update the Mxy and Mz line plots."""
        try:
            # Safely clear plots while preserving persistent items
            persistent_mxy = [self.spatial_mxy_time_line] + self.spatial_slice_lines["mxy"]
            persistent_mz = [self.spatial_mz_time_line] + self.spatial_slice_lines["mz"]
            self._safe_clear_plot(self.spatial_mxy_plot, persistent_mxy)
            self._safe_clear_plot(self.spatial_mz_plot, persistent_mz)

            selected_components = self.spatial_component_combo.get_selected_items()

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
                
                if "Magnitude" in selected_components:
                    self.spatial_mxy_plot.plot(position, mxy, pen=pg.mkPen('b', width=3), name='|Mxy| mean')
                if "Real" in selected_components:
                    mx_mean = np.mean(mx_display, axis=1)
                    self.spatial_mxy_plot.plot(position, mx_mean, pen=pg.mkPen('r', style=Qt.DashLine, width=2), name='Mx mean')
                if "Imaginary" in selected_components:
                    my_mean = np.mean(my_display, axis=1)
                    self.spatial_mxy_plot.plot(position, my_mean, pen=pg.mkPen('g', style=Qt.DotLine, width=2), name='My mean')
                if "Phase" in selected_components:
                    mx_mean = np.mean(mx_display, axis=1)
                    my_mean = np.mean(my_display, axis=1)
                    phase = np.angle(mx_mean + 1j * my_mean) / np.pi
                    self.spatial_mxy_plot.plot(position, phase, pen=pg.mkPen('c', width=2), name='Phase mean')
                
                if "Phase (unwrapped)" in selected_components:
                    mx_mean = np.mean(mx_display, axis=1)
                    my_mean = np.mean(my_display, axis=1)
                    phase_unwrapped = np.unwrap(np.angle(mx_mean + 1j * my_mean)) / np.pi
                    self.spatial_mxy_plot.plot(position, phase_unwrapped, pen=pg.mkPen('y', width=2), name='Phase (unwrapped) mean')

                self.spatial_mz_plot.plot(position, mz, pen=pg.mkPen('m', width=3), name='Mz mean')
            else:
                self._reset_legend(self.spatial_mxy_plot, "spatial_mxy_legend", True)
                self._reset_legend(self.spatial_mz_plot, "spatial_mz_legend", False)

                self.spatial_mxy_plot.setLabel('left', 'Mxy (transverse)')
                
                if "Phase" in selected_components:
                    if mx_display is not None and my_display is not None:
                        if spatial_mode == "Individual freq" and mx_display.ndim == 2:
                            phase = np.angle(mx_display[:, freq_sel] + 1j * my_display[:, freq_sel]) / np.pi if freq_sel < mx_display.shape[1] else np.zeros(mx_display.shape[0])
                        else:
                            mx_mean = np.mean(mx_display, axis=1)
                            my_mean = np.mean(my_display, axis=1)
                            phase = np.angle(mx_mean + 1j * my_mean) / np.pi
                        self.spatial_mxy_plot.plot(position, phase, pen=pg.mkPen('c', width=2), name='Phase')

                if "Phase (unwrapped)" in selected_components:
                    if mx_display is not None and my_display is not None:
                        if spatial_mode == "Individual freq" and mx_display.ndim == 2:
                            phase_unwrapped = np.unwrap(np.angle(mx_display[:, freq_sel] + 1j * my_display[:, freq_sel])) / np.pi if freq_sel < mx_display.shape[1] else np.zeros(mx_display.shape[0])
                        else:
                            mx_mean = np.mean(mx_display, axis=1)
                            my_mean = np.mean(my_display, axis=1)
                            phase_unwrapped = np.unwrap(np.angle(mx_mean + 1j * my_mean)) / np.pi
                        self.spatial_mxy_plot.plot(position, phase_unwrapped, pen=pg.mkPen('y', width=2), name='Phase (unwrapped)')

                if "Magnitude" in selected_components:
                    self.spatial_mxy_plot.plot(position, mxy, pen=pg.mkPen('b', width=2), name='|Mxy|')
                
                if "Real" in selected_components:
                    if mx_display is not None:
                        if spatial_mode == "Individual freq" and mx_display.ndim == 2 and freq_sel < mx_display.shape[1]:
                            mx_line = mx_display[:, freq_sel]
                        elif mx_display.ndim == 2:
                            mx_line = np.mean(mx_display, axis=1)
                        else:
                            mx_line = mx_display
                        self.spatial_mxy_plot.plot(position, mx_line, pen=pg.mkPen('r', style=Qt.DashLine, width=2), name='Mx')

                if "Imaginary" in selected_components:
                    if my_display is not None:
                        if spatial_mode == "Individual freq" and my_display.ndim == 2 and freq_sel < my_display.shape[1]:
                            my_line = my_display[:, freq_sel]
                        elif my_display.ndim == 2:
                            my_line = np.mean(my_display, axis=1)
                        else:
                            my_line = my_display
                        self.spatial_mxy_plot.plot(position, my_line, pen=pg.mkPen('g', style=Qt.DotLine, width=2), name='My')

                self.spatial_mz_plot.plot(position, mz, pen=pg.mkPen('m', width=2))
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

            is_only_phase = all(c in ["Phase", "Phase (unwrapped)"] for c in selected_components) and len(selected_components) > 0
            has_magnitude = any(c in selected_components for c in ["Magnitude", "Real", "Imaginary"])
            has_unwrapped = "Phase (unwrapped)" in selected_components

            if has_unwrapped:
                self.spatial_mxy_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            elif is_only_phase:
                self.spatial_mxy_plot.setYRange(-1.1, 1.1, padding=0)
            else:
                mxy_ymin, mxy_ymax = padded_range(mxy_min_all, mxy_max_all, scale=1.1)
                self.spatial_mxy_plot.setYRange(mxy_ymin, mxy_ymax)

            mz_ymin, mz_ymax = padded_range(mz_min_all, mz_max_all, scale=1.1) if not is_only_phase else (-1.1, 1.1)

            # Use pre-calculated stable ranges, adapted for the current selection
            if has_unwrapped:
                pass # Already set to auto
            elif is_only_phase:
                pass # Already set fixed
            elif hasattr(self, 'spatial_mxy_yrange'):
                # If only "Magnitude" is selected (no Real/Imag), use [0, max]
                if len(selected_components) == 1 and selected_components[0] == "Magnitude":
                    mxy_ymin, mxy_ymax = -0.05 * self.spatial_mxy_yrange[1], self.spatial_mxy_yrange[1]
                else:
                    mxy_ymin, mxy_ymax = self.spatial_mxy_yrange
                self.spatial_mxy_plot.setYRange(mxy_ymin, mxy_ymax)
            
            if hasattr(self, 'spatial_mz_yrange') and not is_only_phase:
                mz_ymin, mz_ymax = self.spatial_mz_yrange
                self.spatial_mz_plot.setYRange(mz_ymin, mz_ymax)

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
                max_markers = min(self.max_traces_spin.value(), 50)  # Limit for spatial markers

                # Broaden condition to show markers even in "Mean" mode if user enabled them
                if nfreq > 0:
                    # Downsample which frequencies to mark
                    if nfreq <= max_markers:
                        freq_indices_to_mark = list(range(nfreq))
                    else:
                        step = nfreq / max_markers
                        freq_indices_to_mark = [int(i * step) for i in range(max_markers)]

                    # Downsample position points to draw stems at
                    if npos <= max_markers:
                        pos_indices_to_mark = list(range(npos))
                    else:
                        step_pos = npos / max_markers
                        pos_indices_to_mark = [int(i * step_pos) for i in range(max_markers)]

                    # Mark selected frequencies with colored stem plots
                    for fi in freq_indices_to_mark:
                        # Ensure color is an RGBA tuple for consistent rendering
                        color = self._color_tuple(self._color_for_index(fi, nfreq))
                        mxy_val = np.sqrt(mx_display[:, fi]**2 + my_display[:, fi]**2) if mx_display.ndim == 2 else mxy
                        mz_val = mz_display[:, fi] if mz_display.ndim == 2 else mz_pos

                        # Downsample position points to draw stems at
                        if npos <= max_markers:
                            pos_indices_to_mark = list(range(npos))
                        else:
                            step_pos = npos / max_markers
                            pos_indices_to_mark = [int(i * step_pos) for i in range(max_markers)]

                        # Draw stem lines from 0 to current value at selected positions
                        for pi in pos_indices_to_mark:
                            pos_val = position[pi]
                            # Mxy markers
                            line_mxy = pg.PlotCurveItem([pos_val, pos_val], [0, mxy_val[pi]],
                                                         pen=pg.mkPen(color=color, width=1.5))
                            self.spatial_mxy_plot.addItem(line_mxy)
                            # Mz markers
                            line_mz = pg.PlotCurveItem([pos_val, pos_val], [0, mz_val[pi]],
                                                        pen=pg.mkPen(color=color, width=1.5))
                            self.spatial_mz_plot.addItem(line_mz)

            self.log_message("Spatial plot: updated successfully")
        except Exception as e:
            self.log_message(f"Spatial plot: error updating plots: {e}")
            import traceback
            self.log_message(f"Spatial plot: traceback: {traceback.format_exc()}")

    def _update_spatial_heatmaps(self, position, mxy_per_freq, mz_per_freq, freq_axis):
        """Render spatial heatmaps (position vs frequency) for |Mxy| and Mz."""
        try:
            if mxy_per_freq is None or mz_per_freq is None:
                return
            pos = np.asarray(position)
            freq_axis = np.asarray(freq_axis)
            if pos.size == 0 or freq_axis.size == 0:
                return
            mxy_arr = np.abs(np.asarray(mxy_per_freq))
            mz_arr = np.asarray(mz_per_freq)
            if mxy_arr.ndim != 2 or mz_arr.ndim != 2:
                return
            # Ensure axis lengths match the data
            npos, nfreq = mxy_arr.shape
            if freq_axis.size != nfreq:
                freq_axis = np.linspace(freq_axis.min() if freq_axis.size else 0.0,
                                        freq_axis.max() if freq_axis.size else float(nfreq - 1),
                                        nfreq)

            pos_min, pos_max = float(np.nanmin(pos)), float(np.nanmax(pos))
            if not np.isfinite(pos_min) or not np.isfinite(pos_max) or np.isclose(pos_min, pos_max):
                pos_min, pos_max = 0.0, float(max(npos - 1, 1))
            x_span = pos_max - pos_min if pos_max != pos_min else 1.0
            freq_min, freq_max = float(np.nanmin(freq_axis)), float(np.nanmax(freq_axis))
            if not np.isfinite(freq_min) or not np.isfinite(freq_max) or np.isclose(freq_min, freq_max):
                freq_min, freq_max = 0.0, float(max(nfreq - 1, 1))
            y_span = freq_max - freq_min if freq_max != freq_min else 1.0

            def _set_heatmap(plot_widget, img_item, colorbar, data, pos_min, y_min, x_span, y_span):
                img_item.setImage(data, autoLevels=True, axisOrder='row-major')
                img_item.setRect(pos_min, y_min, x_span, y_span)
                plot_widget.setXRange(pos_min, pos_max, padding=0)
                plot_widget.setYRange(y_min, y_max, padding=0)
                if colorbar is not None:
                    finite_vals = data[np.isfinite(data)]
                    if finite_vals.size:
                        with np.errstate(invalid='ignore'):
                            vmin, vmax = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
                        if np.isfinite(vmin) and np.isfinite(vmax) and not np.isclose(vmax, vmin):
                            colorbar.setLevels([vmin, vmax])

            # ImageItem expects (rows, cols) = (y, x)
            _set_heatmap(self.spatial_heatmap_mxy, self.spatial_heatmap_mxy_item,
                         getattr(self, "spatial_heatmap_mxy_colorbar", None),
                         mxy_arr.T, pos_min, freq_min, x_span, y_span)
            _set_heatmap(self.spatial_heatmap_mz, self.spatial_heatmap_mz_item,
                         getattr(self, "spatial_heatmap_mz_colorbar", None),
                         mz_arr.T, pos_min, freq_min, x_span, y_span)
        except Exception as exc:
            self.log_message(f"Spatial heatmap update failed: {exc}")

    def _update_spatial_time_heatmaps(self, position, time_axis, mxy_time, mz_time, freq_sel):
        """Render spatial heatmaps (position vs time) for a selected frequency."""
        try:
            pos = np.asarray(position)
            time_axis = np.asarray(time_axis) if time_axis is not None else np.arange(mxy_time.shape[0])
            if pos.size == 0 or time_axis.size == 0:
                return
            if mxy_time.ndim != 2 or mz_time.ndim != 2:
                return
            ntime, npos = mxy_time.shape
            if time_axis.size != ntime:
                time_axis = np.linspace(time_axis.min() if time_axis.size else 0.0,
                                        time_axis.max() if time_axis.size else float(ntime - 1),
                                        ntime)
            time_ms = time_axis * 1000.0

            pos_min, pos_max = float(np.nanmin(pos)), float(np.nanmax(pos))
            if not np.isfinite(pos_min) or not np.isfinite(pos_max) or np.isclose(pos_min, pos_max):
                pos_min, pos_max = 0.0, float(max(npos - 1, 1))
            x_span = pos_max - pos_min if pos_max != pos_min else 1.0
            t_min, t_max = float(np.nanmin(time_ms)), float(np.nanmax(time_ms))
            if not np.isfinite(t_min) or not np.isfinite(t_max) or np.isclose(t_min, t_max):
                t_min, t_max = 0.0, float(max(ntime - 1, 1))
            t_span = t_max - t_min if t_max != t_min else 1.0

            def _set_heatmap(plot_widget, img_item, colorbar, data, title):
                img_item.setImage(data, autoLevels=True, axisOrder='row-major')
                img_item.setRect(pos_min, t_min, x_span, t_span)
                plot_widget.setXRange(pos_min, pos_max, padding=0)
                plot_widget.setYRange(t_min, t_max, padding=0)
                plot_widget.setLabel('bottom', 'Position', 'm')
                plot_widget.setLabel('left', 'Time', 'ms')
                plot_widget.setTitle(title)
                if colorbar is not None:
                    finite_vals = data[np.isfinite(data)]
                    if finite_vals.size:
                        vmin = float(finite_vals.min())
                        vmax = float(finite_vals.max())
                        if np.isfinite(vmin) and np.isfinite(vmax) and vmax != vmin:
                            colorbar.setLevels((vmin, vmax))

            _set_heatmap(
                self.spatial_heatmap_mxy,
                self.spatial_heatmap_mxy_item,
                getattr(self, "spatial_heatmap_mxy_colorbar", None),
                np.abs(mxy_time),
                f"|Mxy| vs time @ freq {freq_sel}",
            )
            _set_heatmap(
                self.spatial_heatmap_mz,
                self.spatial_heatmap_mz_item,
                getattr(self, "spatial_heatmap_mz_colorbar", None),
                mz_time,
                f"Mz vs time @ freq {freq_sel}",
            )
        except Exception as exc:
            self.log_message(f"Spatial time heatmap update failed: {exc}")

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
            self.pos_spin,
            self.pos_range,
            self.freq_spin,
            self.freq_range,
            self.rf_designer.duration,
            self.time_step_spin,
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
        if "duration" in presets:
            self.rf_designer.duration.setValue(presets["duration"])

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
        # Optional simulation grid presets
        if "num_positions" in presets:
            self.pos_spin.setValue(int(presets["num_positions"]))
        if "position_range_cm" in presets:
            self.pos_range.setValue(float(presets["position_range_cm"]))
        if "num_frequencies" in presets:
            self.freq_spin.setValue(int(presets["num_frequencies"]))
        if "frequency_range_hz" in presets:
            self.freq_range.setValue(float(presets["frequency_range_hz"]))
        if "time_step" in presets:
            self.time_step_spin.setValue(float(presets["time_step"]))

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

    def _sync_preview_checkboxes(self, checked: bool):
        """Keep preview checkboxes in sync between footer and status bar."""
        if getattr(self, "_syncing_preview", False):
            return
        self._syncing_preview = True
        try:
            if hasattr(self, "preview_checkbox") and self.preview_checkbox.isChecked() != checked:
                self.preview_checkbox.setChecked(checked)
            if hasattr(self, "status_preview_checkbox") and self.status_preview_checkbox.isChecked() != checked:
                self.status_preview_checkbox.setChecked(checked)
            if hasattr(self, "toolbar_preview_action") and self.toolbar_preview_action.isChecked() != checked:
                # block to avoid recursive signals
                was_blocked = self.toolbar_preview_action.blockSignals(True)
                self.toolbar_preview_action.setChecked(checked)
                self.toolbar_preview_action.blockSignals(was_blocked)
        finally:
            self._syncing_preview = False

    def _build_status_run_bar(self):
        """Add always-visible run controls to the status bar."""
        bar = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)

        self.status_run_button = QPushButton("Run Simulation")
        self.status_run_button.clicked.connect(self.run_simulation)
        self.status_run_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.status_run_button)

        self.status_cancel_button = QPushButton("Cancel")
        self.status_cancel_button.clicked.connect(self.cancel_simulation)
        self.status_cancel_button.setEnabled(False)
        self.status_cancel_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.status_cancel_button)

        self.status_preview_checkbox = QCheckBox("Preview")
        self.status_preview_checkbox.setChecked(self.preview_checkbox.isChecked())
        self.status_preview_checkbox.toggled.connect(lambda val: self._sync_preview_checkboxes(val))
        layout.addWidget(self.status_preview_checkbox)

        self.status_progress = QProgressBar()
        self.status_progress.setFixedWidth(180)
        self.status_progress.setValue(self.progress_bar.value())
        layout.addWidget(self.status_progress)

        layout.addStretch()
        bar.setLayout(layout)
        self.statusBar().addPermanentWidget(bar, 1)
        self.status_run_bar = bar

    def _build_toolbar_run_bar(self):
        """Add a top toolbar with run controls to keep them visible."""
        tb = QToolBar("Run Controls")
        tb.setMovable(False)
        tb.setFloatable(False)

        self.toolbar_run_action = tb.addAction("Run Simulation")
        self.toolbar_run_action.triggered.connect(self.run_simulation)

        self.toolbar_cancel_action = tb.addAction("Cancel")
        self.toolbar_cancel_action.triggered.connect(self.cancel_simulation)
        self.toolbar_cancel_action.setEnabled(False)

        self.toolbar_preview_action = tb.addAction("Preview")
        self.toolbar_preview_action.setCheckable(True)
        self.toolbar_preview_action.setChecked(self.preview_checkbox.isChecked())
        self.toolbar_preview_action.toggled.connect(lambda val: self._sync_preview_checkboxes(val))

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tb.addWidget(spacer)

        self.toolbar_progress = QProgressBar()
        self.toolbar_progress.setFixedWidth(160)
        self.toolbar_progress.setValue(self.progress_bar.value())
        tb.addWidget(self.toolbar_progress)

        self.addToolBar(tb)
        self.toolbar_run_bar = tb
        
    def run_simulation(self):
        """Run the Bloch simulation."""
        self.statusBar().showMessage("Running simulation...")
        self.simulate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        if hasattr(self, "status_run_button"):
            self.status_run_button.setEnabled(False)
        if hasattr(self, "status_cancel_button"):
            self.status_cancel_button.setEnabled(True)
        if hasattr(self, "toolbar_run_action"):
            self.toolbar_run_action.setEnabled(False)
        if hasattr(self, "toolbar_cancel_action"):
            self.toolbar_cancel_action.setEnabled(True)
        self.progress_bar.setValue(0)
        if hasattr(self, "status_progress"):
            self.status_progress.setValue(0)
        if hasattr(self, "toolbar_progress"):
            self.toolbar_progress.setValue(0)
        # Keep preview toggles in sync once more in case caller only touched the status bar
        if hasattr(self, "status_preview_checkbox"):
            self._sync_preview_checkboxes(self.status_preview_checkbox.isChecked())

        self.log_message("Starting simulation...")
        
        # Get parameters
        tissue = self.tissue_widget.get_parameters()
        pulse = self.rf_designer.get_pulse()
        dt_s = max(self.time_step_spin.value(), 0.1) * 1e-6
        sequence_tuple = self.sequence_designer.compile_sequence(custom_pulse=pulse, dt=dt_s, log_info=True)
        # Optionally extend sequence with a zero tail to keep sampling after gradients/RF
        tail_ms = self.extra_tail_spin.value()
        b1_orig_len = len(sequence_tuple[0])
        sequence_tuple = self._extend_sequence_with_tail(sequence_tuple, tail_ms, dt_s)
        if len(sequence_tuple[0]) > b1_orig_len:
            added = len(sequence_tuple[0]) - b1_orig_len
            self.log_message(f"Appended {tail_ms:.3f} ms tail ({added} pts) after sequence.")

        # Always work with explicit arrays so pulse visualization matches the run
        b1_arr, gradients_arr, time_arr = sequence_tuple
        if not self._sweep_mode:
            self.sequence_designer._render_sequence_diagram(b1_arr, gradients_arr, time_arr)
        self.last_b1 = np.asarray(b1_arr)
        self.last_gradients = np.asarray(gradients_arr)
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
        preview_on = self.preview_checkbox.isChecked() or (hasattr(self, "status_preview_checkbox") and self.status_preview_checkbox.isChecked())
        if preview_on:
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
        if hasattr(self, "status_progress"):
            self.simulation_thread.progress.connect(self.status_progress.setValue)
        self.simulation_thread.start()

    def cancel_simulation(self):
        """Request cancellation of the current simulation."""
        if hasattr(self, "simulation_thread") and self.simulation_thread.isRunning():
            self.simulation_thread.request_cancel()
            self.statusBar().showMessage("Cancellation requested...")
            self.log_message("User requested simulation cancel.")
        if hasattr(self, "status_cancel_button"):
            self.status_cancel_button.setEnabled(False)

    def on_simulation_finished(self, result):
        """Handle simulation completion."""
        self.statusBar().showMessage("Simulation complete")
        self.simulate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(100)
        if hasattr(self, "status_run_button"):
            self.status_run_button.setEnabled(True)
        if hasattr(self, "status_cancel_button"):
            self.status_cancel_button.setEnabled(False)
        if hasattr(self, "status_progress"):
            self.status_progress.setValue(100)
        if hasattr(self, "toolbar_run_action"):
            self.toolbar_run_action.setEnabled(True)
        if hasattr(self, "toolbar_cancel_action"):
            self.toolbar_cancel_action.setEnabled(False)
        if hasattr(self, "toolbar_progress"):
            self.toolbar_progress.setValue(100)
        self.log_message("Simulation completed successfully.")
        
        # Update plots
        if result.get('mx', {}).ndim > 2: # Time-resolved
            self._precompute_plot_ranges(result)
        self.mag_3d.last_positions = self.last_positions
        self.mag_3d.last_frequencies = self.last_frequencies
        self.update_plots(result)
        self.log_message("Magnetization plots show Mx/My/Mz over time; Signal shows received complex signal (per frequency).")
        
    def on_simulation_error(self, error_msg):
        """Handle simulation error."""
        self.statusBar().showMessage("Simulation failed")
        self.simulate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(0)
        if hasattr(self, "status_run_button"):
            self.status_run_button.setEnabled(True)
        if hasattr(self, "status_cancel_button"):
            self.status_cancel_button.setEnabled(False)
        if hasattr(self, "status_progress"):
            self.status_progress.setValue(0)
        if hasattr(self, "toolbar_run_action"):
            self.toolbar_run_action.setEnabled(True)
        if hasattr(self, "toolbar_cancel_action"):
            self.toolbar_cancel_action.setEnabled(False)
        if hasattr(self, "toolbar_progress"):
            self.toolbar_progress.setValue(0)
        self.log_message(f"Simulation error: {error_msg}")
        
        QMessageBox.critical(self, "Simulation Error", error_msg)

    def on_simulation_cancelled(self):
        """Handle user cancellation."""
        self.statusBar().showMessage("Simulation cancelled")
        self.simulate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(0)
        if hasattr(self, "status_run_button"):
            self.status_run_button.setEnabled(True)
        if hasattr(self, "status_cancel_button"):
            self.status_cancel_button.setEnabled(False)
        if hasattr(self, "status_progress"):
            self.status_progress.setValue(0)
        if hasattr(self, "toolbar_run_action"):
            self.toolbar_run_action.setEnabled(True)
        if hasattr(self, "toolbar_cancel_action"):
            self.toolbar_cancel_action.setEnabled(False)
        if hasattr(self, "toolbar_progress"):
            self.toolbar_progress.setValue(0)
        self.log_message("Simulation cancelled.")
        
    def update_plots(self, result):
        """Update all visualization plots."""
        # Invalidate plot caches when new simulation results arrive
        self._invalidate_plot_caches()

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
        # Precompute final-spectrum range for consistent y-limits
        try:
            self._compute_final_spectrum_range(signal_arr, self.last_time)
        except Exception:
            self._spectrum_final_range = None

        # Reset sliders to default values (0 Hz, 0 cm) if this is a new simulation
        # Use helper to find index closest to 0
        def _get_zero_idx(arr, length):
            if arr is None or len(arr) == 0:
                return 0
            if len(arr) != length:
                return 0
            return int(np.argmin(np.abs(arr)))

        # Default frequencies
        f_idx = _get_zero_idx(self.last_frequencies, freq_len)
        if hasattr(self, 'spatial_freq_slider'):
             self.spatial_freq_slider.setValue(f_idx)
        
        # Default positions
        p_idx = _get_zero_idx(self.last_positions[:, 2] * 100 if self.last_positions is not None else None, pos_len)
        if hasattr(self, 'spectrum_pos_slider'):
             self.spectrum_pos_slider.setValue(p_idx)
        
        # Default view selectors
        if hasattr(self, 'mag_view_selector'):
             # If showing frequencies, default to 0 Hz index. If showing positions, default to 0 cm index.
             # Logic is handled dynamically in update_mag_selector, but we can preset value
             # We just set it to 0 as a safe default, or the relevant zero index if we knew the mode.
             # Since mode can change, resetting to 0 is often safe, but specific zero-index is better.
             # Let's just reset to 0 for now as 'All spins' usually starts at 0.
             self.mag_view_selector.setValue(0)
             
        if hasattr(self, 'signal_view_selector'):
             self.signal_view_selector.setValue(0)

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

        # Skip heavy plotting during parameter sweeps; keep data cached only
        if getattr(self, "_sweep_mode", False):
            self._spectrum_needs_update = True
            self._spatial_needs_update = True
            return
        
        # Handle different result shapes
        if mx_arr.ndim == 2:
            # Endpoint mode: show last values as flat lines
            self.log_message("Endpoint mode: showing final magnetization values.")
            mx_all = mx_arr  # shape (npos, nfreq)
            my_all = my_arr
            mz_all = mz_arr
            t_plot = np.array([time[0], time[-1]]) if len(time) > 1 else np.array([0, 1])

            self._safe_clear_plot(self.mxy_plot)
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

            self._safe_clear_plot(self.mz_plot)
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
            self._safe_clear_plot(self.signal_plot)
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

            # Spatial plot for endpoint
            self.update_spatial_plot_from_last_result()

            # Hide universal time control in endpoint mode
            self.playback_indices = None
            self.playback_time = None
            self.playback_time_ms = None
            self.anim_vectors_full = None
            self.time_control.set_time_range(None)
            self.time_control.setEnabled(False)
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
            mz_all = mz_arr.copy() # Make a copy to avoid modifying cache
            ntime, npos, nfreq = mx_all.shape
            mean_only = self.mean_only_checkbox.isChecked()
            # self.anim_time = time # This is handled by playback_time_ms

            # Update magnetization plots
            self._update_mag_selector_limits(npos, nfreq, disable=mean_only)
            self._update_signal_selector_limits(npos, nfreq, disable=mean_only)
            view_mode, selector = self._current_mag_filter(npos, nfreq)

            self._safe_clear_plot(self.mxy_plot)
            if mean_only:
                self.mxy_plot.setTitle("Mean Transverse Magnetization")
                self._reset_legend(self.mxy_plot, "mxy_legend", False)
                mean_mx = np.mean(mx_all, axis=(1, 2))
                mean_my = np.mean(my_all, axis=(1, 2))
                self.mxy_plot.plot(time, mean_mx, pen=pg.mkPen('c', width=4), name="Mean Mx")
                self.mxy_plot.plot(time, mean_my, pen=pg.mkPen('c', width=4, style=Qt.DashLine), name="Mean My")
            else:
                if view_mode == "Positions @ freq":
                    fi = min(selector, nfreq - 1)
                    freq_val = self.last_frequencies[fi] if self.last_frequencies is not None and fi < len(self.last_frequencies) else fi
                    self.mxy_plot.setTitle(f"Mx/My vs Time for all Positions @ Freq: {freq_val:.1f} Hz")
                    total_series = npos
                    indices_to_plot = self._get_trace_indices_to_plot(total_series)
                    self._reset_legend(self.mxy_plot, "mxy_legend", total_series > 1)
                    for pi in indices_to_plot:
                        color = self._color_for_index(pi, total_series)
                        self.mxy_plot.plot(time, mx_all[:, pi, fi], pen=pg.mkPen(color, width=2), name=f"p{pi} Mx @ f{fi}")
                        self.mxy_plot.plot(time, my_all[:, pi, fi], pen=pg.mkPen(color, style=Qt.DashLine, width=2), name=f"p{pi} My @ f{fi}")
                    mean_mx = np.mean(mx_all[:, :, fi], axis=1)
                    mean_my = np.mean(my_all[:, :, fi], axis=1)
                elif view_mode == "Freqs @ position":
                    pi = min(selector, npos - 1)
                    pos_val = self.last_positions[pi, 2] * 100 if self.last_positions is not None and pi < len(self.last_positions) else pi
                    self.mxy_plot.setTitle(f"Mx/My vs Time for all Frequencies @ Position: {pos_val:.2f} cm")
                    total_series = nfreq
                    indices_to_plot = self._get_trace_indices_to_plot(total_series)
                    self._reset_legend(self.mxy_plot, "mxy_legend", total_series > 1)
                    for fi in indices_to_plot:
                        color = self._color_for_index(fi, total_series)
                        self.mxy_plot.plot(time, mx_all[:, pi, fi], pen=pg.mkPen(color, width=2), name=f"f{fi} Mx @ p{pi}")
                        self.mxy_plot.plot(time, my_all[:, pi, fi], pen=pg.mkPen(color, style=Qt.DashLine, width=2), name=f"f{fi} My @ p{pi}")
                    mean_mx = np.mean(mx_all[:, pi, :], axis=1)
                    mean_my = np.mean(my_all[:, pi, :], axis=1)
                else:
                    total_series = npos * nfreq
                    self.mxy_plot.setTitle("Mx/My vs Time (All Spins)")
                    indices_to_plot = self._get_trace_indices_to_plot(total_series)
                    self._reset_legend(self.mxy_plot, "mxy_legend", len(indices_to_plot) > 1)

                    # Log performance optimization if downsampling
                    if len(indices_to_plot) < total_series:
                        self.log_message(f"Performance: Plotting {len(indices_to_plot)}/{total_series} traces (limit: {self.max_traces_spin.value()})")

                    # Plot only selected subset
                    for plot_idx, linear_idx in enumerate(indices_to_plot):
                        pi = linear_idx // nfreq
                        fi = linear_idx % nfreq
                        color = self._color_for_index(linear_idx, total_series)
                        self.mxy_plot.plot(time, mx_all[:, pi, fi], pen=pg.mkPen(color, width=2), name=f"p{pi} f{fi} Mx")
                        self.mxy_plot.plot(time, my_all[:, pi, fi], pen=pg.mkPen(color, style=Qt.DashLine, width=2), name=f"p{pi} f{fi} My")

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

            self._safe_clear_plot(self.mz_plot)
            if mean_only:
                self.mz_plot.setTitle("Mean Longitudinal Magnetization")
                self._reset_legend(self.mz_plot, "mz_legend", False)
                mean_mz = np.mean(mz_all, axis=(1, 2))
                self.mz_plot.plot(time, mean_mz, pen=pg.mkPen('c', width=4), name="Mean Mz")
            else:
                if view_mode == "Positions @ freq":
                    freq_val = self.last_frequencies[fi] if self.last_frequencies is not None and fi < len(self.last_frequencies) else fi
                    self.mz_plot.setTitle(f"Mz vs Time for all Positions @ Freq: {freq_val:.1f} Hz")
                    self._reset_legend(self.mz_plot, "mz_legend", total_series > 1)
                    indices_to_plot = self._get_trace_indices_to_plot(total_series)
                    for pi in indices_to_plot:
                        color = self._color_for_index(pi, total_series)
                        self.mz_plot.plot(time, mz_all[:, pi, fi], pen=pg.mkPen(color, width=2), name=f"p{pi} Mz @ f{fi}")
                    mean_mz = np.mean(mz_all[:, :, fi], axis=1)
                elif view_mode == "Freqs @ position":
                    pos_val = self.last_positions[pi, 2] * 100 if self.last_positions is not None and pi < len(self.last_positions) else pi
                    self.mz_plot.setTitle(f"Mz vs Time for all Frequencies @ Position: {pos_val:.2f} cm")
                    self._reset_legend(self.mz_plot, "mz_legend", total_series > 1)
                    indices_to_plot = self._get_trace_indices_to_plot(total_series)
                    for fi in indices_to_plot:
                        color = self._color_for_index(fi, total_series)
                        self.mz_plot.plot(time, mz_all[:, pi, fi], pen=pg.mkPen(color, width=2), name=f"f{fi} Mz @ p{pi}")
                    mean_mz = np.mean(mz_all[:, pi, :], axis=1)
                else:
                    # Use same downsampled indices from Mxy plot
                    self.mz_plot.setTitle("Mz vs Time (All Spins)")
                    self._reset_legend(self.mz_plot, "mz_legend", len(indices_to_plot) > 1)

                    for plot_idx, linear_idx in enumerate(indices_to_plot):
                        pi = linear_idx // nfreq
                        fi = linear_idx % nfreq
                        color = self._color_for_index(linear_idx, total_series)
                        self.mz_plot.plot(time, mz_all[:, pi, fi], pen=pg.mkPen(color, width=2), name=f"p{pi} f{fi} Mz")

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
            self._safe_clear_plot(self.signal_plot)

            # Apply view filter to signal
            if mean_only:
                self.signal_plot.setTitle("Mean Signal Evolution")
                sig_to_plot = np.mean(signal, axis=(1, 2))
                total_series_sig = 1
                indices_to_plot_sig = [0]
            else:
                if view_mode == "Positions @ freq":
                    fi = min(selector, nfreq - 1)
                    freq_val = self.last_frequencies[fi] if self.last_frequencies is not None and fi < len(self.last_frequencies) else fi
                    self.signal_plot.setTitle(f"Signal vs Time for all Positions @ Freq: {freq_val:.1f} Hz")
                    sig_to_plot = signal[:, :, fi] # (ntime, npos)
                    total_series_sig = npos
                elif view_mode == "Freqs @ position":
                    pi = min(selector, npos - 1)
                    pos_val = self.last_positions[pi, 2] * 100 if self.last_positions is not None and pi < len(self.last_positions) else pi
                    self.signal_plot.setTitle(f"Signal vs Time for all Frequencies @ Position: {pos_val:.2f} cm")
                    sig_to_plot = signal[:, pi, :] # (ntime, nfreq)
                    total_series_sig = nfreq
                else: # All
                    self.signal_plot.setTitle("Signal vs Time (All Spins)")
                    sig_to_plot = signal.reshape(ntime, -1) # (ntime, npos*nfreq)
                    total_series_sig = npos * nfreq
                
                indices_to_plot_sig = self._get_trace_indices_to_plot(total_series_sig)

            if mean_only:
                self._reset_legend(self.signal_plot, "signal_legend", False)
                mean_sig = sig_to_plot
                self.signal_plot.plot(time, np.abs(mean_sig), pen=pg.mkPen('c', width=4), name='|S| mean')
            else:
                self._reset_legend(self.signal_plot, "signal_legend", len(indices_to_plot_sig) > 1)

                # Plot only selected subset (3 lines each: magnitude, real, imaginary)
                for plot_idx, linear_idx in enumerate(indices_to_plot_sig):
                    color = self._color_for_index(linear_idx, total_series_sig)
                    trace = sig_to_plot[:, linear_idx]
                    name_prefix = f"Trace {linear_idx}"
                    self.signal_plot.plot(time, np.abs(trace), pen=color, name=f'|S| {name_prefix}')
                    self.signal_plot.plot(time, np.real(trace), pen=pg.mkPen(color, style=Qt.DashLine), name=f'Re {name_prefix}')
                    self.signal_plot.plot(time, np.imag(trace), pen=pg.mkPen(color, style=Qt.DotLine), name=f'Im {name_prefix}')

                mean_sig = np.mean(sig_to_plot, axis=1)
                self.signal_plot.plot(time, np.abs(mean_sig), pen=pg.mkPen('c', width=4), name='|S| mean')
            self._apply_pulse_region(self.signal_plot, "signal_region")
            sig_ymin, sig_ymax = self._calc_symmetric_limits(
                np.abs(signal), np.real(signal), np.imag(signal), np.abs(mean_sig),
                base=self.initial_mz
            )
            self._set_plot_ranges(self.signal_plot, x_min, x_max, sig_ymin, sig_ymax)

        # Update spectrum using the shared helper (avoids SciPy dependency)
        try:
            self._refresh_spectrum(time_idx=len(time) - 1, skip_fft=False)
            self._spectrum_needs_update = False
        except Exception as exc:
            self.log_message(f"Spectrum update failed: {exc}")

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
        self._reset_playback_anchor(0)
        self.time_control.setEnabled(True)

        # Ensure signal tab x-range set even if spectrum-only
        self.signal_plot.setXRange(x_min, x_max, padding=0)

        # Update heatmaps if they are the active view (default)
        if hasattr(self, 'mag_plot_type') and self.mag_plot_type.currentText() == "Heatmap":
            self._update_mag_heatmaps()
        if hasattr(self, 'signal_plot_type') and self.signal_plot_type.currentText() == "Heatmap":
            self._update_signal_heatmaps()

    def _precompute_plot_ranges(self, result):
        """Pre-calculate stable Y-ranges for plots based on the full dataset."""
        mx = result.get('mx')
        my = result.get('my')
        mz = result.get('mz')
        signal = result.get('signal')
        initial_mag = self.initial_mz

        # Spatial plot ranges
        if mx is not None and my is not None and mz is not None:
            # Mxy (magnitude and components)
            max_abs_mxy = float(np.nanmax(np.abs(np.sqrt(mx**2 + my**2))))
            max_abs_mx = float(np.nanmax(np.abs(mx)))
            max_abs_my = float(np.nanmax(np.abs(my)))
            mxy_limit = max(initial_mag, max_abs_mxy, max_abs_mx, max_abs_my) * 1.05
            self.spatial_mxy_yrange = (-mxy_limit, mxy_limit)

            # Mz
            max_abs_mz = float(np.nanmax(np.abs(mz)))
            mz_limit = max(initial_mag, max_abs_mz) * 1.05
            self.spatial_mz_yrange = (-mz_limit, mz_limit) if np.nanmin(mz) < -1e-6 else (-0.05 * mz_limit, mz_limit)

        # Spectrum plot ranges
        if signal is not None and self.last_time is not None and len(self.last_time) > 1:
            try:
                dt = self.last_time[1] - self.last_time[0]
                sig_mean = np.mean(signal, axis=tuple(range(1, signal.ndim)))
                n_fft = self._spectrum_fft_len(len(sig_mean))
                spec = np.fft.fftshift(np.fft.fft(sig_mean, n=n_fft))
                max_abs_spec = float(np.nanmax(np.abs(spec)))
                spec_limit = max_abs_spec * 1.05
                self.spectrum_yrange = (-spec_limit, spec_limit)

            except Exception:
                self.spectrum_yrange = (-1.1, 1.1)
                self.off_res_spectrum_yrange = (-1.1, 1.1)

            # Also pre-calculate range for off-resonant spins view
            if signal is not None:
                max_abs_signal = float(np.nanmax(np.abs(signal)))
                self.off_res_spectrum_yrange = (-max_abs_signal * 1.1, max_abs_signal * 1.1)
            else:
                self.off_res_spectrum_yrange = (-1.1, 1.1)
        else:
            self.spectrum_yrange = (-1.1, 1.1)
            self.off_res_spectrum_yrange = (-1.1, 1.1)

    def load_parameters(self):
        """Load simulation parameters from file."""
        export_dir = self._get_export_directory()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Parameters", str(export_dir), "JSON Files (*.json);;All Files (*)"
        )
        if not filename:
            return

        try:
            with open(filename, 'r') as f:
                state = json.load(f)

            # 1. Restore Tissue Parameters
            t_state = state.get("tissue", {})
            self.tissue_widget.preset_combo.blockSignals(True)
            self.tissue_widget.field_combo.blockSignals(True)
            self.tissue_widget.preset_combo.setCurrentText(t_state.get("preset", "Custom"))
            self.tissue_widget.field_combo.setCurrentText(t_state.get("field", "3.0T"))
            self.tissue_widget.preset_combo.blockSignals(False)
            self.tissue_widget.field_combo.blockSignals(False)
            
            self.tissue_widget.t1_spin.setValue(t_state.get("t1_ms", 1000))
            self.tissue_widget.t2_spin.setValue(t_state.get("t2_ms", 100))
            self.tissue_widget.t2s_spin.setValue(t_state.get("t2s_ms", 50))
            self.tissue_widget.pd_spin.setValue(t_state.get("pd", 1.0))
            self.tissue_widget.m0_spin.setValue(t_state.get("m0", 1.0))

            # 2. Restore RF Pulse Parameters
            if "rf" in state:
                self.rf_designer.set_state(state["rf"])

            # 3. Restore Sequence Parameters
            s_state = state.get("sequence", {})
            # Disable preset auto-loading to prevent overwriting loaded values
            old_presets_enabled = self.tissue_widget.sequence_presets_enabled
            self.tissue_widget.sequence_presets_enabled = False
            
            try:
                self.sequence_designer.sequence_type.setCurrentText(s_state.get("type", "Free Induction Decay"))
                self.sequence_designer.te_spin.setValue(s_state.get("te", 10))
                self.sequence_designer.tr_spin.setValue(s_state.get("tr", 100))
                self.sequence_designer.ti_spin.setValue(s_state.get("ti", 400))
                self.sequence_designer.spin_echo_echoes.setValue(s_state.get("echo_count", 1))
                self.sequence_designer.slice_thickness_spin.setValue(s_state.get("slice_thickness", 5.0))
                self.sequence_designer.slice_gradient_spin.setValue(s_state.get("slice_gradient", 0.0))
                
                # SSFP
                self.sequence_designer.ssfp_repeats.setValue(s_state.get("ssfp_repeats", 16))
                self.sequence_designer.ssfp_amp.setValue(s_state.get("ssfp_amp", 0.05))
                self.sequence_designer.ssfp_phase.setValue(s_state.get("ssfp_phase", 0.0))
                self.sequence_designer.ssfp_dur.setValue(s_state.get("ssfp_dur", 1.0))
                self.sequence_designer.ssfp_start_delay.setValue(s_state.get("ssfp_start_delay", 0.0))
                self.sequence_designer.ssfp_start_amp.setValue(s_state.get("ssfp_start_amp", 0.025))
                self.sequence_designer.ssfp_start_phase.setValue(s_state.get("ssfp_start_phase", 180.0))
                self.sequence_designer.ssfp_alternate_phase.setChecked(s_state.get("ssfp_alternate", True))
                
                # Rephase
                self.sequence_designer.rephase_percentage.setValue(s_state.get("rephase_pct", 50.0))
            finally:
                self.tissue_widget.sequence_presets_enabled = old_presets_enabled

            # 4. Restore Simulation Grid
            sim_state = state.get("simulation", {})
            self.mode_combo.setCurrentText(sim_state.get("mode", "Time-resolved"))
            self.pos_spin.setValue(sim_state.get("num_pos", 1))
            self.pos_range.setValue(sim_state.get("pos_range", 2.0))
            self.freq_spin.setValue(sim_state.get("num_freq", 31))
            self.freq_range.setValue(sim_state.get("freq_range", 100.0))
            self.time_step_spin.setValue(sim_state.get("time_step", 1.0))
            self.extra_tail_spin.setValue(sim_state.get("extra_tail", 5.0))
            self.max_traces_spin.setValue(sim_state.get("max_traces", 50))

            self.log_message(f"Parameters loaded successfully from {Path(filename).name}")
            self.statusBar().showMessage(f"Parameters loaded from {Path(filename).name}")
            
            # Trigger diagram update
            self.sequence_designer.update_diagram()

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load parameters:\n{str(e)}")
            self.log_message(f"Error loading parameters: {e}")

    def _start_vector_animation(self):
        """Start or restart the 3D vector animation if data exists."""
        # Prevent animation during parameter sweeps
        if getattr(self, "_sweep_mode", False):
            self._sync_play_toggle(False)
            return

        if self.anim_data is None or len(self.anim_data) == 0:
            self.anim_timer.stop()
            self._sync_play_toggle(False)
            return
        if self.mag_3d.track_path:
            self.mag_3d._clear_path()
        if self.anim_index >= len(self.anim_data):
            self.anim_index = 0
        self._reset_playback_anchor(self.anim_index)

        # CRITICAL: Disable updates on Magnetization and Signal plots during animation
        # This prevents pyqtgraph from rendering them every frame, which causes slowness
        self.mxy_plot.setUpdatesEnabled(False)
        self.mz_plot.setUpdatesEnabled(False)
        self.signal_plot.setUpdatesEnabled(False)

        # Always recompute interval using current speed control
        self._recompute_anim_interval(self.time_control.speed_spin.value() if hasattr(self, 'time_control') else None)
        self.anim_timer.start(self.anim_interval_ms)
        self._sync_play_toggle(True)

    def _resume_vector_animation(self):
        """Resume playback of the 3D vector."""
        self._start_vector_animation()

    def _pause_vector_animation(self):
        """Pause the 3D vector animation."""
        self.anim_timer.stop()
        self._sync_play_toggle(False)

        # Re-enable updates on Magnetization and Signal plots after animation stops
        self.mxy_plot.setUpdatesEnabled(True)
        self.mz_plot.setUpdatesEnabled(True)
        self.signal_plot.setUpdatesEnabled(True)

        # When paused, refresh plots with current frame (full update)
        if hasattr(self, 'anim_index'):
            self._on_universal_time_changed(self.anim_index, skip_expensive_updates=False)

    def _reset_vector_animation(self):
        """Reset the 3D vector to the first available frame."""
        self.anim_timer.stop()
        self._sync_play_toggle(False)

        # Re-enable updates on Magnetization and Signal plots after animation stops
        self.mxy_plot.setUpdatesEnabled(True)
        self.mz_plot.setUpdatesEnabled(True)
        self.signal_plot.setUpdatesEnabled(True)

        self.anim_index = 0
        self.mag_3d._clear_path()
        self._reset_playback_anchor(0)
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
        interval_ms = max(min_interval, desired_interval_ms)
        self.anim_interval_ms = max(1, int(round(interval_ms)))
        self._frame_step = 1

    def _update_playback_speed(self, sim_ms_per_s: float):
        """Adjust playback speed (simulation ms per real second)."""
        self._recompute_anim_interval(sim_ms_per_s)
        self._reset_playback_anchor(self.anim_index)
        if self.anim_timer.isActive():
            self.anim_timer.start(self.anim_interval_ms)

    def _set_animation_index_from_slider(self, idx: int, reset_anchor=True):
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
        if reset_anchor:
            self._reset_playback_anchor(idx)

    def _animate_vector(self):
        """Advance the 3D vector animation if data is available."""
        if self.anim_data is None or self.playback_time_ms is None or len(self.playback_time_ms) == 0:
            return
        now = time.monotonic()
        if self._last_render_wall is not None:
            delta_ms = (now - self._last_render_wall) * 1000.0
            if delta_ms < self._min_render_interval_ms:
                return
        if self._playback_anchor_wall is None or self._playback_anchor_time_ms is None:
            self._reset_playback_anchor(self.anim_index)
        if len(self.playback_time_ms) == 1:
            self.time_control.set_time_index(0)
            self._on_universal_time_changed(0)
            return

        elapsed_s = max(0.0, now - (self._playback_anchor_wall or now))
        sim_ms_per_s = max(self.time_control.speed_spin.value(), 0.001)

        start_ms = float(self.playback_time_ms[0])
        end_ms = float(self.playback_time_ms[-1])
        duration_ms = max(end_ms - start_ms, 1e-9)

        target_ms = float(self._playback_anchor_time_ms or start_ms) + elapsed_s * sim_ms_per_s
        wrapped = False
        if target_ms > end_ms:
            # Loop seamlessly: wrap to start while preserving speed
            target_rel = (target_ms - start_ms) % duration_ms
            target_ms = start_ms + target_rel
            wrapped = True
            self._playback_anchor_wall = now
            self._playback_anchor_time_ms = target_ms

        idx = int(np.searchsorted(self.playback_time_ms, target_ms, side="left"))
        idx = min(max(idx, 0), len(self.playback_time_ms) - 1)
        if idx > 0:
            prev_ms = float(self.playback_time_ms[idx - 1])
            curr_ms = float(self.playback_time_ms[idx])
            if abs(target_ms - prev_ms) < abs(curr_ms - target_ms):
                idx -= 1

        if wrapped and self.mag_3d.track_path:
            self.mag_3d._clear_path()

        # Move universal time control (label/slider) then propagate to all views
        self.time_control.set_time_index(idx)
        # During animation, skip expensive plot redraws (spatial FFT, spectrum FFT)
        # Only update time cursors and 3D view for smooth playback
        self._on_universal_time_changed(idx, skip_expensive_updates=True, reset_anchor=False)
        self._last_render_wall = now

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
        export_dir = self._get_export_directory()
        seq_type = self.sequence_designer.sequence_type.currentText().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "").lower()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_filename = f"bloch_params_{seq_type}_{timestamp}.json"
        default_path = export_dir / default_filename

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", str(default_path), "JSON Files (*.json)"
        )
        if not filename:
            return

        try:
            state = {
                "version": "1.1",
                "timestamp": timestamp,
                "tissue": {
                    "preset": self.tissue_widget.preset_combo.currentText(),
                    "field": self.tissue_widget.field_combo.currentText(),
                    "t1_ms": self.tissue_widget.t1_spin.value(),
                    "t2_ms": self.tissue_widget.t2_spin.value(),
                    "t2s_ms": self.tissue_widget.t2s_spin.value(),
                    "pd": self.tissue_widget.pd_spin.value(),
                    "m0": self.tissue_widget.m0_spin.value(),
                },
                "rf": self.rf_designer.get_state(),
                "sequence": {
                    "type": self.sequence_designer.sequence_type.currentText(),
                    "te": self.sequence_designer.te_spin.value(),
                    "tr": self.sequence_designer.tr_spin.value(),
                    "ti": self.sequence_designer.ti_spin.value(),
                    "echo_count": self.sequence_designer.spin_echo_echoes.value(),
                    "slice_thickness": self.sequence_designer.slice_thickness_spin.value(),
                    "slice_gradient": self.sequence_designer.slice_gradient_spin.value(),
                    "ssfp_repeats": self.sequence_designer.ssfp_repeats.value(),
                    "ssfp_amp": self.sequence_designer.ssfp_amp.value(),
                    "ssfp_phase": self.sequence_designer.ssfp_phase.value(),
                    "ssfp_dur": self.sequence_designer.ssfp_dur.value(),
                    "ssfp_start_delay": self.sequence_designer.ssfp_start_delay.value(),
                    "ssfp_start_amp": self.sequence_designer.ssfp_start_amp.value(),
                    "ssfp_start_phase": self.sequence_designer.ssfp_start_phase.value(),
                    "ssfp_alternate": self.sequence_designer.ssfp_alternate_phase.isChecked(),
                    "rephase_pct": self.sequence_designer.rephase_percentage.value(),
                },
                "simulation": {
                    "mode": self.mode_combo.currentText(),
                    "num_pos": self.pos_spin.value(),
                    "pos_range": self.pos_range.value(),
                    "num_freq": self.freq_spin.value(),
                    "freq_range": self.freq_range.value(),
                    "time_step": self.time_step_spin.value(),
                    "extra_tail": self.extra_tail_spin.value(),
                    "max_traces": self.max_traces_spin.value(),
                }
            }

            with open(filename, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.log_message(f"Parameters saved to {Path(filename).name}")
            self.statusBar().showMessage(f"Parameters saved to {Path(filename).name}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save parameters:\n{str(e)}")
            self.log_message(f"Error saving parameters: {e}")
                
    def export_results(self):
        """Export simulation results and parameters using the new multi-format dialog."""
        if self.last_result is None and not hasattr(self, 'last_b1'):
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return

        # Create the dialog
        dialog = ExportDataDialog(
            self, 
            default_filename="simulation_results", 
            default_directory=self._get_export_directory()
        )

        if dialog.exec_() != QDialog.Accepted:
            return

        options = dialog.get_export_options()
        base_path = options['base_path']
        
        # Collect parameters
        sequence_params = self._collect_sequence_parameters()
        simulation_params = self._collect_simulation_parameters()
        
        # 1. HDF5 Export
        h5_path = f"{base_path}.h5"
        if options['hdf5']:
            try:
                self.simulator.save_results(h5_path, sequence_params, simulation_params)
                self.log_message(f"Exported HDF5: {h5_path}")
            except Exception as e:
                QMessageBox.critical(self, "HDF5 Export Error", f"Failed to export HDF5:\n{str(e)}")
                return # Stop if primary data export fails

        # 2. Notebook Exports
        if options['notebook_analysis'] or options['notebook_repro']:
            try:
                from .notebook_exporter import export_notebook
            except ImportError:
                QMessageBox.warning(self, "Missing Dependency", "Notebook export requires 'nbformat'.\nPlease install it: pip install nbformat")
            else:
                tissue_params = {
                    'name': self.tissue_widget.preset_combo.currentText(),
                    't1': self.tissue_widget.t1_spin.value() / 1000,
                    't2': self.tissue_widget.t2_spin.value() / 1000,
                    't2_star': self.tissue_widget.t2s_spin.value() / 1000,
                    'density': self.tissue_widget.pd_spin.value()
                }
                
                if options['notebook_analysis']:
                    nb_path = f"{base_path}_analysis.ipynb"
                    try:
                        export_notebook(
                            mode='load_data',
                            filename=nb_path,
                            sequence_params=sequence_params,
                            simulation_params=simulation_params,
                            tissue_params=tissue_params,
                            h5_filename=Path(h5_path).name, # Use relative path assuming same dir
                            title="Bloch Simulation Analysis"
                        )
                        self.log_message(f"Exported Analysis Notebook: {nb_path}")
                    except Exception as e:
                        self.log_message(f"Failed to export analysis notebook: {e}")

                if options['notebook_repro']:
                    nb_path = f"{base_path}_repro.ipynb"
                    wf_path = f"{base_path}_waveforms.npz"
                    rf_waveform = None
                    if hasattr(self, 'last_b1') and self.last_b1 is not None:
                        rf_waveform = (self.last_b1, self.last_time)
                    
                    try:
                        export_notebook(
                            mode='resimulate',
                            filename=nb_path,
                            sequence_params=sequence_params,
                            simulation_params=simulation_params,
                            tissue_params=tissue_params,
                            rf_waveform=rf_waveform,
                            title="Bloch Simulation - Reproducible",
                            waveform_filename=wf_path
                        )
                        self.log_message(f"Exported Repro Notebook: {nb_path}")
                        self.log_message(f"Exported Waveforms: {wf_path}")
                    except Exception as e:
                        self.log_message(f"Failed to export repro notebook: {e}")

        # 3. CSV/Text Export
        if options['csv']:
            fmt = options['csv_format']
            csv_path = f"{base_path}_data.{fmt}" if fmt != 'npy' else f"{base_path}_data.npy"
            try:
                # Use DatasetExporter
                exporter = DatasetExporter()
                
                # We need time, mx, my, mz from last_result
                if self.last_result:
                    time = self.last_result.get('time')
                    mx = self.last_result.get('mx')
                    my = self.last_result.get('my')
                    mz = self.last_result.get('mz')
                    pos = self.last_result.get('positions')
                    freq = self.last_result.get('frequencies')
                    
                    if time is not None and mx is not None:
                        exporter.export_magnetization(
                            time, mx, my, mz, pos, freq, csv_path, format=fmt
                        )
                        self.log_message(f"Exported Data ({fmt}): {csv_path}")
                    else:
                        self.log_message("Skipping CSV export: missing magnetization data.")
            except Exception as e:
                self.log_message(f"Failed to export CSV/Text data: {e}")

        QMessageBox.information(self, "Export Complete", f"Export process finished.\nFiles saved to: {Path(base_path).parent}")



    def _export_final_state_data(self):
        """Export the final magnetization state (Mx, My, Mz) for all positions/frequencies."""
        if self.last_result is None:
            return
            
        # Get data
        mx = self.last_result.get('mx')
        my = self.last_result.get('my')
        mz = self.last_result.get('mz')
        
        if mx is None:
            return

        # Extract final state
        if mx.ndim == 3: # (ntime, npos, nfreq)
            mx_final = mx[-1]
            my_final = my[-1]
            mz_final = mz[-1]
        else:
            mx_final = mx
            my_final = my
            mz_final = mz
            
        # Prompt for file
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Final State", "", "CSV Files (*.csv);;NumPy Archive (*.npz)"
        )
        if not filename:
            return
            
        path = Path(filename)
        if "csv" in selected_filter.lower():
            if path.suffix.lower() != ".csv":
                path = path.with_suffix(".csv")
            
            import csv
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Pos_Index", "Freq_Index", "Position_m", "Frequency_Hz", "Mx", "My", "Mz", "Mxy_Complex"])
                    
                    npos, nfreq = mx_final.shape
                    positions = self.last_positions[:, 2] if self.last_positions is not None else np.zeros(npos)
                    frequencies = self.last_frequencies if self.last_frequencies is not None else np.zeros(nfreq)
                    
                    for p in range(npos):
                        pos_val = positions[p] if p < len(positions) else 0
                        for f_idx in range(nfreq):
                            freq_val = frequencies[f_idx] if f_idx < len(frequencies) else 0
                            val_mx = float(mx_final[p, f_idx])
                            val_my = float(my_final[p, f_idx])
                            val_mz = float(mz_final[p, f_idx])
                            val_mxy_complex = complex(val_mx, val_my)
                            
                            writer.writerow([p, f_idx, pos_val, freq_val, val_mx, val_my, val_mz, str(val_mxy_complex)])
                            
                self.statusBar().showMessage(f"Final state exported to {path}")
                QMessageBox.information(self, "Export Successful", f"Final state data saved to:\n{path.name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export CSV:\n{str(e)}")
            
        else: # NPZ
            if path.suffix.lower() != ".npz":
                path = path.with_suffix(".npz")
            
            mxy_complex = mx_final + 1j * my_final
            try:
                np.savez(path, 
                        mx=mx_final, my=my_final, mz=mz_final,
                        mxy=mxy_complex,
                        positions=self.last_positions,
                        frequencies=self.last_frequencies)
                self.statusBar().showMessage(f"Final state exported to {path}")
                QMessageBox.information(self, "Export Successful", f"Final state data saved to:\n{path.name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export NPZ:\n{str(e)}")

    def _export_full_simulation_data(self):
        """Export full simulation data arrays."""
        if self.last_result is None:
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Full Simulation Data", "", "NumPy Archive (*.npz)"
        )
        if not filename:
            return
            
        path = Path(filename)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
            
        mx = self.last_result.get('mx')
        my = self.last_result.get('my')
        mxy = None
        if mx is not None and my is not None:
            mxy = mx + 1j * my

        try:
            # Save all arrays
            np.savez(path,
                    time=self.last_time,
                    mx=mx,
                    my=my,
                    mz=self.last_result.get('mz'),
                    mxy=mxy,
                    signal=self.last_result.get('signal'),
                    positions=self.last_positions,
                    frequencies=self.last_frequencies)
                    
            self.statusBar().showMessage(f"Full simulation data exported to {path}")
            QMessageBox.information(self, "Export Successful", f"Full simulation data saved to:\n{path.name}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export NPZ:\n{str(e)}")

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
            if hasattr(self, 'last_gradients'):
                params['gradients_waveform'] = self.last_gradients

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
        override = os.environ.get("BLOCH_EXPORT_DIR")
        if override:
            export_dir = Path(override).expanduser()
        else:
            export_dir = get_app_data_dir() / "exports"
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback to working directory if preferred location is unavailable
            export_dir = Path.cwd()
            export_dir.mkdir(parents=True, exist_ok=True)
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

    def _calculate_spectrum_data(self, time_idx=None, compute_fft=True):
        """Compute spectrum arrays used for plotting or export."""
        if self.last_result is None:
            return None
        signal = self.last_result.get('signal')
        if signal is None:
            return None
        time = self.last_time if self.last_time is not None else self.last_result.get('time', None)

        time = np.asarray(time)
        sig_arr = np.asarray(signal)
        if sig_arr.ndim == 1:
            sig_arr = sig_arr[:, None, None]
        elif sig_arr.ndim == 2:
            sig_arr = sig_arr[:, :, None]
        if time_idx is None:
            time_idx = sig_arr.shape[0] - 1
        time_idx = int(max(0, min(time_idx, sig_arr.shape[0] - 1)))
        sig_slice = sig_arr[:time_idx + 1]

        if time is None or len(time) < 2:
            time_slice = np.arange(sig_slice.shape[0])
            dt = 1.0
        else:
            time_slice = time[:time_idx + 1]
            if len(time_slice) < 2:
                return None
        dt = (time_slice[1] - time_slice[0]) # seconds per sample

        spectrum_mode = self.spectrum_view_mode.currentText() if hasattr(self, 'spectrum_view_mode') else "Mean over positions"
        pos_count = sig_slice.shape[1]

        # Fallback for original slider
        pos_sel = min(self.spectrum_pos_slider.value(), pos_count - 1) if pos_count > 0 else 0
        
        actual_pos_cm = 0.0
        if self.last_positions is not None and pos_sel < len(self.last_positions):
            actual_pos_cm = self.last_positions[pos_sel, 2] * 100
            self.spectrum_pos_label.setText(f"Pos: {actual_pos_cm:.3f} cm")
        else:
            self.spectrum_pos_label.setText(f"Pos idx: {pos_sel}")

        spectrum = None
        spec_mean = None
        freq = None

        if compute_fft:
            if spectrum_mode == "Mean over positions":
                sig_for_fft = np.mean(sig_slice, axis=tuple(range(1, sig_slice.ndim)))
                spec_mean = np.fft.fftshift(np.fft.fft(sig_for_fft, n=self._spectrum_fft_len(len(sig_for_fft))))
            elif spectrum_mode == "Mean + individuals":
                sig_for_fft = np.mean(sig_slice, axis=tuple(range(1, sig_slice.ndim)))
                n_fft = self._spectrum_fft_len(len(sig_for_fft))
                spec_mean = np.fft.fftshift(np.fft.fft(sig_for_fft, n=n_fft))
            else:
                sig_for_fft = np.mean(sig_slice[:, pos_sel, :], axis=1)
                spec_mean = None

            n_fft = self._spectrum_fft_len(len(sig_for_fft))
            spectrum = np.fft.fftshift(np.fft.fft(sig_for_fft, n=n_fft))
            freq = np.fft.fftshift(np.fft.fftfreq(n_fft, dt))

        return {
            "freq": freq,
            "spectrum": spectrum,
            "spec_mean": spec_mean,
            "mode": spectrum_mode,
            "pos_count": pos_count,
            "pos_sel": pos_sel,
            "time_idx": time_idx,
            "time_slice": time_slice,
            "signal_slice": sig_slice,
        }

    def _update_spectrum_heatmap(self, time_idx=None):
        """Render a heatmap of spectra across all position/frequency spins."""
        if self.last_result is None:
            return
        signal = self.last_result.get('signal')
        time = self.last_time if self.last_time is not None else self.last_result.get('time', None)
        if signal is None or time is None:
            self.log_message("Spectrum heatmap: missing signal or time axis")
            return
        
        # Determine heatmap mode
        mode_text = self.spectrum_heatmap_mode.currentText() if hasattr(self, "spectrum_heatmap_mode") else "Spin vs Frequency (FFT)"
        show_evolution = (mode_text == "Spin vs Time (Evolution)")

        sig_arr = np.asarray(signal)
        if sig_arr.ndim == 1:
            sig_arr = sig_arr[:, None, None]
        elif sig_arr.ndim == 2:
            sig_arr = sig_arr[:, :, None]

        ntime = sig_arr.shape[0]
        if ntime < 2:
            self.log_message("Spectrum heatmap: need at least two time points")
            return

        time = np.asarray(time)
        if time.size < 2:
            self.log_message("Spectrum heatmap: invalid time array")
            return

        if time_idx is None:
            time_idx = ntime - 1
        time_idx = int(max(1, min(time_idx, ntime - 1)))

        sig_slice = sig_arr[:time_idx + 1]  # (ntime, npos, nfreq)
        npos, nfreq = sig_arr.shape[1], sig_arr.shape[2]
        
        if npos > 1 and nfreq == 1:
            y_label = "Position index"
        elif npos == 1 and nfreq > 1:
            y_label = "Frequency index (spin off-res)"
        else:
            y_label = "Spin index (pos×freq)"

        if show_evolution:
            # Direct spin magnitudes over time (no FFT)
            time_ms = time[:time_idx + 1] * 1000.0
            
            # Reshape to (ntime, spin_count)
            mags = np.abs(sig_slice).reshape(sig_slice.shape[0], -1) 
            data = mags.T  # (spin_count, ntime)
            
            spin_count = data.shape[0]
            try:
                self.spectrum_heatmap_item.setImage(data, autoLevels=True, axisOrder='row-major')
                time_span = float(time_ms[-1] - time_ms[0]) if len(time_ms) > 1 else 1.0
                self.spectrum_heatmap_item.setRect(float(time_ms[0]), 0,
                                                   time_span if time_span != 0 else 1.0,
                                                   spin_count)
                self.spectrum_heatmap.setLabel('left', y_label)
                self.spectrum_heatmap.setLabel('bottom', 'Time', 'ms')
                self.spectrum_heatmap.setTitle("Temporal Evolution (|Mxy| over time)")
                self.spectrum_heatmap.setXRange(float(time_ms[0]), float(time_ms[-1]), padding=0)
                self.spectrum_heatmap.setYRange(0, spin_count, padding=0)
                if hasattr(self, "spectrum_heatmap_colorbar") and self.spectrum_heatmap_colorbar is not None:
                    finite_mag = data[np.isfinite(data)]
                    if finite_mag.size:
                        vmin = float(finite_mag.min())
                        vmax = float(finite_mag.max())
                        if np.isfinite(vmin) and np.isfinite(vmax) and vmax != vmin:
                            self.spectrum_heatmap_colorbar.setLevels((vmin, vmax))
            except Exception as exc:
                self.log_message(f"Spectrum heatmap update failed: {exc}")
        else:
            # FFT Mode: Stack of spectra
            dt = float(time[1] - time[0]) if len(time) > 1 else 1e-3
            n_fft = self._spectrum_fft_len(sig_slice.shape[0])
            freq_axis = np.fft.fftshift(np.fft.fftfreq(n_fft, dt))

            sig_flat = sig_slice.reshape(sig_slice.shape[0], -1)  # (ntime, spin)
            spec = np.fft.fftshift(np.fft.fft(sig_flat, n=n_fft, axis=0), axes=0)  # (nfreqbins, spin)
            magnitude = np.abs(spec).T  # (spin, nfreqbins)

            spin_count = magnitude.shape[0]
            try:
                self.spectrum_heatmap_item.setImage(magnitude, autoLevels=True, axisOrder='row-major')
                span = float(freq_axis[-1] - freq_axis[0])
                self.spectrum_heatmap_item.setRect(float(freq_axis[0]), 0, span if span != 0 else 1.0, spin_count)
                self.spectrum_heatmap.setLabel('left', y_label)
                self.spectrum_heatmap.setLabel('bottom', 'Frequency (from signal FFT)', 'Hz')
                self.spectrum_heatmap.setTitle("Spectra Stack (FFT of signal per spin)")
                self.spectrum_heatmap.setXRange(float(freq_axis[0]), float(freq_axis[-1]), padding=0)
                self.spectrum_heatmap.setYRange(0, spin_count, padding=0)
                if hasattr(self, "spectrum_heatmap_colorbar") and self.spectrum_heatmap_colorbar is not None:
                    finite_mag = magnitude[np.isfinite(magnitude)]
                    if finite_mag.size:
                        vmin = float(finite_mag.min())
                        vmax = float(finite_mag.max())
                        if np.isfinite(vmin) and np.isfinite(vmax) and vmax != vmin:
                            self.spectrum_heatmap_colorbar.setLevels((vmin, vmax))
            except Exception as exc:
                self.log_message(f"Spectrum heatmap update failed: {exc}")

    def _toggle_spectrum_3d_mode(self, checked):
        """Toggle between 2D and 3D spectrum visualization."""
        if checked:
            self.spectrum_plot.hide()
            self.spectrum_heatmap_layout.hide()
            self.spectrum_plot_3d.show()
            # Disable other controls that might not apply
            if hasattr(self, 'spectrum_plot_type'):
                self.spectrum_plot_type.setEnabled(False)
        else:
            self.spectrum_plot_3d.hide()
            if hasattr(self, 'spectrum_plot_type'):
                self.spectrum_plot_type.setEnabled(True)
            
            # Restore based on plot type
            is_heatmap = self.spectrum_plot_type.currentText() == "Heatmap"
            self.spectrum_plot.setVisible(not is_heatmap)
            self.spectrum_heatmap_layout.setVisible(is_heatmap)
            
        self._refresh_spectrum()

    def _update_spectrum_3d(self, time_idx=None):
        """Update the 3D spectrum plot."""
        self.spectrum_plot_3d.clear()
        
        # Add simple grid
        gx = gl.GLGridItem()
        gx.setSize(x=20, y=20, z=20)
        gx.rotate(90, 0, 1, 0)
        self.spectrum_plot_3d.addItem(gx)
        
        gy = gl.GLGridItem()
        gy.setSize(x=20, y=20, z=20)
        gy.rotate(90, 1, 0, 0)
        self.spectrum_plot_3d.addItem(gy)
        
        gz = gl.GLGridItem()
        gz.setSize(x=20, y=20, z=20)
        self.spectrum_plot_3d.addItem(gz)

        freqs = None
        data = None
        
        # 1. Try Off-Resonant (Direct Frequency Axis)
        if self.last_result is not None and self.last_frequencies is not None:
            sig = self.last_result.get('signal')
            if sig is not None:
                sig_arr = np.asarray(sig)
                ntime = sig_arr.shape[0]
                if ntime > 0:
                    if sig_arr.ndim == 1: sig_arr = sig_arr[:, None, None]
                    elif sig_arr.ndim == 2:
                        # Assume (time, freq) or (time, pos)
                        if self.last_positions is not None and sig_arr.shape[1] == self.last_positions.shape[0]:
                             sig_arr = sig_arr[:, :, None] # (time, pos, freq=1)
                        else:
                             sig_arr = sig_arr[:, None, :] # (time, pos=1, freq)
                    
                    if time_idx is None: time_idx = ntime - 1
                    t_idx = int(max(0, min(time_idx, ntime - 1)))
                    
                    pos_count = sig_arr.shape[1]
                    pos_sel = min(self.spectrum_pos_slider.value(), pos_count - 1) if pos_count > 0 else 0
                    
                    snapshot = sig_arr[t_idx] # (npos, nfreq)
                    
                    if pos_count > 0:
                        data = snapshot[pos_sel]
                    else:
                        data = snapshot[0]
                        
                    freqs = np.asarray(self.last_frequencies)

        # 2. If not found, Try FFT
        if data is None:
            # We must force compute_fft=True here to get data for 3D plot
            spec_data = self._calculate_spectrum_data(time_idx, compute_fft=True)
            if spec_data and spec_data.get('spectrum') is not None:
                data = spec_data['spectrum']
                freqs = spec_data['freq']
        
        if data is None or freqs is None or data.size != freqs.size:
            return

        # Prepare Points
        # Normalize Frequency for display [-10, 10]
        f_min, f_max = freqs.min(), freqs.max()
        f_range = f_max - f_min
        if f_range == 0: f_range = 1.0
        
        freq_norm = (freqs - f_min) / f_range * 20.0 - 10.0
        
        # Scale Magnitude for display
        mag_scale = 5.0
        
        # x=freq, y=real, z=imag
        pts = np.vstack([freq_norm, np.real(data) * mag_scale, np.imag(data) * mag_scale]).transpose()
        
        # Main signal line
        line = gl.GLLinePlotItem(pos=pts, color=(0, 1, 1, 1), width=2, antialias=True)
        self.spectrum_plot_3d.addItem(line)
        
        # Frequency baseline (Real/Imag = 0)
        baseline_pts = np.vstack([freq_norm, np.zeros_like(freqs), np.zeros_like(freqs)]).transpose()
        baseline = gl.GLLinePlotItem(pos=baseline_pts, color=(0.5, 0.5, 0.5, 0.5), width=1)
        self.spectrum_plot_3d.addItem(baseline)

        # Labels (using TextItem if available or printing info)
        # GLTextItem is tricky in older pyqtgraph. simpler to just rely on grid.

    def _refresh_spectrum(self, time_idx=None, skip_fft=False):
        """Update spectrum plot using data up to the specified time index."""
        # 3D Mode Check
        if hasattr(self, 'spectrum_3d_toggle') and self.spectrum_3d_toggle.isChecked():
            self._update_spectrum_3d(time_idx)
            return

        spec_data = self._calculate_spectrum_data(time_idx, compute_fft=not skip_fft)
        if spec_data is None:
            return

        spectrum_mode = spec_data["mode"]
        pos_count = spec_data["pos_count"]
        pos_sel = spec_data["pos_sel"]
        time_idx = spec_data.get("time_idx", time_idx)

        self.spectrum_pos_slider.setMaximum(max(0, pos_count - 1))
        # Disable slider based on mode and position count instead of hiding
        if hasattr(self, 'spectrum_pos_slider'):
            is_individual = (spectrum_mode == "Individual position")
            self.spectrum_pos_slider.setEnabled(is_individual and pos_count > 1)
            # Update tooltip to explain why disabled
            if not is_individual:
                self.spectrum_pos_slider.setToolTip("Switch to 'Individual position' view to select position")
            elif pos_count <= 1:
                self.spectrum_pos_slider.setToolTip("Only one position available")
            else:
                self.spectrum_pos_slider.setToolTip("Select position to view")

        self.spectrum_pos_slider.blockSignals(True)
        self.spectrum_pos_slider.setValue(pos_sel)
        self.spectrum_pos_slider.blockSignals(False)
        plot_type = self.spectrum_plot_type.currentText() if hasattr(self, 'spectrum_plot_type') else "Line"
        is_heatmap = (plot_type == "Heatmap")
        
        # Show/hide heatmap mode selector
        if hasattr(self, "spectrum_heatmap_mode"):
            self.spectrum_heatmap_mode.setVisible(is_heatmap)
            self.spectrum_heatmap_mode_label.setVisible(is_heatmap)
        
        # Show/hide component selector (only for line plots)
        if hasattr(self, "spectrum_component_combo"):
            self.spectrum_component_combo.setVisible(not is_heatmap)
            if hasattr(self, "spectrum_component_label"):
                self.spectrum_component_label.setVisible(not is_heatmap)

        self.spectrum_plot.setVisible(not is_heatmap)
        if hasattr(self, "spectrum_heatmap"):
            self.spectrum_heatmap_layout.setVisible(is_heatmap)
            
        # Add colorbar to heatmap
        if is_heatmap and hasattr(self, "spectrum_heatmap_colorbar"):
            self.spectrum_heatmap_layout.addItem(self.spectrum_heatmap_colorbar, row=0, col=1)
        
        if is_heatmap:
            self._update_spectrum_heatmap(time_idx=time_idx)
        else:
            self._plot_off_resonant_spins(time_idx=time_idx)

    def _compute_final_spectrum_range(self, signal, time):
        """Compute final-spectrum magnitude range for consistent y-limits."""
        self._spectrum_final_range = None
        if signal is None or time is None:
            return
        time = np.asarray(time)
        sig_arr = np.asarray(signal)
        if sig_arr.ndim == 1:
            sig_arr = sig_arr[:, None, None]
        elif sig_arr.ndim == 2:
            sig_arr = sig_arr[:, :, None]
        if len(time) < 2 or sig_arr.shape[0] != len(time):
            return
        try:
            # Use mean across positions/frequencies
            sig_for_fft = np.mean(sig_arr, axis=tuple(range(1, sig_arr.ndim)))
            dt = float(time[1] - time[0])
            n_fft = self._spectrum_fft_len(len(sig_for_fft))
            spectrum = np.fft.fftshift(np.fft.fft(sig_for_fft, n=n_fft))
            mag = np.abs(spectrum)
            if mag.size:
                mag_min = float(np.nanmin(mag))
                mag_max = float(np.nanmax(mag))
                if not np.isfinite(mag_min):
                    mag_min = 0.0
                if not np.isfinite(mag_max) or mag_max <= 0:
                    mag_max = 1.0
                self._spectrum_final_range = (mag_min, mag_max * 1.05)
        except Exception:
            self._spectrum_final_range = None

    def _plot_off_resonant_spins(self, time_idx=None) -> bool:
        """Plot a spectrum built directly from the simulated off-resonant spins (no FFT)."""
        if self.last_result is None or self.last_frequencies is None:
            return False
        sig = self.last_result.get('signal')
        if sig is None:
            return False
        sig_arr = np.asarray(sig)
        ntime = sig_arr.shape[0]
        if ntime == 0:
            return False
        if sig_arr.ndim == 1:
            sig_arr = sig_arr[:, None, None]
        elif sig_arr.ndim == 2:
            # Assume (time, freq) or (time, pos)
            if self.last_positions is not None and sig_arr.shape[1] == self.last_positions.shape[0]:
                 sig_arr = sig_arr[:, :, None] # (time, pos, freq=1)
            else:
                 sig_arr = sig_arr[:, None, :] # (time, pos=1, freq)

        time_axis = self.last_time if self.last_time is not None else np.arange(ntime, dtype=float)
        if time_idx is None:
            time_idx = ntime - 1
        t_idx = int(max(0, min(time_idx, ntime - 1)))

        freq_axis = np.asarray(self.last_frequencies)
        if freq_axis.shape[0] != sig_arr.shape[2]:
            freq_axis = np.linspace(-0.5, 0.5, sig_arr.shape[2])

        pos_count = sig_arr.shape[1]
        spectrum_mode = self.spectrum_mode.currentText() if hasattr(self, 'spectrum_mode') else "Mean only"
        pos_sel = min(self.spectrum_pos_slider.value(), pos_count - 1) if pos_count > 0 else 0

        # Snapshot spectrum at the selected time index
        snapshot = sig_arr[t_idx]  # (npos, nfreq)
        mean_series = np.mean(snapshot, axis=0) if pos_count > 0 else snapshot
        selected_series = mean_series
        selected_label = "Mean"
        if spectrum_mode == "Mean + individuals":
            selected_series = snapshot[pos_sel] if pos_count else mean_series
            selected_label = f"Pos {pos_sel}"
        elif spectrum_mode == "Individual (select pos)":
            selected_series = snapshot[pos_sel] if pos_count else mean_series
            selected_label = f"Pos {pos_sel}"
            mean_series = None

        self.spectrum_plot.clear()
        selected_components = self.spectrum_component_combo.get_selected_items()

        def get_component(data, comp):
            if comp == "Magnitude": return np.abs(data)
            if comp == "Phase": return np.angle(data) / np.pi
            if comp == "Phase (unwrapped)": return np.unwrap(np.angle(data)) / np.pi
            if comp == "Real": return np.real(data)
            if comp == "Imaginary": return np.imag(data)
            return np.abs(data)

        # Plot each selected component
        for component in selected_components:
            if mean_series is not None:
                # Use fixed colors for components in the Mean plot
                color = 'c' # Default Mean Magnitude
                if component == "Real": color = 'r'
                elif component == "Imaginary": color = 'g'
                elif component == "Phase" or component == "Phase (unwrapped)": color = 'y'
                
                self.spectrum_plot.plot(freq_axis, get_component(mean_series, component), 
                                        pen=pg.mkPen(color, width=3), 
                                        name=f"Mean {component}")

            # Selected position plot
            if pos_count > 0:
                sel_color = self._color_for_index(pos_sel, max(pos_count, 1)) if pos_count > 1 else 'w'
                
                # If only one component is selected, use the position-cycling color.
                # If multiple components are selected, use fixed colors but different styles.
                if len(selected_components) == 1:
                    pen = pg.mkPen(sel_color, width=2)
                else:
                    color = 'b' # Selected Magnitude
                    if component == "Real": color = 'r'
                    elif component == "Imaginary": color = 'g'
                    elif component == "Phase" or component == "Phase (unwrapped)": color = 'y'
                    pen = pg.mkPen(color, width=2)
                    
                if component == "Real": pen.setStyle(Qt.DashLine)
                elif component == "Imaginary": pen.setStyle(Qt.DotLine)
                elif component == "Phase (unwrapped)": pen.setStyle(Qt.SolidLine)
                
                self.spectrum_plot.plot(freq_axis, get_component(selected_series, component), 
                                        pen=pen, name=f"{selected_label} {component}")

        self.spectrum_plot.setLabel('bottom', 'Frequency', 'Hz')
        y_label = "Signal" if len(selected_components) > 1 else selected_components[0] if selected_components else ""
        if ("Phase" in selected_components or "Phase (unwrapped)" in selected_components) and len(selected_components) == 1:
            y_label = f"{selected_components[0]} (units of π)"
        self.spectrum_plot.setLabel('left', y_label)

        if freq_axis is not None and freq_axis.size > 0:
            self.spectrum_plot.setXRange(float(np.nanmin(freq_axis)), float(np.nanmax(freq_axis)), padding=0.05)

        y_min, y_max = -1.1, 1.1
        if hasattr(self, 'off_res_spectrum_yrange') and self.off_res_spectrum_yrange is not None:
            y_min, y_max = self.off_res_spectrum_yrange

        if "Phase (unwrapped)" in selected_components:
            # Unwrapped phase can be large, use auto-range
            self.spectrum_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        elif "Phase" in selected_components and len(selected_components) == 1:
            self.spectrum_plot.setYRange(-1.1, 1.1, padding=0)
        elif "Magnitude" in selected_components and len(selected_components) == 1:
            self.spectrum_plot.setYRange(0, y_max * 1.1, padding=0)
        else: # Mixed or Real/Imag
            self.spectrum_plot.setYRange(-y_max * 1.1, y_max * 1.1, padding=0)

        export_entry = {
            "frequency": freq_axis,
            "selected_magnitude": np.abs(selected_series),
            "selected_phase_rad": np.angle(selected_series),
            "mode": "off_res_spins_freq",
            "time_idx": t_idx,
            "time_s": float(time_axis[t_idx]) if t_idx < len(time_axis) else None,
        }
        if mean_series is not None:
            export_entry["mean_magnitude"] = np.abs(mean_series)
            export_entry["mean_phase_rad"] = np.angle(mean_series)
        self._last_spectrum_export = export_entry
        return True

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

    def _grab_pyqtgraph_array(self, plot_widget: pg.PlotWidget, width: int = None, height: int = None) -> np.ndarray:
        """Grab a pyqtgraph plot as an RGB numpy array using its faster internal exporter."""
        exporter = pg.exporters.ImageExporter(plot_widget.getPlotItem())
        if width:
            exporter.parameters()['width'] = width
        if height:
            exporter.parameters()['height'] = height
        
        image = exporter.export(toBytes=True)
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

    def _compute_playback_indices(self, time_array, start_idx, end_idx, fps):
        """Compute indices to match the current playback speed setting."""
        speed_ms_per_s = self.time_control.speed_spin.value()
        if speed_ms_per_s <= 1e-6:
            speed_ms_per_s = 50.0

        t_start = time_array[start_idx]
        t_end = time_array[end_idx]

        # Simulation duration in seconds
        sim_dur_s = t_end - t_start

        # Real duration in seconds = (Sim ms) / (ms/s)
        real_dur_s = (sim_dur_s * 1000.0) / speed_ms_per_s

        # Total frames
        n_frames = int(max(2, np.ceil(real_dur_s * fps)))

        # Target times
        target_times = np.linspace(t_start, t_end, n_frames)

        # Find indices
        indices = np.searchsorted(time_array, target_times)
        indices = np.clip(indices, start_idx, end_idx)

        self.log_message(f"Exporting animation: {n_frames} frames to match {speed_ms_per_s} ms/s at {fps} FPS.")

        return indices

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
        dialog.include_sequence_checkbox.setVisible(False)

        if dialog.exec_() != QDialog.Accepted:
            return
        params = dialog.get_export_params()
        
        time_array = self.playback_time if self.playback_time is not None else self.last_time
        indices = self._compute_playback_indices(time_array, params['start_idx'], params['end_idx'], params['fps'])

        fmt = params['format']
        filepath = Path(params['filename'])
        if filepath.suffix.lower() != f".{fmt}":
            filepath = filepath.with_suffix(f".{fmt}")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if vz_imageio is None:
            QMessageBox.warning(self, "Missing Dependency", "Animation export requires the 'imageio' package.")
            return

        exporter = AnimationExporter()
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
            ui_update_interval = max(1, len(indices) // 50)  # keep UI responsive without slowing loop
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
                target_w = params['width'] if params['width'] else None
                for w in widgets:
                    if isinstance(w, pg.PlotWidget):
                        frames.append(self._grab_pyqtgraph_array(w, width=target_w, height=target_h))
                    else:
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
                    # QImage bits are often 32-bit aligned/padded or RGBA even if Format_RGB888 was requested
                    # Check size to determine channels
                    n_bytes = qimg.byteCount()
                    n_pixels = qimg.width() * qimg.height()
                    n_channels = n_bytes // n_pixels
                    
                    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((qimg.height(), qimg.width(), n_channels))
                    # Keep only RGB channels (drop alpha if present)
                    combined = arr[:, :, :3]
                combined = self._ensure_even_frame(combined)
                writer.append_data(combined)
                progress.setValue(i + 1)
                if (i % ui_update_interval) == 0:
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
        dialog.include_sequence_checkbox.setVisible(False)

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

        indices = self._compute_playback_indices(time_s, start_idx, end_idx, params['fps'])
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

        exporter = AnimationExporter()
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
        indices = self._compute_playback_indices(self.playback_time, params['start_idx'], params['end_idx'], params['fps'])

        # Prepare writers (main + optional sequence-only)
        fmt = params['format']
        filepath = Path(params['filename'])
        if filepath.suffix.lower() != f".{fmt}":
            filepath = filepath.with_suffix(f".{fmt}")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        exporter = AnimationExporter()
        def _make_writer(target_path: Path):
            if fmt == 'gif':
                return vz_imageio.get_writer(str(target_path), mode='I', fps=params['fps'], format='GIF')
            return vz_imageio.get_writer(
                str(target_path),
                fps=params['fps'],
                format='FFMPEG',
                codec='libx264',
                bitrate=exporter.default_bitrate,
                quality=8,
                macro_block_size=None
            )

        writer = _make_writer(filepath)
        seq_writer = None
        seq_filepath = None
        if params.get('include_sequence', False):
            seq_filepath = filepath.with_name(f"{filepath.stem}_sequence{filepath.suffix}")
            seq_filepath.parent.mkdir(parents=True, exist_ok=True)
            seq_writer = _make_writer(seq_filepath)

        progress = QProgressDialog("Exporting 3D animation...", "Cancel", 0, len(indices), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        # Save state and pause playback to avoid interference
        was_running = self.anim_timer.isActive()
        current_idx = self.anim_index
        self.anim_timer.stop()


        def grab_sequence_frame():
            seq_pixmap = self.sequence_designer.diagram_widget.grab()
            seq_image = seq_pixmap.toImage()
            target_w = params['width'] if params['width'] else seq_image.width()
            target_h = params['height'] if params['height'] else seq_image.height()
            return _qimage_to_rgb(seq_image, target_w=target_w, target_h=target_h)

        try:
            ui_update_interval = max(1, len(indices) // 50)
            for i, idx in enumerate(indices):
                if progress.wasCanceled():
                    raise RuntimeError("Animation export cancelled")
                self._set_animation_index_from_slider(int(idx))
                QApplication.processEvents() # Allow UI to update with new vector positions

                # Use fast off-screen rendering for the 3D view
                target_w = params['width'] if params['width'] else self.mag_3d.gl_widget.width()
                target_h = params['height'] if params['height'] else self.mag_3d.gl_widget.height()
                frame = self.mag_3d.gl_widget.renderToArray((target_w, target_h))
                frame = self._ensure_even_frame(frame) # Ensure dimensions are even for video codecs
                writer.append_data(frame)
                if seq_writer is not None:
                    seq_frame = grab_sequence_frame()
                    seq_writer.append_data(seq_frame)
                progress.setValue(i + 1)
                QApplication.processEvents()
        finally:
            writer.close()
            if seq_writer is not None:
                seq_writer.close()
            # Restore playback state
            self._set_animation_index_from_slider(current_idx)
            if was_running:
                self._resume_vector_animation()
            progress.setValue(progress.maximum())

        msg = f"3D animation exported successfully:\n{filepath.name}"
        if seq_filepath is not None:
            msg += f"\nSequence diagram exported:\n{seq_filepath.name}"
        QMessageBox.information(self, "Export Successful", msg)
        self.log_message(f"3D animation exported to {filepath}")
        if seq_filepath is not None:
            self.log_message(f"Sequence diagram exported to {seq_filepath}")

    def _export_sequence_diagram_animation(self):
        """Export the sequence diagram only (no combined views)."""
        self._export_widget_animation([self.sequence_designer.diagram_widget], default_filename="sequence")

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
        dialog.include_sequence_checkbox.setVisible(False)
        if dialog.exec_() != QDialog.Accepted:
            return

        params = dialog.get_export_params()
        start_idx = min(params['start_idx'], nframes - 1)
        end_idx = max(start_idx, min(params['end_idx'], nframes - 1))
        mean_only = params['mean_only']
        indices = self._compute_playback_indices(time_s, start_idx, end_idx, params['fps'])

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

        exporter = AnimationExporter()
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
            self._refresh_spectrum(time_idx=actual_idx, skip_fft=False)
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
        self._refresh_spectrum(time_idx=actual_idx, skip_fft=False)
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

    def _update_mag_heatmaps(self):
        """Update magnetization heatmaps (Time vs Position/Frequency)."""
        if self.last_result is None:
            return

        mx_all = self.last_result['mx']
        my_all = self.last_result['my']
        mz_all = self.last_result['mz']
        time = self.last_time if self.last_time is not None else self.last_result.get('time', None)

        if time is None or mx_all.ndim != 3:
            self.log_message("Heatmap requires time-resolved 3D data")
            return

        ntime, npos, nfreq = mx_all.shape
        time_ms = time * 1000  # Convert to ms

        # Get view mode
        view_mode = self.mag_view_mode.currentText() if hasattr(self, 'mag_view_mode') else "All positions x freqs"
        selector = self.mag_view_selector.value() if hasattr(self, 'mag_view_selector') else 0

        # Determine if view mode selector should be visible
        # Hide selector for single position/frequency cases
        if view_mode == "Positions @ freq" and npos == 1:
            # Only 1 position - no point showing position selector
            if hasattr(self, 'mag_view_selector'):
                self.mag_view_selector.setVisible(False)
                self.mag_view_selector_label.setVisible(False)
        elif view_mode == "Freqs @ position" and nfreq == 1:
            # Only 1 frequency - no point showing frequency selector
            if hasattr(self, 'mag_view_selector'):
                self.mag_view_selector.setVisible(False)
                self.mag_view_selector_label.setVisible(False)
        elif view_mode == "All positions x freqs" and npos * nfreq == 1:
            # Only 1 spin total
            if hasattr(self, 'mag_view_selector'):
                self.mag_view_selector.setVisible(False)
                self.mag_view_selector_label.setVisible(False)

        # Get component selection
        component = self.mag_component.currentText() if hasattr(self, 'mag_component') else "Magnitude"

        # Prepare data slices based on view mode
        y_min, y_max = 0, 0
        
        if view_mode == "Positions @ freq" and nfreq > 1:
            # Show all positions at selected frequency
            fi = min(selector, nfreq - 1)
            mx_slice = mx_all[:, :, fi]  # (ntime, npos)
            my_slice = my_all[:, :, fi]
            mz_slice = mz_all[:, :, fi]
            
            y_label = "Position (cm)"
            n_y = npos
            if self.last_positions is not None:
                # Assume Z-axis varying
                pos_vals = self.last_positions[:, 2] * 100
                y_min, y_max = pos_vals[0], pos_vals[-1]
            else:
                y_label = "Position Index"
                y_min, y_max = 0, npos

        elif view_mode == "Freqs @ position" and npos > 1:
            # Show all frequencies at selected position
            pi = min(selector, npos - 1)
            mx_slice = mx_all[:, pi, :]  # (ntime, nfreq)
            my_slice = my_all[:, pi, :]
            mz_slice = mz_all[:, pi, :]
            
            y_label = "Frequency (Hz)"
            n_y = nfreq
            if self.last_frequencies is not None:
                y_min, y_max = self.last_frequencies[0], self.last_frequencies[-1]
            else:
                y_label = "Frequency Index"
                y_min, y_max = 0, nfreq
        else:
            # Show all spins (flatten position x frequency)
            mx_slice = mx_all.reshape(ntime, -1)  # (ntime, npos*nfreq)
            my_slice = my_all.reshape(ntime, -1)
            mz_slice = mz_all.reshape(ntime, -1)
            y_label = "Spin Index (pos×freq)"
            n_y = npos * nfreq
            y_min, y_max = 0, n_y

        # Handle degenerate range
        if np.isclose(y_min, y_max):
             y_max = y_min + (1.0 if n_y <= 1 else n_y)

        # Compute the selected component
        if component == "Magnitude":
            mxy_data = np.sqrt(mx_slice**2 + my_slice**2)  # (ntime, n_y)
            mz_data = np.abs(mz_slice)  # (ntime, n_y)
            mxy_title = '|Mxy| Heatmap'
            mz_title = '|Mz| Heatmap'
        elif component == "Real (Mx/Re)":
            mxy_data = mx_slice  # (ntime, n_y)
            mz_data = mz_slice  # (ntime, n_y)
            mxy_title = 'Mx Heatmap'
            mz_title = 'Mz Heatmap'
        elif component == "Imaginary (My/Im)":
            mxy_data = my_slice  # (ntime, n_y)
            mz_data = mz_slice  # (ntime, n_y)
            mxy_title = 'My Heatmap'
            mz_title = 'Mz Heatmap'
        elif component == "Phase":
            mxy_data = np.angle(mx_slice + 1j * my_slice)  # (ntime, n_y)
            mz_data = mz_slice  # (ntime, n_y)
            mxy_title = 'Mxy Phase Heatmap'
            mz_title = 'Mz Heatmap'
        elif component == "Mz":
            mxy_data = mz_slice  # (ntime, n_y)
            mz_data = mz_slice  # (ntime, n_y)
            mxy_title = 'Mz Heatmap'
            mz_title = 'Mz Heatmap'
        else:
            # Default to magnitude
            mxy_data = np.sqrt(mx_slice**2 + my_slice**2)
            mz_data = np.abs(mz_slice)
            mxy_title = '|Mxy| Heatmap'
            mz_title = '|Mz| Heatmap'

        # Update Mxy heatmap
        # pyqtgraph ImageItem convention: data[row, col] where row=Y, col=X
        # We have mxy_data as (ntime, n_y) meaning data[time, spin]
        # We want: X-axis = Time, Y-axis = Spin/Pos/Freq
        # So we need to transpose to get data[spin, time] = data[Y, X]
        self.mxy_heatmap_item.setImage(mxy_data.T, autoLevels=True, axisOrder='row-major')
        # Now set the coordinate mapping using setRect(x, y, width, height)
        self.mxy_heatmap_item.setRect(time_ms[0], y_min, time_ms[-1] - time_ms[0], y_max - y_min)
        self.mxy_heatmap.setLabel('left', y_label)
        self.mxy_heatmap.setLabel('bottom', 'Time', 'ms')
        self.mxy_heatmap.setTitle(mxy_title)
        # Set view limits to show only actual data
        self.mxy_heatmap.setXRange(time_ms[0], time_ms[-1], padding=0)
        self.mxy_heatmap.setYRange(y_min, y_max, padding=0)

        # Update Mz heatmap
        self.mz_heatmap_item.setImage(mz_data.T, autoLevels=True, axisOrder='row-major')
        self.mz_heatmap_item.setRect(time_ms[0], y_min, time_ms[-1] - time_ms[0], y_max - y_min)
        self.mz_heatmap.setLabel('left', y_label)
        self.mz_heatmap.setLabel('bottom', 'Time', 'ms')
        self.mz_heatmap.setTitle(mz_title)
        # Set view limits to show only actual data
        self.mz_heatmap.setXRange(time_ms[0], time_ms[-1], padding=0)
        self.mz_heatmap.setYRange(y_min, y_max, padding=0)

    def _update_signal_heatmaps(self):
        """Update signal heatmaps (Time vs Position/Frequency)."""
        if self.last_result is None:
            return

        signal_all = self.last_result['signal']
        time = self.last_time if self.last_time is not None else self.last_result.get('time', None)

        if time is None or signal_all.ndim != 3:
            self.log_message("Heatmap requires time-resolved 3D data")
            return

        ntime, npos, nfreq = signal_all.shape
        time_ms = time * 1000  # Convert to ms

        # Get view mode
        view_mode = self.signal_view_mode.currentText() if hasattr(self, 'signal_view_mode') else "All positions x freqs"
        selector = self.signal_view_selector.value() if hasattr(self, 'signal_view_selector') else 0

        # Determine if view mode selector should be visible
        # Hide selector for single position/frequency cases
        if view_mode == "Positions @ freq" and npos == 1:
            # Only 1 position - no point showing position selector
            if hasattr(self, 'signal_view_selector'):
                self.signal_view_selector.setVisible(False)
                self.signal_view_selector_label.setVisible(False)
        elif view_mode == "Freqs @ position" and nfreq == 1:
            # Only 1 frequency - no point showing frequency selector
            if hasattr(self, 'signal_view_selector'):
                self.signal_view_selector.setVisible(False)
                self.signal_view_selector_label.setVisible(False)
        elif view_mode == "All positions x freqs" and npos * nfreq == 1:
            # Only 1 spin total
            if hasattr(self, 'signal_view_selector'):
                self.signal_view_selector.setVisible(False)
                self.signal_view_selector_label.setVisible(False)

        # Get component selection
        component = self.signal_component.currentText() if hasattr(self, 'signal_component') else "Magnitude"

        # Prepare data slices based on view mode
        y_min, y_max = 0, 0
        
        if view_mode == "Positions @ freq" and nfreq > 1:
            # Show all positions at selected frequency
            fi = min(selector, nfreq - 1)
            signal_slice = signal_all[:, :, fi]  # (ntime, npos)
            
            y_label = "Position (cm)"
            n_y = npos
            if self.last_positions is not None:
                pos_vals = self.last_positions[:, 2] * 100
                y_min, y_max = pos_vals[0], pos_vals[-1]
            else:
                y_label = "Position Index"
                y_min, y_max = 0, npos

        elif view_mode == "Freqs @ position" and npos > 1:
            # Show all frequencies at selected position
            pi = min(selector, npos - 1)
            signal_slice = signal_all[:, pi, :]  # (ntime, nfreq)
            
            y_label = "Frequency (Hz)"
            n_y = nfreq
            if self.last_frequencies is not None:
                y_min, y_max = self.last_frequencies[0], self.last_frequencies[-1]
            else:
                y_label = "Frequency Index"
                y_min, y_max = 0, nfreq
        else:
            # Show all spins (flatten position x frequency)
            signal_slice = signal_all.reshape(ntime, -1)  # (ntime, npos*nfreq)
            y_label = "Spin Index (pos×freq)"
            n_y = npos * nfreq
            y_min, y_max = 0, n_y

        # Handle degenerate range
        if np.isclose(y_min, y_max):
             y_max = y_min + (1.0 if n_y <= 1 else n_y)

        # Compute the selected component
        if component == "Magnitude":
            signal_data = np.abs(signal_slice)  # (ntime, n_y)
            title = '|Signal| Heatmap'
        elif component == "Real":
            signal_data = np.real(signal_slice)  # (ntime, n_y)
            title = 'Re(Signal) Heatmap'
        elif component == "Imaginary":
            signal_data = np.imag(signal_slice)  # (ntime, n_y)
            title = 'Im(Signal) Heatmap'
        elif component == "Phase":
            signal_data = np.angle(signal_slice)  # (ntime, n_y)
            title = 'Phase(Signal) Heatmap'
        else:
            # Default to magnitude
            signal_data = np.abs(signal_slice)
            title = '|Signal| Heatmap'

        # Update signal heatmap
        # pyqtgraph ImageItem convention: data[row, col] where row=Y, col=X
        # We have signal_data as (ntime, n_y) meaning data[time, spin]
        # We want: X-axis = Time, Y-axis = Spin
        # So we need to transpose to get data[spin, time] = data[Y, X]
        self.signal_heatmap_item.setImage(signal_data.T, autoLevels=True, axisOrder='row-major')
        # Set proper scale and position
        self.signal_heatmap_item.setRect(time_ms[0], y_min, time_ms[-1] - time_ms[0], y_max - y_min)
        self.signal_heatmap.setLabel('left', y_label)
        self.signal_heatmap.setLabel('bottom', 'Time', 'ms')
        self.signal_heatmap.setTitle(title)
        # Set view limits to show only actual data
        self.signal_heatmap.setXRange(time_ms[0], time_ms[-1], padding=0)
        self.signal_heatmap.setYRange(y_min, y_max, padding=0)

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About Bloch Simulator",
            "Bloch Equation Simulator\n\n"
            "A Python implementation of the Bloch equation solver\n"
            "originally developed by Brian Hargreaves.\n\n"
            "This GUI provides interactive visualization and\n"
            "parameter control for MRI pulse sequence simulation.\n\n"
            f"Version {__version__}")


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
