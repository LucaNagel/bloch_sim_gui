"""
kspace_widget.py - K-Space Simulation Widget for Bloch Simulator GUI

This widget integrates k-space simulation into the existing BlochSimulatorGUI,
following the same pattern as PhantomWidget.

Features:
- Uses existing phantom from PhantomWidget
- Trajectory generation (Cartesian, EPI, CSI)
- Eddy current simulation
- B0 inhomogeneity effects
- Visualization of k-space, gradients, and reconstructed images

Integration:
    1. Copy kspace_widget.py to your project directory
    2. Copy kspace_simulator.py and spectral_phantom.py
    3. Add import and tab creation to bloch_gui.py (see patch at bottom of file)

Author: Luca Nagel
Date: 2025
"""

import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QTabWidget, QSplitter,
    QProgressBar, QCheckBox, QSlider, QFileDialog, QMessageBox,
    QGridLayout, QFrame, QScrollArea, QSizePolicy, QProgressDialog,
    QApplication, QDialog, QRadioButton, QButtonGroup, QMenu
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

import pyqtgraph as pg

from .kspace import (
    KSpaceSimulator,
    EddyCurrentModel,
    EPIParameters,
    CSIParameters,
    TrajectoryType
)

# Import spectral phantom for CSI
from .spectral_phantom import (
    SpectralPhantom,
    SpectralPhantomFactory,
    ChemicalSpecies
)


class KSpaceSimulationThread(QThread):
    """Thread for running k-space simulation without blocking the GUI."""
    
    progress = pyqtSignal(int, str)  # progress percentage, status message
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, simulator, phantom, trajectory, 
                 magnetization=None, eddy_model=None, b0_map=None, noise_std=0.0):
        super().__init__()
        self.simulator = simulator
        self.phantom = phantom
        self.trajectory = trajectory
        self.magnetization = magnetization
        self.eddy_model = eddy_model
        self.b0_map = b0_map
        self.noise_std = noise_std
        self._cancel_requested = False
    
    def request_cancel(self):
        self._cancel_requested = True
    
    def run(self):
        try:
            if self._cancel_requested:
                return
            
            self.progress.emit(10, "Simulating k-space signal...")
            
            # Set up progress callback
            def log_callback(msg):
                self.progress.emit(50, msg)
            
            self.simulator.set_log_callback(log_callback)
            
            # Run simulation
            result = self.simulator.simulate_signal(
                self.phantom,
                self.trajectory,
                magnetization=self.magnetization,
                eddy_model=self.eddy_model,
                b0_map=self.b0_map,
                noise_std=self.noise_std
            )
            
            if self._cancel_requested:
                return
            
            self.progress.emit(80, "Reconstructing...")
            
            # Reconstruct based on trajectory type
            traj_type = self.trajectory.get('trajectory_type', TrajectoryType.CARTESIAN)
            
            if traj_type == TrajectoryType.CSI:
                recon = self.simulator.reconstruct_csi(result)
                result['reconstruction'] = recon
                result['recon_type'] = 'csi'
            elif traj_type == TrajectoryType.EPI:
                image = self.simulator.reconstruct_epi(result)
                result['reconstruction'] = image
                result['recon_type'] = 'image'
            else:
                image = self.simulator.reconstruct_cartesian(result)
                result['reconstruction'] = image
                result['recon_type'] = 'image'
            
            self.progress.emit(100, "Complete")
            self.finished.emit(result)
            
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class TrajectoryWidget(QGroupBox):
    """Widget for k-space trajectory settings."""
    
    settings_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__("K-Space Trajectory")
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Trajectory type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Cartesian", "EPI", "CSI"])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self.type_combo.currentTextChanged.connect(self.settings_changed.emit)
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()
        layout.addLayout(type_layout)
        
        # Common spatial settings
        spatial_group = QGroupBox("Spatial Encoding")
        spatial_layout = QGridLayout()
        
        spatial_layout.addWidget(QLabel("Matrix X:"), 0, 0)
        self.matrix_x_spin = QSpinBox()
        self.matrix_x_spin.setRange(8, 256)
        self.matrix_x_spin.setValue(32)
        self.matrix_x_spin.valueChanged.connect(self.settings_changed.emit)
        spatial_layout.addWidget(self.matrix_x_spin, 0, 1)
        
        spatial_layout.addWidget(QLabel("Matrix Y:"), 0, 2)
        self.matrix_y_spin = QSpinBox()
        self.matrix_y_spin.setRange(8, 256)
        self.matrix_y_spin.setValue(32)
        self.matrix_y_spin.valueChanged.connect(self.settings_changed.emit)
        spatial_layout.addWidget(self.matrix_y_spin, 0, 3)
        
        spatial_layout.addWidget(QLabel("FOV X (cm):"), 1, 0)
        self.fov_x_spin = QDoubleSpinBox()
        self.fov_x_spin.setRange(1, 50)
        self.fov_x_spin.setValue(24)
        self.fov_x_spin.setDecimals(1)
        self.fov_x_spin.valueChanged.connect(self.settings_changed.emit)
        spatial_layout.addWidget(self.fov_x_spin, 1, 1)
        
        spatial_layout.addWidget(QLabel("FOV Y (cm):"), 1, 2)
        self.fov_y_spin = QDoubleSpinBox()
        self.fov_y_spin.setRange(1, 50)
        self.fov_y_spin.setValue(24)
        self.fov_y_spin.setDecimals(1)
        self.fov_y_spin.valueChanged.connect(self.settings_changed.emit)
        spatial_layout.addWidget(self.fov_y_spin, 1, 3)
        
        # Match phantom button
        self.match_phantom_btn = QPushButton("Match Phantom")
        self.match_phantom_btn.setToolTip("Copy matrix size and FOV from current phantom")
        spatial_layout.addWidget(self.match_phantom_btn, 2, 0, 1, 2)
        
        spatial_group.setLayout(spatial_layout)
        layout.addWidget(spatial_group)
        
        # EPI-specific settings
        self.epi_group = QGroupBox("EPI Parameters")
        epi_layout = QGridLayout()
        
        epi_layout.addWidget(QLabel("Echo spacing (ms):"), 0, 0)
        self.echo_spacing_spin = QDoubleSpinBox()
        self.echo_spacing_spin.setRange(0.1, 5.0)
        self.echo_spacing_spin.setValue(0.5)
        self.echo_spacing_spin.setDecimals(2)
        self.echo_spacing_spin.valueChanged.connect(self.settings_changed.emit)
        epi_layout.addWidget(self.echo_spacing_spin, 0, 1)
        
        epi_layout.addWidget(QLabel("Ramp time (ms):"), 0, 2)
        self.ramp_time_spin = QDoubleSpinBox()
        self.ramp_time_spin.setRange(0.01, 1.0)
        self.ramp_time_spin.setValue(0.1)
        self.ramp_time_spin.setDecimals(2)
        self.ramp_time_spin.valueChanged.connect(self.settings_changed.emit)
        epi_layout.addWidget(self.ramp_time_spin, 0, 3)
        
        epi_layout.addWidget(QLabel("Shots:"), 1, 0)
        self.shots_spin = QSpinBox()
        self.shots_spin.setRange(1, 32)
        self.shots_spin.setValue(1)
        self.shots_spin.valueChanged.connect(self.settings_changed.emit)
        epi_layout.addWidget(self.shots_spin, 1, 1)
        
        self.flyback_check = QCheckBox("Flyback (unipolar)")
        self.flyback_check.setToolTip("Flyback avoids Nyquist ghosting but is slower")
        self.flyback_check.toggled.connect(self.settings_changed.emit)
        epi_layout.addWidget(self.flyback_check, 1, 2, 1, 2)
        
        self.epi_group.setLayout(epi_layout)
        layout.addWidget(self.epi_group)
        
        # CSI-specific settings
        self.csi_group = QGroupBox("CSI Parameters")
        csi_layout = QGridLayout()
        
        csi_layout.addWidget(QLabel("Spectral points:"), 0, 0)
        self.spectral_points_spin = QSpinBox()
        self.spectral_points_spin.setRange(64, 4096)
        self.spectral_points_spin.setValue(1024)
        self.spectral_points_spin.setSingleStep(256)
        self.spectral_points_spin.valueChanged.connect(self._update_csi_info)
        self.spectral_points_spin.valueChanged.connect(self.settings_changed.emit)
        csi_layout.addWidget(self.spectral_points_spin, 0, 1)
        
        csi_layout.addWidget(QLabel("Bandwidth (Hz):"), 0, 2)
        self.bandwidth_spin = QDoubleSpinBox()
        self.bandwidth_spin.setRange(500, 10000)
        self.bandwidth_spin.setValue(2000)
        self.bandwidth_spin.setDecimals(0)
        self.bandwidth_spin.valueChanged.connect(self._update_csi_info)
        self.bandwidth_spin.valueChanged.connect(self.settings_changed.emit)
        csi_layout.addWidget(self.bandwidth_spin, 0, 3)
        
        csi_layout.addWidget(QLabel("Dwell time:"), 1, 0)
        self.dwell_time_label = QLabel("0.5 ms")
        csi_layout.addWidget(self.dwell_time_label, 1, 1)
        
        csi_layout.addWidget(QLabel("Acq. time:"), 1, 2)
        self.acq_time_label = QLabel("512 ms")
        csi_layout.addWidget(self.acq_time_label, 1, 3)
        
        self.csi_group.setLayout(csi_layout)
        layout.addWidget(self.csi_group)
        
        self.setLayout(layout)
        
        # Initial visibility
        self._on_type_changed(self.type_combo.currentText())
        self._update_csi_info()
    
    def _on_type_changed(self, traj_type: str):
        """Show/hide trajectory-specific settings."""
        self.epi_group.setVisible(traj_type == "EPI")
        self.csi_group.setVisible(traj_type == "CSI")
    
    def _update_csi_info(self):
        """Update calculated CSI timing display."""
        bw = self.bandwidth_spin.value()
        n_spec = self.spectral_points_spin.value()
        dwell = 1000.0 / bw  # ms
        acq_time = n_spec * dwell
        self.dwell_time_label.setText(f"{dwell:.3f} ms")
        self.acq_time_label.setText(f"{acq_time:.1f} ms")
    
    def get_type(self) -> str:
        return self.type_combo.currentText()
    
    def get_matrix_size(self) -> Tuple[int, int]:
        return (self.matrix_x_spin.value(), self.matrix_y_spin.value())
    
    def get_fov(self) -> Tuple[float, float]:
        """Get FOV in meters."""
        return (self.fov_x_spin.value() / 100, self.fov_y_spin.value() / 100)
    
    def set_from_phantom(self, phantom):
        """Set matrix and FOV to match phantom."""
        if phantom is None:
            return
        
        if hasattr(phantom, 'shape') and len(phantom.shape) >= 2:
            self.matrix_x_spin.setValue(phantom.shape[0])
            self.matrix_y_spin.setValue(phantom.shape[1])
        
        if hasattr(phantom, 'fov') and len(phantom.fov) >= 2:
            self.fov_x_spin.setValue(phantom.fov[0] * 100)  # m to cm
            self.fov_y_spin.setValue(phantom.fov[1] * 100)
    
    def get_epi_params(self) -> EPIParameters:
        return EPIParameters(
            matrix_size=self.get_matrix_size(),
            fov=self.get_fov(),
            echo_spacing=self.echo_spacing_spin.value() * 1e-3,
            ramp_time=self.ramp_time_spin.value() * 1e-3,
            n_shots=self.shots_spin.value(),
            flyback=self.flyback_check.isChecked()
        )
    
    def get_csi_params(self) -> CSIParameters:
        return CSIParameters(
            matrix_size=self.get_matrix_size(),
            fov=self.get_fov(),
            spectral_points=self.spectral_points_spin.value(),
            spectral_bandwidth=self.bandwidth_spin.value()
        )


class EddyCurrentWidget(QGroupBox):
    """Widget for eddy current settings."""
    
    settings_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__("Eddy Currents")
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Enable checkbox
        self.enable_check = QCheckBox("Enable eddy current simulation")
        self.enable_check.toggled.connect(self._on_enable_toggled)
        self.enable_check.toggled.connect(self.settings_changed.emit)
        layout.addWidget(self.enable_check)
        
        # Parameters frame
        self.params_frame = QFrame()
        params_layout = QGridLayout()
        
        # Component 1 (fast)
        params_layout.addWidget(QLabel("Fast:"), 0, 0)
        params_layout.addWidget(QLabel("A₁ (%):"), 0, 1)
        self.amp1_spin = QDoubleSpinBox()
        self.amp1_spin.setRange(0, 20)
        self.amp1_spin.setValue(2.0)
        self.amp1_spin.setDecimals(2)
        self.amp1_spin.setToolTip("Amplitude as % of gradient step")
        self.amp1_spin.valueChanged.connect(self.settings_changed.emit)
        params_layout.addWidget(self.amp1_spin, 0, 2)
        
        params_layout.addWidget(QLabel("τ₁ (ms):"), 0, 3)
        self.tau1_spin = QDoubleSpinBox()
        self.tau1_spin.setRange(0.01, 10)
        self.tau1_spin.setValue(0.1)
        self.tau1_spin.setDecimals(2)
        self.tau1_spin.setToolTip("Time constant in ms")
        self.tau1_spin.valueChanged.connect(self.settings_changed.emit)
        params_layout.addWidget(self.tau1_spin, 0, 4)
        
        # Component 2 (slow)
        params_layout.addWidget(QLabel("Slow:"), 1, 0)
        params_layout.addWidget(QLabel("A₂ (%):"), 1, 1)
        self.amp2_spin = QDoubleSpinBox()
        self.amp2_spin.setRange(0, 20)
        self.amp2_spin.setValue(0.5)
        self.amp2_spin.setDecimals(2)
        self.amp2_spin.valueChanged.connect(self.settings_changed.emit)
        params_layout.addWidget(self.amp2_spin, 1, 2)
        
        params_layout.addWidget(QLabel("τ₂ (ms):"), 1, 3)
        self.tau2_spin = QDoubleSpinBox()
        self.tau2_spin.setRange(0.1, 100)
        self.tau2_spin.setValue(1.0)
        self.tau2_spin.setDecimals(2)
        self.tau2_spin.valueChanged.connect(self.settings_changed.emit)
        params_layout.addWidget(self.tau2_spin, 1, 4)
        
        # Presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom",
            "Mild (good shim)",
            "Moderate (typical)",
            "Severe (poor shim)"
        ])
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        params_layout.addLayout(preset_layout, 2, 0, 1, 5)
        
        self.params_frame.setLayout(params_layout)
        layout.addWidget(self.params_frame)
        
        self.setLayout(layout)
        
        # Initially disabled
        self.params_frame.setEnabled(False)
    
    def _on_enable_toggled(self, enabled: bool):
        self.params_frame.setEnabled(enabled)
    
    def _apply_preset(self, preset: str):
        if preset == "Custom":
            return
        
        presets = {
            "Mild (good shim)": (1.0, 0.1, 0.2, 1.0),
            "Moderate (typical)": (2.0, 0.1, 0.5, 1.0),
            "Severe (poor shim)": (5.0, 0.1, 2.0, 2.0),
        }
        
        if preset in presets:
            a1, t1, a2, t2 = presets[preset]
            self.amp1_spin.setValue(a1)
            self.tau1_spin.setValue(t1)
            self.amp2_spin.setValue(a2)
            self.tau2_spin.setValue(t2)
    
    def is_enabled(self) -> bool:
        return self.enable_check.isChecked()
    
    def get_eddy_model(self) -> Optional[EddyCurrentModel]:
        if not self.enable_check.isChecked():
            return None
        
        return EddyCurrentModel(
            amplitudes=[
                self.amp1_spin.value() / 100,
                self.amp2_spin.value() / 100
            ],
            time_constants=[
                self.tau1_spin.value() * 1e-3,
                self.tau2_spin.value() * 1e-3
            ]
        )


class B0Widget(QGroupBox):
    """Widget for B0 inhomogeneity settings."""
    
    settings_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__("B0 Inhomogeneity")
        self._phantom_b0 = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.enable_check = QCheckBox("Enable B0 inhomogeneity")
        self.enable_check.toggled.connect(self._on_enable_toggled)
        self.enable_check.toggled.connect(self.settings_changed.emit)
        layout.addWidget(self.enable_check)
        
        self.params_frame = QFrame()
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Linear gradient",
            "Quadratic",
            "Sinusoidal",
            "From phantom df_map"
        ])
        self.model_combo.currentTextChanged.connect(self.settings_changed.emit)
        params_layout.addWidget(self.model_combo, 0, 1)
        
        params_layout.addWidget(QLabel("Gradient (Hz/cm):"), 1, 0)
        self.gradient_spin = QDoubleSpinBox()
        self.gradient_spin.setRange(-200, 200)
        self.gradient_spin.setValue(20)
        self.gradient_spin.setDecimals(1)
        self.gradient_spin.valueChanged.connect(self.settings_changed.emit)
        params_layout.addWidget(self.gradient_spin, 1, 1)
        
        params_layout.addWidget(QLabel("Direction:"), 1, 2)
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["X", "Y", "Diagonal"])
        self.direction_combo.setCurrentIndex(1)
        self.direction_combo.currentTextChanged.connect(self.settings_changed.emit)
        params_layout.addWidget(self.direction_combo, 1, 3)
        
        self.params_frame.setLayout(params_layout)
        layout.addWidget(self.params_frame)
        
        self.setLayout(layout)
        self.params_frame.setEnabled(False)
    
    def _on_enable_toggled(self, enabled: bool):
        self.params_frame.setEnabled(enabled)
    
    def set_phantom_b0(self, b0_map: np.ndarray):
        """Set B0 map from phantom."""
        self._phantom_b0 = b0_map
    
    def is_enabled(self) -> bool:
        return self.enable_check.isChecked()
    
    def generate_b0_map(self, shape: Tuple[int, int], 
                        fov: Tuple[float, float]) -> Optional[np.ndarray]:
        """Generate B0 map based on current settings."""
        if not self.enable_check.isChecked():
            return None
        
        nx, ny = shape
        fov_x, fov_y = fov
        
        x = np.linspace(-fov_x/2, fov_x/2, nx)
        y = np.linspace(-fov_y/2, fov_y/2, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        model = self.model_combo.currentText()
        grad = self.gradient_spin.value() * 100  # Hz/cm to Hz/m
        direction = self.direction_combo.currentText()
        
        if model == "From phantom df_map":
            if self._phantom_b0 is not None:
                # Resize if needed
                if self._phantom_b0.shape[:2] == shape:
                    return self._phantom_b0
                else:
                    from scipy.ndimage import zoom
                    factors = (shape[0] / self._phantom_b0.shape[0],
                              shape[1] / self._phantom_b0.shape[1])
                    return zoom(self._phantom_b0, factors)
            else:
                return np.zeros(shape)
        
        elif model == "Linear gradient":
            if direction == "X":
                b0 = grad * X
            elif direction == "Y":
                b0 = grad * Y
            else:
                b0 = grad * (X + Y) / np.sqrt(2)
        
        elif model == "Quadratic":
            b0 = grad * 0.5 * (X**2 + Y**2) / (fov_x * fov_y)
        
        else:  # Sinusoidal
            b0 = grad * np.sin(2 * np.pi * X / fov_x) * np.cos(2 * np.pi * Y / fov_y)
        
        return b0


class KSpaceViewer(QWidget):
    """Widget for visualizing k-space simulation results."""
    
    def __init__(self):
        super().__init__()
        self.current_result = None
        self.current_trajectory = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Tabs for different views
        self.tabs = QTabWidget()
        
        # Trajectory tab
        traj_widget = QWidget()
        traj_layout = QVBoxLayout()
        self.traj_plot = pg.PlotWidget()
        self.traj_plot.setLabel('left', 'ky (cycles/m)')
        self.traj_plot.setLabel('bottom', 'kx (cycles/m)')
        self.traj_plot.setAspectLocked(True)
        traj_layout.addWidget(self.traj_plot)
        traj_widget.setLayout(traj_layout)
        self.tabs.addTab(traj_widget, "Trajectory")
        
        # Gradients tab
        grad_widget = QWidget()
        grad_layout = QVBoxLayout()
        self.grad_plot = pg.PlotWidget()
        self.grad_plot.setLabel('left', 'Gradient (G/cm)')
        self.grad_plot.setLabel('bottom', 'Time (ms)')
        self.grad_plot.addLegend()
        grad_layout.addWidget(self.grad_plot)
        grad_widget.setLayout(grad_layout)
        self.tabs.addTab(grad_widget, "Gradients")
        
        # K-space data tab
        kspace_widget = QWidget()
        kspace_layout = QVBoxLayout()
        
        # Controls
        kspace_controls = QHBoxLayout()
        kspace_controls.addWidget(QLabel("Display:"))
        self.kspace_display_combo = QComboBox()
        self.kspace_display_combo.addItems(["Magnitude", "Phase", "Real", "Imaginary"])
        self.kspace_display_combo.currentTextChanged.connect(self._update_kspace_view)
        kspace_controls.addWidget(self.kspace_display_combo)
        
        self.kspace_log_check = QCheckBox("Log scale")
        self.kspace_log_check.setChecked(True)
        self.kspace_log_check.toggled.connect(self._update_kspace_view)
        kspace_controls.addWidget(self.kspace_log_check)
        kspace_controls.addStretch()
        kspace_layout.addLayout(kspace_controls)
        
        self.kspace_image = pg.ImageView()
        kspace_layout.addWidget(self.kspace_image)
        kspace_widget.setLayout(kspace_layout)
        self.tabs.addTab(kspace_widget, "K-Space")
        
        # Reconstructed image tab
        image_widget = QWidget()
        image_layout = QVBoxLayout()
        
        # Controls
        image_controls = QHBoxLayout()
        image_controls.addWidget(QLabel("Display:"))
        self.image_display_combo = QComboBox()
        self.image_display_combo.addItems(["Magnitude", "Phase", "Real", "Imaginary"])
        self.image_display_combo.currentTextChanged.connect(self._update_image_view)
        image_controls.addWidget(self.image_display_combo)
        image_controls.addStretch()
        image_layout.addLayout(image_controls)
        
        self.recon_image = pg.ImageView()
        image_layout.addWidget(self.recon_image)
        image_widget.setLayout(image_layout)
        self.tabs.addTab(image_widget, "Image")
        
        # Spectrum tab (for CSI)
        spectrum_widget = QWidget()
        spectrum_layout = QVBoxLayout()
        
        # Voxel selector
        spectrum_controls = QHBoxLayout()
        spectrum_controls.addWidget(QLabel("Voxel X:"))
        self.spectrum_x_spin = QSpinBox()
        self.spectrum_x_spin.setRange(0, 0)
        self.spectrum_x_spin.valueChanged.connect(self._update_spectrum_view)
        spectrum_controls.addWidget(self.spectrum_x_spin)
        
        spectrum_controls.addWidget(QLabel("Y:"))
        self.spectrum_y_spin = QSpinBox()
        self.spectrum_y_spin.setRange(0, 0)
        self.spectrum_y_spin.valueChanged.connect(self._update_spectrum_view)
        spectrum_controls.addWidget(self.spectrum_y_spin)
        spectrum_controls.addStretch()
        spectrum_layout.addLayout(spectrum_controls)
        
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel('left', 'Magnitude')
        self.spectrum_plot.setLabel('bottom', 'Frequency (Hz)')
        spectrum_layout.addWidget(self.spectrum_plot)
        spectrum_widget.setLayout(spectrum_layout)
        self.tabs.addTab(spectrum_widget, "Spectrum")
        
        # Comparison tab (with/without artifacts)
        compare_widget = QWidget()
        compare_layout = QVBoxLayout()
        
        compare_plots = QHBoxLayout()
        
        self.compare_clean = pg.ImageView()
        compare_plots.addWidget(self.compare_clean)
        
        self.compare_artifact = pg.ImageView()
        compare_plots.addWidget(self.compare_artifact)
        
        compare_layout.addLayout(compare_plots)
        compare_widget.setLayout(compare_layout)
        self.tabs.addTab(compare_widget, "Comparison")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def set_trajectory(self, trajectory: Dict):
        """Display k-space trajectory."""
        self.current_trajectory = trajectory
        
        traj_type = trajectory.get('trajectory_type', TrajectoryType.CARTESIAN)
        
        # Clear and plot trajectory
        self.traj_plot.clear()
        
        if traj_type == TrajectoryType.CSI:
            # CSI: phase encoding grid
            kx = trajectory['kx_spatial']
            ky = trajectory['ky_spatial']
            self.traj_plot.plot(kx, ky, pen=None, symbol='o', 
                               symbolSize=8, symbolBrush='b')
        else:
            kx = trajectory['kx']
            ky = trajectory['ky']
            
            if traj_type == TrajectoryType.EPI:
                # EPI: line trajectory
                self.traj_plot.plot(kx, ky, pen=pg.mkPen('b', width=1))
                # Mark start/end
                self.traj_plot.plot([kx[0]], [ky[0]], pen=None, symbol='o',
                                   symbolSize=10, symbolBrush='g')
                self.traj_plot.plot([kx[-1]], [ky[-1]], pen=None, symbol='o',
                                   symbolSize=10, symbolBrush='r')
            else:
                # Cartesian: scatter
                self.traj_plot.plot(kx, ky, pen=None, symbol='o',
                                   symbolSize=3, symbolBrush='b')
        
        # Plot gradients if available
        self.grad_plot.clear()
        if 'gradients' in trajectory:
            grad = trajectory['gradients']
            dt = trajectory.get('dt', 4e-6)
            t_ms = np.arange(len(grad)) * dt * 1000
            
            self.grad_plot.plot(t_ms, grad[:, 0], pen='b', name='Gx (read)')
            self.grad_plot.plot(t_ms, grad[:, 1], pen='r', name='Gy (phase)')
            if grad.shape[1] > 2:
                self.grad_plot.plot(t_ms, grad[:, 2], pen='g', name='Gz (slice)')
    
    def set_result(self, result: Dict):
        """Display simulation result."""
        self.current_result = result
        
        recon_type = result.get('recon_type', 'image')
        
        if recon_type == 'csi':
            # CSI result
            recon = result['reconstruction']
            spectra = recon['spectra']
            
            # Update spectrum selector ranges
            if len(spectra.shape) >= 2:
                self.spectrum_x_spin.setRange(0, spectra.shape[0] - 1)
                self.spectrum_y_spin.setRange(0, spectra.shape[1] - 1)
            
            # Show spatial image (integrated spectrum)
            spatial_image = recon.get('spatial_image', np.sum(np.abs(spectra), axis=-1))
            self.recon_image.setImage(spatial_image.T)
            
            # Show spectrum for selected voxel
            self._update_spectrum_view()
            
            # Enable spectrum tab
            self.tabs.setTabEnabled(4, True)
        else:
            # Image result
            recon = result['reconstruction']
            self._display_image(self.recon_image, recon, 
                               self.image_display_combo.currentText())
            
            # Disable spectrum tab
            self.tabs.setTabEnabled(4, False)
        
        # Update k-space view
        self._update_kspace_view()
    
    def set_comparison(self, clean_image: np.ndarray, artifact_image: np.ndarray):
        """Set comparison images (clean vs with artifacts)."""
        self.compare_clean.setImage(np.abs(clean_image).T)
        self.compare_artifact.setImage(np.abs(artifact_image).T)
    
    def _update_kspace_view(self):
        """Update k-space display."""
        if self.current_result is None:
            return
        
        signal = self.current_result.get('signal')
        if signal is None:
            return
        
        trajectory = self.current_result.get('trajectory', self.current_trajectory)
        if trajectory is None:
            return
        
        # Reshape signal to k-space grid
        traj_type = trajectory.get('trajectory_type', TrajectoryType.CARTESIAN)
        
        if traj_type == TrajectoryType.CSI:
            # CSI: (n_spatial, n_spectral) - show first spectral point
            kspace = signal[:, 0].reshape(trajectory['params'].matrix_size)
        else:
            matrix = trajectory.get('matrix_size', (32, 32))
            if len(signal) == np.prod(matrix):
                kspace = signal.reshape(matrix)
            else:
                # Can't reshape - just show as 1D
                self.kspace_image.setImage(np.abs(signal)[np.newaxis, :])
                return
        
        display = self.kspace_display_combo.currentText()
        use_log = self.kspace_log_check.isChecked()
        
        if display == "Magnitude":
            data = np.abs(kspace)
            if use_log:
                data = np.log1p(data)
        elif display == "Phase":
            data = np.angle(kspace)
        elif display == "Real":
            data = np.real(kspace)
        else:
            data = np.imag(kspace)
        
        self.kspace_image.setImage(data.T)
    
    def _update_image_view(self):
        """Update reconstructed image display."""
        if self.current_result is None:
            return
        
        recon = self.current_result.get('reconstruction')
        if recon is None:
            return
        
        if isinstance(recon, dict):
            # CSI - use spatial image
            recon = recon.get('spatial_image')
        
        if recon is not None:
            self._display_image(self.recon_image, recon,
                               self.image_display_combo.currentText())
    
    def _display_image(self, image_view: pg.ImageView, data: np.ndarray, display_type: str):
        """Display image with specified display type."""
        if display_type == "Magnitude":
            img = np.abs(data)
        elif display_type == "Phase":
            img = np.angle(data)
        elif display_type == "Real":
            img = np.real(data)
        else:
            img = np.imag(data)
        
        image_view.setImage(img.T)
    
    def _update_spectrum_view(self):
        """Update spectrum display for CSI."""
        if self.current_result is None:
            return
        
        recon = self.current_result.get('reconstruction')
        if not isinstance(recon, dict) or 'spectra' not in recon:
            return
        
        spectra = recon['spectra']
        freq = recon.get('frequency', np.arange(spectra.shape[-1]))
        
        x = self.spectrum_x_spin.value()
        y = self.spectrum_y_spin.value()
        
        if x < spectra.shape[0] and y < spectra.shape[1]:
            spectrum = np.abs(spectra[x, y, :])
            
            self.spectrum_plot.clear()
            self.spectrum_plot.plot(freq, spectrum, pen='b')


class KSpaceWidget(QWidget):
    """
    Main k-space simulation widget.
    
    This integrates with BlochSimulatorGUI similar to PhantomWidget.
    """
    
    simulation_started = pyqtSignal()
    simulation_finished = pyqtSignal(dict)
    
    def __init__(self, parent=None, get_phantom_callback=None, 
                 get_magnetization_callback=None):
        """
        Initialize the K-Space widget.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget (usually BlochSimulatorGUI)
        get_phantom_callback : callable, optional
            Function to get current phantom from PhantomWidget
        get_magnetization_callback : callable, optional
            Function to get magnetization from Bloch simulation
        """
        super().__init__(parent)
        self.parent_gui = parent
        self.get_phantom_callback = get_phantom_callback
        self.get_magnetization_callback = get_magnetization_callback
        
        self.simulator = KSpaceSimulator(verbose=False)
        self.current_phantom = None
        self.current_trajectory = None
        self.current_result = None
        self.simulation_thread = None
        
        self.init_ui()
    
    def init_ui(self):
        main_layout = QHBoxLayout()
        
        # Left panel: Settings
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setMaximumWidth(380)
        
        # Scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        settings_widget = QWidget()
        settings_layout = QVBoxLayout()
        
        # Phantom info
        phantom_group = QGroupBox("Phantom")
        phantom_layout = QVBoxLayout()
        
        self.phantom_info_label = QLabel("No phantom loaded")
        phantom_layout.addWidget(self.phantom_info_label)
        
        phantom_btn_layout = QHBoxLayout()
        self.load_phantom_btn = QPushButton("Load from Phantom Tab")
        self.load_phantom_btn.clicked.connect(self._load_phantom_from_tab)
        phantom_btn_layout.addWidget(self.load_phantom_btn)
        
        self.create_spectral_btn = QPushButton("Create Spectral...")
        self.create_spectral_btn.clicked.connect(self._create_spectral_phantom)
        phantom_btn_layout.addWidget(self.create_spectral_btn)
        phantom_layout.addLayout(phantom_btn_layout)
        
        phantom_group.setLayout(phantom_layout)
        settings_layout.addWidget(phantom_group)
        
        # Trajectory settings
        self.trajectory_widget = TrajectoryWidget()
        self.trajectory_widget.match_phantom_btn.clicked.connect(self._match_phantom)
        self.trajectory_widget.settings_changed.connect(self._update_trajectory_preview)
        settings_layout.addWidget(self.trajectory_widget)
        
        # Eddy currents
        self.eddy_widget = EddyCurrentWidget()
        settings_layout.addWidget(self.eddy_widget)
        
        # B0 inhomogeneity
        self.b0_widget = B0Widget()
        settings_layout.addWidget(self.b0_widget)
        
        # Noise
        noise_group = QGroupBox("Noise")
        noise_layout = QHBoxLayout()
        self.noise_check = QCheckBox("Add noise")
        noise_layout.addWidget(self.noise_check)
        noise_layout.addWidget(QLabel("SNR:"))
        self.snr_spin = QSpinBox()
        self.snr_spin.setRange(1, 1000)
        self.snr_spin.setValue(50)
        self.snr_spin.setEnabled(False)
        self.noise_check.toggled.connect(self.snr_spin.setEnabled)
        noise_layout.addWidget(self.snr_spin)
        noise_layout.addStretch()
        noise_group.setLayout(noise_layout)
        settings_layout.addWidget(noise_group)
        
        settings_layout.addStretch()
        
        scroll.setWidget(settings_widget)
        settings_widget.setLayout(settings_layout)
        left_layout.addWidget(scroll)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.simulate_btn = QPushButton("▶ Simulate K-Space")
        self.simulate_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.simulate_btn.clicked.connect(self.run_simulation)
        btn_layout.addWidget(self.simulate_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_simulation)
        btn_layout.addWidget(self.cancel_btn)
        
        left_layout.addLayout(btn_layout)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready - Load a phantom to begin")
        left_layout.addWidget(self.status_label)
        
        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)
        
        # Right panel: Visualization
        self.viewer = KSpaceViewer()
        main_layout.addWidget(self.viewer, stretch=1)
        
        self.setLayout(main_layout)
    
    def _load_phantom_from_tab(self):
        """Load phantom from the PhantomWidget tab."""
        phantom = None
        
        # First try the callback
        if self.get_phantom_callback is not None:
            phantom = self.get_phantom_callback()
        
        # If no callback or callback returned None, try to access parent GUI directly
        if phantom is None and self.parent_gui is not None:
            if hasattr(self.parent_gui, 'phantom_widget') and self.parent_gui.phantom_widget is not None:
                pw = self.parent_gui.phantom_widget
                # PhantomWidget stores phantom in current_phantom
                if hasattr(pw, 'current_phantom') and pw.current_phantom is not None:
                    phantom = pw.current_phantom
                # Also check phantom_creator.current_phantom as fallback
                elif hasattr(pw, 'phantom_creator') and hasattr(pw.phantom_creator, 'current_phantom'):
                    phantom = pw.phantom_creator.current_phantom
        
        if phantom is not None:
            self._set_phantom(phantom)
            return
        
        QMessageBox.warning(self, "K-Space",
            "No phantom available. Please create one in the Phantom tab first.")
    
    def _set_phantom(self, phantom):
        """Set the current phantom."""
        self.current_phantom = phantom
        
        # Update info label
        if hasattr(phantom, 'shape'):
            shape_str = 'x'.join(map(str, phantom.shape))
        else:
            shape_str = "unknown"
        
        if hasattr(phantom, 'name'):
            name = phantom.name
        else:
            name = type(phantom).__name__
        
        self.phantom_info_label.setText(f"{name}\nShape: {shape_str}")
        
        # Update B0 widget if phantom has df_map
        if hasattr(phantom, 'df_map') and phantom.df_map is not None:
            self.b0_widget.set_phantom_b0(phantom.df_map)
        
        self.status_label.setText(f"Phantom loaded: {shape_str}")
    
    def _create_spectral_phantom(self):
        """Create a spectral phantom for CSI simulation."""
        # Show dialog to select spectral phantom type
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Spectral Phantom")
        dialog.setModal(True)
        
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Select phantom type:"))
        
        type_combo = QComboBox()
        type_combo.addItems([
            "Brain MRS (single voxel)",
            "Brain CSI (16x16)",
            "Fat-Water (32x32)",
        ])
        layout.addWidget(type_combo)
        
        # Field strength
        field_layout = QHBoxLayout()
        field_layout.addWidget(QLabel("Field strength:"))
        field_combo = QComboBox()
        field_combo.addItems(["1.5T", "3.0T", "7.0T"])
        field_combo.setCurrentText("3.0T")
        field_layout.addWidget(field_combo)
        layout.addLayout(field_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Create")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            field_map = {"1.5T": 1.5, "3.0T": 3.0, "7.0T": 7.0}
            field = field_map.get(field_combo.currentText(), 3.0)
            
            phantom_type = type_combo.currentText()
            
            if "Brain MRS" in phantom_type:
                phantom = SpectralPhantomFactory.brain_mrs_voxel(field)
            elif "Brain CSI" in phantom_type:
                phantom = SpectralPhantomFactory.brain_csi_grid(
                    matrix_size=(16, 16), field_strength=field
                )
            else:  # Fat-Water
                phantom = SpectralPhantomFactory.fat_water_phantom(
                    matrix_size=(32, 32), field_strength=field
                )
            
            self._set_phantom(phantom)
            
            # Auto-select CSI trajectory for spectral phantoms
            if "CSI" in phantom_type or "MRS" in phantom_type:
                self.trajectory_widget.type_combo.setCurrentText("CSI")
    
    def _match_phantom(self):
        """Match trajectory settings to phantom."""
        if self.current_phantom is not None:
            self.trajectory_widget.set_from_phantom(self.current_phantom)
    
    def _update_trajectory_preview(self):
        """Update trajectory preview in viewer."""
        try:
            traj_type = self.trajectory_widget.get_type()
            matrix = self.trajectory_widget.get_matrix_size()
            fov = self.trajectory_widget.get_fov()
            
            if traj_type == "Cartesian":
                traj = self.simulator.generate_cartesian_trajectory(matrix, fov)
            elif traj_type == "EPI":
                params = self.trajectory_widget.get_epi_params()
                traj = self.simulator.generate_epi_trajectory(params)
            else:  # CSI
                params = self.trajectory_widget.get_csi_params()
                traj = self.simulator.generate_csi_trajectory(params)
            
            self.current_trajectory = traj
            self.viewer.set_trajectory(traj)
            
        except Exception as e:
            self.status_label.setText(f"Trajectory error: {str(e)}")
    
    def run_simulation(self):
        """Run k-space simulation."""
        if self.current_phantom is None:
            QMessageBox.warning(self, "K-Space Simulation",
                "Please load a phantom first.")
            return
        
        # Generate trajectory
        self._update_trajectory_preview()
        if self.current_trajectory is None:
            QMessageBox.warning(self, "K-Space Simulation",
                "Failed to generate trajectory.")
            return
        
        # Get eddy current model
        eddy_model = self.eddy_widget.get_eddy_model()
        
        # Get B0 map
        matrix = self.trajectory_widget.get_matrix_size()
        fov = self.trajectory_widget.get_fov()
        b0_map = self.b0_widget.generate_b0_map(matrix, fov)
        
        # Get noise level
        noise_std = 0.0
        if self.noise_check.isChecked():
            noise_std = 1.0 / self.snr_spin.value()
        
        # Get magnetization if available
        magnetization = None
        if self.get_magnetization_callback is not None:
            magnetization = self.get_magnetization_callback()
        
        # Update UI
        self.simulate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting simulation...")
        
        # Create and start worker thread
        self.simulation_thread = KSpaceSimulationThread(
            self.simulator,
            self.current_phantom,
            self.current_trajectory,
            magnetization=magnetization,
            eddy_model=eddy_model,
            b0_map=b0_map,
            noise_std=noise_std
        )
        
        self.simulation_thread.progress.connect(self._on_progress)
        self.simulation_thread.finished.connect(self._on_finished)
        self.simulation_thread.error.connect(self._on_error)
        
        self.simulation_thread.start()
        self.simulation_started.emit()
    
    def cancel_simulation(self):
        """Cancel running simulation."""
        if self.simulation_thread is not None:
            self.simulation_thread.request_cancel()
            self.simulation_thread.quit()
            self.simulation_thread.wait()
            self.simulation_thread = None
        
        self._reset_controls()
        self.status_label.setText("Cancelled")
    
    def _on_progress(self, value: int, message: str):
        """Handle progress updates."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def _on_finished(self, result: Dict):
        """Handle simulation completion."""
        self.current_result = result
        self._reset_controls()
        
        # Update viewer
        self.viewer.set_result(result)
        
        self.status_label.setText("Simulation complete")
        self.simulation_finished.emit(result)
    
    def _on_error(self, error_msg: str):
        """Handle simulation error."""
        self._reset_controls()
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Simulation Error", error_msg)
    
    def _reset_controls(self):
        """Reset UI controls after simulation."""
        self.simulate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)


# =============================================================================
# INTEGRATION PATCH FOR bloch_gui.py
# =============================================================================
"""
To integrate KSpaceWidget into your existing bloch_gui.py:

1. Add import at the top of bloch_gui.py:

    # Import k-space simulation widget
    try:
        from kspace_widget import KSpaceWidget
        KSPACE_AVAILABLE = True
    except ImportError:
        KSPACE_AVAILABLE = False
        print("K-Space module not available - k-space tab will be disabled")

2. In BlochSimulatorGUI.init_ui(), after the phantom tab creation (around line 2433),
   add:

    # === K-SPACE TAB ===
    if KSPACE_AVAILABLE:
        def get_phantom_for_kspace():
            if self.phantom_widget is not None:
                return self.phantom_widget.creator.current_phantom
            return None
        
        def get_magnetization_for_kspace():
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

That's it! The K-Space tab will now appear in your GUI.
"""


if __name__ == "__main__":
    # Test the widget standalone
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create standalone window
    window = QWidget()
    window.setWindowTitle("K-Space Simulation Test")
    window.resize(1200, 800)
    
    layout = QVBoxLayout()
    
    widget = KSpaceWidget()
    layout.addWidget(widget)
    
    window.setLayout(layout)
    window.show()
    
    sys.exit(app.exec_())
