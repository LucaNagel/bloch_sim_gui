"""
phantom_widget.py - GUI Widget for phantom-based MRI simulation

This module provides a PyQt5 widget for creating, visualizing, and simulating
MRI phantoms using the EXISTING pulse sequences from the main GUI.

The widget integrates with BlochSimulatorGUI by:
1. Using the same sequence designer (RF pulses, gradients, timing)
2. Applying sequences to 2D/3D phantoms with spatially-varying T1/T2
3. Displaying results as images

Author: Luca Nagel
Date: 2024/2025
"""

import numpy as np
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QTabWidget, QSplitter,
    QProgressBar, QCheckBox, QSlider, QFileDialog, QMessageBox,
    QGridLayout, QFrame, QScrollArea, QSizePolicy, QProgressDialog,
    QApplication, QDialog, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

import pyqtgraph as pg

# Import phantom module
from .phantom import Phantom, PhantomFactory

# Import simulator
from .simulator import BlochSimulator


class PhantomSimulationThread(QThread):
    """Thread for running phantom simulations without blocking the GUI."""
    
    progress = pyqtSignal(int, str)  # progress percentage, status message
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, simulator, phantom, sequence, dt, mode=0):
        super().__init__()
        self.simulator = simulator
        self.phantom = phantom
        self.sequence = sequence
        self.dt = dt
        self.mode = mode
        self._cancel_requested = False
    
    def request_cancel(self):
        self._cancel_requested = True
    
    def run(self):
        try:
            if self._cancel_requested:
                return
            
            self.progress.emit(5, "Preparing phantom...")
            
            # Run simulation using the simulator's simulate_phantom method
            result = self.simulator.simulate_phantom(
                self.phantom,
                self.sequence,
                dt=self.dt,
                mode=self.mode
            )
            
            if self._cancel_requested:
                return
            
            self.progress.emit(100, "Complete")
            self.finished.emit(result)
            
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class PhantomCreatorWidget(QGroupBox):
    """Widget for creating and configuring phantoms."""
    
    phantom_created = pyqtSignal(object)  # Emits Phantom object
    
    def __init__(self):
        super().__init__("Phantom Configuration")
        self.current_phantom = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Phantom type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "Shepp-Logan 2D",
            "Cylindrical 2D",
            "Multi-Tissue 2D",
            "Chemical Shift (Water/Fat)",
            "Spherical 3D",
            "Load from File..."
        ])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)
        
        # Resolution
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Matrix Size:"))
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(8, 256)
        self.resolution_spin.setValue(32)
        self.resolution_spin.setSingleStep(8)
        self.resolution_spin.setToolTip("Image matrix size (NxN). Larger = slower but more detailed.")
        res_layout.addWidget(self.resolution_spin)
        layout.addLayout(res_layout)
        
        # FOV
        fov_layout = QHBoxLayout()
        fov_layout.addWidget(QLabel("FOV:"))
        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(1, 100)
        self.fov_spin.setValue(24)
        self.fov_spin.setSingleStep(1)
        self.fov_spin.setSuffix(" cm")
        self.fov_spin.setDecimals(1)
        self.fov_spin.setToolTip("Field of view in centimeters")
        fov_layout.addWidget(self.fov_spin)
        layout.addLayout(fov_layout)
        
        # Field strength
        field_layout = QHBoxLayout()
        field_layout.addWidget(QLabel("B0:"))
        self.field_combo = QComboBox()
        self.field_combo.addItems(["1.5T", "3.0T", "7.0T"])
        self.field_combo.setCurrentText("3.0T")
        self.field_combo.setToolTip("Main magnetic field strength (affects T1/T2 values)")
        field_layout.addWidget(self.field_combo)
        layout.addLayout(field_layout)
        
        # Tissue type (for single-tissue phantoms)
        self.tissue_layout = QHBoxLayout()
        self.tissue_label = QLabel("Tissue:")
        self.tissue_layout.addWidget(self.tissue_label)
        self.tissue_combo = QComboBox()
        self.tissue_combo.addItems([
            "gray_matter", "white_matter", "csf", "fat", "muscle", "blood"
        ])
        self.tissue_combo.setCurrentText("gray_matter")
        self.tissue_layout.addWidget(self.tissue_combo)
        layout.addLayout(self.tissue_layout)
        
        # Create button
        btn_layout = QHBoxLayout()
        self.create_btn = QPushButton("Create Phantom")
        self.create_btn.clicked.connect(self.create_phantom)
        self.create_btn.setStyleSheet("font-weight: bold;")
        btn_layout.addWidget(self.create_btn)
        
        self.save_btn = QPushButton("Save...")
        self.save_btn.clicked.connect(self.save_phantom)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)
        
        # Info display
        self.info_label = QLabel("Click 'Create Phantom' to begin")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: gray; font-style: italic; padding: 5px;")
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
        self._on_type_changed(self.type_combo.currentText())
    
    def _on_type_changed(self, type_name: str):
        """Update UI based on selected phantom type."""
        # Show/hide tissue selector
        single_tissue_types = ["Cylindrical 2D", "Spherical 3D"]
        show_tissue = type_name in single_tissue_types
        self.tissue_label.setVisible(show_tissue)
        self.tissue_combo.setVisible(show_tissue)
        
        # Limit resolution for 3D
        if "3D" in type_name:
            self.resolution_spin.setMaximum(64)
            if self.resolution_spin.value() > 64:
                self.resolution_spin.setValue(32)
        else:
            self.resolution_spin.setMaximum(256)
    
    def get_field_strength(self) -> float:
        return float(self.field_combo.currentText().replace("T", ""))
    
    def create_phantom(self):
        """Create phantom based on current settings."""
        phantom_type = self.type_combo.currentText()
        
        if phantom_type == "Load from File...":
            self.load_phantom()
            return
        
        try:
            n = self.resolution_spin.value()
            fov_cm = self.fov_spin.value()
            fov_m = fov_cm / 100.0  # Convert cm to m
            field = self.get_field_strength()
            tissue = self.tissue_combo.currentText()
            
            if phantom_type == "Shepp-Logan 2D":
                phantom = PhantomFactory.shepp_logan_2d(n, fov_m, field)
            elif phantom_type == "Cylindrical 2D":
                phantom = PhantomFactory.cylindrical_2d(n, fov_m, tissue, field)
            elif phantom_type == "Multi-Tissue 2D":
                phantom = PhantomFactory.multi_tissue_2d(n, fov_m, field)
            elif phantom_type == "Chemical Shift (Water/Fat)":
                phantom = PhantomFactory.chemical_shift_phantom(n, fov_m, field)
            elif phantom_type == "Spherical 3D":
                phantom = PhantomFactory.spherical_3d(n, fov_m, tissue, field)
            else:
                return
            
            self.current_phantom = phantom
            self._update_info()
            self.save_btn.setEnabled(True)
            self.phantom_created.emit(phantom)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create phantom:\n{e}")
    
    def load_phantom(self):
        """Load phantom from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Phantom", "",
            "Phantom Files (*.npz *.h5 *.hdf5);;All Files (*)"
        )
        if filename:
            try:
                phantom = Phantom.load(filename)
                self.current_phantom = phantom
                self._update_info()
                self.save_btn.setEnabled(True)
                self.phantom_created.emit(phantom)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load phantom:\n{e}")
    
    def save_phantom(self):
        """Save current phantom to file."""
        if self.current_phantom is None:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Phantom", f"{self.current_phantom.name}.npz",
            "NumPy Archive (*.npz);;HDF5 File (*.h5)"
        )
        if filename:
            try:
                self.current_phantom.save(filename)
                QMessageBox.information(self, "Saved", f"Phantom saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save phantom:\n{e}")
    
    def _update_info(self):
        """Update info label with phantom details."""
        if self.current_phantom:
            p = self.current_phantom
            t1_active = p.t1_map[p.mask]
            t2_active = p.t2_map[p.mask]
            self.info_label.setText(
                f"<b>{p.name}</b><br>"
                f"Shape: {p.shape}<br>"
                f"Active voxels: {p.n_active} / {p.nvoxels}<br>"
                f"T1: {t1_active.min()*1000:.0f} - {t1_active.max()*1000:.0f} ms<br>"
                f"T2: {t2_active.min()*1000:.0f} - {t2_active.max()*1000:.0f} ms"
            )
            self.info_label.setStyleSheet("color: #006400; padding: 5px; background: #f0fff0; border-radius: 3px;")
        else:
            self.info_label.setText("Click 'Create Phantom' to begin")
            self.info_label.setStyleSheet("color: gray; font-style: italic; padding: 5px;")


class PhantomViewerWidget(QWidget):
    """Widget for visualizing phantom property maps and simulation results."""
    
    def __init__(self):
        super().__init__()
        self.phantom = None
        self.result = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        
        # === PHANTOM PROPERTIES TAB ===
        prop_widget = QWidget()
        prop_layout = QVBoxLayout()
        
        # Property selector
        prop_select_layout = QHBoxLayout()
        prop_select_layout.addWidget(QLabel("Display:"))
        self.prop_combo = QComboBox()
        self.prop_combo.addItems(["T1 Map", "T2 Map", "Proton Density", "Off-resonance (dF)", "Mask"])
        self.prop_combo.currentTextChanged.connect(self._update_property_view)
        prop_select_layout.addWidget(self.prop_combo)
        prop_select_layout.addStretch()
        prop_layout.addLayout(prop_select_layout)
        
        # Property image view
        self.prop_image = pg.ImageView()
        self.prop_image.ui.roiBtn.hide()
        self.prop_image.ui.menuBtn.hide()
        prop_layout.addWidget(self.prop_image)
        
        self.prop_info = QLabel("")
        self.prop_info.setAlignment(Qt.AlignCenter)
        prop_layout.addWidget(self.prop_info)
        
        prop_widget.setLayout(prop_layout)
        self.tabs.addTab(prop_widget, "Phantom Properties")
        
        # === SIMULATION RESULTS TAB ===
        result_widget = QWidget()
        result_layout = QVBoxLayout()
        
        # Result selector
        result_select_layout = QHBoxLayout()
        result_select_layout.addWidget(QLabel("Display:"))
        self.result_combo = QComboBox()
        self.result_combo.addItems(["|Mxy| (Transverse)", "Mx", "My", "Mz (Longitudinal)", 
                                    "|Signal|", "Phase (rad)"])
        self.result_combo.currentTextChanged.connect(self._update_result_view)
        result_select_layout.addWidget(self.result_combo)
        result_select_layout.addStretch()
        result_layout.addLayout(result_select_layout)
        
        # Time slider (for time-resolved)
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time:"))
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.setValue(0)
        self.time_slider.valueChanged.connect(self._update_result_view)
        self.time_slider.setEnabled(False)
        time_layout.addWidget(self.time_slider)
        self.time_label = QLabel("Endpoint")
        time_layout.addWidget(self.time_label)
        result_layout.addLayout(time_layout)
        
        # Result image view
        self.result_image = pg.ImageView()
        self.result_image.ui.roiBtn.hide()
        self.result_image.ui.menuBtn.hide()
        result_layout.addWidget(self.result_image)
        
        self.result_info = QLabel("")
        self.result_info.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.result_info)
        
        result_widget.setLayout(result_layout)
        self.tabs.addTab(result_widget, "Spatial Maps")
        
        # === RECEIVED SIGNAL TAB ===
        signal_widget = QWidget()
        signal_layout = QVBoxLayout()
        
        # Info label
        signal_info_layout = QHBoxLayout()
        signal_info_layout.addWidget(QLabel("<b>Received Signal</b> - Sum of all voxel signals (what an RF coil would measure)"))
        signal_info_layout.addStretch()
        signal_layout.addLayout(signal_info_layout)
        
        # Signal magnitude plot
        self.signal_mag_plot = pg.PlotWidget(title="Signal Magnitude |S(t)|")
        self.signal_mag_plot.setLabel('left', '|Signal|', units='a.u.')
        self.signal_mag_plot.setLabel('bottom', 'Time', units='ms')
        self.signal_mag_plot.showGrid(x=True, y=True, alpha=0.3)
        self.signal_mag_plot.addLegend()
        signal_layout.addWidget(self.signal_mag_plot)
        
        # Signal components plot (real/imag)
        self.signal_components_plot = pg.PlotWidget(title="Signal Components")
        self.signal_components_plot.setLabel('left', 'Signal', units='a.u.')
        self.signal_components_plot.setLabel('bottom', 'Time', units='ms')
        self.signal_components_plot.showGrid(x=True, y=True, alpha=0.3)
        self.signal_components_plot.addLegend()
        signal_layout.addWidget(self.signal_components_plot)
        
        # Signal statistics
        self.signal_stats_label = QLabel("")
        self.signal_stats_label.setAlignment(Qt.AlignCenter)
        self.signal_stats_label.setStyleSheet("padding: 5px; background: #f5f5f5; border-radius: 3px;")
        signal_layout.addWidget(self.signal_stats_label)
        
        signal_widget.setLayout(signal_layout)
        self.tabs.addTab(signal_widget, "Received Signal")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def set_phantom(self, phantom: Optional[Phantom]):
        """Set phantom to display."""
        self.phantom = phantom
        self._update_property_view()
        
        # Switch to properties tab
        self.tabs.setCurrentIndex(0)
    
    def set_result(self, result: Optional[Dict], phantom: Optional[Phantom] = None):
        """Set simulation result to display."""
        self.result = result
        if phantom is not None:
            self.phantom = phantom
        
        if result is None:
            self.result_image.clear()
            self.time_slider.setEnabled(False)
            self.result_info.setText("")
            return
        
        # Check if time-resolved
        mx = result.get('mx')
        if mx is not None and self.phantom is not None:
            if mx.ndim > len(self.phantom.shape):
                # Time-resolved
                ntime = mx.shape[0]
                self.time_slider.setEnabled(True)
                self.time_slider.setRange(0, ntime - 1)
                self.time_slider.setValue(ntime - 1)
            else:
                self.time_slider.setEnabled(False)
                self.time_slider.setRange(0, 0)
        
        self._update_result_view()
        self._update_signal_plot()
        
        # Switch to signal tab for time-resolved, results tab for endpoint
        if self.time_slider.isEnabled():
            self.tabs.setCurrentIndex(2)  # Signal tab
        else:
            self.tabs.setCurrentIndex(1)  # Spatial maps tab
    
    def _update_property_view(self):
        """Update property map display."""
        if self.phantom is None:
            self.prop_image.clear()
            self.prop_info.setText("No phantom loaded")
            return
        
        view = self.prop_combo.currentText()
        
        if "T1" in view:
            data = self.phantom.t1_map * 1000  # Convert to ms
            unit = "ms"
        elif "T2" in view:
            data = self.phantom.t2_map * 1000  # Convert to ms
            unit = "ms"
        elif "Proton" in view:
            data = self.phantom.pd_map
            unit = ""
        elif "dF" in view or "Off" in view:
            data = self.phantom.df_map
            unit = "Hz"
        elif "Mask" in view:
            data = self.phantom.mask.astype(float)
            unit = ""
        else:
            return
        
        # Handle 3D (show middle slice)
        display_data = data.copy()
        mask_2d = self.phantom.mask
        
        if data.ndim == 3:
            slice_idx = data.shape[2] // 2
            display_data = data[:, :, slice_idx]
            mask_2d = self.phantom.mask[:, :, slice_idx]
        
        # Mask background
        display_data = np.where(mask_2d, display_data, np.nan)
        
        self.prop_image.setImage(display_data.T, autoLevels=True)
        
        # Update info
        valid = data[self.phantom.mask]
        if len(valid) > 0:
            self.prop_info.setText(f"Range: {valid.min():.2f} - {valid.max():.2f} {unit}")
        else:
            self.prop_info.setText("No data")
    
    def _update_result_view(self):
        """Update simulation result display."""
        if self.result is None or self.phantom is None:
            self.result_image.clear()
            self.result_info.setText("No simulation results")
            return
        
        mx = self.result.get('mx')
        my = self.result.get('my')
        mz = self.result.get('mz')
        signal = self.result.get('signal')
        time_arr = self.result.get('time')
        
        if mx is None:
            return
        
        # Get time index
        time_idx = self.time_slider.value() if self.time_slider.isEnabled() else -1
        
        # Extract data at time point
        if mx.ndim > len(self.phantom.shape) and time_idx >= 0:
            mx_t = mx[time_idx]
            my_t = my[time_idx]
            mz_t = mz[time_idx]
            sig_t = signal[time_idx] if signal is not None and signal.ndim > len(self.phantom.shape) else signal
            
            # Update time label
            if time_arr is not None and time_idx < len(time_arr):
                self.time_label.setText(f"{time_arr[time_idx]*1000:.2f} ms")
        else:
            mx_t, my_t, mz_t = mx, my, mz
            sig_t = signal
            self.time_label.setText("Endpoint")
        
        # Calculate display data
        view = self.result_combo.currentText()
        
        if "|Mxy|" in view or "Transverse" in view:
            data = np.sqrt(mx_t**2 + my_t**2)
            label = "|Mxy|"
        elif view == "Mx":
            data = mx_t
            label = "Mx"
        elif view == "My":
            data = my_t
            label = "My"
        elif "Mz" in view:
            data = mz_t
            label = "Mz"
        elif "|Signal|" in view:
            if sig_t is not None:
                data = np.abs(sig_t)
            else:
                data = np.sqrt(mx_t**2 + my_t**2) * self.phantom.pd_map
            label = "|Signal|"
        elif "Phase" in view:
            data = np.arctan2(my_t, mx_t)
            label = "Phase"
        else:
            return
        
        # Handle 3D
        mask_2d = self.phantom.mask
        if data.ndim == 3:
            slice_idx = data.shape[2] // 2
            data = data[:, :, slice_idx]
            mask_2d = self.phantom.mask[:, :, slice_idx]
        
        # Mask background
        display_data = np.where(mask_2d, data, np.nan)
        
        self.result_image.setImage(display_data.T, autoLevels=True)
        
        # Update info
        valid = data[mask_2d]
        if len(valid) > 0:
            self.result_info.setText(f"{label}: min={valid.min():.4f}, max={valid.max():.4f}, mean={valid.mean():.4f}")
        else:
            self.result_info.setText("No data")
    
    def _update_signal_plot(self):
        """Update the received signal plots."""
        if self.result is None:
            self.signal_mag_plot.clear()
            self.signal_components_plot.clear()
            self.signal_stats_label.setText("No simulation results")
            return
        
        # Get received signal (summed over all voxels)
        received_signal = self.result.get('received_signal')
        time_arr = self.result.get('time')
        
        if received_signal is None or time_arr is None:
            self.signal_stats_label.setText("No received signal data (use time-resolved mode)")
            return
        
        # Check if it's a scalar (endpoint mode)
        if np.isscalar(received_signal) or received_signal.ndim == 0:
            self.signal_mag_plot.clear()
            self.signal_components_plot.clear()
            self.signal_stats_label.setText(
                f"Endpoint signal: |S| = {np.abs(received_signal):.4f}, "
                f"Phase = {np.angle(received_signal)*180/np.pi:.1f}°"
            )
            return
        
        # Time in ms
        time_ms = np.asarray(time_arr) * 1000
        
        # Ensure arrays match
        npts = min(len(time_ms), len(received_signal))
        time_ms = time_ms[:npts]
        received_signal = received_signal[:npts]
        
        # Clear plots
        self.signal_mag_plot.clear()
        self.signal_components_plot.clear()
        
        # Plot magnitude
        magnitude = np.abs(received_signal)
        self.signal_mag_plot.plot(time_ms, magnitude, pen=pg.mkPen('c', width=2), name='|S(t)|')
        
        # Plot real and imaginary components
        self.signal_components_plot.plot(time_ms, np.real(received_signal), 
                                         pen=pg.mkPen('g', width=2), name='Real')
        self.signal_components_plot.plot(time_ms, np.imag(received_signal), 
                                         pen=pg.mkPen('r', width=2), name='Imag')
        
        # Statistics
        max_mag = np.max(magnitude)
        max_idx = np.argmax(magnitude)
        max_time = time_ms[max_idx] if max_idx < len(time_ms) else 0
        final_mag = magnitude[-1] if len(magnitude) > 0 else 0
        
        self.signal_stats_label.setText(
            f"Peak: |S|={max_mag:.4f} at t={max_time:.2f} ms | "
            f"Final: |S|={final_mag:.4f} | "
            f"Total points: {npts}"
        )


class PhantomWidget(QWidget):
    """
    Main widget for phantom-based MRI simulation.
    
    This widget integrates with the main BlochSimulatorGUI to use the existing
    pulse sequence configuration while applying it to 2D/3D phantoms.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget (typically BlochSimulatorGUI)
    get_sequence_callback : callable, optional
        Function that returns (b1, gradients, time) tuple from main GUI
    get_dt_callback : callable, optional
        Function that returns time step dt from main GUI
    log_callback : callable, optional
        Function to log messages to main GUI console
    """
    
    def __init__(self, parent=None, get_sequence_callback=None, 
                 get_dt_callback=None, log_callback=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.get_sequence = get_sequence_callback
        self.get_dt = get_dt_callback
        self.log = log_callback or print
        
        self.simulator = BlochSimulator(use_parallel=True)
        self.current_phantom = None
        self.last_result = None
        self.sim_thread = None
        
        self.init_ui()
    
    def init_ui(self):
        main_layout = QHBoxLayout()
        
        # === LEFT PANEL - Configuration ===
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Phantom creator
        self.phantom_creator = PhantomCreatorWidget()
        self.phantom_creator.phantom_created.connect(self._on_phantom_created)
        left_layout.addWidget(self.phantom_creator)
        
        # Simulation settings
        sim_group = QGroupBox("Simulation Settings")
        sim_layout = QVBoxLayout()
        
        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Output:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Endpoint only (fast)", "Time-resolved (slow)"])
        self.mode_combo.setToolTip("Endpoint: final magnetization only. Time-resolved: full evolution.")
        mode_layout.addWidget(self.mode_combo)
        sim_layout.addLayout(mode_layout)
        
        # Info about sequence source
        self.seq_info = QLabel("<i>Uses sequence from main GUI panel</i>")
        self.seq_info.setWordWrap(True)
        self.seq_info.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        sim_layout.addWidget(self.seq_info)
        
        sim_group.setLayout(sim_layout)
        left_layout.addWidget(sim_group)
        
        # Run controls
        run_group = QGroupBox("Run")
        run_layout = QVBoxLayout()
        
        self.run_btn = QPushButton("Run Phantom Simulation")
        self.run_btn.clicked.connect(self.run_simulation)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.run_btn.setToolTip("Run the current sequence on the phantom")
        run_layout.addWidget(self.run_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_simulation)
        self.cancel_btn.setEnabled(False)
        run_layout.addWidget(self.cancel_btn)
        
        self.progress_bar = QProgressBar()
        run_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Create a phantom to begin")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: gray; padding: 3px;")
        run_layout.addWidget(self.status_label)
        
        run_group.setLayout(run_layout)
        left_layout.addWidget(run_group)
        
        left_layout.addStretch()
        
        # Make left panel scrollable
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        left_scroll.setMaximumWidth(320)
        left_scroll.setMinimumWidth(250)
        
        # === RIGHT PANEL - Visualization ===
        self.viewer = PhantomViewerWidget()
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
    
    def _on_phantom_created(self, phantom: Phantom):
        """Handle new phantom creation."""
        self.current_phantom = phantom
        self.viewer.set_phantom(phantom)
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"Phantom ready: {phantom.n_active} voxels")
        self.status_label.setStyleSheet("color: green; padding: 3px;")
        self.log(f"Phantom created: {phantom.name}, {phantom.shape}, {phantom.n_active} active voxels")
    
    def _get_sequence_from_parent(self):
        """Get the current sequence from the parent GUI."""
        if self.get_sequence is not None:
            return self.get_sequence()
        
        # Try to access parent GUI directly
        if self.parent_gui is not None:
            try:
                # Get pulse from RF designer
                pulse = self.parent_gui.rf_designer.get_pulse()
                dt_s = max(self.parent_gui.time_step_spin.value(), 0.1) * 1e-6
                
                # Compile sequence
                sequence = self.parent_gui.sequence_designer.compile_sequence(
                    custom_pulse=pulse, dt=dt_s
                )
                
                # Add tail if specified
                tail_ms = self.parent_gui.extra_tail_spin.value()
                if tail_ms > 0:
                    sequence = self.parent_gui._extend_sequence_with_tail(sequence, tail_ms, dt_s)
                
                return sequence, dt_s
            except Exception as e:
                self.log(f"Error getting sequence from GUI: {e}")
                return None, None
        
        return None, None
    
    def run_simulation(self):
        """Run phantom simulation using the current sequence from the main GUI."""
        if self.current_phantom is None:
            QMessageBox.warning(self, "No Phantom", "Please create a phantom first.")
            return
        
        # Get sequence from parent GUI
        result = self._get_sequence_from_parent()
        if result[0] is None:
            QMessageBox.warning(self, "No Sequence", 
                "Could not get pulse sequence from main GUI.\n"
                "Please configure a sequence in the main panel first.")
            return
        
        sequence, dt = result
        
        # Update sequence info
        b1, grads, time = sequence
        dur_ms = (time[-1] - time[0]) * 1000 if len(time) > 1 else 0
        self.seq_info.setText(
            f"Sequence: {len(b1)} points, {dur_ms:.2f} ms, dt={dt*1e6:.1f} us"
        )
        
        # Check memory for time-resolved
        mode = 0 if "Endpoint" in self.mode_combo.currentText() else 2
        nvoxels = self.current_phantom.n_active
        ntime = len(b1)
        
        if mode == 2:
            total_elements = nvoxels * ntime * 3
            if total_elements > 5e7:  # 50M elements warning
                reply = QMessageBox.question(
                    self, "Large Simulation",
                    f"Time-resolved simulation will generate {total_elements/1e6:.1f}M data points\n"
                    f"({nvoxels} voxels x {ntime} time points x 3 components).\n\n"
                    "This may be slow. Continue?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
        
        try:
            # Disable controls
            self.run_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("Starting simulation...")
            self.status_label.setStyleSheet("color: blue; padding: 3px;")
            
            self.log(f"Starting phantom simulation: {nvoxels} voxels, mode={mode}")
            
            # Start simulation thread
            self.sim_thread = PhantomSimulationThread(
                self.simulator,
                self.current_phantom,
                sequence,
                dt,
                mode
            )
            self.sim_thread.progress.connect(self._on_progress)
            self.sim_thread.finished.connect(self._on_simulation_finished)
            self.sim_thread.error.connect(self._on_simulation_error)
            self.sim_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start simulation:\n{e}")
            self._reset_controls()
    
    def cancel_simulation(self):
        """Cancel running simulation."""
        if self.sim_thread and self.sim_thread.isRunning():
            self.sim_thread.request_cancel()
            self.sim_thread.wait(2000)
            self._reset_controls()
            self.status_label.setText("Simulation cancelled")
            self.status_label.setStyleSheet("color: orange; padding: 3px;")
            self.log("Phantom simulation cancelled by user")
    
    def _on_progress(self, value: int, message: str):
        """Handle progress updates."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def _on_simulation_finished(self, result: Dict):
        """Handle simulation completion."""
        self.last_result = result
        self.viewer.set_result(result, self.current_phantom)
        self._reset_controls()
        
        # Calculate summary statistics
        mx = result.get('mx')
        my = result.get('my')
        mz = result.get('mz')
        
        if mx is not None:
            # Get endpoint values
            if mx.ndim > len(self.current_phantom.shape):
                mx_end = mx[-1]
                my_end = my[-1]
                mz_end = mz[-1]
            else:
                mx_end = mx
                my_end = my
                mz_end = mz
            
            mxy = np.sqrt(mx_end**2 + my_end**2)
            mask = self.current_phantom.mask
            
            # Handle dimension mismatch for 3D
            if mask.ndim != mxy.ndim:
                if mxy.ndim == 2 and mask.ndim == 3:
                    mask = mask[:, :, mask.shape[2]//2]
            
            # Ensure shapes match
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(mxy.shape, mask.shape))
            mxy_crop = mxy[:min_shape[0], :min_shape[1]] if mxy.ndim >= 2 else mxy
            mz_crop = mz_end[:min_shape[0], :min_shape[1]] if mz_end.ndim >= 2 else mz_end
            mask_crop = mask[:min_shape[0], :min_shape[1]]
            
            mxy_active = mxy_crop[mask_crop]
            mz_active = mz_crop[mask_crop]
            
            self.status_label.setText(
                f"Complete | |Mxy|: {mxy_active.mean():.3f}, Mz: {mz_active.mean():.3f}"
            )
            self.status_label.setStyleSheet("color: green; padding: 3px;")
            
            self.log(f"Phantom simulation complete: |Mxy| mean={mxy_active.mean():.4f}, Mz mean={mz_active.mean():.4f}")
    
    def _on_simulation_error(self, error_msg: str):
        """Handle simulation error."""
        QMessageBox.critical(self, "Simulation Error", f"Simulation failed:\n{error_msg}")
        self._reset_controls()
        self.status_label.setText(f"Error: {error_msg[:80]}...")
        self.status_label.setStyleSheet("color: red; padding: 3px;")
        self.log(f"Phantom simulation error: {error_msg}")
    
    def _reset_controls(self):
        """Reset control states after simulation."""
        self.run_btn.setEnabled(self.current_phantom is not None)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(0)


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Create a simple test sequence callback
    def test_sequence_callback():
        """Generate a simple 90 degree pulse for testing."""
        from blochsimulator import design_rf_pulse
        
        duration = 1e-3  # 1 ms
        flip_angle = 90
        dt = 10e-6  # 10 us
        npoints = int(duration / dt)
        
        b1, time = design_rf_pulse('rect', duration, flip_angle, npoints)
        gradients = np.zeros((len(b1), 3))
        
        return (b1, gradients, time), dt
    
    window = QWidget()
    window.setWindowTitle("Phantom Simulation - Standalone Test")
    window.resize(1200, 800)
    
    layout = QVBoxLayout()
    
    # Create widget with test callback
    phantom_widget = PhantomWidget(
        get_sequence_callback=lambda: test_sequence_callback()[0],
        get_dt_callback=lambda: test_sequence_callback()[1],
        log_callback=print
    )
    layout.addWidget(phantom_widget)
    
    window.setLayout(layout)
    window.show()
    
    sys.exit(app.exec_())
