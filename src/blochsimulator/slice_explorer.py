"""
slice_explorer.py - Interactive explorer for Slice Selection profiles.

This module provides a widget for designing slice-selective RF pulses
and simulating their excitation profiles.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QDoubleSpinBox,
    QComboBox,
    QSpinBox,
    QSplitter,
    QSizePolicy,
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg

from .simulator import (
    BlochSimulator,
    TissueParameters,
    SliceSelectRephase,
    design_rf_pulse,
)


class SliceSelectionExplorer(QWidget):
    """
    Widget for exploring slice selection profiles.
    Allows user to configure RF pulse and gradient parameters and visualizes
    the resulting magnetization profile across the slice.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.simulator = BlochSimulator()
        self.init_ui()
        # Trigger initial simulation
        self.run_simulation()

    def init_ui(self):
        layout = QHBoxLayout()

        # Left Panel: Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(350)

        # Pulse Parameters Group
        pulse_group = QGroupBox("Pulse Parameters")
        pulse_layout = QVBoxLayout()

        # Flip Angle
        row_flip = QHBoxLayout()
        row_flip.addWidget(QLabel("Flip Angle (°):"))
        self.flip_angle = QDoubleSpinBox()
        self.flip_angle.setRange(0, 180)
        self.flip_angle.setValue(90)
        row_flip.addWidget(self.flip_angle)
        pulse_layout.addLayout(row_flip)

        # Duration
        row_dur = QHBoxLayout()
        row_dur.addWidget(QLabel("Duration (ms):"))
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.1, 20.0)
        self.duration.setValue(2.0)
        self.duration.setSingleStep(0.1)
        row_dur.addWidget(self.duration)
        pulse_layout.addLayout(row_dur)

        # Time-Bandwidth Product
        row_tbw = QHBoxLayout()
        row_tbw.addWidget(QLabel("Time-BW Product:"))
        self.tbw = QDoubleSpinBox()
        self.tbw.setRange(1.0, 16.0)
        self.tbw.setValue(4.0)
        self.tbw.setSingleStep(0.5)
        row_tbw.addWidget(self.tbw)
        pulse_layout.addLayout(row_tbw)

        # Apodization
        row_apod = QHBoxLayout()
        row_apod.addWidget(QLabel("Apodization:"))
        self.apodization = QComboBox()
        self.apodization.addItems(["None", "Hamming", "Hanning", "Blackman"])
        self.apodization.setCurrentText("None")
        row_apod.addWidget(self.apodization)
        pulse_layout.addLayout(row_apod)

        pulse_group.setLayout(pulse_layout)
        control_layout.addWidget(pulse_group)

        # Slice Parameters Group
        slice_group = QGroupBox("Slice Parameters")
        slice_layout = QVBoxLayout()

        # Slice Thickness
        row_thick = QHBoxLayout()
        row_thick.addWidget(QLabel("Thickness (mm):"))
        self.thickness = QDoubleSpinBox()
        self.thickness.setRange(0.1, 20.0)
        self.thickness.setValue(5.0)
        self.thickness.setSingleStep(0.5)
        row_thick.addWidget(self.thickness)
        slice_layout.addLayout(row_thick)

        # Rephasing
        row_rephase = QHBoxLayout()
        self.use_rephase = QComboBox()
        self.use_rephase.addItems(["Rephase (50%)", "No Rephase"])
        row_rephase.addWidget(QLabel("Gradient:"))
        row_rephase.addWidget(self.use_rephase)
        slice_layout.addLayout(row_rephase)

        slice_group.setLayout(slice_layout)
        control_layout.addWidget(slice_group)

        # Simulation Parameters Group
        sim_group = QGroupBox("Simulation Grid")
        sim_layout = QVBoxLayout()

        # Position Range
        row_range = QHBoxLayout()
        row_range.addWidget(QLabel("Range (cm):"))
        self.pos_range = QDoubleSpinBox()
        self.pos_range.setRange(0.5, 20.0)
        self.pos_range.setValue(4.0)
        row_range.addWidget(self.pos_range)
        sim_layout.addLayout(row_range)

        # Number of Points
        row_points = QHBoxLayout()
        row_points.addWidget(QLabel("Points:"))
        self.num_points = QSpinBox()
        self.num_points.setRange(50, 2000)
        self.num_points.setValue(201)
        row_points.addWidget(self.num_points)
        sim_layout.addLayout(row_points)

        sim_group.setLayout(sim_layout)
        control_layout.addWidget(sim_group)

        # Action Buttons
        self.btn_simulate = QPushButton("Simulate Profile")
        self.btn_simulate.clicked.connect(self.run_simulation)
        control_layout.addWidget(self.btn_simulate)

        control_layout.addStretch()

        # Right Panel: Visualization
        viz_panel = QSplitter(Qt.Vertical)

        # RF Pulse Plot
        self.plot_rf = pg.PlotWidget(title="RF Pulse & Gradient")
        self.plot_rf.setLabel("left", "B1 (G) / Gz (G/cm)")
        self.plot_rf.setLabel("bottom", "Time (ms)")
        self.plot_rf.addLegend()
        viz_panel.addWidget(self.plot_rf)

        # Slice Profile Plot
        self.plot_profile = pg.PlotWidget(title="Excitation Profile (Mz vs Position)")
        self.plot_profile.setLabel("left", "Mz")
        self.plot_profile.setLabel("bottom", "Position (cm)")
        self.plot_profile.setYRange(-1.1, 1.1)
        self.plot_profile.addLegend()
        viz_panel.addWidget(self.plot_profile)

        layout.addWidget(control_panel)
        layout.addWidget(viz_panel)
        self.setLayout(layout)

        # Connect changes to auto-update (optional, maybe just button is safer for performance)
        # For now, let's auto-update on changes for responsiveness, unless it's too slow
        self.flip_angle.valueChanged.connect(self.run_simulation)
        self.duration.valueChanged.connect(self.run_simulation)
        self.tbw.valueChanged.connect(self.run_simulation)
        self.apodization.currentTextChanged.connect(self.run_simulation)
        self.thickness.valueChanged.connect(self.run_simulation)
        self.use_rephase.currentIndexChanged.connect(self.run_simulation)
        self.pos_range.valueChanged.connect(self.run_simulation)
        # self.num_points.valueChanged.connect(self.run_simulation) # Don't auto-update on points change while typing

    def run_simulation(self):
        """Build sequence and run Bloch simulation."""

        # 1. Gather Parameters
        flip = self.flip_angle.value()
        dur_s = self.duration.value() / 1000.0
        tbw = self.tbw.value()
        apod = self.apodization.currentText()
        thick_m = self.thickness.value() / 1000.0
        do_rephase = self.use_rephase.currentIndex() == 0

        range_cm = self.pos_range.value()
        n_points = self.num_points.value()

        # 2. Design RF Pulse (Sinc)
        # We use a custom design flow here to support apodization easily on the base shape
        # Or we can use design_rf_pulse and modify it.
        # design_rf_pulse in simulator.py doesn't support apodization args directly,
        # but the GUI's RFPulseDesigner does it manually. Let's mimic that.

        dt = 1e-5  # 10 us time step
        n_rf_pts = int(np.ceil(dur_s / dt))

        b1_base, time_rf = design_rf_pulse(
            pulse_type="sinc",
            duration=dur_s,
            flip_angle=flip,
            time_bw_product=tbw,
            npoints=n_rf_pts,
        )

        # Apply Apodization
        if apod != "None" and len(b1_base) > 1:
            if apod == "Hamming":
                win = np.hamming(len(b1_base))
            elif apod == "Hanning":
                win = np.hanning(len(b1_base))
            elif apod == "Blackman":
                win = np.blackman(len(b1_base))
            else:
                win = np.ones(len(b1_base))
            b1_base = b1_base * win

            # Re-scale to maintain flip angle after apodization
            # (Simplified re-scaling)
            # Ideally we'd integrate and re-scale.
            gamma = (
                4258.0 * 2 * np.pi
            )  # rad/s/G - wait, gamma in simulator.py is Hz/G usually?
            # simulator.py: gamma = 4258.0 Hz/G
            # target area (G*s) = flip_rad / (gamma_Hz_per_G * 2 * pi)

            target_area = np.deg2rad(flip) / (4258.0 * 2 * np.pi)
            current_area = np.trapz(np.abs(b1_base), dx=dt)
            if current_area > 0:
                scale = target_area / current_area
                b1_base *= scale

        # 3. Create Sequence
        # We can construct a SliceSelectRephase, passing our custom pulse
        rephase_dur = 0.0
        if do_rephase:
            rephase_dur = dur_s / 2.0  # Simple default, usually sufficient
            # But the class calculates rephase lobe based on slice gradient area.
            # SliceSelectRephase handles the rephase lobe creation.

        # We will manually construct the sequence tuple to have full control if needed,
        # or rely on SliceSelectRephase logic.
        # SliceSelectRephase logic for rephase_duration creates a lobe.
        # If we want "No Rephase", we can pass rephase_duration very small or handle gradients manually.

        if do_rephase:
            seq_obj = SliceSelectRephase(
                flip_angle=flip,
                pulse_duration=dur_s,
                time_bw_product=tbw,
                rephase_duration=0.5e-3,  # Fixed rephase time (0.5ms)
                slice_thickness=thick_m,
                custom_pulse=(b1_base, time_rf),
            )
            b1, grads, time = seq_obj.compile(dt=dt)
        else:
            # Custom construction without rephase
            # Calculate Slice Gradient
            bw_hz = tbw / dur_s
            gamma_hz_per_g = 4258.0
            gz_amp = bw_hz / (gamma_hz_per_g * (thick_m * 100))  # G/cm

            n_total = len(b1_base) + 10
            b1 = np.zeros(n_total, dtype=complex)
            b1[: len(b1_base)] = b1_base

            grads = np.zeros((n_total, 3))
            grads[: len(b1_base), 2] = gz_amp

            time = np.arange(n_total) * dt

        # 4. Define Spatial Grid
        half_range = range_cm / 2.0
        positions = np.zeros((n_points, 3))
        # Z-axis varies (slice direction)
        positions[:, 2] = np.linspace(
            -half_range / 100.0, half_range / 100.0, n_points
        )  # meters

        # 5. Run Simulation
        # Tissue: Long T1/T2 to ignore relaxation effects on profile shape
        tissue = TissueParameters(name="Water", t1=2.0, t2=2.0)

        result = self.simulator.simulate(
            sequence=(b1, grads, time),
            tissue=tissue,
            positions=positions,
            mode=0,  # Endpoint only
        )

        # 6. Update Plots
        self._update_plots(time, b1, grads, positions, result)

    def _update_plots(self, time, b1, grads, positions, result):
        self.plot_rf.clear()
        self.plot_profile.clear()

        # RF Plot
        t_ms = time * 1000.0
        self.plot_rf.plot(t_ms, np.abs(b1), pen="b", name="|B1| (G)")

        # Gradient Plot (Gz)
        if grads is not None and grads.shape[1] > 2:
            gz = grads[:, 2]
            # Scale Gz for visibility if needed, or plot on separate axis.
            # For now, just plot it directly. B1 is ~0.1G, Gz might be ~1G/cm.
            self.plot_rf.plot(t_ms, gz, pen="r", name="Gz (G/cm)")

        # Profile Plot
        pos_cm = positions[:, 2] * 100.0
        mz = result["mz"]
        mxy = np.sqrt(result["mx"] ** 2 + result["my"] ** 2)

        self.plot_profile.plot(pos_cm, mz, pen="g", name="Mz")
        self.plot_profile.plot(pos_cm, mxy, pen="y", name="|Mxy|")

        # Add slice boundaries indicators
        half_thick_cm = (self.thickness.value() / 2.0) / 10.0  # mm -> cm
        line_neg = pg.InfiniteLine(
            pos=-half_thick_cm, angle=90, pen=pg.mkPen("w", style=Qt.DashLine)
        )
        line_pos = pg.InfiniteLine(
            pos=half_thick_cm, angle=90, pen=pg.mkPen("w", style=Qt.DashLine)
        )
        self.plot_profile.addItem(line_neg)
        self.plot_profile.addItem(line_pos)
