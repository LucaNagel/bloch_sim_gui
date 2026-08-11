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
    RF_PULSE_TYPE_OPTIONS,
    TissueParameters,
    SliceSelectRephase,
    apply_rf_carrier,
    design_rf_pulse,
)


class SliceSelectionExplorer(QWidget):
    """
    Widget for exploring slice selection profiles.
    Allows user to configure RF pulse and gradient parameters and visualizes
    the resulting magnetization profile across the slice.
    """

    def __init__(self, parent=None, rf_designer=None):
        super().__init__(parent)
        self.simulator = BlochSimulator()
        self.rf_designer = rf_designer
        self.last_b1 = None
        self.last_gradients = None
        self.last_time = None
        self.init_ui()
        self._connect_rf_designer()
        self._update_pulse_controls()
        # Trigger initial simulation
        self.run_simulation()

    def init_ui(self):
        layout = QVBoxLayout()
        title = QLabel("Slice Explorer")
        title_font = title.font()
        title_font.setBold(True)
        title_font.setPointSize(max(title_font.pointSize() + 2, 12))
        title.setFont(title_font)
        title.setVisible(False)
        self.page_title = title
        layout.addWidget(title)

        content_layout = QHBoxLayout()
        layout.addLayout(content_layout, 1)

        # Left Panel: Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setMinimumWidth(400)
        control_panel.setMaximumWidth(400)
        self.control_panel = control_panel

        # Pulse Parameters Group
        pulse_group = QGroupBox("Pulse Parameters")
        pulse_layout = QVBoxLayout()

        # Pulse source/type
        row_type = QHBoxLayout()
        row_type.addWidget(QLabel("Pulse:"))
        self.pulse_source = QComboBox()
        self.pulse_source.setObjectName("slice_explorer_pulse_source")
        self.pulse_source.addItems(["Use RF Design", *RF_PULSE_TYPE_OPTIONS])
        self.pulse_source.setToolTip(
            "Use the current waveform from RF Design, or generate the selected "
            "pulse shape from the Slice Explorer parameters."
        )
        row_type.addWidget(self.pulse_source)
        pulse_layout.addLayout(row_type)
        self.pulse_status = QLabel()
        self.pulse_status.setWordWrap(True)
        self.pulse_status.setStyleSheet("color: gray;")
        pulse_layout.addWidget(self.pulse_status)

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
        row_range.addWidget(QLabel("Range (mm):"))
        self.pos_range = QDoubleSpinBox()
        self.pos_range.setRange(5.0, 200.0)
        self.pos_range.setValue(40.0)
        self.pos_range.setSuffix(" mm")
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

        for field in (
            self.pulse_source,
            self.flip_angle,
            self.duration,
            self.tbw,
            self.apodization,
            self.thickness,
            self.use_rephase,
            self.pos_range,
            self.num_points,
        ):
            field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

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
        self.plot_profile.setLabel("bottom", "Position (mm)")
        self.plot_profile.setYRange(-1.1, 1.1)
        self.plot_profile.addLegend()
        viz_panel.addWidget(self.plot_profile)

        content_layout.addWidget(control_panel)
        content_layout.addWidget(viz_panel, 1)
        self.setLayout(layout)

        # Connect changes to auto-update (optional, maybe just button is safer for performance)
        # For now, let's auto-update on changes for responsiveness, unless it's too slow
        self.pulse_source.currentTextChanged.connect(self._pulse_source_changed)
        self.flip_angle.valueChanged.connect(self.run_simulation)
        self.duration.valueChanged.connect(self.run_simulation)
        self.tbw.valueChanged.connect(self.run_simulation)
        self.apodization.currentTextChanged.connect(self.run_simulation)
        self.thickness.valueChanged.connect(self.run_simulation)
        self.use_rephase.currentIndexChanged.connect(self.run_simulation)
        self.pos_range.valueChanged.connect(self.run_simulation)
        # self.num_points.valueChanged.connect(self.run_simulation) # Don't auto-update on points change while typing

    def _connect_rf_designer(self):
        """Use the main-window RF Designer when one is available."""
        if self.rf_designer is None:
            blocked = self.pulse_source.blockSignals(True)
            self.pulse_source.setCurrentText("Sinc")
            self.pulse_source.blockSignals(blocked)
            return
        self.rf_designer.pulse_changed.connect(self._rf_designer_pulse_changed)

    def _rf_designer_pulse_changed(self, *_):
        """Refresh an active Slice Explorer when its shared waveform changes."""
        if self.pulse_source.currentText() == "Use RF Design" and self.isVisible():
            self.run_simulation()

    def _pulse_source_changed(self, *_):
        self._update_pulse_controls()
        self.run_simulation()

    def _update_pulse_controls(self):
        """Enable local pulse controls only for locally generated waveforms."""
        use_rf_design = self.pulse_source.currentText() == "Use RF Design"
        for control in (
            self.flip_angle,
            self.duration,
            self.tbw,
            self.apodization,
        ):
            control.setEnabled(not use_rf_design)

    @staticmethod
    def _validate_pulse(pulse):
        if pulse is None or len(pulse) != 2:
            raise ValueError("Design a valid RF pulse in the RF Design tab first.")
        b1, time_rf = pulse
        b1 = np.asarray(b1, dtype=np.complex128).reshape(-1)
        time_rf = np.asarray(time_rf, dtype=float).reshape(-1)
        if (
            b1.size == 0
            or b1.size != time_rf.size
            or not np.all(np.isfinite(b1))
            or not np.all(np.isfinite(time_rf))
        ):
            raise ValueError("RF waveform and time axis must be finite and aligned.")
        time_rf = time_rf - time_rf[0]
        if time_rf.size > 1 and np.any(np.diff(time_rf) <= 0):
            raise ValueError("RF waveform time points must be strictly increasing.")
        return b1, time_rf

    @staticmethod
    def _apply_window_and_flip_scaling(b1, time_rf, window_type, flip_angle):
        """Apply optional apodization and scale a waveform to a target flip."""
        b1 = np.asarray(b1, dtype=np.complex128)
        if window_type != "None" and b1.size > 1:
            windows = {
                "Hamming": np.hamming,
                "Hanning": np.hanning,
                "Blackman": np.blackman,
            }
            window = windows.get(window_type, lambda count: np.ones(count))(b1.size)
            b1 = b1 * window

        area = np.trapezoid(b1, x=time_rf)
        if not np.isfinite(area) or abs(area) < 1e-12:
            raise ValueError("RF waveform integral is too small for flip scaling.")
        target_area = np.deg2rad(flip_angle) / (4258.0 * 2.0 * np.pi)
        return b1 * (target_area / area)

    def _custom_pulse_from_rf_design(self, duration_s, flip_angle, window_type):
        """Build a local Custom pulse from the waveform loaded in RF Design."""
        designer = self.rf_designer
        loaded_b1 = getattr(designer, "loaded_pulse_b1", None)
        loaded_time = getattr(designer, "loaded_pulse_time", None)
        if designer is None or loaded_b1 is None or loaded_time is None:
            raise ValueError(
                "Load a Custom waveform in RF Design before selecting Custom here."
            )
        b1, time_rf = self._validate_pulse((loaded_b1, loaded_time))
        source_duration = float(time_rf[-1]) if time_rf.size > 1 else 0.0
        if source_duration <= 0:
            raise ValueError("The loaded Custom waveform has no usable duration.")
        time_rf = time_rf * (duration_s / source_duration)
        b1 = self._apply_window_and_flip_scaling(b1, time_rf, window_type, flip_angle)
        return b1, time_rf

    def _resolve_pulse(self):
        """Return waveform and slice-gradient parameters for the selected source."""
        source = self.pulse_source.currentText()
        if source == "Use RF Design":
            if self.rf_designer is None:
                raise ValueError("RF Design is unavailable in this window.")
            b1, time_rf = self._validate_pulse(self.rf_designer.get_pulse())
            state = self.rf_designer.get_state()
            duration_s = float(state.get("duration", 0.0)) / 1000.0
            if duration_s <= 0:
                duration_s = (
                    float(np.median(np.diff(time_rf))) * time_rf.size
                    if time_rf.size > 1
                    else 1e-5
                )
            tbw = max(float(self.rf_designer.tbw.value()), 1e-6)
            flip_angle = float(state.get("flip_angle", 90.0))
            frequency_offset_hz = float(state.get("freq_offset", 0.0))
            b1 = apply_rf_carrier(b1, time_rf, frequency_offset_hz)
            dt = float(np.median(np.diff(time_rf))) if time_rf.size > 1 else duration_s
            label = f"RF Design ({state.get('pulse_type', 'Custom')})"
            return b1, time_rf, flip_angle, duration_s, tbw, dt, label

        flip_angle = self.flip_angle.value()
        duration_s = self.duration.value() / 1000.0
        tbw = self.tbw.value()
        window_type = self.apodization.currentText()
        dt = 1e-5

        if source == "Custom":
            b1, time_rf = self._custom_pulse_from_rf_design(
                duration_s, flip_angle, window_type
            )
            if time_rf.size > 1:
                dt = float(np.median(np.diff(time_rf)))
        else:
            pulse_types = {
                "Rectangle": "rect",
                "Sinc": "sinc",
                "Gaussian": "gaussian",
                "Hermite": "hermite",
                "Adiabatic Half Passage": "adiabatic_half",
                "Adiabatic Full Passage": "adiabatic_full",
                "BIR-4": "bir4",
            }
            pulse_type = pulse_types[source]
            n_rf_pts = max(8, int(np.ceil(duration_s / dt)))
            if pulse_type == "bir4":
                n_rf_pts = int(np.ceil(n_rf_pts / 4.0) * 4)
            b1, time_rf = design_rf_pulse(
                pulse_type=pulse_type,
                duration=duration_s,
                flip_angle=flip_angle,
                time_bw_product=tbw,
                npoints=n_rf_pts,
            )
            if window_type != "None":
                b1 = self._apply_window_and_flip_scaling(
                    b1, time_rf, window_type, flip_angle
                )

        return b1, time_rf, flip_angle, duration_s, tbw, dt, source

    def run_simulation(self):
        """Build sequence and run Bloch simulation."""

        try:
            (
                b1_base,
                time_rf,
                flip,
                dur_s,
                tbw,
                dt,
                pulse_label,
            ) = self._resolve_pulse()
        except (KeyError, TypeError, ValueError) as exc:
            self.last_b1 = None
            self.last_gradients = None
            self.last_time = None
            self.plot_rf.clear()
            self.plot_profile.clear()
            self.pulse_status.setText(str(exc))
            return

        self.pulse_status.setText(f"Using {pulse_label}")
        thick_m = self.thickness.value() / 1000.0
        do_rephase = self.use_rephase.currentIndex() == 0

        range_mm = self.pos_range.value()
        n_points = self.num_points.value()

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
        half_range = range_mm / 2.0
        positions = np.zeros((n_points, 3))
        # Z-axis varies (slice direction)
        positions[:, 2] = np.linspace(
            -half_range / 1000.0, half_range / 1000.0, n_points
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
        self.last_b1 = np.asarray(b1).copy()
        self.last_gradients = np.asarray(grads).copy()
        self.last_time = np.asarray(time).copy()
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
        pos_mm = positions[:, 2] * 1000.0
        mz = np.squeeze(result["mz"])
        mx = np.squeeze(result["mx"])
        my = np.squeeze(result["my"])
        mxy = np.sqrt(mx**2 + my**2)

        self.plot_profile.plot(pos_mm, mz.flatten(), pen="g", name="Mz")
        self.plot_profile.plot(pos_mm, mxy.flatten(), pen="y", name="|Mxy|")

        # Add slice boundaries indicators
        half_thick_mm = self.thickness.value() / 2.0
        line_neg = pg.InfiniteLine(
            pos=-half_thick_mm, angle=90, pen=pg.mkPen("w", style=Qt.DashLine)
        )
        line_pos = pg.InfiniteLine(
            pos=half_thick_mm, angle=90, pen=pg.mkPen("w", style=Qt.DashLine)
        )
        self.plot_profile.addItem(line_neg)
        self.plot_profile.addItem(line_pos)
