from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QComboBox,
    QWidget,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
)
import numpy as np
import math
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from typing import Optional

from ..simulator import (
    PulseSequence,
    SpinEcho,
    SpinEchoTipAxis,
    GradientEcho,
    SliceSelectRephase,
    InversionRecovery,
    design_rf_pulse,
)


class SequenceDesigner(QGroupBox):
    """Widget for designing pulse sequences."""

    def __init__(self):
        super().__init__("Sequence Design")
        self.default_dt = 1e-5  # 10 us
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
        self.sequence_type.setObjectName("sequence_type_combo")
        self.sequence_type.addItems(
            [
                "Free Induction Decay",
                "Spin Echo",
                "Spin Echo (Tip-axis 180)",
                "Slice Select + Rephase",
                "SSFP (Loop)",
                "Inversion Recovery",
                "Custom",
            ]
        )
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
        self.spin_echo_echoes.setObjectName("spin_echo_echoes")
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
        self.ssfp_repeats.setObjectName("ssfp_repeats")
        self.ssfp_repeats.setRange(1, 10000)
        self.ssfp_repeats.setValue(16)
        self.ssfp_repeats.valueChanged.connect(lambda _: self.update_diagram())
        row1.addWidget(self.ssfp_repeats)
        ssfp_layout.addLayout(row1)

        row3 = QHBoxLayout()
        # Pulse duration is now taken from RF Pulse Designer
        # self.ssfp_dur = QDoubleSpinBox()... (removed)

        row3.addWidget(QLabel("Start delay (ms):"))
        self.ssfp_start_delay = QDoubleSpinBox()
        self.ssfp_start_delay.setObjectName("ssfp_start_delay")
        self.ssfp_start_delay.setRange(0.0, 10000.0)
        self.ssfp_start_delay.setDecimals(3)
        self.ssfp_start_delay.setValue(0.0)
        self.ssfp_start_delay.valueChanged.connect(lambda _: self.update_diagram())
        row3.addWidget(self.ssfp_start_delay)
        ssfp_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Start Flip (°):"))
        self.ssfp_start_flip = QDoubleSpinBox()
        self.ssfp_start_flip.setObjectName("ssfp_start_flip")
        self.ssfp_start_flip.setRange(0.0, 360.0)
        self.ssfp_start_flip.setDecimals(2)
        self.ssfp_start_flip.setValue(45.0)
        self.ssfp_start_flip.valueChanged.connect(lambda _: self.update_diagram())
        row4.addWidget(self.ssfp_start_flip)
        row4.addWidget(QLabel("Start phase (deg):"))
        self.ssfp_start_phase = QDoubleSpinBox()
        self.ssfp_start_phase.setObjectName("ssfp_start_phase")
        self.ssfp_start_phase.setRange(-3600, 3600)
        self.ssfp_start_phase.setDecimals(2)
        self.ssfp_start_phase.setValue(180.0)
        self.ssfp_start_phase.valueChanged.connect(lambda _: self.update_diagram())
        row4.addWidget(self.ssfp_start_phase)
        ssfp_layout.addLayout(row4)

        # Alternating phase option (common bSSFP scheme: 0/180/0/180 ...)
        self.ssfp_alternate_phase = QCheckBox("Alternate phase each TR (0/180°)")
        self.ssfp_alternate_phase.setObjectName("ssfp_alternate_phase")
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
        self.rephase_percentage.setObjectName("rephase_percentage")
        self.rephase_percentage.setRange(-200.0, 200.0)
        self.rephase_percentage.setValue(50.0)
        self.rephase_percentage.setSingleStep(1.0)
        self.rephase_percentage.setToolTip(
            "Percentage of slice select gradient area to rewind (50% = half area)"
        )
        self.rephase_percentage.valueChanged.connect(lambda _: self.update_diagram())
        sr_layout.addWidget(self.rephase_percentage)
        self.slice_rephase_opts.setLayout(sr_layout)
        self.options_container.addWidget(self.slice_rephase_opts)

        layout.addLayout(self.options_container)
        self.spin_echo_opts.setVisible(False)
        self.ssfp_opts.setVisible(False)
        self.slice_rephase_opts.setVisible(False)

        # TE parameter
        self.te_container = QWidget()
        te_layout = QHBoxLayout(self.te_container)
        te_layout.setContentsMargins(0, 0, 0, 0)
        te_layout.addWidget(QLabel("TE (ms):"))
        self.te_spin = QDoubleSpinBox()
        self.te_spin.setObjectName("te_spin")
        self.te_spin.setRange(0.1, 200)
        self.te_spin.setValue(20)
        te_layout.addWidget(self.te_spin)
        layout.addWidget(self.te_container)
        self.te_spin.valueChanged.connect(lambda _: self.update_diagram())

        # Slice thickness and Gradient overrides (grouped for easy hiding)
        self.gradient_opts_container = QWidget()
        grad_layout = QVBoxLayout()
        grad_layout.setContentsMargins(0, 0, 0, 0)

        thick_layout = QHBoxLayout()
        thick_layout.addWidget(QLabel("Slice thickness (mm):"))
        self.slice_thickness_spin = QDoubleSpinBox()
        self.slice_thickness_spin.setObjectName("slice_thickness_spin")
        self.slice_thickness_spin.setRange(0.05, 50.0)
        self.slice_thickness_spin.setValue(5.0)
        self.slice_thickness_spin.setDecimals(2)
        self.slice_thickness_spin.setSingleStep(0.1)
        self.slice_thickness_spin.valueChanged.connect(lambda _: self.update_diagram())
        thick_layout.addWidget(self.slice_thickness_spin)
        grad_layout.addLayout(thick_layout)

        # Manual slice gradient override
        g_layout = QHBoxLayout()
        g_layout.addWidget(QLabel("Slice G override (G/cm, 0=auto):"))
        self.slice_gradient_spin = QDoubleSpinBox()
        self.slice_gradient_spin.setObjectName("slice_gradient_spin")
        self.slice_gradient_spin.setRange(0.0, 99999.0)
        self.slice_gradient_spin.setDecimals(3)
        self.slice_gradient_spin.setSingleStep(0.1)
        self.slice_gradient_spin.setValue(0.0)
        self.slice_gradient_spin.valueChanged.connect(lambda _: self.update_diagram())
        g_layout.addWidget(self.slice_gradient_spin)
        grad_layout.addLayout(g_layout)

        self.gradient_opts_container.setLayout(grad_layout)
        layout.addWidget(self.gradient_opts_container)

        # TR parameter
        tr_layout = QHBoxLayout()
        tr_layout.addWidget(QLabel("TR (ms):"))
        self.tr_spin = QDoubleSpinBox()
        self.tr_spin.setObjectName("tr_spin")
        self.tr_spin.setRange(1, 10000)
        self.tr_spin.setValue(10)
        tr_layout.addWidget(self.tr_spin)
        self.tr_actual_label = QLabel("")
        self.tr_actual_label.setStyleSheet("color: #666; font-style: italic;")
        tr_layout.addWidget(self.tr_actual_label)
        layout.addLayout(tr_layout)
        self.tr_spin.valueChanged.connect(lambda _: self.update_diagram())

        # TI parameter (for IR)
        ti_layout = QHBoxLayout()
        ti_layout.addWidget(QLabel("TI (ms):"))
        self.ti_spin = QDoubleSpinBox()
        self.ti_spin.setObjectName("ti_spin")
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
        self.diagram_widget.setLabel("left", "")
        self.diagram_widget.setLabel("bottom", "Time", "ms")
        self.diagram_widget.setMinimumHeight(250)
        layout.addWidget(self.diagram_widget)
        self.diagram_arrows = []
        self.playhead_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("y", width=2))
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
        self.spin_echo_opts.setVisible(
            seq_type in ("Spin Echo", "Spin Echo (Tip-axis 180)")
        )
        self.ssfp_opts.setVisible(seq_type == "SSFP (Loop)")
        self.slice_rephase_opts.setVisible(seq_type == "Slice Select + Rephase")
        self.ti_widget.setVisible(seq_type == "Inversion Recovery")

        # Hide gradient options for SSFP as it's typically a 0D/1D simulation without slice gradients in this context
        if hasattr(self, "gradient_opts_container"):
            self.gradient_opts_container.setVisible(seq_type != "SSFP (Loop)")
            self.gradient_opts_container.setVisible(seq_type != "Inversion Recovery")
            self.gradient_opts_container.setVisible(seq_type != "Spin Echo")
            self.gradient_opts_container.setVisible(
                seq_type != "Spin Echo (Tip-axis 180)"
            )
            self.gradient_opts_container.setVisible(seq_type != "Free Induction Decay")

        # TE is irrelevant for continuous SSFP loop (which uses TR/Dur)
        if hasattr(self, "te_container"):
            self.te_container.setVisible(False)
            self.te_container.setVisible(
                seq_type in ("Spin Echo", "Spin Echo (Tip-axis 180)")
            )

        # Update pulse list based on sequence type
        self.pulse_list.blockSignals(True)
        self.pulse_list.clear()
        self.pulse_waveforms.clear()
        self.current_role = None

        roles = []
        if seq_type in ("Spin Echo", "Spin Echo (Tip-axis 180)"):
            # Generate Refocusing first, then Excitation.
            # This ensures rf_designer is left in Excitation state (90 deg)
            # which is safer if it's picked up as a default anywhere.
            generation_roles = ["Refocusing", "Excitation"]
            roles = ["Excitation", "Refocusing"]  # Display order

            if hasattr(self, "parent_gui") and hasattr(self.parent_gui, "rf_designer"):
                current_state = self.parent_gui.rf_designer.get_state()
                presets = self.get_sequence_preset_params(seq_type)
                default_duration = presets.get(
                    "duration", current_state.get("duration", 1.0)
                )

                # Excitation: 90 degrees, same type, correct duration, reset B1 override, reset phase
                exc_state = current_state.copy()
                exc_state["flip_angle"] = 90.0
                exc_state["duration"] = default_duration
                exc_state["b1_amplitude"] = 0.0
                exc_state["phase"] = 0.0  # Explicitly reset phase
                self.pulse_states["Excitation"] = exc_state

                # Refocusing: 180 degrees, same type, correct duration, reset B1 override
                ref_state = current_state.copy()
                if seq_type == "Spin Echo (Tip-axis 180)":
                    ref_state["flip_angle"] = 180.0
                    ref_state["phase"] = 90.0
                else:
                    ref_state["flip_angle"] = 180.0
                    ref_state["phase"] = 0.0
                ref_state["duration"] = default_duration
                ref_state["b1_amplitude"] = 0.0
                self.pulse_states["Refocusing"] = ref_state

                # Pre-generate waveforms
                for role in generation_roles:
                    self.current_role = role
                    self.parent_gui.rf_designer.set_state(self.pulse_states[role])

        elif seq_type == "Inversion Recovery":
            generation_roles = ["Inversion", "Excitation"]
            roles = ["Inversion", "Excitation"]

            # Pre-populate states to ensure they are both Sinc (or match current designer type)
            if hasattr(self, "parent_gui") and hasattr(self.parent_gui, "rf_designer"):
                current_state = self.parent_gui.rf_designer.get_state()
                presets = self.get_sequence_preset_params(seq_type)
                default_duration = presets.get(
                    "duration", current_state.get("duration", 1.0)
                )

                # Inversion: 180 degrees, reset B1 override, reset phase
                inv_state = current_state.copy()
                inv_state["flip_angle"] = 180.0
                inv_state["duration"] = default_duration
                inv_state["b1_amplitude"] = 0.0
                inv_state["phase"] = 0.0  # Explicitly reset phase
                # Ensure it's a Sinc if the user hasn't explicitly set a type for this role yet
                # Or just force it to match the current designer type (which is usually what users expect)
                self.pulse_states["Inversion"] = inv_state

                # Excitation: 90 degrees, reset B1 override, reset phase
                exc_state = current_state.copy()
                exc_state["flip_angle"] = 90.0
                exc_state["duration"] = default_duration
                exc_state["b1_amplitude"] = 0.0
                exc_state["phase"] = 0.0  # Explicitly reset phase
                self.pulse_states["Excitation"] = exc_state

                # Pre-generate waveforms
                for role in generation_roles:
                    self.current_role = role
                    self.parent_gui.rf_designer.set_state(self.pulse_states[role])

        elif seq_type in ("Gradient Echo", "Free Induction Decay", "FLASH", "EPI"):
            roles = ["Excitation"]
            if hasattr(self, "parent_gui") and hasattr(self.parent_gui, "rf_designer"):
                current_state = self.parent_gui.rf_designer.get_state()
                exc_state = current_state.copy()
                exc_state["phase"] = 0.0  # Reset phase for standard FID/GRE
                self.pulse_states["Excitation"] = exc_state
                self.parent_gui.rf_designer.set_state(exc_state)
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
                "b1_amplitude": 0.0,
            },
            "Spin Echo": {
                "te_ms": 10,
                "tr_ms": 20,
                "num_positions": 1,
                "num_frequencies": 201,
                "frequency_range_hz": 100,
                "duration": 1.0,  # ms
                "b1_amplitude": 0.0,
            },
            "Spin Echo (Tip-axis 180)": {
                "te_ms": 10,
                "tr_ms": 20,
                "num_positions": 1,
                "num_frequencies": 201,
                "frequency_range_hz": 100,
                "duration": 1.0,  # ms
                "b1_amplitude": 0.0,
            },
            "Gradient Echo": {
                "te_ms": 5,
                "tr_ms": 20,
                "flip_angle": 30,
                "duration": 1.0,  # ms
                "b1_amplitude": 0.0,
            },
            "Slice Select + Rephase": {
                "te_ms": 5,
                "tr_ms": 20,
                "num_positions": 99,
                "num_frequencies": 3,
                "duration": 1.0,  # ms
                "flip_angle": 90,
                "b1_amplitude": 0.0,
            },
            "SSFP (Loop)": {
                "te_ms": 2,
                "tr_ms": 5,
                "flip_angle": 30,
                "ssfp_repeats": 100,
                "ssfp_dur": 1.0,
                "ssfp_start_delay": 0.0,
                "ssfp_start_flip": 15.0,
                "ssfp_start_phase": 0.0,
                "ssfp_alternate_phase": True,
                "pulse_type": "gaussian",
                "num_positions": 1,
                "num_frequencies": 101,
                "frequency_range_hz": 1000,
                "duration": 1.0,  # ms
                "time_step": 10.0,
                "b1_amplitude": 0.0,
            },
            "Inversion Recovery": {
                "te_ms": 10,
                "tr_ms": 100,
                "ti_ms": 50,
                "num_positions": 1,
                "num_frequencies": 51,
                "duration": 1.0,  # ms
                "b1_amplitude": 0.0,
            },
            "FLASH": {
                "te_ms": 3,
                "tr_ms": 10,
                "flip_angle": 15,
                "duration": 1.0,  # ms
                "b1_amplitude": 0.0,
            },
            "EPI": {
                "te_ms": 25,
                "tr_ms": 100,
                "num_positions": 51,
                "num_frequencies": 3,
                "duration": 1.0,  # ms
                "b1_amplitude": 0.0,
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
        """
        seq_type = self.sequence_type.currentText()
        te = self.te_spin.value() / 1000  # Convert to seconds
        tr = self.tr_spin.value() / 1000
        ti = self.ti_spin.value() / 1000

        # Use explicit B1/gradient arrays when we can so RF designer changes take effect
        if (
            seq_type == "Free Induction Decay" or seq_type == "Custom"
        ) and custom_pulse is not None:
            b1, time = custom_pulse
            b1 = np.asarray(b1, dtype=complex)
            time = np.asarray(time, dtype=float)
            if b1.shape[0] != time.shape[0]:
                raise ValueError(
                    "Custom pulse B1 and time arrays must have the same length."
                )

            # Extend to full TR
            dt = time[1] - time[0] if len(time) > 1 else self.default_dt
            current_dur = time[-1] if len(time) > 0 else 0
            target_dur = max(tr, current_dur)

            if target_dur > current_dur + dt / 2:
                n_extra = int(np.ceil((target_dur - current_dur) / dt))
                # Clamp to avoid huge allocations if TR is very large relative to dt
                n_extra = min(n_extra, 1000000)
                if n_extra > 0:
                    b1 = np.pad(b1, (0, n_extra), "constant")
                    extra_time = current_dur + np.arange(1, n_extra + 1) * dt
                    time = np.concatenate([time, extra_time])

            gradients = np.zeros((len(time), 3))
            return (b1, gradients, time)

        if seq_type == "Spin Echo":
            # Get RF frequency offset from RF designer
            rf_freq_offset = (
                self.parent_gui.rf_designer.freq_offset.value()
                if hasattr(self, "parent_gui")
                and hasattr(self.parent_gui, "rf_designer")
                else 0.0
            )

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
            # Get RF frequency offset from RF designer
            rf_freq_offset = (
                self.parent_gui.rf_designer.freq_offset.value()
                if hasattr(self, "parent_gui")
                and hasattr(self.parent_gui, "rf_designer")
                else 0.0
            )
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
                rf_freq_offset=rf_freq_offset,
            )
        elif seq_type == "Gradient Echo":
            # Get RF frequency offset from RF designer
            rf_freq_offset = (
                self.parent_gui.rf_designer.freq_offset.value()
                if hasattr(self, "parent_gui")
                and hasattr(self.parent_gui, "rf_designer")
                else 0.0
            )
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
            total_duration = max(te, tr, 0.01)  # cover at least 10 ms, TE or TR
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
                wave_duration = float(
                    t_wave[-1] - t_wave[0] + (t_wave[1] - t_wave[0])
                    if len(t_wave) > 1
                    else dt
                )
                n_wave = max(1, int(np.ceil(wave_duration / dt)))
                t_resample = np.arange(0, n_wave) * dt
                real_part = np.interp(t_resample, t_wave - t_wave[0], np.real(b1_wave))
                imag_part = np.interp(t_resample, t_wave - t_wave[0], np.imag(b1_wave))
                b1_exc = real_part + 1j * imag_part
            else:
                exc_duration = 1e-3
                n_exc = max(int(np.ceil(exc_duration / dt)), 16)
                flip = (
                    self.parent_gui.rf_designer.flip_angle.value()
                    if hasattr(self, "parent_gui") and self.parent_gui is not None
                    else 90.0
                )
                b1_exc, _ = design_rf_pulse(
                    "rect", duration=n_exc * dt, flip_angle=flip, npoints=n_exc
                )

            ntime = max(len(b1_exc), int(np.ceil(total_duration / dt)))
            ntime = min(max(ntime, 1000), 20000)  # keep reasonable bounds
            b1 = np.zeros(ntime, dtype=complex)
            gradients = np.zeros((ntime, 3))
            b1[: min(len(b1_exc), ntime)] = b1_exc[: min(len(b1_exc), ntime)]
            time = np.arange(ntime) * dt
            return (b1, gradients, time)

    def compile_sequence(
        self, custom_pulse=None, dt: float = None, log_info: bool = False
    ):
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
            if self._slice_gradient_override() is None and seq_type in (
                "Spin Echo",
                "Spin Echo (Tip-axis 180)",
                "Gradient Echo",
            ):
                scale = self._effective_tbw() / 4.0
                gradients = np.array(gradients, copy=True)
                gradients[:, 2] *= scale
            return b1, gradients, time
        b1, gradients, time = seq
        return (
            np.asarray(b1, dtype=complex),
            np.asarray(gradients, dtype=float),
            np.asarray(time, dtype=float),
        )

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
            exc_b1, _ = design_rf_pulse(
                "sinc", duration=n_exc * dt, flip_angle=90, npoints=n_exc
            )
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
        pre_time = (
            exc_pts + slice_gap_pts + rephase_pts + prephase_pts + settle_pts
        ) * dt
        train_start_time = max(pre_time, te - mid_echo_time)
        train_start_pts = int(np.round(train_start_time / dt))

        # Use a start index that respects prephasing blocks
        actual_train_start = max(
            train_start_pts,
            exc_pts + slice_gap_pts + rephase_pts + prephase_pts + settle_pts,
        )
        train_end_pts = (
            actual_train_start + n_lines * ro_pts + (n_lines - 1) * (blip_pts + gap_pts)
        )
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
        slice_g = self._slice_gradient_override() or (
            bw_hz / (gamma_hz_per_g * thickness_cm)
        )  # G/cm
        gradients[:n_exc, 2] = slice_g

        # Slice rephasing (half area rewind)
        rephase_start = n_exc + slice_gap_pts
        if rephase_start < npoints:
            area_exc = slice_g * n_exc * dt
            rephase_amp = -(0.5 * area_exc) / (rephase_pts * dt)
            gradients[rephase_start : rephase_start + rephase_pts, 2] = rephase_amp

        # Readout prephaser to move to -kmax
        prephase_start = rephase_start + rephase_pts
        read_amp = 8e-3  # readout gradient amplitude
        if prephase_start < npoints:
            prephase_amp = -0.5 * read_amp * (ro_pts / max(prephase_pts, 1))
            gradients[prephase_start : prephase_start + prephase_pts, 0] = prephase_amp

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

        # Determine which pulse is which
        # If the user is designing a pulse, 'custom_pulse' is that live pulse.
        # We need to assign it to the correct role (Inversion or Excitation)
        # and retrieve the OTHER pulse from the stored waveforms.

        current_role = getattr(self, "current_role", None)
        inv_pulse = None
        exc_pulse = None

        if current_role == "Inversion":
            inv_pulse = custom_pulse
            exc_pulse = self.pulse_waveforms.get("Excitation")
        elif current_role == "Excitation":
            exc_pulse = custom_pulse
            inv_pulse = self.pulse_waveforms.get("Inversion")
        else:
            # Fallback if roles are somehow ambiguous
            inv_pulse = self.pulse_waveforms.get("Inversion")
            exc_pulse = self.pulse_waveforms.get("Excitation")

        seq = InversionRecovery(
            ti=ti,
            tr=tr,
            te=te,
            pulse_type="sinc",
            slice_thickness=self._slice_thickness_m(),
            slice_gradient_override=self._slice_gradient_override(),
            custom_inversion=inv_pulse,
            custom_excitation=exc_pulse,
        )

        return seq.compile(dt)

    def _build_ssfp(self, custom_pulse, dt):
        """
        Build a simple balanced-SSFP-style pulse train: identical RF pulses every TR,
        with an optional distinct first pulse (amplitude/phase/delay).
        """
        dt = max(dt, 1e-6)
        tr = self.tr_spin.value() / 1000.0
        n_reps = max(1, self.ssfp_repeats.value())
        # Use duration and flip from RF designer (shared state)
        if hasattr(self, "parent_gui") and self.parent_gui:
            main_flip = self.parent_gui.rf_designer.flip_angle.value()
            pulse_dur = self.parent_gui.rf_designer.duration.value() / 1000.0
        else:
            main_flip = 30.0
            pulse_dur = 1e-3

        start_flip = self.ssfp_start_flip.value()
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
                raise ValueError(
                    "Custom pulse B1 and time arrays must have the same length."
                )
            if len(t_wave) > 1:
                wave_dt = np.median(np.diff(t_wave))
                wave_duration = len(t_wave) * wave_dt
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
            base_peak = (
                float(np.max(np.abs(custom_b1))) if np.any(np.abs(custom_b1)) else 1.0
            )

        def _place_pulse(start_s, amp, phase):
            start_idx = int(np.round(start_s / dt))
            n_dur = max(1, int(np.round(pulse_dur / dt)))
            end_idx = min(start_idx + n_dur, npoints)
            if custom_b1 is not None:
                seg = custom_b1
                seg_len = min(end_idx - start_idx, len(seg))
                # Scale/rotate custom waveform by amp/phase controls
                scale = amp / base_peak if base_peak else 1.0
                # Note: custom_b1 already has the phase from the designer baked in.
                # However, for SSFP we might want to rotate it further if alternating phase is used.
                b1[start_idx : start_idx + seg_len] = (
                    seg[:seg_len] * scale * np.exp(1j * phase)
                )
            else:
                b1[start_idx:end_idx] = amp * np.exp(1j * phase)

        # Optional distinct first pulse
        # Calculate start amplitude from start flip angle
        start_scale = start_flip / main_flip if main_flip > 0 else 0.5
        start_amp = base_peak * start_scale if base_peak is not None else 0.025
        _place_pulse(start_delay, start_amp, start_phase)

        # Remaining pulses evenly spaced by TR
        for k in range(1, n_reps):
            t0 = start_delay + k * tr
            # Main pulses use base_peak (the flip angle from designer)
            # and additional phase rotation for alternating bSSFP if needed.
            extra_phase = math.pi if (k % 2 == 1 and alternate) else 0.0
            _place_pulse(t0, base_peak if base_peak is not None else 0.05, extra_phase)

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
            exc_b1, _ = design_rf_pulse(
                "sinc", duration=n_exc * dt, flip_angle=90, npoints=n_exc
            )

        exc_pts = len(exc_b1)
        exc_duration = exc_pts * dt

        # Slice gradient
        thickness_cm = self._slice_thickness_m() * 100.0
        gamma_hz_per_g = 4258.0
        tbw = self._effective_tbw()
        bw_hz = tbw / max(exc_duration, dt)
        slice_g = self._slice_gradient_override() or (
            bw_hz / (gamma_hz_per_g * thickness_cm)
        )

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
        if not hasattr(self, "parent_gui") or not hasattr(
            self.parent_gui, "rf_designer"
        ):
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
                state = self.pulse_states[curr_role]
                self.parent_gui.rf_designer.set_state(state)
                # Also update the panel view if it exists
                if hasattr(self.parent_gui, "rf_designer_panel"):
                    self.parent_gui.rf_designer_panel.set_state(state)
            else:
                # Apply defaults for new roles
                defaults = {}
                if curr_role in ("Refocusing", "Inversion"):
                    defaults = {
                        "flip_angle": 180.0,
                        "duration": 2.0,
                        "b1_amplitude": 0.0,
                        "phase": 0.0,  # Explicit default phase
                    }
                elif curr_role == "Excitation":
                    defaults = {
                        "flip_angle": 90.0,
                        "duration": 1.0,
                        "b1_amplitude": 0.0,
                        "phase": 0.0,  # Explicit default phase
                    }

                if defaults:
                    self.parent_gui.rf_designer.set_state(defaults)
                    # Initialize the state in our local cache so it's not lost
                    self.pulse_states[curr_role] = (
                        self.parent_gui.rf_designer.get_state()
                    )

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
                # Sync Start Flip to half of the designer's flip angle
                main_flip = self.parent_gui.rf_designer.flip_angle.value()
                self.ssfp_start_flip.setValue(main_flip / 2.0)
            if t_wave is not None and len(t_wave) > 1:
                # Calculate dt from the first two points (assuming uniform spacing)
                dt = float(t_wave[1] - t_wave[0])
                # Duration is span + 1 dt (because samples are 0..N-1)
                duration_s = float(t_wave[-1] - t_wave[0]) + dt
                # self.ssfp_dur.setValue(...) # Removed as redundant
        self.update_diagram()

    def update_diagram(self, custom_pulse=None):
        """Render the sequence diagram so users can see the selected waveform."""
        custom = custom_pulse if custom_pulse is not None else self.custom_pulse
        try:
            b1, gradients, time = self.compile_sequence(
                custom_pulse=custom, dt=self.default_dt
            )
        except ValueError as e:
            # Handle validation errors (e.g. TE too short)
            self.diagram_widget.clear()
            self.diagram_widget.setLabel("bottom", "")
            self.diagram_widget.setTitle(f"Invalid Sequence: {str(e)}")
            text = pg.TextItem(text=f"Error:\n{str(e)}", color="r", anchor=(0.5, 0.5))
            # Put text roughly in center
            self.diagram_widget.addItem(text)
            text.setPos(0.5, 0.5)
            # Need to set arbitrary range to show text
            self.diagram_widget.setXRange(0, 1)
            self.diagram_widget.setYRange(0, 1)
            return
        except Exception as e:
            self.diagram_widget.clear()
            self.diagram_widget.setTitle(f"Error: {str(e)}")
            return

        self.diagram_widget.setTitle(None)
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
            ("RF Mag", 4.0, "b"),
            ("RF Phase", 3.0, "c"),
            ("Gz (slice/readout)", 2.0, "m"),
            ("Gy (phase1)", 1.0, "g"),
            ("Gx (phase2)", 0.0, "r"),
        ]

        # Draw horizontal grid lines for clarity
        for _, y, _ in lanes:
            line = pg.InfiniteLine(
                pos=y, angle=0, pen=pg.mkPen((180, 180, 180, 120), width=1)
            )
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
        self.diagram_widget.plot(
            time_ms,
            rf_mag_y + b1_mag * rf_scale,
            pen=pg.mkPen("b", width=2),
            name="RF Mag",
        )

        # RF Phase lane (convert radians to normalized display: -π to π → -0.8 to 0.8)
        rf_phase_y = lanes[1][1]
        # Only plot phase where there's significant RF (avoid noise)
        phase_mask = (
            b1_mag > (b1_mag.max() * 0.01)
            if b1_mag.max() > 0
            else np.zeros_like(b1_mag, dtype=bool)
        )
        if np.any(phase_mask):
            phase_display = b1_phase / np.pi * 0.8  # Normalize to ±0.8 for display
            # Create connected segments only where RF is active
            self.diagram_widget.plot(
                time_ms,
                rf_phase_y + phase_display,
                pen=pg.mkPen("c", width=2),
                name="RF Phase",
                connect="finite",
            )

            # Add phase reference markers at -π, 0, +π
            phase_ref_pen = pg.mkPen((150, 150, 150, 100), width=1, style=Qt.DashLine)
            for phase_val, label_text in [(-np.pi, "-π"), (0, "0"), (np.pi, "+π")]:
                y_pos = rf_phase_y + (phase_val / np.pi * 0.8)
                ref_line = pg.InfiniteLine(pos=y_pos, angle=0, pen=phase_ref_pen)
                self.diagram_widget.addItem(ref_line)
                # Add small label at the right edge
                if len(time_ms) > 0:
                    phase_label = pg.TextItem(
                        text=label_text, color=(150, 150, 150), anchor=(1, 0.5), angle=0
                    )
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
                name=label,
            )
            nonzero = np.where(np.abs(g) > 0)[0]
            if nonzero.size:
                mid = nonzero[nonzero.size // 2]
                x = time_ms[mid]
                angle = 90 if g[mid] > 0 else -90
                arr = pg.ArrowItem(
                    pos=(x, y + g[mid] * scale), angle=angle, brush=color
                )
                self.diagram_widget.addItem(arr)
                self.diagram_arrows.append(arr)

        # TE / TR markers
        te_ms = self.te_spin.value()
        tr_ms = self.tr_spin.value()
        actual_tr_ms = time_ms[-1] if len(time_ms) > 0 else 0

        # Remove actual TR label (user request)
        self.tr_actual_label.setText("")

        if actual_tr_ms > 0:
            tr_line = pg.InfiniteLine(
                pos=actual_tr_ms,
                angle=90,
                pen=pg.mkPen((120, 120, 120), style=Qt.DashLine),
            )
            self.diagram_widget.addItem(tr_line)
            # Removed "Actual TR" text item

        if te_ms > 0 and te_ms <= actual_tr_ms:
            # Calculate actual echo position based on sequence type
            seq_type = self.sequence_type.currentText()
            echo_pos_ms = te_ms

            if seq_type in ("Spin Echo", "Spin Echo (Tip-axis 180)", "Gradient Echo"):
                # Estimate excitation duration
                exc = self.pulse_waveforms.get("Excitation", self.custom_pulse)
                if exc is not None and len(exc[1]) > 1:
                    exc_dur_ms = (exc[1][-1] - exc[1][0]) * 1000.0
                else:
                    exc_dur_ms = 1.0  # default 1ms

                # Echo happens at exc_duration/2 + TE
                echo_pos_ms = exc_dur_ms / 2.0 + te_ms

            if echo_pos_ms <= actual_tr_ms:
                te_line = pg.InfiniteLine(
                    pos=echo_pos_ms,
                    angle=90,
                    pen=pg.mkPen((200, 150, 0), style=Qt.DotLine, width=2),
                )
                self.diagram_widget.addItem(te_line)
                te_lbl = pg.TextItem(
                    text=f"TE={te_ms:.1f}ms", color=(200, 150, 0), anchor=(0, 1)
                )
                te_lbl.setPos(echo_pos_ms, 4.8)
                self.diagram_widget.addItem(te_lbl)
                self.diagram_labels.append(te_lbl)

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
