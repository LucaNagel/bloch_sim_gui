"""Live mode of the 3D Vector view: play pulses and gradients by hand.

The panel drives a :class:`~blochsimulator.live_bloch.LiveSpinSystem` from a
timer. Pulses are rectangular pulses of the chosen length: they play in
simulation time, slowed down on screen so that no pulse turns faster than
``PULSE_DEG_PER_S``. In between, the spins precess in simulation time scaled
by the speed setting. The RF and gradient history is kept for the field
traces below the signal trace.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..live_bloch import LiveSpinSystem
from ..position_sampling import AXIS_NAMES

PULSE_PHASES_DEG = {"x": 0.0, "y": 90.0, "-x": 180.0, "-y": 270.0}
# Seconds on screen the live signal trace scrolls through.
LIVE_TRACE_WINDOW_S = 10.0
# Field changes kept for the RF and gradient traces.
FIELD_HISTORY_LENGTH = 4000
LIVE_BUTTON_STYLE = (
    "QPushButton#mag_3d_live_mode {"
    " padding: 3px 12px; border: 1px solid palette(mid); border-radius: 5px;"
    " background-color: palette(button); color: palette(button-text); }"
    "QPushButton#mag_3d_live_mode:checked {"
    " background-color: #a5e3a1; border-color: #3f9b3a; color: #10330e;"
    " font-weight: bold; }"
)


@dataclass
class LiveContext:
    """Spins and tissue the live mode simulates."""

    positions_m: np.ndarray
    offsets_hz: np.ndarray
    gamma_hz_per_t: float
    t1_s: float = np.inf
    t2_s: float = np.inf

    def same_spins(self, other) -> bool:
        """Whether ``other`` has the same spins (relaxation may differ)."""
        return (
            other is not None
            and self.positions_m.shape == other.positions_m.shape
            and np.allclose(self.positions_m, other.positions_m)
            and self.offsets_hz.shape == other.offsets_hz.shape
            and np.allclose(self.offsets_hz, other.offsets_hz)
            and self.gamma_hz_per_t == other.gamma_hz_per_t
        )


class LiveModePanel(QWidget):
    """Controls for live pulses and gradients in the 3D Vector view.

    Parameters
    ----------
    context_provider : callable
        Returns the current :class:`LiveContext`; called on every tick so
        geometry or tissue changes take effect at once.
    render : callable
        ``render(panel)`` draws the current state of ``panel.system``.
    """

    TICK_MS = 30
    MAX_TICK_S = 0.1
    PULSE_DEG_PER_S = 180.0

    active_changed = pyqtSignal(bool)

    def __init__(self, context_provider, render, parent=None):
        super().__init__(parent)
        self._context_provider = context_provider
        self._render = render
        self.system = None
        self._context = None
        self._pulses = deque()
        # (time ms, B1x µT, B1y µT, Gx, Gy, Gz mT/m) at every field change;
        # each field holds until the next entry.
        self.field_history = deque(maxlen=FIELD_HISTORY_LENGTH)
        self._last_wall = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)

        # Two short rows keep the 3D Vector tab narrow enough for laptop
        # screens: pulses and gradients first, timing and status below.
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.live_button = QPushButton("Live mode")
        self.live_button.setObjectName("mag_3d_live_mode")
        self.live_button.setCheckable(True)
        self.live_button.setToolTip(
            "Play pulses and gradients by hand and watch the spins evolve.\n"
            "Keys while live: 1 = 90°, 2 = 180°, X/Y/Z = gradient, R = reset."
        )
        self.live_button.setStyleSheet(LIVE_BUTTON_STYLE)
        self.live_button.toggled.connect(self.set_active)
        row.addWidget(self.live_button)

        self.controls = QWidget()
        rows = QVBoxLayout(self.controls)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(2)
        controls = QHBoxLayout()
        rows.addLayout(controls)
        self.pulse_buttons = {}
        for flip in (90, 180):
            button = QPushButton(f"{flip}°")
            button.setObjectName(f"mag_3d_live_pulse_{flip}")
            button.setToolTip(
                f"Rectangular {flip}° pulse about the chosen axis, of the "
                "chosen pulse length"
            )
            button.clicked.connect(lambda _, f=flip: self.queue_pulse(f))
            controls.addWidget(button)
            self.pulse_buttons[flip] = button
        self.phase_combo = QComboBox()
        self.phase_combo.setObjectName("mag_3d_live_pulse_phase")
        self.phase_combo.addItems(list(PULSE_PHASES_DEG))
        self.phase_combo.setToolTip(
            "Pulse phase: x tips +z towards +y, y tips +z towards +x"
        )
        controls.addWidget(self.phase_combo)

        controls.addSpacing(12)
        self.gradient_buttons = []
        for axis in AXIS_NAMES:
            button = QPushButton(f"G{axis}")
            button.setObjectName(f"mag_3d_live_g{axis}")
            button.setCheckable(True)
            button.setToolTip(f"Switch the {axis} gradient on or off")
            button.toggled.connect(lambda _: self._fields_changed())
            controls.addWidget(button)
            self.gradient_buttons.append(button)
        self.gradient_spin = QDoubleSpinBox()
        self.gradient_spin.setObjectName("mag_3d_live_gradient")
        self.gradient_spin.setRange(-200.0, 200.0)
        self.gradient_spin.setDecimals(2)
        self.gradient_spin.setSingleStep(0.5)
        self.gradient_spin.setValue(2.0)
        self.gradient_spin.setSuffix(" mT/m")
        self.gradient_spin.setToolTip(
            "Strength of every switched-on gradient; negative values rewind"
        )
        self.gradient_spin.valueChanged.connect(lambda _: self._fields_changed())
        controls.addWidget(self.gradient_spin)
        controls.addStretch()

        controls = QHBoxLayout()
        rows.addLayout(controls)
        speed_label = QLabel("Speed:")
        controls.addWidget(speed_label)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setObjectName("mag_3d_live_speed")
        self.speed_spin.setRange(0.01, 1000.0)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSuffix(" ms/s")
        self.speed_spin.setToolTip("Simulated milliseconds per second on screen")
        speed_label.setBuddy(self.speed_spin)
        controls.addWidget(self.speed_spin)
        pulse_label = QLabel("Pulse:")
        controls.addWidget(pulse_label)
        self.pulse_length_spin = QDoubleSpinBox()
        self.pulse_length_spin.setObjectName("mag_3d_live_pulse_length")
        self.pulse_length_spin.setRange(0.01, 50.0)
        self.pulse_length_spin.setDecimals(2)
        self.pulse_length_spin.setSingleStep(0.05)
        self.pulse_length_spin.setValue(0.2)
        self.pulse_length_spin.setSuffix(" ms")
        self.pulse_length_spin.setToolTip(
            "Length of a pulse; B1 = flip angle / (gamma * length). Off-resonance "
            "and switched-on gradients act during the pulse."
        )
        pulse_label.setBuddy(self.pulse_length_spin)
        controls.addWidget(self.pulse_length_spin)
        self.relax_checkbox = QCheckBox("Relaxation")
        self.relax_checkbox.setObjectName("mag_3d_live_relaxation")
        self.relax_checkbox.setToolTip(
            "Relax with the tissue T1/T2 towards Mz = 1 between pulses"
        )
        controls.addWidget(self.relax_checkbox)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("mag_3d_live_reset")
        self.reset_button.setToolTip("Return all spins to equilibrium at t = 0")
        self.reset_button.clicked.connect(self.reset)
        controls.addWidget(self.reset_button)
        self.time_label = QLabel()
        self.time_label.setObjectName("mag_3d_live_time")
        self.time_label.setMinimumWidth(90)
        controls.addWidget(self.time_label)
        self.hint_label = QLabel("Single position: choose a line, plane or volume")
        self.hint_label.setObjectName("mag_3d_live_hint")
        self.hint_label.setToolTip(
            "Gradients only spread spins that sit at different positions"
        )
        self.hint_label.setWordWrap(True)
        # The hint must never widen the window; it wraps or clips instead.
        self.hint_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.hint_label.setVisible(False)
        controls.addWidget(self.hint_label, 2)
        controls.addStretch(1)
        row.addWidget(self.controls, 1)
        # While live mode is off, one line explains it instead of a block of
        # disabled controls.
        self.intro_label = QLabel("Play 90°/180° pulses and Gx/Gy/Gz gradients by hand")
        self.intro_label.setObjectName("mag_3d_live_intro")
        self.intro_label.setEnabled(False)
        row.addWidget(self.intro_label, 1)
        self._set_controls_enabled(False)

        # Window-wide while live; text fields still take typed digits.
        self._shortcuts = []
        for key, slot in (
            ("1", lambda: self.queue_pulse(90)),
            ("2", lambda: self.queue_pulse(180)),
            ("X", lambda: self.gradient_buttons[0].toggle()),
            ("Y", lambda: self.gradient_buttons[1].toggle()),
            ("Z", lambda: self.gradient_buttons[2].toggle()),
            ("R", self.reset),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.setEnabled(False)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)
        self._update_time_label()

    # -- state -------------------------------------------------------------

    def is_active(self) -> bool:
        return self.timer.isActive()

    def set_active(self, active: bool):
        """Start or stop the live mode."""
        active = bool(active)
        if active == self.is_active():
            return
        if self.live_button.isChecked() != active:
            self.live_button.blockSignals(True)
            self.live_button.setChecked(active)
            self.live_button.blockSignals(False)
        self._set_controls_enabled(active)
        for shortcut in self._shortcuts:
            shortcut.setEnabled(active)
        if active:
            self._sync_context(force=True)
            self.field_history.clear()
            self._record_fields()
            self._last_wall = time.perf_counter()
            self.timer.start(self.TICK_MS)
            self.active_changed.emit(True)
            self._draw()
        else:
            self.timer.stop()
            self._pulses.clear()
            self.active_changed.emit(False)

    def _set_controls_enabled(self, enabled: bool):
        self.controls.setEnabled(enabled)
        self.controls.setVisible(enabled)
        self.intro_label.setVisible(not enabled)

    def reset(self):
        """Return the spins to equilibrium and drop queued pulses."""
        self._pulses.clear()
        self._sync_context(force=True)
        self.field_history.clear()
        self._record_fields()
        self._draw()

    def queue_pulse(self, flip_deg: float):
        """Play a pulse after the ones still turning."""
        if not self.is_active():
            return
        phase = PULSE_PHASES_DEG[self.phase_combo.currentText()]
        length_s = self.pulse_length_spin.value() * 1e-3
        # [flip deg, phase deg, length s, remaining s]
        self._pulses.append([float(flip_deg), phase, length_s, length_s])
        self._draw()

    def gradient_t_per_m(self) -> np.ndarray:
        """Switched-on gradient (gx, gy, gz) in T/m."""
        on = np.array([b.isChecked() for b in self.gradient_buttons], dtype=float)
        return on * self.gradient_spin.value() * 1e-3

    def current_pulse(self):
        """(flip, phase) in degrees of the pulse being played, or None."""
        if not self._pulses:
            return None
        flip, phase = self._pulses[0][:2]
        return flip, phase

    @staticmethod
    def _rabi_hz(pulse) -> float:
        flip, _, length_s, _ = pulse
        return flip / 360.0 / length_s

    def b1_ut(self) -> complex:
        """RF field (B1x + i B1y) in µT of the pulse being played, else 0."""
        if not self._pulses or self.system is None:
            return 0j
        pulse = self._pulses[0]
        amplitude = self._rabi_hz(pulse) / self.system.gamma_hz_per_t * 1e6
        return amplitude * np.exp(1j * np.radians(pulse[1]))

    def _fields_changed(self):
        self._record_fields()
        self._draw()

    def _record_fields(self):
        """Log the current RF and gradient at the current time."""
        if self.system is None:
            return
        b1 = self.b1_ut()
        gradient = self.gradient_t_per_m() * 1e3
        entry = (self.system.time_s * 1e3, b1.real, b1.imag, *gradient)
        history = self.field_history
        if history and history[-1][1:] == entry[1:]:
            return
        if history and history[-1][0] >= entry[0]:
            history.pop()
            if history and history[-1][1:] == entry[1:]:
                return
        history.append(entry)

    def field_readout(self) -> str:
        """Pulse and gradient currently played, for the field label."""
        parts = []
        pulse = self.current_pulse()
        if pulse is not None:
            parts.append(f"{pulse[0]:.0f}°{self.phase_combo_name(pulse[1])}")
        gradient = self.gradient_t_per_m() * 1e3
        parts += [
            f"G{AXIS_NAMES[a]} {gradient[a]:.2f} mT/m"
            for a in range(3)
            if self.gradient_buttons[a].isChecked()
        ]
        return "   ".join(parts)

    @staticmethod
    def phase_combo_name(phase_deg: float) -> str:
        for name, value in PULSE_PHASES_DEG.items():
            if np.isclose(value, phase_deg % 360.0):
                return name
        return f"({phase_deg:.0f}°)"

    # -- evolution ---------------------------------------------------------

    def _sync_context(self, force: bool = False):
        """Rebuild the spins when geometry, offsets or nucleus changed."""
        context = self._context_provider()
        if context is None:
            return
        if force or self.system is None or not context.same_spins(self._context):
            self.field_history.clear()
            self.system = LiveSpinSystem(
                context.positions_m,
                context.offsets_hz,
                context.gamma_hz_per_t,
                context.t1_s,
                context.t2_s,
            )
        else:
            self.system.t1_s = float(context.t1_s)
            self.system.t2_s = float(context.t2_s)
        self._context = context
        self.hint_label.setVisible(len(context.positions_m) < 2)

    def _pulse_rate(self, pulse) -> float:
        """Simulated seconds per second on screen while ``pulse`` plays."""
        flip, _, length_s, _ = pulse
        slowest = length_s * self.PULSE_DEG_PER_S / max(flip, 1e-9)
        return min(self.speed_spin.value() * 1e-3, slowest)

    def advance(self, wall_dt_s: float):
        """Advance by ``wall_dt_s`` seconds on screen."""
        if self.system is None:
            return
        wall_left = min(max(float(wall_dt_s), 0.0), self.MAX_TICK_S)
        gradient = self.gradient_t_per_m()
        relax = self.relax_checkbox.isChecked()
        self._record_fields()
        while self._pulses and wall_left > 0.0:
            pulse = self._pulses[0]
            rate = self._pulse_rate(pulse)
            dt = min(pulse[3], wall_left * rate)
            self.system.drive(dt, self._rabi_hz(pulse), pulse[1], gradient, relax)
            pulse[3] -= dt
            wall_left -= dt / rate
            if pulse[3] <= 1e-12 * pulse[2]:
                self._pulses.popleft()
                wall_left = max(wall_left, 0.0)
            self._record_fields()
        if self._pulses or wall_left <= 0.0:
            return
        self.system.evolve(
            wall_left * self.speed_spin.value() * 1e-3, gradient, relax=relax
        )

    def tick(self):
        now = time.perf_counter()
        wall_dt = 0.0 if self._last_wall is None else now - self._last_wall
        self._last_wall = now
        self._sync_context()
        self.advance(wall_dt)
        if self.isVisible():
            self._draw()

    def _update_time_label(self):
        t_ms = 0.0 if self.system is None else self.system.time_s * 1e3
        self.time_label.setText(f"t = {t_ms:.2f} ms")

    def _draw(self):
        self._update_time_label()
        if self.is_active() and self.system is not None:
            self._render(self)
