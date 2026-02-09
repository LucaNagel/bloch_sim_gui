from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QSlider,
    QPushButton,
    QDoubleSpinBox,
)
from PyQt5.QtCore import Qt, pyqtSignal


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
        self.time_slider.setObjectName("playback_time_slider")
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.time_slider, 1)

        self.time_label = QLabel("0.0 ms")
        self.time_label.setFixedWidth(90)
        layout.addWidget(self.time_label)

        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.setObjectName("playback_play_btn")
        self.play_pause_button.setCheckable(True)
        self.play_pause_button.toggled.connect(self._update_play_pause_label)
        layout.addWidget(self.play_pause_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("playback_reset_btn")
        layout.addWidget(self.reset_button)

        layout.addWidget(QLabel("Speed (ms/s):"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setObjectName("playback_speed_spin")
        self.speed_spin.setRange(0.001, 1000.0)
        self.speed_spin.setValue(1.0)  # Default to 50 ms of sim per real second
        self.speed_spin.setSuffix(" ms/s")
        self.speed_spin.setSingleStep(0.1)
        layout.addWidget(self.speed_spin)

        # Backwards compatibility for existing signal connections
        self.play_button = self.play_pause_button
        self.pause_button = self.play_pause_button

        self.setLayout(layout)
        # self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.time_array = None  # Will store time array in seconds

    def set_time_range(self, time_array):
        """Set the time range from a time array (in seconds)."""
        import numpy as np

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
