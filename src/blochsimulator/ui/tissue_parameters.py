from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSlider,
    QCheckBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from ..simulator import TissueParameters
from ..units import NUCLEUS_GAMMA_HZ_PER_T
from .styles import BOLD_GROUP_TITLES_STYLE


class TissueParameterWidget(QGroupBox):
    """Widget for setting tissue parameters."""

    field_strength_changed = pyqtSignal(float)
    nucleus_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__("Single-Spin / Ensemble Tissue")
        self.setStyleSheet(BOLD_GROUP_TITLES_STYLE)
        self.setToolTip(
            "Parameters for the classic single-spin/ensemble simulation. "
            "Sequence Simulation uses the selected phantom maps instead."
        )
        self.sequence_presets_enabled = True  # Default: auto-load presets
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Preset selector
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setObjectName("tissue_preset_combo")
        self.preset_combo.addItems(
            [
                "Custom",
                "Gray Matter",
                "White Matter",
                "CSF",
                "Muscle",
                "Fat",
                "Blood",
                "Liver",
                "Hyperpolarized 13C Pyruvate",
            ]
        )
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        preset_layout.addWidget(self.preset_combo)
        layout.addLayout(preset_layout)

        # Shared scanner frequency reference
        reference_layout = QHBoxLayout()
        reference_layout.addWidget(QLabel("Field:"))
        self.field_combo = QComboBox()
        self.field_combo.setObjectName("field_strength_combo")
        self.field_combo.addItems(["1.5T", "3.0T", "7.0T"])
        self.field_combo.setCurrentText("3.0T")
        self.field_combo.currentTextChanged.connect(self._field_strength_changed)
        reference_layout.addWidget(self.field_combo)

        reference_layout.addWidget(QLabel("Nucleus:"))
        self.nucleus_combo = QComboBox()
        self.nucleus_combo.setObjectName("nucleus_combo")
        self.nucleus_combo.addItems(sorted(NUCLEUS_GAMMA_HZ_PER_T))
        self.nucleus_combo.setCurrentText("H1")
        self.nucleus_combo.setToolTip(
            "Shared reference nucleus used by Phantom and Sequence Mode and "
            "recorded in Free Mode exports."
        )
        self.nucleus_combo.currentTextChanged.connect(self.nucleus_changed)
        reference_layout.addWidget(self.nucleus_combo)
        reference_layout.addStretch()
        layout.addLayout(reference_layout)

        # Sequence-specific presets toggle
        seq_preset_layout = QHBoxLayout()
        self.seq_preset_checkbox = QCheckBox("Auto-load sequence presets")
        self.seq_preset_checkbox.setObjectName("seq_preset_checkbox")
        self.seq_preset_checkbox.setChecked(True)
        self.seq_preset_checkbox.setToolTip(
            "Automatically load TE/TR/TI presets when sequence changes"
        )
        self.seq_preset_checkbox.toggled.connect(self._toggle_sequence_presets)
        seq_preset_layout.addWidget(self.seq_preset_checkbox)
        layout.addLayout(seq_preset_layout)

        # T1 parameter
        t1_layout = QHBoxLayout()
        t1_layout.addWidget(QLabel("T1 (ms):"))
        self.t1_spin = QDoubleSpinBox()
        self.t1_spin.setObjectName("t1_spin")
        self.t1_spin.setRange(1, 180000)
        self.t1_spin.setValue(1000)
        self.t1_spin.setSuffix(" ms")
        t1_layout.addWidget(self.t1_spin)

        self.t1_slider = QSlider(Qt.Horizontal)
        self.t1_slider.setObjectName("t1_slider")
        self.t1_slider.setRange(1, 180000)
        self.t1_slider.setValue(1000)
        self.t1_slider.valueChanged.connect(lambda v: self.t1_spin.setValue(v))
        self.t1_spin.valueChanged.connect(lambda v: self.t1_slider.setValue(int(v)))
        t1_layout.addWidget(self.t1_slider)
        layout.addLayout(t1_layout)

        # T2 parameter
        t2_layout = QHBoxLayout()
        t2_layout.addWidget(QLabel("T2 (ms):"))
        self.t2_spin = QDoubleSpinBox()
        self.t2_spin.setObjectName("t2_spin")
        self.t2_spin.setRange(1, 20000)
        self.t2_spin.setValue(100)
        self.t2_spin.setSuffix(" ms")
        t2_layout.addWidget(self.t2_spin)

        self.t2_slider = QSlider(Qt.Horizontal)
        self.t2_slider.setObjectName("t2_slider")
        self.t2_slider.setRange(1, 20000)
        self.t2_slider.setValue(100)
        self.t2_slider.valueChanged.connect(lambda v: self.t2_spin.setValue(v))
        self.t2_spin.valueChanged.connect(lambda v: self.t2_slider.setValue(int(v)))
        t2_layout.addWidget(self.t2_slider)
        layout.addLayout(t2_layout)

        # Initial magnetization (Mz)
        m0_layout = QHBoxLayout()
        m0_layout.addWidget(QLabel("Initial Mz:"))
        self.m0_spin = QDoubleSpinBox()
        self.m0_spin.setObjectName("m0_spin")
        self.m0_spin.setRange(-1e9, 1e9)
        self.m0_spin.setDecimals(3)
        self.m0_spin.setValue(1.0)
        m0_layout.addWidget(self.m0_spin)
        layout.addLayout(m0_layout)

        self.setLayout(layout)

    def get_field_strength(self) -> float:
        """Return the numeric main-field value shown by the widget."""
        text = self.field_combo.currentText().removesuffix("T").strip()
        try:
            return float(text)
        except ValueError:
            return 3.0

    def set_field_strength(self, value_t: float) -> None:
        """Apply a shared B0 value, including non-preset scanner fields."""
        value_t = float(value_t)
        matching_index = -1
        for index in range(self.field_combo.count()):
            try:
                candidate = float(
                    self.field_combo.itemText(index).removesuffix("T").strip()
                )
            except ValueError:
                continue
            if abs(candidate - value_t) <= 1e-9 * max(1.0, abs(value_t)):
                matching_index = index
                break
        if matching_index < 0:
            self.field_combo.addItem(f"{value_t:g}T")
            matching_index = self.field_combo.count() - 1
        self.field_combo.setCurrentIndex(matching_index)

    def get_nucleus(self) -> str:
        """Return the shared reference nucleus shown by the widget."""
        nucleus = self.nucleus_combo.currentText().strip()
        return nucleus if nucleus in NUCLEUS_GAMMA_HZ_PER_T else "H1"

    def set_nucleus(self, nucleus: str) -> None:
        """Apply the shared reference nucleus without inventing new entries."""
        nucleus = str(nucleus).strip()
        index = self.nucleus_combo.findText(nucleus)
        if index >= 0:
            self.nucleus_combo.setCurrentIndex(index)

    def _field_strength_changed(self, *_):
        self.load_preset()
        self.field_strength_changed.emit(self.get_field_strength())

    def load_preset(self):
        """Load tissue parameter preset."""
        preset = self.preset_combo.currentText()
        field = self.get_field_strength()

        preset_factories = {
            "Gray Matter": TissueParameters.gray_matter,
            "White Matter": TissueParameters.white_matter,
            "CSF": TissueParameters.csf,
        }
        if preset in preset_factories:
            try:
                tissue = preset_factories[preset](field)
            except ValueError:
                # Arbitrary scanner fields (for example 9.4 T) are valid for
                # ppm/Hz conversion even when no relaxation preset exists.
                # Preserve the explicitly configured T1/T2 values in that case.
                return
        elif preset == "Hyperpolarized 13C Pyruvate":
            # Typical HP 13C pyruvate values (approx.): long T1, slower decay
            self.t1_spin.setValue(25000)  # 25 s
            self.t2_spin.setValue(300)  # 0.3 s
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
        )

    def get_initial_mz(self) -> float:
        """Return the initial longitudinal magnetization."""
        return float(self.m0_spin.value())

    def _toggle_sequence_presets(self, enabled: bool):
        """Toggle automatic loading of sequence presets."""
        self.sequence_presets_enabled = enabled

    def get_state(self) -> dict:
        """Return current widget state as a dictionary."""
        return {
            "preset": self.preset_combo.currentText(),
            "field": self.field_combo.currentText(),
            "field_strength_t": self.get_field_strength(),
            "nucleus": self.get_nucleus(),
            "t1_ms": self.t1_spin.value(),
            "t2_ms": self.t2_spin.value(),
            "m0": self.m0_spin.value(),
            "auto_load_presets": self.seq_preset_checkbox.isChecked(),
        }

    def set_state(self, state: dict):
        """Restore widget state from a dictionary."""
        if not state:
            return

        # Block signals to avoid triggering multiple preset loads
        self.preset_combo.blockSignals(True)
        self.field_combo.blockSignals(True)
        self.nucleus_combo.blockSignals(True)
        self.seq_preset_checkbox.blockSignals(True)

        try:
            if "preset" in state:
                self.preset_combo.setCurrentText(state["preset"])
            if "field_strength_t" in state:
                self.set_field_strength(state["field_strength_t"])
            elif "field" in state:
                field = str(state["field"])
                try:
                    self.set_field_strength(float(field.removesuffix("T").strip()))
                except ValueError:
                    self.field_combo.setCurrentText(field)
            if "nucleus" in state:
                self.set_nucleus(state["nucleus"])
            if "t1_ms" in state:
                self.t1_spin.setValue(state["t1_ms"])
            if "t2_ms" in state:
                self.t2_spin.setValue(state["t2_ms"])
            if "m0" in state:
                self.m0_spin.setValue(state["m0"])
            if "auto_load_presets" in state:
                self.seq_preset_checkbox.setChecked(state["auto_load_presets"])
                self.sequence_presets_enabled = state["auto_load_presets"]
        finally:
            self.preset_combo.blockSignals(False)
            self.field_combo.blockSignals(False)
            self.nucleus_combo.blockSignals(False)
            self.seq_preset_checkbox.blockSignals(False)
