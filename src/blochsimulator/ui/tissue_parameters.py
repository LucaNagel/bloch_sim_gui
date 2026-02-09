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
from PyQt5.QtCore import Qt
from ..simulator import TissueParameters


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

        # Field strength
        preset_layout.addWidget(QLabel("Field:"))
        self.field_combo = QComboBox()
        self.field_combo.setObjectName("field_strength_combo")
        self.field_combo.addItems(["1.5T", "3.0T", "7.0T"])
        self.field_combo.setCurrentText("3.0T")
        self.field_combo.currentTextChanged.connect(self.load_preset)
        preset_layout.addWidget(self.field_combo)
        layout.addLayout(preset_layout)

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
        self.t1_spin.setRange(1, 5000)
        self.t1_spin.setValue(1000)
        self.t1_spin.setSuffix(" ms")
        t1_layout.addWidget(self.t1_spin)

        self.t1_slider = QSlider(Qt.Horizontal)
        self.t1_slider.setObjectName("t1_slider")
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
        self.t2_spin.setObjectName("t2_spin")
        self.t2_spin.setRange(1, 2000)
        self.t2_spin.setValue(100)
        self.t2_spin.setSuffix(" ms")
        t2_layout.addWidget(self.t2_spin)

        self.t2_slider = QSlider(Qt.Horizontal)
        self.t2_slider.setObjectName("t2_slider")
        self.t2_slider.setRange(1, 2000)
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
            self.t2_spin.setValue(1000)  # 1 s
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
