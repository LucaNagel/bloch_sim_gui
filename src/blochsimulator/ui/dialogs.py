from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QDialogButtonBox,
    QFileDialog,
    QCheckBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QWidget,
)
from typing import Optional
from pathlib import Path

from ..memory import MemoryPolicy, format_bytes, resolve_memory_budget
from ..sequence.scanner import ScannerParameters
from ..units import NUCLEUS_GAMMA_HZ_PER_T
from .default_settings import WorkspaceDefaults


class SettingsDialog(QDialog):
    """Configure persistent application, scanner and simulation settings."""

    MODES = (
        ("Automatic reserve (recommended)", "automatic"),
        ("Custom free-memory reserve", "custom_reserve"),
        ("Fixed simulation limit", "fixed_limit"),
    )
    SEQUENCE_KERNELS = (
        ("Optimized (recommended)", "optimized"),
        ("Reference", "reference"),
    )
    DYNAMIC_SEQUENCE_KERNELS = (
        ("Optimized NumPy (recommended)", "optimized"),
        ("Native RF-block parallel (experimental)", "native_parallel"),
        ("Native RF-block serial (experimental)", "native_serial"),
        ("Reference", "reference"),
    )
    SEQUENCE_TIMESTEP_PRESETS = (
        ("Accurate — 1 µs", "accurate", 1.0),
        ("Balanced — 5 µs", "balanced", 5.0),
        ("Fast — 10 µs", "fast", 10.0),
        ("Custom", "custom", None),
    )

    def __init__(
        self,
        policy: MemoryPolicy,
        export_directory: Path,
        tooltips_enabled: bool,
        parent=None,
        initial_tab: str = "general",
        sequence_live_progress_enabled: bool = True,
        sequence_kernel: str = "optimized",
        dynamic_sequence_kernel: str = "optimized",
        sequence_timestep_preset: str = "balanced",
        sequence_timestep_us: float = 5.0,
        thread_mode: str = "automatic",
        manual_thread_count: int = 4,
        detected_thread_count: Optional[int] = None,
        scanner_parameters: Optional[ScannerParameters] = None,
        workspace_defaults: Optional[WorkspaceDefaults] = None,
    ):
        super().__init__(parent)
        scanner_parameters = ScannerParameters.from_mapping(scanner_parameters)
        workspace_defaults = workspace_defaults or WorkspaceDefaults()
        self.setWindowTitle("Settings")
        self.setMinimumWidth(600)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("settings_tabs")
        layout.addWidget(self.tabs)

        general_tab = QWidget()
        general_form = QFormLayout(general_tab)
        export_layout = QHBoxLayout()
        self.export_directory_edit = QLineEdit(str(export_directory))
        self.export_directory_edit.setObjectName("default_export_directory")
        self.export_directory_edit.setToolTip(
            "Default folder offered by image, data, animation and notebook export dialogs."
        )
        export_layout.addWidget(self.export_directory_edit, 1)
        self.export_browse_button = QPushButton("Browse...")
        self.export_browse_button.setToolTip(
            "Choose an existing folder as the default export location."
        )
        self.export_browse_button.clicked.connect(self._browse_export_directory)
        export_layout.addWidget(self.export_browse_button)
        general_form.addRow("Default export directory:", export_layout)
        self.tabs.addTab(general_tab, "General")

        defaults_tab = QWidget()
        defaults_form = QFormLayout(defaults_tab)

        self.sequence_fov_spins = []
        for axis, value in zip("XYZ", workspace_defaults.sequence_fov_mm):
            spin = QDoubleSpinBox()
            spin.setObjectName(f"default_sequence_fov_{axis.lower()}_mm")
            spin.setRange(0.1, 10000.0)
            spin.setDecimals(3)
            spin.setSuffix(" mm")
            spin.setValue(float(value))
            spin.setToolTip(
                f"Default {axis}-axis FOV for generated sequences and the built-in quick object."
            )
            defaults_form.addRow(f"Sequence FOV {axis}:", spin)
            self.sequence_fov_spins.append(spin)

        self.phantom_fov_spins = []
        for axis, value in zip("XYZ", workspace_defaults.phantom_fov_mm):
            spin = QDoubleSpinBox()
            spin.setObjectName(f"default_phantom_fov_{axis.lower()}_mm")
            spin.setRange(0.01, 1000.0)
            spin.setDecimals(3)
            spin.setSuffix(" mm")
            spin.setValue(float(value))
            spin.setToolTip(f"Default {axis}-axis FOV for newly designed phantoms.")
            defaults_form.addRow(f"Phantom FOV {axis}:", spin)
            self.phantom_fov_spins.append(spin)

        self.phantom_nucleus_combo = QComboBox()
        self.phantom_nucleus_combo.setObjectName("default_phantom_nucleus")
        self.phantom_nucleus_combo.addItem("Automatic (H1 static / C13 dynamic)", None)
        for nucleus in sorted(NUCLEUS_GAMMA_HZ_PER_T):
            self.phantom_nucleus_combo.addItem(nucleus, nucleus)
        nucleus_index = self.phantom_nucleus_combo.findData(
            workspace_defaults.phantom_nucleus
        )
        self.phantom_nucleus_combo.setCurrentIndex(max(0, nucleus_index))
        self.phantom_nucleus_combo.setToolTip(
            "Default nucleus for new Phantom Designer projects. Automatic keeps "
            "the existing H1 static and C13 dynamic behavior."
        )
        defaults_form.addRow("Phantom nucleus:", self.phantom_nucleus_combo)

        self.default_field_strength_spin = QDoubleSpinBox()
        self.default_field_strength_spin.setObjectName("default_field_strength_t")
        self.default_field_strength_spin.setRange(0.01, 30.0)
        self.default_field_strength_spin.setDecimals(4)
        self.default_field_strength_spin.setSuffix(" T")
        self.default_field_strength_spin.setValue(workspace_defaults.field_strength_t)
        self.default_field_strength_spin.setToolTip(
            "Default B0 field strength for Sequence Simulation and newly created phantoms."
        )
        defaults_form.addRow("B0 field strength:", self.default_field_strength_spin)
        self.tabs.addTab(defaults_tab, "Defaults")

        simulation_tab = QWidget()
        simulation_form = QFormLayout(simulation_tab)

        self.sequence_timestep_preset_combo = QComboBox()
        self.sequence_timestep_preset_combo.setObjectName("sequence_timestep_preset")
        for label, preset, _ in self.SEQUENCE_TIMESTEP_PRESETS:
            self.sequence_timestep_preset_combo.addItem(label, preset)
        preset_index = self.sequence_timestep_preset_combo.findData(
            sequence_timestep_preset
        )
        self.sequence_timestep_preset_combo.setCurrentIndex(max(0, preset_index))
        self.sequence_timestep_preset_combo.setToolTip(
            "Larger RF-active time steps reduce runtime by averaging RF and "
            "simultaneous gradients. ADC times and event boundaries remain exact."
        )
        simulation_form.addRow(
            "Sequence time-step preset:", self.sequence_timestep_preset_combo
        )

        self.sequence_timestep_us_spin = QDoubleSpinBox()
        self.sequence_timestep_us_spin.setObjectName("sequence_timestep_us")
        self.sequence_timestep_us_spin.setRange(0.1, 1000.0)
        self.sequence_timestep_us_spin.setDecimals(2)
        self.sequence_timestep_us_spin.setSingleStep(0.1)
        self.sequence_timestep_us_spin.setSuffix(" µs")
        self.sequence_timestep_us_spin.setValue(float(sequence_timestep_us))
        self.sequence_timestep_us_spin.setToolTip(
            "Custom maximum interval while RF is active, in microseconds."
        )
        simulation_form.addRow(
            "RF-active simulation time step:", self.sequence_timestep_us_spin
        )

        detected_threads = max(1, int(detected_thread_count or 1))
        self.thread_mode_combo = QComboBox()
        self.thread_mode_combo.setObjectName("simulation_thread_mode")
        self.thread_mode_combo.addItem(
            f"Automatic ({detected_threads} threads)", "automatic"
        )
        self.thread_mode_combo.addItem("Manual", "manual")
        thread_mode_index = self.thread_mode_combo.findData(thread_mode)
        self.thread_mode_combo.setCurrentIndex(max(0, thread_mode_index))
        self.thread_mode_combo.setToolTip(
            "Automatic uses all logical processors reported by the operating system."
        )
        simulation_form.addRow("CPU threads:", self.thread_mode_combo)

        self.manual_thread_count_spin = QSpinBox()
        self.manual_thread_count_spin.setObjectName("simulation_manual_threads")
        self.manual_thread_count_spin.setRange(1, max(256, detected_threads * 4))
        self.manual_thread_count_spin.setValue(max(1, int(manual_thread_count)))
        self.manual_thread_count_spin.setToolTip(
            "Native worker count used when CPU thread selection is manual."
        )
        simulation_form.addRow("Manual thread count:", self.manual_thread_count_spin)

        self.sequence_kernel_combo = QComboBox()
        self.sequence_kernel_combo.setObjectName("sequence_simulation_kernel")
        for label, kernel in self.SEQUENCE_KERNELS:
            self.sequence_kernel_combo.addItem(label, kernel)
        kernel_index = self.sequence_kernel_combo.findData(sequence_kernel)
        self.sequence_kernel_combo.setCurrentIndex(max(0, kernel_index))
        self.sequence_kernel_combo.setToolTip(
            "The optimized Bloch kernel uses equivalent RF-free and uniform-"
            "relaxation fast paths. Reference keeps the original propagation "
            "path for numerical comparisons."
        )
        simulation_form.addRow("Sequence Bloch kernel:", self.sequence_kernel_combo)

        self.dynamic_sequence_kernel_combo = QComboBox()
        self.dynamic_sequence_kernel_combo.setObjectName(
            "dynamic_sequence_simulation_kernel"
        )
        for label, kernel in self.DYNAMIC_SEQUENCE_KERNELS:
            self.dynamic_sequence_kernel_combo.addItem(label, kernel)
        dynamic_kernel_index = self.dynamic_sequence_kernel_combo.findData(
            dynamic_sequence_kernel
        )
        self.dynamic_sequence_kernel_combo.setCurrentIndex(max(0, dynamic_kernel_index))
        self.dynamic_sequence_kernel_combo.setToolTip(
            "Kernel for dynamic two-pool pyruvate/lactate phantoms. Native "
            "RF-block kernels remove temporary RF rotation arrays; the parallel "
            "variant also uses multiple CPU cores for supported static-B0 cases. "
            "Inflow and dynamic B0 currently fall back safely to optimized NumPy."
        )
        simulation_form.addRow(
            "Dynamic two-pool kernel:", self.dynamic_sequence_kernel_combo
        )
        self.tabs.addTab(simulation_tab, "Simulation")

        scanner_tab = QWidget()
        scanner_layout = QVBoxLayout(scanner_tab)
        scanner_form = QFormLayout()

        self.scanner_max_grad_spin = QDoubleSpinBox()
        self.scanner_max_grad_spin.setObjectName("scanner_max_grad_mtm")
        self.scanner_max_grad_spin.setRange(0.1, 1000.0)
        self.scanner_max_grad_spin.setDecimals(2)
        self.scanner_max_grad_spin.setSingleStep(1.0)
        self.scanner_max_grad_spin.setSuffix(" mT/m")
        self.scanner_max_grad_spin.setValue(scanner_parameters.max_grad_mtm)
        self.scanner_max_grad_spin.setToolTip(
            "Maximum gradient amplitude used when generated Pulseq sequences are designed."
        )
        scanner_form.addRow("Maximum gradient:", self.scanner_max_grad_spin)

        self.scanner_max_slew_spin = QDoubleSpinBox()
        self.scanner_max_slew_spin.setObjectName("scanner_max_slew_tms")
        self.scanner_max_slew_spin.setRange(0.1, 10000.0)
        self.scanner_max_slew_spin.setDecimals(2)
        self.scanner_max_slew_spin.setSingleStep(5.0)
        self.scanner_max_slew_spin.setSuffix(" T/m/s")
        self.scanner_max_slew_spin.setValue(scanner_parameters.max_slew_tms)
        self.scanner_max_slew_spin.setToolTip(
            "Maximum gradient slew rate used for generated Pulseq waveforms."
        )
        scanner_form.addRow("Maximum slew rate:", self.scanner_max_slew_spin)

        def timing_spin(
            object_name: str,
            value_s: float,
            *,
            allow_zero: bool = False,
        ) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setObjectName(object_name)
            spin.setRange(0.0 if allow_zero else 0.001, 1_000_000.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setSuffix(" µs")
            spin.setValue(float(value_s) * 1e6)
            return spin

        self.scanner_grad_raster_spin = timing_spin(
            "scanner_grad_raster_time_us", scanner_parameters.grad_raster_time_s
        )
        self.scanner_grad_raster_spin.setToolTip(
            "Gradient waveform raster interval used by the scanner."
        )
        scanner_form.addRow("Gradient raster time:", self.scanner_grad_raster_spin)

        self.scanner_rf_raster_spin = timing_spin(
            "scanner_rf_raster_time_us", scanner_parameters.rf_raster_time_s
        )
        self.scanner_rf_raster_spin.setToolTip(
            "RF waveform raster interval used by the scanner."
        )
        scanner_form.addRow("RF raster time:", self.scanner_rf_raster_spin)

        self.scanner_adc_raster_spin = timing_spin(
            "scanner_adc_raster_time_us", scanner_parameters.adc_raster_time_s
        )
        self.scanner_adc_raster_spin.setToolTip(
            "ADC dwell-time raster used to quantize generated acquisitions."
        )
        scanner_form.addRow("ADC raster time:", self.scanner_adc_raster_spin)

        self.scanner_block_raster_spin = timing_spin(
            "scanner_block_duration_raster_us",
            scanner_parameters.block_duration_raster_s,
        )
        self.scanner_block_raster_spin.setToolTip(
            "Raster to which complete Pulseq block durations are aligned."
        )
        scanner_form.addRow("Block-duration raster:", self.scanner_block_raster_spin)

        self.scanner_rf_ringdown_spin = timing_spin(
            "scanner_rf_ringdown_time_us",
            scanner_parameters.rf_ringdown_time_s,
            allow_zero=True,
        )
        self.scanner_rf_ringdown_spin.setToolTip(
            "Required RF ringdown interval after transmit events."
        )
        scanner_form.addRow("RF ringdown time:", self.scanner_rf_ringdown_spin)

        self.scanner_rf_dead_time_spin = timing_spin(
            "scanner_rf_dead_time_us",
            scanner_parameters.rf_dead_time_s,
            allow_zero=True,
        )
        self.scanner_rf_dead_time_spin.setToolTip(
            "Required scanner dead time before RF events."
        )
        scanner_form.addRow("RF dead time:", self.scanner_rf_dead_time_spin)

        self.scanner_adc_dead_time_spin = timing_spin(
            "scanner_adc_dead_time_us",
            scanner_parameters.adc_dead_time_s,
            allow_zero=True,
        )
        self.scanner_adc_dead_time_spin.setToolTip(
            "Required scanner dead time before ADC sampling."
        )
        scanner_form.addRow("ADC dead time:", self.scanner_adc_dead_time_spin)

        scanner_layout.addLayout(scanner_form)
        scanner_explanation = QLabel(
            "These limits are applied to all EPI, spiral, CSI and bSSFP "
            "sequences generated after the settings are saved. Imported "
            "Pulseq files keep their own event timing."
        )
        scanner_explanation.setWordWrap(True)
        scanner_layout.addWidget(scanner_explanation)
        scanner_layout.addStretch()
        self.tabs.addTab(scanner_tab, "Scanner")

        memory_tab = QWidget()
        memory_layout = QVBoxLayout(memory_tab)
        form = QFormLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("memory_policy_mode")
        for label, mode in self.MODES:
            self.mode_combo.addItem(label, mode)
        mode_index = self.mode_combo.findData(policy.mode)
        self.mode_combo.setCurrentIndex(max(0, mode_index))
        self.mode_combo.setToolTip(
            "Choose automatic RAM reservation, a custom amount of memory kept "
            "free, or a fixed maximum allocation per simulation."
        )
        form.addRow("Policy:", self.mode_combo)

        self.reserve_spin = QDoubleSpinBox()
        self.reserve_spin.setObjectName("memory_reserve_gib")
        self.reserve_spin.setRange(0.25, 256.0)
        self.reserve_spin.setDecimals(2)
        self.reserve_spin.setSingleStep(0.5)
        self.reserve_spin.setSuffix(" GiB")
        self.reserve_spin.setValue(policy.reserve_bytes / 1024**3)
        self.reserve_spin.setToolTip(
            "Amount of currently available RAM that must remain unused by the simulation."
        )
        form.addRow("Keep free:", self.reserve_spin)

        self.limit_spin = QDoubleSpinBox()
        self.limit_spin.setObjectName("memory_limit_gib")
        self.limit_spin.setRange(0.25, 1024.0)
        self.limit_spin.setDecimals(2)
        self.limit_spin.setSingleStep(1.0)
        self.limit_spin.setSuffix(" GiB")
        self.limit_spin.setValue(policy.limit_bytes / 1024**3)
        self.limit_spin.setToolTip(
            "Maximum estimated RAM allocation permitted for one simulation."
        )
        form.addRow("Maximum per simulation:", self.limit_spin)

        memory_layout.addLayout(form)

        explanation = QLabel(
            "Automatic mode keeps at least 2 GiB or 10% of total system RAM "
            "free, whichever is larger. The fixed limit still preserves a "
            "512 MiB emergency reserve."
        )
        explanation.setWordWrap(True)
        memory_layout.addWidget(explanation)

        self.status_label = QLabel()
        self.status_label.setObjectName("memory_budget_summary")
        self.status_label.setWordWrap(True)
        memory_layout.addWidget(self.status_label)
        memory_layout.addStretch()
        self.tabs.addTab(memory_tab, "Memory")

        interface_tab = QWidget()
        interface_layout = QVBoxLayout(interface_tab)
        self.tooltips_checkbox = QCheckBox("Show explanatory tooltips")
        self.tooltips_checkbox.setObjectName("tooltips_enabled")
        self.tooltips_checkbox.setChecked(bool(tooltips_enabled))
        self.tooltips_checkbox.setToolTip(
            "Show short explanations when the pointer rests over controls and input fields."
        )
        interface_layout.addWidget(self.tooltips_checkbox)
        self.sequence_live_progress_checkbox = QCheckBox(
            "Show live progress during sequence simulations"
        )
        self.sequence_live_progress_checkbox.setObjectName(
            "sequence_live_progress_enabled"
        )
        self.sequence_live_progress_checkbox.setChecked(
            bool(sequence_live_progress_enabled)
        )
        self.sequence_live_progress_checkbox.setToolTip(
            "Move a cursor through the sequence and update k-space and the "
            "intermediate reconstruction while a sequence simulation runs."
        )
        interface_layout.addWidget(self.sequence_live_progress_checkbox)
        interface_layout.addStretch()
        self.tabs.addTab(interface_tab, "Interface")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.mode_combo.currentIndexChanged.connect(self._update_summary)
        self.reserve_spin.valueChanged.connect(self._update_summary)
        self.limit_spin.valueChanged.connect(self._update_summary)
        self.sequence_timestep_preset_combo.currentIndexChanged.connect(
            self._update_simulation_controls
        )
        self.thread_mode_combo.currentIndexChanged.connect(
            self._update_simulation_controls
        )
        self._update_summary()
        self._update_simulation_controls()

        tab_names = {
            "general": 0,
            "defaults": 1,
            "simulation": 2,
            "scanner": 3,
            "memory": 4,
            "interface": 5,
        }
        self.tabs.setCurrentIndex(tab_names.get(initial_tab, 0))

    def get_policy(self) -> MemoryPolicy:
        return MemoryPolicy(
            mode=str(self.mode_combo.currentData()),
            reserve_bytes=int(self.reserve_spin.value() * 1024**3),
            limit_bytes=int(self.limit_spin.value() * 1024**3),
        )

    def get_export_directory(self) -> Path:
        return Path(self.export_directory_edit.text()).expanduser()

    def tooltips_enabled(self) -> bool:
        return self.tooltips_checkbox.isChecked()

    def sequence_live_progress_enabled(self) -> bool:
        return self.sequence_live_progress_checkbox.isChecked()

    def sequence_kernel(self) -> str:
        return str(self.sequence_kernel_combo.currentData())

    def dynamic_sequence_kernel(self) -> str:
        return str(self.dynamic_sequence_kernel_combo.currentData())

    def sequence_timestep_preset(self) -> str:
        return str(self.sequence_timestep_preset_combo.currentData())

    def sequence_timestep_us(self) -> float:
        return float(self.sequence_timestep_us_spin.value())

    def thread_mode(self) -> str:
        return str(self.thread_mode_combo.currentData())

    def manual_thread_count(self) -> int:
        return int(self.manual_thread_count_spin.value())

    def scanner_parameters(self) -> ScannerParameters:
        """Return the validated scanner hardware profile selected in the dialog."""
        return ScannerParameters(
            max_grad_mtm=float(self.scanner_max_grad_spin.value()),
            max_slew_tms=float(self.scanner_max_slew_spin.value()),
            grad_raster_time_s=float(self.scanner_grad_raster_spin.value()) * 1e-6,
            rf_raster_time_s=float(self.scanner_rf_raster_spin.value()) * 1e-6,
            adc_raster_time_s=float(self.scanner_adc_raster_spin.value()) * 1e-6,
            block_duration_raster_s=(
                float(self.scanner_block_raster_spin.value()) * 1e-6
            ),
            rf_ringdown_time_s=(float(self.scanner_rf_ringdown_spin.value()) * 1e-6),
            rf_dead_time_s=float(self.scanner_rf_dead_time_spin.value()) * 1e-6,
            adc_dead_time_s=float(self.scanner_adc_dead_time_spin.value()) * 1e-6,
        )

    def workspace_defaults(self) -> WorkspaceDefaults:
        """Return defaults used for newly created sequences and phantoms."""
        return WorkspaceDefaults(
            sequence_fov_mm=tuple(
                float(spin.value()) for spin in self.sequence_fov_spins
            ),
            phantom_fov_mm=tuple(
                float(spin.value()) for spin in self.phantom_fov_spins
            ),
            phantom_nucleus=self.phantom_nucleus_combo.currentData(),
            field_strength_t=float(self.default_field_strength_spin.value()),
        )

    def _update_simulation_controls(self):
        preset = self.sequence_timestep_preset()
        preset_values = {key: value for _, key, value in self.SEQUENCE_TIMESTEP_PRESETS}
        if preset != "custom":
            self.sequence_timestep_us_spin.setValue(preset_values[preset])
        self.sequence_timestep_us_spin.setEnabled(preset == "custom")
        self.manual_thread_count_spin.setEnabled(self.thread_mode() == "manual")

    def _browse_export_directory(self):
        current = str(self.get_export_directory())
        selected = QFileDialog.getExistingDirectory(
            self, "Select Default Export Directory", current
        )
        if selected:
            self.export_directory_edit.setText(selected)

    def _update_summary(self):
        mode = str(self.mode_combo.currentData())
        self.reserve_spin.setEnabled(mode == "custom_reserve")
        self.limit_spin.setEnabled(mode == "fixed_limit")

        budget = resolve_memory_budget(policy=self.get_policy())
        if budget.available_bytes is None:
            summary = (
                f"Hardware memory could not be detected. Effective budget: "
                f"{format_bytes(budget.limit_bytes)}."
            )
        else:
            summary = (
                f"System RAM: {format_bytes(budget.total_bytes or 0)} total, "
                f"{format_bytes(budget.available_bytes)} currently available. "
                f"Effective budget now: {format_bytes(budget.limit_bytes)}."
            )
        self.status_label.setText(summary)


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
        self.layout_mode.setObjectName("import_layout_mode")
        self.layout_mode.addItems(
            [
                "Interleaved: amp, phase, amp, phase",
                "Interleaved: phase, amp, phase, amp",
                "Columns: amp | phase per row",
            ]
        )
        self.layout_mode.setCurrentIndex(0)
        self.layout_mode.setToolTip(
            "Describe how amplitude and phase values are arranged in the imported text file."
        )
        form.addRow("Data layout:", self.layout_mode)

        self.amp_unit = QComboBox()
        self.amp_unit.setObjectName("import_amp_unit")
        self.amp_unit.addItems(
            [
                "Percent (0-100)",
                "Fraction (0-1)",
                "Gauss",
                "mT",
                "uT",
            ]
        )
        self.amp_unit.setCurrentIndex(0)
        self.amp_unit.setToolTip(
            "Physical or relative unit used by the imported RF amplitude values."
        )
        form.addRow("Amplitude unit:", self.amp_unit)

        self.phase_unit = QComboBox()
        self.phase_unit.setObjectName("import_phase_unit")
        self.phase_unit.addItems(["Degrees", "Radians"])
        self.phase_unit.setCurrentIndex(0)
        self.phase_unit.setToolTip("Angular unit used by imported RF phase values.")
        form.addRow("Phase unit:", self.phase_unit)

        self.duration_ms = QDoubleSpinBox()
        self.duration_ms.setObjectName("import_duration_ms")
        self.duration_ms.setRange(0.001, 100000.0)
        self.duration_ms.setDecimals(3)
        self.duration_ms.setSingleStep(0.1)
        self.duration_ms.setValue(1.0)
        self.duration_ms.setToolTip(
            "Total duration assigned to the imported RF waveform, in milliseconds."
        )
        form.addRow("Duration (ms):", self.duration_ms)

        layout.addLayout(form)
        layout.addWidget(
            QLabel(
                "Tip: Percent/fraction amplitudes are treated as relative and rescaled from flip angle."
            )
        )

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

        phase_unit = (
            "deg" if self.phase_unit.currentText().lower().startswith("deg") else "rad"
        )

        return {
            "layout": layout,
            "amp_unit": amp_unit,
            "phase_unit": phase_unit,
            "duration_s": float(self.duration_ms.value()) / 1000.0,
        }
