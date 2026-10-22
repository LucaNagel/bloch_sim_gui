import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PyQt5.QtCore import QPointF, QSettings, Qt
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QLineEdit,
    QRadioButton,
    QSlider,
    QSpinBox,
)
from blochsimulator.gradient_lobes import GradientLobe, gradient_moments_mt_ms_per_m
from blochsimulator.kernel_capabilities import GUI_KERNEL_NAMES
from blochsimulator.memory import GIB, MemoryPolicy
from blochsimulator.project_io import load_project, save_project
from blochsimulator.pulse_loader import load_amp_phase_dat, load_pulse_from_file
from blochsimulator.sequence import ScannerParameters
from blochsimulator.simulator import (
    GradientEcho,
    InversionRecovery,
    SliceSelectRephase,
    SpinEcho,
    SpinEchoTipAxis,
)
from blochsimulator.ui.dialogs import PulseImportDialog, SettingsDialog
from blochsimulator.ui.default_settings import WorkspaceDefaults
from blochsimulator.ui.magnetization_viewer import (
    MagnetizationViewer,
    centered_stride_indices,
)
from blochsimulator.visualization import ImageExporter
from blochsimulator.position_sampling import (
    build_line_positions,
    build_plane_positions,
    infer_layout,
)
from blochsimulator.ui.main_window import (
    BlochSimulatorGUI,
    _apply_platform_style,
    _capture_widget_state,
    _restore_widget_state,
)
from blochsimulator.ui.rf_pulse_designer import RFPulseDesigner
from blochsimulator.ui.sequence_designer import SequenceDesigner
from blochsimulator.ui.sequence_diagram import (
    DiagramAnnotations,
    SequenceDiagramWidget,
    TimingMarker,
    find_pulse_segments,
)
from blochsimulator.units import LEGACY_GAMMA_RAD_PER_S_PER_GAUSS


def test_application_style_uses_native_macos_controls():
    app = MagicMock()

    _apply_platform_style(app, platform="darwin")

    app.setStyle.assert_not_called()


def test_application_style_can_force_fusion_on_macos():
    app = MagicMock()

    _apply_platform_style(app, platform="darwin", style_mode="fusion")

    app.setStyle.assert_called_once_with("Fusion")


def test_application_style_can_restore_remembered_system_style():
    app = MagicMock()

    _apply_platform_style(
        app,
        platform="darwin",
        style_mode="system",
        system_style_name="macintosh",
    )

    app.setStyle.assert_called_once_with("macintosh")


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_application_style_retains_fusion_elsewhere(platform):
    app = MagicMock()

    _apply_platform_style(app, platform=platform)

    app.setStyle.assert_called_once_with("Fusion")


def test_log_messages_include_local_timestamp():
    window = MagicMock()
    cursor = window.log_widget.textCursor.return_value

    with patch(
        "blochsimulator.ui.main_window.time.strftime",
        return_value="2026-07-07 14:05:09",
    ):
        BlochSimulatorGUI.log_message(window, "Compiling sequence…")

    window.log_widget.append.assert_called_once_with(
        "[2026-07-07 14:05:09] Compiling sequence…"
    )
    window.log_widget.moveCursor.assert_called_once_with(cursor.End)


@pytest.mark.parametrize(
    ("component", "expected"),
    [
        ("Magnitude", [5.0, 5.0]),
        ("Real", [3.0, -3.0]),
        ("Imaginary", [4.0, 4.0]),
        ("Phase", [np.arctan2(4.0, 3.0), np.arctan2(4.0, -3.0)]),
    ],
)
def test_signal_line_plot_uses_selected_component(component, expected):
    window = MagicMock()
    window.last_result = {"signal": np.array([3.0 + 4.0j, -3.0 + 4.0j])}
    window.last_time = np.array([0.0, 0.001])
    window.signal_component.currentText.return_value = component

    BlochSimulatorGUI._render_signal_lines(window)

    plotted_values = window.signal_plot.plot.call_args.args[1]
    assert np.allclose(plotted_values, expected)
    assert window.signal_plot.plot.call_args.kwargs["name"] == component


@pytest.mark.parametrize("time_axis", [None, np.array([]), np.array([0.0])])
def test_spectrum_data_handles_missing_or_single_sample_time_axis(time_axis):
    window = MagicMock()
    window.last_result = {"signal": np.array([3.0 + 4.0j]), "time": time_axis}
    window.last_time = None
    window.last_positions = None
    window.spectrum_view_mode.currentText.return_value = "Mean over positions"
    window.spectrum_pos_slider.value.return_value = 0
    window._spectrum_fft_len.side_effect = (
        lambda n: BlochSimulatorGUI._spectrum_fft_len(window, n)
    )

    spectrum_data = BlochSimulatorGUI._calculate_spectrum_data(window, time_idx=0)

    assert spectrum_data is not None
    assert np.array_equal(spectrum_data["time_slice"], np.array([0.0]))
    assert spectrum_data["freq"].shape == (512,)
    assert np.all(np.isfinite(spectrum_data["freq"]))


def test_gui_instantiation():
    """Smoke test to ensure the main window can be instantiated without crashing."""
    # Ensure a QApplication exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    try:
        window = BlochSimulatorGUI()
        assert window is not None
        assert window.windowTitle() == "Bloch Equation Simulator"

        # Check if tutorial manager is initialized
        assert hasattr(window, "tutorial_manager")
        assert window.tutorial_manager is not None
        assert window.findChild(QAction, "action_settings") is not None
        assert window.findChild(QAction, "action_project_explorer") is not None
        assert window.time_step_spin.minimum() == pytest.approx(0.01)

    finally:
        # We don't want to show the window or start the event loop
        pass


@pytest.mark.parametrize(
    "pulse_type", ["Adiabatic Half Passage", "Adiabatic Full Passage"]
)
def test_adiabatic_passage_uses_direct_b1_and_ten_ms_default(pulse_type):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    designer = RFPulseDesigner(compact=True)
    designer.duration.setValue(1.0)
    designer.pulse_type.setCurrentText(pulse_type)

    assert designer.duration.value() == pytest.approx(10.0)
    assert not designer.flip_angle.isEnabled()
    assert designer.adiabatic_parameter_hint.isVisibleTo(designer)
    assert "directly" in designer.adiabatic_parameter_hint.text()
    assert np.max(np.abs(designer.get_pulse()[0])) == pytest.approx(0.0)

    designer.b1_amplitude.setValue(0.08)
    pulse_before = designer.get_pulse()[0].copy()
    assert np.max(np.abs(pulse_before)) == pytest.approx(0.08)

    designer.flip_angle.setValue(321.0)
    assert np.allclose(designer.get_pulse()[0], pulse_before)


def test_rf_designer_only_shows_shape_specific_controls():
    app = QApplication.instance() or QApplication(sys.argv)
    designer = RFPulseDesigner(compact=True)

    designer.pulse_type.setCurrentText("SLR")
    assert isinstance(designer.slr_sharpness, QSpinBox)
    assert designer.slr_sharpness.minimum() == 1
    assert not designer.slr_sharpness_container.isHidden()
    assert designer.apodization_container.isHidden()
    assert designer.lobes_container.isHidden()
    designer.set_state({"slr_sharpness": 5.0})
    assert designer.slr_sharpness.value() == 5

    designer.pulse_type.setCurrentText("Sinc")
    assert designer.slr_sharpness_container.isHidden()
    assert not designer.apodization_container.isHidden()
    assert not designer.lobes_container.isHidden()

    designer.close()
    designer.deleteLater()
    app.processEvents()


def test_loaded_slr_auto_b1_matches_flip_angle_after_free_mode_resampling():
    app = QApplication.instance() or QApplication(sys.argv)
    pulse_path = Path(__file__).parents[1] / "rfpulses" / "SLR_sharpness_1.txt"
    b1, time, metadata = load_amp_phase_dat(
        pulse_path,
        duration_s=2.333e-3,
        amplitude_unit="percent",
        phase_unit="deg",
        layout="amp_phase_interleaved",
    )
    designer = RFPulseDesigner(compact=True)
    designer.loaded_pulse_b1 = b1
    designer.loaded_pulse_time = time
    designer.loaded_pulse_metadata = metadata
    designer.pulse_type.setCurrentText("Custom")
    designer.duration.setValue(2.333)
    designer.flip_angle.setValue(90.0)
    designer.b1_amplitude.setValue(0.0)
    designer.update_pulse()

    designed_b1, designed_time = designer.get_pulse()
    designed_dt = float(np.median(np.diff(designed_time)))

    def rotation_deg(waveform, raster_s):
        area_gauss_s = abs(np.sum(waveform) * raster_s)
        return np.rad2deg(LEGACY_GAMMA_RAD_PER_S_PER_GAUSS * area_gauss_s)

    assert designed_b1.size * designed_dt == pytest.approx(2.333e-3)
    assert rotation_deg(designed_b1, designed_dt) == pytest.approx(90.0, abs=0.05)

    resampled_b1 = SequenceDesigner._resample_custom_pulse(designer.get_pulse(), 10e-6)
    assert rotation_deg(resampled_b1, 10e-6) == pytest.approx(90.0, abs=0.05)

    designer.close()
    designer.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("pulse_type", ["Rectangle", "Sinc", "Gaussian"])
def test_designed_pulse_flip_matches_piecewise_constant_simulation(pulse_type):
    app = QApplication.instance() or QApplication(sys.argv)
    designer = RFPulseDesigner(compact=True)
    designer.set_time_step(10e-6)
    designer.pulse_type.setCurrentText(pulse_type)
    designer.duration.setValue(1.0)
    designer.flip_angle.setValue(90.0)
    designer.b1_amplitude.setValue(0.0)
    designer.update_pulse()

    b1, time = designer.get_pulse()
    dt = float(np.median(np.diff(time)))
    flip = np.rad2deg(LEGACY_GAMMA_RAD_PER_S_PER_GAUSS * abs(np.sum(b1) * dt))

    # 1 ms on a 10 us raster is exactly 100 samples, not 101.
    assert b1.size == 100
    # A trapezoid scaling over-flipped a hard pulse by 1/(n-1).
    assert flip == pytest.approx(90.0, abs=0.05)

    designer.close()
    designer.deleteLater()
    app.processEvents()


@pytest.mark.parametrize(
    "build, pulse_index",
    [
        (lambda pulse: SpinEcho(te=10e-3, tr=20e-3, custom_excitation=pulse), 0),
        (
            lambda pulse: SpinEchoTipAxis(te=10e-3, tr=20e-3, custom_excitation=pulse),
            0,
        ),
        (lambda pulse: GradientEcho(te=5e-3, tr=20e-3, custom_excitation=pulse), 0),
        # The generated 180 deg inversion comes first.
        (
            lambda pulse: InversionRecovery(
                ti=20e-3, tr=50e-3, te=5e-3, custom_excitation=pulse
            ),
            1,
        ),
        (lambda pulse: SliceSelectRephase(custom_pulse=pulse), 0),
    ],
    ids=["spin-echo", "spin-echo-tip-axis", "gradient-echo", "ir", "slice-select"],
)
def test_sequence_builders_keep_custom_pulse_flip_on_other_raster(build, pulse_index):
    # 0.995 ms block on a 9.95 us raster: neither duration nor raster matches
    # the 10 us simulation step.
    source_dt = 9.95e-6
    time = (np.arange(100) + 0.5) * source_dt
    area = np.deg2rad(90.0) / LEGACY_GAMMA_RAD_PER_S_PER_GAUSS
    pulse = (np.full(100, area / (100 * source_dt), dtype=complex), time)

    b1, _, sim_time = build(pulse).compile(dt=10e-6)

    segment = find_pulse_segments(b1, sim_time)[pulse_index]
    assert segment.flip_deg == pytest.approx(90.0, abs=0.05)


def test_shared_pulse_loader_dispatches_headerless_txt_import_options():
    pulse_path = Path(__file__).parents[1] / "rfpulses" / "SLR_sharpness_1.txt"
    options = {
        "duration_s": 2.333e-3,
        "amplitude_unit": "percent",
        "phase_unit": "deg",
        "layout": "amp_phase_interleaved",
    }

    shared_b1, shared_time, shared_metadata = load_pulse_from_file(
        pulse_path, **options
    )
    direct_b1, direct_time, direct_metadata = load_amp_phase_dat(pulse_path, **options)

    assert shared_b1 == pytest.approx(direct_b1)
    assert shared_time == pytest.approx(direct_time)
    assert shared_metadata.duration == pytest.approx(direct_metadata.duration)


def test_adiabatic_free_mode_does_not_force_global_ten_ns_time_step():
    window = MagicMock()
    window.workspace_mode = "free"
    window._adiabatic_timestep_warning_shown = False
    window.rf_designer.is_adiabatic_passage.return_value = True
    window.rf_designer.pulse_type.currentText.return_value = "Adiabatic Full Passage"
    window.rf_designer.b1_amplitude.value.return_value = 0.08
    window.time_step_spin.value.return_value = 1.0

    dialog = MagicMock()
    with patch("blochsimulator.ui.main_window.QMessageBox", return_value=dialog) as box:
        proceed = BlochSimulatorGUI._confirm_adiabatic_free_mode_run(window)
        proceed_again = BlochSimulatorGUI._confirm_adiabatic_free_mode_run(window)

    assert proceed
    assert proceed_again
    box.assert_not_called()
    window.time_step_spin.setValue.assert_not_called()


def test_adiabatic_free_mode_skips_prompt_at_recommended_time_step():
    window = MagicMock()
    window.workspace_mode = "free"
    window._adiabatic_timestep_warning_shown = False
    window.rf_designer.is_adiabatic_passage.return_value = True
    window.rf_designer.pulse_type.currentText.return_value = "Adiabatic Half Passage"
    window.rf_designer.b1_amplitude.value.return_value = 0.08
    window.time_step_spin.value.return_value = 0.01

    with patch("blochsimulator.ui.main_window.QMessageBox") as box:
        proceed = BlochSimulatorGUI._confirm_adiabatic_free_mode_run(window)

    assert proceed
    box.assert_not_called()


def test_adiabatic_free_mode_run_requires_direct_b1():
    window = MagicMock()
    window.workspace_mode = "free"
    window.rf_designer.is_adiabatic_passage.return_value = True
    window.rf_designer.pulse_type.currentText.return_value = "Adiabatic Half Passage"
    window.rf_designer.b1_amplitude.value.return_value = 0.0

    with patch("blochsimulator.ui.main_window.QMessageBox.warning") as warning:
        proceed = BlochSimulatorGUI._confirm_adiabatic_free_mode_run(window)

    assert not proceed
    warning.assert_called_once()
    window.rf_designer.b1_amplitude.setFocus.assert_called_once()


def test_fid_extension_preserves_sample_centres_and_exact_tr_duration():
    designer = SequenceDesigner()
    designer.sequence_type.setCurrentText("Free Induction Decay")
    designer.tr_spin.setValue(10.0)
    dt = 1e-6
    pulse_time = (np.arange(1000) + 0.5) * dt
    b1, gradients, time_axis = designer.compile_sequence(
        custom_pulse=(np.ones(1000, complex), pulse_time), dt=dt
    )
    assert len(time_axis) == 10000
    assert time_axis[-1] + dt / 2 == pytest.approx(0.01)
    np.testing.assert_allclose(np.diff(time_axis), dt, atol=1e-15)
    assert np.all(b1[1000:] == 0)
    assert np.all(gradients == 0)


def test_settings_dialog_returns_selected_values(tmp_path):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = SettingsDialog(
        MemoryPolicy(), tmp_path, tooltips_enabled=True, detected_thread_count=8
    )
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData("custom_reserve"))
    dialog.reserve_spin.setValue(3.5)

    policy = dialog.get_policy()
    assert policy.mode == "custom_reserve"
    assert policy.reserve_bytes == int(3.5 * GIB)
    assert dialog.reserve_spin.isEnabled()
    assert dialog.memory_reserve_control_label.isEnabled()
    assert not dialog.limit_spin.isEnabled()
    assert not dialog.memory_limit_control_label.isEnabled()
    dialog.animation_memory_budget_spin.setValue(2048.0)
    assert dialog.animation_memory_budget_bytes() == 2048 * 1024**2
    assert dialog.get_export_directory() == tmp_path
    assert dialog.tooltips_enabled()
    assert dialog.interface_style() == "automatic"
    dialog.interface_style_combo.setCurrentIndex(
        dialog.interface_style_combo.findData("fusion")
    )
    assert dialog.interface_style() == "fusion"
    assert dialog.sequence_live_progress_enabled()
    assert dialog.sequence_kernel() == "optimized"
    assert dialog.sequence_kernel_combo.findData("reference") == -1
    dialog.sequence_kernel_combo.setCurrentIndex(
        dialog.sequence_kernel_combo.findData("strict_reference")
    )
    assert dialog.sequence_kernel() == "strict_reference"
    assert dialog.dynamic_sequence_kernel() == "native_parallel"
    for kernel in ("native_serial", "reference", "metal_hybrid"):
        assert dialog.dynamic_sequence_kernel_combo.findData(kernel) == -1
    dialog.dynamic_sequence_kernel_combo.setCurrentIndex(
        dialog.dynamic_sequence_kernel_combo.findData("optimized")
    )
    assert dialog.dynamic_sequence_kernel() == "optimized"
    assert not any(spin.isEnabled() for spin in dialog.subvoxel_spin_count_spins)
    dialog.dynamic_sequence_kernel_combo.setCurrentIndex(
        dialog.dynamic_sequence_kernel_combo.findData("native_parallel")
    )
    assert dialog.dynamic_sequence_kernel() == "native_parallel"
    assert dialog.sequence_timestep_preset() == "balanced"
    assert dialog.sequence_timestep_us() == pytest.approx(5.0)
    assert not dialog.sequence_timestep_us_spin.isEnabled()
    assert not dialog.sequence_timestep_control_label.isEnabled()
    dialog.sequence_timestep_preset_combo.setCurrentIndex(
        dialog.sequence_timestep_preset_combo.findData("custom")
    )
    dialog.sequence_timestep_us_spin.setValue(7.5)
    assert dialog.sequence_timestep_us() == pytest.approx(7.5)
    assert dialog.sequence_timestep_us_spin.isEnabled()
    assert dialog.sequence_timestep_control_label.isEnabled()
    assert dialog.sequence_spoiler_mode() == "ideal"
    assert dialog.subvoxel_spin_counts() == (1, 1, 1)
    assert dialog.subvoxel_sampling_method() == "midpoint"
    assert not any(spin.isEnabled() for spin in dialog.subvoxel_spin_count_spins)
    assert not dialog.subvoxel_sampling_method_combo.isEnabled()
    assert not any(field.isEnabled() for field in dialog.subvoxel_control_fields)
    assert not any(label.isEnabled() for label in dialog.subvoxel_control_labels)
    dialog.sequence_spoiler_mode_combo.setCurrentIndex(
        dialog.sequence_spoiler_mode_combo.findData("gradient")
    )
    for spin, value in zip(dialog.subvoxel_spin_count_spins, (3, 5, 11)):
        spin.setValue(value)
    assert dialog.sequence_spoiler_mode() == "gradient"
    assert dialog.subvoxel_spin_counts() == (3, 5, 11)
    dialog.subvoxel_sampling_method_combo.setCurrentIndex(
        dialog.subvoxel_sampling_method_combo.findData("stratified")
    )
    assert dialog.subvoxel_sampling_method() == "stratified"
    assert all(spin.isEnabled() for spin in dialog.subvoxel_spin_count_spins)
    assert dialog.subvoxel_sampling_method_combo.isEnabled()
    assert all(field.isEnabled() for field in dialog.subvoxel_control_fields)
    assert all(label.isEnabled() for label in dialog.subvoxel_control_labels)
    assert dialog.thread_mode() == "automatic"
    assert not dialog.manual_thread_count_spin.isEnabled()
    assert not dialog.manual_thread_count_control_label.isEnabled()
    dialog.thread_mode_combo.setCurrentIndex(
        dialog.thread_mode_combo.findData("manual")
    )
    dialog.manual_thread_count_spin.setValue(3)
    assert dialog.thread_mode() == "manual"
    assert dialog.manual_thread_count() == 3
    assert dialog.manual_thread_count_spin.isEnabled()
    assert dialog.manual_thread_count_control_label.isEnabled()
    dialog.scanner_max_grad_spin.setValue(40.0)
    dialog.scanner_max_slew_spin.setValue(180.0)
    scanner = dialog.scanner_parameters()
    assert scanner.max_grad_mtm == pytest.approx(40.0)
    assert scanner.max_slew_tms == pytest.approx(180.0)
    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
        "General",
        "Defaults",
        "Simulation",
        "Scanner",
        "Memory",
        "Interface",
    ]


def test_settings_dialog_prefills_and_labels_loaded_pulseq_recommendations(tmp_path):
    app = QApplication.instance() or QApplication(sys.argv)
    recommendation = {
        "sequence_name": "custom.seq",
        "spoiler_mode": "gradient",
        "sampling_method": "midpoint",
        "counts_xyz": (2, 3, 11),
        "detail": "Train-wide recommendation for custom.seq.",
    }

    dialog = SettingsDialog(
        MemoryPolicy(),
        tmp_path,
        tooltips_enabled=True,
        sequence_spoiler_mode="ideal",
        subvoxel_spin_counts=(1, 1, 1),
        subvoxel_sampling_method="stratified",
        loaded_pulseq_recommendation=recommendation,
    )

    assert dialog.sequence_spoiler_mode() == "gradient"
    assert dialog.subvoxel_sampling_method() == "midpoint"
    assert dialog.subvoxel_spin_counts() == (2, 3, 11)
    assert dialog.sequence_spoiler_recommendation_label.text() == (
        "Loaded .seq: Gradient waveform"
    )
    assert dialog.subvoxel_sampling_recommendation_label.text() == (
        "Loaded .seq: Regular midpoint grid"
    )
    assert [label.text() for label in dialog.subvoxel_spin_recommendation_labels] == [
        "Loaded .seq: 2 spins/voxel",
        "Loaded .seq: 3 spins/voxel",
        "Loaded .seq: 11 spins/voxel",
    ]
    assert all(
        label.toolTip() == recommendation["detail"]
        for label in dialog.subvoxel_spin_recommendation_labels
    )
    assert all(spin.isEnabled() for spin in dialog.subvoxel_spin_count_spins)

    dialog.close()
    dialog.deleteLater()
    app.processEvents()


def test_settings_dialog_groups_kernel_extras_and_has_english_tooltips(tmp_path):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = SettingsDialog(MemoryPolicy(), tmp_path, tooltips_enabled=True)

    def offered(combo):
        return tuple(
            combo.itemData(index)
            for index in range(combo.count())
            if combo.itemData(index) is not None
        )

    sequence = dialog.sequence_kernel_combo
    assert offered(sequence) == GUI_KERNEL_NAMES["sequence"]
    sequence_optimized = sequence.findData("optimized")
    sequence_strict_reference = sequence.findData("strict_reference")
    assert sequence.itemText(sequence_optimized) == "Optimized (recommended)"
    assert (
        sequence.itemText(sequence_strict_reference) == "Strict reference (validation)"
    )
    assert sequence_strict_reference > sequence_optimized + 1
    assert "ordinary Bloch simulations" in sequence.itemData(
        sequence_optimized, Qt.ToolTipRole
    )
    assert "strict numerical baseline" in sequence.itemData(
        sequence_strict_reference, Qt.ToolTipRole
    )

    dynamic = dialog.dynamic_sequence_kernel_combo
    assert offered(dynamic) == GUI_KERNEL_NAMES["dynamic"]
    expected_labels = {
        "native_parallel": "Native automatic (recommended)",
        "optimized": "Optimized NumPy (compatibility and validation)",
    }
    indices = {kernel: dynamic.findData(kernel) for kernel in expected_labels}
    assert indices["optimized"] > indices["native_parallel"] + 1
    for kernel, label in expected_labels.items():
        index = indices[kernel]
        assert dynamic.itemText(index) == label
        assert dynamic.itemData(index, Qt.ToolTipRole)

    assert "multiple CPU workers" in dynamic.itemData(
        indices["native_parallel"], Qt.ToolTipRole
    )
    assert "automatic fallback" in dynamic.itemData(
        indices["optimized"], Qt.ToolTipRole
    )


def test_memory_limit_uses_warning_instead_of_generic_error():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    window._show_memory_limit_warning = MagicMock()
    message = "Memory limit exceeded: test details"

    with patch("blochsimulator.ui.main_window.QMessageBox.critical") as critical:
        window.on_simulation_error(message)

    window._show_memory_limit_warning.assert_called_once_with(message)
    critical.assert_not_called()


def test_sequence_diagram_draws_slice_gradient_in_gz_lane_after_error_view():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    designer = window.sequence_designer
    diagram = designer.diagram_widget
    diagram.show_error("broken")
    assert "broken" in diagram.error_text()
    time = np.linspace(0, 1e-3, 11)
    b1 = np.zeros(time.size, complex)
    b1[1:4] = 0.05
    gradients = np.zeros((time.size, 3))
    gradients[:5, 2] = 1.0

    designer._render_sequence_diagram(b1, gradients, time)

    assert diagram.error_text() == ""
    _, gz = diagram.lane_data("gz")
    _, gx = diagram.lane_data("gx")
    assert np.ptp(gz) == pytest.approx(10.0)  # 1 G/cm = 10 mT/m
    assert np.ptp(gx) == 0
    assert diagram.collapsed == {
        "rf": False,
        "phase": False,
        "gz": False,
        "gy": True,
        "gx": True,
    }
    y_min, y_max = diagram.plots["gz"].getViewBox().viewRange()[1]
    assert y_min < -10 and y_max > 10
    assert diagram.time_axis_plot.getAxis("bottom").labelText == "Time"


def test_sequence_diagram_annotates_pulses_and_scales_gradients_together():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    diagram = SequenceDiagramWidget()
    diagram.resize(400, 600)
    diagram.show()
    dt = 1e-5
    time = np.arange(1000) * dt
    b1 = np.zeros(time.size, complex)
    # 90 deg about x, then 180 deg about y at TE/2 = 4 ms.
    area_90 = 0.25 / 4257.7
    b1[10:110] = area_90 / (100 * dt)
    b1[400:500] = 1j * 2 * area_90 / (100 * dt)
    gradients = np.zeros((time.size, 3))
    gradients[10:110, 2] = 1.0
    gradients[120:170, 2] = -1.0
    gradients[600:900, 0] = 0.05

    diagram.set_sequence(
        b1,
        gradients,
        time,
        DiagramAnnotations(
            brackets=[TimingMarker("TE 8 ms", 0.6e-3, 8.6e-3)],
            first_tr_window_s=(0.0, 5e-3),
        ),
    )
    app.processEvents()

    assert diagram.visible_pulse_labels() == ["90°x", "180°y"]
    assert diagram.gradient_zoomed["gx"] and not diagram.gradient_zoomed["gz"]
    assert diagram.collapsed["gy"] and not diagram.collapsed["gx"]
    lo, hi = diagram.visible_time_range_ms()
    assert lo < 0 and hi > 9.99
    diagram.show_first_tr()
    lo, hi = diagram.visible_time_range_ms()
    assert lo < 0 and 5 < hi < 5.5
    diagram.close()


def test_sequence_diagram_click_moves_playback_to_that_time():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    designer = window.sequence_designer
    diagram = designer.diagram_widget
    time = np.linspace(0, 10e-3, 101)
    b1 = np.zeros(time.size, complex)
    b1[1:5] = 0.05
    designer._render_sequence_diagram(b1, np.zeros((time.size, 3)), time)
    diagram.resize(400, 600)
    diagram.show()
    app.processEvents()

    view = diagram.plots["rf"].getViewBox()
    event = MagicMock()
    event.button.return_value = Qt.LeftButton
    event.double.return_value = False
    event.scenePos.return_value = view.mapViewToScene(QPointF(4.0, 0.5))
    window.playback_time = 1.0 + time
    window.last_time = 1.0 + time
    with patch.object(window, "_on_3d_vector_position_changed") as seek:
        diagram._on_scene_clicked(event)

    assert diagram.playhead.value() == pytest.approx(4.0, abs=0.1)
    seek.assert_called_once_with(40)
    diagram.close()


def test_slice_axis_moves_the_slice_gradient_and_defaults_change_nothing():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    designer = window.sequence_designer
    designer.sequence_type.setCurrentText("Slice Select + Rephase")
    dt = designer.default_dt
    _, base_gradients, base_time = designer._compile_base_sequence(None, dt, False)

    assert designer.slice_axis_combo.currentText() == "z"
    assert designer.gradient_lobes() == []
    _, gradients, time = designer.compile_sequence(dt=dt)
    np.testing.assert_array_equal(time, base_time)
    np.testing.assert_array_equal(gradients, base_gradients)
    assert np.ptp(base_gradients[:, 2]) > 0

    designer.slice_axis_combo.setCurrentText("x")
    _, gradients, _ = designer.compile_sequence(dt=dt)
    np.testing.assert_array_equal(gradients[:, 0], base_gradients[:, 2])
    assert np.all(gradients[:, 2] == 0.0)
    _, gx = designer.diagram_widget.lane_data("gx")
    _, gz = designer.diagram_widget.lane_data("gz")
    assert np.ptp(gx) > 0 and np.ptp(gz) == 0


def test_gradient_lobe_table_extends_the_sequence_and_reports_its_moment():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    designer = window.sequence_designer
    designer.sequence_type.setCurrentText("Free Induction Decay")
    dt = designer.default_dt
    _, _, base_time = designer.compile_sequence(dt=dt)
    base_ms = (base_time[-1] - base_time[0] + dt) * 1e3

    lobe = GradientLobe("y", base_ms + 1.0, 2.0, 5.0, 0.2)
    designer.set_gradient_lobes([lobe])
    b1, gradients, time = designer.compile_sequence(dt=dt)

    assert time[-1] - time[0] + dt == pytest.approx((base_ms + 3.0) * 1e-3, abs=dt)
    assert np.all(b1[base_time.size :] == 0.0)
    moments = gradient_moments_mt_ms_per_m(gradients, time)
    assert moments[1] == pytest.approx(9.0)
    label = designer.gradient_moment_label.text()
    assert "y 9" in label and "Phase twist (cycles/mm): y 0.383" in label

    table = designer.gradient_lobe_table
    designer.add_gradient_lobe_button.click()
    assert table.rowCount() == 2
    added = designer.gradient_lobes()[1]
    assert added.axis == "x" and added.start_ms == pytest.approx(lobe.end_ms)

    table.item(1, 3).setData(Qt.EditRole, -20.0)
    assert designer.gradient_lobes()[1].amplitude_mt_per_m == -20.0
    assert "x -18" in designer.gradient_moment_label.text()

    table.selectRow(0)
    designer.remove_gradient_lobe_button.click()
    assert designer.gradient_lobes() == [replace(added, amplitude_mt_per_m=-20.0)]


def test_gradient_options_round_trip_and_legacy_states_reset():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    designer = window.sequence_designer
    lobes = [GradientLobe("z", 0.5, 1.0, -8.0, 0.1), GradientLobe("x", 2.0)]
    designer.slice_axis_combo.setCurrentText("y")
    designer.set_gradient_lobes(lobes)
    state = designer.get_state()

    restored = BlochSimulatorGUI().sequence_designer
    restored.set_state(state)
    assert restored.slice_axis_combo.currentText() == "y"
    assert restored.gradient_lobes() == lobes

    legacy = {
        key: value
        for key, value in state.items()
        if key not in ("slice_axis", "gradient_lobes")
    }
    restored.set_state(legacy)
    assert restored.slice_axis_combo.currentText() == "z"
    assert restored.gradient_lobes() == []


def test_free_mode_run_simulates_gradient_lobes_on_an_x_line():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    designer = window.sequence_designer
    designer.sequence_type.setCurrentText("Free Induction Decay")
    window.pos_axis_combo.setCurrentText("x")
    window.pos_spin.setValue(5)
    lobe = GradientLobe("x", 50.0, 2.0, 10.0, 0.25)
    designer.set_gradient_lobes([lobe])

    with patch("blochsimulator.ui.main_window.SimulationThread") as thread:
        window.run_simulation()

    (_, (b1, gradients, time), _, positions, *_), _ = thread.call_args
    assert positions.shape == (5, 3) and np.ptp(positions[:, 0]) > 0
    moments = gradient_moments_mt_ms_per_m(gradients, time)
    assert moments[0] == pytest.approx(lobe.moment_mt_ms_per_m)
    np.testing.assert_array_equal(window.last_gradients, gradients)


def test_simulation_controls_use_non_overlapping_grid_rows():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    layout = window.simulation_controls_layout
    assert isinstance(layout, QGridLayout)

    def position(widget):
        return layout.getItemPosition(layout.indexOf(widget))

    assert position(window.pos_geometry_combo) == (1, 1, 1, 1)
    # Axis (1D) and plane (2D) share one cell; only one is visible.
    assert position(window.pos_axis_combo) == (2, 1, 1, 1)
    assert position(window.pos_plane_combo) == (2, 1, 1, 1)
    assert position(window.pos_spin) == (3, 1, 1, 1)
    assert position(window.pos_range) == (4, 1, 1, 1)
    assert position(window.pos2_spin) == (5, 1, 1, 1)
    assert position(window.pos2_range) == (6, 1, 1, 1)
    assert position(window.pos_center_widget) == (7, 1, 1, 1)
    assert position(window.position_summary_label) == (8, 0, 1, 2)
    assert position(window.freq_spin) == (9, 1, 1, 1)
    assert position(window.freq_center) == (10, 1, 1, 1)
    assert position(window.freq_range) == (11, 1, 1, 1)
    assert position(window.freq_axis_mode) == (12, 1, 1, 1)


def test_free_mode_and_slice_explorer_spatial_controls_use_mm():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    window.pos_spin.setValue(3)
    window.pos_range.setValue(20.0)
    window._synchronize_workspace_nucleus("C13")

    assert window.pos_range.suffix() == " mm"
    assert window._collect_simulation_parameters(internal_format=True)[
        "pos_range_mm"
    ] == pytest.approx(20.0)
    assert (
        window._collect_simulation_parameters(internal_format=True)["nucleus"] == "C13"
    )
    parameters = window._collect_simulation_parameters()
    assert parameters["nucleus"] == "C13"
    assert parameters["position_range_mm"] == pytest.approx(20.0)
    assert parameters["position_range_cm"] == pytest.approx(2.0)
    assert parameters["position_axis"][:, 2] == pytest.approx([-0.01, 0.0, 0.01])

    explorer = window.slice_explorer
    assert explorer.pos_range.suffix() == " mm"
    assert explorer.pos_range.value() == pytest.approx(40.0)
    assert explorer.plot_profile.getAxis("bottom").labelText == "Position (mm)"
    explorer.num_points.setValue(51)
    explorer.run_simulation()
    profile_x, _ = explorer.plot_profile.listDataItems()[0].getData()
    assert profile_x[[0, -1]] == pytest.approx([-20.0, 20.0])


def test_free_mode_export_uses_parameters_captured_for_completed_run(tmp_path):
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    captured = {
        "tissue": {"field_strength_t": 7.0, "t1_ms": 1234.0},
        "sequence": {"sequence_type": "Spin Echo", "te_s": 0.02},
        "simulation": {
            "mode": "endpoint",
            "time_step_us": 4.0,
            "num_positions": 3,
            "field_strength_t": 7.0,
        },
    }
    window._last_export_parameters = captured
    window.last_result = {"time": np.array([0.0]), "mx": np.array([[0.0]])}
    window.last_time = np.array([0.0])
    window.pos_spin.setValue(99)
    window.time_step_spin.setValue(25.0)
    window.tissue_widget.set_field_strength(1.5)

    dialog = MagicMock()
    dialog.exec_.return_value = QDialog.Accepted
    dialog.get_export_options.return_value = {
        "base_path": str(tmp_path / "run"),
        "image": False,
        "image_format": "png",
        "animation": False,
        "animation_format": "gif",
        "animation_fps": 30,
        "include_sequence": False,
        "hdf5": True,
        "notebook_analysis": False,
        "notebook_repro": False,
        "csv": False,
        "csv_format": "csv",
    }
    with (
        patch("blochsimulator.ui.main_window.ExportDataDialog", return_value=dialog),
        patch.object(window.simulator, "save_results") as save_results,
        patch("blochsimulator.ui.main_window.QMessageBox.information"),
    ):
        window.export_results()

    assert save_results.call_args.args[1] is captured["sequence"]
    assert save_results.call_args.args[2] is captured["simulation"]
    assert save_results.call_args.args[2]["num_positions"] == 3
    assert save_results.call_args.args[2]["field_strength_t"] == 7.0


def test_file_export_in_sequence_mode_opens_sequence_export(tmp_path):
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    window.set_workspace_mode("sequence")
    window._ensure_sequence_simulation_workspace(window.sequence_simulation_tab_index)
    window.sequence_simulation_widget.result = object()
    export_action = window.findChild(QAction, "action_export_results")

    with (
        patch("blochsimulator.ui.main_window.ExportDataDialog") as free_dialog,
        patch.object(
            window.sequence_simulation_widget, "export_results"
        ) as sequence_export,
    ):
        export_action.trigger()

    sequence_export.assert_called_once_with()
    free_dialog.assert_not_called()


def test_file_export_in_sequence_mode_without_result_warns(tmp_path):
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    window.set_workspace_mode("sequence")
    window.sequence_simulation_widget = None

    with (
        patch("blochsimulator.ui.main_window.ExportDataDialog") as free_dialog,
        patch("blochsimulator.ui.main_window.QMessageBox.warning") as warning,
    ):
        window.export_results()

    free_dialog.assert_not_called()
    assert "sequence simulation" in warning.call_args.args[2]


def test_free_mode_export_after_released_result_warns(tmp_path):
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    window.set_workspace_mode("free")
    window.last_result = None
    window.last_b1 = None

    with (
        patch("blochsimulator.ui.main_window.ExportDataDialog") as free_dialog,
        patch("blochsimulator.ui.main_window.QMessageBox.warning") as warning,
    ):
        window.export_results()

    free_dialog.assert_not_called()
    warning.assert_called_once()


def test_free_mode_analysis_notebook_writes_its_hdf5(tmp_path):
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    window.set_workspace_mode("free")
    window._last_export_parameters = {
        "tissue": {},
        "sequence": {"sequence_type": "Spin Echo"},
        "simulation": {"mode": "endpoint"},
    }
    window.last_result = {"time": np.array([0.0]), "mx": np.array([[0.0]])}
    window.last_time = np.array([0.0])
    dialog = MagicMock()
    dialog.exec_.return_value = QDialog.Accepted
    dialog.get_export_options.return_value = {
        "base_path": str(tmp_path / "run"),
        "image": False,
        "image_format": "png",
        "animation": False,
        "animation_format": "gif",
        "animation_fps": 30,
        "include_sequence": False,
        "hdf5": False,
        "notebook_analysis": True,
        "notebook_repro": False,
        "csv": False,
        "csv_format": "csv",
    }
    with (
        patch("blochsimulator.ui.main_window.ExportDataDialog", return_value=dialog),
        patch.object(window.simulator, "save_results") as save_results,
        patch("blochsimulator.notebook_exporter.export_notebook") as export_notebook,
        patch("blochsimulator.ui.main_window.QMessageBox.information"),
    ):
        window.export_results()

    assert save_results.call_args.args[0] == str(tmp_path / "run.h5")
    assert export_notebook.call_args.kwargs["h5_filename"] == "run.h5"


def test_free_mode_defaults_to_a_z_line_and_hides_plane_controls():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()

    assert window.pos_geometry_combo.currentText() == "1D line"
    assert window.pos_axis_combo.currentText() == "z"
    assert window.pos_plane_combo.isHidden()
    assert window.pos2_spin.isHidden() and window.pos2_range.isHidden()
    assert window.pos_count_label.text() == "Positions along z:"
    window.pos_spin.setValue(3)
    np.testing.assert_allclose(
        window._build_position_grid(), [[0, 0, -0.01], [0, 0, 0], [0, 0, 0.01]]
    )
    assert window.position_summary_label.text().startswith(
        "3 positions (3 along z) × 31 frequencies = 93 spins"
    )


def test_free_mode_line_along_x_with_center_offset():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.pos_axis_combo.setCurrentText("x")
    window.pos_spin.setValue(3)
    window.pos_range.setValue(10.0)
    window.pos_center_z.setValue(2.0)

    assert window.pos_range_label.text() == "Range along x:"
    np.testing.assert_allclose(
        window._build_position_grid(),
        [[-0.005, 0, 0.002], [0, 0, 0.002], [0.005, 0, 0.002]],
    )
    parameters = window._collect_simulation_parameters()
    assert parameters["position_geometry"] == "line"
    assert parameters["position_axes"] == "x"
    assert parameters["position_center_mm"] == [0.0, 0.0, 2.0]


def test_switching_free_mode_to_a_plane_applies_the_starting_grid():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    index = window.pos_geometry_combo.findText("2D plane")
    window.pos_geometry_combo.setCurrentIndex(index)
    window.pos_geometry_combo.activated.emit(index)

    assert window.pos_plane_combo.currentText() == "xy"
    assert not window.pos_plane_combo.isHidden()
    assert window.pos_axis_combo.isHidden()
    assert not window.pos2_spin.isHidden()
    assert window.pos_axis_label.text() == "Plane:"
    assert window.pos_count_label.text() == "Positions along x:"
    assert window.pos2_count_label.text() == "Positions along y:"
    assert (window.pos_spin.value(), window.pos2_spin.value()) == (12, 12)
    assert window.freq_spin.value() == 1
    # The 3D view follows with one origin per position and no tip clutter.
    viewer = window.mag_3d
    assert viewer.get_vector_arrangement() == viewer.ARRANGEMENT_ALONG_POSITION
    assert not viewer.tip_markers_checkbox.isChecked()
    assert not viewer.track_checkbox.isChecked()
    assert window.position_summary_label.text().startswith(
        "144 positions (12 × 12 in xy) × 1 frequency = 144 spins"
    )

    positions = window._build_position_grid()
    assert positions.shape == (144, 3)
    np.testing.assert_allclose(positions[:12, 1], -0.01)
    np.testing.assert_allclose(positions[:12, 0], np.linspace(-0.01, 0.01, 12))
    assert np.all(positions[:, 2] == 0.0)
    parameters = window._collect_simulation_parameters()
    assert parameters["num_positions"] == 144
    assert parameters["position_geometry"] == "plane"
    assert parameters["position_axes"] == "xy"
    assert parameters["position_counts"] == [12, 12]


def test_plane_preview_strides_each_axis_and_keeps_the_grid():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.pos_geometry_combo.setCurrentText("2D plane")
    window.pos_plane_combo.setCurrentText("yz")
    window.pos_spin.setValue(20)
    window.pos2_spin.setValue(5)

    positions = window._build_position_grid()
    preview = window._preview_positions(positions, [20, 5])
    layout = infer_layout(preview)

    assert layout.plane_name == "yz"
    assert layout.counts == (7, 5)
    np.testing.assert_allclose(layout.coordinates[0], positions[:20:3, 1])


def test_plane_keeps_its_grid_when_sequence_presets_load():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.pos_geometry_combo.setCurrentText("2D plane")
    window.pos_spin.setValue(12)
    window.pos2_spin.setValue(12)

    window.sequence_designer.sequence_type.setCurrentText("Slice Select + Rephase")

    assert (window.pos_spin.value(), window.pos2_spin.value()) == (12, 12)


def test_position_geometry_round_trips_through_project_controls():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.pos_geometry_combo.setCurrentText("2D plane")
    window.pos_plane_combo.setCurrentText("xz")
    window.pos_spin.setValue(4)
    window.pos2_spin.setValue(6)
    window.pos_center_y.setValue(-3.0)
    state = _capture_widget_state(window)

    restored = BlochSimulatorGUI()
    _restore_widget_state(restored, state)

    assert restored.pos_geometry_combo.currentText() == "2D plane"
    assert restored.pos_plane_combo.currentText() == "xz"
    assert (restored.pos_spin.value(), restored.pos2_spin.value()) == (4, 6)
    np.testing.assert_allclose(
        restored._build_position_grid(), window._build_position_grid()
    )
    assert not restored.pos2_spin.isHidden()


def test_projects_without_position_geometry_open_as_a_z_line(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.pos_spin.setValue(5)
    path = tmp_path / "current.blochproj"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    window.save_project()
    project = load_project(path)
    controls = project["state"]["main_controls"]
    for name in (
        "pos_geometry_combo",
        "pos_axis_combo",
        "pos_plane_combo",
        "pos2_spin",
        "pos2_range",
        "pos_center_x",
        "pos_center_y",
        "pos_center_z",
    ):
        controls.pop(name, None)
    legacy_path = tmp_path / "legacy.blochproj"
    save_project(
        legacy_path,
        project["state"],
        project["phantom"],
        project["tx_field"],
        project["rx_field"],
        project["program"],
        project["legacy_result"],
        project["sequence_result"],
    )

    reopened = BlochSimulatorGUI()
    reopened.pos_geometry_combo.setCurrentText("2D plane")
    reopened.pos_center_x.setValue(4.0)
    reopened.open_project(str(legacy_path))

    assert reopened.pos_geometry_combo.currentText() == "1D line"
    assert reopened.pos_axis_combo.currentText() == "z"
    positions = reopened._build_position_grid()
    assert positions.shape == (5, 3)
    assert np.all(positions[:, :2] == 0.0)


def test_free_mode_position_labels_follow_the_simulated_axis():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.last_positions = build_plane_positions((3, 2), (0.02, 0.01), "xy")

    assert window._position_label(4) == "(x, y) = (0.00, 5.00) mm"
    assert window._center_position_index() in (1, 4)
    label, values = window._position_plot_axis()
    assert label == "Position index (x fastest)" and values is None

    window.last_positions = build_line_positions(3, 0.02, "y")
    label, values = window._position_plot_axis()
    assert label == "Position y (mm)"
    np.testing.assert_allclose(values, [-10.0, 0.0, 10.0])
    assert window._position_label(2) == "y = 10.00 mm"


def test_simulation_controls_have_explanatory_tooltips():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    controls = (
        window.mode_combo,
        window.pos_spin,
        window.pos_range,
        window.freq_spin,
        window.freq_center,
        window.freq_range,
        window.freq_axis_mode,
        window.time_step_spin,
        window.max_traces_spin,
    )

    registered = dict(window._tooltip_registry)
    assert all(registered.get(control, "").strip() for control in controls)


def test_all_main_window_fields_have_tooltips():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    field_types = (
        QAbstractSpinBox,
        QCheckBox,
        QComboBox,
        QRadioButton,
        QSlider,
    )
    registered = {field for field, _ in window._tooltip_registry}
    missing = [
        field for field in window.findChildren(field_types) if field not in registered
    ]

    assert not missing


def test_free_mode_result_controls_use_consistent_headers_and_rows():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()

    assert window.tab_widget.tabBar().font().bold()
    assert window.workspace_header.isHidden()
    assert window.workspace_switch.parentWidget() is window.free_mode_colormap_controls

    vector_layout = window.mag_3d.layout()
    assert vector_layout.indexOf(
        window.mag_3d.control_container
    ) < vector_layout.indexOf(window.mag_3d.gl_widget)
    assert window.mag_3d.vector_palette_combo.findText("Viridis") >= 0
    assert [
        window.mag_3d.spin_display_combo.itemText(index)
        for index in range(window.mag_3d.spin_display_combo.count())
    ] == ["Show all spins", "Show all spins + mean", "Show only mean"]
    assert window.mag_3d.track_checkbox.isChecked()
    assert window.mag_3d.spin_display_combo.currentText() == "Show all spins + mean"
    rainbow = window.mag_3d.color_for_index(0, 5).name()
    window.mag_3d.vector_palette_combo.setCurrentText("Viridis")
    assert window.mag_3d.color_for_index(0, 5).name() != rainbow
    window.mag_3d.vector_palette_combo.setCurrentText("Gray")
    grays = [window.mag_3d.color_for_index(i, 5) for i in range(5)]
    assert all(c.red() == c.green() == c.blue() for c in grays)
    assert grays[0].red() < grays[-1].red() == 255
    window.mag_3d.vector_palette_combo.setCurrentText("White")
    assert {window.mag_3d.color_for_index(i, 5).name() for i in range(5)} == {"#ffffff"}

    assert window.spectrum_controls_layout.indexOf(window.spectrum_component_combo) >= 0
    for control in (
        window.mean_only_checkbox,
        window.spatial_markers_checkbox,
        window.spatial_component_combo,
    ):
        assert window.spatial_options_layout.indexOf(control) >= 0

    assert window.mag_3d.page_title.isHidden()
    assert window.mag_3d.header_container.isHidden()
    assert window.mag_component.width() == window.signal_component.width() == 220
    assert window.rf_designer_tab.page_title.isHidden()
    assert window.slice_explorer.page_title.isHidden()
    assert window.param_sweep_widget.page_title.isHidden()
    assert window.rf_designer_tab.control_panel.minimumWidth() == 400
    assert window.slice_explorer.control_panel.minimumWidth() == 400
    assert window.rf_designer_tab.control_panel.maximumWidth() == 400
    assert window.slice_explorer.control_panel.maximumWidth() == 400


@pytest.mark.parametrize(
    ("display_mode", "show_spins", "show_mean"),
    [
        ("Show all spins", True, False),
        ("Show all spins + mean", True, True),
        ("Show only mean", False, True),
    ],
)
def test_3d_spin_display_mode_does_not_change_tip_path_tracking(
    display_mode, show_spins, show_mean
):
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = MagnetizationViewer()
    viewer.vector_plot = MagicMock()
    viewer.mean_vector = MagicMock()
    viewer.spin_display_combo.setCurrentText(display_mode)
    viewer.vector_plot.reset_mock()
    viewer.mean_vector.reset_mock()
    viewer._clear_path()

    viewer.update_magnetization(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    )

    viewer.vector_plot.setVisible.assert_called_with(show_spins)
    viewer.mean_vector.setVisible.assert_called_with(show_mean)
    assert viewer.track_checkbox.isChecked()
    assert len(viewer.path_points) == 1


def _spatial_viewer(positions_mm, spins_per_position):
    viewer = MagnetizationViewer()
    viewer.set_spatial_layout_available(True)
    viewer.arrangement_combo.setCurrentText(viewer.ARRANGEMENT_ALONG_POSITION)
    viewer.set_spatial_groups(positions_mm, spins_per_position)
    return viewer


def test_3d_spatial_layout_draws_each_position_at_its_own_origin():
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = _spatial_viewer([-10.0, 0.0, 10.0], 2)
    viewer.spin_display_combo.setCurrentText("Show all spins + mean")
    spins = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        ]
    )
    means = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [-0.5, -0.5, 0.0]])
    net = np.array([0.0, 0.0, 0.25])

    assert not viewer.tip_curve_checkbox.isChecked()
    assert not viewer.net_vector_checkbox.isChecked()
    viewer.update_magnetization(spins.reshape(-1, 3), position_means=means, net=net)

    # Positions are stacked along the display z axis.
    spacing = 1.0 / viewer.vector_length_spin.value()
    origins = np.array([[0, 0, -spacing], [0, 0, 0], [0, 0, spacing]])
    np.testing.assert_allclose(viewer.vector_plot.pos[0::2], np.repeat(origins, 2, 0))
    np.testing.assert_allclose(
        viewer.vector_plot.pos[1::2], np.repeat(origins, 2, 0) + spins.reshape(-1, 3)
    )
    np.testing.assert_allclose(viewer.position_mean_plot.pos[1::2], origins + means)
    assert viewer.vector_plot.visible()
    assert viewer.position_mean_plot.visible()
    assert not viewer.tip_curve_item.visible()
    # The net vector and its path stay hidden until requested.
    assert not viewer.mean_vector.visible()
    assert not viewer.path_item.visible()
    assert viewer.path_points == []

    viewer.net_vector_checkbox.setChecked(True)
    viewer.tip_curve_checkbox.setChecked(True)
    np.testing.assert_allclose(viewer.tip_curve_item.pos, origins + means)
    viewer.update_magnetization(spins.reshape(-1, 3), position_means=means, net=net)
    np.testing.assert_allclose(viewer.mean_vector.pos, [[0, 0, 0], net])
    transform = viewer.mean_vector.transform()
    assert transform.column(3).x() == pytest.approx(-viewer.FIELD_GAP - viewer.NET_GAP)
    # The net vector is drawn enlarged.
    assert transform.column(0).x() == pytest.approx(viewer.NET_VECTOR_SCALE)
    assert viewer.mean_vector.visible()
    assert viewer.tip_curve_item.visible()
    assert len(viewer.path_points) == 1

    viewer.spin_display_combo.setCurrentText("Show only mean")
    assert not viewer.vector_plot.visible()
    assert viewer.position_mean_plot.visible()
    viewer.spin_display_combo.setCurrentText("Show all spins")
    assert viewer.vector_plot.visible()
    assert not viewer.position_mean_plot.visible()
    assert viewer.mean_vector.visible()
    assert len(viewer.path_points) == 1


def test_3d_spatial_layout_tracks_one_tip_path_per_position():
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = _spatial_viewer([0.0, 5.0], 2)
    first = np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0], [0, 0, 1.0]])
    second = np.array([[0, 1.0, 0], [0, 1.0, 0], [0, 1.0, 0], [0, 1.0, 0]])

    viewer.update_magnetization(first)
    assert not viewer.position_path_item.visible()
    viewer.update_magnetization(second)

    spacing = 1.0 / viewer.vector_length_spin.value()
    origins = np.array([[0, 0, -0.5 * spacing], [0, 0, 0.5 * spacing]])
    # One segment per position from the previous to the current mean tip.
    np.testing.assert_allclose(
        viewer.position_path_item.pos,
        [
            origins[0] + [1, 0, 0],
            origins[0] + [0, 1, 0],
            origins[1] + [0, 0, 1],
            origins[1] + [0, 1, 0],
        ],
    )
    assert viewer.position_path_item.visible()

    viewer.track_checkbox.setChecked(False)
    assert not viewer.position_path_item.visible()


def test_3d_spatial_layout_shows_b1_and_gradient_once():
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = _spatial_viewer([-10.0, 0.0, 10.0], 1)
    b1 = np.array([0.0, 0.1j, 0.05])
    gradients = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    positions_m = np.array([[0, 0, -0.01], [0, 0, 0.0], [0, 0, 0.01]])
    viewer.set_spatial_fields(b1, gradients, positions_m)
    viewer.update_magnetization(np.tile([0.0, 0.0, 1.0], (3, 1)))
    origin = np.array([-viewer.FIELD_GAP, 0.0, 0.0])
    np.testing.assert_allclose(viewer._field_origin, origin)

    viewer.show_field_frame(0)
    assert not viewer.b1_arrow_item.visible()
    assert not viewer.gradient_arrow_item.visible()
    assert viewer.field_ring_item.visible()

    viewer.show_field_frame(1)
    # One arrow at full length for the largest B1, oriented like the overlay.
    np.testing.assert_allclose(
        viewer.b1_arrow_item.pos, [origin, origin + [0, 1, 0]], atol=1e-6
    )
    assert viewer.b1_arrow_item.visible()
    assert not viewer.gradient_arrow_item.visible()

    viewer.show_field_frame(2)
    # The gradient vector, scaled to the largest gradient of the sequence.
    np.testing.assert_allclose(
        viewer.gradient_arrow_item.pos, [origin, origin + [0, 0, 1]]
    )
    assert viewer.gradient_arrow_item.visible()
    assert len(viewer.field_tip_item.pos) == 2
    assert "10.00 mT/m" in viewer.field_label_item.text
    assert "5.00 µT" in viewer.field_label_item.text

    viewer.gradient_field_checkbox.setChecked(False)
    viewer.b1_field_checkbox.setChecked(False)
    assert not viewer.gradient_arrow_item.visible()
    assert not viewer.b1_arrow_item.visible()
    assert not viewer.field_ring_item.visible()
    assert not viewer.field_tip_item.visible()
    assert not viewer.field_label_item.visible()

    viewer.gradient_field_checkbox.setChecked(True)
    viewer.set_spatial_groups(None)
    assert not viewer.gradient_arrow_item.visible()


def _label_texts(viewer):
    return [item.text for item in viewer.position_label_items]


@pytest.mark.parametrize(("axis", "azimuth"), [("x", -90.0), ("y", 0.0)])
def test_3d_spatial_line_along_x_or_y_is_a_row_on_its_scene_axis(axis, azimuth):
    app = QApplication.instance() or QApplication(sys.argv)
    positions_m = build_line_positions(3, 0.02, axis=axis)
    viewer = _spatial_viewer(positions_m * 1000.0, 1)
    column = "xyz".index(axis)

    spacing = 1.0 / viewer.vector_length_spin.value()
    origins = np.zeros((3, 3))
    origins[:, column] = [-spacing, 0.0, spacing]
    np.testing.assert_allclose(viewer._spatial_origins, origins)
    assert viewer._scene_kind() == "row"
    # Face-on default camera: x rows from the front, y rows from the side.
    assert viewer.gl_widget.opts["azimuth"] == pytest.approx(azimuth)
    assert _label_texts(viewer)[:4] == ["-10.0", "0.0", "10.0", f"{axis} (mm)"]
    assert viewer.max_positions_label.text() == "Max positions:"

    means = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    viewer.update_magnetization(means)
    np.testing.assert_allclose(viewer.position_mean_plot.pos[0::2], origins)
    np.testing.assert_allclose(viewer.position_mean_plot.pos[1::2], origins + means)
    viewer.net_vector_checkbox.setChecked(True)
    assert viewer.mean_vector.transform().column(3)[column] == pytest.approx(
        -spacing - viewer.FIELD_GAP - viewer.NET_GAP
    )

    # One gradient arrow before the first position, along the row axis.
    gradients = np.zeros((1, 3))
    gradients[0, column] = 1.0
    viewer.set_spatial_fields(np.zeros(1), gradients, positions_m)
    viewer.show_field_frame(0)
    field_origin = np.zeros(3)
    field_origin[column] = -spacing - viewer.FIELD_GAP
    np.testing.assert_allclose(
        viewer.gradient_arrow_item.pos,
        [field_origin, field_origin + gradients[0]],
    )
    assert viewer.gradient_arrow_item.visible()
    assert "G 10.00 mT/m" in viewer.field_label_item.text


def test_3d_spatial_plane_draws_a_lattice_on_the_floor():
    app = QApplication.instance() or QApplication(sys.argv)
    positions_m = build_plane_positions((3, 2), (0.02, 0.01), "xy")
    viewer = _spatial_viewer(positions_m * 1000.0, 1)

    spacing = 1.0 / viewer.vector_length_spin.value()
    xs = np.array([-spacing, 0.0, spacing])
    ys = np.array([-0.5, 0.5]) * spacing
    origins = np.column_stack([np.tile(xs, 2), np.repeat(ys, 3), np.zeros(6)])
    np.testing.assert_allclose(viewer._spatial_origins, origins)
    assert viewer._scene_kind() == "plane"
    assert viewer.max_positions_label.text() == "Max per axis:"
    # The lattice replaces the floor grid; the camera looks from above.
    assert not viewer.grid_item.visible()
    assert viewer.gl_widget.opts["elevation"] == pytest.approx(22.0)
    assert viewer.gl_widget.opts["azimuth"] == pytest.approx(-55.0)
    lattice = viewer.position_axis_item.pos.reshape(-1, 2, 3)
    assert len(lattice) == 2 + 3
    np.testing.assert_allclose(lattice[..., 2], 0.0)
    texts = _label_texts(viewer)
    assert texts[:4] == ["-10.0", "0.0", "10.0", "x (mm)"]
    assert texts[4:7] == ["-5.0", "5.0", "y (mm)"]

    # Mz points up from every lattice point.
    means = np.tile([0.0, 0.0, 1.0], (6, 1))
    means[4] = [1.0, 0.0, 0.0]
    viewer.tip_curve_checkbox.setChecked(True)
    viewer.update_magnetization(means)
    np.testing.assert_allclose(viewer.position_mean_plot.pos[1::2], origins + means)
    # The tip mesh joins neighbours along x and along y.
    pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
    np.testing.assert_allclose(
        viewer.tip_curve_item.pos,
        (origins + means)[np.array(pairs)].reshape(-1, 3),
    )
    # Position colors follow the first plane axis.
    colors = viewer._position_colors(6)
    np.testing.assert_allclose(colors[:3], colors[3:])
    assert not np.allclose(colors[0], colors[1])

    # The gradient is one arrow before the first column of the lattice.
    viewer.set_spatial_fields(np.zeros(1), np.array([[1.0, 0.0, 0.0]]), positions_m)
    viewer.show_field_frame(0)
    field_origin = np.array([xs[0] - viewer.FIELD_GAP, 0.0, 0.0])
    np.testing.assert_allclose(
        viewer.gradient_arrow_item.pos, [field_origin, field_origin + [1, 0, 0]]
    )
    assert viewer.gradient_arrow_item.visible()
    assert "Gx 10.00 Gy 0.00 mT/m" in viewer.field_label_item.text

    viewer.camera_buttons["Top"].click()
    assert viewer.gl_widget.opts["elevation"] == pytest.approx(90.0)

    viewer.set_spatial_groups(None)
    assert viewer.grid_item.visible()
    assert not viewer.gradient_arrow_item.visible()
    assert viewer.max_positions_label.text() == "Max positions:"


@pytest.mark.parametrize(("plane", "azimuth"), [("xz", -90.0), ("yz", 0.0)])
def test_3d_spatial_upright_plane_keeps_its_scene_axes(plane, azimuth):
    app = QApplication.instance() or QApplication(sys.argv)
    positions_m = build_plane_positions((3, 2), (0.02, 0.01), plane)
    viewer = _spatial_viewer(positions_m * 1000.0, 1)

    horizontal = "xyz".index(plane[0])
    np.testing.assert_allclose(viewer._spatial_origins[:, 3 - horizontal - 2], 0.0)
    assert np.ptp(viewer._spatial_origins[:, horizontal]) > 0
    assert np.ptp(viewer._spatial_origins[:, 2]) > 0
    assert viewer.gl_widget.opts["azimuth"] == pytest.approx(azimuth)
    assert viewer.grid_item.visible()
    # The floor grid sits below the lowest row and its labels.
    floor = viewer.grid_item.transform().column(3).z()
    assert floor < viewer._spatial_origins[:, 2].min() - viewer.LABEL_GAP


def test_free_mode_vector_view_subsamples_a_plane_per_axis():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    nframes, nfreq = 3, 2
    positions = build_plane_positions((7, 5), (0.03, 0.02), "xy")
    npos = len(positions)
    rng = np.random.default_rng(1)
    window.anim_vectors_full = rng.normal(size=(nframes, npos, nfreq, 3))
    window.playback_indices = np.arange(nframes)
    window.playback_time = np.linspace(0.0, 2e-3, nframes)
    window.playback_time_ms = window.playback_time * 1000.0
    window.last_positions = positions
    window.anim_b1 = np.zeros(nframes, dtype=complex)
    window.anim_gradients = np.tile([1.0, 0.0, 0.0], (nframes, 1))
    window.mag_3d.set_selector_limits(npos, nfreq)
    window.mag_3d.max_positions_spin.setValue(4)
    window.mag_3d.arrangement_combo.setCurrentText(
        window.mag_3d.ARRANGEMENT_ALONG_POSITION
    )

    window._refresh_vector_view(mean_only=False, restart=False)

    shown = infer_layout(positions).grid_indices(4)
    assert len(shown) == 9
    np.testing.assert_allclose(
        window.mag_3d._spatial_positions_mm, positions[shown] * 1000.0
    )
    assert window.mag_3d._scene_kind() == "plane"
    assert window.mag_3d._spatial_layout.counts == (3, 3)
    assert window.mag_3d.max_positions_label.text() == "Max per axis:"
    assert window.anim_data.shape == (nframes, len(shown) * nfreq, 3)
    np.testing.assert_allclose(window.mag_3d._field_gradients, window.anim_gradients)


def _plane_result(positions, nframes):
    """Magnetization whose |Mxy| grows with the x index and phase/Mz with y."""
    n_first, n_second = infer_layout(positions).counts
    ix = np.tile(np.arange(n_first), n_second)
    iy = np.repeat(np.arange(n_second), n_first)
    scale = np.linspace(0.5, 1.0, nframes)[:, None, None]
    ones = np.ones((1, 1, 2))
    amp = scale * (0.2 * ix)[None, :, None] * ones
    phi = np.radians(40.0 * iy - 60.0)[None, :, None]
    mz = scale * (-0.5 + 0.4 * iy)[None, :, None] * ones
    return {"mx": amp * np.cos(phi), "my": amp * np.sin(phi), "mz": mz}, ix, iy


def _show_spatial_result(window, positions, result):
    nframes = result["mx"].shape[0]
    window.last_positions = positions
    window.last_result = result
    window.last_time = np.linspace(0.0, 1e-3, nframes)
    window.last_frequencies = np.zeros(result["mx"].shape[2])
    window._precompute_plot_ranges(result)
    window.update_spatial_plot_from_last_result()


def _spatial_curve(plot, name):
    """Unclipped data of a named curve (the plots clip to their view)."""
    item = next(item for item in plot.listDataItems() if item.name() == name)
    return item.xData, item.yData


def test_free_mode_spatial_map_shows_plane_and_profile():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    positions = build_plane_positions((4, 3), (0.03, 0.02), "xy")
    result, ix, iy = _plane_result(positions, nframes=3)
    _show_spatial_result(window, positions, result)

    assert not window.spatial_profile_controls.isHidden()
    combo = window.spatial_plot_type
    assert combo.model().item(combo.findText("Map")).isEnabled()
    combo.setCurrentText("Map")
    assert not window.spatial_map_widget.isHidden()
    assert not window.spatial_mxy_plot.isHidden()
    assert window.spatial_heatmap_container.isHidden()

    # Images are indexed [y, x]; phase is masked where |Mxy| vanishes (x0)
    images = {key: item.image for key, item in window.spatial_map_items.items()}
    np.testing.assert_allclose(images["mxy"], (0.2 * ix).reshape(3, 4))
    assert np.isnan(images["phase"][:, 0]).all()
    np.testing.assert_allclose(
        images["phase"][:, 1:], (40.0 * iy - 60.0).reshape(3, 4)[:, 1:]
    )
    np.testing.assert_allclose(images["mz"], (-0.5 + 0.4 * iy).reshape(3, 4))
    item = window.spatial_map_items["mxy"]
    rect = item.mapRectToParent(item.boundingRect())
    assert (rect.left(), rect.top(), rect.width(), rect.height()) == pytest.approx(
        (-20.0, -15.0, 40.0, 30.0)
    )

    # Earlier frames keep the levels of the whole result
    window.update_spatial_plot_from_last_result(time_idx=0)
    np.testing.assert_allclose(
        window.spatial_map_items["mxy"].image, 0.5 * (0.2 * ix).reshape(3, 4)
    )
    assert window.spatial_map_colorbars["mxy"].levels() == pytest.approx((0.0, 0.6))
    assert window.spatial_map_colorbars["mz"].levels() == pytest.approx((-0.5, 0.5))
    assert window.spatial_map_colorbars["phase"].levels() == pytest.approx(
        (-180.0, 180.0)
    )

    # The profile starts along x through the grid centre (y = 0)
    assert window.spatial_profile_label.text() == "y = 0.00 mm"
    x, y = _spatial_curve(window.spatial_mxy_plot, "|Mxy|")
    np.testing.assert_allclose(x, [-15.0, -5.0, 5.0, 15.0])
    np.testing.assert_allclose(y, 0.5 * 0.2 * np.arange(4))
    assert window.spatial_mxy_plot.getAxis("bottom").labelText == "Position x"
    marker = window.spatial_map_profile_lines["mxy"][1]
    assert marker.angle == 0
    assert marker.value() == pytest.approx(0.0)

    window.spatial_profile_axis.setCurrentIndex(1)
    assert window.spatial_profile_slider.maximum() == 3
    assert window.spatial_profile_label.text() == "x = -5.00 mm"
    x, y = _spatial_curve(window.spatial_mxy_plot, "|Mxy|")
    np.testing.assert_allclose(x, [-10.0, 0.0, 10.0])
    np.testing.assert_allclose(y, 0.2)
    assert window.spatial_mxy_plot.getAxis("bottom").labelText == "Position y"
    assert marker.angle == 90
    assert marker.value() == pytest.approx(-5.0)

    export = window._last_spatial_export
    assert export["position_axis"] == "y"
    assert export["grid_axes"] == ("x", "y")
    np.testing.assert_array_equal(export["profile_indices"], [1, 5, 9])
    np.testing.assert_allclose(export["position_m"], [-0.01, 0.0, 0.01])
    assert export["mxy_per_freq"].shape == (3, 2)
    assert export["mz_map"].shape == (3, 4)


def test_free_mode_spatial_line_along_x_uses_its_axis():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    plane = build_plane_positions((4, 3), (0.03, 0.02), "xy")
    _show_spatial_result(window, plane, _plane_result(plane, nframes=2)[0])
    window.spatial_plot_type.setCurrentText("Map")

    positions = build_line_positions(5, 0.02, "x")
    amp = np.linspace(0.0, 1.0, 5)[None, :, None] * np.ones((2, 1, 1))
    result = {"mx": amp, "my": np.zeros_like(amp), "mz": 1.0 - amp}
    _show_spatial_result(window, positions, result)

    combo = window.spatial_plot_type
    assert combo.currentText() == "Line"
    assert not combo.model().item(combo.findText("Map")).isEnabled()
    assert window.spatial_profile_controls.isHidden()
    assert window.spatial_map_widget.isHidden()
    x, y = _spatial_curve(window.spatial_mxy_plot, "|Mxy|")
    np.testing.assert_allclose(x, [-10.0, -5.0, 0.0, 5.0, 10.0])
    np.testing.assert_allclose(y, amp[-1, :, 0])
    assert window.spatial_mz_plot.getAxis("bottom").labelText == "Position x"
    assert window._last_spatial_export["position_axis"] == "x"

    # Slice-thickness guides belong to the slice-select axis only
    guides = window.spatial_slice_lines["mxy"]
    assert not any(line.isVisible() for line in guides)
    window.sequence_designer.slice_axis_combo.setCurrentText("x")
    window._refresh_spatial_view()
    if window.sequence_designer._slice_thickness_m():
        assert all(line.isVisible() for line in guides)


def _read_export_columns(path):
    """Column names and values of a CSV written by DatasetExporter."""
    lines = Path(path).read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    values = np.loadtxt(path, delimiter=",", skiprows=header + 1, ndmin=2)
    return lines[header].split(","), values, lines[:header]


def test_free_mode_plane_spatial_export_writes_every_grid_position(tmp_path):
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    positions = build_plane_positions((4, 3), (0.03, 0.02), "xy")
    result, ix, iy = _plane_result(positions, nframes=3)
    _show_spatial_result(window, positions, result)
    window.spatial_plot_type.setCurrentText("Map")
    cache = window._last_spatial_export

    path = window._write_spatial_export(cache, tmp_path / "plane.csv", "csv")

    names, values, header = _read_export_columns(path)
    assert names == ["x_m", "y_m", "z_m", "mxy", "phase_deg", "mz"]
    assert "#   grid: 4 x 3 in xy (x fastest)" in header
    assert f"#   time_s: {cache['time_s']}" in header
    t = cache["time_idx"]
    np.testing.assert_allclose(values[:, :3], positions)
    np.testing.assert_allclose(
        values[:, 3], np.hypot(result["mx"][t, :, 0], result["my"][t, :, 0])
    )
    np.testing.assert_allclose(values[:, 5], result["mz"][t, :, 0])
    phase = values[:, 4]
    assert np.isnan(phase[ix == 0]).all()
    np.testing.assert_allclose(phase[ix > 0], (40.0 * iy - 60.0)[ix > 0])

    # The map panel exports as one image next to the profile plots
    maps_png = ImageExporter().export_pyqtgraph_plot(
        window.spatial_map_widget, str(tmp_path / "maps.png"), width=600
    )
    assert maps_png and Path(maps_png).stat().st_size > 0


def test_free_mode_line_spatial_export_names_its_axis(tmp_path):
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    positions = build_line_positions(5, 0.02, "x")
    amp = np.linspace(0.0, 1.0, 5)[None, :, None] * np.ones((2, 1, 1))
    _show_spatial_result(
        window, positions, {"mx": amp, "my": np.zeros_like(amp), "mz": 1.0 - amp}
    )

    path = window._write_spatial_export(
        window._last_spatial_export, tmp_path / "line.csv", "csv"
    )

    names, values, header = _read_export_columns(path)
    assert names[:3] == ["position_m", "mxy", "mz"]
    assert "#   position_axis: x" in header
    np.testing.assert_allclose(values[:, 0], positions[:, 0])
    np.testing.assert_allclose(values[:, 1], amp[-1, :, 0])


def test_parameter_sweep_records_plane_geometry_and_gradient_lobes():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.pos_geometry_combo.setCurrentText("2D plane")
    window.pos_plane_combo.setCurrentText("xz")
    window.pos_spin.setValue(4)
    window.pos2_spin.setValue(3)
    window.pos_range.setValue(30.0)
    window.pos2_range.setValue(20.0)
    window.pos_center_y.setValue(2.0)
    designer = window.sequence_designer
    designer.slice_axis_combo.setCurrentText("y")
    lobe = GradientLobe(axis="x", start_ms=1.0, duration_ms=2.0)
    designer.set_gradient_lobes([lobe])

    params = window.param_sweep_widget._constant_params()

    assert params["num_positions"] == 12
    assert params["position_geometry"] == "plane"
    assert params["position_axes"] == "xz"
    assert params["position_counts"] == [4, 3]
    assert params["position_ranges_mm"] == [30.0, 20.0]
    assert params["position_center_mm"] == [0.0, 2.0, 0.0]
    assert params["slice_axis"] == "y"
    assert params["gradient_lobes"] == [lobe.to_state()]
    assert json.loads(json.dumps(params)) == params


def test_3d_spatial_layout_returns_to_overlay():
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = _spatial_viewer([-1.0, 1.0], 1)
    viewer.update_magnetization(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert viewer.position_axis_item.visible()

    viewer.set_spatial_groups(None)
    viewer.update_magnetization(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    assert not viewer.position_axis_item.visible()
    assert not viewer.position_mean_plot.visible()
    assert not viewer.position_label_items
    np.testing.assert_allclose(viewer.vector_plot.pos[0::2], 0.0)
    assert viewer.mean_vector.transform().column(3).x() == pytest.approx(0.0)
    assert viewer.axis_item.transform().column(3).z() == pytest.approx(0.0)


def test_3d_arrangement_controls_are_hidden_without_positions():
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = MagnetizationViewer()
    viewer.arrangement_combo.setCurrentText(viewer.ARRANGEMENT_ALONG_POSITION)

    assert viewer.spatial_controls.isHidden()
    assert viewer.get_vector_arrangement() == viewer.ARRANGEMENT_OVERLAID


@pytest.mark.parametrize(
    ("total", "limit", "expected"),
    [
        (5, 32, [0, 1, 2, 3, 4]),
        (41, 32, list(range(0, 41, 2))),
        (40, 32, list(range(1, 40, 2))),
        (1100, 32, list(range(9, 1100, 36))),
        (9, 1, [4]),
    ],
)
def test_centered_stride_indices_keep_uniform_spacing(total, limit, expected):
    indices = centered_stride_indices(total, limit)

    assert indices.tolist() == expected
    assert len(indices) <= limit


def test_3d_phase_colors_follow_transverse_phase():
    colors = MagnetizationViewer.phase_colors(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(colors[1], [0.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(colors[2], [1.0, 1.0, 1.0, 1.0])


def test_free_mode_vector_view_arranges_subsampled_positions():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    nframes, npos, nfreq = 4, 9, 5
    rng = np.random.default_rng(0)
    window.anim_vectors_full = rng.normal(size=(nframes, npos, nfreq, 3))
    window.playback_indices = np.arange(nframes)
    window.playback_time = np.linspace(0.0, 3e-3, nframes)
    window.playback_time_ms = window.playback_time * 1000.0
    window.last_positions = np.zeros((npos, 3))
    window.last_positions[:, 2] = np.linspace(-0.02, 0.02, npos)
    window.anim_b1 = np.full(nframes, 0.1 + 0j)
    window.anim_gradients = np.zeros((nframes, 3))
    window.anim_gradients[:, 2] = 1.0
    window.mag_3d.set_selector_limits(npos, nfreq)
    window.mag_3d.max_positions_spin.setValue(5)
    window.mag_3d.arrangement_combo.setCurrentText(
        window.mag_3d.ARRANGEMENT_ALONG_POSITION
    )

    window._refresh_vector_view(mean_only=False, restart=False)

    shown = window._display_subsample_indices(npos, 5)
    np.testing.assert_allclose(
        window.mag_3d._spatial_positions_mm, window.last_positions[shown] * 1000
    )
    assert window.anim_data.shape == (nframes, len(shown) * nfreq, 3)
    np.testing.assert_allclose(
        window.anim_position_means,
        window.anim_vectors_full[:, shown].mean(axis=2),
    )
    np.testing.assert_allclose(
        window.anim_net, window.anim_vectors_full.mean(axis=(1, 2))
    )
    # The global gradient vector of every playback frame.
    np.testing.assert_allclose(
        window.mag_3d._field_gradients, np.asarray(window.anim_gradients)
    )
    window._update_b1_arrow(1)
    assert not window.mag_3d.b1_arrow.visible()
    assert window.mag_3d._field_frame_idx == 1

    window.mag_3d.view_mode_combo.setCurrentText("Freqs @ position")
    window._refresh_vector_view(mean_only=False, restart=False)
    assert not window.mag_3d.is_spatial_layout_active()
    assert window.anim_net is None


def test_core_dialog_fields_have_tooltips(tmp_path):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialogs = (
        SettingsDialog(MemoryPolicy(), tmp_path, tooltips_enabled=True),
        PulseImportDialog(),
    )
    field_types = (QAbstractSpinBox, QCheckBox, QComboBox, QLineEdit)
    missing = [
        field
        for dialog in dialogs
        for field in dialog.findChildren(field_types)
        if not (
            isinstance(field, QLineEdit)
            and isinstance(field.parentWidget(), QAbstractSpinBox)
        )
        if not field.toolTip().strip()
    ]

    assert not missing


def test_tooltips_can_be_disabled_and_restored():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = BlochSimulatorGUI()
    window._set_tooltips_enabled(False)
    assert all(not widget.toolTip() for widget, _ in window._tooltip_registry)
    assert all(
        not window.tab_widget.tabToolTip(index)
        for index, _ in window._tab_tooltip_registry
    )

    window._set_tooltips_enabled(True)
    assert all(widget.toolTip() == text for widget, text in window._tooltip_registry)
    assert all(
        window.tab_widget.tabToolTip(index) == text
        for index, text in window._tab_tooltip_registry
    )


def test_configured_export_directory_is_used(tmp_path, monkeypatch):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    monkeypatch.delenv("BLOCH_EXPORT_DIR", raising=False)
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    export_directory = tmp_path / "custom-exports"
    window.app_settings.setValue("general/export_directory", str(export_directory))
    window.app_settings.setValue("sequence/kernel", "strict_reference")
    window.app_settings.setValue("sequence/dynamic_kernel", "native_parallel")
    window.app_settings.setValue("sequence/timestep_preset", "fast")
    window.app_settings.setValue("sequence/timestep_us", 10.0)
    window.app_settings.setValue("sequence/spoiler_mode", "gradient")
    window.app_settings.setValue("sequence/subvoxel_spins_x", 3)
    window.app_settings.setValue("sequence/subvoxel_spins_y", 5)
    window.app_settings.setValue("sequence/subvoxel_spins_z", 11)
    window.app_settings.setValue("simulation/thread_mode", "manual")
    window.app_settings.setValue("simulation/manual_threads", 3)

    assert window._get_export_directory() == export_directory
    assert export_directory.is_dir()
    assert window._load_sequence_kernel() == "strict_reference"
    assert window._load_dynamic_sequence_kernel() == "native_parallel"
    assert window._load_sequence_timestep_preset() == "fast"
    assert window._load_sequence_timestep_us() == pytest.approx(10.0)
    assert window._load_sequence_spoiler_mode() == "gradient"
    assert window._load_subvoxel_spin_counts() == (3, 5, 11)
    assert window._load_configured_num_threads() == 3

    window._ensure_sequence_simulation_workspace(window.sequence_simulation_tab_index)
    assert window.sequence_simulation_widget._export_directory() == export_directory
    window.sequence_simulation_widget.result = object()
    with patch(
        "blochsimulator.ui.sequence_simulation_widget.SequenceExportDialog"
    ) as sequence_export_dialog:
        sequence_export_dialog.return_value.exec_.return_value = 0
        window.sequence_simulation_widget._export_results()
    assert sequence_export_dialog.call_args.kwargs["directory"] == export_directory

    window.app_settings.setValue("sequence/kernel", "invalid")
    assert window._load_sequence_kernel() == "optimized"
    window.app_settings.setValue("sequence/kernel", "reference")
    assert window._load_sequence_kernel() == "optimized"
    window.app_settings.setValue("sequence/dynamic_kernel", "invalid")
    assert window._load_dynamic_sequence_kernel() == "native_parallel"
    for removed in ("native_serial", "reference", "metal_hybrid"):
        window.app_settings.setValue("sequence/dynamic_kernel", removed)
        assert window._load_dynamic_sequence_kernel() == "native_parallel"
    # Settings saved before revision 2 stored Optimized NumPy as the old default.
    window.app_settings.setValue("sequence/dynamic_kernel", "optimized")
    assert window._load_dynamic_sequence_kernel() == "native_parallel"
    window.app_settings.setValue("sequence/kernel_settings_revision", 2)
    assert window._load_dynamic_sequence_kernel() == "optimized"
    window.app_settings.setValue("sequence/spoiler_mode", "invalid")
    assert window._load_sequence_spoiler_mode() == "ideal"


def test_simulation_settings_are_persisted_and_applied(tmp_path):
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    window.sequence_simulation_widget = MagicMock()
    dialog = MagicMock()
    dialog.exec_.return_value = QDialog.Accepted
    dialog.get_export_directory.return_value = tmp_path
    dialog.get_policy.return_value = MemoryPolicy()
    dialog.tooltips_enabled.return_value = True
    dialog.interface_style.return_value = "fusion"
    dialog.sequence_timing_visible.return_value = False
    dialog.sequence_live_progress_enabled.return_value = False
    dialog.sequence_kernel.return_value = "strict_reference"
    dialog.dynamic_sequence_kernel.return_value = "optimized"
    dialog.sequence_timestep_preset.return_value = "fast"
    dialog.sequence_timestep_us.return_value = 10.0
    dialog.sequence_spoiler_mode.return_value = "gradient"
    dialog.subvoxel_spin_counts.return_value = (3, 5, 11)
    dialog.subvoxel_sampling_method.return_value = "stratified"
    dialog.thread_mode.return_value = "manual"
    dialog.manual_thread_count.return_value = 2
    dialog.animation_memory_budget_bytes.return_value = 2048 * 1024**2
    scanner_parameters = ScannerParameters(max_grad_mtm=40.0, max_slew_tms=180.0)
    dialog.scanner_parameters.return_value = scanner_parameters
    workspace_defaults = WorkspaceDefaults(
        sequence_fov_mm=(180.0, 170.0, 80.0),
        phantom_fov_mm=(60.0, 50.0, 40.0),
        phantom_nucleus="C13",
        field_strength_t=7.0,
        sequence_reference_ppm_by_nucleus={"C13": 175.5},
    )
    dialog.workspace_defaults.return_value = workspace_defaults

    with patch("blochsimulator.ui.main_window.SettingsDialog", return_value=dialog):
        window.show_settings(initial_tab="simulation")

    assert window.app_settings.value("sequence/timestep_preset") == "fast"
    assert float(window.app_settings.value("sequence/timestep_us")) == 10.0
    assert window.app_settings.value("sequence/spoiler_mode") == "gradient"
    assert int(window.app_settings.value("sequence/subvoxel_spins_x")) == 3
    assert int(window.app_settings.value("sequence/subvoxel_spins_y")) == 5
    assert int(window.app_settings.value("sequence/subvoxel_spins_z")) == 11
    assert (
        window.app_settings.value("sequence/subvoxel_sampling_method") == "stratified"
    )
    assert window.app_settings.value("simulation/thread_mode") == "manual"
    assert int(window.app_settings.value("simulation/manual_threads")) == 2
    assert int(window.app_settings.value("memory/animation_replay_mib")) == 2048
    assert window.app_settings.value("interface/style") == "fusion"
    assert (
        window.app_settings.value("interface/show_sequence_timing", type=bool) is False
    )
    window.sequence_simulation_widget.set_sequence_timing_visible.assert_called_once_with(
        False
    )
    assert float(window.app_settings.value("scanner/max_grad_mtm")) == 40.0
    assert float(window.app_settings.value("scanner/max_slew_tms")) == 180.0
    assert float(window.app_settings.value("defaults/sequence_fov_x_mm")) == 180.0
    assert window.app_settings.value("defaults/phantom_nucleus") == "C13"
    assert float(window.app_settings.value("defaults/field_strength_t")) == 7.0
    assert (
        float(window.app_settings.value("defaults/sequence_reference_ppm/C13")) == 175.5
    )
    assert window.app_settings.value("sequence/kernel") == "strict_reference"
    assert window.app_settings.value("sequence/dynamic_kernel") == "optimized"
    assert int(window.app_settings.value("sequence/kernel_settings_revision")) == 2
    assert window._load_dynamic_sequence_kernel() == "optimized"
    assert window.simulator.sequence_kernel == "strict_reference"
    assert window.simulator.dynamic_sequence_kernel == "optimized"
    assert window.simulator.num_threads == 2
    window.sequence_simulation_widget.set_sequence_timestep_us.assert_called_once_with(
        10.0
    )
    window.sequence_simulation_widget.set_dynamic_sequence_kernel.assert_called_once_with(
        "optimized"
    )
    window.sequence_simulation_widget.set_spoiler_configuration.assert_called_once_with(
        "gradient", (3, 5, 11), "stratified"
    )
    window.sequence_simulation_widget.set_thread_configuration.assert_called_once_with(
        "manual", 2
    )
    window.sequence_simulation_widget.set_animation_memory_budget_bytes.assert_called_once_with(
        2048 * 1024**2
    )
    window.sequence_simulation_widget.set_scanner_parameters.assert_called_once_with(
        scanner_parameters
    )
    window.sequence_simulation_widget.set_workspace_defaults.assert_called_once_with(
        workspace_defaults
    )


def test_changing_only_spoiler_settings_preserves_active_sequence_parameters(
    tmp_path,
):
    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    window.sequence_simulation_widget = MagicMock()
    scanner_parameters = window._load_scanner_parameters()
    workspace_defaults = WorkspaceDefaults.from_settings(window.app_settings)
    dialog = MagicMock()
    dialog.exec_.return_value = QDialog.Accepted
    dialog.get_export_directory.return_value = tmp_path
    dialog.get_policy.return_value = MemoryPolicy()
    dialog.tooltips_enabled.return_value = True
    dialog.interface_style.return_value = "automatic"
    dialog.sequence_live_progress_enabled.return_value = True
    dialog.sequence_kernel.return_value = "optimized"
    dialog.dynamic_sequence_kernel.return_value = "optimized"
    dialog.sequence_timestep_preset.return_value = "balanced"
    dialog.sequence_timestep_us.return_value = 5.0
    dialog.sequence_spoiler_mode.return_value = "gradient"
    dialog.subvoxel_spin_counts.return_value = (1, 1, 17)
    dialog.subvoxel_sampling_method.return_value = "midpoint"
    dialog.thread_mode.return_value = "automatic"
    dialog.manual_thread_count.return_value = 4
    dialog.animation_memory_budget_bytes.return_value = 512 * 1024**2
    dialog.scanner_parameters.return_value = scanner_parameters
    dialog.workspace_defaults.return_value = workspace_defaults

    with patch("blochsimulator.ui.main_window.SettingsDialog", return_value=dialog):
        window.show_settings(initial_tab="simulation")

    window.sequence_simulation_widget.set_spoiler_configuration.assert_called_once_with(
        "gradient", (1, 1, 17), "midpoint"
    )
    window.sequence_simulation_widget.set_scanner_parameters.assert_not_called()
    window.sequence_simulation_widget.set_workspace_defaults.assert_not_called()


def test_settings_dialog_sequence_timing_preference(tmp_path):
    dialog = SettingsDialog(MemoryPolicy(), tmp_path, tooltips_enabled=True)
    assert dialog.sequence_timing_visible() is True
    checkbox = dialog.sequence_timing_checkbox
    interface_tab = dialog.tabs.widget(dialog.tabs.count() - 1)
    assert interface_tab.isAncestorOf(checkbox)
    checkbox.setChecked(False)
    assert dialog.sequence_timing_visible() is False
    reopened = SettingsDialog(
        MemoryPolicy(), tmp_path, tooltips_enabled=True, sequence_timing_visible=False
    )
    assert reopened.sequence_timing_visible() is False


def test_settings_dialog_saves_sequence_reference_defaults_per_nucleus(tmp_path):
    original = WorkspaceDefaults(phantom_nucleus="C13")
    dialog = SettingsDialog(
        MemoryPolicy(), tmp_path, tooltips_enabled=True, workspace_defaults=original
    )
    assert dialog.default_sequence_reference_spin.value() == 171.0
    dialog.default_sequence_reference_spin.setValue(183.35)
    dialog.phantom_nucleus_combo.setCurrentText("H1")
    assert dialog.default_sequence_reference_spin.value() == 0.0
    dialog.default_sequence_reference_spin.setValue(4.7)
    dialog.phantom_nucleus_combo.setCurrentText("C13")
    assert dialog.default_sequence_reference_spin.value() == 183.35
    assert original.sequence_reference_ppm("C13") == 171.0

    settings = QSettings(str(tmp_path / "references.ini"), QSettings.IniFormat)
    dialog.workspace_defaults().save(settings)
    settings.sync()
    reloaded_settings = QSettings(str(tmp_path / "references.ini"), QSettings.IniFormat)
    restored = WorkspaceDefaults.from_settings(reloaded_settings)
    reopened = SettingsDialog(
        MemoryPolicy(), tmp_path, tooltips_enabled=True, workspace_defaults=restored
    )
    assert reopened.default_sequence_reference_spin.value() == 183.35
    reopened.phantom_nucleus_combo.setCurrentText("H1")
    assert reopened.default_sequence_reference_spin.value() == 4.7


def test_workspace_defaults_round_trip_through_settings(tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    expected = WorkspaceDefaults(
        sequence_fov_mm=(210.0, 190.0, 75.0),
        phantom_fov_mm=(80.0, 70.0, 60.0),
        phantom_nucleus="P31",
        field_strength_t=9.4,
        sequence_reference_ppm_by_nucleus={"C13": 183.35, "P31": -10.0},
    )
    expected.save(settings)
    settings.sync()

    assert WorkspaceDefaults.from_settings(settings) == expected


@pytest.mark.parametrize("invalid_reference", ["invalid", "nan", "inf", -10001, 10001])
def test_invalid_reference_default_falls_back_for_each_nucleus(
    tmp_path, invalid_reference
):
    settings = QSettings(str(tmp_path / "invalid.ini"), QSettings.IniFormat)
    for nucleus in ("H1", "C13"):
        settings.setValue(
            f"defaults/sequence_reference_ppm/{nucleus}", invalid_reference
        )
    defaults = WorkspaceDefaults.from_settings(settings)
    assert defaults.sequence_reference_ppm("C13") == 171.0
    assert defaults.sequence_reference_ppm("H1") == 0.0


def test_legacy_automatic_nucleus_default_migrates_to_shared_h1(tmp_path):
    settings = QSettings(str(tmp_path / "legacy-settings.ini"), QSettings.IniFormat)
    settings.setValue("defaults/phantom_nucleus", "auto")

    assert WorkspaceDefaults.from_settings(settings).phantom_nucleus == "H1"


if __name__ == "__main__":
    pytest.main([__file__])


def test_free_mode_repetitions_play_the_sequence_once_per_tr():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    designer = window.sequence_designer
    designer.sequence_type.setCurrentText("Free Induction Decay")
    designer.tr_spin.setValue(10.0)
    assert designer.repetitions_spin.value() == 1
    assert not hasattr(window, "extra_tail_spin")
    pulse = window.rf_designer.get_pulse()
    dt = 1e-5
    b1, gradients, time = designer.compile_sequence(custom_pulse=pulse, dt=dt)
    period = time[-1] - time[0] + dt
    assert period == pytest.approx(10e-3, rel=1e-3)

    designer.repetitions_spin.setValue(3)
    b1_rep, gradients_rep, time_rep = designer.compile_sequence(
        custom_pulse=pulse, dt=dt
    )
    assert len(b1_rep) == 3 * len(b1)
    np.testing.assert_allclose(b1_rep, np.tile(b1, 3))
    np.testing.assert_allclose(gradients_rep, np.tile(gradients, (3, 1)))
    np.testing.assert_allclose(np.diff(time_rep), dt, rtol=1e-6)
    assert designer.get_state()["repetitions"] == 3

    # The SSFP loop keeps its own repeat count and hides the control.
    designer.sequence_type.setCurrentText("SSFP (Loop)")
    assert designer.repetitions_widget.isHidden()
    restored = designer.get_state()
    designer.set_state({"tr": 10.0})
    assert designer.repetitions_spin.value() == 1
    designer.set_state(restored)
    assert designer.repetitions_spin.value() == 3
