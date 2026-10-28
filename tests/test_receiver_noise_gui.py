"""Exercise receiver controls through the actual sequence worker and project UI."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QStyle,
    QStyleOptionSpinBox,
)

from blochsimulator import BlochSimulator
from blochsimulator.b1_fields import create_b1_preset
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import ADCEvent, RFEvent, SequenceProgram, ReceiverNoise
from blochsimulator.sequence.receiver import (
    ReceiverCalibration,
    ideal_fid_amplitude,
    receiver_acquisition_metadata,
)
from blochsimulator.sequence.reference_scan import (
    ReferenceScan,
    reference_signal_efficiency,
)
from blochsimulator.spectral_phantom import ChemicalSpecies, SpectralPhantom
from blochsimulator.ui.main_window import BlochSimulatorGUI
from blochsimulator.ui.receiver_noise import ReceiverNoiseWidget
from blochsimulator.ui.receiver_noise_preview import (
    REFERENCE_T2STAR_S,
    SNRBudgetPanel,
    reference_fid,
)
from blochsimulator.ui.sequence_simulation_widget import (
    SequenceSimulationThread,
    SequenceSimulationWidget,
)
from blochsimulator.units import thermal_polarization


def _inputs():
    shape = (2, 2, 2)
    obj = Phantom(
        shape,
        (0.02,) * 3,
        np.ones(shape),
        np.ones(shape),
    )
    program = SequenceProgram(
        (RFEvent(0, np.array([250.0]), 1e-3), ADCEvent(1e-3, 32, 1e-4)),
        duration_s=0.005,
    )
    return obj, program


def _fill_urea_calibration(panel, amplitude="200", std="2"):
    panel.mode.setCurrentIndex(2)
    panel.amplitude.setText(amplitude)
    panel.reference_std.setText(std)
    panel.concentration.setText("2000")
    panel.volume.setText("50")


def _budget_rows(budget):
    """Budget tree values keyed by (section, quantity)."""
    rows = {}
    for index in range(budget.tree.topLevelItemCount()):
        section = budget.tree.topLevelItem(index)
        for child in range(section.childCount()):
            item = section.child(child)
            rows[(section.text(0), item.text(0))] = item.text(1)
    return rows


def _budget_text(budget):
    texts = []
    for index in range(budget.tree.topLevelItemCount()):
        section = budget.tree.topLevelItem(index)
        texts.append(section.text(0))
        texts.extend(section.child(row).text(0) for row in range(section.childCount()))
    return "\n".join(texts)


def _pyruvate_phantom():
    """80 mM hyperpolarized pyruvate (20 %) in 200 ul at body temperature."""
    shape = (2, 2, 2)
    obj = Phantom(
        shape,
        ((0.2e-6) ** (1 / 3),) * 3,
        np.full(shape, 1e12),
        np.full(shape, 1e12),
        pd_map=np.full(shape, 80.0),
    )
    obj.m0_map[..., 2] = 0.2 / thermal_polarization("C13", 7.0, 310.15)
    return obj


def test_calibration_requires_explicit_capture_and_round_trips():
    panel = ReceiverNoiseWidget()
    _fill_urea_calibration(panel)
    with pytest.raises(ValueError, match="Set the calibration"):
        panel.settings()
    panel.averages.setValue(4)
    panel.bandwidth_khz.setValue(5)
    panel.capture_calibration(nucleus="C13", field_strength_t=7)
    calibration = panel.settings().calibration
    assert calibration.concentration_mm == 2000
    assert calibration.dwell_s == pytest.approx(2e-4)
    assert calibration.single_scan_snr == 50
    assert calibration.temperature_k == pytest.approx(293.15)
    assert calibration.target_temperature_k == pytest.approx(310.15)
    assert "100 mmol nuclei" in panel.info.text()
    restored = ReceiverNoiseWidget()
    restored.restore_state(panel.get_state())
    assert restored.settings().calibration == calibration
    assert restored.averages.value() == 4
    with pytest.raises(ValueError, match="same nucleus"):
        panel.settings(nucleus="C13", field_strength_t=3)
    panel.reference_std.setText("4")
    with pytest.raises(ValueError, match="changing"):
        panel.settings()
    panel.reference_std.setText("2")
    panel.target_temperature.setValue(20)
    with pytest.raises(ValueError, match="changing"):
        panel.settings()
    panel.volume.setText("")
    assert "reference volume" in panel.info.text()
    with pytest.raises(ValueError, match="changing"):
        panel.settings()
    restored.restore_state(None)
    assert restored.settings().mode == "off"


def test_worker_applies_calibrated_noise_with_run_bandwidth():
    obj, program = _inputs()
    calibration = ReceiverCalibration(100.0, 1.0, 1000.0, 8.0, "C13", 7.0, dwell_s=4e-4)
    outputs, errors = [], []
    worker = SequenceSimulationThread(
        BlochSimulator(use_parallel=False),
        program,
        obj,
        (),
        signal_weighting="voxel_volume",
        live_preview=False,
        nucleus="C13",
        field_strength_t=7.0,
        receiver_noise=ReceiverNoise("calibrated", seed=4, calibration=calibration),
    )
    worker.result_ready.connect(outputs.append)
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors
    measured = outputs[0]
    sigma = calibration.noise_std(1e-4)
    assert sigma == pytest.approx(2 * calibration.noise_std(4e-4))
    rng = np.random.Generator(np.random.PCG64(4))
    noise = sigma * (rng.standard_normal(32) + 1j * rng.standard_normal(32))
    np.testing.assert_allclose(
        measured.signal - measured.signal_noiseless, noise, atol=1e-20
    )
    assert measured.metadata["receiver_noise"]["noise_std_per_quadrature"] == [
        pytest.approx(sigma)
    ]


@pytest.mark.parametrize("kind", ["static", "spectral", "dynamic"])
def test_worker_applies_noise_once_after_all_components(kind):
    obj, program = _inputs()
    if kind == "spectral":
        obj = SpectralPhantom(
            shape=obj.shape,
            fov=obj.fov,
            species=[ChemicalSpecies("a", 0, 1, 1), ChemicalSpecies("b", 0, 1, 1)],
            concentration_maps={
                "a": np.full(obj.shape, 0.25),
                "b": np.full(obj.shape, 0.75),
            },
        )
    elif kind == "dynamic":
        obj = DynamicSpectralPhantom(
            shape=obj.shape,
            fov=obj.fov,
            pools=(
                ChemicalSpecies("Pyruvate", 0, 1, 1),
                ChemicalSpecies("Lactate", 12, 1, 1),
            ),
            initial_concentration_maps={
                "Pyruvate": np.ones(obj.shape),
                "Lactate": np.zeros(obj.shape),
            },
            kpl_map_s_inv=np.zeros(obj.shape),
        )
    outputs, errors = [], []
    worker = SequenceSimulationThread(
        BlochSimulator(use_parallel=False),
        program,
        obj,
        (),
        signal_weighting="voxel_volume",
        live_preview=False,
        receiver_noise=ReceiverNoise("relative", 25.0, 23, reference_dwell_s=4e-4),
    )
    worker.result_ready.connect(outputs.append)
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors
    assert len(outputs) == 1
    measured = outputs[0]
    amplitude = ideal_fid_amplitude(obj)
    # All three objects hold 8e-6 mol-equivalents; ADC at 4x the reference BW.
    assert amplitude == pytest.approx(8e-6)
    assert measured.metadata["receiver_acquisition"][
        "ideal_fid_amplitude"
    ] == pytest.approx(amplitude)
    sigma = 2 * amplitude / 25
    rng = np.random.Generator(np.random.PCG64(23))
    noise = sigma * (rng.standard_normal(32) + 1j * rng.standard_normal(32))
    np.testing.assert_allclose(
        measured.signal - measured.signal_noiseless, noise, atol=1e-20
    )
    assert measured.metadata["receiver_acquisition"]["adc_dwell_s"] == [1e-4]
    assert measured.metadata["receiver_acquisition"]["adc_samples"] == [32]


def test_workspace_reapplies_noise_without_running_bloch_again():
    obj, program = _inputs()
    widget = SequenceSimulationWidget()
    widget.program, widget.phantom = program, obj
    simulated = widget.simulator.simulate_sequence(program, obj)

    def with_context(phantom):
        context = receiver_acquisition_metadata(
            program,
            nucleus="C13",
            field_strength_t=7.0,
            phantom=phantom,
            signal_weighting="voxel",
        )
        metadata = {**simulated.metadata, "receiver_acquisition": context}
        return replace(simulated, metadata=metadata)

    clean = with_context(obj)
    widget._finished(clean, record_run=False)
    panel = widget.receiver_noise
    panel.mode.setCurrentIndex(1)
    panel.snr.setText("80")
    with patch.object(widget.simulator, "simulate_sequence") as simulate:
        panel.apply_button.click()
        first = widget.result.signal.copy()
        panel.new_button.click()
        second = widget.result.signal.copy()
        simulate.assert_not_called()
    assert panel.seed.value() == 1
    assert not np.array_equal(first, second)
    receiver = widget.result.metadata["receiver_noise"]
    assert receiver["noise_std_per_quadrature"] == [pytest.approx(8 / 80)]
    np.testing.assert_array_equal(widget.result.signal_noiseless, clean.signal)
    assert any(
        curve.name() == "Noiseless magnitude"
        for curve in widget.signal_plot.listDataItems()
    )
    panel.mode.setCurrentIndex(0)
    panel.apply_button.click()
    np.testing.assert_array_equal(widget.result.signal, clean.signal)
    # Results from before relative SNR lack the reference amplitude.
    widget._finished(with_context(None), record_run=False)
    panel.mode.setCurrentIndex(1)
    with patch.object(widget, "_show_receiver_noise_error") as warning:
        panel.apply_button.click()
    assert "Rerun the simulation" in warning.call_args.args[0]
    widget.close()


def test_workspace_calibration_forces_volume_weighting_and_survives_project(tmp_path):
    window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    widget = window.sequence_simulation_widget
    widget.object_source.setCurrentIndex(1)
    widget.matrix_size.setValue(4)
    widget.z_matrix_size.setValue(1)
    widget.program = _inputs()[1]
    panel = widget.receiver_noise
    _fill_urea_calibration(panel, "1000", "5")
    widget._capture_noise_calibration()
    assert panel.calibration.nucleus == widget.nucleus.currentText()
    assert widget._signal_weighting_mode() == "voxel_volume"
    assert not widget.signal_weighting.isEnabled()
    expected = panel.get_state()
    from blochsimulator.project_io import save_project

    path = tmp_path / "receiver.blochproj"
    save_project(
        path,
        window._complete_project_state(),
        widget.phantom,
        None,
        None,
        widget.program,
    )
    panel.restore_state(None)
    window.open_project(str(path))
    assert panel.get_state() == expected
    nucleus = panel.calibration.nucleus
    field = panel.calibration.field_strength_t
    # 0.1 mol of thermal nuclei at 20 degC measured with SNR 200.
    assert panel.settings().calibration.noise_std(1e-4) == pytest.approx(
        0.1
        / 200
        * thermal_polarization(nucleus, field, 293.15)
        / thermal_polarization(nucleus, field, 310.15)
    )
    window.close()


def test_run_rejects_changed_field_strength_before_worker_starts(monkeypatch):
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.program = _inputs()[1]
    _fill_urea_calibration(widget.receiver_noise)
    widget._capture_noise_calibration()
    widget.field_strength_t.setValue(widget.field_strength_t.value() + 1)
    monkeypatch.setattr(widget, "_ensure_current_generated_sequence", lambda: True)
    with patch.object(widget, "_show_receiver_noise_error") as error:
        assert widget._run() is False
    assert "field strength" in error.call_args.args[0]
    assert widget.worker is None
    widget.close()


def test_reference_fid_scales_noise_and_covers_the_decay():
    time_s, ideal, noisy = reference_fid(2.0, 0.05, 1e-4, seed=41)
    _, _, same = reference_fid(2.0, 0.05, 1e-4, seed=41)
    _, _, stronger = reference_fid(2.0, 0.1, 1e-4, seed=41)
    assert time_s.size == 1000
    assert ideal[0] == 2
    assert ideal[-1] == pytest.approx(2 * np.exp(-time_s[-1] / REFERENCE_T2STAR_S))
    np.testing.assert_array_equal(noisy, same)
    np.testing.assert_allclose(stronger - ideal, 2 * (noisy - ideal), atol=1e-15)
    assert np.std(noisy - ideal) == pytest.approx(0.05, rel=0.1)
    assert reference_fid(1.0, 1.0, 1.0)[0].size == 64
    assert reference_fid(1.0, 1.0, 1e-9)[0].size == 8192


def test_budget_updates_while_typing_relative_snr():
    panel = ReceiverNoiseWidget()
    budget = SNRBudgetPanel(panel)
    assert [section.title for section in budget.sections] == ["Receiver noise is off"]
    assert budget.reference is None
    panel.mode.setCurrentIndex(1)
    panel.snr.setText("20")
    assert budget.reference == (1.0, 0.05, 1e-4)
    assert _budget_rows(budget)[("Receiver setting", "Relative SNR")] == "20"
    assert "Choose or design a phantom" in _budget_text(budget)
    panel.snr.setText("2")
    assert budget.reference[1] == 0.5
    assert _budget_rows(budget)[("Receiver setting", "Relative SNR")] == "2"
    first = budget.noisy_curve.yData.copy()
    assert panel.new_button.isEnabled()
    panel.new_button.click()
    assert not np.array_equal(first, budget.noisy_curve.yData)
    panel.snr.setText("-")
    assert budget.reference is None
    assert budget.noisy_curve.yData is None
    assert "Relative SNR must be positive" in _budget_text(budget)
    panel.snr.setText("0")
    assert budget.reference is None
    assert "positive" in budget.plot_info.text()
    assert "positive" in panel.info.text()
    with pytest.raises(ValueError, match="Relative SNR"):
        panel.settings()


def test_budget_previews_entered_calibration_until_it_is_set():
    context = {"phantom": _pyruvate_phantom(), "nucleus": "C13", "field_strength_t": 7}
    panel = ReceiverNoiseWidget()
    budget = SNRBudgetPanel(panel, lambda: context)
    panel.averages.setValue(4)
    _fill_urea_calibration(panel, "1000", "5")
    assert budget.sections[0].status == "preview — not set"
    assert budget.reference == (1000.0, 5.0, 1e-4)
    rows = _budget_rows(budget)
    assert rows[("Reference phantom", "Single-scan SNR")] == "100"
    assert rows[("Simulated object", "Fully polarized equivalent")] == "3.2 µmol"
    assert rows[("Simulated object", "Whole-object FID SNR")] == "521.7"
    before = budget.noisy_curve.yData.copy()
    panel.capture_calibration(nucleus="C13", field_strength_t=7)
    assert budget.sections[0].status == "set"
    np.testing.assert_array_equal(budget.noisy_curve.yData, before)
    assert _budget_rows(budget)[("Simulated object", "Whole-object FID SNR")] == (
        "521.7"
    )
    # Without a sequence workspace the preview has no nucleus and field.
    unset = ReceiverNoiseWidget()
    _fill_urea_calibration(unset)
    assert "needs the nucleus and field" in _budget_text(SNRBudgetPanel(unset))


def test_budget_context_follows_the_sequence_workspace():
    obj, program = _inputs()
    widget = SequenceSimulationWidget()
    widget.program = program
    widget.object_source.setCurrentIndex(1)
    context = widget.receiver_budget_context()
    assert context["phantom"] is not None
    assert context["program"] is program
    assert context["result"] is None
    assert context["signal_weighting"] == widget._signal_weighting_mode()
    assert context["nucleus"] == widget.nucleus.currentText()
    assert context["field_strength_t"] == widget.field_strength_t.value()
    panel = widget.receiver_noise
    budget = SNRBudgetPanel(panel, widget.receiver_budget_context)
    panel.mode.setCurrentIndex(1)
    panel.snr.setText("50")
    rows = _budget_rows(budget)
    assert rows[("Simulated object", "Whole-object FID SNR")] == "50"
    assert rows[("Sequence readout", "Whole object at 10 kHz")] == "50"
    widget.close()


def test_receiver_tab_lives_between_phantom_b1_and_simulations(qt_application):
    window = BlochSimulatorGUI()
    index = window.receiver_noise_tab_index
    assert index == window.b1_combo_tab_index + 1
    assert window.spoiling_analysis_tab_index == index + 1
    assert window.psf_analysis_tab_index == index + 2
    assert window.kspace_tab_index == index + 3
    assert window.simulation_explorer_tab_index == index + 4
    assert not window.tab_widget.isTabVisible(index)
    window.set_workspace_mode("sequence")
    sequence = window.sequence_simulation_widget
    controls = sequence.receiver_noise
    assert window.receiver_noise_workspace is None
    assert controls.isHidden()
    assert window.tab_widget.isTabVisible(index)
    window.tab_widget.setCurrentIndex(index)
    workspace = window.receiver_noise_workspace
    assert workspace.preview.controls is controls
    assert not sequence.isAncestorOf(controls)
    assert not controls.isHidden()
    controls.mode.setCurrentIndex(1)
    controls.snr.setText("8")
    assert workspace.preview.reference[1] == 0.125
    assert _budget_rows(workspace.preview)[("Receiver setting", "Relative SNR")] == (
        "8"
    )
    window.tab_widget.setCurrentIndex(window.sequence_simulation_tab_index)
    window.set_workspace_mode("free")
    assert not window.tab_widget.isTabVisible(index)
    window.set_workspace_mode("sequence")
    window.tab_widget.setCurrentIndex(index)
    assert window.receiver_noise_workspace is workspace
    assert controls.snr.text() == "8"
    window.close()
    qt_application.processEvents()


def test_std_controls_support_arrow_keys_and_native_spin_buttons(qt_application):
    panel = ReceiverNoiseWidget()
    preview = SNRBudgetPanel(panel)
    panel.mode.setCurrentIndex(1)
    panel.show()
    field = panel.snr
    assert isinstance(field, QDoubleSpinBox)
    assert field.buttonSymbols() == QAbstractSpinBox.UpDownArrows
    field.setText("50")
    field.setFocus()
    QTest.keyClick(field, Qt.Key_Up)
    assert field.value() == 51
    assert field.text() == "51"
    assert preview.reference[1] == 1 / 51
    QTest.keyClick(field, Qt.Key_Down)
    assert field.value() == 50
    option = QStyleOptionSpinBox()
    field.initStyleOption(option)
    arrow = field.style().subControlRect(
        QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxUp, field
    )
    QTest.mouseClick(field, Qt.LeftButton, pos=arrow.center())
    assert field.value() == 51
    assert preview.reference[1] == 1 / 51
    panel.mode.setCurrentIndex(2)
    panel.reference_std.setText("0.05")
    QTest.keyClick(panel.reference_std, Qt.Key_Up)
    assert panel.reference_std.value() == 0.051
    assert preview.reference[1] == 0.051
    panel.close()
    qt_application.processEvents()


@pytest.mark.parametrize("value", [0.01, 1e-12, 1e6, 1e300])
def test_snr_control_preserves_scientific_values_and_saved_state(value):
    panel = ReceiverNoiseWidget()
    panel.mode.setCurrentIndex(1)
    panel.bandwidth_khz.setValue(5)
    panel.snr.setText(str(value))
    settings = panel.settings()
    assert settings.mode == "relative"
    assert settings.snr == value
    assert settings.reference_dwell_s == pytest.approx(2e-4)
    assert panel.snr.value() == value
    panel.snr.stepUp()
    assert panel.snr.value() == pytest.approx(value * 1.1, rel=1e-12, abs=0)
    panel.snr.stepDown()
    assert panel.snr.value() == value
    restored = ReceiverNoiseWidget()
    restored.restore_state(panel.get_state())
    assert restored.settings() == settings


@pytest.mark.parametrize("value", [1e-12, 1e-100, 1e-300])
def test_reference_std_preserves_small_scientific_values(value):
    panel = ReceiverNoiseWidget()
    panel.mode.setCurrentIndex(2)
    panel.reference_std.setText(str(value))
    assert panel.reference_std.value() == value
    panel.reference_std.stepUp()
    panel.reference_std.stepDown()
    assert panel.reference_std.value() == value
    restored = ReceiverNoiseWidget()
    restored.restore_state(panel.get_state())
    assert restored.reference_std.value() == value


def test_relative_mode_shows_only_its_controls_and_old_direct_state_turns_off():
    panel = ReceiverNoiseWidget()
    panel.mode.setCurrentIndex(1)
    assert panel.mode.currentData() == "relative"
    assert panel.snr.isEnabled() and panel.bandwidth_khz.isEnabled()
    assert not panel.amplitude.isEnabled()
    assert "ideal FID amplitude" in panel.info.text()
    panel.mode.setCurrentIndex(2)
    assert not panel.snr.isEnabled() and panel.bandwidth_khz.isEnabled()
    panel.mode.setCurrentIndex(0)
    assert not panel.snr.isEnabled() and not panel.bandwidth_khz.isEnabled()
    restored = ReceiverNoiseWidget()
    restored.restore_state({"mode": "direct", "noise_std": "0.2", "seed": 4})
    assert restored.settings().mode == "off"
    assert restored.snr.value() == 100


def test_snr_arrow_from_zero_has_a_useful_step_and_cannot_go_negative():
    panel = ReceiverNoiseWidget()
    panel.mode.setCurrentIndex(1)
    panel.snr.setValue(0)
    panel.snr.stepDown()
    assert panel.snr.value() == 0
    panel.snr.stepUp()
    assert panel.snr.value() == 0.001


def test_signal_unit_settings_round_trip_and_cancel(tmp_path):
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import QComboBox, QDialog, QWidget
    from blochsimulator.ui.dialogs import SettingsDialog

    filename = str(tmp_path / "units.ini")
    settings = QSettings(filename, QSettings.IniFormat)
    with patch("blochsimulator.ui.main_window.QSettings", return_value=settings):
        window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    widget = window.sequence_simulation_widget
    assert widget.signal_unit == "physical"

    def accept(dialog):
        combo = dialog.findChild(QComboBox, "signal_unit")
        assert combo is dialog.signal_unit_combo
        assert dialog.signal_unit() == "physical"
        combo.setCurrentIndex(combo.findData("snr"))
        return QDialog.Accepted

    with patch.object(SettingsDialog, "exec_", accept):
        window.show_settings(initial_tab="interface")
    assert settings.value("sequence/signal_unit") == "snr"
    assert widget.signal_unit == "snr"
    assert widget.receiver_budget_context()["signal_unit"] == "snr"
    parent = QWidget()
    parent.app_settings = QSettings(filename, QSettings.IniFormat)
    restored = SequenceSimulationWidget(parent)
    assert restored.signal_unit == "snr"

    def cancel(dialog):
        assert dialog.signal_unit() == "snr"
        dialog.signal_unit_combo.setCurrentIndex(0)
        return QDialog.Rejected

    with patch.object(SettingsDialog, "exec_", cancel):
        window.show_settings(initial_tab="interface")
    assert settings.value("sequence/signal_unit") == "snr"
    assert widget.signal_unit == "snr"
    parent.close()
    window.close()


def _result_for_unit_display(widget):
    from blochsimulator.sequence import SequenceCompiler, SequenceSimulationResult
    from blochsimulator.sequence.receiver import apply_receiver_noise

    compiled = SequenceCompiler().compile(widget.program)
    count = compiled.adc_times_s.size
    clean = SequenceSimulationResult(
        signal=np.full(count, 1 + 0.5j),
        adc_times_s=compiled.adc_times_s,
        adc_gradient_moment_cyc_per_m=compiled.adc_gradient_moment_cyc_per_m,
        final_magnetization=np.zeros(widget.phantom.shape + (3,)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        pool_names=("Pyruvate", "Lactate"),
        species_signal=np.full((2, count), 0.5 + 0.25j),
        metadata={
            "signal_weighting": "voxel_volume",
            "receiver_acquisition": receiver_acquisition_metadata(
                widget.program,
                nucleus="C13",
                field_strength_t=7.0,
                phantom=widget.phantom,
                signal_weighting="voxel_volume",
            ),
        },
    )
    calibration = ReceiverCalibration(100.0, 1.0, 1000.0, 8.0, "C13", 7.0, dwell_s=1e-4)
    return apply_receiver_noise(
        clean, ReceiverNoise("calibrated", calibration=calibration)
    )


def test_signal_unit_redraw_preserves_result_and_run_state():
    from blochsimulator.sequence.signal_units import signal_display

    widget = SequenceSimulationWidget()
    widget.phantom, widget.program = _inputs()
    widget.acquisition = widget.acquisition_frames = widget.acquisition_volumes = None
    widget.spectroscopic_acquisition = widget.spiral_acquisition = None
    result = _result_for_unit_display(widget)
    original = result.signal.copy()
    widget._finished(result, record_run=False)
    animation = object()
    widget.magnetization_animation = animation
    widget._active_session_run_id = "keep-this-run"
    for unit in ("snr", "physical", "snr"):
        with patch.object(widget.simulator, "simulate_sequence") as simulate:
            with patch.object(widget, "_register_session_simulation_run") as register:
                widget.set_signal_unit(unit)
        simulate.assert_not_called()
        register.assert_not_called()
        display = signal_display(result, unit)
        axis = widget.signal_plot.getAxis("left")
        assert axis.labelText == display.axis_label
        assert axis.labelUnits == (display.axis_units or "")
        curves = {curve.name(): curve for curve in widget.signal_plot.listDataItems()}
        np.testing.assert_allclose(
            curves["Real"].getData()[1], display.scale(original).real
        )
        np.testing.assert_allclose(
            curves["|Pyruvate|"].getData()[1],
            np.abs(display.scale(result.species_signal)[0]),
        )
        assert widget.result is result
        assert widget.magnetization_animation is animation
        assert widget._active_session_run_id == "keep-this-run"
        np.testing.assert_array_equal(result.signal, original)
    widget._show_live_signal(original, result.adc_times_s, len(original), "live")
    assert widget.signal_plot.getAxis("left").labelUnits == "a.u."
    widget.close()


@pytest.mark.parametrize("mode", ["relative", "calibrated"])
def test_snr_reference_plot_normalizes_curves_and_sigma_lines(mode):
    controls = ReceiverNoiseWidget()
    context = {"signal_unit": "physical"}
    panel = SNRBudgetPanel(controls, lambda: context)
    if mode == "relative":
        controls.mode.setCurrentIndex(1)
        controls.snr.setText("100")
    else:
        _fill_urea_calibration(controls)
    panel.refresh()
    amplitude, std, dwell = panel.reference
    _, ideal, noisy = reference_fid(amplitude, std, dwell, seed=controls.seed.value())
    context["signal_unit"] = "snr"
    panel.refresh()
    np.testing.assert_allclose(panel.ideal_curve.getData()[1], ideal / std)
    np.testing.assert_allclose(panel.noisy_curve.getData()[1], noisy / std)
    assert [line.value() for line in panel.sigma_lines] == [1, -1]
    assert panel.plot.getAxis("left").labelText == "Signal / σ"
    context["signal_unit"] = "physical"
    panel.refresh()
    np.testing.assert_allclose(panel.noisy_curve.getData()[1], noisy)
    assert [line.value() for line in panel.sigma_lines] == [std, -std]
    panel.close()
    controls.close()


def test_csi_units_apply_to_both_spectrum_and_split_fid():
    from blochsimulator.sequence import SpectroscopicAcquisition
    from blochsimulator.sequence.result import SequenceSimulationResult
    from blochsimulator.sequence.signal_units import signal_display

    widget = SequenceSimulationWidget()
    csi = SpectroscopicAcquisition(
        (4, 3),
        (0.2, 0.2),
        16,
        1e-4,
        tuple((x, y) for y in range(3) for x in range(4)),
    )
    widget.spectroscopic_acquisition = csi
    time = np.concatenate(
        [event * 0.02 + csi.spectral_time_s for event in range(csi.num_encodings)]
    )
    signal = np.random.default_rng(8).normal(size=csi.num_samples).astype(complex)
    widget.result = SequenceSimulationResult(
        signal=signal,
        adc_times_s=time,
        final_magnetization=np.zeros((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        metadata={
            "receiver_noise": {
                "mode": "calibrated",
                "noise_std_per_quadrature": [2.0],
                "calibration_derived": {"target_thermal_polarization": 6e-6},
            }
        },
    )
    widget._configure_spectroscopy_selectors()
    for unit in ("snr", "physical"):
        widget.set_signal_unit(unit)
        display = signal_display(widget.result, unit)
        expected = csi.reconstruct_spectra(
            display.scale(signal), norm=display.image_norm
        )
        expected *= display.spectrum_scale(csi.spectral_points)
        x, y = widget.spectrum_x_selector.value(), widget.spectrum_y_selector.value()
        np.testing.assert_allclose(
            widget.signal_plot.listDataItems()[0].getData()[1], np.abs(expected[y, x])
        )
        assert widget.signal_plot.getAxis("left").labelText == display.axis_label
        widget.split_signal_source.setCurrentText("FID")
        fid = csi.reconstruct_spatial(display.scale(signal), norm=display.image_norm)
        np.testing.assert_allclose(
            widget.split_signal_plot.listDataItems()[0].getData()[1], np.abs(fid[y, x])
        )
        assert widget.split_signal_plot.getAxis("left").labelUnits == (
            display.axis_units or ""
        )
        widget.split_signal_source.setCurrentText("Spectrum")
    widget.close()


@pytest.mark.parametrize("trajectory", ["cartesian", "frames", "volume", "spiral"])
def test_image_units_follow_frame_and_volume_selection(trajectory, tmp_path):
    import runpy
    from pathlib import Path
    from blochsimulator.sequence.signal_units import (
        display_result,
        noise_only_result,
        realized_noise_std,
        signal_display,
    )

    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    if trajectory == "volume":
        generate = runpy.run_path(
            str(
                Path(__file__).parents[1]
                / "sequences/scripts/generate_3d_bssfp_spectral_selective.py"
            )
        )["main"]
        path = tmp_path / "units_3d.seq"
        generate(
            write_seq=True,
            seq_filename=str(path),
            n_read=4,
            n_phase=2,
            n_partition=2,
            n_repetition=2,
            dummy_repetitions=0,
            use_alpha_half=False,
            target_tr=8e-3,
        )
        widget._load_pulseq_path(path)
    else:
        widget.sequence_live_preview.setChecked(True)
        widget.sequence_source.setCurrentIndex(1)
        widget.read_matrix.setValue(8)
        widget.phase_matrix.setValue(8)
        widget.epi_slice_count.setValue(1 if trajectory == "cartesian" else 2)
        if trajectory == "spiral":
            widget.epi_spiral_turns.setValue(4.0)
            widget.epi_readout_trajectory.setCurrentText("Spiral")
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(2)
    widget._build_phantom()
    result = _result_for_unit_display(widget)
    widget.result = result
    widget._configure_frame_selector()
    for unit in ("snr", "physical"):
        widget.set_signal_unit(unit)
        display = signal_display(result, unit)
        shown = display_result(result, display)
        # Inspect every single frame and the montage, with the same raw result.
        for index in range(widget.frame_selector.count()):
            widget.frame_selector.setCurrentIndex(index)
            selected = widget.frame_selector.currentData()
            assert "unavailable" not in widget.reconstruction_info.text()
            assert display.description in widget.reconstruction_info.text()
            if selected < 0:
                continue
            if trajectory == "spiral":
                image = widget.spiral_acquisition.reconstruct(shown, selected)
                if display.snr:
                    noise = widget.spiral_acquisition.reconstruct(
                        noise_only_result(shown), selected
                    )
                    image /= realized_noise_std(noise)
                    assert "estimated per frame" in widget.reconstruction_info.text()
            elif trajectory == "volume":
                volume = widget.acquisition_volumes.reconstruct(
                    shown, selected, norm=display.image_norm
                )
                image = volume[widget.acquisition_volumes.partition_matrix // 2]
            elif widget.acquisition_frames is not None:
                image = widget.acquisition_frames.reconstruct(
                    shown, selected, norm=display.image_norm
                )
            else:
                image = shown.reconstruct_cartesian(
                    widget.acquisition, norm=display.image_norm
                )
            np.testing.assert_allclose(
                widget.reconstruction_view.image, np.abs(image).T
            )
    assert widget.result is result
    widget.close()


def _fill_saturated_reference(panel):
    """Urea reference at 60 deg, 4 averages after 2 dummy scans, read at 0.5 ms."""
    _fill_urea_calibration(panel, "1000", "5")
    panel.averages.setValue(4)
    panel.flip_angle.setValue(60)
    panel.reference_scan_box.setChecked(True)
    panel.reference_t1.setValue(1.2)
    panel.reference_dummy_scans.setValue(2)
    panel.reference_t2star.setValue(20)
    panel.reference_delay.setValue(0.5)


def _coil_fields():
    shape, fov = (16, 16, 16), (0.08,) * 3
    return (
        create_b1_preset("birdcage_cp", shape, fov),
        create_b1_preset("surface_loop", shape, fov, kind="receive"),
    )


def test_reference_scan_model_corrects_the_calibration_and_round_trips():
    context = {"phantom": _pyruvate_phantom(), "nucleus": "C13", "field_strength_t": 7}
    panel = ReceiverNoiseWidget()
    budget = SNRBudgetPanel(panel, lambda: context)
    _fill_saturated_reference(panel)
    # Repeated excitations with T1 need the TR before anything is computed.
    assert "Enter the reference TR" in panel.reference_info.text()
    assert "Complete the reference scan model" in panel.info.text()
    panel.reference_tr.setValue(1.0)
    expected = reference_signal_efficiency(
        ReferenceScan(
            volume_ml=50,
            flip_angle_deg=60,
            averages=4,
            dummy_scans=2,
            repetition_time_s=1.0,
            t1_s=1.2,
            t2star_s=0.02,
            amplitude_delay_s=5e-4,
        )
    ).efficiency
    assert expected < 0.7
    assert f"η = {expected:.4g}" in panel.reference_info.text()
    # The geometry only matters inside coil fields.
    assert not panel.reference_geometry.isEnabled()
    assert "reference T2* 20 ms" in budget.plot_info.text()
    rows = _budget_rows(budget)
    assert rows[("Reference phantom", "Reference signal fraction")] == (
        f"{expected:.4g}"
    )
    panel.capture_calibration(nucleus="C13", field_strength_t=7)
    calibration = panel.settings().calibration
    assert calibration.signal_efficiency == pytest.approx(expected, rel=1e-12)
    assert calibration.reference_scan["t1_s"] == 1.2
    assert calibration.reference_scan["coil_model"] is None
    # Without the model the calibration assumes sin(alpha) of the ideal FID.
    ideal = replace(calibration, signal_efficiency=None, reference_scan=None)
    assert calibration.noise_std(1e-4) == pytest.approx(
        ideal.noise_std(1e-4) * expected / np.sin(np.deg2rad(60))
    )
    assert "per ideal 90° FID" in panel.info.text()
    restored = ReceiverNoiseWidget()
    restored.restore_state(panel.get_state())
    assert restored.reference_scan_box.isChecked()
    assert restored.settings().calibration == calibration
    panel.reference_t1.setValue(1.0)
    with pytest.raises(ValueError, match="changing"):
        panel.settings()
    panel.reference_t1.setValue(1.2)
    panel.settings()
    panel.reference_scan_box.setChecked(False)
    assert panel.reference_info.text() == ""
    with pytest.raises(ValueError, match="changing"):
        panel.settings()
    # Projects without the reference model restore it switched off.
    old = panel.get_state()
    for key in [key for key in old if key.startswith("reference_")]:
        if key != "reference_std":
            del old[key]
    restored.restore_state({**old, "calibration": ideal.to_metadata()})
    assert not restored.reference_scan_box.isChecked()
    assert restored.settings().calibration == ideal


def test_reference_scan_samples_the_coil_fields_at_the_phantom_position():
    tx_field, rx_field = _coil_fields()
    panel = ReceiverNoiseWidget()
    _fill_saturated_reference(panel)
    panel.reference_tr.setValue(1.0)
    panel.reference_use_coil.setChecked(True)
    assert "Define a B1 field" in panel.reference_info.text()
    with pytest.raises(ValueError, match="Define a B1 field"):
        panel.capture_calibration(nucleus="C13", field_strength_t=7)
    fields = [tx_field, rx_field]
    panel.coil_fields = lambda: tuple(fields)
    panel.coil_fields_changed()
    assert panel.reference_geometry.isEnabled()
    assert not panel.reference_diameter.isEnabled()
    panel.reference_position[0].setValue(10)
    scan = ReferenceScan(
        volume_ml=50,
        flip_angle_deg=60,
        averages=4,
        dummy_scans=2,
        repetition_time_s=1.0,
        t1_s=1.2,
        t2star_s=0.02,
        amplitude_delay_s=5e-4,
        position_m=(0.01, 0.0, 0.0),
    )
    expected = reference_signal_efficiency(scan, tx_field, rx_field)
    assert expected.efficiency < 0.9 * expected.homogeneous
    assert f"η = {expected.efficiency:.4g}" in panel.reference_info.text()
    assert "Rx Circular surface loop" in panel.reference_info.text()
    panel.capture_calibration(nucleus="C13", field_strength_t=7)
    calibration = panel.settings().calibration
    assert calibration.signal_efficiency == pytest.approx(expected.efficiency)
    assert calibration.reference_scan["coil_model"] == expected.coil_model
    assert calibration.reference_scan["position_m"] == [0.01, 0.0, 0.0]
    panel.reference_geometry.setCurrentIndex(1)
    assert panel.reference_diameter.isEnabled()
    with pytest.raises(ValueError, match="changing"):
        panel.settings()
    panel.reference_geometry.setCurrentIndex(0)
    panel.settings()
    # Editing the coil in the B1 workspace invalidates the calibration.
    fields[1] = create_b1_preset("uniform", (16, 16, 16), (0.08,) * 3, kind="receive")
    panel.coil_fields_changed()
    with pytest.raises(ValueError, match="changing"):
        panel.settings()


def test_main_window_feeds_the_b1_workspace_coil_to_the_reference_model():
    tx_field, rx_field = _coil_fields()
    window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    panel = window.sequence_simulation_widget.receiver_noise
    assert panel.coil_fields() == (None, None)
    _fill_saturated_reference(panel)
    panel.reference_tr.setValue(1.0)
    panel.reference_use_coil.setChecked(True)
    assert "Define a B1 field" in panel.reference_info.text()
    window._ensure_b1_workspace(window.b1_tab_index)
    window.b1_widget.tx_editor.set_field(tx_field)
    window.b1_widget.rx_editor.set_field(rx_field)
    tx, rx = panel.coil_fields()
    assert tx is window.b1_widget.tx_field and rx is window.b1_widget.rx_field
    assert "Rx Circular surface loop" in panel.reference_info.text()
    window.close()


def test_run_rejects_a_coil_calibration_for_an_object_without_b1_maps(monkeypatch):
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.program = _inputs()[1]
    panel = widget.receiver_noise
    panel.coil_fields = _coil_fields
    _fill_saturated_reference(panel)
    panel.reference_tr.setValue(1.0)
    panel.reference_use_coil.setChecked(True)
    widget._capture_noise_calibration()
    monkeypatch.setattr(widget, "_ensure_current_generated_sequence", lambda: True)
    with patch.object(widget, "_show_receiver_noise_error") as error:
        assert widget._run() is False
    assert "no B1 maps" in error.call_args.args[0]
    assert widget.worker is None
    widget.close()


def _click_open_receiver_noise(dialog):
    (button,) = [b for b in dialog.buttons() if b.text() == "Open Receiver / Noise"]
    button.click()


def test_receiver_noise_error_links_to_the_receiver_tab(qt_application):
    window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    widget = window.sequence_simulation_widget
    widget.object_source.setCurrentIndex(1)
    widget.program = _inputs()[1]
    _fill_urea_calibration(widget.receiver_noise)
    widget._capture_noise_calibration()
    widget.field_strength_t.setValue(widget.field_strength_t.value() + 1)
    widget._ensure_current_generated_sequence = lambda: True
    shown = []

    def exec_(dialog):
        shown.append(dialog.text())
        _click_open_receiver_noise(dialog)

    with patch("blochsimulator.ui.sequence_simulation_widget.QMessageBox.exec_", exec_):
        assert widget._run() is False
    assert len(shown) == 1 and "Receiver / Noise tab" in shown[0]
    assert widget.worker is None
    assert window.tab_widget.currentIndex() == window.receiver_noise_tab_index
    window.close()
    qt_application.processEvents()
