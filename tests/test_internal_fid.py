"""Numerical and desktop contracts for the configurable internal FID."""

import numpy as np
import pytest

from blochsimulator.sequence import simulate_reference_sequence
from blochsimulator.sequence.internal_fid import make_internal_fid
from blochsimulator.simulator import BlochSimulator
from blochsimulator.ui.main_window import _capture_widget_state, _restore_widget_state
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget


def test_default_fid_preserves_original_excitation_and_sampling():
    program = make_internal_fid()
    assert program.source == "internal-fid"
    assert program.duration_s == pytest.approx(0.0266)
    assert not program.gradient_events
    rf = program.rf_events[0]
    np.testing.assert_allclose(rf.samples_hz, [250.0])
    assert rf.raster_s == pytest.approx(0.001)
    adc = program.adc_events[0]
    assert adc.num_samples == 256
    assert adc.dwell_s == pytest.approx(100e-6)
    assert adc.start_s == pytest.approx(1.05e-3)


@pytest.mark.parametrize("pulse_type", ["block", "sinc", "gaussian", "slr"])
def test_fid_pulse_shapes_apply_flip_phase_and_frequency(pulse_type):
    program = make_internal_fid(
        rf_pulse_type=pulse_type,
        rf_duration_s=2e-3,
        flip_angle_deg=37.0,
        rf_phase_deg=43.0,
        rf_frequency_offset_hz=123.0,
        receiver_frequency_offset_hz=-12.0,
    )
    rf = program.rf_events[0]
    assert abs(np.sum(rf.samples_hz) * rf.raster_s) * 360 == pytest.approx(37)
    assert rf.end_s == pytest.approx(0.002)
    assert rf.phase_offset_rad == pytest.approx(np.deg2rad(43.0))
    assert rf.frequency_offset_hz == 123.0
    assert program.adc_events[0].frequency_offset_hz == -12.0
    if pulse_type != "block":
        assert np.ptp(rf.samples_hz.real) > 0


@pytest.mark.parametrize("explicit_tr", [0.0, 0.1])
def test_fid_repetition_timing_and_complete_adc_windows(explicit_tr):
    program = make_internal_fid(
        rf_duration_s=2e-3,
        num_samples=17,
        sampling_bandwidth_hz=2000,
        adc_delay_s=3e-3,
        repetitions=3,
        repetition_time_s=explicit_tr,
    )
    acquisition_end = 0.002 + 0.003 + 17 / 2000
    tr = explicit_tr or acquisition_end
    assert len(program.rf_events) == len(program.adc_events) == 3
    assert program.duration_s == pytest.approx(2 * tr + acquisition_end)
    for index, (rf, adc) in enumerate(zip(program.rf_events, program.adc_events)):
        assert rf.start_s == pytest.approx(index * tr)
        np.testing.assert_allclose(
            adc.sample_times_s,
            index * tr + 0.005 + (np.arange(17) + 0.5) / 2000,
        )
        assert adc.end_s + adc.dwell_s / 2 <= (index + 1) * tr + 1e-12


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"repetitions": 0}, "repetitions"),
        ({"num_samples": 1.5}, "num_samples"),
        ({"sampling_bandwidth_hz": 0}, "sampling_bandwidth_hz"),
        ({"adc_delay_s": -1}, "adc_delay_s"),
        ({"repetition_time_s": 0.01}, "TR must be at least 26.6 ms"),
    ],
)
def test_invalid_fid_parameters_are_rejected(parameters, message):
    with pytest.raises(ValueError, match=message):
        make_internal_fid(**parameters)


def test_repeated_fid_accumulates_excitation_without_reset_or_spoiling():
    program = make_internal_fid(
        flip_angle_deg=30,
        num_samples=1,
        repetitions=3,
        repetition_time_s=0.05,
    )
    reference = simulate_reference_sequence(program, t1_s=1e12, t2_s=1e12)
    np.testing.assert_allclose(
        reference.adc_magnetization[:, 0, 1],
        np.sin(np.deg2rad([30, 60, 90])),
        atol=2e-12,
    )
    native = BlochSimulator(use_parallel=False).simulate_sequence_probes(
        program,
        np.zeros((1, 3)),
        np.zeros(1),
        checkpoints_s=[event.end_s for event in program.rf_events],
        t1_s=1e12,
        t2_s=1e12,
    )
    np.testing.assert_allclose(
        native.magnetization[:, 0, 0, 1],
        np.sin(np.deg2rad([30, 60, 90])),
        atol=2e-12,
    )


def test_fid_controls_generate_and_live_refresh_without_stale_probe_program():
    widget = SequenceSimulationWidget()
    original = widget.program
    assert widget.generate_sequence_button.isEnabled()
    assert widget.sequence_live_preview.isEnabled()
    assert not widget.export_pulseq_button.isEnabled()
    widget.fid_rf_pulse_type.setCurrentText("Gaussian")
    widget.fid_flip_angle_deg.setValue(45)
    widget.fid_points.setValue(64)
    widget.fid_bandwidth_khz.setValue(20)
    widget.fid_repetitions.setValue(3)
    widget.fid_repetition_time_ms.setValue(50)
    assert widget._sequence_generation_pending
    assert widget.program is original
    widget.generate_sequence_button.click()
    assert not widget._sequence_generation_pending
    assert len(widget.program.adc_events) == 3
    assert widget.program.adc_events[0].num_samples == 64
    assert widget.program.adc_events[0].dwell_s == pytest.approx(50e-6)
    assert "192 total samples" in widget.fid_sampling_info.text()
    widget.sequence_live_preview.setChecked(True)
    widget.fid_points.setValue(32)
    assert widget.program.adc_events[0].num_samples == 32
    widget.sequence_live_preview.setChecked(False)
    widget.fid_flip_angle_deg.setValue(20)
    widget._apply_probe_defaults_from_program = lambda: None
    assert widget._can_start_probe()
    assert widget.program.metadata["definitions"]["FlipAngleDeg"] == 20


def test_short_fid_tr_preserves_last_valid_program_and_blocks_execution():
    widget = SequenceSimulationWidget()
    original = widget.program
    widget.fid_repetition_time_ms.setValue(1)
    assert not widget._ensure_current_generated_sequence()
    assert "TR must be at least" in widget._generation_error
    assert widget.program is original
    assert widget._sequence_generation_pending
    widget.fid_repetition_time_ms.setValue(30)
    assert widget._ensure_current_generated_sequence()
    assert not widget._sequence_generation_pending


def test_fid_controls_survive_source_switch_and_project_control_restore():
    widget = SequenceSimulationWidget()
    widget.fid_rf_pulse_type.setCurrentText("Sinc")
    widget.fid_rf_sinc_lobes.setValue(5)
    widget.fid_flip_angle_deg.setValue(25)
    widget.fid_points.setValue(99)
    widget.fid_repetitions.setValue(4)
    widget.fid_adc_delay_ms.setValue(2)
    state = _capture_widget_state(widget)
    widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)
    assert widget.fid_group.isHidden()
    widget.sequence_source.setCurrentIndex(widget.INTERNAL_SOURCE)
    assert not widget.fid_group.isHidden()
    assert len(widget.program.adc_events) == 4
    assert widget.program.adc_events[0].num_samples == 99
    restored = SequenceSimulationWidget()
    _restore_widget_state(restored, state)
    assert restored._ensure_current_generated_sequence()
    assert restored.fid_rf_pulse_type.currentText() == "Sinc"
    assert restored.fid_rf_sinc_lobes.value() == 5
    assert restored.program.metadata["definitions"]["FlipAngleDeg"] == 25
    assert restored.program.adc_events[0].num_samples == 99
    assert len(restored.program.adc_events) == 4


def test_fid_accepts_imported_designer_pulse_and_refreshes_its_duration():
    widget = SequenceSimulationWidget()
    widget.set_rf_designer_pulse(
        (np.array([0.01, 0.02, 0.01]), np.array([0, 1e-3, 2e-3])),
        state={"duration": 3.0, "flip_angle": 60.0, "freq_offset": 75},
    )
    widget.fid_rf_pulse_type.setCurrentText("RF Pulse Designer")
    widget.fid_flip_angle_deg.setValue(30)
    assert widget._ensure_current_generated_sequence()
    assert widget.program.rf_events[0].end_s == pytest.approx(0.003)
    assert widget.program.rf_events[0].frequency_offset_hz == 75
    assert not widget.fid_rf_duration_ms.isEnabled()
    assert widget.fid_rf_pulse_type.findText("Spectral-spatial Designer") == -1
