import numpy as np
import pytest

from blochsimulator.sequence import (
    INVERSION_PULSE_TYPES,
    load_pulseq,
    fit_inversion_recovery_t1,
    inversion_recovery_signal,
    make_pulseq_epi,
    make_pulseq_flash,
)
from blochsimulator.simulator import BlochSimulator


def _ir_epi(**overrides):
    parameters = dict(
        matrix=(4, 4),
        inversion_times_s=(0.05, 0.3, 0.8, 2.0),
        repetitions=4,
        repetition_time_s=8.0,
        echo_time_s=10e-3,
    )
    parameters.update(overrides)
    return make_pulseq_epi(**parameters)


def _ir_flash(**overrides):
    parameters = dict(
        matrix=(4, 4),
        n_slices=2,
        slice_gap_m=1e-3,
        inversion_times_s=(0.1, 0.5),
        repetitions=2,
        repetition_time_s=10e-3,
        echo_time_s=4e-3,
        acquisition_interval_s=3.0,
    )
    parameters.update(overrides)
    return make_pulseq_flash(**parameters)


def test_ir_epi_bloch_simulation_recovers_t1(tmp_path):
    sequence = _ir_epi()
    assert sequence.check_timing()[0]
    path = tmp_path / "ir_epi.seq"
    sequence.write(path)
    program = load_pulseq(path)

    # Inversion and excitation alternate; probe Mz just before each excitation.
    excitation_starts = tuple(event.start_s for event in program.rf_events[1::2])
    probes = BlochSimulator(use_parallel=False).simulate_sequence_probes(
        program,
        np.zeros((1, 3)),
        np.zeros(1),
        checkpoints_s=excitation_starts,
        t1_s=1.0,
        t2_s=1.0,
        simulation_timestep_s=10e-6,
    )
    mz = probes.magnetization[:, 0, 0, 2]
    inversion_times = np.asarray(sequence.definitions["InversionTimes"])
    fit = fit_inversion_recovery_t1(inversion_times, mz, magnitude=False)

    assert float(fit.t1_s) == pytest.approx(1.0, rel=5e-3)
    assert float(fit.inversion_efficiency) > 0.95
    assert mz[0] < 0 < mz[-1]


@pytest.mark.parametrize("builder", [_ir_epi, _ir_flash])
def test_inversion_definitions_and_crusher_spoiling(builder):
    sequence = builder()
    definitions = sequence.definitions

    assert sequence.check_timing()[0]
    assert definitions["InversionRecovery"] is True
    assert definitions["InversionPulseType"] == "hyperbolic_secant"
    assert definitions["InversionSelectivity"] == "nonselective"
    assert definitions["InversionTimes"] == pytest.approx(
        definitions["RequestedInversionTimes"], abs=1e-5
    )
    offsets = definitions["InversionSliceTimeOffsets"]
    assert offsets[0] == 0.0
    assert all(later > earlier for earlier, later in zip(offsets, offsets[1:]))
    # Each inversion crusher ends the transverse state before the TI delay.
    repetitions = len(definitions["InversionTimes"])
    assert len(definitions["IdealSpoilerEndTimes"]) >= repetitions


@pytest.mark.parametrize("builder", [_ir_epi, _ir_flash])
def test_inversion_time_minimum_is_reported_and_enforced(builder):
    plan = builder(timing_only=True)
    minimum = plan.minimum_inversion_time_s
    assert minimum > 0

    sequence = builder(inversion_times_s=minimum, repetitions=1)
    assert sequence.definitions["InversionTimes"] == pytest.approx([minimum], abs=1e-6)
    with pytest.raises(ValueError, match="inversion time .* too short"):
        builder(inversion_times_s=0.5 * minimum, repetitions=1)


def test_inversion_times_cycle_and_the_repetition_time_covers_the_longest():
    sequence = _ir_epi(inversion_times_s=(0.1, 0.4), repetitions=5)
    assert sequence.definitions["InversionTimes"] == pytest.approx(
        [0.1, 0.4, 0.1, 0.4, 0.1], abs=1e-5
    )
    with pytest.raises(ValueError, match="repetition_time_s is too short"):
        _ir_epi(inversion_times_s=(0.1, 2.0), repetition_time_s=1.0)


def test_flash_inversion_time_targets_the_centre_line():
    sequence = _ir_flash(inversion_times_s=0.2, repetitions=1, n_slices=1)
    path_rf = [block for block in sequence.block_events if sequence.get_block(block).rf]
    inversion_block, *excitation_blocks = path_rf
    times = np.cumsum(
        [0.0] + [float(duration) for duration in sequence.block_durations.values()]
    )
    block_ids = list(sequence.block_durations)
    inversion = sequence.get_block(inversion_block).rf
    centre_excitation = sequence.get_block(excitation_blocks[2]).rf
    inversion_centre = (
        times[block_ids.index(inversion_block)]
        + inversion.delay
        + float(inversion.shape_dur) / 2
    )
    excitation_centre = (
        times[block_ids.index(excitation_blocks[2])]
        + centre_excitation.delay
        + float(centre_excitation.shape_dur) / 2
    )
    assert excitation_centre - inversion_centre == pytest.approx(0.2, abs=2e-5)


@pytest.mark.parametrize("pulse_type", INVERSION_PULSE_TYPES)
def test_every_inversion_pulse_type_builds(pulse_type):
    sequence = _ir_epi(
        inversion_pulse_type=pulse_type, inversion_times_s=0.5, repetitions=1
    )
    assert sequence.check_timing()[0]
    assert sequence.definitions["InversionPulseType"] == pulse_type


def test_sequences_without_inversion_are_unchanged():
    assert make_pulseq_epi(matrix=(4, 4)).definitions["InversionRecovery"] is False
    with pytest.raises(ValueError):
        _ir_epi(inversion_times_s=(0.1, -0.2))
    with pytest.raises(ValueError):
        _ir_epi(inversion_pulse_type="gaussian")


def test_t1_fit_recovers_complex_and_magnitude_series():
    rng = np.random.default_rng(3)
    inversion_times = np.array([0.05, 0.15, 0.4, 0.8, 1.5, 3.0, 6.0])
    t1 = rng.uniform(0.3, 2.5, size=(6, 5))
    phase = rng.uniform(-np.pi, np.pi, size=t1.shape)
    real = inversion_recovery_signal(
        inversion_times, t1[..., None], a=2.0, inversion_efficiency=0.95
    )
    complex_signal = real * np.exp(1j * phase)[..., None]

    complex_fit = fit_inversion_recovery_t1(inversion_times, complex_signal)
    magnitude_fit = fit_inversion_recovery_t1(inversion_times, np.abs(real))
    np.testing.assert_allclose(complex_fit.t1_s, t1, rtol=1e-4)
    np.testing.assert_allclose(magnitude_fit.t1_s, t1, rtol=1e-4)
    np.testing.assert_allclose(complex_fit.inversion_efficiency, 0.95, atol=1e-3)
    expected_flips = np.sum(real < 0, axis=-1)
    np.testing.assert_array_equal(magnitude_fit.polarity_flips, expected_flips)

    noisy = np.abs(real + rng.normal(scale=0.01, size=real.shape))
    noisy_fit = fit_inversion_recovery_t1(inversion_times, noisy)
    assert np.median(np.abs(noisy_fit.t1_s / t1 - 1)) < 0.03


def test_t1_fit_masks_background_and_validates_input():
    inversion_times = np.array([0.1, 0.5, 1.0, 3.0])
    signal = np.stack(
        [inversion_recovery_signal(inversion_times, 1.0), np.zeros(4)], axis=0
    )
    fit = fit_inversion_recovery_t1(inversion_times, signal, magnitude=False)
    assert float(fit.t1_s[0]) == pytest.approx(1.0, rel=1e-4)
    assert np.isnan(fit.t1_s[1])

    with pytest.raises(ValueError, match="last axis"):
        fit_inversion_recovery_t1(inversion_times, np.zeros((4, 3)))
    with pytest.raises(ValueError, match="three"):
        fit_inversion_recovery_t1(inversion_times[:2], np.zeros(2))


def test_gui_inversion_controls_feed_epi_and_flash(qt_application):
    from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget

    widget = SequenceSimulationWidget()
    try:
        widget.sequence_live_preview.setChecked(True)
        widget.sequence_source.setCurrentIndex(widget.EPI_SOURCE)
        widget.read_matrix.setValue(4)
        widget.phase_matrix.setValue(4)
        widget.epi_repetitions.setValue(4)
        assert not widget.epi_inversion_times_ms.isEnabled()
        widget.epi_inversion_times_ms.setText("50, 500")
        widget.epi_inversion_enabled.setChecked(True)
        qt_application.processEvents()

        assert widget.epi_inversion_times_ms.isEnabled()
        definitions = widget.program.metadata["definitions"]
        assert definitions["InversionRecovery"]
        assert definitions["InversionTimes"] == pytest.approx(
            [0.05, 0.5, 0.05, 0.5], abs=1e-5
        )
        # The interval is raised so the longest TI fits.
        assert widget.epi_repetition_time_ms.value() >= 500.0
        assert "Shortest TI" in widget.epi_inversion_info.text()
        assert "2 complete series" in widget.epi_inversion_info.text()

        widget.epi_inversion_times_ms.setText("50, abc")
        with pytest.raises(ValueError, match="Inversion times"):
            widget._epi_pulseq_parameters()
        widget.epi_inversion_times_ms.setText("50, 500")

        widget.epi_readout_trajectory.setCurrentText("Spiral")
        qt_application.processEvents()
        assert not widget.epi_inversion_enabled.isEnabled()
        assert "inversion_times_s" not in widget._spiral_pulseq_parameters()
        assert not widget._spectral_spatial_schedule_supported("epi")
        widget.epi_readout_trajectory.setCurrentText("Cartesian EPI")
        assert widget._spectral_spatial_schedule_supported("epi")

        widget.sequence_source.setCurrentIndex(widget.FLASH_SOURCE)
        widget.flash_inversion_times_ms.setText("3000")
        widget.flash_inversion_pulse_type.setCurrentIndex(1)
        widget.flash_inversion_enabled.setChecked(True)
        qt_application.processEvents()
        definitions = widget.program.metadata["definitions"]
        assert definitions["InversionPulseType"] == "sinc"
        assert np.atleast_1d(definitions["InversionTimes"])[0] == pytest.approx(
            3.0, abs=1e-5
        )
        # FLASH measures TI to the centre line, so its minimum spans half
        # the phase-encoding train.
        minimum_text = widget.flash_inversion_info.text()
        minimum_ms = float(minimum_text.split("Shortest TI ")[1].split(" ms")[0])
        assert minimum_ms > widget.flash_repetition_time_ms.value()
    finally:
        widget.close()
        widget.deleteLater()
        qt_application.processEvents()
