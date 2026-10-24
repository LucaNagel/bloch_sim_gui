import numpy as np
import pytest

from blochsimulator.sequence import (
    INVERSION_PULSE_TYPES,
    INVERSION_RECOVERY_READOUTS,
    InversionRecoveryTimingPlan,
    load_pulseq,
    fit_inversion_recovery_t1,
    inversion_recovery_signal,
    make_pulseq_epi,
    make_pulseq_flash,
    make_pulseq_inversion_recovery,
    slice_acquisition_order,
    slice_inversion_times,
)
from blochsimulator.sequence.inversion_recovery import (
    default_inversion_recovery_repetitions,
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


def _ir(**overrides):
    parameters = dict(
        readout="epi",
        matrix=(4, 4),
        n_slices=3,
        slice_gap_m=1e-3,
        inversion_times_s=(0.05, 0.3, 0.8, 2.0),
        repetition_time_s=8.0,
        echo_time_s=10e-3,
    )
    parameters.update(overrides)
    return make_pulseq_inversion_recovery(**parameters)


def _rf_block_count(sequence):
    return sum(1 for block in sequence.block_events if sequence.get_block(block).rf)


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


def test_global_inversion_delays_each_later_slice_of_the_train():
    sequence = _ir()
    definitions = sequence.definitions

    assert sequence.check_timing()[0]
    assert definitions["Name"] == "ir_epi_2d"
    assert definitions["SequenceFamily"] == "inversion_recovery"
    assert definitions["InversionMode"] == "global"
    assert definitions["InversionSelectivity"] == "nonselective"
    # One non-selective 180° is followed by the excitation of every slice.
    assert _rf_block_count(sequence) == 4 * (1 + 3)
    offsets = np.asarray(definitions["InversionSliceTimeOffsets"])
    assert offsets[0] == 0.0
    assert np.all(np.diff(offsets) > 0)
    times = slice_inversion_times(definitions)
    assert times.shape == (3, 4)
    np.testing.assert_allclose(
        times,
        np.asarray(definitions["InversionTimes"])[None, :] + offsets[:, None],
        atol=1e-6,
    )


def test_slice_selective_inversion_gives_every_slice_the_requested_ti():
    sequence = _ir(inversion_mode="slice_selective", inversion_slab_factor=1.5)
    definitions = sequence.definitions

    assert sequence.check_timing()[0]
    assert definitions["InversionSelectivity"] == "slice_selective"
    assert definitions["InversionSlabThickness"] == pytest.approx(
        1.5 * definitions["SliceThickness"]
    )
    # Every slice has its own inversion right before its excitation.
    assert _rf_block_count(sequence) == 4 * 3 * 2
    np.testing.assert_allclose(
        slice_inversion_times(definitions),
        np.broadcast_to(definitions["RequestedInversionTimes"], (3, 4)),
        atol=1e-5,
    )
    # The slice train no longer overlaps the TI delays, so it is longer.
    global_plan = _ir(timing_only=True)
    selective_plan = _ir(inversion_mode="slice_selective", timing_only=True)
    assert (
        selective_plan.minimum_repetition_time_s
        > 2.5 * global_plan.minimum_repetition_time_s
    )


def test_slice_selective_inversion_limits_are_explained():
    with pytest.raises(ValueError, match="slice-selective inversion needs"):
        _ir(
            inversion_mode="slice_selective",
            inversion_slab_factor=0.1,
            inversion_duration_s=2e-3,
        )
    with pytest.raises(ValueError, match="block inversion has no slice profile"):
        _ir(inversion_mode="slice_selective", inversion_pulse_type="block")


@pytest.mark.parametrize("inversion_mode", ["global", "slice_selective"])
def test_multislice_inversion_bloch_simulation_matches_the_slice_tis(
    tmp_path, inversion_mode
):
    sequence = _ir(
        n_slices=2,
        slice_gap_m=3e-3,
        inversion_mode=inversion_mode,
        inversion_slab_factor=1.5,
    )
    path = tmp_path / "ir.seq"
    sequence.write(path)
    program = load_pulseq(path)
    definitions = sequence.definitions
    positions = np.zeros((2, 3))
    positions[:, 2] = definitions["SlicePositions"]

    # Per repetition: [inversion,] excitation 1, [inversion,] excitation 2.
    group = 3 if inversion_mode == "global" else 4
    starts = [event.start_s for event in program.rf_events]
    excitations = [
        [starts[index] for index in range(slice_rf, len(starts), group)]
        for slice_rf in ((1, 2) if inversion_mode == "global" else (1, 3))
    ]
    probes = BlochSimulator(use_parallel=False).simulate_sequence_probes(
        program,
        positions,
        np.zeros(1),
        checkpoints_s=tuple(sorted(excitations[0] + excitations[1])),
        t1_s=1.0,
        t2_s=0.1,
        simulation_timestep_s=10e-6,
    )
    times = slice_inversion_times(definitions)
    checkpoints = sorted(excitations[0] + excitations[1])
    for slice_index in range(2):
        rows = [checkpoints.index(start) for start in excitations[slice_index]]
        mz = probes.magnetization[rows, slice_index, 0, 2]
        fit = fit_inversion_recovery_t1(times[slice_index], mz, magnitude=False)
        assert float(fit.t1_s) == pytest.approx(1.0, rel=2e-2)
        assert mz[0] < 0 < mz[-1]
    if inversion_mode == "global":
        assert times[1, 0] > times[0, 0]
    else:
        np.testing.assert_allclose(times[0], times[1], atol=1e-6)


def test_slice_orders_place_every_slice_in_the_train():
    assert slice_acquisition_order(4, "interleaved") == (0, 2, 1, 3)
    assert slice_acquisition_order(3, "cyclic", repetition=2, series_length=2) == (
        1,
        2,
        0,
    )

    interleaved = _ir(n_slices=4, slice_order="interleaved", repetitions=1)
    times = slice_inversion_times(interleaved.definitions)[:, 0]
    assert np.all(np.diff(times[[0, 2, 1, 3]]) > 0)

    inversion_times = (0.1, 0.5)
    cyclic = _ir(slice_order="cyclic", inversion_times_s=inversion_times)
    definitions = cyclic.definitions
    assert definitions["SliceOrder"] == "cyclic"
    # One TI series per slice by default.
    assert len(definitions["InversionTimes"]) == 6
    offsets = np.sort(definitions["InversionSliceTimeOffsets"])
    times = slice_inversion_times(definitions)
    requested = np.tile(inversion_times, 3)
    for slice_times in times:
        for value in inversion_times:
            np.testing.assert_allclose(
                np.sort(slice_times[requested == value] - value), offsets, atol=1e-6
            )


def test_gre_readout_inverts_before_every_phase_encoding_line():
    sequence = _ir(
        readout="gre",
        inversion_times_s=(0.05, 0.5),
        echo_time_s=4e-3,
        readout_repetition_time_s=10e-3,
        repetition_time_s=None,
    )
    definitions = sequence.definitions

    assert sequence.check_timing()[0]
    assert definitions["Name"] == "ir_gre_2d"
    assert definitions["InversionInterval"] == "line"
    # Two TIs × four lines, each with one inversion and three excitations.
    assert _rf_block_count(sequence) == 2 * 4 * (1 + 3)
    np.testing.assert_allclose(
        np.diff(definitions["InversionSliceTimeOffsets"]), 10e-3, atol=1e-6
    )


def test_inversion_recovery_wrapper_plans_and_defaults():
    for readout in INVERSION_RECOVERY_READOUTS:
        plan = _ir(readout=readout, echo_time_s=4e-3, timing_only=True)
        assert isinstance(plan, InversionRecoveryTimingPlan)
        assert plan.readout == readout
        assert (plan.minimum_readout_repetition_time_s is None) == (readout == "epi")
        assert plan.minimum_repetition_time_s > 2.0
        assert plan.minimum_inversion_time_s > 0

    turboflash = _ir(readout="turboflash", echo_time_s=4e-3)
    assert turboflash.definitions["FlipAngleDeg"] == pytest.approx(10.0)
    assert turboflash.definitions["InversionInterval"] == "image"
    assert _ir(readout="FLASH", echo_time_s=4e-3).definitions["Name"] == "ir_gre_2d"

    assert default_inversion_recovery_repetitions((0.1, 0.5), n_slices=3) == 2
    assert (
        default_inversion_recovery_repetitions(
            (0.1, 0.5), n_slices=3, slice_order="cyclic"
        )
        == 6
    )
    with pytest.raises(TypeError, match="inversion_interval"):
        _ir(inversion_interval="line")
    with pytest.raises(ValueError, match="readout must be one of"):
        _ir(readout="spiral")
    with pytest.raises(ValueError, match="slice_order must be one of"):
        _ir(slice_order="random")


def test_slice_inversion_times_reshape_and_broadcast_into_the_fit():
    definitions = {
        "InversionTimes": [0.1, 0.5],
        "SlicePositions": [-1e-3, 1e-3],
        "SliceInversionTimes": [0.1, 0.13, 0.5, 0.53],
    }
    np.testing.assert_allclose(
        slice_inversion_times(definitions), [[0.1, 0.5], [0.13, 0.53]]
    )
    np.testing.assert_allclose(
        slice_inversion_times(definitions, slice_axis="last"),
        [[0.1, 0.13], [0.5, 0.53]],
    )
    single = {"InversionTimes": [0.1, 0.5], "SlicePositions": [-1e-3, 1e-3]}
    np.testing.assert_allclose(slice_inversion_times(single), [[0.1, 0.5]] * 2)
    with pytest.raises(ValueError, match="no inversion recovery"):
        slice_inversion_times({})
    with pytest.raises(ValueError, match="slice_axis"):
        slice_inversion_times(definitions, slice_axis="middle")

    # (slice, y, x, TI) images of a global inversion with a 40 ms slice step.
    nominal = np.array([0.05, 0.2, 0.6, 1.5, 4.0])
    times = nominal[None, :] + np.array([0.0, 0.04, 0.08])[:, None]
    t1 = np.array([0.8, 1.2, 2.0])[:, None, None]
    signal = inversion_recovery_signal(
        times[:, None, None, :], t1[..., None] * np.ones((1, 2, 2, 1))
    )
    fit = fit_inversion_recovery_t1(times[:, None, None, :], signal, magnitude=False)
    np.testing.assert_allclose(fit.t1_s, np.broadcast_to(t1, (3, 2, 2)), rtol=1e-4)
    # a - b exp(-(TI + d) / T1) keeps its form with b exp(-d / T1), so the
    # three-parameter fit with the nominal TIs still finds T1 and only reports
    # a smaller apparent inversion efficiency for later slices.
    nominal_fit = fit_inversion_recovery_t1(nominal, signal, magnitude=False)
    np.testing.assert_allclose(
        nominal_fit.t1_s, np.broadcast_to(t1, (3, 2, 2)), rtol=1e-4
    )
    efficiency = nominal_fit.inversion_efficiency[:, 0, 0]
    assert np.all(np.diff(efficiency) < 0)


def test_gui_inversion_recovery_source_builds_exports_and_fits(
    qt_application, tmp_path
):
    import nbformat

    from blochsimulator.notebook_exporter import export_pulseq_generation_notebook
    from blochsimulator.phantom import Phantom
    from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget

    widget = SequenceSimulationWidget()
    try:
        widget.sequence_live_preview.setChecked(True)
        widget.sequence_source.setCurrentIndex(widget.INVERSION_RECOVERY_SOURCE)
        widget.ir_read_matrix.setValue(16)
        widget.ir_phase_matrix.setValue(16)
        widget.ir_read_fov_mm.setValue(40.0)
        widget.ir_phase_fov_mm.setValue(40.0)
        widget.ir_sampling_bandwidth_khz.setValue(25.0)
        qt_application.processEvents()

        assert widget.ir_group.isVisibleTo(widget)
        assert widget._ensure_current_generated_sequence(), widget._generation_error
        definitions = widget.program.metadata["definitions"]
        assert definitions["SequenceFamily"] == "inversion_recovery"
        assert definitions["InversionMode"] == "global"
        assert widget.ir_slice_ti_info.text().startswith("TI + 0 / ")
        assert "4 inversions, 1 complete TI series" in (widget.ir_inversion_info.text())
        minimums = widget._timing_minimums_ms()
        assert widget.ir_line_repetition_time_ms not in minimums
        assert minimums[widget.ir_repetition_time_ms] > 2500.0
        assert not widget.ir_inversion_slab_factor.isEnabled()

        shape = (16, 16, 9)
        phantom = Phantom(
            shape, (0.04, 0.04, 0.02), np.full(shape, 1.0), np.full(shape, 0.08)
        )
        result = widget.simulator.simulate_sequence(
            widget.program, phantom, simulation_timestep_s=20e-6
        )
        widget.phantom = phantom
        widget._finished(result, record_run=False)
        assert "montage of 12 frames" in widget.reconstruction_info.text()
        times = slice_inversion_times(definitions)
        centre = np.zeros(times.shape, dtype=complex)
        for frame in range(widget.acquisition_frames.num_frames):
            image = widget.acquisition_frames.reconstruct(result, frame)
            repetition, slice_index = divmod(frame, times.shape[0])
            centre[slice_index, repetition] = image[6:10, 6:10].mean()
        fit = fit_inversion_recovery_t1(times, np.abs(centre), magnitude=True)
        np.testing.assert_allclose(fit.t1_s, 1.0, rtol=3e-2)

        # The longer selective slice train raises the inversion TR.
        widget.ir_inversion_mode.setCurrentIndex(
            widget.ir_inversion_mode.findData("slice_selective")
        )
        qt_application.processEvents()
        assert widget._ensure_current_generated_sequence(), widget._generation_error
        assert widget.ir_inversion_slab_factor.isEnabled()
        assert widget.ir_repetition_time_ms.value() > 7000.0
        assert widget.ir_slice_ti_info.text() == "Every slice sees the requested TI."

        # Readout defaults follow the readout unless the user changed them.
        widget.ir_inversion_mode.setCurrentIndex(
            widget.ir_inversion_mode.findData("global")
        )
        widget.ir_readout.setCurrentIndex(widget.ir_readout.findData("turboflash"))
        qt_application.processEvents()
        assert widget.ir_flip_angle_deg.value() == pytest.approx(10.0)
        assert widget.ir_echo_time_ms.value() == pytest.approx(4.0)
        assert widget.ir_inversion_times_ms.text() == "250, 500, 1000, 2500"
        assert widget.ir_line_repetition_time_ms.isEnabled()
        assert not widget.ir_epi_readout_strategy.isEnabled()
        widget.ir_flip_angle_deg.setValue(12.0)
        widget.ir_readout.setCurrentIndex(widget.ir_readout.findData("gre"))
        qt_application.processEvents()
        assert widget.ir_flip_angle_deg.value() == pytest.approx(12.0)
        assert widget.ir_inversion_times_ms.text() == "100, 400, 1000, 2500"
        assert widget.ir_line_repetition_time_ms in widget._timing_minimums_ms()
        assert widget._ensure_current_generated_sequence(), widget._generation_error

        kind, parameters, filename = widget._pulseq_export_spec()
        assert (kind, filename) == ("inversion_recovery", "ir_gre_2d.seq")
        assert parameters["readout"] == "gre"
        assert widget._write_pulseq_path(tmp_path / "ir").name == "ir.seq"
        notebook = export_pulseq_generation_notebook(
            tmp_path / "ir.ipynb", kind, parameters, seq_filename=filename
        )
        code = "\n".join(
            cell.source for cell in nbformat.read(notebook, as_version=4).cells
        )
        assert "make_pulseq_inversion_recovery" in code
    finally:
        widget.close()
        widget.deleteLater()
        qt_application.processEvents()
