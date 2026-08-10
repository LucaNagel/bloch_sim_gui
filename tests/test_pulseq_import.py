import numpy as np
import pytest
import runpy
from pathlib import Path

import blochsimulator.sequence.pulseq as pulseq_import_module

pypulseq = pytest.importorskip("pypulseq")

from blochsimulator import BlochSimulator
from blochsimulator.phantom import PhantomFactory
from blochsimulator.sequence import (
    ADCEvent,
    AcquisitionDimensions,
    GradientEvent,
    RFEvent,
    SequenceCompiler,
    UnsupportedPulseqVersionError,
    infer_cartesian_acquisition,
    infer_cartesian_acquisition_frames,
    load_pulseq,
)


EXAMPLE_MAIN = runpy.run_path(
    str(Path(__file__).parents[1] / "sequences" / "scripts" / "generate_epi.py")
)["main"]


def _write_reference_sequence(path):
    system = pypulseq.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=0,
        rf_dead_time=0,
        adc_dead_time=0,
    )
    sequence = pypulseq.Sequence(system)
    rf = pypulseq.make_block_pulse(
        flip_angle=np.pi / 6,
        duration=100e-6,
        freq_offset=25.0,
        phase_offset=0.1,
        system=system,
    )
    gradient = pypulseq.make_trapezoid(
        "x",
        amplitude=1000.0,
        rise_time=100e-6,
        flat_time=400e-6,
        fall_time=100e-6,
        system=system,
    )
    adc = pypulseq.make_adc(
        num_samples=4,
        dwell=50e-6,
        delay=100e-6,
        freq_offset=10.0,
        phase_offset=0.2,
        system=system,
    )
    sequence.add_block(rf)
    sequence.add_block(gradient, adc)
    sequence.write(str(path))
    return sequence, gradient


def test_load_pulseq_preserves_events_duration_and_adc_times(tmp_path):
    path = tmp_path / "reference.seq"
    reference, gradient = _write_reference_sequence(path)
    program = load_pulseq(path)
    assert program.version == "1.5.0"
    assert len(program.rf_events) == 1
    assert len(program.gradient_events) == 1
    assert len(program.adc_events) == 1
    assert isinstance(program.rf_events[0], RFEvent)
    assert isinstance(program.gradient_events[0], GradientEvent)
    assert isinstance(program.adc_events[0], ADCEvent)
    assert program.duration_s == pytest.approx(reference.duration()[0], abs=1e-12)

    compiled = SequenceCompiler().compile(program)
    expected_adc, _ = reference.adc_times()
    assert np.allclose(compiled.adc_times_s, expected_adc, atol=1e-12)
    area = np.sum(compiled.gradient_hz_per_m[:, 0] * compiled.dt_s)
    assert area == pytest.approx(gradient.area, rel=1e-10, abs=1e-12)


def test_load_pulseq_reuses_identical_gradient_rasterizations(tmp_path, monkeypatch):
    system = pypulseq.Opts(max_grad=1e6, max_slew=1e12)
    sequence = pypulseq.Sequence(system)
    gradient = pypulseq.make_trapezoid(
        "x",
        amplitude=1200.0,
        rise_time=40e-6,
        flat_time=80e-6,
        fall_time=40e-6,
        system=system,
    )
    for _ in range(12):
        sequence.add_block(gradient)
    path = tmp_path / "repeated_gradient.seq"
    sequence.write(str(path))
    original = pulseq_import_module._gradient_samples
    calls = []

    def counting_gradient_samples(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        pulseq_import_module, "_gradient_samples", counting_gradient_samples
    )

    program = load_pulseq(path)

    assert len(program.gradient_events) == 12
    assert len(calls) == 1
    assert all(
        np.array_equal(
            event.samples_hz_per_m,
            program.gradient_events[0].samples_hz_per_m,
        )
        for event in program.gradient_events
    )


def test_load_pulseq_defaults_missing_optional_ppm_fields_to_zero(
    tmp_path, monkeypatch
):
    path = tmp_path / "legacy_offsets.seq"
    _write_reference_sequence(path)
    original_get_block = pypulseq.Sequence.get_block

    def get_block_without_ppm(sequence, block_index):
        block = original_get_block(sequence, block_index)
        for event_name in ("rf", "adc"):
            event = getattr(block, event_name, None)
            if event is not None:
                for field in ("freq_ppm", "phase_ppm"):
                    if hasattr(event, field):
                        delattr(event, field)
        return block

    monkeypatch.setattr(pypulseq.Sequence, "get_block", get_block_without_ppm)
    program = load_pulseq(path)

    assert program.rf_events[0].frequency_offset_hz == pytest.approx(25.0)
    assert program.rf_events[0].phase_offset_rad == pytest.approx(
        0.1 + 2 * np.pi * 25.0 * program.rf_events[0].raster_s / 2
    )
    assert program.adc_events[0].frequency_offset_hz == pytest.approx(10.0)
    assert program.adc_events[0].phase_offset_rad == pytest.approx(
        0.2 + 2 * np.pi * 10.0 * (100e-6 + 50e-6 / 2)
    )


def test_pulseq_to_adc_signal_end_to_end(tmp_path):
    path = tmp_path / "reference.seq"
    _write_reference_sequence(path)
    program = load_pulseq(path)
    phantom = PhantomFactory.uniform((2, 2, 2), (0.02, 0.02, 0.02), 1.0, 0.1)
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    assert result.signal.shape == (4,)
    assert result.final_magnetization.shape == (2, 2, 2, 3)
    assert np.all(np.isfinite(result.signal))


def test_pulseq_1_5_1_rejected_before_parser(tmp_path):
    path = tmp_path / "future.seq"
    path.write_text(
        "[VERSION]\nmajor 1\nminor 5\nrevision 1\n\n[BLOCKS]\n",
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedPulseqVersionError, match="1.5.1"):
        load_pulseq(path)


def test_pulseq_trigger_is_retained_as_metadata(tmp_path):
    sequence = pypulseq.Sequence()
    sequence.add_block(pypulseq.make_trigger("physio1", duration=1e-3, delay=0.0))
    path = tmp_path / "trigger.seq"
    sequence.write(str(path))
    with pytest.warns(RuntimeWarning, match="trigger retained"):
        program = load_pulseq(path)
    assert len(program.metadata["triggers"]) == 1


def test_pulseq_numeric_array_string_definitions_are_normalized(tmp_path):
    sequence = pypulseq.Sequence()
    sequence.set_definition("MatrixSize", "[32. 32.  6.]")
    sequence.set_definition("EncodingBasisXYZ", "[1. 0. 0. 0. 1. 0. 0. 0. 1.]")
    sequence.add_block(pypulseq.make_delay(1e-3))
    path = tmp_path / "string_definitions.seq"
    sequence.write(str(path))

    program = load_pulseq(path)

    assert program.metadata["definitions"]["MatrixSize"] == pytest.approx(
        [32.0, 32.0, 6.0]
    )
    assert program.metadata["definitions"]["EncodingBasisXYZ"] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    )


def test_pulseq_labels_define_outer_indices_at_each_adc_event(tmp_path):
    sequence = pypulseq.Sequence()
    adc = pypulseq.make_adc(num_samples=2, dwell=10e-6)
    sequence.add_block(
        pypulseq.make_label("SLC", "SET", 3),
        pypulseq.make_label("REP", "SET", 0),
        adc,
    )
    sequence.add_block(
        pypulseq.make_label("REP", "INC", 1),
        pypulseq.make_label("ECO", "SET", 2),
        adc,
    )
    path = tmp_path / "labels.seq"
    sequence.write(str(path))

    program = load_pulseq(path)
    dimensions = AcquisitionDimensions.from_program(program)
    assert program.metadata["adc_label_values"] == {
        "SLC": (3, 3),
        "REP": (0, 1),
        "ECO": (0, 2),
    }
    assert dimensions.slice_indices == (3, 3)
    assert dimensions.repetition_indices == (0, 1)
    assert dimensions.echo_indices == (0, 2)
    assert program.metadata["labels"][0]["events"][0] == {
        "type": "labelset",
        "name": "SLC",
        "value": 3,
    }


def test_pulseq_extended_gradient_preserves_area(tmp_path):
    system = pypulseq.Opts(max_grad=1e6, max_slew=1e12)
    sequence = pypulseq.Sequence(system)
    gradient = pypulseq.make_extended_trapezoid(
        "z",
        times=np.array([0.0, 10e-6, 30e-6]),
        amplitudes=np.array([0.0, 2000.0, 0.0]),
        system=system,
    )
    sequence.add_block(gradient)
    path = tmp_path / "extended.seq"
    sequence.write(str(path))
    compiled = SequenceCompiler().compile(load_pulseq(path))
    imported_area = np.sum(compiled.gradient_hz_per_m[:, 2] * compiled.dt_s)
    assert imported_area == pytest.approx(gradient.area, rel=1e-12)


def test_pulseq_triangular_gradient_preserves_area(tmp_path):
    system = pypulseq.Opts(max_grad=1e6, max_slew=1e12)
    sequence = pypulseq.Sequence(system)
    gradient = pypulseq.make_trapezoid(
        "y",
        amplitude=2000.0,
        rise_time=50e-6,
        flat_time=0.0,
        fall_time=50e-6,
        system=system,
    )
    assert gradient.flat_time == 0
    sequence.add_block(gradient)
    path = tmp_path / "triangle.seq"
    sequence.write(str(path))
    compiled = SequenceCompiler().compile(load_pulseq(path))
    imported_area = np.sum(compiled.gradient_hz_per_m[:, 1] * compiled.dt_s)
    assert imported_area == pytest.approx(gradient.area, rel=1e-12)


def test_pulseq_multiblock_epi_readout_import(tmp_path):
    system = pypulseq.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
    )
    sequence = pypulseq.Sequence(system)
    for line in range(4):
        gx = pypulseq.make_trapezoid(
            "x",
            amplitude=1000.0 if line % 2 == 0 else -1000.0,
            rise_time=50e-6,
            flat_time=200e-6,
            fall_time=50e-6,
            system=system,
        )
        adc = pypulseq.make_adc(
            num_samples=4,
            dwell=50e-6,
            delay=50e-6,
            system=system,
        )
        sequence.add_block(gx, adc)
        if line < 3:
            gy = pypulseq.make_trapezoid(
                "y",
                amplitude=500.0,
                rise_time=50e-6,
                flat_time=50e-6,
                fall_time=50e-6,
                system=system,
            )
            sequence.add_block(gy)
    path = tmp_path / "epi.seq"
    sequence.write(str(path))
    program = load_pulseq(path)
    compiled = SequenceCompiler().compile(program)
    assert len(program.gradient_events) == 7
    assert len(program.adc_events) == 4
    assert compiled.adc_times_s.size == 16
    assert np.all(np.diff(compiled.adc_times_s) > 0)


def test_generated_epi_infers_one_cartesian_grid(tmp_path):
    path = tmp_path / "generated_epi.seq"
    EXAMPLE_MAIN(
        write_seq=True,
        seq_filename=str(path),
        fov=(0.22, 0.24),
        n_x=8,
        n_y=6,
        flip_angle_deg=30.0,
        slice_thickness=4e-3,
        n_slices=1,
    )
    program = load_pulseq(path)
    compiled = SequenceCompiler().compile(program)
    acquisition = infer_cartesian_acquisition(program, compiled=compiled)

    assert acquisition.read_matrix == 8
    assert acquisition.phase_matrix == 6
    assert acquisition.fov_m == pytest.approx((0.22, 0.24))
    assert acquisition.dwell_s == pytest.approx(4e-6)
    assert acquisition.kx_offset_cells == pytest.approx(0.5)
    assert acquisition.ky_offset_cells == pytest.approx(0.0)
    assert program.metadata["definitions"]["FlipAngleDeg"] == pytest.approx(30.0)
    assert 360.0 * abs(np.sum(compiled.rf_hz * compiled.dt_s)) == pytest.approx(
        30.0, rel=1e-5
    )
    acquisition.validate_gradient_moments(compiled.adc_gradient_moment_cyc_per_m)


def test_generated_multislice_epi_infers_excitation_relative_2d_frames(tmp_path):
    path = tmp_path / "multislice_epi.seq"
    EXAMPLE_MAIN(
        write_seq=True,
        seq_filename=str(path),
        fov=(0.22, 0.24),
        n_x=8,
        n_y=6,
        slice_thickness=4e-3,
        n_slices=3,
    )
    program = load_pulseq(path)
    compiled = SequenceCompiler().compile(program)
    frames = infer_cartesian_acquisition_frames(program, compiled=compiled)

    assert frames.num_frames == 3
    assert frames.varying_axes == ("slice",)
    assert frames.dimensions.source == "rf_frequency_offsets"
    assert [frames.frame_label(index) for index in range(3)] == [
        "slice=0",
        "slice=1",
        "slice=2",
    ]
    assert all(item.read_matrix == 8 for item in frames.acquisitions)
    assert all(item.phase_matrix == 6 for item in frames.acquisitions)
    assert all(
        item.kx_offset_cells == pytest.approx(0.5) for item in frames.acquisitions
    )
    assert all(
        item.ky_offset_cells == pytest.approx(0.0) for item in frames.acquisitions
    )
    assert all(
        origin[0] == pytest.approx(0.0, abs=5e-6)
        and origin[1] == pytest.approx(0.0, abs=5e-6)
        for origin in frames.moment_origins_cyc_per_m
    )


def test_generated_multislice_epi_supports_edge_to_edge_slice_gaps(tmp_path):
    path = tmp_path / "multislice_epi_gap.seq"
    EXAMPLE_MAIN(
        write_seq=True,
        seq_filename=str(path),
        fov=(0.22, 0.24),
        n_x=4,
        n_y=4,
        slice_thickness=4e-3,
        slice_gap=2e-3,
        n_slices=3,
    )
    program = load_pulseq(path)
    definitions = program.metadata["definitions"]

    assert definitions["SliceThickness"] == pytest.approx(4e-3)
    assert definitions["SliceGap"] == pytest.approx(2e-3)
    assert definitions["SliceSpacing"] == pytest.approx(6e-3)
    assert definitions["SlicePositions"] == pytest.approx([-6e-3, 0.0, 6e-3])
    assert definitions["FOV"] == pytest.approx([0.22, 0.24, 16e-3])
    assert definitions["UseSliceLabels"] == pytest.approx(0.0)
    assert bool(definitions["SpoilAfterSlice"]) is True
    assert definitions["SpoilerCyclesPerSlice"] == pytest.approx(8.0)
    assert definitions["SpoilerCyclesPerVoxel"] == pytest.approx(0.0)
    assert definitions["SpoilerDuration"] == pytest.approx(4e-3)
    assert definitions["SpoilerAxes"] == "z"
    assert np.asarray(definitions["SpoilerEndTimes"]).size == 3
    rf_offsets = sorted({event.frequency_offset_hz for event in program.rf_events})
    assert len(rf_offsets) == 3
    assert rf_offsets[1] == pytest.approx(0.0)
    assert rf_offsets[2] - rf_offsets[1] == pytest.approx(rf_offsets[1] - rf_offsets[0])


def test_generated_epi_supports_in_plane_only_spoilers(tmp_path):
    path = tmp_path / "epi_in_plane_spoiler.seq"
    fov = (0.22, 0.24)
    sequence = EXAMPLE_MAIN(
        write_seq=True,
        seq_filename=str(path),
        fov=fov,
        n_x=4,
        n_y=2,
        n_slices=1,
        spoiler_cycles_per_slice=0.0,
        spoiler_cycles_per_voxel=0.5,
        spoiler_duration=2e-3,
    )
    assert sequence.check_timing()[0]
    program = load_pulseq(path)
    definitions = program.metadata["definitions"]

    assert definitions["SpoilerAxes"] == "xy"
    spoiler_end = float(np.asarray(definitions["SpoilerEndTimes"]).reshape(-1)[0])
    events = [
        event
        for event in program.gradient_events
        if np.isclose(event.end_s, spoiler_end, rtol=0.0, atol=1e-9)
    ]
    assert {event.axis for event in events} == {"x", "y"}
    voxel_sizes = {"x": fov[0] / 4, "y": fov[1] / 2}
    for event in events:
        moment = np.sum(event.samples_hz_per_m) * event.raster_s
        assert moment * voxel_sizes[event.axis] == pytest.approx(0.5, abs=2e-5)


def test_generated_multislice_epi_supports_repetitions_and_repetition_time(tmp_path):
    path = tmp_path / "multislice_repeated_epi.seq"
    sequence = EXAMPLE_MAIN(
        write_seq=True,
        seq_filename=str(path),
        n_x=4,
        n_y=4,
        slice_thickness=4e-3,
        n_slices=2,
        n_repetitions=3,
        repetition_time=100e-3,
    )
    program = load_pulseq(path)
    compiled = SequenceCompiler().compile(program)
    frames = infer_cartesian_acquisition_frames(program, compiled=compiled)
    definitions = program.metadata["definitions"]

    assert sequence.duration()[0] == pytest.approx(300e-3)
    assert definitions["Repetitions"] == pytest.approx(3)
    assert definitions["RepetitionTime"] == pytest.approx(100e-3)
    assert definitions["MinimumRepetitionTime"] < definitions["RepetitionTime"]
    assert frames.num_frames == 6
    assert frames.varying_axes == ("slice", "repetition")
    assert frames.dimensions.source == "rf_frequency_offsets_and_repetitions"
    assert [frames.frame_label(index) for index in range(6)] == [
        "slice=0, repetition=0",
        "slice=1, repetition=0",
        "slice=0, repetition=1",
        "slice=1, repetition=1",
        "slice=0, repetition=2",
        "slice=1, repetition=2",
    ]


def test_generated_epi_rejects_negative_slice_gap():
    with pytest.raises(ValueError, match="slice_gap"):
        EXAMPLE_MAIN(slice_gap=-1e-3)


def test_generated_epi_rejects_invalid_repetition_settings():
    with pytest.raises(ValueError, match="flip_angle_deg"):
        EXAMPLE_MAIN(flip_angle_deg=0.0)
    with pytest.raises(ValueError, match="n_repetitions"):
        EXAMPLE_MAIN(n_repetitions=0)
    with pytest.raises(ValueError, match="shorter than the minimum"):
        EXAMPLE_MAIN(
            n_x=4,
            n_y=4,
            n_slices=2,
            n_repetitions=2,
            repetition_time=1e-3,
        )
