import numpy as np
import pytest

pypulseq = pytest.importorskip("pypulseq")

from blochsimulator import BlochSimulator
from blochsimulator.phantom import PhantomFactory
from blochsimulator.sequence import (
    ADCEvent,
    GradientEvent,
    RFEvent,
    SequenceCompiler,
    UnsupportedPulseqVersionError,
    load_pulseq,
)


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
