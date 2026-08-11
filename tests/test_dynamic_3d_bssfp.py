import runpy
from pathlib import Path

import numpy as np
import pytest

pypulseq = pytest.importorskip("pypulseq")

from blochsimulator.sequence import (
    AcquisitionDimensions,
    SequenceCompiler,
    infer_cartesian_acquisition_frames,
    infer_cartesian_acquisition_volumes,
    load_pulseq,
)


DYNAMIC_BSSFP_MAIN = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "sequences"
        / "scripts"
        / "generate_3d_bssfp_dynamic.py"
    )
)["main"]


def test_dynamic_3d_bssfp_uses_pulseq_frame_and_partition_labels(tmp_path):
    sequence = DYNAMIC_BSSFP_MAIN(
        n_read=4,
        n_phase=2,
        n_partition=2,
        n_repetition=2,
        rf_frequency_offsets_hz=(-245.0, 735.0),
        dummy_repetitions=0,
    )
    ok, errors = sequence.check_timing()
    assert ok, errors

    path = tmp_path / "dynamic_3d_bssfp.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)
    dimensions = AcquisitionDimensions.from_program(program)

    assert dimensions.repetition_indices == (0, 0, 0, 0, 1, 1, 1, 1)
    assert dimensions.partition_indices == (0, 0, 1, 1, 0, 0, 1, 1)
    assert sorted(
        {round(event.frequency_offset_hz) for event in program.rf_events}
    ) == [
        -245,
        735,
    ]
    assert sorted(
        {round(event.frequency_offset_hz) for event in program.adc_events}
    ) == [
        -245,
        735,
    ]
    definitions = program.metadata["definitions"]
    assert definitions["DynamicFrames"] == 2
    assert definitions["RFPulseType"] == "slr"
    assert definitions["RFPulseFile"].endswith("rfpulses/SLR_sharpness_5.txt")
    assert definitions["EndImageSpoilerCyclesPerFOV"] == pytest.approx(4.0)
    assert definitions["EndImageSpoilerDuration"] == pytest.approx(1e-3)
    assert definitions["EndImageSpoilerAxes"] == "xyz"

    rf_magnitudes = np.abs(program.rf_events[0].samples_hz)
    assert rf_magnitudes.size > 100
    assert not np.allclose(rf_magnitudes, rf_magnitudes[0])


def test_dynamic_3d_bssfp_adds_configurable_spoiler_after_each_volume(tmp_path):
    fov = (0.08, 0.06, 0.04)
    sequence = DYNAMIC_BSSFP_MAIN(
        fov=fov,
        n_read=4,
        n_phase=2,
        n_partition=1,
        n_repetition=2,
        dummy_repetitions=0,
        use_alpha_half=False,
        end_image_spoiler_cycles_per_fov=3.5,
        end_image_spoiler_duration=2e-3,
    )
    path = tmp_path / "dynamic_3d_bssfp_spoiler.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)

    definitions = program.metadata["definitions"]
    spoiler_end_times = np.asarray(
        definitions["EndImageSpoilerEndTimes"], dtype=float
    ).reshape(-1)
    ideal_spoiler_end_times = np.asarray(
        definitions["IdealSpoilerEndTimes"], dtype=float
    ).reshape(-1)
    assert spoiler_end_times.size == 2
    assert ideal_spoiler_end_times == pytest.approx(spoiler_end_times)
    assert SequenceCompiler().compile(
        program
    ).transverse_crush_times_s == pytest.approx(spoiler_end_times)
    for spoiler_end in spoiler_end_times:
        events = [
            event
            for event in program.gradient_events
            if np.isclose(event.end_s, spoiler_end, rtol=0.0, atol=1e-9)
        ]
        assert {event.axis for event in events} == {"x", "y", "z"}
        for event in events:
            axis_index = "xyz".index(event.axis)
            moment = np.sum(event.samples_hz_per_m) * event.raster_s
            assert moment * fov[axis_index] == pytest.approx(3.5, abs=2e-5)


def test_dynamic_3d_bssfp_spoiler_can_be_disabled():
    sequence = DYNAMIC_BSSFP_MAIN(
        n_read=2,
        n_phase=1,
        n_partition=1,
        n_repetition=1,
        dummy_repetitions=0,
        use_alpha_half=False,
        end_image_spoiler_cycles_per_fov=0.0,
    )

    assert sequence.definitions["EndImageSpoilerEndTimes"] == []
    assert sequence.definitions["IdealSpoilerEndTimes"] == []


def test_dynamic_3d_bssfp_can_encode_readout_on_scanner_z(tmp_path):
    sequence = DYNAMIC_BSSFP_MAIN(
        n_read=4,
        n_phase=2,
        n_partition=2,
        n_repetition=1,
        dummy_repetitions=0,
        use_alpha_half=False,
        encoding_axes=("+z", "+y", "-x"),
    )
    path = tmp_path / "dynamic_3d_bssfp_read_z.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)
    compiled = SequenceCompiler().compile_acquisition(program)
    frames = infer_cartesian_acquisition_frames(program, compiled=compiled)
    volumes = infer_cartesian_acquisition_volumes(
        program, compiled=compiled, frames=frames
    )

    assert volumes.encoding_frame.axis_codes == ("+z", "+y", "-x")
    assert program.metadata["definitions"]["ReadoutAxis"] == "+z"


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_dynamic_3d_bssfp_rejects_invalid_frame_count(value):
    with pytest.raises(ValueError, match="n_repetition"):
        DYNAMIC_BSSFP_MAIN(n_repetition=value)
