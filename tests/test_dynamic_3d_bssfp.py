import runpy
from pathlib import Path

import numpy as np
import pytest

pypulseq = pytest.importorskip("pypulseq")

from blochsimulator.sequence import AcquisitionDimensions, load_pulseq


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

    rf_magnitudes = np.abs(program.rf_events[0].samples_hz)
    assert rf_magnitudes.size > 100
    assert not np.allclose(rf_magnitudes, rf_magnitudes[0])


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_dynamic_3d_bssfp_rejects_invalid_frame_count(value):
    with pytest.raises(ValueError, match="n_repetition"):
        DYNAMIC_BSSFP_MAIN(n_repetition=value)
