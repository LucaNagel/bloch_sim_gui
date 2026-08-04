import numpy as np
import pytest

from blochsimulator.sequence import EncodingFrame


def test_read_phase_axes_derive_right_handed_partition_direction():
    frame = EncodingFrame.from_read_phase_axes("z", "y")

    assert frame.axis_codes == ("+z", "+y", "-x")
    assert np.linalg.det(frame.matrix) == pytest.approx(1.0)


def test_encoding_frame_round_trips_vectors_and_metadata():
    frame = EncodingFrame.from_axis_codes(("+z", "+x", "+y"))
    logical = np.asarray([[2.0, -3.0, 4.0], [-1.0, 0.5, 7.0]])

    scanner = frame.encoding_to_scanner(logical)

    assert scanner == pytest.approx(np.asarray([[-3.0, 4.0, 2.0], [0.5, 7.0, -1.0]]))
    assert frame.scanner_to_encoding(scanner) == pytest.approx(logical)
    assert EncodingFrame.from_metadata(frame.to_metadata()) == frame
    assert EncodingFrame.from_definitions(frame.pulseq_definitions()) == frame


def test_encoding_frame_rejects_reflections_and_reused_axes():
    with pytest.raises(ValueError, match="right-handed"):
        EncodingFrame.from_axis_codes(("x", "y", "-z"))
    with pytest.raises(ValueError, match="orthonormal"):
        EncodingFrame.from_axis_codes(("x", "x", "z"))


def test_required_encoding_extents_follow_axis_mapping():
    frame = EncodingFrame.from_axis_codes(("+z", "+y", "-x"))

    assert frame.required_encoding_extents((0.10, 0.20, 0.30)) == pytest.approx(
        (0.30, 0.20, 0.10)
    )
