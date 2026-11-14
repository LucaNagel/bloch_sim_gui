"""Acquisition detection keeps pattern mismatches quiet and reports real errors."""

import re
import warnings

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import ChemicalSpecies, DynamicSpectralPhantom
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import (
    AcquisitionDetectionWarning,
    AcquisitionNotRecognized,
    SequenceCompiler,
    detect_acquisitions,
    load_pulseq,
)
from blochsimulator.sequence import acquisition
from blochsimulator.sequence.pulseq_builders import (
    make_pulseq_bssfp,
    make_pulseq_flash,
    make_pulseq_spiral,
)

FOV_ERROR = "spiral FOV values must be positive and finite"


def _static_phantom():
    return Phantom(
        shape=(1, 1, 1),
        fov=(0.01, 0.01, 0.01),
        t1_map=np.ones((1, 1, 1)),
        t2_map=np.full((1, 1, 1), 0.1),
    )


def _dynamic_phantom():
    shape = (1, 1, 1)
    return DynamicSpectralPhantom(
        shape=shape,
        fov=(0.01, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.zeros(shape),
        nucleus="C13",
    )


def _load(sequence, path, edit=None):
    sequence.write(str(path), v141_compat=True)
    if edit is not None:
        path.write_text(edit(path.read_text()))
    return load_pulseq(path)


def _spiral_with_negative_fov(tmp_path):
    """A spiral file whose FOV definition is corrupt (negative read width)."""
    return _load(
        make_pulseq_spiral(fov_m=(0.032, 0.032), matrix=(16, 16)),
        tmp_path / "spiral.seq",
        edit=lambda text: re.sub(r"(?m)^FOV\s+\S+", "FOV -0.032", text, count=1),
    )


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_real_detection_error_warns_and_is_recorded(tmp_path, dynamic):
    program = _spiral_with_negative_fov(tmp_path)
    simulator = BlochSimulator(use_parallel=False)

    with pytest.warns(AcquisitionDetectionWarning, match=FOV_ERROR):
        if dynamic:
            result = simulator.simulate_dynamic_sequence(program, _dynamic_phantom())
        else:
            result = simulator.simulate_sequence(program, _static_phantom())

    detection = result.metadata["acquisition_detection"]
    assert result.metadata["spiral_acquisition"] is None
    assert detection["errors"] == {"spiral_acquisition": f"ValueError: {FOV_ERROR}"}
    assert "spiral_acquisition" not in detection["not_recognized"]


def test_real_detection_error_can_be_made_fatal(tmp_path):
    program = _spiral_with_negative_fov(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", AcquisitionDetectionWarning)
        with pytest.raises(AcquisitionDetectionWarning, match=FOV_ERROR):
            BlochSimulator(use_parallel=False).simulate_sequence(
                program, _static_phantom()
            )


def test_pattern_mismatch_is_quiet_and_listed(tmp_path):
    program = _load(
        make_pulseq_flash(fov_m=(0.08, 0.06), matrix=(4, 4)), tmp_path / "flash.seq"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", AcquisitionDetectionWarning)
        result = BlochSimulator(use_parallel=False).simulate_sequence(
            program, _static_phantom()
        )

    detection = result.metadata["acquisition_detection"]
    assert result.metadata["cartesian_acquisition"] is not None
    assert detection["errors"] == {}
    assert set(detection["not_recognized"]) == {
        "spectroscopic_acquisition",
        "spiral_acquisition",
    }


def test_detect_acquisitions_separates_mismatch_from_error(monkeypatch, tmp_path):
    program = _load(
        make_pulseq_flash(fov_m=(0.08, 0.06), matrix=(4, 4)), tmp_path / "flash.seq"
    )
    compiled = SequenceCompiler().compile(program)

    def not_recognized(*args, **kwargs):
        raise AcquisitionNotRecognized("no spectral readout")

    def broken(*args, **kwargs):
        raise ValueError("definitions are inconsistent")

    monkeypatch.setattr(acquisition, "infer_spectroscopic_acquisition", not_recognized)
    monkeypatch.setattr(acquisition, "infer_cartesian_acquisition", broken)

    with pytest.warns(AcquisitionDetectionWarning) as caught:
        detected = detect_acquisitions(program, compiled=compiled)

    assert [str(item.message) for item in caught] == [
        "cartesian acquisition detection failed: definitions are inconsistent"
    ]
    assert detected["acquisition_detection"]["not_recognized"][
        "spectroscopic_acquisition"
    ] == ("no spectral readout")
    assert detected["acquisition_detection"]["errors"] == {
        "cartesian_acquisition": "ValueError: definitions are inconsistent"
    }
    assert detected["cartesian_acquisition"] is None
    # The Cartesian frame inference still runs after the single-image error.
    assert "cartesian_acquisition_frames" in (
        detected["acquisition_detection"]["not_recognized"]
    )


def test_not_recognized_is_a_value_error():
    assert issubclass(AcquisitionNotRecognized, ValueError)
    assert issubclass(AcquisitionDetectionWarning, RuntimeWarning)


@pytest.mark.parametrize(
    ("matrix", "reason"),
    [
        ((4, 8, 4), "LIN and PAR labels do not cover the CSI grid"),
        ((8, 8, 8), "ADC samples move through k-space"),
    ],
)
def test_3d_cartesian_matching_csi_counts_is_not_an_error(tmp_path, matrix, reason):
    """MatrixSize nx ny nz with nz read samples matches the CSI counts."""
    program = _load(
        make_pulseq_bssfp(matrix=matrix, dummy_repetitions=0), tmp_path / "b.seq"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", AcquisitionDetectionWarning)
        detected = detect_acquisitions(
            program, compiled=SequenceCompiler().compile(program)
        )

    detection = detected["acquisition_detection"]
    assert detection["errors"] == {}
    assert reason in detection["not_recognized"]["spectroscopic_acquisition"]
    assert detected["cartesian_acquisition_volumes"] is not None
