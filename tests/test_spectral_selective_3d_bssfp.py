import runpy
from pathlib import Path

import numpy as np
import pytest

pypulseq = pytest.importorskip("pypulseq")

from blochsimulator.sequence import AcquisitionDimensions, load_pulseq


SPECTRAL_BSSFP_MAIN = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "sequences"
        / "scripts"
        / "generate_3d_bssfp_spectral_selective.py"
    )
)["main"]


def test_spectral_selective_3d_bssfp_cycles_rf_and_adc_offsets(tmp_path):
    sequence = SPECTRAL_BSSFP_MAIN(
        n_read=4,
        n_phase=2,
        n_partition=2,
        n_repetition=2,
        target_frequency_offsets_hz=(-245.0, 735.0),
        target_metabolite_names=("Py", "Lac"),
        flip_angle_deg=30.0,
        spectral_rf_duration=4e-3,
        spectral_rf_bandwidth_hz=150.0,
        dummy_repetitions=0,
        use_alpha_half=False,
        target_tr=15.3e-3,
    )
    ok, errors = sequence.check_timing()
    assert ok, errors

    path = tmp_path / "spectral_selective_3d_bssfp.seq"
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
    assert definitions["SpectralTargetOffsetsHz"] == pytest.approx([-245.0, 735.0])
    assert definitions["FlipAngleDeg"] == pytest.approx([30.0, 30.0])
    assert definitions["SpectralRFBandwidthHz"] == pytest.approx(150.0)
    assert definitions["SpectralRFDuration"] == pytest.approx(4e-3)
    assert definitions["SpectralRFPulseType"] == "slr"
    assert definitions["SpectralSLRSharpness"] == pytest.approx(1.0)
    assert definitions["SpectralRFPulseFile"].endswith("rfpulses/SLR_sharpness_1.txt")

    rf_magnitudes = np.abs(program.rf_events[0].samples_hz)
    assert rf_magnitudes.size > 100
    assert not np.allclose(rf_magnitudes, rf_magnitudes[0])


def test_spectral_selective_3d_bssfp_defaults_match_skinner_paper():
    sequence = SPECTRAL_BSSFP_MAIN(
        n_phase=2,
        n_partition=2,
        n_repetition=1,
        dummy_repetitions=0,
    )
    ok, errors = sequence.check_timing()
    assert ok, errors

    definitions = sequence.definitions
    assert definitions["TR"] == pytest.approx(6.29e-3)
    assert definitions["TE"] == pytest.approx(3.145e-3)
    assert definitions["FOV"] == pytest.approx([56e-3, 28e-3, 21e-3])
    assert definitions["MatrixSize"] == [32, 2, 2]
    assert definitions["SpectralTargetOffsetsHz"] == pytest.approx([1655.0, -245.0])
    assert definitions["SpectralTargetNames"] == ["Lac", "Py"]
    assert definitions["FlipAngleDeg"] == pytest.approx([90.0, 4.0])
    assert definitions["SpectralRFDuration"] == pytest.approx(2.33e-3)
    assert definitions["SpectralRFBandwidthFactorHzMs"] == pytest.approx(2100.0)
    assert definitions["SpectralRFBandwidthHz"] == pytest.approx(2100.0 / 2.33)
    assert definitions["SpectralRFFWHM"] == pytest.approx(900.0)
    assert definitions["ReadoutBandwidthHz"] == pytest.approx(10_000.0)


def test_spectral_selective_3d_bssfp_uses_metabolite_specific_flip_angles(tmp_path):
    sequence = SPECTRAL_BSSFP_MAIN(
        n_phase=1,
        n_partition=1,
        n_repetition=2,
        dummy_repetitions=0,
        use_alpha_half=False,
    )
    path = tmp_path / "spectral_selective_3d_bssfp_flip_angles.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)

    by_offset = {}
    for event in program.rf_events:
        by_offset.setdefault(round(event.frequency_offset_hz), []).append(
            np.max(np.abs(event.samples_hz))
        )

    assert max(by_offset[1655]) / max(by_offset[-245]) == pytest.approx(
        90.0 / 4.0,
        rel=0.02,
    )


def test_spectral_selective_3d_bssfp_accepts_readout_bandwidth():
    sequence = SPECTRAL_BSSFP_MAIN(
        n_read=32,
        n_phase=2,
        n_partition=2,
        n_repetition=1,
        readout_bandwidth_hz=20_000.0,
        dummy_repetitions=0,
    )
    ok, errors = sequence.check_timing()
    assert ok, errors

    definitions = sequence.definitions
    assert definitions["ReadoutBandwidthHz"] == pytest.approx(20_000.0)
    assert definitions["ADCDwell"] == pytest.approx(50e-6)


@pytest.mark.parametrize("value", [(), (float("nan"),), (float("inf"),)])
def test_spectral_selective_3d_bssfp_rejects_invalid_offsets(value):
    with pytest.raises(ValueError, match="target_frequency_offsets_hz"):
        SPECTRAL_BSSFP_MAIN(target_frequency_offsets_hz=value)


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_spectral_selective_3d_bssfp_rejects_invalid_frame_count(value):
    with pytest.raises(ValueError, match="n_repetition"):
        SPECTRAL_BSSFP_MAIN(n_repetition=value)
