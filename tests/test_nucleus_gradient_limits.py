import numpy as np
import pytest

from blochsimulator.sequence.advanced_bssfp_builders import (
    make_pulseq_me_bssfp,
    make_pulseq_spectral_selective_bssfp,
)
from blochsimulator.sequence.pulseq_builders import (
    make_pulseq_bssfp,
    make_pulseq_epi,
    make_pulseq_epsi_mge,
    make_pulseq_flash,
    make_pulseq_spiral,
)
from blochsimulator.sequence.scanner import ScannerParameters
from blochsimulator.units import NUCLEUS_GAMMA_HZ_PER_T

SCANNER = {"max_grad_mtm": 32.0, "max_slew_tms": 130.0}
CARBON_SLICE = dict(slice_thickness_m=10e-3, rf_duration_s=3e-3)


def _physical_peaks(sequence, nucleus):
    """Peak gradient (mT/m) and slew rate (T/m/s) of the imaged nucleus."""
    gamma = NUCLEUS_GAMMA_HZ_PER_T[nucleus]
    waveforms = sequence.waveforms_and_times()[0][:3]
    peak_gradient = 0.0
    peak_slew = 0.0
    for wave in waveforms:
        if not wave.size:
            continue
        times, amplitudes = wave
        peak_gradient = max(peak_gradient, float(np.abs(amplitudes).max()))
        step = np.diff(times)
        valid = step > 1e-9
        slew = np.abs(np.diff(amplitudes)[valid] / step[valid])
        if slew.size:
            peak_slew = max(peak_slew, float(slew.max()))
    return peak_gradient / gamma * 1e3, peak_slew / gamma


@pytest.mark.parametrize(
    "builder, kwargs",
    [
        (
            make_pulseq_flash,
            dict(CARBON_SLICE, matrix=(16, 16), sampling_bandwidth_hz=25e3),
        ),
        (make_pulseq_epi, CARBON_SLICE),
        (make_pulseq_spiral, CARBON_SLICE),
        (make_pulseq_bssfp, {}),
        (make_pulseq_spectral_selective_bssfp, dict(repetition_time_s=9e-3)),
        (
            make_pulseq_me_bssfp,
            dict(
                fov_m=(0.12, 0.12, 0.12),
                sampling_bandwidth_hz=25e3,
                echo_spacing_s=3e-3,
                encoding_duration_s=1e-3,
                repetition_time_s=20e-3,
            ),
        ),
    ],
)
def test_carbon_gradients_stay_within_the_physical_limits(builder, kwargs):
    sequence = builder(nucleus="C13", scanner_parameters=SCANNER, **kwargs)

    gradient, slew = _physical_peaks(sequence, "C13")

    assert 0 < gradient <= SCANNER["max_grad_mtm"] * (1 + 1e-6)
    assert slew <= SCANNER["max_slew_tms"] * (1 + 1e-3)


def _adc_dwell(sequence):
    for index in sequence.block_events:
        block = sequence.get_block(index)
        if block.adc is not None:
            return float(block.adc.dwell)
    raise AssertionError("no ADC")


def test_carbon_spiral_is_slower_than_the_proton_spiral():
    kwargs = dict(CARBON_SLICE, sampling_bandwidth_hz=100e3, scanner_parameters=SCANNER)
    proton = make_pulseq_spiral(nucleus="H1", **kwargs)
    carbon = make_pulseq_spiral(nucleus="C13", **kwargs)

    # The 1H limit would let the 13C spiral play four times the gradient.
    assert _physical_peaks(proton, "C13")[0] > SCANNER["max_grad_mtm"]
    assert _adc_dwell(carbon) > 1.5 * _adc_dwell(proton)


def test_infeasible_carbon_slice_reports_the_physical_gradient():
    with pytest.raises(ValueError, match=r"needs 41 mT/m .* above the 32 mT/m limit"):
        make_pulseq_epi(
            nucleus="C13",
            scanner_parameters=SCANNER,
            slice_thickness_m=3e-3,
            rf_duration_s=3e-3,
        )


@pytest.mark.parametrize(
    "builder", [make_pulseq_flash, make_pulseq_spiral, make_pulseq_me_bssfp]
)
def test_unknown_nucleus_is_rejected(builder):
    with pytest.raises(ValueError, match="unknown nucleus"):
        builder(nucleus="X99")


@pytest.mark.parametrize("bandwidth_hz", [142858.0, 250e3])
@pytest.mark.parametrize(
    "builder",
    [make_pulseq_flash, make_pulseq_epi, make_pulseq_epsi_mge, make_pulseq_bssfp],
)
def test_fast_readouts_keep_the_adc_centred_on_a_raster_plateau(builder, bandwidth_hz):
    # On a preclinical system the slew-limited ramp is shorter than the ADC
    # dead time; the plateau must still be centred and on the block raster.
    sequence = builder(
        scanner_parameters=ScannerParameters(), sampling_bandwidth_hz=bandwidth_hz
    )
    ok, errors = sequence.check_timing()
    assert ok, errors[:3]
    checked = 0
    for index in sequence.block_events:
        block = sequence.get_block(index)
        if block.adc is None:
            continue
        plateaus = [
            gradient
            for gradient in (block.gx, block.gy, block.gz)
            if gradient is not None and getattr(gradient, "flat_time", 0) > 0
        ]
        assert len(plateaus) == 1
        read = plateaus[0]
        adc_centre = block.adc.delay + block.adc.num_samples * block.adc.dwell / 2
        plateau_centre = read.delay + read.rise_time + read.flat_time / 2
        assert adc_centre == pytest.approx(plateau_centre, abs=1e-9)
        assert block.adc.delay >= read.delay + read.rise_time - 1e-12
        checked += 1
    assert checked
