"""Display units preserve amounts and complex per-quadrature noise statistics."""

from dataclasses import replace

import numpy as np
import pytest

from blochsimulator.sequence import CartesianAcquisition, SpectroscopicAcquisition
from blochsimulator.sequence.result import SequenceSimulationResult
from blochsimulator.sequence.signal_units import (
    SignalDisplay,
    display_result,
    noise_only_result,
    noise_std_per_sample,
    normalize_signal_unit,
    realized_noise_std,
    signal_display,
)
from blochsimulator.units import thermal_polarization


def _result(signal, sigmas=(2.0,), samples=None, mode="relative"):
    count = np.shape(signal)[-1]
    return SequenceSimulationResult(
        signal=np.asarray(signal),
        signal_noiseless=np.zeros_like(signal),
        adc_times_s=np.arange(count) * 1e-4,
        final_magnetization=np.zeros((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        metadata={
            "receiver_noise": {
                "mode": mode,
                "noise_std_per_quadrature": list(sigmas),
            },
            "receiver_acquisition": {"adc_samples": list(samples or (count,))},
        },
    )


@pytest.mark.parametrize("unit", ["physical", "snr"])
@pytest.mark.parametrize("mode", ["off", "relative"])
def test_fallbacks_are_explicit(unit, mode):
    result = _result(np.ones(5), mode=mode)
    display = signal_display(result, unit)
    if unit == "snr" and mode == "relative":
        assert display.snr
    else:
        assert display.unit == "simulation"
        assert "showing simulation units" in display.note
        assert display_result(result, display) is result
    assert normalize_signal_unit("unknown") == "physical"
    assert normalize_signal_unit(None) == "physical"


@pytest.mark.parametrize(
    "sigmas,samples", [((0,), (4,)), ((np.nan,), (4,)), ((2, 4), (1, 1))]
)
def test_missing_or_invalid_noise_metadata_falls_back(sigmas, samples):
    result = _result(np.ones(4), sigmas, samples)
    assert noise_std_per_sample(result) is None
    assert signal_display(result, "snr").unit == "simulation"


def test_per_event_scaling_broadcasts_over_species_and_coils_without_mutation():
    signal = np.full((1, 5), 8 + 4j)
    result = _result(signal, (2, 4), (2, 3))
    species = np.stack((signal, 2 * signal))
    result = replace(result, species_signal=species, pool_names=("A", "B"))
    display = signal_display(result, "snr")
    shown = display_result(result, display)
    np.testing.assert_array_equal(shown.signal, [[4 + 2j] * 2 + [2 + 1j] * 3])
    np.testing.assert_array_equal(shown.species_signal[1], 2 * shown.signal)
    np.testing.assert_array_equal(result.signal, signal)
    np.testing.assert_array_equal(result.species_signal, species)
    np.testing.assert_array_equal(noise_only_result(shown).signal, shown.signal)
    assert shown.final_magnetization is result.final_magnetization
    assert shown.metadata is result.metadata


def test_physical_pyruvate_amount_and_formatting():
    pth = thermal_polarization("C13", 7, 310.15)
    amount = 80 * 200e-9 * 0.2  # mM = mol/m³; 200 µl in m³.
    result = _result(np.full(8, amount / pth), mode="calibrated")
    result.metadata["receiver_noise"]["calibration_derived"] = {
        "target_thermal_polarization": pth
    }
    display = signal_display(result, "physical")
    shown = display_result(result, display)
    assert shown.signal[0] == pytest.approx(3.2e-6)
    assert display.format_value(shown.signal[0]) == "3.2 µmol"
    assert display.axis_units == "mol"
    assert display.image_norm is None
    assert display.spectrum_scale(64) == 1 / 64
    assert SignalDisplay("snr").axis_label == "Signal / σ"


@pytest.mark.parametrize("domain", ["adc", "cartesian", "csi_fid", "csi_spectrum"])
def test_unit_noise_std_survives_complete_fourier_reconstruction(domain):
    nx, ny, ns = 32, 32, 64
    rng = np.random.default_rng(71)
    count = nx * ny * (1 if domain == "cartesian" else ns)
    # Mixed ADC bandwidths make an omitted per-event scaling visible.
    sigmas = (2.0, 6.0)
    sigma = np.repeat(sigmas, count // 2)
    noise = sigma * (rng.normal(size=count) + 1j * rng.normal(size=count))
    result = _result(noise, sigmas, (count // 2, count // 2))
    display = signal_display(result, "snr")
    shown = display_result(result, display)
    if domain == "cartesian":
        acq = CartesianAcquisition(nx, ny, (0.2, 0.2), 1e-4)
        values = acq.reconstruct(shown.signal, norm=display.image_norm)
    elif domain.startswith("csi"):
        acq = SpectroscopicAcquisition(
            (nx, ny),
            (0.2, 0.2),
            ns,
            1e-4,
            tuple((x, y) for y in range(ny) for x in range(nx)),
        )
        if domain == "csi_fid":
            values = acq.reconstruct_spatial(shown.signal, norm=display.image_norm)
        else:
            values = acq.reconstruct_spectra(shown.signal, norm=display.image_norm)
            values *= display.spectrum_scale(ns)
    else:
        values = shown.signal
    assert np.std(values.real) == pytest.approx(1, rel=0.07)
    assert np.std(values.imag) == pytest.approx(1, rel=0.07)
    assert realized_noise_std(values) == pytest.approx(1, rel=0.05)


def test_physical_spatial_amount_and_spectral_peak_are_preserved():
    nx, ny, ns = 4, 3, 16
    amount = 3.2e-6
    pth = thermal_polarization("C13", 7, 310.15)
    csi = SpectroscopicAcquisition(
        (nx, ny),
        (0.2, 0.2),
        ns,
        1e-4,
        tuple((x, y) for y in range(ny) for x in range(nx)),
    )
    # A constant FID in every image voxel yields only central k-space signal.
    kspace = np.zeros((ny, nx, ns), dtype=complex)
    kspace[ny // 2, nx // 2, :] = amount / pth
    result = _result(kspace.reshape(-1), mode="calibrated")
    result.metadata["receiver_noise"]["calibration_derived"] = {
        "target_thermal_polarization": pth
    }
    display = signal_display(result, "physical")
    shown = display_result(result, display)
    fid = csi.reconstruct_spatial(shown.signal)
    spectra = csi.reconstruct_spectra(shown.signal) * display.spectrum_scale(ns)
    assert np.sum(np.abs(fid[..., 0])) == pytest.approx(amount)
    np.testing.assert_allclose(np.max(np.abs(spectra), axis=-1), np.abs(fid[..., 0]))
