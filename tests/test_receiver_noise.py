"""Scientific contracts for the manual single-channel receiver model."""

from dataclasses import replace
import json

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import (
    DynamicSpectralPhantom,
    PyruvateInflow,
    TimeCurve,
)
from blochsimulator.phantom import Phantom
from blochsimulator.project_io import load_project, save_project
from blochsimulator.sequence import (
    ADCEvent,
    ReceiverCalibration,
    ReceiverNoise,
    RFEvent,
    SequenceProgram,
    SequenceSimulationResult,
    apply_receiver_noise,
    receiver_acquisition_metadata,
)
from blochsimulator.sequence.receiver import ideal_fid_amplitude
from blochsimulator.spectral_phantom import ChemicalSpecies, SpectralPhantom
from blochsimulator.units import (
    BOLTZMANN_J_PER_K,
    NUCLEUS_GAMMA_HZ_PER_T,
    PLANCK_J_S,
    thermal_polarization,
)


def phantom(shape=(2, 2, 2), density=1.0, fov=(0.02, 0.02, 0.02)):
    return Phantom(
        shape=shape,
        fov=fov,
        t1_map=np.full(shape, 1e12),
        t2_map=np.full(shape, 1e12),
        pd_map=np.full(shape, density),
    )


def fid_program(flip=90.0, count=32, dwell=1e-4):
    duration = 1e-3
    return SequenceProgram(
        (
            RFEvent(0.0, np.array([flip / 360 / duration]), duration),
            ADCEvent(duration, count, dwell),
        ),
        duration_s=duration + count * dwell,
    )


def result(count=100000):
    return SequenceSimulationResult(
        signal=np.full(count, 3 + 2j),
        adc_times_s=np.arange(count) * 1e-4,
        final_magnetization=np.ones((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        metadata={"signal_weighting": "voxel_volume"},
    )


def calibration(**kwargs):
    """2 M 13C urea in 50 ml at 20 degC, measured with SNR 200."""
    values = dict(
        measured_amplitude=1000.0,
        measured_noise_std=5.0,
        concentration_mm=2000.0,
        volume_ml=50.0,
        nucleus="C13",
        field_strength_t=7.0,
        dwell_s=1e-4,
    )
    values.update(kwargs)
    return ReceiverCalibration(**values)


def with_context(
    clean,
    program=None,
    nucleus="C13",
    field_strength_t=7.0,
    *,
    obj=None,
    amplitude=None,
):
    """Attach the run context; ``amplitude`` stands in for a simulated object."""
    metadata = dict(clean.metadata)
    context = receiver_acquisition_metadata(
        program or fid_program(count=clean.signal.shape[-1]),
        nucleus=nucleus,
        field_strength_t=field_strength_t,
        phantom=obj,
        signal_weighting=metadata.get("signal_weighting"),
    )
    if amplitude is not None:
        context["ideal_fid_amplitude"] = amplitude
    metadata["receiver_acquisition"] = context
    return replace(clean, metadata=metadata)


def inflow_phantom(polarization_curve):
    """Empty object filled by a step-rate bolus: 2 mM/s for 0.5 s, then 1 mM/s."""
    shape = (2, 2, 2)
    return DynamicSpectralPhantom(
        shape=shape,
        fov=(0.02,) * 3,
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 1e15, 1e15),
            ChemicalSpecies("Lactate", 12.0, 1e15, 1e15),
        ),
        initial_concentration_maps={
            "Pyruvate": np.zeros(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.zeros(shape),
        pyruvate_inflow=PyruvateInflow(
            TimeCurve((0.0, 0.5, 1.0), (2.0, 1.0, 1.0), "step", "zero"),
            np.linspace(0.5, 1.5, 8).reshape(shape),
            polarization_curve,
        ),
        nucleus="C13",
    )


def test_noise_statistics_reproducibility_and_no_accumulation():
    clean = with_context(result(), amplitude=2.0)
    settings = ReceiverNoise("relative", snr=10.0, seed=71)
    noisy = apply_receiver_noise(clean, settings)
    difference = noisy.signal - clean.signal
    assert np.mean(difference.real) == pytest.approx(0, abs=0.003)
    assert np.mean(difference.imag) == pytest.approx(0, abs=0.003)
    assert np.std(difference.real) == pytest.approx(0.2, rel=0.01)
    assert np.std(difference.imag) == pytest.approx(0.2, rel=0.01)
    assert np.mean(np.abs(difference) ** 2) == pytest.approx(2 * 0.2**2, rel=0.01)
    assert abs(np.corrcoef(difference.real, difference.imag)[0, 1]) < 0.01
    np.testing.assert_array_equal(
        apply_receiver_noise(noisy, settings).signal, noisy.signal
    )
    np.testing.assert_array_equal(noisy.signal_noiseless, clean.signal)
    assert noisy.final_magnetization is clean.final_magnetization
    assert "receiver_noise" not in clean.metadata
    different = apply_receiver_noise(noisy, replace(settings, seed=72))
    assert not np.array_equal(different.signal, noisy.signal)
    np.testing.assert_array_equal(
        apply_receiver_noise(noisy, ReceiverNoise()).signal, clean.signal
    )
    assert apply_receiver_noise(clean, ReceiverNoise()) is clean


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_invalid_relative_snr_and_reference_dwell_are_rejected(value):
    with pytest.raises(ValueError, match="Relative SNR"):
        ReceiverNoise("relative", snr=value)
    with pytest.raises(ValueError, match="Reference dwell"):
        ReceiverNoise("relative", reference_dwell_s=value)
    with pytest.raises(ValueError, match="Unknown"):
        ReceiverNoise("direct")


@pytest.mark.parametrize("seed", [-1, 1.2, True])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        ReceiverNoise("relative", seed=seed)


def test_thermal_polarization_follows_boltzmann_statistics():
    x = PLANCK_J_S * NUCLEUS_GAMMA_HZ_PER_T["C13"] * 7.0 / (BOLTZMANN_J_PER_K * 293.15)
    assert thermal_polarization("C13", 7.0, 293.15) == pytest.approx(
        np.tanh(x / 2), rel=1e-12
    )
    assert thermal_polarization("C13", 7.0, 293.15) == pytest.approx(6.13e-6, rel=1e-3)
    assert thermal_polarization("H1", 3.0, 310.15) == pytest.approx(9.88e-6, rel=1e-3)
    assert thermal_polarization("C13", 3.0, 310.15) / thermal_polarization(
        "C13", 7.0, 293.15
    ) == pytest.approx(3 * 293.15 / (7 * 310.15), rel=1e-9)
    # Spin 3/2 in the high-temperature limit: (I + 1) / 3 * h*gamma*B0 / (k*T).
    x_na = (
        PLANCK_J_S * NUCLEUS_GAMMA_HZ_PER_T["Na23"] * 3.0 / (BOLTZMANN_J_PER_K * 310.15)
    )
    assert thermal_polarization("Na23", 3.0, 310.15) == pytest.approx(
        5 / 6 * x_na, rel=1e-9
    )
    for args in (("C13", 0, 300), ("C13", 7, -1), ("X1", 7, 300)):
        with pytest.raises(ValueError):
            thermal_polarization(*args)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"concentration_mm": 0}, "concentration"),
        ({"volume_ml": -1}, "volume"),
        ({"measured_noise_std": 0}, "noise std"),
        ({"dwell_s": 0}, "dwell"),
        ({"averages": 0}, "Averages"),
        ({"averages": 1.5}, "Averages"),
        ({"flip_angle_deg": 0}, "flip angle"),
        ({"flip_angle_deg": 180}, "flip angle"),
        ({"enrichment": 1.2}, "Enrichment"),
        ({"polarization": 0}, "polarization"),
        ({"nucleus": "X1"}, "nucleus"),
    ],
)
def test_calibration_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        calibration(**kwargs)


def test_thermal_phantom_simulated_at_its_temperature_reproduces_measured_snr():
    # 1 mM in a 2 cm cube (8 ml): the measured SNR is that phantom's SNR.
    obj = phantom()
    ref = calibration(concentration_mm=1.0, volume_ml=8.0, target_temperature_k=293.15)
    assert ref.reference_moles == pytest.approx(ideal_fid_amplitude(obj))
    assert ref.noise_std(1e-4) == pytest.approx(ideal_fid_amplitude(obj) / 200)
    simulator = BlochSimulator(use_parallel=False)
    for flip, snr in ((90, 200), (30, 100)):
        clean = simulator.simulate_sequence(
            fid_program(flip), obj, signal_weighting="voxel_volume"
        )
        sigma = ref.noise_std(1e-4)
        assert abs(clean.signal[0]) / sigma == pytest.approx(snr, rel=1e-5)


def test_calibration_scales_with_bandwidth_averages_flip_and_spin_content():
    base = calibration()
    sigma = base.noise_std(1e-4)
    assert base.noise_std(4e-4) == pytest.approx(sigma / 2)
    assert base.noise_std(2.5e-5) == pytest.approx(2 * sigma)
    # Same measured SNR from 4 averages means half the single-scan SNR.
    assert calibration(averages=4).noise_std(1e-4) == pytest.approx(2 * sigma)
    assert calibration(flip_angle_deg=30).noise_std(1e-4) == pytest.approx(sigma / 2)
    assert calibration(enrichment=0.011).noise_std(1e-4) == pytest.approx(0.011 * sigma)
    assert calibration(nuclei_per_molecule=2).noise_std(1e-4) == pytest.approx(
        2 * sigma
    )
    assert calibration(noise_factor=1.5).noise_std(1e-4) == pytest.approx(1.5 * sigma)
    # A reference dwell different from the target dwell only rescales sqrt(BW).
    assert calibration(dwell_s=4e-4).noise_std(4e-4) == pytest.approx(sigma)
    thermal_20 = thermal_polarization("C13", 7.0, 293.15)
    assert calibration(polarization=thermal_20).noise_std(1e-4) == pytest.approx(sigma)
    assert calibration(target_temperature_k=293.15).noise_std(1e-4) == pytest.approx(
        0.1 / 200
    )


def test_urea_calibration_predicts_hyperpolarized_pyruvate_snr():
    """2 M urea, 50 ml, SNR 200 at 4 averages -> 80 mM pyruvate, 200 ul, 20 %."""
    ref = calibration(averages=4)
    assert ref.reference_moles == pytest.approx(0.1)
    assert ref.single_scan_snr == pytest.approx(100)
    thermal_target = thermal_polarization("C13", 7.0, 310.15)
    side = (0.2e-6) ** (1 / 3)
    for shape in ((1, 1, 1), (2, 2, 2)):
        obj = phantom(shape=shape, density=80.0, fov=(side,) * 3)
        obj.m0_map[..., 2] = 0.2 / thermal_target
        clean = BlochSimulator(use_parallel=False).simulate_sequence(
            fid_program(), obj, signal_weighting="voxel_volume"
        )
        noisy = apply_receiver_noise(
            with_context(clean), ReceiverNoise("calibrated", seed=5, calibration=ref)
        )
        sigma = noisy.metadata["receiver_noise"]["noise_std_per_quadrature"][0]
        # 16 umol x 20 % = 3.2 umol versus 0.1 mol x 6.13e-6 = 0.613 umol.
        expected = (
            100 * (80 * 0.2e-6 * 0.2) / (0.1 * thermal_polarization("C13", 7.0, 293.15))
        )
        assert expected == pytest.approx(521.7, rel=1e-3)
        assert abs(clean.signal[0]) / sigma == pytest.approx(expected, rel=1e-5)


def test_calibrated_noise_follows_each_adc_event_bandwidth():
    program = SequenceProgram(
        (ADCEvent(0.0, 3, 1e-4), ADCEvent(1e-3, 5, 4e-4)), duration_s=4e-3
    )
    ref = calibration()
    settings = ReceiverNoise("calibrated", seed=8, calibration=ref)
    clean = with_context(result(8), program)
    noisy = apply_receiver_noise(clean, settings)
    sigma = np.repeat([ref.noise_std(1e-4), ref.noise_std(4e-4)], [3, 5])
    rng = np.random.Generator(np.random.PCG64(8))
    expected = sigma * (rng.standard_normal(8) + 1j * rng.standard_normal(8))
    np.testing.assert_allclose(noisy.signal - clean.signal, expected, atol=1e-15)
    receiver = noisy.metadata["receiver_noise"]
    assert receiver["noise_std_scope"] == "per_adc_event"
    assert receiver["noise_std_per_quadrature"] == pytest.approx(
        [ref.noise_std(1e-4), ref.noise_std(4e-4)]
    )
    assert receiver["calibration"] == ref.to_metadata()
    assert receiver["calibration_derived"]["single_scan_snr"] == pytest.approx(200)
    with pytest.raises(ValueError, match="signal length"):
        apply_receiver_noise(with_context(result(9), program), settings)
    with pytest.raises(ValueError, match="nucleus, field strength"):
        apply_receiver_noise(result(8), settings)
    with pytest.raises(ValueError, match="same nucleus"):
        apply_receiver_noise(with_context(result(8), program, "H1"), settings)
    with pytest.raises(ValueError, match="same nucleus"):
        apply_receiver_noise(
            with_context(result(8), program, field_strength_t=3.0), settings
        )
    with pytest.raises(ValueError, match="weighting"):
        apply_receiver_noise(
            replace(clean, metadata={**clean.metadata, "signal_weighting": "voxel"}),
            settings,
        )
    with pytest.raises(ValueError, match="calibration"):
        ReceiverNoise("calibrated")


def test_calibrated_noise_monte_carlo_matches_per_quadrature_std():
    ref = calibration()
    noisy = apply_receiver_noise(
        with_context(result()), ReceiverNoise("calibrated", seed=13, calibration=ref)
    )
    difference = noisy.signal - noisy.signal_noiseless
    sigma = ref.noise_std(1e-4)
    assert np.std(difference.real) == pytest.approx(sigma, rel=0.01)
    assert np.std(difference.imag) == pytest.approx(sigma, rel=0.01)
    assert np.mean(np.abs(difference) ** 2) == pytest.approx(2 * sigma**2, rel=0.01)


@pytest.mark.parametrize("weighting", ["voxel", "voxel_volume"])
def test_relative_snr_is_the_simulated_whole_object_fid_snr(weighting):
    simulator = BlochSimulator(use_parallel=False)
    settings = ReceiverNoise("relative", snr=40.0, seed=3)
    sigmas = []
    for obj in (phantom(), phantom(shape=(4, 4, 4))):
        for flip, snr in ((90, 40), (30, 20)):
            clean = simulator.simulate_sequence(
                fid_program(flip), obj, signal_weighting=weighting
            )
            noisy = apply_receiver_noise(with_context(clean, obj=obj), settings)
            sigma = noisy.metadata["receiver_noise"]["noise_std_per_quadrature"][0]
            assert abs(clean.signal[0]) / sigma == pytest.approx(snr, rel=1e-5)
        sigmas.append(sigma)
    # Physical weighting gives one std for both grids; per-voxel weighting
    # sums more voxels on the finer grid, and the std follows the amplitude.
    expected = 1.0 if weighting == "voxel_volume" else 8.0
    assert sigmas[1] / sigmas[0] == pytest.approx(expected)
    relative = noisy.metadata["receiver_noise"]["relative_snr"]
    assert relative == {
        "snr": 40.0,
        "reference_dwell_s": 1e-4,
        "ideal_fid_amplitude": pytest.approx(ideal_fid_amplitude(obj, weighting)),
    }
    assert noisy.metadata["receiver_noise"]["calibration"] is None


def test_relative_noise_follows_each_adc_event_bandwidth():
    program = SequenceProgram(
        (ADCEvent(0.0, 3, 1e-4), ADCEvent(1e-3, 5, 4e-4)), duration_s=4e-3
    )
    clean = with_context(result(8), program, amplitude=2.0)
    noisy = apply_receiver_noise(clean, ReceiverNoise("relative", 10.0, 8))
    sigma = np.repeat([0.2, 0.1], [3, 5])
    rng = np.random.Generator(np.random.PCG64(8))
    expected = sigma * (rng.standard_normal(8) + 1j * rng.standard_normal(8))
    np.testing.assert_allclose(noisy.signal - clean.signal, expected, atol=1e-15)
    # SNR 10 stated at the slower ADC: that event gets amplitude / 10.
    slow_reference = ReceiverNoise("relative", 10.0, 8, reference_dwell_s=4e-4)
    receiver = apply_receiver_noise(clean, slow_reference).metadata["receiver_noise"]
    assert receiver["noise_std_per_quadrature"] == pytest.approx([0.4, 0.2])
    assert slow_reference.noise_std_for_dwell(4e-4, 2.0) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="ideal FID amplitude"):
        slow_reference.noise_std_for_dwell(4e-4)
    with pytest.raises(ValueError, match="signal length"):
        apply_receiver_noise(
            with_context(result(9), program, amplitude=2.0), slow_reference
        )


def test_relative_snr_requires_a_recorded_positive_reference():
    settings = ReceiverNoise("relative", 10.0)
    with pytest.raises(ValueError, match="nucleus, field strength and ADC dwell"):
        apply_receiver_noise(result(32), settings)
    with pytest.raises(ValueError, match="Rerun the simulation"):
        apply_receiver_noise(with_context(result(32)), settings)
    empty = phantom(density=0.0)
    context = receiver_acquisition_metadata(
        fid_program(), nucleus="C13", field_strength_t=7.0, phantom=empty
    )
    assert context["ideal_fid_amplitude"] is None
    with pytest.raises(ValueError, match="initial or delivered magnetization"):
        ideal_fid_amplitude(empty)
    with pytest.raises(ValueError, match="initial or delivered magnetization"):
        apply_receiver_noise(with_context(result(32), obj=empty), settings)
    with pytest.raises(ValueError, match="3D phantom"):
        ideal_fid_amplitude(phantom(shape=(2, 2), fov=(0.02, 0.02)))
    flat = phantom(shape=(2, 2), fov=(0.02, 0.02))
    assert ideal_fid_amplitude(flat, "voxel") == pytest.approx(4.0)
    with pytest.raises(ValueError, match="signal_weighting"):
        ideal_fid_amplitude(flat, "volume")


def test_off_mode_restores_clean_signal_and_records_zero_noise():
    program = SequenceProgram(
        (ADCEvent(0.0, 3, 1e-4), ADCEvent(1e-3, 5, 4e-4)), duration_s=4e-3
    )
    clean = with_context(result(8), program, amplitude=2.0)
    noisy = apply_receiver_noise(clean, ReceiverNoise("relative", 10.0, 8))
    restored = apply_receiver_noise(noisy, ReceiverNoise(seed=8))
    np.testing.assert_array_equal(restored.signal, clean.signal)
    receiver = restored.metadata["receiver_noise"]
    assert receiver["mode"] == "off"
    assert receiver["noise_std_per_quadrature"] == [0.0, 0.0]
    assert receiver["relative_snr"] is None


@pytest.mark.parametrize(
    "polarization, dose",
    [
        # Step rate x linear polarization 0.3 -> 0.1 over 1 s:
        # 2 * (0.15 - 0.1 * 0.5**2) + (0.15 - 0.1 * (1 - 0.5**2)) = 0.325.
        (TimeCurve((0.0, 1.0), (0.3, 0.1), "linear", "zero"), 0.325),
        # Legacy direct-Mz source: the rate integral alone, 2 * 0.5 + 1 * 0.5.
        (None, 1.5),
    ],
)
@pytest.mark.parametrize("weighting", ["voxel", "voxel_volume"])
def test_relative_reference_includes_delivered_bolus_dose(
    polarization, dose, weighting
):
    obj = inflow_phantom(polarization)
    scale = 1.0 if weighting == "voxel" else 1e-6
    # The delivery map sums to 8 over the eight voxels.
    assert ideal_fid_amplitude(obj, weighting) == pytest.approx(
        8 * dose * scale, rel=1e-14
    )
    # Without relaxation or conversion, a 90-degree pulse after the bolus
    # reads exactly the delivered dose.
    duration = 1e-3
    program = SequenceProgram(
        (
            RFEvent(1.0, np.array([90 / 360 / duration]), duration),
            ADCEvent(1.0 + duration, 4, 1e-4),
        ),
        duration_s=1.0 + duration + 4e-4,
    )
    clean = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, obj, signal_weighting=weighting
    )
    assert abs(clean.signal[0]) == pytest.approx(
        ideal_fid_amplitude(obj, weighting), rel=1e-12
    )


def test_reference_grid_invariance_and_initial_polarization():
    coarse = phantom()
    fine = phantom(shape=(4, 4, 4))
    assert ideal_fid_amplitude(coarse) == pytest.approx(ideal_fid_amplitude(fine))
    polarized = phantom()
    polarized.m0_map[..., 2] = 1000
    assert ideal_fid_amplitude(polarized) == pytest.approx(
        1000 * ideal_fid_amplitude(coarse)
    )


def test_receiver_rejects_multiple_channels_without_changing_off_mode():
    clean = replace(result(20), signal=np.zeros((2, 20), complex))
    assert apply_receiver_noise(clean, ReceiverNoise()) is clean
    with pytest.raises(ValueError, match="one receive"):
        apply_receiver_noise(clean, ReceiverNoise("relative", 10.0))


def test_noise_propagates_through_kspace_and_image_reconstruction():
    from blochsimulator.sequence import CartesianAcquisition

    acquisition = CartesianAcquisition(2, 2, (0.02, 0.02), 1e-4)
    clean = with_context(
        replace(
            result(4),
            metadata={"cartesian_acquisition": acquisition.to_metadata()},
        ),
        amplitude=1.0,
    )
    measured = apply_receiver_noise(clean, ReceiverNoise("relative", 5.0, 41))
    exported = measured.to_xarray()
    np.testing.assert_array_equal(
        exported.cartesian_kspace, acquisition.reshape_signal(measured.signal)
    )
    np.testing.assert_array_equal(
        exported.cartesian_image, measured.reconstruct_cartesian(acquisition)
    )
    difference = measured.reconstruct_cartesian(
        acquisition, norm="ortho"
    ) - clean.reconstruct_cartesian(acquisition, norm="ortho")
    assert np.sum(np.abs(difference) ** 2) == pytest.approx(
        np.sum(np.abs(measured.signal - clean.signal) ** 2)
    )


def test_spectral_partitioning_does_not_multiply_noise():
    shape = (2, 2, 2)
    species = [ChemicalSpecies("a", 0, 1e12, 1e12), ChemicalSpecies("b", 0, 1e12, 1e12)]
    spectral = SpectralPhantom(
        shape=shape,
        fov=(0.02,) * 3,
        species=species,
        concentration_maps={"a": np.full(shape, 0.3), "b": np.full(shape, 0.7)},
    )
    assert ideal_fid_amplitude(spectral) == pytest.approx(
        ideal_fid_amplitude(phantom())
    )
    simulator = BlochSimulator(use_parallel=False)
    static_result = simulator.simulate_sequence(
        fid_program(), phantom(), signal_weighting="voxel_volume"
    )
    spectral_result = simulator.simulate_spectral_sequence(
        fid_program(), spectral, signal_weighting="voxel_volume"
    )
    settings = ReceiverNoise("relative", 50.0, 19)
    static_noisy = apply_receiver_noise(
        with_context(static_result, obj=phantom()), settings
    )
    spectral_noisy = apply_receiver_noise(
        with_context(spectral_result, obj=spectral), settings
    )
    np.testing.assert_allclose(static_noisy.signal, spectral_noisy.signal, atol=1e-16)
    assert spectral_noisy.species_signal is spectral_result.species_signal


def test_chunk_size_does_not_change_noise_and_sequence_signal_can_change():
    simulator = BlochSimulator(use_parallel=False)
    obj = phantom()
    settings = ReceiverNoise("calibrated", seed=99, calibration=calibration())
    first = simulator.simulate_sequence(
        fid_program(), obj, signal_weighting="voxel_volume", chunk_voxels=1
    )
    second = simulator.simulate_sequence(
        fid_program(), obj, signal_weighting="voxel_volume", chunk_voxels=8
    )
    np.testing.assert_allclose(
        apply_receiver_noise(with_context(first), settings).signal,
        apply_receiver_noise(with_context(second), settings).signal,
        atol=1e-18,
    )
    low = simulator.simulate_sequence(
        fid_program(30), obj, signal_weighting="voxel_volume"
    )
    sigma = settings.noise_std_for_dwell(1e-4)
    assert abs(low.signal[0]) / abs(first.signal[0]) == pytest.approx(0.5, rel=1e-5)
    assert abs(first.signal[0]) / sigma == pytest.approx(
        200
        * 8e-6
        / 0.1
        * thermal_polarization("C13", 7, 310.15)
        / thermal_polarization("C13", 7, 293.15),
        rel=1e-5,
    )


def test_zero_degree_acquisition_contains_only_receiver_noise():
    clean = BlochSimulator(use_parallel=False).simulate_sequence(
        fid_program(0), phantom()
    )
    np.testing.assert_array_equal(clean.signal, 0)
    # The 90-degree reference of the same object still defines the noise.
    noisy = apply_receiver_noise(
        with_context(clean, fid_program(0), obj=phantom()),
        ReceiverNoise("relative", 80.0, 12),
    )
    assert noisy.metadata["receiver_noise"]["noise_std_per_quadrature"] == [
        pytest.approx(0.1)
    ]
    assert np.any(noisy.signal.real != 0)
    assert np.any(noisy.signal.imag != 0)
    np.testing.assert_array_equal(noisy.signal_noiseless, 0)


def budget_row(sections, title, label):
    section = next(section for section in sections if section.title == title)
    return next(row for row in section.rows if row.label == label)


def test_ideal_amplitude_map_sums_to_the_whole_object_fid():
    from blochsimulator.sequence.receiver import ideal_amplitude_map

    static = phantom(density=2)
    static.rx_sensitivity_maps = np.full((1, 2, 2, 2), 2j)
    bolus = inflow_phantom(TimeCurve((0.0, 1.0), (0.3, 0.1), "linear", "zero"))
    for obj, weighting in (
        (static, "voxel"),
        (static, "voxel_volume"),
        (bolus, "voxel"),
        (bolus, "voxel_volume"),
    ):
        amplitudes = ideal_amplitude_map(obj, weighting)
        assert amplitudes.shape == obj.shape
        assert abs(np.sum(amplitudes)) == pytest.approx(
            ideal_fid_amplitude(obj, weighting), rel=1e-14
        )
    # The bolus delivery map, not the empty initial state, sets each voxel.
    np.testing.assert_allclose(
        np.abs(ideal_amplitude_map(bolus, "voxel")),
        0.325 * np.linspace(0.5, 1.5, 8).reshape(2, 2, 2),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="signal_weighting must"):
        ideal_amplitude_map(static, "pixel")


def test_budget_voxels_match_epi_reconstruction_and_sum_rule():
    """Fully sampled EPI: voxel SNRs follow from the matched-filter total."""
    from blochsimulator.sequence import CartesianAcquisition
    from blochsimulator.sequence.acquisition import make_cartesian_epi
    from blochsimulator.sequence.receiver import ideal_amplitude_map
    from blochsimulator.sequence.snr_budget import (
        acquisition_snr,
        grid_voxel_amplitudes,
        voxel_shares,
    )

    shape = (4, 4, 1)
    density = np.arange(1, 17, dtype=float).reshape(shape)
    obj = phantom(shape=shape, density=density, fov=(0.02, 0.02, 0.005))
    acquisition = CartesianAcquisition(4, 4, (0.02, 0.02), 1e-4)
    program = make_cartesian_epi(acquisition)
    clean = BlochSimulator(use_parallel=False).simulate_sequence(
        program, obj, signal_weighting="voxel"
    )
    clean = with_context(clean, program, "H1", 3.0, obj=obj)
    cells, outside = grid_voxel_amplitudes(
        ideal_amplitude_map(obj, "voxel"),
        obj.positions,
        (4, 4),
        acquisition.fov_m,
        acquisition.encoding_frame.matrix,
    )
    assert outside == 0
    image = clean.reconstruct_cartesian(acquisition, norm="ortho")
    # Without relaxation the unitary reconstruction is the binned object.
    np.testing.assert_allclose(np.abs(image), 4 * np.abs(cells).T, atol=1e-12)
    shares = voxel_shares(cells, ideal_fid_amplitude(obj, "voxel"), outside)
    assert shares.voxels == 16
    assert shares.brightest_share == pytest.approx(16 / 136)
    assert shares.effective_voxels == pytest.approx(136**2 / np.sum(np.arange(17) ** 2))

    noise = ReceiverNoise("relative", 50.0, 3)
    sigma = noise.noise_std_for_dwell(1e-4, ideal_fid_amplitude(obj, "voxel"))
    measured = acquisition_snr(clean, noise)
    assert measured.samples == 16
    assert measured.peak == pytest.approx(50)
    assert measured.total**2 == pytest.approx(np.sum(np.abs(image) ** 2) / sigma**2)
    assert measured.total * shares.sum_rule_share == pytest.approx(
        np.abs(image).max() / sigma
    )


def test_matched_filter_snr_matches_monte_carlo_across_bandwidths():
    from blochsimulator.sequence.snr_budget import acquisition_snr

    program = SequenceProgram(
        (ADCEvent(0.0, 3, 1e-4), ADCEvent(1e-3, 5, 4e-4)), duration_s=4e-3
    )
    clean = with_context(
        replace(
            result(8),
            signal=np.array([3, 2j, 1 + 1j, 0.5, -1, 2, 0.25j, 1], complex),
            adc_times_s=np.arange(8) * 1e-4,
        ),
        program,
        amplitude=4.0,
    )
    noise = ReceiverNoise("relative", 10.0)
    measured = acquisition_snr(clean, noise)
    sigma = np.repeat([0.4, 0.2], [3, 5])
    assert measured.total == pytest.approx(
        np.sqrt(np.sum(np.abs(clean.signal) ** 2 / sigma**2))
    )
    assert measured.peak == pytest.approx(2 / 0.2)
    assert measured.peak_time_s == pytest.approx(5e-4)
    weights = np.conj(clean.signal) / sigma**2 / measured.total
    statistics = [
        np.sum(weights * apply_receiver_noise(clean, replace(noise, seed=seed)).signal)
        for seed in range(1500)
    ]
    statistics = np.real(statistics)
    assert np.mean(statistics) == pytest.approx(measured.total, abs=0.1)
    assert np.std(statistics) == pytest.approx(1, rel=0.07)


def test_acquisition_snr_reports_species_and_requires_physical_weighting():
    from blochsimulator.sequence.snr_budget import acquisition_snr

    species = np.array([[3, 0, 0, 0], [0, 4j, 0, 0]], complex)
    clean = with_context(
        replace(
            result(4),
            signal=species.sum(axis=0),
            pool_names=("Pyruvate", "Lactate"),
            species_signal=species[:, None, :],
        ),
        amplitude=1.0,
    )
    measured = acquisition_snr(clean, ReceiverNoise("relative", 10.0))
    assert measured.species == (
        ("Pyruvate", pytest.approx(30)),
        ("Lactate", pytest.approx(40)),
    )
    assert measured.total == pytest.approx(50)
    with pytest.raises(ValueError, match="noise is off"):
        acquisition_snr(clean, ReceiverNoise())
    voxel_weighted = replace(
        clean, metadata={**clean.metadata, "signal_weighting": "voxel"}
    )
    with pytest.raises(ValueError, match="voxel-volume"):
        acquisition_snr(
            voxel_weighted, ReceiverNoise("calibrated", calibration=calibration())
        )


def test_budget_carries_urea_calibration_to_pyruvate_snr():
    from blochsimulator.sequence.snr_budget import snr_budget

    side = (0.2e-6) ** (1 / 3)
    obj = phantom(density=80.0, fov=(side,) * 3)
    obj.m0_map[..., 2] = 0.2 / thermal_polarization("C13", 7.0, 310.15)
    program = fid_program()
    clean = BlochSimulator(use_parallel=False).simulate_sequence(
        program, obj, signal_weighting="voxel_volume"
    )
    noise = ReceiverNoise("calibrated", seed=5, calibration=calibration(averages=4))
    sections = snr_budget(
        noise,
        phantom=obj,
        program=program,
        result=with_context(clean, program, obj=obj),
        nucleus="C13",
        field_strength_t=7.0,
    )
    assert [section.title for section in sections] == [
        "Reference phantom",
        "Simulation units",
        "Simulated object",
        "Sequence readout",
        "Image voxels",
        "Last run",
    ]
    assert sections[0].status == "set"
    nuclei = budget_row(sections, "Reference phantom", "Observed nuclei")
    assert nuclei.value == "100 mmol"
    single = budget_row(sections, "Reference phantom", "Single-scan SNR")
    assert single.number == pytest.approx(100)
    equivalent = budget_row(sections, "Simulated object", "Fully polarized equivalent")
    assert equivalent.value == "3.2 µmol"
    ceiling = budget_row(sections, "Simulated object", "Whole-object FID SNR")
    assert ceiling.number == pytest.approx(521.7, rel=1e-3)
    readout = budget_row(sections, "Sequence readout", "Whole object at 10 kHz")
    assert readout.number == pytest.approx(ceiling.number)
    strongest = budget_row(sections, "Last run", "Strongest sample")
    assert strongest.number == pytest.approx(ceiling.number, rel=1e-6)
    combined = budget_row(sections, "Last run", "All samples combined")
    assert combined.number == pytest.approx(ceiling.number * np.sqrt(32), rel=1e-6)

    # The same object at another nucleus does not match the calibration.
    mismatch = snr_budget(noise, phantom=obj, nucleus="H1", field_strength_t=7.0)
    message = budget_row(
        mismatch,
        "Simulated object",
        "The receiver calibration requires the same nucleus and field strength "
        "(C13, 7 T)",
    )
    assert message.value == ""


def test_budget_scales_relative_snr_with_bandwidth_and_grid():
    from blochsimulator.sequence import EncodingFrame
    from blochsimulator.sequence.snr_budget import snr_budget

    off = snr_budget(ReceiverNoise())
    assert [section.title for section in off] == ["Receiver noise is off"]
    program = SequenceProgram(
        (ADCEvent(0.0, 3, 1e-4), ADCEvent(1e-3, 5, 4e-4)), duration_s=4e-3
    )
    obj = phantom(shape=(4, 4, 1), fov=(0.02, 0.02, 0.005))
    noise = ReceiverNoise("relative", 40.0)
    sections = snr_budget(noise, phantom=obj, program=program)
    assert budget_row(sections, "Simulated object", "Whole-object FID SNR").number == (
        pytest.approx(40)
    )
    assert budget_row(
        sections, "Sequence readout", "Whole object at 2.5 kHz"
    ).number == pytest.approx(80)
    assert sections[-2].title == "Image voxels"
    assert "no spatial encoding grid" in sections[-2].rows[0].label
    assert sections[-1].rows[0].label.startswith("Run the simulation")

    # Half of a 4 x 4 object lies outside a 2 x 4 grid of half the read FOV.
    grid = ((2, 4), (0.01, 0.02, 0.005), EncodingFrame())
    sections = snr_budget(noise, phantom=obj, program=program, grid=grid)
    assert budget_row(sections, "Image voxels", "Voxels with signal").number == 8
    outside = budget_row(sections, "Image voxels", "Outside the FOV")
    assert outside.number == pytest.approx(0.5)
    one_sample = budget_row(sections, "Image voxels", "Brightest voxel, one sample")
    assert one_sample.number == pytest.approx(40 / 16)

    unweighted = snr_budget(noise, program=program)
    assert budget_row(
        unweighted, "Sequence readout", "Whole object at 2.5 kHz"
    ).number == pytest.approx(80)
    assert (
        "σ × 0.5"
        in budget_row(unweighted, "Sequence readout", "Whole object at 2.5 kHz").note
    )


def test_spectral_and_dynamic_reference_polarization_is_integrated_once():
    from blochsimulator.dynamic_phantom import DynamicSpectralPhantom

    shape = (2, 2, 2)
    volume = (0.02) ** 3
    spectral = SpectralPhantom(
        shape=shape,
        fov=(0.02,) * 3,
        species=[ChemicalSpecies("a", 0, 1, 1), ChemicalSpecies("b", 0, 1, 1)],
        concentration_maps={"a": np.full(shape, 0.3), "b": np.full(shape, 0.7)},
        initial_mz_maps={"a": np.full(shape, 2), "b": np.full(shape, 3)},
    )
    assert ideal_fid_amplitude(spectral) == pytest.approx((0.3 * 2 + 0.7 * 3) * volume)
    static = phantom(density=2)
    static.m0_map[..., 2] = 3
    static.rx_sensitivity_maps = np.full((1, *shape), 2j)
    assert ideal_fid_amplitude(static) == pytest.approx(12 * volume)
    dynamic = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.02,) * 3,
        pools=(
            ChemicalSpecies("Pyruvate", 0, 1, 1),
            ChemicalSpecies("Lactate", 12, 1, 1),
        ),
        initial_concentration_maps={
            "Pyruvate": np.full(shape, 6),
            "Lactate": np.zeros(shape),
        },
        initial_spin_density_maps={
            "Pyruvate": np.full(shape, 2),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.zeros(shape),
    )
    assert ideal_fid_amplitude(dynamic) == pytest.approx(6 * volume)


@pytest.mark.parametrize("extension", ["npz", "h5", "nc"])
def test_clean_signal_and_noise_metadata_are_exported(tmp_path, extension):
    noisy = apply_receiver_noise(
        with_context(result(32), amplitude=2.0), ReceiverNoise("relative", 10.0, 42)
    )
    path = noisy.save(tmp_path / f"noise.{extension}")
    if extension == "npz":
        with np.load(path) as exported:
            np.testing.assert_array_equal(
                exported["signal_noiseless"], noisy.signal_noiseless
            )
            np.testing.assert_array_equal(exported["signal"], noisy.signal)
    elif extension == "h5":
        import h5py

        with h5py.File(path) as exported:
            np.testing.assert_array_equal(
                exported["signal_noiseless"], noisy.signal_noiseless
            )
    else:
        from blochsimulator.sequence.reconstruction import load_reconstruction_dataset

        exported = load_reconstruction_dataset(path)
        np.testing.assert_array_equal(exported.signal_noiseless, noisy.signal_noiseless)
        assert (
            json.loads(exported.attrs["metadata_json"])["receiver_noise"]["seed"] == 42
        )


def test_clean_signal_and_calibration_round_trip_in_project(tmp_path):
    obj = phantom()
    ref = calibration(averages=4, temperature_k=295.0, enrichment=0.99)
    settings = ReceiverNoise("calibrated", seed=3, calibration=ref)
    noisy = apply_receiver_noise(with_context(result(16)), settings)
    path = tmp_path / "noise.blochproj"
    save_project(
        path,
        {"receiver_noise": {"calibration": ref.to_metadata()}},
        obj,
        None,
        None,
        fid_program(count=16),
        sequence_result=noisy,
    )
    loaded = load_project(path)
    restored = loaded["sequence_result"]
    np.testing.assert_array_equal(restored.signal_noiseless, noisy.signal_noiseless)
    np.testing.assert_array_equal(restored.signal, noisy.signal)
    restored_calibration = ReceiverCalibration(
        **loaded["state"]["receiver_noise"]["calibration"]
    )
    assert restored_calibration == ref
    np.testing.assert_array_equal(
        apply_receiver_noise(
            restored,
            ReceiverNoise("calibrated", seed=3, calibration=restored_calibration),
        ).signal,
        noisy.signal,
    )


def test_object_enrichment_scales_noise_and_follows_the_phantom_metadata():
    base = calibration(target_temperature_k=293.15)
    sigma = base.noise_std(1e-4)
    labelled = calibration(target_temperature_k=293.15, target_enrichment=0.5)
    assert labelled.target_signal_scale == pytest.approx(0.5 * base.target_signal_scale)
    # Half the observed nuclei per simulated mol: noise is twice as large in
    # simulated units.
    assert labelled.noise_std(1e-4) == pytest.approx(2 * sigma)
    noise = ReceiverNoise(mode="calibrated", calibration=base)
    applied = noise.for_object({"temperature_k": 293.15, "isotope_enrichment": 0.5})
    assert applied.calibration.target_enrichment == 0.5
    assert applied.calibration.noise_std(1e-4) == pytest.approx(2 * sigma)
    assert noise.for_object({}) is noise
    with pytest.raises(ValueError, match="enrichment"):
        calibration(target_enrichment=0.0)
    with pytest.raises(ValueError, match="enrichment"):
        calibration(target_enrichment=1.5)
