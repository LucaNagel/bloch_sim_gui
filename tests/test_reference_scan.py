"""Signal model of the receiver calibration's phantom FID scan."""

import json

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.b1_fields import create_b1_preset
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import (
    ReceiverCalibration,
    ReferenceScan,
    reference_signal_efficiency,
)
from blochsimulator.sequence.internal_fid import make_internal_fid

# Sphere of 12 cells across its diameter, centred one radius off-axis in x.
CELLS = 12
VOLUME_ML = 4.0
RADIUS = (3 * VOLUME_ML * 1e-6 / (4 * np.pi)) ** (1 / 3)
CELL = 2 * RADIUS / CELLS


def coil(tx="birdcage_cp", rx="surface_loop"):
    shape, fov = (32, 32, 32), (0.06, 0.06, 0.06)
    options = {"ramp_mode": "phase"}
    return (
        create_b1_preset(tx, shape, fov, **options),
        create_b1_preset(rx, shape, fov, kind="receive", **options),
    )


def sphere_phantom(tx_field, rx_field, *, t1, t2, density=1.0):
    """Voxel grid whose active cells are exactly the model's sample points."""
    shape = (2 * CELLS, CELLS, CELLS)
    fov = tuple(count * CELL for count in shape)
    obj = Phantom(
        shape=shape,
        fov=fov,
        t1_map=np.full(shape, t1),
        t2_map=np.full(shape, t2),
        pd_map=np.full(shape, density),
    )
    offset = np.array([RADIUS, 0.0, 0.0])
    inside = np.sum((obj.positions - offset) ** 2, axis=1) <= RADIUS**2
    obj.mask = inside.reshape(shape)
    obj.pd_map = np.where(obj.mask, density, 0.0)
    obj.tx_sensitivity_map = tx_field.resample_to_phantom(obj)[0]
    obj.rx_sensitivity_maps = rx_field.resample_to_phantom(obj)
    return obj


def scan(**kwargs):
    values = dict(volume_ml=VOLUME_ML, position_m=(RADIUS, 0.0, 0.0))
    values.update(kwargs)
    return ReferenceScan(**values)


def test_ideal_homogeneous_scan_is_sin_flip():
    for flip in (90.0, 30.0, 150.0):
        result = reference_signal_efficiency(scan(flip_angle_deg=flip))
        assert result.efficiency == pytest.approx(np.sin(np.deg2rad(flip)))
        assert result.homogeneous == result.efficiency
        assert result.coverage == 1.0
    # Full recovery between repetitions (no T1) never saturates.
    relaxed = scan(averages=16, dummy_scans=2, repetition_time_s=0.1)
    assert reference_signal_efficiency(relaxed).efficiency == pytest.approx(1.0)


def test_saturation_matches_the_spoiled_recursion():
    reference = scan(
        flip_angle_deg=60.0,
        averages=7,
        dummy_scans=3,
        repetition_time_s=0.8,
        t1_s=2.5,
        equilibrium_fraction=0.9,
    )
    flip = np.deg2rad(np.array([0.0, 20.0, 60.0, 90.0, 140.0]))
    e1 = np.exp(-0.8 / 2.5)
    mz, acquired = np.ones_like(flip), []
    for repetition in range(10):
        if repetition >= 3:
            acquired.append(mz.copy())
        mz = e1 * np.cos(flip) * mz + (1 - e1) * 0.9
    np.testing.assert_allclose(
        reference.mean_longitudinal(flip), np.mean(acquired, axis=0), rtol=1e-13
    )
    # A long TR relaxes fully again; one scan never saturates.
    assert scan(averages=4, repetition_time_s=1e3, t1_s=1.0).mean_longitudinal(
        1.0
    ) == pytest.approx(1.0)
    assert scan(t1_s=1.0).mean_longitudinal(np.pi / 2) == 1.0


def test_t2star_decay_until_the_measured_point():
    decayed = scan(t2star_s=0.02, amplitude_delay_s=0.005)
    assert reference_signal_efficiency(decayed).efficiency == pytest.approx(
        np.exp(-0.25)
    )
    assert scan(amplitude_delay_s=0.005).decay == 1.0


def test_sample_points_fill_sphere_and_cylinder_at_the_position():
    sphere = scan().sample_points(24)
    assert np.mean(sphere, axis=0) == pytest.approx([RADIUS, 0.0, 0.0], abs=1e-12)
    assert np.max(np.linalg.norm(sphere - [RADIUS, 0, 0], axis=1)) <= RADIUS
    cylinder = ReferenceScan(volume_ml=50.0, geometry="cylinder", diameter_m=0.028)
    assert cylinder.length_m == pytest.approx(50e-6 / (np.pi * 0.014**2))
    points = cylinder.sample_points(24)
    assert np.ptp(points[:, 2]) == pytest.approx(
        cylinder.length_m * (1 - 1 / 24), rel=0.05
    )
    assert np.max(np.hypot(points[:, 0], points[:, 1])) <= 0.014


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"volume_ml": 0}, "volume"),
        ({"flip_angle_deg": 0}, "flip angle"),
        ({"averages": 0}, "Averages"),
        ({"dummy_scans": -1}, "Dummy"),
        ({"t1_s": 1.0, "averages": 2}, "TR"),
        ({"t2star_s": 0}, "T2"),
        ({"amplitude_delay_s": -1}, "delay"),
        ({"geometry": "cube"}, "geometry"),
        ({"geometry": "cylinder"}, "diameter"),
        ({"position_m": (0, 0)}, "position"),
    ],
)
def test_reference_scan_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        scan(**kwargs)


@pytest.mark.parametrize(
    "tx, rx",
    [
        ("birdcage_cp", "surface_loop"),
        # Equal phase ramps cancel only with the conjugate transmit phase.
        ("linear_ramp", "linear_ramp"),
    ],
)
def test_model_matches_bloch_simulation_in_coil_fields(tx, rx):
    """Offset sphere in coil fields, saturated by dummy scans and averages."""
    tx_field, rx_field = coil(tx, rx)
    t1, t2, tr, delay = 0.6, 0.01, 0.5, 2e-3
    dummy, averages, flip = 2, 3, 70.0
    obj = sphere_phantom(tx_field, rx_field, t1=t1, t2=t2)
    duration, dwell = 1e-6, 1e-7
    program = make_internal_fid(
        flip_angle_deg=flip,
        rf_duration_s=duration,
        num_samples=1,
        sampling_bandwidth_hz=1 / dwell,
        adc_delay_s=delay,
        repetitions=dummy + averages,
        repetition_time_s=tr,
    )
    signal = BlochSimulator(use_parallel=False).simulate_sequence(program, obj)
    acquired = np.asarray(signal.signal).reshape(-1)[dummy:]
    simulated = abs(np.mean(acquired)) / np.count_nonzero(obj.mask)

    model = reference_signal_efficiency(
        scan(
            flip_angle_deg=flip,
            averages=averages,
            dummy_scans=dummy,
            repetition_time_s=tr,
            t1_s=t1,
            t2star_s=t2,
            amplitude_delay_s=duration / 2 + delay + dwell / 2,
        ),
        tx_field,
        rx_field,
        points_per_axis=CELLS,
    )
    assert model.sample_points == np.count_nonzero(obj.mask)
    assert model.efficiency == pytest.approx(simulated, rel=1e-3)
    assert model.coverage == 1.0
    if tx == "birdcage_cp":
        # The coil reduces the signal well below the homogeneous prediction.
        assert model.efficiency < 0.8 * model.homogeneous
        assert "birdcage" in model.coil_model.lower()
    else:
        # Interpolating a rotating phase lowers |B1| slightly between nodes.
        assert model.efficiency == pytest.approx(model.homogeneous, rel=1e-2)


def test_calibration_from_a_simulated_scan_recovers_the_true_noise():
    """A thermal 13C scan with saturation, decay and coil maps, then noise."""
    tx_field, rx_field = coil()
    concentration, t1, t2, tr, averages = 3000.0, 1.2, 0.02, 1.0, 4
    obj = sphere_phantom(tx_field, rx_field, t1=t1, t2=t2, density=concentration)
    program = make_internal_fid(
        flip_angle_deg=90.0,
        rf_duration_s=1e-6,
        num_samples=1,
        sampling_bandwidth_hz=1e7,
        repetitions=averages,
        repetition_time_s=tr,
    )
    clean = BlochSimulator(use_parallel=False).simulate_sequence(
        program, obj, signal_weighting="voxel_volume"
    )
    amplitude = abs(np.mean(np.asarray(clean.signal).reshape(-1)))
    true_sigma = 1e-9
    reference = scan(
        averages=averages,
        repetition_time_s=tr,
        t1_s=t1,
        t2star_s=t2,
        amplitude_delay_s=0.5e-6 + 0.5e-7,
    )
    efficiency = reference_signal_efficiency(
        reference, tx_field, rx_field, points_per_axis=CELLS
    ).efficiency
    calibration = ReceiverCalibration(
        measured_amplitude=amplitude,
        measured_noise_std=true_sigma / np.sqrt(averages),
        concentration_mm=concentration,
        volume_ml=np.count_nonzero(obj.mask) * obj.voxel_volume_m3 * 1e6,
        nucleus="C13",
        field_strength_t=7.0,
        dwell_s=1e-4,
        averages=averages,
        target_temperature_k=293.15,
        signal_efficiency=efficiency,
        reference_scan=reference.to_metadata(),
    )
    assert calibration.noise_std(1e-4) == pytest.approx(true_sigma, rel=2e-3)
    # The ideal-reference assumption mistakes the lost signal for lower gain.
    ideal = ReceiverCalibration(
        **{
            **calibration.to_metadata(),
            "signal_efficiency": None,
            "reference_scan": None,
        }
    )
    assert ideal.noise_std(1e-4) / calibration.noise_std(1e-4) == pytest.approx(
        1 / efficiency
    )
    assert calibration.derived_metadata()["reference_signal_fraction"] == efficiency
    # Calibration metadata survives a JSON project round trip unchanged.
    restored = ReceiverCalibration(**json.loads(json.dumps(calibration.to_metadata())))
    assert restored == calibration


def test_multichannel_coil_is_rejected():
    array = create_b1_preset(
        "circular_array", (16, 16, 16), (0.06,) * 3, kind="receive"
    )
    with pytest.raises(ValueError, match="one receive channel"):
        reference_signal_efficiency(scan(), None, array)
