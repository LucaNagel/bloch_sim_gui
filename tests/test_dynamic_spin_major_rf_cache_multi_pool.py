"""Cached RF propagators of the spin-major executor for N-pool phantoms.

Repeated pulses run as cached per-spin affine propagators for two to six
pools, also with per-spin inflow arrival delays, and pulses that differ only
by a constant phase (RF spoiling) reuse one entry rotated about z. The
interval-major native plan (``_native_spin_major=False``) is the reference at
a relative error of 1e-12.
"""

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator import dynamic_phantom as dynamic_module
from blochsimulator.dynamic_phantom import (
    DynamicSpectralPhantom,
    PyruvateInflow,
    TimeCurve,
)
from blochsimulator.sequence import (
    ADCEvent,
    GradientEvent,
    RFEvent,
    SequenceProgram,
    SpinSampling,
)
from blochsimulator.spectral_phantom import ChemicalSpecies

pytest.importorskip("blochsimulator.dynamic_spin_major_cy")

# A dyadic raster keeps the plan times of repeated pulses exact, so their RF
# segments have bitwise equal keys up to the RF phase.
RASTER = 2.0**-16
PULSE = np.array([25.0 + 8.0j, 60.0 - 6.0j, 85.0, 60.0 + 6.0j, 25.0 - 8.0j])
PERIOD = 10
REPETITIONS = 48
OUTPUTS = (
    "signal",
    "species_signal",
    "final_magnetization",
    "final_pool_magnetization",
    "checkpoint_magnetization",
    "checkpoint_pool_magnetization",
)
POOLS = (
    ("Pyruvate", 0.0),
    ("Lactate", 12.0),
    ("Alanine", 6.0),
    ("Bicarbonate", -7.0),
    ("Urea", 3.0),
    ("Glutamate", -11.0),
)


def _spoil_phase(repetition):
    """Quadratic RF spoiling phase of one repetition (117 degree increment)."""
    return np.deg2rad((117.0 * repetition * (repetition + 1) / 2) % 360.0)


def _phantom(
    n_pools=3,
    shape=(40, 1, 1),
    *,
    inflow="polarized",
    tracked=True,
    conversion_start=0.0,
    max_delay_s=40 * RASTER,
    interpolation="linear",
    outside="hold",
):
    grid = np.linspace(0.0, 1.0, int(np.prod(shape))).reshape(shape)
    pools = POOLS[:n_pools]
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.06, 0.01, 0.01),
        pools=tuple(
            ChemicalSpecies(name, shift, 30.0 - 3.0 * index, 1.0)
            for index, (name, shift) in enumerate(pools)
        ),
        initial_concentration_maps={
            name: (1.0 + 0.2 * grid if index == 0 else 0.04 / index + 0.03 * grid)
            for index, (name, _) in enumerate(pools)
        },
        kpl_map_s_inv=20.0 + 60.0 * grid,
        b0_map=-40.0 + 80.0 * grid,
        conversion_start_s=conversion_start,
        nucleus="C13",
    )
    if n_pools > 2:
        # Opposite gradient to kPL, so both conversion maps stay per spin.
        phantom.kpa_map_s_inv = 10.0 + 50.0 * grid[::-1]
    if tracked:
        phantom.initial_spin_density_maps = {
            name: (1.0 + 0.1 * grid if index == 0 else np.full(shape, 0.1 / index))
            for index, (name, _) in enumerate(pools)
        }
        phantom.equilibrium_polarization = 0.02
    # Knots inside pulses: occurrences whose source queries cross one run
    # stepwise, per occurrence with global sources and per tile with delays.
    rate = TimeCurve(
        (0.0, 60 * RASTER, 203 * RASTER, 330 * RASTER),
        (50.0, 600.0, 200.0, 400.0),
        interpolation,
        outside,
    )
    polarization = TimeCurve(
        (0.0, 152 * RASTER, 300 * RASTER), (0.5, 0.3, 0.45), interpolation, outside
    )
    delays = (
        {}
        if max_delay_s is None
        else dict(arrival_delay_map_s=max_delay_s * grid, max_step_s=RASTER)
    )
    if inflow == "direct":
        phantom.pyruvate_inflow = PyruvateInflow(rate, 0.8 + 0.4 * grid, **delays)
    elif inflow == "polarized":
        phantom.pyruvate_inflow = PyruvateInflow(
            rate, 0.8 + 0.4 * grid, polarization, **delays
        )
    return phantom


def _program(scales=(1.0,) * REPETITIONS, *, gradient=False, spoiled=True):
    events = []
    for repetition, scale in enumerate(scales):
        start = repetition * PERIOD * RASTER
        phase = _spoil_phase(repetition) if spoiled else 0.0
        events.append(RFEvent(start, scale * PULSE * np.exp(1j * phase), RASTER))
        if gradient:
            events.append(GradientEvent("x", start, np.full(PULSE.size, 90.0), RASTER))
        events.append(
            ADCEvent(
                start + (PULSE.size + 1) * RASTER,
                2,
                RASTER,
                phase_offset_rad=phase + 0.1 * repetition,
            )
        )
    return SequenceProgram(tuple(events), duration_s=len(scales) * PERIOD * RASTER)


def _run(phantom, program, kernel="native_serial", **kwargs):
    kwargs.setdefault("checkpoints_s", (7 * RASTER, program.duration_s))
    kwargs.setdefault("simulation_timestep_s", RASTER)
    kwargs.setdefault("spoiler_mode", "ideal")
    simulator = BlochSimulator(
        use_parallel=kwargs.pop("use_parallel", False),
        num_threads=kwargs.pop("num_threads", 1),
    )
    return simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel=kernel, **kwargs
    )


def _rf_cache(result, name):
    return result.metadata[f"native_spin_major_rf_propagator_{name}"]


def _assert_close(actual, expected, tolerance=1e-12):
    for name in OUTPUTS:
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=tolerance,
            atol=tolerance,
            err_msg=name,
        )


def _assert_equal(actual, expected):
    for name in OUTPUTS:
        np.testing.assert_array_equal(
            getattr(actual, name), getattr(expected, name), err_msg=name
        )


@pytest.mark.parametrize(
    "kernel, n_pools, inflow, tracked, max_delay_s, interpolation, outside, start",
    [
        ("native_serial", 3, "polarized", True, 40 * RASTER, "linear", "hold", 0.0),
        ("native_parallel", 3, "polarized", True, 40 * RASTER, "step", "zero", 0.0),
        ("native_serial", 3, "direct", False, 40 * RASTER, "linear", "zero", 0.0),
        (
            "native_parallel",
            3,
            "polarized",
            False,
            40 * RASTER,
            "linear",
            "hold",
            (24 * PERIOD + 2) * RASTER,
        ),
        ("native_serial", 3, "polarized", True, None, "linear", "hold", 0.0),
        ("native_serial", 3, None, True, None, "linear", "hold", 0.0),
        ("native_serial", 2, "polarized", True, 40 * RASTER, "linear", "hold", 0.0),
        ("native_parallel", 4, "polarized", True, 40 * RASTER, "linear", "zero", 0.0),
        (
            "native_parallel",
            6,
            "direct",
            False,
            40 * RASTER,
            "step",
            "hold",
            (24 * PERIOD + 2) * RASTER,
        ),
    ],
)
def test_rf_cache_matches_interval_major_for_spoiled_multi_pool_plans(
    kernel, n_pools, inflow, tracked, max_delay_s, interpolation, outside, start
):
    phantom = _phantom(
        n_pools,
        inflow=inflow,
        tracked=tracked,
        conversion_start=start,
        max_delay_s=max_delay_s,
        interpolation=interpolation,
        outside=outside,
    )
    program = _program()
    kwargs = dict(
        spin_sampling=SpinSampling((4, 1, 1)),
        use_parallel=kernel == "native_parallel",
        num_threads=3,
    )

    interval_major = _run(phantom, program, kernel, _native_spin_major=False, **kwargs)
    stepwise = _run(
        phantom, program, kernel, _native_spin_major_rf_cache=False, **kwargs
    )
    cached = _run(phantom, program, kernel, **kwargs)

    assert interval_major.metadata["native_spin_major_enabled"] is False
    assert cached.metadata["native_spin_major_enabled"] is True
    assert _rf_cache(cached, "enabled") is True
    assert _rf_cache(cached, "fallback_reason") is None
    # Spoiling turns nearly every hit; a phase equal to the build's is not. A
    # conversion start inside a pulse splits the pulses into two keys.
    hits = _rf_cache(cached, "applications") - _rf_cache(cached, "builds")
    assert _rf_cache(cached, "keys") == (1 if start == 0.0 else 2)
    assert _rf_cache(cached, "builds") == _rf_cache(cached, "keys")
    assert hits - 3 <= _rf_cache(cached, "rotated_applications") <= hits
    assert _rf_cache(cached, "applications") > REPETITIONS // 2
    assert (
        _rf_cache(cached, "applications") + _rf_cache(cached, "stepwise_segments")
        == REPETITIONS
    )
    if inflow is not None and max_delay_s is None:
        assert _rf_cache(cached, "stepwise_segments") >= 1
    if inflow is not None and max_delay_s is not None:
        assert _rf_cache(cached, "stepwise_tiles") > 0
    else:
        assert _rf_cache(cached, "stepwise_tiles") == 0
    assert cached.species_signal.shape[0] == n_pools
    _assert_close(cached, interval_major)
    _assert_close(cached, stepwise)


def test_rf_cache_supports_spatial_tx_coils_and_slice_selection_with_delays():
    shape = (40, 1, 1)
    grid = np.linspace(0.0, 1.0, int(np.prod(shape))).reshape(shape)
    phantom = _phantom(3, shape, interpolation="step", outside="zero")
    phantom.tx_sensitivity_map = (0.75 + 0.3 * grid) * np.exp(1j * (-0.2 + 0.3 * grid))
    phantom.rx_sensitivity_maps = np.stack(
        (
            np.ones(shape, dtype=np.complex128),
            (0.8 + 0.1 * grid) * np.exp(1j * (0.1 + 0.2 * grid)),
        )
    )
    program = _program(gradient=True)
    kwargs = dict(
        spin_sampling=SpinSampling((4, 1, 1)),
        parent_voxel_chunk_size=24,
        spoiler_mode="gradient",
        use_parallel=True,
        num_threads=3,
    )

    interval_major = _run(
        phantom, program, "native_parallel", _native_spin_major=False, **kwargs
    )
    cached = _run(phantom, program, "native_parallel", **kwargs)

    assert cached.metadata["native_spin_major_parent_blocks"] == 2
    assert _rf_cache(cached, "enabled") is True
    assert _rf_cache(cached, "rotated_applications") > 0
    assert _rf_cache(cached, "stepwise_tiles") > 0
    assert cached.signal.shape[0] == 2
    _assert_close(cached, interval_major)


def test_rf_cache_keeps_pulses_with_phase_ramps_apart():
    # An off-resonant copy of the pulse has a phase ramp, not a constant
    # phase, so it cannot reuse the entry of the on-resonant pulse.
    ramp = np.exp(0.4j * np.arange(PULSE.size))
    scales = tuple(
        1.0 if repetition % 2 == 0 else ramp for repetition in range(REPETITIONS)
    )
    phantom = _phantom(3)
    program = _program(scales)

    interval_major = _run(phantom, program, _native_spin_major=False)
    cached = _run(phantom, program)

    assert _rf_cache(cached, "keys") == 2
    assert _rf_cache(cached, "builds") == 2
    assert _rf_cache(cached, "rotated_applications") > 0
    _assert_close(cached, interval_major)


def test_rf_phasor_requires_a_constant_phase_to_rounding():
    phasor = dynamic_module._spin_major_rf_phasor
    turned = PULSE * np.exp(0.7j)

    assert phasor(PULSE, PULSE) == 1.0
    assert phasor(turned, PULSE) == pytest.approx(np.exp(0.7j), abs=1e-15)
    assert phasor(1j * PULSE, PULSE) == 1j
    assert phasor(turned * (1.0 + 1e-13), PULSE) is None
    assert phasor(PULSE * np.exp(0.4j * np.arange(PULSE.size)), PULSE) is None
    assert phasor(np.zeros(3, dtype=complex), np.zeros(3, dtype=complex)) == 1.0
    assert phasor(PULSE, np.zeros(PULSE.size, dtype=complex)) is None


def test_rf_cache_multi_pool_result_does_not_depend_on_slices_or_threads(monkeypatch):
    phantom = _phantom(3, (12, 10, 10), interpolation="step", outside="zero")
    program = _program((1.0,) * 20 + (0.5,) * 20 + (1.0,) * 20)
    kwargs = dict(use_parallel=True)

    serial = _run(phantom, program, "native_serial")
    parallel = _run(phantom, program, "native_parallel", num_threads=4, **kwargs)
    # Slices shorter than one pulse end at cached segment boundaries.
    monkeypatch.setattr(dynamic_module, "_SPIN_MAJOR_SLICE_INTERVALS", 4)
    monkeypatch.setattr(dynamic_module, "_SPIN_MAJOR_SLICE_ADC_VALUES", 2)
    sliced = _run(phantom, program, "native_parallel", num_threads=3, **kwargs)
    # One entry rebuilds the first pulse from another spoiling phase, which
    # changes rounding only.
    monkeypatch.setattr(dynamic_module, "_SPIN_MAJOR_RF_PROPAGATOR_ENTRIES", 1)
    evicting = _run(phantom, program, "native_parallel", num_threads=3, **kwargs)

    assert _rf_cache(serial, "keys") == 2
    assert _rf_cache(serial, "builds") == 2
    assert _rf_cache(evicting, "builds") == 3
    assert _rf_cache(serial, "stepwise_tiles") > 0
    assert sliced.metadata["native_spin_major_chunks"] > (
        parallel.metadata["native_spin_major_chunks"]
    )
    for result in (parallel, sliced):
        assert _rf_cache(result, "enabled") is True
        assert _rf_cache(result, "stepwise_tiles") == _rf_cache(
            serial, "stepwise_tiles"
        )
        _assert_equal(result, serial)
    _assert_close(evicting, serial)


def _with_washout(phantom, sources):
    """Add a voxel loss gradient and, optionally, two source compartments."""
    shape = phantom.shape
    grid = np.linspace(0.0, 1.0, int(np.prod(shape))).reshape(shape)
    n_pools = len(phantom.pools)
    washout = dict(loss_map_s_inv=20.0 + 200.0 * grid[::-1])
    if sources:
        compartments = np.full(shape, -1, dtype=np.int32)
        compartments.reshape(-1)[-2:] = (0, 1)
        source_mask = compartments >= 0
        phantom.kpl_map_s_inv = np.where(source_mask, 0.0, phantom.kpl_map_s_inv)
        if getattr(phantom, "kpa_map_s_inv", None) is not None:
            phantom.kpa_map_s_inv = np.where(source_mask, 0.0, phantom.kpa_map_s_inv)
        concentration = np.stack(
            (
                np.broadcast_to((30.0, 300.0, 120.0), (n_pools, 3)),
                np.broadcast_to((200.0, 0.0, 500.0), (n_pools, 3)),
            )
        )
        washout.update(
            compartment_map=compartments,
            times_s=np.asarray((40 * RASTER, 205 * RASTER, 400 * RASTER)),
            concentration_rates=concentration,
            magnetization_rates=concentration * np.asarray((4.0, 2.5))[:, None, None],
        )
    phantom.voxel_washout = dynamic_module.VoxelWashout(**washout)
    phantom.voxel_washout.validate(phantom.shape, n_pools)
    return phantom


@pytest.mark.parametrize(
    "kernel, n_pools, tracked, sources",
    [
        ("native_serial", 3, True, True),
        ("native_parallel", 3, False, False),
        ("native_serial", 2, True, True),
    ],
)
def test_rf_cache_carries_voxel_washout_loss_and_steps_source_tiles(
    kernel, n_pools, tracked, sources
):
    phantom = _with_washout(
        _phantom(n_pools, inflow=None, tracked=tracked, max_delay_s=None), sources
    )
    program = _program()
    kwargs = dict(
        spin_sampling=SpinSampling((4, 1, 1)),
        use_parallel=kernel == "native_parallel",
        num_threads=3,
    )

    reference = _run(phantom, program, "optimized")
    stepwise = _run(
        phantom, program, kernel, _native_spin_major_rf_cache=False, **kwargs
    )
    cached = _run(phantom, program, kernel, **kwargs)

    assert cached.metadata["sequence_kernel"] == kernel
    assert _rf_cache(cached, "enabled") is True
    assert _rf_cache(cached, "applications") > REPETITIONS // 2
    # Only the tile holding the source voxels steps through the pulses.
    if sources:
        assert (
            0 < _rf_cache(cached, "stepwise_tiles") <= _rf_cache(cached, "applications")
        )
    else:
        assert _rf_cache(cached, "stepwise_tiles") == 0
    _assert_close(cached, stepwise)
    _assert_close(cached, reference, 1e-10)
