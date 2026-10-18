"""Spin-major executor for N-pool phantoms and spatially delayed inflow.

The interval-major native plan (``_native_spin_major=False``) is the reference:
its multi-pool driver for three or more pools and its spatial source tables for
per-voxel inflow arrival delays. The acceptance bound for the spin-major path
is a relative error of 1e-9; the comparisons below use the tighter 1e-12 of
the two-pool spin-major tests.
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
from blochsimulator.kernel_capabilities import detect_kernel_capabilities
from blochsimulator.sequence import (
    ADCEvent,
    GradientEvent,
    RFEvent,
    SequenceProgram,
    SpinSampling,
)
from blochsimulator.spectral_phantom import ChemicalSpecies

pytest.importorskip("blochsimulator.dynamic_spin_major_cy")

RASTER = 20e-6
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
    ("Hydrate", 9.0),
)


def _phantom(
    n_pools=3,
    shape=(8, 1, 1),
    *,
    inflow=None,
    tracked=False,
    conversion_start=0.0,
    max_delay_s=None,
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
        kpl_map_s_inv=0.01 + 0.09 * grid,
        b0_map=-5.0 + 10.0 * grid,
        conversion_start_s=conversion_start,
        nucleus="C13",
    )
    if n_pools > 2:
        # Opposite gradient to kPL, so both conversion maps stay per spin.
        phantom.kpa_map_s_inv = 0.02 + 0.05 * grid[::-1]
    if tracked:
        phantom.initial_spin_density_maps = {
            name: (1.0 + 0.1 * grid if index == 0 else np.full(shape, 0.1 / index))
            for index, (name, _) in enumerate(pools)
        }
        phantom.equilibrium_polarization = 0.02
    rate = TimeCurve(
        (0.0, 3 * RASTER, 8 * RASTER, 13 * RASTER),
        (0.05, 0.6, 0.2, 0.4),
        interpolation,
        outside,
    )
    polarization = TimeCurve(
        (0.0, 6 * RASTER, 11 * RASTER), (0.5, 0.3, 0.45), interpolation, outside
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


def _program():
    return SequenceProgram(
        (
            RFEvent(0.0, np.linspace(40.0 + 6.0j, 85.0 - 12.0j, 10), RASTER),
            GradientEvent("x", 0.0, np.linspace(-110.0, 150.0, 20), RASTER),
            RFEvent(14 * RASTER, np.full(3, 60.0 - 20.0j), RASTER),
            ADCEvent(3 * RASTER, 9, 2 * RASTER, phase_offset_rad=0.29),
        ),
        duration_s=25 * RASTER,
        metadata={"definitions": {"IdealSpoilerEndTimes": [12 * RASTER]}},
    )


def _run(phantom, program, kernel="native_serial", **kwargs):
    kwargs.setdefault("checkpoints_s", (7 * RASTER, 25 * RASTER))
    kwargs.setdefault("simulation_timestep_s", RASTER)
    kwargs.setdefault("spoiler_mode", "ideal")
    simulator = BlochSimulator(
        use_parallel=kwargs.pop("use_parallel", False),
        num_threads=kwargs.pop("num_threads", 1),
    )
    return simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel=kernel, **kwargs
    )


def _assert_close(actual, expected, tolerance=1e-12):
    for name in OUTPUTS:
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=tolerance,
            atol=tolerance,
            err_msg=name,
        )


def _assert_spin_major(result):
    assert result.metadata["native_spin_major_enabled"] is True
    assert result.metadata["native_spin_major_fallback_reason"] is None
    assert result.metadata["native_spin_major_intervals"] == (
        result.metadata["compiled_interval_count"]
        * result.metadata.get("native_spin_major_parent_blocks", 1)
    )


def _assert_interval_major(result):
    assert result.metadata["native_spin_major_enabled"] is False
    assert result.metadata["native_spin_major_fallback_reason"] == (
        "disabled by request"
    )
    assert result.metadata["native_full_plan_enabled"] is True


@pytest.mark.parametrize("kernel", ["native_serial", "native_parallel"])
@pytest.mark.parametrize(
    "n_pools, inflow, tracked, conversion_start",
    [
        (3, None, False, 0.0),
        (3, None, True, 0.0),
        (3, "direct", False, 0.0),
        (3, "polarized", False, 0.0),
        (3, "polarized", True, 0.0),
        (3, "polarized", True, 9 * RASTER),
        (4, "polarized", True, 0.0),
        (6, "direct", False, 9 * RASTER),
    ],
)
def test_spin_major_matches_multi_pool_plan(
    kernel, n_pools, inflow, tracked, conversion_start
):
    phantom = _phantom(
        n_pools, inflow=inflow, tracked=tracked, conversion_start=conversion_start
    )
    program = _program()
    kwargs = dict(use_parallel=kernel == "native_parallel", num_threads=2)

    reference = _run(phantom, program, "optimized")
    interval_major = _run(phantom, program, kernel, _native_spin_major=False, **kwargs)
    spin_major = _run(phantom, program, kernel, **kwargs)

    _assert_spin_major(spin_major)
    _assert_interval_major(interval_major)
    assert interval_major.metadata["native_full_plan_model"].startswith("multi-pool-")
    assert spin_major.species_signal.shape[0] == n_pools
    _assert_close(spin_major, interval_major)
    _assert_close(spin_major, reference)


@pytest.mark.parametrize("n_pools", [2, 3])
@pytest.mark.parametrize(
    "inflow, tracked",
    [("direct", False), ("polarized", False), ("polarized", True)],
)
@pytest.mark.parametrize(
    "interpolation, outside",
    [("linear", "hold"), ("linear", "zero"), ("step", "hold"), ("step", "zero")],
)
def test_spin_major_matches_spatial_inflow_plan(
    n_pools, inflow, tracked, interpolation, outside
):
    phantom = _phantom(
        n_pools,
        inflow=inflow,
        tracked=tracked,
        max_delay_s=6 * RASTER,
        interpolation=interpolation,
        outside=outside,
    )
    program = _program()

    reference = _run(phantom, program, "optimized")
    interval_major = _run(phantom, program, _native_spin_major=False)
    spin_major = _run(phantom, program)

    _assert_spin_major(spin_major)
    _assert_interval_major(interval_major)
    for result in (interval_major, spin_major):
        assert result.metadata["spatial_inflow_arrival_delays"] is True
    _assert_close(spin_major, interval_major)
    _assert_close(spin_major, reference)


@pytest.mark.parametrize(
    "n_pools, inflow, tracked, interpolation, outside",
    [
        (3, "polarized", True, "step", "zero"),
        (3, "direct", False, "linear", "zero"),
        (2, "polarized", False, "linear", "zero"),
        (3, "polarized", False, "step", "hold"),
    ],
)
def test_spin_major_spatial_inflow_over_tiles_with_tx_and_coils(
    n_pools, inflow, tracked, interpolation, outside
):
    # 160 spins fill three tiles. Delays up to 30 rasters exceed the sequence,
    # so late tiles see no inflow and early tiles outlive zero-outside curves.
    shape = (40, 1, 1)
    grid = np.linspace(0.0, 1.0, int(np.prod(shape))).reshape(shape)
    phantom = _phantom(
        n_pools,
        shape,
        inflow=inflow,
        tracked=tracked,
        max_delay_s=30 * RASTER,
        interpolation=interpolation,
        outside=outside,
    )
    phantom.tx_sensitivity_map = (0.75 + 0.3 * grid) * np.exp(1j * (-0.2 + 0.3 * grid))
    phantom.rx_sensitivity_maps = np.stack(
        (
            np.ones(shape, dtype=np.complex128),
            (0.8 + 0.1 * grid) * np.exp(1j * (0.1 + 0.2 * grid)),
        )
    )
    program = _program()
    kwargs = dict(
        spin_sampling=SpinSampling((4, 1, 1)),
        parent_voxel_chunk_size=24,
        spoiler_mode="gradient",
    )
    parallel = dict(use_parallel=True, num_threads=3)

    reference = _run(phantom, program, "optimized", **kwargs)
    interval_major = _run(
        phantom,
        program,
        "native_parallel",
        _native_spin_major=False,
        **parallel,
        **kwargs,
    )
    spin_major = _run(phantom, program, "native_parallel", **parallel, **kwargs)

    assert spin_major.metadata["native_spin_major_enabled"] is True
    assert spin_major.metadata["native_spin_major_parent_blocks"] == 2
    _assert_interval_major(interval_major)
    _assert_close(spin_major, interval_major)
    _assert_close(spin_major, reference)


def test_spin_major_multi_pool_result_does_not_depend_on_slices_or_threads(
    monkeypatch,
):
    phantom = _phantom(
        3,
        (12, 10, 10),
        inflow="polarized",
        tracked=True,
        max_delay_s=6 * RASTER,
        interpolation="step",
        outside="zero",
    )
    program = _program()
    kwargs = dict(use_parallel=True)

    serial = _run(phantom, program, "native_serial")
    parallel = _run(phantom, program, "native_parallel", num_threads=4, **kwargs)
    monkeypatch.setattr(dynamic_module, "_SPIN_MAJOR_SLICE_INTERVALS", 4)
    monkeypatch.setattr(dynamic_module, "_SPIN_MAJOR_SLICE_ADC_VALUES", 2)
    sliced = _run(phantom, program, "native_parallel", num_threads=3, **kwargs)

    assert serial.metadata["native_spin_major_enabled"] is True
    assert parallel.metadata["native_spin_major_threads"] == 4
    assert sliced.metadata["native_spin_major_chunks"] > (
        parallel.metadata["native_spin_major_chunks"]
    )
    for name in OUTPUTS:
        np.testing.assert_array_equal(getattr(parallel, name), getattr(serial, name))
        np.testing.assert_array_equal(getattr(sliced, name), getattr(serial, name))


def test_spin_major_falls_back_beyond_six_pools():
    phantom = _phantom(7, inflow="polarized", tracked=True)

    result = _run(phantom, _program())

    assert result.metadata["native_spin_major_enabled"] is False
    assert result.metadata["native_spin_major_fallback_reason"] == (
        "more than six pools are unsupported"
    )


def test_spin_major_slice_limits_scale_with_pools_and_inflow():
    # A budget that binds both limits for 16 receive channels on one thread.
    budget = 32 * 1024**2
    limits = dynamic_module._spin_major_slice_limits
    two_pool = limits(budget, 16, 1)

    assert limits(budget, 16, 1, 2, False) == two_pool
    three_pool = limits(budget, 16, 1, 3, True)
    six_pool = limits(budget, 16, 1, 6, True)
    assert three_pool[0] < two_pool[0] and three_pool[1] < two_pool[1]
    assert six_pool[0] < three_pool[0] and six_pool[1] < three_pool[1]
    assert limits(budget, 16, 16, 6, True) is None
    assert limits(None, 16, 8, 6, True) == limits(None, 16, 8)


def test_kernel_capabilities_report_spin_major_extensions():
    native = detect_kernel_capabilities()["dynamic"]["native_parallel"]

    assert native.details["spin_major"]
    assert native.details["spin_major_multi_pool"]
    assert native.details["spin_major_spatial_inflow"]
    assert native.details["spin_major_rf_cache_multi_pool"]
    assert native.details["spin_major_rf_cache_phase"]
