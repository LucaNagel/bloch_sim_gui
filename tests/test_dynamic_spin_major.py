import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator import dynamic_phantom as dynamic_module
from blochsimulator.dynamic_phantom import (
    DynamicB0,
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

RASTER = 20e-6
OUTPUTS = (
    "signal",
    "species_signal",
    "final_magnetization",
    "final_pool_magnetization",
    "checkpoint_magnetization",
    "checkpoint_pool_magnetization",
)


def _phantom(shape=(6, 1, 1), *, inflow=None, tracked=False, conversion_start=0.0):
    grid = np.linspace(0.0, 1.0, int(np.prod(shape))).reshape(shape)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.06, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": 1.0 + 0.2 * grid,
            "Lactate": 0.04 + 0.03 * grid,
        },
        kpl_map_s_inv=0.01 + 0.09 * grid,
        b0_map=-5.0 + 10.0 * grid,
        conversion_start_s=conversion_start,
        nucleus="C13",
    )
    if tracked:
        phantom.initial_spin_density_maps = {
            "Pyruvate": 1.0 + 0.1 * grid,
            "Lactate": np.full(shape, 0.1),
        }
        phantom.equilibrium_polarization = 0.02
    rate = TimeCurve((0.0, 2e-4, 5e-4), (0.05, 0.6, 0.2), "linear", "hold")
    polarization = TimeCurve((0.0, 5e-4), (0.5, 0.3), "linear", "hold")
    if inflow == "direct":
        phantom.pyruvate_inflow = PyruvateInflow(rate, 0.8 + 0.4 * grid)
    elif inflow == "polarized":
        phantom.pyruvate_inflow = PyruvateInflow(rate, 0.8 + 0.4 * grid, polarization)
    return phantom


def _program(n_rf=10, n_adc=9):
    return SequenceProgram(
        (
            RFEvent(0.0, np.linspace(40.0 + 6.0j, 85.0 - 12.0j, n_rf), RASTER),
            GradientEvent("x", 0.0, np.linspace(-110.0, 150.0, 20), RASTER),
            RFEvent(14 * RASTER, np.full(3, 60.0 - 20.0j), RASTER),
            ADCEvent(3 * RASTER, n_adc, 2 * RASTER, phase_offset_rad=0.29),
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


@pytest.mark.parametrize("kernel", ["native_serial", "native_parallel"])
@pytest.mark.parametrize(
    "inflow, tracked, conversion_start",
    [
        (None, False, 0.0),
        (None, False, 9 * RASTER),
        (None, True, 0.0),
        ("direct", False, 0.0),
        ("polarized", False, 0.0),
        ("polarized", True, 0.0),
        ("polarized", True, 9 * RASTER),
    ],
)
def test_spin_major_matches_reference_for_two_pool_models(
    kernel, inflow, tracked, conversion_start
):
    phantom = _phantom(
        inflow=inflow, tracked=tracked, conversion_start=conversion_start
    )
    program = _program()

    reference = _run(phantom, program, "optimized")
    voxel_major = _run(phantom, program, kernel, _native_spin_major=False)
    spin_major = _run(phantom, program, kernel)

    assert spin_major.metadata["native_spin_major_enabled"] is True
    assert spin_major.metadata["native_spin_major_fallback_reason"] is None
    assert spin_major.metadata["native_full_plan_enabled"] is True
    assert spin_major.metadata["native_spin_major_intervals"] == (
        spin_major.metadata["compiled_interval_count"]
    )
    assert spin_major.metadata["numerical_contract"] == "float64-close"
    assert voxel_major.metadata["native_spin_major_enabled"] is False
    assert voxel_major.metadata["native_spin_major_fallback_reason"] == (
        "disabled by request"
    )
    _assert_close(spin_major, reference)
    _assert_close(spin_major, voxel_major)


def test_spin_major_supports_constant_tx_multi_rx_and_subvoxels():
    phantom = _phantom(inflow="polarized", tracked=True)
    grid = np.linspace(0.0, 1.0, phantom.nvoxels).reshape(phantom.shape)
    phantom.tx_sensitivity_map = np.full(phantom.shape, 0.85 * np.exp(0.3j))
    phantom.rx_sensitivity_maps = np.stack(
        (
            np.ones(phantom.shape, dtype=np.complex128),
            (0.7 + 0.2 * grid) * np.exp(1j * (0.1 + 0.3 * grid)),
        )
    )
    program = _program()
    kwargs = dict(spoiler_mode="gradient", spin_sampling=SpinSampling((2, 2, 1)))

    reference = _run(phantom, program, "optimized", **kwargs)
    spin_major = _run(phantom, program, **kwargs)

    assert spin_major.metadata["native_spin_major_enabled"] is True
    assert spin_major.signal.shape == reference.signal.shape
    assert spin_major.signal.shape[0] == 2
    _assert_close(spin_major, reference)


@pytest.mark.parametrize("kernel", ["native_serial", "native_parallel"])
@pytest.mark.parametrize(
    "inflow,tracked",
    [(None, False), (None, True), ("direct", False), ("polarized", True)],
)
def test_spatial_tx_matches_numpy_with_cache_coils_subspins_and_zero_tx(
    kernel, inflow, tracked
):
    phantom = _phantom(inflow=inflow, tracked=tracked)
    grid = np.linspace(0, 1, phantom.nvoxels).reshape(phantom.shape)
    phantom.tx_sensitivity_map = (2 * grid) * np.exp(1j * (-2 + 4 * grid))
    phantom.rx_sensitivity_maps = np.stack(
        (np.ones(phantom.shape), (0.5 + grid) * np.exp(0.3j * grid))
    )
    program = _repeated_program(ALTERNATING, gradient=True)
    kwargs = dict(
        spoiler_mode="gradient",
        spin_sampling=SpinSampling((2, 2, 1)),
        num_threads=2,
        use_parallel=True,
    )
    reference = _run_cached(phantom, program, "optimized", **kwargs)
    stepwise = _run_cached(
        phantom, program, kernel, _native_spin_major_rf_cache=False, **kwargs
    )
    cached = _run_cached(phantom, program, kernel, **kwargs)
    assert stepwise.metadata["native_spin_major_enabled"]
    assert cached.metadata["native_spin_major_enabled"]
    assert _rf_cache(cached, "enabled")
    _assert_close(stepwise, reference)
    _assert_close(cached, reference)


def test_old_extension_without_spatial_tx_capability_falls_back(monkeypatch):
    from blochsimulator import dynamic_spin_major_cy as extension

    monkeypatch.delattr(extension, "SPATIAL_TX_SUPPORTED")
    phantom = _phantom()
    phantom.tx_sensitivity_map = np.linspace(0, 1, phantom.nvoxels).reshape(
        phantom.shape
    )
    result = _run(phantom, _program())
    assert not result.metadata["native_spin_major_enabled"]
    assert "installed extension" in result.metadata["native_spin_major_fallback_reason"]


def test_stale_spin_major_extension_uses_previous_native_driver(monkeypatch):
    from blochsimulator import dynamic_spin_major_cy as extension

    monkeypatch.setattr(extension, "API_VERSION", 1)
    result = _run(_phantom(), _program())
    assert result.metadata["sequence_kernel"] == "native_serial"
    assert not result.metadata["native_spin_major_enabled"]
    assert (
        result.metadata["native_spin_major_fallback_reason"]
        == "extension is unavailable"
    )


def test_spin_major_result_does_not_depend_on_slices_or_threads(monkeypatch):
    phantom = _phantom(shape=(12, 10, 10), inflow="polarized", tracked=True)
    program = _program()
    kwargs = dict(use_parallel=True)

    serial = _run(phantom, program, "native_serial")
    parallel = _run(phantom, program, "native_parallel", num_threads=4, **kwargs)
    monkeypatch.setattr(dynamic_module, "_SPIN_MAJOR_SLICE_INTERVALS", 4)
    monkeypatch.setattr(dynamic_module, "_SPIN_MAJOR_SLICE_ADC_VALUES", 2)
    sliced = _run(phantom, program, "native_parallel", num_threads=3, **kwargs)

    assert serial.metadata["native_spin_major_threads"] == 1
    assert parallel.metadata["native_spin_major_threads"] == 4
    assert sliced.metadata["native_spin_major_chunks"] > (
        parallel.metadata["native_spin_major_chunks"]
    )
    for name in OUTPUTS:
        np.testing.assert_array_equal(getattr(parallel, name), getattr(serial, name))
        np.testing.assert_array_equal(getattr(sliced, name), getattr(serial, name))


def test_spin_major_parent_chunks_reduce_one_row_per_block():
    phantom = _phantom(inflow="polarized", tracked=True)
    program = _program()
    kwargs = dict(spoiler_mode="gradient", spin_sampling=SpinSampling((2, 1, 1)))

    reference = _run(phantom, program, "optimized", **kwargs)
    whole = _run(phantom, program, **kwargs)
    chunked = _run(phantom, program, parent_voxel_chunk_size=2, **kwargs)

    assert chunked.metadata["dynamic_parent_chunking"] is True
    assert chunked.metadata["dynamic_parent_chunk_count"] == 3
    assert chunked.metadata["native_spin_major_parent_blocks"] == 3
    assert chunked.metadata["dynamic_reduction_group_parent_voxels"] == 2
    assert chunked.metadata["dynamic_signal_reduction"] == (
        "fixed-block-streaming-pairwise"
    )
    _assert_close(chunked, reference)
    _assert_close(chunked, whole)


def test_spin_major_memory_plan_counts_adc_rows_per_block():
    # Receive channels create ADC-buffer pressure without a long propagation
    # whose accumulated libm/NumPy rounding can dominate this memory-plan test.
    n_adc = 1000
    n_rx_coils = 40
    program = SequenceProgram(
        (
            RFEvent(0.0, np.full(3, 60.0 - 20.0j), RASTER),
            ADCEvent(4 * RASTER, n_adc, RASTER),
        ),
        duration_s=(n_adc + 6) * RASTER,
    )
    phantom = _phantom(inflow="polarized", tracked=True)
    coil_indices = np.arange(n_rx_coils)
    sensitivities = (0.8 + 0.01 * coil_indices) * np.exp(0.03j * coil_indices)
    phantom.rx_sensitivity_maps = np.broadcast_to(
        sensitivities[:, None, None, None], (n_rx_coils, *phantom.shape)
    ).copy()
    kwargs = dict(
        spoiler_mode="gradient",
        spin_sampling=SpinSampling((2, 1, 1)),
        memory_budget_bytes=8 * 1024**2,
    )

    reference = _run(phantom, program, "optimized", **kwargs)
    spin_major = _run(phantom, program, **kwargs)
    voxel_major = _run(phantom, program, _native_spin_major=False, **kwargs)
    explicit = _run(phantom, program, parent_voxel_chunk_size=3, **kwargs)

    adc_row_bytes = phantom.n_species * n_rx_coils * n_adc * 16
    # One ADC row per parent voxel forces the voxel-major path into blocks.
    assert voxel_major.metadata["dynamic_parent_chunk_count"] == 2
    assert voxel_major.metadata["dynamic_block_work_bytes_estimate"] == 0
    assert spin_major.metadata["native_spin_major_enabled"] is True
    assert not spin_major.metadata.get("dynamic_parent_chunking", False)
    assert explicit.metadata["dynamic_block_work_bytes_estimate"] == adc_row_bytes
    assert explicit.metadata["dynamic_parent_work_bytes_estimate"] == (
        voxel_major.metadata["dynamic_parent_work_bytes_estimate"] - adc_row_bytes
    )
    _assert_close(spin_major, voxel_major)
    _assert_close(explicit, voxel_major)
    for result in (spin_major, voxel_major, explicit):
        _assert_close(result, reference)
    _assert_close(explicit, spin_major)


def test_spin_major_switch_follows_environment(monkeypatch):
    phantom = _phantom()
    program = _program()

    monkeypatch.setenv("BLOCHSIM_DYNAMIC_SPIN_MAJOR", "off")
    disabled = _run(phantom, program)
    enabled = _run(phantom, program, _native_spin_major=True)

    assert disabled.metadata["native_spin_major_enabled"] is False
    assert disabled.metadata["native_spin_major_fallback_reason"] == (
        "disabled by request"
    )
    assert disabled.metadata["native_full_plan_enabled"] is True
    assert enabled.metadata["native_spin_major_enabled"] is True


def _three_pool_phantom():
    shape = (3, 1, 1)
    return DynamicSpectralPhantom(
        shape=shape,
        fov=(0.03, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
            ChemicalSpecies("Alanine", 6.0, 20.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.full(shape, 0.05),
            "Alanine": np.full(shape, 0.02),
        },
        kpl_map_s_inv=np.full(shape, 0.05),
        nucleus="C13",
    )


@pytest.mark.parametrize("case", ["spatial_delay", "three_pools"])
def test_spin_major_runs_models_it_used_to_refuse(case):
    if case == "spatial_delay":
        phantom = _phantom(inflow="polarized", tracked=True)
        inflow = phantom.pyruvate_inflow
        phantom.pyruvate_inflow = PyruvateInflow(
            inflow.rate_curve_s_inv,
            inflow.delivery_map,
            inflow.polarization_curve,
            arrival_delay_map_s=np.linspace(0.0, 4 * RASTER, phantom.nvoxels).reshape(
                phantom.shape
            ),
        )
    else:
        phantom = _three_pool_phantom()
    program = _program()

    spin_major = _run(phantom, program)
    interval_major = _run(phantom, program, _native_spin_major=False)

    assert spin_major.metadata["native_spin_major_enabled"] is True
    assert spin_major.metadata["native_spin_major_chunks"] > 0
    assert spin_major.metadata["native_spin_major_fallback_reason"] is None
    assert interval_major.metadata["native_spin_major_enabled"] is False
    _assert_close(spin_major, interval_major)


@pytest.mark.parametrize(
    "case, reason",
    [
        ("dynamic_b0", "dynamic B0 is unsupported"),
        ("float32_checkpoints", "requires float64 state and checkpoints"),
        ("tiny_budget", "memory budget is too small"),
    ],
)
def test_spin_major_falls_back_for_unsupported_models(case, reason):
    phantom = _phantom(inflow="polarized", tracked=True)
    program = _program()
    kwargs = {}
    if case == "dynamic_b0":
        phantom.dynamic_b0 = DynamicB0(
            TimeCurve((0.0, 25 * RASTER), (-3.0, 4.0), "linear", "hold"),
            np.ones(phantom.shape),
        )
    elif case == "spatial_tx":
        phantom.tx_sensitivity_map = np.linspace(0.8, 1.1, phantom.nvoxels).reshape(
            phantom.shape
        ) * np.exp(0.2j)
    elif case == "float32_checkpoints":
        kwargs["checkpoint_dtype"] = np.float32
    else:
        kwargs["memory_budget_bytes"] = 1

    result = _run(phantom, program, **kwargs)

    assert result.metadata["native_spin_major_enabled"] is False
    assert result.metadata["native_spin_major_chunks"] == 0
    assert result.metadata["native_spin_major_fallback_reason"] == reason


def test_optimized_kernel_does_not_report_spin_major_fallback():
    result = _run(_phantom(), _program(), "optimized")

    assert result.metadata["native_spin_major_enabled"] is False
    assert result.metadata["native_spin_major_fallback_reason"] is None


# A dyadic raster keeps the plan times of repeated pulses exact, so their RF
# segments have bitwise equal propagator keys.
CACHE_RASTER = 2.0**-16
CACHE_PULSE = np.array([25.0 + 8.0j, 60.0 - 6.0j, 85.0, 60.0 + 6.0j, 25.0 - 8.0j])
CACHE_PERIOD = 10
ALTERNATING = tuple((-1.0) ** repetition for repetition in range(48))


def _repeated_program(signs, *, gradient=False):
    events = []
    for repetition, sign in enumerate(signs):
        start = repetition * CACHE_PERIOD * CACHE_RASTER
        events.append(RFEvent(start, sign * CACHE_PULSE, CACHE_RASTER))
        if gradient:
            events.append(
                GradientEvent("x", start, np.full(CACHE_PULSE.size, 90.0), CACHE_RASTER)
            )
        events.append(
            ADCEvent(
                start + (CACHE_PULSE.size + 1) * CACHE_RASTER,
                2,
                CACHE_RASTER,
                phase_offset_rad=0.1 * repetition,
            )
        )
    return SequenceProgram(
        tuple(events), duration_s=len(signs) * CACHE_PERIOD * CACHE_RASTER
    )


def _run_cached(phantom, program, kernel="native_serial", **kwargs):
    kwargs.setdefault("simulation_timestep_s", CACHE_RASTER)
    kwargs.setdefault("checkpoints_s", (7 * CACHE_RASTER, program.duration_s))
    return _run(phantom, program, kernel, **kwargs)


def _rf_cache(result, name):
    return result.metadata[f"native_spin_major_rf_propagator_{name}"]


def _assert_equal(actual, expected):
    for name in OUTPUTS:
        np.testing.assert_array_equal(
            getattr(actual, name), getattr(expected, name), err_msg=name
        )


@pytest.mark.parametrize("kernel", ["native_serial", "native_parallel"])
@pytest.mark.parametrize(
    "inflow, tracked, conversion_start",
    [
        (None, False, 0.0),
        (None, False, 9 * CACHE_RASTER),
        (None, True, 0.0),
        ("direct", False, 0.0),
        ("polarized", False, 0.0),
        ("polarized", True, 0.0),
        ("polarized", True, 9 * CACHE_RASTER),
    ],
)
def test_rf_cache_matches_stepwise_rf_for_two_pool_models(
    kernel, inflow, tracked, conversion_start
):
    phantom = _phantom(
        inflow=inflow, tracked=tracked, conversion_start=conversion_start
    )
    program = _repeated_program(ALTERNATING)

    reference = _run_cached(phantom, program, "optimized")
    stepwise = _run_cached(phantom, program, kernel, _native_spin_major_rf_cache=False)
    cached = _run_cached(phantom, program, kernel)

    assert cached.metadata["native_spin_major_enabled"] is True
    assert _rf_cache(cached, "enabled") is True
    assert _rf_cache(cached, "fallback_reason") is None
    assert _rf_cache(cached, "keys") == 2
    assert _rf_cache(cached, "builds") == 2
    assert _rf_cache(cached, "bytes") > 0
    assert _rf_cache(cached, "applications") > len(ALTERNATING) // 2
    assert _rf_cache(cached, "applications") + _rf_cache(
        cached, "stepwise_segments"
    ) == len(ALTERNATING)
    assert _rf_cache(stepwise, "enabled") is False
    assert _rf_cache(stepwise, "fallback_reason") == "disabled by request"
    assert _rf_cache(stepwise, "stepwise_segments") == len(ALTERNATING)
    _assert_close(cached, reference)
    _assert_close(cached, stepwise)


def test_rf_cache_runs_occurrences_with_source_knots_stepwise():
    phantom = _phantom()
    grid = np.linspace(0.8, 1.2, phantom.nvoxels).reshape(phantom.shape)
    # The knot lies on an interval boundary inside the fourth pulse, so that
    # occurrence keeps its key but its source is not quadratic.
    knot = (3 * CACHE_PERIOD + 2) * CACHE_RASTER
    phantom.pyruvate_inflow = PyruvateInflow(
        TimeCurve((0.0, knot), (0.1, 0.7), "linear", "hold"), grid
    )
    program = _repeated_program(ALTERNATING)

    reference = _run_cached(phantom, program, "optimized")
    stepwise = _run_cached(phantom, program, _native_spin_major_rf_cache=False)
    cached = _run_cached(phantom, program)
    without_inflow = _run_cached(_phantom(), program)

    assert cached.metadata["compiled_interval_count"] == (
        without_inflow.metadata["compiled_interval_count"]
    )
    assert _rf_cache(cached, "keys") == 2
    assert _rf_cache(cached, "stepwise_segments") == 1
    assert _rf_cache(cached, "applications") == len(ALTERNATING) - 1
    _assert_close(cached, reference)
    _assert_close(cached, stepwise)


def test_rf_cache_keys_conversion_state_of_each_occurrence():
    signs = ALTERNATING * 2
    # Conversion starts inside one pulse, between two cached halves of the run.
    conversion_start = (len(ALTERNATING) * CACHE_PERIOD + 2) * CACHE_RASTER
    phantom = _phantom(
        inflow="polarized", tracked=True, conversion_start=conversion_start
    )
    program = _repeated_program(signs)

    reference = _run_cached(phantom, program, "optimized")
    stepwise = _run_cached(phantom, program, _native_spin_major_rf_cache=False)
    cached = _run_cached(phantom, program)

    assert _rf_cache(cached, "keys") == 4
    assert _rf_cache(cached, "builds") == 4
    assert _rf_cache(cached, "stepwise_segments") >= 1
    assert _rf_cache(cached, "applications") + _rf_cache(
        cached, "stepwise_segments"
    ) == len(signs)
    _assert_close(cached, reference)
    _assert_close(cached, stepwise)


def test_rf_cache_supports_slice_selective_pulses_constant_tx_multi_rx_and_subvoxels():
    phantom = _phantom(inflow="polarized", tracked=True)
    grid = np.linspace(0.0, 1.0, phantom.nvoxels).reshape(phantom.shape)
    phantom.tx_sensitivity_map = np.full(phantom.shape, 0.85 * np.exp(0.3j))
    phantom.rx_sensitivity_maps = np.stack(
        (
            np.ones(phantom.shape, dtype=np.complex128),
            (0.7 + 0.2 * grid) * np.exp(1j * (0.1 + 0.3 * grid)),
        )
    )
    program = _repeated_program(ALTERNATING, gradient=True)
    kwargs = dict(spoiler_mode="gradient", spin_sampling=SpinSampling((2, 2, 1)))

    reference = _run_cached(phantom, program, "optimized", **kwargs)
    stepwise = _run_cached(
        phantom, program, _native_spin_major_rf_cache=False, **kwargs
    )
    cached = _run_cached(phantom, program, **kwargs)

    assert _rf_cache(cached, "enabled") is True
    assert _rf_cache(cached, "keys") == 2
    assert cached.signal.shape[0] == 2
    _assert_close(cached, reference)
    _assert_close(cached, stepwise)


@pytest.mark.parametrize("spatial_tx", [False, True])
def test_rf_cache_result_does_not_depend_on_slices_threads_or_capacity(
    monkeypatch, spatial_tx
):
    phantom = _phantom(shape=(12, 10, 10), inflow="polarized", tracked=True)
    if spatial_tx:
        grid = np.linspace(0, 1, phantom.nvoxels).reshape(phantom.shape)
        phantom.tx_sensitivity_map = (0.5 + grid) * np.exp(1j * (-1 + 2 * grid))
    program = _repeated_program((1.0,) * 20 + (-1.0,) * 20 + (1.0,) * 20)
    kwargs = dict(use_parallel=True)

    serial = _run_cached(phantom, program, "native_serial")
    parallel = _run_cached(phantom, program, "native_parallel", num_threads=4, **kwargs)
    # Slices shorter than one pulse end at cached segment boundaries.
    monkeypatch.setattr(dynamic_module, "_SPIN_MAJOR_SLICE_INTERVALS", 4)
    monkeypatch.setattr(dynamic_module, "_SPIN_MAJOR_SLICE_ADC_VALUES", 2)
    sliced = _run_cached(phantom, program, "native_parallel", num_threads=3, **kwargs)
    monkeypatch.setattr(dynamic_module, "_SPIN_MAJOR_RF_PROPAGATOR_ENTRIES", 1)
    evicting = _run_cached(phantom, program, "native_parallel", num_threads=3, **kwargs)

    assert _rf_cache(serial, "keys") == 2
    assert _rf_cache(serial, "builds") == 2
    assert _rf_cache(serial, "bytes") == 2 * _rf_cache(evicting, "bytes")
    assert _rf_cache(evicting, "builds") == 3
    assert sliced.metadata["native_spin_major_chunks"] > (
        parallel.metadata["native_spin_major_chunks"]
    )
    for result in (parallel, sliced, evicting):
        assert _rf_cache(result, "enabled") is True
        _assert_equal(result, serial)


@pytest.mark.parametrize(
    "case, reason",
    [
        ("single", "no repeated RF segments"),
        ("rare", "RF segments repeat too rarely for the cache capacity"),
        ("tiny_budget", "memory budget is too small"),
    ],
)
def test_rf_cache_falls_back_to_stepwise_rf(monkeypatch, case, reason):
    phantom = _phantom(inflow="polarized", tracked=True)
    program = _program() if case == "single" else _repeated_program(ALTERNATING)
    kwargs = {}
    if case == "rare":
        # Alternating keys with one entry rebuild at every occurrence.
        monkeypatch.setattr(dynamic_module, "_SPIN_MAJOR_RF_PROPAGATOR_ENTRIES", 1)
    elif case == "tiny_budget":
        kwargs["_spin_major_rf_propagator_budget_bytes"] = 1
    run = _run if case == "single" else _run_cached

    stepwise = run(phantom, program, _native_spin_major_rf_cache=False)
    result = run(phantom, program, **kwargs)

    assert result.metadata["native_spin_major_enabled"] is True
    assert _rf_cache(result, "enabled") is False
    assert _rf_cache(result, "fallback_reason") == reason
    assert _rf_cache(result, "builds") == 0
    assert _rf_cache(result, "applications") == 0
    assert _rf_cache(result, "bytes") == 0
    assert _rf_cache(result, "stepwise_segments") == (
        _rf_cache(stepwise, "stepwise_segments")
    )
    _assert_equal(result, stepwise)


def test_rf_cache_switch_follows_environment_and_keyword(monkeypatch):
    phantom = _phantom(inflow="polarized", tracked=True)
    program = _repeated_program(ALTERNATING)

    default = _run_cached(phantom, program)
    disabled = _run_cached(phantom, program, _native_spin_major_rf_cache=False)
    monkeypatch.setenv("BLOCHSIM_DYNAMIC_SPIN_MAJOR_RF_CACHE", "0")
    environment = _run_cached(phantom, program)
    enabled = _run_cached(phantom, program, _native_spin_major_rf_cache=True)

    assert _rf_cache(default, "enabled") is True
    assert _rf_cache(enabled, "enabled") is True
    for result in (disabled, environment):
        assert _rf_cache(result, "enabled") is False
        assert _rf_cache(result, "fallback_reason") == "disabled by request"
    _assert_equal(environment, disabled)
    _assert_equal(enabled, default)
