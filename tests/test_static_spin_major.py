"""The private static spin-major adapter (kernel §5d C5).

``_static_spin_major=True`` runs ``simulate_sequence`` and
``simulate_spectral_sequence`` on the dynamic spin-major executor. It is a
float64-close path: the dynamic kernel splits free precession around the RF
rotation, so the gap to the static kernel is a time-step error that shrinks
with the step, while the default static path stays bit-identical.
"""

import dataclasses
import warnings

import numpy as np
import pytest

from blochsimulator import BlochSimulator, KernelFallbackWarning
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom
from blochsimulator.phantom import Phantom
from blochsimulator.sequence.model import (
    ADCEvent,
    GradientEvent,
    RFEvent,
    SequenceProgram,
)
from blochsimulator.sequence.spin_sampling import SpinSampling
from blochsimulator.spectral_phantom import ChemicalSpecies, SpectralPhantom
from blochsimulator.static_spin_major import (
    StaticSpinMajorFallback,
    run_static_components,
)

pytest.importorskip("blochsimulator.dynamic_spin_major_cy")

SHAPE = (1, 1, 5)
FOV_M = (0.01, 0.01, 0.025)
STEP_S = 20e-6
CHECKPOINTS_S = (2e-3, 5e-3)
# Static vs spin-major at a 20 us step with a 1 us RF raster, measured on an
# M3: at most 5.8e-4 (signal) and 7.6e-4 (magnetization) relative.
TOLERANCE = 2e-3


def _train(repetitions=2, repetition_s=4e-3, definitions=None):
    """Return phase-cycled 1 ms sinc excitations under a slice gradient.

    The RF raster (1 us) is finer than every tested step, so the compile
    subdivides the pulse and the step error is visible.
    """
    lobes = np.linspace(-2.0, 2.0, 1000)
    pulse_hz = 30.0 / 360.0 / 1e-3 / np.mean(np.sinc(lobes)) * np.sinc(lobes)
    events = []
    for index in range(repetitions):
        start = index * repetition_s
        phase = np.pi * index
        events.append(RFEvent(start, pulse_hz * np.exp(1j * phase), 1e-6))
        events.append(GradientEvent("z", start, np.full(100, 1.0e5), 10e-6))
        events.append(GradientEvent("z", start + 1.0e-3, np.full(50, -1.0e5), 10e-6))
        events.append(ADCEvent(start + 1.6e-3, 32, 50e-6, phase_offset_rad=phase))
    metadata = {} if definitions is None else {"definitions": dict(definitions)}
    return SequenceProgram(tuple(events), repetitions * repetition_s, metadata=metadata)


def _grid():
    return np.linspace(0.0, 1.0, int(np.prod(SHAPE))).reshape(SHAPE)


def _phantom(**overrides):
    grid = _grid()
    values = dict(
        shape=SHAPE,
        fov=FOV_M,
        t1_map=1.0 + grid,
        t2_map=0.05 + 0.03 * (grid > 0.5),
        pd_map=0.5 + grid,
        b0_map=-100.0 + 200.0 * grid,
    )
    values.update(overrides)
    return Phantom(**values)


def _simulator():
    return BlochSimulator(use_parallel=False, num_threads=1)


def _pair(phantom, program=None, **kwargs):
    """Return (static, spin-major) results without any fallback."""
    program = _train() if program is None else program
    kwargs.setdefault("simulation_timestep_s", STEP_S)
    kwargs.setdefault("checkpoints_s", CHECKPOINTS_S)
    simulator = _simulator()
    static = simulator.simulate_sequence(program, phantom, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("error", KernelFallbackWarning)
        spin_major = simulator.simulate_sequence(
            program, phantom, _static_spin_major=True, **kwargs
        )
    return static, spin_major


def _relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    scale = max(float(np.max(np.abs(a))), 1e-300)
    return float(np.max(np.abs(a - b))) / scale


def _assert_close(static, spin_major, tolerance=TOLERANCE):
    assert spin_major.signal.shape == static.signal.shape
    assert spin_major.final_magnetization.shape == static.final_magnetization.shape
    assert _relative(static.signal, spin_major.signal) < tolerance
    assert (
        _relative(static.final_magnetization, spin_major.final_magnetization)
        < tolerance
    )
    if static.checkpoint_magnetization is not None:
        assert (
            spin_major.checkpoint_magnetization.dtype
            == static.checkpoint_magnetization.dtype
        )
        assert (
            _relative(
                static.checkpoint_magnetization.astype(np.float64),
                spin_major.checkpoint_magnetization.astype(np.float64),
            )
            < tolerance
        )


def test_default_path_is_unchanged():
    result = _simulator().simulate_sequence(
        _train(), _phantom(), simulation_timestep_s=STEP_S
    )
    assert "static_spin_major" not in result.metadata
    assert "progress_unit" not in result.metadata


def test_single_object_is_close_and_reports_the_path():
    static, spin_major = _pair(_phantom())
    _assert_close(static, spin_major)
    metadata = spin_major.metadata
    assert metadata["static_spin_major"] is True
    assert metadata["native_sequence_kernel"] == "spin_major"
    assert metadata["numerical_contract"] == "float64-close"
    assert metadata["progress_unit"] == "intervals"
    assert metadata["static_spin_major_passes"] == 1
    assert metadata["native_spin_major_enabled"]
    assert metadata["static_spin_major_intervals"] > 0
    # Five distinct T1/T2 rows: one pool constant and four groups.
    assert metadata["static_spin_major_relaxation_groups"] == [4]


def test_gap_to_the_static_kernel_shrinks_with_the_step():
    program = _train(repetitions=1)
    errors = []
    for step_s in (20e-6, 5e-6, 1e-6):
        static, spin_major = _pair(
            _phantom(),
            program=program,
            simulation_timestep_s=step_s,
            checkpoints_s=(2e-3,),
        )
        errors.append(
            max(
                _relative(static.signal, spin_major.signal),
                _relative(static.final_magnetization, spin_major.final_magnetization),
            )
        )
    assert errors[0] > 4.0 * errors[1] > 16.0 * errors[2]
    assert errors[2] < 2e-5


def test_matches_the_dynamic_run_of_the_same_object():
    grid = _grid()
    b0_hz = -100.0 + 200.0 * grid
    species = ChemicalSpecies("A", 0.0, 2.0, 0.08)
    ones, zeros = np.ones(SHAPE), np.zeros(SHAPE)
    dynamic = DynamicSpectralPhantom(
        shape=SHAPE,
        fov=FOV_M,
        pools=(species, ChemicalSpecies("B", 0.0, 2.0, 0.08)),
        initial_concentration_maps={"A": ones, "B": zeros},
        initial_spin_density_maps={"A": ones, "B": zeros},
        equilibrium_polarization=1.0,
        kpl_map_s_inv=zeros,
        b0_map=b0_hz,
        field_strength=3.0,
        nucleus="C13",
    )
    static = Phantom(
        shape=SHAPE,
        fov=FOV_M,
        t1_map=np.full(SHAPE, 2.0),
        t2_map=np.full(SHAPE, 0.08),
        b0_map=b0_hz,
    )
    simulator = _simulator()
    program = _train()
    reference = simulator.simulate_dynamic_sequence(
        program,
        dynamic,
        simulation_timestep_s=STEP_S,
        checkpoints_s=CHECKPOINTS_S,
        sequence_kernel="native_serial",
    )
    adapted = simulator.simulate_sequence(
        program,
        static,
        simulation_timestep_s=STEP_S,
        checkpoints_s=CHECKPOINTS_S,
        _static_spin_major=True,
    )
    # Same executor and splitting: only the summation order differs.
    assert _relative(reference.signal, adapted.signal) < 1e-12
    assert (
        _relative(
            reference.final_pool_magnetization[0].reshape(-1, 3),
            adapted.final_magnetization,
        )
        < 1e-12
    )


def test_spectral_species_become_pools_and_split_into_passes():
    grid = _grid()
    names = [f"S{index}" for index in range(8)]
    species = [
        ChemicalSpecies(name, 0.0, 20.0 + index, 0.04 + 0.002 * index)
        for index, name in enumerate(names)
    ]
    # Distinct constant shifts per species, one species absent in two voxels.
    species = [
        dataclasses.replace(item, frequency_offset_hz=15.0 * index)
        for index, item in enumerate(species)
    ]
    maps = {name: 1.0 + 0.1 * index + grid for index, name in enumerate(names)}
    maps["S3"] = maps["S3"] * (grid > 0.3)
    phantom = SpectralPhantom(
        shape=SHAPE,
        fov=FOV_M,
        species=species,
        concentration_maps=maps,
        b0_map=-50.0 + 100.0 * grid,
        field_strength=3.0,
        nucleus="C13",
    )
    simulator = _simulator()
    options = dict(
        field_strength_t=3.0,
        nucleus="C13",
        simulation_timestep_s=STEP_S,
        checkpoints_s=CHECKPOINTS_S,
    )
    static = simulator.simulate_spectral_sequence(_train(), phantom, **options)
    with warnings.catch_warnings():
        warnings.simplefilter("error", KernelFallbackWarning)
        spin_major = simulator.simulate_spectral_sequence(
            _train(), phantom, _static_spin_major=True, **options
        )
    _assert_close(static, spin_major)
    assert _relative(static.species_signal, spin_major.species_signal) < TOLERANCE
    assert (
        _relative(static.final_pool_magnetization, spin_major.final_pool_magnetization)
        < TOLERANCE
    )
    assert spin_major.metadata["static_spin_major_passes"] == 2
    assert "static_spin_major" not in static.metadata


def test_relaxation_groups_beyond_the_limit_fall_back_with_a_reason():
    shape = (300,)
    grid = np.linspace(0.0, 1.0, shape[0])
    phantom = Phantom(
        shape=shape,
        fov=(0.03,),
        t1_map=1.0 + grid,
        t2_map=np.full(shape, 0.05),
    )
    program = SequenceProgram(
        (
            RFEvent(0.0, np.full(10, 500.0), 10e-6),
            ADCEvent(2e-4, 4, 20e-6),
        ),
        5e-4,
    )
    with pytest.warns(KernelFallbackWarning, match="relaxation groups") as caught:
        result = _simulator().simulate_sequence(
            program, phantom, simulation_timestep_s=STEP_S, _static_spin_major=True
        )
    assert caught[0].filename == __file__
    assert result.metadata["static_spin_major"] is False
    assert "relaxation groups" in result.metadata["static_spin_major_fallback_reason"]
    reference = _simulator().simulate_sequence(
        program, phantom, simulation_timestep_s=STEP_S
    )
    np.testing.assert_array_equal(result.signal, reference.signal)


@pytest.mark.parametrize(
    "options, match",
    [
        ({"sequence_kernel": "strict_reference"}, "strict reference"),
        ({"equilibrium_mz": -0.5}, "negative equilibrium"),
    ],
)
def test_fallbacks_warn_at_the_call_site(options, match):
    with pytest.warns(KernelFallbackWarning, match=match) as caught:
        result = _simulator().simulate_sequence(
            _train(repetitions=1),
            _phantom(),
            simulation_timestep_s=STEP_S,
            _static_spin_major=True,
            **options,
        )
    assert caught[0].filename == __file__
    assert result.metadata["static_spin_major"] is False
    assert result.metadata["static_spin_major_fallback_reason"]


def test_memory_budget_below_one_spin_tile_falls_back():
    with pytest.raises(StaticSpinMajorFallback, match="memory budget"):
        run_static_components(
            _train(repetitions=1),
            [("static", _phantom())],
            use_parallel=False,
            memory_budget_bytes=1,
            simulation_timestep_s=STEP_S,
        )


def test_equilibrium_scalar_and_map():
    grid = _grid()
    for equilibrium in (0.4, 0.2 + 0.6 * grid):
        static, spin_major = _pair(_phantom(), equilibrium_mz=equilibrium)
        _assert_close(static, spin_major)


def test_coils_tx_map_and_transverse_initial_state():
    grid = _grid()
    coils = np.stack([np.ones(SHAPE), 0.5 + 0.5j * grid, 1.0 - 0.3 * grid]).astype(
        np.complex128
    )
    m0 = np.zeros(SHAPE + (3,))
    m0[..., 0] = 0.3 * grid
    m0[..., 2] = 0.9
    phantom = _phantom(
        tx_sensitivity_map=(0.8 + 0.4 * grid) * np.exp(0.2j * grid),
        rx_sensitivity_maps=coils,
        m0_map=m0,
    )
    static, spin_major = _pair(phantom)
    assert static.signal.shape[0] == 3
    _assert_close(static, spin_major)


def test_inactive_voxels_and_voxel_volume_weighting():
    grid = _grid()
    phantom = _phantom(pd_map=np.where(grid > 0.4, 0.5 + grid, 0.0))
    static, spin_major = _pair(phantom, signal_weighting="voxel_volume")
    _assert_close(static, spin_major)


def test_subvoxel_spins_with_gradient_spoiling():
    for method in ("midpoint", "stratified"):
        static, spin_major = _pair(
            _phantom(),
            spoiler_mode="gradient",
            spin_sampling=SpinSampling((1, 1, 3), method=method),
        )
        _assert_close(static, spin_major)


def test_ideal_crushers_and_small_checkpoint_dtypes():
    program = _train(definitions={"IdealSpoilerEndTimes": (3.5e-3,)})
    for dtype in (np.float32, np.float16):
        static, spin_major = _pair(_phantom(), program=program, checkpoint_dtype=dtype)
        assert spin_major.checkpoint_magnetization.dtype == np.dtype(dtype)
        tolerance = 1e-2 if dtype == np.float16 else TOLERANCE
        _assert_close(static, spin_major, tolerance=tolerance)


def test_adc_at_time_zero():
    program = SequenceProgram(
        (
            ADCEvent(0.0, 4, 20e-6),
            RFEvent(1e-4, np.full(10, 500.0), 10e-6),
            ADCEvent(3e-4, 4, 20e-6),
        ),
        6e-4,
    )
    m0 = np.zeros(SHAPE + (3,))
    m0[..., 0] = 1.0
    static, spin_major = _pair(_phantom(m0_map=m0), program=program, checkpoints_s=())
    _assert_close(static, spin_major)


def test_progress_preview_and_cancel():
    progress, previews = [], []
    simulator = _simulator()
    result = simulator.simulate_sequence(
        _train(),
        _phantom(),
        simulation_timestep_s=STEP_S,
        _static_spin_major=True,
        progress_callback=lambda done, total: progress.append((done, total)),
        preview_callback=lambda fraction, signal: previews.append(
            (fraction, np.shape(signal))
        ),
    )
    total = result.metadata["static_spin_major_intervals"]
    assert progress[-1] == (total, total)
    assert all(later[0] >= earlier[0] for earlier, later in zip(progress, progress[1:]))
    assert previews and previews[-1][0] == pytest.approx(1.0)
    assert previews[-1][1] == result.signal.shape
    with pytest.raises(RuntimeError, match="cancelled"):
        simulator.simulate_sequence(
            _train(),
            _phantom(),
            simulation_timestep_s=STEP_S,
            _static_spin_major=True,
            cancel_callback=lambda: True,
        )


def _probes(n_spins=4):
    from blochsimulator.sequence.convergence import SpinProbeEnsemble

    rng = np.random.default_rng(1)
    return SpinProbeEnsemble(
        positions_m=np.column_stack(
            [np.zeros(n_spins), np.zeros(n_spins), rng.uniform(-4e-3, 4e-3, n_spins)]
        ),
        frequency_offsets_hz=rng.uniform(-200.0, 200.0, n_spins),
        b1_scales=np.ones(n_spins),
        t1_s=np.full(n_spins, 3.0),
        t2_s=np.full(n_spins, 0.08),
        initial_magnetization=(0.0, 0.0, 1.0),
    )


def test_second_order_against_the_gold_reference():
    from blochsimulator.sequence.convergence import run_timestep_convergence

    steps = (20e-6, 10e-6, 5e-6)
    errors = {}
    for spin_major in (False, True):
        result = run_timestep_convergence(
            _train(repetitions=1),
            _probes(),
            timesteps_s=steps,
            simulator=_simulator(),
            _static_spin_major=spin_major,
        )
        errors[spin_major] = [point.max_vector_error for point in result.points]
    orders = np.log(np.divide(errors[True][:-1], errors[True][1:])) / np.log(2.0)
    assert np.all(orders > 1.8)
    # Symmetric splitting halves the static kernel's error on this train.
    assert errors[True][0] < 0.75 * errors[False][0]


def test_timestep_estimate_measures_the_kernel_that_runs():
    from blochsimulator.sequence.convergence import estimate_timestep_error

    class Recording(BlochSimulator):
        paths = []

        def simulate_sequence(self, *args, **kwargs):
            result = super().simulate_sequence(*args, **kwargs)
            self.paths.append(result.metadata.get("static_spin_major", False))
            return result

    simulator = Recording(use_parallel=False, num_threads=1)
    estimates = {}
    for spin_major in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("error", KernelFallbackWarning)
            estimates[spin_major] = estimate_timestep_error(
                _train(repetitions=1),
                STEP_S,
                positions_m=((0.0, 0.0, -4e-3), (0.0, 0.0, 4e-3)),
                frequency_offsets_hz=(-100.0, 100.0),
                line_samples=3,
                simulator=simulator,
                _static_spin_major=spin_major,
            )
    assert Recording.paths == [False, False, True, True]
    assert estimates[True].max_vector_error != estimates[False].max_vector_error
    assert 0.0 < estimates[True].max_vector_error < 1e-2


_BUILDER_FOV_M = (0.08, 0.08)


def _builders():
    from blochsimulator.sequence import pulseq_builders as builders

    return {
        "flash": lambda: builders.make_pulseq_flash(
            fov_m=_BUILDER_FOV_M, matrix=(4, 4)
        ),
        "epi": lambda: builders.make_pulseq_epi(fov_m=_BUILDER_FOV_M, matrix=(4, 4)),
        "csi": lambda: builders.make_pulseq_csi(
            fov_m=_BUILDER_FOV_M,
            matrix=(2, 2),
            spectral_points=8,
            spectral_bandwidth_hz=2000.0,
            repetition_time_s=30e-3,
        ),
        "epsi": lambda: builders.make_pulseq_epsi_mge(
            fov_m=_BUILDER_FOV_M, matrix=(4, 4), echoes=2
        ),
        "spiral": lambda: builders.make_pulseq_spiral(
            fov_m=_BUILDER_FOV_M, matrix=(8, 8)
        ),
    }


def _reconstruct(result):
    if result.cartesian_acquisition_frames is not None:
        return result.cartesian_acquisition_frames.reconstruct(result, 0)
    if result.cartesian_acquisition is not None:
        return result.cartesian_acquisition.reconstruct(result.signal)
    if result.spectroscopic_acquisition is not None:
        return result.spectroscopic_acquisition.reconstruct_spectra(result.signal)
    return result.spiral_acquisition.reconstruct(result, 0)


# Keys whose value names the path; every other key of the static run must
# be unchanged, and the spin-major run may only add keys.
_PATH_KEYS = {"numerical_contract", "native_sequence_kernel"}


@pytest.mark.parametrize("name", ["flash", "epi", "csi", "epsi", "spiral"])
def test_builders_keep_metadata_and_reconstruction(name, tmp_path):
    from blochsimulator.sequence import load_pulseq

    path = tmp_path / f"{name}.seq"
    _builders()[name]().write(str(path), v141_compat=True)
    program = load_pulseq(path)
    shape = (6, 6, 1)
    x = np.linspace(-1.0, 1.0, 6)
    phantom = Phantom(
        shape=shape,
        fov=_BUILDER_FOV_M + (3e-3,),
        t1_map=np.full(shape, 1.0),
        t2_map=np.full(shape, 0.05),
        pd_map=np.add.outer(1.0 + 0.2 * x, 0.1 * x)[..., None],
    )
    static, spin_major = _pair(phantom, program=program, checkpoints_s=())
    assert set(static.metadata) <= set(spin_major.metadata)
    for key, value in static.metadata.items():
        if key not in _PATH_KEYS:
            np.testing.assert_equal(spin_major.metadata[key], value, err_msg=key)
    assert static.metadata["numerical_contract"] == "legacy-fast-math"
    assert spin_major.metadata["numerical_contract"] == "float64-close"
    _assert_close(static, spin_major)
    image, spin_major_image = _reconstruct(static), _reconstruct(spin_major)
    assert spin_major_image.shape == image.shape
    assert spin_major_image.dtype == image.dtype
    assert _relative(image, spin_major_image) < TOLERANCE
