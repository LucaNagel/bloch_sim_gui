"""Gold reference: Hargreaves replica, steady state, pools and kinetics."""

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import (
    DynamicSpectralPhantom,
    PyruvateInflow,
    TimeCurve,
)
from blochsimulator.sequence import GradientEvent, RFEvent, SequenceProgram
from blochsimulator.sequence.reference import (
    ReferencePool,
    simulate_reference_sequence,
)
from blochsimulator.spectral_phantom import ChemicalSpecies

RASTER = 20e-6


def _bloch_c_literal(b1_hz, grad_hz_per_m, dt, t1, t2, df, position, m0):
    """Literal Python transcription of the loop in Hargreaves' bloch.c.

    Matrices are stored column-major in flat lists exactly as in the C code;
    gamma is already folded into the Hz units.
    """

    def calcrotmat(nx, ny, nz):
        phi = np.sqrt(nx * nx + ny * ny + nz * nz)
        if phi == 0.0:
            return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        hp = phi / 2
        cp = np.cos(hp)
        sp = np.sin(hp) / phi
        ar, ai, br, bi = cp, -nz * sp, ny * sp, -nx * sp
        arar, aiai, brbr, bibi = ar * ar, ai * ai, br * br, bi * bi
        arai2, brbi2, arbi2 = 2 * ar * ai, 2 * br * bi, 2 * ar * bi
        aibr2, arbr2, aibi2 = 2 * ai * br, 2 * ar * br, 2 * ai * bi
        return [
            arar - aiai - brbr + bibi,
            -arai2 - brbi2,
            -arbr2 + aibi2,
            arai2 - brbi2,
            arar - aiai + brbr - bibi,
            -aibr2 - arbi2,
            arbr2 + aibi2,
            arbi2 - aibr2,
            arar + aiai - brbr - bibi,
        ]

    def multmatvec(mat, vec):
        return [
            mat[0] * vec[0] + mat[3] * vec[1] + mat[6] * vec[2],
            mat[1] * vec[0] + mat[4] * vec[1] + mat[7] * vec[2],
            mat[2] * vec[0] + mat[5] * vec[1] + mat[8] * vec[2],
        ]

    twopi = 2.0 * np.pi
    m = list(m0)
    for b1, grad in zip(b1_hz, grad_hz_per_m):
        rotz = -(twopi * float(np.dot(grad, position)) + df * twopi) * dt
        rotx = -b1.real * twopi * dt
        roty = +b1.imag * twopi * dt
        m = multmatvec(calcrotmat(rotx, roty, rotz), m)
        e1, e2 = np.exp(-dt / t1), np.exp(-dt / t2)
        m = [e2 * m[0], e2 * m[1], e1 * m[2] + 1 - e1]
    return np.asarray(m)


def _random_program(seed=3, samples=40, raster=10e-6):
    rng = np.random.default_rng(seed)
    rf = rng.normal(0.0, 600.0, samples) + 1j * rng.normal(0.0, 600.0, samples)
    gradients = rng.normal(0.0, 2000.0, (samples, 3))
    events = [RFEvent(0.0, rf, raster)]
    events.extend(
        GradientEvent(axis, 0.0, gradients[:, index], raster)
        for index, axis in enumerate("xyz")
    )
    return SequenceProgram(tuple(events), duration_s=samples * raster), rf, gradients


def test_hargreaves_mode_reproduces_bloch_c_time_stepping():
    program, rf, gradients = _random_program()
    position = np.array([0.004, -0.007, 0.002])
    result = simulate_reference_sequence(
        program,
        method="hargreaves",
        positions_m=position,
        frequency_offsets_hz=37.0,
        t1_s=0.05,
        t2_s=0.01,
        initial_magnetization=(0.1, -0.2, 0.9),
    )
    expected = _bloch_c_literal(
        rf, gradients, 10e-6, 0.05, 0.01, 37.0, position, (0.1, -0.2, 0.9)
    )
    assert result.method == "hargreaves"
    assert result.final_magnetization[0] == pytest.approx(expected, abs=1e-14)


def test_hargreaves_equals_exact_without_rf():
    program = SequenceProgram(
        (GradientEvent("x", 0.0, np.linspace(-800.0, 900.0, 25), RASTER),),
        duration_s=25 * RASTER,
    )
    kwargs = dict(
        positions_m=[[0.003, 0, 0], [-0.006, 0, 0]],
        frequency_offsets_hz=[12.0, -40.0],
        t1_s=[0.02, 0.4],
        t2_s=[0.004, 0.03],
        initial_magnetization=(0.5, 0.3, 0.2),
    )
    exact = simulate_reference_sequence(program, **kwargs)
    hargreaves = simulate_reference_sequence(program, method="hargreaves", **kwargs)
    assert hargreaves.final_magnetization == pytest.approx(
        exact.final_magnetization, abs=1e-14
    )


def test_hargreaves_converges_first_order_to_exact():
    errors = []
    for repeat in (1, 2, 4):
        program = SequenceProgram(
            (
                RFEvent(
                    0.0,
                    np.repeat([400.0 + 150.0j, -250.0j, 300.0], repeat),
                    RASTER / repeat,
                ),
            ),
            duration_s=3 * RASTER,
        )
        kwargs = dict(frequency_offsets_hz=150.0, t1_s=2e-3, t2_s=5e-4)
        exact = simulate_reference_sequence(program, **kwargs)
        hargreaves = simulate_reference_sequence(program, method="hargreaves", **kwargs)
        errors.append(
            np.abs(hargreaves.final_magnetization - exact.final_magnetization).max()
        )
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.1)
    assert errors[1] / errors[2] == pytest.approx(2.0, rel=0.1)


@pytest.mark.parametrize("method", ["exact", "hargreaves"])
def test_steady_state_is_fixed_point_and_limit_of_transient(method):
    tr = 10 * RASTER
    program = SequenceProgram(
        (RFEvent(0.0, np.full(2, 1500.0 + 0j), RASTER),), duration_s=tr
    )
    kwargs = dict(
        method=method,
        frequency_offsets_hz=[0.0, 900.0],
        t1_s=2e-3,
        t2_s=1e-3,
    )
    steady = simulate_reference_sequence(program, steady_state=True, **kwargs)
    assert steady.final_magnetization == pytest.approx(
        steady.initial_magnetization, abs=1e-13
    )

    state = np.tile([0.0, 0.0, 1.0], (2, 1))
    for _ in range(400):
        state = simulate_reference_sequence(
            program, initial_magnetization=state, **kwargs
        ).final_magnetization
    assert state == pytest.approx(steady.initial_magnetization, abs=1e-10)


def test_two_pool_conversion_matches_analytic_solution():
    duration = 2.0
    kpl, r1p, r1l = 0.3, 1 / 30.0, 1 / 25.0
    pools = (
        ReferencePool(0.0, 30.0, 1.0, 0.0, (0.0, 0.0, 1.0)),
        ReferencePool(400.0, 25.0, 1.0, 0.0, (0.0, 0.0, 0.0)),
    )
    times = (0.0, 0.5, duration)
    result = simulate_reference_sequence(
        SequenceProgram((), duration_s=duration),
        pools=pools,
        exchange_rates_per_s=[[-kpl, 0.0], [kpl, 0.0]],
        checkpoints_s=times,
    )
    t = np.asarray(times)
    pyruvate = np.exp(-(kpl + r1p) * t)
    lactate = kpl / (r1l - r1p - kpl) * (np.exp(-(kpl + r1p) * t) - np.exp(-r1l * t))
    assert result.checkpoint_magnetization.shape == (3, 1, 2, 3)
    assert result.checkpoint_magnetization[:, 0, 0, 2] == pytest.approx(
        pyruvate, abs=1e-13
    )
    assert result.checkpoint_magnetization[:, 0, 1, 2] == pytest.approx(
        lactate, abs=1e-13
    )


def test_linear_source_is_integrated_exactly_between_knots():
    duration = 3e-3
    knots = (0.0, 1e-3, duration)
    values = (0.0, 600.0, 150.0)
    result = simulate_reference_sequence(
        SequenceProgram((), duration_s=duration),
        pools=(ReferencePool(t1_s=1e12, t2_s=1.0, initial_magnetization=(0, 0, 0)),),
        longitudinal_source=lambda t: np.interp(t, knots, values),
        source_knots_s=knots,
    )
    expected = 0.5 * 1e-3 * 600.0 + 0.5 * 2e-3 * (600.0 + 150.0)
    assert result.final_magnetization[0, 0, 2] == pytest.approx(expected, rel=1e-12)


def test_hargreaves_rejects_kinetics_and_pools_reject_scalar_relaxation():
    program = SequenceProgram((), duration_s=1e-3)
    pools = (ReferencePool(), ReferencePool(frequency_offset_hz=50.0))
    with pytest.raises(ValueError, match="hargreaves"):
        simulate_reference_sequence(
            program,
            method="hargreaves",
            pools=pools,
            exchange_rates_per_s=[[-1.0, 0.0], [1.0, 0.0]],
        )
    with pytest.raises(ValueError, match="through pools"):
        simulate_reference_sequence(program, pools=pools, t1_s=1.0)
    with pytest.raises(ValueError, match="method"):
        simulate_reference_sequence(program, method="euler")


def _kinetic_phantom():
    shape = (2, 1, 1)
    return DynamicSpectralPhantom(
        shape=shape,
        fov=(0.02, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.asarray([0.5, 3.0]).reshape(shape),
        nucleus="C13",
    )


def _dynamic_versus_gold(rf_scale, repeat):
    knots = (0.0, 6 * RASTER, 12 * RASTER)
    values = (0.0, 200.0, 100.0)
    delivery = (1.0, 0.5)
    phantom = _kinetic_phantom()
    phantom.pyruvate_inflow = PyruvateInflow(
        TimeCurve(knots, values, "linear", "hold"),
        np.asarray(delivery).reshape(phantom.shape),
    )
    raster = RASTER / repeat
    rf = rf_scale * np.asarray([800 + 200j, 1200 - 100j, 600 + 50j, 0j])
    program = SequenceProgram(
        (
            RFEvent(2 * RASTER, np.repeat(rf, repeat), raster),
            GradientEvent(
                "x", 0.0, np.repeat(np.linspace(-1500.0, 2000.0, 12), repeat), raster
            ),
        ),
        duration_s=12 * RASTER,
    )
    checkpoints = (0.0, 7 * RASTER, 12 * RASTER)
    dynamic = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom, checkpoints_s=checkpoints, simulation_timestep_s=raster
    )
    offsets = dynamic.metadata["pool_frequency_offsets_hz"]
    pools = (
        ReferencePool(offsets[0], 30.0, 1.0, 0.0, (0.0, 0.0, 1.0)),
        ReferencePool(offsets[1], 25.0, 1.0, 0.0, (0.0, 0.0, 0.0)),
    )
    error = 0.0
    for voxel, (x, kpl) in enumerate(((-0.005, 0.5), (0.005, 3.0))):
        gold = simulate_reference_sequence(
            program,
            positions_m=(x, 0.0, 0.0),
            pools=pools,
            exchange_rates_per_s=[[-kpl, 0.0], [kpl, 0.0]],
            longitudinal_source=lambda t, d=delivery[voxel]: (
                d * np.interp(t, knots, values),
                0.0,
            ),
            source_knots_s=knots,
            checkpoints_s=checkpoints,
        )
        actual = dynamic.checkpoint_pool_magnetization[:, :, voxel, 0, 0]
        error = max(error, np.abs(gold.checkpoint_magnetization[:, 0] - actual).max())
    return error


def test_dynamic_kernel_matches_gold_kinetics_without_rf():
    assert _dynamic_versus_gold(0.0, 1) < 1e-13


def test_dynamic_kernel_converges_second_order_to_gold_with_rf():
    errors = [_dynamic_versus_gold(1.0, repeat) for repeat in (1, 2, 4)]
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.05)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.05)
