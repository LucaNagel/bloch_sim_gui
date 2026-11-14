"""The dynamic kernel converges in second order with exchange and inflow.

The dynamic kernel splits free precession, relaxation, conversion, inflow and
washout symmetrically around each RF rotation. Only the RF-active intervals
follow the time step; free intervals are integrated exactly. The time-step
error must therefore fall by a factor of four per halving of the step, also
with kPL > 0, pyruvate inflow (with a co-labelled pool) and voxel washout
(with a source compartment).

Errors are measured against the same kernel at 1 us, the RF raster. Against
that reference a second-order error falls by ``(h1**2 - 1) / (h2**2 - 1)``
(h in us): 4.03 from 20 to 10 us and 4.12 from 10 to 5 us.
"""

import warnings

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import (
    DynamicSpectralPhantom,
    PyruvateInflow,
    TimeCurve,
    VoxelWashout,
)
from blochsimulator.kernel_capabilities import (
    GUI_DYNAMIC_KERNEL_NAMES,
    KernelFallbackWarning,
)
from blochsimulator.sequence.model import (
    ADCEvent,
    GradientEvent,
    RFEvent,
    SequenceProgram,
)
from blochsimulator.spectral_phantom import ChemicalSpecies

SHAPE = (1, 1, 6)
FOV_M = (0.01, 0.01, 0.03)
FIELD_T = 3.0
NUCLEUS = "C13"
RASTER_S = 1e-6
STEPS_S = (20e-6, 10e-6, 5e-6)
REFERENCE_S = 1e-6
# Second-order ratios against the 1 us reference (see the module docstring).
EXPECTED_RATIOS = (399.0 / 99.0, 99.0 / 24.0)


def _slice_train(repetitions=4, repetition_s=4e-3):
    """Phase-cycled 30 degree, 1 ms sinc pulses under a slice gradient."""
    lobes = np.linspace(-2.0, 2.0, int(round(1e-3 / RASTER_S)))
    shape = np.sinc(lobes)
    pulse_hz = 30.0 / 360.0 / 1e-3 / np.mean(shape) * shape
    events = []
    for index in range(repetitions):
        start = index * repetition_s
        phase = np.pi * index
        events.append(RFEvent(start, pulse_hz * np.exp(1j * phase), RASTER_S))
        events.append(GradientEvent("z", start, np.full(100, 1.0e5), 10e-6))
        events.append(GradientEvent("z", start + 1.0e-3, np.full(50, -1.0e5), 10e-6))
        events.append(ADCEvent(start + 1.6e-3, 32, 50e-6, phase_offset_rad=phase))
    return SequenceProgram(tuple(events), repetitions * repetition_s)


def _block_train(repetitions=2, repetition_s=5e-3):
    """On-resonance 90 degree, 2 ms block pulses without gradients."""
    count = int(round(2e-3 / RASTER_S))
    amplitude_hz = 90.0 / 360.0 / 2e-3
    events = []
    for index in range(repetitions):
        start = index * repetition_s
        samples = np.full(count, amplitude_hz * np.exp(0.5j * index))
        events.append(RFEvent(start, samples, RASTER_S))
        events.append(ADCEvent(start + 2.5e-3, 16, 50e-6))
    return SequenceProgram(tuple(events), repetitions * repetition_s)


def _phantom(
    *,
    kpl_s_inv=(2.0, 20.0),
    inflow=True,
    washout=True,
    shifts_ppm=(0.0, 12.0, 9.0),
    b0_hz=(-150.0, 150.0),
):
    """Pyruvate, lactate and a co-labelled pool on a line of six voxels.

    The last voxel is a washout source compartment and therefore does not
    convert. Rates are far above in vivo values so that conversion, inflow
    and washout change the result within the 16 ms slice train.
    """
    grid = np.linspace(0.0, 1.0, int(np.prod(SHAPE))).reshape(SHAPE)
    density = {
        "Pyruvate": 1.0 + 0.2 * grid,
        "Lactate": 0.1 + 0.05 * grid,
        "C1": 0.5 + 0.1 * grid,
    }
    kpl = kpl_s_inv[0] + (kpl_s_inv[1] - kpl_s_inv[0]) * grid
    source = np.zeros(SHAPE, dtype=bool)
    if washout:
        source.reshape(-1)[-1] = True
        kpl = np.where(source, 0.0, kpl)
    phantom = DynamicSpectralPhantom(
        shape=SHAPE,
        fov=FOV_M,
        pools=tuple(
            ChemicalSpecies(name, shift, t1, t2)
            for name, shift, t1, t2 in zip(
                density, shifts_ppm, (30.0, 25.0, 20.0), (1.0, 0.8, 0.6)
            )
        ),
        initial_concentration_maps={name: values for name, values in density.items()},
        initial_spin_density_maps={name: values for name, values in density.items()},
        kpl_map_s_inv=kpl,
        b0_map=b0_hz[0] + (b0_hz[1] - b0_hz[0]) * grid,
        field_strength=FIELD_T,
        nucleus=NUCLEUS,
        spectral_reference_ppm=0.0,
    )
    if inflow:
        phantom.pyruvate_inflow = PyruvateInflow(
            TimeCurve((0.0, 8e-3, 20e-3, 32e-3), (20.0, 300.0, 100.0, 200.0)),
            0.8 + 0.4 * grid,
            TimeCurve((0.0, 16e-3, 32e-3), (0.5, 0.3, 0.45)),
            colabelled_pools=("C1",),
        )
    if washout:
        compartments = np.where(source, 0, -1).astype(np.int32)
        rates = np.asarray((30.0, 300.0, 120.0))
        concentration = np.broadcast_to(rates, (1, 3, 3)).copy()
        phantom.voxel_washout = VoxelWashout(
            loss_map_s_inv=5.0 + 40.0 * grid[..., ::-1],
            compartment_map=compartments,
            times_s=np.asarray((2e-3, 12e-3, 30e-3)),
            concentration_rates=concentration,
            magnetization_rates=0.4 * concentration,
        )
    phantom.__post_init__()
    return phantom


def _run(kernel, phantom, program, step_s):
    simulator = BlochSimulator(
        use_parallel=kernel == "native_parallel",
        dynamic_sequence_kernel=kernel,
    )
    with warnings.catch_warnings():
        # The requested kernel must run, not a fallback.
        warnings.simplefilter("error", KernelFallbackWarning)
        return simulator.simulate_dynamic_sequence(
            program,
            phantom,
            simulation_timestep_s=step_s,
            field_strength_t=FIELD_T,
            nucleus=NUCLEUS,
            sequence_reference_ppm=0.0,
        )


def _errors(kernel, phantom, program):
    """Return (signal, final magnetization) errors of STEPS_S against 1 us."""
    reference = _run(kernel, phantom, program, REFERENCE_S)
    signal = np.asarray(reference.signal).ravel()
    final = np.asarray(reference.final_pool_magnetization)
    signal_errors, final_errors = [], []
    for step_s in STEPS_S:
        result = _run(kernel, phantom, program, step_s)
        signal_errors.append(
            np.abs(np.asarray(result.signal).ravel() - signal).max()
            / np.abs(signal).max()
        )
        final_errors.append(
            np.abs(np.asarray(result.final_pool_magnetization) - final).max()
            / np.abs(final).max()
        )
    return np.asarray(signal_errors), np.asarray(final_errors)


def _assert_second_order(errors):
    ratios = errors[:-1] / errors[1:]
    np.testing.assert_allclose(ratios, EXPECTED_RATIOS, rtol=0.05)


@pytest.mark.parametrize("kernel", GUI_DYNAMIC_KERNEL_NAMES)
def test_second_order_with_exchange_inflow_and_washout(kernel):
    program = _slice_train()
    phantom = _phantom()
    result = _run(kernel, phantom, program, STEPS_S[0])
    assert result.metadata["colabelled_pools"] == ["C1"]
    assert result.metadata["voxel_washout_model"]
    plain = _run(
        kernel,
        _phantom(kpl_s_inv=(0.0, 0.0), inflow=False, washout=False),
        program,
        STEPS_S[0],
    )
    # Conversion, inflow and washout change the magnetization substantially.
    assert (
        np.abs(
            np.asarray(result.final_pool_magnetization)
            - np.asarray(plain.final_pool_magnetization)
        ).max()
        > 0.05 * np.abs(np.asarray(plain.final_pool_magnetization)).max()
    )

    signal_errors, final_errors = _errors(kernel, phantom, program)
    # Measured on an M3: 6.1e-4 (signal) and 8.5e-4 (final) at 20 us.
    assert 1e-4 < signal_errors[0] < 2e-3
    assert 3e-4 < final_errors[0] < 3e-3
    _assert_second_order(signal_errors)
    _assert_second_order(final_errors)


@pytest.mark.parametrize("kernel", GUI_DYNAMIC_KERNEL_NAMES)
def test_exchange_and_inflow_splitting_error_is_second_order(kernel):
    """On resonance the RF rotation is exact per step.

    The remaining step error comes from splitting relaxation, conversion,
    inflow and washout sources around the rotation. Relaxation alone leaves
    almost nothing; the coupled terms dominate and still fall in second
    order. The washout loss itself commutes with RF and adds no error.
    """
    program = _block_train()
    on_resonance = dict(shifts_ppm=(0.0, 0.0, 0.0), b0_hz=(0.0, 0.0))
    _, relaxation_only = _errors(
        kernel,
        _phantom(kpl_s_inv=(0.0, 0.0), inflow=False, washout=False, **on_resonance),
        program,
    )
    _, coupled = _errors(
        kernel,
        _phantom(kpl_s_inv=(50.0, 150.0), **on_resonance),
        program,
    )
    # Measured on an M3 at 20 us: 6e-9 (relaxation only), 6e-6 (coupled).
    assert relaxation_only[0] < 1e-7
    assert coupled[0] > 100.0 * relaxation_only[0]
    _assert_second_order(coupled)
