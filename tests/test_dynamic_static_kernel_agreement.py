"""The dynamic and static sequence kernels agree on one relaxing pool.

The static kernel rotates exactly and then relaxes each step; the dynamic
kernel splits free precession symmetrically around the RF rotation. Without
exchange both describe the same Bloch equation, so their difference is a
pure time-step error that must shrink as the step shrinks.
"""

import warnings

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom
from blochsimulator.kernel_capabilities import (
    GUI_DYNAMIC_KERNEL_NAMES,
    GUI_SEQUENCE_KERNEL_NAMES,
    KernelFallbackWarning,
)
from blochsimulator.sequence.model import (
    ADCEvent,
    GradientEvent,
    RFEvent,
    SequenceProgram,
)
from blochsimulator.spectral_phantom import ChemicalSpecies, SpectralPhantom

SHAPE = (1, 1, 5)
FOV_M = (0.01, 0.01, 0.025)
FIELD_T = 3.0
NUCLEUS = "C13"
STEPS_S = (20e-6, 5e-6, 1e-6)
# Measured on an M3: 7.3e-4 at 20 us, 6.6e-5 at 5 us and 7.1e-6 at 1 us.
SIGNAL_TOLERANCE_AT_1_US = 2e-5
MAGNETIZATION_TOLERANCE_AT_1_US = 1e-5


def _slice_train(repetitions=4, repetition_s=4e-3):
    """Return phase-cycled 1 ms sinc excitations under a slice gradient."""
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
    return SequenceProgram(tuple(events), repetitions * repetition_s)


def _phantoms():
    """Return a one-pool dynamic phantom and its static spectral twin."""
    b0_hz = np.linspace(-200.0, 200.0, 5).reshape(SHAPE)
    species = ChemicalSpecies("A", 3.0, 2.0, 0.08)
    ones, zeros = np.ones(SHAPE), np.zeros(SHAPE)
    # The second pool stays empty; the dynamic phantom needs two pools.
    # Thermal relaxation toward M0 needs the spin density maps, otherwise
    # the dynamic kernel treats Mz as hyperpolarized and relaxes it to zero.
    dynamic = DynamicSpectralPhantom(
        shape=SHAPE,
        fov=FOV_M,
        pools=(species, ChemicalSpecies("B", 13.0, 2.0, 0.08)),
        initial_concentration_maps={"A": ones, "B": zeros},
        initial_spin_density_maps={"A": ones, "B": zeros},
        equilibrium_polarization=1.0,
        kpl_map_s_inv=zeros,
        b0_map=b0_hz,
        field_strength=FIELD_T,
        nucleus=NUCLEUS,
        spectral_reference_ppm=0.0,
    )
    static = SpectralPhantom(
        shape=SHAPE,
        fov=FOV_M,
        species=[species],
        concentration_maps={"A": ones},
        b0_map=b0_hz,
        field_strength=FIELD_T,
        nucleus=NUCLEUS,
    )
    return dynamic, static


@pytest.mark.parametrize("static_kernel", GUI_SEQUENCE_KERNEL_NAMES)
@pytest.mark.parametrize("dynamic_kernel", GUI_DYNAMIC_KERNEL_NAMES)
def test_kernel_difference_shrinks_with_the_time_step(dynamic_kernel, static_kernel):
    simulator = BlochSimulator(
        use_parallel=False,
        sequence_kernel=static_kernel,
        dynamic_sequence_kernel=dynamic_kernel,
    )
    program = _slice_train()
    dynamic_phantom, static_phantom = _phantoms()
    options = dict(
        field_strength_t=FIELD_T, nucleus=NUCLEUS, sequence_reference_ppm=0.0
    )
    signal_errors, magnetization_errors = [], []
    for step_s in STEPS_S:
        with warnings.catch_warnings():
            # The requested kernels must run, not a fallback.
            warnings.simplefilter("error", KernelFallbackWarning)
            dynamic = simulator.simulate_dynamic_sequence(
                program, dynamic_phantom, simulation_timestep_s=step_s, **options
            )
            static = simulator.simulate_spectral_sequence(
                program, static_phantom, simulation_timestep_s=step_s, **options
            )
        static_signal = np.asarray(static.signal).ravel()
        dynamic_signal = np.asarray(dynamic.signal).ravel()
        assert dynamic_signal.shape == static_signal.shape
        signal_errors.append(
            np.abs(dynamic_signal - static_signal).max() / np.abs(static_signal).max()
        )
        static_final = np.asarray(static.final_magnetization)
        dynamic_final = np.asarray(dynamic.final_magnetization).reshape(
            static_final.shape
        )
        magnetization_errors.append(np.abs(dynamic_final - static_final).max())

    if static_kernel == dynamic_kernel == "production":
        # Static production runs on the same spin-major executor.
        assert max(signal_errors + magnetization_errors) < 1e-12
        return
    for errors in (signal_errors, magnetization_errors):
        assert errors[0] > 4.0 * errors[1] > 16.0 * errors[2]
    assert signal_errors[-1] < SIGNAL_TOLERANCE_AT_1_US
    assert magnetization_errors[-1] < MAGNETIZATION_TOLERANCE_AT_1_US
