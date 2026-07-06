"""Small native examples for the sparse 3D sequence simulator."""

from __future__ import annotations

import numpy as np

from blochsimulator import BlochSimulator
from blochsimulator.phantom import Phantom, PhantomFactory
from blochsimulator.sequence import (
    ADCEvent,
    CartesianAcquisition,
    GradientEvent,
    RFEvent,
    SequenceProgram,
    make_cartesian_epi,
)


def fid_program(samples=256, dwell_s=100e-6):
    """90-degree excitation followed by an ADC FID."""
    pulse_duration = 1e-3
    adc_start = pulse_duration + dwell_s / 2
    return SequenceProgram(
        (
            RFEvent(0.0, np.array([250.0]), pulse_duration),
            ADCEvent(adc_start, samples, dwell_s),
        ),
        duration_s=pulse_duration + samples * dwell_s,
        source="example-fid",
    )


def spin_echo_program(te_s=50e-3, samples=256, dwell_s=100e-6):
    """90/180 spin echo with ADC centred on TE."""
    pulse_duration = 1e-3
    adc_start = te_s - samples * dwell_s / 2 + dwell_s / 2
    duration = max(te_s + samples * dwell_s / 2, te_s + dwell_s)
    return SequenceProgram(
        (
            RFEvent(0.0, np.array([250.0]), pulse_duration),
            RFEvent(te_s / 2, np.array([500.0]), pulse_duration),
            ADCEvent(adc_start, samples, dwell_s),
        ),
        duration_s=duration,
        source="example-spin-echo",
    )


def gradient_echo_program(samples=128, dwell_s=20e-6):
    """30-degree excitation, readout prephaser, and frequency-encoded ADC."""
    pulse_duration = 1e-3
    prephase_duration = 1e-3
    readout_duration = samples * dwell_s
    readout_start = pulse_duration + prephase_duration
    readout_gradient = 12_000.0  # Hz/m
    prephase_gradient = -readout_gradient * readout_duration / (2 * prephase_duration)
    return SequenceProgram(
        (
            RFEvent(0.0, np.array([250.0 / 3.0]), pulse_duration),
            GradientEvent(
                "x", pulse_duration, np.array([prephase_gradient]), prephase_duration
            ),
            GradientEvent(
                "x", readout_start, np.array([readout_gradient]), readout_duration
            ),
            ADCEvent(readout_start + dwell_s / 2, samples, dwell_s),
        ),
        duration_s=readout_start + readout_duration,
        source="example-gradient-echo",
    )


def run_example(program=None):
    """Simulate one program on a small 3D uniform phantom."""
    program = fid_program() if program is None else program
    phantom = PhantomFactory.uniform(
        shape=(8, 8, 8),
        fov=(0.16, 0.16, 0.16),
        t1=1.0,
        t2=0.1,
        pd=1.0,
    )
    return BlochSimulator(use_parallel=True).simulate_sequence(program, phantom)


def reconstruct_cartesian_epi_example(matrix=(16, 16), dwell_s=20e-6):
    """Simulate Cartesian EPI and return raw ADC, k-space, and FFT image."""
    fov = (0.16, 0.16)
    pd = np.zeros(matrix)
    pd[matrix[0] // 4 : 3 * matrix[0] // 4, matrix[1] // 3 : 2 * matrix[1] // 3] = 1
    phantom = Phantom(
        shape=matrix,
        fov=fov,
        t1_map=np.full(matrix, 1e9),
        t2_map=np.full(matrix, 1e9),
        pd_map=pd,
    )
    acquisition = CartesianAcquisition.epi(
        read_matrix=matrix[0],
        phase_matrix=matrix[1],
        fov_m=fov,
        dwell_s=dwell_s,
    )
    program = make_cartesian_epi(acquisition)
    result = BlochSimulator(use_parallel=True).simulate_sequence(program, phantom)
    kspace = result.to_cartesian_kspace(acquisition)
    image = result.reconstruct_cartesian(acquisition)
    return result, kspace, image


if __name__ == "__main__":
    result = run_example()
    print(
        f"ADC samples: {result.signal.size}; "
        f"peak signal: {np.max(np.abs(result.signal)):.6g}"
    )
