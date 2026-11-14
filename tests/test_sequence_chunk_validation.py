"""The native sequence chunk rejects state indices its cursors cannot follow."""

import importlib

import numpy as np
import pytest

N_INTERVALS = 12
N_SPINS = 3

KERNELS = [
    ("blochsimulator.blochsimulator_cy", "optimized"),
    ("blochsimulator.blochsimulator_cy", "reference"),
    ("blochsimulator.blochsimulator_strict_cy", "reference"),
]


def _chunk_function(module_name):
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        pytest.skip(f"{module_name} is not built")
    return module.simulate_sequence_chunk


def _run(module_name, kernel, adc, checkpoints, crushers=(), num_threads=1):
    rng = np.random.default_rng(3)
    rf = np.zeros(N_INTERVALS, dtype=complex)
    rf[1] = 250.0 + 40.0j
    gradients = np.zeros((N_INTERVALS, 3))
    gradients[2:, 0] = 1.0e3
    adc = np.asarray(adc, dtype=np.int32)
    return _chunk_function(module_name)(
        rf,
        gradients,
        np.full(N_INTERVALS, 1.0e-4),
        np.full(N_SPINS, 1.0),
        np.full(N_SPINS, 0.05),
        rng.normal(0.0, 20.0, N_SPINS),
        rng.uniform(-0.01, 0.01, (N_SPINS, 3)),
        np.ones(N_SPINS),
        np.ones(N_SPINS, dtype=complex),
        np.ones((1, N_SPINS), dtype=complex),
        np.tile([0.0, 0.0, 1.0], (N_SPINS, 1)),
        adc,
        np.ones(adc.size, dtype=complex),
        np.asarray(checkpoints, dtype=np.int32),
        np.asarray(crushers, dtype=np.int32),
        num_threads,
        kernel,
    )


VALID_ADC = [2, 4, 4, 6, 8, 12]
VALID_CHECKPOINTS = [0, 5, 5, 12]
BROKEN = {
    "negative": [-1, 4, 6],
    "after_last_state": [2, 4, N_INTERVALS + 1],
    "falls_back": [2, 6, 4],
}


@pytest.mark.parametrize(("module_name", "kernel"), KERNELS)
@pytest.mark.parametrize("broken", sorted(BROKEN))
def test_invalid_adc_indices_raise(module_name, kernel, broken):
    with pytest.raises(ValueError, match="ADC state indices"):
        _run(module_name, kernel, BROKEN[broken], VALID_CHECKPOINTS)


@pytest.mark.parametrize(("module_name", "kernel"), KERNELS)
@pytest.mark.parametrize("broken", sorted(BROKEN))
def test_invalid_checkpoint_indices_raise(module_name, kernel, broken):
    with pytest.raises(ValueError, match="checkpoint state indices"):
        _run(module_name, kernel, VALID_ADC, BROKEN[broken])


@pytest.mark.parametrize(("module_name", "kernel"), KERNELS)
@pytest.mark.parametrize("num_threads", [1, 2])
def test_valid_indices_fill_every_sample_and_checkpoint(
    module_name, kernel, num_threads
):
    signal, final, checkpoints = _run(
        module_name,
        kernel,
        VALID_ADC,
        VALID_CHECKPOINTS,
        crushers=[7],
        num_threads=num_threads,
    )

    assert signal.shape == (1, len(VALID_ADC))
    assert np.all(signal[:, :4] != 0.0)
    # The crusher at state 7 removes the only transverse magnetization.
    np.testing.assert_array_equal(signal[:, 4:], 0.0)
    # Repeated indices sample the same state.
    np.testing.assert_array_equal(signal[:, 1], signal[:, 2])
    np.testing.assert_array_equal(checkpoints[1], checkpoints[2])
    np.testing.assert_array_equal(checkpoints[0], np.tile([0.0, 0.0, 1.0], (3, 1)))
    np.testing.assert_array_equal(checkpoints[-1], final)
