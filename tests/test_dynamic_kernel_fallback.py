"""Mistyped options and replaced kernels must not pass silently."""

import warnings

import numpy as np
import pytest

from blochsimulator import (
    BlochSimulator,
    KernelFallbackWarning,
    dynamic_kernel_fallback,
)
from blochsimulator.dynamic_phantom import (
    DynamicSpectralPhantom,
    simulate_dynamic_sequence,
)
from blochsimulator.sequence import ADCEvent, RFEvent, SequenceProgram
from blochsimulator.spectral_phantom import ChemicalSpecies

RASTER = 20e-6


def _phantom():
    shape = (4, 1, 1)
    grid = np.linspace(0.0, 1.0, 4).reshape(shape)
    return DynamicSpectralPhantom(
        shape=shape,
        fov=(0.04, 0.01, 0.01),
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
        nucleus="C13",
    )


def _program():
    return SequenceProgram(
        (
            RFEvent(0.0, np.full(5, 60.0 - 20.0j), RASTER),
            ADCEvent(6 * RASTER, 4, 2 * RASTER),
        ),
        duration_s=16 * RASTER,
    )


def _run(kernel, **kwargs):
    kwargs.setdefault("simulation_timestep_s", RASTER)
    simulator = BlochSimulator(use_parallel=False, num_threads=1)
    return simulator.simulate_dynamic_sequence(
        _program(), _phantom(), sequence_kernel=kernel, **kwargs
    )


def test_misspelled_option_is_rejected():
    with pytest.raises(TypeError, match="simulation_timestep"):
        _run("optimized", simulation_timestep=1e-5)
    # The static solver's block option has no meaning for a dynamic phantom.
    with pytest.raises(TypeError, match="chunk_voxels"):
        simulate_dynamic_sequence(
            _program(), _phantom(), sequence_kernel="optimized", chunk_voxels=2
        )


def test_native_run_and_deliberate_switch_do_not_warn():
    pytest.importorskip("blochsimulator.dynamic_spin_major_cy")
    with warnings.catch_warnings():
        warnings.simplefilter("error", KernelFallbackWarning)
        native = _run("native_serial")
        disabled = _run("native_serial", _native_spin_major=False)
    assert native.metadata["native_spin_major_enabled"]
    assert not disabled.metadata["native_spin_major_enabled"]
    assert dynamic_kernel_fallback(disabled.metadata) is None


def test_lost_spin_major_executor_warns_at_the_call_site():
    pytest.importorskip("blochsimulator.dynamic_spin_major_cy")
    with pytest.warns(KernelFallbackWarning, match="float64") as caught:
        result = _run("native_serial", checkpoint_dtype=np.float32)
    assert not result.metadata["native_spin_major_enabled"]
    assert "spin-major" in str(caught[0].message)

    with pytest.warns(KernelFallbackWarning) as caught:
        simulate_dynamic_sequence(
            _program(),
            _phantom(),
            sequence_kernel="native_serial",
            simulation_timestep_s=RASTER,
            checkpoint_dtype=np.float32,
        )
    assert caught[0].filename == __file__

    # The reference and NumPy kernels have no faster path to lose.
    with warnings.catch_warnings():
        warnings.simplefilter("error", KernelFallbackWarning)
        _run("optimized", checkpoint_dtype=np.float32)


def test_fallback_can_be_turned_into_an_error():
    pytest.importorskip("blochsimulator.dynamic_spin_major_cy")
    with warnings.catch_warnings():
        warnings.simplefilter("error", KernelFallbackWarning)
        with pytest.raises(KernelFallbackWarning):
            _run("native_serial", checkpoint_dtype=np.float32)


def test_fallback_summary_covers_each_replaced_path():
    assert dynamic_kernel_fallback(None) is None
    assert (
        dynamic_kernel_fallback(
            {
                "requested_sequence_kernel": "native_parallel",
                "sequence_kernel": "native_parallel",
                "native_spin_major_enabled": True,
            }
        )
        is None
    )
    replaced = dynamic_kernel_fallback(
        {
            "requested_sequence_kernel": "native_parallel",
            "sequence_kernel": "optimized",
            "native_fallback_reason": "strict native extension is unavailable",
        }
    )
    assert replaced == (
        "requested the native_parallel kernel, ran optimized: "
        "strict native extension is unavailable"
    )
    slower = dynamic_kernel_fallback(
        {
            "requested_sequence_kernel": "native_parallel",
            "sequence_kernel": "native_parallel",
            "native_spin_major_enabled": False,
            "native_spin_major_fallback_reason": "dynamic B0 is unsupported",
        }
    )
    assert slower == (
        "the native_parallel kernel ran without its spin-major executor: "
        "dynamic B0 is unsupported"
    )
    contrast = dynamic_kernel_fallback(
        {
            "requested_sequence_kernel": "reference",
            "sequence_kernel": "reference",
            "contrast_kernel_fallback_from": "native_parallel",
            "contrast_kernel_fallback_reason": "memory budget is too small",
        }
    )
    assert contrast == (
        "requested the native_parallel kernel, ran reference: contrast relaxation "
        "needs the spin-major executor (memory budget is too small)"
    )
