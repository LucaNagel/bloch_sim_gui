"""Voxelwise pool shifts and co-labelled inflow pools.

``chemical_shift_maps_ppm`` adds a per-voxel shift to individual pools (e.g. a
pH-dependent resonance). The spin-major, interval-major and reference paths
must agree on it. ``PyruvateInflow.colabelled_pools`` gives a further pool the
precursor's inflow; the superposition must equal a phantom in which that pool
is the precursor itself.
"""

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import (
    DynamicSpectralPhantom,
    PyruvateInflow,
    TimeCurve,
)
from blochsimulator.sequence import ADCEvent, GradientEvent, RFEvent, SequenceProgram
from blochsimulator.spectral_phantom import ChemicalSpecies

RASTER = 20e-6
OUTPUTS = (
    "signal",
    "species_signal",
    "final_magnetization",
    "final_pool_magnetization",
    "checkpoint_magnetization",
    "checkpoint_pool_magnetization",
)


def _phantom(*, shift_maps=True, colabelled=(), shape=(8, 1, 1)):
    grid = np.linspace(0.0, 1.0, int(np.prod(shape))).reshape(shape)
    pools = (
        ChemicalSpecies("C5", 3.0, 30.0, 1.0),
        ChemicalSpecies("Product", 12.0, 27.0, 1.0),
        ChemicalSpecies("C1", -4.0, 24.0, 1.0),
    )
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.06, 0.01, 0.01),
        pools=pools,
        initial_concentration_maps={
            "C5": 0.5 + 0.2 * grid,
            "Product": 0.02 + 0.03 * grid,
            "C1": np.zeros(shape),
        },
        kpl_map_s_inv=0.01 + 0.09 * grid,
        b0_map=-5.0 + 10.0 * grid,
        nucleus="C13",
        chemical_shift_maps_ppm=(
            {"C5": -2.0 + 4.0 * grid, "C1": 0.6 * grid[::-1]} if shift_maps else None
        ),
    )
    rate = TimeCurve((0.0, 3 * RASTER, 8 * RASTER, 13 * RASTER), (0.05, 0.6, 0.2, 0.4))
    polarization = TimeCurve((0.0, 6 * RASTER, 11 * RASTER), (0.5, 0.3, 0.45))
    phantom.pyruvate_inflow = PyruvateInflow(
        rate, 0.8 + 0.4 * grid, polarization, colabelled_pools=tuple(colabelled)
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


def _run(phantom, kernel="native_serial", **kwargs):
    return BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        _program(),
        phantom,
        sequence_kernel=kernel,
        checkpoints_s=(7 * RASTER, 25 * RASTER),
        simulation_timestep_s=RASTER,
        spoiler_mode="ideal",
        **kwargs,
    )


def _assert_close(actual, expected, tolerance=1e-10):
    for name in OUTPUTS:
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=tolerance,
            atol=tolerance,
            err_msg=name,
        )


def test_shift_maps_agree_across_solver_paths():
    phantom = _phantom()
    reference = _run(phantom, "reference")
    interval_major = _run(phantom, _native_spin_major=False)
    spin_major = _run(phantom)
    assert spin_major.metadata["native_spin_major_enabled"] is True
    assert interval_major.metadata["native_spin_major_enabled"] is False
    _assert_close(interval_major, reference, 1e-9)
    _assert_close(spin_major, interval_major)
    unshifted = _run(_phantom(shift_maps=False))
    assert not np.allclose(unshifted.species_signal[0], spin_major.species_signal[0])


def test_shift_map_equals_uniform_pool_shift():
    shape = (4, 1, 1)
    shifted = _phantom(shape=shape)
    shifted.chemical_shift_maps_ppm = {"C5": np.full(shape, 1.5)}
    moved = _phantom(shape=shape, shift_maps=False)
    moved.pools = (ChemicalSpecies("C5", 4.5, 30.0, 1.0),) + moved.pools[1:]
    _assert_close(_run(shifted), _run(moved))


def test_colabelled_pool_matches_standalone_precursor():
    phantom = _phantom(colabelled=("C1",))
    combined = _run(phantom)
    assert combined.metadata["colabelled_pools"] == ["C1"]
    assert np.max(np.abs(combined.species_signal[2])) > 0
    # The superposed passes keep the names of every pool.
    assert combined.pool_names == tuple(pool.name for pool in phantom.pools)
    assert combined.to_xarray().sizes["pool"] == phantom.n_species

    standalone = _phantom()
    standalone.pools = (standalone.pools[2], standalone.pools[0], standalone.pools[1])
    standalone.initial_concentration_maps = {
        name: np.zeros(standalone.shape) for name in ("C1", "C5", "Product")
    }
    standalone.kpl_map_s_inv = np.zeros(standalone.shape)
    expected = _run(standalone)
    np.testing.assert_allclose(
        combined.species_signal[2], expected.species_signal[0], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        combined.final_pool_magnetization[2],
        expected.final_pool_magnetization[0],
        rtol=1e-10,
        atol=1e-12,
    )
    primary = _run(_phantom())
    np.testing.assert_allclose(
        combined.species_signal[:2], primary.species_signal[:2], rtol=1e-12
    )
    np.testing.assert_allclose(
        combined.signal, combined.species_signal.sum(axis=0), rtol=1e-12
    )


def test_colabelled_pool_validation():
    with pytest.raises(ValueError, match="must not receive conversion"):
        _phantom(colabelled=("Product",)).__post_init__()
    with pytest.raises(ValueError, match="does not exist"):
        _phantom(colabelled=("Missing",)).__post_init__()


@pytest.mark.parametrize("suffix", [".npz", ".h5", ".nc"])
def test_shift_maps_and_colabelled_pools_round_trip(tmp_path, suffix):
    if suffix == ".nc":
        pytest.importorskip("netCDF4")
    if suffix == ".h5":
        pytest.importorskip("h5py")
    phantom = _phantom(colabelled=("C1",))
    path = phantom.save(tmp_path / f"phantom{suffix}")
    loaded = DynamicSpectralPhantom.load(path)
    assert loaded.pyruvate_inflow.colabelled_pools == ("C1",)
    assert set(loaded.chemical_shift_maps_ppm) == {"C5", "C1"}
    for name, values in phantom.chemical_shift_maps_ppm.items():
        np.testing.assert_allclose(loaded.chemical_shift_maps_ppm[name], values)


def test_spectrum_preview_includes_shift_map():
    phantom = _phantom()
    phantom.initial_concentration_maps["C1"] = np.zeros(phantom.shape)
    phantom.initial_concentration_maps["Product"] = np.zeros(phantom.shape)
    ppm, spectrum = phantom.spectrum_at_ppm((7, 0, 0), absolute=False)
    b0_ppm = phantom.get_b0_offset_map_ppm()[7, 0, 0]
    assert ppm[np.argmax(spectrum)] == pytest.approx(3.0 + 2.0 + b0_ppm, abs=0.05)
