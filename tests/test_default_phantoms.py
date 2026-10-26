import numpy as np
import pytest

from blochsimulator.default_phantoms import (
    DEFAULT_COMPOUNDS,
    build_default_phantom,
    default_phantom_grid,
    is_default_phantom,
)
from blochsimulator.units import NUCLEUS_GAMMA_HZ_PER_T


def test_every_nucleus_has_a_default_compound():
    assert set(DEFAULT_COMPOUNDS) == set(NUCLEUS_GAMMA_HZ_PER_T)
    assert DEFAULT_COMPOUNDS["H1"].name == "Water"
    assert DEFAULT_COMPOUNDS["C13"].name == "Pyruvate"


@pytest.mark.parametrize("nucleus", sorted(NUCLEUS_GAMMA_HZ_PER_T))
def test_default_sphere_is_on_resonance_at_the_sequence_reference(nucleus):
    reference_ppm = 171.0 if nucleus == "C13" else 0.0
    phantom = build_default_phantom(
        nucleus,
        field_strength_t=3.0,
        sequence_reference_ppm=reference_ppm,
        side_m=0.1,
        matrix=32,
    )

    assert is_default_phantom(phantom)
    assert phantom.nucleus == nucleus
    assert phantom.spectral_reference_ppm == pytest.approx(reference_ppm)
    assert [species.chemical_shift_ppm for species in phantom.species] == [0.0]
    assert phantom.shape == (32, 32, 32)
    assert phantom.fov == pytest.approx((0.1, 0.1, 0.1))
    # A centred sphere with 60 % diameter fills about π/6 · 0.6³ of the cube;
    # supersampled partial-volume edge cells count as active.
    filled = phantom.n_active / np.prod(phantom.shape)
    assert filled == pytest.approx(np.pi / 6 * 0.6**3, rel=0.2)
    assert not phantom.mask[0, 0, 0]
    assert phantom.mask[16, 16, 16]


def test_default_grid_fits_the_smallest_encoded_fov_at_sequence_resolution():
    assert default_phantom_grid() == (0.22, (64, 64, 64))
    assert default_phantom_grid((0.22, 0.22), (64, 64)) == (0.22, (64, 64, 64))
    # 3D: smallest FOV, finest pixel, rounded up to a multiple of eight.
    side, shape = default_phantom_grid((0.22, 0.22, 0.16), (100, 100, 20))
    assert side == pytest.approx(0.16)
    assert shape == (80, 80, 80)
    # Coarse sequences keep a minimum grid; fine ones are capped.
    assert default_phantom_grid((0.03, 0.03), (8, 8))[1] == (32, 32, 32)
    assert default_phantom_grid((0.2, 0.2), (512, 512))[1] == (128, 128, 128)


def test_default_grid_shrinks_for_oblique_encoding():
    angle = np.deg2rad(45.0)
    frame = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    side, _ = default_phantom_grid((0.2, 0.2), (32, 32), frame)
    assert side == pytest.approx(0.2 / np.sqrt(2.0))
    required = np.abs(frame).T @ np.full(3, side)
    assert np.all(required[:2] <= 0.2 + 1e-12)


def test_default_grid_resolves_the_slice_thickness():
    side, shape = default_phantom_grid(
        (0.22, 0.22),
        (16, 16),
        np.eye(3),
        slice_thickness_m=0.01,
        slice_normal=(0.0, 0.0, 1.0),
    )
    assert side == pytest.approx(0.22)
    assert shape[:2] == (32, 32)
    assert side / shape[2] <= 0.005 + 1e-12
    # Very thin slices cap the slice axis only.
    _, thin = default_phantom_grid(
        (0.22, 0.22),
        (16, 16),
        np.eye(3),
        slice_thickness_m=0.0005,
        slice_normal=(1.0, 0.0, 0.0),
    )
    assert thin == (256, 32, 32)


def test_default_sphere_uses_5_ppm_and_257_points():
    phantom = build_default_phantom(
        "C13", field_strength_t=3.0, sequence_reference_ppm=171.0, matrix=8
    )
    design = phantom.metadata["phantom_design"]
    assert design["spectral_bandwidth_ppm"] == pytest.approx(5.0)
    assert design["spectral_points"] == 257
    assert design["isotope_enrichment"] == 1.0
    assert phantom.metadata["isotope_enrichment"] == 1.0
