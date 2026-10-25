import json

import numpy as np
import pytest
import xarray as xr

from blochsimulator.sequence.reconstruction import SequenceReconstructionModel
from blochsimulator.spectral_phantom import ChemicalSpecies, SpectralPhantom
from blochsimulator.ui.reconstruction_explorer import SequenceReconstructionExplorer
from blochsimulator.ui.reconstruction_phantom import (
    ImageGrid,
    contour_rgba,
    grid_description,
    planar_image_grid,
    resample_phantom,
    volume_display,
    volume_image_grid,
    volume_section_grid,
    voxel_composition,
)
from blochsimulator.ui.sequence_geometry_viewer import object_contrasts

MM = 1e-3


def _two_sphere_phantom():
    """Pyruvate sphere (r = 6 mm) at x = -8 mm, lactate (r = 4 mm) at +8 mm."""
    n = 20
    coordinate = ((np.arange(n) + 0.5) / n - 0.5) * 0.04
    x, y, z = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    left = (x + 0.008) ** 2 + y**2 + z**2 <= 0.006**2
    right = (x - 0.008) ** 2 + y**2 + z**2 <= 0.004**2
    species = [
        ChemicalSpecies(name="Pyruvate", chemical_shift_ppm=0.0, t1=30.0, t2=0.5),
        ChemicalSpecies(name="Lactate", chemical_shift_ppm=12.0, t1=30.0, t2=0.5),
    ]
    return SpectralPhantom(
        shape=(n, n, n),
        fov=(0.04, 0.04, 0.04),
        species=species,
        concentration_maps={"Pyruvate": left * 10.0, "Lactate": right * 10.0},
    )


def _species_contrast(phantom):
    return next(item for item in object_contrasts(phantom) if item.key == "species")


def _region_name(contrast, resampled, index):
    label = resampled.labels[int(resampled.values[index])]
    return contrast.categories[label][0]


def _planar_dataset(slice_offset_m=0.0):
    """A 16 x 16 axial image, 2.5 mm pixels, rows along y and columns along x."""
    count = 16
    positions = ((np.arange(count) + 0.5) / count - 0.5) * 0.04
    image = np.ones((count, count), dtype=complex)
    metadata = {
        "sequence_definitions": {
            "SliceThickness": 0.004,
            "SliceOffset": slice_offset_m,
            "FOV": [0.04, 0.04],
        }
    }
    return xr.Dataset(
        {
            "cartesian_image": (("phase_y", "read_x"), image),
            "cartesian_kspace": (("phase_y", "read_x"), image),
        },
        coords={
            "phase_y": np.arange(count),
            "read_x": np.arange(count),
            "cartesian_read_position_m": ("read_x", positions),
            "cartesian_phase_position_m": ("phase_y", positions),
        },
        attrs={"metadata_json": json.dumps(metadata)},
    )


def _volume_dataset():
    """A 5 x 4 x 3 (x, y, z) volume with 8 mm voxels centred on the phantom."""
    shape = (3, 4, 5)
    image = np.ones(shape, dtype=complex)
    return xr.Dataset(
        {
            "cartesian_3d_image": (("partition_z", "phase_y", "read_x"), image),
            "cartesian_3d_kspace": (("partition_z", "phase_y", "read_x"), image),
        },
        coords={
            "partition_z": np.arange(3),
            "phase_y": np.arange(4),
            "read_x": np.arange(5),
            "cartesian_k_partition_cyc_per_m": ("partition_z", [-42.0, 0.0, 42.0]),
            "cartesian_k_phase_cyc_per_m": ("phase_y", [-62.5, -31.25, 0.0, 31.25]),
            "cartesian_k_read_cyc_per_m": ("read_x", [-50.0, -25.0, 0.0, 25.0, 50.0]),
            "cartesian_x_position_m": ("read_x", np.arange(-2, 3) * 8 * MM),
            "cartesian_y_position_m": ("phase_y", (np.arange(4) - 1.5) * 8 * MM),
            "cartesian_z_position_m": ("partition_z", np.arange(-1, 2) * 8 * MM),
        },
        attrs={"cartesian_encoding_axes": "+x +y +z"},
    )


def test_planar_grid_follows_image_rows_columns_and_slice():
    phantom = _two_sphere_phantom()
    model = SequenceReconstructionModel(_planar_dataset(slice_offset_m=0.006))
    grid = planar_image_grid(model, ("phase_y", "read_x"), phantom)

    assert grid.shape == (16, 16, 1)
    assert not grid.projection
    # Rows step along y, columns along x, the 4 mm slab sits at z = 6 mm.
    assert grid.center((0, 0, 0)) == pytest.approx((-18.75 * MM, -18.75 * MM, 6 * MM))
    assert grid.center((1, 2, 0)) == pytest.approx((-13.75 * MM, -16.25 * MM, 6 * MM))
    assert abs(grid.steps_m[2][2]) == pytest.approx(4 * MM)
    assert "-0.0" not in grid_description(grid)
    assert "4 mm slab" in grid_description(grid)


def test_resampled_regions_contours_and_voxel_composition():
    phantom = _two_sphere_phantom()
    contrast = _species_contrast(phantom)
    grid = ImageGrid(
        origin_m=np.array([-18.75, -18.75, 0.0]) * MM,
        steps_m=np.diag([2.5, 2.5, 4.0]) * MM,
        shape=(16, 16, 1),
        planar=True,
    )
    resampled = resample_phantom(phantom, grid, contrast)
    up = resampled.upsampling
    # Pixel 4 is centred at -8.75 mm (pyruvate), pixel 11 at +8.75 mm (lactate).
    left = (4 * up[0] + up[0] // 2, 7 * up[1] + up[1] // 2)
    right = (11 * up[0] + up[0] // 2, 7 * up[1] + up[1] // 2)
    assert _region_name(contrast, resampled, left) == "Pyruvate"
    assert _region_name(contrast, resampled, right) == "Lactate"
    assert resampled.values[0, 0] == -1
    assert resampled.fraction[left] == pytest.approx(1.0)

    contours = contour_rgba(resampled)
    assert contours.shape == resampled.fraction.shape + (4,)
    assert contours[left][3] == 0 and contours[0, 0][3] == 0
    assert np.count_nonzero(contours[..., 3]) > 0

    inside = voxel_composition(phantom, grid.voxel_box((4, 7, 0)), [contrast], contrast)
    assert inside.object_fraction == pytest.approx(1.0)
    assert inside.regions[0][1][0][0] == "Pyruvate"
    assert "Pyruvate" in inside.html()
    outside = voxel_composition(
        phantom, grid.voxel_box((0, 0, 0)), [contrast], contrast
    )
    assert outside.object_fraction == 0.0


def test_volume_grid_sections_and_display():
    phantom = _two_sphere_phantom()
    contrast = _species_contrast(phantom)
    model = SequenceReconstructionModel(_volume_dataset())
    grid = volume_image_grid(model)
    volume, _fov = model.scanner_volume(model.select(model.image_name(), {}))

    assert grid.shape == volume.shape == (5, 4, 3)
    assert grid.center((1, 2, 1)) == pytest.approx((-8 * MM, 4 * MM, 0.0))
    values, levels, lookup = volume_display(resample_phantom(phantom, grid, contrast))
    assert values[0, 0, 0] == 0.0
    # Category rows: outside first, then the regions in their legend order.
    assert lookup.shape[0] == len(contrast.categories) + 1
    assert levels == (-0.5, len(contrast.categories) + 0.5)
    pyruvate = list(contrast.categories).index(
        next(
            label
            for label, item in contrast.categories.items()
            if item[0] == "Pyruvate"
        )
    )
    assert values[1, 2, 1] == pyruvate + 1

    section = volume_section_grid(grid, "xz", 2)
    assert section.planar and section.shape == (5, 3, 1)
    assert section.center((1, 1, 0)) == pytest.approx((-8 * MM, 4 * MM, 0.0))


def test_explorer_shows_phantom_next_to_planar_image(qt_application):
    phantom = _two_sphere_phantom()
    explorer = SequenceReconstructionExplorer()
    explorer.set_model(SequenceReconstructionModel(_planar_dataset()), phantom=phantom)

    assert not explorer.phantom_panel.isHidden()
    assert explorer.phantom_note.text() == ""
    explorer.phantom_panel.pixel_selected.emit(4, 8)
    assert explorer._voxel == (4, 8)
    assert "Pyruvate" in explorer.phantom_composition.text()
    assert "Dominant species" in explorer.phantom_legend.text()

    explorer.phantom_contours.setChecked(True)
    assert explorer.image_contours.image is not None
    explorer.phantom_contrast_combo.setCurrentIndex(
        explorer.phantom_contrast_combo.findData("density")
    )
    state = explorer.get_state()
    assert state["phantom"] == {"show": True, "contours": True, "contrast": "density"}

    explorer.phantom_show.setChecked(False)
    assert explorer.phantom_panel.isHidden()
    # The voxel summary stays available without the panel.
    assert "Selected voxel" in explorer.phantom_composition.text()

    restored = SequenceReconstructionExplorer()
    restored.set_model(SequenceReconstructionModel(_planar_dataset()), phantom=phantom)
    restored.restore_state(state)
    assert restored.phantom_contours.isChecked()
    assert restored.phantom_contrast_combo.currentData() == "density"

    restored.set_dataset(_planar_dataset())
    assert restored.phantom_panel.isHidden()
    assert not restored.phantom_show.isEnabled()
    assert "No phantom" in restored.phantom_note.text()


def test_explorer_phantom_volume_follows_reconstruction_indices(qt_application):
    phantom = _two_sphere_phantom()
    explorer = SequenceReconstructionExplorer()
    explorer.set_model(SequenceReconstructionModel(_volume_dataset()), phantom=phantom)

    tab = explorer.volume_pages.indexOf(explorer.phantom_volume)
    assert explorer.volume_pages.isTabVisible(tab)
    assert explorer.phantom_volume.data.shape == (5, 4, 3)

    explorer.image_volume.set_indices((1, 2, 1))
    assert explorer.phantom_volume.indices == (1, 2, 1)
    assert "(1, 2, 1)" in explorer.phantom_composition.text()
    assert "Pyruvate" in explorer.phantom_composition.text()
    explorer.phantom_volume.set_indices((3, 1, 0))
    assert explorer.image_volume.indices == (3, 1, 0)

    explorer.phantom_contours.setChecked(True)
    drawn = {
        (viewer is explorer.image_volume, plane): item.image is not None
        for viewer, plane, item in explorer._volume_contour_items
    }
    assert all(drawn.values())

    explorer.phantom_show.setChecked(False)
    assert not explorer.volume_pages.isTabVisible(tab)
