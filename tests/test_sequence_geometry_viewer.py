import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator.default_phantoms import is_default_phantom
from blochsimulator.mouse_phantom import (
    MOUSE_TISSUE_COLORS,
    MousePerfusionConfig,
    MousePerfusionPhantom,
)
from blochsimulator.phantom import Phantom
from blochsimulator.sequence.encoding import slice_package_positions
from blochsimulator.spectral_phantom import ChemicalSpecies, SpectralPhantom
from blochsimulator.ui.sequence_geometry_viewer import (
    OrientedBox,
    SequenceGeometry,
    SequenceGeometryViewer,
    automatic_contrast,
    configured_object_box,
    contrast_projections,
    convex_hull_2d,
    coverage_notes,
    object_contrasts,
    phantom_grid_box,
    projected_outline_mm,
    sequence_geometry_from_definitions,
)
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget


def test_projected_outline_of_rotated_box_is_its_convex_hull():
    angle = np.deg2rad(30.0)
    axes = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    box = OrientedBox((0.0, 0.0, 0.01), (0.1, 0.05, 0.02), axes)
    outline = projected_outline_mm(box, (0, 1))
    assert len(outline) == 5
    assert np.allclose(outline[0], outline[-1])
    # The XZ outline spans the rotated in-plane extent and the slab.
    xz = projected_outline_mm(box, (0, 2))
    width = 100.0 * np.cos(angle) + 50.0 * np.sin(angle)
    assert np.ptp(xz[:, 0]) == pytest.approx(width)
    assert xz[:, 1].min() == pytest.approx(0.0)
    assert xz[:, 1].max() == pytest.approx(20.0)
    assert len(convex_hull_2d(np.array([[0.0, 0.0], [1.0, 1.0]]))) == 3


def test_two_dimensional_geometry_spans_its_slice_package():
    geometry = sequence_geometry_from_definitions(
        {"SliceThickness": 0.004, "SlicePositions": [-0.01, 0.0, 0.01]},
        np.eye(3),
        fov_m=(0.2, 0.18),
        matrix=(64, 48),
    )
    assert geometry.is_2d
    fov_box = geometry.fov_box(0.3)
    assert fov_box.size_m == pytest.approx((0.2, 0.18, 0.024))
    slices = geometry.slice_boxes((0.3, 0.3, 0.3))
    assert [box.center_m[2] for box in slices] == pytest.approx([-0.01, 0.0, 0.01])
    assert slices[0].size_m == pytest.approx((0.2, 0.18, 0.004))

    unencoded = sequence_geometry_from_definitions(
        {"SliceThickness": 0.003, "SliceOffset": 0.02}, np.eye(3)
    )
    assert unencoded.fov_box(0.3) is None
    assert unencoded.slice_positions_m == (0.02,)
    assert unencoded.slice_boxes((0.3, 0.2, 0.1))[0].size_m == pytest.approx(
        (0.3, 0.2, 0.003)
    )


def test_coverage_notes_report_aliasing_and_truncated_mouse():
    mouse = MousePerfusionConfig(
        shape=(8, 8, 16), mouse_size_m=(0.03, 0.03, 0.12)
    ).build()
    grid = phantom_grid_box(mouse)
    assert grid.size_m == pytest.approx(mouse.config.fov_m)
    object_box = configured_object_box(mouse)
    assert object_box.size_m == pytest.approx((0.03, 0.03, 0.12))

    geometry = sequence_geometry_from_definitions(
        {}, np.eye(3), fov_m=(0.03, 0.04, 0.2), matrix=(8, 8, 8)
    )
    notes = coverage_notes(mouse, geometry, object_box)
    assert any("beyond the simulated grid along z" in note for note in notes)
    assert any(note.startswith("Read FOV 30 mm") for note in notes)
    assert not any(note.startswith("Phase FOV") for note in notes)
    assert any("Phantom cells" in note and "read pixel" in note for note in notes)


def test_read_phase_offsets_shift_the_fov_and_slices():
    frame = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    definitions = {
        "SliceThickness": 0.003,
        "SlicePositions": [-0.002, 0.002],
        "ReadOffset": 0.01,
        "PhaseOffset": -0.005,
    }
    geometry = sequence_geometry_from_definitions(
        definitions, frame, fov_m=(0.04, 0.03), matrix=(32, 24)
    )
    assert geometry.in_plane_center_m == pytest.approx((0.01, -0.005))
    shift = 0.01 * frame[:, 0] - 0.005 * frame[:, 1]
    assert geometry.fov_box(0.3).center_m == pytest.approx(shift)
    slices = geometry.slice_boxes((0.3, 0.3, 0.3))
    assert [box.center_m for box in slices] == [
        pytest.approx(shift + position * frame[:, 2]) for position in (-0.002, 0.002)
    ]
    assert geometry.center_text() == "centre read 10, phase -5 mm"

    centred = sequence_geometry_from_definitions(
        {**definitions, "ReadOffset": 0.0, "PhaseOffset": 0.0},
        frame,
        fov_m=(0.04, 0.03),
        matrix=(32, 24),
    )
    assert centred.center_text() == ""
    assert not centred.covers_same_region(geometry)


def test_coverage_notes_report_a_grid_beyond_the_offset_fov():
    phantom = Phantom(
        shape=(16, 16, 1),
        fov=(0.032, 0.032, 0.003),
        t1_map=np.ones((16, 16, 1)),
        t2_map=np.ones((16, 16, 1)),
        pd_map=np.ones((16, 16, 1)),
    )
    grid = phantom_grid_box(phantom)

    def notes(read_offset_m):
        geometry = sequence_geometry_from_definitions(
            {"SliceThickness": 0.003, "ReadOffset": read_offset_m},
            np.eye(3),
            fov_m=(0.04, 0.04),
            matrix=(16, 16),
        )
        return coverage_notes(phantom, geometry, grid)

    assert not any("offset" in note for note in notes(0.003))
    shifted = [note for note in notes(0.006) if "offset read FOV" in note]
    assert shifted == [
        "The simulated grid reaches 2 mm beyond the offset read FOV: signal "
        "outside the FOV aliases."
    ]


def test_sequence_workspace_uses_default_sphere_and_shows_geometry_tab():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    assert widget.views.tabText(widget.geometry_tab_index) == "Geometry"
    assert widget.geometry_tab_index == widget.sequence_tab_index + 1
    # The workspace opens on the geometry preview of the default FLASH.
    assert widget.views.currentIndex() == widget.geometry_tab_index
    app.processEvents()
    assert "Sequence FOV" in widget.geometry_viewer.summary.text()

    phantom = widget._selected_designed_phantom()
    assert is_default_phantom(phantom)
    assert phantom.nucleus == "H1"
    assert widget._selected_designed_phantom() is phantom
    rows = dict(widget._simulation_object_summary_rows())
    assert "automatic default" in rows["Selected phantom"]

    widget.nucleus.setCurrentText("C13")
    c13 = widget._selected_designed_phantom()
    assert c13.name == "Default sphere (C13 pyruvate)"
    assert c13.spectral_reference_ppm == pytest.approx(
        widget.sequence_reference_ppm.value()
    )

    widget.sequence_source.setCurrentIndex(widget.FLASH_SOURCE)
    widget.read_matrix_flash = None
    assert widget._reload_selected_generated_sequence()
    matrix, fov_m, _frame = widget._sequence_spatial_grid_for_resolution_warning()
    fitted = widget._selected_designed_phantom()
    assert max(fitted.fov) <= min(fov_m) + 1e-12
    assert (
        fitted.fov[0] / fitted.shape[0]
        <= min(np.asarray(fov_m) / np.asarray(matrix)) + 1e-12
    )
    widget.phantom = fitted
    assert widget._confirm_generated_sequence_fov()
    assert widget._confirm_phantom_resolution_for_run()

    widget.views.setCurrentIndex(widget.geometry_tab_index)
    app.processEvents()
    text = widget.geometry_viewer.summary.text()
    assert "Default sphere (C13 pyruvate)" in text
    assert "Sequence FOV" in text and "Slice" in text

    loaded = SimpleNamespace(name="Loaded")
    widget.window().phantom_widget = SimpleNamespace(current_phantom=loaded)
    assert widget._selected_designed_phantom() is loaded

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_geometry_tab_previews_sequence_parameter_changes():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_live_preview.setChecked(False)
    widget.sequence_source.setCurrentIndex(widget.FLASH_SOURCE)
    assert widget._reload_selected_generated_sequence()
    widget.views.setCurrentIndex(widget.geometry_tab_index)
    app.processEvents()
    notice = widget.geometry_viewer.stale_notice
    assert notice.isHidden()
    assert "Generated (dotted)" not in widget.geometry_viewer.summary.text()

    widget.flash_slice_offset_mm.setValue(4.0)
    app.processEvents()
    assert widget._sequence_generation_pending
    assert not notice.isHidden()
    label = widget.geometry_viewer.stale_notice_label.text()
    assert "changed" in label and "Generate the sequence now?" in label
    # The slice follows the form; the generated one stays as a reference.
    summary = widget.geometry_viewer.summary.text()
    assert "1 × 3 mm at 4 mm" in summary
    generated_row = summary[summary.index("Generated (dotted)") :]
    assert " at 0 mm" in generated_row

    widget.geometry_viewer.update_button.click()
    app.processEvents()
    assert not widget._sequence_generation_pending
    assert notice.isHidden()
    summary = widget.geometry_viewer.summary.text()
    assert " at 4 mm" in summary
    assert "Generated (dotted)" not in summary

    widget.flash_read_offset_mm.setValue(6.0)
    widget.flash_phase_offset_mm.setValue(-4.0)
    app.processEvents()
    summary = widget.geometry_viewer.summary.text()
    assert "centre read 6, phase -4 mm" in summary
    generated_row = summary[summary.index("Generated (dotted)") :]
    assert "centre read" not in generated_row

    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_slice_package_is_centred_on_its_offset():
    assert slice_package_positions(1, 0.003) == pytest.approx([0.0])
    assert slice_package_positions(3, 0.004, 0.001, 0.002) == pytest.approx(
        [-0.003, 0.002, 0.007]
    )
    assert slice_package_positions(2, 0.002, 0.0, -0.01) == pytest.approx(
        [-0.011, -0.009]
    )


def _shift_slice_package(widget, prefix):
    getattr(widget, f"{prefix}_slice_count").setValue(3)
    getattr(widget, f"{prefix}_slice_gap_mm").setValue(1.0)
    getattr(widget, f"{prefix}_slice_offset_mm").setValue(5.0)


def test_planned_geometry_matches_the_generated_sequence():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_live_preview.setChecked(False)
    cases = {
        "EPI": (widget.EPI_SOURCE, "_epi_pulseq_parameters"),
        "Spiral": (widget.EPI_SOURCE, "_spiral_pulseq_parameters"),
        "CSI": (widget.CSI_SOURCE, "_csi_pulseq_parameters"),
        "bSSFP": (widget.BSSFP_SOURCE, "_bssfp_pulseq_parameters"),
        "Radial": (widget.RADIAL_ME_BSSFP_SOURCE, "_radial_me_bssfp_pulseq_parameters"),
        "ME-bSSFP": (widget.ME_BSSFP_SOURCE, "_me_bssfp_pulseq_parameters"),
        "FLASH": (widget.FLASH_SOURCE, "_flash_pulseq_parameters"),
        "EPSI": (widget.EPSI_MGE_SOURCE, "_epsi_mge_pulseq_parameters"),
        "Slice": (widget.SLICE_SELECTION_SOURCE, "_slice_selection_pulseq_parameters"),
    }
    _shift_slice_package(widget, "epi")
    _shift_slice_package(widget, "epsi")
    _shift_slice_package(widget, "flash")
    widget.flash_read_offset_mm.setValue(6.0)
    widget.flash_phase_offset_mm.setValue(-4.0)
    widget.csi_slice_offset_mm.setValue(-6.0)
    widget.bssfp_phase_matrix.setValue(6)
    widget.slice_selection_offset_mm.setValue(7.0)
    widget.slice_selection_axis.setCurrentText("-Y")
    for label, (source, builder) in cases.items():
        widget.epi_readout_trajectory.setCurrentText(
            "Spiral" if label == "Spiral" else "Cartesian EPI"
        )
        widget.sequence_source.setCurrentIndex(source)
        widget._sequence_generation_pending = True
        planned = widget._planned_sequence_geometry()
        assert planned is not None, label
        # The Pulseq builders receive the geometry entries unchanged.
        parameters = getattr(widget, builder)()
        for key, value in widget._sequence_geometry_parameters(source).items():
            assert parameters[key] == value, (label, key)

        assert widget._reload_selected_generated_sequence(), label
        assert widget._planned_sequence_geometry() is None, label
        generated = widget._sequence_geometry()
        assert planned.covers_same_region(generated), label
        assert planned.matrix == generated.matrix, label
        if label == "FLASH":
            assert generated.in_plane_center_m == pytest.approx((0.006, -0.004))
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_sequence_overlay_update_keeps_contrast_and_marks_generated_outline():
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = SequenceGeometryViewer()
    generated = SequenceGeometry(
        frame=np.eye(3),
        fov_m=(0.03, 0.03),
        matrix=(16, 16),
        slice_positions_m=(0.0,),
        slice_thickness_m=0.003,
    )
    viewer.update_geometry(_two_sphere_phantom(), generated, object_label="Spheres")
    combo = viewer.contrast_combo
    combo.setCurrentIndex(combo.findData("concentration:Lactate"))
    app.processEvents()

    same = SequenceGeometry(
        frame=np.eye(3),
        fov_m=(0.03, 0.03),
        matrix=(32, 32),
        slice_positions_m=(0.0,),
        slice_thickness_m=0.003,
    )
    viewer.set_sequence_geometry(same, generated_sequence=generated)
    assert "Generated (dotted)" not in viewer.summary.text()

    moved = SequenceGeometry(
        frame=np.eye(3),
        fov_m=(0.03, 0.03),
        matrix=(16, 16),
        slice_positions_m=(0.004,),
        slice_thickness_m=0.003,
    )
    viewer.set_sequence_geometry(moved, generated_sequence=generated)
    summary = viewer.summary.text()
    assert "1 × 3 mm at 4 mm" in summary
    assert "Generated (dotted)" in summary
    assert viewer.contrast.key == "concentration:Lactate"
    assert "Lactate concentration" in summary

    viewer.set_sequence_geometry(None, generated_sequence=generated)
    assert "Generated (dotted)" not in viewer.summary.text()
    viewer.deleteLater()
    app.processEvents()


def _two_sphere_phantom(lactate_scale=1.0):
    n = 20
    coordinate = (np.arange(n) + 0.5) / n - 0.5
    x, y, z = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    left = (x + 0.2) ** 2 + y**2 + z**2 <= 0.15**2
    right = (x - 0.2) ** 2 + y**2 + z**2 <= 0.1**2
    species = [
        ChemicalSpecies(name="Pyruvate", chemical_shift_ppm=0.0, t1=30.0, t2=0.5),
        ChemicalSpecies(name="Lactate", chemical_shift_ppm=12.0, t1=30.0, t2=0.5),
    ]
    return SpectralPhantom(
        shape=(n, n, n),
        fov=(0.04, 0.04, 0.04),
        species=species,
        concentration_maps={
            "Pyruvate": left * 10.0,
            "Lactate": right * 10.0 * lactate_scale,
        },
    )


def test_anatomy_labels_drive_the_automatic_mouse_contrast():
    mouse = MousePerfusionPhantom.from_config(MousePerfusionConfig(shape=(16, 16, 24)))
    contrasts = object_contrasts(mouse)
    contrast = automatic_contrast(contrasts)
    assert contrast.key == "anatomy"
    assert 0 not in contrast.categories
    names = [name for name, _color in contrast.categories.values()]
    assert names[0] == "Soft tissue"  # Largest region, painted first.
    assert "Liver" in names
    # kPL stays selectable and differs between organs and body.
    assert any(item.key == "kpl" and item.score > 0.5 for item in contrasts)

    projections, shown_range = contrast_projections(mouse, contrast)
    assert shown_range is None
    xy, _rect = projections[0]
    assert xy.dtype == np.uint8 and xy.shape == (16, 16, 4)
    colors = {tuple(pixel[:3]) for pixel in xy.reshape(-1, 4) if pixel[3]}
    assert MOUSE_TISSUE_COLORS[6] in colors  # Liver
    assert set(np.unique(xy[..., 3])) <= {0, 255}


def test_species_regions_or_widest_map_set_the_contrast():
    # Two compartments of different species show as species regions.
    two_species = automatic_contrast(object_contrasts(_two_sphere_phantom()))
    assert two_species.key == "species"
    assert [name for name, _ in two_species.categories.values()] == [
        "Pyruvate",
        "Lactate",
    ]

    # Without a second region, the most varying map is shown instead.
    phantom = _two_sphere_phantom(lactate_scale=0.0)
    phantom.concentration_maps["Pyruvate"][
        phantom.concentration_maps["Pyruvate"] > 0
    ] = 10.0
    phantom.concentration_maps["Pyruvate"][:, :, :8] *= 0.5
    contrasts = object_contrasts(phantom)
    assert "species" not in {item.key for item in contrasts}
    contrast = automatic_contrast(contrasts)
    assert contrast.key == "concentration"
    assert contrast.value_range == pytest.approx((5.0, 10.0))
    projections, shown_range = contrast_projections(phantom, contrast)
    assert 5.0 <= shown_range[0] < shown_range[1] <= 10.0
    xz, _rect = projections[1]
    assert np.any(xz[..., 3] == 0) and np.any(xz[..., 3] == 255)

    # Relaxation maps of a plain phantom work the same way.
    shape = (8, 8, 8)
    t1 = np.full(shape, 1.0)
    t1[:4] = 0.5
    plain = Phantom(shape=shape, fov=(0.02,) * 3, t1_map=t1, t2_map=np.full(shape, 0.1))
    assert automatic_contrast(object_contrasts(plain)).key == "t1"

    # Uniform objects keep the plain density projection.
    uniform = Phantom(
        shape=shape, fov=(0.02,) * 3, t1_map=np.ones(shape), t2_map=np.ones(shape)
    )
    assert automatic_contrast(object_contrasts(uniform)) is None


def test_geometry_viewer_contrast_selector_overrides_auto():
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = SequenceGeometryViewer()
    combo = viewer.contrast_combo
    viewer.update_geometry(None, None)
    assert not combo.isEnabled()
    assert combo.itemText(0) == "Auto: object density"

    viewer.update_geometry(_two_sphere_phantom(), None, object_label="Spheres")
    assert combo.isEnabled()
    assert combo.itemText(0) == "Auto: Dominant species"
    assert viewer.contrast.key == "species"
    assert "Pyruvate" in viewer.summary.text()
    combo.setCurrentIndex(combo.findData("concentration:Lactate"))
    app.processEvents()
    assert viewer.contrast.key == "concentration:Lactate"
    assert "Lactate concentration" in viewer.summary.text()
    assert viewer.images[0].image.shape[-1] == 4

    # A redraw keeps the chosen property while the object still has it.
    viewer.update_geometry(_two_sphere_phantom(), None, object_label="Spheres")
    assert viewer.contrast.key == "concentration:Lactate"
    viewer.deleteLater()
    app.processEvents()
