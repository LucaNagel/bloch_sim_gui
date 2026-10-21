import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator.default_phantoms import is_default_phantom
from blochsimulator.mouse_phantom import MousePerfusionConfig
from blochsimulator.ui.sequence_geometry_viewer import (
    OrientedBox,
    configured_object_box,
    convex_hull_2d,
    coverage_notes,
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
