from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDialog, QHeaderView, QWidget
from unittest.mock import MagicMock, patch

from blochsimulator import BlochSimulator
from blochsimulator.phantom_design import (
    PhantomDesign,
    ShapeDefinition,
    SpectralPeakDefinition,
)
from blochsimulator.sequence import ADCEvent, RFEvent, SequenceProgram
from blochsimulator.spectral_phantom import SpectralPhantom
from blochsimulator.ui.phantom_designer import SpectralPhantomDesignerDialog
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget
from blochsimulator.ui.volume_viewer import VolumeViewerWidget
from blochsimulator.phantom_widget import PhantomCreatorWidget, PhantomViewerWidget
from blochsimulator.units import hz_to_ppm, ppm_to_hz


def _spectral_design():
    return PhantomDesign(
        name="Two peak phantom",
        shape=(6, 6, 4),
        fov_m=(0.06, 0.06, 0.004),
        shapes=[
            ShapeDefinition(
                name="Object",
                kind="box",
                center=(0.5, 0.5, 0.5),
                size=(0.6, 0.6, 1.0),
                t1_s=1.2,
                b0_ppm=float(hz_to_ppm(10.0, 3.0)),
                peaks=[
                    SpectralPeakDefinition(
                        "Water", 1.0, float(hz_to_ppm(-100.0, 3.0)), 0.020
                    ),
                    SpectralPeakDefinition(
                        "Metabolite", 0.25, float(hz_to_ppm(80.0, 3.0)), 0.010
                    ),
                ],
            )
        ],
    )


def test_shape_design_builds_lorentzian_components():
    phantom = _spectral_design().build()
    assert phantom.shape == (6, 6, 4)
    assert phantom.n_species == 2
    assert phantom.n_active > 0
    assert np.allclose(phantom.positions.mean(axis=0), 0.0)

    species = phantom.species[0]
    centre = phantom.get_frequency_offset(species.name) + float(
        phantom.get_b0_offset_map_hz()[3, 3, 2]
    )
    half_width = 1.0 / (2 * np.pi * species.t2_star)
    frequency = np.asarray([centre, centre + half_width])
    _, spectrum = phantom.spectrum_at((3, 3, 2), frequency_hz=frequency)
    # The distant second peak contributes slightly at both points.
    assert spectrum[0] == pytest.approx(1.0, rel=2e-3)
    assert spectrum[1] == pytest.approx(0.5, rel=1e-2)


def test_peak_specific_t1_overrides_shape_default_and_round_trips():
    design = _spectral_design()
    design.shapes[0].peaks[0].t1_s = 0.7
    design.shapes[0].peaks[1].t1_s = None

    phantom = design.build()
    restored = PhantomDesign.from_dict(design.to_dict())

    assert [species.t1 for species in phantom.species] == pytest.approx([0.7, 1.2])
    assert restored.shapes[0].peaks[0].t1_s == pytest.approx(0.7)
    assert restored.shapes[0].peaks[1].t1_s is None


def test_shape_design_builds_hyperpolarized_initial_mz_maps():
    design = PhantomDesign(
        name="Hyperpolarized voxel",
        shape=(2, 2, 2),
        fov_m=(0.02, 0.02, 0.02),
        shapes=[
            ShapeDefinition(
                name="HP",
                kind="box",
                size=(1.0, 1.0, 1.0),
                initial_mz=50.0,
                peaks=[SpectralPeakDefinition("Pyruvate", 2.0, 0.0, 0.020)],
            )
        ],
    )
    phantom = design.build()
    component_name, component = phantom.to_component_phantoms()[0]
    frequency = np.asarray([phantom.get_frequency_offset(component_name)])
    _, spectrum = phantom.spectrum_at((0, 0, 0), frequency_hz=frequency)

    assert phantom.concentration_maps[component_name][0, 0, 0] == pytest.approx(2.0)
    assert phantom.initial_mz_maps[component_name][0, 0, 0] == pytest.approx(50.0)
    assert component.m0_map[0, 0, 0, 2] == pytest.approx(50.0)
    assert component.pd_map[0, 0, 0] == pytest.approx(2.0)
    assert spectrum[0] == pytest.approx(100.0)


def test_spectral_ppm_offsets_are_converted_at_simulation_field_strength():
    phantom = _spectral_design().build()
    at_3t = phantom.to_component_phantoms(field_strength=3.0)[0][1]
    at_7t = phantom.to_component_phantoms(field_strength=7.0)[0][1]

    expected_ppm = phantom.species[0].chemical_shift_ppm
    assert at_3t.chemical_shift_map[3, 3, 2] == pytest.approx(
        ppm_to_hz(expected_ppm, 3.0)
    )
    assert at_7t.chemical_shift_map[3, 3, 2] == pytest.approx(
        ppm_to_hz(expected_ppm, 7.0)
    )
    assert at_7t.b0_map[3, 3, 2] / at_3t.b0_map[3, 3, 2] == pytest.approx(7.0 / 3.0)


def test_spectral_reference_keeps_simulation_offsets_centered_on_zero_ppm():
    design = PhantomDesign(
        name="C13 absolute ppm example",
        shape=(2, 2, 2),
        fov_m=(0.02, 0.02, 0.02),
        spectral_reference_ppm=175.0,
        spectral_bandwidth_ppm=12.0,
        spectral_points=129,
        shapes=[
            ShapeDefinition(
                name="Voxel",
                kind="box",
                peaks=[
                    SpectralPeakDefinition("Peak170", 1.0, 170.0 - 175.0, 0.02),
                    SpectralPeakDefinition("Peak180", 1.0, 180.0 - 175.0, 0.02),
                ],
            )
        ],
    )
    phantom = design.build()
    component_offsets = [
        component.chemical_shift_map[0, 0, 0]
        for _, component in phantom.to_component_phantoms(
            field_strength=3.0, nucleus="C13"
        )
    ]
    frequency, _ = phantom.spectrum_at((0, 0, 0))

    assert phantom.spectral_reference_ppm == 175.0
    assert component_offsets == pytest.approx(
        [ppm_to_hz(-5.0, 3.0, "C13"), ppm_to_hz(5.0, 3.0, "C13")]
    )
    assert frequency.size == 129
    assert frequency[0] == pytest.approx(ppm_to_hz(-6.0, 3.0))
    assert frequency[-1] == pytest.approx(ppm_to_hz(6.0, 3.0))


def test_spectral_preview_uses_absolute_ppm_by_default_and_optional_hz_conversion():
    design = PhantomDesign(
        name="C13 pyruvate lactate preview",
        shape=(2, 2, 2),
        fov_m=(0.02, 0.02, 0.02),
        spectral_reference_ppm=171.0,
        spectral_bandwidth_ppm=30.0,
        spectral_points=1024,
        shapes=[
            ShapeDefinition(
                name="Mixture",
                kind="box",
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 1.0, 171.0 - 171.0, 0.3),
                    SpectralPeakDefinition("Lactate", 1.0, 183.35 - 171.0, 0.3),
                ],
            )
        ],
    )
    phantom = design.build()
    ppm_axis, ppm_spectrum = phantom.spectrum_at_ppm((0, 0, 0))
    hz_axis, hz_spectrum = phantom.spectrum_at(
        (0, 0, 0),
        field_strength=7.0,
        nucleus="C13",
    )

    assert ppm_axis[0] == pytest.approx(156.0)
    assert ppm_axis[-1] == pytest.approx(186.0)
    assert ppm_axis[np.argmax(ppm_spectrum[: ppm_spectrum.size // 2])] == pytest.approx(
        171.0,
        abs=0.02,
    )
    assert ppm_axis[
        ppm_spectrum.size // 2 + np.argmax(ppm_spectrum[ppm_spectrum.size // 2 :])
    ] == pytest.approx(183.35, abs=0.02)
    assert hz_axis[hz_axis.size // 2] == pytest.approx(0.0, abs=2.0)
    assert ppm_to_hz(183.35 - 171.0, 7.0, "C13") == pytest.approx(925.0, rel=5e-3)
    assert hz_spectrum.max() > 0


@pytest.mark.parametrize(
    "mode,constant_axis",
    [("linear_x", 0), ("linear_y", 1), ("linear_z", 2), ("radial_xy", 2)],
)
def test_designer_builds_spatial_b0_inhomogeneity(mode, constant_axis):
    design = _spectral_design()
    design.b0_inhomogeneity_mode = mode
    design.b0_inhomogeneity_ppm = 2.0
    added = design.rasterize_b0_inhomogeneity()
    phantom = design.build()

    assert added.shape == design.shape
    assert np.ptp(added) > 0
    if mode.startswith("linear"):
        for axis in range(3):
            if axis != constant_axis:
                assert np.allclose(np.diff(added, axis=axis), 0.0)
    assert phantom.b0_map_ppm is not None
    assert PhantomDesign.from_phantom(phantom).b0_inhomogeneity_mode == mode


def test_spatial_b0_inhomogeneity_changes_the_simulated_adc_signal():
    reference_design = _spectral_design()
    inhomogeneous_design = _spectral_design()
    inhomogeneous_design.b0_inhomogeneity_mode = "linear_x"
    inhomogeneous_design.b0_inhomogeneity_ppm = 2.0
    program = SequenceProgram(
        events=(
            RFEvent(0.0, np.asarray([250.0]), 1e-3),
            ADCEvent(10e-3, 1, 100e-6),
        ),
        duration_s=11e-3,
    )
    simulator = BlochSimulator(use_parallel=False)

    reference = simulator.simulate_spectral_sequence(program, reference_design.build())
    inhomogeneous = simulator.simulate_spectral_sequence(
        program, inhomogeneous_design.build()
    )

    assert not np.allclose(inhomogeneous.signal, reference.signal)


def test_legacy_hz_design_metadata_is_migrated_using_saved_field_strength():
    design = PhantomDesign.from_dict(
        {
            "shape": (2, 2, 2),
            "fov_m": (0.02, 0.02, 0.02),
            "shapes": [
                {
                    "name": "Legacy",
                    "b0_hz": 10.0,
                    "peaks": [
                        {
                            "name": "Peak",
                            "amplitude": 1.0,
                            "frequency_hz": -100.0,
                            "t2_star_s": 0.02,
                        }
                    ],
                }
            ],
        },
        legacy_field_strength_t=3.0,
        legacy_nucleus="H1",
    )
    phantom = design.build()

    assert ppm_to_hz(design.shapes[0].b0_ppm, 3.0) == pytest.approx(10.0)
    assert ppm_to_hz(design.shapes[0].peaks[0].frequency_ppm, 3.0) == pytest.approx(
        -100.0
    )
    assert phantom.b0_map is None
    assert phantom.b0_map_ppm is not None


@pytest.mark.parametrize("suffix", [".npz", ".h5", ".nc"])
def test_spectral_phantom_round_trip_preserves_design(tmp_path, suffix):
    design = _spectral_design()
    design.shapes[0].initial_mz = 25.0
    phantom = design.build()
    path = tmp_path / f"spectral{suffix}"
    phantom.save(path)
    loaded = SpectralPhantom.load(path)

    assert loaded.shape == phantom.shape
    assert loaded.fov == phantom.fov
    assert [item.name for item in loaded.species] == [
        item.name for item in phantom.species
    ]
    assert PhantomDesign.from_phantom(loaded).to_dict() == design.to_dict()
    for name in phantom.concentration_maps:
        assert np.array_equal(
            loaded.concentration_maps[name], phantom.concentration_maps[name]
        )
        assert np.array_equal(
            loaded.initial_mz_maps[name], phantom.initial_mz_maps[name]
        )
    assert loaded.coordinate_system == "object_xyz"
    assert np.array_equal(loaded.affine_ijk_to_xyz_m, phantom.affine_ijk_to_xyz_m)


def test_spectral_phantom_xarray_dataset_labels_species_and_coordinates():
    phantom = _spectral_design().build()
    dataset = phantom.to_xarray()

    assert dataset["concentration"].dims == ("species", "x", "y", "z")
    assert dataset["initial_mz"].dims == ("species", "x", "y", "z")
    assert list(dataset.coords["species"].values) == [
        "Object: Water",
        "Object: Metabolite",
    ]
    assert dataset.attrs["coordinate_system"] == "object_xyz"
    assert dataset.attrs["affine_ijk_to_xyz_m"].shape == (16,)
    assert np.asarray(dataset.coords["x"]) == pytest.approx(
        [-0.025, -0.015, -0.005, 0.005, 0.015, 0.025]
    )


def test_spectral_sequence_signal_is_sum_of_independent_components():
    phantom = _spectral_design().build()
    program = SequenceProgram(
        events=(
            RFEvent(0.0, np.asarray([250.0]), 1e-3),
            ADCEvent(1.05e-3, 5, 100e-6),
        ),
        duration_s=1.5e-3,
    )
    simulator = BlochSimulator(use_parallel=False)
    result = simulator.simulate_spectral_sequence(program, phantom)
    component_results = [
        simulator.simulate_sequence(program, component)
        for _, component in phantom.to_component_phantoms()
    ]

    assert np.allclose(result.signal, sum(item.signal for item in component_results))
    assert result.final_magnetization.shape == phantom.shape + (3,)
    assert result.metadata["spectral_component_count"] == 2


def test_designer_and_sequence_workspace_accept_spectral_phantom():
    app = QApplication.instance() or QApplication([])
    dialog = SpectralPhantomDesignerDialog(design=_spectral_design())
    dialog._preview()
    assert dialog.phantom.n_species == 2
    assert dialog.inspector.volume.data.shape == dialog.phantom.shape
    dialog.inspector.volume._select_plane_coordinates("xy", 15.0, -5.0)
    assert dialog.inspector.volume.indices == (4, 2, 2)
    assert "Voxel (4, 2, 2)" in dialog.inspector.spectrum_info.text()
    marker_x, marker_y = dialog.inspector.volume.slice_markers["xy"].getData()
    assert marker_x[0] == pytest.approx(15.0)
    assert marker_y[0] == pytest.approx(-5.0)

    host = QWidget()
    host.phantom_widget = SimpleNamespace(current_phantom=dialog.phantom)
    widget = SequenceSimulationWidget(host)
    widget._build_phantom()
    assert widget.phantom is dialog.phantom
    dialog.close()
    host.close()
    app.processEvents()


def test_designer_supports_exact_numeric_xy_shape_placement():
    app = QApplication.instance() or QApplication([])
    dialog = SpectralPhantomDesignerDialog(design=_spectral_design())
    dialog.x_center.setValue(25.0)
    dialog.y_center.setValue(75.0)
    dialog.x_size.setValue(40.0)
    dialog.y_size.setValue(20.0)

    shape = dialog.design.shapes[0]
    assert shape.center[:2] == pytest.approx((0.25, 0.75))
    assert shape.size[:2] == pytest.approx((0.4, 0.2))
    assert "centre=(-15, 15) mm" in dialog.xy_info.text()
    dialog.close()
    app.processEvents()


def test_designer_uses_compact_shape_controls_and_wide_peak_columns():
    app = QApplication.instance() or QApplication([])
    dialog = SpectralPhantomDesignerDialog(design=_spectral_design())
    shape_panel = dialog.shape_splitter.widget(0)
    shape_button_grid = shape_panel.layout().itemAt(2).layout()

    assert shape_panel.maximumWidth() == 300
    assert shape_button_grid.rowCount() == 5
    assert shape_button_grid.columnCount() == 1
    assert (
        dialog.peak_table.horizontalHeader().sectionResizeMode(0) == QHeaderView.Stretch
    )
    assert (
        dialog.peak_table.horizontalHeader().sectionResizeMode(1)
        == QHeaderView.ResizeToContents
    )
    dialog.close()
    app.processEvents()


def test_designer_selects_shape_by_roi_and_highlights_current_shape():
    app = QApplication.instance() or QApplication([])
    design = _spectral_design()
    design.shapes.append(
        ShapeDefinition(
            name="Second",
            kind="box",
            center=(0.25, 0.25, 0.5),
            size=(0.2, 0.2, 1.0),
            initial_mz=10.0,
        )
    )
    dialog = SpectralPhantomDesignerDialog(design=design)

    dialog._select_roi(dialog._rois[1])

    assert dialog.shape_list.currentRow() == 1
    assert dialog.initial_mz.value() == pytest.approx(10.0)
    assert dialog._rois[1].pen.width() > dialog._rois[0].pen.width()
    dialog.close()
    app.processEvents()


def test_designer_creates_shape_from_mouse_drawn_bounds():
    app = QApplication.instance() or QApplication([])
    dialog = SpectralPhantomDesignerDialog(design=_spectral_design())
    dialog.show()
    app.processEvents()
    original_count = len(dialog.design.shapes)

    dialog._start_shape_drawing("box")
    assert dialog.canvas.drawing_kind == "box"
    view_box = dialog.canvas.plotItem.vb
    start = dialog.canvas.mapFromScene(view_box.mapViewToScene(QPointF(0.1, 0.2)))
    end = dialog.canvas.mapFromScene(view_box.mapViewToScene(QPointF(0.4, 0.6)))
    QTest.mousePress(dialog.canvas.viewport(), Qt.LeftButton, pos=start)
    QTest.mouseMove(dialog.canvas.viewport(), end, 20)
    QTest.mouseRelease(dialog.canvas.viewport(), Qt.LeftButton, pos=end)
    app.processEvents()

    shape = dialog.design.shapes[-1]
    assert len(dialog.design.shapes) == original_count + 1
    assert shape.kind == "box"
    assert shape.center == pytest.approx((0.25, 0.4, 0.5), abs=0.01)
    assert shape.size == pytest.approx((0.3, 0.4, 0.5), abs=0.01)
    assert dialog._rois[-1].pos().x() == pytest.approx(0.1, abs=0.01)
    assert dialog._rois[-1].pos().y() == pytest.approx(0.2, abs=0.01)
    dialog.close()
    app.processEvents()


def test_designer_displays_absolute_peak_ppm_but_stores_relative_offsets():
    app = QApplication.instance() or QApplication([])
    design = PhantomDesign(
        spectral_reference_ppm=175.0,
        shape=(2, 2, 2),
        fov_m=(0.02, 0.02, 0.02),
        shapes=[
            ShapeDefinition(
                name="C13",
                peaks=[
                    SpectralPeakDefinition("P170", 1.0, -5.0, 0.02),
                    SpectralPeakDefinition("P180", 1.0, 5.0, 0.02),
                ],
            )
        ],
    )
    dialog = SpectralPhantomDesignerDialog(design=design)

    assert float(dialog.peak_table.item(0, 2).text()) == pytest.approx(170.0)
    assert float(dialog.peak_table.item(1, 2).text()) == pytest.approx(180.0)
    assert [peak.frequency_ppm for peak in dialog._read_peak_table()] == pytest.approx(
        [-5.0, 5.0]
    )
    dialog.close()
    app.processEvents()


def test_designer_edits_optional_metabolite_t1_and_preserves_default_fallback():
    app = QApplication.instance() or QApplication([])
    design = _spectral_design()
    design.shapes[0].peaks[0].t1_s = 0.7
    design.shapes[0].peaks[1].t1_s = None
    dialog = SpectralPhantomDesignerDialog(design=design)

    assert float(dialog.peak_table.item(0, 3).text()) == pytest.approx(700.0)
    assert dialog.peak_table.item(1, 3).text() == ""
    peaks = dialog._read_peak_table()
    assert peaks[0].t1_s == pytest.approx(0.7)
    assert peaks[1].t1_s is None
    assert (
        "initial pool weight"
        in dialog.peak_table.horizontalHeaderItem(1).text().lower()
    )

    dialog.close()
    app.processEvents()


def test_phantom_creator_retains_spectral_designer_dialog_lifetime():
    app = QApplication.instance() or QApplication([])
    creator = PhantomCreatorWidget()
    phantom = _spectral_design().build()
    dialog = MagicMock()
    dialog.exec_.return_value = QDialog.Accepted
    dialog.get_phantom.return_value = phantom
    creator.type_combo.setCurrentText("Spectral Shape Designer...")
    assert not creator.resolution_spin.isEnabled()
    assert not creator.fov_spin.isEnabled()
    assert not creator.field_combo.isEnabled()

    with patch(
        "blochsimulator.phantom_widget.SpectralPhantomDesignerDialog",
        return_value=dialog,
    ):
        creator.create_phantom()

    assert creator.current_phantom is phantom
    assert creator._retained_spectral_designer_dialogs == [dialog]
    assert not creator.edit_btn.isHidden()

    edited_design = _spectral_design()
    edited_design.name = "Edited in memory"
    edited_phantom = edited_design.build()
    edit_dialog = MagicMock()
    edit_dialog.exec_.return_value = QDialog.Accepted
    edit_dialog.get_phantom.return_value = edited_phantom
    with patch(
        "blochsimulator.phantom_widget.SpectralPhantomDesignerDialog",
        return_value=edit_dialog,
    ) as designer_class:
        creator.edit_current_phantom()

    reopened_design = designer_class.call_args.kwargs["design"]
    assert reopened_design.to_dict() == _spectral_design().to_dict()
    assert creator.current_phantom.name == "Edited in memory"
    creator.close()
    app.processEvents()


def test_spectral_phantom_property_image_is_finite_and_fills_xy_view():
    app = QApplication.instance() or QApplication([])
    viewer = PhantomViewerWidget()
    phantom = _spectral_design().build()
    viewer.set_phantom(phantom)
    viewer.tabs.setCurrentIndex(0)
    viewer.prop_combo.setCurrentText("T1 Map")

    assert viewer.prop_image.image.shape == phantom.shape[:2]
    assert np.all(np.isfinite(viewer.prop_image.image))
    transform = viewer.prop_image.getImageItem().transform()
    assert transform.m11() == pytest.approx(10.0)
    assert transform.m22() == pytest.approx(10.0)
    assert "Range: 1200.00 - 1200.00 ms" in viewer.prop_info.text()
    assert viewer.prop_image.ui.histogram.maximumWidth() == 88
    assert set(viewer.property_overview_views) == {
        "T1 Map",
        "T2 Map",
        "Proton Density",
        "Off-resonance (dF)",
        "Mask",
    }
    for image_view in viewer.property_overview_views.values():
        assert image_view.image.shape == phantom.shape[:2]
        assert image_view.ui.histogram.maximumWidth() == 34
    viewer.close()
    app.processEvents()


def test_volume_viewer_normalizes_2d_mask_and_resets_stale_indices():
    app = QApplication.instance() or QApplication([])
    viewer = VolumeViewerWidget()

    viewer.set_volume(
        np.ones((64, 8, 4)),
        mask=np.ones((64, 8, 4), dtype=bool),
        fov_m=(0.064, 0.008, 0.004),
    )
    transform = viewer.xy_view.getImageItem().transform()
    assert transform.m11() == pytest.approx(1.0)
    assert transform.m22() == pytest.approx(1.0)
    assert "mm" in viewer.index_labels[0].text()

    viewer._select_plane_coordinates("xy", -31.5, 2.5)
    assert viewer.indices == (0, 6, 2)
    viewer._select_plane_coordinates("xz", 10.5, -1.5)
    assert viewer.indices == (42, 6, 0)
    viewer._select_plane_coordinates("yz", -3.5, 1.5)
    assert viewer.indices == (42, 0, 3)
    marker_x, marker_z = viewer.slice_markers["xz"].getData()
    assert marker_x[0] == pytest.approx(10.5)
    assert marker_z[0] == pytest.approx(1.5)

    viewer.sliders[0].setValue(48)

    data = np.arange(64.0)[None, :]
    viewer.set_volume(data, mask=np.ones(data.shape, dtype=bool))

    assert viewer.data.shape == (1, 64, 1)
    assert viewer.mask.shape == viewer.data.shape
    assert viewer.indices == (0, 32, 0)
    viewer._indices_updated()
    viewer.close()
    app.processEvents()


def test_volume_viewer_handles_fully_masked_and_nonfinite_slices():
    app = QApplication.instance() or QApplication([])
    viewer = VolumeViewerWidget()
    data = np.zeros((3, 3, 3), dtype=float)
    data[1, 1, 1] = 5.0
    data[2, 2, 2] = np.inf
    mask = np.zeros(data.shape, dtype=bool)
    mask[1, 1, 1] = True
    mask[2, 2, 2] = True

    viewer.set_volume(data, mask=mask)
    previous = viewer.sliders[0].blockSignals(True)
    viewer.sliders[0].setValue(0)
    viewer.sliders[0].blockSignals(previous)
    viewer._indices_updated()

    assert np.all(np.isfinite(viewer.xy_view.image))
    assert np.all(np.isfinite(viewer.xz_view.image))
    assert np.all(np.isfinite(viewer.yz_view.image))
    assert viewer.yz_view.getLevels() == pytest.approx((4.999995, 5.000005))
    viewer.close()
    app.processEvents()
