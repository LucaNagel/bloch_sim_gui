import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtWidgets import QSlider

from blochsimulator.sequence import AcquisitionDimensions, SequenceSimulationResult
from blochsimulator.sequence.reconstruction import (
    SequenceReconstructionModel,
    ideal_separate,
)
from blochsimulator.ui.reconstruction_explorer import SequenceReconstructionExplorer
from blochsimulator.ui.simulation_explorer import SessionSimulationRun


def _session_run(dataset, run_id, name):
    return SessionSimulationRun(
        run_id=run_id,
        created_at_utc="2026-10-11T10:00:00+00:00",
        sequence_name="test-sequence",
        phantom_name="test-phantom",
        phantom_shape=(2, 2, 1),
        sequence_duration_s=0.01,
        adc_samples=4,
        runtime_s=0.1,
        kernel="reference",
        result=SimpleNamespace(metadata={}, to_xarray=lambda: dataset),
        display_name=name,
    )


def test_explorer_selects_retained_runs_with_different_reconstruction_geometry(
    qt_application,
):
    explorer = SequenceReconstructionExplorer()
    first = _session_run(_framed_cartesian_dataset(), "first", "Loop 1/2 · flip = 30°")
    second = _session_run(_cartesian_3d_dataset(), "second", "Loop 2/2 · flip = 60°")
    for run in (first, second):
        explorer.set_result(run.result)
        explorer.add_session_run(run)

    assert explorer.simulation_combo.count() == 2
    assert explorer.simulation_combo.currentData() == second.run_id
    assert explorer.model.kind == "cartesian_3d"
    explorer.simulation_combo.setCurrentIndex(0)
    assert explorer.model.kind == "cartesian_2d"
    assert explorer._result is first.result
    assert explorer.pages.currentWidget() is explorer.two_d_page
    np.testing.assert_allclose(
        explorer._current_display,
        np.abs(_framed_cartesian_dataset().cartesian_image[0]),
    )
    explorer.simulation_combo.setCurrentIndex(1)
    assert explorer._result is second.result
    assert explorer.pages.currentWidget() is explorer.three_d_page
    explorer.set_result(first.result)
    assert explorer.simulation_combo.currentData() == first.run_id


def test_explorer_run_list_survives_clear_and_tracks_rename_and_delete(qt_application):
    explorer = SequenceReconstructionExplorer()
    first = _session_run(_framed_cartesian_dataset(), "first", "First run")
    second = _session_run(_cartesian_3d_dataset(), "second", "Second run")
    explorer.add_session_run(first)
    explorer.add_session_run(second)
    explorer.simulation_combo.setCurrentIndex(0)

    first.display_name = "Baseline"
    explorer.add_session_run(first)
    assert explorer.simulation_combo.count() == 2
    assert explorer.simulation_combo.currentText() == "Baseline"
    explorer.remove_session_run(second)
    assert explorer._result is first.result
    assert explorer.simulation_combo.currentData() == first.run_id

    explorer.clear("Next simulation running")
    assert explorer.simulation_combo.currentIndex() == -1
    assert explorer.simulation_combo.count() == 1
    explorer.simulation_combo.setCurrentIndex(0)
    assert explorer._result is first.result

    explorer.set_dataset(_cartesian_3d_dataset(), source="imported.nc")
    assert explorer.simulation_combo.currentIndex() == -1
    explorer.simulation_combo.setCurrentIndex(0)
    assert explorer._result is first.result
    explorer.remove_session_run(first)
    assert explorer.model is None
    assert explorer._result is None
    assert not explorer.simulation_combo.isEnabled()
    assert not explorer._session_runs


def _framed_cartesian_dataset():
    frames, ny, nx = 4, 3, 5
    image = np.arange(frames * ny * nx, dtype=float).reshape(frames, ny, nx) + 1j
    kspace = np.fft.fft2(image, axes=(-2, -1))
    return xr.Dataset(
        {
            "cartesian_image": (
                ("cartesian_frame", "phase_y", "read_x"),
                image,
            ),
            "cartesian_kspace": (
                ("cartesian_frame", "phase_y", "read_x"),
                kspace,
            ),
            "species_cartesian_image": (
                ("cartesian_frame", "pool", "phase_y", "read_x"),
                np.stack((image, 2.0 * image), axis=1),
            ),
            "species_cartesian_kspace": (
                ("cartesian_frame", "pool", "phase_y", "read_x"),
                np.stack((kspace, 2.0 * kspace), axis=1),
            ),
        },
        coords={
            "cartesian_frame": np.arange(frames),
            "phase_y": np.arange(ny),
            "read_x": np.arange(nx),
            "cartesian_frame_echo_index": (
                "cartesian_frame",
                [0, 1, 0, 1],
            ),
            "cartesian_frame_repetition_index": (
                "cartesian_frame",
                [0, 0, 1, 1],
            ),
            "pool": ["Pyruvate", "Lactate"],
        },
        attrs={
            "echo_times_s": "0.001,0.002",
            "pool_frequency_offsets_hz": "0,100",
        },
    )


def _sliced_cartesian_dataset():
    slices, ny, nx = 3, 4, 5
    image = np.arange(slices * ny * nx, dtype=float).reshape(slices, ny, nx)
    kspace = np.fft.fft2(image, axes=(-2, -1))
    return xr.Dataset(
        {
            "cartesian_image": (
                ("cartesian_frame", "phase_y", "read_x"),
                image,
            ),
            "cartesian_kspace": (
                ("cartesian_frame", "phase_y", "read_x"),
                kspace,
            ),
        },
        coords={
            "cartesian_frame": np.arange(slices),
            "phase_y": np.arange(ny),
            "read_x": np.arange(nx),
            "cartesian_frame_slice_index": (
                "cartesian_frame",
                np.arange(slices),
            ),
        },
    )


def _cartesian_3d_dataset():
    shape = (3, 4, 5)
    image = np.arange(np.prod(shape), dtype=float).reshape(shape).astype(complex)
    return xr.Dataset(
        {
            "cartesian_3d_image": (
                ("partition_z", "phase_y", "read_x"),
                image,
            ),
            "cartesian_3d_kspace": (
                ("partition_z", "phase_y", "read_x"),
                np.fft.fftn(image),
            ),
        },
        coords={
            "partition_z": np.arange(shape[0]),
            "phase_y": np.arange(shape[1]),
            "read_x": np.arange(shape[2]),
            "cartesian_k_partition_cyc_per_m": (
                "partition_z",
                [-10.0, 0.0, 10.0],
            ),
            "cartesian_k_phase_cyc_per_m": (
                "phase_y",
                [-15.0, -5.0, 5.0, 15.0],
            ),
            "cartesian_k_read_cyc_per_m": (
                "read_x",
                [-20.0, -10.0, 0.0, 10.0, 20.0],
            ),
        },
        attrs={"cartesian_encoding_axes": "+x +y +z"},
    )


def test_known_frequency_ideal_separates_complex_echo_images():
    echo_times = np.array([0.0, 1e-3, 2e-3])
    offsets = np.array([0.0, 125.0])
    species = np.array(
        [
            [[1.0 + 0.25j, 2.0 - 0.5j]],
            [[0.4 - 0.1j, 0.75 + 0.2j]],
        ]
    )
    encoding = np.exp(2j * np.pi * echo_times[:, None] * offsets[None, :])
    echoes = np.einsum("es,syx->eyx", encoding, species)

    separated = ideal_separate(echoes, echo_times, offsets)

    np.testing.assert_allclose(separated, species, atol=1e-12)


def test_model_exposes_virtual_echo_and_repetition_dimensions():
    model = SequenceReconstructionModel(_framed_cartesian_dataset())

    assert model.kind == "cartesian_2d"
    assert [item.name for item in model.outer_dimensions] == ["echo", "repetition"]
    assert model.ideal_species_names == ("Pyruvate", "Lactate")
    selected = model.select(
        model.image_name(pool=True),
        {"echo": 1, "repetition": 1},
        pool_index=1,
    )
    expected = _framed_cartesian_dataset().species_cartesian_image.isel(
        cartesian_frame=3, pool=1
    )
    np.testing.assert_allclose(selected, expected)


def test_explorer_restores_multidimensional_selection(qt_application):
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(_framed_cartesian_dataset(), source="example.nc")

    assert isinstance(explorer.outer_controls["echo"], QSlider)
    assert isinstance(explorer.outer_controls["repetition"], QSlider)

    explorer.restore_state(
        {
            "outer": {"echo": 1, "repetition": 1},
            "pool": ["pool", 1],
            "coil": "rss",
            "component": "real",
            "voxel": [2, 1],
        }
    )

    state = explorer.get_state()
    assert state["outer"] == {"echo": 1, "repetition": 1}
    assert state["pool"] == ["pool", 1]
    assert state["component"] == "real"
    assert state["voxel"] == [2, 1]
    assert explorer._current_display.shape == (3, 5)


def test_explorer_applies_and_restores_image_display_options(qt_application):
    explorer = SequenceReconstructionExplorer()
    dataset = _framed_cartesian_dataset()
    explorer.set_dataset(dataset)

    selected_kspace = dataset.cartesian_kspace.isel(cartesian_frame=0)
    assert explorer.kspace_scale_combo.currentData() == "log"
    np.testing.assert_allclose(
        explorer.kspace_panel.image.image,
        np.log1p(np.abs(selected_kspace)).T,
    )

    explorer.colormap_combo.setCurrentText("magma")
    explorer.display_strength.setValue(1.8)
    explorer._set_combo_data(explorer.interpolation_combo, "cubic")
    explorer._set_combo_data(explorer.kspace_scale_combo, "magnitude")

    state = explorer.get_state()
    assert state["colormap"] == "magma"
    assert state["display_strength"] == 1.8
    assert state["interpolation"] == "cubic"
    assert state["kspace_scale"] == "magnitude"
    np.testing.assert_allclose(
        explorer.kspace_panel.image.image,
        np.abs(selected_kspace).T,
    )
    assert explorer.kspace_panel.info.text().startswith("|gridded k-space|")
    # Cubic interpolation changes display pixels, not the reconstruction array.
    assert explorer.image_panel.image.image.shape == (40, 24)
    assert explorer._current_display.shape == (3, 5)

    restored = SequenceReconstructionExplorer()
    restored.set_dataset(_framed_cartesian_dataset())
    restored.restore_state(state)

    assert restored.colormap_combo.currentText() == "magma"
    assert restored.display_strength.value() == 1.8
    assert restored.interpolation_combo.currentData() == "cubic"
    assert restored.kspace_scale_combo.currentData() == "magnitude"
    np.testing.assert_array_equal(
        restored.image_panel.image.lut, explorer.image_panel.image.lut
    )


def test_explorer_uses_two_handle_contrast_slider(qt_application):
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(_framed_cartesian_dataset())

    full_low, full_high = explorer.contrast_slider.values()
    selected = (
        full_low + 0.2 * (full_high - full_low),
        full_low + 0.8 * (full_high - full_low),
    )
    explorer.auto_contrast.setChecked(False)
    explorer.contrast_slider.set_values(*selected)

    assert explorer.contrast_slider.isEnabled()
    assert explorer.get_state()["contrast_range"] == list(selected)
    assert explorer.contrast_min_label.text().startswith("Minimum ")
    assert explorer.contrast_max_label.text().startswith("Maximum ")
    np.testing.assert_allclose(explorer.image_panel.image.levels, selected)

    restored = SequenceReconstructionExplorer()
    restored.set_dataset(_framed_cartesian_dataset())
    restored.restore_state(explorer.get_state())

    assert not restored.auto_contrast.isChecked()
    np.testing.assert_allclose(restored.contrast_slider.values(), selected)


def test_auto_contrast_tracks_minimum_and_maximum_of_selected_image(qt_application):
    dataset = _framed_cartesian_dataset()
    explorer = SequenceReconstructionExplorer()
    explorer.auto_contrast.setChecked(True)
    explorer.set_dataset(dataset)

    first_image = np.abs(dataset.cartesian_image.isel(cartesian_frame=0))
    expected = (first_image.min().item(), first_image.max().item())
    np.testing.assert_allclose(explorer.contrast_slider.values(), expected)
    np.testing.assert_allclose(explorer.image_panel.image.levels, expected)

    explorer.outer_controls["repetition"].setValue(1)

    selected_image = np.abs(dataset.cartesian_image.isel(cartesian_frame=2))
    expected = (selected_image.min().item(), selected_image.max().item())
    np.testing.assert_allclose(explorer.contrast_slider.values(), expected)
    np.testing.assert_allclose(explorer.image_panel.image.levels, expected)


def test_auto_contrast_includes_negative_values_of_selected_component(
    qt_application,
):
    dataset = _framed_cartesian_dataset()
    explorer = SequenceReconstructionExplorer()
    explorer.auto_contrast.setChecked(True)
    explorer.set_dataset(dataset)

    explorer._set_combo_data(explorer.component_combo, "real")
    explorer.outer_controls["repetition"].setValue(1)

    selected_image = dataset.cartesian_image.real.isel(cartesian_frame=2)
    expected = (selected_image.min().item(), selected_image.max().item())
    np.testing.assert_allclose(explorer.contrast_slider.values(), expected)
    np.testing.assert_allclose(explorer.image_panel.image.levels, expected)


def test_manual_contrast_is_default_and_starts_at_the_full_dataset_range(
    qt_application,
):
    dataset = _framed_cartesian_dataset()
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(dataset)

    assert not explorer.auto_contrast.isChecked()
    assert explorer.contrast_slider.isEnabled()
    high = 1.1 * np.abs(dataset.cartesian_image).max().item()
    np.testing.assert_allclose(explorer.contrast_slider.values(), (0.0, high))

    # A much weaker result does not collapse the kept range to one value.
    weak = dataset.copy(deep=True)
    weak["cartesian_image"] = weak.cartesian_image * 1e-6
    explorer.set_dataset(weak)
    np.testing.assert_allclose(explorer.contrast_slider.values(), (0.0, 1e-6 * high))


def test_manual_contrast_survives_navigation_from_strong_to_weak_echo(
    qt_application,
):
    dataset = _framed_cartesian_dataset()
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(dataset)
    echo_control = explorer.outer_controls["echo"]

    echo_control.setValue(1)
    explorer.auto_contrast.setChecked(False)
    explorer.contrast_slider.set_values(50.0, 60.0)

    echo_control.setValue(0)

    expected_domain = (0.0, 1.1 * np.abs(dataset.cartesian_image).max().item())
    assert explorer.contrast_slider._domain == expected_domain
    assert explorer.contrast_slider.values() == (50.0, 60.0)
    np.testing.assert_allclose(explorer.image_panel.image.levels, (50.0, 60.0))


def test_explorer_uses_labelled_slider_for_2d_slice_series(qt_application):
    dataset = _sliced_cartesian_dataset()
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(dataset)

    slice_control = explorer.outer_controls["slice"]
    assert isinstance(slice_control, QSlider)
    assert slice_control.minimum() == 0
    assert slice_control.maximum() == 2

    slice_control.setValue(2)

    assert explorer.get_state()["outer"]["slice"] == 2
    assert explorer.outer_value_labels["slice"].text() == "2  (3/3)"
    np.testing.assert_allclose(
        explorer._current_display,
        np.abs(dataset.cartesian_image.isel(cartesian_frame=2)),
    )


def test_wheel_over_2d_image_steps_the_slice_slider_instead_of_zooming(
    qt_application,
):
    dataset = _sliced_cartesian_dataset()
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(dataset)
    slice_control = explorer.outer_controls["slice"]
    view_range = explorer.image_panel.plot.getViewBox().viewRange()

    class _Wheel:
        def __init__(self, delta, modifiers=0):
            self._delta, self._modifiers = delta, modifiers

        def modifiers(self):
            return self._modifiers

        def delta(self):
            return self._delta

        def accept(self):
            pass

        def scenePos(self):
            from PyQt5.QtCore import QPointF

            return QPointF(0, 0)

    box = explorer.image_panel.plot.getViewBox()
    box.wheelEvent(_Wheel(120))
    assert slice_control.value() == 1
    box.wheelEvent(_Wheel(-120))
    box.wheelEvent(_Wheel(-120))
    assert slice_control.value() == 0
    assert explorer.image_panel.plot.getViewBox().viewRange() == view_range

    # A short two-finger swipe sends many small deltas and must not race
    # through every slice: it takes a long travel per slice.
    for _ in range(10):
        box.wheelEvent(_Wheel(8))
    assert slice_control.value() == 0
    for _ in range(40):
        box.wheelEvent(_Wheel(8))
    assert slice_control.value() == 1


def test_time_course_never_shows_negative_repetitions(qt_application):
    dataset = _framed_cartesian_dataset().assign_coords(
        cartesian_frame_first_adc_time_s=("cartesian_frame", [0.0, 0.01, 1.0, 1.01])
    )
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(dataset)
    explorer.voxel_plot_combo.setCurrentIndex(
        explorer.voxel_plot_combo.findData("time_course")
    )
    explorer._voxel_selected(4, 2)

    assert explorer.spectrum_plot.getViewBox().state["limits"]["xLimits"][0] == 0.0


def test_manual_contrast_domain_covers_all_slices_with_headroom_and_survives_navigation(
    qt_application,
):
    dataset = _sliced_cartesian_dataset()
    explorer = SequenceReconstructionExplorer()
    explorer.auto_contrast.setChecked(True)
    explorer.set_dataset(dataset)
    slice_control = explorer.outer_controls["slice"]

    assert explorer.contrast_slider.values() == (0.0, 19.0)
    explorer.auto_contrast.setChecked(False)
    explorer.contrast_slider.set_values(5.0, 55.0)

    slice_control.setValue(0)
    assert explorer.contrast_slider._domain == (0.0, 64.9)
    assert explorer.contrast_slider.values() == (5.0, 55.0)
    slice_control.setValue(2)
    assert explorer.contrast_slider._domain == (0.0, 64.9)
    assert explorer.contrast_slider.values() == (5.0, 55.0)


def test_explorer_keeps_3d_image_and_kspace_slice_positions_independent(
    qt_application,
):
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(_cartesian_3d_dataset())

    # Image, k-space and a phantom tab that stays hidden without a phantom.
    assert explorer.volume_pages.count() == 3
    assert not explorer.volume_pages.isTabVisible(
        explorer.volume_pages.indexOf(explorer.phantom_volume)
    )
    explorer._set_volume_indices(explorer.image_volume, (0, 1, 2))
    explorer._set_volume_indices(explorer.kspace_volume, (4, 3, 0))
    explorer.image_volume.sliders[0].setValue(1)
    explorer.component_combo.setCurrentIndex(2)
    explorer.volume_pages.setCurrentIndex(1)

    assert explorer.image_volume.indices == (1, 1, 2)
    assert explorer.kspace_volume.indices == (4, 3, 0)
    state = explorer.get_state()
    assert state["image_volume_indices"] == [1, 1, 2]
    assert state["kspace_volume_indices"] == [4, 3, 0]
    assert state["volume_page"] == 1

    restored = SequenceReconstructionExplorer()
    restored.set_dataset(_cartesian_3d_dataset())
    restored.restore_state(state)

    assert restored.image_volume.indices == (1, 1, 2)
    assert restored.kspace_volume.indices == (4, 3, 0)
    assert restored.volume_pages.currentIndex() == 1

    legacy = SequenceReconstructionExplorer()
    legacy.set_dataset(_cartesian_3d_dataset())
    legacy.restore_state({"volume_indices": [3, 2, 1]})
    assert legacy.image_volume.indices == (3, 2, 1)
    assert legacy.kspace_volume.indices == (3, 2, 1)


def test_exported_netcdf_complex_pairs_can_be_reopened(tmp_path):
    source = _framed_cartesian_dataset().drop_vars(
        ["species_cartesian_image", "species_cartesian_kspace"]
    )
    stored = source.drop_vars(["cartesian_image", "cartesian_kspace"]).assign(
        cartesian_image_real=source.cartesian_image.real,
        cartesian_image_imag=source.cartesian_image.imag,
        cartesian_kspace_real=source.cartesian_kspace.real,
        cartesian_kspace_imag=source.cartesian_kspace.imag,
    )
    path = tmp_path / "sequence_result.nc"
    stored.to_netcdf(path)

    model = SequenceReconstructionModel.from_file(path)

    assert model.kind == "cartesian_2d"
    np.testing.assert_allclose(model.dataset.cartesian_image, source.cartesian_image)


def test_csi_explorer_links_voxel_selection_to_spectrum(qt_application):
    shape = (2, 2, 3, 4)
    values = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1j
    dataset = xr.Dataset(
        {
            "csi_kspace": (
                ("repetition", "phase_y", "phase_x", "spectral_point"),
                values,
            ),
            "csi_spatial_fid": (
                ("repetition", "phase_y", "phase_x", "spectral_point"),
                values,
            ),
            "csi_spectrum": (
                ("repetition", "phase_y", "phase_x", "spectral_point"),
                np.fft.fftshift(np.fft.fft(values, axis=-1), axes=-1),
            ),
        },
        coords={
            "repetition": [0, 1],
            "phase_y": np.arange(2),
            "phase_x": np.arange(3),
            "spectral_point": np.arange(4),
            "spectral_frequency_hz": (
                "spectral_point",
                [-200.0, -100.0, 0.0, 100.0],
            ),
        },
    )
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(dataset)

    assert isinstance(explorer.outer_controls["repetition"], QSlider)
    assert isinstance(explorer.spectral_point, QSlider)
    explorer.outer_controls["repetition"].setValue(1)
    explorer.spectral_point.setValue(2)
    explorer._voxel_selected(2, 1)

    assert explorer.model.kind == "csi"
    assert explorer.get_state()["outer"]["repetition"] == 1
    assert explorer.get_state()["spectral_point"] == 2
    assert explorer.spectral_value_label.text() == "2  (3/4) · 0 Hz"
    assert explorer.get_state()["voxel"] == [2, 1]
    assert explorer._current_display.shape == (2, 3)
    assert len(explorer.spectrum_plot.listDataItems()) == 3

    modes = [
        explorer.voxel_plot_combo.itemData(index)
        for index in range(explorer.voxel_plot_combo.count())
    ]
    assert modes == ["spectrum", "fid", "time_course"]
    spectrum = explorer.spectrum_plot.listDataItems()[0]
    np.testing.assert_allclose(spectrum.xData, [-200.0, -100.0, 0.0, 100.0])

    explorer.voxel_plot_combo.setCurrentIndex(1)
    fid = explorer.spectrum_plot.listDataItems()[0]
    np.testing.assert_allclose(fid.yData, np.abs(values[1, 1, 2]))

    explorer.voxel_plot_combo.setCurrentIndex(2)
    course = explorer.spectrum_plot.listDataItems()[0]
    np.testing.assert_allclose(course.yData, np.abs(values[:, 1, 2, 2]))
    assert explorer.get_state()["voxel_plot"] == "time_course"


def test_cartesian_3d_model_maps_logical_axes_back_to_scanner_coordinates():
    logical = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(complex)
    dataset = xr.Dataset(
        {
            "cartesian_3d_image": (
                ("partition_x", "phase_y", "read_z"),
                logical,
            ),
            "cartesian_3d_kspace": (
                ("partition_x", "phase_y", "read_z"),
                np.fft.fftn(logical),
            ),
        },
        coords={
            "partition_x": np.arange(2),
            "phase_y": np.arange(3),
            "read_z": np.arange(4),
            "cartesian_k_partition_cyc_per_m": (
                "partition_x",
                [-5.0, 5.0],
            ),
            "cartesian_k_phase_cyc_per_m": (
                "phase_y",
                [-10.0, 0.0, 10.0],
            ),
            "cartesian_k_read_cyc_per_m": (
                "read_z",
                [-15.0, -5.0, 5.0, 15.0],
            ),
        },
        attrs={"cartesian_encoding_axes": "+z +y -x"},
    )
    model = SequenceReconstructionModel(dataset)

    scanner, fov = model.scanner_volume(dataset.cartesian_3d_image)

    np.testing.assert_array_equal(scanner, np.flip(logical, axis=0))
    assert fov == (0.1, 0.1, 0.1)


def test_radial_multi_echo_result_exports_dimensioned_3d_reconstruction(tmp_path):
    dimensions = AcquisitionDimensions(
        adc_event_sample_counts=(4, 4),
        echo_indices=(0, 1),
    )
    line = np.array([-1.5, -0.5, 0.5, 1.5])
    trajectory = np.vstack(
        (
            np.column_stack((line, np.zeros(4), np.zeros(4))),
            np.column_stack((np.zeros(4), line, np.zeros(4))),
        )
    )
    result = SequenceSimulationResult(
        signal=np.ones(8, dtype=np.complex128),
        adc_times_s=np.arange(8, dtype=float) * 1e-4,
        final_magnetization=np.zeros((1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        adc_gradient_moment_cyc_per_m=trajectory,
        metadata={
            "acquisition_dimensions": dimensions.to_metadata(),
            "sequence_definitions": {
                "TrajectoryType": "radial_3d_spiral_phyllotaxis",
                "MatrixSize": [4, 4, 4],
                "FOV": [1.0, 1.0, 1.0],
            },
        },
    )

    dataset = result.to_xarray()

    assert dataset.radial_3d_image.dims == (
        "echo",
        "radial_z",
        "radial_y",
        "radial_x",
    )
    assert dataset.radial_3d_image.shape == (2, 4, 4, 4)
    assert np.isfinite(dataset.radial_3d_image).all()
    output = result.save(tmp_path / "radial_result.npz")
    with np.load(output) as stored:
        assert stored["radial_3d_image"].shape == (2, 4, 4, 4)
        assert stored["radial_3d_echo_index"].tolist() == [0, 1]


def test_dynamic_explorer_plots_voxel_time_course(qt_application):
    dataset = _framed_cartesian_dataset()
    dataset = dataset.assign_coords(
        cartesian_frame_first_adc_time_s=("cartesian_frame", [0.0, 0.01, 1.0, 1.01])
    )
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(dataset)

    modes = [
        explorer.voxel_plot_combo.itemData(index)
        for index in range(explorer.voxel_plot_combo.count())
    ]
    assert modes == ["echoes", "echo_spectrum", "time_course"]
    explorer.voxel_plot_combo.setCurrentIndex(2)
    assert explorer.spectrum_plot.isVisibleTo(explorer)
    explorer.outer_controls["echo"].setValue(1)
    explorer._voxel_selected(4, 2)

    course = explorer.spectrum_plot.listDataItems()[0]
    image = np.asarray(dataset["cartesian_image"])
    np.testing.assert_allclose(course.xData, [0.0, 1.0])
    np.testing.assert_allclose(course.yData, np.abs(image[[1, 3], 2, 4]))


def test_clicking_the_time_course_selects_the_nearest_repetition(qt_application):
    dataset = _framed_cartesian_dataset().assign_coords(
        cartesian_frame_first_adc_time_s=("cartesian_frame", [0.0, 0.01, 1.0, 1.01])
    )
    explorer = SequenceReconstructionExplorer()
    explorer.resize(1200, 800)
    explorer.show()
    explorer.set_dataset(dataset)
    explorer.voxel_plot_combo.setCurrentIndex(
        explorer.voxel_plot_combo.findData("time_course")
    )
    qt_application.processEvents()
    repetition = explorer.outer_controls["repetition"]
    assert repetition.value() == 0

    view_box = explorer.spectrum_plot.getViewBox()
    y = float(np.mean(view_box.viewRange()[1]))
    event = SimpleNamespace(
        button=lambda: Qt.LeftButton,
        scenePos=lambda: view_box.mapViewToScene(QPointF(0.8, y)),
        accept=lambda: None,
    )
    explorer._time_course_clicked(explorer.spectrum_plot, event)

    assert repetition.value() == 1
    explorer.close()


def test_reappearing_voxel_plot_gets_its_share_of_the_height(qt_application):
    dataset = _framed_cartesian_dataset().assign_coords(
        cartesian_frame_first_adc_time_s=("cartesian_frame", [0.0, 0.01, 1.0, 1.01])
    )
    explorer = SequenceReconstructionExplorer()
    explorer.resize(1400, 900)
    explorer.show()
    explorer.set_dataset(_sliced_cartesian_dataset())
    qt_application.processEvents()
    assert explorer.voxel_plot_panel.isHidden()

    explorer.set_dataset(dataset)
    qt_application.processEvents()
    images, plot = explorer.result_splitter.sizes()
    assert plot == pytest.approx(0.4 * (images + plot), abs=2)
    explorer.close()


def _epsi_dataset(echoes=16, spacing_s=1e-3, offset_hz=125.0):
    ny, nx = 3, 4
    times = 2e-3 + spacing_s * np.arange(echoes)
    train = np.exp(2j * np.pi * offset_hz * times)
    image = np.ones((ny, nx))[None] * train[:, None, None]
    return (
        xr.Dataset(
            {
                "cartesian_image": (("cartesian_frame", "phase_y", "read_x"), image),
                "cartesian_kspace": (
                    ("cartesian_frame", "phase_y", "read_x"),
                    np.fft.fft2(image, axes=(-2, -1)),
                ),
            },
            coords={
                "cartesian_frame": np.arange(echoes),
                "phase_y": np.arange(ny),
                "read_x": np.arange(nx),
                "cartesian_frame_echo_index": ("cartesian_frame", np.arange(echoes)),
            },
            attrs={"echo_times_s": ",".join(f"{value:.17g}" for value in times)},
        ),
        times,
        train,
    )


def test_epsi_explorer_switches_between_echo_train_and_spectrum(qt_application):
    dataset, times, train = _epsi_dataset()
    explorer = SequenceReconstructionExplorer()
    explorer.resize(1200, 800)
    explorer.show()
    explorer.set_dataset(dataset)
    qt_application.processEvents()

    # Long trains open on the spectrum; the bar above the plot switches.
    assert explorer.voxel_plot_combo.currentData() == "echo_spectrum"
    buttons = explorer.voxel_plot_mode_buttons.buttons()
    assert [button.text() for button in buttons] == ["Spectrum", "Echoes"]
    assert explorer.voxel_plot_mode_bar.isVisibleTo(explorer)
    explorer._voxel_selected(1, 2)
    magnitude = explorer.spectrum_plot.listDataItems()[0]
    peak = magnitude.xData[int(np.argmax(magnitude.yData))]
    assert peak == pytest.approx(125.0)
    assert "62.5 Hz resolution" in explorer.spectrum_plot.plotItem.titleLabel.text

    buttons[1].click()
    assert explorer.voxel_plot_combo.currentData() == "echoes"
    echo_train = explorer.spectrum_plot.listDataItems()
    np.testing.assert_allclose(echo_train[0].xData, times * 1e3)
    np.testing.assert_allclose(echo_train[1].yData, train.real)

    # Clicking into the echo train selects that echo.
    view_box = explorer.spectrum_plot.getViewBox()
    y = float(np.mean(view_box.viewRange()[1]))
    event = SimpleNamespace(
        button=lambda: Qt.LeftButton,
        scenePos=lambda: view_box.mapViewToScene(QPointF(times[5] * 1e3, y)),
        accept=lambda: None,
    )
    explorer._time_course_clicked(explorer.spectrum_plot, event)
    assert explorer.outer_controls["echo"].value() == 5
    assert buttons[1].isChecked()
    explorer.close()


def test_csi_voxel_plot_has_spectrum_and_fid_buttons(qt_application):
    values = np.ones((2, 3, 4), dtype=complex)
    dataset = xr.Dataset(
        {
            name: (("phase_y", "phase_x", "spectral_point"), values)
            for name in ("csi_kspace", "csi_spatial_fid", "csi_spectrum")
        },
        coords={"spectral_point": np.arange(4)},
    )
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(dataset)
    buttons = explorer.voxel_plot_mode_buttons.buttons()
    assert [button.text() for button in buttons] == ["Spectrum", "FID"]
    buttons[1].click()
    assert explorer.voxel_plot_combo.currentData() == "fid"
    assert explorer.spectrum_plot.plotItem.titleLabel.text.startswith("FID")


def test_single_frame_explorer_hides_voxel_plot(qt_application):
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(_sliced_cartesian_dataset())

    assert explorer.voxel_plot_combo.count() == 0
    assert not explorer.spectrum_plot.isVisibleTo(explorer)


def test_model_averages_complex_repetitions_of_framed_data():
    dataset = _framed_cartesian_dataset()
    model = SequenceReconstructionModel(dataset)
    frames = dataset.cartesian_frame_repetition_index.values
    echoes = dataset.cartesian_frame_echo_index.values

    averaged = model.select(
        model.image_name(),
        {"echo": 1, "repetition": 0},
        average_dimensions=("repetition",),
    )

    selected = np.flatnonzero(echoes == 1)
    assert selected.size > 1 and set(frames[selected]) == {0, 1}
    expected = dataset.cartesian_image.isel(cartesian_frame=selected)
    np.testing.assert_allclose(averaged, expected.mean("cartesian_frame"))
    # A complex mean differs from the mean of the magnitudes.
    assert not np.allclose(np.abs(averaged), np.abs(expected).mean("cartesian_frame"))


def test_model_averages_complex_repetitions_before_combining_coils():
    shape = (2, 2, 3, 4, 2)
    rng = np.random.default_rng(3)
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    dims = ("repetition", "phase_y", "phase_x", "spectral_point", "coil")
    dataset = xr.Dataset(
        {
            "csi_kspace": (dims, values),
            "csi_spatial_fid": (dims, values),
            "csi_spectrum": (dims, values),
        },
        coords={
            "repetition": [0, 1],
            "phase_y": np.arange(2),
            "phase_x": np.arange(3),
            "spectral_point": np.arange(4),
            "coil": np.arange(2),
        },
    )
    model = SequenceReconstructionModel(dataset)
    assert [item.name for item in model.outer_dimensions] == ["repetition"]

    averaged = model.select(
        "csi_spatial_fid", {}, coil_mode="rss", average_dimensions=("repetition",)
    )

    expected = np.sqrt((np.abs(values.mean(axis=0)) ** 2).sum(axis=-1))
    np.testing.assert_allclose(averaged, expected)


def test_explorer_averages_repetitions_independent_of_displayed_component(
    qt_application,
):
    dataset = _framed_cartesian_dataset()
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(dataset)
    assert explorer.average_repetitions.isEnabled()
    echoes = dataset.cartesian_frame_echo_index.values
    selected = np.flatnonzero(echoes == 0)
    mean = dataset.cartesian_image.isel(cartesian_frame=selected).mean(
        "cartesian_frame"
    )

    explorer.average_repetitions.setChecked(True)
    np.testing.assert_allclose(explorer._current_display, np.abs(mean))
    assert not explorer.outer_controls["repetition"].isEnabled()
    assert explorer.get_state()["average_repetitions"] is True

    explorer._set_combo_data(explorer.component_combo, "real")
    np.testing.assert_allclose(explorer._current_display, np.real(mean))

    restored = SequenceReconstructionExplorer()
    restored.set_dataset(dataset)
    restored.restore_state(explorer.get_state())
    assert restored.average_repetitions.isChecked()
    np.testing.assert_allclose(restored._current_display, np.real(mean))

    explorer.average_repetitions.setChecked(False)
    assert explorer.outer_controls["repetition"].isEnabled()


def test_csi_explorer_average_keeps_the_time_course_per_repetition(qt_application):
    shape = (3, 2, 3, 4)
    values = np.arange(np.prod(shape), dtype=float).reshape(shape) * (1 + 0.5j)
    dims = ("repetition", "phase_y", "phase_x", "spectral_point")
    dataset = xr.Dataset(
        {name: (dims, values) for name in ("csi_kspace", "csi_spatial_fid")}
        | {"csi_spectrum": (dims, values)},
        coords={
            "repetition": [0, 1, 2],
            "phase_y": np.arange(2),
            "phase_x": np.arange(3),
            "spectral_point": np.arange(4),
        },
    )
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(dataset)
    explorer._voxel_selected(1, 1)
    explorer._set_combo_data(explorer.voxel_plot_combo, "time_course")

    explorer.average_repetitions.setChecked(True)

    expected = np.abs(values.mean(axis=0)[:, :, 0])
    np.testing.assert_allclose(explorer._current_display, expected)
    (curve,) = explorer.spectrum_plot.listDataItems()
    assert len(curve.getData()[1]) == 3


def _dynamic_cartesian_3d_dataset(definitions, repetitions=4):
    base = _cartesian_3d_dataset()
    volume = np.asarray(base["cartesian_3d_image"])
    # Even repetitions are weak (lactate-like), odd ones strong (pyruvate-like).
    scale = np.asarray(
        [1.0 if value % 2 == 0 else 100.0 for value in range(repetitions)]
    )
    image = (
        scale[:, None, None, None]
        * (1.0 + np.arange(repetitions))[:, None, None, None]
        * (volume + 1.0)
    )
    samples = 3
    return xr.Dataset(
        {
            "cartesian_3d_image": (
                ("repetition", "partition_z", "phase_y", "read_x"),
                image,
            ),
            "cartesian_3d_kspace": (
                ("repetition", "partition_z", "phase_y", "read_x"),
                np.fft.fftn(image, axes=(-3, -2, -1)),
            ),
        },
        coords={
            **{name: base.coords[name] for name in base.coords},
            "repetition": np.arange(repetitions),
            "repetition_index": (
                "adc_sample",
                np.repeat(np.arange(repetitions), samples),
            ),
            "adc_time_s": (
                "adc_sample",
                np.repeat(0.5 * np.arange(repetitions), samples)
                + np.tile(np.arange(samples) * 1e-3, repetitions),
            ),
        },
        attrs={
            "cartesian_encoding_axes": "+x +y +z",
            "metadata_json": json.dumps({"sequence_definitions": definitions}),
        },
    )


def test_model_labels_repetitions_of_spectrally_selective_sequences():
    ss_bssfp = SequenceReconstructionModel(
        _dynamic_cartesian_3d_dataset({"SpectralTargetNames": "Lac Py"})
    )
    assert ss_bssfp.repetition_metabolites == ("Lac", "Py", "Lac", "Py")
    np.testing.assert_allclose(
        ss_bssfp.outer_dimension_times_s("repetition"), [0.0, 0.5, 1.0, 1.5]
    )

    schedule = {
        "SPSPScheduleFrameIndices": "0 1 2 0",
        "SPSPScheduleTargetNames": json.dumps(["Pyruvate", "Lactate", ""]),
        "SPSPScheduleDisplayNames": json.dumps(["Pyr", "Lac", "Alanine"]),
        "SPSPScheduleSelectionLevel": "frame",
    }
    spsp = SequenceReconstructionModel(_dynamic_cartesian_3d_dataset(schedule))
    assert spsp.repetition_metabolites == ("Pyruvate", "Lactate", "Alanine", "Pyruvate")

    mixed = SequenceReconstructionModel(
        _dynamic_cartesian_3d_dataset({**schedule, "SPSPScheduleSelectionLevel": "tr"})
    )
    assert mixed.repetition_metabolites == ()
    plain = SequenceReconstructionModel(_dynamic_cartesian_3d_dataset({}))
    assert plain.repetition_metabolites == ()


def test_model_averages_a_subset_of_volume_repetitions():
    dataset = _dynamic_cartesian_3d_dataset({})
    model = SequenceReconstructionModel(dataset)
    averaged = model.select(
        model.image_name(),
        {},
        average_dimensions=("repetition",),
        average_values={"repetition": [1, 3]},
    )
    expected = np.abs(
        dataset.cartesian_3d_image.isel(repetition=[1, 3]).mean("repetition")
    )
    np.testing.assert_allclose(np.asarray(averaged), np.asarray(expected))


def test_explorer_colours_and_filters_metabolite_repetitions(qt_application):
    dataset = _dynamic_cartesian_3d_dataset({"SpectralTargetNames": "Lac Py"})
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(dataset)
    image = np.abs(np.asarray(dataset.cartesian_3d_image))

    assert explorer.frame_bar.isVisibleTo(explorer)
    assert [
        explorer.metabolite_filter.itemData(index)
        for index in range(explorer.metabolite_filter.count())
    ] == [None, "Lac", "Py"]
    assert explorer._metabolite_colors["Py"] != explorer._metabolite_colors["Lac"]
    slider = explorer.outer_controls["repetition"]
    assert "Lac" in explorer.outer_value_labels["repetition"].text()
    # Weak lactate images are not scaled to the strong pyruvate ones.
    assert explorer.contrast_slider._domain[1] == pytest.approx(1.1 * image[2].max())
    slider.setValue(1)
    assert "Py" in explorer.summary.text()
    assert explorer.contrast_slider._domain[1] == pytest.approx(1.1 * image[3].max())

    curves = explorer.volume_plot.listDataItems()
    assert explorer.volume_plot.isVisibleTo(explorer)
    assert [curve.name() for curve in curves] == ["Lac", "Py"]
    np.testing.assert_allclose(curves[1].xData, [0.5, 1.5])
    x, y, z = explorer.image_volume.indices
    np.testing.assert_allclose(curves[1].yData, image[[1, 3], z, y, x])

    # Averaging mixed contrasts is refused until one metabolite is chosen.
    assert not explorer.average_repetitions.isEnabled()
    explorer.metabolite_filter.setCurrentIndex(
        explorer.metabolite_filter.findData("Py")
    )
    assert explorer.outer_values["repetition"] == (1, 3)
    assert slider.maximum() == 1
    assert [curve.name() for curve in explorer.volume_plot.listDataItems()] == ["Py"]
    explorer.average_repetitions.setChecked(True)
    np.testing.assert_allclose(
        explorer.image_volume.data,
        explorer.model.scanner_volume(
            xr.DataArray(
                image[[1, 3]].mean(axis=0), dims=dataset.cartesian_3d_image.dims[1:]
            )
        )[0],
    )

    state = explorer.get_state()
    assert state["metabolite_filter"] == "Py"
    restored = SequenceReconstructionExplorer()
    restored.set_dataset(dataset)
    restored.restore_state(state)
    assert restored.outer_values["repetition"] == (1, 3)
    assert restored.average_repetitions.isChecked()


def test_explorer_plays_repetitions_and_keeps_3d_slices(qt_application):
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(_dynamic_cartesian_3d_dataset({}))
    slider = explorer.outer_controls["repetition"]
    explorer.image_volume.set_indices((1, 2, 0))

    explorer.play_button.setChecked(True)
    assert explorer._play_timer.isActive()
    for expected in (1, 2, 3, 0):
        explorer._play_step()
        assert slider.value() == expected
    assert explorer.image_volume.indices == (1, 2, 0)
    assert [curve.name() for curve in explorer.volume_plot.listDataItems()] == [
        "Magnitude"
    ]

    explorer.average_repetitions.setChecked(True)
    assert not explorer.play_button.isChecked()
    explorer.clear()
    assert not explorer.frame_bar.isVisibleTo(explorer)


def _raw_signal_dataset(samples=64, dwell_s=1e-3, offset_hz=50.0):
    times = 1e-3 + dwell_s * np.arange(samples)
    signal = np.exp(2j * np.pi * offset_hz * times) * np.exp(-times / 0.05)
    return xr.Dataset(
        {
            "signal": (("adc",), signal),
            "species_signal": (("adc", "pool"), signal[:, None]),
        },
        coords={
            "adc_time_s": ("adc", times),
            "adc_event_index": ("adc", np.zeros(samples, dtype=int)),
            "pool": ["Pyruvate"],
        },
        attrs={"duration_s": 0.08},
    )


def test_single_pulse_is_shown_as_fid_and_spectrum(qt_application):
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(_raw_signal_dataset())

    assert explorer.model.kind == "raw_signal"
    assert explorer.pages.currentWidget() is explorer.signal_page
    fid = explorer.signal_fid_plot.listDataItems()
    spectrum = explorer.signal_spectrum_plot.listDataItems()
    assert fid and spectrum
    frequency, amplitude = spectrum[0].getData()
    assert frequency[np.argmax(amplitude)] == pytest.approx(50.0, abs=16.0)
    # A single excitation needs no excitation selector.
    assert not explorer.signal_event_spin.isVisibleTo(explorer)

    explorer.clear()
    assert not explorer.signal_fid_plot.listDataItems()


def test_explorer_switches_spiral_reconstruction_method_and_gradient_errors(
    qt_application, tmp_path
):
    from blochsimulator import BlochSimulator
    from blochsimulator.sequence import load_pulseq
    from blochsimulator.sequence.kspace_trajectory import (
        GradientImperfections,
        gradient_error,
    )
    from blochsimulator.sequence.psf import PointSource, simulate_point_sources
    from blochsimulator.sequence.pulseq_builders import make_pulseq_spiral

    sequence = make_pulseq_spiral(
        fov_m=(0.08, 0.08),
        matrix=(16, 16),
        sampling_bandwidth_hz=50e3,
        read_offset_m=0.0,
    )
    path = tmp_path / "spiral.seq"
    sequence.write(str(path), v141_compat=True)
    program = load_pulseq(path)
    result = simulate_point_sources(
        program,
        (PointSource(position_m=(0.012, -0.007, 0.0)),),
        simulator=BlochSimulator(use_parallel=False),
    )
    delta = gradient_error(program, GradientImperfections(delay_s=(8e-6, 8e-6, 0.0)))

    explorer = SequenceReconstructionExplorer()
    explorer.set_result(result)
    assert not explorer.reconstruction_group.isHidden()
    assert explorer.reconstruction_method_combo.currentData() == "nufft"
    assert not explorer.reconstruction_trajectory_combo.isEnabled()
    nufft = np.array(explorer._current_display)

    explorer.reconstruction_method_combo.setCurrentIndex(
        explorer.reconstruction_method_combo.findData("cg")
    )
    assert explorer._result is result
    assert not explorer.reconstruction_group.isHidden()
    cg = np.array(explorer._current_display)
    assert cg.shape == nufft.shape
    assert not np.allclose(cg, nufft)
    assert np.unravel_index(cg.argmax(), cg.shape) == np.unravel_index(
        nufft.argmax(), nufft.shape
    )

    explorer.gradient_error_provider = lambda _result: delta
    explorer.set_result(result)
    assert explorer.reconstruction_trajectory_combo.isEnabled()
    explorer.reconstruction_trajectory_combo.setCurrentIndex(1)
    shifted = np.array(explorer._current_display)
    # An 8 µs delay misplaces the spiral by about half a k-space cell.
    assert np.abs(shifted - nufft).max() > 0.05 * nufft.max()
    assert explorer.reconstruction_message.text() == ""

    def missing(_result):
        raise ValueError("No gradient imperfections are set")

    explorer.gradient_error_provider = missing
    explorer.reconstruction_trajectory_combo.setCurrentIndex(0)
    explorer.reconstruction_trajectory_combo.setCurrentIndex(1)
    assert "ideal gradients" in explorer.reconstruction_message.text()
    np.testing.assert_allclose(explorer._current_display, nufft)
