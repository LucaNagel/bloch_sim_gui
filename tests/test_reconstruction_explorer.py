import numpy as np
import xarray as xr
from PyQt5.QtWidgets import QSlider

from blochsimulator.sequence import AcquisitionDimensions, SequenceSimulationResult
from blochsimulator.sequence.reconstruction import (
    SequenceReconstructionModel,
    ideal_separate,
)
from blochsimulator.ui.reconstruction_explorer import SequenceReconstructionExplorer


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


def test_explorer_keeps_3d_image_and_kspace_slice_positions_independent(
    qt_application,
):
    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(_cartesian_3d_dataset())

    assert explorer.volume_pages.count() == 2
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
