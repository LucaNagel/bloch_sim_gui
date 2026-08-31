import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QGroupBox, QSizePolicy

from blochsimulator.b1_fields import (
    B1Field,
    b1_preset_options,
    create_b1_preset,
    load_b1_field,
)
from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom
from blochsimulator.paths import workspace_directory
from blochsimulator.phantom import PhantomFactory
from blochsimulator.sequence import ADCEvent, RFEvent, SequenceProgram
from blochsimulator.spectral_phantom import ChemicalSpecies, SpectralPhantom
from blochsimulator.ui.b1_widgets import (
    B1PhantomCombinationWidget,
    B1WorkspaceWidget,
)
from blochsimulator.ui.main_window import BlochSimulatorGUI


def test_b1_workspace_directory_follows_configured_root(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOCHSIMULATOR_DATA_DIR", str(tmp_path))

    assert workspace_directory("b1_fields") == tmp_path / "b1_fields"


def test_uniform_3d_b1_resamples_exactly_to_matching_phantom_grid():
    phantom = PhantomFactory.uniform((3, 4, 2), (0.03, 0.04, 0.02), 1.0, 0.1)
    value = 0.8 + 0.25j
    field = B1Field.uniform(
        phantom.shape,
        phantom.fov,
        kind="transmit",
        value=value,
    )

    sampled = field.resample_to_phantom(phantom)

    assert sampled.shape == (1, *phantom.shape)
    np.testing.assert_allclose(sampled[0], value)


@pytest.mark.parametrize(
    ("shape", "fov"),
    [
        ((11, 9), (0.22, 0.18)),
        ((9, 7, 5), (0.18, 0.14, 0.10)),
    ],
)
def test_default_b1_presets_generate_finite_complex_2d_and_3d_fields(shape, fov):
    for kind in ("transmit", "receive"):
        options = dict(b1_preset_options(kind))
        assert set(options) >= {
            "uniform",
            "birdcage_cp",
            "surface_loop",
            "linear_ramp",
        }
        assert ("circular_array" in options) == (kind == "receive")
        for preset in options:
            field = create_b1_preset(
                preset,
                shape,
                fov,
                kind=kind,
                ramp_axis="y",
            )
            assert field.spatial_shape == shape
            assert field.spatial_ndim == len(shape)
            assert field.n_channels == (
                8 if kind == "receive" and preset == "circular_array" else 1
            )
            assert np.iscomplexobj(field.data)
            assert np.all(np.isfinite(field.data))


def test_physical_3d_presets_have_expected_spatial_and_channel_profiles():
    shape = (21, 9, 7)
    fov = (0.21, 0.09, 0.07)
    loop = create_b1_preset("surface_loop", shape, fov, kind="transmit")
    loop_magnitude = np.abs(loop.values)
    assert loop_magnitude[:3].mean() > loop_magnitude[-3:].mean()

    birdcage = create_b1_preset(
        "birdcage_cp", shape, fov, kind="transmit", phase_deg=35.0
    )
    center = tuple(count // 2 for count in shape)
    assert abs(birdcage.values[center]) == pytest.approx(1.0)
    assert np.angle(birdcage.values[center], deg=True) == pytest.approx(35.0)

    receive = create_b1_preset("circular_array", shape, fov, kind="receive")
    center_rss = np.sqrt(np.sum(np.abs(receive.data[(slice(None), *center)]) ** 2))
    assert receive.n_channels == 8
    assert center_rss == pytest.approx(1.0)


def test_linear_ramp_presets_control_magnitude_and_phase_axis():
    magnitude = create_b1_preset(
        "linear_ramp",
        (9, 7),
        (0.09, 0.07),
        ramp_axis="x",
        ramp_mode="magnitude",
    )
    assert np.abs(magnitude.values[0]).mean() < np.abs(magnitude.values[-1]).mean()

    phase = create_b1_preset(
        "linear_ramp",
        (9, 7),
        (0.09, 0.07),
        ramp_axis="y",
        ramp_mode="phase",
    )
    unwrapped = np.unwrap(np.angle(phase.values[4]))
    assert np.all(np.diff(unwrapped) > 0)


def test_2d_field_rotation_and_stretch_use_object_space_coordinates():
    values = np.repeat(np.arange(1.0, 4.0)[:, None], 3, axis=1)
    field = B1Field(values, (0.03, 0.03), kind="transmit")

    field.set_transform((1.0, 1.0, 1.0), (0.0, 0.0, 90.0))
    rotated = field.sample_world(np.asarray([[0.0, 0.01, 0.0]]))
    assert rotated[0, 0] == pytest.approx(3.0)

    compact = B1Field(np.ones((2, 2)), (0.02, 0.02), kind="transmit")
    outside = compact.sample_world(np.asarray([[0.009, 0.0, 0.0]]))
    compact.set_transform((2.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    stretched = compact.sample_world(np.asarray([[0.009, 0.0, 0.0]]))
    assert outside[0, 0] == 0.0
    assert stretched[0, 0] > 0.0


def test_receive_field_loader_preserves_channels_and_geometry(tmp_path):
    path = tmp_path / "receive.npz"
    receive = np.stack(
        [np.ones((2, 3, 4)), np.full((2, 3, 4), 2j)],
        axis=0,
    )
    np.savez(
        path,
        rx_sensitivity_maps=receive,
        fov_mm=np.asarray([20.0, 30.0, 40.0]),
        scale_xyz=np.asarray([1.5, 0.75, 2.0]),
        rotation_deg_xyz=np.asarray([10.0, 20.0, 30.0]),
    )

    field = load_b1_field(path, kind="receive")

    assert field.spatial_shape == (2, 3, 4)
    assert field.n_channels == 2
    assert field.fov_m == pytest.approx((0.02, 0.03, 0.04))
    assert field.scale_xyz == pytest.approx((1.5, 0.75, 2.0))
    assert field.rotation_deg_xyz == pytest.approx((10.0, 20.0, 30.0))


def test_spectral_components_inherit_applied_b1_fields():
    shape = (2, 2)
    species = ChemicalSpecies(
        name="Water",
        chemical_shift_ppm=0.0,
        t1=1.0,
        t2=0.1,
        t2_star=0.08,
    )
    phantom = SpectralPhantom(
        shape=shape,
        fov=(0.02, 0.02),
        species=[species],
        concentration_maps={"Water": np.ones(shape)},
    )
    phantom.tx_sensitivity_map = np.full(shape, 0.75 + 0.1j)
    phantom.rx_sensitivity_maps = np.full((2, *shape), 0.5 - 0.2j)

    _, component = phantom.to_component_phantoms()[0]

    np.testing.assert_array_equal(
        component.tx_sensitivity_map, phantom.tx_sensitivity_map
    )
    np.testing.assert_array_equal(
        component.rx_sensitivity_maps, phantom.rx_sensitivity_maps
    )


def test_dynamic_3d_sequence_uses_spatial_tx_and_multi_receive_fields():
    shape = (2, 1, 1)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.02, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.zeros(shape),
        nucleus="C13",
    )
    phantom.tx_sensitivity_map = np.asarray([1.0, 0.0]).reshape(shape)
    phantom.rx_sensitivity_maps = np.asarray(
        [
            np.asarray([1.0, 0.0]).reshape(shape),
            np.asarray([0.0, 1.0]).reshape(shape),
        ],
        dtype=np.complex128,
    )
    program = SequenceProgram(
        (
            RFEvent(0.0, np.asarray([250.0]), 1e-3),
            ADCEvent(1e-3, 1, 1e-6),
        ),
        duration_s=1.001e-3,
    )

    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom
    )

    assert result.signal.shape == (2, 1)
    assert result.species_signal.shape == (2, 2, 1)
    assert abs(result.signal[0, 0]) > 0.5
    assert result.signal[1, 0] == pytest.approx(0.0, abs=1e-12)
    assert result.metadata["n_rx_coils"] == 2


def test_b1_workspace_applies_tx_and_multi_receive_maps_to_phantom():
    app = QApplication.instance() or QApplication([])
    phantom = PhantomFactory.uniform((3, 3, 2), (0.03, 0.03, 0.02), 1.0, 0.1)
    workspace = B1WorkspaceWidget()
    workspace.set_phantom(phantom)
    workspace.tx_editor.set_field(
        B1Field.uniform(phantom.shape, phantom.fov, value=0.7 + 0.1j)
    )
    workspace.rx_editor.set_field(
        B1Field.uniform(
            phantom.shape,
            phantom.fov,
            kind="receive",
            value=0.5j,
            channels=2,
        )
    )

    workspace.apply_fields()

    np.testing.assert_allclose(phantom.tx_sensitivity_map, 0.7 + 0.1j)
    assert phantom.rx_sensitivity_maps.shape == (2, *phantom.shape)
    np.testing.assert_allclose(phantom.rx_sensitivity_maps, 0.5j)
    assert "2 receive channel(s)" in workspace.apply_status.text()
    workspace.close()
    app.processEvents()


def test_b1_workspace_layout_is_compact_and_field_specific():
    app = QApplication.instance() or QApplication([])
    workspace = B1WorkspaceWidget()

    assert workspace.splitter.count() == 2
    assert (
        workspace.layout().stretch(workspace.layout().indexOf(workspace.splitter)) == 1
    )
    assert workspace.header.sizePolicy().verticalPolicy() == QSizePolicy.Fixed
    assert workspace.header.maximumHeight() < 60
    assert workspace.tx_preview.channel_label.isHidden()
    assert workspace.tx_preview.channel_combo.isHidden()
    assert not workspace.rx_preview.channel_label.isHidden()
    assert workspace.tx_editor.fov_spins[0].minimumWidth() >= 125
    assert workspace.tx_editor.scale_spins[0].minimumWidth() >= 90
    assert "QGroupBox { font-weight: bold; }" in workspace.styleSheet()
    titled_groups = {
        group.title(): group
        for group in workspace.findChildren(QGroupBox)
        if group.title()
    }
    assert titled_groups["Field source"].fontInfo().bold()
    assert titled_groups["Spatial geometry"].fontInfo().bold()

    workspace.editor_tabs.setCurrentIndex(1)
    assert workspace.preview_tabs.currentIndex() == 1
    workspace.preview_tabs.setCurrentIndex(0)
    assert workspace.editor_tabs.currentIndex() == 1
    workspace.editor_tabs.setCurrentIndex(0)
    assert workspace.preview_tabs.currentIndex() == 0
    workspace.close()
    app.processEvents()


def test_b1_editors_offer_and_create_kind_specific_3d_presets():
    app = QApplication.instance() or QApplication([])
    phantom = PhantomFactory.uniform((7, 5, 3), (0.07, 0.05, 0.03), 1.0, 0.1)
    workspace = B1WorkspaceWidget()
    workspace.set_phantom(phantom)

    tx_labels = [
        workspace.tx_editor.preset_combo.itemText(index)
        for index in range(workspace.tx_editor.preset_combo.count())
    ]
    rx_labels = [
        workspace.rx_editor.preset_combo.itemText(index)
        for index in range(workspace.rx_editor.preset_combo.count())
    ]
    assert "Birdcage CP" in tx_labels
    assert "8-channel circular array" not in tx_labels
    assert "8-channel circular array" in rx_labels

    tx_index = workspace.tx_editor.preset_combo.findData("birdcage_cp")
    workspace.tx_editor.preset_combo.setCurrentIndex(tx_index)
    workspace.tx_editor.dimension_combo.setCurrentText("3D")
    workspace.tx_editor._create_preset()
    assert workspace.tx_field.name == "Birdcage CP"
    assert workspace.tx_field.spatial_ndim == 3
    assert workspace.tx_field.spatial_shape == (64, 64, 64)

    rx_index = workspace.rx_editor.preset_combo.findData("circular_array")
    workspace.rx_editor.preset_combo.setCurrentIndex(rx_index)
    workspace.rx_editor.dimension_combo.setCurrentText("3D")
    workspace.rx_editor._create_preset()
    assert workspace.rx_field.n_channels == 8
    assert workspace.rx_field.spatial_ndim == 3
    assert workspace.rx_field.spatial_shape == (64, 64, 64)
    workspace.close()
    app.processEvents()


def test_phantom_b1_combination_shows_transmit_and_receive_vertically():
    app = QApplication.instance() or QApplication([])
    phantom = PhantomFactory.uniform((3, 3, 3), (0.03, 0.03, 0.03), 1.0, 0.1)
    tx = B1Field.uniform(phantom.shape, phantom.fov, kind="transmit")
    rx = B1Field.uniform(phantom.shape, phantom.fov, kind="receive", channels=2)
    combination = B1PhantomCombinationWidget()
    combination.set_phantom(phantom)
    combination.set_fields(tx, rx)

    assert not hasattr(combination, "field_combo")
    assert combination.view_splitter.orientation() == Qt.Vertical
    assert combination.view_splitter.widget(0) is combination.tx_panel
    assert combination.view_splitter.widget(1) is combination.rx_panel
    assert combination.channel_combo.count() == 2
    assert "Uniform B1+" in combination.tx_info.text()
    assert "Uniform B1−" in combination.rx_info.text()
    combination.close()
    app.processEvents()


def test_sequence_mode_exposes_b1_and_combination_tabs():
    app = QApplication.instance() or QApplication([])
    window = BlochSimulatorGUI()

    window.set_workspace_mode("sequence")
    labels = [
        window.tab_widget.tabText(index) for index in range(window.tab_widget.count())
    ]

    assert "B1 Fields" in labels
    assert "Phantom + B1" in labels
    assert not window.tab_widget.isTabVisible(window.magnetization_tab_index)
    assert window.tab_widget.isTabVisible(window.b1_tab_index)
    assert window.tab_widget.isTabVisible(window.b1_combo_tab_index)
    window.close()
    app.processEvents()
