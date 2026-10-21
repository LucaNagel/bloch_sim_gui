import numpy as np
import pytest
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QGridLayout
from unittest.mock import MagicMock, patch

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom
from blochsimulator.dynamic_phantom import PyruvateInflow, TimeCurve
from blochsimulator.mouse_phantom import (
    MousePerfusionConfig,
    MousePerfusionPhantom,
    reference_mouse_vascular_graph,
)
from blochsimulator.phantom_widget import PhantomCreatorWidget
from blochsimulator.sequence import SequenceProgram
from blochsimulator.spectral_phantom import ChemicalSpecies
from blochsimulator.ui.phantom_designer import load_any_phantom
from blochsimulator.ui.mouse_phantom_designer import (
    MousePhantomDesignerDialog,
    MouseSlicePreviewWidget,
)


def test_reference_mouse_builds_anatomy_vessels_kidney_maps_and_conserves_dose():
    config = MousePerfusionConfig()
    phantom = config.build()

    assert isinstance(phantom, DynamicSpectralPhantom)
    assert phantom.shape == config.shape
    assert tuple(pool.name for pool in phantom.pools) == (
        "Pyruvate",
        "Lactate",
        "Alanine",
    )
    assert {2, 3, 4, 5, 6, 9, 10, 11} <= set(np.unique(phantom.anatomy_labels))
    assert np.allclose(
        phantom.kpl_map_s_inv[phantom.anatomy_labels == 2],
        config.left_cortex_kpl_s_inv,
    )
    assert np.allclose(
        phantom.kpl_map_s_inv[phantom.anatomy_labels == 5],
        config.right_medulla_kpl_s_inv,
    )
    assert np.allclose(
        phantom.ph_map[phantom.anatomy_labels == 3], config.left_medulla_ph
    )

    curve = phantom.pyruvate_inflow.rate_curve_s_inv
    delivered_mol = curve.integral(curve.times_s[0], curve.times_s[-1]) * (
        np.sum(phantom.pyruvate_inflow.delivery_map) * phantom.voxel_volume_m3
    )
    assert delivered_mol == pytest.approx(config.injection_amount_umol * 1e-6)
    assert phantom.metadata["transport_solver"] == "voxelwise_arrival_delay"
    assert phantom.metadata["ph_model"] == "ground_truth_map_only"
    assert np.allclose(
        phantom.kpa_map_s_inv[phantom.anatomy_labels == 6],
        config.liver_kpa_s_inv,
    )


def test_mouse_physical_size_is_independent_from_fov_and_old_configs_migrate():
    baseline = MousePerfusionConfig(
        shape=(48, 32, 96),
        fov_m=(0.045, 0.032, 0.105),
        mouse_size_m=(0.045, 0.032, 0.105),
    ).build()
    cropped = MousePerfusionConfig(
        shape=(48, 32, 96),
        fov_m=(0.045, 0.032, 0.080),
        mouse_size_m=(0.045, 0.032, 0.105),
    ).build()
    smaller_mouse = MousePerfusionConfig(
        shape=(48, 32, 96),
        fov_m=(0.045, 0.032, 0.105),
        mouse_size_m=(0.036, 0.026, 0.084),
    ).build()

    assert not np.array_equal(cropped.anatomy_labels, baseline.anatomy_labels)
    assert np.count_nonzero(cropped.anatomy_labels) > np.count_nonzero(
        baseline.anatomy_labels
    )
    assert np.count_nonzero(smaller_mouse.anatomy_labels) < np.count_nonzero(
        baseline.anatomy_labels
    )

    legacy_values = MousePerfusionConfig().to_dict()
    legacy_values.pop("mouse_size_m")
    legacy_values["fov_m"] = (0.050, 0.030, 0.090)
    restored = MousePerfusionConfig.from_dict(legacy_values)
    assert restored.mouse_size_m == pytest.approx((0.031, 0.0156, 0.10125))


def test_mouse_size_describes_visible_body_bounds():
    phantom = MousePerfusionConfig(
        shape=(48, 40, 96),
        fov_m=(0.032, 0.025, 0.075),
        mouse_size_m=(0.032, 0.025, 0.075),
    ).build()

    for axis in range(3):
        assert np.any(np.take(phantom.anatomy_labels, 0, axis=axis))
        assert np.any(np.take(phantom.anatomy_labels, -1, axis=axis))


def test_mouse_position_translates_anatomy_in_physical_coordinates():
    common = dict(
        shape=(64, 48, 96),
        fov_m=(0.080, 0.060, 0.140),
        mouse_size_m=(0.032, 0.025, 0.075),
    )
    centered = MousePerfusionConfig(**common).build()
    offset_m = np.asarray((0.010, -0.006, 0.018))
    shifted = MousePerfusionConfig(**common, mouse_offset_m=tuple(offset_m)).build()

    def physical_center(phantom):
        coordinates = tuple(
            (np.arange(count) + 0.5) / count * extent - 0.5 * extent
            for count, extent in zip(phantom.shape, phantom.fov)
        )
        mesh = np.meshgrid(*coordinates, indexing="ij")
        body = phantom.anatomy_labels > 0
        return np.asarray([np.mean(axis[body]) for axis in mesh])

    measured_translation = physical_center(shifted) - physical_center(centered)
    voxel_size = np.asarray(common["fov_m"]) / np.asarray(common["shape"])
    assert measured_translation == pytest.approx(offset_m, abs=max(voxel_size))


def test_spatial_bolus_preview_uses_arrival_map_and_breathing_is_periodic():
    config = MousePerfusionConfig(
        breathing_enabled=True,
        breathing_rate_hz=2.0,
        breathing_amplitude_mm=1.5,
    )
    phantom = config.build()

    source = phantom.bolus_rate_map(
        config.injection_start_s
        + config.circulation_delay_s
        + 0.45 * config.injection_duration_s
    )
    active = source > 0
    assert np.any(active)
    normalized_source = source[active] / phantom.pyruvate_inflow.delivery_map[active]
    assert np.ptp(normalized_source) > 0

    assert np.allclose(phantom.displacement_map_m(0.0), 0.0)
    quarter_period = 0.25 / config.breathing_rate_hz
    displacement = phantom.displacement_map_m(quarter_period)
    assert np.max(displacement[..., 2]) == pytest.approx(
        config.breathing_amplitude_mm * 1e-3,
        rel=2e-3,
    )
    assert np.allclose(displacement[..., :2], 0.0)
    assert np.allclose(
        phantom.displacement_map_m(1.0 / config.breathing_rate_hz),
        phantom.displacement_map_m(0.0),
        atol=1e-15,
    )


def _mouse_axial_position(config, z_index):
    """Return a voxel centre's z in units of the mouse half-length."""
    fov_z = config.fov_m[2]
    z_m = (z_index + 0.5) / config.shape[2] * fov_z - 0.5 * fov_z
    return (z_m - config.mouse_offset_m[2]) / (0.5 * config.mouse_size_m[2])


def test_bolus_enters_at_tail_vein_before_cardiopulmonary_circulation():
    config = MousePerfusionConfig(shape=(24, 16, 48))
    phantom = config.build()
    earliest = np.unravel_index(
        np.nanargmin(phantom.perfusion_arrival_time_s), phantom.shape
    )
    earliest_z = _mouse_axial_position(config, earliest[2])

    # The default mouse no longer fills the FOV, so measure along the body.
    assert earliest_z < -0.75
    assert phantom.anatomy_labels[earliest] == 10
    assert phantom.perfusion_arrival_time_s[earliest] == pytest.approx(
        config.injection_start_s
    )

    early_frame = phantom.injected_concentration_map(
        config.injection_start_s + 0.1 * config.injection_duration_s
    )
    maximum = np.unravel_index(np.argmax(early_frame), phantom.shape)
    assert _mouse_axial_position(config, maximum[2]) < -0.75


def test_mouse_phantom_round_trip_restores_editable_configuration(tmp_path):
    config = MousePerfusionConfig(
        name="Renal mouse",
        left_cortex_kpl_s_inv=0.123,
        right_medulla_ph=6.75,
        breathing_enabled=True,
        mouse_offset_m=(0.003, -0.002, 0.004),
    )
    phantom = config.build()
    path = tmp_path / "renal_mouse.npz"
    phantom.save(path)

    loaded = load_any_phantom(path)

    assert isinstance(loaded, MousePerfusionPhantom)
    assert loaded.config.left_cortex_kpl_s_inv == pytest.approx(0.123)
    assert loaded.config.right_medulla_ph == pytest.approx(6.75)
    assert loaded.config.breathing_enabled
    assert loaded.config.mouse_offset_m == pytest.approx((0.003, -0.002, 0.004))
    assert np.array_equal(loaded.anatomy_labels, phantom.anatomy_labels)
    assert np.array_equal(loaded.kpl_map_s_inv, phantom.kpl_map_s_inv)
    assert np.array_equal(loaded.kpa_map_s_inv, phantom.kpa_map_s_inv)
    assert np.allclose(
        loaded.pyruvate_inflow.arrival_delay_map_s,
        phantom.pyruvate_inflow.arrival_delay_map_s,
        equal_nan=True,
    )
    assert loaded.name == path.stem


@pytest.mark.parametrize("kernel", ["reference", "optimized"])
def test_dynamic_solver_applies_voxelwise_inflow_arrival_delays(kernel):
    shape = (2, 1, 1)
    zeros = np.zeros(shape)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.02, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 1e15, 1e15),
            ChemicalSpecies("Lactate", 12.0, 1e15, 1e15),
        ),
        initial_concentration_maps={"Pyruvate": zeros.copy(), "Lactate": zeros.copy()},
        initial_spin_density_maps={"Pyruvate": zeros.copy(), "Lactate": zeros.copy()},
        kpl_map_s_inv=zeros.copy(),
        pyruvate_inflow=PyruvateInflow(
            rate_curve_s_inv=TimeCurve(
                (0.0, 0.5), (1.0, 1.0), interpolation="step", outside="zero"
            ),
            delivery_map=np.ones(shape),
            polarization_curve=TimeCurve(
                (0.0, 0.5), (1.0, 1.0), interpolation="step", outside="hold"
            ),
            arrival_delay_map_s=np.asarray([0.0, 0.5]).reshape(shape),
            max_step_s=0.05,
        ),
        nucleus="C13",
    )

    result = BlochSimulator(
        use_parallel=False, dynamic_sequence_kernel=kernel
    ).simulate_dynamic_sequence(
        SequenceProgram((), duration_s=1.2),
        phantom,
        checkpoints_s=(0.25, 0.75, 1.2),
    )
    pyruvate = result.checkpoint_pool_magnetization[:, 0, :, 0, 0, 2]

    assert pyruvate[0, 0] == pytest.approx(0.25, abs=1e-12)
    assert pyruvate[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert pyruvate[1, 0] == pytest.approx(0.5, abs=1e-12)
    assert pyruvate[1, 1] == pytest.approx(0.25, abs=1e-12)
    assert pyruvate[2] == pytest.approx((0.5, 0.5), abs=1e-12)
    assert result.metadata["spatial_inflow_arrival_delays"]


def test_reference_vascular_graph_is_directed_and_reaches_both_kidneys():
    graph = reference_mouse_vascular_graph()
    graph.validate()
    destinations = {edge.destination for edge in graph.edges}

    assert {"left_kidney", "right_kidney", "brain", "liver"} <= destinations
    assert any(edge.kind == "arterial" for edge in graph.edges)
    assert any(edge.kind == "venous" for edge in graph.edges)


@pytest.mark.parametrize(
    "compound, expected_first_pool",
    [
        ("pyruvate", "Pyruvate"),
        ("lactate", "Lactate"),
        ("z_ompd", "Z-OMPD C5"),
        ("contrast_agent", "Contrast agent"),
    ],
)
def test_injection_compound_presets_build_transportable_dynamic_pools(
    compound, expected_first_pool
):
    phantom = MousePerfusionConfig(
        shape=(24, 16, 48), injection_compound=compound
    ).build()

    assert phantom.pools[0].name == expected_first_pool
    assert phantom.metadata["injection_compound"] == compound
    assert np.any(phantom.pyruvate_inflow.delivery_map > 0)
    if compound != "pyruvate":
        assert not np.any(phantom.kpl_map_s_inv)


def test_injected_concentration_and_organ_curves_follow_delayed_bolus():
    config = MousePerfusionConfig(shape=(24, 16, 48))
    phantom = config.build()
    start, end = phantom.preview_time_bounds_s()
    times = np.linspace(start, end, 21)

    frames = [phantom.injected_concentration_map(time) for time in times]
    assert np.max(frames[0]) == pytest.approx(0.0)
    assert max(float(np.max(frame)) for frame in frames[1:]) > 0.0
    curves = phantom.organ_bolus_curves(times)
    assert set(curves) == {"Brain", "Heart", "Liver", "Kidneys", "Bladder"}
    assert all(values.shape == times.shape for values in curves.values())
    assert all(np.max(values) > 0 for values in curves.values())


def test_renal_excretion_moves_injected_dose_from_kidneys_to_bladder():
    config = MousePerfusionConfig(shape=(24, 16, 48))
    phantom = config.build()
    labels = phantom.anatomy_labels
    bladder = labels == 12
    ureters = np.isin(labels, (13, 14))
    kidneys = np.isin(labels, (2, 3, 4, 5))
    assert np.any(bladder) and np.any(ureters)
    assert not np.any(phantom.pyruvate_inflow.delivery_map[bladder | ureters])

    volume = phantom.voxel_volume_m3
    _, end = phantom.preview_time_bounds_s()
    early = phantom.injected_concentration_map(config.injection_start_s + 5.0)
    late = phantom.injected_concentration_map(end)
    assert np.sum(early[bladder]) == pytest.approx(0.0)
    assert np.sum(late[bladder]) * volume > 0.03 * config.injection_amount_umol * 1e-6
    assert np.sum(late[kidneys]) < 0.01 * np.sum(late[bladder])

    # Once the vascular bolus has passed, excreted tracer is conserved.
    totals = [
        np.sum(phantom.injected_concentration_map(time)) * volume
        for time in (end - 40.0, end - 20.0, end)
    ]
    renal_and_urinary = bladder | ureters | kidneys
    excreted = [
        np.sum(phantom.injected_concentration_map(time)[renal_and_urinary])
        for time in (end - 20.0, end)
    ]
    assert excreted[1] * volume == pytest.approx(totals[2], rel=0.05)
    assert totals[1] == pytest.approx(totals[2], rel=0.02)

    bladder_voxel = tuple(np.argwhere(bladder)[0])
    pools = phantom.preview_pool_curves_at(bladder_voxel, np.array([end]))
    assert pools["Pyruvate"][-1] + pools["Lactate"][-1] == pytest.approx(
        late[bladder_voxel], rel=0.02
    )


def test_renal_excretion_can_be_disabled_and_legacy_configs_keep_it_off():
    config = MousePerfusionConfig(shape=(24, 16, 48), renal_excretion_s_inv=0.0)
    phantom = config.build()
    _, end = phantom.preview_time_bounds_s()
    frame = phantom.injected_concentration_map(end)
    assert not np.any(frame[np.isin(phantom.anatomy_labels, (12, 13, 14))])

    legacy = MousePerfusionConfig().to_dict()
    del legacy["renal_excretion_s_inv"]
    assert MousePerfusionConfig.from_dict(legacy).renal_excretion_s_inv == 0.0
    with pytest.raises(ValueError, match="renal excretion"):
        MousePerfusionConfig(ureter_transit_s=-1.0).validate()


def test_voxel_preview_curves_include_configured_conversion_products():
    phantom = MousePerfusionConfig(shape=(24, 16, 48)).build()
    start, end = phantom.preview_time_bounds_s()
    times = np.linspace(start, end, 400)
    renal_voxel = tuple(np.argwhere(phantom.anatomy_labels == 2)[0])
    liver_voxel = tuple(np.argwhere(phantom.anatomy_labels == 6)[0])

    renal = phantom.preview_pool_curves_at(renal_voxel, times)
    liver = phantom.preview_pool_curves_at(liver_voxel, times)

    assert np.max(renal["Pyruvate"]) > 0
    assert np.max(renal["Lactate"]) > 0
    assert np.max(renal["Alanine"]) == pytest.approx(0.0)
    assert np.max(liver["Alanine"]) > 0


def test_heart_and_brain_have_literature_kinetics_and_legacy_configs_keep_them_off():
    config = MousePerfusionConfig(shape=(24, 16, 48))
    phantom = config.build()
    labels = phantom.anatomy_labels
    heart = labels == 8
    brain = labels == 11
    np.testing.assert_allclose(phantom.kpl_map_s_inv[heart], config.heart_kpl_s_inv)
    np.testing.assert_allclose(phantom.kpa_map_s_inv[heart], config.heart_kpa_s_inv)
    np.testing.assert_allclose(phantom.kpl_map_s_inv[brain], config.brain_kpl_s_inv)
    delivery = phantom.pyruvate_inflow.delivery_map
    background = labels == 1
    assert np.mean(delivery[heart]) > np.mean(delivery[brain])
    assert np.mean(delivery[heart]) > np.mean(delivery[background])

    start, end = phantom.preview_time_bounds_s()
    times = np.linspace(start, end, 300)
    heart_curves = phantom.preview_pool_curves_at(tuple(np.argwhere(heart)[0]), times)
    assert np.max(heart_curves["Lactate"]) > 0
    assert np.max(heart_curves["Alanine"]) > 0

    legacy = config.to_dict()
    for key in (
        "heart_perfusion_weight",
        "heart_kpl_s_inv",
        "heart_kpa_s_inv",
        "brain_kpl_s_inv",
        "brain_kpa_s_inv",
    ):
        del legacy[key]
    restored = MousePerfusionConfig.from_dict(legacy)
    assert restored.heart_perfusion_weight == restored.background_perfusion_weight
    assert restored.heart_kpl_s_inv == restored.brain_kpl_s_inv == 0.0
    with pytest.raises(ValueError, match="kPL and kPA"):
        MousePerfusionConfig(brain_kpl_s_inv=-0.1).validate()


@pytest.mark.parametrize("kernel", ["reference", "optimized"])
def test_mouse_sequence_solver_converts_pyruvate_to_alanine_in_liver(kernel):
    phantom = MousePerfusionConfig(shape=(24, 16, 48)).build()
    result = BlochSimulator(
        use_parallel=False, dynamic_sequence_kernel=kernel
    ).simulate_dynamic_sequence(
        SequenceProgram((), duration_s=3.0),
        phantom,
        checkpoints_s=(3.0,),
    )

    alanine = result.checkpoint_pool_magnetization[0, 2, ..., 2]
    liver = phantom.anatomy_labels == 6
    converting = phantom.kpa_map_s_inv > 0
    assert np.max(alanine[liver]) > 0
    assert np.allclose(alanine[~converting], 0.0, atol=1e-12)
    assert result.metadata["additional_product_conversion"]


def test_mouse_type_opens_nonblocking_editor_and_installs_result():
    app = QApplication.instance() or QApplication([])
    creator = PhantomCreatorWidget()
    creator.type_combo.setCurrentText(creator.MOUSE_PERFUSION_TYPE)
    phantom = MousePerfusionConfig(shape=(24, 16, 48)).build()
    dialog = MagicMock()
    dialog.get_phantom.return_value = phantom
    dialog.field_strength = MagicMock()

    with patch(
        "blochsimulator.phantom_widget.MousePhantomDesignerDialog",
        return_value=dialog,
    ):
        creator.create_phantom()

    assert creator.current_phantom is None
    assert creator._retained_mouse_designer_dialogs == [dialog]
    dialog.open.assert_called_once_with()
    dialog.accepted.connect.call_args.args[0]()
    assert creator.current_phantom is phantom
    assert creator.type_combo.currentText() == creator.MOUSE_PERFUSION_TYPE
    assert not creator.edit_btn.isHidden()
    creator.close()
    app.processEvents()


def test_mouse_designer_exposes_animation_curves_and_hd_anatomy_controls():
    app = QApplication.instance() or QApplication([])
    dialog = MousePhantomDesignerDialog(config=MousePerfusionConfig(shape=(24, 16, 48)))

    tabs = [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())]
    assert "Anatomy" in tabs
    assert "Injected perfusion" in tabs
    assert "Anatomy + perfusion" in tabs
    assert dialog.anatomy_resolution.findData("hd") >= 0
    assert dialog.anatomy_resolution.currentData() == "hd"
    assert dialog.combined_anatomy_resolution.currentData() == "hd"
    assert dialog.animation_speed.value() == pytest.approx(0.5)
    assert dialog.perfusion_animation_speed.value() == pytest.approx(0.5)
    assert "alpha" in dialog.bolus_alpha.toolTip()
    assert "beta" in dialog.bolus_beta.toolTip()
    assert [spin.value() for spin in dialog.mouse_size_spins] == pytest.approx(
        [28.0, 28.0, 90.0]
    )
    assert [spin.value() for spin in dialog.mouse_offset_spins] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    for spins in (
        dialog.matrix_spins,
        dialog.fov_spins,
        dialog.mouse_size_spins,
        dialog.mouse_offset_spins,
    ):
        layout = spins[0].parentWidget().layout()
        assert isinstance(layout, QGridLayout)
        for axis, spin in enumerate(spins):
            assert layout.getItemPosition(layout.indexOf(spin)) == (
                0,
                2 * axis + 1,
                1,
                1,
            )
    assert dialog.animation_time.value() < dialog.config.injection_start_s
    dialog.anatomy_resolution.setCurrentIndex(dialog.anatomy_resolution.findData("hd"))
    assert dialog.anatomy_preview.items[0][0].image.shape == (16 * 8, 24 * 8)
    axial_item = dialog.anatomy_preview.items[0][0]
    assert axial_item.transform().m11() * axial_item.image.shape[0] == pytest.approx(
        dialog.config.fov_m[1]
    )
    assert axial_item.transform().m22() * axial_item.image.shape[1] == pytest.approx(
        dialog.config.fov_m[0]
    )
    dialog.perfusion_animation_slider.setValue(400)
    assert dialog.animation_slider.value() == 400
    dialog._animation_slider_changed(500)
    assert dialog.perfusion_preview.perfusion.max() > 0
    assert dialog.organ_curve_plot.listDataItems()
    assert dialog._selected_voxel is not None
    assert "Voxel" in dialog.voxel_spectrum_label.text()

    axial_image = dialog.combined_preview.items[0][0]
    axial_image.clicked.emit(
        float(axial_image.image.shape[0] / 2),
        float(axial_image.image.shape[1] / 2),
    )
    assert "Voxel" in dialog.voxel_spectrum_label.text()

    dialog.animation_time.setValue(3.0)
    combined_anatomy = dialog.combined_preview.items[0][0].image
    combined_perfusion = dialog.combined_preview.items[0][1].image
    assert combined_perfusion.shape == combined_anatomy.shape
    assert np.max(combined_perfusion) > 0

    renal_voxel = tuple(np.argwhere(dialog.phantom.anatomy_labels == 2)[0])
    dialog._voxel_selected(renal_voxel)
    spectrum_times, spectrum_curves = dialog._voxel_pool_curves
    peak_index = int(np.argmax(spectrum_curves["Lactate"]))
    dialog.animation_time.setValue(float(spectrum_times[peak_index]))
    assert "Lactate" in dialog.voxel_spectrum_label.text()
    assert len(dialog.voxel_spectrum_plot.listDataItems()) >= 2

    dialog.injection_concentration.setValue(10.0)
    dialog.injection_volume.setValue(50.0)
    assert dialog._live_preview_timer.isActive()
    assert "0.5 µmol" in dialog.injection_dose_label.text()
    QTest.qWait(400)
    assert dialog.phantom.config.injection_amount_umol == pytest.approx(0.5)

    dialog.relaxation_spins["injected"][2].setValue(15.0)
    dialog.transverse_decay.setCurrentIndex(dialog.transverse_decay.findData("t2"))
    QTest.qWait(400)
    config = dialog.phantom.config
    assert config.injected_t2_star_s == pytest.approx(0.015)
    assert config.transverse_decay == "t2"
    assert dialog.relaxation_spins["metabolite"][0].isEnabled()

    dialog.close()
    app.processEvents()


def test_mouse_slice_preview_preserves_voxel_marker_when_perfusion_is_rendered():
    preview = MouseSlicePreviewWidget(show_anatomy=False)
    shape = (24, 16, 48)
    preview.set_anatomy(np.ones(shape, dtype=np.uint8))
    preview.set_selected_voxel((12, 8, 24))

    preview.set_perfusion(np.ones(shape, dtype=float))

    for marker in preview.selection_items:
        assert marker.data["x"].shape == (1,)
        assert marker.data["y"].shape == (1,)

    preview.close()


def _small_z_ompd_config(**overrides):
    values = dict(injection_compound="z_ompd", shape=(16, 12, 32))
    values.update(overrides)
    return MousePerfusionConfig(**values)


def test_z_ompd_shifts_follow_grashei_titration_per_voxel():
    from blochsimulator.mouse_phantom import z_ompd_shift_ppm

    config = _small_z_ompd_config()
    phantom = config.build()
    c5, c1 = phantom.pools
    reference = config.spectral_reference_ppm

    # Grashei et al.: C5 and C1 at pH 7.4 relative to 163.5 ppm urea.
    assert c5.chemical_shift_ppm + reference == pytest.approx(178.874, abs=1e-3)
    assert c1.chemical_shift_ppm + reference == pytest.approx(171.584, abs=1e-3)
    assert z_ompd_shift_ppm("C5", 6.545) == pytest.approx(163.5 + 12.4801 + 1.64875)
    assert phantom.pyruvate_inflow.colabelled_pools == (c1.name,)
    assert phantom.metadata["ph_model"] == "z_ompd_grashei_titration_shift_maps"

    labels = phantom.anatomy_labels
    for label, ph in (
        (2, config.left_cortex_ph),
        (3, config.left_medulla_ph),
        (12, config.urine_ph),
    ):
        mask = labels == label
        assert np.allclose(phantom.ph_map[mask], ph)
        absolute_c5 = (
            c5.chemical_shift_ppm
            + reference
            + phantom.chemical_shift_maps_ppm[c5.name][mask]
        )
        absolute_c1 = (
            c1.chemical_shift_ppm
            + reference
            + phantom.chemical_shift_maps_ppm[c1.name][mask]
        )
        assert np.allclose(absolute_c5, z_ompd_shift_ppm("C5", ph))
        assert np.allclose(absolute_c1, z_ompd_shift_ppm("C1", ph))
    assert np.allclose(phantom.ph_map[phantom.arterial_mask], config.blood_ph)
    cortex = phantom.chemical_shift_maps_ppm[c5.name][labels == 2].mean()
    medulla = phantom.chemical_shift_maps_ppm[c5.name][labels == 3].mean()
    assert cortex - medulla == pytest.approx(0.566, abs=0.01)


def test_z_ompd_c1_receives_the_same_inflow_as_c5():
    phantom = _small_z_ompd_config(shape=(8, 8, 16)).build()
    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        SequenceProgram((), duration_s=6.0),
        phantom,
        checkpoints_s=(3.0, 6.0),
    )
    c5 = result.checkpoint_pool_magnetization[:, 0, ..., 2]
    c1 = result.checkpoint_pool_magnetization[:, 1, ..., 2]
    assert np.max(c5) > 0
    np.testing.assert_allclose(c1, c5, rtol=1e-10, atol=1e-15)


def test_legacy_z_ompd_phantom_gains_titration_model_on_reload():
    config = _small_z_ompd_config()
    legacy = config.build()
    legacy.pools = (
        ChemicalSpecies("Z-OMPD C5", 3.0, 25.0, 0.3, t2_star=0.3),
        ChemicalSpecies("Z-OMPD C1 reference", 0.4, 25.0, 0.3, t2_star=0.3),
    )
    legacy.chemical_shift_maps_ppm = None
    legacy.pyruvate_inflow.colabelled_pools = ()
    legacy.metadata["ph_model"] = "z_ompd_ground_truth_context"
    old_config = config.to_dict()
    del old_config["blood_ph"], old_config["urine_ph"]
    legacy.metadata["mouse_perfusion_config"] = old_config

    restored = MousePerfusionPhantom.from_dynamic_phantom(legacy)

    assert restored.config.blood_ph == pytest.approx(config.background_ph)
    assert restored.config.urine_ph == pytest.approx(config.background_ph)
    assert restored.pools[0].chemical_shift_ppm == pytest.approx(
        178.874 - config.spectral_reference_ppm, abs=1e-3
    )
    assert restored.pyruvate_inflow.colabelled_pools == ("Z-OMPD C1 reference",)
    assert set(restored.chemical_shift_maps_ppm) == {
        "Z-OMPD C5",
        "Z-OMPD C1 reference",
    }


def test_mouse_config_dose_and_transverse_decay():
    config = MousePerfusionConfig(
        injection_concentration_mm=80.0, injection_volume_ul=250.0
    )
    assert config.injection_amount_umol == pytest.approx(20.0)
    assert config.solver_t2_s(0.3, 0.02) == pytest.approx(0.02)
    assert MousePerfusionConfig(transverse_decay="t2").solver_t2_s(
        0.3, 0.02
    ) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        MousePerfusionConfig(injected_t2_s=0.01, injected_t2_star_s=0.02).validate()
    with pytest.raises(ValueError):
        MousePerfusionConfig(transverse_decay="bogus").validate()


def test_mouse_config_migrates_legacy_amount_and_relaxation():
    legacy = MousePerfusionConfig().to_dict()
    for key in (
        "injection_concentration_mm",
        "injection_volume_ul",
        "injected_t1_s",
        "injected_t2_s",
        "injected_t2_star_s",
        "metabolite_t1_s",
        "metabolite_t2_s",
        "metabolite_t2_star_s",
        "transverse_decay",
    ):
        legacy.pop(key)
    legacy.update(
        injection_amount_umol=0.25,
        pyruvate_t1_s=30.0,
        pyruvate_t2_s=0.4,
        lactate_t1_s=20.0,
        lactate_t2_s=0.2,
    )
    config = MousePerfusionConfig.from_dict(legacy)
    assert config.injection_amount_umol == pytest.approx(0.25)
    assert config.injected_t1_s == pytest.approx(30.0)
    assert config.metabolite_t1_s == pytest.approx(20.0)
    # Legacy phantoms decayed with T2; keep that behaviour on load.
    assert config.solver_t2_s(config.injected_t2_s, config.injected_t2_star_s) == (
        pytest.approx(0.4)
    )


def test_mouse_phantom_pools_use_t2_star_by_default():
    phantom = MousePerfusionPhantom.from_config(MousePerfusionConfig(shape=(8, 8, 12)))
    for pool in phantom.pools:
        assert pool.t2 == pytest.approx(pool.t2_star)
