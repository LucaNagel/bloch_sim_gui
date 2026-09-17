import numpy as np
import pytest
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
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
from blochsimulator.ui.mouse_phantom_designer import MousePhantomDesignerDialog


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


def test_bolus_enters_at_tail_vein_before_cardiopulmonary_circulation():
    config = MousePerfusionConfig(shape=(24, 16, 48))
    phantom = config.build()
    earliest = np.unravel_index(
        np.nanargmin(phantom.perfusion_arrival_time_s), phantom.shape
    )
    earliest_z = (earliest[2] + 0.5) / phantom.shape[2] * 2.0 - 1.0

    assert earliest_z < -0.8
    assert phantom.anatomy_labels[earliest] == 10
    assert phantom.perfusion_arrival_time_s[earliest] == pytest.approx(
        config.injection_start_s
    )

    early_frame = phantom.injected_concentration_map(
        config.injection_start_s + 0.1 * config.injection_duration_s
    )
    maximum = np.unravel_index(np.argmax(early_frame), phantom.shape)
    maximum_z = (maximum[2] + 0.5) / phantom.shape[2] * 2.0 - 1.0
    assert maximum_z < -0.8


def test_mouse_phantom_round_trip_restores_editable_configuration(tmp_path):
    config = MousePerfusionConfig(
        name="Renal mouse",
        left_cortex_kpl_s_inv=0.123,
        right_medulla_ph=6.75,
        breathing_enabled=True,
    )
    phantom = config.build()
    path = tmp_path / "renal_mouse.npz"
    phantom.save(path)

    loaded = load_any_phantom(path)

    assert isinstance(loaded, MousePerfusionPhantom)
    assert loaded.config.left_cortex_kpl_s_inv == pytest.approx(0.123)
    assert loaded.config.right_medulla_ph == pytest.approx(6.75)
    assert loaded.config.breathing_enabled
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
    assert set(curves) == {"Brain", "Liver", "Kidneys"}
    assert all(values.shape == times.shape for values in curves.values())
    assert all(np.max(values) > 0 for values in curves.values())


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
    assert np.max(alanine[liver]) > 0
    assert np.allclose(alanine[~liver], 0.0, atol=1e-12)
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
    assert dialog.animation_time.value() < dialog.config.injection_start_s
    dialog.anatomy_resolution.setCurrentIndex(dialog.anatomy_resolution.findData("hd"))
    assert dialog.anatomy_preview.items[0][0].image.shape == (16 * 8, 24 * 8)
    dialog.perfusion_animation_slider.setValue(400)
    assert dialog.animation_slider.value() == 400
    dialog._animation_slider_changed(500)
    assert dialog.perfusion_preview.perfusion.max() > 0
    assert dialog.organ_curve_plot.listDataItems()

    dialog.injection_amount.setValue(0.5)
    assert dialog._live_preview_timer.isActive()
    QTest.qWait(400)
    assert dialog.phantom.config.injection_amount_umol == pytest.approx(0.5)

    dialog.close()
    app.processEvents()
