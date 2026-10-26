import pytest

from blochsimulator.mouse_phantom import MousePerfusionConfig
from blochsimulator.phantom_design import (
    PhantomDesign,
    ShapeDefinition,
    SpectralPeakDefinition,
)
from blochsimulator.phantom_loop import phantom_loop_parameters, phantom_with_parameter


def _design():
    return PhantomDesign(
        name="Loop phantom",
        shape=(4, 4, 1),
        fov_m=(0.02, 0.02, 0.005),
        shapes=[
            ShapeDefinition(
                name="Object",
                kind="box",
                size=(0.8, 0.8, 1.0),
                t1_s=1.2,
                peaks=[
                    SpectralPeakDefinition("Water", 1.0, 0.0, 0.020),
                    SpectralPeakDefinition("Fat", 0.25, -3.4, 0.010, t1_s=0.4),
                ],
            )
        ],
    )


def test_mouse_phantom_exposes_perfusion_parameters_and_rebuilds_with_change():
    phantom = MousePerfusionConfig(shape=(12, 12, 16)).build()
    phantom.metadata["sequence_reference_ppm"] = 171.0
    parameters = {item.key: item for item in phantom_loop_parameters(phantom)}

    delay = parameters["mouse.circulation_delay_s"]
    assert delay.label == "Circulation delay"
    assert delay.unit == "s"
    assert delay.value == pytest.approx(MousePerfusionConfig().circulation_delay_s)
    assert "mouse.field_strength_t" not in parameters
    assert "mouse.shape" not in parameters

    rebuilt = phantom_with_parameter(phantom, delay.key, 4.5)

    config = rebuilt.metadata["mouse_perfusion_config"]
    assert config["circulation_delay_s"] == pytest.approx(4.5)
    assert tuple(config["shape"]) == (12, 12, 16)
    assert rebuilt.metadata["sequence_reference_ppm"] == pytest.approx(171.0)
    # The source phantom keeps its own configuration.
    assert phantom.metadata["mouse_perfusion_config"][
        "circulation_delay_s"
    ] == pytest.approx(delay.value)


def test_designed_phantom_exposes_shape_and_peak_parameters():
    phantom = _design().build()
    parameters = {item.key: item for item in phantom_loop_parameters(phantom)}

    assert parameters["design.shape[0].t1_s"].label == "Object: T1"
    assert parameters["design.shape[0].peak[0].t1_s"].value == pytest.approx(1.2)
    assert parameters["design.shape[0].peak[1].t1_s"].value == pytest.approx(0.4)
    assert "design.default_kpl_s_inv" not in parameters

    rebuilt = phantom_with_parameter(phantom, "design.shape[0].peak[0].t2_star_s", 0.05)

    restored = PhantomDesign.from_phantom(rebuilt)
    assert restored.shapes[0].peaks[0].t2_star_s == pytest.approx(0.05)
    assert restored.shapes[0].peaks[1].t2_star_s == pytest.approx(0.010)


def test_unknown_phantom_loop_parameter_is_rejected():
    phantom = _design().build()
    with pytest.raises(KeyError):
        phantom_with_parameter(phantom, "design.shape[0].peaks", 1.0)
    with pytest.raises(KeyError):
        phantom_with_parameter(phantom, "mouse.field_strength_t", 3.0)
    assert phantom_loop_parameters(None) == []


def test_plain_phantom_loops_scale_relaxation_maps():
    import numpy as np

    from blochsimulator.phantom import PhantomFactory

    phantom = PhantomFactory.shepp_logan_2d(16)
    parameters = {
        parameter.key: parameter for parameter in phantom_loop_parameters(phantom)
    }
    assert {"relax.t1_map", "relax.t2_map"} <= set(parameters)
    reference = parameters["relax.t1_map"].value
    original = phantom.t1_map.copy()

    changed = phantom_with_parameter(phantom, "relax.t1_map", 2 * reference)

    np.testing.assert_allclose(changed.t1_map, 2 * original)
    np.testing.assert_array_equal(phantom.t1_map, original)
    with pytest.raises(ValueError):
        phantom_with_parameter(phantom, "relax.t2_map", 0.0)


def test_pool_phantoms_loop_over_pool_relaxation_without_mutation():
    from blochsimulator.spectral_phantom import SpectralPhantomFactory

    phantom = SpectralPhantomFactory.brain_mrs_voxel()
    parameters = {
        parameter.key: parameter for parameter in phantom_loop_parameters(phantom)
    }
    assert "relax.pool[0].t1" in parameters and "relax.pool[0].t2" in parameters
    original_t2 = phantom.species[0].t2

    changed = phantom_with_parameter(phantom, "relax.pool[0].t2", 0.5 * original_t2)

    assert changed.species[0].t2 == pytest.approx(0.5 * original_t2)
    assert changed.species[0].t2_star <= changed.species[0].t2
    assert phantom.species[0].t2 == original_t2


def test_h1_mouse_loops_over_tissue_water_relaxation():
    mouse = MousePerfusionConfig(shape=(8, 8, 8), nucleus="H1").build()
    keys = {parameter.key for parameter in phantom_loop_parameters(mouse)}
    assert any(key.startswith("relax.pool[") for key in keys)
