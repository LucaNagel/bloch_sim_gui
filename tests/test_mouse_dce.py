"""Gadolinium DCE of the 1H mouse: kinetics, contrast relaxation and solvers."""

import copy
from dataclasses import replace

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator import dynamic_phantom as dynamic_module
from blochsimulator.dynamic_phantom import ContrastRelaxation
from blochsimulator.mouse_dce import (
    ARTERIAL_COMPARTMENT,
    VENOUS_COMPARTMENT,
    _first_order_response,
)
from blochsimulator.mouse_phantom import (
    BLADDER_LABEL,
    MousePerfusionConfig,
    MousePerfusionPhantom,
)
from blochsimulator.sequence import ADCEvent, RFEvent, SequenceProgram

RASTER = 20e-6
TR = 5e-3
SMALL = (8, 8, 14)
# Large enough to resolve the bladder and ureters.
ANATOMY = (16, 16, 26)


def _dce_config(**overrides):
    values = dict(
        shape=SMALL,
        nucleus="H1",
        injection_compound="contrast_agent",
        injection_concentration_mm=50.0,
        injection_volume_ul=50.0,
    )
    values.update(overrides)
    return MousePerfusionConfig(**values)


def _spgr(n_tr, flip_deg=15.0, rf_duration_s=200e-6, adc_s=1.0e-3):
    samples = int(round(rf_duration_s / RASTER))
    amplitude_hz = np.deg2rad(flip_deg) / (2 * np.pi * rf_duration_s)
    events, spoilers = [], []
    for index in range(n_tr):
        start = index * TR
        events.append(RFEvent(start, np.full(samples, amplitude_hz + 0j), RASTER))
        events.append(ADCEvent(start + adc_s, 1, RASTER))
        spoilers.append(start + 2.0e-3)
    return SequenceProgram(
        tuple(events),
        duration_s=n_tr * TR,
        metadata={"definitions": {"IdealSpoilerEndTimes": spoilers}},
    )


def _run(phantom, program, kernel):
    return dynamic_module.simulate_dynamic_sequence(
        program,
        phantom,
        sequence_kernel=kernel,
        simulation_timestep_s=RASTER,
        use_parallel=False,
    )


def test_first_order_response_is_exact_for_piecewise_linear_input():
    times = np.linspace(0.0, 10.0, 41)
    inputs = np.stack([np.ones_like(times), times], axis=1)
    rates = np.array([0.3, 0.0])
    result = _first_order_response(times, inputs, rates)
    np.testing.assert_allclose(
        result[:, 0], (1 - np.exp(-0.3 * times)) / 0.3, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(result[:, 1], 0.5 * times**2, rtol=1e-12)


def test_dce_kinetics_conserve_dose_and_follow_tofts():
    phantom = _dce_config(shape=ANATOMY).build()
    kinetics = phantom.dce_kinetics
    assert kinetics.dose_umol == pytest.approx(2.5)
    assert phantom.config.contrast_dose_mmol_per_kg == pytest.approx(0.1)
    # Whole-body amounts plus the excreted amount equal the dose.
    total = kinetics.body_amounts_umol.sum(axis=1)
    injection_end = (
        phantom.config.injection_start_s + phantom.config.injection_duration_s
    )
    after = kinetics.times_s > injection_end + 0.1
    np.testing.assert_allclose(total[after], 2.5, rtol=1e-12)
    excreted = kinetics.body_amounts_umol[:, 3]
    assert 0.0 < excreted[-1] < 2.5
    urinary = sum(kinetics.urinary_amounts_umol.values())
    np.testing.assert_allclose(urinary, excreted, atol=1e-4 * 2.5)
    # First pass peak, then washout of the plasma.
    arterial = kinetics.arterial_plasma_mm
    assert kinetics.times_s[np.argmax(arterial)] < 5.0
    assert arterial[-1] < 0.1 * arterial.max()

    concentration = kinetics.concentration_mm
    hct = phantom.config.hematocrit
    assert np.max(concentration[:, ARTERIAL_COMPARTMENT]) == pytest.approx(
        (1 - hct) * arterial.max(), rel=1e-2
    )
    assert np.all(concentration[:, VENOUS_COMPARTMENT] >= 0)
    # Intact blood-brain barrier: brain only carries its plasma volume.
    brain = concentration[:, 10]
    vp = phantom.config.dce_tofts_parameters["brain"][2]
    assert brain.max() == pytest.approx(vp * arterial.max(), rel=0.05)
    # Urine accumulates in the bladder.
    bladder = concentration[:, BLADDER_LABEL - 1]
    assert bladder[0] == 0.0 and bladder[-1] > arterial[-1]

    relaxation = phantom.contrast_relaxation
    assert relaxation.n_compartments == 14
    np.testing.assert_allclose(
        relaxation.delta_r1_s_inv, phantom.config.contrast_r1_mm_s_inv * concentration
    )
    np.testing.assert_array_equal(relaxation.compartment_map, kinetics.compartment_map)
    assert np.all((relaxation.compartment_map >= 0) == (phantom.anatomy_labels > 0))
    assert phantom.metadata["contrast_model"].startswith("three_compartment")


def test_dce_previews_show_the_contrast_concentration():
    phantom = _dce_config(shape=ANATOMY).build()
    frame = phantom.injected_concentration_map(30.0)
    expected = phantom.dce_kinetics.concentration_map(30.0)
    np.testing.assert_array_equal(frame, expected)
    assert frame.max() > 0
    curves = phantom.organ_bolus_curves(np.array([0.0, 30.0, 600.0]))
    assert curves["Kidneys"][1] > curves["Brain"][1] > 0
    assert curves["Bladder"][2] > curves["Bladder"][1]
    start, end = phantom.preview_time_bounds_s()
    assert end == pytest.approx(phantom.config.contrast_horizon_s)


def test_dce_is_limited_to_the_h1_contrast_agent_mouse():
    assert _dce_config(nucleus="C13").build().contrast_relaxation is None
    assert _dce_config(injection_compound="pyruvate").build().dce_kinetics is None
    with pytest.raises(ValueError, match="ve \\+ vp"):
        _dce_config(dce_tofts_parameters={"liver": (1.0, 0.8, 0.5)}).validate()
    with pytest.raises(ValueError, match="contrast agent"):
        _dce_config(contrast_agent="unknown").validate()
    # Configurations saved before DCE existed load with the defaults.
    values = _dce_config().to_dict()
    for key in ("contrast_r1_mm_s_inv", "dce_tofts_parameters", "gfr_ml_min"):
        values.pop(key)
    restored = MousePerfusionConfig.from_dict(values)
    restored.validate()
    assert restored.contrast_r1_mm_s_inv == 3.0


def test_dce_phantom_round_trips_through_save(tmp_path):
    phantom = _dce_config().build()
    path = phantom.save(tmp_path / "dce.npz")
    loaded = MousePerfusionPhantom.load(path)
    np.testing.assert_array_equal(
        loaded.contrast_relaxation.delta_r2_s_inv,
        phantom.contrast_relaxation.delta_r2_s_inv,
    )
    np.testing.assert_array_equal(
        loaded.dce_kinetics.concentration_mm, phantom.dce_kinetics.concentration_mm
    )


def test_contrast_relaxation_rejects_converting_pools():
    phantom = MousePerfusionConfig(shape=SMALL).build()
    relaxation = ContrastRelaxation(
        np.zeros(phantom.shape, dtype=np.int32),
        np.array([0.0]),
        np.array([[1.0]]),
        np.array([[1.0]]),
    )
    with pytest.raises(ValueError, match="contrast relaxation"):
        replace(phantom, contrast_relaxation=relaxation)


@pytest.mark.parametrize("kernel", ["native_serial", "reference"])
def test_constant_contrast_equals_shortened_relaxation(kernel):
    pytest.importorskip("blochsimulator.dynamic_spin_major_cy")
    phantom = MousePerfusionConfig(shape=SMALL, nucleus="H1").build()
    delta_r1, delta_r2 = 2.0, 15.0
    contrast = copy.copy(phantom)
    contrast.contrast_relaxation = ContrastRelaxation(
        np.where(phantom.anatomy_labels > 0, 0, -1).astype(np.int32),
        np.array([0.0]),
        np.array([[delta_r1]]),
        np.array([[delta_r2]]),
    )
    shortened = copy.copy(phantom)
    shortened.pools = tuple(
        replace(pool, t1=1 / (1 / pool.t1 + delta_r1), t2=1 / (1 / pool.t2 + delta_r2))
        for pool in phantom.pools
    )
    program = _spgr(80)
    expected = _run(shortened, program, kernel).signal
    actual = _run(contrast, program, kernel).signal
    scale = np.max(np.abs(expected))
    assert np.max(np.abs(actual - expected)) <= 1e-12 * scale
    baseline = _run(phantom, program, kernel).signal
    assert np.max(np.abs(baseline - expected)) > 1e-3 * scale


def test_dce_mouse_spin_major_matches_reference_across_epochs():
    pytest.importorskip("blochsimulator.dynamic_spin_major_cy")
    phantom = _dce_config().build()
    program = _spgr(400)  # 2 s: four epochs during the first pass
    native = _run(phantom, program, "native_serial")
    reference = _run(phantom, program, "reference")
    assert native.metadata["independent_pool_passes"]
    scale = np.max(np.abs(reference.signal))
    assert np.max(np.abs(native.signal - reference.signal)) <= 1e-12 * scale
    np.testing.assert_allclose(
        native.final_pool_magnetization,
        reference.final_pool_magnetization,
        atol=1e-12,
    )
    # The bolus raises the T1-weighted signal.
    baseline = _run(
        MousePerfusionConfig(shape=SMALL, nucleus="H1").build(),
        program,
        "native_serial",
    )
    late = slice(-20, None)
    assert np.mean(np.abs(native.signal[late])) > 1.05 * np.mean(
        np.abs(baseline.signal[late])
    )


def test_spgr_steady_state_matches_the_analytic_signal_with_contrast():
    pytest.importorskip("blochsimulator.dynamic_spin_major_cy")
    phantom = MousePerfusionConfig(shape=SMALL, nucleus="H1").build()
    muscle = phantom.anatomy_labels == 1
    flip = np.deg2rad(15.0)
    adc_s, rf_s = 1.0e-3, 200e-6

    def steady_state(delta_r1, delta_r2):
        contrast = copy.copy(phantom)
        contrast.contrast_relaxation = ContrastRelaxation(
            np.where(muscle, 0, -1).astype(np.int32),
            np.array([0.0]),
            np.array([[delta_r1]]),
            np.array([[delta_r2]]),
        )
        density = copy.copy(contrast)
        density.initial_spin_density_maps = {
            name: np.where(muscle, values, 0.0)
            for name, values in phantom.initial_spin_density_maps.items()
        }
        density.initial_concentration_maps = dict(density.initial_spin_density_maps)
        signal = _run(density, _spgr(300), "native_serial").signal
        return abs(np.asarray(signal).reshape(300, -1)[-1].sum())

    pool = phantom.pools[0]

    def analytic(delta_r1, delta_r2):
        e1 = np.exp(-TR * (1 / pool.t1 + delta_r1))
        echo = adc_s - 0.5 * rf_s
        return (
            np.sin(flip)
            * (1 - e1)
            / (1 - np.cos(flip) * e1)
            * np.exp(-echo * (1 / pool.t2 + delta_r2))
        )

    ratio = steady_state(4.0, 20.0) / steady_state(0.0, 0.0)
    assert ratio == pytest.approx(analytic(4.0, 20.0) / analytic(0.0, 0.0), rel=2e-3)


def test_designer_edits_dce_parameters():
    QApplication.instance() or QApplication([])
    from blochsimulator.ui.mouse_phantom_designer import MousePhantomDesignerDialog

    dialog = MousePhantomDesignerDialog(config=_dce_config(shape=(16, 16, 26)))
    assert not dialog.contrast_group.isHidden()
    assert "mmol/kg" in dialog.injection_dose_label.text()
    dialog.contrast_agent.setCurrentIndex(dialog.contrast_agent.findData("gadobutrol"))
    assert dialog.contrast_r1.value() == 4.5 and not dialog.contrast_r1.isEnabled()
    dialog.tofts_spins["liver"][0].setValue(0.7)
    dialog.gfr.setValue(0.3)
    config = dialog._read_config()
    assert config.contrast_agent == "gadobutrol"
    assert config.dce_tofts_parameters["liver"][0] == pytest.approx(0.7)
    assert config.gfr_ml_min == pytest.approx(0.3)
    dialog._preview()
    assert dialog.phantom.contrast_relaxation is not None

    c13 = MousePhantomDesignerDialog(config=_dce_config(nucleus="C13"))
    assert c13.contrast_group.isHidden()
