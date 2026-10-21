"""Absolute-polarization authoring keeps the solver's thermal-relative contract."""

from dataclasses import replace

import numpy as np
import pytest

from blochsimulator.dynamic_phantom import TimeCurve
from blochsimulator.mouse_phantom import MousePerfusionConfig
from blochsimulator.phantom_design import (
    PhantomDesign,
    ShapeDefinition,
    SpectralPeakDefinition,
)
from blochsimulator.sequence import (
    ADCEvent,
    ReceiverCalibration,
    ReceiverNoise,
    SequenceProgram,
    SequenceSimulationResult,
)
from blochsimulator.sequence.receiver import (
    apply_receiver_noise,
    ideal_fid_amplitude,
    receiver_acquisition_metadata,
)
from blochsimulator.sequence.signal_units import signal_display
from blochsimulator.sequence.snr_budget import snr_budget
from blochsimulator.ui.mouse_phantom_designer import MousePhantomDesignerDialog
from blochsimulator.ui.phantom_designer import SpectralPhantomDesignerDialog
from blochsimulator.units import (
    polarization_percent_to_relative,
    relative_polarization_to_percent,
    thermal_polarization,
)


@pytest.mark.parametrize("nucleus", ["H1", "C13", "P31", "F19", "Na23"])
@pytest.mark.parametrize("temperature", [293.15, 310.15])
def test_percent_conversion_round_trip(nucleus, temperature):
    percentages = np.array([0.0, 0.001, 20.0, 100.0])
    relative = polarization_percent_to_relative(percentages, nucleus, 7.0, temperature)
    np.testing.assert_allclose(
        relative_polarization_to_percent(relative, nucleus, 7.0, temperature),
        percentages,
    )
    assert relative_polarization_to_percent(
        1.0, nucleus, 7.0, temperature
    ) == pytest.approx(100 * thermal_polarization(nucleus, 7.0, temperature))


@pytest.mark.parametrize("value", [-1, 100.01, np.nan, np.inf])
def test_invalid_absolute_polarization_rejected(value):
    with pytest.raises(ValueError, match="0 and 100"):
        polarization_percent_to_relative(value, "C13", 7, 310.15)


def _design():
    return PhantomDesign(
        shape=(2, 2, 2),
        nucleus="C13",
        field_strength_t=7.0,
        temperature_k=310.15,
        supersampling_enabled=False,
        shapes=[
            ShapeDefinition(
                "Dose",
                kind="box",
                size=(1, 1, 1),
                peaks=[
                    SpectralPeakDefinition(
                        "Pyruvate", amplitude=80.0, initial_polarization=1.0
                    ),
                    SpectralPeakDefinition("Lactate", amplitude=0.0),
                ],
            )
        ],
    )


def test_designer_stores_temperature_and_shows_thermal_percent():
    design = _design()
    dialog = SpectralPhantomDesignerDialog(design=design)
    assert dialog.temperature_c.value() == pytest.approx(37.0)
    pth = thermal_polarization("C13", 7, 310.15)
    assert f"{100 * pth:.4g} %" in dialog.thermal_polarization_info.text()
    dialog.field_strength_t.setValue(3)
    dialog.temperature_c.setValue(20)
    dialog.nucleus.setCurrentIndex(dialog.nucleus.findData("H1"))
    dialog._sync_global()
    pth = thermal_polarization("H1", 3, 293.15)
    assert f"{100 * pth:.4g} %" in dialog.thermal_polarization_info.text()
    assert "H1" in dialog.thermal_polarization_info.text()
    # Polarization stays authored in multiples of thermal equilibrium.
    assert design.shapes[0].peaks[0].initial_polarization == 1.0
    assert design.temperature_k == pytest.approx(293.15)
    assert design.build().metadata["temperature_k"] == pytest.approx(293.15)
    dialog.close()


def test_legacy_design_round_trip_preserves_initial_maps():
    data = _design().to_dict()
    data.pop("temperature_k")
    data["shapes"][0]["initial_mz"] = 10000.123456789
    data["shapes"][0]["peaks"][0]["initial_polarization"] = 12345.6789012345
    design = PhantomDesign.from_dict(data)
    before = design.build()
    dialog = SpectralPhantomDesignerDialog(design=design)
    dialog._sync_global()
    after = PhantomDesign.from_dict(design.to_dict()).build()
    for name in before.initial_mz_maps:
        np.testing.assert_array_equal(
            after.initial_mz_maps[name], before.initial_mz_maps[name]
        )
        np.testing.assert_array_equal(
            after.concentration_maps[name], before.concentration_maps[name]
        )
    assert design.temperature_k == 310.15
    dialog.close()


def test_mouse_recipe_conserves_amount_and_polarized_dose():
    config = MousePerfusionConfig(
        shape=(8, 8, 16),
        injection_concentration_mm=80,
        injection_volume_ul=200,
        injection_polarization=polarization_percent_to_relative(20, "C13", 7, 310.15),
    )
    phantom = config.build()
    assert config.injection_amount_umol == 16
    pth = thermal_polarization("C13", 7, 310.15)
    assert ideal_fid_amplitude(phantom, "voxel_volume") * pth == pytest.approx(3.2e-6)
    restored = MousePerfusionConfig.from_dict(config.to_dict())
    assert restored.injection_concentration_mm == 80
    assert restored.injection_volume_ul == 200
    assert restored.temperature_k == 310.15
    assert restored.to_dict() == config.to_dict()


@pytest.mark.parametrize(
    "recipe",
    [
        dict(injection_concentration_mm=-1, injection_volume_ul=200),
        dict(injection_concentration_mm=80, injection_volume_ul=np.nan),
        dict(temperature_k=0.0),
        dict(injection_polarization=-1.0),
    ],
)
def test_invalid_mouse_recipe_rejected(recipe):
    with pytest.raises(ValueError):
        MousePerfusionConfig(**recipe).validate()


def test_mouse_dialog_keeps_unedited_exact_polarization():
    config = MousePerfusionConfig(
        shape=(8, 8, 16),
        injection_concentration_mm=10,
        injection_volume_ul=25,
        injection_polarization=12345.6789012345,
    )
    dialog = MousePhantomDesignerDialog(config=config)
    restored = dialog._read_config()
    assert restored.injection_amount_umol == pytest.approx(0.25)
    assert restored.injection_polarization == config.injection_polarization
    dialog.injection_concentration.setValue(80)
    dialog.injection_volume.setValue(200)
    dialog.injection_polarization.setValue(20)
    restored = dialog._read_config()
    assert restored.injection_amount_umol == pytest.approx(16)
    assert "3.2 µmol fully polarized" in dialog.injection_dose_label.text()
    assert restored.injection_polarization == pytest.approx(
        0.2 / thermal_polarization("C13", 7, 310.15)
    )
    assert "× thermal" in dialog.injection_polarization_thermal.text()
    dialog.close()


def test_receiver_uses_explicit_phantom_temperature_for_calibration_and_display():
    design = _design()
    design.temperature_k = 293.15
    phantom = design.build()
    program = SequenceProgram((ADCEvent(0, 16, 1e-4),), duration_s=0.002)
    context = receiver_acquisition_metadata(
        program,
        nucleus="C13",
        field_strength_t=7,
        phantom=phantom,
        signal_weighting="voxel_volume",
    )
    assert context["temperature_k"] == 293.15
    result = SequenceSimulationResult(
        signal=np.ones(16, dtype=complex),
        adc_times_s=np.arange(16) * 1e-4,
        final_magnetization=np.zeros((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        metadata={"signal_weighting": "voxel_volume", "receiver_acquisition": context},
    )
    calibration = ReceiverCalibration(200, 2, 2000, 50, "C13", 7, 1e-4)
    noise = ReceiverNoise("calibrated", calibration=calibration)
    measured = apply_receiver_noise(result, noise)
    effective = replace(calibration, target_temperature_k=293.15)
    assert (
        measured.metadata["receiver_noise"]["calibration"]["target_temperature_k"]
        == 293.15
    )
    assert measured.metadata["receiver_noise"]["noise_std_per_quadrature"] == [
        pytest.approx(effective.noise_std(1e-4))
    ]
    assert (
        signal_display(measured, "physical").sample_scale
        == effective.target_thermal_polarization
    )
    sections = snr_budget(noise, phantom=phantom, nucleus="C13", field_strength_t=7)
    units = next(section for section in sections if section.title == "Simulation units")
    assert units.rows[0].number == effective.target_thermal_polarization
    assert calibration.target_temperature_k == 310.15


@pytest.mark.parametrize("family", ["static", "dynamic", "mouse"])
def test_workspace_rebase_preserves_absolute_states_and_authored_metadata(family):
    from blochsimulator.phantom_polarization import rebase_phantom_polarization

    if family == "mouse":
        phantom = MousePerfusionConfig(
            shape=(8, 8, 16), injection_concentration_mm=80, injection_volume_ul=200
        ).build()
    else:
        design = _design()
        if family == "dynamic":
            design.dynamic_enabled = True
            design.pyruvate_inflow_curve = TimeCurve((0, 1), (1, 1), "linear", "zero")
            design.pyruvate_inflow_polarization_curve = TimeCurve(
                (0, 1), (10000, 20000), "linear", "hold"
            )
        phantom = design.build()
    old_pth = thermal_polarization("C13", 7, 310.15)
    old_signal = ideal_fid_amplitude(phantom, "voxel_volume") * old_pth
    density = {
        name: values.copy()
        for name, values in (
            phantom.concentration_maps
            if family == "static"
            else phantom.initial_spin_density_maps
        ).items()
    }
    rebase_phantom_polarization(phantom, field_strength_t=3, nucleus="C13")
    phantom.field_strength = 3
    # The workspace updates these metadata fields after rebasing.
    if family != "mouse":
        phantom.metadata["phantom_design"]["field_strength_t"] = 3
        phantom.metadata["phantom_design"]["nucleus"] = "C13"
        rebuilt = PhantomDesign.from_dict(phantom.metadata["phantom_design"]).build()
    else:
        rebuilt = MousePerfusionConfig.from_dict(
            phantom.metadata["mouse_perfusion_config"]
        ).build()
    pth = thermal_polarization("C13", 3, 310.15)
    assert ideal_fid_amplitude(phantom, "voxel_volume") * pth == pytest.approx(
        old_signal
    )
    assert ideal_fid_amplitude(rebuilt, "voxel_volume") * pth == pytest.approx(
        old_signal
    )
    after_density = (
        phantom.concentration_maps
        if family == "static"
        else phantom.initial_spin_density_maps
    )
    for name in density:
        np.testing.assert_array_equal(after_density[name], density[name])


def test_shared_workspace_field_change_uses_polarization_rebase(tmp_path):
    from unittest.mock import patch
    from PyQt5.QtCore import QSettings
    from blochsimulator.ui.main_window import BlochSimulatorGUI

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    with patch("blochsimulator.ui.main_window.QSettings", return_value=settings):
        window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    window._ensure_phantom_workspace(window.phantom_tab_index)
    design = _design()
    design.shapes[0].peaks[0].initial_polarization = polarization_percent_to_relative(
        20, "C13", 7, 310.15
    )
    phantom = design.build()
    window.phantom_widget.current_phantom = phantom
    window.phantom_widget.phantom_creator.current_phantom = phantom
    window._synchronize_workspace_field_strength(3)
    pth = thermal_polarization(phantom.nucleus, 3, 310.15)
    np.testing.assert_allclose(phantom.initial_mz_maps["Dose: Pyruvate"] * pth, 0.2)
    window._synchronize_workspace_nucleus("H1")
    pth = thermal_polarization("H1", 3, 310.15)
    np.testing.assert_allclose(phantom.initial_mz_maps["Dose: Pyruvate"] * pth, 0.2)
    window.close()


@pytest.mark.parametrize("locale_name, entered", [("C", "12.5 %"), ("de_DE", "12,5 %")])
def test_percent_spin_shows_compact_values_and_accepts_locale_decimal(
    locale_name, entered
):
    from PyQt5.QtCore import QLocale
    from blochsimulator.ui.widgets import PolarizationPercentSpinBox

    spin = PolarizationPercentSpinBox()
    spin.setLocale(QLocale(locale_name))
    spin.setValue(20)
    assert spin.text() == "20 %"
    spin.lineEdit().setText(entered)
    spin.interpretText()
    assert spin.value() == 12.5
    spin.setValue(100 * thermal_polarization("C13", 7, 310.15))
    assert spin.value() > 0
    assert "e-" not in spin.text()
    spin.close()


def test_new_mouse_dialog_starts_with_example_recipe_and_absolute_percent():
    from unittest.mock import patch

    with patch.object(MousePhantomDesignerDialog, "_preview"):
        dialog = MousePhantomDesignerDialog()
    assert dialog.injection_concentration.value() == 80
    assert dialog.injection_volume.value() == 250
    assert dialog.injection_polarization.text() == "20 %"
    assert dialog._read_config().injection_amount_umol == pytest.approx(20)
    assert "4 µmol fully polarized" in dialog.injection_dose_label.text()
    dialog.close()
