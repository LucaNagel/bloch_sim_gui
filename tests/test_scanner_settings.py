import pytest
from PyQt5.QtCore import QSettings

from blochsimulator.sequence import (
    ScannerParameters,
    load_scanner_parameters,
    make_pulseq_epi,
    save_scanner_parameters,
)


def test_scanner_parameters_round_trip_through_settings(tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    expected = ScannerParameters(
        max_grad_mtm=40.0,
        max_slew_tms=180.0,
        grad_raster_time_s=8e-6,
        rf_raster_time_s=2e-6,
        adc_raster_time_s=0.2e-6,
        block_duration_raster_s=8e-6,
        rf_ringdown_time_s=25e-6,
        rf_dead_time_s=80e-6,
        adc_dead_time_s=12e-6,
    )

    save_scanner_parameters(settings, expected)
    settings.sync()

    assert load_scanner_parameters(settings) == expected
    assert float(settings.value("scanner/max_grad_mtm")) == pytest.approx(40.0)
    assert float(settings.value("scanner/grad_raster_time_us")) == pytest.approx(8.0)


def test_invalid_scanner_setting_falls_back_to_default(tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    settings.setValue("scanner/max_slew_tms", -1.0)

    assert load_scanner_parameters(settings) == ScannerParameters()


def test_generated_pulseq_uses_configured_scanner_limits():
    pytest.importorskip("pypulseq")
    scanner = ScannerParameters(
        max_grad_mtm=40.0,
        max_slew_tms=180.0,
        adc_raster_time_s=0.2e-6,
        rf_ringdown_time_s=25e-6,
        rf_dead_time_s=80e-6,
        adc_dead_time_s=10e-6,
    )

    sequence = make_pulseq_epi(
        matrix=(4, 3),
        sampling_bandwidth_hz=25_000.0,
        repetition_time_s=50e-3,
        scanner_parameters=scanner,
    )

    system = sequence.system
    assert system.max_grad / system.gamma * 1e3 == pytest.approx(40.0)
    assert system.max_slew / system.gamma == pytest.approx(180.0)
    assert system.adc_raster_time == pytest.approx(0.2e-6)
    assert system.rf_ringdown_time == pytest.approx(25e-6)
    assert system.rf_dead_time == pytest.approx(80e-6)
    assert system.adc_dead_time == pytest.approx(10e-6)
    assert sequence.check_timing()[0]


def test_scanner_parameters_reject_unknown_or_non_physical_values():
    with pytest.raises(ValueError, match="max_slew_tms"):
        ScannerParameters(max_slew_tms=0.0)
    with pytest.raises(ValueError, match="unknown scanner parameter"):
        ScannerParameters.from_mapping({"gradient_coil_color": 1.0})
