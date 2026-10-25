import json

import pytest

from blochsimulator.ui.sequence_presets import (
    PRESET_FORMAT,
    STARTUP_PRESET_KEY,
    SequencePresetStore,
    is_sequence_control,
    preset_file_name,
    read_preset_file,
)


def test_preset_controls_leave_out_phantom_probe_and_viewer_state():
    for name in ("flash_flip_angle_deg", "press_repetition_time_ms", "ir_readout"):
        assert is_sequence_control(name)
    for name in (
        "sequence_source",
        "sampling_bandwidth_khz",
        "simulation_timestep_us",
    ):
        assert is_sequence_control(name)
    for name in (
        "object_source",
        "probe_t1_ms",
        "parameter_loop_start",
        "frame_slider",
        "nucleus",
        "field_strength_t",
        "fov_mm",
    ):
        assert not is_sequence_control(name)


def test_preset_store_names_files_and_rejects_foreign_json(tmp_path):
    store = SequencePresetStore(tmp_path)
    assert store.names() == []
    store.save("FLASH 7T: mouse/brain", {"format": PRESET_FORMAT})
    assert store.names() == ["FLASH 7T_ mouse_brain"]
    assert preset_file_name("a/b") == "a_b.json"
    with pytest.raises(ValueError):
        preset_file_name("  ")
    foreign = tmp_path / "other.json"
    foreign.write_text(json.dumps({"format": "project"}))
    with pytest.raises(ValueError):
        read_preset_file(foreign)


def test_gui_sequence_preset_round_trip_and_startup(
    qt_application, tmp_path, monkeypatch
):
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import QWidget

    from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget

    widget = SequenceSimulationWidget()
    widget.sequence_preset_store = SequencePresetStore(tmp_path)
    widget.sequence_source.setCurrentIndex(widget.PRESS_SOURCE)
    widget.press_repetition_time_ms.setValue(1234.0)
    widget.flash_flip_angle_deg.setValue(33.0)
    path = widget.save_sequence_preset("press long TR")
    saved = json.loads(path.read_text())
    assert saved["format"] == PRESET_FORMAT
    assert "object_source" not in saved["controls"]
    assert "receiver_noise" in saved and "rf_pulse" in saved

    other = SequenceSimulationWidget()
    other.sequence_preset_store = SequencePresetStore(tmp_path)
    assert other.sequence_source.currentIndex() == other.FLASH_SOURCE
    other.load_sequence_preset("press long TR")
    assert other.sequence_source.currentIndex() == other.PRESS_SOURCE
    assert other.press_repetition_time_ms.value() == pytest.approx(1234.0)
    assert other.flash_flip_angle_deg.value() == pytest.approx(33.0)
    assert other._press_pulseq_parameters()["repetition_time_s"] == pytest.approx(1.234)
    assert other.program is not None

    # A startup preset replaces the built-in FLASH default on construction.
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    settings.setValue(STARTUP_PRESET_KEY, "press long TR")
    host = QWidget()
    host.app_settings = settings
    monkeypatch.setenv("BLOCH_APP_DIR", str(tmp_path / "app"))
    SequencePresetStore().save("press long TR", saved)
    started = SequenceSimulationWidget(host)
    assert started.sequence_source.currentIndex() == started.PRESS_SOURCE
    assert started.press_repetition_time_ms.value() == pytest.approx(1234.0)
