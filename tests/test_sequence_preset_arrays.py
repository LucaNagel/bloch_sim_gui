import numpy as np

from blochsimulator.ui.sequence_presets import read_preset_file, write_preset_file


def test_preset_file_roundtrips_numpy_arrays(tmp_path):
    wave = np.array([1 + 2j, 3 - 1j])
    preset = {
        "format": "blochsimulator-sequence-preset",
        "rf_pulse": {"waveform_hz": wave, "x": np.arange(3), "n": np.float64(2.5)},
    }
    path = tmp_path / "p.json"
    write_preset_file(path, preset)
    loaded = read_preset_file(path)
    np.testing.assert_allclose(loaded["rf_pulse"]["waveform_hz"], wave)
    np.testing.assert_array_equal(loaded["rf_pulse"]["x"], np.arange(3))
    assert loaded["rf_pulse"]["n"] == 2.5
