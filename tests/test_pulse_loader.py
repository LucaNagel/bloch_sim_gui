import hashlib
from pathlib import Path

import numpy as np
import pytest

from blochsimulator import pulse_loader
from blochsimulator.pulse_loader import (
    PulseLibrary,
    load_amp_phase_dat,
    load_pulse,
    load_pulse_from_file,
)
from blochsimulator.simulator import CustomPulse

REPO_ROOT = Path(__file__).parents[1]
PULSE_SUFFIXES = {".exc", ".dat", ".json", ".txt"}


def _pulse_file_digests(directory):
    digests = {}
    for path in directory.rglob("*"):
        if path.suffix.lower() in PULSE_SUFFIXES:
            name = path.relative_to(directory).as_posix()
            digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def test_default_library_lists_shipped_pulses_from_source():
    library = PulseLibrary()

    assert library.base_path == Path(pulse_loader.__file__).parent / "rfpulses"
    assert library.list_pulses() == [
        "AHP_5ms_Hsn",
        "bruker/13C_Ultimate_SPSP_Pulse_QuEMRT",
    ]


def test_library_names_load_through_load_pulse_and_custom_pulse():
    b1, time, metadata = load_pulse("bruker/13C_Ultimate_SPSP_Pulse_QuEMRT")

    assert b1.size == 2473
    assert metadata.duration == pytest.approx(24.72e-3)
    assert time[-1] < metadata.duration

    pulse = CustomPulse("bruker/13C_Ultimate_SPSP_Pulse_QuEMRT")
    np.testing.assert_array_equal(pulse.b1, b1)


def test_packaged_pulse_library_matches_repository_rfpulses():
    packaged = _pulse_file_digests(REPO_ROOT / "src" / "blochsimulator" / "rfpulses")
    repository = _pulse_file_digests(REPO_ROOT / "rfpulses")

    assert packaged == repository


@pytest.mark.parametrize(
    "filename, expected_duration_s",
    [
        ("pulse_2.5ms.dat", 2.5e-3),
        ("HS_0.25MS_adiabatic.dat", 0.25e-3),
        ("AHP_5ms_Hsn.dat", 5e-3),
    ],
)
def test_amp_phase_dat_reads_duration_from_filename(
    tmp_path, filename, expected_duration_s
):
    path = tmp_path / filename
    np.savetxt(path, [[0.5, 0.0], [1.0, 90.0], [0.5, 180.0], [0.25, 0.0]])

    b1, time, metadata = load_amp_phase_dat(path)

    assert metadata.duration == pytest.approx(expected_duration_s)
    np.testing.assert_allclose(np.diff(time), expected_duration_s / 4)


def test_library_ahp_percent_amplitudes_peak_at_max_b1():
    b1, time, metadata = PulseLibrary().load("AHP_5ms_Hsn")

    assert np.max(np.abs(b1)) == pytest.approx(1.0)
    assert metadata.max_b1 == pytest.approx(1.0)
    assert metadata.duration == pytest.approx(5e-3)
    assert b1.size == 1024

    for source in ("AHP_5ms_Hsn", REPO_ROOT / "rfpulses" / "AHP_5ms_Hsn.dat"):
        pulse = CustomPulse(source)
        assert np.max(np.abs(pulse.b1)) == pytest.approx(1.0)


def test_library_dat_sidecar_sets_import_options(tmp_path):
    np.savetxt(tmp_path / "ramp_1ms.dat", [[50.0, 0.0], [100.0, 90.0]])
    (tmp_path / "ramp_1ms.json").write_text(
        '{"amplitude_unit": "percent", "max_b1_gauss": 0.2}'
    )

    library = PulseLibrary(tmp_path)
    b1, time, metadata = library.load("ramp_1ms")

    assert library.list_pulses() == ["ramp_1ms"]
    np.testing.assert_allclose(b1, [0.1, 0.2j], atol=1e-15)
    assert metadata.max_b1 == pytest.approx(0.2)

    b1_override, _, _ = load_pulse_from_file(
        tmp_path / "ramp_1ms.dat", max_b1_gauss=1.0
    )
    np.testing.assert_allclose(b1_override, [0.5, 1.0j], atol=1e-15)
