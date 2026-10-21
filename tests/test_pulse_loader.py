import hashlib
from pathlib import Path

import numpy as np
import pytest

from blochsimulator import pulse_loader
from blochsimulator.pulse_loader import PulseLibrary, load_pulse
from blochsimulator.simulator import CustomPulse

REPO_ROOT = Path(__file__).parents[1]
PULSE_SUFFIXES = {".exc", ".dat", ".txt"}


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
