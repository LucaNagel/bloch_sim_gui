import importlib.util
from pathlib import Path

import numpy as np

from blochsimulator.phantom import Phantom
from blochsimulator.project_io import load_project, save_project
from blochsimulator.sequence import ADCEvent, GradientEvent, RFEvent, SequenceProgram


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_flash_flip_angle_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("run_flash_flip_angle_sweep", SCRIPT_PATH)
SWEEP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SWEEP)


def _small_project(path):
    phantom = Phantom(
        shape=(1, 1, 1),
        fov=(0.01, 0.01, 0.01),
        t1_map=np.ones((1, 1, 1)),
        t2_map=np.full((1, 1, 1), 0.1),
        pd_map=np.ones((1, 1, 1)),
        name="Sweep test phantom",
    )
    program = SequenceProgram(
        events=(
            RFEvent(
                start_s=0.0,
                samples_hz=np.asarray([100.0 + 20.0j, 80.0 - 10.0j]),
                raster_s=1e-4,
                phase_offset_rad=0.25,
            ),
            GradientEvent("x", 2e-4, np.asarray([10.0]), 1e-4),
            ADCEvent(3e-4, 2, 1e-4),
        ),
        duration_s=5e-4,
        source="internal-flash-2d",
        metadata={"definitions": {"Name": "flash_2d", "FlipAngleDeg": 10.0}},
    )
    state = {
        "workspace_mode": "sequence",
        "sequence_controls": {
            "flash_flip_angle_deg": {"type": "value", "value": 10.0},
            "simulation_timestep_us": {"type": "value", "value": 20.0},
        },
    }
    save_project(path, state, phantom=phantom, program=program)
    return program


def test_flip_angle_grid_is_inclusive():
    np.testing.assert_allclose(SWEEP.flip_angle_values(5, 150, 5), np.arange(5, 151, 5))


def test_program_scaling_changes_only_rf_amplitude_and_metadata(tmp_path):
    program = _small_project(tmp_path / "input.blochproj")
    scaled = SWEEP.program_with_flip_angle(program, 25.0, 10.0)

    np.testing.assert_allclose(
        scaled.rf_events[0].samples_hz,
        program.rf_events[0].samples_hz * 2.5,
    )
    assert scaled.rf_events[0].phase_offset_rad == program.rf_events[0].phase_offset_rad
    assert scaled.gradient_events[0] is program.gradient_events[0]
    assert scaled.adc_events[0] is program.adc_events[0]
    assert scaled.metadata["definitions"]["FlipAngleDeg"] == 25.0
    assert program.metadata["definitions"]["FlipAngleDeg"] == 10.0


def test_small_sweep_writes_resumable_projects_and_aggregate(tmp_path):
    project_path = tmp_path / "input.blochproj"
    _small_project(project_path)
    output_dir = tmp_path / "sweep"
    arguments = [
        str(project_path),
        "--output-dir",
        str(output_dir),
        "--start",
        "5",
        "--stop",
        "10",
        "--step",
        "5",
        "--spin-counts",
        "1",
        "1",
        "1",
        "--chunk-voxels",
        "1",
    ]

    assert SWEEP.main(arguments) == 0
    assert (output_dir / "sweep_summary.csv").is_file()
    assert (output_dir / "sweep_signals.npz").is_file()

    five = load_project(output_dir / "fa_005deg.blochproj")
    ten = load_project(output_dir / "fa_010deg.blochproj")
    assert five["program"].metadata["definitions"]["FlipAngleDeg"] == 5.0
    assert ten["program"].metadata["definitions"]["FlipAngleDeg"] == 10.0
    assert five["state"]["sequence_controls"]["flash_flip_angle_deg"]["value"] == 5.0
    assert five["sequence_result"] is not None
    assert ten["sequence_result"] is not None

    with np.load(output_dir / "sweep_signals.npz", allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["flip_angles_deg"], [5.0, 10.0])
        assert archive["signal"].shape[0] == 2

    # A second run validates and reuses both completed project files.
    assert SWEEP.main(arguments) == 0
