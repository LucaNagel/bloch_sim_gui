from pathlib import Path

import h5py
import nbformat


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def test_spin_echo_example_notebooks_are_linked_and_parameterized():
    reproduction_path = EXAMPLES / "spin_echo_reproduction.ipynb"
    analysis_path = EXAMPLES / "spin_echo_analysis.ipynb"

    reproduction = nbformat.read(reproduction_path, as_version=4)
    analysis = nbformat.read(analysis_path, as_version=4)
    reproduction_text = nbformat.writes(reproduction)
    analysis_text = nbformat.writes(analysis)

    assert "Spin-Echo Simulation — Reproduction" in reproduction_text
    assert "sequence_type = 'Spin Echo'" in reproduction_text
    assert "te = 0.020000" in reproduction_text
    assert "tr = 0.100000" in reproduction_text
    assert "dt=time_step_us * 1e-6" in reproduction_text

    assert "Spin-Echo Simulation — Analysis" in analysis_text
    assert "spin_echo_analysis_data.h5" in analysis_text
    assert "Xarray Dataset" in analysis_text

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "examples/spin_echo_reproduction.ipynb" in readme
    assert "examples/spin_echo_analysis.ipynb" in readme
    assert "Parameter sweeps" in readme


def test_spin_echo_analysis_data_matches_notebook_parameters():
    data_path = EXAMPLES / "spin_echo_analysis_data.h5"
    with h5py.File(data_path, "r") as data:
        assert data["mx"].shape == (501, 2, 7)
        assert data["signal"].shape == (501, 2, 7)
        assert data["sequence_parameters"].attrs["sequence_type"] == "Spin Echo"
        assert data["sequence_parameters"].attrs["te"] == 0.020
        assert data["sequence_parameters"].attrs["tr"] == 0.100
        assert data["simulation_parameters"].attrs["time_step_us"] == 200.0
