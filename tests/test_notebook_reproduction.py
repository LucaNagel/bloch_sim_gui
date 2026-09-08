"""Behavioral regression tests for Free Mode reproduction notebooks."""

from pathlib import Path

import matplotlib
import nbformat
import numpy as np
import pytest

from blochsimulator import BlochSimulator, TissueParameters, design_rf_pulse
from blochsimulator.notebook_exporter import export_notebook


matplotlib.use("Agg")


def _code_cell(notebook, marker: str) -> str:
    return next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and marker in cell.source
    )


def _export_fid_reproduction_notebook(tmp_path: Path):
    """Export with frozen arrays, matching the GUI path that caused the bug."""
    time = np.arange(750, dtype=float) * 20e-6
    b1 = np.zeros(time.size, dtype=complex)
    b1[:50] = 0.01
    gradients = np.zeros((time.size, 3), dtype=float)
    sequence_params = {
        "sequence_type": "Free Induction Decay",
        "type": "Free Induction Decay",
        # GUI state stores these three values in milliseconds and also exports
        # the unambiguous *_s variants used by the notebook.
        "te": 3.0,
        "tr": 10.0,
        "ti": 400.0,
        "te_s": 0.003,
        "tr_s": 0.010,
        "ti_s": 0.400,
        "ssfp_flip_ratio": 0.5,
        "ssfp_tr_ratio": 0.5,
        "rf_pulse_type": "Gaussian",
        "rf_flip_angle": 90.0,
        "rf_duration_s": 0.001,
        "rf_time_bw_product": 2.5,
        "rf_phase": 0.0,
        "rf_freq_offset": 0.0,
        "rf_b1_amplitude": 0.0,
        "rf_sinc_lobes": 3,
        "rf_slr_sharpness": 1,
        "rf_apodization": "None",
        # These arrays are deliberately present. Before the regression fix,
        # they silently overrode edits to rf_flip_angle in the notebook.
        "b1_waveform": b1,
        "time_waveform": time,
        "gradients_waveform": gradients,
    }
    simulation_params = {
        "mode": "time-resolved",
        "num_positions": 1,
        "num_frequencies": 1,
        "time_step_us": 20.0,
        "extra_tail_ms": 5.0,
        "position_range_mm": 0.0,
        "frequency_range_hz": 0.0,
    }
    tissue_params = {"name": "Test", "t1": 1.0, "t2": 0.1, "density": 1.0}
    notebook_path = tmp_path / "fid_repro.ipynb"
    export_notebook(
        "resimulate",
        str(notebook_path),
        sequence_params=sequence_params,
        simulation_params=simulation_params,
        tissue_params=tissue_params,
        waveform_filename=str(tmp_path / "fid_waveforms.npz"),
    )
    return nbformat.read(notebook_path, as_version=4)


def _fid_namespace():
    return {
        "np": np,
        "Path": Path,
        "BlochSimulator": BlochSimulator,
        "TissueParameters": TissueParameters,
        "design_rf_pulse": design_rf_pulse,
    }


def _run_fid_notebook(notebook, parameter_code: str):
    """Execute the generated computational path with one parameter cell."""
    namespace = _fid_namespace()
    sources = (
        parameter_code,
        _code_cell(notebook, "# Create simulator"),
        _code_cell(notebook, "# Create Free Induction Decay"),
        _code_cell(notebook, "# Define spatial positions"),
        _code_cell(notebook, "# Run simulation"),
    )
    for source in sources:
        exec(compile(source, "fid_repro.ipynb", "exec"), namespace)
    return (
        np.array(namespace["b1"], copy=True),
        np.array(namespace["result"]["signal"], copy=True),
        dict(namespace["sequence_params"]),
    )


def test_exported_fid_flip_angle_changes_waveform_and_simulation(tmp_path, monkeypatch):
    """Editing rf_flip_angle must alter downstream notebook behavior."""
    notebook = _export_fid_reproduction_notebook(tmp_path)
    monkeypatch.chdir(tmp_path)
    parameter_code = _code_cell(notebook, "# Define simulation parameters")
    edited_parameter_code = parameter_code.replace(
        "rf_flip_angle = 90  # degrees",
        "rf_flip_angle = 45  # degrees",
        1,
    )
    assert edited_parameter_code != parameter_code
    assert parameter_code.count("rf_flip_angle = 90") == 1
    assert "'rf_flip_angle': rf_flip_angle," in parameter_code

    waveform_90, signal_90, params_90 = _run_fid_notebook(notebook, parameter_code)
    waveform_45, signal_45, params_45 = _run_fid_notebook(
        notebook, edited_parameter_code
    )

    assert params_90["rf_flip_angle"] == 90.0
    assert params_45["rf_flip_angle"] == 45.0
    assert not np.allclose(waveform_90, waveform_45)
    assert abs(np.sum(waveform_45)) / abs(np.sum(waveform_90)) == pytest.approx(0.5)
    assert not np.allclose(signal_90, signal_45)
    assert np.max(np.abs(signal_45)) < np.max(np.abs(signal_90))


def test_fid_notebook_marks_irrelevant_parameters_and_plots_sequence(
    tmp_path, monkeypatch
):
    notebook = _export_fid_reproduction_notebook(tmp_path)
    parameter_code = _code_cell(notebook, "# Define simulation parameters")
    assert "'ssfp_flip_ratio': 0.5,  # not used for Free Induction Decay" in (
        parameter_code
    )
    assert "'ssfp_tr_ratio': 0.5,  # not used for Free Induction Decay" in (
        parameter_code
    )
    assert "te = 0.003000  # seconds  # not used for Free Induction Decay" in (
        parameter_code
    )

    monkeypatch.chdir(tmp_path)
    namespace = _fid_namespace()
    for marker in (
        "# Define simulation parameters",
        "# Create Free Induction Decay",
    ):
        source = _code_cell(notebook, marker)
        exec(compile(source, "fid_repro.ipynb", "exec"), namespace)

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    namespace["plt"] = plt
    visualization_code = _code_cell(notebook, "# Plot the exact arrays")
    exec(compile(visualization_code, "fid_repro.ipynb", "exec"), namespace)

    figure = namespace["fig"]
    assert len(figure.axes) == 4
    np.testing.assert_allclose(
        figure.axes[0].lines[0].get_xdata(), namespace["time"] * 1e3
    )
    np.testing.assert_allclose(
        figure.axes[0].lines[0].get_ydata(), np.abs(namespace["b1"])
    )
    plt.close(figure)
