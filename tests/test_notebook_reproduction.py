"""Behavioral regression tests for Free Mode reproduction notebooks."""

from pathlib import Path

import matplotlib
import nbformat
import numpy as np
import pytest

from blochsimulator import BlochSimulator, TissueParameters, design_rf_pulse
from blochsimulator.gradient_lobes import GradientLobe, add_gradient_lobes
from blochsimulator.notebook_exporter import export_notebook


matplotlib.use("Agg")


def _code_cell(notebook, marker: str) -> str:
    return next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and marker in cell.source
    )


def _export_fid_reproduction_notebook(tmp_path: Path, gradient_lobes=None):
    """Export with frozen arrays, matching the GUI path that caused the bug."""
    time = np.arange(750, dtype=float) * 20e-6
    b1 = np.zeros(time.size, dtype=complex)
    b1[:50] = 0.01
    gradients = np.zeros((time.size, 3), dtype=float)
    for lobe in gradient_lobes or []:
        _, gradients, _ = add_gradient_lobes(b1, gradients, time, [lobe], 20e-6)
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
        "slice_axis": "z",
        "gradient_lobes": [lobe.to_state() for lobe in gradient_lobes or []],
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
        _code_cell(notebook, "# Use the exact sampled axes"),
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
    parameter_code = _code_cell(notebook, "# Canonical simulation parameters")
    edited_parameter_code = parameter_code.replace(
        "'flip_angle_deg': 90.0",
        "'flip_angle_deg': 45.0",
        1,
    )
    assert edited_parameter_code != parameter_code
    assert parameter_code.count("'flip_angle_deg': 90.0") == 1
    assert "'sequence_definition_source': 'regenerated_from_parameters'" in (
        parameter_code
    )

    waveform_90, signal_90, params_90 = _run_fid_notebook(notebook, parameter_code)
    waveform_45, signal_45, params_45 = _run_fid_notebook(
        notebook, edited_parameter_code
    )

    assert params_90["rf_designer_snapshot"]["flip_angle_deg"] == 90.0
    assert params_45["rf_designer_snapshot"]["flip_angle_deg"] == 45.0
    assert not np.allclose(waveform_90, waveform_45)
    assert abs(np.sum(waveform_45)) / abs(np.sum(waveform_90)) == pytest.approx(0.5)
    assert not np.allclose(signal_90, signal_45)
    assert np.max(np.abs(signal_45)) < np.max(np.abs(signal_90))


def test_fid_notebook_runs_adaptive_ahp_with_independent_saved_spacing(
    tmp_path, monkeypatch
):
    notebook = _export_fid_reproduction_notebook(tmp_path)
    monkeypatch.chdir(tmp_path)
    namespace = _fid_namespace()
    exec(_code_cell(notebook, "# Canonical simulation parameters"), namespace)
    namespace["sequence_params"]["rf_designer_snapshot"].update(
        pulse_type="Adiabatic Half Passage",
        duration_s=0.005,
        b1_amplitude_g=5.0,
    )
    namespace["simulation_params"].update(
        adaptive_rf=True,
        time_step_us=0.1,
        output_step_us=1.0,
    )
    for marker in (
        "# Create simulator",
        "# Create Free Induction Decay",
        "# Use the exact sampled axes",
        "# Run simulation",
    ):
        exec(_code_cell(notebook, marker), namespace)
    result = namespace["result"]
    assert result["integration"]["solver"] == "DOP853"
    assert len(result["time"]) == 15001
    assert len(namespace["b1"]) > 100000
    np.testing.assert_allclose(np.diff(result["time"]), 1e-6, atol=1e-15)


def test_fid_notebook_marks_irrelevant_parameters_and_plots_sequence(
    tmp_path, monkeypatch
):
    notebook = _export_fid_reproduction_notebook(tmp_path)
    parameter_code = _code_cell(notebook, "# Canonical simulation parameters")
    assert "'ssfp_flip_ratio'" not in parameter_code
    assert "'ssfp_tr_ratio'" not in parameter_code
    assert "'te_s'" not in parameter_code
    assert "'tr_s': 0.01" in parameter_code

    monkeypatch.chdir(tmp_path)
    namespace = _fid_namespace()
    for marker in (
        "# Canonical simulation parameters",
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


def test_fid_notebook_with_gradient_lobes_uses_the_exported_waveforms(
    tmp_path, monkeypatch
):
    lobe = GradientLobe("x", 2.0, 4.0, 10.0, 0.5)
    notebook = _export_fid_reproduction_notebook(tmp_path, [lobe])
    monkeypatch.chdir(tmp_path)
    parameter_code = _code_cell(notebook, "# Canonical simulation parameters")
    assert "'sequence_definition_source': 'exact_exported_waveforms'" in (
        parameter_code
    )
    assert "'gradient_lobes': [{'axis': 'x'" in parameter_code
    assert "'slice_axis'" not in parameter_code

    namespace = _fid_namespace()
    for marker in (
        "# Canonical simulation parameters",
        "# Create Free Induction Decay",
    ):
        exec(
            compile(_code_cell(notebook, marker), "fid_repro.ipynb", "exec"), namespace
        )

    gradients = namespace["gradients"]
    assert gradients.shape == (750, 3)
    area_mt_ms_per_m = gradients[:, 0].sum() * 20e-6 / 0.1 * 1e3
    assert area_mt_ms_per_m == pytest.approx(lobe.moment_mt_ms_per_m)


def test_fid_notebook_without_gradient_lobes_drops_the_empty_table(tmp_path):
    notebook = _export_fid_reproduction_notebook(tmp_path)
    parameter_code = _code_cell(notebook, "# Canonical simulation parameters")
    assert "'gradient_lobes'" not in parameter_code
    assert "'sequence_definition_source': 'regenerated_from_parameters'" in (
        parameter_code
    )
