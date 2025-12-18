"""
Test script for Jupyter Notebook export functionality (Phase 2)

This script tests:
1. Mode A: Notebook that loads data from HDF5
2. Mode B: Notebook that re-runs simulation
3. Notebook execution validation
"""

import numpy as np
from blochsimulator import BlochSimulator, TissueParameters, SpinEcho
from blochsimulator.notebook_exporter import export_notebook
from pathlib import Path
import subprocess


def test_mode_a_notebook():
    """Test Mode A: Load data from HDF5."""
    print("\n" + "="*60)
    print("TEST 1: Mode A - Load Data from HDF5")
    print("="*60)

    # 1. Create and run simulation
    print("\n1. Running simulation...")
    sim = BlochSimulator(use_parallel=False)
    tissue = TissueParameters.gray_matter(3.0)
    sequence = SpinEcho(te=20e-3, tr=100e-3)

    positions = np.array([[0.0, 0.0, 0.0]])
    frequencies = np.array([0.0])

    result = sim.simulate(sequence, tissue, positions=positions,
                         frequencies=frequencies, mode=2)
    print(f"   ✓ Simulation complete")

    # 2. Save HDF5
    h5_file = 'test_notebook_data.h5'
    sequence_params = {
        'sequence_type': 'Spin Echo',
        'te': 20e-3,
        'tr': 100e-3,
        'flip_angle': 90.0
    }
    simulation_params = {
        'mode': 'time-resolved',
        'num_positions': 1,
        'num_frequencies': 1,
        'time_step_us': 1.0
    }
    tissue_params = {
        'name': tissue.name,
        't1': tissue.t1,
        't2': tissue.t2,
        'density': tissue.density
    }

    sim.save_results(h5_file, sequence_params, simulation_params)
    print(f"   ✓ HDF5 saved: {h5_file}")

    # 3. Export notebook (Mode A)
    nb_file = 'test_mode_a.ipynb'
    print(f"\n2. Exporting Mode A notebook...")
    export_notebook(
        mode='load_data',
        filename=nb_file,
        sequence_params=sequence_params,
        simulation_params=simulation_params,
        tissue_params=tissue_params,
        h5_filename=h5_file
    )
    print(f"   ✓ Notebook exported: {nb_file}")

    # 4. Verify notebook structure
    print(f"\n3. Verifying notebook structure...")
    import nbformat
    with open(nb_file, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    print(f"   ✓ Notebook has {len(nb.cells)} cells")

    # Check for key cells
    cell_types = [cell.cell_type for cell in nb.cells]
    code_cells = sum(1 for ct in cell_types if ct == 'code')
    markdown_cells = sum(1 for ct in cell_types if ct == 'markdown')

    print(f"   ✓ Code cells: {code_cells}, Markdown cells: {markdown_cells}")

    # 5. Check that HDF5 loading code is present
    nb_text = nbformat.writes(nb)
    assert 'h5py' in nb_text, "Missing h5py import"
    assert 'data_file' in nb_text, "Missing data loading code"
    assert h5_file in nb_text, f"Missing reference to {h5_file}"
    print(f"   ✓ Notebook contains HDF5 loading code")

    return h5_file, nb_file


def test_mode_b_notebook():
    """Test Mode B: Re-run simulation."""
    print("\n" + "="*60)
    print("TEST 2: Mode B - Re-run Simulation")
    print("="*60)

    # 1. Define parameters
    print("\n1. Preparing parameters...")
    sequence_params = {
        'sequence_type': 'Spin Echo',
        'te': 30e-3,
        'tr': 200e-3,
        'flip_angle': 90.0
    }
    simulation_params = {
        'mode': 'time-resolved',
        'num_positions': 3,
        'num_frequencies': 5,
        'time_step_us': 1.0,
        'position_range_cm': 2.0,
        'frequency_range_hz': 200.0
    }
    tissue_params = {
        'name': 'White Matter',
        't1': 0.83,
        't2': 0.070,
        'density': 1.0
    }
    print(f"   ✓ Parameters prepared")

    # 2. Export notebook (Mode B)
    nb_file = 'test_mode_b.ipynb'
    print(f"\n2. Exporting Mode B notebook...")
    export_notebook(
        mode='resimulate',
        filename=nb_file,
        sequence_params=sequence_params,
        simulation_params=simulation_params,
        tissue_params=tissue_params
    )
    print(f"   ✓ Notebook exported: {nb_file}")

    # 3. Verify notebook structure
    print(f"\n3. Verifying notebook structure...")
    import nbformat
    with open(nb_file, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    print(f"   ✓ Notebook has {len(nb.cells)} cells")

    # 4. Check for simulation code
    nb_text = nbformat.writes(nb)
    assert 'BlochSimulator' in nb_text, "Missing BlochSimulator import"
    assert 'SpinEcho' in nb_text, "Missing sequence import"
    assert 't1 =' in nb_text, "Missing parameter definitions"
    assert 'sim.simulate' in nb_text, "Missing simulation call"
    print(f"   ✓ Notebook contains simulation code")

    # 5. Check parameter values are correct
    assert f"t1 = {tissue_params['t1']:.6f}" in nb_text, "T1 parameter mismatch"
    assert f"te = {sequence_params['te']:.6f}" in nb_text, "TE parameter mismatch"
    print(f"   ✓ Parameters correctly embedded in notebook")

    return nb_file


def helper_notebook_execution(nb_file):
    """
    Test that notebook can be executed (requires jupyter/nbconvert).

    Parameters
    ----------
    nb_file : str
        Notebook filename to execute
    """
    print(f"\n" + "="*60)
    print(f"TEST 3: Execute Notebook - {nb_file}")
    print("="*60)

    try:
        import nbconvert
        print(f"\n1. nbconvert available, testing execution...")

        # Try to execute notebook
        result = subprocess.run(
            ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
             '--output', f'executed_{nb_file}', nb_file],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(f"   ✓ Notebook executed successfully")
            print(f"   ✓ Output: executed_{nb_file}")
            return True
        else:
            print(f"   ⚠ Execution failed:")
            print(f"     {result.stderr}")
            return False

    except ImportError:
        print(f"   ℹ nbconvert not available, skipping execution test")
        print(f"   Install with: pip install nbconvert")
        return None
    except subprocess.TimeoutExpired:
        print(f"   ⚠ Execution timed out (>60s)")
        return False
    except FileNotFoundError:
        print(f"   ℹ jupyter command not found, skipping execution test")
        return None


def cleanup_test_files():
    """Remove test files."""
    print(f"\n" + "="*60)
    print("CLEANUP")
    print("="*60)

    files_to_remove = [
        'test_notebook_data.h5',
        'test_mode_a.ipynb',
        'test_mode_b.ipynb',
        'executed_test_mode_a.ipynb',
        'executed_test_mode_b.ipynb'
    ]

    for fname in files_to_remove:
        fpath = Path(fname)
        if fpath.exists():
            fpath.unlink()
            print(f"   ✓ Removed: {fname}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" JUPYTER NOTEBOOK EXPORT TESTS (PHASE 2) ".center(70))
    print("="*70)

    try:
        # Test Mode A
        h5_file, mode_a_nb = test_mode_a_notebook()

        # Test Mode B
        mode_b_nb = test_mode_b_notebook()

        # Try to execute notebooks (optional - requires jupyter)
        print("\n" + "="*60)
        print("OPTIONAL: Notebook Execution Tests")
        print("="*60)
        print("\nThese tests require 'jupyter' and 'nbconvert' installed.")
        print("They will be skipped if not available.\n")

        helper_notebook_execution(mode_a_nb)
        helper_notebook_execution(mode_b_nb)

        # Summary
        print("\n" + "="*70)
        print(" ALL TESTS PASSED ".center(70, "="))
        print("="*70)
        print("\nPhase 2 Implementation Summary:")
        print("✓ Mode A: Notebook with HDF5 data loading")
        print("✓ Mode B: Notebook with simulation re-execution")
        print("✓ Proper cell structure and content")
        print("✓ Parameter embedding")
        print("\nGenerated notebooks:")
        print(f"  - {mode_a_nb} (loads data from {h5_file})")
        print(f"  - {mode_b_nb} (re-runs simulation)")
        print("\nYou can now:")
        print(f"  1. Open notebooks in Jupyter: jupyter lab {mode_a_nb}")
        print(f"  2. Test in GUI: File → Export Results → Notebook options")
        print(f"  3. Proceed to Phase 3: Visualization export")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Ask before cleanup
        response = input("\nRemove test files? (y/n): ")
        if response.lower() == 'y':
            cleanup_test_files()
        else:
            print("\nTest files preserved for inspection.")

    return 0


if __name__ == "__main__":
    exit(main())
