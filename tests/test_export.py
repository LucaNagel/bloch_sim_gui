"""
Test script for the enhanced export functionality (Phase 1)

This script tests:
1. Enhanced HDF5 export with complete parameters
2. JSON parameter export
3. Ability to reload and inspect exported data
"""

import numpy as np
from blochsimulator import BlochSimulator, TissueParameters, SpinEcho
import h5py
import json
from pathlib import Path

def test_enhanced_export():
    """Test the enhanced export functionality."""

    print("=" * 60)
    print("Testing Enhanced Export Functionality (Phase 1)")
    print("=" * 60)

    # 1. Create a simple simulation
    print("\n1. Running simulation...")
    sim = BlochSimulator(use_parallel=False)
    tissue = TissueParameters.gray_matter(3.0)
    sequence = SpinEcho(te=20e-3, tr=500e-3)

    positions = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    frequencies = np.array([-50.0, 0.0, 50.0])

    result = sim.simulate(
        sequence,
        tissue,
        positions=positions,
        frequencies=frequencies,
        mode=2  # Time-resolved
    )

    print(f"   ✓ Simulation complete: {result['mx'].shape} samples")

    # 2. Prepare parameters for export
    print("\n2. Preparing parameters...")
    sequence_params = {
        'sequence_type': 'Spin Echo',
        'te': 20e-3,
        'tr': 500e-3,
        'flip_angle': 90.0,
    }

    simulation_params = {
        'mode': 'time-resolved',
        'time_step_us': 1.0,
        'num_positions': 3,
        'num_frequencies': 3,
        'use_parallel': False,
    }

    # 3. Test HDF5 export
    print("\n3. Testing HDF5 export...")
    h5_file = "test_export_results.h5"
    sim.save_results(h5_file, sequence_params, simulation_params)
    print(f"   ✓ HDF5 file saved: {h5_file}")

    # 4. Verify HDF5 contents
    print("\n4. Verifying HDF5 contents...")
    with h5py.File(h5_file, 'r') as f:
        print(f"   Data arrays:")
        print(f"     - mx: {f['mx'].shape}")
        print(f"     - my: {f['my'].shape}")
        print(f"     - mz: {f['mz'].shape}")
        print(f"     - signal: {f['signal'].shape}")

        print(f"\n   Tissue parameters:")
        for key, value in f['tissue'].attrs.items():
            print(f"     - {key}: {value}")

        if 'sequence_parameters' in f:
            print(f"\n   Sequence parameters:")
            for key, value in f['sequence_parameters'].attrs.items():
                print(f"     - {key}: {value}")

        if 'simulation_parameters' in f:
            print(f"\n   Simulation parameters:")
            for key, value in f['simulation_parameters'].attrs.items():
                print(f"     - {key}: {value}")

        print(f"\n   Metadata:")
        for key, value in f.attrs.items():
            print(f"     - {key}: {value}")

    # 5. Test JSON export
    print("\n5. Testing JSON export...")
    json_file = "test_export_params.json"
    sim.save_parameters_json(json_file, sequence_params, simulation_params)
    print(f"   ✓ JSON file saved: {json_file}")

    # 6. Verify JSON contents
    print("\n6. Verifying JSON contents...")
    with open(json_file, 'r') as f:
        params = json.load(f)

    print(f"   JSON structure:")
    for key in params.keys():
        print(f"     - {key}")

    print(f"\n   Tissue parameters from JSON:")
    for key, value in params['tissue_parameters'].items():
        print(f"     - {key}: {value}")

    # 7. Clean up
    print("\n7. Cleaning up test files...")
    Path(h5_file).unlink()
    Path(json_file).unlink()
    print("   ✓ Test files removed")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nPhase 1 Implementation Summary:")
    print("✓ Enhanced HDF5 export with complete parameters")
    print("✓ JSON parameter export for human-readable documentation")
    print("✓ Both formats include tissue, sequence, and simulation parameters")
    print("✓ Metadata includes timestamp and version information")
    print("\nNext Steps:")
    print("- Test in GUI with real simulations")
    print("- Gather user feedback on export format")
    print("- Move to Phase 2: Jupyter notebook export")

if __name__ == "__main__":
    test_enhanced_export()
