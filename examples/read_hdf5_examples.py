"""
Examples for reading exported HDF5 files from Bloch Simulator

This script demonstrates how to:
1. Load HDF5 files exported from the Bloch Simulator
2. Access magnetization data
3. Read all parameters (tissue, sequence, simulation)
4. Recreate visualizations
5. Perform custom analysis

Requirements:
    pip install h5py numpy matplotlib
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def read_hdf5_complete(filename):
    """
    Read all data and parameters from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to HDF5 file

    Returns
    -------
    dict
        Complete dataset with all arrays and parameters
    """
    data = {}

    with h5py.File(filename, 'r') as f:
        print(f"Reading file: {filename}")
        print("=" * 60)

        # 1. Load magnetization data
        print("\n1. Loading magnetization data...")
        data['mx'] = f['mx'][...]
        data['my'] = f['my'][...]
        data['mz'] = f['mz'][...]
        data['signal'] = f['signal'][...]
        print(f"   Magnetization shape: {data['mx'].shape}")
        print(f"   Data type: {data['mx'].dtype}")

        # 2. Load time and space arrays
        print("\n2. Loading coordinate arrays...")
        data['time'] = f['time'][...]
        data['positions'] = f['positions'][...]
        data['frequencies'] = f['frequencies'][...]
        print(f"   Time points: {len(data['time'])}")
        print(f"   Duration: {data['time'][-1]*1000:.3f} ms")
        print(f"   Positions: {data['positions'].shape}")
        print(f"   Frequencies: {data['frequencies'].shape}")

        # 3. Load tissue parameters
        print("\n3. Loading tissue parameters...")
        tissue_params = {}
        for key, value in f['tissue'].attrs.items():
            tissue_params[key] = value
            print(f"   {key}: {value}")
        data['tissue'] = tissue_params

        # 4. Load sequence parameters (if available)
        if 'sequence_parameters' in f:
            print("\n4. Loading sequence parameters...")
            seq_params = {}
            seq_group = f['sequence_parameters']

            # Read attributes
            for key, value in seq_group.attrs.items():
                seq_params[key] = value
                print(f"   {key}: {value}")

            # Read datasets (like waveforms)
            for key in seq_group.keys():
                seq_params[key] = seq_group[key][...]
                print(f"   {key}: shape={seq_group[key].shape}")

            data['sequence_parameters'] = seq_params
        else:
            print("\n4. No sequence parameters found in file")
            data['sequence_parameters'] = {}

        # 5. Load simulation parameters (if available)
        if 'simulation_parameters' in f:
            print("\n5. Loading simulation parameters...")
            sim_params = {}
            sim_group = f['simulation_parameters']

            # Read attributes
            for key, value in sim_group.attrs.items():
                sim_params[key] = value
                print(f"   {key}: {value}")

            # Read datasets
            for key in sim_group.keys():
                sim_params[key] = sim_group[key][...]
                print(f"   {key}: shape={sim_group[key].shape}")

            data['simulation_parameters'] = sim_params
        else:
            print("\n5. No simulation parameters found in file")
            data['simulation_parameters'] = {}

        # 6. Load metadata
        print("\n6. Loading metadata...")
        metadata = {}
        for key, value in f.attrs.items():
            metadata[key] = value
            print(f"   {key}: {value}")
        data['metadata'] = metadata

        print("\n" + "=" * 60)
        print("Loading complete!\n")

    return data


def plot_magnetization_evolution(data, position_idx=0, freq_idx=0):
    """
    Plot magnetization evolution over time.

    Parameters
    ----------
    data : dict
        Data loaded from read_hdf5_complete()
    position_idx : int
        Position index to plot
    freq_idx : int
        Frequency index to plot
    """
    time_ms = data['time'] * 1000  # Convert to ms

    # Check if time-resolved or endpoint data
    if data['mx'].ndim == 3:  # Time-resolved
        mx = data['mx'][:, position_idx, freq_idx]
        my = data['my'][:, position_idx, freq_idx]
        mz = data['mz'][:, position_idx, freq_idx]
    elif data['mx'].ndim == 2:  # Endpoint only
        print("This is endpoint data (no time evolution)")
        return
    else:
        print(f"Unexpected data shape: {data['mx'].shape}")
        return

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Mx
    axes[0, 0].plot(time_ms, mx, 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Mx')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Transverse Magnetization (x)')

    # My
    axes[0, 1].plot(time_ms, my, 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('My')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Transverse Magnetization (y)')

    # Mz
    axes[1, 0].plot(time_ms, mz, 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Mz')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Longitudinal Magnetization')

    # Magnitude
    mxy = np.sqrt(mx**2 + my**2)
    axes[1, 1].plot(time_ms, mxy, 'purple', linewidth=1.5)
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('|Mxy|')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Transverse Magnitude')

    tissue_name = data['tissue'].get('name', 'Unknown')
    plt.suptitle(f'Magnetization Evolution - {tissue_name}\n'
                 f'Position {position_idx}, Frequency {freq_idx}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_signal(data, position_idx=0, freq_idx=0):
    """
    Plot complex signal over time.

    Parameters
    ----------
    data : dict
        Data loaded from read_hdf5_complete()
    position_idx : int
        Position index to plot
    freq_idx : int
        Frequency index to plot
    """
    time_ms = data['time'] * 1000

    if data['signal'].ndim == 3:
        signal = data['signal'][:, position_idx, freq_idx]
    elif data['signal'].ndim == 2:
        print("Endpoint data - no time evolution")
        return
    else:
        print(f"Unexpected signal shape: {data['signal'].shape}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Real and imaginary parts
    axes[0].plot(time_ms, np.real(signal), 'b-', label='Real', linewidth=1.5)
    axes[0].plot(time_ms, np.imag(signal), 'r-', label='Imaginary', linewidth=1.5)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Signal')
    axes[0].set_title('Complex Signal Components')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Magnitude
    axes[1].plot(time_ms, np.abs(signal), 'purple', linewidth=1.5)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('|Signal|')
    axes[1].set_title('Signal Magnitude')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'MRI Signal - Position {position_idx}, Frequency {freq_idx}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_spatial_profile(data, freq_idx=0, time_idx=-1):
    """
    Plot spatial profile of magnetization.

    Parameters
    ----------
    data : dict
        Data loaded from read_hdf5_complete()
    freq_idx : int
        Frequency index to plot
    time_idx : int
        Time index to plot (default: -1 for final state)
    """
    positions = data['positions']

    if data['mz'].ndim == 3:  # Time-resolved
        mz = data['mz'][time_idx, :, freq_idx]
        mxy = np.sqrt(data['mx'][time_idx, :, freq_idx]**2 +
                      data['my'][time_idx, :, freq_idx]**2)
    elif data['mz'].ndim == 2:  # Endpoint
        mz = data['mz'][:, freq_idx]
        mxy = np.sqrt(data['mx'][:, freq_idx]**2 +
                      data['my'][:, freq_idx]**2)
    else:
        print(f"Unexpected data shape: {data['mz'].shape}")
        return

    # Plot along z-axis (typical slice-select direction)
    z_pos = positions[:, 2] * 100  # Convert to cm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Mz profile
    ax1.plot(z_pos, mz, 'go-', linewidth=2, markersize=6)
    ax1.set_xlabel('Position (cm)')
    ax1.set_ylabel('Mz')
    ax1.set_title('Longitudinal Magnetization Profile')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Mxy profile
    ax2.plot(z_pos, mxy, 'mo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Position (cm)')
    ax2.set_ylabel('|Mxy|')
    ax2.set_title('Transverse Magnetization Profile')
    ax2.grid(True, alpha=0.3)

    freq = data['frequencies'][freq_idx]
    time_ms = data['time'][time_idx] * 1000 if data['mz'].ndim == 3 else 0
    plt.suptitle(f'Spatial Profile - Frequency: {freq:.1f} Hz, Time: {time_ms:.3f} ms',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def print_file_info(filename):
    """
    Print a summary of what's in the HDF5 file.

    Parameters
    ----------
    filename : str
        Path to HDF5 file
    """
    with h5py.File(filename, 'r') as f:
        print(f"\n{'='*60}")
        print(f"HDF5 File Summary: {Path(filename).name}")
        print(f"{'='*60}\n")

        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📊 {name.split('/')[-1]}: {obj.shape} {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}📁 {name.split('/')[-1]}/")

        f.visititems(print_structure)

        print(f"\n{'='*60}\n")


def quick_analysis(data):
    """
    Perform quick analysis and print summary statistics.

    Parameters
    ----------
    data : dict
        Data loaded from read_hdf5_complete()
    """
    print("\n" + "="*60)
    print("QUICK ANALYSIS")
    print("="*60)

    # Time info
    print(f"\nTime Domain:")
    print(f"  Duration: {data['time'][-1]*1000:.3f} ms")
    print(f"  Time points: {len(data['time'])}")
    print(f"  Time step: {np.mean(np.diff(data['time']))*1e6:.3f} µs")

    # Spatial info
    print(f"\nSpatial Domain:")
    print(f"  Positions: {data['positions'].shape[0]}")
    print(f"  Position range: {data['positions'][:, 2].min()*100:.3f} to {data['positions'][:, 2].max()*100:.3f} cm")

    # Frequency info
    print(f"\nFrequency Domain:")
    print(f"  Frequencies: {len(data['frequencies'])}")
    print(f"  Frequency range: {data['frequencies'].min():.1f} to {data['frequencies'].max():.1f} Hz")

    # Magnetization statistics
    if data['mx'].ndim == 3:
        print(f"\nMagnetization Statistics (final time point):")
        print(f"  Mx: min={data['mx'][-1].min():.4f}, max={data['mx'][-1].max():.4f}")
        print(f"  My: min={data['my'][-1].min():.4f}, max={data['my'][-1].max():.4f}")
        print(f"  Mz: min={data['mz'][-1].min():.4f}, max={data['mz'][-1].max():.4f}")

        # Find peak transverse magnetization
        mxy = np.sqrt(data['mx']**2 + data['my']**2)
        max_mxy = mxy.max()
        max_idx = np.unravel_index(mxy.argmax(), mxy.shape)
        print(f"\nPeak Transverse Magnetization:")
        print(f"  |Mxy|_max: {max_mxy:.4f}")
        print(f"  At time index: {max_idx[0]} ({data['time'][max_idx[0]]*1000:.3f} ms)")
        print(f"  At position index: {max_idx[1]}")
        print(f"  At frequency index: {max_idx[2]}")

    # Tissue parameters
    print(f"\nTissue Parameters:")
    for key, value in data['tissue'].items():
        if key in ['t1', 't2', 't2_star']:
            print(f"  {key.upper()}: {value*1000:.1f} ms")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*60 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_basic():
    """Basic example: Load and inspect HDF5 file."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Loading and Inspection")
    print("="*60)

    # Create a test file first
    from bloch_simulator import BlochSimulator, TissueParameters, SpinEcho

    print("\nCreating test simulation...")
    sim = BlochSimulator(use_parallel=False)
    tissue = TissueParameters.gray_matter(3.0)
    sequence = SpinEcho(te=20e-3, tr=100e-3)

    positions = np.array([[0.0, 0.0, 0.0]])
    frequencies = np.array([0.0])

    result = sim.simulate(sequence, tissue, positions=positions,
                         frequencies=frequencies, mode=2)

    # Save with parameters
    seq_params = {'sequence_type': 'Spin Echo', 'te': 20e-3, 'tr': 100e-3}
    sim_params = {'mode': 'time-resolved', 'num_positions': 1}

    filename = 'example_data.h5'
    sim.save_results(filename, seq_params, sim_params)
    print(f"Saved to: {filename}")

    # Now load and inspect
    print("\n" + "-"*60)
    print("Loading HDF5 file...")
    print("-"*60)
    data = read_hdf5_complete(filename)

    # Quick analysis
    quick_analysis(data)

    # Visualization
    print("Creating plots...")
    plot_magnetization_evolution(data)
    plot_signal(data)

    # Clean up
    Path(filename).unlink()
    print(f"\nCleaned up: {filename}")


def example_multi_position():
    """Example with multiple positions and frequencies."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Position, Multi-Frequency")
    print("="*60)

    from bloch_simulator import BlochSimulator, TissueParameters, SpinEcho

    print("\nCreating simulation with spatial and frequency variation...")
    sim = BlochSimulator(use_parallel=False)
    tissue = TissueParameters.white_matter(3.0)
    sequence = SpinEcho(te=30e-3, tr=200e-3)

    # Multiple positions along z-axis
    positions = np.zeros((11, 3))
    positions[:, 2] = np.linspace(-0.01, 0.01, 11)  # ±1 cm

    # Multiple off-resonance frequencies
    frequencies = np.linspace(-100, 100, 5)  # ±100 Hz

    result = sim.simulate(sequence, tissue, positions=positions,
                         frequencies=frequencies, mode=2)

    # Save
    seq_params = {
        'sequence_type': 'Spin Echo',
        'te': 30e-3,
        'tr': 200e-3,
        'flip_angle': 90.0
    }
    sim_params = {
        'mode': 'time-resolved',
        'num_positions': 11,
        'num_frequencies': 5,
        'position_range_cm': 2.0,
        'frequency_range_hz': 200.0
    }

    filename = 'example_multi.h5'
    sim.save_results(filename, seq_params, sim_params)
    print(f"Saved to: {filename}")

    # Load and analyze
    print("\n" + "-"*60)
    print("Loading and analyzing...")
    print("-"*60)
    data = read_hdf5_complete(filename)
    quick_analysis(data)

    # Plot different views
    print("\nCreating plots...")
    plot_magnetization_evolution(data, position_idx=5, freq_idx=2)  # Center position
    plot_spatial_profile(data, freq_idx=2)  # Spatial profile at center frequency

    # Clean up
    Path(filename).unlink()
    print(f"\nCleaned up: {filename}")


def example_custom_analysis(filename):
    """
    Example of custom analysis on exported data.

    Parameters
    ----------
    filename : str
        Path to exported HDF5 file
    """
    data = read_hdf5_complete(filename)

    print("\n" + "="*60)
    print("CUSTOM ANALYSIS EXAMPLE")
    print("="*60)

    # Calculate T2 relaxation curve fit
    if data['mx'].ndim == 3:
        time = data['time']
        mxy = np.sqrt(data['mx'][:, 0, 0]**2 + data['my'][:, 0, 0]**2)

        # Find peak and fit exponential decay after it
        peak_idx = np.argmax(mxy)
        if peak_idx < len(time) - 10:
            t_decay = time[peak_idx:] - time[peak_idx]
            mxy_decay = mxy[peak_idx:]

            # Simple exponential fit
            from scipy.optimize import curve_fit
            def exp_decay(t, M0, T2):
                return M0 * np.exp(-t / T2)

            try:
                popt, _ = curve_fit(exp_decay, t_decay, mxy_decay,
                                   p0=[mxy[peak_idx], data['tissue']['t2']])

                print(f"\nT2 Decay Analysis:")
                print(f"  Fitted T2: {popt[1]*1000:.2f} ms")
                print(f"  Expected T2: {data['tissue']['t2']*1000:.2f} ms")
                print(f"  Difference: {abs(popt[1] - data['tissue']['t2'])*1000:.2f} ms")
            except:
                print("\nCould not fit T2 decay curve")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" HDF5 READING EXAMPLES FOR BLOCH SIMULATOR ".center(70))
    print("="*70)

    # Run examples
    example_basic()
    print("\n\n")
    example_multi_position()

    print("\n" + "="*70)
    print(" ALL EXAMPLES COMPLETED ".center(70))
    print("="*70)

    print("\n📚 Usage Summary:")
    print("   1. Load data: data = read_hdf5_complete('filename.h5')")
    print("   2. Quick stats: quick_analysis(data)")
    print("   3. Plot magnetization: plot_magnetization_evolution(data)")
    print("   4. Plot signal: plot_signal(data)")
    print("   5. Plot spatial profile: plot_spatial_profile(data)")
    print("   6. File structure: print_file_info('filename.h5')")
    print("\n   Access data arrays: data['mx'], data['my'], data['mz'], data['signal']")
    print("   Access parameters: data['tissue'], data['sequence_parameters'], etc.")
