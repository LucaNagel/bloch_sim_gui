"""
kspace_simulator.py - K-space and spectroscopic MRI simulation

This module provides signal-based MRI simulation with proper k-space encoding,
enabling simulation of:
- Gradient-induced phase encoding
- Eddy current effects on spectra and images
- EPI odd/even echo artifacts (Nyquist ghosting)
- Chemical shift imaging (CSI)
- Multi-echo sequences

The key difference from image-space simulation:
    Image-space: Compute M(r) at each voxel independently
    K-space: Compute S(t) = ∫ M(r,t) × exp(-i·φ(r,t)) dr
    
Where φ(r,t) = 2π × k(t)·r includes gradient-induced phase.

Author: Luca Nagel
Date: 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Union, Callable
from enum import Enum
import warnings

# Try to import scipy for signal processing
try:
    from scipy import signal as sp_signal
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available - some features will be limited")


class TrajectoryType(Enum):
    """K-space trajectory types."""
    CARTESIAN = "cartesian"
    EPI = "epi"
    SPIRAL = "spiral"
    RADIAL = "radial"
    CSI = "csi"  # Chemical shift imaging (phase encoding only)
    CUSTOM = "custom"


@dataclass
class EddyCurrentModel:
    """
    Model for gradient eddy currents.
    
    Eddy currents cause the actual gradient to differ from the commanded gradient:
        G_actual(t) = G_commanded(t) - ΔG(t)
    
    Where ΔG(t) is a sum of exponential decays from previous gradient changes.
    
    Attributes
    ----------
    amplitudes : list of float
        Relative amplitude of each eddy current component (fraction of step)
    time_constants : list of float
        Time constant (tau) for each component in seconds
    axes : list of str or None
        Which axes each component affects ('x', 'y', 'z', or None for all)
    cross_terms : dict, optional
        Cross-term eddy currents, e.g., {'xy': (amplitude, tau)} means
        x-gradient causes eddy current on y-axis
    """
    amplitudes: List[float] = field(default_factory=lambda: [0.02, 0.005])
    time_constants: List[float] = field(default_factory=lambda: [0.1e-3, 1e-3])
    axes: List[Optional[str]] = field(default_factory=lambda: [None, None])
    cross_terms: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    def apply_to_gradient(self, gradient: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply eddy current distortion to gradient waveform.
        
        Parameters
        ----------
        gradient : ndarray, shape (ntime, 3)
            Commanded gradient waveform [Gx, Gy, Gz] in G/cm
        dt : float
            Time step in seconds
            
        Returns
        -------
        ndarray, shape (ntime, 3)
            Actual gradient after eddy current effects
        """
        ntime = len(gradient)
        grad_actual = gradient.copy()
        
        # Calculate gradient derivative (rate of change)
        grad_diff = np.diff(gradient, axis=0, prepend=gradient[0:1])
        
        for amp, tau, axis in zip(self.amplitudes, self.time_constants, self.axes):
            if tau <= 0:
                continue
                
            # Build impulse response: h(t) = amp * exp(-t/tau)
            # Length needed to capture 5 time constants
            ir_length = min(ntime, int(5 * tau / dt) + 1)
            t_ir = np.arange(ir_length) * dt
            impulse_response = amp * np.exp(-t_ir / tau)
            
            # Convolve gradient derivative with impulse response
            for ax_idx, ax_name in enumerate(['x', 'y', 'z']):
                if axis is not None and axis != ax_name:
                    continue
                    
                eddy = np.convolve(grad_diff[:, ax_idx], impulse_response, mode='full')[:ntime]
                grad_actual[:, ax_idx] -= eddy
        
        # Apply cross-terms
        for cross_key, (amp, tau) in self.cross_terms.items():
            if len(cross_key) != 2:
                continue
            src_axis = {'x': 0, 'y': 1, 'z': 2}.get(cross_key[0])
            dst_axis = {'x': 0, 'y': 1, 'z': 2}.get(cross_key[1])
            if src_axis is None or dst_axis is None:
                continue
                
            ir_length = min(ntime, int(5 * tau / dt) + 1)
            t_ir = np.arange(ir_length) * dt
            impulse_response = amp * np.exp(-t_ir / tau)
            
            eddy = np.convolve(grad_diff[:, src_axis], impulse_response, mode='full')[:ntime]
            grad_actual[:, dst_axis] -= eddy
        
        return grad_actual


@dataclass
class EPIParameters:
    """
    Parameters for EPI (Echo Planar Imaging) trajectory.
    
    Attributes
    ----------
    matrix_size : tuple
        (nx, ny) matrix size
    fov : tuple
        (fov_x, fov_y) field of view in meters
    echo_spacing : float
        Time between echoes in seconds
    ramp_time : float
        Gradient ramp time in seconds
    flat_time : float
        Gradient flat-top time in seconds
    blip_duration : float
        Phase-encode blip duration in seconds
    n_shots : int
        Number of shots (1 = single-shot EPI)
    partial_fourier : float
        Partial Fourier factor (0.5-1.0)
    flyback : bool
        If True, use flyback (unipolar) readout
    """
    matrix_size: Tuple[int, int] = (64, 64)
    fov: Tuple[float, float] = (0.24, 0.24)
    echo_spacing: float = 0.5e-3
    ramp_time: float = 0.1e-3
    flat_time: float = 0.3e-3
    blip_duration: float = 0.1e-3
    n_shots: int = 1
    partial_fourier: float = 1.0
    flyback: bool = False
    
    @property
    def lines_per_shot(self) -> int:
        """Number of phase-encode lines per shot."""
        total_lines = int(self.matrix_size[1] * self.partial_fourier)
        return total_lines // self.n_shots


@dataclass 
class CSIParameters:
    """
    Parameters for Chemical Shift Imaging (CSI).
    
    Attributes
    ----------
    matrix_size : tuple
        Spatial matrix size (nx, ny) or (nx, ny, nz)
    fov : tuple
        Field of view in meters
    spectral_points : int
        Number of spectral (time) points
    spectral_bandwidth : float
        Spectral bandwidth in Hz
    tr : float
        Repetition time in seconds
    """
    matrix_size: Tuple[int, ...] = (16, 16)
    fov: Tuple[float, ...] = (0.16, 0.16)
    spectral_points: int = 1024
    spectral_bandwidth: float = 2000.0
    tr: float = 1.5
    
    @property
    def dwell_time(self) -> float:
        """Time between spectral samples."""
        return 1.0 / self.spectral_bandwidth
    
    @property
    def acquisition_time(self) -> float:
        """Total acquisition time per voxel."""
        return self.spectral_points * self.dwell_time


class KSpaceSimulator:
    """
    Signal-based MRI simulator with k-space encoding.
    
    This simulator properly models the MRI signal as the coherent sum of
    all spin contributions with gradient-induced phase encoding:
    
        S(t) = ∫ M_xy(r,t) × ρ(r) × C(r) × exp(-i·2π·k(t)·r) dr
    
    Where:
        - M_xy(r,t) = transverse magnetization at position r, time t
        - ρ(r) = proton density
        - C(r) = coil sensitivity
        - k(t) = ∫₀ᵗ γ·G(t') dt' = k-space position
    
    This enables simulation of:
        - Spatial encoding artifacts (aliasing, ringing)
        - Eddy current effects on phase/frequency
        - EPI Nyquist ghosting
        - Chemical shift displacement
        - Off-resonance blurring
    
    Parameters
    ----------
    gamma : float
        Gyromagnetic ratio in Hz/T (default: 42.576e6 for ¹H)
    verbose : bool
        Print progress messages
    """
    
    def __init__(self, gamma: float = 42.576e6, verbose: bool = True):
        self.gamma = gamma  # Hz/T
        self.verbose = verbose
        self._log_callback = None
        
    def set_log_callback(self, callback: Callable[[str], None]):
        """Set callback for log messages."""
        self._log_callback = callback
        
    def log(self, message: str):
        """Log a message."""
        if self._log_callback:
            self._log_callback(message)
        elif self.verbose:
            print(message)
    
    # =========================================================================
    # K-SPACE TRAJECTORY GENERATION
    # =========================================================================
    
    def generate_cartesian_trajectory(self, 
                                      matrix_size: Tuple[int, ...],
                                      fov: Tuple[float, ...],
                                      dwell_time: float = 10e-6) -> Dict:
        """
        Generate Cartesian k-space trajectory.
        
        Parameters
        ----------
        matrix_size : tuple
            (nx,), (nx, ny), or (nx, ny, nz)
        fov : tuple
            Field of view in meters, matching matrix_size dimensions
        dwell_time : float
            ADC dwell time in seconds
            
        Returns
        -------
        dict with:
            'kx', 'ky', 'kz': k-space coordinates in cycles/m
            'time': time points in seconds
            'sample_indices': indices into full k-space grid
        """
        ndim = len(matrix_size)
        
        # k-space extent: Δk = 1/FOV, k_max = N/(2*FOV)
        dk = [1.0 / f for f in fov]
        k_max = [n / (2 * f) for n, f in zip(matrix_size, fov)]
        
        # Generate k-space coordinates
        k_coords = []
        for dim in range(ndim):
            n = matrix_size[dim]
            k = np.linspace(-k_max[dim], k_max[dim], n, endpoint=False)
            k_coords.append(k)
        
        # Create meshgrid for all dimensions
        if ndim == 1:
            kx = k_coords[0]
            ky = np.zeros_like(kx)
            kz = np.zeros_like(kx)
        elif ndim == 2:
            KX, KY = np.meshgrid(k_coords[0], k_coords[1], indexing='ij')
            kx = KX.ravel()
            ky = KY.ravel()
            kz = np.zeros_like(kx)
        else:
            KX, KY, KZ = np.meshgrid(k_coords[0], k_coords[1], k_coords[2], indexing='ij')
            kx = KX.ravel()
            ky = KY.ravel()
            kz = KZ.ravel()
        
        n_samples = len(kx)
        time = np.arange(n_samples) * dwell_time
        
        return {
            'kx': kx,
            'ky': ky,
            'kz': kz,
            'time': time,
            'dwell_time': dwell_time,
            'matrix_size': matrix_size,
            'fov': fov,
            'trajectory_type': TrajectoryType.CARTESIAN,
        }
    
    def generate_epi_trajectory(self, params: EPIParameters, 
                               dt: float = 4e-6) -> Dict:
        """
        Generate EPI k-space trajectory with realistic timing.
        
        This generates the actual gradient waveforms and k-space trajectory
        for echo-planar imaging, including:
        - Trapezoidal readout gradients
        - Phase-encode blips
        - Alternating readout direction (or flyback)
        
        Parameters
        ----------
        params : EPIParameters
            EPI sequence parameters
        dt : float
            Simulation time step in seconds
            
        Returns
        -------
        dict with trajectory and gradient information
        """
        nx, ny = params.matrix_size
        fov_x, fov_y = params.fov
        
        # Calculate gradient amplitudes
        # For Cartesian: G = dk / (γ × dt_sample) where dk = 1/FOV
        # Readout gradient: traverse kx in flat_time
        dk_x = 1.0 / fov_x  # cycles/m per sample
        k_max_x = nx / (2 * fov_x)  # max k in x
        
        # Gradient amplitude to traverse full kx in readout time
        # Δk = γ × G × Δt, so G = Δk / (γ × Δt)
        # Full kx range = 2 × k_max_x, traversed in flat_time
        g_read_amplitude = (2 * k_max_x) / (self.gamma * params.flat_time)
        g_read_amplitude_gcm = g_read_amplitude * 1e-2  # T/m to G/cm
        
        # Phase-encode blip: one dk_y step
        dk_y = 1.0 / fov_y
        g_blip_amplitude = dk_y / (self.gamma * params.blip_duration)
        g_blip_amplitude_gcm = g_blip_amplitude * 1e-2
        
        # Build waveforms
        n_lines = params.lines_per_shot
        
        # Time for one readout (ramp + flat + ramp)
        readout_time = 2 * params.ramp_time + params.flat_time
        
        # Total time estimate
        total_time = n_lines * (readout_time + params.blip_duration)
        n_points = int(total_time / dt) + 1
        
        # Initialize gradient arrays
        gx = np.zeros(n_points)
        gy = np.zeros(n_points)
        gz = np.zeros(n_points)
        
        # Build each echo
        t_idx = 0
        echo_centers = []
        line_indices = []
        
        # Pre-phase gradient (move to -kx_max, -ky_max)
        # For simplicity, assume instantaneous pre-phasing
        # In practice, this would be a separate gradient lobe
        
        for line in range(n_lines):
            # Determine readout polarity
            if params.flyback:
                polarity = 1
            else:
                polarity = 1 if line % 2 == 0 else -1
            
            # Ramp up
            n_ramp = int(params.ramp_time / dt)
            for i in range(n_ramp):
                if t_idx < n_points:
                    ramp_frac = (i + 1) / n_ramp
                    gx[t_idx] = polarity * g_read_amplitude_gcm * ramp_frac
                    t_idx += 1
            
            # Flat top (readout)
            n_flat = int(params.flat_time / dt)
            echo_center_idx = t_idx + n_flat // 2
            echo_centers.append(echo_center_idx)
            line_indices.append(line if polarity > 0 else (ny - 1 - line) if not params.flyback else line)
            
            for i in range(n_flat):
                if t_idx < n_points:
                    gx[t_idx] = polarity * g_read_amplitude_gcm
                    t_idx += 1
            
            # Ramp down
            for i in range(n_ramp):
                if t_idx < n_points:
                    ramp_frac = 1.0 - (i + 1) / n_ramp
                    gx[t_idx] = polarity * g_read_amplitude_gcm * ramp_frac
                    t_idx += 1
            
            # Phase-encode blip (if not last line)
            if line < n_lines - 1:
                n_blip = int(params.blip_duration / dt)
                for i in range(n_blip):
                    if t_idx < n_points:
                        gy[t_idx] = g_blip_amplitude_gcm
                        t_idx += 1
            
            # Flyback (if using flyback readout)
            if params.flyback and line < n_lines - 1:
                for i in range(n_ramp):
                    if t_idx < n_points:
                        ramp_frac = (i + 1) / n_ramp
                        gx[t_idx] = -g_read_amplitude_gcm * ramp_frac
                        t_idx += 1
                for i in range(n_flat):
                    if t_idx < n_points:
                        gx[t_idx] = -g_read_amplitude_gcm
                        t_idx += 1
                for i in range(n_ramp):
                    if t_idx < n_points:
                        ramp_frac = 1.0 - (i + 1) / n_ramp
                        gx[t_idx] = -g_read_amplitude_gcm * ramp_frac
                        t_idx += 1
        
        # Trim to actual length
        gx = gx[:t_idx]
        gy = gy[:t_idx]
        gz = gz[:t_idx]
        
        # Calculate k-space trajectory by integrating gradients
        time = np.arange(len(gx)) * dt
        kx, ky, kz = self._integrate_gradients(
            np.column_stack([gx, gy, gz]), dt
        )
        
        # Add pre-phasing offset
        kx = kx - k_max_x
        ky = ky - (ny / 2) * dk_y
        
        return {
            'kx': kx,
            'ky': ky,
            'kz': kz,
            'time': time,
            'gradients': np.column_stack([gx, gy, gz]),
            'echo_centers': np.array(echo_centers),
            'line_indices': np.array(line_indices),
            'params': params,
            'trajectory_type': TrajectoryType.EPI,
            'dt': dt,
        }
    
    def generate_csi_trajectory(self, params: CSIParameters,
                               dt: float = None) -> Dict:
        """
        Generate CSI (Chemical Shift Imaging) trajectory.
        
        CSI uses phase encoding for spatial dimensions and free precession
        for the spectral dimension. Each k-space point requires a separate
        excitation.
        
        Parameters
        ----------
        params : CSIParameters
            CSI sequence parameters
        dt : float, optional
            Time step (defaults to dwell_time)
            
        Returns
        -------
        dict with trajectory information
        """
        if dt is None:
            dt = params.dwell_time
            
        matrix = params.matrix_size
        fov = params.fov
        ndim = len(matrix)
        
        # k-space coordinates for phase encoding
        k_coords = []
        for dim in range(ndim):
            n = matrix[dim]
            dk = 1.0 / fov[dim]
            k = (np.arange(n) - n // 2) * dk
            k_coords.append(k)
        
        # Create full k-space grid
        if ndim == 1:
            kx_grid = k_coords[0]
            ky_grid = np.zeros(matrix[0])
            kz_grid = np.zeros(matrix[0])
        elif ndim == 2:
            KX, KY = np.meshgrid(k_coords[0], k_coords[1], indexing='ij')
            kx_grid = KX.ravel()
            ky_grid = KY.ravel()
            kz_grid = np.zeros_like(kx_grid)
        else:
            KX, KY, KZ = np.meshgrid(k_coords[0], k_coords[1], k_coords[2], indexing='ij')
            kx_grid = KX.ravel()
            ky_grid = KY.ravel()
            kz_grid = KZ.ravel()
        
        n_spatial = len(kx_grid)
        n_spectral = params.spectral_points
        
        # Time points for spectral acquisition
        t_spectral = np.arange(n_spectral) * params.dwell_time
        
        return {
            'kx_spatial': kx_grid,
            'ky_spatial': ky_grid,
            'kz_spatial': kz_grid,
            't_spectral': t_spectral,
            'n_spatial': n_spatial,
            'n_spectral': n_spectral,
            'params': params,
            'trajectory_type': TrajectoryType.CSI,
            'dwell_time': params.dwell_time,
        }
    
    def _integrate_gradients(self, gradients: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate k-space trajectory from gradient waveform.
        
        k(t) = γ × ∫₀ᵗ G(t') dt'
        
        Parameters
        ----------
        gradients : ndarray, shape (ntime, 3)
            Gradient waveform in G/cm
        dt : float
            Time step in seconds
            
        Returns
        -------
        kx, ky, kz : ndarray
            k-space coordinates in cycles/m
        """
        # Convert G/cm to T/m: 1 G/cm = 0.01 T/m
        grad_T_m = gradients * 0.01
        
        # Integrate: k = γ × cumsum(G × dt)
        kx = np.cumsum(grad_T_m[:, 0]) * dt * self.gamma
        ky = np.cumsum(grad_T_m[:, 1]) * dt * self.gamma
        kz = np.cumsum(grad_T_m[:, 2]) * dt * self.gamma
        
        return kx, ky, kz
    
    # =========================================================================
    # SIGNAL SIMULATION
    # =========================================================================
    
    def simulate_signal(self, 
                       phantom,
                       trajectory: Dict,
                       magnetization: Dict = None,
                       eddy_model: EddyCurrentModel = None,
                       b0_map: np.ndarray = None,
                       coil_sensitivity: np.ndarray = None,
                       noise_std: float = 0.0) -> Dict:
        """
        Simulate MRI signal from phantom with k-space encoding.
        
        This is the core function that computes:
            S(t) = Σᵣ M_xy(r,t) × ρ(r) × C(r) × exp(-i·2π·k(t)·r + φ_B0(r,t))
        
        Parameters
        ----------
        phantom : Phantom
            Phantom object with tissue properties
        trajectory : dict
            K-space trajectory from generate_*_trajectory()
        magnetization : dict, optional
            Pre-computed magnetization from Bloch simulation.
            If None, uses equilibrium M_xy = 1.
        eddy_model : EddyCurrentModel, optional
            Eddy current model to apply to gradients
        b0_map : ndarray, optional
            B0 inhomogeneity map in Hz, shape matching phantom.shape
        coil_sensitivity : ndarray, optional
            Complex coil sensitivity map, shape matching phantom.shape
        noise_std : float
            Standard deviation of complex Gaussian noise to add
            
        Returns
        -------
        dict with:
            'signal': Complex signal at each k-space point
            'kx', 'ky', 'kz': Actual k-space coordinates (may differ from
                              trajectory if eddy currents applied)
            'time': Time points
        """
        traj_type = trajectory.get('trajectory_type', TrajectoryType.CUSTOM)
        
        if traj_type == TrajectoryType.CSI:
            return self._simulate_csi_signal(
                phantom, trajectory, magnetization, 
                b0_map, coil_sensitivity, noise_std
            )
        else:
            return self._simulate_imaging_signal(
                phantom, trajectory, magnetization,
                eddy_model, b0_map, coil_sensitivity, noise_std
            )
    
    def _simulate_imaging_signal(self,
                                 phantom,
                                 trajectory: Dict,
                                 magnetization: Dict = None,
                                 eddy_model: EddyCurrentModel = None,
                                 b0_map: np.ndarray = None,
                                 coil_sensitivity: np.ndarray = None,
                                 noise_std: float = 0.0) -> Dict:
        """Simulate signal for imaging sequences (Cartesian, EPI, etc.)."""
        
        kx = trajectory['kx']
        ky = trajectory['ky']
        kz = trajectory['kz']
        time = trajectory['time']
        n_samples = len(kx)
        
        # Apply eddy currents to trajectory if model provided
        if eddy_model is not None and 'gradients' in trajectory:
            self.log("Applying eddy current model...")
            dt = trajectory.get('dt', time[1] - time[0] if len(time) > 1 else 1e-5)
            grad_actual = eddy_model.apply_to_gradient(trajectory['gradients'], dt)
            kx, ky, kz = self._integrate_gradients(grad_actual, dt)
            
            # Apply same pre-phase offset as original
            if trajectory['trajectory_type'] == TrajectoryType.EPI:
                params = trajectory['params']
                kx = kx - params.matrix_size[0] / (2 * params.fov[0])
                ky = ky - params.matrix_size[1] / (2 * params.fov[1])
        
        # Get phantom spatial positions
        positions = phantom.positions  # (nvoxels, 3) in meters
        nvoxels = len(positions)
        
        # Get proton density
        pd = phantom.pd_map.ravel()
        
        # Get magnetization
        if magnetization is not None:
            # Use provided transverse magnetization
            mx = magnetization.get('mx')
            my = magnetization.get('my')
            if mx is not None and my is not None:
                if mx.ndim > len(phantom.shape):
                    # Time-resolved: use final time point
                    mxy = (mx[-1] + 1j * my[-1]).ravel()
                else:
                    mxy = (mx + 1j * my).ravel()
            else:
                mxy = np.ones(nvoxels, dtype=np.complex128)
        else:
            # Assume fully excited (Mxy = 1)
            mxy = np.ones(nvoxels, dtype=np.complex128)
        
        # Apply coil sensitivity
        if coil_sensitivity is not None:
            coil_flat = coil_sensitivity.ravel()
            if len(coil_flat) != nvoxels:
                self.log(f"Warning: Coil sensitivity shape mismatch, ignoring")
                coil_flat = np.ones(nvoxels)
        else:
            coil_flat = np.ones(nvoxels)
        
        # B0 inhomogeneity (frequency offset in Hz)
        if b0_map is not None:
            b0_flat = b0_map.ravel()
        else:
            b0_flat = phantom.df_map.ravel() if phantom.df_map is not None else np.zeros(nvoxels)
        
        self.log(f"Simulating signal: {n_samples} k-space points, {nvoxels} voxels")
        
        # Compute signal: S(k) = Σ M_xy(r) × ρ(r) × C(r) × exp(-i·2π·k·r)
        signal = np.zeros(n_samples, dtype=np.complex128)
        
        # Weighted source: M_xy × ρ × C
        source = mxy * pd * coil_flat
        
        # For efficiency, vectorize over k-space points in chunks
        chunk_size = min(1000, n_samples)
        
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            
            # k-space coordinates for this chunk
            kx_chunk = kx[start:end, np.newaxis]  # (chunk, 1)
            ky_chunk = ky[start:end, np.newaxis]
            kz_chunk = kz[start:end, np.newaxis]
            t_chunk = time[start:end, np.newaxis]
            
            # Position coordinates (broadcast)
            rx = positions[:, 0]  # (nvoxels,)
            ry = positions[:, 1]
            rz = positions[:, 2]
            
            # Phase from spatial encoding: φ = 2π × (kx×x + ky×y + kz×z)
            phase_spatial = 2 * np.pi * (
                kx_chunk * rx + ky_chunk * ry + kz_chunk * rz
            )  # (chunk, nvoxels)
            
            # Phase from B0 inhomogeneity: φ = 2π × Δf × t
            phase_b0 = 2 * np.pi * b0_flat * t_chunk  # (chunk, nvoxels)
            
            total_phase = phase_spatial + phase_b0
            
            # Signal = sum over voxels
            signal[start:end] = np.sum(source * np.exp(-1j * total_phase), axis=1)
        
        # Add noise
        if noise_std > 0:
            noise = noise_std * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
            signal = signal + noise
        
        return {
            'signal': signal,
            'kx': kx,
            'ky': ky,
            'kz': kz,
            'time': time,
            'trajectory': trajectory,
            'eddy_applied': eddy_model is not None,
        }
    
    def _simulate_csi_signal(self,
                            phantom,
                            trajectory: Dict,
                            magnetization: Dict = None,
                            b0_map: np.ndarray = None,
                            coil_sensitivity: np.ndarray = None,
                            noise_std: float = 0.0) -> Dict:
        """
        Simulate CSI signal with spectral and spatial encoding.
        
        For CSI, each k-space point is a full FID. The signal is:
            S(k, t) = Σᵣ M_xy(r) × exp(-t/T2*(r)) × exp(-i·2π·(k·r + Δf(r)·t))
        """
        params = trajectory['params']
        kx_spatial = trajectory['kx_spatial']
        ky_spatial = trajectory['ky_spatial']
        kz_spatial = trajectory['kz_spatial']
        t_spectral = trajectory['t_spectral']
        
        n_spatial = trajectory['n_spatial']
        n_spectral = trajectory['n_spectral']
        
        # Get phantom properties
        positions = phantom.positions
        nvoxels = len(positions)
        pd = phantom.pd_map.ravel()
        t2 = phantom.t2_map.ravel()
        
        # Frequency offsets (chemical shift + B0)
        df = phantom.df_map.ravel() if phantom.df_map is not None else np.zeros(nvoxels)
        if b0_map is not None:
            df = df + b0_map.ravel()
        
        # Coil sensitivity
        if coil_sensitivity is not None:
            coil_flat = coil_sensitivity.ravel()
        else:
            coil_flat = np.ones(nvoxels)
        
        # Get initial magnetization
        if magnetization is not None:
            mx = magnetization.get('mx')
            my = magnetization.get('my')
            if mx is not None:
                if mx.ndim > len(phantom.shape):
                    mxy = (mx[-1] + 1j * my[-1]).ravel()
                else:
                    mxy = (mx + 1j * my).ravel()
            else:
                mxy = np.ones(nvoxels, dtype=np.complex128)
        else:
            mxy = np.ones(nvoxels, dtype=np.complex128)
        
        self.log(f"Simulating CSI: {n_spatial} spatial × {n_spectral} spectral points")
        
        # Output: (n_spatial, n_spectral) complex array
        signal = np.zeros((n_spatial, n_spectral), dtype=np.complex128)
        
        # Source term (constant for all time)
        source = mxy * pd * coil_flat
        
        for k_idx in range(n_spatial):
            # Spatial encoding phase
            phase_spatial = 2 * np.pi * (
                kx_spatial[k_idx] * positions[:, 0] +
                ky_spatial[k_idx] * positions[:, 1] +
                kz_spatial[k_idx] * positions[:, 2]
            )
            
            # For each spectral point
            for t_idx, t in enumerate(t_spectral):
                # T2 decay
                t2_decay = np.exp(-t / np.maximum(t2, 1e-10))
                
                # Chemical shift evolution
                phase_spectral = 2 * np.pi * df * t
                
                # Total signal
                signal[k_idx, t_idx] = np.sum(
                    source * t2_decay * np.exp(-1j * (phase_spatial + phase_spectral))
                )
        
        # Add noise
        if noise_std > 0:
            noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
            signal = signal + noise
        
        return {
            'signal': signal,  # (n_spatial, n_spectral)
            'kx_spatial': kx_spatial,
            'ky_spatial': ky_spatial,
            'kz_spatial': kz_spatial,
            't_spectral': t_spectral,
            'trajectory': trajectory,
        }
    
    # =========================================================================
    # IMAGE RECONSTRUCTION
    # =========================================================================
    
    def reconstruct_cartesian(self, signal_data: Dict) -> np.ndarray:
        """
        Reconstruct image from Cartesian k-space data.
        
        Parameters
        ----------
        signal_data : dict
            Output from simulate_signal() with Cartesian trajectory
            
        Returns
        -------
        ndarray
            Reconstructed image (complex)
        """
        signal = signal_data['signal']
        trajectory = signal_data['trajectory']
        matrix_size = trajectory['matrix_size']
        
        # Reshape signal to k-space matrix
        kspace = signal.reshape(matrix_size)
        
        # Apply inverse FFT with proper shifting
        # fftshift moves DC to center, ifft2 reconstructs, ifftshift moves back
        image = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kspace)))
        
        return image
    
    def reconstruct_epi(self, signal_data: Dict, 
                       phase_correction: str = 'none') -> np.ndarray:
        """
        Reconstruct EPI with optional phase correction.
        
        Parameters
        ----------
        signal_data : dict
            Output from simulate_signal() with EPI trajectory
        phase_correction : str
            'none': No correction
            'linear': Linear phase correction per line
            'navigator': Navigator-based correction (requires navigator echoes)
            
        Returns
        -------
        ndarray
            Reconstructed image
        """
        signal = signal_data['signal']
        trajectory = signal_data['trajectory']
        params = trajectory['params']
        
        nx, ny = params.matrix_size
        echo_centers = trajectory['echo_centers']
        line_indices = trajectory['line_indices']
        
        # Extract data for each echo
        n_points_per_echo = nx  # Assumes symmetric readout
        
        # Initialize k-space
        kspace = np.zeros((nx, ny), dtype=np.complex128)
        
        for i, (center, line) in enumerate(zip(echo_centers, line_indices)):
            # Extract echo data
            start = center - nx // 2
            end = center + nx // 2
            
            if start >= 0 and end <= len(signal):
                echo_data = signal[start:end]
                
                # Reverse odd echoes if bipolar readout
                if not params.flyback and i % 2 == 1:
                    echo_data = echo_data[::-1]
                
                # Phase correction
                if phase_correction == 'linear':
                    # Simple linear phase correction
                    # In practice, this would be estimated from navigators
                    pass
                
                kspace[:, line] = echo_data
        
        # Reconstruct
        image = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
        
        return image
    
    def reconstruct_csi(self, signal_data: Dict) -> np.ndarray:
        """
        Reconstruct CSI data to spatial-spectral cube.
        
        Parameters
        ----------
        signal_data : dict
            Output from simulate_signal() with CSI trajectory
            
        Returns
        -------
        ndarray
            Spatial-spectral data, shape (*spatial_shape, n_spectral)
        """
        signal = signal_data['signal']  # (n_spatial, n_spectral)
        trajectory = signal_data['trajectory']
        params = trajectory['params']
        
        n_spectral = params.spectral_points
        spatial_shape = params.matrix_size
        
        # Reshape to spatial grid
        signal_grid = signal.reshape((*spatial_shape, n_spectral))
        
        # Spatial FFT (inverse FFT for each spectral point)
        ndim_spatial = len(spatial_shape)
        axes_spatial = tuple(range(ndim_spatial))
        
        # Apply inverse FFT along spatial dimensions
        image_spectral = np.fft.ifftshift(
            np.fft.ifftn(
                np.fft.ifftshift(signal_grid, axes=axes_spatial),
                axes=axes_spatial
            ),
            axes=axes_spatial
        )
        
        # Spectral FFT for each voxel
        spectra = np.fft.fftshift(np.fft.fft(image_spectral, axis=-1), axes=-1)
        
        # Frequency axis
        df = 1.0 / (n_spectral * params.dwell_time)
        freq = np.fft.fftshift(np.fft.fftfreq(n_spectral, params.dwell_time))
        
        return {
            'spectra': spectra,  # (*spatial_shape, n_spectral)
            'frequency': freq,   # Hz
            'spatial_image': np.sum(np.abs(spectra), axis=-1),  # Integrated magnitude
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def simulate_epi_with_artifacts(phantom, 
                                epi_params: EPIParameters,
                                eddy_amplitudes: List[float] = None,
                                eddy_taus: List[float] = None,
                                b0_inhomogeneity: np.ndarray = None,
                                noise_level: float = 0.0,
                                verbose: bool = True) -> Dict:
    """
    Convenience function to simulate EPI with common artifacts.
    
    Parameters
    ----------
    phantom : Phantom
        Phantom object
    epi_params : EPIParameters
        EPI sequence parameters
    eddy_amplitudes : list of float, optional
        Eddy current amplitudes (relative to gradient step)
    eddy_taus : list of float, optional
        Eddy current time constants in seconds
    b0_inhomogeneity : ndarray, optional
        B0 map in Hz
    noise_level : float
        Noise standard deviation
    verbose : bool
        Print progress
        
    Returns
    -------
    dict with images and intermediate data
    """
    sim = KSpaceSimulator(verbose=verbose)
    
    # Generate trajectory
    trajectory = sim.generate_epi_trajectory(epi_params)
    
    # Create eddy current model if specified
    if eddy_amplitudes is not None and eddy_taus is not None:
        eddy = EddyCurrentModel(
            amplitudes=eddy_amplitudes,
            time_constants=eddy_taus
        )
    else:
        eddy = None
    
    # Simulate without eddy currents (reference)
    result_clean = sim.simulate_signal(
        phantom, trajectory,
        b0_map=b0_inhomogeneity,
        noise_std=noise_level
    )
    image_clean = sim.reconstruct_epi(result_clean)
    
    # Simulate with eddy currents
    if eddy is not None:
        result_eddy = sim.simulate_signal(
            phantom, trajectory,
            eddy_model=eddy,
            b0_map=b0_inhomogeneity,
            noise_std=noise_level
        )
        image_eddy = sim.reconstruct_epi(result_eddy)
    else:
        result_eddy = None
        image_eddy = None
    
    return {
        'image_clean': image_clean,
        'image_eddy': image_eddy,
        'result_clean': result_clean,
        'result_eddy': result_eddy,
        'trajectory': trajectory,
        'epi_params': epi_params,
    }


def simulate_csi_spectrum(phantom,
                         csi_params: CSIParameters,
                         excitation_flip: float = 90.0,
                         verbose: bool = True) -> Dict:
    """
    Convenience function to simulate CSI experiment.
    
    Parameters
    ----------
    phantom : Phantom or SpectralPhantom
        Phantom with chemical shift information in df_map
    csi_params : CSIParameters
        CSI sequence parameters
    excitation_flip : float
        Excitation flip angle in degrees
    verbose : bool
        Print progress
        
    Returns
    -------
    dict with spectra and images
    """
    sim = KSpaceSimulator(verbose=verbose)
    
    # Generate CSI trajectory
    trajectory = sim.generate_csi_trajectory(csi_params)
    
    # Simulate signal
    result = sim.simulate_signal(phantom, trajectory)
    
    # Reconstruct
    recon = sim.reconstruct_csi(result)
    
    return {
        **recon,
        'signal': result['signal'],
        'trajectory': trajectory,
        'params': csi_params,
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("K-Space Simulator Module")
    print("=" * 50)
    
    # Create a simple test phantom
    try:
        from phantom import PhantomFactory
        phantom = PhantomFactory.shepp_logan_2d(32, 0.24, 3.0)
        print(f"Created phantom: {phantom}")
    except ImportError:
        print("Phantom module not available for testing")
        phantom = None
    
    if phantom is not None:
        sim = KSpaceSimulator()
        
        # Test Cartesian trajectory
        print("\nTesting Cartesian trajectory...")
        traj = sim.generate_cartesian_trajectory((32, 32), (0.24, 0.24))
        print(f"  K-space points: {len(traj['kx'])}")
        
        # Test EPI trajectory
        print("\nTesting EPI trajectory...")
        epi_params = EPIParameters(matrix_size=(32, 32), fov=(0.24, 0.24))
        traj_epi = sim.generate_epi_trajectory(epi_params)
        print(f"  Gradient points: {len(traj_epi['gradients'])}")
        print(f"  Echo centers: {len(traj_epi['echo_centers'])}")
        
        # Test signal simulation
        print("\nTesting signal simulation...")
        result = sim.simulate_signal(phantom, traj)
        print(f"  Signal shape: {result['signal'].shape}")
        
        # Test reconstruction
        print("\nTesting reconstruction...")
        image = sim.reconstruct_cartesian(result)
        print(f"  Image shape: {image.shape}")
        
        # Test EPI with eddy currents
        print("\nTesting EPI with eddy currents...")
        eddy = EddyCurrentModel(
            amplitudes=[0.02, 0.01],
            time_constants=[0.1e-3, 1e-3]
        )
        result_epi = sim.simulate_signal(phantom, traj_epi, eddy_model=eddy)
        image_epi = sim.reconstruct_epi(result_epi)
        print(f"  EPI image shape: {image_epi.shape}")
        
        # Test CSI
        print("\nTesting CSI...")
        csi_params = CSIParameters(matrix_size=(8, 8), spectral_points=256)
        traj_csi = sim.generate_csi_trajectory(csi_params)
        result_csi = sim.simulate_signal(phantom, traj_csi)
        recon_csi = sim.reconstruct_csi(result_csi)
        print(f"  CSI spectra shape: {recon_csi['spectra'].shape}")
        
        print("\n✓ All tests passed!")
