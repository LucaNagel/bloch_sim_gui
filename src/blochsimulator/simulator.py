"""
blochsimulator.py - High-level Python API for Bloch equation simulations

This module provides user-friendly classes and functions for MRI pulse sequence
simulation using the Bloch equations.

Author: Your Name
Date: 2024
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict
from scipy import signal
from dataclasses import dataclass
import h5py
from pathlib import Path
from . import __version__

# Import the Cython extension (will be available after building)
try:
    from .blochsimulator_cy import (
        simulate_bloch, 
        simulate_bloch_parallel,
        calculate_signal,
        design_rf_pulse
    )
except ImportError:
    print("Warning: Cython extension not built. Run 'python setup.py build_ext --inplace' first.")

    # Define dummy functions for testing
    def simulate_bloch(*args, **kwargs):
        raise NotImplementedError("Build the Cython extension first")
    def simulate_bloch_parallel(*args, **kwargs):
        raise NotImplementedError("Build the Cython extension first")
    def calculate_signal(mx, my, mz, receiver_phase=0.0):
        phase_factor = np.exp(-1j * receiver_phase)
        return (mx + 1j * my) * phase_factor
def design_rf_pulse(pulse_type='rect', duration=1e-3, flip_angle=90,
                    time_bw_product=4, npoints=100, freq_offset=0.0):
        """
        Pure-Python fallback for RF design so imports work even without the extension.

        Parameters
        ----------
        pulse_type : str
            Type of pulse ('rect', 'sinc', 'gaussian', 'adiabatic_half', 'adiabatic_full', 'bir4')
        duration : float
            Pulse duration in seconds
        flip_angle : float
            Flip angle in degrees
        time_bw_product : float
            Time-bandwidth product for sinc/gaussian pulses
        npoints : int
            Number of time points
        freq_offset : float
            Frequency offset in Hz (default 0). Applies phase modulation: B1 * exp(2πi*f*t)
            Positive offset shifts the pulse frequency higher.

        Returns
        -------
        b1 : complex ndarray
            Complex B1 field in Gauss
        time : ndarray
            Time points in seconds
        """
        time = np.linspace(0, duration, npoints, endpoint=False)
        dt = duration / npoints
        gamma = 4258.0  # Hz/Gauss for protons
        flip_rad = np.deg2rad(flip_angle)
        target_area = flip_rad / (gamma * 2 * np.pi)  # integral of B1 over time (Gauss * s)
        if pulse_type == 'rect':
            b1 = np.ones(npoints) * (target_area / duration)
        elif pulse_type == 'sinc':
            t_centered = time - duration/2
            bw = time_bw_product / duration
            envelope = np.sinc(bw * t_centered)
            area = np.trapezoid(envelope, time)
            b1 = envelope * (target_area / area)
        elif pulse_type == 'gaussian':
            t_centered = time - duration/2
            sigma = duration / (2 * np.sqrt(2 * np.log(2)) * time_bw_product)
            envelope = np.exp(-t_centered**2 / (2 * sigma**2))
            area = np.trapezoid(envelope, time)
            b1 = envelope * (target_area / area)
        elif pulse_type == 'adiabatic_half':
            # Adiabatic Half Passage (AHP): 90° excitation pulse
            # Uses hyperbolic secant amplitude + tanh frequency modulation
            # Magnetization tracks effective field through 90° rotation
            #
            # For adiabatic pulses, the flip angle is determined by the adiabaticity
            # parameter κ = γ·B1_max·T / β, NOT by the pulse area.
            # The flip_angle parameter controls B1_max to achieve the desired rotation.
            t_centered = time - duration/2
            beta = time_bw_product  # Modulation parameter (typically 4-8)

            # HS amplitude modulation: A(t) = A0 * sech(beta * t / T)
            # Normalized amplitude envelope (max = 1.0)
            amplitude = 1.0 / np.cosh(beta * t_centered / (duration/2))

            # Frequency modulation using tanh (standard for HS pulses)
            # Frequency sweeps from +BW/2 to -BW/2 (or vice versa)
            # Delta_omega(t) = -omega_max * tanh(beta * t / T)
            bandwidth_hz = time_bw_product / duration
            omega_max = np.pi * bandwidth_hz  # Max frequency offset (rad/s)

            # Frequency sweep using tanh
            freq_modulation = -omega_max * np.tanh(beta * t_centered / (duration/2))

            # Integrate to get phase: phi(t) = integral(omega(t) dt)
            dt = duration / npoints
            instantaneous_phase = np.cumsum(freq_modulation * dt)

            # Complex B1: A(t) * exp(i*phi(t))
            b1_complex = amplitude * np.exp(1j * instantaneous_phase)

            # For adiabatic pulses, scale by flip_angle to control B1_max directly
            # AHP typically achieves 90° when adiabaticity κ ≈ 5-10
            # User adjusts flip_angle to control the RF amplitude (B1_max in Gauss)
            # flip_angle here acts as a B1 scaling factor, not a target rotation
            target_flip_rad = np.deg2rad(flip_angle)
            b1_max_gauss = target_flip_rad / (gamma * 2 * np.pi * duration)
            b1 = b1_complex * b1_max_gauss

        elif pulse_type == 'adiabatic_full':
            # Adiabatic Full Passage (AFP): 180° inversion pulse
            # Uses hyperbolic secant amplitude + tanh frequency modulation
            # Magnetization follows effective field through full 180° inversion
            #
            # For adiabatic pulses, the flip angle is determined by the adiabaticity
            # parameter κ = γ·B1_max·T / β, NOT by the pulse area.
            # The flip_angle parameter controls B1_max to achieve the desired rotation.
            t_centered = time - duration/2
            beta = time_bw_product  # Typically 4-8 for good adiabatic condition

            # HS amplitude modulation: A(t) = A0 * sech(beta * t / T)
            # Normalized amplitude envelope (max = 1.0)
            amplitude = 1.0 / np.cosh(beta * t_centered / (duration/2))

            # Frequency modulation using tanh (sweeps through full resonance)
            bandwidth_hz = time_bw_product / duration
            omega_max = np.pi * bandwidth_hz

            # Full sweep: omega goes from +omega_max to -omega_max
            freq_modulation = -omega_max * np.tanh(beta * t_centered / (duration/2))

            # Integrate to get phase
            dt = duration / npoints
            instantaneous_phase = np.cumsum(freq_modulation * dt)

            # Complex B1
            b1_complex = amplitude * np.exp(1j * instantaneous_phase)

            # For adiabatic pulses, scale by flip_angle to control B1_max directly
            # AFP typically achieves 180° when adiabaticity κ ≈ 5-10
            # User adjusts flip_angle to control the RF amplitude (B1_max in Gauss)
            # flip_angle here acts as a B1 scaling factor, not a target rotation
            target_flip_rad = np.deg2rad(flip_angle)
            b1_max_gauss = target_flip_rad / (gamma * 2 * np.pi * duration)
            b1 = b1_complex * b1_max_gauss

        elif pulse_type == 'bir4':
            # BIR-4 (B1-Insensitive Rotation): Composite adiabatic pulse for arbitrary flip angles
            # Structure: 4 segments that produce plane rotation insensitive to B1 inhomogeneity
            # Composed of: AHP - 180° - AHP_inverse - 180°
            # This implementation uses a simplified HS-based BIR-4
            #
            # For adiabatic pulses, the flip angle is determined by the adiabaticity
            # parameter κ = γ·B1_max·T / β, NOT by the pulse area.
            # The flip_angle parameter controls B1_max to achieve the desired rotation.

            beta = time_bw_product

            # Divide pulse into 4 segments
            n_seg = npoints // 4
            t_seg = duration / 4

            # Segment times
            t1 = time[:n_seg] - time[n_seg//2]
            t2 = time[n_seg:2*n_seg] - time[3*n_seg//2]
            t3 = time[2*n_seg:3*n_seg] - time[5*n_seg//2]
            t4 = time[3*n_seg:] - time[7*n_seg//2]

            bandwidth_hz = time_bw_product / t_seg
            omega_max = np.pi * bandwidth_hz

            # Segment 1: AHP (90°)
            amp1 = 1.0 / np.cosh(beta * t1 / (t_seg/2))
            freq1 = -omega_max * np.tanh(beta * t1 / (t_seg/2))
            phase1 = np.cumsum(freq1 * (t_seg / n_seg))
            b1_seg1 = amp1 * np.exp(1j * phase1)

            # Segment 2: 180° phase shift + reverse AHP
            amp2 = 1.0 / np.cosh(beta * t2 / (t_seg/2))
            freq2 = omega_max * np.tanh(beta * t2 / (t_seg/2))  # Reversed
            phase2 = np.cumsum(freq2 * (t_seg / n_seg)) + phase1[-1]
            b1_seg2 = amp2 * np.exp(1j * (phase2 + np.pi))  # 180° phase shift

            # Segment 3: Inverse AHP
            amp3 = 1.0 / np.cosh(beta * t3 / (t_seg/2))
            freq3 = omega_max * np.tanh(beta * t3 / (t_seg/2))
            phase3 = np.cumsum(freq3 * (t_seg / n_seg)) + phase2[-1]
            b1_seg3 = amp3 * np.exp(1j * phase3)

            # Segment 4: 180° phase shift + AHP
            amp4 = 1.0 / np.cosh(beta * t4 / (t_seg/2))
            freq4 = -omega_max * np.tanh(beta * t4 / (t_seg/2))
            phase4 = np.cumsum(freq4 * (t_seg / n_seg)) + phase3[-1]
            b1_seg4 = amp4 * np.exp(1j * (phase4 + np.pi))  # 180° phase shift

            # Concatenate segments
            b1_complex = np.concatenate([b1_seg1, b1_seg2, b1_seg3, b1_seg4])

            # Pad if needed due to rounding
            if len(b1_complex) < npoints:
                b1_complex = np.pad(b1_complex, (0, npoints - len(b1_complex)), mode='edge')
            elif len(b1_complex) > npoints:
                b1_complex = b1_complex[:npoints]

            # For adiabatic pulses, scale by flip_angle to control B1_max directly
            # User adjusts flip_angle to control the RF amplitude (B1_max in Gauss)
            # flip_angle here acts as a B1 scaling factor, not a target rotation
            target_flip_rad = np.deg2rad(flip_angle)
            b1_max_gauss = target_flip_rad / (gamma * 2 * np.pi * duration)
            b1 = b1_complex * b1_max_gauss
        else:
            raise ValueError(f"Unknown pulse type: {pulse_type}")

        # Apply frequency offset as phase modulation
        if freq_offset != 0.0:
            phase_modulation = np.exp(2j * np.pi * freq_offset * time)
            b1 = b1 * phase_modulation

        return b1.astype(complex), time


# Import pulse loader (optional - gracefully handle if module not available)
try:
    from pulse_loader import load_pulse, load_pulse_from_file, get_pulse_library
except ImportError:
    # Define dummy functions if pulse_loader not available
    def load_pulse(*args, **kwargs):
        raise ImportError("pulse_loader module not available")
    def load_pulse_from_file(*args, **kwargs):
        raise ImportError("pulse_loader module not available")
    def get_pulse_library(*args, **kwargs):
        raise ImportError("pulse_loader module not available")


@dataclass
class TissueParameters:
    """
    Container for tissue parameters.
    
    Attributes
    ----------
    name : str
        Tissue name
    t1 : float
        T1 relaxation time in seconds
    t2 : float
        T2 relaxation time in seconds
    t2_star : float
        T2* relaxation time in seconds
    density : float
        Proton density (relative)
    """
    name: str
    t1: float
    t2: float
    t2_star: float = None
    density: float = 1.0
    
    @classmethod
    def gray_matter(cls, field_strength=3.0):
        """Gray matter parameters at different field strengths."""
        if field_strength == 1.5:
            return cls("Gray Matter", t1=0.95, t2=0.100)
        elif field_strength == 3.0:
            return cls("Gray Matter", t1=1.33, t2=0.083)
        elif field_strength == 7.0:
            return cls("Gray Matter", t1=1.92, t2=0.047)
        else:
            raise ValueError(f"No data for {field_strength}T")
    
    @classmethod
    def white_matter(cls, field_strength=3.0):
        """White matter parameters at different field strengths."""
        if field_strength == 1.5:
            return cls("White Matter", t1=0.65, t2=0.070)
        elif field_strength == 3.0:
            return cls("White Matter", t1=0.83, t2=0.070)
        elif field_strength == 7.0:
            return cls("White Matter", t1=1.22, t2=0.046)
        else:
            raise ValueError(f"No data for {field_strength}T")
    
    @classmethod
    def csf(cls, field_strength=3.0):
        """CSF parameters at different field strengths."""
        if field_strength == 1.5:
            return cls("CSF", t1=2.5, t2=2.0)
        elif field_strength == 3.0:
            return cls("CSF", t1=3.8, t2=2.0)
        elif field_strength == 7.0:
            return cls("CSF", t1=4.4, t2=1.5)
        else:
            raise ValueError(f"No data for {field_strength}T")


class PulseSequence:
    """
    Base class for MRI pulse sequences.
    """
    
    def __init__(self, fov: float = 0.24, matrix_size: int = 256,
                 slice_thickness: float = 0.005, **kwargs):
        """
        Initialize pulse sequence.
        
        Parameters
        ----------
        fov : float
            Field of view in meters
        matrix_size : int
            Matrix size (assumes square matrix)
        slice_thickness : float
            Slice thickness in meters
        """
        self.fov = fov
        self.matrix_size = matrix_size
        self.slice_thickness = slice_thickness
        self.gamma = 42.576e6  # Hz/T for protons
        
        # Calculate resolution
        self.resolution = fov / matrix_size
        
        # Initialize sequence components
        self.rf_pulses = []
        self.gradients = []
        self.adc_times = []
        self.time_points = []
        
    def add_rf_pulse(self, b1: np.ndarray, time: np.ndarray, 
                     phase: float = 0.0):
        """Add an RF pulse to the sequence."""
        self.rf_pulses.append({
            'b1': b1 * np.exp(1j * phase),
            'time': time
        })
        
    def add_gradient(self, axis: str, amplitude: float, 
                    duration: float, time: float):
        """Add a gradient to the sequence."""
        self.gradients.append({
            'axis': axis,
            'amplitude': amplitude,
            'duration': duration,
            'time': time
        })
        
    def compile(self, dt: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compile the sequence into arrays for simulation.
        
        Returns
        -------
        b1 : ndarray
            Complex B1 field
        gradients : ndarray
            Gradient waveforms [Gx, Gy, Gz]
        time : ndarray
            Time points
        """
        # Implementation depends on specific sequence
        raise NotImplementedError("Subclasses must implement compile()")


class SpinEcho(PulseSequence):
    """
    Spin echo pulse sequence.
    """
    
    def __init__(self, te: float, tr: float, custom_excitation=None, custom_refocusing=None, slice_thickness: float = 0.005,
                 slice_gradient_override: Optional[float] = None, echo_count: int = 1, rf_freq_offset: float = 0.0, **kwargs):
        """
        Initialize spin echo sequence.

        Parameters
        ----------
        te : float
            Echo time in seconds
        tr : float
            Repetition time in seconds
        rf_freq_offset : float
            RF frequency offset in Hz (default 0)
        """
        super().__init__(slice_thickness=slice_thickness, **kwargs)
        self.te = te
        self.tr = tr
        self.custom_excitation = custom_excitation
        self.custom_refocusing = custom_refocusing
        self.slice_gradient_override = slice_gradient_override
        self.echo_count = max(1, int(echo_count))
        self.rf_freq_offset = rf_freq_offset
        
    def compile(self, dt: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compile spin echo sequence."""
        # Ensure timeline covers all requested echoes (echo spacing = TE)
        min_duration = (self.echo_count + 0.5) * self.te + 1e-3  # include some buffer
        total_duration = max(self.tr, min_duration)
        npoints = int(np.ceil(total_duration / dt))
        
        # Initialize arrays
        b1 = np.zeros(npoints, dtype=complex)
        gradients = np.zeros((npoints, 3))
        time = np.arange(npoints) * dt
        
        # 90-degree excitation pulse
        if self.custom_excitation is not None:
            exc_b1, exc_time = self.custom_excitation
            exc_b1 = np.asarray(exc_b1, dtype=complex)
            n_exc = min(len(exc_b1), npoints)
            b1[:n_exc] = exc_b1[:n_exc]
            exc_duration = n_exc * dt
        else:
            exc_pulse, _ = design_rf_pulse('sinc', duration=1e-3,
                                          flip_angle=90, npoints=int(1e-3/dt),
                                          freq_offset=self.rf_freq_offset)
            n_exc = len(exc_pulse)
            b1[:n_exc] = exc_pulse
            exc_duration = 1e-3
            
        # Refocusing pulse
        if self.custom_refocusing is not None:
            ref_b1, _ = self.custom_refocusing
            ref_pulse = np.asarray(ref_b1, dtype=complex)
        else:
            # Default refocusing: classic 180° sinc
            ref_pulse, _ = design_rf_pulse('sinc', duration=2e-3,
                                          flip_angle=180, npoints=int(2e-3/dt),
                                          freq_offset=self.rf_freq_offset)

        for echo_idx in range(self.echo_count):
            ref_time = (0.5 + echo_idx) * self.te
            ref_start = int(ref_time / dt)
            if ref_start + len(ref_pulse) < npoints:
                b1[ref_start:ref_start+len(ref_pulse)] = ref_pulse
        
        # Add slice selection gradients
        # (simplified - real implementation would be more complex)
        # Slice gradient G (G/cm) = BW(Hz) / (gamma(Hz/G) * thickness(cm))
        bw_hz = 4.0 / max(exc_duration, dt)
        gamma_hz_per_g = 4258.0
        thickness_cm = max(self.slice_thickness, 1e-3) * 100.0
        if self.slice_gradient_override is not None and self.slice_gradient_override > 0:
            gz_amp = self.slice_gradient_override
        else:
            gz_amp = bw_hz / (gamma_hz_per_g * thickness_cm)
        gradients[:max(n_exc, 1), 2] = gz_amp
        for echo_idx in range(self.echo_count):
            ref_time = (0.5 + echo_idx) * self.te
            ref_start = int(ref_time / dt)
            if ref_start + len(ref_pulse) < npoints:
                gradients[ref_start:ref_start+len(ref_pulse), 2] = gz_amp
        
        return b1, gradients, time


class SpinEchoTipAxis(PulseSequence):
    """
    Spin echo where the refocusing 180 is applied around the axis of the tipped magnetization.

    Implemented by phase-shifting the 180 pulse by +90 degrees relative to the excitation phase
    (CPMG-style: 90° about X, 180° about Y).
    """

    def __init__(self, te: float, tr: float, custom_excitation=None, custom_refocusing=None, slice_thickness: float = 0.005,
                 slice_gradient_override: Optional[float] = None, echo_count: int = 1, **kwargs):
        super().__init__(slice_thickness=slice_thickness, **kwargs)
        self.te = te
        self.tr = tr
        self.custom_excitation = custom_excitation
        self.custom_refocusing = custom_refocusing
        self.slice_gradient_override = slice_gradient_override
        self.echo_count = max(1, int(echo_count))

    def compile(self, dt: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_duration = (self.echo_count + 0.5) * self.te + 1e-3
        total_duration = max(self.tr, min_duration)
        npoints = int(np.ceil(total_duration / dt))

        b1 = np.zeros(npoints, dtype=complex)
        gradients = np.zeros((npoints, 3))
        time = np.arange(npoints) * dt

        # Excitation pulse
        if self.custom_excitation is not None:
            exc_b1, exc_time = self.custom_excitation
            exc_b1 = np.asarray(exc_b1, dtype=complex)
            n_exc = min(len(exc_b1), npoints)
            b1[:n_exc] = exc_b1[:n_exc]
            exc_duration = n_exc * dt
        else:
            exc_b1, _ = design_rf_pulse('sinc', duration=1e-3,
                                       flip_angle=90, npoints=int(1e-3 / dt))
            n_exc = len(exc_b1)
            b1[:n_exc] = exc_b1
            exc_duration = 1e-3
        # Build a proper 180° refocusing pulse (independent of excitation shape)
        if self.custom_refocusing is not None:
            ref_b1, _ = self.custom_refocusing
            ref_pulse = np.asarray(ref_b1, dtype=complex)
        else:
            ref_pulse, _ = design_rf_pulse('sinc', duration=2e-3,
                                          flip_angle=180, npoints=int(2e-3 / dt))

        # Estimate excitation phase from non-zero samples; default to 0
        if np.any(np.abs(b1[:n_exc]) > 0):
            exc_phase = np.angle(np.mean(b1[:n_exc][np.abs(b1[:n_exc]) > 0]))
        else:
            exc_phase = 0.0

        # 180° refocusing pulses every TE, phase-shifted by +90° relative to excitation
        phase_shift = np.exp(1j * (exc_phase + np.pi / 2.0))
        ref_pulse = ref_pulse * phase_shift
        for echo_idx in range(self.echo_count):
            ref_time = (0.5 + echo_idx) * self.te
            ref_start = int(ref_time / dt)
            if ref_start + len(ref_pulse) < npoints:
                b1[ref_start:ref_start + len(ref_pulse)] = ref_pulse

        # Slice-select gradients (reuse SpinEcho logic)
        bw_hz = 4.0 / max(exc_duration, dt)
        gamma_hz_per_g = 4258.0
        thickness_cm = max(self.slice_thickness, 1e-3) * 100.0
        if self.slice_gradient_override is not None and self.slice_gradient_override > 0:
            gz_amp = self.slice_gradient_override
        else:
            gz_amp = bw_hz / (gamma_hz_per_g * thickness_cm)
        gradients[:max(n_exc, 1), 2] = gz_amp
        for echo_idx in range(self.echo_count):
            ref_time = (0.5 + echo_idx) * self.te
            ref_start = int(ref_time / dt)
            if ref_start + len(ref_pulse) < npoints:
                gradients[ref_start:ref_start + len(ref_pulse), 2] = gz_amp

        return b1, gradients, time


class GradientEcho(PulseSequence):
    """
    Gradient echo pulse sequence.
    """
    
    def __init__(self, te: float, tr: float, flip_angle: float = 30, custom_excitation=None,
                 slice_thickness: float = 0.005, slice_gradient_override: Optional[float] = None, rf_freq_offset: float = 0.0, **kwargs):
        """
        Initialize gradient echo sequence.

        Parameters
        ----------
        te : float
            Echo time in seconds
        tr : float
            Repetition time in seconds
        flip_angle : float
            Flip angle in degrees
        rf_freq_offset : float
            RF frequency offset in Hz (default 0)
        """
        super().__init__(slice_thickness=slice_thickness, **kwargs)
        self.te = te
        self.tr = tr
        self.flip_angle = flip_angle
        self.custom_excitation = custom_excitation
        self.slice_gradient_override = slice_gradient_override
        self.rf_freq_offset = rf_freq_offset
        
    def compile(self, dt: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compile gradient echo sequence."""
        npoints = int(self.tr / dt)
        
        # Initialize arrays
        b1 = np.zeros(npoints, dtype=complex)
        gradients = np.zeros((npoints, 3))
        time = np.arange(npoints) * dt
        
        # Excitation pulse
        if self.custom_excitation is not None:
            exc_b1, exc_time = self.custom_excitation
            exc_b1 = np.asarray(exc_b1, dtype=complex)
            n_exc = min(len(exc_b1), npoints)
            b1[:n_exc] = exc_b1[:n_exc]
            exc_duration = n_exc * dt
        else:
            exc_pulse, _ = design_rf_pulse('sinc', duration=1e-3,
                                          flip_angle=self.flip_angle,
                                          npoints=int(1e-3/dt),
                                          freq_offset=self.rf_freq_offset)
            b1[:len(exc_pulse)] = exc_pulse
            exc_duration = 1e-3
        
        # Slice selection gradient
        thickness_cm = max(self.slice_thickness, 1e-3) * 100.0
        bw_hz = 4.0 / max(exc_duration, dt)
        gamma_hz_per_g = 4258.0
        if self.slice_gradient_override is not None and self.slice_gradient_override > 0:
            gz_amp = self.slice_gradient_override
        else:
            gz_amp = bw_hz / (gamma_hz_per_g * thickness_cm)
        n_exc = np.count_nonzero(np.abs(b1) > 0)
        gradients[:max(n_exc, 1), 2] = gz_amp
        
        # Readout gradient (simplified)
        readout_start = int(self.te / dt) - int(0.5e-3 / dt)
        readout_duration = int(1e-3 / dt)
        if readout_start + readout_duration < npoints:
            gradients[readout_start:readout_start+readout_duration, 0] = 5e-3
        
        return b1, gradients, time


class SliceSelectRephase(PulseSequence):
    """
    Simple slice-select pulse followed by a rephasing gradient lobe.
    """

    def __init__(
        self,
        flip_angle: float = 90,
        pulse_duration: float = 3e-3,
        time_bw_product: float = 4.0,
        rephase_duration: float = 0.6e-3,
        slice_gradient_override: Optional[float] = None,
        custom_pulse: Optional[Tuple] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.flip_angle = flip_angle
        self.pulse_duration = pulse_duration
        self.time_bw_product = time_bw_product
        self.rephase_duration = rephase_duration
        self.slice_gradient_override = slice_gradient_override
        self.custom_pulse = custom_pulse

    def compile(self, dt: float = 5e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compile a slice-select + rephase waveform.

        The slice gradient area during the RF pulse is rewound with a
        negative lobe of half the area.
        """
        dt = max(dt, 1e-6)

        # Use custom pulse if provided
        if self.custom_pulse is not None:
            b1, pulse_time = self.custom_pulse
            b1 = np.asarray(b1, dtype=complex)
            n_rf = len(b1)
            # Use actual pulse duration from custom pulse
            pulse_duration = pulse_time[-1] - pulse_time[0] if len(pulse_time) > 1 else self.pulse_duration
        else:
            n_rf = max(8, int(np.ceil(self.pulse_duration / dt)))
            pulse_duration = self.pulse_duration
            # RF pulse
            b1, _ = design_rf_pulse(
                "sinc",
                duration=self.pulse_duration,
                flip_angle=self.flip_angle,
                time_bw_product=self.time_bw_product,
                npoints=n_rf,
            )
            b1 = np.asarray(b1, dtype=complex)

        n_rephase = max(4, int(np.ceil(self.rephase_duration / dt)))
        gap_pts = max(2, int(np.ceil(0.2e-3 / dt)))
        n_time = n_rf + gap_pts + n_rephase

        # Gradients (Gauss/cm)
        gradients = np.zeros((n_time, 3), dtype=float)
        bw_hz = self.time_bw_product / pulse_duration
        gamma_hz_per_g = 4258.0
        thickness_cm = max(self.slice_thickness, 1e-3) * 100.0
        if self.slice_gradient_override is not None and self.slice_gradient_override > 0:
            gz_gauss_per_cm = self.slice_gradient_override
        else:
            gz_gauss_per_cm = bw_hz / (gamma_hz_per_g * thickness_cm)
        gradients[:n_rf, 2] = gz_gauss_per_cm

        # Rephasing lobe with half the area of the excitation lobe
        area_exc = gz_gauss_per_cm * pulse_duration
        rephase_amp = -(0.5 * area_exc) / (n_rephase * dt)
        start_rephase = n_rf + gap_pts
        gradients[start_rephase:start_rephase + n_rephase, 2] = rephase_amp
        # Zero-pad B1 to match total time length
        b1_full = np.zeros(n_time, dtype=complex)
        b1_full[:n_rf] = b1

        time = np.arange(n_time) * dt
        return b1_full, gradients, time


class CustomPulse(PulseSequence):
    """
    Custom pulse sequence loaded from a file.

    Supports Bruker JCAMP-DX format (.exc) and other waveform files.
    """

    def __init__(
        self,
        pulse_source: Union[str, Path],
        gradients: Optional[np.ndarray] = None,
        slice_gradient_override: Optional[float] = None,
        scale_b1: float = 1.0,
        **kwargs,
    ):
        """
        Initialize custom pulse sequence.

        Parameters
        ----------
        pulse_source : str or Path
            Either a pulse name (e.g., 'bruker/13C_Ultimate_SPSP_Pulse_QuEMRT')
            or a file path to an RF pulse file
        gradients : ndarray, optional
            Custom gradient waveforms (ntime, 3). If None, no gradients applied.
        slice_gradient_override : float, optional
            Override slice gradient amplitude (Gauss/cm)
        scale_b1 : float, optional
            Scale factor for B1 amplitude (default: 1.0)
        **kwargs : optional
            Additional arguments passed to PulseSequence
        """
        super().__init__(**kwargs)

        self.pulse_source = pulse_source
        self.custom_gradients = gradients
        self.slice_gradient_override = slice_gradient_override
        self.scale_b1 = scale_b1
        self.metadata = None

        # Load the pulse
        self._load_pulse()

    def _load_pulse(self):
        """Load pulse from file or library."""
        pulse_source = str(self.pulse_source)

        # Try to load from library first (if it looks like a library name)
        if not Path(pulse_source).exists():
            try:
                self.b1, self.time, self.metadata = load_pulse(pulse_source)
                return
            except (ImportError, ValueError):
                pass

        # Try to load as a direct file path
        try:
            self.b1, self.time, self.metadata = load_pulse_from_file(pulse_source)
        except (ImportError, FileNotFoundError) as e:
            raise ValueError(
                f"Could not load pulse from '{pulse_source}'. "
                f"Ensure the file exists or the pulse name is in the library."
            ) from e

    def compile(self, dt: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compile the custom pulse sequence.

        Parameters
        ----------
        dt : float
            Time step for resampling (if needed)

        Returns
        -------
        b1 : ndarray
            Complex B1 field (Gauss)
        gradients : ndarray
            Gradient waveforms (ntime, 3) in Gauss/cm
        time : ndarray
            Time points in seconds
        """
        # Resample B1 if needed
        b1 = self.b1.copy()
        time = self.time.copy()

        # Check if resampling is needed
        if len(time) > 1:
            actual_dt = time[1] - time[0]
        else:
            actual_dt = dt

        if actual_dt != dt and dt > 0:
            # Resample to new dt
            new_npoints = int(np.ceil((time[-1] - time[0]) / dt))
            new_time = np.linspace(time[0], time[-1], new_npoints)
            # Simple linear interpolation
            b1 = np.interp(new_time, time, b1)
            time = new_time

        # Apply B1 scaling
        b1 = b1 * self.scale_b1

        # Handle gradients
        if self.custom_gradients is not None:
            gradients = np.asarray(self.custom_gradients, dtype=np.float64)
        else:
            # No gradients
            gradients = np.zeros((len(b1), 3), dtype=np.float64)

        # Ensure gradient shape matches B1 length
        if gradients.shape[0] != len(b1):
            if gradients.shape[0] == 1:
                # Broadcast single gradient to all points
                gradients = np.tile(gradients, (len(b1), 1))
            else:
                # Resample gradients to match B1 length
                old_time = np.linspace(0, gradients.shape[0] - 1, gradients.shape[0])
                new_time_idx = np.linspace(0, gradients.shape[0] - 1, len(b1))
                gradients_resampled = np.zeros((len(b1), 3))
                for i in range(3):
                    gradients_resampled[:, i] = np.interp(
                        new_time_idx, old_time, gradients[:, i]
                    )
                gradients = gradients_resampled

        return b1, gradients, time


class BlochSimulator:
    """
    High-level interface for Bloch equation simulations.
    """
    
    def __init__(self, use_parallel: bool = True, num_threads: int = 4, verbose: bool = False):
        """
        Initialize the Bloch simulator.
        
        Parameters
        ----------
        use_parallel : bool
            Use parallel processing
        num_threads : int
            Number of threads for parallel processing
        verbose : bool
            Print progress messages
        """
        self.use_parallel = use_parallel
        self.num_threads = num_threads
        self.verbose = verbose
        self.last_result = None
    
    def log_message(self, message: str):
        """Print a message if verbose mode is enabled."""
        if self.verbose:
            print(message)
        
    def simulate(self, 
                 sequence: Union[PulseSequence, Tuple],
                 tissue: TissueParameters,
                 positions: Optional[np.ndarray] = None,
                 frequencies: Optional[np.ndarray] = None,
                 initial_magnetization: Optional[np.ndarray] = None,
                 dt: float = 1e-6,
                 mode: int = 0) -> Dict:
        """
        Simulate MRI signal using Bloch equations.
        
        Parameters
        ----------
        sequence : PulseSequence or tuple
            Pulse sequence object or (b1, gradients, time) tuple
        tissue : TissueParameters
            Tissue parameters
        positions : ndarray, optional
            Spatial positions [x, y, z] in meters
        frequencies : ndarray, optional  
            Off-resonance frequencies in Hz
        initial_magnetization : ndarray, optional
            Initial magnetization state
        dt : float
            Time step for compilation
        mode : int
            Simulation mode (0: endpoint, 2: time-resolved)
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'mx', 'my', 'mz': Magnetization components
            - 'signal': Complex MRI signal
            - 'time': Time points
            - 'positions': Positions used
            - 'frequencies': Frequencies used
        """
        
        # Compile sequence if needed
        if isinstance(sequence, PulseSequence):
            b1, gradients, time = sequence.compile(dt)
        else:
            b1, gradients, time = sequence
        
        # Sanitize and standardize inputs to avoid buffer errors from Cython
        b1 = np.asarray(b1, dtype=np.complex128)
        b1 = np.squeeze(b1)
        if b1.ndim != 1:
            raise ValueError(f"B1 array must be 1D after squeezing, got shape {b1.shape}")
        # design_rf_pulse already returns Gauss; keep units unchanged
        b1_gauss = np.ascontiguousarray(b1)

        gradients = np.asarray(gradients, dtype=np.float64)
        if gradients.ndim == 1:
            gradients = gradients.reshape(-1, 1)
        if gradients.ndim != 2:
            raise ValueError(f"Gradients must be 2D, got {gradients.ndim}D with shape {gradients.shape}")
        if gradients.shape[1] < 3:
            gradients = np.pad(gradients, ((0, 0), (0, 3 - gradients.shape[1])), mode='constant')
        elif gradients.shape[1] > 3:
            gradients = gradients[:, :3]
        if gradients.shape[0] != b1_gauss.shape[0]:
            raise ValueError(f"Gradients length ({gradients.shape[0]}) must match B1 length ({b1_gauss.shape[0]})")
        # Gradients expected in Gauss/cm already
        gradients_gauss = np.ascontiguousarray(gradients)

        time = np.asarray(time, dtype=np.float64)
        time = np.squeeze(time)
        if time.ndim != 1:
            raise ValueError(f"Time array must be 1D after squeezing, got shape {time.shape}")
        if time.shape[0] != b1_gauss.shape[0]:
            raise ValueError(f"Time length ({time.shape[0]}) must match B1 length ({b1_gauss.shape[0]})")
        time = np.ascontiguousarray(time)

        # Default positions and frequencies
        if positions is None:
            positions = np.array([[0.0, 0.0, 0.0]])
        positions = np.asarray(positions, dtype=np.float64)
        positions = np.atleast_2d(positions)
        if positions.shape[1] < 3:
            positions = np.pad(positions, ((0, 0), (0, 3 - positions.shape[1])), mode='constant')
        elif positions.shape[1] > 3:
            positions = positions[:, :3]
        positions_cm = np.ascontiguousarray(positions * 100)  # m -> cm

        if frequencies is None:
            frequencies = np.array([0.0])
        frequencies = np.asarray(frequencies, dtype=np.float64)
        frequencies = np.ravel(frequencies)
        frequencies = np.ascontiguousarray(frequencies)

        # Prepare initial magnetization if provided (shape expected: 3 x (npos*nfreq))
        m_init = None
        if initial_magnetization is not None:
            init_arr = np.asarray(initial_magnetization, dtype=np.float64)
            nfnpos = positions.shape[0] * frequencies.shape[0]
            if init_arr.ndim == 0:
                vec = np.array([0.0, 0.0, float(init_arr)], dtype=np.float64)
                m_init = np.tile(vec[:, None], (1, nfnpos))
            elif init_arr.ndim == 1:
                if init_arr.size == 3:
                    vec = init_arr.reshape(3, 1)
                    m_init = np.tile(vec, (1, nfnpos))
                else:
                    raise ValueError("Initial magnetization must be scalar or length-3 vector.")
            elif init_arr.ndim == 2:
                # Accept (3, nfnpos) or (nfnpos, 3)
                if init_arr.shape == (3, nfnpos):
                    m_init = init_arr
                elif init_arr.shape == (nfnpos, 3):
                    m_init = init_arr.T
                else:
                    raise ValueError(f"Initial magnetization shape must be (3, npos*nfreq); got {init_arr.shape}.")
            else:
                raise ValueError("Initial magnetization must be scalar, length-3 vector, or (3, npos*nfreq) array.")
            m_init = np.ascontiguousarray(m_init, dtype=np.float64)

        # Guard against unreasonably large allocations in time-resolved mode
        ntout = len(time) if (mode & 2) else 1
        total_points = ntout * positions.shape[0] * frequencies.shape[0]
        if total_points > 5e7:
            raise ValueError(
                f"Requested simulation is too large ({total_points:.1e} samples). "
                "Increase the time step, reduce positions/frequencies, or use Endpoint mode."
            )
            
        # Time intervals
        if len(time) > 1:
            dt_array = np.diff(time)
            dt_array = np.append(dt_array, dt_array[-1])
        else:
            dt_array = np.array([dt])
            
        # Run simulation
        # The OpenMP path can be unstable on some macOS/Python builds; keep the
        # threshold high to avoid crashes for small/medium workloads.
        parallel_threshold = 256
        if self.use_parallel and len(positions) * len(frequencies) >= parallel_threshold:
            mx, my, mz = simulate_bloch_parallel(
                b1_gauss, gradients_gauss, dt_array,
                tissue.t1, tissue.t2, frequencies,
                positions_cm, initial_magnetization,
                mode, self.num_threads
            )
        else:
            mx, my, mz = simulate_bloch(
                b1_gauss, gradients_gauss, dt_array,
                tissue.t1, tissue.t2, frequencies,
                positions_cm, initial_magnetization,
                mode
            )
            
        # Calculate complex signal
        signal = calculate_signal(mx, my, mz)
        
        # Store result
        self.last_result = {
            'mx': mx,
            'my': my,
            'mz': mz,
            'signal': signal,
            'time': time,
            'positions': positions,
            'frequencies': frequencies,
            'tissue': tissue
        }
        
        return self.last_result
    
    def simulate_phantom(self, phantom, sequence: Union[PulseSequence, Tuple],
                        dt: float = 1e-5,
                        mode: int = 0,
                        additional_frequencies: Optional[np.ndarray] = None,
                        use_grouped: bool = True) -> Dict:
        """
        Simulate Bloch equations for a heterogeneous phantom.
        
        This method simulates MRI physics for phantoms with spatially-varying
        tissue properties (T1, T2, proton density, frequency offset). Each voxel
        can have different parameters, enabling realistic imaging simulation.
        
        Parameters
        ----------
        phantom : Phantom
            Phantom object with tissue property maps (T1, T2, PD, df).
            See phantom.py for Phantom class and PhantomFactory.
        sequence : PulseSequence or tuple
            Either a PulseSequence object or tuple of (b1, gradients, time)
        dt : float
            Time step for sequence compilation (if using PulseSequence)
        mode : int
            Simulation mode:
            - 0: Endpoint only (faster, returns final magnetization)
            - 2: Time-resolved (returns magnetization at all time points)
        additional_frequencies : ndarray, optional
            Extra frequency offsets to simulate (Hz). These are added to
            each voxel's df_map value. Useful for multi-frequency/spectroscopic
            imaging.
        use_grouped : bool
            If True and phantom has discrete tissue labels, use optimized
            grouped simulation (faster for segmented phantoms).
        
        Returns
        -------
        dict
            Simulation results containing:
            - 'mx', 'my', 'mz': Magnetization components
              Shape: (*phantom.shape,) for mode=0, or (ntime, *phantom.shape) for mode=2
            - 'signal': Complex transverse magnetization (mx + 1j*my)
            - 'time': Time array from sequence
            - 'phantom': The input phantom object
            - 'pd_weighted_signal': Signal weighted by proton density
        
        Examples
        --------
        >>> from phantom import PhantomFactory
        >>> # Create Shepp-Logan phantom
        >>> phantom = PhantomFactory.shepp_logan_2d(64, 0.24, 3.0)
        >>> # Create excitation pulse
        >>> seq = PulseSequence()
        >>> seq.add_rf_pulse(flip_angle=90, duration=1e-3)
        >>> # Simulate
        >>> result = simulator.simulate_phantom(phantom, seq, mode=0)
        >>> # Result shape matches phantom
        >>> print(result['mx'].shape)  # (64, 64)
        """
        # Import Phantom class (avoid circular import)
        try:
            from phantom import Phantom
        except ImportError:
            raise ImportError("Phantom module not found. Ensure phantom.py is available.")
        
        if not isinstance(phantom, Phantom):
            raise TypeError(f"Expected Phantom object, got {type(phantom)}")
        
        self.log_message(f"Simulating phantom: {phantom}")
        self.log_message(f"Active voxels: {phantom.n_active} / {phantom.nvoxels}")
        
        # Compile sequence
        if isinstance(sequence, PulseSequence):
            b1, gradients, time = sequence.compile(dt)
        else:
            b1, gradients, time = sequence
        
        # Prepare arrays (same sanitization as simulate())
        b1 = np.asarray(b1, dtype=np.complex128)
        b1 = np.squeeze(b1)
        if b1.ndim != 1:
            raise ValueError(f"B1 array must be 1D, got shape {b1.shape}")
        b1_gauss = np.ascontiguousarray(b1)
        
        gradients = np.asarray(gradients, dtype=np.float64)
        if gradients.ndim == 1:
            gradients = gradients.reshape(-1, 1)
        if gradients.shape[1] < 3:
            gradients = np.pad(gradients, ((0, 0), (0, 3 - gradients.shape[1])), mode='constant')
        elif gradients.shape[1] > 3:
            gradients = gradients[:, :3]
        gradients_gauss = np.ascontiguousarray(gradients)
        
        time = np.asarray(time, dtype=np.float64).ravel()
        time = np.ascontiguousarray(time)
        
        # Time intervals
        if len(time) > 1:
            dt_array = np.diff(time)
            dt_array = np.append(dt_array, dt_array[-1])
        else:
            dt_array = np.array([dt])
        dt_array = np.ascontiguousarray(dt_array, dtype=np.float64)
        
        # Get phantom properties (active voxels only for efficiency)
        props = phantom.get_active_properties()
        n_active = len(props['t1'])
        
        if n_active == 0:
            self.log_message("Warning: No active voxels in phantom (all masked)")
            # Return zeros
            if mode & 2:
                shape = (len(time),) + phantom.shape
            else:
                shape = phantom.shape
            zeros = np.zeros(shape, dtype=np.float64)
            return {
                'mx': zeros,
                'my': zeros,
                'mz': zeros,
                'signal': np.zeros(shape, dtype=np.complex128),
                'time': time,
                'phantom': phantom,
                'pd_weighted_signal': np.zeros(shape, dtype=np.complex128),
            }
        
        # Convert positions from meters to cm (Bloch core uses Gauss/cm)
        positions_cm = props['positions'] * 100  # m -> cm
        positions_cm = np.ascontiguousarray(positions_cm, dtype=np.float64)
        
        # Frequency offsets
        df_array = np.ascontiguousarray(props['df'], dtype=np.float64)
        
        # Initial magnetization
        m_init = np.ascontiguousarray(props['m0'], dtype=np.float64)
        
        # Check memory requirements
        ntout = len(time) if (mode & 2) else 1
        total_samples = ntout * n_active
        if total_samples > 1e8:
            raise ValueError(
                f"Phantom simulation too large ({total_samples:.1e} samples for {n_active} active voxels). "
                f"Use smaller phantom, reduce time points, or use endpoint mode (mode=0)."
            )
        
        self.log_message(f"Simulation size: {n_active} voxels × {ntout} time points = {total_samples:.1e} samples")
        
        # Import wrapper function
        try:
            from .blochsimulator_cy import simulate_phantom as simulate_phantom_core
        except ImportError:
            raise ImportError("blochsimulator_cy not compiled. Run: python setup.py build_ext --inplace")
        
        # Run simulation
        t1_array = np.ascontiguousarray(props['t1'], dtype=np.float64)
        t2_array = np.ascontiguousarray(props['t2'], dtype=np.float64)
        
        self.log_message("Running heterogeneous Bloch simulation...")
        mx, my, mz = simulate_phantom_core(
            b1_gauss, gradients_gauss, dt_array,
            t1_array, t2_array, df_array,
            positions_cm, m_init,
            mode, self.num_threads
        )
        
        # Reconstruct full phantom shape from active voxels
        indices = props['indices']
        
        if mode & 2:
            # Time-resolved: (ntime, n_active) -> (ntime, *phantom.shape)
            mx_full = phantom.reconstruct_from_active(mx, indices, has_time=True, fill_value=0.0)
            my_full = phantom.reconstruct_from_active(my, indices, has_time=True, fill_value=0.0)
            mz_full = phantom.reconstruct_from_active(mz, indices, has_time=True, fill_value=0.0)
        else:
            # Endpoint: (n_active,) -> (*phantom.shape,)
            mx_full = phantom.reconstruct_from_active(mx, indices, has_time=False, fill_value=0.0)
            my_full = phantom.reconstruct_from_active(my, indices, has_time=False, fill_value=0.0)
            mz_full = phantom.reconstruct_from_active(mz, indices, has_time=False, fill_value=0.0)
        
        # Complex signal per voxel (image-space magnetization)
        signal_per_voxel = mx_full + 1j * my_full
        
        # Apply proton density weighting
        pd_map = phantom.pd_map
        if mode & 2:
            # Broadcast pd_map to (ntime, *shape)
            pd_weighted = signal_per_voxel * pd_map[np.newaxis, ...]
        else:
            pd_weighted = signal_per_voxel * pd_map
        
        # Calculate RECEIVED SIGNAL (sum over all voxels)
        # This is what an RF coil would measure - the coherent sum of all spins
        # S(t) = Σ [Mxy(r,t) * PD(r)] for all positions r
        if mode & 2:
            # Time-resolved: sum over spatial dimensions, keep time
            # pd_weighted shape: (ntime, *spatial_shape)
            spatial_axes = tuple(range(1, pd_weighted.ndim))
            received_signal = np.sum(pd_weighted, axis=spatial_axes)
            self.log_message(f"Received signal shape: {received_signal.shape} (sum over {pd_weighted.shape[1:]})")
        else:
            # Endpoint: sum over all spatial dimensions
            received_signal = np.sum(pd_weighted)
            self.log_message(f"Received signal (endpoint): {received_signal}")
        
        # Store result
        self.last_result = {
            'mx': mx_full,
            'my': my_full,
            'mz': mz_full,
            'signal': signal_per_voxel,           # Per-voxel signal (for imaging)
            'time': time,
            'phantom': phantom,
            'pd_weighted_signal': pd_weighted,    # Per-voxel signal * PD
            'received_signal': received_signal,   # Total signal (what coil measures)
        }
        
        self.log_message(f"Simulation complete. Output shape: {mx_full.shape}")
        
        return self.last_result
    
    def plot_magnetization(self, component: str = 'all', 
                          position_idx: int = 0, 
                          freq_idx: int = 0):
        """
        Plot magnetization evolution.
        
        Parameters
        ----------
        component : str
            'mx', 'my', 'mz', 'magnitude', or 'all'
        position_idx : int
            Position index to plot
        freq_idx : int
            Frequency index to plot
        """
        if self.last_result is None:
            raise ValueError("No simulation results available")
            
        result = self.last_result
        time = result['time']
        
        if len(result['mx'].shape) == 2:
            # Single time point
            print("Single time point result - no time evolution to plot")
            return
            
        # Extract data for specific position and frequency
        mx = result['mx'][:, position_idx, freq_idx]
        my = result['my'][:, position_idx, freq_idx]
        mz = result['mz'][:, position_idx, freq_idx]
        self.log_message(f"result = {result}")  # Debugging line
        
        import matplotlib.pyplot as plt

        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Mx
        axes[0, 0].plot(time * 1000, mx)
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Mx')
        axes[0, 0].grid(True)
        axes[0, 0].set_title('Transverse Magnetization (x)')
        
        # My
        axes[0, 1].plot(time * 1000, my)
        axes[0, 1].set_xlabel('Time (ms)')
        axes[0, 1].set_ylabel('My')
        axes[0, 1].grid(True)
        axes[0, 1].set_title('Transverse Magnetization (y)')
        
        # Mz
        axes[1, 0].plot(time * 1000, mz)
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Mz')
        axes[1, 0].grid(True)
        axes[1, 0].set_title('Longitudinal Magnetization')
        
        # Magnitude
        magnitude = np.sqrt(mx**2 + my**2)
        axes[1, 1].plot(time * 1000, magnitude)
        axes[1, 1].set_xlabel('Time (ms)')
        axes[1, 1].set_ylabel('|Mxy|')
        axes[1, 1].grid(True)
        axes[1, 1].set_title('Transverse Magnitude')
        
        plt.suptitle(f'Magnetization Evolution\n{result["tissue"].name}')
        plt.tight_layout()
        plt.show()
        
    def save_results(self, filename: str, sequence_params: Optional[Dict] = None,
                     simulation_params: Optional[Dict] = None):
        """
        Save simulation results to HDF5 file with complete parameters.

        Parameters
        ----------
        filename : str
            Output HDF5 filename
        sequence_params : dict, optional
            Pulse sequence parameters (TE, TR, flip angle, etc.)
        simulation_params : dict, optional
            Simulation settings (mode, dt, parallel settings, etc.)
        """
        if self.last_result is None:
            raise ValueError("No simulation results available")

        with h5py.File(filename, 'w') as f:
            # Save magnetization data
            f.create_dataset('mx', data=self.last_result['mx'])
            f.create_dataset('my', data=self.last_result['my'])
            f.create_dataset('mz', data=self.last_result['mz'])
            f.create_dataset('signal', data=self.last_result['signal'])

            # Save parameters
            f.create_dataset('time', data=self.last_result['time'])
            f.create_dataset('positions', data=self.last_result['positions'])
            f.create_dataset('frequencies', data=self.last_result['frequencies'])

            # Save tissue parameters
            tissue_group = f.create_group('tissue')
            tissue = self.last_result['tissue']
            tissue_group.attrs['name'] = tissue.name
            tissue_group.attrs['t1'] = tissue.t1
            tissue_group.attrs['t2'] = tissue.t2
            tissue_group.attrs['density'] = tissue.density
            if tissue.t2_star is not None:
                tissue_group.attrs['t2_star'] = tissue.t2_star

            # Save pulse sequence parameters if provided
            if sequence_params is not None:
                seq_group = f.create_group('sequence_parameters')
                for key, value in sequence_params.items():
                    if value is not None:
                        if isinstance(value, (np.ndarray, list, tuple)):
                            seq_group.create_dataset(key, data=value)
                        else:
                            seq_group.attrs[key] = value

            # Save simulation parameters if provided
            if simulation_params is not None:
                sim_group = f.create_group('simulation_parameters')
                for key, value in simulation_params.items():
                    if value is not None:
                        if isinstance(value, (np.ndarray, list, tuple)):
                            sim_group.create_dataset(key, data=value)
                        else:
                            sim_group.attrs[key] = value

            # Add metadata
            f.attrs['export_timestamp'] = str(np.datetime64('now'))
            f.attrs['simulator_version'] = __version__

    def save_parameters_json(self, filename: str, sequence_params: Optional[Dict] = None,
                             simulation_params: Optional[Dict] = None,
                             include_waveforms: bool = False):
        """
        Save simulation parameters to JSON file.

        Parameters
        ----------
        filename : str
            Output JSON filename
        sequence_params : dict, optional
            Pulse sequence parameters
        simulation_params : dict, optional
            Simulation settings
        include_waveforms : bool, optional
            If True, include RF pulse and gradient waveforms (can be large)
        """
        if self.last_result is None:
            raise ValueError("No simulation results available")

        import json

        params_dict = {
            'metadata': {
                'export_timestamp': str(np.datetime64('now')),
                'simulator_version': __version__
            },
            'tissue_parameters': {
                'name': self.last_result['tissue'].name,
                't1': float(self.last_result['tissue'].t1),
                't2': float(self.last_result['tissue'].t2),
                'density': float(self.last_result['tissue'].density),
                't2_star': float(self.last_result['tissue'].t2_star) if self.last_result['tissue'].t2_star else None
            },
            'positions': self.last_result['positions'].tolist(),
            'frequencies': self.last_result['frequencies'].tolist(),
            'time_points': int(len(self.last_result['time'])),
            'duration': float(self.last_result['time'][-1]) if len(self.last_result['time']) > 0 else 0.0
        }

        # Add sequence parameters
        if sequence_params is not None:
            params_dict['sequence_parameters'] = {}
            for key, value in sequence_params.items():
                if value is not None:
                    if isinstance(value, np.ndarray):
                        if include_waveforms:
                            params_dict['sequence_parameters'][key] = value.tolist()
                        else:
                            params_dict['sequence_parameters'][key] = f"<array shape={value.shape}>"
                    elif isinstance(value, (list, tuple)):
                        params_dict['sequence_parameters'][key] = list(value)
                    else:
                        params_dict['sequence_parameters'][key] = value

        # Add simulation parameters
        if simulation_params is not None:
            params_dict['simulation_parameters'] = {}
            for key, value in simulation_params.items():
                if value is not None:
                    if isinstance(value, np.ndarray):
                        params_dict['simulation_parameters'][key] = value.tolist()
                    elif isinstance(value, (list, tuple)):
                        params_dict['simulation_parameters'][key] = list(value)
                    else:
                        params_dict['simulation_parameters'][key] = value

        # Write to file
        with open(filename, 'w') as f:
            json.dump(params_dict, f, indent=2)

    def load_results(self, filename: str):
        """Load simulation results from HDF5 file."""
        with h5py.File(filename, 'r') as f:
            self.last_result = {
                'mx': f['mx'][...],
                'my': f['my'][...],
                'mz': f['mz'][...],
                'signal': f['signal'][...],
                'time': f['time'][...],
                'positions': f['positions'][...],
                'frequencies': f['frequencies'][...],
                'tissue': TissueParameters(
                    name=f['tissue'].attrs['name'],
                    t1=f['tissue'].attrs['t1'],
                    t2=f['tissue'].attrs['t2'],
                    density=f['tissue'].attrs['density']
                )
            }


# Example usage functions
def example_fid():
    """Example: Free Induction Decay simulation."""
    # Create simulator
    sim = BlochSimulator()
    
    # Define tissue
    tissue = TissueParameters.gray_matter(3.0)
    
    # Simple FID sequence
    ntime = 1000
    dt = 1e-5  # 10 microseconds
    time = np.arange(ntime) * dt
    
    # 90-degree pulse
    b1 = np.zeros(ntime, dtype=complex)
    b1[0] = 0.01  # Short hard pulse
    
    # No gradients
    gradients = np.zeros((ntime, 3))
    
    # Single position, multiple frequencies
    positions = np.array([[0, 0, 0]])
    frequencies = np.linspace(-100, 100, 21)  # -100 to 100 Hz
    
    # Simulate
    result = sim.simulate(
        (b1, gradients, time),
        tissue,
        positions=positions,
        frequencies=frequencies,
        mode=2  # Time-resolved
    )
    
    return result


def example_spin_echo():
    """Example: Spin echo simulation."""
    sim = BlochSimulator()
    
    # Create spin echo sequence
    sequence = SpinEcho(te=20e-3, tr=500e-3)
    
    # Define tissue
    tissue = TissueParameters.white_matter(3.0)
    
    # Simulate
    result = sim.simulate(sequence, tissue, mode=2)
    
    return result


if __name__ == "__main__":
    print("Bloch Simulator Python API")
    print("==========================")
    print("This module provides high-level functions for MRI simulation.")
    print("\nExample usage:")
    print("  from blochsimulator import BlochSimulator, TissueParameters")
    print("  sim = BlochSimulator()")
    print("  tissue = TissueParameters.gray_matter(3.0)")
    print("  # ... define sequence ...")
    print("  result = sim.simulate(sequence, tissue)")
