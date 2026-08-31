"""
blochsimulator.py - High-level Python API for Bloch equation simulations

This module provides user-friendly classes and functions for MRI pulse sequence
simulation using the Bloch equations.

Author: Your Name
Date: 2024
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
import h5py
import xarray as xr
import json
import os
from pathlib import Path
from . import __version__
from .memory import (
    MemoryPolicy,
    enforce_memory_budget,
    enforce_sequence_memory,
    resolve_memory_budget,
)

RF_PULSE_TYPE_OPTIONS = (
    "Rectangle",
    "Sinc",
    "SLR",
    "Gaussian",
    "Hermite",
    "Adiabatic Half Passage",
    "Adiabatic Full Passage",
    "BIR-4",
    "Custom",
)

# Import the Cython extension (will be available after building)
HAS_CYTHON = False
try:
    from .blochsimulator_cy import (
        simulate_bloch,
        simulate_bloch_parallel,
        calculate_signal,
    )

    HAS_CYTHON = True
except ImportError:
    print(
        "Warning: Cython extension not built. Run 'python setup.py build_ext --inplace' first."
    )

    # Define dummy functions for testing
    def simulate_bloch(*args, **kwargs):
        raise NotImplementedError("Build the Cython extension first")

    def simulate_bloch_parallel(*args, **kwargs):
        raise NotImplementedError("Build the Cython extension first")

    def calculate_signal(mx, my, mz, receiver_phase=0.0):
        phase_factor = np.exp(-1j * receiver_phase)
        return (mx + 1j * my) * phase_factor


def resolve_num_threads(num_threads: Optional[int] = None) -> int:
    """Resolve an explicit or automatic native worker count."""
    if num_threads is None:
        return max(1, int(os.cpu_count() or 1))
    if isinstance(num_threads, bool):
        raise ValueError("num_threads must be a positive integer, zero, or None")
    if num_threads == 0:
        return max(1, int(os.cpu_count() or 1))
    if int(num_threads) != num_threads:
        raise ValueError("num_threads must be a positive integer, zero, or None")
    if num_threads < 0:
        raise ValueError("num_threads must be a positive integer, zero, or None")
    return int(num_threads)


def design_rf_pulse(
    pulse_type="rect",
    duration=1e-3,
    flip_angle=90,
    time_bw_product=4,
    npoints=100,
    freq_offset=0.0,
    slr_sharpness=1.0,
):
    """
    Pure-Python fallback for RF design so imports work even without the extension.

    Parameters
    ----------
    pulse_type : str
        Type of pulse ('rect', 'sinc', 'slr', 'gaussian', 'hermite',
        'adiabatic_half', 'adiabatic_full', 'bir4')
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
    slr_sharpness : float
        SLR transition sharpness. Higher values generate progressively more
        temporal lobes through the shared RF envelope designer.

    Returns
    -------
    b1 : complex ndarray
        Complex B1 field in Gauss
    time : ndarray
        Time points in seconds
    """
    npoints = int(npoints)
    if npoints <= 0:
        raise ValueError("npoints must be positive")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    dt = duration / npoints
    # RF amplitudes represent raster intervals, so report their sample centres.
    # This also keeps every analytic pulse exactly symmetric for even and odd
    # point counts instead of giving one Sinc edge an unmatched half sample.
    time = (np.arange(npoints, dtype=float) + 0.5) * dt
    gamma = 4258.0  # Hz/Gauss for protons
    flip_rad = np.deg2rad(flip_angle)
    target_area = flip_rad / (gamma * 2 * np.pi)  # integral of B1 over time (Gauss * s)
    shared_type = {
        "rect": "block",
        "rectangle": "block",
        "block": "block",
        "sinc": "sinc",
        "slr": "slr",
        "gaussian": "gaussian",
    }.get(str(pulse_type).lower())
    if shared_type is not None:
        # Free Mode and generated Sequence Mode pulses use the same baseband
        # envelope factory. Free Mode may still apply its selected display/design
        # apodization afterwards; the underlying analytic sampling is identical.
        from .sequence.rf_pulses import design_rf_envelope

        envelope, _, _, _ = design_rf_envelope(
            pulse_type=shared_type,
            duration_s=duration,
            raster_s=dt,
            time_bandwidth_product=time_bw_product,
            apodization=0.0,
            slr_sharpness=slr_sharpness,
        )
        area = np.sum(envelope) * dt
        if abs(area) < 1e-15:
            raise ValueError(
                f"{shared_type} pulse integral is too small for flip scaling"
            )
        b1 = envelope * (target_area / abs(area))
    elif pulse_type == "hermite":
        t_centered = time - duration / 2
        sigma = duration / max(2.0 * time_bw_product, 1e-9)
        normalized_time = t_centered / sigma
        envelope = (1.0 - 0.5 * normalized_time**2) * np.exp(-0.5 * normalized_time**2)
        area = np.trapezoid(envelope, time)
        if abs(area) < 1e-12:
            raise ValueError("Hermite pulse integral is too small for flip scaling")
        b1 = envelope * (target_area / area)
    elif pulse_type == "adiabatic_half":
        # Adiabatic Half Passage (AHP): 90° excitation pulse
        # Sweeps from off-resonance to on-resonance.
        # Magnetization tracks effective field from Z to Transverse plane.

        # Time variable for AHP (Half of a full passage)
        # Map time [0, duration] to [-duration, 0] relative to crossing
        t_arg = (time - duration) / duration

        beta = time_bw_product  # Modulation parameter (typically 4-8)

        # HS amplitude modulation: A(t) = A0 * sech(beta * t)
        # Grows from ~0 to 1.0
        amplitude = 1.0 / np.cosh(beta * t_arg)

        # Frequency modulation using tanh
        # Delta_omega(t) = -omega_max * tanh(beta * t)
        # Sweeps from +Omega_max (at t=0) to 0 (at t=duration)
        bandwidth_hz = time_bw_product / duration
        omega_max = np.pi * bandwidth_hz

        freq_modulation = -omega_max * np.tanh(beta * t_arg)

        # Integrate to get phase: phi(t) = integral(omega(t) dt)
        dt = duration / npoints
        instantaneous_phase = np.cumsum(freq_modulation * dt)

        # Complex B1: A(t) * exp(i*phi(t))
        b1_complex = amplitude * np.exp(1j * instantaneous_phase)

        # For adiabatic pulses, scale by flip_angle to control B1_max directly
        # AHP typically achieves 90° when adiabaticity κ ≈ 5-10
        # User adjusts flip_angle to control the RF amplitude (B1_max in Gauss)
        target_flip_rad = np.deg2rad(flip_angle)
        b1_max_gauss = target_flip_rad / (gamma * 2 * np.pi * duration)
        b1 = b1_complex * b1_max_gauss

    elif pulse_type == "adiabatic_full":
        # Adiabatic Full Passage (AFP): 180° inversion pulse
        # Uses hyperbolic secant amplitude + tanh frequency modulation
        # Magnetization follows effective field through full 180° inversion
        #
        # For adiabatic pulses, the flip angle is determined by the adiabaticity
        # parameter κ = γ·B1_max·T / β, NOT by the pulse area.
        # The flip_angle parameter controls B1_max to achieve the desired rotation.
        t_centered = time - duration / 2
        beta = time_bw_product  # Typically 4-8 for good adiabatic condition

        # HS amplitude modulation: A(t) = A0 * sech(beta * t / T)
        # Normalized amplitude envelope (max = 1.0)
        amplitude = 1.0 / np.cosh(beta * t_centered / (duration / 2))

        # Frequency modulation using tanh (sweeps through full resonance)
        bandwidth_hz = time_bw_product / duration
        omega_max = np.pi * bandwidth_hz

        # Full sweep: omega goes from +omega_max to -omega_max
        freq_modulation = -omega_max * np.tanh(beta * t_centered / (duration / 2))

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

    elif pulse_type == "bir4":
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
        t1 = time[:n_seg] - time[n_seg // 2]
        t2 = time[n_seg : 2 * n_seg] - time[3 * n_seg // 2]
        t3 = time[2 * n_seg : 3 * n_seg] - time[5 * n_seg // 2]
        t4 = time[3 * n_seg :] - time[7 * n_seg // 2]

        bandwidth_hz = time_bw_product / t_seg
        omega_max = np.pi * bandwidth_hz

        # Segment 1: AHP (90°)
        amp1 = 1.0 / np.cosh(beta * t1 / (t_seg / 2))
        freq1 = -omega_max * np.tanh(beta * t1 / (t_seg / 2))
        phase1 = np.cumsum(freq1 * (t_seg / n_seg))
        b1_seg1 = amp1 * np.exp(1j * phase1)

        # Segment 2: 180° phase shift + reverse AHP
        amp2 = 1.0 / np.cosh(beta * t2 / (t_seg / 2))
        freq2 = omega_max * np.tanh(beta * t2 / (t_seg / 2))  # Reversed
        phase2 = np.cumsum(freq2 * (t_seg / n_seg)) + phase1[-1]
        b1_seg2 = amp2 * np.exp(1j * (phase2 + np.pi))  # 180° phase shift

        # Segment 3: Inverse AHP
        amp3 = 1.0 / np.cosh(beta * t3 / (t_seg / 2))
        freq3 = omega_max * np.tanh(beta * t3 / (t_seg / 2))
        phase3 = np.cumsum(freq3 * (t_seg / n_seg)) + phase2[-1]
        b1_seg3 = amp3 * np.exp(1j * phase3)

        # Segment 4: 180° phase shift + AHP
        amp4 = 1.0 / np.cosh(beta * t4 / (t_seg / 2))
        freq4 = -omega_max * np.tanh(beta * t4 / (t_seg / 2))
        phase4 = np.cumsum(freq4 * (t_seg / n_seg)) + phase3[-1]
        b1_seg4 = amp4 * np.exp(1j * (phase4 + np.pi))  # 180° phase shift

        # Concatenate segments
        b1_complex = np.concatenate([b1_seg1, b1_seg2, b1_seg3, b1_seg4])

        # Pad if needed due to rounding
        if len(b1_complex) < npoints:
            b1_complex = np.pad(b1_complex, (0, npoints - len(b1_complex)), mode="edge")
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


def apply_rf_carrier(
    b1: np.ndarray, time: np.ndarray, frequency_offset_hz: float
) -> np.ndarray:
    """Apply an RF carrier using absolute sequence time.

    ``b1`` is treated as a complex baseband waveform.  Using the absolute
    sequence time here (rather than pulse-local time) preserves carrier phase
    across separated pulses and phase-cycled pulse trains.
    """
    b1_arr = np.asarray(b1, dtype=complex)
    time_arr = np.asarray(time, dtype=float)
    if b1_arr.shape != time_arr.shape:
        raise ValueError(
            f"B1 and time must have identical shapes, got {b1_arr.shape} and {time_arr.shape}."
        )
    if frequency_offset_hz == 0.0:
        return b1_arr.copy()
    return b1_arr * np.exp(2j * np.pi * float(frequency_offset_hz) * time_arr)


# Import pulse loader (optional - gracefully handle if module not available)
try:
    from .pulse_loader import load_pulse, load_pulse_from_file, get_pulse_library
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

    def __init__(
        self,
        fov: float = 0.24,
        matrix_size: int = 256,
        slice_thickness: float = 0.005,
        **kwargs,
    ):
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

    def add_rf_pulse(self, b1: np.ndarray, time: np.ndarray, phase: float = 0.0):
        """Add an RF pulse to the sequence."""
        self.rf_pulses.append({"b1": b1 * np.exp(1j * phase), "time": time})

    def add_gradient(self, axis: str, amplitude: float, duration: float, time: float):
        """Add a gradient to the sequence."""
        self.gradients.append(
            {"axis": axis, "amplitude": amplitude, "duration": duration, "time": time}
        )

    def compile(self, dt: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def __init__(
        self,
        te: float,
        tr: float,
        custom_excitation=None,
        custom_refocusing=None,
        slice_thickness: float = 0.005,
        slice_gradient_override: Optional[float] = None,
        echo_count: int = 1,
        rf_freq_offset: float = 0.0,
        **kwargs,
    ):
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

    def compile(self, dt: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compile spin echo sequence."""

        def _resample_pulse(pulse_data, target_dt, duration):
            """Resample pulse to match simulation dt."""
            b1_in, time_in = pulse_data
            b1_in = np.asarray(b1_in, dtype=complex)
            time_in = np.asarray(time_in, dtype=float)

            if len(time_in) < 2:
                return b1_in

            # Target time points
            num_points = int(np.round(duration / target_dt))
            if num_points <= 0:
                return np.array([], dtype=complex)

            # Original time (normalized to start at 0)
            t_orig = time_in - time_in[0]
            # Target time
            t_new = np.linspace(0, t_orig[-1], num_points)

            # Interpolate complex
            b1_real = np.interp(t_new, t_orig, np.real(b1_in))
            b1_imag = np.interp(t_new, t_orig, np.imag(b1_in))
            return b1_real + 1j * b1_imag

        # Determine pulse and readout durations first to validate TE
        if self.custom_excitation is not None:
            exc_b1_in, exc_time_in = self.custom_excitation
            # Use actual length from time array
            if len(exc_time_in) > 1:
                exc_duration = exc_time_in[-1] - exc_time_in[0]
            else:
                exc_duration = len(exc_b1_in) * dt
        else:
            exc_duration = 1e-3

        if self.custom_refocusing is not None:
            ref_b1_in, ref_time_in = self.custom_refocusing
            if len(ref_time_in) > 1:
                ref_duration = ref_time_in[-1] - ref_time_in[0]
            else:
                ref_duration = len(ref_b1_in) * dt
        else:
            ref_duration = 2e-3

        # Validate TE
        # Center-to-center spacing is TE/2.
        # This requires TE/2 >= (exc_duration/2 + ref_duration/2)
        # So TE >= exc_duration + ref_duration
        # We add a small buffer for safety
        min_te = (exc_duration + ref_duration) * 1.01
        if self.te < min_te:
            raise ValueError(
                f"TE ({self.te*1000:.2f} ms) is too short for selected pulses. "
                f"Minimum TE ≈ {min_te*1000:.2f} ms (Exc: {exc_duration*1000:.1f}ms, Ref: {ref_duration*1000:.1f}ms)"
            )

        # Ensure timeline covers all requested echoes (echo spacing = TE)
        min_duration = (
            exc_duration / 2.0 + (self.echo_count + 0.5) * self.te + ref_duration
        )  # include buffer for last echo
        total_duration = max(self.tr, min_duration)
        npoints = int(np.ceil(total_duration / dt))

        # Initialize arrays
        enforce_sequence_memory(npoints)
        b1 = np.zeros(npoints, dtype=complex)
        gradients = np.zeros((npoints, 3))
        time = np.arange(npoints) * dt

        # 90-degree excitation pulse
        if self.custom_excitation is not None:
            exc_pulse = _resample_pulse(self.custom_excitation, dt, exc_duration)
            n_exc = min(len(exc_pulse), npoints)
            b1[:n_exc] = exc_pulse[:n_exc]
        else:
            exc_pulse, _ = design_rf_pulse(
                "sinc",
                duration=exc_duration,
                flip_angle=90,
                npoints=int(exc_duration / dt),
                freq_offset=0.0,
            )
            n_exc = len(exc_pulse)
            b1[:n_exc] = exc_pulse

        # Refocusing pulse
        if self.custom_refocusing is not None:
            ref_pulse = _resample_pulse(self.custom_refocusing, dt, ref_duration)
        else:
            # Default refocusing: classic 180° sinc
            ref_pulse, _ = design_rf_pulse(
                "sinc",
                duration=ref_duration,
                flip_angle=180,
                npoints=int(ref_duration / dt),
                freq_offset=0.0,
            )

        for echo_idx in range(self.echo_count):
            # Center of refocusing pulse at exc_duration/2 + (0.5 + idx) * TE
            ref_center_time = exc_duration / 2.0 + (0.5 + echo_idx) * self.te
            ref_start_time = ref_center_time - ref_duration / 2
            ref_start = int(ref_start_time / dt)

            if ref_start >= 0 and ref_start + len(ref_pulse) <= npoints:
                b1[ref_start : ref_start + len(ref_pulse)] = ref_pulse

        # Add slice selection gradients
        # (simplified - real implementation would be more complex)
        # Slice gradient G (G/cm) = BW(Hz) / (gamma(Hz/G) * thickness(cm))
        bw_hz = 4.0 / max(exc_duration, dt)
        gamma_hz_per_g = 4258.0
        thickness_cm = max(self.slice_thickness, 1e-3) * 100.0
        if (
            self.slice_gradient_override is not None
            and self.slice_gradient_override > 0
        ):
            gz_amp = self.slice_gradient_override
        else:
            gz_amp = bw_hz / (gamma_hz_per_g * thickness_cm)
        gradients[: max(n_exc, 1), 2] = gz_amp
        for echo_idx in range(self.echo_count):
            ref_center_time = exc_duration / 2.0 + (0.5 + echo_idx) * self.te
            ref_start_time = ref_center_time - ref_duration / 2
            ref_start = int(ref_start_time / dt)

            if ref_start >= 0 and ref_start + len(ref_pulse) <= npoints:
                gradients[ref_start : ref_start + len(ref_pulse), 2] = gz_amp

        return apply_rf_carrier(b1, time, self.rf_freq_offset), gradients, time


class SpinEchoTipAxis(PulseSequence):
    """
    Spin echo where the refocusing 180 is applied around the axis of the tipped magnetization.

    Implemented by phase-shifting the 180 pulse by +90 degrees relative to the excitation phase
    (CPMG-style: 90° about X, 180° about Y).
    """

    def __init__(
        self,
        te: float,
        tr: float,
        custom_excitation=None,
        custom_refocusing=None,
        slice_thickness: float = 0.005,
        slice_gradient_override: Optional[float] = None,
        echo_count: int = 1,
        rf_freq_offset: float = 0.0,
        **kwargs,
    ):
        super().__init__(slice_thickness=slice_thickness, **kwargs)
        self.te = te
        self.tr = tr
        self.custom_excitation = custom_excitation
        self.custom_refocusing = custom_refocusing
        self.slice_gradient_override = slice_gradient_override
        self.echo_count = max(1, int(echo_count))
        self.rf_freq_offset = rf_freq_offset

    def compile(self, dt: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        def _resample_pulse(pulse_data, target_dt, duration):
            """Resample pulse to match simulation dt."""
            b1_in, time_in = pulse_data
            b1_in = np.asarray(b1_in, dtype=complex)
            time_in = np.asarray(time_in, dtype=float)

            if len(time_in) < 2:
                return b1_in

            # Target time points
            num_points = int(np.round(duration / target_dt))
            if num_points <= 0:
                return np.array([], dtype=complex)

            # Original time (normalized to start at 0)
            t_orig = time_in - time_in[0]
            # Target time
            t_new = np.linspace(0, t_orig[-1], num_points)

            # Interpolate complex
            b1_real = np.interp(t_new, t_orig, np.real(b1_in))
            b1_imag = np.interp(t_new, t_orig, np.imag(b1_in))
            return b1_real + 1j * b1_imag

        # Determine pulse durations first
        if self.custom_excitation is not None:
            exc_b1_in, exc_time_in = self.custom_excitation
            if len(exc_time_in) > 1:
                exc_duration = exc_time_in[-1] - exc_time_in[0]
            else:
                exc_duration = len(exc_b1_in) * dt
        else:
            exc_duration = 1e-3

        if self.custom_refocusing is not None:
            ref_b1_in, ref_time_in = self.custom_refocusing
            if len(ref_time_in) > 1:
                ref_duration = ref_time_in[-1] - ref_time_in[0]
            else:
                ref_duration = len(ref_b1_in) * dt
        else:
            ref_duration = 2e-3

        min_duration = exc_duration / 2.0 + (self.echo_count + 0.5) * self.te + 1e-3
        total_duration = max(self.tr, min_duration)
        npoints = int(np.ceil(total_duration / dt))

        enforce_sequence_memory(npoints)
        b1 = np.zeros(npoints, dtype=complex)
        gradients = np.zeros((npoints, 3))
        time = np.arange(npoints) * dt

        # Excitation pulse
        if self.custom_excitation is not None:
            exc_pulse = _resample_pulse(self.custom_excitation, dt, exc_duration)
            n_exc = min(len(exc_pulse), npoints)
            b1[:n_exc] = exc_pulse[:n_exc]
        else:
            exc_pulse, _ = design_rf_pulse(
                "sinc",
                duration=1e-3,
                flip_angle=90,
                npoints=int(1e-3 / dt),
                freq_offset=0.0,
            )
            n_exc = len(exc_pulse)
            b1[:n_exc] = exc_pulse

        # Build a proper 180° refocusing pulse (independent of excitation shape)
        if self.custom_refocusing is not None:
            ref_pulse = _resample_pulse(self.custom_refocusing, dt, ref_duration)
            if np.any(np.abs(ref_pulse) > 0):
                peak_idx = np.argmax(np.abs(ref_pulse))
                ref_phase = np.angle(ref_pulse[peak_idx])
        else:
            ref_pulse, _ = design_rf_pulse(
                "sinc",
                duration=2e-3,
                flip_angle=180,
                npoints=int(2e-3 / dt),
                freq_offset=0.0,
            )

        # Estimate excitation phase from non-zero samples; default to 0
        if np.any(np.abs(b1[:n_exc]) > 0):
            exc_phase = np.angle(np.mean(b1[:n_exc][np.abs(b1[:n_exc]) > 0]))
        else:
            exc_phase = 0.0

        # 180° refocusing pulses every TE
        # Phase shift is typically +90° relative to excitation (CPMG),
        # but we now expect the custom_refocusing pulse to already contain the desired phase
        # (synced from the UI).
        for echo_idx in range(self.echo_count):
            # Center of refocusing pulse at exc_duration/2 + (0.5 + idx) * TE
            ref_center_time = exc_duration / 2.0 + (0.5 + echo_idx) * self.te
            ref_start_time = ref_center_time - len(ref_pulse) * dt / 2.0
            ref_start = int(ref_start_time / dt)

            if ref_start >= 0 and ref_start + len(ref_pulse) <= npoints:
                b1[ref_start : ref_start + len(ref_pulse)] = ref_pulse

        # Slice-select gradients (reuse SpinEcho logic)
        bw_hz = 4.0 / max(exc_duration, dt)
        gamma_hz_per_g = 4258.0
        thickness_cm = max(self.slice_thickness, 1e-3) * 100.0
        if (
            self.slice_gradient_override is not None
            and self.slice_gradient_override > 0
        ):
            gz_amp = self.slice_gradient_override
        else:
            gz_amp = bw_hz / (gamma_hz_per_g * thickness_cm)
        gradients[: max(n_exc, 1), 2] = gz_amp
        for echo_idx in range(self.echo_count):
            ref_center_time = exc_duration / 2.0 + (0.5 + echo_idx) * self.te
            ref_start_time = ref_center_time - len(ref_pulse) * dt / 2.0
            ref_start = int(ref_start_time / dt)

            if ref_start >= 0 and ref_start + len(ref_pulse) <= npoints:
                gradients[ref_start : ref_start + len(ref_pulse), 2] = gz_amp

        return apply_rf_carrier(b1, time, self.rf_freq_offset), gradients, time


class GradientEcho(PulseSequence):
    """
    Gradient echo pulse sequence.
    """

    def __init__(
        self,
        te: float,
        tr: float,
        flip_angle: float = 30,
        custom_excitation=None,
        slice_thickness: float = 0.005,
        slice_gradient_override: Optional[float] = None,
        rf_freq_offset: float = 0.0,
        **kwargs,
    ):
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

    def compile(self, dt: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compile gradient echo sequence."""

        # Determine pulse and readout durations first to validate TE
        if self.custom_excitation is not None:
            exc_b1_in, exc_time_in = self.custom_excitation
            exc_duration = (
                len(exc_b1_in) * dt
                if len(exc_time_in) <= 1
                else exc_time_in[-1] - exc_time_in[0]
            )
        else:
            exc_duration = 1e-3

        readout_duration = 1e-3  # Default readout duration

        # Validate TE
        # Center of excitation to center of readout is TE.
        # TE >= exc_duration/2 + readout_duration/2
        min_te = (exc_duration + readout_duration) / 2.0 * 1.01  # Add buffer
        if self.te < min_te:
            raise ValueError(
                f"TE ({self.te*1000:.2f} ms) is too short. "
                f"Minimum TE ≈ {min_te*1000:.2f} ms (Exc: {exc_duration*1000:.1f}ms, Read: {readout_duration*1000:.1f}ms)"
            )

        # Determine total duration
        # Must cover readout end: TE + readout_duration/2
        min_duration = self.te + readout_duration / 2.0 + 1e-3  # small buffer
        total_duration = max(self.tr, min_duration)
        npoints = int(np.ceil(total_duration / dt))

        # Initialize arrays
        enforce_sequence_memory(npoints)
        b1 = np.zeros(npoints, dtype=complex)
        gradients = np.zeros((npoints, 3))
        time = np.arange(npoints) * dt

        # Excitation pulse
        if self.custom_excitation is not None:
            exc_b1, exc_time = self.custom_excitation
            exc_b1 = np.asarray(exc_b1, dtype=complex)
            n_exc = min(len(exc_b1), npoints)
            b1[:n_exc] = exc_b1[:n_exc]
        else:
            exc_pulse, _ = design_rf_pulse(
                "sinc",
                duration=exc_duration,
                flip_angle=self.flip_angle,
                npoints=int(exc_duration / dt),
                freq_offset=0.0,
            )
            n_exc = len(exc_pulse)
            b1[:n_exc] = exc_pulse

        # Slice selection gradient
        thickness_cm = max(self.slice_thickness, 1e-3) * 100.0
        bw_hz = 4.0 / max(exc_duration, dt)
        gamma_hz_per_g = 4258.0
        if (
            self.slice_gradient_override is not None
            and self.slice_gradient_override > 0
        ):
            gz_amp = self.slice_gradient_override
        else:
            gz_amp = bw_hz / (gamma_hz_per_g * thickness_cm)
        n_exc_active = np.count_nonzero(np.abs(b1) > 0)
        gradients[: max(n_exc_active, 1), 2] = gz_amp

        # Readout gradient
        # Center at TE
        readout_start_time = self.te - readout_duration / 2.0
        readout_start = int(readout_start_time / dt)
        readout_pts = int(readout_duration / dt)

        if readout_start >= 0 and readout_start + readout_pts <= npoints:
            gradients[readout_start : readout_start + readout_pts, 0] = 5e-3

        return apply_rf_carrier(b1, time, self.rf_freq_offset), gradients, time


class InversionRecovery(PulseSequence):
    """
    Inversion recovery pulse sequence (180 -> TI -> 90).
    """

    def __init__(
        self,
        ti: float,
        tr: float,
        te: float = 0.0,
        pulse_type: str = "sinc",
        slice_thickness: float = 0.005,
        slice_gradient_override: Optional[float] = None,
        custom_inversion=None,
        custom_excitation=None,
        rf_freq_offset: float = 0.0,
        **kwargs,
    ):
        """
        Initialize inversion recovery sequence.

        Parameters
        ----------
        ti : float
            Inversion time (center of 180 to center of 90) in seconds
        tr : float
            Repetition time in seconds
        te : float
            Echo time (time from 90 to readout center) in seconds.
        pulse_type : str
            Type of pulses to use ('sinc', 'rect', 'gaussian', etc.) if custom pulses are not provided.
            Ensures both pulses are of the same kind.
        """
        super().__init__(slice_thickness=slice_thickness, **kwargs)
        self.ti = ti
        self.tr = tr
        self.te = te
        self.pulse_type = pulse_type
        self.slice_gradient_override = slice_gradient_override
        self.custom_inversion = custom_inversion
        self.custom_excitation = custom_excitation
        self.rf_freq_offset = rf_freq_offset

    def compile(self, dt: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compile inversion recovery sequence."""

        def _resample_pulse(pulse_data, target_dt, duration):
            """Resample pulse to match simulation dt."""
            b1_in, time_in = pulse_data
            b1_in = np.asarray(b1_in, dtype=complex)
            time_in = np.asarray(time_in, dtype=float)

            if len(time_in) < 2:
                return b1_in

            # Target time points
            num_points = int(np.round(duration / target_dt))
            if num_points <= 0:
                return np.array([], dtype=complex)

            # Original time (normalized to start at 0)
            t_orig = time_in - time_in[0]
            # Target time
            t_new = np.linspace(0, t_orig[-1], num_points)

            # Interpolate complex
            b1_real = np.interp(t_new, t_orig, np.real(b1_in))
            b1_imag = np.interp(t_new, t_orig, np.imag(b1_in))
            return b1_real + 1j * b1_imag

        # Calculate durations
        if self.custom_inversion is not None:
            inv_b1_in, inv_time_in = self.custom_inversion
            if len(inv_time_in) > 1:
                inv_duration = inv_time_in[-1] - inv_time_in[0]
            else:
                inv_duration = len(inv_b1_in) * dt
        else:
            inv_duration = 2e-3

        if self.custom_excitation is not None:
            exc_b1_in, exc_time_in = self.custom_excitation
            if len(exc_time_in) > 1:
                exc_duration = exc_time_in[-1] - exc_time_in[0]
            else:
                exc_duration = len(exc_b1_in) * dt
        else:
            exc_duration = 1e-3

        # Ensure minimal duration
        min_duration = self.ti + self.te + 5e-3
        total_duration = max(self.tr, min_duration)
        npoints = int(np.ceil(total_duration / dt))

        enforce_sequence_memory(npoints)
        b1 = np.zeros(npoints, dtype=complex)
        gradients = np.zeros((npoints, 3))
        time = np.arange(npoints) * dt

        # --- 1. Inversion Pulse (180) ---
        if self.custom_inversion is not None:
            inv_pulse = _resample_pulse(self.custom_inversion, dt, inv_duration)
            n_inv = min(len(inv_pulse), npoints)
            b1[:n_inv] = inv_pulse[:n_inv]
        else:
            # Generate 180 of the specified type
            inv_pulse, _ = design_rf_pulse(
                self.pulse_type,
                duration=inv_duration,
                flip_angle=180,
                npoints=int(inv_duration / dt),
                freq_offset=0.0,
            )
            n_inv = min(len(inv_pulse), npoints)
            b1[:n_inv] = inv_pulse[:n_inv]

        inv_center_time = (n_inv * dt) / 2.0  # Approximate center

        # Slice gradient for inversion
        thickness_cm = max(self.slice_thickness, 1e-3) * 100.0
        bw_hz = 4.0 / max(inv_duration, dt)  # approx
        gamma_hz_per_g = 4258.0

        if (
            self.slice_gradient_override is not None
            and self.slice_gradient_override > 0
        ):
            gz_amp = self.slice_gradient_override
        else:
            gz_amp = bw_hz / (gamma_hz_per_g * thickness_cm)

        gradients[:n_inv, 2] = gz_amp

        # --- 2. Excitation Pulse (90) ---
        if self.custom_excitation is not None:
            exc_pulse = _resample_pulse(self.custom_excitation, dt, exc_duration)
        else:
            # Generate 90 of the SAME type
            exc_pulse, _ = design_rf_pulse(
                self.pulse_type,
                duration=exc_duration,
                flip_angle=90,
                npoints=int(exc_duration / dt),
                freq_offset=0.0,
            )

        n_exc = len(exc_pulse)
        exc_center_time = (n_exc * dt) / 2.0

        # Calculate start time for excitation to match TI (center-to-center)
        # TI = (exc_start + exc_center) - inv_center
        # exc_start = TI + inv_center - exc_center
        exc_start_time = self.ti + inv_center_time - exc_center_time
        exc_start_idx = int(exc_start_time / dt)

        # Safety check: don't overlap
        if exc_start_idx < n_inv:
            exc_start_idx = n_inv + 10  # minimal gap

        if exc_start_idx + n_exc < npoints:
            b1[exc_start_idx : exc_start_idx + n_exc] = exc_pulse

            # Slice gradient for excitation
            bw_hz_exc = 4.0 / max(exc_duration, dt)
            if (
                self.slice_gradient_override is not None
                and self.slice_gradient_override > 0
            ):
                gz_amp_exc = self.slice_gradient_override
            else:
                gz_amp_exc = bw_hz_exc / (gamma_hz_per_g * thickness_cm)
            gradients[exc_start_idx : exc_start_idx + n_exc, 2] = gz_amp_exc

        # --- 3. Readout ---
        # Assuming simple FID readout starting after excitation or at TE
        # Center of excitation is at exc_start_time + exc_center_time
        # We want readout center at TE after that? Or TE relative to excitation center?
        # Usually TE in IR is defined if there is a refocusing pulse (IR-SE).
        # If it's IR-FID, TE might just mean "start acquisition".
        # Let's assume readout starts shortly after excitation for FID.

        if self.te > 0:
            # If TE provided, maybe we want a gradient echo or just wait?
            # For simplicity, let's put a readout gradient lobe at TE
            ro_center = (exc_start_idx * dt) + exc_center_time + self.te
            ro_start = int((ro_center - 0.5e-3) / dt)
        else:
            ro_start = exc_start_idx + n_exc + 10

        ro_dur = int(1e-3 / dt)
        if ro_start + ro_dur < npoints:
            gradients[ro_start : ro_start + ro_dur, 0] = 5e-3  # Readout gradient

        return apply_rf_carrier(b1, time, self.rf_freq_offset), gradients, time


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

    def compile(self, dt: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            pulse_duration = (
                pulse_time[-1] - pulse_time[0]
                if len(pulse_time) > 1
                else self.pulse_duration
            )
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
        if (
            self.slice_gradient_override is not None
            and self.slice_gradient_override > 0
        ):
            gz_gauss_per_cm = self.slice_gradient_override
        else:
            gz_gauss_per_cm = bw_hz / (gamma_hz_per_g * thickness_cm)
        gradients[:n_rf, 2] = gz_gauss_per_cm

        # Rephasing lobe with half the area of the excitation lobe
        area_exc = gz_gauss_per_cm * pulse_duration
        rephase_amp = -(0.5 * area_exc) / (n_rephase * dt)
        start_rephase = n_rf + gap_pts
        gradients[start_rephase : start_rephase + n_rephase, 2] = rephase_amp
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

    def compile(self, dt: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
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
            enforce_sequence_memory(new_npoints)
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

    def __init__(
        self,
        use_parallel: bool = True,
        num_threads: Optional[int] = None,
        verbose: bool = False,
        memory_limit_bytes: Optional[int] = None,
        memory_policy: Optional[MemoryPolicy] = None,
        sequence_kernel: str = "optimized",
        dynamic_sequence_kernel: str = "optimized",
        dynamic_sequence_precision: str = "float64",
    ):
        """
        Initialize the Bloch simulator.

        Parameters
        ----------
        use_parallel : bool
            Use parallel processing
        num_threads : int, optional
            Number of threads for parallel processing. ``None`` or zero uses
            all logical processors reported by the operating system.
        verbose : bool
            Print progress messages
        memory_limit_bytes : int, optional
            Explicit RAM budget for one simulation. If omitted, the budget is
            derived from currently available system RAM.
        memory_policy : MemoryPolicy, optional
            Reserve-based or fixed-limit RAM policy. The process-wide GUI policy
            is used when omitted.
        sequence_kernel : {"optimized", "reference"}
            Native kernel used by event-based sequence simulations. The reference
            kernel remains available for numerical comparisons.
        dynamic_sequence_kernel : {"optimized", "native_parallel", "native_serial", "metal_hybrid", "reference"}
            Kernel used specifically for dynamic two-pool sequence simulations.
        dynamic_sequence_precision : {"float64", "float32"}
            Arithmetic used by the optimized dynamic CPU path. ``float64`` is
            the bit-exact default; ``float32`` is an experimental GPU-precision
            validation path.
        """
        self.use_parallel = use_parallel
        self.num_threads = resolve_num_threads(num_threads)
        self.verbose = verbose
        self.memory_limit_bytes = memory_limit_bytes
        self.memory_policy = memory_policy
        if sequence_kernel not in {"optimized", "reference"}:
            raise ValueError("sequence_kernel must be 'optimized' or 'reference'")
        self.sequence_kernel = sequence_kernel
        if dynamic_sequence_kernel not in {
            "optimized",
            "native_parallel",
            "native_serial",
            "metal_hybrid",
            "reference",
        }:
            raise ValueError(
                "dynamic_sequence_kernel must be 'optimized', 'native_parallel', "
                "'native_serial', 'metal_hybrid', or 'reference'"
            )
        self.dynamic_sequence_kernel = dynamic_sequence_kernel
        if dynamic_sequence_precision not in {"float64", "float32"}:
            raise ValueError(
                "dynamic_sequence_precision must be 'float64' or 'float32'"
            )
        if (
            dynamic_sequence_precision == "float32"
            and dynamic_sequence_kernel != "optimized"
        ):
            raise ValueError(
                "float32 dynamic precision currently requires the optimized kernel"
            )
        self.dynamic_sequence_precision = dynamic_sequence_precision
        self.last_result = None

        # Validate settings immediately instead of waiting for the first run.
        self._memory_budget()

    def _memory_budget(self):
        return resolve_memory_budget(
            policy=self.memory_policy,
            explicit_limit_bytes=self.memory_limit_bytes,
        )

    def _check_standard_simulation_memory(
        self, ntime: int, npos: int, nfreq: int, mode: int
    ) -> None:
        """Check peak memory before the Cython output buffers are allocated."""
        ntout = ntime if (mode & 2) else 1
        output_samples = ntout * npos * nfreq
        spin_count = npos * nfreq

        # Peak estimate includes Mx/My/Mz, the complex signal and NumPy/Cython
        # temporaries. Sequence working arrays and initial magnetization are
        # included separately. This is intentionally conservative.
        estimated_bytes = output_samples * 64 + ntime * 64 + spin_count * 56
        if mode & 2:
            suggestions = (
                "Use Endpoint mode, increase the time step, reduce positions or "
                "frequencies, or enable Preview"
            )
        else:
            suggestions = (
                "Increase the time step, reduce positions or frequencies, or "
                "enable Preview"
            )
        enforce_memory_budget(
            estimated_bytes,
            self._memory_budget(),
            description=(
                f"Requested {ntime:,} time points × {npos:,} positions × "
                f"{nfreq:,} frequencies ({output_samples:,} output samples)"
            ),
            suggestions=suggestions,
        )

    def _check_phantom_simulation_memory(
        self, ntime: int, n_active: int, n_voxels: int, mode: int
    ) -> None:
        """Check peak memory for active buffers plus full phantom reconstruction."""
        ntout = ntime if (mode & 2) else 1
        active_samples = ntout * n_active
        full_samples = ntout * n_voxels
        estimated_bytes = (
            active_samples * 48 + full_samples * 56 + ntime * 64 + n_active * 96
        )
        enforce_memory_budget(
            estimated_bytes,
            self._memory_budget(),
            description=(
                f"Requested {ntime:,} time points × {n_active:,} active voxels "
                f"in a {n_voxels:,}-voxel phantom"
            ),
            suggestions=(
                "Use Endpoint mode, increase the time step, or reduce the phantom "
                "resolution"
            ),
        )

    def log_message(self, message: str):
        """Print a message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def simulate(
        self,
        sequence: Union[PulseSequence, Tuple],
        tissue: TissueParameters,
        positions: Optional[np.ndarray] = None,
        frequencies: Optional[np.ndarray] = None,
        initial_magnetization: Optional[np.ndarray] = None,
        dt: float = 1e-5,
        mode: int = 0,
        rf_carrier_offset: Optional[float] = None,
    ) -> Dict:
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
        rf_carrier_offset : float, optional
            RF carrier offset in Hz. Used to report effective detuning as
            ``frequencies - rf_carrier_offset``. For PulseSequence instances,
            the sequence value is inferred when omitted.

        Returns
        -------
        dict
            Dictionary containing:
            - 'mx', 'my', 'mz': Magnetization components
            - 'signal': Complex MRI signal
            - 'time': Time points
            - 'positions': Positions used
            - 'frequencies': Laboratory-frame spin offsets used
            - 'effective_frequencies': Spin offsets relative to the RF carrier
            - 'rf_carrier_offset': RF carrier offset in Hz
        """

        if rf_carrier_offset is None:
            rf_carrier_offset = float(getattr(sequence, "rf_freq_offset", 0.0))
        else:
            rf_carrier_offset = float(rf_carrier_offset)

        # Compile sequence if needed
        if isinstance(sequence, PulseSequence):
            b1, gradients, time = sequence.compile(dt)
        else:
            b1, gradients, time = sequence

        # Sanitize and standardize inputs to avoid buffer errors from Cython
        b1 = np.asarray(b1, dtype=np.complex128)
        b1 = np.squeeze(b1)
        if b1.ndim != 1:
            raise ValueError(
                f"B1 array must be 1D after squeezing, got shape {b1.shape}"
            )
        # design_rf_pulse already returns Gauss; keep units unchanged
        b1_gauss = np.ascontiguousarray(b1)

        gradients = np.asarray(gradients, dtype=np.float64)
        if gradients.ndim == 1:
            gradients = gradients.reshape(-1, 1)
        if gradients.ndim != 2:
            raise ValueError(
                f"Gradients must be 2D, got {gradients.ndim}D with shape {gradients.shape}"
            )
        if gradients.shape[1] < 3:
            gradients = np.pad(
                gradients, ((0, 0), (0, 3 - gradients.shape[1])), mode="constant"
            )
        elif gradients.shape[1] > 3:
            gradients = gradients[:, :3]
        if gradients.shape[0] != b1_gauss.shape[0]:
            raise ValueError(
                f"Gradients length ({gradients.shape[0]}) must match B1 length ({b1_gauss.shape[0]})"
            )
        # Gradients expected in Gauss/cm already
        gradients_gauss = np.ascontiguousarray(gradients)

        time = np.asarray(time, dtype=np.float64)
        time = np.squeeze(time)
        if time.ndim != 1:
            raise ValueError(
                f"Time array must be 1D after squeezing, got shape {time.shape}"
            )
        if time.shape[0] != b1_gauss.shape[0]:
            raise ValueError(
                f"Time length ({time.shape[0]}) must match B1 length ({b1_gauss.shape[0]})"
            )
        time = np.ascontiguousarray(time)

        # Default positions and frequencies
        if positions is None:
            positions = np.array([[0.0, 0.0, 0.0]])
        positions = np.asarray(positions, dtype=np.float64)
        positions = np.atleast_2d(positions)
        if positions.shape[1] < 3:
            positions = np.pad(
                positions, ((0, 0), (0, 3 - positions.shape[1])), mode="constant"
            )
        elif positions.shape[1] > 3:
            positions = positions[:, :3]
        positions_cm = np.ascontiguousarray(positions * 100)  # m -> cm

        if frequencies is None:
            frequencies = np.array([0.0])
        frequencies = np.asarray(frequencies, dtype=np.float64)
        frequencies = np.ravel(frequencies)
        frequencies = np.ascontiguousarray(frequencies)

        self._check_standard_simulation_memory(
            len(time), positions.shape[0], frequencies.shape[0], mode
        )

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
                    raise ValueError(
                        "Initial magnetization must be scalar or length-3 vector."
                    )
            elif init_arr.ndim == 2:
                # Accept (3, nfnpos) or (nfnpos, 3)
                if init_arr.shape == (3, nfnpos):
                    m_init = init_arr
                elif init_arr.shape == (nfnpos, 3):
                    m_init = init_arr.T
                else:
                    raise ValueError(
                        f"Initial magnetization shape must be (3, npos*nfreq); got {init_arr.shape}."
                    )
            else:
                raise ValueError(
                    "Initial magnetization must be scalar, length-3 vector, or (3, npos*nfreq) array."
                )
            m_init = np.ascontiguousarray(m_init, dtype=np.float64)

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
        if (
            self.use_parallel
            and len(positions) * len(frequencies) >= parallel_threshold
        ):
            mx, my, mz = simulate_bloch_parallel(
                b1_gauss,
                gradients_gauss,
                dt_array,
                tissue.t1,
                tissue.t2,
                frequencies,
                positions_cm,
                m_init,
                mode,
                self.num_threads,
            )
        else:
            mx, my, mz = simulate_bloch(
                b1_gauss,
                gradients_gauss,
                dt_array,
                tissue.t1,
                tissue.t2,
                frequencies,
                positions_cm,
                m_init,
                mode,
            )

        # Calculate complex signal
        signal = calculate_signal(mx, my, mz)

        # Store result
        self.last_result = {
            "mx": mx,
            "my": my,
            "mz": mz,
            "signal": signal,
            "time": time,
            "positions": positions,
            "frequencies": frequencies,
            "effective_frequencies": frequencies - rf_carrier_offset,
            "rf_carrier_offset": rf_carrier_offset,
            "tissue": tissue,
        }

        return self.last_result

    def simulate_phantom(
        self,
        phantom,
        sequence: Union[PulseSequence, Tuple],
        dt: float = 1e-5,
        mode: int = 0,
        additional_frequencies: Optional[np.ndarray] = None,
        use_grouped: bool = True,
    ) -> Dict:
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
            from .phantom import Phantom
        except ImportError:
            raise ImportError(
                "Phantom module not found. Ensure phantom.py is available."
            )

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
            gradients = np.pad(
                gradients, ((0, 0), (0, 3 - gradients.shape[1])), mode="constant"
            )
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
        n_active = len(props["t1"])

        if n_active == 0:
            self.log_message("Warning: No active voxels in phantom (all masked)")
            # Return zeros
            if mode & 2:
                shape = (len(time),) + phantom.shape
            else:
                shape = phantom.shape
            zeros = np.zeros(shape, dtype=np.float64)
            return {
                "mx": zeros,
                "my": zeros,
                "mz": zeros,
                "signal": np.zeros(shape, dtype=np.complex128),
                "time": time,
                "phantom": phantom,
                "pd_weighted_signal": np.zeros(shape, dtype=np.complex128),
            }

        self._check_phantom_simulation_memory(
            len(time), n_active, phantom.nvoxels, mode
        )

        # Convert positions from meters to cm (Bloch core uses Gauss/cm)
        positions_cm = props["positions"] * 100  # m -> cm
        positions_cm = np.ascontiguousarray(positions_cm, dtype=np.float64)

        # Frequency offsets
        df_array = np.ascontiguousarray(props["df"], dtype=np.float64)

        # Initial magnetization
        m_init = np.ascontiguousarray(props["m0"], dtype=np.float64)

        # Log the output size after the memory check has accepted it.
        ntout = len(time) if (mode & 2) else 1
        total_samples = ntout * n_active

        self.log_message(
            f"Simulation size: {n_active} voxels × {ntout} time points = {total_samples:.1e} samples"
        )

        # Import wrapper function
        try:
            from .blochsimulator_cy import simulate_phantom as simulate_phantom_core
        except ImportError:
            raise ImportError(
                "blochsimulator_cy not compiled. Run: python setup.py build_ext --inplace"
            )

        # Run simulation
        t1_array = np.ascontiguousarray(props["t1"], dtype=np.float64)
        t2_array = np.ascontiguousarray(props["t2"], dtype=np.float64)

        self.log_message("Running heterogeneous Bloch simulation...")
        mx, my, mz = simulate_phantom_core(
            b1_gauss,
            gradients_gauss,
            dt_array,
            t1_array,
            t2_array,
            df_array,
            positions_cm,
            m_init,
            mode,
            self.num_threads,
        )

        # Reconstruct full phantom shape from active voxels
        indices = props["indices"]

        if mode & 2:
            # Time-resolved: (ntime, n_active) -> (ntime, *phantom.shape)
            mx_full = phantom.reconstruct_from_active(
                mx, indices, has_time=True, fill_value=0.0
            )
            my_full = phantom.reconstruct_from_active(
                my, indices, has_time=True, fill_value=0.0
            )
            mz_full = phantom.reconstruct_from_active(
                mz, indices, has_time=True, fill_value=0.0
            )
        else:
            # Endpoint: (n_active,) -> (*phantom.shape,)
            mx_full = phantom.reconstruct_from_active(
                mx, indices, has_time=False, fill_value=0.0
            )
            my_full = phantom.reconstruct_from_active(
                my, indices, has_time=False, fill_value=0.0
            )
            mz_full = phantom.reconstruct_from_active(
                mz, indices, has_time=False, fill_value=0.0
            )

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
            self.log_message(
                f"Received signal shape: {received_signal.shape} (sum over {pd_weighted.shape[1:]})"
            )
        else:
            # Endpoint: sum over all spatial dimensions
            received_signal = np.sum(pd_weighted)
            self.log_message(f"Received signal (endpoint): {received_signal}")

        # Store result
        self.last_result = {
            "mx": mx_full,
            "my": my_full,
            "mz": mz_full,
            "signal": signal_per_voxel,  # Per-voxel signal (for imaging)
            "time": time,
            "phantom": phantom,
            "pd_weighted_signal": pd_weighted,  # Per-voxel signal * PD
            "received_signal": received_signal,  # Total signal (what coil measures)
        }

        self.log_message(f"Simulation complete. Output shape: {mx_full.shape}")

        return self.last_result

    def simulate_spectral_sequence(
        self,
        program,
        phantom,
        *,
        checkpoints_s=(),
        chunk_voxels: Optional[int] = None,
        signal_weighting: str = "voxel",
        field_strength_t: Optional[float] = None,
        nucleus: Optional[str] = None,
        sequence_reference_ppm: Optional[float] = None,
        progress_callback=None,
        preview_callback=None,
        cancel_callback=None,
        status_callback=None,
        simulation_timestep_s=None,
        sequence_kernel=None,
        spin_sampling=None,
        spoiler_mode: str = "ideal",
        checkpoint_dtype=None,
    ):
        """Simulate independent Lorentzian spectral components and sum signals.

        Each spectral peak is represented by an independent isochromat with its
        own centre frequency and ``T2*`` transverse decay. This produces a
        Lorentzian frequency-domain line. Chemical exchange and J-coupling are
        intentionally not part of this independent-component model.
        """
        from .sequence import SequenceSimulationResult
        from .spectral_phantom import SpectralPhantom

        if not isinstance(phantom, SpectralPhantom):
            raise TypeError(f"phantom must be SpectralPhantom, got {type(phantom)}")
        effective_field = (
            phantom.field_strength
            if field_strength_t is None
            else float(field_strength_t)
        )
        effective_nucleus = phantom.nucleus if nucleus is None else str(nucleus)
        effective_reference_ppm = (
            phantom.spectral_reference_ppm
            if sequence_reference_ppm is None
            else float(sequence_reference_ppm)
        )
        components = phantom.to_component_phantoms(
            effective_field,
            effective_nucleus,
            sequence_reference_ppm=effective_reference_ppm,
        )
        if not components:
            raise ValueError("spectral phantom has no active components")

        combined_signal = None
        component_signals = []
        component_final_magnetizations = []
        component_checkpoint_magnetizations = []
        component_active_voxels = {}
        component_simulated_spins = {}
        final_numerator = np.zeros(phantom.shape + (3,), dtype=np.float64)
        checkpoint_numerator = None
        total_concentration = phantom.get_total_concentration()
        first_result = None

        for component_index, (name, component) in enumerate(components):
            if cancel_callback is not None and cancel_callback():
                raise RuntimeError("sequence simulation cancelled")

            def component_preview(fraction, partial_signal):
                if preview_callback is None:
                    return
                signal = np.asarray(partial_signal)
                if combined_signal is not None:
                    signal = combined_signal + signal
                preview_callback(
                    (component_index + float(fraction)) / len(components),
                    signal,
                )

            def component_status(message):
                if status_callback is not None:
                    status_callback(
                        f"Component {component_index + 1}/{len(components)} "
                        f"({name}): {message}"
                    )

            result = self.simulate_sequence(
                program,
                component,
                checkpoints_s=checkpoints_s,
                chunk_voxels=chunk_voxels,
                signal_weighting=signal_weighting,
                preview_callback=(component_preview if preview_callback else None),
                cancel_callback=cancel_callback,
                status_callback=component_status,
                simulation_timestep_s=simulation_timestep_s,
                sequence_kernel=sequence_kernel,
                spin_sampling=spin_sampling,
                spoiler_mode=spoiler_mode,
                checkpoint_dtype=checkpoint_dtype,
            )
            if first_result is None:
                first_result = result
                combined_signal = np.zeros_like(result.signal)
                if result.checkpoint_magnetization is not None:
                    checkpoint_numerator = np.zeros_like(
                        result.checkpoint_magnetization
                    )
            component_signal = np.asarray(result.signal)
            component_signals.append(component_signal)
            component_active_voxels[name] = int(
                result.metadata.get("n_active_voxels", component.n_active)
            )
            component_simulated_spins[name] = int(
                result.metadata.get("n_simulated_spins", component.n_active)
            )
            component_final_magnetizations.append(
                np.asarray(result.final_magnetization)
            )
            if result.checkpoint_magnetization is not None:
                component_checkpoint_magnetizations.append(
                    np.asarray(result.checkpoint_magnetization)
                )
            combined_signal += component_signal
            concentration = component.pd_map
            final_numerator += result.final_magnetization * concentration[..., None]
            if checkpoint_numerator is not None:
                checkpoint_numerator += (
                    result.checkpoint_magnetization * concentration[None, ..., None]
                )
            if progress_callback is not None:
                progress_callback(component_index + 1, len(components))

        denominator = total_concentration[..., None]
        final_magnetization = np.divide(
            final_numerator,
            denominator,
            out=np.zeros_like(final_numerator),
            where=denominator > 0,
        )
        checkpoint_magnetization = None
        if checkpoint_numerator is not None:
            checkpoint_magnetization = np.divide(
                checkpoint_numerator,
                denominator[None, ...],
                out=np.zeros_like(checkpoint_numerator),
                where=denominator[None, ...] > 0,
            )
        metadata = dict(first_result.metadata)
        metadata.update(
            {
                "spectral_phantom": True,
                "spectral_components": [name for name, _ in components],
                "spectral_component_count": len(components),
                "spectral_component_active_voxels": component_active_voxels,
                "spectral_component_simulated_spins": component_simulated_spins,
                "n_active_voxels": int(phantom.n_active),
                "n_simulated_spins": int(sum(component_simulated_spins.values())),
                "spectral_model": "independent Lorentzian T2* components",
                "field_strength_t": effective_field,
                "nucleus": effective_nucleus,
                "frequency_input_unit": "ppm",
                "spectral_reference_ppm": phantom.spectral_reference_ppm,
                "sequence_reference_ppm": effective_reference_ppm,
                "spectral_window_center_ppm": phantom.spectral_window_center_ppm,
                "spectral_bandwidth_ppm": phantom.spectral_bandwidth_ppm,
                "spectral_points": phantom.spectral_points,
            }
        )
        return SequenceSimulationResult(
            signal=combined_signal,
            adc_times_s=first_result.adc_times_s,
            final_magnetization=final_magnetization,
            checkpoint_magnetization=checkpoint_magnetization,
            checkpoint_times_s=first_result.checkpoint_times_s,
            metadata=metadata,
            adc_gradient_moment_cyc_per_m=(first_result.adc_gradient_moment_cyc_per_m),
            pool_names=tuple(name for name, _ in components),
            species_signal=np.stack(component_signals, axis=0),
            final_pool_magnetization=np.stack(component_final_magnetizations, axis=0),
            checkpoint_pool_magnetization=(
                np.stack(component_checkpoint_magnetizations, axis=1)
                if component_checkpoint_magnetizations
                else None
            ),
            sequence_waveforms=first_result.sequence_waveforms,
            physical_field_maps=first_result.physical_field_maps,
        )

    def simulate_dynamic_sequence(self, program, phantom, **kwargs):
        """Simulate a regional two-pool hyperpolarized dynamic phantom."""
        if self.dynamic_sequence_kernel == "metal_hybrid":
            from .dynamic_metal_backend import run_metal_hybrid_sequence

            kwargs.setdefault("memory_budget_bytes", self._memory_budget().limit_bytes)
            return run_metal_hybrid_sequence(program, phantom, **kwargs)

        from .dynamic_phantom import simulate_dynamic_sequence

        kwargs.setdefault("sequence_kernel", self.dynamic_sequence_kernel)
        kwargs.setdefault("simulation_precision", self.dynamic_sequence_precision)
        kwargs.setdefault("use_parallel", self.use_parallel)
        kwargs.setdefault("num_threads", self.num_threads)
        kwargs.setdefault("memory_budget_bytes", self._memory_budget().limit_bytes)
        return simulate_dynamic_sequence(program, phantom, **kwargs)

    def simulate_sequence_probes(
        self,
        program,
        positions_m,
        frequency_offsets_hz,
        *,
        checkpoints_s,
        t1_s: float = 25.0,
        t2_s: float = 0.3,
        initial_magnetization=(0.0, 0.0, 1.0),
        chunk_voxels: Optional[int] = None,
        signal_weighting: str = "voxel",
        progress_callback=None,
        preview_callback=None,
        cancel_callback=None,
        status_callback=None,
        simulation_timestep_s=None,
        sequence_kernel=None,
    ):
        """Simulate explicit position/frequency spin probes over a sequence.

        ``positions_m`` is ``(position, xyz)`` and ``frequency_offsets_hz`` is
        one-dimensional.  The simulator evaluates the Cartesian product of both
        axes and returns time-resolved magnetization at ``checkpoints_s``.
        """
        from .phantom import Phantom
        from .sequence import SequenceProbeResult, SequenceProgram

        if not isinstance(program, SequenceProgram):
            raise TypeError(f"program must be SequenceProgram, got {type(program)}")
        checkpoints = np.asarray(tuple(checkpoints_s), dtype=float)
        if checkpoints.ndim != 1 or checkpoints.size == 0:
            raise ValueError("checkpoints_s must contain at least one time point")
        if not np.all(np.isfinite(checkpoints)):
            raise ValueError("checkpoints_s must be finite")
        if np.any(checkpoints < 0) or np.any(checkpoints > program.duration_s):
            raise ValueError("checkpoints_s values must lie within the sequence")

        positions = np.asarray(positions_m, dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions_m must have shape (position, 3)")
        if positions.shape[0] == 0 or not np.all(np.isfinite(positions)):
            raise ValueError("positions_m must contain finite positions")

        frequencies = np.asarray(frequency_offsets_hz, dtype=float)
        if frequencies.ndim == 0:
            frequencies = frequencies.reshape(1)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequency_offsets_hz must be a non-empty 1D array")
        if not np.all(np.isfinite(frequencies)):
            raise ValueError("frequency_offsets_hz must be finite")
        if not np.isfinite(t1_s) or t1_s <= 0 or not np.isfinite(t2_s) or t2_s <= 0:
            raise ValueError("t1_s and t2_s must be positive and finite")

        n_positions = positions.shape[0]
        n_frequencies = frequencies.size
        n_spins = n_positions * n_frequencies
        repeated_positions = np.repeat(positions, n_frequencies, axis=0)
        tiled_frequencies = np.tile(frequencies, n_positions)
        m0 = np.asarray(initial_magnetization, dtype=float)
        if m0.shape == (3,):
            m0_map = np.broadcast_to(m0, (n_spins, 3)).copy()
        elif m0.shape == (n_spins, 3):
            m0_map = m0.copy()
        else:
            raise ValueError(
                "initial_magnetization must have shape (3,) or (n_spins, 3)"
            )
        if not np.all(np.isfinite(m0_map)):
            raise ValueError("initial_magnetization must be finite")

        phantom = Phantom(
            shape=(n_spins,),
            fov=(1.0,),
            t1_map=np.full((n_spins,), float(t1_s), dtype=float),
            t2_map=np.full((n_spins,), float(t2_s), dtype=float),
            pd_map=np.ones((n_spins,), dtype=float),
            chemical_shift_map=tiled_frequencies,
            m0_map=m0_map,
            mask=np.ones((n_spins,), dtype=bool),
            name="Sequence spin probes",
            metadata={
                "probe_phantom": True,
                "n_probe_positions": n_positions,
                "n_probe_frequencies": n_frequencies,
            },
        )
        # Probe coordinates are not constrained to an axis-aligned grid.
        phantom.positions = repeated_positions.copy()
        phantom.x = repeated_positions[:, 0].copy()
        phantom.y = repeated_positions[:, 1].copy()
        phantom.z = repeated_positions[:, 2].copy()

        result = self.simulate_sequence(
            program,
            phantom,
            checkpoints_s=tuple(float(value) for value in checkpoints),
            chunk_voxels=chunk_voxels,
            signal_weighting=signal_weighting,
            progress_callback=progress_callback,
            preview_callback=preview_callback,
            cancel_callback=cancel_callback,
            status_callback=status_callback,
            simulation_timestep_s=simulation_timestep_s,
            sequence_kernel=sequence_kernel,
        )
        if result.checkpoint_magnetization is None:
            raise RuntimeError("probe simulation did not return checkpoint states")
        magnetization = np.asarray(result.checkpoint_magnetization).reshape(
            checkpoints.size,
            n_positions,
            n_frequencies,
            3,
        )
        return SequenceProbeResult(
            time_s=np.asarray(result.checkpoint_times_s, dtype=float),
            positions_m=positions,
            frequency_offsets_hz=frequencies,
            magnetization=magnetization,
            metadata={
                "sequence_source": program.source,
                "sequence_version": program.version,
                "duration_s": program.duration_s,
                "n_positions": n_positions,
                "n_frequencies": n_frequencies,
                "t1_s": float(t1_s),
                "t2_s": float(t2_s),
                "initial_magnetization": (
                    m0.tolist() if m0.shape == (3,) else "per-spin"
                ),
                "simulation_timestep_s": simulation_timestep_s,
                "sequence_kernel": result.metadata.get("sequence_kernel"),
                "probe_type": (
                    "spectral"
                    if n_positions == 1 and n_frequencies > 1
                    else (
                        "geometry" if n_positions > 1 and n_frequencies == 1 else "grid"
                    )
                ),
            },
        )

    def simulate_sequence(
        self,
        program,
        phantom,
        *,
        checkpoints_s=(),
        chunk_voxels: Optional[int] = None,
        signal_weighting: str = "voxel",
        progress_callback=None,
        preview_callback=None,
        cancel_callback=None,
        status_callback=None,
        simulation_timestep_s=None,
        sequence_kernel=None,
        spin_sampling=None,
        spoiler_mode: str = "ideal",
        checkpoint_dtype=None,
    ):
        """Simulate an event-based sequence on a heterogeneous 1D/2D/3D object.

        The streaming path stores only ADC samples, explicitly requested
        checkpoint states, and final magnetization. Canonical sequence units are
        Hz, Hz/m, metres, and seconds.

        ``signal_weighting='voxel'`` preserves the historical relative signal
        sum. ``'voxel_volume'`` additionally multiplies proton density by the
        physical voxel volume of a 3D phantom. ``simulation_timestep_s`` sets
        the maximum compiler interval while RF is active; event boundaries and
        ADC observation times are always retained exactly. ``sequence_kernel``
        overrides the simulator's ``"optimized"`` or ``"reference"`` kernel
        selection for this call. ``spin_sampling`` controls deterministic
        intravoxel position sampling in gradient-waveform mode.
        ``spoiler_mode='ideal'`` always uses one spin per voxel and applies
        declared transverse crushers, while ``'gradient'`` derives spoiling
        only from the simulated gradient waveform and spin positions.
        """
        from .phantom import Phantom
        from .sequence import (
            SequenceCompiler,
            SequenceProgram,
            SequenceSimulationResult,
        )
        from .sequence.spin_sampling import (
            coerce_spin_sampling,
            phantom_voxel_basis_m,
        )

        if not isinstance(program, SequenceProgram):
            raise TypeError(f"program must be SequenceProgram, got {type(program)}")
        if not isinstance(phantom, Phantom):
            raise TypeError(f"phantom must be Phantom, got {type(phantom)}")

        spoiler_mode = str(spoiler_mode).strip().lower()
        if spoiler_mode not in {"ideal", "gradient"}:
            raise ValueError("spoiler_mode must be 'ideal' or 'gradient'")
        sampling = coerce_spin_sampling(
            spin_sampling if spoiler_mode == "gradient" else None
        )
        sampling.validate_phantom_dimensions(phantom.ndim)

        compiled = SequenceCompiler().compile(
            program,
            checkpoints_s=checkpoints_s,
            simulation_timestep_s=simulation_timestep_s,
            status_callback=status_callback,
        )
        if status_callback is not None:
            status_callback(
                f"Compiled {compiled.n_intervals:,} intervals and "
                f"{compiled.adc_times_s.size:,} ADC samples."
            )
        props = phantom.get_active_properties()
        n_active = int(props["indices"].size)
        n_checkpoints = int(compiled.checkpoint_times_s.size)
        n_adc = int(compiled.adc_times_s.size)
        n_rx_coils = int(props["rx_sensitivities"].shape[0])
        spins_per_voxel = sampling.spins_per_voxel
        n_simulated_spins = n_active * spins_per_voxel
        native_threads = self.num_threads if self.use_parallel else 1
        selected_kernel = (
            self.sequence_kernel if sequence_kernel is None else sequence_kernel
        )
        if selected_kernel not in {"optimized", "reference"}:
            raise ValueError("sequence_kernel must be 'optimized' or 'reference'")
        if n_active == 0:
            raise ValueError("phantom has no active voxels")
        if signal_weighting not in {"voxel", "voxel_volume"}:
            raise ValueError("signal_weighting must be 'voxel' or 'voxel_volume'")
        checkpoint_dtype = np.dtype(
            np.float64 if checkpoint_dtype is None else checkpoint_dtype
        )
        if checkpoint_dtype not in {
            np.dtype(np.float16),
            np.dtype(np.float32),
            np.dtype(np.float64),
        }:
            raise ValueError("checkpoint_dtype must be float16, float32, or float64")

        arrays_to_validate = {
            "T1": props["t1"],
            "T2": props["t2"],
            "proton density": props["pd"],
            "off-resonance": props["df"],
            "Tx sensitivity": props["tx_sensitivity"],
            "Rx sensitivities": props["rx_sensitivities"],
            "positions": props["positions"],
            "initial magnetization": props["m0"],
        }
        for name, values in arrays_to_validate.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"active phantom {name} contains NaN or infinite values"
                )
        if np.any(props["t1"] <= 0) or np.any(props["t2"] <= 0):
            raise ValueError("active phantom voxels require T1 > 0 and T2 > 0")
        if np.any(props["pd"] < 0):
            raise ValueError("active phantom proton density must not be negative")

        if chunk_voxels is None:
            chunk_voxels = min(n_active, max(1, 65536 // spins_per_voxel))
        if int(chunk_voxels) != chunk_voxels or chunk_voxels <= 0:
            raise ValueError("chunk_voxels must be a positive integer")
        chunk_voxels = min(int(chunk_voxels), n_active)
        chunk_spins = chunk_voxels * spins_per_voxel
        if chunk_spins > np.iinfo(np.int32).max:
            raise ValueError(
                "one voxel chunk contains too many subvoxel spins for the native kernel"
            )

        # Final/checkpoint reconstructions dominate persistent output. Thread-local
        # ADC accumulators and one native chunk are the main working allocations.
        persistent_bytes = (
            phantom.nvoxels * 3 * 8
            + n_checkpoints * phantom.nvoxels * 3 * checkpoint_dtype.itemsize
            + n_rx_coils * n_adc * 16
            + n_active * (3 * 8 + 7 * 8 + 16)
            + n_rx_coils * n_active * 16
        )
        if sampling.enabled:
            # Expanded inputs, Cython contiguous component copies, native final
            # states, and sparse checkpoint output exist only for one parent-
            # voxel chunk. This intentionally overestimates small temporaries.
            expanded_bytes_per_spin = 184 + 48 * n_rx_coils
            working_bytes = (
                native_threads * n_rx_coils * max(1, n_adc) * 2 * 8
                + chunk_spins * (expanded_bytes_per_spin + n_checkpoints * 3 * 8)
                + compiled.n_intervals * 6 * 8
                + chunk_spins * 4
                + min(chunk_spins, 64) * compiled.n_intervals * 2 * 8
            )
        else:
            working_bytes = (
                native_threads * n_rx_coils * max(1, n_adc) * 2 * 8
                + chunk_voxels * (3 * 8 + n_checkpoints * 3 * 8)
                + compiled.n_intervals * 6 * 8
                + chunk_voxels * 4
                + min(chunk_voxels, 64) * compiled.n_intervals * 2 * 8
            )
        enforce_memory_budget(
            persistent_bytes + working_bytes,
            self._memory_budget(),
            description=(
                f"Streaming sequence with {compiled.n_intervals:,} intervals, "
                f"{n_active:,} active voxels, {n_simulated_spins:,} spins, "
                f"{n_adc:,} ADC samples, and "
                f"{n_checkpoints:,} checkpoints"
            ),
            suggestions=(
                "remove checkpoints, reduce object resolution or ADC samples, "
                "or choose a smaller voxel chunk"
            ),
        )

        try:
            from .blochsimulator_cy import simulate_sequence_chunk
        except ImportError as exc:
            raise ImportError(
                "blochsimulator_cy does not contain the streaming sequence kernel; "
                "rebuild the extension"
            ) from exc

        coil_signal = np.zeros((n_rx_coils, n_adc), dtype=np.complex128)
        final_active = np.empty((n_active, 3), dtype=np.float64)
        checkpoints_active = np.empty(
            (n_checkpoints, n_active, 3), dtype=checkpoint_dtype
        )
        positions = np.ascontiguousarray(props["positions"], dtype=np.float64)
        t1 = np.ascontiguousarray(props["t1"], dtype=np.float64)
        t2 = np.ascontiguousarray(props["t2"], dtype=np.float64)
        df = np.ascontiguousarray(props["df"], dtype=np.float64)
        pd = np.ascontiguousarray(props["pd"], dtype=np.float64)
        if signal_weighting == "voxel_volume":
            pd = pd * phantom.voxel_volume_m3
        tx_sensitivity = np.ascontiguousarray(
            props["tx_sensitivity"], dtype=np.complex128
        )
        rx_sensitivities = np.ascontiguousarray(
            props["rx_sensitivities"], dtype=np.complex128
        )
        m0 = np.ascontiguousarray(props["m0"], dtype=np.float64)
        subvoxel_offsets_m, subvoxel_weights = sampling.offsets_m(
            phantom_voxel_basis_m(phantom)
        )
        crush_state_indices = (
            compiled.transverse_crush_state_indices
            if spoiler_mode == "ideal"
            else np.zeros(0, dtype=np.int32)
        )

        chunks = (n_active + chunk_voxels - 1) // chunk_voxels
        if status_callback is not None:
            status_callback(
                f"Starting {compiled.n_intervals * n_simulated_spins:,} "
                f"spin-interval "
                f"updates in {chunks:,} chunk(s) on {native_threads} thread(s)."
            )
        for chunk_index, start in enumerate(range(0, n_active, chunk_voxels)):
            if cancel_callback is not None and cancel_callback():
                raise RuntimeError("sequence simulation cancelled")
            end = min(start + chunk_voxels, n_active)
            if status_callback is not None:
                status_callback(
                    f"Simulating chunk {chunk_index + 1}/{chunks} "
                    f"({end - start:,} active voxels)…"
                )
            parent_count = end - start
            if sampling.enabled:
                chunk_positions = np.repeat(
                    positions[start:end], spins_per_voxel, axis=0
                ) + np.tile(subvoxel_offsets_m, (parent_count, 1))
                chunk_t1 = np.repeat(t1[start:end], spins_per_voxel)
                chunk_t2 = np.repeat(t2[start:end], spins_per_voxel)
                chunk_df = np.repeat(df[start:end], spins_per_voxel)
                chunk_pd = np.repeat(pd[start:end], spins_per_voxel) * np.tile(
                    subvoxel_weights, parent_count
                )
                chunk_tx = np.repeat(tx_sensitivity[start:end], spins_per_voxel)
                chunk_rx = np.repeat(
                    rx_sensitivities[:, start:end], spins_per_voxel, axis=1
                )
                chunk_m0 = np.repeat(m0[start:end], spins_per_voxel, axis=0)
            else:
                chunk_positions = positions[start:end]
                chunk_t1 = t1[start:end]
                chunk_t2 = t2[start:end]
                chunk_df = df[start:end]
                chunk_pd = pd[start:end]
                chunk_tx = tx_sensitivity[start:end]
                chunk_rx = rx_sensitivities[:, start:end]
                chunk_m0 = m0[start:end]

            chunk_signal, chunk_final, chunk_checkpoints = simulate_sequence_chunk(
                compiled.rf_hz,
                compiled.gradient_hz_per_m,
                compiled.dt_s,
                chunk_t1,
                chunk_t2,
                chunk_df,
                chunk_positions,
                chunk_pd,
                chunk_tx,
                chunk_rx,
                chunk_m0,
                compiled.adc_state_indices,
                compiled.adc_demodulation,
                compiled.checkpoint_state_indices,
                crush_state_indices,
                native_threads,
                selected_kernel,
            )
            coil_signal += chunk_signal
            if sampling.enabled:
                final_active[start:end] = np.einsum(
                    "s,vsd->vd",
                    subvoxel_weights,
                    chunk_final.reshape(parent_count, spins_per_voxel, 3),
                    optimize=True,
                )
                if n_checkpoints:
                    checkpoints_active[:, start:end] = np.einsum(
                        "s,cvsd->cvd",
                        subvoxel_weights,
                        chunk_checkpoints.reshape(
                            n_checkpoints, parent_count, spins_per_voxel, 3
                        ),
                        optimize=True,
                    )
            else:
                final_active[start:end] = chunk_final
                if n_checkpoints:
                    checkpoints_active[:, start:end] = chunk_checkpoints
            if progress_callback is not None:
                progress_callback(chunk_index + 1, chunks)
            if preview_callback is not None:
                partial_signal = coil_signal[0] if n_rx_coils == 1 else coil_signal
                preview_callback(
                    (chunk_index + 1) / chunks,
                    np.array(partial_signal, copy=True),
                )

        active_indices = props["indices"]
        final_flat = np.zeros((phantom.nvoxels, 3), dtype=np.float64)
        final_flat[active_indices] = final_active
        final_magnetization = final_flat.reshape(phantom.shape + (3,))
        checkpoint_magnetization = None
        if n_checkpoints:
            checkpoint_flat = np.zeros(
                (n_checkpoints, phantom.nvoxels, 3), dtype=checkpoint_dtype
            )
            checkpoint_flat[:, active_indices] = checkpoints_active
            checkpoint_magnetization = checkpoint_flat.reshape(
                (n_checkpoints,) + phantom.shape + (3,)
            )

        signal = coil_signal[0] if n_rx_coils == 1 else coil_signal
        from .sequence import (
            AcquisitionDimensions,
            physical_b1_field_arrays,
            physical_sequence_waveforms,
        )
        from .sequence.acquisition import (
            CartesianAcquisitionFrames,
            infer_cartesian_acquisition,
            infer_cartesian_acquisition_frames,
            infer_cartesian_acquisition_volumes,
            infer_spectroscopic_acquisition,
            infer_spiral_acquisition,
        )

        acquisition_dimensions = AcquisitionDimensions.from_program(program)
        spectroscopic_metadata = program.metadata.get("spectroscopic_acquisition")
        if spectroscopic_metadata is None:
            try:
                spectroscopic_metadata = infer_spectroscopic_acquisition(
                    program, compiled=compiled
                ).to_metadata()
            except ValueError:
                spectroscopic_metadata = None
        cartesian_metadata = program.metadata.get("cartesian_acquisition")
        spiral_metadata = program.metadata.get("spiral_acquisition")
        if spiral_metadata is None:
            try:
                spiral_metadata = infer_spiral_acquisition(
                    program, compiled=compiled
                ).to_metadata()
            except ValueError:
                spiral_metadata = None
        if cartesian_metadata is None:
            program_acquisition = program.metadata.get("acquisition")
            if (
                isinstance(program_acquisition, dict)
                and program_acquisition.get("type") == "cartesian_2d"
            ):
                cartesian_metadata = program_acquisition
        cartesian_frame_metadata = program.metadata.get("cartesian_acquisition_frames")
        cartesian_volume_metadata = program.metadata.get(
            "cartesian_acquisition_volumes"
        )
        if (
            spectroscopic_metadata is None
            and spiral_metadata is None
            and cartesian_metadata is None
        ):
            try:
                cartesian_metadata = infer_cartesian_acquisition(
                    program, compiled=compiled
                ).to_metadata()
            except ValueError:
                if cartesian_frame_metadata is None:
                    try:
                        cartesian_frame_metadata = infer_cartesian_acquisition_frames(
                            program, compiled=compiled
                        ).to_metadata()
                    except ValueError:
                        cartesian_frame_metadata = None
        if (
            spectroscopic_metadata is None
            and cartesian_metadata is None
            and cartesian_frame_metadata is not None
            and cartesian_volume_metadata is None
        ):
            try:
                cartesian_volume_metadata = infer_cartesian_acquisition_volumes(
                    program,
                    compiled=compiled,
                    frames=CartesianAcquisitionFrames.from_metadata(
                        cartesian_frame_metadata
                    ),
                ).to_metadata()
            except ValueError:
                cartesian_volume_metadata = None
        definitions = dict(program.metadata.get("definitions", {}))
        physical_nucleus = str(
            phantom.metadata.get("nucleus") or definitions.get("nucleus") or "H1"
        )
        from .units import NUCLEUS_GAMMA_HZ_PER_T

        if physical_nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
            physical_nucleus = "H1"
        sequence_waveforms = physical_sequence_waveforms(program, physical_nucleus)
        physical_field_maps = physical_b1_field_arrays(phantom, sequence_waveforms)
        result = SequenceSimulationResult(
            signal=signal,
            adc_times_s=compiled.adc_times_s,
            final_magnetization=final_magnetization,
            checkpoint_magnetization=checkpoint_magnetization,
            checkpoint_times_s=compiled.checkpoint_times_s,
            adc_gradient_moment_cyc_per_m=(compiled.adc_gradient_moment_cyc_per_m),
            metadata={
                "sequence_source": program.source,
                "sequence_version": program.version,
                "duration_s": program.duration_s,
                "n_intervals": compiled.n_intervals,
                "n_active_voxels": n_active,
                "n_simulated_spins": n_simulated_spins,
                "spin_sampling": sampling.to_metadata(),
                "subvoxel_spin_counts_xyz": sampling.counts_xyz,
                "subvoxel_spins_per_voxel": spins_per_voxel,
                "n_rx_coils": n_rx_coils,
                "chunk_voxels": chunk_voxels,
                "chunk_spins": chunk_voxels * spins_per_voxel,
                "simulation_timestep_s": simulation_timestep_s,
                "signal_weighting": signal_weighting,
                "sequence_kernel": selected_kernel,
                "phantom_metadata": dict(phantom.metadata),
                "field_strength_t": phantom.metadata.get("field_strength_t"),
                "nucleus": phantom.metadata.get("nucleus"),
                "voxel_volume_m3": (
                    phantom.voxel_volume_m3
                    if signal_weighting == "voxel_volume"
                    else None
                ),
                "acquisition_dimensions": acquisition_dimensions.to_metadata(),
                "spectroscopic_acquisition": spectroscopic_metadata,
                "cartesian_acquisition": cartesian_metadata,
                "cartesian_acquisition_frames": cartesian_frame_metadata,
                "cartesian_acquisition_volumes": cartesian_volume_metadata,
                "spiral_acquisition": spiral_metadata,
                "sequence_definitions": definitions,
                "physical_waveform_nucleus": physical_nucleus,
                "physical_rf_unit": "G",
                "physical_gradient_unit": "T/m",
                "spoiler_mode": spoiler_mode,
                "ideal_spoiling_applied": bool(
                    spoiler_mode == "ideal" and compiled.transverse_crush_times_s.size
                ),
                "ideal_spoiler_end_times_s": (
                    compiled.transverse_crush_times_s.tolist()
                    if spoiler_mode == "ideal"
                    else []
                ),
                "declared_ideal_spoiler_end_times_s": (
                    compiled.transverse_crush_times_s.tolist()
                ),
                "units": {
                    "time": "s",
                    "position": "m",
                    "rf": "Hz",
                    "gradient": "Hz/m",
                    "off_resonance": "Hz",
                    "tx_sensitivity": "dimensionless complex B1+ scale",
                    "rx_sensitivity": "dimensionless complex receive scale",
                },
            },
            sequence_waveforms=sequence_waveforms,
            physical_field_maps=physical_field_maps,
        )
        self.last_result = result.to_dict()
        self.last_result["phantom"] = phantom
        self.last_result["program"] = program
        return result

    def plot_magnetization(
        self, component: str = "all", position_idx: int = 0, freq_idx: int = 0
    ):
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
        time = result["time"]

        if len(result["mx"].shape) == 2:
            # Single time point
            print("Single time point result - no time evolution to plot")
            return

        # Extract data for specific position and frequency
        mx = result["mx"][:, position_idx, freq_idx]
        my = result["my"][:, position_idx, freq_idx]
        mz = result["mz"][:, position_idx, freq_idx]
        self.log_message(f"result = {result}")  # Debugging line

        import matplotlib.pyplot as plt

        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Mx
        axes[0, 0].plot(time * 1000, mx)
        axes[0, 0].set_xlabel("Time (ms)")
        axes[0, 0].set_ylabel("Mx")
        axes[0, 0].grid(True)
        axes[0, 0].set_title("Transverse Magnetization (x)")

        # My
        axes[0, 1].plot(time * 1000, my)
        axes[0, 1].set_xlabel("Time (ms)")
        axes[0, 1].set_ylabel("My")
        axes[0, 1].grid(True)
        axes[0, 1].set_title("Transverse Magnetization (y)")

        # Mz
        axes[1, 0].plot(time * 1000, mz)
        axes[1, 0].set_xlabel("Time (ms)")
        axes[1, 0].set_ylabel("Mz")
        axes[1, 0].grid(True)
        axes[1, 0].set_title("Longitudinal Magnetization")

        # Magnitude
        magnitude = np.sqrt(mx**2 + my**2)
        axes[1, 1].plot(time * 1000, magnitude)
        axes[1, 1].set_xlabel("Time (ms)")
        axes[1, 1].set_ylabel("|Mxy|")
        axes[1, 1].grid(True)
        axes[1, 1].set_title("Transverse Magnitude")

        plt.suptitle(f'Magnetization Evolution\n{result["tissue"].name}')
        plt.tight_layout()
        plt.show()

    def get_results_as_xarray(self) -> "xr.Dataset":
        """
        Convert the last simulation result to an xarray Dataset.

        Returns
        -------
        xarray.Dataset
            Dataset containing magnetization and signal data with metadata attributes.
            Dimensions will be (time, position, frequency) or (position, frequency)
            depending on the simulation mode.
        """
        if self.last_result is None:
            raise ValueError("No simulation results available")

        result = self.last_result
        if "adc_times_s" in result:
            phantom = result.get("phantom")
            final = result["final_magnetization"]
            spatial_dims = (
                list(("x", "y", "z")[: phantom.ndim])
                if phantom is not None
                else [f"spatial_{index}" for index in range(final.ndim - 1)]
            )
            coords = {"adc": np.arange(len(result["adc_times_s"]))}
            coords["adc_time_s"] = ("adc", result["adc_times_s"])
            signal = np.asarray(result["signal"])
            signal_dims = ("adc",) if signal.ndim == 1 else ("coil", "adc")
            data_vars = {
                "signal": (signal_dims, signal),
                "mx": (spatial_dims, result["mx"]),
                "my": (spatial_dims, result["my"]),
                "mz": (spatial_dims, result["mz"]),
            }
            if signal.ndim == 2:
                coords["coil"] = np.arange(signal.shape[0])
            acquisition_metadata = result.get("metadata", {}).get(
                "acquisition_dimensions"
            )
            if acquisition_metadata is not None:
                from .sequence import AcquisitionDimensions

                acquisition_dimensions = AcquisitionDimensions.from_metadata(
                    acquisition_metadata
                )
                if acquisition_dimensions.num_samples != len(result["adc_times_s"]):
                    raise ValueError(
                        "acquisition dimension metadata does not match the ADC stream"
                    )
                for axis in acquisition_dimensions.AXIS_NAMES:
                    coords[f"{axis}_index"] = (
                        "adc",
                        acquisition_dimensions.sample_indices(axis),
                    )
            moments = result.get("adc_gradient_moment_cyc_per_m")
            if moments is not None:
                coords["gradient_axis"] = ["x", "y", "z"]
                data_vars["adc_gradient_moment_cyc_per_m"] = (
                    ("adc", "gradient_axis"),
                    moments,
                )
            checkpoints = result.get("checkpoint_magnetization")
            if checkpoints is not None:
                coords["checkpoint"] = result["checkpoint_times_s"]
                data_vars["checkpoint_magnetization"] = (
                    ["checkpoint"] + spatial_dims + ["component"],
                    checkpoints,
                )
                coords["component"] = ["mx", "my", "mz"]
            attrs = {
                "simulator_version": __version__,
                "export_timestamp": str(np.datetime64("now")),
                **{
                    key: value
                    for key, value in result.get("metadata", {}).items()
                    if isinstance(value, (str, int, float, bool))
                },
            }
            return xr.Dataset(data_vars, coords=coords, attrs=attrs)

        mx = result["mx"]
        my = result["my"]
        mz = result["mz"]
        signal = result["signal"]
        time = result["time"]

        # Prepare attributes
        attrs = {
            "simulator_version": __version__,
            "export_timestamp": str(np.datetime64("now")),
            "rf_carrier_offset_hz": float(result.get("rf_carrier_offset", 0.0)),
        }
        if "tissue" in result:
            tissue = result["tissue"]
            attrs.update(
                {
                    "tissue_name": tissue.name,
                    "t1": tissue.t1,
                    "t2": tissue.t2,
                    "density": tissue.density,
                }
            )
            if tissue.t2_star is not None:
                attrs["t2_star"] = tissue.t2_star

        # Handle Phantom Simulation
        if "phantom" in result:
            phantom = result["phantom"]

            # Determine dimensions
            # mx shape is either (*phantom.shape) or (ntime, *phantom.shape)
            phantom_dim_names = ["z", "y", "x"][
                -phantom.ndim :
            ]  # e.g. ['y', 'x'] for 2D

            dims = []
            coords = {}

            # Check if time dimension exists
            # We assume if mx has one more dim than phantom, the first one is time
            if mx.ndim == phantom.ndim + 1:
                dims.append("time")
                coords["time"] = time

            dims.extend(phantom_dim_names)

            # Create Dataset
            data_vars = {
                "mx": (dims, mx),
                "my": (dims, my),
                "mz": (dims, mz),
                "signal": (dims, signal),
            }

            if "pd_weighted_signal" in result:
                data_vars["pd_weighted_signal"] = (dims, result["pd_weighted_signal"])

            # Add spatial coords if available in phantom?
            # Phantom usually implies a grid, so indices are sufficient or we could add real-world coords if phantom has FOV.
            # Assuming basic indices for now.

            ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
            return ds

        # Handle Point Simulation
        positions = result["positions"]  # (N_pos, 3)
        frequencies = result["frequencies"]  # (N_freq,)

        n_pos = positions.shape[0]
        n_freq = frequencies.shape[0]
        n_time = len(time)

        # Logic to determine dimensions based on shape
        # Expected max shape: (time, position, frequency)

        dims = []
        coords = {}

        if mx.ndim == 3:
            # Full (time, pos, freq)
            dims = ["time", "position", "frequency"]
        elif mx.ndim == 2:
            # (pos, freq) or (time, pos) or (time, freq)?
            # If endpoint mode (time not returned in shape), usually (pos, freq)
            # Check against sizes
            if mx.shape == (n_pos, n_freq):
                dims = ["position", "frequency"]
            elif mx.shape == (n_time, n_pos):  # Single freq
                dims = ["time", "position"]
            elif mx.shape == (n_time, n_freq):  # Single pos
                dims = ["time", "frequency"]
            else:
                # Fallback / Ambiguous
                # Assume (pos, freq) if n_time is arguably 1 (endpoint)
                dims = ["dim_0", "dim_1"]
        elif mx.ndim == 1:
            if n_pos > 1 and mx.shape[0] == n_pos:
                dims = ["position"]
            elif n_freq > 1 and mx.shape[0] == n_freq:
                dims = ["frequency"]
            elif n_time > 1 and mx.shape[0] == n_time:
                dims = ["time"]
            else:
                dims = ["dim_0"]
        else:
            # Scalar?
            dims = []

        # Populate coordinates
        if "time" in dims:
            coords["time"] = time
        if "position" in dims:
            coords["position"] = np.arange(n_pos)
            # Add spatial coordinates
            coords["x"] = ("position", positions[:, 0])
            coords["y"] = ("position", positions[:, 1])
            coords["z"] = ("position", positions[:, 2])
        if "frequency" in dims:
            coords["frequency"] = frequencies
            coords["effective_frequency"] = (
                "frequency",
                result.get("effective_frequencies", frequencies),
            )

        ds = xr.Dataset(
            {
                "mx": (dims, mx),
                "my": (dims, my),
                "mz": (dims, mz),
                "signal": (dims, signal),
            },
            coords=coords,
            attrs=attrs,
        )
        return ds

    def save_results(
        self,
        filename: str,
        sequence_params: Optional[Dict] = None,
        simulation_params: Optional[Dict] = None,
    ):
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

        with h5py.File(filename, "w") as f:
            if "adc_times_s" in self.last_result:
                result = self.last_result
                f.create_dataset("signal", data=result["signal"])
                f.create_dataset("adc_times_s", data=result["adc_times_s"])
                if result.get("adc_gradient_moment_cyc_per_m") is not None:
                    f.create_dataset(
                        "adc_gradient_moment_cyc_per_m",
                        data=result["adc_gradient_moment_cyc_per_m"],
                    )
                f.create_dataset(
                    "final_magnetization", data=result["final_magnetization"]
                )
                f.create_dataset("mx", data=result["mx"])
                f.create_dataset("my", data=result["my"])
                f.create_dataset("mz", data=result["mz"])
                if result.get("checkpoint_magnetization") is not None:
                    f.create_dataset(
                        "checkpoint_magnetization",
                        data=result["checkpoint_magnetization"],
                    )
                f.create_dataset(
                    "checkpoint_times_s", data=result["checkpoint_times_s"]
                )
                f.attrs["metadata_json"] = json.dumps(
                    result.get("metadata", {}), default=str
                )
                f.attrs["export_timestamp"] = str(np.datetime64("now"))
                f.attrs["simulator_version"] = __version__
                return

            # Save magnetization data
            f.create_dataset("mx", data=self.last_result["mx"])
            f.create_dataset("my", data=self.last_result["my"])
            f.create_dataset("mz", data=self.last_result["mz"])
            f.create_dataset("signal", data=self.last_result["signal"])

            # Save parameters
            f.create_dataset("time", data=self.last_result["time"])
            f.create_dataset("positions", data=self.last_result["positions"])
            f.create_dataset("frequencies", data=self.last_result["frequencies"])
            f.create_dataset(
                "effective_frequencies",
                data=self.last_result.get(
                    "effective_frequencies", self.last_result["frequencies"]
                ),
            )
            f.attrs["rf_carrier_offset_hz"] = float(
                self.last_result.get("rf_carrier_offset", 0.0)
            )

            # Save tissue parameters
            tissue_group = f.create_group("tissue")
            tissue = self.last_result["tissue"]
            tissue_group.attrs["name"] = tissue.name
            tissue_group.attrs["t1"] = tissue.t1
            tissue_group.attrs["t2"] = tissue.t2
            tissue_group.attrs["density"] = tissue.density
            if tissue.t2_star is not None:
                tissue_group.attrs["t2_star"] = tissue.t2_star

            # Save pulse sequence parameters if provided
            if sequence_params is not None:
                seq_group = f.create_group("sequence_parameters")
                for key, value in sequence_params.items():
                    if value is not None:
                        if isinstance(value, (np.ndarray, list, tuple)):
                            try:
                                seq_group.create_dataset(key, data=value)
                            except (TypeError, ValueError):
                                # Fallback for object arrays or other incompatible types
                                try:
                                    # Try converting to string array
                                    str_data = np.asarray(value).astype(str)
                                    seq_group.create_dataset(key, data=str_data)
                                except Exception:
                                    # Last resort: JSON string
                                    seq_group.attrs[key] = json.dumps(value)
                        elif isinstance(value, dict):
                            seq_group.attrs[key] = json.dumps(value)
                        else:
                            try:
                                seq_group.attrs[key] = value
                            except Exception:
                                seq_group.attrs[key] = str(value)

            # Save simulation parameters if provided
            if simulation_params is not None:
                sim_group = f.create_group("simulation_parameters")
                for key, value in simulation_params.items():
                    if value is not None:
                        if isinstance(value, (np.ndarray, list, tuple)):
                            try:
                                sim_group.create_dataset(key, data=value)
                            except (TypeError, ValueError):
                                try:
                                    str_data = np.asarray(value).astype(str)
                                    sim_group.create_dataset(key, data=str_data)
                                except Exception:
                                    sim_group.attrs[key] = json.dumps(value)
                        elif isinstance(value, dict):
                            sim_group.attrs[key] = json.dumps(value)
                        else:
                            try:
                                sim_group.attrs[key] = value
                            except Exception:
                                sim_group.attrs[key] = str(value)

            # Add metadata
            f.attrs["export_timestamp"] = str(np.datetime64("now"))
            f.attrs["simulator_version"] = __version__

    def save_parameters_json(
        self,
        filename: str,
        sequence_params: Optional[Dict] = None,
        simulation_params: Optional[Dict] = None,
        include_waveforms: bool = False,
    ):
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
            "metadata": {
                "export_timestamp": str(np.datetime64("now")),
                "simulator_version": __version__,
            },
            "tissue_parameters": {
                "name": self.last_result["tissue"].name,
                "t1": float(self.last_result["tissue"].t1),
                "t2": float(self.last_result["tissue"].t2),
                "density": float(self.last_result["tissue"].density),
                "t2_star": (
                    float(self.last_result["tissue"].t2_star)
                    if self.last_result["tissue"].t2_star
                    else None
                ),
            },
            "positions": self.last_result["positions"].tolist(),
            "frequencies": self.last_result["frequencies"].tolist(),
            "effective_frequencies": self.last_result.get(
                "effective_frequencies", self.last_result["frequencies"]
            ).tolist(),
            "rf_carrier_offset_hz": float(
                self.last_result.get("rf_carrier_offset", 0.0)
            ),
            "time_points": int(len(self.last_result["time"])),
            "duration": (
                float(self.last_result["time"][-1])
                if len(self.last_result["time"]) > 0
                else 0.0
            ),
        }

        # Add sequence parameters
        if sequence_params is not None:
            params_dict["sequence_parameters"] = {}
            for key, value in sequence_params.items():
                if value is not None:
                    if isinstance(value, np.ndarray):
                        if include_waveforms:
                            params_dict["sequence_parameters"][key] = value.tolist()
                        else:
                            params_dict["sequence_parameters"][
                                key
                            ] = f"<array shape={value.shape}>"
                    elif isinstance(value, (list, tuple)):
                        params_dict["sequence_parameters"][key] = list(value)
                    else:
                        params_dict["sequence_parameters"][key] = value

        # Add simulation parameters
        if simulation_params is not None:
            params_dict["simulation_parameters"] = {}
            for key, value in simulation_params.items():
                if value is not None:
                    if isinstance(value, np.ndarray):
                        params_dict["simulation_parameters"][key] = value.tolist()
                    elif isinstance(value, (list, tuple)):
                        params_dict["simulation_parameters"][key] = list(value)
                    else:
                        params_dict["simulation_parameters"][key] = value

        # Write to file
        with open(filename, "w") as f:
            json.dump(params_dict, f, indent=2)

    def load_results(self, filename: str):
        """Load simulation results from HDF5 file."""
        with h5py.File(filename, "r") as f:
            if "adc_times_s" in f:
                final = f["final_magnetization"][...]
                metadata = json.loads(f.attrs.get("metadata_json", "{}"))
                self.last_result = {
                    "signal": f["signal"][...],
                    "adc_times_s": f["adc_times_s"][...],
                    "adc_gradient_moment_cyc_per_m": (
                        f["adc_gradient_moment_cyc_per_m"][...]
                        if "adc_gradient_moment_cyc_per_m" in f
                        else None
                    ),
                    "final_magnetization": final,
                    "mx": final[..., 0],
                    "my": final[..., 1],
                    "mz": final[..., 2],
                    "checkpoint_magnetization": (
                        f["checkpoint_magnetization"][...]
                        if "checkpoint_magnetization" in f
                        else None
                    ),
                    "checkpoint_times_s": f["checkpoint_times_s"][...],
                    "metadata": metadata,
                }
                return
            frequencies = f["frequencies"][...]
            rf_carrier_offset = float(f.attrs.get("rf_carrier_offset_hz", 0.0))
            self.last_result = {
                "mx": f["mx"][...],
                "my": f["my"][...],
                "mz": f["mz"][...],
                "signal": f["signal"][...],
                "time": f["time"][...],
                "positions": f["positions"][...],
                "frequencies": frequencies,
                "effective_frequencies": (
                    f["effective_frequencies"][...]
                    if "effective_frequencies" in f
                    else frequencies - rf_carrier_offset
                ),
                "rf_carrier_offset": rf_carrier_offset,
                "tissue": TissueParameters(
                    name=f["tissue"].attrs["name"],
                    t1=f["tissue"].attrs["t1"],
                    t2=f["tissue"].attrs["t2"],
                    density=f["tissue"].attrs["density"],
                ),
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
        mode=2,  # Time-resolved
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
