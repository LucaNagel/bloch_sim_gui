"""
pulse_loader.py - RF Pulse loading and processing pipeline

This module provides utilities to load and process RF pulses from various formats,
including Bruker JCAMP-DX format (.exc files).

Author: Your Name
Date: 2024
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
from dataclasses import dataclass
import re


@dataclass
class PulseMetadata:
    """Container for RF pulse metadata."""

    # Basic properties
    name: str = ""
    title: str = ""
    origin: str = ""

    # Pulse parameters
    flip_angle: float = 90.0  # degrees
    duration: float = 0.0  # seconds
    npoints: int = 0
    nucleus: str = "Proton"
    field_strength: float = 3.0  # Tesla

    # B1 properties
    max_b1: float = 0.0  # Tesla or uT
    shape_type: str = "conventional"
    shape_mode: str = "Excitation"

    # Gradient/slice properties
    max_grad: float = 0.0  # mT/m
    max_slew: float = 0.0  # T/m/s
    slice_width: float = 0.0  # cm
    slice_offset: float = 0.0  # mm

    # Integration/rephasing factors
    integfac: float = 1.0
    rephfac: float = 1.0
    bwfac: float = 1.0

    # Raw JCAMP metadata
    minx: float = 0.0
    maxx: float = 100.0
    miny: float = 0.0
    maxy: float = 360.0

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'title': self.title,
            'origin': self.origin,
            'flip_angle': self.flip_angle,
            'duration': self.duration,
            'npoints': self.npoints,
            'nucleus': self.nucleus,
            'field_strength': self.field_strength,
            'max_b1': self.max_b1,
            'shape_type': self.shape_type,
            'shape_mode': self.shape_mode,
            'max_grad': self.max_grad,
            'max_slew': self.max_slew,
            'slice_width': self.slice_width,
            'slice_offset': self.slice_offset,
        }


class JCAMPPulseLoader:
    """Load RF pulses from Bruker JCAMP-DX format (.exc files)."""

    @staticmethod
    def load(filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, PulseMetadata]:
        """
        Load an RF pulse from a JCAMP-DX file.

        Parameters
        ----------
        filepath : str or Path
            Path to the .exc file

        Returns
        -------
        b1 : ndarray
            Complex B1 field (Gauss)
        time : ndarray
            Time array (seconds)
        metadata : PulseMetadata
            Metadata extracted from the file
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Pulse file not found: {filepath}")

        # Read file
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse metadata and data
        metadata = PulseMetadata()
        metadata.name = filepath.stem

        # Parse header
        data_section_start = JCAMPPulseLoader._parse_header(lines, metadata)

        # Parse XY data points
        xy_data = JCAMPPulseLoader._parse_xy_data(lines[data_section_start:])

        # Convert to B1 and time
        b1, time = JCAMPPulseLoader._xy_to_b1(xy_data, metadata)

        return b1, time, metadata

    @staticmethod
    def _parse_header(lines: list, metadata: PulseMetadata) -> int:
        """Parse JCAMP-DX header and populate metadata."""
        data_start = 0

        for i, line in enumerate(lines):
            line = line.strip()

            if not line or line.startswith('##'):
                # Parse header line
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Map JCAMP keys to metadata
                    if key == '##TITLE':
                        metadata.title = value
                    elif key == '##ORIGIN':
                        metadata.origin = value
                    elif key == '##$SHAPE_TOTROT':
                        metadata.flip_angle = float(value)
                    elif key == '##$SHAPE_EXMODE':
                        metadata.shape_mode = value
                    elif key == '##$SHAPE_TYPE':
                        metadata.shape_type = value
                    elif key == '##NPOINTS':
                        metadata.npoints = int(value)
                    elif key == '##DURATION':
                        # Parse duration (e.g., "2.47200e+01 ms")
                        parts = value.split()
                        duration_val = float(parts[0])
                        if len(parts) > 1:
                            unit = parts[1].lower()
                            if unit == 'ms':
                                duration_val /= 1000.0
                            elif unit == 'us':
                                duration_val /= 1e6
                        metadata.duration = duration_val
                    elif key == '##NUCLEUS':
                        metadata.nucleus = value
                    elif key == '##FIELD':
                        # Parse field strength (e.g., "3.00000e+00 T")
                        parts = value.split()
                        metadata.field_strength = float(parts[0])
                    elif key == '##MAXB1':
                        # Parse max B1 (e.g., "8.42971e+00 uT")
                        parts = value.split()
                        b1_val = float(parts[0])
                        if len(parts) > 1:
                            unit = parts[1].lower()
                            if unit == 'ut':
                                # Convert microTesla to Gauss (1 uT = 0.01 Gauss = 1e-5 T)
                                b1_val = b1_val * 0.01  # uT to Gauss
                            elif unit == 'mt':
                                # Convert milliTesla to Gauss (1 mT = 10 Gauss)
                                b1_val = b1_val * 10.0
                            elif unit == 't':
                                # Convert Tesla to Gauss (1 T = 10000 Gauss)
                                b1_val = b1_val * 10000.0
                        metadata.max_b1 = b1_val
                    elif key == '##MAXGRAD':
                        # Parse max gradient (e.g., "5.00000e+01 mT/m")
                        parts = value.split()
                        metadata.max_grad = float(parts[0])
                    elif key == '##MAXSLEW':
                        # Parse max slew rate
                        parts = value.split()
                        metadata.max_slew = float(parts[0])
                    elif key == '##SLICEWIDTH':
                        # Parse slice width (e.g., "1.00000e+00 cm")
                        parts = value.split()
                        metadata.slice_width = float(parts[0])
                    elif key == '##SLICEOFFSET':
                        # Parse slice offset
                        parts = value.split()
                        metadata.slice_offset = float(parts[0])
                    elif key == '##$SHAPE_INTEGFAC':
                        metadata.integfac = float(value)
                    elif key == '##$SHAPE_REPHFAC':
                        metadata.rephfac = float(value)
                    elif key == '##$SHAPE_BWFAC':
                        metadata.bwfac = float(value)
                    elif key == '##MINX':
                        metadata.minx = float(value)
                    elif key == '##MAXX':
                        metadata.maxx = float(value)
                    elif key == '##MINY':
                        metadata.miny = float(value)
                    elif key == '##MAXY':
                        metadata.maxy = float(value)

            # Check if we've reached the data section
            if '##XYPOINTS=' in line:
                data_start = i + 1
                break

        return data_start

    @staticmethod
    def _parse_xy_data(lines: list) -> Tuple[np.ndarray, np.ndarray]:
        """Parse XY data points from JCAMP file."""
        amp_list = []
        phase_list = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('##'):
                continue

            # Parse comma-separated values
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    amp = float(parts[0].strip())
                    phase = float(parts[1].strip())
                    amp_list.append(amp)
                    phase_list.append(phase)
                except ValueError:
                    # Skip lines that can't be parsed
                    continue

        return np.array(amp_list), np.array(phase_list)

    @staticmethod
    def _xy_to_b1(xy_data: Tuple[np.ndarray, np.ndarray],
                   metadata: PulseMetadata) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert XY data to B1 waveform and time array.

        In Bruker JCAMP format:
        - X (amplitude) is in percent (0-100) of maximum
        - Y (phase) is in degrees (0-360)

        Returns
        -------
        b1 : ndarray
            Complex B1 field in Gauss
        time : ndarray
            Time array in seconds
        """
        amp_percent, phase_deg = xy_data

        # Normalize amplitude to [0, 1]
        amp_norm = amp_percent / 100.0

        # Scale to maximum B1 (convert from Gauss to Gauss - already in Gauss)
        amp_gauss = amp_norm * metadata.max_b1

        # Convert phase from degrees to radians
        phase_rad = np.deg2rad(phase_deg)

        # Create complex B1
        b1 = amp_gauss * np.exp(1j * phase_rad)

        # Create time array
        if metadata.duration > 0 and metadata.npoints > 0:
            time = np.linspace(0, metadata.duration, len(b1), endpoint=False)
        else:
            time = np.arange(len(b1))

        return b1.astype(np.complex128), time.astype(np.float64)


class PulseLibrary:
    """Manage a library of RF pulses."""

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize pulse library.

        Parameters
        ----------
        base_path : str or Path, optional
            Base directory for pulse files. If None, uses './rfpulses'
        """
        if base_path is None:
            base_path = Path(__file__).parent / 'rfpulses'
        else:
            base_path = Path(base_path)

        self.base_path = base_path
        self.cache = {}
        self._index = {}

        # Build index of available pulses
        if self.base_path.exists():
            self._build_index()

    def _build_index(self):
        """Build index of available pulse files."""
        for pulse_file in self.base_path.rglob('*'):
            if pulse_file.suffix.lower() not in {'.exc', '.dat'}:
                continue
            rel_path = pulse_file.relative_to(self.base_path)
            key = str(rel_path).replace('\\', '/')
            key = key[: -len(pulse_file.suffix)]  # strip extension
            self._index[key] = pulse_file

    def list_pulses(self) -> list:
        """List all available pulses."""
        return sorted(self._index.keys())

    def load(self, pulse_name: str, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, PulseMetadata]:
        """
        Load a pulse by name.

        Parameters
        ----------
        pulse_name : str
            Name of the pulse (without .exc extension)
            Can use '/' as path separator, e.g., 'bruker/13C_Ultimate_SPSP_Pulse_QuEMRT'
        use_cache : bool
            Whether to cache loaded pulses

        Returns
        -------
        b1 : ndarray
            Complex B1 field (Gauss)
        time : ndarray
            Time array (seconds)
        metadata : PulseMetadata
            Metadata from the pulse file
        """
        if use_cache and pulse_name in self.cache:
            return self.cache[pulse_name]

        if pulse_name not in self._index:
            available = '\n  '.join(self.list_pulses())
            raise ValueError(
                f"Pulse '{pulse_name}' not found.\n"
                f"Available pulses:\n  {available}"
            )

        filepath = self._index[pulse_name]
        suffix = filepath.suffix.lower()
        if suffix == '.exc':
            b1, time, metadata = JCAMPPulseLoader.load(filepath)
        elif suffix == '.dat':
            b1, time, metadata = load_amp_phase_dat(filepath)
        else:
            raise ValueError(f"Unsupported pulse format for {filepath}")

        if use_cache:
            self.cache[pulse_name] = (b1, time, metadata)

        return b1, time, metadata

    def load_from_file(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, PulseMetadata]:
        """Load a pulse directly from a file path."""
        return JCAMPPulseLoader.load(filepath)


# Global pulse library instance
_global_library = None

def get_pulse_library(base_path: Optional[Union[str, Path]] = None) -> PulseLibrary:
    """Get or create the global pulse library."""
    global _global_library
    if _global_library is None:
        _global_library = PulseLibrary(base_path)
    return _global_library


def load_pulse(pulse_name: str) -> Tuple[np.ndarray, np.ndarray, PulseMetadata]:
    """
    Load a pulse from the global library.

    Parameters
    ----------
    pulse_name : str
        Name of the pulse, e.g., 'bruker/13C_Ultimate_SPSP_Pulse_QuEMRT'

    Returns
    -------
    b1 : ndarray
        Complex B1 field (Gauss)
    time : ndarray
        Time array (seconds)
    metadata : PulseMetadata
        Metadata from the pulse file
    """
    lib = get_pulse_library()
    return lib.load(pulse_name)


def load_pulse_from_file(filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, PulseMetadata]:
    """Load a pulse directly from a file path."""
    filepath = Path(filepath)
    if filepath.suffix.lower() == '.dat':
        return load_amp_phase_dat(filepath)
    return JCAMPPulseLoader.load(filepath)


def load_amp_phase_dat(filepath: Union[str, Path],
                       duration_s: Optional[float] = None,
                       max_b1_gauss: Optional[float] = None,
                       amplitude_unit: str = "relative",
                       phase_unit: str = "deg",
                       layout: str = "columns") -> Tuple[np.ndarray, np.ndarray, PulseMetadata]:
    """
    Load a simple two-column .dat pulse where column 0 is relative amplitude (1.0 = 100%)
    and column 1 is phase in degrees.

    Parameters
    ----------
    filepath : str or Path
        Path to the .dat file.
    duration_s : float, optional
        Total pulse duration in seconds. If not provided, try to parse `<number>ms`
        from the filename (e.g., `AHP_5ms_Hsn.dat`). Fallback is index-based timing.
    max_b1_gauss : float, optional
        Peak B1 amplitude in Gauss corresponding to amplitude=1.0. Defaults to 1.0 G.
    amplitude_unit : str, optional
        One of {"relative", "percent", "fraction", "gauss", "mt", "ut"}.
        - "relative"/"fraction": column values are 0..1 scaling of max_b1_gauss
        - "percent": column values are 0..100 percent of max_b1_gauss
        - "gauss"/"mt"/"ut": column values are absolute amplitudes in those units
    phase_unit : str, optional
        "deg" or "rad".
    layout : str, optional
        "columns" for amp|phase per row,
        "amp_phase_interleaved" for amp, phase, amp, phase...,
        "phase_amp_interleaved" for phase, amp, phase, amp...

    Returns
    -------
    b1 : ndarray (complex128)
        Complex B1 envelope in Gauss.
    time : ndarray (float64)
        Time array in seconds (uniformly spaced).
    metadata : PulseMetadata
        Basic metadata (duration/npoints/name/max_b1).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Pulse file not found: {filepath}")

    # Load data tolerating comma or whitespace delimiters.
    try:
        data = np.loadtxt(path, delimiter=',')
    except Exception:
        data = np.loadtxt(path)

    if data.ndim == 0:
        raise ValueError("Pulse file must have at least two values (amp, phase).")
    if data.ndim > 1:
        flat = data.ravel()
    else:
        flat = data

    amp_arr = None
    phase_arr = None
    layout_key = (layout or "columns").lower()

    if layout_key == "columns":
        if data.ndim == 1:
            if data.size < 2:
                raise ValueError("Pulse file must have at least two columns (amp, phase).")
            data = data.reshape((-1, 2))
        if data.shape[1] < 2:
            raise ValueError(f"Expected at least 2 columns (amp, phase) in {filepath}, got {data.shape[1]}")
        amp_arr = data[:, 0]
        phase_arr = data[:, 1]
    elif layout_key == "amp_phase_interleaved":
        if flat.size % 2 != 0:
            raise ValueError("Interleaved data must have an even number of entries (amp, phase, ...).")
        amp_arr = flat[0::2]
        phase_arr = flat[1::2]
    elif layout_key == "phase_amp_interleaved":
        if flat.size % 2 != 0:
            raise ValueError("Interleaved data must have an even number of entries (phase, amp, ...).")
        phase_arr = flat[0::2]
        amp_arr = flat[1::2]
    else:
        raise ValueError(f"Unsupported layout '{layout}'.")

    # Scale amplitude to Gauss.
    unit = (amplitude_unit or "relative").lower()
    if unit in ("percent", "percentage"):
        peak = max_b1_gauss if max_b1_gauss is not None else 1.0
        amp_gauss = (amp_arr / 100.0) * peak
        max_b1_used = float(peak)
    elif unit in ("fraction", "relative"):
        peak = max_b1_gauss if max_b1_gauss is not None else 1.0
        amp_gauss = amp_arr * peak
        max_b1_used = float(peak)
    elif unit == "mt":
        amp_gauss = amp_arr * 10.0
        max_b1_used = float(np.max(np.abs(amp_gauss))) if amp_gauss.size else max_b1_gauss
    elif unit == "ut":
        amp_gauss = amp_arr * 0.01
        max_b1_used = float(np.max(np.abs(amp_gauss))) if amp_gauss.size else max_b1_gauss
    else:  # gauss or fallback
        amp_gauss = amp_arr
        max_b1_used = float(np.max(np.abs(amp_gauss))) if amp_gauss.size else (max_b1_gauss if max_b1_gauss is not None else 0.0)

    # Phase to radians.
    if (phase_unit or "deg").lower().startswith("rad"):
        phase_rad = phase_arr.astype(float)
    else:
        phase_rad = np.deg2rad(phase_arr)
    b1 = amp_gauss * np.exp(1j * phase_rad)

    # Derive duration if not provided (look for '<number>ms' in filename).
    pulse_duration = duration_s
    if pulse_duration is None:
        m = re.search(r'(\d+(?:\\.\\d+)?)ms', path.stem, re.IGNORECASE)
        if m:
            pulse_duration = float(m.group(1)) / 1000.0
    if pulse_duration is not None:
        time = np.linspace(0.0, pulse_duration, len(b1), endpoint=False, dtype=np.float64)
    else:
        time = np.arange(len(b1), dtype=np.float64)

    metadata = PulseMetadata(
        name=path.stem,
        title=path.stem,
        origin=str(path),
        duration=pulse_duration if pulse_duration is not None else 0.0,
        npoints=len(b1),
        max_b1=max_b1_used,
        shape_type="custom",
        shape_mode="Excitation",
    )

    return b1.astype(np.complex128), time, metadata


if __name__ == "__main__":
    # Example usage
    print("Pulse Loader - Example Usage")
    print("=" * 60)

    # List available pulses
    lib = get_pulse_library()
    print(f"Found {len(lib.list_pulses())} pulse(s):")
    for pulse_name in lib.list_pulses():
        print(f"  - {pulse_name}")

    # Load a pulse
    if lib.list_pulses():
        pulse_name = lib.list_pulses()[0]
        print(f"\nLoading pulse: {pulse_name}")
        b1, time, metadata = lib.load(pulse_name)

        print(f"  Name: {metadata.name}")
        print(f"  Flip angle: {metadata.flip_angle}°")
        print(f"  Duration: {metadata.duration * 1000:.3f} ms")
        print(f"  Points: {len(b1)}")
        print(f"  Max B1: {metadata.max_b1:.4f} Gauss")
        print(f"  Nucleus: {metadata.nucleus}")
        print(f"  Field: {metadata.field_strength} T")
        print(f"  B1 shape: {b1.shape}")
        print(f"  B1 dtype: {b1.dtype}")
        print(f"  B1 range: [{np.min(np.abs(b1)):.4e}, {np.max(np.abs(b1)):.4e}] Gauss")
