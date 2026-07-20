"""Cartesian acquisition layout, reconstruction, and reference sequence builders."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar, Mapping, Optional, Tuple

import numpy as np

from .model import ADCEvent, GradientEvent, RFEvent, SequenceProgram


@dataclass(frozen=True)
class AcquisitionDimensions:
    """Map chronological ADC events to explicit outer acquisition indices.

    The indices are deliberately separate from a Cartesian or non-Cartesian
    trajectory. Each value applies to one complete :class:`ADCEvent`; sample
    coordinates can be expanded without changing the chronological signal.
    """

    AXIS_NAMES: ClassVar[Tuple[str, ...]] = (
        "slice",
        "echo",
        "repetition",
        "segment",
        "partition",
    )
    PULSEQ_LABELS: ClassVar[Mapping[str, str]] = {
        "slice": "SLC",
        "echo": "ECO",
        "repetition": "REP",
        "segment": "SEG",
        "partition": "PAR",
    }

    adc_event_sample_counts: Tuple[int, ...]
    slice_indices: Tuple[int, ...] = ()
    echo_indices: Tuple[int, ...] = ()
    repetition_indices: Tuple[int, ...] = ()
    segment_indices: Tuple[int, ...] = ()
    partition_indices: Tuple[int, ...] = ()
    source: str = "default"

    def __post_init__(self) -> None:
        counts = tuple(
            _positive_integer(value, "ADC event sample count")
            for value in self.adc_event_sample_counts
        )
        object.__setattr__(self, "adc_event_sample_counts", counts)
        event_count = len(counts)
        for axis in self.AXIS_NAMES:
            field_name = f"{axis}_indices"
            values = tuple(getattr(self, field_name))
            if not values and event_count:
                values = (0,) * event_count
            if len(values) != event_count:
                raise ValueError(f"{field_name} must contain one value per ADC event")
            normalized = []
            for value in values:
                integer = int(value)
                if integer != value:
                    raise ValueError(f"{field_name} must contain integer values")
                normalized.append(integer)
            object.__setattr__(self, field_name, tuple(normalized))
        object.__setattr__(self, "source", str(self.source))

    @property
    def num_adc_events(self) -> int:
        return len(self.adc_event_sample_counts)

    @property
    def num_samples(self) -> int:
        return int(sum(self.adc_event_sample_counts))

    @property
    def varying_axes(self) -> Tuple[str, ...]:
        return tuple(
            axis for axis in self.AXIS_NAMES if len(set(self.event_indices(axis))) > 1
        )

    def event_indices(self, axis: str) -> Tuple[int, ...]:
        """Return one index per ADC event for a canonical outer axis."""
        name = str(axis).lower()
        if name not in self.AXIS_NAMES:
            raise ValueError(
                f"axis must be one of {', '.join(self.AXIS_NAMES)}, got {axis!r}"
            )
        return getattr(self, f"{name}_indices")

    def sample_indices(self, axis: str) -> np.ndarray:
        """Expand event indices to one read-only value per ADC sample."""
        values = np.repeat(
            np.asarray(self.event_indices(axis), dtype=np.int64),
            np.asarray(self.adc_event_sample_counts, dtype=np.int64),
        )
        values.setflags(write=False)
        return values

    def to_metadata(self) -> dict:
        """Return a JSON-compatible representation."""
        return {
            "type": "acquisition_outer_dimensions",
            "source": self.source,
            "adc_event_sample_counts": self.adc_event_sample_counts,
            "event_indices": {
                axis: self.event_indices(axis) for axis in self.AXIS_NAMES
            },
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> "AcquisitionDimensions":
        """Restore an explicit acquisition-dimension metadata mapping."""
        if not isinstance(metadata, Mapping):
            raise TypeError("acquisition dimension metadata must be a mapping")
        kind = metadata.get("type", "acquisition_outer_dimensions")
        if kind != "acquisition_outer_dimensions":
            raise ValueError(f"unsupported acquisition dimension type {kind!r}")
        indices = metadata.get("event_indices", {})
        if not isinstance(indices, Mapping):
            raise ValueError("event_indices must be a mapping")
        return cls(
            adc_event_sample_counts=tuple(metadata.get("adc_event_sample_counts", ())),
            slice_indices=tuple(indices.get("slice", ())),
            echo_indices=tuple(indices.get("echo", ())),
            repetition_indices=tuple(indices.get("repetition", ())),
            segment_indices=tuple(indices.get("segment", ())),
            partition_indices=tuple(indices.get("partition", ())),
            source=str(metadata.get("source", "metadata")),
        )

    @classmethod
    def from_program(cls, program: SequenceProgram) -> "AcquisitionDimensions":
        """Use explicit metadata or Pulseq label state for each ADC event."""
        sample_counts = tuple(event.num_samples for event in program.adc_events)
        explicit = program.metadata.get("acquisition_dimensions")
        if explicit is not None:
            dimensions = cls.from_metadata(explicit)
            if dimensions.adc_event_sample_counts != sample_counts:
                raise ValueError(
                    "acquisition dimension metadata does not match ADC events"
                )
            return dimensions

        label_values = program.metadata.get("adc_label_values", {})
        if not isinstance(label_values, Mapping):
            raise ValueError("adc_label_values must be a mapping")
        axis_values = {}
        used_labels = False
        for axis, label in cls.PULSEQ_LABELS.items():
            values = label_values.get(label)
            if values is None:
                axis_values[axis] = (0,) * len(sample_counts)
                continue
            values = tuple(values)
            if len(values) != len(sample_counts):
                raise ValueError(
                    f"Pulseq {label} label count does not match ADC events"
                )
            axis_values[axis] = values
            used_labels = True
        return cls(
            adc_event_sample_counts=sample_counts,
            slice_indices=axis_values["slice"],
            echo_indices=axis_values["echo"],
            repetition_indices=axis_values["repetition"],
            segment_indices=axis_values["segment"],
            partition_indices=axis_values["partition"],
            source="pulseq_labels" if used_labels else "default",
        )


@dataclass(frozen=True)
class CartesianAcquisitionFrames:
    """Validated Cartesian 2D frames within one chronological ADC stream."""

    acquisitions: Tuple["CartesianAcquisition", ...]
    sample_indices: Tuple[Tuple[int, ...], ...]
    frame_indices: Tuple[Tuple[int, ...], ...]
    dimensions: AcquisitionDimensions
    moment_origins_cyc_per_m: Tuple[Tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        acquisitions = tuple(self.acquisitions)
        samples = tuple(
            tuple(int(value) for value in item) for item in self.sample_indices
        )
        frames = tuple(
            tuple(int(value) for value in item) for item in self.frame_indices
        )
        origins = tuple(
            tuple(float(value) for value in item)
            for item in self.moment_origins_cyc_per_m
        )
        if not origins:
            origins = tuple((0.0, 0.0, 0.0) for _ in acquisitions)
        if (
            not acquisitions
            or len(acquisitions) != len(samples)
            or len(frames) != len(samples)
        ):
            raise ValueError(
                "Cartesian frame metadata must contain equal non-zero lengths"
            )
        if any(len(item) != len(AcquisitionDimensions.AXIS_NAMES) for item in frames):
            raise ValueError("each Cartesian frame index must contain all outer axes")
        if len(origins) != len(acquisitions) or any(len(item) != 3 for item in origins):
            raise ValueError("each Cartesian frame requires one 3D moment origin")
        flattened = [value for item in samples for value in item]
        if len(flattened) != self.dimensions.num_samples:
            raise ValueError("Cartesian frames do not cover the complete ADC stream")
        if sorted(flattened) != list(range(self.dimensions.num_samples)):
            raise ValueError("Cartesian frame sample indices must be a permutation")
        for acquisition, item in zip(acquisitions, samples):
            if len(item) != acquisition.num_samples:
                raise ValueError(
                    "Cartesian frame sample count does not match its layout"
                )
        object.__setattr__(self, "acquisitions", acquisitions)
        object.__setattr__(self, "sample_indices", samples)
        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "moment_origins_cyc_per_m", origins)

    @property
    def num_frames(self) -> int:
        return len(self.acquisitions)

    @property
    def varying_axes(self) -> Tuple[str, ...]:
        return tuple(
            axis
            for index, axis in enumerate(AcquisitionDimensions.AXIS_NAMES)
            if len({frame[index] for frame in self.frame_indices}) > 1
        )

    def frame_label(self, frame: int) -> str:
        values = self.frame_indices[int(frame)]
        axes = self.varying_axes or ("frame",)
        if axes == ("frame",):
            return f"frame={int(frame)}"
        return ", ".join(
            f"{axis}={values[AcquisitionDimensions.AXIS_NAMES.index(axis)]}"
            for axis in axes
        )

    def _frame_values(self, values, frame: int) -> np.ndarray:
        array = np.asarray(values)
        if array.shape[-1] != self.dimensions.num_samples:
            raise ValueError(
                "values do not match the complete chronological ADC stream"
            )
        return np.take(array, self.sample_indices[int(frame)], axis=-1)

    def to_cartesian_kspace(self, result, frame: int) -> np.ndarray:
        """Validate and reshape one selected 2D frame from a result."""
        frame = int(frame)
        acquisition = self.acquisitions[frame]
        times = self._frame_values(result.adc_times_s, frame)
        acquisition.validate_adc_times(times)
        if result.adc_gradient_moment_cyc_per_m is not None:
            moments = np.take(
                np.asarray(result.adc_gradient_moment_cyc_per_m),
                self.sample_indices[frame],
                axis=0,
            )
            moments = moments - np.asarray(
                self.moment_origins_cyc_per_m[frame], dtype=float
            )
            acquisition.validate_gradient_moments(moments)
        return acquisition.reshape_signal(self._frame_values(result.signal, frame))

    def reconstruct(
        self,
        result,
        frame: int,
        *,
        norm: Optional[str] = None,
        coil_combine: Optional[str] = None,
        voxel_centered: bool = True,
    ) -> np.ndarray:
        """Reconstruct one selected validated Cartesian frame."""
        frame = int(frame)
        self.to_cartesian_kspace(result, frame)
        return self.acquisitions[frame].reconstruct(
            self._frame_values(result.signal, frame),
            norm=norm,
            coil_combine=coil_combine,
            voxel_centered=voxel_centered,
        )

    def to_metadata(self) -> dict:
        return {
            "type": "cartesian_2d_frames",
            "acquisitions": [
                acquisition.to_metadata() for acquisition in self.acquisitions
            ],
            "sample_indices": self.sample_indices,
            "frame_indices": self.frame_indices,
            "dimensions": self.dimensions.to_metadata(),
            "moment_origins_cyc_per_m": self.moment_origins_cyc_per_m,
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> "CartesianAcquisitionFrames":
        if metadata.get("type") != "cartesian_2d_frames":
            raise ValueError("unsupported Cartesian frame metadata")
        return cls(
            acquisitions=tuple(
                CartesianAcquisition.from_metadata(item)
                for item in metadata["acquisitions"]
            ),
            sample_indices=tuple(
                tuple(int(value) for value in item)
                for item in metadata["sample_indices"]
            ),
            frame_indices=tuple(
                tuple(int(value) for value in item)
                for item in metadata["frame_indices"]
            ),
            dimensions=AcquisitionDimensions.from_metadata(metadata["dimensions"]),
            moment_origins_cyc_per_m=tuple(
                tuple(float(value) for value in item)
                for item in metadata.get("moment_origins_cyc_per_m", ())
            ),
        )


@dataclass(frozen=True)
class SpectroscopicAcquisition:
    """Map phase-encoded 2D CSI data to ``(ky, kx, spectral_point)``.

    Unlike a Cartesian imaging readout, every ADC event is an FID acquired at
    one fixed spatial k-space coordinate.  Treating the spectral samples as a
    readout axis would therefore produce a physically invalid reconstruction.
    """

    matrix: Tuple[int, int]
    fov_m: Tuple[float, float]
    spectral_points: int
    dwell_s: float
    encoding_indices: Tuple[Tuple[int, int], ...]
    moment_origins_cyc_per_m: Tuple[Tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        matrix = tuple(
            _positive_integer(value, "CSI matrix size") for value in self.matrix
        )
        if len(matrix) != 2:
            raise ValueError("CSI matrix must contain x and y sizes")
        fov = tuple(float(value) for value in self.fov_m)
        if len(fov) != 2 or not np.all(np.isfinite(fov)) or min(fov) <= 0:
            raise ValueError("CSI fov_m must contain two positive finite values")
        points = _positive_integer(self.spectral_points, "spectral_points")
        dwell = float(self.dwell_s)
        if not np.isfinite(dwell) or dwell <= 0:
            raise ValueError("CSI dwell_s must be positive and finite")
        indices = tuple((int(x), int(y)) for x, y in self.encoding_indices)
        expected = {(x, y) for y in range(matrix[1]) for x in range(matrix[0])}
        if len(indices) != matrix[0] * matrix[1] or set(indices) != expected:
            raise ValueError("CSI encoding_indices must cover the spatial grid once")
        origins = tuple(
            tuple(float(value) for value in origin)
            for origin in self.moment_origins_cyc_per_m
        )
        if not origins:
            origins = tuple((0.0, 0.0, 0.0) for _ in indices)
        if len(origins) != len(indices) or any(len(origin) != 3 for origin in origins):
            raise ValueError("CSI requires one 3D gradient-moment origin per FID")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "fov_m", fov)
        object.__setattr__(self, "spectral_points", points)
        object.__setattr__(self, "dwell_s", dwell)
        object.__setattr__(self, "encoding_indices", indices)
        object.__setattr__(self, "moment_origins_cyc_per_m", origins)

    @property
    def num_encodings(self) -> int:
        return self.matrix[0] * self.matrix[1]

    @property
    def num_samples(self) -> int:
        return self.num_encodings * self.spectral_points

    @property
    def spectral_bandwidth_hz(self) -> float:
        return 1.0 / self.dwell_s

    @property
    def spectral_resolution_hz(self) -> float:
        return self.spectral_bandwidth_hz / self.spectral_points

    @property
    def kx_cyc_per_m(self) -> np.ndarray:
        return (
            np.arange(self.matrix[0], dtype=float) - self.matrix[0] // 2
        ) / self.fov_m[0]

    @property
    def ky_cyc_per_m(self) -> np.ndarray:
        return (
            np.arange(self.matrix[1], dtype=float) - self.matrix[1] // 2
        ) / self.fov_m[1]

    @property
    def spectral_time_s(self) -> np.ndarray:
        return np.arange(self.spectral_points, dtype=float) * self.dwell_s

    @property
    def frequency_hz(self) -> np.ndarray:
        return np.fft.fftshift(np.fft.fftfreq(self.spectral_points, self.dwell_s))

    def reshape_signal(self, signal: np.ndarray) -> np.ndarray:
        """Return chronological signal as ``(..., ky, kx, spectral_point)``."""
        values = np.asarray(signal)
        if values.ndim not in (1, 2) or values.shape[-1] != self.num_samples:
            raise ValueError(
                f"signal must end with {self.num_samples} chronological CSI samples"
            )
        raw = values.reshape(
            values.shape[:-1] + (self.num_encodings, self.spectral_points)
        )
        grid = np.empty(
            values.shape[:-1] + (self.matrix[1], self.matrix[0], self.spectral_points),
            dtype=values.dtype,
        )
        for acquired, (x_index, y_index) in enumerate(self.encoding_indices):
            grid[..., y_index, x_index, :] = raw[..., acquired, :]
        return grid

    def reconstruct_spatial(
        self,
        signal: np.ndarray,
        *,
        norm: Optional[str] = None,
        voxel_centered: bool = True,
    ) -> np.ndarray:
        """Apply the spatial 2D inverse FFT while retaining the complete FID."""
        kspace = self.reshape_signal(signal)
        if voxel_centered:
            dx = self.fov_m[0] / self.matrix[0]
            dy = self.fov_m[1] / self.matrix[1]
            phase = np.exp(
                2j
                * np.pi
                * (
                    self.ky_cyc_per_m[:, None] * dy / 2
                    + self.kx_cyc_per_m[None, :] * dx / 2
                )
            )
            kspace = kspace * phase[..., None]
        axes = (-3, -2)
        return np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm=norm),
            axes=axes,
        )

    def reconstruct_spectra(
        self,
        signal: np.ndarray,
        *,
        norm: Optional[str] = None,
        voxel_centered: bool = True,
    ) -> np.ndarray:
        """Return spatially reconstructed complex spectra as ``(..., y, x, f)``."""
        fid = self.reconstruct_spatial(signal, norm=norm, voxel_centered=voxel_centered)
        return np.fft.fftshift(np.fft.fft(fid, axis=-1), axes=-1)

    def validate_adc_times(self, adc_times_s: np.ndarray) -> None:
        times = np.asarray(adc_times_s, dtype=float)
        if times.shape != (self.num_samples,) or not np.all(np.isfinite(times)):
            raise ValueError("ADC times do not match the CSI acquisition")
        fids = times.reshape(self.num_encodings, self.spectral_points)
        if self.spectral_points > 1 and not np.allclose(
            np.diff(fids, axis=1), self.dwell_s, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "CSI spectral dwell spacing does not match the acquisition"
            )

    def validate_gradient_moments(self, moments_cyc_per_m: np.ndarray) -> None:
        moments = np.asarray(moments_cyc_per_m, dtype=float)
        if moments.shape != (self.num_samples, 3):
            raise ValueError("CSI gradient moments must have shape (num_samples, 3)")
        raw = moments.reshape(self.num_encodings, self.spectral_points, 3)
        origins = np.asarray(self.moment_origins_cyc_per_m)[:, None, :]
        relative = raw - origins
        tolerance_x = max(1e-9, 1e-3 / self.fov_m[0])
        tolerance_y = max(1e-9, 1e-3 / self.fov_m[1])
        for acquired, (x_index, y_index) in enumerate(self.encoding_indices):
            if not np.allclose(
                relative[acquired, :, 0],
                self.kx_cyc_per_m[x_index],
                rtol=0.0,
                atol=tolerance_x,
            ):
                raise ValueError(
                    "CSI x phase encoding does not match the declared grid"
                )
            if not np.allclose(
                relative[acquired, :, 1],
                self.ky_cyc_per_m[y_index],
                rtol=0.0,
                atol=tolerance_y,
            ):
                raise ValueError(
                    "CSI y phase encoding does not match the declared grid"
                )

    def to_metadata(self) -> dict:
        return {
            "type": "csi_2d",
            "matrix": self.matrix,
            "fov_m": self.fov_m,
            "spectral_points": self.spectral_points,
            "dwell_s": self.dwell_s,
            "encoding_indices": self.encoding_indices,
            "moment_origins_cyc_per_m": self.moment_origins_cyc_per_m,
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> "SpectroscopicAcquisition":
        if metadata.get("type") != "csi_2d":
            raise ValueError("unsupported spectroscopic acquisition metadata")
        return cls(
            matrix=tuple(metadata["matrix"]),
            fov_m=tuple(metadata["fov_m"]),
            spectral_points=metadata["spectral_points"],
            dwell_s=metadata["dwell_s"],
            encoding_indices=tuple(
                tuple(value) for value in metadata["encoding_indices"]
            ),
            moment_origins_cyc_per_m=tuple(
                tuple(value) for value in metadata.get("moment_origins_cyc_per_m", ())
            ),
        )


@dataclass(frozen=True)
class CartesianAcquisition:
    """Describe how a chronological ADC stream maps to a 2D Cartesian grid."""

    read_matrix: int
    phase_matrix: int
    fov_m: Tuple[float, float]
    dwell_s: float
    phase_indices: Optional[Tuple[int, ...]] = None
    readout_directions: Optional[Tuple[int, ...]] = None
    kx_offset_cells: float = 0.0
    ky_offset_cells: float = 0.0

    def __post_init__(self) -> None:
        read_matrix = _positive_integer(self.read_matrix, "read_matrix")
        phase_matrix = _positive_integer(self.phase_matrix, "phase_matrix")
        fov = tuple(float(value) for value in self.fov_m)
        if len(fov) != 2 or not np.all(np.isfinite(fov)) or min(fov) <= 0:
            raise ValueError("fov_m must contain two positive finite values")
        dwell = float(self.dwell_s)
        if not np.isfinite(dwell) or dwell <= 0:
            raise ValueError("dwell_s must be positive and finite")
        kx_offset = float(self.kx_offset_cells)
        ky_offset = float(self.ky_offset_cells)
        if not np.isfinite(kx_offset) or not np.isfinite(ky_offset):
            raise ValueError("Cartesian k-space offsets must be finite")

        phase_indices = (
            tuple(range(phase_matrix))
            if self.phase_indices is None
            else tuple(int(value) for value in self.phase_indices)
        )
        if len(phase_indices) != phase_matrix or sorted(phase_indices) != list(
            range(phase_matrix)
        ):
            raise ValueError("phase_indices must be a permutation of phase rows")

        directions = (
            tuple(1 for _ in range(phase_matrix))
            if self.readout_directions is None
            else tuple(int(value) for value in self.readout_directions)
        )
        if len(directions) != phase_matrix or any(
            value not in (-1, 1) for value in directions
        ):
            raise ValueError("readout_directions must contain one +1/-1 per row")

        object.__setattr__(self, "read_matrix", read_matrix)
        object.__setattr__(self, "phase_matrix", phase_matrix)
        object.__setattr__(self, "fov_m", fov)
        object.__setattr__(self, "dwell_s", dwell)
        object.__setattr__(self, "phase_indices", phase_indices)
        object.__setattr__(self, "readout_directions", directions)
        object.__setattr__(self, "kx_offset_cells", kx_offset)
        object.__setattr__(self, "ky_offset_cells", ky_offset)

    @classmethod
    def epi(
        cls,
        read_matrix: int,
        phase_matrix: int,
        fov_m: Tuple[float, float],
        dwell_s: float,
        *,
        phase_indices: Optional[Tuple[int, ...]] = None,
        first_readout_direction: int = 1,
    ) -> "CartesianAcquisition":
        """Create a Cartesian layout with alternating EPI readout directions."""
        if first_readout_direction not in (-1, 1):
            raise ValueError("first_readout_direction must be +1 or -1")
        directions = tuple(
            first_readout_direction * (-1 if line % 2 else 1)
            for line in range(int(phase_matrix))
        )
        return cls(
            read_matrix=read_matrix,
            phase_matrix=phase_matrix,
            fov_m=fov_m,
            dwell_s=dwell_s,
            phase_indices=phase_indices,
            readout_directions=directions,
        )

    @property
    def num_samples(self) -> int:
        return self.read_matrix * self.phase_matrix

    @property
    def sampling_bandwidth_hz(self) -> float:
        """Complex receiver sampling bandwidth (spectral width)."""
        return 1.0 / self.dwell_s

    @property
    def pixel_bandwidth_hz(self) -> float:
        """Nominal receiver bandwidth per readout pixel."""
        return self.sampling_bandwidth_hz / self.read_matrix

    @property
    def kx_cyc_per_m(self) -> np.ndarray:
        return (
            np.arange(self.read_matrix, dtype=float)
            - self.read_matrix // 2
            + self.kx_offset_cells
        ) / self.fov_m[0]

    @property
    def ky_cyc_per_m(self) -> np.ndarray:
        return (
            np.arange(self.phase_matrix, dtype=float)
            - self.phase_matrix // 2
            + self.ky_offset_cells
        ) / self.fov_m[1]

    def reshape_signal(self, signal: np.ndarray) -> np.ndarray:
        """Map chronological ADC data to ``(..., phase, read)`` k-space."""
        values = np.asarray(signal)
        if values.ndim not in (1, 2) or values.shape[-1] != self.num_samples:
            raise ValueError(
                f"signal must end with {self.num_samples} chronological samples"
            )
        raw = values.reshape(values.shape[:-1] + (self.phase_matrix, self.read_matrix))
        grid = np.empty_like(raw)
        for acquired_line, phase_index in enumerate(self.phase_indices):
            line = raw[..., acquired_line, :]
            if self.readout_directions[acquired_line] < 0:
                line = line[..., ::-1]
            grid[..., phase_index, :] = line
        return grid

    def reconstruct(
        self,
        signal: np.ndarray,
        *,
        norm: Optional[str] = None,
        coil_combine: Optional[str] = None,
        voxel_centered: bool = True,
    ) -> np.ndarray:
        """Reconstruct Cartesian data with a centred inverse 2D FFT."""
        kspace = self.reshape_signal(signal)
        if voxel_centered:
            dx = self.fov_m[0] / self.read_matrix
            dy = self.fov_m[1] / self.phase_matrix
            centre_phase = np.exp(
                2j
                * np.pi
                * (
                    self.ky_cyc_per_m[:, None] * dy / 2
                    + self.kx_cyc_per_m[None, :] * dx / 2
                )
            )
            kspace = kspace * centre_phase
        axes = (-2, -1)
        image = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm=norm),
            axes=axes,
        )
        if coil_combine is None:
            return image
        if image.ndim != 3:
            raise ValueError("coil combination requires signal shape (coil, adc)")
        if coil_combine == "rss":
            return np.sqrt(np.sum(np.abs(image) ** 2, axis=0))
        if coil_combine == "sum":
            return np.sum(image, axis=0)
        raise ValueError("coil_combine must be None, 'rss', or 'sum'")

    def validate_adc_times(self, adc_times_s: np.ndarray) -> None:
        """Validate sample count and within-line dwell spacing."""
        times = np.asarray(adc_times_s, dtype=float)
        if times.shape != (self.num_samples,) or not np.all(np.isfinite(times)):
            raise ValueError("ADC times do not match the Cartesian acquisition")
        lines = times.reshape(self.phase_matrix, self.read_matrix)
        if self.read_matrix > 1 and not np.allclose(
            np.diff(lines, axis=1), self.dwell_s, rtol=0.0, atol=1e-12
        ):
            raise ValueError("ADC dwell spacing does not match the acquisition")

    def validate_gradient_moments(self, moments_cyc_per_m: np.ndarray) -> None:
        """Validate that ADC gradient moments lie on the described 2D grid."""
        moments = np.asarray(moments_cyc_per_m, dtype=float)
        if moments.shape != (self.num_samples, 3):
            raise ValueError("gradient moments must have shape (num_samples, 3)")
        raw = moments.reshape(self.phase_matrix, self.read_matrix, 3)
        for acquired_line, phase_index in enumerate(self.phase_indices):
            expected_x = self.kx_cyc_per_m
            if self.readout_directions[acquired_line] < 0:
                expected_x = expected_x[::-1]
            expected_y = self.ky_cyc_per_m[phase_index]
            x_tolerance = max(1e-9, 1e-3 / self.fov_m[0])
            y_tolerance = max(1e-9, 1e-3 / self.fov_m[1])
            if not np.allclose(
                raw[acquired_line, :, 0], expected_x, rtol=0.0, atol=x_tolerance
            ):
                raise ValueError("readout gradient moments do not match the grid")
            if not np.allclose(
                raw[acquired_line, :, 1], expected_y, rtol=0.0, atol=y_tolerance
            ):
                raise ValueError("phase gradient moments do not match the grid")

    def to_metadata(self) -> dict:
        return {
            "type": "cartesian_2d",
            "read_matrix": self.read_matrix,
            "phase_matrix": self.phase_matrix,
            "fov_m": self.fov_m,
            "dwell_s": self.dwell_s,
            "sampling_bandwidth_hz": self.sampling_bandwidth_hz,
            "pixel_bandwidth_hz": self.pixel_bandwidth_hz,
            "phase_indices": self.phase_indices,
            "readout_directions": self.readout_directions,
            "kx_offset_cells": self.kx_offset_cells,
            "ky_offset_cells": self.ky_offset_cells,
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> "CartesianAcquisition":
        if metadata.get("type") != "cartesian_2d":
            raise ValueError("unsupported Cartesian acquisition metadata")
        return cls(
            read_matrix=metadata["read_matrix"],
            phase_matrix=metadata["phase_matrix"],
            fov_m=tuple(metadata["fov_m"]),
            dwell_s=metadata["dwell_s"],
            phase_indices=tuple(metadata["phase_indices"]),
            readout_directions=tuple(metadata["readout_directions"]),
            kx_offset_cells=metadata.get("kx_offset_cells", 0.0),
            ky_offset_cells=metadata.get("ky_offset_cells", 0.0),
        )


def infer_cartesian_acquisition(
    program: SequenceProgram,
    *,
    compiled=None,
) -> CartesianAcquisition:
    """Infer one regular 2D Cartesian acquisition from a sequence program.

    The conservative inference currently accepts one chronological ADC event
    per phase line, x readout, y phase encoding, and an explicit Pulseq FOV
    definition. Multi-slice, repeated, segmented, or non-Cartesian streams are
    rejected instead of being reshaped ambiguously.
    """
    from .compiler import SequenceCompiler

    adc_events = program.adc_events
    if not adc_events:
        raise ValueError("sequence contains no ADC events")
    outer_dimensions = AcquisitionDimensions.from_program(program)
    if outer_dimensions.varying_axes:
        axes = ", ".join(outer_dimensions.varying_axes)
        raise ValueError(
            "single 2D Cartesian inference does not support varying outer "
            f"acquisition dimensions: {axes}"
        )
    read_matrix = adc_events[0].num_samples
    if read_matrix < 2:
        raise ValueError("Cartesian inference requires at least two samples per line")
    if any(event.num_samples != read_matrix for event in adc_events):
        raise ValueError("ADC events do not have a common read matrix")
    dwell_s = adc_events[0].dwell_s
    if any(
        not np.isclose(event.dwell_s, dwell_s, rtol=0.0, atol=1e-15)
        for event in adc_events
    ):
        raise ValueError("ADC events do not have a common dwell time")
    if any(
        current.start_s <= previous.start_s
        for previous, current in zip(adc_events, adc_events[1:])
    ):
        raise ValueError("ADC events are not strictly chronological")

    definitions = dict(program.metadata.get("definitions", {}))
    fov_value = next(
        (value for key, value in definitions.items() if str(key).lower() == "fov"),
        None,
    )
    if fov_value is None:
        raise ValueError("Pulseq sequence has no FOV definition")
    fov = np.asarray(fov_value, dtype=float).reshape(-1)
    if fov.size < 2 or not np.all(np.isfinite(fov[:2])) or np.any(fov[:2] <= 0):
        raise ValueError("Pulseq FOV definition does not contain valid x/y values")
    fov_x, fov_y = (float(fov[0]), float(fov[1]))

    compiled = SequenceCompiler().compile(program) if compiled is None else compiled
    expected_times = np.concatenate([event.sample_times_s for event in adc_events])
    if compiled.adc_times_s.shape != expected_times.shape or not np.allclose(
        compiled.adc_times_s, expected_times, rtol=0.0, atol=1e-12
    ):
        raise ValueError("ADC samples cannot be grouped into chronological lines")

    phase_matrix = len(adc_events)
    moments = np.asarray(compiled.adc_gradient_moment_cyc_per_m, dtype=float)
    if moments.shape != (phase_matrix * read_matrix, 3):
        raise ValueError("compiled ADC gradient moments have an invalid shape")
    raw = moments.reshape(phase_matrix, read_matrix, 3)
    tolerance_x = max(1e-9, 1e-3 / fov_x)
    tolerance_y = max(1e-9, 1e-3 / fov_y)

    delta_x = np.diff(raw[:, :, 0], axis=1)
    mean_delta_x = np.mean(delta_x, axis=1)
    if np.any(np.abs(mean_delta_x) <= tolerance_x):
        raise ValueError("ADC events do not contain an x readout gradient")
    directions = np.where(mean_delta_x > 0, 1, -1)
    expected_delta_x = directions[:, None] / fov_x
    if not np.allclose(delta_x, expected_delta_x, rtol=0.0, atol=tolerance_x):
        raise ValueError("ADC readout samples are not on a regular x grid")
    if not np.allclose(np.diff(raw[:, :, 1], axis=1), 0.0, rtol=0.0, atol=tolerance_y):
        raise ValueError("phase gradient changes during an ADC line")
    z_scale = max(1.0, float(np.max(np.abs(raw[:, :, 2]))))
    if not np.allclose(
        np.diff(raw[:, :, 2], axis=1), 0.0, rtol=0.0, atol=1e-9 * z_scale
    ):
        raise ValueError("slice gradient changes during an ADC line")

    ordered_x = np.stack(
        [
            raw[line, :, 0] if directions[line] > 0 else raw[line, ::-1, 0]
            for line in range(phase_matrix)
        ]
    )
    common_x = np.mean(ordered_x, axis=0)
    if not np.allclose(ordered_x, common_x, rtol=0.0, atol=tolerance_x):
        raise ValueError("EPI readout lines do not share one Cartesian kx grid")

    line_y = np.mean(raw[:, :, 1], axis=1)
    phase_order = np.argsort(line_y)
    sorted_y = line_y[phase_order]
    if phase_matrix > 1 and not np.allclose(
        np.diff(sorted_y), 1.0 / fov_y, rtol=0.0, atol=tolerance_y
    ):
        raise ValueError("ADC lines do not form one regular Cartesian ky grid")
    phase_indices = np.empty(phase_matrix, dtype=int)
    phase_indices[phase_order] = np.arange(phase_matrix)

    base_x = np.arange(read_matrix, dtype=float) - read_matrix // 2
    kx_offset = float(np.mean(common_x * fov_x - base_x))
    if not np.allclose(
        common_x * fov_x,
        base_x + kx_offset,
        rtol=0.0,
        atol=tolerance_x * fov_x,
    ):
        raise ValueError("kx coordinates cannot be represented by one Cartesian grid")
    rounded_kx_offset = round(2.0 * kx_offset) / 2.0
    if np.allclose(
        ordered_x * fov_x,
        base_x + rounded_kx_offset,
        rtol=0.0,
        atol=tolerance_x * fov_x,
    ):
        kx_offset = rounded_kx_offset
    base_y = np.arange(phase_matrix, dtype=float) - phase_matrix // 2
    ky_offset = float(np.mean(sorted_y * fov_y - base_y))
    if not np.allclose(
        sorted_y * fov_y,
        base_y + ky_offset,
        rtol=0.0,
        atol=tolerance_y * fov_y,
    ):
        raise ValueError("ky coordinates cannot be represented by one Cartesian grid")
    rounded_ky_offset = round(2.0 * ky_offset) / 2.0
    if np.allclose(
        sorted_y * fov_y,
        base_y + rounded_ky_offset,
        rtol=0.0,
        atol=tolerance_y * fov_y,
    ):
        ky_offset = rounded_ky_offset

    acquisition = CartesianAcquisition(
        read_matrix=read_matrix,
        phase_matrix=phase_matrix,
        fov_m=(fov_x, fov_y),
        dwell_s=dwell_s,
        phase_indices=tuple(int(value) for value in phase_indices),
        readout_directions=tuple(int(value) for value in directions),
        kx_offset_cells=kx_offset,
        ky_offset_cells=ky_offset,
    )
    acquisition.validate_adc_times(compiled.adc_times_s)
    acquisition.validate_gradient_moments(moments)
    return acquisition


def infer_spectroscopic_acquisition(
    program: SequenceProgram,
    *,
    compiled=None,
) -> SpectroscopicAcquisition:
    """Infer a labelled phase-encoded 2D CSI acquisition.

    CSI is identified conservatively from ``MatrixSize``/``SpectralPoints``
    definitions plus Pulseq ``LIN`` and ``PAR`` labels.  The gradient moments
    are validated relative to the RF event preceding each FID so that moments
    accumulated during earlier repetitions are not mistaken for encoding.
    """
    from .compiler import SequenceCompiler

    adc_events = program.adc_events
    if not adc_events:
        raise ValueError("sequence contains no ADC events")
    definitions = {
        str(key).lower(): value
        for key, value in dict(program.metadata.get("definitions", {})).items()
    }
    matrix_value = definitions.get("matrixsize")
    spectral_value = definitions.get("spectralpoints")
    fov_value = definitions.get("fov")
    if matrix_value is None or fov_value is None:
        raise ValueError("CSI inference requires MatrixSize and FOV definitions")
    matrix_values = np.asarray(matrix_value, dtype=float).reshape(-1)
    if matrix_values.size < 3:
        raise ValueError("CSI MatrixSize must contain x, y, and spectral sizes")
    nx, ny, matrix_spectral = (
        _positive_integer(value, "CSI MatrixSize") for value in matrix_values[:3]
    )
    spectral_points = (
        matrix_spectral
        if spectral_value is None
        else _positive_integer(spectral_value, "SpectralPoints")
    )
    if spectral_points != matrix_spectral:
        raise ValueError("CSI MatrixSize and SpectralPoints definitions disagree")
    if len(adc_events) != nx * ny:
        raise ValueError("ADC event count does not match the declared CSI grid")
    if any(event.num_samples != spectral_points for event in adc_events):
        raise ValueError("ADC event sizes do not match CSI SpectralPoints")
    dwell_s = adc_events[0].dwell_s
    if any(
        not np.isclose(event.dwell_s, dwell_s, rtol=0.0, atol=1e-15)
        for event in adc_events
    ):
        raise ValueError("CSI ADC events do not have a common spectral dwell")

    fov = np.asarray(fov_value, dtype=float).reshape(-1)
    if fov.size < 2 or not np.all(np.isfinite(fov[:2])) or np.any(fov[:2] <= 0):
        raise ValueError("CSI FOV definition does not contain valid x/y values")
    labels = program.metadata.get("adc_label_values", {})
    lin = tuple(labels.get("LIN", ()))
    par = tuple(labels.get("PAR", ()))
    if len(lin) != len(adc_events) or len(par) != len(adc_events):
        raise ValueError("CSI inference requires one LIN and PAR label per FID")
    encoding_indices = tuple((int(x), int(y)) for x, y in zip(lin, par))

    compiled = SequenceCompiler().compile(program) if compiled is None else compiled
    origins = tuple(
        tuple(
            float(value)
            for value in _frame_gradient_moment_origin(program, compiled, event.start_s)
        )
        for event in adc_events
    )
    acquisition = SpectroscopicAcquisition(
        matrix=(nx, ny),
        fov_m=(float(fov[0]), float(fov[1])),
        spectral_points=spectral_points,
        dwell_s=dwell_s,
        encoding_indices=encoding_indices,
        moment_origins_cyc_per_m=origins,
    )
    acquisition.validate_adc_times(compiled.adc_times_s)
    acquisition.validate_gradient_moments(compiled.adc_gradient_moment_cyc_per_m)
    return acquisition


def infer_cartesian_acquisition_frames(
    program: SequenceProgram,
    *,
    compiled=None,
) -> CartesianAcquisitionFrames:
    """Infer complete Cartesian 2D frames grouped by outer dimensions.

    Pulseq outer labels are preferred. If they are absent, multiple frames are
    derived only when at least two RF-delimited groups each contain multiple ADC
    lines. Distinct RF frequency offsets are interpreted as slice indices;
    otherwise the groups are repetitions. Every group is independently passed
    through the strict single-frame Cartesian validator.
    """
    from .compiler import SequenceCompiler

    adc_events = program.adc_events
    if not adc_events:
        raise ValueError("sequence contains no ADC events")
    compiled = SequenceCompiler().compile(program) if compiled is None else compiled
    dimensions = AcquisitionDimensions.from_program(program)

    if dimensions.varying_axes:
        event_frames = [
            tuple(
                dimensions.event_indices(axis)[index] for axis in dimensions.AXIS_NAMES
            )
            for index in range(dimensions.num_adc_events)
        ]
    else:
        event_frames, dimensions = _derive_rf_delimited_dimensions(program, dimensions)

    grouped_events = {}
    for event_index, frame_index in enumerate(event_frames):
        grouped_events.setdefault(frame_index, []).append(event_index)
    if len(grouped_events) < 2:
        raise ValueError("sequence does not contain multiple explicit 2D frames")

    event_offsets = np.concatenate(
        ([0], np.cumsum(dimensions.adc_event_sample_counts, dtype=np.int64))
    )
    acquisitions = []
    frame_samples = []
    frame_indices = []
    moment_origins = []
    for frame_index, event_indices in grouped_events.items():
        sample_indices = tuple(
            value
            for event_index in event_indices
            for value in range(
                int(event_offsets[event_index]), int(event_offsets[event_index + 1])
            )
        )
        subset_dimensions = AcquisitionDimensions(
            adc_event_sample_counts=tuple(
                adc_events[index].num_samples for index in event_indices
            ),
            slice_indices=tuple(frame_index[0] for _ in event_indices),
            echo_indices=tuple(frame_index[1] for _ in event_indices),
            repetition_indices=tuple(frame_index[2] for _ in event_indices),
            segment_indices=tuple(frame_index[3] for _ in event_indices),
            partition_indices=tuple(frame_index[4] for _ in event_indices),
            source=dimensions.source,
        )
        subset_metadata = dict(program.metadata)
        subset_metadata["acquisition_dimensions"] = subset_dimensions.to_metadata()
        subset_program = SequenceProgram(
            events=tuple(adc_events[index] for index in event_indices),
            duration_s=program.duration_s,
            source=program.source,
            version=program.version,
            metadata=subset_metadata,
        )
        moment_origin = _frame_gradient_moment_origin(
            program, compiled, adc_events[event_indices[0]].start_s
        )
        subset_compiled = SimpleNamespace(
            adc_times_s=np.take(compiled.adc_times_s, sample_indices),
            adc_gradient_moment_cyc_per_m=(
                np.take(compiled.adc_gradient_moment_cyc_per_m, sample_indices, axis=0)
                - moment_origin
            ),
        )
        acquisitions.append(
            infer_cartesian_acquisition(subset_program, compiled=subset_compiled)
        )
        frame_samples.append(sample_indices)
        frame_indices.append(frame_index)
        moment_origins.append(tuple(float(value) for value in moment_origin))

    return CartesianAcquisitionFrames(
        acquisitions=tuple(acquisitions),
        sample_indices=tuple(frame_samples),
        frame_indices=tuple(frame_indices),
        dimensions=dimensions,
        moment_origins_cyc_per_m=tuple(moment_origins),
    )


def _frame_gradient_moment_origin(
    program: SequenceProgram, compiled, first_adc_start_s: float
) -> np.ndarray:
    preceding_rf = [
        rf for rf in program.rf_events if rf.end_s <= first_adc_start_s + 1e-15
    ]
    if not preceding_rf:
        return np.zeros(3, dtype=float)
    reference_time = preceding_rf[-1].start_s
    boundaries = np.concatenate(([0.0], np.asarray(compiled.interval_end_s)))
    moments = np.vstack(
        (
            np.zeros((1, 3), dtype=float),
            np.cumsum(
                np.asarray(compiled.gradient_hz_per_m)
                * np.asarray(compiled.dt_s)[:, None],
                axis=0,
            ),
        )
    )
    index = int(np.argmin(np.abs(boundaries - reference_time)))
    tolerance = max(1e-12, abs(reference_time) * 1e-10)
    if abs(boundaries[index] - reference_time) > tolerance:
        raise ValueError("RF reference time is not a compiled sequence boundary")
    return moments[index]


def _derive_rf_delimited_dimensions(
    program: SequenceProgram,
    dimensions: AcquisitionDimensions,
):
    rf_events = program.rf_events
    if len(rf_events) < 2:
        raise ValueError("outer dimensions are absent and cannot be derived")

    rf_group_for_event = []
    for adc in program.adc_events:
        preceding = [
            index
            for index, rf in enumerate(rf_events)
            if rf.end_s <= adc.start_s + 1e-15
        ]
        if not preceding:
            raise ValueError("an ADC event precedes the first RF excitation")
        rf_group_for_event.append(preceding[-1])
    used_groups = tuple(dict.fromkeys(rf_group_for_event))
    counts = [rf_group_for_event.count(group) for group in used_groups]
    if len(used_groups) < 2 or min(counts) < 2:
        raise ValueError(
            "RF-delimited frame derivation requires multiple ADC lines per frame"
        )

    offsets = np.asarray(
        [rf_events[group].frequency_offset_hz for group in used_groups], dtype=float
    )
    slice_selective = all(
        any(
            gradient.axis == "z"
            and gradient.start_s < rf_events[group].end_s
            and gradient.end_s > rf_events[group].start_s
            and np.any(gradient.samples_hz_per_m != 0)
            for gradient in program.gradient_events
        )
        for group in used_groups
    )
    if slice_selective and np.unique(np.round(offsets, decimals=9)).size == len(
        used_groups
    ):
        ordered = np.argsort(offsets)
        group_values = {
            used_groups[int(group_position)]: int(rank)
            for rank, group_position in enumerate(ordered)
        }
        axis = "slice"
        source = "rf_frequency_offsets"
    else:
        group_values = {group: index for index, group in enumerate(used_groups)}
        axis = "repetition"
        source = "rf_delimited_repetitions"

    derived = {name: [0] * dimensions.num_adc_events for name in dimensions.AXIS_NAMES}
    derived[axis] = [group_values[group] for group in rf_group_for_event]
    updated = AcquisitionDimensions(
        adc_event_sample_counts=dimensions.adc_event_sample_counts,
        slice_indices=tuple(derived["slice"]),
        echo_indices=tuple(derived["echo"]),
        repetition_indices=tuple(derived["repetition"]),
        segment_indices=tuple(derived["segment"]),
        partition_indices=tuple(derived["partition"]),
        source=source,
    )
    event_frames = [
        tuple(updated.event_indices(name)[index] for name in updated.AXIS_NAMES)
        for index in range(updated.num_adc_events)
    ]
    return event_frames, updated


def make_cartesian_epi(
    acquisition: CartesianAcquisition,
    *,
    flip_angle_deg: float = 90.0,
    rf_duration_s: float = 1e-3,
    prephaser_duration_s: float = 1e-3,
    blip_duration_s: float = 100e-6,
    delay_after_prephaser_s: float = 0.0,
    tail_s: float = 0.0,
) -> SequenceProgram:
    """Build a non-slice-selective single-shot Cartesian EPI program."""
    for name, value, allow_zero in (
        ("rf_duration_s", rf_duration_s, False),
        ("prephaser_duration_s", prephaser_duration_s, False),
        ("blip_duration_s", blip_duration_s, False),
        ("delay_after_prephaser_s", delay_after_prephaser_s, True),
        ("tail_s", tail_s, True),
    ):
        if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} has an invalid duration")
    if not np.isfinite(flip_angle_deg):
        raise ValueError("flip_angle_deg must be finite")

    nx = acquisition.read_matrix
    ny = acquisition.phase_matrix
    fov_x, fov_y = acquisition.fov_m
    dwell = acquisition.dwell_s
    readout_duration = nx * dwell
    events = []

    rf_hz = np.deg2rad(flip_angle_deg) / (2 * np.pi * rf_duration_s)
    events.append(RFEvent(0.0, np.asarray([rf_hz]), rf_duration_s))

    first_direction = acquisition.readout_directions[0]
    first_kx = (
        acquisition.kx_cyc_per_m[0]
        if first_direction > 0
        else acquisition.kx_cyc_per_m[-1]
    )
    current_x = first_kx - first_direction * 0.5 / fov_x
    first_phase_index = acquisition.phase_indices[0]
    current_y = acquisition.ky_cyc_per_m[first_phase_index]
    prephaser_start = rf_duration_s
    if current_x != 0:
        events.append(
            GradientEvent(
                "x",
                prephaser_start,
                np.asarray([current_x / prephaser_duration_s]),
                prephaser_duration_s,
            )
        )
    if current_y != 0:
        events.append(
            GradientEvent(
                "y",
                prephaser_start,
                np.asarray([current_y / prephaser_duration_s]),
                prephaser_duration_s,
            )
        )

    time_s = prephaser_start + prephaser_duration_s + delay_after_prephaser_s
    for line in range(ny):
        direction = acquisition.readout_directions[line]
        read_gradient = direction / (fov_x * dwell)
        events.append(
            GradientEvent("x", time_s, np.asarray([read_gradient]), readout_duration)
        )
        events.append(
            ADCEvent(
                start_s=time_s + dwell / 2,
                num_samples=nx,
                dwell_s=dwell,
            )
        )
        current_x += direction * nx / fov_x
        time_s += readout_duration

        if line == ny - 1:
            continue
        next_direction = acquisition.readout_directions[line + 1]
        next_first_kx = (
            acquisition.kx_cyc_per_m[0]
            if next_direction > 0
            else acquisition.kx_cyc_per_m[-1]
        )
        desired_x = next_first_kx - next_direction * 0.5 / fov_x
        next_phase_index = acquisition.phase_indices[line + 1]
        desired_y = acquisition.ky_cyc_per_m[next_phase_index]
        x_area = desired_x - current_x
        y_area = desired_y - current_y
        if x_area != 0:
            events.append(
                GradientEvent(
                    "x",
                    time_s,
                    np.asarray([x_area / blip_duration_s]),
                    blip_duration_s,
                )
            )
        if y_area != 0:
            events.append(
                GradientEvent(
                    "y",
                    time_s,
                    np.asarray([y_area / blip_duration_s]),
                    blip_duration_s,
                )
            )
        current_x = desired_x
        current_y = desired_y
        time_s += blip_duration_s

    return SequenceProgram(
        events=tuple(events),
        duration_s=time_s + tail_s,
        source="internal-cartesian-epi",
        metadata={"acquisition": acquisition.to_metadata()},
    )


def _positive_integer(value: int, name: str) -> int:
    if int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
