"""Cartesian acquisition layout, reconstruction, and reference sequence builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from types import SimpleNamespace
from typing import ClassVar, Mapping, Optional, Sequence, Tuple

import numpy as np

from .flip_angles import VFA_REFERENCE_DOI, variable_flip_angle_schedule
from .encoding import EncodingFrame, numeric_definition_array
from .model import ADCEvent, GradientEvent, RFEvent, SequenceProgram
from .rf_pulses import design_rf_envelope, scale_rf_envelope_to_flip


# Pulseq's text serialization rounds gradient amplitudes. In long balanced
# trains the resulting sub-microcycle residual can accumulate across lines,
# even though it remains far below one percent of a Cartesian grid cell.
_CARTESIAN_GRID_TOLERANCE_CELLS = 1e-2


def _cartesian_grid_tolerance(fov_m: float) -> float:
    return max(1e-9, _CARTESIAN_GRID_TOLERANCE_CELLS / float(fov_m))


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
class SpiralAcquisition:
    """Single-shot 2D spiral frames within one chronological ADC stream.

    The trajectory itself remains in the compiled/result ADC gradient moments.
    This object records how those samples are split into slices and repetitions
    and provides a lightweight linear gridding reconstruction.
    """

    matrix: Tuple[int, int]
    fov_m: Tuple[float, float]
    dwell_s: float
    sample_indices: Tuple[Tuple[int, ...], ...]
    frame_indices: Tuple[Tuple[int, ...], ...]
    dimensions: AcquisitionDimensions
    moment_origins_cyc_per_m: Tuple[Tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        matrix = tuple(
            _positive_integer(value, "spiral matrix size") for value in self.matrix
        )
        if len(matrix) != 2:
            raise ValueError("spiral matrix must contain x and y sizes")
        fov = tuple(float(value) for value in self.fov_m)
        if len(fov) != 2 or not np.all(np.isfinite(fov)) or min(fov) <= 0:
            raise ValueError("spiral fov_m must contain two positive finite values")
        dwell = float(self.dwell_s)
        if not np.isfinite(dwell) or dwell <= 0:
            raise ValueError("spiral dwell_s must be positive and finite")
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
        if not samples or len(samples) != len(frames) or len(samples) != len(origins):
            raise ValueError(
                "spiral frame metadata must contain equal non-zero lengths"
            )
        if any(len(item) != matrix[0] * matrix[1] for item in samples):
            raise ValueError(
                "each spiral frame must contain matrix_x * matrix_y samples"
            )
        if any(len(item) != len(AcquisitionDimensions.AXIS_NAMES) for item in frames):
            raise ValueError("each spiral frame index must contain all outer axes")
        if any(len(item) != 3 or not np.all(np.isfinite(item)) for item in origins):
            raise ValueError("each spiral frame requires one finite 3D moment origin")
        flattened = [value for item in samples for value in item]
        if sorted(flattened) != list(range(self.dimensions.num_samples)):
            raise ValueError("spiral frames must cover the complete ADC stream once")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "fov_m", fov)
        object.__setattr__(self, "dwell_s", dwell)
        object.__setattr__(self, "sample_indices", samples)
        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "moment_origins_cyc_per_m", origins)

    @property
    def num_frames(self) -> int:
        return len(self.sample_indices)

    @property
    def samples_per_frame(self) -> int:
        return self.matrix[0] * self.matrix[1]

    @property
    def num_samples(self) -> int:
        return self.dimensions.num_samples

    @property
    def sampling_bandwidth_hz(self) -> float:
        return 1.0 / self.dwell_s

    @property
    def varying_axes(self) -> Tuple[str, ...]:
        return tuple(
            axis
            for index, axis in enumerate(AcquisitionDimensions.AXIS_NAMES)
            if len({frame[index] for frame in self.frame_indices}) > 1
        )

    @property
    def kx_grid_cyc_per_m(self) -> np.ndarray:
        return (
            np.arange(self.matrix[0], dtype=float) - self.matrix[0] // 2
        ) / self.fov_m[0]

    @property
    def ky_grid_cyc_per_m(self) -> np.ndarray:
        return (
            np.arange(self.matrix[1], dtype=float) - self.matrix[1] // 2
        ) / self.fov_m[1]

    def frame_label(self, frame: int) -> str:
        values = self.frame_indices[int(frame)]
        axes = self.varying_axes
        if not axes:
            return "single spiral frame"
        return ", ".join(
            f"{axis}={values[AcquisitionDimensions.AXIS_NAMES.index(axis)]}"
            for axis in axes
        )

    def _frame_values(self, values, frame: int) -> np.ndarray:
        array = np.asarray(values)
        if array.shape[-1] != self.num_samples:
            raise ValueError("values do not match the complete spiral ADC stream")
        return np.take(array, self.sample_indices[int(frame)], axis=-1)

    def trajectory(self, moments_cyc_per_m, frame: int) -> np.ndarray:
        moments = np.asarray(moments_cyc_per_m, dtype=float)
        if moments.shape != (self.num_samples, 3):
            raise ValueError("spiral gradient moments must have shape (num_samples, 3)")
        selected = np.take(moments, self.sample_indices[int(frame)], axis=0)
        relative = selected - np.asarray(
            self.moment_origins_cyc_per_m[int(frame)], dtype=float
        )
        if not np.all(np.isfinite(relative)):
            raise ValueError("spiral trajectory contains non-finite values")
        if np.ptp(relative[:, 0]) <= 0 or np.ptp(relative[:, 1]) <= 0:
            raise ValueError("spiral trajectory must vary on both in-plane axes")
        return relative

    def grid_kspace(self, result, frame: int) -> np.ndarray:
        """Linearly grid one non-Cartesian frame onto its nominal matrix."""
        if result.adc_gradient_moment_cyc_per_m is None:
            raise ValueError("spiral gridding requires ADC gradient moments")
        trajectory = self.trajectory(result.adc_gradient_moment_cyc_per_m, frame)
        values = self._frame_values(result.signal, frame)
        from scipy.interpolate import griddata

        target_x, target_y = np.meshgrid(
            self.kx_grid_cyc_per_m,
            self.ky_grid_cyc_per_m,
            indexing="xy",
        )
        points = trajectory[:, :2]
        leading_shape = values.shape[:-1]
        flattened = values.reshape((-1, self.samples_per_frame))
        grids = []
        for channel in flattened:
            try:
                grid = griddata(
                    points,
                    channel,
                    (target_x, target_y),
                    method="linear",
                    fill_value=0.0,
                )
            except Exception:
                grid = griddata(
                    points,
                    channel,
                    (target_x, target_y),
                    method="nearest",
                    fill_value=0.0,
                )
            grids.append(np.asarray(grid))
        return np.stack(grids).reshape(leading_shape + (self.matrix[1], self.matrix[0]))

    def reconstruct(
        self,
        result,
        frame: int,
        *,
        norm: Optional[str] = None,
        coil_combine: Optional[str] = None,
    ) -> np.ndarray:
        """Grid and reconstruct one spiral frame with a centred 2D IFFT."""
        kspace = self.grid_kspace(result, frame)
        image = np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1), norm=norm
            ),
            axes=(-2, -1),
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

    def to_metadata(self) -> dict:
        return {
            "type": "spiral_2d_frames",
            "matrix": self.matrix,
            "fov_m": self.fov_m,
            "dwell_s": self.dwell_s,
            "sample_indices": self.sample_indices,
            "frame_indices": self.frame_indices,
            "dimensions": self.dimensions.to_metadata(),
            "moment_origins_cyc_per_m": self.moment_origins_cyc_per_m,
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> "SpiralAcquisition":
        if metadata.get("type") != "spiral_2d_frames":
            raise ValueError("unsupported spiral acquisition metadata")
        return cls(
            matrix=tuple(metadata["matrix"]),
            fov_m=tuple(metadata["fov_m"]),
            dwell_s=metadata["dwell_s"],
            sample_indices=tuple(tuple(item) for item in metadata["sample_indices"]),
            frame_indices=tuple(tuple(item) for item in metadata["frame_indices"]),
            dimensions=AcquisitionDimensions.from_metadata(metadata["dimensions"]),
            moment_origins_cyc_per_m=tuple(
                tuple(item) for item in metadata["moment_origins_cyc_per_m"]
            ),
        )


@dataclass(frozen=True)
class CartesianAcquisitionVolumes:
    """Validated Cartesian 3D volumes assembled from sorted 2D k-space planes.

    ``volume_frame_indices`` contains one frame index for every increasing kz
    plane. ``volume_indices`` retains the non-spatial Pulseq dimensions in the
    canonical order ``(slice, echo, repetition, segment)``.  The chronological
    ADC stream therefore remains untouched while an explicit index mapping
    provides arrays shaped as ``(..., partition_z, phase_y, read_x)``.
    """

    OUTER_AXIS_NAMES: ClassVar[Tuple[str, ...]] = tuple(
        axis for axis in AcquisitionDimensions.AXIS_NAMES if axis != "partition"
    )

    frames: CartesianAcquisitionFrames
    volume_frame_indices: Tuple[Tuple[int, ...], ...]
    volume_indices: Tuple[Tuple[int, ...], ...]
    partition_matrix: int
    fov_z_m: float
    kz_offset_cells: float = 0.0

    def __post_init__(self) -> None:
        frame_indices = tuple(
            tuple(int(value) for value in item) for item in self.volume_frame_indices
        )
        volume_indices = tuple(
            tuple(int(value) for value in item) for item in self.volume_indices
        )
        partition_matrix = _positive_integer(self.partition_matrix, "partition_matrix")
        fov_z = float(self.fov_z_m)
        kz_offset = float(self.kz_offset_cells)
        if not np.isfinite(fov_z) or fov_z <= 0:
            raise ValueError("fov_z_m must be positive and finite")
        if not np.isfinite(kz_offset):
            raise ValueError("kz_offset_cells must be finite")
        if (
            not frame_indices
            or len(frame_indices) != len(volume_indices)
            or any(len(item) != partition_matrix for item in frame_indices)
        ):
            raise ValueError(
                "Cartesian volumes require one complete ordered frame set per volume"
            )
        if any(len(item) != len(self.OUTER_AXIS_NAMES) for item in volume_indices):
            raise ValueError("each Cartesian volume index must contain all outer axes")
        if len(set(volume_indices)) != len(volume_indices):
            raise ValueError("Cartesian volume indices must be unique")
        flattened = [value for item in frame_indices for value in item]
        if sorted(flattened) != list(range(self.frames.num_frames)):
            raise ValueError("Cartesian volumes must use every 2D frame exactly once")

        first = self.frames.acquisitions[0]
        tolerance_x = _cartesian_grid_tolerance(first.fov_m[0])
        tolerance_y = _cartesian_grid_tolerance(first.fov_m[1])
        for acquisition in self.frames.acquisitions[1:]:
            if (
                acquisition.read_matrix != first.read_matrix
                or acquisition.phase_matrix != first.phase_matrix
                or acquisition.encoding_frame != first.encoding_frame
                or not np.allclose(acquisition.fov_m, first.fov_m)
                or not np.allclose(
                    acquisition.kx_cyc_per_m,
                    first.kx_cyc_per_m,
                    rtol=0.0,
                    atol=tolerance_x,
                )
                or not np.allclose(
                    acquisition.ky_cyc_per_m,
                    first.ky_cyc_per_m,
                    rtol=0.0,
                    atol=tolerance_y,
                )
            ):
                raise ValueError("Cartesian volume planes do not share one xy grid")

        varying = self._varying_axes(volume_indices)
        expected = set(
            product(*[self._axis_values(volume_indices, axis) for axis in varying])
        )
        actual = {
            tuple(item[self.OUTER_AXIS_NAMES.index(axis)] for axis in varying)
            for item in volume_indices
        }
        if actual != expected:
            raise ValueError(
                "Cartesian volumes do not cover the outer acquisition grid exactly once"
            )

        object.__setattr__(self, "volume_frame_indices", frame_indices)
        object.__setattr__(self, "volume_indices", volume_indices)
        object.__setattr__(self, "partition_matrix", partition_matrix)
        object.__setattr__(self, "fov_z_m", fov_z)
        object.__setattr__(self, "kz_offset_cells", kz_offset)

    @staticmethod
    def _varying_axes(volume_indices) -> Tuple[str, ...]:
        return tuple(
            axis
            for index, axis in enumerate(CartesianAcquisitionVolumes.OUTER_AXIS_NAMES)
            if len({item[index] for item in volume_indices}) > 1
        )

    @staticmethod
    def _axis_values(volume_indices, axis: str) -> Tuple[int, ...]:
        index = CartesianAcquisitionVolumes.OUTER_AXIS_NAMES.index(axis)
        return tuple(sorted({item[index] for item in volume_indices}))

    @property
    def num_volumes(self) -> int:
        return len(self.volume_frame_indices)

    @property
    def varying_axes(self) -> Tuple[str, ...]:
        return self._varying_axes(self.volume_indices)

    def axis_values(self, axis: str) -> Tuple[int, ...]:
        if axis not in self.OUTER_AXIS_NAMES:
            raise ValueError(
                f"axis must be one of {', '.join(self.OUTER_AXIS_NAMES)}, got {axis!r}"
            )
        return self._axis_values(self.volume_indices, axis)

    @property
    def outer_shape(self) -> Tuple[int, ...]:
        return tuple(len(self.axis_values(axis)) for axis in self.varying_axes)

    @property
    def read_matrix(self) -> int:
        return self.frames.acquisitions[0].read_matrix

    @property
    def phase_matrix(self) -> int:
        return self.frames.acquisitions[0].phase_matrix

    @property
    def fov_m(self) -> Tuple[float, float, float]:
        in_plane = self.frames.acquisitions[0].fov_m
        return (in_plane[0], in_plane[1], self.fov_z_m)

    @property
    def encoding_frame(self) -> EncodingFrame:
        return self.frames.acquisitions[0].encoding_frame

    @property
    def matrix(self) -> Tuple[int, int, int]:
        return (self.read_matrix, self.phase_matrix, self.partition_matrix)

    @property
    def kx_cyc_per_m(self) -> np.ndarray:
        return np.mean(
            np.stack([item.kx_cyc_per_m for item in self.frames.acquisitions], axis=0),
            axis=0,
        )

    @property
    def ky_cyc_per_m(self) -> np.ndarray:
        return np.mean(
            np.stack([item.ky_cyc_per_m for item in self.frames.acquisitions], axis=0),
            axis=0,
        )

    @property
    def kz_cyc_per_m(self) -> np.ndarray:
        """Logical partition coordinates (legacy kz-compatible name)."""
        return (
            np.arange(self.partition_matrix, dtype=float)
            - self.partition_matrix // 2
            + self.kz_offset_cells
        ) / self.fov_z_m

    @property
    def k_read_cyc_per_m(self) -> np.ndarray:
        return self.kx_cyc_per_m

    @property
    def k_phase_cyc_per_m(self) -> np.ndarray:
        return self.ky_cyc_per_m

    @property
    def k_partition_cyc_per_m(self) -> np.ndarray:
        return self.kz_cyc_per_m

    @property
    def read_dimension(self) -> str:
        return self.encoding_frame.dimension_name("read")

    @property
    def phase_dimension(self) -> str:
        return self.encoding_frame.dimension_name("phase")

    @property
    def partition_dimension(self) -> str:
        return self.encoding_frame.dimension_name("partition")

    def volume_label(self, volume: int) -> str:
        values = self.volume_indices[int(volume)]
        axes = self.varying_axes or ("volume",)
        if axes == ("volume",):
            return f"volume={int(volume)}"
        return ", ".join(
            f"{axis}={values[self.OUTER_AXIS_NAMES.index(axis)]}" for axis in axes
        )

    def _volume_grid_position(self, volume: int) -> Tuple[int, ...]:
        values = self.volume_indices[int(volume)]
        return tuple(
            self.axis_values(axis).index(values[self.OUTER_AXIS_NAMES.index(axis)])
            for axis in self.varying_axes
        )

    def _validate_frame_kz(self, result, frame: int, z_index: int) -> None:
        if result.adc_gradient_moment_cyc_per_m is None:
            return
        moments = np.take(
            np.asarray(result.adc_gradient_moment_cyc_per_m),
            self.frames.sample_indices[int(frame)],
            axis=0,
        )
        moments = moments - np.asarray(
            self.frames.moment_origins_cyc_per_m[int(frame)], dtype=float
        )
        moments = self.encoding_frame.scanner_to_encoding(moments)
        tolerance = _cartesian_grid_tolerance(self.fov_z_m)
        if not np.allclose(
            moments[:, 2],
            self.kz_cyc_per_m[int(z_index)],
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("partition gradient moments do not match the 3D grid")

    def to_cartesian_kspace(self, result, volume: int) -> np.ndarray:
        """Return one volume as ``(..., partition_z, phase_y, read_x)``."""
        planes = []
        for z_index, frame in enumerate(self.volume_frame_indices[int(volume)]):
            self._validate_frame_kz(result, frame, z_index)
            planes.append(self.frames.to_cartesian_kspace(result, frame))
        return np.stack(planes, axis=-3)

    def reconstruct(
        self,
        result,
        volume: int,
        *,
        norm: Optional[str] = None,
        coil_combine: Optional[str] = None,
        voxel_centered: bool = True,
    ) -> np.ndarray:
        """Reconstruct one validated Cartesian volume with a centred 3D IFFT."""
        kspace = self.to_cartesian_kspace(result, volume)
        if voxel_centered:
            dx = self.fov_m[0] / self.read_matrix
            dy = self.fov_m[1] / self.phase_matrix
            dz = self.fov_m[2] / self.partition_matrix
            centre_phase = np.exp(
                2j
                * np.pi
                * (
                    self.kz_cyc_per_m[:, None, None] * dz / 2
                    + self.ky_cyc_per_m[None, :, None] * dy / 2
                    + self.kx_cyc_per_m[None, None, :] * dx / 2
                )
            )
            kspace = kspace * centre_phase
        axes = (-3, -2, -1)
        image = np.fft.fftshift(
            np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm=norm),
            axes=axes,
        )
        if coil_combine is None:
            return image
        if image.ndim != 4:
            raise ValueError("coil combination requires signal shape (coil, adc)")
        if coil_combine == "rss":
            return np.sqrt(np.sum(np.abs(image) ** 2, axis=0))
        if coil_combine == "sum":
            return np.sum(image, axis=0)
        raise ValueError("coil_combine must be None, 'rss', or 'sum'")

    def dimensioned_kspace(self, result) -> np.ndarray:
        """Return all volumes with explicit non-spatial dimensions."""
        volumes = [
            self.to_cartesian_kspace(result, item) for item in range(self.num_volumes)
        ]
        if not self.varying_axes:
            return volumes[0]
        output = np.empty(self.outer_shape + volumes[0].shape, dtype=volumes[0].dtype)
        for volume, values in enumerate(volumes):
            output[self._volume_grid_position(volume)] = values
        return output

    def dimensioned_reconstruction(
        self,
        result,
        *,
        norm: Optional[str] = None,
        coil_combine: Optional[str] = None,
        voxel_centered: bool = True,
    ) -> np.ndarray:
        """Return reconstructed volumes with explicit non-spatial dimensions."""
        images = [
            self.reconstruct(
                result,
                item,
                norm=norm,
                coil_combine=coil_combine,
                voxel_centered=voxel_centered,
            )
            for item in range(self.num_volumes)
        ]
        if not self.varying_axes:
            return images[0]
        output = np.empty(self.outer_shape + images[0].shape, dtype=images[0].dtype)
        for volume, values in enumerate(images):
            output[self._volume_grid_position(volume)] = values
        return output

    def to_metadata(self) -> dict:
        return {
            "type": "cartesian_3d_volumes",
            "frames": self.frames.to_metadata(),
            "volume_frame_indices": self.volume_frame_indices,
            "volume_indices": self.volume_indices,
            "outer_axis_names": self.OUTER_AXIS_NAMES,
            "partition_matrix": self.partition_matrix,
            "fov_z_m": self.fov_z_m,
            "kz_offset_cells": self.kz_offset_cells,
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> "CartesianAcquisitionVolumes":
        if metadata.get("type") != "cartesian_3d_volumes":
            raise ValueError("unsupported Cartesian volume metadata")
        return cls(
            frames=CartesianAcquisitionFrames.from_metadata(metadata["frames"]),
            volume_frame_indices=tuple(
                tuple(int(value) for value in item)
                for item in metadata["volume_frame_indices"]
            ),
            volume_indices=tuple(
                tuple(int(value) for value in item)
                for item in metadata["volume_indices"]
            ),
            partition_matrix=metadata["partition_matrix"],
            fov_z_m=metadata["fov_z_m"],
            kz_offset_cells=metadata.get("kz_offset_cells", 0.0),
        )


@dataclass(frozen=True)
class SpectroscopicAcquisition:
    """Map repeated phase-encoded CSI data to spatial grids plus FIDs.

    Unlike a Cartesian imaging readout, every ADC event is an FID acquired at
    one fixed spatial k-space coordinate.  Treating the spectral samples as a
    readout axis would therefore produce a physically invalid reconstruction.
    For multiple repetitions the returned arrays have an explicit leading
    ``repetition`` dimension before ``(ky, kx, spectral_point)``.
    """

    matrix: Tuple[int, int]
    fov_m: Tuple[float, float]
    spectral_points: int
    dwell_s: float
    encoding_indices: Tuple[Tuple[int, int], ...]
    repetition_indices: Tuple[int, ...] = ()
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
        repetitions = tuple(int(value) for value in self.repetition_indices)
        if not repetitions and indices:
            repetitions = (0,) * len(indices)
        if len(repetitions) != len(indices) or any(value < 0 for value in repetitions):
            raise ValueError("CSI requires one non-negative repetition index per FID")
        repetition_values = tuple(sorted(set(repetitions)))
        if not repetition_values:
            raise ValueError("CSI requires at least one complete repetition")
        if repetition_values != tuple(range(len(repetition_values))):
            raise ValueError("CSI repetition indices must be contiguous from zero")
        if len(indices) != matrix[0] * matrix[1] * len(repetition_values):
            raise ValueError("CSI FID count does not match its repetitions and matrix")
        for repetition in repetition_values:
            repetition_grid = {
                index
                for index, value in zip(indices, repetitions)
                if value == repetition
            }
            if repetition_grid != expected or repetitions.count(repetition) != len(
                expected
            ):
                raise ValueError(
                    "each CSI repetition must cover the spatial grid exactly once"
                )
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
        object.__setattr__(self, "repetition_indices", repetitions)
        object.__setattr__(self, "moment_origins_cyc_per_m", origins)

    @property
    def num_encodings(self) -> int:
        return len(self.encoding_indices)

    @property
    def num_repetitions(self) -> int:
        return len(set(self.repetition_indices))

    @property
    def encodings_per_repetition(self) -> int:
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
        """Return chronological data with an explicit repetition dimension."""
        values = np.asarray(signal)
        if values.ndim < 1 or values.shape[-1] != self.num_samples:
            raise ValueError(
                f"signal must end with {self.num_samples} chronological CSI samples"
            )
        raw = values.reshape(
            values.shape[:-1] + (self.num_encodings, self.spectral_points)
        )
        grid = np.empty(
            values.shape[:-1]
            + (
                self.num_repetitions,
                self.matrix[1],
                self.matrix[0],
                self.spectral_points,
            ),
            dtype=values.dtype,
        )
        for acquired, ((x_index, y_index), repetition) in enumerate(
            zip(self.encoding_indices, self.repetition_indices)
        ):
            grid[..., repetition, y_index, x_index, :] = raw[..., acquired, :]
        return grid[..., 0, :, :, :] if self.num_repetitions == 1 else grid

    def encoding_event_index(self, repetition: int, x_index: int, y_index: int) -> int:
        """Return the chronological FID index for one CSI grid location."""
        target = (int(x_index), int(y_index))
        repetition = int(repetition)
        for event, (index, value) in enumerate(
            zip(self.encoding_indices, self.repetition_indices)
        ):
            if index == target and value == repetition:
                return event
        raise ValueError("CSI repetition/grid location is not present")

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
            "repetition_indices": self.repetition_indices,
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
            repetition_indices=tuple(metadata.get("repetition_indices", ())),
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
    encoding_frame: EncodingFrame = field(default_factory=EncodingFrame.identity)
    moment_origins_cyc_per_m: Tuple[Tuple[float, float, float], ...] = ()

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
        if not isinstance(self.encoding_frame, EncodingFrame):
            raise TypeError("encoding_frame must be an EncodingFrame")
        moment_origins = tuple(
            tuple(float(value) for value in origin)
            for origin in self.moment_origins_cyc_per_m
        )
        if not moment_origins:
            moment_origins = tuple((0.0, 0.0, 0.0) for _ in range(phase_matrix))
        if len(moment_origins) != phase_matrix or any(
            len(origin) != 3 or not np.all(np.isfinite(origin))
            for origin in moment_origins
        ):
            raise ValueError(
                "Cartesian acquisition requires one finite 3D moment origin per line"
            )

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
        object.__setattr__(self, "moment_origins_cyc_per_m", moment_origins)

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
        """Logical read coordinates (legacy name for x-readout compatibility)."""
        return (
            np.arange(self.read_matrix, dtype=float)
            - self.read_matrix // 2
            + self.kx_offset_cells
        ) / self.fov_m[0]

    @property
    def ky_cyc_per_m(self) -> np.ndarray:
        """Logical phase coordinates (legacy name for y-phase compatibility)."""
        return (
            np.arange(self.phase_matrix, dtype=float)
            - self.phase_matrix // 2
            + self.ky_offset_cells
        ) / self.fov_m[1]

    @property
    def k_read_cyc_per_m(self) -> np.ndarray:
        return self.kx_cyc_per_m

    @property
    def k_phase_cyc_per_m(self) -> np.ndarray:
        return self.ky_cyc_per_m

    @property
    def read_dimension(self) -> str:
        return self.encoding_frame.dimension_name("read")

    @property
    def phase_dimension(self) -> str:
        return self.encoding_frame.dimension_name("phase")

    def reshape_signal(self, signal: np.ndarray) -> np.ndarray:
        """Map chronological ADC data to ``(..., phase, read)`` k-space."""
        values = np.asarray(signal)
        if values.ndim < 1 or values.shape[-1] != self.num_samples:
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
        raw = raw - np.asarray(self.moment_origins_cyc_per_m)[:, None, :]
        raw = self.encoding_frame.scanner_to_encoding(raw).reshape(
            self.phase_matrix, self.read_matrix, 3
        )
        for acquired_line, phase_index in enumerate(self.phase_indices):
            expected_x = self.kx_cyc_per_m
            if self.readout_directions[acquired_line] < 0:
                expected_x = expected_x[::-1]
            expected_y = self.ky_cyc_per_m[phase_index]
            x_tolerance = _cartesian_grid_tolerance(self.fov_m[0])
            y_tolerance = _cartesian_grid_tolerance(self.fov_m[1])
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
            "encoding_frame": self.encoding_frame.to_metadata(),
            "moment_origins_cyc_per_m": self.moment_origins_cyc_per_m,
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
            encoding_frame=EncodingFrame.from_metadata(
                metadata.get("encoding_frame", {})
            ),
            moment_origins_cyc_per_m=tuple(
                tuple(value) for value in metadata.get("moment_origins_cyc_per_m", ())
            ),
        )


def infer_spiral_acquisition(
    program: SequenceProgram,
    *,
    compiled=None,
) -> SpiralAcquisition:
    """Infer generated single-interleaf 2D spiral frames from ADC moments."""
    from .compiler import SequenceCompiler

    definitions = {
        str(key).lower(): value
        for key, value in dict(program.metadata.get("definitions", {})).items()
    }
    trajectory_name = str(definitions.get("trajectory", "")).strip().lower()
    sequence_name = str(definitions.get("name", "")).strip().lower()
    if trajectory_name != "spiral" and "spiral" not in sequence_name:
        raise ValueError("sequence is not declared as a spiral acquisition")
    matrix_value = numeric_definition_array(
        definitions.get("matrixsize", ()), "spiral MatrixSize"
    )
    fov_value = numeric_definition_array(definitions.get("fov", ()), "spiral FOV")
    if matrix_value.size < 2:
        raise ValueError("spiral acquisition requires a 2D MatrixSize definition")
    if fov_value.size < 2:
        raise ValueError("spiral acquisition requires a 2D FOV definition")
    matrix = tuple(
        _positive_integer(value, "spiral MatrixSize") for value in matrix_value[:2]
    )
    fov = tuple(float(value) for value in fov_value[:2])
    if not np.all(np.isfinite(fov)) or min(fov) <= 0:
        raise ValueError("spiral FOV values must be positive and finite")

    adc_events = program.adc_events
    if not adc_events:
        raise ValueError("spiral sequence contains no ADC events")
    samples_per_frame = matrix[0] * matrix[1]
    if any(event.num_samples != samples_per_frame for event in adc_events):
        raise ValueError(
            "each spiral ADC event must contain matrix_x * matrix_y samples"
        )
    dwell = adc_events[0].dwell_s
    if any(
        not np.isclose(event.dwell_s, dwell, rtol=0.0, atol=1e-15)
        for event in adc_events
    ):
        raise ValueError("spiral ADC events do not share one dwell time")

    dimensions = AcquisitionDimensions.from_program(program)
    if dimensions.num_adc_events != len(adc_events):
        raise ValueError("spiral acquisition dimensions do not match ADC events")
    compiled = (
        SequenceCompiler().compile_acquisition(program)
        if compiled is None
        else compiled
    )
    moments = np.asarray(compiled.adc_gradient_moment_cyc_per_m, dtype=float)
    if moments.shape != (dimensions.num_samples, 3):
        raise ValueError("compiled spiral ADC moments have an invalid shape")

    sample_indices = []
    frame_indices = []
    origins = []
    cursor = 0
    for event_index, count in enumerate(dimensions.adc_event_sample_counts):
        indices = tuple(range(cursor, cursor + count))
        sample_indices.append(indices)
        frame_indices.append(
            tuple(
                int(dimensions.event_indices(axis)[event_index])
                for axis in dimensions.AXIS_NAMES
            )
        )
        origins.append(tuple(float(value) for value in moments[cursor]))
        cursor += count

    acquisition = SpiralAcquisition(
        matrix=matrix,
        fov_m=fov,
        dwell_s=dwell,
        sample_indices=tuple(sample_indices),
        frame_indices=tuple(frame_indices),
        dimensions=dimensions,
        moment_origins_cyc_per_m=tuple(origins),
    )
    reference = acquisition.trajectory(moments, 0)
    scale = np.asarray((matrix[0] / (2 * fov[0]), matrix[1] / (2 * fov[1])))
    normalized_radius = np.hypot(reference[:, 0] / scale[0], reference[:, 1] / scale[1])
    if float(np.max(normalized_radius)) < 0.75:
        raise ValueError("spiral trajectory does not reach the declared matrix extent")
    for frame in range(1, acquisition.num_frames):
        candidate = acquisition.trajectory(moments, frame)
        tolerance = max(1e-8, 1e-5 * float(np.max(np.abs(reference))))
        if not np.allclose(
            candidate[:, :2], reference[:, :2], rtol=0.0, atol=tolerance
        ):
            raise ValueError("spiral trajectory changes between acquisition frames")
    return acquisition


def infer_cartesian_acquisition(
    program: SequenceProgram,
    *,
    compiled=None,
    moment_origins_cyc_per_m=None,
) -> CartesianAcquisition:
    """Infer one regular 2D Cartesian acquisition from a sequence program.

    The conservative inference accepts one chronological ADC event per phase
    line and an explicit Pulseq FOV definition. Generated-sequence orientation
    metadata projects physical scanner moments into logical read/phase/
    partition coordinates before the regular grid is validated. Multi-slice,
    repeated, segmented, or non-Cartesian streams are rejected instead of
    being reshaped ambiguously.
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
    encoding_frame = EncodingFrame.from_definitions(definitions)
    fov_value = next(
        (
            value
            for key, value in definitions.items()
            if str(key).lower() == "encodingfov"
        ),
        None,
    )
    if fov_value is None:
        fov_value = next(
            (value for key, value in definitions.items() if str(key).lower() == "fov"),
            None,
        )
    if fov_value is None:
        raise ValueError("Pulseq sequence has no FOV definition")
    fov = numeric_definition_array(fov_value, "Cartesian FOV")
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
    if moment_origins_cyc_per_m is None:
        moment_origins_cyc_per_m = _adc_gradient_moment_origins(
            program, compiled, adc_events
        )
    moment_origins = np.asarray(moment_origins_cyc_per_m, dtype=float)
    if moment_origins.shape != (phase_matrix, 3) or not np.all(
        np.isfinite(moment_origins)
    ):
        raise ValueError(
            "Cartesian inference requires one finite 3D moment origin per ADC line"
        )
    relative_moments = moments.reshape(phase_matrix, read_matrix, 3)
    relative_moments = relative_moments - moment_origins[:, None, :]
    raw = encoding_frame.scanner_to_encoding(relative_moments).reshape(
        phase_matrix, read_matrix, 3
    )
    tolerance_x = _cartesian_grid_tolerance(fov_x)
    tolerance_y = _cartesian_grid_tolerance(fov_y)

    delta_x = np.diff(raw[:, :, 0], axis=1)
    mean_delta_x = np.mean(delta_x, axis=1)
    if np.any(np.abs(mean_delta_x) <= tolerance_x):
        raise ValueError("ADC events do not contain a readout gradient")
    directions = np.where(mean_delta_x > 0, 1, -1)
    expected_delta_x = directions[:, None] / fov_x
    if not np.allclose(delta_x, expected_delta_x, rtol=0.0, atol=tolerance_x):
        raise ValueError("ADC readout samples are not on a regular read grid")
    if not np.allclose(np.diff(raw[:, :, 1], axis=1), 0.0, rtol=0.0, atol=tolerance_y):
        raise ValueError("phase gradient changes during an ADC line")
    z_scale = max(1.0, float(np.max(np.abs(raw[:, :, 2]))))
    if not np.allclose(
        np.diff(raw[:, :, 2], axis=1), 0.0, rtol=0.0, atol=1e-9 * z_scale
    ):
        raise ValueError("partition gradient changes during an ADC line")

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
        encoding_frame=encoding_frame,
        moment_origins_cyc_per_m=tuple(
            tuple(float(value) for value in origin) for origin in moment_origins
        ),
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
    matrix_values = numeric_definition_array(matrix_value, "CSI MatrixSize")
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
    repetitions_value = definitions.get("repetitions", 1)
    repetitions = _positive_integer(repetitions_value, "CSI Repetitions")
    if len(adc_events) != nx * ny * repetitions:
        raise ValueError(
            "ADC event count does not match the declared CSI grid and repetitions"
        )
    if any(event.num_samples != spectral_points for event in adc_events):
        raise ValueError("ADC event sizes do not match CSI SpectralPoints")
    dwell_s = adc_events[0].dwell_s
    if any(
        not np.isclose(event.dwell_s, dwell_s, rtol=0.0, atol=1e-15)
        for event in adc_events
    ):
        raise ValueError("CSI ADC events do not have a common spectral dwell")

    fov = numeric_definition_array(fov_value, "CSI FOV")
    if fov.size < 2 or not np.all(np.isfinite(fov[:2])) or np.any(fov[:2] <= 0):
        raise ValueError("CSI FOV definition does not contain valid x/y values")
    labels = program.metadata.get("adc_label_values", {})
    lin = tuple(labels.get("LIN", ()))
    par = tuple(labels.get("PAR", ()))
    if len(lin) != len(adc_events) or len(par) != len(adc_events):
        raise ValueError("CSI inference requires one LIN and PAR label per FID")
    encoding_indices = tuple((int(x), int(y)) for x, y in zip(lin, par))
    if repetitions == 1:
        repetition_indices = (0,) * len(adc_events)
    else:
        repetition_indices = tuple(int(value) for value in labels.get("REP", ()))
        if len(repetition_indices) != len(adc_events):
            raise ValueError("repeated CSI requires one REP label per FID")

    compiled = SequenceCompiler().compile(program) if compiled is None else compiled
    origins = tuple(
        tuple(float(value) for value in origin)
        for origin in _adc_gradient_moment_origins(program, compiled, adc_events)
    )
    acquisition = SpectroscopicAcquisition(
        matrix=(nx, ny),
        fov_m=(float(fov[0]), float(fov[1])),
        spectral_points=spectral_points,
        dwell_s=dwell_s,
        encoding_indices=encoding_indices,
        repetition_indices=repetition_indices,
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
    recurring offsets additionally identify repetitions. Otherwise the groups
    are repetitions. Every group is independently passed through the strict
    single-frame Cartesian validator.
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
    adc_moment_origins = _adc_gradient_moment_origins(program, compiled, adc_events)
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
        moment_origin = adc_moment_origins[event_indices[0]]
        line_moment_origins = tuple(
            tuple(
                float(value)
                for value in (adc_moment_origins[event_index] - moment_origin)
            )
            for event_index in event_indices
        )
        subset_compiled = SimpleNamespace(
            adc_times_s=np.take(compiled.adc_times_s, sample_indices),
            adc_gradient_moment_cyc_per_m=(
                np.take(compiled.adc_gradient_moment_cyc_per_m, sample_indices, axis=0)
                - moment_origin
            ),
        )
        acquisitions.append(
            infer_cartesian_acquisition(
                subset_program,
                compiled=subset_compiled,
                moment_origins_cyc_per_m=line_moment_origins,
            )
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


def infer_cartesian_acquisition_volumes(
    program: SequenceProgram,
    *,
    compiled=None,
    frames: Optional[CartesianAcquisitionFrames] = None,
) -> CartesianAcquisitionVolumes:
    """Infer complete Cartesian 3D volumes from ADC k-space coordinates.

    The Pulseq ``MatrixSize`` and ``FOV`` definitions declare the intended
    regular grid.  Chronological 2D frames are sorted by their RF-relative kz
    moments, while ``PAR`` labels are retained only as a consistency check.
    Non-spatial dimensions such as repetition remain explicit outer axes.
    """
    from .compiler import SequenceCompiler

    definitions = {
        str(key).lower(): value
        for key, value in dict(program.metadata.get("definitions", {})).items()
    }
    matrix_value = definitions.get("encodingmatrixsize", definitions.get("matrixsize"))
    fov_value = definitions.get("encodingfov", definitions.get("fov"))
    if matrix_value is None or fov_value is None:
        raise ValueError(
            "3D Cartesian inference requires MatrixSize and FOV definitions"
        )
    matrix = numeric_definition_array(matrix_value, "3D Cartesian MatrixSize")
    fov = numeric_definition_array(fov_value, "3D Cartesian FOV")
    if matrix.size < 3:
        raise ValueError("3D Cartesian MatrixSize must contain x, y, and z sizes")
    if fov.size < 3 or not np.all(np.isfinite(fov[:3])) or np.any(fov[:3] <= 0):
        raise ValueError("3D Cartesian FOV must contain positive x, y, and z values")
    read_matrix, phase_matrix, partition_matrix = (
        _positive_integer(value, "3D Cartesian MatrixSize") for value in matrix[:3]
    )
    if partition_matrix < 2:
        raise ValueError("3D Cartesian inference requires at least two kz partitions")

    compiled = SequenceCompiler().compile(program) if compiled is None else compiled
    frames = (
        infer_cartesian_acquisition_frames(program, compiled=compiled)
        if frames is None
        else frames
    )
    if frames.dimensions.num_samples != compiled.adc_times_s.size:
        raise ValueError("Cartesian frames do not match the compiled ADC stream")
    for acquisition in frames.acquisitions:
        if (
            acquisition.read_matrix != read_matrix
            or acquisition.phase_matrix != phase_matrix
        ):
            raise ValueError("2D frame grid does not match the declared 3D MatrixSize")
        if not np.allclose(acquisition.fov_m, fov[:2], rtol=0.0, atol=1e-12):
            raise ValueError("2D frame FOV does not match the declared 3D FOV")

    moments = np.asarray(compiled.adc_gradient_moment_cyc_per_m, dtype=float)
    if moments.shape != (frames.dimensions.num_samples, 3):
        raise ValueError("compiled ADC gradient moments have an invalid shape")

    partition_axis = AcquisitionDimensions.AXIS_NAMES.index("partition")
    outer_axes = tuple(
        index
        for index, axis in enumerate(AcquisitionDimensions.AXIS_NAMES)
        if axis != "partition"
    )
    grouped_frames = {}
    for frame, frame_index in enumerate(frames.frame_indices):
        outer_index = tuple(frame_index[index] for index in outer_axes)
        grouped_frames.setdefault(outer_index, []).append(frame)
    if any(len(items) != partition_matrix for items in grouped_frames.values()):
        raise ValueError(
            "each Cartesian volume must contain the declared number of kz partitions"
        )

    tolerance_z = _cartesian_grid_tolerance(float(fov[2]))
    volume_frame_indices = []
    volume_indices = []
    volume_kz = []
    partition_orders = []
    for outer_index, frame_group in grouped_frames.items():
        frame_coordinates = []
        for frame in frame_group:
            relative = np.take(moments, frames.sample_indices[frame], axis=0)
            relative = relative - np.asarray(
                frames.moment_origins_cyc_per_m[frame], dtype=float
            )
            acquisition = frames.acquisitions[frame]
            line_origins = np.repeat(
                np.asarray(acquisition.moment_origins_cyc_per_m, dtype=float),
                acquisition.read_matrix,
                axis=0,
            )
            relative = relative - line_origins
            relative = acquisition.encoding_frame.scanner_to_encoding(relative)
            kz = float(np.mean(relative[:, 2]))
            if not np.allclose(relative[:, 2], kz, rtol=0.0, atol=tolerance_z):
                raise ValueError("kz changes within one Cartesian partition frame")
            frame_coordinates.append((kz, frame))
        frame_coordinates.sort(key=lambda item: item[0])
        sorted_kz = np.asarray([item[0] for item in frame_coordinates], dtype=float)
        if partition_matrix > 1 and not np.allclose(
            np.diff(sorted_kz),
            1.0 / float(fov[2]),
            rtol=0.0,
            atol=tolerance_z,
        ):
            raise ValueError("ADC frames do not form one regular Cartesian kz grid")
        ordered_frames = tuple(item[1] for item in frame_coordinates)
        partition_order = tuple(
            frames.frame_indices[frame][partition_axis] for frame in ordered_frames
        )
        if len(set(partition_order)) != partition_matrix:
            raise ValueError(
                "PAR labels do not identify every kz partition exactly once"
            )
        volume_frame_indices.append(ordered_frames)
        volume_indices.append(tuple(int(value) for value in outer_index))
        volume_kz.append(sorted_kz)
        partition_orders.append(partition_order)

    if any(order != partition_orders[0] for order in partition_orders[1:]):
        raise ValueError("PAR-to-kz ordering changes between Cartesian volumes")
    common_kz = np.mean(np.stack(volume_kz, axis=0), axis=0)
    if not all(
        np.allclose(values, common_kz, rtol=0.0, atol=tolerance_z)
        for values in volume_kz
    ):
        raise ValueError("Cartesian volumes do not share one common kz grid")

    base_z = np.arange(partition_matrix, dtype=float) - partition_matrix // 2
    kz_offset = float(np.mean(common_kz * float(fov[2]) - base_z))
    if not np.allclose(
        common_kz * float(fov[2]),
        base_z + kz_offset,
        rtol=0.0,
        atol=tolerance_z * float(fov[2]),
    ):
        raise ValueError("kz coordinates cannot be represented by one Cartesian grid")
    rounded_kz_offset = round(2.0 * kz_offset) / 2.0
    if np.allclose(
        common_kz * float(fov[2]),
        base_z + rounded_kz_offset,
        rtol=0.0,
        atol=tolerance_z * float(fov[2]),
    ):
        kz_offset = rounded_kz_offset

    volumes = CartesianAcquisitionVolumes(
        frames=frames,
        volume_frame_indices=tuple(volume_frame_indices),
        volume_indices=tuple(volume_indices),
        partition_matrix=partition_matrix,
        fov_z_m=float(fov[2]),
        kz_offset_cells=kz_offset,
    )
    expected = (read_matrix, phase_matrix, partition_matrix)
    if volumes.matrix != expected:
        raise ValueError("inferred Cartesian volume matrix is inconsistent")
    return volumes


def _adc_gradient_moment_origins(
    program: SequenceProgram, compiled, adc_events=None
) -> np.ndarray:
    """Return cumulative scanner moments at the RF preceding each ADC event."""
    adc_events = tuple(program.adc_events if adc_events is None else adc_events)
    origins = np.zeros((len(adc_events), 3), dtype=float)
    rf_events = tuple(program.rf_events)
    if not adc_events or not rf_events:
        return origins

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
    rf_index = -1
    boundary_indices = {}
    for adc_index, adc in enumerate(adc_events):
        while (
            rf_index + 1 < len(rf_events)
            and rf_events[rf_index + 1].end_s <= adc.start_s + 1e-15
        ):
            rf_index += 1
        if rf_index < 0:
            continue
        reference_time = rf_events[rf_index].start_s
        boundary_index = boundary_indices.get(reference_time)
        if boundary_index is None:
            boundary_index = int(np.argmin(np.abs(boundaries - reference_time)))
            tolerance = max(1e-12, abs(reference_time) * 1e-10)
            if abs(boundaries[boundary_index] - reference_time) > tolerance:
                raise ValueError(
                    "RF reference time is not a compiled sequence boundary"
                )
            boundary_indices[reference_time] = boundary_index
        origins[adc_index] = moments[boundary_index]
    return origins


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
    derived = {name: [0] * dimensions.num_adc_events for name in dimensions.AXIS_NAMES}
    rounded_offsets = np.round(offsets, decimals=9)
    unique_offsets = np.unique(rounded_offsets)
    inferred_slice_repetitions = False
    if slice_selective and unique_offsets.size > 1:
        slice_rank = {
            float(value): rank for rank, value in enumerate(sorted(unique_offsets))
        }
        group_slices = {
            group: slice_rank[float(offset)]
            for group, offset in zip(used_groups, rounded_offsets)
        }
        occurrences = {rank: 0 for rank in range(unique_offsets.size)}
        group_repetitions = {}
        for group in used_groups:
            slice_index = group_slices[group]
            group_repetitions[group] = occurrences[slice_index]
            occurrences[slice_index] += 1
        inferred_slice_repetitions = len(set(occurrences.values())) == 1
        if inferred_slice_repetitions:
            derived["slice"] = [group_slices[group] for group in rf_group_for_event]
            derived["repetition"] = [
                group_repetitions[group] for group in rf_group_for_event
            ]
            source = (
                "rf_frequency_offsets"
                if max(group_repetitions.values()) == 0
                else "rf_frequency_offsets_and_repetitions"
            )

    if not inferred_slice_repetitions:
        group_values = {group: index for index, group in enumerate(used_groups)}
        derived["repetition"] = [group_values[group] for group in rf_group_for_event]
        source = "rf_delimited_repetitions"
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
    variable_flip_angle: bool = False,
    vfa_final_flip_angle_deg: float = 90.0,
    rf_pulse_type: str = "block",
    rf_duration_s: float = 1e-3,
    rf_time_bandwidth_product: float = 4.0,
    rf_apodization: float = 0.5,
    rf_slr_sharpness: float = 1.0,
    rf_raster_s: float = 1e-6,
    rf_custom_waveform_hz: Optional[Sequence[complex]] = None,
    rf_custom_raster_s: Optional[float] = None,
    rf_custom_flip_angle_deg: Optional[float] = None,
    rf_custom_name: Optional[str] = None,
    rf_frequency_offset_hz: float = 0.0,
    prephaser_duration_s: float = 1e-3,
    blip_duration_s: float = 100e-6,
    delay_after_prephaser_s: float = 0.0,
    tail_s: float = 0.0,
    n_slices: int = 1,
    slice_thickness_m: Optional[float] = None,
    slice_gap_m: float = 0.0,
    repetitions: int = 1,
    repetition_time_s: Optional[float] = None,
    spoil_after_slice: bool = False,
    spoiler_cycles_per_slice: float = 8.0,
    spoiler_cycles_per_voxel: float = 0.0,
    spoiler_duration_s: float = 4e-3,
) -> SequenceProgram:
    """Build a single-shot Cartesian EPI program.

    Passing ``slice_thickness_m`` enables a slice-selective RF pulse. ``sinc``,
    ``slr``, and ``block`` envelopes are supported. Multiple slices are centred
    around z=0 and played sequentially within each repetition;
    ``slice_gap_m`` is their edge-to-edge separation.
    ``repetition_time_s`` is the time between the
    first slice excitations of consecutive repetitions; when omitted, the
    shortest possible repetition time is used. ``spoil_after_slice`` adds a
    rectangular spoiler after the rewinder. Its z moment is measured in cycles
    across the slice thickness and its optional x/y moment in cycles across an
    acquired voxel. With ``variable_flip_angle`` enabled, one flip angle is
    calculated per repetition and shared by all slices in that repetition.
    """
    for name, value, allow_zero in (
        ("rf_duration_s", rf_duration_s, False),
        ("rf_raster_s", rf_raster_s, False),
        ("prephaser_duration_s", prephaser_duration_s, False),
        ("blip_duration_s", blip_duration_s, False),
        ("delay_after_prephaser_s", delay_after_prephaser_s, True),
        ("tail_s", tail_s, True),
        ("spoiler_duration_s", spoiler_duration_s, False),
    ):
        if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} has an invalid duration")
    if not np.isfinite(flip_angle_deg):
        raise ValueError("flip_angle_deg must be finite")
    if not np.isfinite(rf_frequency_offset_hz):
        raise ValueError("rf_frequency_offset_hz must be finite")
    n_slices = _positive_integer(n_slices, "n_slices")
    repetitions = _positive_integer(repetitions, "repetitions")
    if slice_thickness_m is None:
        if n_slices != 1:
            raise ValueError("multiple slices require slice_thickness_m")
        slice_positions = np.asarray([0.0])
    else:
        slice_thickness_m = float(slice_thickness_m)
        if not np.isfinite(slice_thickness_m) or slice_thickness_m <= 0:
            raise ValueError("slice_thickness_m must be positive and finite")
        if not np.isfinite(slice_gap_m) or slice_gap_m < 0:
            raise ValueError("slice_gap_m must be finite and non-negative")
        slice_spacing = slice_thickness_m + float(slice_gap_m)
        slice_positions = (
            np.arange(n_slices, dtype=float) - (n_slices - 1) / 2.0
        ) * slice_spacing
    if repetition_time_s is not None and (
        not np.isfinite(repetition_time_s) or repetition_time_s <= 0
    ):
        raise ValueError("repetition_time_s must be positive and finite")
    if not np.isfinite(spoiler_cycles_per_slice) or spoiler_cycles_per_slice < 0:
        raise ValueError("spoiler_cycles_per_slice must be finite and non-negative")
    if not np.isfinite(spoiler_cycles_per_voxel) or spoiler_cycles_per_voxel < 0:
        raise ValueError("spoiler_cycles_per_voxel must be finite and non-negative")
    if spoil_after_slice and spoiler_cycles_per_slice > 0 and slice_thickness_m is None:
        raise ValueError(
            "a through-slice spoiler requires slice_thickness_m; set "
            "spoiler_cycles_per_slice to zero for an in-plane-only spoiler"
        )

    nx = acquisition.read_matrix
    ny = acquisition.phase_matrix
    fov_x, fov_y = acquisition.fov_m
    dwell = acquisition.dwell_s
    readout_duration = nx * dwell
    if variable_flip_angle:
        flip_angle_schedule_deg = variable_flip_angle_schedule(
            repetitions,
            final_flip_angle_deg=vfa_final_flip_angle_deg,
        )
    else:
        flip_angle_schedule_deg = np.full(repetitions, flip_angle_deg, dtype=float)
    requested_rf_duration_s = float(rf_duration_s)
    rf_envelope, rf_duration_s, effective_rf_tbw, rf_pulse_type = design_rf_envelope(
        pulse_type=rf_pulse_type,
        duration_s=rf_duration_s,
        raster_s=rf_raster_s,
        time_bandwidth_product=rf_time_bandwidth_product,
        apodization=rf_apodization,
        slr_sharpness=rf_slr_sharpness,
        custom_waveform=rf_custom_waveform_hz,
        custom_raster_s=rf_custom_raster_s,
    )
    rf_samples_by_repetition = [
        scale_rf_envelope_to_flip(
            rf_envelope,
            flip_angle_deg=angle_deg,
            raster_s=rf_raster_s,
            reference_flip_angle_deg=(
                rf_custom_flip_angle_deg if rf_pulse_type == "designer" else None
            ),
        )
        for angle_deg in flip_angle_schedule_deg
    ]
    slice_selective = slice_thickness_m is not None
    slice_gradient_hz_per_m = (
        0.0
        if not slice_selective
        else effective_rf_tbw / (rf_duration_s * slice_thickness_m)
    )
    spoiler_enabled = bool(spoil_after_slice) and (
        spoiler_cycles_per_slice > 0 or spoiler_cycles_per_voxel > 0
    )
    rewind_after_frame = (
        slice_selective or n_slices > 1 or repetitions > 1 or spoiler_enabled
    )
    frame_duration_s = (
        rf_duration_s
        + prephaser_duration_s
        + delay_after_prephaser_s
        + ny * readout_duration
        + max(ny - 1, 0) * blip_duration_s
        + (prephaser_duration_s if rewind_after_frame else 0.0)
        + (spoiler_duration_s if spoiler_enabled else 0.0)
    )
    minimum_repetition_time_s = n_slices * frame_duration_s
    actual_repetition_time_s = (
        minimum_repetition_time_s
        if repetition_time_s is None
        else float(repetition_time_s)
    )
    tolerance = max(1e-12, minimum_repetition_time_s * 1e-10)
    if actual_repetition_time_s < minimum_repetition_time_s - tolerance:
        raise ValueError(
            "repetition_time_s is shorter than the minimum multi-slice "
            f"acquisition time ({minimum_repetition_time_s:.9g} s)"
        )

    events = []
    adc_counts = []
    slice_indices = []
    repetition_indices = []
    spoiler_end_times = []

    for repetition in range(repetitions):
        rf_samples_hz = rf_samples_by_repetition[repetition]
        repetition_start = repetition * actual_repetition_time_s
        for slice_index, slice_position in enumerate(slice_positions):
            frame_start = repetition_start + slice_index * frame_duration_s
            slice_frequency_offset_hz = slice_gradient_hz_per_m * slice_position
            frequency_offset_hz = rf_frequency_offset_hz + slice_frequency_offset_hz
            events.append(
                RFEvent(
                    frame_start,
                    rf_samples_hz,
                    rf_raster_s,
                    frequency_offset_hz=frequency_offset_hz,
                    phase_offset_rad=(
                        -2 * np.pi * slice_frequency_offset_hz * rf_duration_s / 2
                    ),
                )
            )
            if slice_selective:
                events.append(
                    GradientEvent(
                        "z",
                        frame_start,
                        np.asarray([slice_gradient_hz_per_m]),
                        rf_duration_s,
                    )
                )

            first_direction = acquisition.readout_directions[0]
            first_kx = (
                acquisition.kx_cyc_per_m[0]
                if first_direction > 0
                else acquisition.kx_cyc_per_m[-1]
            )
            current_x = first_kx - first_direction * 0.5 / fov_x
            first_phase_index = acquisition.phase_indices[0]
            current_y = acquisition.ky_cyc_per_m[first_phase_index]
            prephaser_start = frame_start + rf_duration_s
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
            if slice_selective:
                events.append(
                    GradientEvent(
                        "z",
                        prephaser_start,
                        np.asarray([-slice_gradient_hz_per_m * rf_duration_s / 2])
                        / prephaser_duration_s,
                        prephaser_duration_s,
                    )
                )

            time_s = prephaser_start + prephaser_duration_s + delay_after_prephaser_s
            for line in range(ny):
                direction = acquisition.readout_directions[line]
                read_gradient = direction / (fov_x * dwell)
                events.append(
                    GradientEvent(
                        "x", time_s, np.asarray([read_gradient]), readout_duration
                    )
                )
                events.append(
                    ADCEvent(
                        start_s=time_s + dwell / 2,
                        num_samples=nx,
                        dwell_s=dwell,
                    )
                )
                adc_counts.append(nx)
                slice_indices.append(slice_index)
                repetition_indices.append(repetition)
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

            if rewind_after_frame:
                if current_x != 0:
                    events.append(
                        GradientEvent(
                            "x",
                            time_s,
                            np.asarray([-current_x / prephaser_duration_s]),
                            prephaser_duration_s,
                        )
                    )
                if current_y != 0:
                    events.append(
                        GradientEvent(
                            "y",
                            time_s,
                            np.asarray([-current_y / prephaser_duration_s]),
                            prephaser_duration_s,
                        )
                    )
                time_s += prephaser_duration_s

            if spoiler_enabled:
                if spoiler_cycles_per_voxel > 0:
                    for axis, voxel_size in zip("xy", (fov_x / nx, fov_y / ny)):
                        events.append(
                            GradientEvent(
                                axis,
                                time_s,
                                np.asarray(
                                    [
                                        spoiler_cycles_per_voxel
                                        / voxel_size
                                        / spoiler_duration_s
                                    ]
                                ),
                                spoiler_duration_s,
                            )
                        )
                if spoiler_cycles_per_slice > 0:
                    events.append(
                        GradientEvent(
                            "z",
                            time_s,
                            np.asarray(
                                [
                                    spoiler_cycles_per_slice
                                    / slice_thickness_m
                                    / spoiler_duration_s
                                ]
                            ),
                            spoiler_duration_s,
                        )
                    )
                spoiler_end_times.append(time_s + spoiler_duration_s)

    dimensions = AcquisitionDimensions(
        adc_event_sample_counts=tuple(adc_counts),
        slice_indices=tuple(slice_indices),
        repetition_indices=tuple(repetition_indices),
        source="internal_cartesian_epi",
    )
    definitions = {
        "FOV": (fov_x, fov_y),
        "MatrixSize": (nx, ny),
        "FlipAngleDeg": float(
            vfa_final_flip_angle_deg if variable_flip_angle else flip_angle_deg
        ),
        "RFPulseType": rf_pulse_type,
        "RFDuration": float(rf_duration_s),
        "RequestedRFDuration": requested_rf_duration_s,
        "RFTimeBandwidthProduct": float(effective_rf_tbw),
        "RFBandwidth": float(effective_rf_tbw / rf_duration_s),
        "RFFrequencyOffset": float(rf_frequency_offset_hz),
        "VariableFlipAngle": bool(variable_flip_angle),
        "Repetitions": repetitions,
        "RepetitionTime": actual_repetition_time_s,
        "MinimumRepetitionTime": minimum_repetition_time_s,
        "SpoilAfterSlice": bool(spoil_after_slice),
        "SpoilerCyclesPerSlice": float(spoiler_cycles_per_slice),
        "SpoilerCyclesPerVoxel": float(spoiler_cycles_per_voxel),
        "SpoilerDuration": float(spoiler_duration_s),
        "SpoilerAxes": (
            ("xy" if spoiler_cycles_per_voxel > 0 else "")
            + ("z" if spoiler_cycles_per_slice > 0 and slice_selective else "")
            if spoiler_enabled
            else "none"
        ),
        "SpoilerEndTimes": tuple(float(value) for value in spoiler_end_times),
        "IdealSpoilerEndTimes": tuple(float(value) for value in spoiler_end_times),
    }
    if rf_pulse_type == "sinc":
        definitions["RFApodization"] = float(rf_apodization)
    if rf_pulse_type == "slr":
        definitions["RFSLRSharpness"] = float(rf_slr_sharpness)
    if rf_pulse_type == "designer":
        definitions["RFDesignerPulseName"] = rf_custom_name or "custom"
        definitions["RFDesignerFlipAngleDeg"] = float(rf_custom_flip_angle_deg)
    if variable_flip_angle:
        definitions.update(
            {
                "VariableFlipAngleDimension": "repetition",
                "VariableFlipAngleFinalDeg": float(vfa_final_flip_angle_deg),
                "FlipAngleScheduleDeg": tuple(
                    float(value) for value in flip_angle_schedule_deg
                ),
                "VariableFlipAngleReferenceDOI": VFA_REFERENCE_DOI,
            }
        )
    if slice_selective:
        slice_extent = n_slices * slice_thickness_m + (n_slices - 1) * float(
            slice_gap_m
        )
        definitions.update(
            {
                "FOV": (fov_x, fov_y, slice_extent),
                "SliceThickness": slice_thickness_m,
                "SliceGap": float(slice_gap_m),
                "SliceSpacing": slice_thickness_m + float(slice_gap_m),
                "SlicePositions": tuple(float(value) for value in slice_positions),
            }
        )

    return SequenceProgram(
        events=tuple(events),
        duration_s=repetitions * actual_repetition_time_s + tail_s,
        source="internal-cartesian-epi",
        metadata={
            "acquisition": acquisition.to_metadata(),
            "acquisition_dimensions": dimensions.to_metadata(),
            "definitions": definitions,
        },
    )


def _positive_integer(value: int, name: str) -> int:
    if int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
