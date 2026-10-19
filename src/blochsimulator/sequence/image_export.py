"""Scanner-geometry image series extracted from simulated sequence results.

The Bruker ``pdata``, DICOM and NIfTI exporters need the same information:
magnitude (and optionally phase) images for every slice, echo and repetition
together with the position and orientation of each pixel.  This module derives
those image series once from a :class:`SequenceReconstructionModel`.

Conventions
-----------
* Positions are given in the simulator's gradient coordinates ``x, y, z`` in
  metres.  For a head-first supine subject these coincide with Bruker's
  subject coordinates and with the patient-based DICOM LPS frame.
* Every frame is a 2D array ``[row, column]``.  For Cartesian data the
  columns run along the readout and the rows along the phase encoding.
* ``row_direction`` is the direction of increasing column index and
  ``column_direction`` the direction of increasing row index (the two
  triplets of DICOM ImageOrientationPatient).
* ``position_m`` is the centre of the first pixel (row 0, column 0), as
  DICOM ImagePositionPatient.  Bruker ``VisuCorePosition`` is the outer
  corner of that pixel, half a pixel further along ``-row_direction`` and
  ``-column_direction``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from ..units import NUCLEUS_GAMMA_HZ_PER_T
from .encoding import EncodingFrame
from .reconstruction import SequenceReconstructionModel


@dataclass(frozen=True)
class ImageFrame:
    """One 2D image of a series with its position in scanner coordinates."""

    magnitude: np.ndarray
    position_m: tuple[float, float, float]
    phase: Optional[np.ndarray] = None
    slice_index: int = 0
    echo_index: int = 0
    repetition_index: int = 0
    echo_time_s: Optional[float] = None
    acquisition_time_s: Optional[float] = None


@dataclass(frozen=True)
class ImageSeries:
    """Frames that share orientation, matrix and pixel spacing.

    Frames are ordered with the slice index running fastest, then the echo
    and then the repetition index (Bruker's ``FG_SLICE``, ``FG_ECHO``,
    ``FG_CYCLE`` order).
    """

    name: str
    description: str
    kind: str
    frames: tuple[ImageFrame, ...]
    row_direction: tuple[float, float, float]
    column_direction: tuple[float, float, float]
    pixel_spacing_m: tuple[float, float]
    slice_thickness_m: float
    acquisition_type: str = "2D"
    is_spectroscopic_map: bool = False
    pool: Optional[str] = None
    spectral_window_hz: Optional[tuple[float, float]] = None

    @property
    def normal(self) -> tuple[float, float, float]:
        normal = np.cross(self.row_direction, self.column_direction)
        return tuple(float(value) for value in normal)

    @property
    def orientation_matrix(self) -> np.ndarray:
        """Rows: row direction, column direction and slice normal."""
        return np.asarray(
            [self.row_direction, self.column_direction, self.normal], dtype=float
        )

    @property
    def rows(self) -> int:
        return int(self.frames[0].magnitude.shape[0]) if self.frames else 0

    @property
    def columns(self) -> int:
        return int(self.frames[0].magnitude.shape[1]) if self.frames else 0

    @property
    def slice_count(self) -> int:
        return len({frame.slice_index for frame in self.frames})

    @property
    def echo_count(self) -> int:
        return len({frame.echo_index for frame in self.frames})

    @property
    def repetition_count(self) -> int:
        return len({frame.repetition_index for frame in self.frames})

    @property
    def has_phase(self) -> bool:
        return bool(self.frames) and all(
            frame.phase is not None for frame in self.frames
        )

    @property
    def slice_positions_m(self) -> np.ndarray:
        """Signed slice positions along the normal, one per slice index."""
        normal = np.asarray(self.normal)
        positions = {}
        for frame in self.frames:
            positions.setdefault(
                frame.slice_index, float(np.dot(frame.position_m, normal))
            )
        return np.asarray([positions[key] for key in sorted(positions)])

    @property
    def slice_spacing_m(self) -> Optional[float]:
        positions = self.slice_positions_m
        if positions.size < 2:
            return None
        return float(abs(np.median(np.diff(positions))))

    def magnitude_stack(self) -> np.ndarray:
        """Return ``(frames, rows, columns)`` magnitude values."""
        return np.stack([frame.magnitude for frame in self.frames], axis=0)

    def phase_stack(self) -> Optional[np.ndarray]:
        if not self.has_phase:
            return None
        return np.stack([frame.phase for frame in self.frames], axis=0)


@dataclass(frozen=True)
class AcquisitionSummary:
    """Protocol parameters shared by the image exporters."""

    sequence_name: str = ""
    repetition_time_s: Optional[float] = None
    echo_time_s: Optional[float] = None
    echo_times_s: tuple[float, ...] = ()
    flip_angle_deg: Optional[float] = None
    field_strength_t: Optional[float] = None
    nucleus: str = "1H"
    imaging_frequency_mhz: Optional[float] = None
    sampling_bandwidth_hz: Optional[float] = None
    readout_samples: Optional[int] = None
    phase_encoding_steps: Optional[int] = None
    repetitions: int = 1
    scan_time_s: Optional[float] = None
    spectral_bandwidth_hz: Optional[float] = None
    spectral_points: Optional[int] = None
    extras: Mapping[str, object] = field(default_factory=dict)

    @property
    def pixel_bandwidth_hz(self) -> Optional[float]:
        if self.sampling_bandwidth_hz is None or not self.readout_samples:
            return None
        return float(self.sampling_bandwidth_hz) / float(self.readout_samples)


def definition_mapping(result) -> dict:
    """Return the result's Pulseq definitions with lower-case keys."""
    metadata = getattr(result, "metadata", None) or {}
    return {
        str(key).lower(): value
        for key, value in dict(metadata.get("sequence_definitions", {}) or {}).items()
    }


def definition_values(definitions: Mapping, *names: str) -> np.ndarray:
    """Return the first numeric definition among ``names`` as a 1D array."""
    for name in names:
        value = definitions.get(name.lower())
        if value is None:
            continue
        parsed = _numeric_values(value)
        if parsed.size:
            return parsed
    return np.empty(0, dtype=float)


def definition_float(definitions: Mapping, *names: str) -> Optional[float]:
    values = definition_values(definitions, *names)
    if values.size == 0 or not np.isfinite(values[0]):
        return None
    return float(values[0])


def _numeric_values(value) -> np.ndarray:
    if isinstance(value, str):
        text = value.strip()
        if text.lower().startswith("array(") and text.endswith(")"):
            text = text[6:-1]
        tokens = text.strip("[](){}").replace(",", " ").replace(";", " ").split()
        try:
            return np.asarray([float(token) for token in tokens], dtype=float)
        except ValueError:
            return np.empty(0, dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=float)


def _normalize_nucleus(value) -> str:
    text = str(value or "").strip().upper().replace("-", "")
    match = re.match(r"^([A-Z][A-Z]?)(\d+)$", text)
    if match:
        return f"{match.group(2)}{match.group(1).capitalize()}"
    match = re.match(r"^(\d+)([A-Z][A-Z]?)$", text)
    if match:
        return f"{match.group(1)}{match.group(2).capitalize()}"
    return "1H"


def gyromagnetic_ratio_hz_per_t(nucleus: str) -> Optional[float]:
    """Return gamma/2pi for a nucleus written as ``1H``/``H1``/``13C``."""
    normalized = _normalize_nucleus(nucleus)
    match = re.match(r"^(\d+)([A-Za-z]+)$", normalized)
    if not match:
        return None
    key = f"{match.group(2)}{match.group(1)}"
    value = NUCLEUS_GAMMA_HZ_PER_T.get(key)
    return None if value is None else float(value)


def acquisition_summary(result) -> AcquisitionSummary:
    """Collect TR, TE, flip angle, field strength and similar parameters."""
    metadata = getattr(result, "metadata", None) or {}
    definitions = definition_mapping(result)
    nucleus = _normalize_nucleus(metadata.get("nucleus") or "H1")
    field_strength = _optional_float(metadata.get("field_strength_t"))
    frequency = None
    gamma = gyromagnetic_ratio_hz_per_t(nucleus)
    if field_strength is not None and gamma is not None:
        frequency = field_strength * gamma / 1e6
    echo_times = definition_values(definitions, "EchoTimes")
    echo_time = definition_float(definitions, "TE")
    if echo_time is None and echo_times.size:
        echo_time = float(echo_times[0])
    repetition_time = definition_float(definitions, "TR", "RepetitionTime")
    if repetition_time is None:
        repetition_time = definition_float(definitions, "VolumeInterval")
    matrix = definition_values(definitions, "MatrixSize", "EncodingMatrixSize")
    spectral_points = definition_float(definitions, "SpectralPoints")
    spectral_bandwidth = definition_float(definitions, "SpectralBandwidth")
    spectroscopic = metadata.get("spectroscopic_acquisition")
    if isinstance(spectroscopic, dict):
        if spectral_points is None:
            spectral_points = _optional_float(spectroscopic.get("spectral_points"))
        dwell = _optional_float(spectroscopic.get("dwell_s"))
        if spectral_bandwidth is None and dwell:
            spectral_bandwidth = 1.0 / dwell
    is_csi = spectral_points is not None and str(
        definitions.get("name", "")
    ).lower().startswith("csi")
    readout_samples = None
    phase_steps = None
    if matrix.size >= 2 and not is_csi:
        readout_samples = int(matrix[0])
        phase_steps = int(matrix[1])
    elif matrix.size >= 2:
        phase_steps = int(matrix[0] * matrix[1])
    repetitions = definition_float(definitions, "Repetitions")
    duration = definition_float(definitions, "TotalDuration")
    adc_times = np.asarray(getattr(result, "adc_times_s", ()), dtype=float)
    if duration is None and adc_times.size:
        duration = float(np.max(adc_times))
    return AcquisitionSummary(
        sequence_name=str(definitions.get("name") or metadata.get("program_name", "")),
        repetition_time_s=repetition_time,
        echo_time_s=echo_time,
        echo_times_s=tuple(float(value) for value in echo_times),
        flip_angle_deg=definition_float(definitions, "FlipAngleDeg", "FlipAngle"),
        field_strength_t=field_strength,
        nucleus=nucleus,
        imaging_frequency_mhz=frequency,
        sampling_bandwidth_hz=definition_float(definitions, "SamplingBandwidth"),
        readout_samples=readout_samples,
        phase_encoding_steps=phase_steps,
        repetitions=max(1, int(repetitions)) if repetitions else 1,
        scan_time_s=duration,
        spectral_bandwidth_hz=spectral_bandwidth,
        spectral_points=None if spectral_points is None else int(spectral_points),
    )


def _optional_float(value) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def extract_image_series(
    result=None,
    *,
    model: Optional[SequenceReconstructionModel] = None,
    definitions: Optional[Mapping] = None,
    include_pools: bool = True,
    include_phase: bool = True,
    default_slice_thickness_m: float = 1e-3,
) -> list[ImageSeries]:
    """Return the reconstructed images of ``result`` as scanner-placed series.

    The first series always holds the combined image (or, for CSI, the total
    signal map).  With ``include_pools`` one series per pool (metabolite)
    follows when pool-resolved data exist.  Results without a supported
    image reconstruction return an empty list.
    """

    if model is None:
        if result is None:
            raise ValueError("extract_image_series needs a result or a model")
        model = SequenceReconstructionModel.from_result(result)
    if definitions is None:
        definitions = definition_mapping(result) if result is not None else {}
        if not definitions:
            definitions = _definitions_from_dataset(model.dataset)
    else:
        definitions = {str(key).lower(): value for key, value in definitions.items()}
    extractor = {
        "cartesian_2d": _cartesian_2d_series,
        "cartesian_3d": _cartesian_3d_series,
        "spiral_2d": _spiral_series,
        "radial_3d": _radial_series,
        "csi": _csi_series,
    }.get(model.kind)
    if extractor is None:
        return []
    context = _Context(
        model=model,
        definitions=definitions,
        include_pools=include_pools,
        include_phase=include_phase,
        default_slice_thickness_m=float(default_slice_thickness_m),
    )
    return [series for series in extractor(context) if series.frames]


def _definitions_from_dataset(dataset) -> dict:
    import json

    try:
        metadata = json.loads(dataset.attrs.get("metadata_json", "{}"))
    except (TypeError, ValueError):
        return {}
    return {
        str(key).lower(): value
        for key, value in dict(metadata.get("sequence_definitions", {}) or {}).items()
    }


@dataclass
class _Context:
    model: SequenceReconstructionModel
    definitions: Mapping
    include_pools: bool
    include_phase: bool
    default_slice_thickness_m: float

    @property
    def dataset(self):
        return self.model.dataset

    def slice_thickness_m(self) -> float:
        value = definition_float(self.definitions, "SliceThickness")
        if value is not None and value > 0:
            return value
        return self.default_slice_thickness_m

    def slice_positions_m(self, count: int) -> np.ndarray:
        """Slice centres along the slice normal, one per slice index."""
        count = max(1, int(count))
        positions = definition_values(self.definitions, "SlicePositions")
        if positions.size == count:
            return positions
        offset = definition_float(self.definitions, "SliceOffset") or 0.0
        if count == 1:
            return np.asarray([positions[0] if positions.size == 1 else offset])
        spacing = definition_float(self.definitions, "SliceSpacing")
        if spacing is None:
            gap = definition_float(self.definitions, "SliceGap") or 0.0
            spacing = self.slice_thickness_m() + gap
        return offset + (np.arange(count) - (count - 1) / 2.0) * spacing

    def echo_time_s(self, echo_index: int) -> Optional[float]:
        echoes = definition_values(self.definitions, "EchoTimes")
        if echoes.size > echo_index:
            return float(echoes[echo_index])
        if echo_index == 0:
            return definition_float(self.definitions, "TE")
        return None


def _unit(vector) -> tuple[float, float, float]:
    values = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(values))
    if norm == 0:
        raise ValueError("direction vector must not be zero")
    return tuple(float(value) + 0.0 for value in values / norm)


def _cartesian_basis(context: _Context) -> np.ndarray:
    """Return the 3x3 basis whose columns are read, phase and partition."""
    attribute = context.dataset.attrs.get("cartesian_encoding_basis_xyz")
    if attribute is not None:
        values = _numeric_values(attribute)
        if values.size == 9:
            return values.reshape(3, 3)
    try:
        return EncodingFrame.from_definitions(dict(context.definitions)).matrix
    except Exception:
        return np.eye(3)


def _spacing(positions: np.ndarray, fallback: float) -> float:
    positions = np.asarray(positions, dtype=float)
    if positions.size > 1:
        step = float(np.median(np.abs(np.diff(positions))))
        if np.isfinite(step) and step > 0:
            return step
    return float(fallback)


def _centered_positions(count: int, fov_m: float) -> np.ndarray:
    return ((np.arange(count) + 0.5) / count - 0.5) * float(fov_m)


def _coordinate(dataset, name: str, dimension: str) -> Optional[np.ndarray]:
    if name in dataset.coords and dataset.coords[name].dims == (dimension,):
        return np.asarray(dataset.coords[name].values, dtype=float)
    return None


def _planes(data, row_dim: str, col_dim: str):
    """Yield ``(selection, magnitude, complex_sum)`` for each 2D plane."""
    outer = [dim for dim in data.dims if dim not in (row_dim, col_dim, "coil")]
    sizes = [int(data.sizes[dim]) for dim in outer]
    order = (["coil"] if "coil" in data.dims else []) + [row_dim, col_dim]
    for index in np.ndindex(*sizes) if sizes else [()]:
        selection = dict(zip(outer, index))
        plane = data.isel(selection) if selection else data
        values = np.asarray(plane.transpose(*order).values)
        if "coil" in data.dims:
            magnitude = np.sqrt(np.sum(np.abs(values) ** 2, axis=0))
            combined = values.sum(axis=0)
        else:
            magnitude = np.abs(values)
            combined = values
        yield selection, magnitude.astype(float), combined


def _phase(combined: np.ndarray, include_phase: bool) -> Optional[np.ndarray]:
    if not include_phase or not np.iscomplexobj(combined):
        return None
    return np.angle(combined).astype(float)


def _fold_index(selection: Mapping[str, int], data, dims: Sequence[str]) -> int:
    """Flatten the indices of ``dims`` into one repetition-like index."""
    dims = [dim for dim in dims if dim in selection]
    if not dims:
        return 0
    return int(
        np.ravel_multi_index(
            [selection[dim] for dim in dims],
            [int(data.sizes[dim]) for dim in dims],
        )
    )


def _sorted_frames(frames: list[ImageFrame]) -> tuple[ImageFrame, ...]:
    return tuple(
        sorted(
            frames,
            key=lambda frame: (
                frame.repetition_index,
                frame.echo_index,
                frame.slice_index,
            ),
        )
    )


def _pool_variants(context: _Context, variable: str):
    """Yield ``(name, description, pool, data)`` for combined and pool data."""
    dataset = context.dataset
    pool_variable = context.model.image_name(pool=True)
    data = dataset[variable]
    if "pool" in data.dims:
        data = data.sum("pool")
    yield "image", "Magnitude", None, data
    if not context.include_pools or not pool_variable:
        return
    pool_data = dataset[pool_variable]
    for index, pool in enumerate(context.model.pool_names):
        yield (
            f"pool_{_safe_name(pool)}",
            f"Pool {pool}",
            pool,
            pool_data.isel(pool=index),
        )


def _safe_name(text) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")
    return cleaned or "pool"


def _cartesian_2d_series(context: _Context) -> list[ImageSeries]:
    model = context.model
    dataset = context.dataset
    variable = model.image_name()
    phase_dim, read_dim = model.spatial_dims
    basis = _cartesian_basis(context)
    read_direction = _unit(basis[:, 0])
    phase_direction = _unit(basis[:, 1])
    normal = np.cross(read_direction, phase_direction)
    read_positions = _coordinate(dataset, "cartesian_read_position_m", read_dim)
    if read_positions is None:
        read_positions = _centered_positions(
            dataset.sizes[read_dim],
            model._fov_from_k_coordinate("cartesian_k_read_cyc_per_m", read_dim),
        )
    phase_positions = _coordinate(dataset, "cartesian_phase_position_m", phase_dim)
    if phase_positions is None:
        phase_positions = _centered_positions(
            dataset.sizes[phase_dim],
            model._fov_from_k_coordinate("cartesian_k_phase_cyc_per_m", phase_dim),
        )
    dx = _spacing(read_positions, 1e-3)
    dy = _spacing(phase_positions, 1e-3)
    in_plane_origin = read_positions[0] * np.asarray(read_direction) + phase_positions[
        0
    ] * np.asarray(phase_direction)
    frame_dim = "cartesian_frame" if "cartesian_frame" in dataset.dims else None

    def frame_index(axis: str, frame: Optional[int]) -> int:
        name = f"cartesian_frame_{axis}_index"
        if frame is None or name not in dataset.coords:
            return 0
        return int(dataset.coords[name].values[frame])

    slice_values = (
        np.unique(dataset.coords["cartesian_frame_slice_index"].values)
        if frame_dim and "cartesian_frame_slice_index" in dataset.coords
        else np.asarray([0])
    )
    slice_rank = {int(value): rank for rank, value in enumerate(slice_values)}
    slice_positions = context.slice_positions_m(len(slice_values))
    echo_values = (
        np.unique(dataset.coords["cartesian_frame_echo_index"].values)
        if frame_dim and "cartesian_frame_echo_index" in dataset.coords
        else np.asarray([0])
    )
    echo_rank = {int(value): rank for rank, value in enumerate(echo_values)}
    repetition_values = (
        np.unique(dataset.coords["cartesian_frame_repetition_index"].values)
        if frame_dim and "cartesian_frame_repetition_index" in dataset.coords
        else np.asarray([0])
    )
    repetition_rank = {int(value): rank for rank, value in enumerate(repetition_values)}
    series = []
    for name, description, pool, data in _pool_variants(context, variable):
        frames = []
        extra_dims = [
            dim
            for dim in data.dims
            if dim not in (phase_dim, read_dim, "coil", frame_dim)
        ]
        for selection, magnitude, combined in _planes(data, phase_dim, read_dim):
            frame = selection.get(frame_dim) if frame_dim else None
            slice_index = slice_rank.get(frame_index("slice", frame), 0)
            echo_index = echo_rank.get(frame_index("echo", frame), 0)
            repetition_index = repetition_rank.get(frame_index("repetition", frame), 0)
            repetition_index += len(repetition_rank) * _fold_index(
                selection, data, extra_dims
            )
            echo_time = None
            if frame is not None and "cartesian_frame_echo_time_s" in dataset.coords:
                echo_time = float(
                    dataset.coords["cartesian_frame_echo_time_s"].values[frame]
                )
            if echo_time is None:
                echo_time = context.echo_time_s(echo_index)
            acquisition_time = None
            if (
                frame is not None
                and "cartesian_frame_first_adc_time_s" in dataset.coords
            ):
                acquisition_time = float(
                    dataset.coords["cartesian_frame_first_adc_time_s"].values[frame]
                )
            position = in_plane_origin + slice_positions[slice_index] * normal
            frames.append(
                ImageFrame(
                    magnitude=magnitude,
                    phase=_phase(combined, context.include_phase),
                    position_m=tuple(float(value) for value in position),
                    slice_index=slice_index,
                    echo_index=echo_index,
                    repetition_index=repetition_index,
                    echo_time_s=echo_time,
                    acquisition_time_s=acquisition_time,
                )
            )
        series.append(
            ImageSeries(
                name=name,
                description=description,
                kind=model.kind,
                frames=_sorted_frames(frames),
                row_direction=read_direction,
                column_direction=phase_direction,
                pixel_spacing_m=(dy, dx),
                slice_thickness_m=context.slice_thickness_m(),
                acquisition_type="2D",
                pool=pool,
            )
        )
    return series


def _cartesian_3d_series(context: _Context) -> list[ImageSeries]:
    model = context.model
    dataset = context.dataset
    variable = model.image_name()
    partition_dim, phase_dim, read_dim = model.spatial_dims
    basis = _cartesian_basis(context)
    read_direction = _unit(basis[:, 0])
    phase_direction = _unit(basis[:, 1])
    normal = np.cross(read_direction, phase_direction)
    positions = {}
    for role, dim in (
        ("read", read_dim),
        ("phase", phase_dim),
        ("partition", partition_dim),
    ):
        values = _coordinate(dataset, f"cartesian_{role}_position_m", dim)
        if values is None:
            values = _centered_positions(
                dataset.sizes[dim],
                model._fov_from_k_coordinate(f"cartesian_k_{role}_cyc_per_m", dim),
            )
        positions[role] = values
    dx = _spacing(positions["read"], 1e-3)
    dy = _spacing(positions["phase"], 1e-3)
    dz = _spacing(positions["partition"], context.slice_thickness_m())
    in_plane_origin = positions["read"][0] * np.asarray(read_direction) + positions[
        "phase"
    ][0] * np.asarray(phase_direction)
    series = []
    for name, description, pool, data in _pool_variants(context, variable):
        frames = []
        repetition_dims = [
            dim
            for dim in data.dims
            if dim not in (partition_dim, phase_dim, read_dim, "coil", "echo")
        ]
        for selection, magnitude, combined in _planes(data, phase_dim, read_dim):
            slice_index = int(selection.get(partition_dim, 0))
            echo_index = int(selection.get("echo", 0))
            position = in_plane_origin + positions["partition"][slice_index] * normal
            frames.append(
                ImageFrame(
                    magnitude=magnitude,
                    phase=_phase(combined, context.include_phase),
                    position_m=tuple(float(value) for value in position),
                    slice_index=slice_index,
                    echo_index=echo_index,
                    repetition_index=_fold_index(selection, data, repetition_dims),
                    echo_time_s=context.echo_time_s(echo_index),
                )
            )
        series.append(
            ImageSeries(
                name=name,
                description=description,
                kind=model.kind,
                frames=_sorted_frames(frames),
                row_direction=read_direction,
                column_direction=phase_direction,
                pixel_spacing_m=(dy, dx),
                slice_thickness_m=dz,
                acquisition_type="3D",
                pool=pool,
            )
        )
    return series


def _spiral_series(context: _Context) -> list[ImageSeries]:
    model = context.model
    dataset = context.dataset
    variable = model.image_name()
    row_dim, col_dim = "phase_y", "read_x"
    kx = _coordinate(dataset, "spiral_grid_kx_cyc_per_m", col_dim)
    ky = _coordinate(dataset, "spiral_grid_ky_cyc_per_m", row_dim)
    nx = int(dataset.sizes[col_dim])
    ny = int(dataset.sizes[row_dim])
    fov_x = model._fov_from_k_coordinate("spiral_grid_kx_cyc_per_m", col_dim)
    fov_y = model._fov_from_k_coordinate("spiral_grid_ky_cyc_per_m", row_dim)
    dx = fov_x / nx
    dy = fov_y / ny
    # The spiral grid places k=0 at index N//2 and reconstructs with
    # fftshift, so image index n sits at (n - N//2) * dx.
    x_positions = (np.arange(nx) - nx // 2) * dx
    y_positions = (np.arange(ny) - ny // 2) * dy
    if kx is None or ky is None:
        x_positions = _centered_positions(nx, fov_x)
        y_positions = _centered_positions(ny, fov_y)
    row_direction = (1.0, 0.0, 0.0)
    column_direction = (0.0, 1.0, 0.0)
    normal = np.asarray((0.0, 0.0, 1.0))
    frame_dim = "spiral_frame" if "spiral_frame" in dataset.dims else None
    slice_values = (
        np.unique(dataset.coords["spiral_frame_slice_index"].values)
        if frame_dim and "spiral_frame_slice_index" in dataset.coords
        else np.asarray([0])
    )
    slice_rank = {int(value): rank for rank, value in enumerate(slice_values)}
    slice_positions = context.slice_positions_m(len(slice_values))

    def frame_index(axis: str, frame: Optional[int]) -> int:
        name = f"spiral_frame_{axis}_index"
        if frame is None or name not in dataset.coords:
            return 0
        return int(dataset.coords[name].values[frame])

    series = []
    for name, description, pool, data in _pool_variants(context, variable):
        frames = []
        extra_dims = [
            dim for dim in data.dims if dim not in (row_dim, col_dim, "coil", frame_dim)
        ]
        for selection, magnitude, combined in _planes(data, row_dim, col_dim):
            frame = selection.get(frame_dim) if frame_dim else None
            slice_index = slice_rank.get(frame_index("slice", frame), 0)
            echo_index = frame_index("echo", frame)
            repetition_index = frame_index("repetition", frame) + frame_index(
                "segment", frame
            )
            repetition_index += _fold_index(selection, data, extra_dims)
            position = (
                x_positions[0] * np.asarray(row_direction)
                + y_positions[0] * np.asarray(column_direction)
                + slice_positions[slice_index] * normal
            )
            frames.append(
                ImageFrame(
                    magnitude=magnitude,
                    phase=_phase(combined, context.include_phase),
                    position_m=tuple(float(value) for value in position),
                    slice_index=slice_index,
                    echo_index=echo_index,
                    repetition_index=repetition_index,
                    echo_time_s=context.echo_time_s(echo_index),
                )
            )
        series.append(
            ImageSeries(
                name=name,
                description=description,
                kind=model.kind,
                frames=_sorted_frames(frames),
                row_direction=row_direction,
                column_direction=column_direction,
                pixel_spacing_m=(dy, dx),
                slice_thickness_m=context.slice_thickness_m(),
                acquisition_type="2D",
                pool=pool,
            )
        )
    return series


def _radial_series(context: _Context) -> list[ImageSeries]:
    model = context.model
    dataset = context.dataset
    variable = model.image_name()
    fov = model._numeric_attr(dataset.attrs.get("radial_fov_m"))
    positions = {}
    for axis, dim in (("x", "radial_x"), ("y", "radial_y"), ("z", "radial_z")):
        values = _coordinate(dataset, f"radial_position_{axis}_m", dim)
        if values is None:
            extent = fov[0] if fov else float(dataset.sizes[dim])
            values = _centered_positions(dataset.sizes[dim], extent)
        positions[axis] = values
    dx = _spacing(positions["x"], 1e-3)
    dy = _spacing(positions["y"], 1e-3)
    dz = _spacing(positions["z"], 1e-3)
    series = []
    for name, description, pool, data in _pool_variants(context, variable):
        frames = []
        repetition_dims = [
            dim
            for dim in data.dims
            if dim not in ("radial_x", "radial_y", "radial_z", "coil", "echo")
        ]
        for selection, magnitude, combined in _planes(data, "radial_y", "radial_x"):
            slice_index = int(selection.get("radial_z", 0))
            echo_index = int(selection.get("echo", 0))
            position = (
                positions["x"][0],
                positions["y"][0],
                positions["z"][slice_index],
            )
            frames.append(
                ImageFrame(
                    magnitude=magnitude,
                    phase=_phase(combined, context.include_phase),
                    position_m=tuple(float(value) for value in position),
                    slice_index=slice_index,
                    echo_index=echo_index,
                    repetition_index=_fold_index(selection, data, repetition_dims),
                    echo_time_s=context.echo_time_s(echo_index),
                )
            )
        series.append(
            ImageSeries(
                name=name,
                description=description,
                kind=model.kind,
                frames=_sorted_frames(frames),
                row_direction=(1.0, 0.0, 0.0),
                column_direction=(0.0, 1.0, 0.0),
                pixel_spacing_m=(dy, dx),
                slice_thickness_m=dz,
                acquisition_type="3D",
                pool=pool,
            )
        )
    return series


def _csi_series(context: _Context) -> list[ImageSeries]:
    """Return signal maps of a phase-encoded CSI acquisition.

    The spatial encoding of the simulated CSI runs along scanner ``x``
    (columns) and ``y`` (rows).  The total map is the magnitude of the first
    FID point.  Pool-resolved results give one map per pool; otherwise peaks
    listed in ``pool_frequency_offsets_hz`` are integrated over a window of
    the magnitude spectrum.
    """

    model = context.model
    dataset = context.dataset
    row_dim, col_dim = "phase_y", "phase_x"
    nx = int(dataset.sizes[col_dim])
    ny = int(dataset.sizes[row_dim])
    fov_x = definition_values(context.definitions, "FOV")
    fov_x_m = model._fov_from_k_coordinate("spatial_kx_cyc_per_m", col_dim)
    fov_y_m = model._fov_from_k_coordinate("spatial_ky_cyc_per_m", row_dim)
    if fov_x.size >= 2 and "spatial_kx_cyc_per_m" not in dataset.coords:
        fov_x_m, fov_y_m = float(fov_x[0]), float(fov_x[1])
    x_positions = _centered_positions(nx, fov_x_m)
    y_positions = _centered_positions(ny, fov_y_m)
    slice_offset = definition_float(context.definitions, "SliceOffset") or 0.0
    position = (float(x_positions[0]), float(y_positions[0]), float(slice_offset))
    common = dict(
        kind=model.kind,
        row_direction=(1.0, 0.0, 0.0),
        column_direction=(0.0, 1.0, 0.0),
        pixel_spacing_m=(fov_y_m / ny, fov_x_m / nx),
        slice_thickness_m=context.slice_thickness_m(),
        acquisition_type="2D",
        is_spectroscopic_map=True,
    )

    def maps(data, *, reduce) -> list[ImageFrame]:
        frames = []
        repetition_dims = [
            dim
            for dim in data.dims
            if dim not in (row_dim, col_dim, "coil", "spectral_point")
        ]
        reduced = reduce(data)
        for selection, magnitude, _ in _planes(reduced, row_dim, col_dim):
            frames.append(
                ImageFrame(
                    magnitude=magnitude,
                    position_m=position,
                    repetition_index=_fold_index(selection, reduced, repetition_dims),
                    echo_time_s=context.echo_time_s(0),
                )
            )
        return list(_sorted_frames(frames))

    def first_point(data):
        return data.isel(spectral_point=0)

    series = [
        ImageSeries(
            name="csi_total",
            description="CSI total signal",
            frames=tuple(maps(dataset["csi_spatial_fid"], reduce=first_point)),
            **common,
        )
    ]
    if not context.include_pools:
        return series
    pool_variable = model.image_name(pool=True)
    if pool_variable and model.pool_names:
        pool_data = dataset[pool_variable]
        for index, pool in enumerate(model.pool_names):
            series.append(
                ImageSeries(
                    name=f"metabolite_{_safe_name(pool)}",
                    description=f"Metabolite {pool}",
                    frames=tuple(maps(pool_data.isel(pool=index), reduce=first_point)),
                    pool=pool,
                    **common,
                )
            )
        return series
    offsets = model._numeric_attr(dataset.attrs.get("pool_frequency_offsets_hz"))
    spectrum_name = model.spectrum_name()
    if (
        not offsets
        or not spectrum_name
        or "spectral_frequency_hz" not in dataset.coords
    ):
        return series
    frequencies = np.asarray(
        dataset.coords["spectral_frequency_hz"].values, dtype=float
    )
    resolution = _spacing(frequencies, 1.0)
    ordered = np.sort(np.asarray(offsets, dtype=float))
    separation = float(np.min(np.diff(ordered))) if ordered.size > 1 else np.inf
    half_width = max(resolution, min(separation / 2.0, 5.0 * resolution))
    spectrum = dataset[spectrum_name]
    for offset in offsets:
        low, high = float(offset) - half_width, float(offset) + half_width
        mask = (frequencies >= low) & (frequencies <= high)
        if not np.any(mask):
            mask = np.zeros_like(frequencies, dtype=bool)
            mask[int(np.argmin(np.abs(frequencies - offset)))] = True
        points = np.flatnonzero(mask)

        def window(data, points=points):
            return np.abs(data.isel(spectral_point=points)).sum("spectral_point")

        series.append(
            ImageSeries(
                name=f"peak_{_safe_name(f'{offset:g}Hz')}",
                description=f"Peak {offset:g} Hz",
                frames=tuple(maps(spectrum, reduce=window)),
                spectral_window_hz=(low, high),
                **common,
            )
        )
    return series
