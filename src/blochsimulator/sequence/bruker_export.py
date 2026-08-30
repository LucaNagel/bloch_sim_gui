"""Bruker-style raw-data export for simulated sequence results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .acquisition import CartesianAcquisition, CartesianAcquisitionFrames
from .model import SequenceProgram
from .result import SequenceSimulationResult


INT32_TARGET = float(2**30)


@dataclass(frozen=True)
class BrukerExportOptions:
    """User-facing Bruker metadata overrides for simulated exports."""

    method_name: str = "Bruker:RARE"
    scan_name: Optional[str] = None
    matrix: Optional[tuple[int, int]] = None
    fov_m: Optional[tuple[float, float]] = None
    slice_thickness_mm: float = 1.0
    read_orientation: str = "L_R"
    slice_orientation: str = "axial"
    patient_position: str = "Head_Supine"
    raw_data_files: str = "fid"


def export_bruker_raw(
    result: SequenceSimulationResult,
    directory,
    *,
    program: Optional[SequenceProgram] = None,
    phantom=None,
    acquisition: Optional[CartesianAcquisition] = None,
    acquisition_frames: Optional[CartesianAcquisitionFrames] = None,
    scale: Optional[float] = None,
    options: Optional[BrukerExportOptions] = None,
    method_name: Optional[str] = None,
    scan_name: Optional[str] = None,
    matrix: Optional[tuple[int, int]] = None,
    fov_m: Optional[tuple[float, float]] = None,
    slice_thickness_mm: Optional[float] = None,
    raw_data_files: Optional[str] = None,
) -> Path:
    """Export a simulated ADC stream as a Bruker-like experiment directory.

    The raw ``fid`` is written as little-endian signed 32-bit integer pairs in
    real/imaginary order, matching ParaVision's ``GO_32BIT_SGN_INT`` raw data
    convention.  The inverse scale is recorded in ``acqp`` and ``method`` as
    ``BLOCHSIM_signal_scale`` so consumers can recover the floating-point
    simulation signal with ``complex_int / scale``.
    """

    output_dir = Path(directory)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Bruker export target is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    signal = _as_coil_signal(result.signal)
    scale = _signal_scale(signal, scale)
    options = _merge_options(
        options,
        method_name=method_name,
        scan_name=scan_name,
        matrix=matrix,
        fov_m=fov_m,
        slice_thickness_mm=slice_thickness_mm,
        raw_data_files=raw_data_files,
    )
    context = _ExportContext(
        result=result,
        program=program,
        phantom=phantom,
        acquisition=acquisition,
        acquisition_frames=acquisition_frames,
        signal=signal,
        scale=scale,
        options=options,
    )
    if context.write_fid:
        _bruker_fid_int32(context).tofile(output_dir / "fid")
    if context.write_rawdata_job0:
        _rawdata_job0_int32(context).tofile(output_dir / "rawdata.job0")

    _write_acqp(output_dir / "acqp", context)
    _write_method(output_dir / "method", context)
    _write_visu_pars(output_dir / "visu_pars", context)
    _write_pulseprogram(output_dir / "pulseprogram", context)
    _write_specpar(output_dir / "specpar", context)
    _write_placeholder_files(output_dir, context)
    _write_reconstructed_pdata(output_dir / "pdata" / "1", context)
    return output_dir


class _ExportContext:
    def __init__(
        self,
        *,
        result,
        program,
        phantom,
        acquisition,
        acquisition_frames,
        signal,
        scale,
        options,
    ):
        self.result = result
        self.program = program
        self.phantom = phantom
        self.acquisition = acquisition
        self.acquisition_frames = acquisition_frames
        self.signal = signal
        self.scale = float(scale)
        self.options = options
        self.created = datetime.now().astimezone()

    @property
    def num_coils(self) -> int:
        return int(self.signal.shape[0])

    @property
    def num_adc_samples(self) -> int:
        return int(self.signal.shape[-1])

    @property
    def adc_event_counts(self) -> tuple[int, ...]:
        if self.program is not None and self.program.adc_events:
            return tuple(int(event.num_samples) for event in self.program.adc_events)
        try:
            dimensions = self.result.acquisition_dimensions
        except Exception:
            dimensions = None
        if dimensions is not None:
            return tuple(int(value) for value in dimensions.adc_event_sample_counts)
        return (self.num_adc_samples,)

    @property
    def readout_samples(self) -> int:
        if self.acquisition is not None:
            return int(self.acquisition.read_matrix)
        spectroscopy = self.spectroscopy
        if spectroscopy is not None:
            return int(spectroscopy.spectral_points)
        counts = self.adc_event_counts
        return int(counts[0]) if counts else self.num_adc_samples

    @property
    def num_readouts(self) -> int:
        readout = max(1, int(self.readout_samples))
        if self.num_adc_samples % readout == 0:
            return max(1, self.num_adc_samples // readout)
        counts = self.adc_event_counts
        return len(counts) if counts else 1

    @property
    def raw_frame_count(self) -> int:
        frames = self.frame_count
        if frames > 1 and self.num_readouts % frames == 0:
            return frames
        return 1

    @property
    def readouts_per_frame(self) -> int:
        return max(1, self.num_readouts // self.raw_frame_count)

    @property
    def matrix(self) -> tuple[int, int]:
        if self.options.matrix is not None:
            return self.options.matrix
        if self.acquisition is not None:
            return (
                int(self.acquisition.read_matrix),
                int(self.acquisition.phase_matrix),
            )
        spectroscopy = self.spectroscopy
        if spectroscopy is not None:
            return (int(spectroscopy.matrix[0]), int(spectroscopy.matrix[1]))
        return (int(self.readout_samples), max(1, int(self.num_readouts)))

    @property
    def frame_count(self) -> int:
        if self.acquisition_frames is not None:
            return int(self.acquisition_frames.num_frames)
        return 1

    @property
    def definitions(self) -> dict:
        if self.program is None:
            return {}
        values = self.program.metadata.get("definitions", {})
        return values if isinstance(values, dict) else {}

    def definition_float(self, name: str) -> Optional[float]:
        value = self.definitions.get(name)
        try:
            scalar = float(np.asarray(value, dtype=float).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            return None
        return scalar if np.isfinite(scalar) else None

    @property
    def repetition_time_ms(self) -> float:
        value = self.definition_float("TR")
        if value is not None and value > 0:
            return value * 1000.0
        spectroscopy = self.spectroscopy
        if spectroscopy is not None and spectroscopy.num_encodings > 0:
            return self.duration_ms / spectroscopy.num_encodings
        return self.duration_ms

    @property
    def echo_time_ms(self) -> float:
        value = self.definition_float("TE")
        return max(0.0, value * 1000.0) if value is not None else 0.0

    @property
    def flip_angle_deg(self) -> float:
        value = self.definition_float("FlipAngleDeg")
        return value if value is not None else 0.0

    @property
    def fov_m(self) -> tuple[float, float]:
        if self.options.fov_m is not None:
            return self.options.fov_m
        if self.acquisition is not None:
            return tuple(float(value) for value in self.acquisition.fov_m)
        spectroscopy = self.spectroscopy
        if spectroscopy is not None:
            return tuple(float(value) for value in spectroscopy.fov_m)
        if self.phantom is not None and hasattr(self.phantom, "fov"):
            fov = tuple(float(value) for value in self.phantom.fov)
            if len(fov) >= 2:
                return (fov[0], fov[1])
            if len(fov) == 1:
                return (fov[0], fov[0])
        return (1.0, 1.0)

    @property
    def dwell_s(self) -> float:
        if self.acquisition is not None:
            return float(self.acquisition.dwell_s)
        spectroscopy = getattr(self.result, "spectroscopic_acquisition", None)
        if spectroscopy is not None:
            return float(spectroscopy.dwell_s)
        if self.program is not None and self.program.adc_events:
            return float(self.program.adc_events[0].dwell_s)
        times = np.asarray(self.result.adc_times_s, dtype=float)
        if times.size > 1:
            diffs = np.diff(times)
            finite_positive = diffs[np.isfinite(diffs) & (diffs > 0)]
            if finite_positive.size:
                return float(np.median(finite_positive))
        return 1.0

    @property
    def duration_ms(self) -> float:
        if self.program is not None:
            return float(self.program.duration_s) * 1000.0
        times = np.asarray(self.result.adc_times_s, dtype=float)
        if times.size:
            return float(np.max(times) - np.min(times) + self.dwell_s) * 1000.0
        return 0.0

    @property
    def source(self) -> str:
        if self.program is not None:
            return str(self.program.source)
        return str(self.result.metadata.get("program_source", "simulated"))

    @property
    def scan_name(self) -> str:
        if self.options.scan_name:
            return self.options.scan_name
        source_name = Path(self.source).name
        if source_name and source_name not in {".", "simulated"}:
            return f"BlochSimulator {source_name}"
        return "BlochSimulator Sequence Simulation"

    @property
    def method_name(self) -> str:
        return self.options.method_name

    @property
    def is_luca_csi4(self) -> bool:
        return (
            self.spectroscopy is not None
            and self.method_name.strip().lower() == "user:lucacsi4"
        )

    @property
    def nucleus(self) -> str:
        return str(self.result.metadata.get("nucleus") or "H1")

    @property
    def bruker_nucleus(self) -> str:
        normalized = self.nucleus.upper().replace("-", "")
        if normalized in {"13C", "C13"}:
            return "13C"
        if normalized in {"1H", "H1"}:
            return "1H"
        return self.nucleus

    @property
    def field_strength_t(self) -> Optional[float]:
        value = self.result.metadata.get("field_strength_t")
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    @property
    def reference_frequency_mhz(self) -> float:
        if self.field_strength_t is None:
            return 0.0
        gamma_hz_per_t = 42_577_478.518
        if self.nucleus.upper().replace("-", "") in {"13C", "C13"}:
            gamma_hz_per_t = 10_708_400.0
        return self.field_strength_t * gamma_hz_per_t / 1e6

    @property
    def spectral_reference_ppm(self) -> float:
        value = self.result.metadata.get("sequence_reference_ppm")
        if value is None:
            value = self.result.metadata.get("spectral_reference_ppm")
        if value is None and self.phantom is not None:
            metadata = getattr(self.phantom, "metadata", {})
            value = metadata.get("sequence_reference_ppm")
        if value is None and self.phantom is not None:
            value = getattr(self.phantom, "spectral_reference_ppm", None)
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return 4.7 if self.bruker_nucleus == "1H" else 0.0
        return scalar if np.isfinite(scalar) else 0.0

    @property
    def working_frequency_mhz(self) -> float:
        return self.reference_frequency_mhz * (1.0 + self.spectral_reference_ppm * 1e-6)

    @property
    def spectral_bandwidth_ppm(self) -> float:
        if self.reference_frequency_mhz <= 0:
            return 0.0
        return (1.0 / self.dwell_s) / self.reference_frequency_mhz

    @property
    def spectroscopy(self):
        try:
            return self.result.spectroscopic_acquisition
        except Exception:
            return None

    @property
    def write_fid(self) -> bool:
        return self.options.raw_data_files in {"fid", "both"}

    @property
    def write_rawdata_job0(self) -> bool:
        return self.options.raw_data_files in {"rawdata.job0", "both"}


def _as_coil_signal(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.complex128)
    if values.ndim == 1:
        return np.ascontiguousarray(values.reshape(1, -1))
    if values.ndim == 2:
        return np.ascontiguousarray(values)
    raise ValueError("Bruker raw export requires signal shape (adc,) or (coil, adc)")


def _merge_options(
    options: Optional[BrukerExportOptions],
    *,
    method_name: Optional[str],
    scan_name: Optional[str],
    matrix: Optional[tuple[int, int]],
    fov_m: Optional[tuple[float, float]],
    slice_thickness_mm: Optional[float],
    raw_data_files: Optional[str],
) -> BrukerExportOptions:
    base = options or BrukerExportOptions()
    merged = BrukerExportOptions(
        method_name=method_name if method_name is not None else base.method_name,
        scan_name=scan_name if scan_name is not None else base.scan_name,
        matrix=matrix if matrix is not None else base.matrix,
        fov_m=fov_m if fov_m is not None else base.fov_m,
        slice_thickness_mm=(
            slice_thickness_mm
            if slice_thickness_mm is not None
            else base.slice_thickness_mm
        ),
        read_orientation=base.read_orientation,
        slice_orientation=base.slice_orientation,
        patient_position=base.patient_position,
        raw_data_files=(
            raw_data_files if raw_data_files is not None else base.raw_data_files
        ),
    )
    return _validate_options(merged)


def _validate_options(options: BrukerExportOptions) -> BrukerExportOptions:
    method = _normalize_angle_text(options.method_name, "method_name")
    scan_name = None
    if options.scan_name is not None:
        scan_name = _normalize_plain_text(options.scan_name, "scan_name")

    matrix = None
    if options.matrix is not None:
        if len(options.matrix) != 2:
            raise ValueError("Bruker matrix override must contain two values")
        matrix = tuple(int(value) for value in options.matrix)
        if any(value <= 0 for value in matrix):
            raise ValueError("Bruker matrix override values must be positive")

    fov = None
    if options.fov_m is not None:
        if len(options.fov_m) != 2:
            raise ValueError("Bruker FOV override must contain two values")
        fov = tuple(float(value) for value in options.fov_m)
        if any(not np.isfinite(value) or value <= 0 for value in fov):
            raise ValueError("Bruker FOV override values must be positive and finite")

    thickness = float(options.slice_thickness_mm)
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("slice_thickness_mm must be positive and finite")

    return BrukerExportOptions(
        method_name=method,
        scan_name=scan_name,
        matrix=matrix,
        fov_m=fov,
        slice_thickness_mm=thickness,
        read_orientation=_normalize_plain_text(
            options.read_orientation, "read_orientation"
        ),
        slice_orientation=_normalize_plain_text(
            options.slice_orientation, "slice_orientation"
        ),
        patient_position=_normalize_plain_text(
            options.patient_position, "patient_position"
        ),
        raw_data_files=_normalize_raw_data_files(options.raw_data_files),
    )


def _normalize_angle_text(value: str, name: str) -> str:
    text = str(value).strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1].strip()
    return _normalize_plain_text(text, name)


def _normalize_plain_text(value: str, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text.replace("<", "_").replace(">", "_")


def _normalize_raw_data_files(value: str) -> str:
    text = str(value).strip().lower()
    aliases = {
        "fid": "fid",
        "rawdata": "rawdata.job0",
        "rawdata.job0": "rawdata.job0",
        "job0": "rawdata.job0",
        "both": "both",
        "fid+rawdata.job0": "both",
        "fid + rawdata.job0": "both",
    }
    if text not in aliases:
        raise ValueError("raw_data_files must be 'fid', 'rawdata.job0', or 'both'")
    return aliases[text]


def _signal_scale(signal: np.ndarray, requested: Optional[float]) -> float:
    if requested is not None:
        scale = float(requested)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be positive and finite")
        return scale
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if not np.isfinite(peak) or peak == 0.0:
        return 1.0
    return INT32_TARGET / peak


def _interleaved_int32(values: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.clip(
        np.rint(np.stack((values.real, values.imag), axis=-1) * scale),
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
    )
    return np.ascontiguousarray(scaled.astype("<i4", copy=False).reshape(-1))


def _bruker_fid_int32(context: _ExportContext) -> np.ndarray:
    readout = context.readout_samples
    if context.num_adc_samples % readout:
        raise ValueError("ADC sample count must be divisible by the readout length")
    readouts = context.num_readouts
    line_values = context.signal.reshape(context.num_coils, readouts, readout)
    line_values = np.moveaxis(line_values, 0, 1).reshape(readouts, -1)
    block_complex = _standard_kblock_complex_size(readout, context.num_coils)
    padded = np.zeros((readouts, block_complex), dtype=np.complex128)
    padded[:, : line_values.shape[1]] = line_values
    return _interleaved_int32(padded, context.scale)


def _rawdata_job0_int32(context: _ExportContext) -> np.ndarray:
    spectroscopy = context.spectroscopy
    if spectroscopy is not None:
        if context.is_luca_csi4:
            chronological = context.signal.reshape(
                context.num_coils,
                spectroscopy.num_encodings,
                spectroscopy.spectral_points,
            )
            line_values = np.moveaxis(chronological, 0, 1).reshape(-1)
            return _interleaved_int32(line_values, context.scale)
        grid = spectroscopy.reshape_signal(context.signal)
        line_values = np.moveaxis(grid, 0, -2) if context.signal.ndim == 2 else grid
        line_values = np.asarray(line_values).reshape(-1)
        return _interleaved_int32(line_values, context.scale)
    line_values = np.moveaxis(context.signal, 0, 1).reshape(-1)
    return _interleaved_int32(line_values, context.scale)


def _standard_kblock_complex_size(readout_samples: int, num_coils: int) -> int:
    bytes_per_int = 4
    acq_size = 2 * int(readout_samples)
    block_ints = np.ceil(acq_size * int(num_coils) * bytes_per_int / 1024.0)
    return int(block_ints * 1024.0 / bytes_per_int / 2)


def _csi_encoding_metadata(context: _ExportContext) -> dict[str, tuple]:
    spectroscopy = context.spectroscopy
    if spectroscopy is None:
        raise ValueError("CSI encoding metadata requires a spectroscopic acquisition")
    if tuple(context.matrix) != tuple(spectroscopy.matrix):
        raise ValueError(
            "CSI export requires PVM_Matrix to match the simulated CSI grid"
        )

    n_x, n_y = context.matrix
    chronological = tuple(
        index
        for index, repetition in zip(
            spectroscopy.encoding_indices, spectroscopy.repetition_indices
        )
        if repetition == 0
    )
    if len(chronological) != n_x * n_y:
        raise ValueError("CSI export requires one complete first repetition")
    for repetition in range(1, spectroscopy.num_repetitions):
        repeated_order = tuple(
            index
            for index, value in zip(
                spectroscopy.encoding_indices, spectroscopy.repetition_indices
            )
            if value == repetition
        )
        if repeated_order != chronological:
            raise ValueError(
                "Bruker CSI export requires the same phase-encoding order in "
                "every repetition"
            )

    steps_x = tuple(int(index - n_x // 2) for index in range(n_x))
    steps_y = tuple(int(index - n_y // 2) for index in range(n_y))
    signed_x = tuple(int(x - n_x // 2) for x, _ in chronological)
    signed_y = tuple(int(y - n_y // 2) for _, y in chronological)
    values_x = tuple(2.0 * value / n_x for value in steps_x)
    values_y = tuple(2.0 * value / n_y for value in steps_y)
    chronological_values_x = tuple(values_x[x] for x, _ in chronological)
    chronological_values_y = tuple(values_y[y] for _, y in chronological)
    fov_mm = tuple(value * 1000.0 for value in context.fov_m)
    distances = tuple(
        float(np.hypot(x / fov_mm[0], y / fov_mm[1]))
        for x, y in zip(signed_x, signed_y)
    )
    if context.is_luca_csi4 and np.any(np.diff(distances) < -1e-12):
        raise ValueError(
            "User:lucaCSI4 export requires a centric CSI acquisition order"
        )
    return {
        "indices": chronological,
        "steps_x": steps_x,
        "steps_y": steps_y,
        "signed_x": signed_x,
        "signed_y": signed_y,
        "values_x": values_x,
        "values_y": values_y,
        "chronological_values_x": chronological_values_x,
        "chronological_values_y": chronological_values_y,
        "distances": distances,
        "reco_x": tuple(x for x, _ in chronological),
        "reco_y": tuple(y for _, y in chronological),
    }


def _write_acqp(path: Path, context: _ExportContext) -> None:
    readout = context.readout_samples
    readouts = context.readouts_per_frame
    frame_count = context.raw_frame_count
    spectroscopy = context.spectroscopy
    if spectroscopy is not None:
        acq_dim = 3
        acq_dim_desc = (
            _bare("Spectroscopic"),
            _bare("Spatial"),
            _bare("Spatial"),
        )
        acq_size = (2 * readout, context.matrix[0], context.matrix[1])
        frame_count = 1
        repetitions = spectroscopy.num_repetitions
    else:
        acq_dim = 2
        acq_dim_desc = (_bare("Spatial"), _bare("Spatial"))
        acq_size = (2 * readout, readouts)
        repetitions = 1
    lines = [
        _jcamp_scalar("ACQ_scan_name", _angle(context.scan_name)),
        _jcamp_scalar(
            "ACQ_time",
            _angle(context.created.strftime("%Y-%m-%dT%H:%M:%S,%f%z")),
        ),
        _jcamp_scalar("ACQ_method", _angle(context.method_name)),
        _jcamp_scalar("PULPROG", _angle("BlochSimulator.ppg")),
        _jcamp_scalar("ACQ_experiment_mode", "SingleExperiment"),
        _jcamp_scalar("ACQ_dim", acq_dim),
        _jcamp_array("ACQ_dim_desc", (acq_dim,), acq_dim_desc),
        _jcamp_array("ACQ_size", (acq_dim,), acq_size),
        _jcamp_scalar("NI", frame_count),
        _jcamp_scalar("NA", 1),
        _jcamp_scalar("NR", repetitions),
        _jcamp_scalar("DS", 0),
        _jcamp_scalar("SW_h", 1.0 / context.dwell_s),
        _jcamp_scalar("DW", context.dwell_s * 1e6),
        _jcamp_scalar("RG", 1),
        _jcamp_scalar("ACQ_patient_pos", context.options.patient_position),
        _jcamp_array("ACQ_grad_matrix", (1, 3, 3), (1, 0, 0, 0, 1, 0, 0, 0, 1)),
        _jcamp_scalar("BYTORDA", "little"),
        _jcamp_scalar("GO_raw_data_format", "GO_32BIT_SGN_INT"),
        _jcamp_scalar("GO_block_size", "Standard_KBlock_Format"),
        _jcamp_scalar("ACQ_ReceiverChannels", context.num_coils),
        _jcamp_scalar("BLOCHSIM_signal_scale", context.scale),
        _jcamp_scalar(
            "BLOCHSIM_signal_order",
            _angle(
                "encoding,coil,spectral_point,real_imag"
                if context.is_luca_csi4
                else "coil,adc,real_imag"
            ),
        ),
        _jcamp_scalar("BLOCHSIM_adc_samples", context.num_adc_samples),
    ]
    if spectroscopy is not None:
        encoding = _csi_encoding_metadata(context)
        lines.extend(
            [
                _jcamp_scalar("ACQ_flip_angle", context.flip_angle_deg),
                _jcamp_array(
                    "ACQ_repetition_time", (1,), (context.repetition_time_ms,)
                ),
                _jcamp_array(
                    "ACQ_phase_encoding_mode",
                    (2,),
                    (
                        _bare("User_Defined_Encoding"),
                        _bare("User_Defined_Encoding"),
                    ),
                ),
                _jcamp_array("ACQ_phase_enc_start", (2,), (-1, -1)),
                _jcamp_scalar(
                    "ACQ_spatial_size_0", spectroscopy.encodings_per_repetition
                ),
                _jcamp_array(
                    "ACQ_spatial_phase_0",
                    (spectroscopy.encodings_per_repetition,),
                    encoding["chronological_values_x"],
                ),
                _jcamp_scalar(
                    "ACQ_spatial_size_1", spectroscopy.encodings_per_repetition
                ),
                _jcamp_array(
                    "ACQ_spatial_phase_1",
                    (spectroscopy.encodings_per_repetition,),
                    encoding["chronological_values_y"],
                ),
                _jcamp_scalar("BF1", context.reference_frequency_mhz),
                _jcamp_scalar("SFO1", context.working_frequency_mhz),
                _jcamp_array("NUC1", (8,), (_angle(context.bruker_nucleus),)),
            ]
        )
    _write_jcamp(
        path, "Parameter List, BlochSimulator Bruker raw export", lines, context
    )


def _write_method(path: Path, context: _ExportContext) -> None:
    matrix = context.matrix
    spectroscopy = context.spectroscopy
    fov_mm = tuple(value * 1000.0 for value in context.fov_m)
    resolution = tuple(fov / max(size, 1) for fov, size in zip(fov_mm, matrix))
    repetitions = spectroscopy.num_repetitions if spectroscopy is not None else 1
    object_count = 1 if spectroscopy is not None else context.frame_count
    bandwidth_hz = 1.0 / context.dwell_s
    lines = [
        _jcamp_scalar("Method", _angle(context.method_name)),
        _jcamp_scalar("BLOCHSIM_Source", _angle(context.source)),
        _jcamp_scalar("BLOCHSIM_signal_scale", context.scale),
        _jcamp_scalar("BLOCHSIM_raw_data_format", _angle("GO_32BIT_SGN_INT")),
        _jcamp_array("PVM_Matrix", (2,), matrix),
        _jcamp_array("PVM_EncMatrix", (2,), matrix),
        _jcamp_array("PVM_Fov", (2,), fov_mm),
        _jcamp_array(
            "PVM_FovCm", (2,), tuple(value * 100.0 for value in context.fov_m)
        ),
        _jcamp_array("PVM_SpatResol", (2,), resolution),
        _jcamp_scalar("PVM_SpatDimEnum", _angle("2D" if matrix[1] > 1 else "1D")),
        _jcamp_scalar("PVM_NAverages", 1),
        _jcamp_scalar("PVM_NRepetitions", repetitions),
        _jcamp_scalar("PVM_NEchoImages", 1),
        _jcamp_scalar("PVM_EncNReceivers", context.num_coils),
        _jcamp_scalar("PVM_EncUseMultiRec", "Yes" if context.num_coils > 1 else "No"),
        _jcamp_array(
            "PVM_EncActReceivers", (context.num_coils,), ("On",) * context.num_coils
        ),
        _jcamp_scalar(
            "PVM_EncSpectroscopy",
            "No" if context.is_luca_csi4 else ("Yes" if spectroscopy else "No"),
        ),
        _jcamp_scalar("PVM_NSPacks", 1),
        _jcamp_array("PVM_SPackArrNSlices", (1,), (object_count,)),
        _jcamp_array("PVM_ObjOrderList", (object_count,), range(object_count)),
        _jcamp_scalar("PVM_SliceThick", context.options.slice_thickness_mm),
        _jcamp_array("PVM_SPackArrSliceGap", (1,), (0.0,)),
        _jcamp_array("PVM_SPackArrReadOffset", (1,), (0.0,)),
        _jcamp_array("PVM_SPackArrPhase0Offset", (1,), (0.0,)),
        _jcamp_array("PVM_SPackArrPhase1Offset", (1,), (0.0,)),
        _jcamp_array("PVM_SPackArrSliceOffset", (1,), (0.0,)),
        _jcamp_array("PVM_SliceOffset", (object_count,), (0.0,) * object_count),
        _jcamp_scalar("PVM_SPackArrReadOrient", context.options.read_orientation),
        _jcamp_scalar("PVM_SPackArrSliceOrient", context.options.slice_orientation),
        _jcamp_scalar("PVM_RepetitionTime", context.repetition_time_ms),
        _jcamp_scalar("PVM_EchoTime", context.echo_time_ms),
        _jcamp_scalar("PVM_ScanTime", context.duration_ms),
        _jcamp_scalar("PVM_EffSwh", bandwidth_hz),
        _jcamp_scalar("PVM_EffSWh", bandwidth_hz),
        _jcamp_scalar("PVM_Dw", context.dwell_s * 1e6),
        _jcamp_scalar("PVM_DigDw", context.dwell_s * 1000.0),
        _jcamp_scalar("PVM_Nucleus1Enum", _angle(context.bruker_nucleus)),
        _jcamp_array("PVM_Nucleus1", (8,), (_angle(context.bruker_nucleus),)),
        _jcamp_array(
            "PVM_FrqRef", (8,), (context.reference_frequency_mhz,) + (0.0,) * 7
        ),
        _jcamp_array(
            "PVM_FrqWork", (8,), (context.working_frequency_mhz,) + (0.0,) * 7
        ),
        _jcamp_array(
            "PVM_FrqWorkOffset",
            (8,),
            (context.spectral_reference_ppm * context.reference_frequency_mhz,)
            + (0.0,) * 7,
        ),
        _jcamp_array("PVM_FrqRefPpm", (8,), (0.0,) * 8),
        _jcamp_array(
            "PVM_FrqWorkOffsetPpm",
            (8,),
            (context.spectral_reference_ppm,) + (0.0,) * 7,
        ),
        _jcamp_array(
            "PVM_FrqWorkPpm",
            (8,),
            (context.spectral_reference_ppm,) + (0.0,) * 7,
        ),
    ]
    if spectroscopy is not None:
        encoding = _csi_encoding_metadata(context)
        spec_acq_time_ms = spectroscopy.spectral_points * spectroscopy.dwell_s * 1000.0
        if context.is_luca_csi4:
            spec_lines = [
                _jcamp_array(
                    "PVM_EncOrder",
                    (2,),
                    (_bare("LINEAR_ENC"), _bare("LINEAR_ENC")),
                ),
                _jcamp_array("PVM_SpecMatrix", (1,), (spectroscopy.spectral_points,)),
                _jcamp_array(
                    "PVM_SpecSWH", (1,), (spectroscopy.spectral_bandwidth_hz,)
                ),
                _jcamp_array("PVM_SpecSW", (1,), (context.spectral_bandwidth_ppm,)),
                _jcamp_array("PVM_SpecDwellTime", (1,), (spectroscopy.dwell_s * 5e5,)),
                _jcamp_array(
                    "PVM_SpecNomRes",
                    (1,),
                    (
                        spectroscopy.spectral_bandwidth_hz
                        / (2.0 * spectroscopy.spectral_points),
                    ),
                ),
            ]
        else:
            spec_lines = [
                _jcamp_scalar("PVM_EncOrder", _bare("LINEAR_ENC LINEAR_ENC")),
                _jcamp_scalar("PVM_SpecMatrix", spectroscopy.spectral_points),
                _jcamp_scalar("PVM_SpecSWH", spectroscopy.spectral_bandwidth_hz),
                _jcamp_scalar("PVM_SpecDwellTime", spectroscopy.dwell_s * 1e6),
            ]
        lines.extend(
            [
                *spec_lines,
                _jcamp_scalar("PVM_SpecDimEnum", _angle("1D")),
                _jcamp_array("PVM_EncSteps0", (matrix[0],), encoding["steps_x"]),
                _jcamp_array("PVM_EncSteps1", (matrix[1],), encoding["steps_y"]),
                _jcamp_array("PVM_EncValues0", (matrix[0],), encoding["values_x"]),
                _jcamp_array("PVM_EncValues1", (matrix[1],), encoding["values_y"]),
                _jcamp_scalar("PVM_EncCentralStep0", 1),
                _jcamp_scalar("PVM_EncCentralStep1", 1),
                _jcamp_array("PVM_EncStart", (2,), (-1, -1)),
                _jcamp_scalar(
                    "PVM_SpecAcquisitionTime",
                    spec_acq_time_ms,
                ),
                _jcamp_scalar("PVM_SpecOffsetHz", 0.0),
                _jcamp_scalar("PVM_SpecOffsetppm", 0.0),
                _jcamp_scalar("PVM_DigNp", spectroscopy.spectral_points),
                _jcamp_scalar("PVM_DigSw", spectroscopy.spectral_bandwidth_hz),
                _jcamp_scalar("PVM_DigDur", spec_acq_time_ms),
            ]
        )
        if not context.is_luca_csi4:
            lines.append(
                _jcamp_scalar(
                    "PVM_SpecNomRes",
                    spectroscopy.spectral_bandwidth_hz
                    / (2.0 * spectroscopy.spectral_points),
                )
            )
        if context.is_luca_csi4:
            count = spectroscopy.encodings_per_repetition
            zeros = (0.0,) * count
            lines.extend(
                [
                    _jcamp_scalar("CentricEncOrder_OnOff", "On"),
                    _jcamp_array(
                        "CentricEncOrderMatrixx", (count,), encoding["signed_x"]
                    ),
                    _jcamp_array(
                        "CentricEncOrderMatrixy", (count,), encoding["signed_y"]
                    ),
                    _jcamp_array(
                        "Distance2kspacecenter", (count,), encoding["distances"]
                    ),
                    _jcamp_array("Angles", (count,), zeros),
                    _jcamp_scalar("SpiralEncOrder_OnOff", "Off"),
                    _jcamp_array("SpiralEncOrderMatrixx", (count,), (0,) * count),
                    _jcamp_array("SpiralEncOrderMatrixy", (count,), (0,) * count),
                    _jcamp_scalar("PhaseEncGrad_OnOff", "On"),
                    _jcamp_scalar("KFiltering", "On"),
                    _jcamp_array("Reco_x", (count,), encoding["reco_x"]),
                    _jcamp_array("Reco_y", (count,), encoding["reco_y"]),
                ]
            )
    if context.field_strength_t is not None:
        lines.append(_jcamp_scalar("BLOCHSIM_FieldStrengthT", context.field_strength_t))
    _write_jcamp(path, "Parameter List, BlochSimulator method", lines, context)


def _write_visu_pars(path: Path, context: _ExportContext) -> None:
    matrix = context.matrix
    fov_mm = tuple(value * 1000.0 for value in context.fov_m)
    spectroscopy = context.spectroscopy
    if spectroscopy is not None:
        core_dim = 3
        core_size = (spectroscopy.spectral_points, matrix[0], matrix[1])
        core_desc = (
            _bare("spectroscopic"),
            _bare("spatial"),
            _bare("spatial"),
        )
        core_extent = (context.spectral_bandwidth_ppm,) + fov_mm
        core_units = (_angle("ppm"), _angle("mm"), _angle("mm"))
        frame_count = spectroscopy.num_repetitions
    else:
        core_dim = 2
        core_size = matrix
        core_desc = (_bare("spatial"), _bare("spatial"))
        core_extent = fov_mm
        core_units = (_angle("mm"), _angle("mm"))
        frame_count = context.frame_count
    lines = [
        _jcamp_scalar("VisuVersion", 3),
        _jcamp_scalar("VisuCreator", _angle("BlochSimulator")),
        _jcamp_scalar("VisuCreatorVersion", _angle("7.0.0")),
        _jcamp_scalar(
            "VisuCreationDate",
            _angle(context.created.strftime("%Y-%m-%dT%H:%M:%S,%f%z")),
        ),
        _jcamp_scalar("VisuInstanceType", "STANDARD_INSTANCE"),
        _jcamp_array("VisuInstanceModality", (65,), (_angle("MR"),)),
        _jcamp_scalar("VisuCoreFrameCount", frame_count),
        _jcamp_scalar("VisuCoreDim", core_dim),
        _jcamp_array("VisuCoreSize", (core_dim,), core_size),
        _jcamp_array("VisuCoreDimDesc", (core_dim,), core_desc),
        _jcamp_array("VisuCoreExtent", (core_dim,), core_extent),
        _jcamp_array("VisuCoreUnits", (core_dim, 65), core_units),
        _jcamp_scalar("VisuCoreWordType", "_32BIT_SGN_INT"),
        _jcamp_scalar("VisuCoreByteOrder", "littleEndian"),
        _jcamp_array("VisuSubjectName", (65,), (_angle("BlochSimulator"),)),
        _jcamp_array("VisuSubjectId", (65,), (_angle("simulated"),)),
        _jcamp_scalar("VisuSubjectType", "Material"),
        _jcamp_scalar("VisuSubjectPosition", context.options.patient_position),
        _jcamp_array("VisuStudyId", (65,), (_angle("BlochSimulator"),)),
        _jcamp_scalar("VisuExperimentNumber", 1),
        _jcamp_scalar("BLOCHSIM_signal_scale", context.scale),
    ]
    _write_jcamp(path, "Parameter List, BlochSimulator visu_pars", lines, context)


def _write_pulseprogram(path: Path, context: _ExportContext) -> None:
    lines = [
        "; BlochSimulator generated Bruker-style metadata",
        f"; Source: {context.source}",
        f"; ADC samples: {context.num_adc_samples}",
        f"; Receiver channels: {context.num_coils}",
        "1 ze",
        "  aq",
        "exit",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_specpar(path: Path, context: _ExportContext) -> None:
    lines = [
        _jcamp_scalar("BF1", context.reference_frequency_mhz),
        _jcamp_scalar("SFO1", context.working_frequency_mhz),
        _jcamp_scalar("NUC1", _angle(context.bruker_nucleus)),
    ]
    _write_jcamp(
        path, "Parameter List, BlochSimulator spectrometer placeholder", lines, context
    )


def _write_placeholder_files(output_dir: Path, context: _ExportContext) -> None:
    _write_jcamp(
        output_dir / "AdjStatePerScan",
        "Parameter List, BlochSimulator adjustment state",
        [_jcamp_scalar("AdjScanStateCompleted", "No")],
        context,
    )
    _write_jcamp(
        output_dir / "configscan",
        "Parameter List, BlochSimulator scan configuration",
        [
            _jcamp_scalar("CONFIG_SCAN_version", 1),
            _jcamp_scalar("BLOCHSIM_receiver_channels", context.num_coils),
        ],
        context,
    )
    (output_dir / "uxnmr.info").write_text(
        "CONFIGURATION INFORMATION\n"
        "=========================\n\n"
        "Generated by BlochSimulator Bruker raw export.\n",
        encoding="utf-8",
    )
    _write_jcamp(
        output_dir / "uxnmr.par",
        "Parameter file, BlochSimulator",
        [_jcamp_scalar("BLOCHSIM_generated", "Yes")],
        context,
    )
    (output_dir / "spnam0").write_text(
        "BlochSimulator sequence export does not include scanner RF shape files.\n",
        encoding="utf-8",
    )


def _write_reconstructed_pdata(path: Path, context: _ExportContext) -> None:
    if context.spectroscopy is not None:
        _write_reconstructed_csi_pdata(path, context)
        return
    image_stack = _reconstruct_image_stack(context)
    if image_stack is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        return
    path.mkdir(parents=True, exist_ok=True)
    magnitude = np.abs(image_stack)
    peak = float(np.max(magnitude)) if magnitude.size else 0.0
    slope = peak / float(np.iinfo(np.int16).max) if peak > 0 else 1.0
    int_image = np.rint(magnitude / slope).clip(0, np.iinfo(np.int16).max)
    np.ascontiguousarray(int_image.astype("<i2", copy=False)).tofile(path / "2dseq")

    frame_count = int_image.shape[0]
    size = (int_image.shape[-1], int_image.shape[-2])
    fov_mm = tuple(value * 1000.0 for value in context.fov_m)
    lines = [
        _jcamp_scalar("RecoDim", 2),
        _jcamp_scalar("RecoObjectsPerRepetition", frame_count),
        _jcamp_array("RECO_size", (2,), size),
        _jcamp_array("RECO_fov", (2,), tuple(value / 10.0 for value in fov_mm)),
        _jcamp_scalar("RECO_wordtype", "_16BIT_SGN_INT"),
        _jcamp_scalar("RECO_byte_order", "littleEndian"),
        _jcamp_scalar("RECO_image_type", "MAGNITUDE_IMAGE"),
        _jcamp_array("RECO_map_slope", (frame_count,), (slope,) * frame_count),
    ]
    _write_jcamp(path / "reco", "Parameter List, BlochSimulator reco", lines, context)
    _write_jcamp(path / "procs", "Parameter List, BlochSimulator procs", [], context)
    _write_jcamp(
        path / "methreco", "Parameter List, BlochSimulator methreco", [], context
    )
    _write_jcamp(path / "id", "Parameter List, BlochSimulator id", [], context)

    visu_lines = [
        _jcamp_scalar("VisuVersion", 3),
        _jcamp_scalar("VisuCreator", _angle("BlochSimulator")),
        _jcamp_scalar("VisuCreatorVersion", _angle("7.0.0")),
        _jcamp_scalar("VisuCoreFrameCount", frame_count),
        _jcamp_scalar("VisuCoreDim", 2),
        _jcamp_array("VisuCoreSize", (2,), size),
        _jcamp_array("VisuCoreDimDesc", (2,), (_bare("spatial"), _bare("spatial"))),
        _jcamp_array("VisuCoreExtent", (2,), fov_mm),
        _jcamp_array("VisuCoreDataSlope", (frame_count,), (slope,) * frame_count),
        _jcamp_array("VisuCoreDataOffs", (frame_count,), (0.0,) * frame_count),
        _jcamp_array("VisuCoreFrameType", (1,), (_bare("MAGNITUDE_IMAGE"),)),
        _jcamp_scalar("VisuCoreWordType", "_16BIT_SGN_INT"),
        _jcamp_scalar("VisuCoreByteOrder", "littleEndian"),
        _jcamp_array("VisuCorePosition", (frame_count, 3), (0.0,) * frame_count * 3),
        _jcamp_scalar("VisuFGOrderDescDim", 1),
        _jcamp_array(
            "VisuFGOrderDesc",
            (1,),
            (_bare(f"({frame_count}, <FG_SLICE>, <>, 0, 2)"),),
        ),
    ]
    _write_jcamp(
        path / "visu_pars",
        "Parameter List, BlochSimulator pdata visu_pars",
        visu_lines,
        context,
    )


def _write_reconstructed_csi_pdata(path: Path, context: _ExportContext) -> None:
    spectroscopy = context.spectroscopy
    spectra = np.asarray(spectroscopy.reconstruct_spectra(context.signal))
    magnitude = np.sqrt(np.sum(np.abs(spectra) ** 2, axis=0))
    if spectroscopy.num_repetitions == 1:
        magnitude = magnitude[None, ...]

    path.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(magnitude)) if magnitude.size else 0.0
    slope = peak / float(np.iinfo(np.int16).max) if peak > 0 else 1.0
    int_spectra = np.rint(magnitude / slope).clip(0, np.iinfo(np.int16).max)
    np.ascontiguousarray(int_spectra.astype("<i2", copy=False)).tofile(path / "2dseq")

    repetitions = spectroscopy.num_repetitions
    n_x, n_y = context.matrix
    size = (spectroscopy.spectral_points, n_x, n_y)
    fov_mm = tuple(value * 1000.0 for value in context.fov_m)
    # ParaVision's spectroscopic reco header repeats the x FOV for the
    # spectroscopic axis before the two spatial FOV entries.
    reco_fov_cm = (
        fov_mm[0] / 10.0,
        fov_mm[0] / 10.0,
        fov_mm[1] / 10.0,
    )
    reco_lines = [
        _jcamp_scalar("RecoDim", 3),
        _jcamp_scalar("RecoObjectsPerRepetition", 1),
        _jcamp_scalar("RecoNumRepetitions", repetitions),
        _jcamp_array("RECO_size", (3,), size),
        _jcamp_array("RECO_inp_size", (3,), size),
        _jcamp_array("RECO_ft_size", (3,), size),
        _jcamp_array("RECO_fov", (3,), reco_fov_cm),
        _jcamp_scalar("RECO_wordtype", "_16BIT_SGN_INT"),
        _jcamp_scalar("RECO_byte_order", "littleEndian"),
        _jcamp_scalar("RECO_image_type", "MAGNITUDE_IMAGE"),
        _jcamp_array("RECO_map_slope", (repetitions,), (slope,) * repetitions),
    ]
    _write_jcamp(
        path / "reco", "Parameter List, BlochSimulator CSI reco", reco_lines, context
    )
    _write_jcamp(
        path / "procs",
        "Parameter List, BlochSimulator CSI procs",
        [
            _jcamp_scalar("SF", context.reference_frequency_mhz),
            _jcamp_scalar(
                "OFFSET",
                context.spectral_reference_ppm + context.spectral_bandwidth_ppm / 2.0,
            ),
            _jcamp_scalar("SW_p", spectroscopy.spectral_bandwidth_hz),
            _jcamp_scalar("SI", spectroscopy.spectral_points),
        ],
        context,
    )
    _write_jcamp(
        path / "methreco", "Parameter List, BlochSimulator methreco", [], context
    )
    _write_jcamp(path / "id", "Parameter List, BlochSimulator id", [], context)

    visu_lines = [
        _jcamp_scalar("VisuVersion", 3),
        _jcamp_scalar("VisuCreator", _angle("BlochSimulator")),
        _jcamp_scalar("VisuCreatorVersion", _angle("7.0.0")),
        _jcamp_scalar("VisuCoreFrameCount", repetitions),
        _jcamp_scalar("VisuCoreDim", 3),
        _jcamp_array("VisuCoreSize", (3,), size),
        _jcamp_array(
            "VisuCoreDimDesc",
            (3,),
            (_bare("spectroscopic"), _bare("spatial"), _bare("spatial")),
        ),
        _jcamp_array(
            "VisuCoreExtent", (3,), (context.spectral_bandwidth_ppm,) + fov_mm
        ),
        _jcamp_array(
            "VisuCoreUnits",
            (3, 65),
            (_angle("ppm"), _angle("mm"), _angle("mm")),
        ),
        _jcamp_array("VisuCoreDataSlope", (repetitions,), (slope,) * repetitions),
        _jcamp_array("VisuCoreDataOffs", (repetitions,), (0.0,) * repetitions),
        _jcamp_array("VisuCoreFrameType", (1,), (_bare("MAGNITUDE_IMAGE"),)),
        _jcamp_scalar("VisuCoreWordType", "_16BIT_SGN_INT"),
        _jcamp_scalar("VisuCoreByteOrder", "littleEndian"),
        _jcamp_array("VisuCorePosition", (repetitions, 3), (0.0,) * repetitions * 3),
        _jcamp_scalar("VisuFGOrderDescDim", 1),
        _jcamp_array(
            "VisuFGOrderDesc",
            (1,),
            (_bare(f"({repetitions}, <FG_MOVIE>, <>, 0, 3)"),),
        ),
    ]
    _write_jcamp(
        path / "visu_pars",
        "Parameter List, BlochSimulator CSI pdata visu_pars",
        visu_lines,
        context,
    )


def _reconstruct_image_stack(context: _ExportContext) -> Optional[np.ndarray]:
    try:
        if context.acquisition_frames is not None:
            frames = []
            for frame in range(context.acquisition_frames.num_frames):
                image = context.acquisition_frames.reconstruct(
                    context.result,
                    frame,
                    coil_combine="rss" if context.signal.shape[0] > 1 else None,
                )
                frames.append(_drop_coil_axis(image))
            return np.stack(frames, axis=0)
        if context.acquisition is not None:
            if context.result.signal.shape[-1] != context.acquisition.num_samples:
                return None
            image = context.result.reconstruct_cartesian(
                context.acquisition,
                coil_combine="rss" if context.signal.shape[0] > 1 else None,
            )
            return _drop_coil_axis(image)[None, ...]
    except Exception:
        return None
    return None


def _drop_coil_axis(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 3:
        return np.sqrt(np.sum(np.abs(values) ** 2, axis=0))
    return values


def _write_jcamp(
    path: Path, title: str, body: list[str], context: _ExportContext
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = context.created.strftime("%Y-%m-%d %H:%M:%S.%f %z")
    lines = [
        f"##TITLE={title}",
        "##JCAMPDX=4.24",
        "##DATATYPE=Parameter Values",
        "##ORIGIN=BlochSimulator",
        "##OWNER=simulated",
        f"$$ {timestamp}  BlochSimulator",
        *body,
        "##END=",
        f"$$ File finished by BlochSimulator at {timestamp}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _jcamp_scalar(name: str, value) -> str:
    return f"##${name}={_format_value(value)}"


def _jcamp_array(name: str, shape: tuple[int, ...], values) -> str:
    shape_text = ", ".join(str(int(value)) for value in shape)
    value_text = _format_values(values)
    return f"##${name}=( {shape_text} )\n{value_text}"


def _format_values(values) -> str:
    if isinstance(values, np.ndarray):
        iterable = values.reshape(-1).tolist()
    else:
        iterable = list(values)
    chunks = []
    for start in range(0, len(iterable), 8):
        chunks.append(
            " ".join(_format_value(value) for value in iterable[start : start + 8])
        )
    return " \n".join(chunks)


def _format_value(value) -> str:
    if isinstance(value, _BareValue):
        return value.text
    if isinstance(value, str):
        return value
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.15g}"
    return str(value)


class _BareValue:
    def __init__(self, text: str):
        self.text = str(text)


def _bare(text: str) -> _BareValue:
    return _BareValue(text)


def _angle(text: str) -> str:
    escaped = str(text).replace(">", "_")
    return f"<{escaped}>"
