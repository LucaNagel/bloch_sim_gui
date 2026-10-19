"""NIfTI export of simulated sequence images.

Each image series of :func:`~blochsimulator.sequence.image_export.extract_image_series`
becomes one NIfTI-1 file with a BIDS-style JSON sidecar.

Data layout
-----------
``data[i, j, k, t]`` with ``i`` along the image columns (readout), ``j`` along
the image rows (phase encoding), ``k`` over slices or partitions and ``t``
over echoes (fastest) and repetitions.  Single-volume series are stored as
3-D images.

The affine maps voxel centres to scanner RAS millimetres.  Simulator
gradient coordinates are patient LPS for a head-first supine subject (the
same convention as the DICOM and Bruker exports), so the affine is the LPS
matrix with its first two rows negated.  :mod:`nibabel` is an optional
dependency (``pip install blochsimulator[export]``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .bruker_export import GENERATOR, build_scan_identity
from .bruker_jcamp import TemplateSource, resolve_template
from .bruker_pdata import ScanIdentity
from .dicom_export import (
    DICOM_PATIENT_POSITIONS,
    scanning_sequence_code,
    sequence_variant_code,
    software_version,
)
from .image_export import (
    AcquisitionSummary,
    ImageSeries,
    acquisition_summary,
    extract_image_series,
)

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass(frozen=True)
class NiftiExportOptions:
    """Settings of a NIfTI export.

    ``base_name`` prefixes every file name; ``template`` only fills the
    scanner fields of the sidecar (manufacturer, institution, station).
    """

    base_name: str = "sequence"
    template: TemplateSource = None
    anonymize: bool = False
    include_pools: bool = True
    include_phase: bool = True
    compress: bool = True
    write_sidecar: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class NiftiSeriesInfo:
    """One written NIfTI image and its sidecar."""

    name: str
    component: str
    image_path: Path
    sidecar_path: Optional[Path]
    shape: tuple[int, ...]


@dataclass(frozen=True)
class NiftiExportResult:
    """Paths and notes of a finished NIfTI export."""

    directory: Path
    series: tuple[NiftiSeriesInfo, ...]
    warnings: tuple[str, ...] = ()

    @property
    def files(self) -> tuple[Path, ...]:
        paths = []
        for info in self.series:
            paths.append(info.image_path)
            if info.sidecar_path is not None:
                paths.append(info.sidecar_path)
        return tuple(paths)


def require_nibabel():
    """Import :mod:`nibabel` or explain how to install it."""

    try:
        import nibabel
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "NIfTI export needs the optional 'nibabel' package. Install it with "
            "'pip install blochsimulator[export]' or 'pip install nibabel'."
        ) from error
    return nibabel


def nifti_affine(
    series: ImageSeries, warnings: Optional[list[str]] = None
) -> np.ndarray:
    """Return the voxel-to-RAS affine (millimetres) of ``series``."""

    return LPS_TO_RAS @ _lps_affine(series, warnings)


def nifti_volume(series: ImageSeries, component: str = "magnitude") -> np.ndarray:
    """Arrange the frames of ``series`` as ``(columns, rows, slices[, volumes])``."""

    phase = component == "phase"
    if phase and not series.has_phase:
        raise ValueError(f"series {series.name!r} has no phase images")
    slices = _ordered_indices(frame.slice_index for frame in series.frames)
    echoes = _ordered_indices(frame.echo_index for frame in series.frames)
    repetitions = _ordered_indices(frame.repetition_index for frame in series.frames)
    volumes = len(echoes) * len(repetitions)
    data = np.zeros(
        (series.columns, series.rows, len(slices), volumes), dtype=np.float32
    )
    for frame in series.frames:
        values = frame.phase if phase else frame.magnitude
        volume = (
            echoes[frame.echo_index] + len(echoes) * repetitions[frame.repetition_index]
        )
        data[:, :, slices[frame.slice_index], volume] = np.asarray(
            values, dtype=np.float32
        ).T
    if volumes == 1:
        data = data[..., 0]
    return np.nan_to_num(data)


def export_nifti(
    result,
    output_dir,
    *,
    options: Optional[NiftiExportOptions] = None,
    series: Optional[Sequence[ImageSeries]] = None,
    summary: Optional[AcquisitionSummary] = None,
    identity: Optional[ScanIdentity] = None,
    **overrides,
) -> NiftiExportResult:
    """Write every image series of ``result`` as NIfTI plus JSON sidecar.

    Files are named ``<base_name>_<series>.nii.gz`` (phase images get a
    ``_phase`` suffix).  ``series``/``summary``/``identity`` replace the
    values derived from ``result`` and the template.
    """

    nibabel = require_nibabel()
    options = replace(options or NiftiExportOptions(), **overrides)
    if series is None:
        if result is None:
            raise ValueError("export_nifti needs a result or explicit image series")
        series = extract_image_series(
            result,
            include_pools=options.include_pools,
            include_phase=options.include_phase,
        )
    series = list(series)
    if not series:
        raise ValueError(
            "the simulation has no reconstructable images; NIfTI export needs a "
            "Cartesian, spiral, radial or CSI acquisition"
        )
    if summary is None:
        summary = (
            acquisition_summary(result) if result is not None else AcquisitionSummary()
        )
    template = resolve_template(options.template)
    if identity is None and template is not None:
        identity = build_scan_identity(template, anonymize=options.anonymize)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    if summary.field_strength_t is None:
        warnings.append(
            "field strength unknown; MagneticFieldStrength left out of the sidecars"
        )
    base = _safe_name(options.base_name) or "sequence"
    suffix = ".nii.gz" if options.compress else ".nii"
    written = []
    for image_series in series:
        components = ["magnitude"]
        if (
            options.include_phase
            and image_series.has_phase
            and not image_series.is_spectroscopic_map
        ):
            components.append("phase")
        for component in components:
            stem = f"{base}_{_safe_name(image_series.name)}"
            if component == "phase":
                stem += "_phase"
            image_path = root / f"{stem}{suffix}"
            sidecar_path = root / f"{stem}.json" if options.write_sidecar else None
            for path in (image_path, sidecar_path):
                if path is not None and path.exists() and not options.overwrite:
                    raise FileExistsError(f"{path} already exists")
            series_warnings: list[str] = []
            data = nifti_volume(image_series, component)
            affine = nifti_affine(image_series, series_warnings)
            image = nibabel.Nifti1Image(data, affine)
            _configure_header(
                image.header, image_series, summary, affine, data, component
            )
            nibabel.save(image, str(image_path))
            if sidecar_path is not None:
                sidecar = _sidecar(
                    image_series,
                    summary,
                    identity,
                    component,
                    data.shape,
                    series_warnings,
                )
                sidecar_path.write_text(
                    json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
                )
            warnings.extend(f"{image_series.name}: {text}" for text in series_warnings)
            written.append(
                NiftiSeriesInfo(
                    name=image_series.name,
                    component=component,
                    image_path=image_path,
                    sidecar_path=sidecar_path,
                    shape=tuple(int(size) for size in data.shape),
                )
            )
    return NiftiExportResult(
        directory=root, series=tuple(written), warnings=tuple(warnings)
    )


# ---------------------------------------------------------------------------
# Implementation


def _ordered_indices(values) -> dict[int, int]:
    return {
        value: index for index, value in enumerate(sorted(set(int(v) for v in values)))
    }


def _lps_affine(series: ImageSeries, warnings: Optional[list[str]]) -> np.ndarray:
    row_spacing_m, column_spacing_m = series.pixel_spacing_m
    i_axis = np.asarray(series.row_direction, dtype=float) * column_spacing_m
    j_axis = np.asarray(series.column_direction, dtype=float) * row_spacing_m
    positions: dict[int, np.ndarray] = {}
    for frame in series.frames:
        positions.setdefault(
            int(frame.slice_index), np.asarray(frame.position_m, dtype=float)
        )
    ordered = [positions[key] for key in sorted(positions)]
    if len(ordered) > 1:
        steps = np.diff(np.asarray(ordered), axis=0)
        k_axis = (ordered[-1] - ordered[0]) / (len(ordered) - 1)
        if (
            warnings is not None
            and np.max(np.linalg.norm(steps - k_axis, axis=1)) > 1e-6
        ):
            warnings.append(
                "slices are not equally spaced; the NIfTI affine uses the mean spacing"
            )
    else:
        k_axis = np.asarray(series.normal, dtype=float) * series.slice_thickness_m
    affine = np.eye(4)
    affine[:3, 0] = i_axis * 1e3
    affine[:3, 1] = j_axis * 1e3
    affine[:3, 2] = k_axis * 1e3
    affine[:3, 3] = ordered[0] * 1e3
    return affine


def _configure_header(header, series, summary, affine, data, component) -> None:
    header.set_xyzt_units("mm", "sec")
    header.set_qform(affine, code=1)
    header.set_sform(affine, code=1)
    zooms = [float(value) for value in np.linalg.norm(affine[:3, :3], axis=0)]
    if data.ndim == 4:
        repetition_time = summary.repetition_time_s or 0.0
        echoes = len({frame.echo_index for frame in series.frames})
        zooms.append(float(repetition_time) if echoes == 1 else 0.0)
    header.set_zooms(zooms[: data.ndim])
    if series.kind in {"cartesian_2d", "cartesian_3d"}:
        header.set_dim_info(freq=0, phase=1, slice=2)
    header["descrip"] = f"{GENERATOR} {series.name} {component}"[:79].encode(
        "ascii", "replace"
    )
    finite = data[np.isfinite(data)]
    if finite.size:
        header["cal_min"] = float(finite.min())
        header["cal_max"] = float(finite.max())


def _sidecar(series, summary, identity, component, shape, warnings) -> dict:
    echo_times = sorted(
        {
            (frame.echo_index, frame.echo_time_s)
            for frame in series.frames
            if frame.echo_time_s is not None
        }
    )
    sidecar: dict = {
        "Modality": "MR",
        "Manufacturer": (identity.manufacturer if identity is not None else GENERATOR),
        "ManufacturersModelName": GENERATOR,
        "SoftwareVersions": f"{GENERATOR} {software_version()}",
    }
    if identity is not None:
        for key, value in (
            ("InstitutionName", identity.institution),
            ("StationName", identity.station),
            ("DeviceSerialNumber", identity.system_order_number),
        ):
            if value:
                sidecar[key] = value
    optional = {
        "MagneticFieldStrength": summary.field_strength_t,
        "ImagingFrequency": summary.imaging_frequency_mhz,
        "ImagedNucleus": summary.nucleus or None,
        "RepetitionTime": summary.repetition_time_s,
        "EchoTime": echo_times[0][1] if echo_times else summary.echo_time_s,
        "FlipAngle": summary.flip_angle_deg,
        "PixelBandwidth": summary.pixel_bandwidth_hz,
    }
    sidecar.update({key: value for key, value in optional.items() if value is not None})
    if len(echo_times) > 1:
        sidecar["EchoTimes"] = [value for _, value in echo_times]
    sidecar.update(
        {
            "MRAcquisitionType": series.acquisition_type,
            "ScanningSequence": scanning_sequence_code(series, summary),
            "SequenceVariant": sequence_variant_code(summary),
            "SequenceName": summary.sequence_name or series.name,
            "SeriesDescription": series.description or series.name,
            "ImageType": [
                "DERIVED",
                "PRIMARY",
                (
                    "P"
                    if component == "phase"
                    else ("METABOLITE_MAP" if series.is_spectroscopic_map else "M")
                ),
            ],
            "SliceThickness": series.slice_thickness_m * 1e3,
        }
    )
    if series.slice_spacing_m:
        sidecar["SpacingBetweenSlices"] = series.slice_spacing_m * 1e3
    if series.kind in {"cartesian_2d", "cartesian_3d"}:
        sidecar["PhaseEncodingAxis"] = "j"
    if identity is not None:
        sidecar["PatientPosition"] = DICOM_PATIENT_POSITIONS.get(
            identity.patient_position, "HFS"
        )
    sidecar["ConversionSoftware"] = GENERATOR
    sidecar["ConversionSoftwareVersion"] = software_version()

    ordered = sorted(
        series.frames,
        key=lambda frame: (frame.repetition_index, frame.echo_index, frame.slice_index),
    )
    volumes = []
    for frame in ordered:
        key = (frame.echo_index, frame.repetition_index)
        if key not in volumes:
            volumes.append(key)
    acquisition_times = [
        frame.acquisition_time_s
        for frame in ordered
        if frame.slice_index == ordered[0].slice_index
        and frame.acquisition_time_s is not None
    ]
    details: dict = {
        "series": series.name,
        "kind": series.kind,
        "component": component,
        "shape": list(shape),
        "axes": ["column", "row", "slice", "volume"][: len(shape)],
        "volume_order": "echo fastest, then repetition",
        "volume_echo_index": [echo for echo, _ in volumes],
        "volume_repetition_index": [repetition for _, repetition in volumes],
        "coordinate_system": "NIfTI RAS; simulator x, y, z are LPS of a head-first supine subject",
        "units": (
            "radians" if component == "phase" else "arbitrary (simulated magnetization)"
        ),
    }
    if acquisition_times and len(acquisition_times) == len(volumes):
        details["volume_acquisition_time_s"] = acquisition_times
    if series.pool:
        details["pool"] = series.pool
    if series.spectral_window_hz is not None:
        details["spectral_window_hz"] = list(series.spectral_window_hz)
    for key, value in (
        ("spectral_bandwidth_hz", summary.spectral_bandwidth_hz),
        ("spectral_points", summary.spectral_points),
        ("scan_time_s", summary.scan_time_s),
    ):
        if value is not None:
            details[key] = value
    if warnings:
        details["warnings"] = list(warnings)
    sidecar[GENERATOR] = details
    return sidecar


def _safe_name(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9.+-]+", "_", str(text))).strip("_")
