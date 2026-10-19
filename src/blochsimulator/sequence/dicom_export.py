"""DICOM export of simulated sequence images.

Every image frame becomes one classic *MR Image Storage* file.  The images
come from :func:`~blochsimulator.sequence.image_export.extract_image_series`,
so the geometry conventions are shared with the Bruker ``pdata`` export:
simulator gradient coordinates (x, y, z in metres) are the patient LPS
coordinates of a head-first supine subject.

Series layout
-------------
* ``image`` – combined magnitude, one file per slice, echo and repetition
* ``pool_<name>`` – the same for every pool of a multi-pool phantom
* phase images – a separate series with ``ImageType ... \\P``
* CSI – the total signal and one metabolite map per pool or spectral peak

Patient, study, user and scanner attributes come from the same
:class:`~blochsimulator.sequence.bruker_pdata.ScanIdentity` the Bruker
exporter writes, so a ParaVision template dataset (``subject`` / ``acqp``)
fills them.  :mod:`pydicom` is an optional dependency
(``pip install blochsimulator[export]``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .bruker_export import GENERATOR, build_scan_identity
from .bruker_jcamp import TemplateSource, generate_uid, resolve_template
from .bruker_pdata import ScanIdentity
from .image_export import (
    AcquisitionSummary,
    ImageFrame,
    ImageSeries,
    acquisition_summary,
    extract_image_series,
)

MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"
EXPLICIT_VR_LITTLE_ENDIAN = "1.2.840.10008.1.2.1"
MAGNITUDE_STORED_MAX = 32767
PHASE_STORED_MAX = 32767

DICOM_PATIENT_POSITIONS = {
    "Head_Supine": "HFS",
    "Head_Prone": "HFP",
    "Head_Left": "HFDL",
    "Head_Right": "HFDR",
    "Foot_Supine": "FFS",
    "Foot_Prone": "FFP",
    "Foot_Left": "FFDL",
    "Foot_Right": "FFDR",
}
_UID_PATTERN = re.compile(r"^(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*$")


@dataclass(frozen=True)
class DicomExportOptions:
    """Settings of a DICOM export.

    ``template`` is a ParaVision dataset (or loaded metadata) whose subject,
    study and user fields fill the patient and study modules.  Without
    ``new_study`` the template's study instance UID is kept, so the images
    join the measured study in a PACS.
    """

    template: TemplateSource = None
    anonymize: bool = False
    new_study: bool = True
    patient_position: Optional[str] = None
    include_pools: bool = True
    include_phase: bool = True
    series_prefix: Optional[str] = None
    first_series_number: int = 1
    overwrite: bool = False


@dataclass(frozen=True)
class DicomSeriesInfo:
    """One written DICOM series."""

    number: int
    description: str
    directory: Path
    series_uid: str
    image_type: str
    files: tuple[Path, ...]


@dataclass(frozen=True)
class DicomExportResult:
    """Paths and notes of a finished DICOM export."""

    directory: Path
    study_uid: str
    series: tuple[DicomSeriesInfo, ...]
    warnings: tuple[str, ...] = ()

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(path for series in self.series for path in series.files)


def require_pydicom():
    """Import :mod:`pydicom` or explain how to install it."""

    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "DICOM export needs the optional 'pydicom' package. Install it with "
            "'pip install blochsimulator[export]' or 'pip install pydicom'."
        ) from error
    return pydicom


def export_dicom(
    result,
    output_dir,
    *,
    options: Optional[DicomExportOptions] = None,
    series: Optional[Sequence[ImageSeries]] = None,
    summary: Optional[AcquisitionSummary] = None,
    identity: Optional[ScanIdentity] = None,
    created: Optional[datetime] = None,
    **overrides,
) -> DicomExportResult:
    """Write the reconstructed images of ``result`` as DICOM series.

    ``series`` and ``summary`` replace the images and acquisition parameters
    derived from ``result`` (``result`` may then be ``None``); ``identity``
    replaces the identity derived from the template.  Keyword ``overrides``
    update ``options``.  Each series is written to
    ``<output_dir>/S<number>_<name>/IM<instance>.dcm``.
    """

    pydicom = require_pydicom()
    options = replace(options or DicomExportOptions(), **overrides)
    if series is None:
        if result is None:
            raise ValueError("export_dicom needs a result or explicit image series")
        series = extract_image_series(
            result,
            include_pools=options.include_pools,
            include_phase=options.include_phase,
        )
    series = list(series)
    if not series:
        raise ValueError(
            "the simulation has no reconstructable images; DICOM export needs a "
            "Cartesian, spiral, radial or CSI acquisition"
        )
    if summary is None:
        summary = (
            acquisition_summary(result) if result is not None else AcquisitionSummary()
        )

    created = created or datetime.now().astimezone()
    template = resolve_template(options.template)
    if identity is None:
        identity = build_scan_identity(
            template,
            patient_position=options.patient_position,
            anonymize=options.anonymize,
            created=created,
            new_study=options.new_study,
        )
    manufacturer = identity.manufacturer if template is not None else GENERATOR

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    study_uid = _valid_uid(identity.study_uid)
    if study_uid != identity.study_uid:
        warnings.append(
            "the template study UID is not a valid DICOM UID; a new one was generated"
        )
    frame_of_reference_uid = generate_uid()
    context = _Context(
        pydicom=pydicom,
        identity=identity,
        summary=summary,
        created=created,
        study_uid=study_uid,
        frame_of_reference_uid=frame_of_reference_uid,
        manufacturer=manufacturer,
        anonymize=bool(options.anonymize),
        prefix=(options.series_prefix or summary.sequence_name or GENERATOR).strip(),
    )
    if summary.field_strength_t is None:
        warnings.append(
            "field strength unknown; MagneticFieldStrength and ImagingFrequency left empty"
        )

    written: list[DicomSeriesInfo] = []
    number = int(options.first_series_number)
    for image_series in series:
        components = ["magnitude"]
        if (
            options.include_phase
            and image_series.has_phase
            and not image_series.is_spectroscopic_map
        ):
            components.append("phase")
        for component in components:
            info = _write_series(
                context,
                image_series,
                component,
                number,
                root,
                overwrite=options.overwrite,
            )
            written.append(info)
            number += 1
    return DicomExportResult(
        directory=root,
        study_uid=study_uid,
        series=tuple(written),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Implementation


@dataclass(frozen=True)
class _Context:
    pydicom: object
    identity: ScanIdentity
    summary: AcquisitionSummary
    created: datetime
    study_uid: str
    frame_of_reference_uid: str
    manufacturer: str
    anonymize: bool
    prefix: str


def _write_series(
    context: _Context,
    series: ImageSeries,
    component: str,
    number: int,
    root: Path,
    *,
    overwrite: bool,
) -> DicomSeriesInfo:
    phase = component == "phase"
    suffix = "_phase" if phase else ""
    directory = root / f"S{number:03d}_{_safe_name(series.name + suffix)}"
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f"{directory} already contains files")
    directory.mkdir(parents=True, exist_ok=True)
    # Images of an earlier, longer export would otherwise stay in the series.
    for stale in directory.glob("IM*.dcm"):
        stale.unlink()

    if phase:
        stack = series.phase_stack()
        slope = float(np.pi / PHASE_STORED_MAX)
        stored = np.clip(
            np.rint(np.nan_to_num(stack) / slope),
            -PHASE_STORED_MAX - 1,
            PHASE_STORED_MAX,
        )
        stored = stored.astype("<i2")
        window = (0.0, float(2.0 * np.pi))
        image_type = ["DERIVED", "PRIMARY", "P"]
        complex_component = "PHASE"
    else:
        stack = np.nan_to_num(series.magnitude_stack())
        peak = float(np.max(np.abs(stack))) if stack.size else 0.0
        slope = peak / MAGNITUDE_STORED_MAX if peak > 0 and np.isfinite(peak) else 1.0
        stored = np.clip(
            np.rint(np.abs(stack) / slope), 0, MAGNITUDE_STORED_MAX
        ).astype("<u2")
        window = (0.5 * peak, max(peak, 1e-12))
        if series.is_spectroscopic_map:
            image_type = ["DERIVED", "PRIMARY", "METABOLITE_MAP"]
        else:
            image_type = ["DERIVED", "PRIMARY", "M"]
        complex_component = "MAGNITUDE"

    description = _series_description(context, series, phase)
    series_uid = generate_uid()
    files = []
    for instance, (frame, pixels) in enumerate(zip(series.frames, stored), start=1):
        dataset = _frame_dataset(
            context,
            series,
            frame,
            pixels,
            number=number,
            instance=instance,
            series_uid=series_uid,
            description=description,
            image_type=image_type,
            complex_component=complex_component,
            slope=slope,
            window=window,
            signed=phase,
        )
        path = directory / f"IM{instance:05d}.dcm"
        _save(context.pydicom, dataset, path)
        files.append(path)
    return DicomSeriesInfo(
        number=number,
        description=description,
        directory=directory,
        series_uid=series_uid,
        image_type="\\".join(image_type),
        files=tuple(files),
    )


def _series_description(context: _Context, series: ImageSeries, phase: bool) -> str:
    text = series.description or series.name
    if phase:
        text = f"{text} phase"
    return _clip(f"{context.prefix} {text}" if context.prefix else text, 64)


def _frame_dataset(
    context: _Context,
    series: ImageSeries,
    frame: ImageFrame,
    pixels: np.ndarray,
    *,
    number: int,
    instance: int,
    series_uid: str,
    description: str,
    image_type: list[str],
    complex_component: str,
    slope: float,
    window: tuple[float, float],
    signed: bool,
):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import PYDICOM_IMPLEMENTATION_UID

    identity = context.identity
    summary = context.summary
    sop_uid = generate_uid()

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MR_IMAGE_STORAGE
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = EXPLICIT_VR_LITTLE_ENDIAN
    meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    meta.ImplementationVersionName = _clip(f"BLOCHSIM_{software_version()}", 16)

    ds = Dataset()
    ds.file_meta = meta
    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.ImageType = image_type
    ds.SOPClassUID = MR_IMAGE_STORAGE
    ds.SOPInstanceUID = sop_uid
    created = context.created
    ds.InstanceCreationDate = _date(created)
    ds.InstanceCreationTime = _time(created)

    # Patient
    ds.PatientName = _clip(identity.subject_name.replace("\\", " "), 64)
    ds.PatientID = _clip(identity.subject_id, 64)
    ds.PatientBirthDate = _parse_date(identity.subject_birth_date)
    ds.PatientSex = {"MALE": "M", "FEMALE": "F"}.get(identity.visu_sex, "")
    if identity.subject_weight_kg and identity.subject_weight_kg > 0.0011:
        ds.PatientWeight = _ds(identity.subject_weight_kg)
    if context.anonymize:
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = _clip(f"{GENERATOR} template anonymization", 64)

    # Study
    study_date = identity.study_date or created
    ds.StudyInstanceUID = context.study_uid
    ds.StudyDate = _date(study_date)
    ds.StudyTime = _time(study_date)
    ds.StudyID = _clip(str(identity.study_number), 16)
    ds.StudyDescription = _clip(identity.study_name, 64)
    ds.AccessionNumber = ""
    ds.ReferringPhysicianName = _clip(
        identity.referring_physician.replace("\\", " "), 64
    )

    # Series and equipment
    ds.Modality = "MR"
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = number
    ds.SeriesDescription = description
    ds.ProtocolName = _clip(summary.sequence_name or series.name, 64)
    ds.SeriesDate = _date(created)
    ds.SeriesTime = _time(created)
    ds.PatientPosition = DICOM_PATIENT_POSITIONS.get(identity.patient_position, "HFS")
    ds.OperatorsName = _clip(identity.operator.replace("\\", " "), 64)
    ds.Manufacturer = _clip(context.manufacturer, 64)
    ds.ManufacturerModelName = GENERATOR
    ds.SoftwareVersions = _clip(f"{GENERATOR} {software_version()}", 64)
    ds.InstitutionName = _clip(identity.institution, 64)
    ds.StationName = _clip(identity.station, 16)
    if identity.system_order_number:
        ds.DeviceSerialNumber = _clip(identity.system_order_number, 64)
    ds.FrameOfReferenceUID = context.frame_of_reference_uid
    ds.PositionReferenceIndicator = ""

    # MR acquisition
    ds.ScanningSequence = scanning_sequence_code(series, summary)
    ds.SequenceVariant = sequence_variant_code(summary)
    ds.ScanOptions = ""
    ds.MRAcquisitionType = (
        series.acquisition_type if series.acquisition_type in {"2D", "3D"} else "2D"
    )
    if summary.sequence_name:
        ds.SequenceName = _clip(summary.sequence_name, 16)
    _set_ds(ds, "RepetitionTime", _ms(summary.repetition_time_s))
    echo_time = (
        frame.echo_time_s if frame.echo_time_s is not None else summary.echo_time_s
    )
    _set_ds(ds, "EchoTime", _ms(echo_time))
    ds.EchoNumbers = int(frame.echo_index) + 1
    if series.echo_count > 1:
        ds.EchoTrainLength = int(series.echo_count)
    ds.NumberOfAverages = _ds(1.0)
    _set_ds(ds, "ImagingFrequency", summary.imaging_frequency_mhz)
    if summary.nucleus:
        ds.ImagedNucleus = _clip(summary.nucleus, 16)
    _set_ds(ds, "MagneticFieldStrength", summary.field_strength_t)
    _set_ds(ds, "FlipAngle", summary.flip_angle_deg)
    _set_ds(ds, "PixelBandwidth", summary.pixel_bandwidth_hz)
    if summary.phase_encoding_steps:
        ds.NumberOfPhaseEncodingSteps = int(summary.phase_encoding_steps)
    if series.kind in {"cartesian_2d", "cartesian_3d"}:
        readout = int(summary.readout_samples or series.columns)
        phase_steps = int(summary.phase_encoding_steps or series.rows)
        if 0 < readout < 65536 and 0 < phase_steps < 65536:
            ds.AcquisitionMatrix = [0, readout, phase_steps, 0]
        ds.InPlanePhaseEncodingDirection = "COL"
    ds.ReceiveCoilName = _clip(identity.coil_receive_name, 16)
    ds.TransmitCoilName = _clip(identity.coil_transmit_name, 16)
    ds.ComplexImageComponent = complex_component
    ds.AcquisitionNumber = int(frame.repetition_index) + 1
    ds.TemporalPositionIdentifier = int(frame.repetition_index) + 1
    ds.NumberOfTemporalPositions = int(series.repetition_count)
    acquired = created
    if frame.acquisition_time_s is not None and np.isfinite(frame.acquisition_time_s):
        acquired = created + timedelta(seconds=float(frame.acquisition_time_s))
    ds.AcquisitionDate = _date(acquired)
    ds.AcquisitionTime = _time(acquired)
    ds.ContentDate = _date(created)
    ds.ContentTime = _time(created)
    ds.InstanceNumber = instance
    ds.ImageComments = _clip(_image_comment(series), 10240)

    # Geometry: positions are LPS millimetres
    position_mm = [float(value) * 1e3 for value in frame.position_m]
    ds.ImagePositionPatient = [_ds(value) for value in position_mm]
    ds.ImageOrientationPatient = [
        _ds(value) for value in (*series.row_direction, *series.column_direction)
    ]
    ds.PixelSpacing = [
        _ds(series.pixel_spacing_m[0] * 1e3),
        _ds(series.pixel_spacing_m[1] * 1e3),
    ]
    ds.SliceThickness = _ds(series.slice_thickness_m * 1e3)
    spacing = series.slice_spacing_m
    if series.slice_count > 1 and spacing:
        ds.SpacingBetweenSlices = _ds(spacing * 1e3)
    ds.SliceLocation = _ds(float(np.dot(series.normal, frame.position_m)) * 1e3)

    # Pixels
    rows, columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = int(rows)
    ds.Columns = int(columns)
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1 if signed else 0
    ds.RescaleIntercept = _ds(0.0)
    ds.RescaleSlope = _ds(slope)
    ds.RescaleType = "US"
    ds.WindowCenter = _ds(window[0])
    ds.WindowWidth = _ds(window[1])
    ds.PixelData = np.ascontiguousarray(pixels).tobytes()
    return ds


def _image_comment(series: ImageSeries) -> str:
    parts = [f"Simulated with {GENERATOR}", f"series={series.name}"]
    if series.pool:
        parts.append(f"pool={series.pool}")
    if series.spectral_window_hz is not None:
        low, high = series.spectral_window_hz
        parts.append(f"spectral window {low:g} to {high:g} Hz")
    return "; ".join(parts)


def scanning_sequence_code(series: ImageSeries, summary: AcquisitionSummary) -> str:
    name = (summary.sequence_name or "").lower()
    if series.is_spectroscopic_map or series.kind == "csi":
        return "RM"
    if "epi" in name and "epsi" not in name:
        return "EP"
    return "GR"


def sequence_variant_code(summary: AcquisitionSummary) -> str:
    name = (summary.sequence_name or "").lower()
    if any(token in name for token in ("flash", "mge", "gre", "spoil")):
        return "SP"
    return "NONE"


def _save(pydicom, dataset, path: Path) -> None:
    major = int(str(pydicom.__version__).split(".")[0])
    if major >= 3:
        dataset.save_as(path, enforce_file_format=True)
    else:  # pragma: no cover - pydicom 2.x
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False
        dataset.save_as(str(path), write_like_original=False)


def _ds(value: float):
    from pydicom.valuerep import DSfloat

    return DSfloat(float(f"{float(value):.10g}"), auto_format=True)


def _set_ds(dataset, keyword: str, value: Optional[float]) -> None:
    if value is None:
        return
    value = float(value)
    if np.isfinite(value):
        setattr(dataset, keyword, _ds(value))


def _ms(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value) * 1e3


def _clip(text, length: int) -> str:
    return str(text or "").strip()[:length]


def _date(moment: datetime) -> str:
    return moment.strftime("%Y%m%d")


def _time(moment: datetime) -> str:
    return moment.strftime("%H%M%S.%f")


def _parse_date(text: str) -> str:
    value = str(text or "").strip().strip("<>")
    if not value:
        return ""
    for pattern in ("%d %b %Y", "%Y-%m-%d", "%Y%m%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(value, pattern).strftime("%Y%m%d")
        except ValueError:
            continue
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", value)
    return "".join(match.groups()) if match else ""


def _valid_uid(uid: str) -> str:
    text = str(uid or "").strip()
    if text and len(text) <= 64 and _UID_PATTERN.match(text):
        return text
    return generate_uid()


def _safe_name(text: str) -> str:
    return (
        re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9.+-]+", "_", str(text))).strip("_")
        or "series"
    )


def software_version() -> str:
    try:
        from .. import __version__
    except ImportError:  # pragma: no cover
        return "unknown"
    return str(__version__)
