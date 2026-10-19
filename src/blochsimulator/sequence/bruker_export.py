"""Bruker ParaVision dataset export for simulated sequence results.

:func:`export_bruker_dataset` writes a ParaVision study folder (``subject``,
``AdjStatePerStudy`` and one numbered scan folder) that ParaVision can import
and that analysis tools such as ``brukerapi`` can read.  A scan folder holds
the raw data (``fid`` and/or ``rawdata.job0``), the acquisition parameters
(``acqp``, ``method``) and reconstructions in ``pdata/<procno>``.

The raw data follow the Bruker block layout of the selected method:
``GO_32BIT_SGN_INT`` little-endian real/imaginary pairs, one block per
readout with all receivers of that readout one after another.  ``fid``
blocks are padded to whole KBytes (``Standard_KBlock_Format``),
``rawdata.job0`` blocks are not.

Scanner, operator and subject metadata come from a measured ParaVision
dataset (``template``) when one is given; see
:func:`~.bruker_jcamp.load_bruker_template`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .acquisition import CartesianAcquisition, CartesianAcquisitionFrames
from .bruker_jcamp import (
    BrukerTemplateMetadata,
    JcampEnum,
    TemplateSource,
    anonymize_template,
    generate_uid,
    load_bruker_template,
    normalize_nucleus,
    resolve_template,
)
from .bruker_pdata import (
    BrukerFileContext,
    PdataFrames,
    ProtocolInfo,
    ScanIdentity,
    frames_from_image_series,
    frames_from_spectra,
    scale_to_int16,
    write_abs_time,
    write_date,
    write_label,
    write_pdata,
    write_visu_pars,
)
from .image_export import (
    definition_float,
    definition_values,
    extract_image_series,
    gyromagnetic_ratio_hz_per_t,
)
from .model import SequenceProgram
from .paravision import (
    DEFAULT_PARAVISION_VERSION,
    ParaVisionProfile,
    get_paravision_profile,
)
from .result import SequenceSimulationResult

INT32_TARGET = float(2**30)
GENERATOR = "BlochSimulator"
DEFAULT_OWNER = "nmrsu"
DEFAULT_FIELD_STRENGTH_T = 7.0
FID_ONLY_JOB = (0, 1, 0, 1, 64, 100, 0, 0)


@dataclass(frozen=True)
class BrukerExportOptions:
    """Settings of a Bruker ParaVision export.

    ``None`` values are derived from the simulation, the template dataset or
    the ParaVision version profile.  ``method_name=None`` selects the method
    from the acquisition (see :func:`suggest_bruker_method`).
    ``conjugate_raw_data`` stores the complex conjugate of the simulated
    signal and marks it with ``RECO_qopts=COMPLEX_CONJUGATE`` as measured
    ParaVision data do.
    """

    method_name: Optional[str] = None
    scan_name: Optional[str] = None
    matrix: Optional[tuple[int, int]] = None
    fov_m: Optional[tuple[float, float]] = None
    slice_thickness_mm: Optional[float] = None
    read_orientation: Optional[str] = None
    slice_orientation: Optional[str] = None
    patient_position: Optional[str] = None
    raw_data_files: Optional[str] = None
    paravision_version: str = DEFAULT_PARAVISION_VERSION
    template: TemplateSource = None
    anonymize: bool = False
    study_layout: str = "new"
    expno: Optional[int] = None
    include_pools: bool = True
    conjugate_raw_data: bool = False
    use_template_frequency: bool = True


@dataclass(frozen=True)
class BrukerExportResult:
    """Paths and notes of a finished Bruker export."""

    study_dir: Path
    scan_dir: Path
    expno: int
    paravision_version: str
    method: str
    files: tuple[Path, ...]
    warnings: tuple[str, ...] = ()


def export_bruker_dataset(
    result: SequenceSimulationResult,
    output_root,
    *,
    program: Optional[SequenceProgram] = None,
    phantom=None,
    acquisition: Optional[CartesianAcquisition] = None,
    acquisition_frames: Optional[CartesianAcquisitionFrames] = None,
    options: Optional[BrukerExportOptions] = None,
    scale: Optional[float] = None,
    **overrides,
) -> BrukerExportResult:
    """Export a simulation as a ParaVision study.

    With ``study_layout="new"`` (default) a new study folder is created in
    ``output_root``.  With ``"append"`` ``output_root`` is an existing study
    and the scan is added as the next free experiment number; its subject
    and scanner metadata serve as template unless another one is given.
    """

    options = _merge_options(options, **overrides)
    created = datetime.now().astimezone()
    root = Path(output_root)
    warnings: list[str] = []
    template = _resolve_export_template(options, root, warnings)
    if options.study_layout == "append":
        if not root.is_dir():
            raise ValueError(f"Bruker study folder does not exist: {root}")
        study_dir = root
        expno = _next_expno(study_dir, options.expno)
    else:
        if root.exists() and not root.is_dir():
            raise ValueError(f"Bruker export target is not a directory: {root}")
        study_dir = None
        expno = int(options.expno or 1)
    identity = _scan_identity(template, options, created, new_study=study_dir is None)
    profile = get_paravision_profile(options.paravision_version)
    if study_dir is None:
        study_dir = root / _study_folder_name(identity, created, profile)
        if study_dir.exists():
            raise ValueError(f"Bruker study folder already exists: {study_dir}")
    study_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    context = BrukerFileContext(
        profile=profile,
        owner=identity.owner,
        user_host=identity.user_host,
        data_root=profile.data_root_for(identity.owner),
        study_folder=study_dir.name,
        expno=expno,
        created=created,
    )
    if options.study_layout != "append":
        files.extend(_write_study_files(study_dir, context, identity))
    scan = _Scan(
        result,
        program=program,
        phantom=phantom,
        acquisition=acquisition,
        acquisition_frames=acquisition_frames,
        options=options,
        template=template,
        scale=scale,
        warnings=warnings,
    )
    scan_dir = study_dir / str(expno)
    files.extend(_write_scan(scan_dir, scan, context, identity))
    return BrukerExportResult(
        study_dir=study_dir,
        scan_dir=scan_dir,
        expno=expno,
        paravision_version=profile.key,
        method=scan.method,
        files=tuple(files),
        warnings=tuple(warnings),
    )


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
    """Export a simulation as a single ParaVision scan folder.

    ``directory`` becomes the scan folder (``<study>/<expno>``); no study
    files are written.  The raw data are little-endian ``GO_32BIT_SGN_INT``
    pairs; the scale between simulated signal and stored integers is kept in
    ``blochsim_provenance.json``.
    """

    output_dir = Path(directory)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Bruker export target is not a directory: {output_dir}")
    options = _merge_options(
        options,
        method_name=method_name,
        scan_name=scan_name,
        matrix=matrix,
        fov_m=fov_m,
        slice_thickness_mm=slice_thickness_mm,
        raw_data_files=raw_data_files,
    )
    created = datetime.now().astimezone()
    warnings: list[str] = []
    template = resolve_template(options.template)
    if template is not None and options.anonymize:
        template = anonymize_template(template)
    identity = _scan_identity(template, options, created, new_study=False)
    profile = get_paravision_profile(options.paravision_version)
    expno = (
        int(output_dir.name) if output_dir.name.isdigit() else int(options.expno or 1)
    )
    context = BrukerFileContext(
        profile=profile,
        owner=identity.owner,
        user_host=identity.user_host,
        data_root=profile.data_root_for(identity.owner),
        study_folder=output_dir.parent.name or "study",
        expno=expno,
        created=created,
    )
    scan = _Scan(
        result,
        program=program,
        phantom=phantom,
        acquisition=acquisition,
        acquisition_frames=acquisition_frames,
        options=options,
        template=template,
        scale=scale,
        warnings=warnings,
    )
    _write_scan(output_dir, scan, context, identity)
    return output_dir


def suggest_bruker_method(
    result: SequenceSimulationResult,
    *,
    acquisition: Optional[CartesianAcquisition] = None,
    acquisition_frames: Optional[CartesianAcquisitionFrames] = None,
) -> str:
    """Return the ParaVision method that matches the simulated acquisition.

    CSI → ``Bruker:CSI``; Cartesian with reversed lines → ``Bruker:EPI``;
    Cartesian with several echoes → ``Bruker:MGE``; other Cartesian 2D/3D →
    ``Bruker:FLASH``; spiral → ``Bruker:SPIRAL``; everything else is
    written as spectroscopy (``Bruker:NSPECT``).
    """

    return _AcquisitionSources.collect(
        result, acquisition, acquisition_frames
    ).suggested_method()


# ---------------------------------------------------------------------------
# Options


_RAW_DATA_ALIASES = {
    "fid": "fid",
    "rawdata": "rawdata.job0",
    "rawdata.job0": "rawdata.job0",
    "job0": "rawdata.job0",
    "both": "both",
    "fid+rawdata.job0": "both",
    "fid + rawdata.job0": "both",
}


def _merge_options(
    options: Optional[BrukerExportOptions], **overrides
) -> BrukerExportOptions:
    base = options or BrukerExportOptions()
    unknown = set(overrides) - set(BrukerExportOptions.__dataclass_fields__)
    if unknown:
        raise TypeError(
            f"unknown Bruker export option(s): {', '.join(sorted(unknown))}"
        )
    values = {key: value for key, value in overrides.items() if value is not None}
    return _validate_options(replace(base, **values))


def _validate_options(options: BrukerExportOptions) -> BrukerExportOptions:
    profile = get_paravision_profile(options.paravision_version)
    method = options.method_name
    if method is not None and str(method).strip().lower() in ("", "auto", "automatic"):
        method = None
    if method is not None:
        method = _normalize_angle_text(method, "method_name")
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

    thickness = options.slice_thickness_mm
    if thickness is not None:
        thickness = float(thickness)
        if not np.isfinite(thickness) or thickness <= 0:
            raise ValueError("slice_thickness_mm must be positive and finite")

    raw = options.raw_data_files
    raw = (
        profile.default_raw_data_files
        if raw is None
        else _normalize_raw_data_files(raw)
    )
    if raw not in profile.raw_data_files:
        allowed = ", ".join(repr(item) for item in profile.raw_data_files)
        raise ValueError(f"{profile.label} raw data must be one of {allowed}")
    layout = str(options.study_layout).strip().lower()
    if layout not in ("new", "append"):
        raise ValueError("study_layout must be 'new' or 'append'")
    expno = options.expno
    if expno is not None:
        expno = int(expno)
        if expno <= 0:
            raise ValueError("expno must be a positive integer")

    def optional_text(value, name):
        return None if value is None else _normalize_plain_text(value, name)

    return replace(
        options,
        method_name=method,
        scan_name=scan_name,
        matrix=matrix,
        fov_m=fov,
        slice_thickness_mm=thickness,
        read_orientation=optional_text(options.read_orientation, "read_orientation"),
        slice_orientation=optional_text(options.slice_orientation, "slice_orientation"),
        patient_position=optional_text(options.patient_position, "patient_position"),
        raw_data_files=raw,
        paravision_version=profile.key,
        study_layout=layout,
        expno=expno,
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
    if text not in _RAW_DATA_ALIASES:
        raise ValueError("raw_data_files must be 'fid', 'rawdata.job0', or 'both'")
    return _RAW_DATA_ALIASES[text]


def _resolve_export_template(
    options: BrukerExportOptions, root: Path, warnings: list[str]
) -> Optional[BrukerTemplateMetadata]:
    template = resolve_template(options.template)
    if template is None and options.study_layout == "append" and root.is_dir():
        try:
            template = load_bruker_template(root)
        except (OSError, ValueError) as error:
            warnings.append(f"study metadata could not be read: {error}")
    if template is not None and options.anonymize:
        template = anonymize_template(template)
    return template


# ---------------------------------------------------------------------------
# Identity and study layout


def _parse_paravision_date(text: str) -> Optional[datetime]:
    value = str(text or "").strip().strip("<>")
    if not value:
        return None
    for pattern in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.replace(",", "."), pattern)
        except ValueError:
            continue
    return None


def build_scan_identity(
    template: TemplateSource = None,
    *,
    patient_position: Optional[str] = None,
    anonymize: bool = False,
    created: Optional[datetime] = None,
    new_study: bool = True,
) -> ScanIdentity:
    """Subject, study and scanner identity of an export.

    The Bruker exporter writes the same identity, so DICOM and NIfTI files of
    one export share user, subject and (for ``new_study``) study metadata.
    ``patient_position`` uses the ``Head_Supine`` style of the Bruker options.
    """

    resolved = resolve_template(template)
    if resolved is not None and anonymize:
        resolved = anonymize_template(resolved)
    options = BrukerExportOptions(patient_position=patient_position or None)
    return _scan_identity(
        resolved, options, created or datetime.now().astimezone(), new_study=new_study
    )


def _scan_identity(
    template: Optional[BrukerTemplateMetadata],
    options: BrukerExportOptions,
    created: datetime,
    *,
    new_study: bool,
) -> ScanIdentity:
    t = template or BrukerTemplateMetadata()
    owner = t.owner or DEFAULT_OWNER
    study_date = None if new_study else _parse_paravision_date(t.study_date)
    entry = t.subject_entry or "SUBJ_ENTRY_HeadFirst"
    position = t.subject_position or "SUBJ_POS_Supine"
    if options.patient_position:
        head, _, pose = options.patient_position.partition("_")
        entry = (
            "SUBJ_ENTRY_HeadFirst"
            if head.lower().startswith("head")
            else "SUBJ_ENTRY_FeetFirst"
        )
        position = f"SUBJ_POS_{pose or 'Supine'}"
    study_number = int(t.study_number or 1)
    if template is not None and new_study:
        study_number += 1
    return ScanIdentity(
        owner=owner,
        user_host=t.user_host or f"{owner}@{GENERATOR}",
        operator=t.operator or owner,
        institution=t.institution,
        station=t.station or GENERATOR,
        system_order_number=t.system_order_number,
        manufacturer=t.manufacturer or "Bruker BioSpin MRI GmbH",
        subject_id=t.subject_id or GENERATOR,
        subject_name=t.subject_name or t.subject_id or GENERATOR,
        subject_uid=t.subject_uid or generate_uid(),
        subject_type=t.subject_type or "Other",
        subject_sex=t.subject_sex or "unknown",
        subject_birth_date=t.subject_birth_date,
        subject_weight_kg=float(t.subject_weight_kg) if t.subject_weight_kg else 0.001,
        subject_created=created,
        subject_entry=entry,
        subject_position=position,
        study_name=t.study_name or "Simulation",
        study_uid=generate_uid() if new_study or not t.study_uid else t.study_uid,
        study_number=study_number,
        study_date=study_date or created,
        referring_physician=t.referring_physician,
        coil_receive_name=t.coil_receive_name or GENERATOR,
        coil_transmit_name=t.coil_transmit_name or t.coil_receive_name or GENERATOR,
        hardware_parameters=t.hardware_parameters,
    )


def _sanitize_name(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", str(text))).strip("_")


def _study_folder_name(identity: ScanIdentity, created: datetime, profile) -> str:
    prefix = created.strftime("%Y%m%d_%H%M%S")
    suffix = f"1_{identity.study_number}"
    body = _sanitize_name(f"{identity.subject_id}_{identity.study_name}") or GENERATOR
    # ParaVision 7 keeps folder names within 63 characters, cutting the body.
    room = max(1, 63 - len(prefix) - len(suffix) - 2)
    return f"{prefix}_{body[:room].strip('_')}_{suffix}"


def _next_expno(study_dir: Path, requested: Optional[int]) -> int:
    used = {
        int(path.name)
        for path in study_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    }
    if requested is not None:
        if requested in used:
            raise ValueError(f"experiment {requested} already exists in {study_dir}")
        return int(requested)
    return max(used, default=0) + 1


def _write_study_files(
    study_dir: Path, context: BrukerFileContext, identity: ScanIdentity
) -> list[Path]:
    profile = context.profile
    subject = context.writer("subject", "parx", footer=False)
    classic = profile.subject_style == "pv5"
    subject.scalar("SUBJECT_version_nr", profile.subject_version_nr)
    subject.string("SUBJECT_id", identity.subject_id, 60)
    subject.string(
        "SUBJECT_patient_instance_uid" if classic else "SUBJECT_instance_uid",
        identity.subject_uid,
        65,
    )
    subject.struct(
        "SUBJECT_instance_creation_date", _abs_time(identity.subject_created)
    )
    subject.string("SUBJECT_name_string", identity.subject_name, 64)
    # ParaVision 5.1 stores the name in the second field of the struct.
    name = ("", identity.subject_name) if classic else (identity.subject_name, "")
    subject.struct("SUBJECT_name", name)
    if identity.subject_birth_date:
        subject.string("SUBJECT_dbirth", identity.subject_birth_date, 12)
    subject.scalar("SUBJECT_type", identity.subject_type or "Other")
    subject.string("SUBJECT_sex", identity.subject_sex, 8)
    subject.scalar("SUBJECT_sex_animal", "UNDEFINED")
    subject.string("SUBJECT_remarks", "", 2048)
    subject.string("SUBJECT_study_name", identity.study_name, 64)
    subject.scalar("SUBJECT_study_nr", identity.study_number)
    subject.string("SUBJECT_study_instance_uid", identity.study_uid, 65)
    subject.scalar("SUBJECT_weight", identity.subject_weight_kg)
    subject.string("SUBJECT_referral", identity.referring_physician, 64)
    subject.string("SUBJECT_purpose", "", 2048)
    subject.scalar("SUBJECT_pos_valid", True)
    subject.scalar("SUBJECT_entry", identity.subject_entry)
    subject.scalar("SUBJECT_position", identity.subject_position)
    write_date(subject, profile, "SUBJECT_date", identity.study_date)
    write_abs_time(subject, profile, "SUBJECT_abs_date", identity.study_date)
    subject.string("SUBJECT_comment", "", 2048)
    for name, value in identity.hardware_parameters:
        subject.raw(name, value)
    files = [subject.write(study_dir / "subject")]
    adjust = context.writer("AdjStatePerStudy", "pipe")
    adjust.scalar("AdjStudyStateCompleted", False)
    files.append(adjust.write(study_dir / "AdjStatePerStudy"))
    return files


def _abs_time(moment: datetime) -> tuple[int, int, int]:
    offset = moment.utcoffset()
    minutes = int(offset.total_seconds() // 60) if offset is not None else 0
    return (int(moment.timestamp()), moment.microsecond // 1000, minutes)


# ---------------------------------------------------------------------------
# Acquisition layouts


def _try(getter):
    try:
        return getter()
    except Exception:
        return None


@dataclass(frozen=True)
class _AcquisitionSources:
    acquisition: Optional[CartesianAcquisition]
    frames: Optional[CartesianAcquisitionFrames]
    volumes: object
    spiral: object
    spectroscopy: object

    @classmethod
    def collect(cls, result, acquisition=None, acquisition_frames=None):
        if acquisition is None:
            acquisition = _try(lambda: result.cartesian_acquisition)
        if acquisition_frames is None:
            acquisition_frames = _try(lambda: result.cartesian_acquisition_frames)
        return cls(
            acquisition=acquisition,
            frames=acquisition_frames,
            volumes=_try(lambda: result.cartesian_acquisition_volumes),
            spiral=_try(lambda: result.spiral_acquisition),
            spectroscopy=_try(lambda: result.spectroscopic_acquisition),
        )

    def cartesian_layouts(self) -> tuple:
        if self.frames is not None:
            return tuple(self.frames.acquisitions)
        if self.acquisition is not None:
            return (self.acquisition,)
        return ()

    def echo_count(self) -> int:
        if self.volumes is not None:
            return len({index[1] for index in self.volumes.volume_indices})
        if self.frames is not None:
            return len({index[1] for index in self.frames.frame_indices})
        return 1

    def suggested_method(self) -> str:
        if self.spectroscopy is not None:
            return "Bruker:CSI"
        if self.volumes is not None:
            return "Bruker:MGE" if self.echo_count() > 1 else "Bruker:FLASH"
        layouts = self.cartesian_layouts()
        if layouts:
            if any(
                direction < 0
                for item in layouts
                for direction in item.readout_directions
            ):
                return "Bruker:EPI"
            return "Bruker:MGE" if self.echo_count() > 1 else "Bruker:FLASH"
        if self.spiral is not None:
            return "Bruker:SPIRAL"
        return "Bruker:NSPECT"


@dataclass
class _RawLayout:
    """Raw data in Bruker file order plus their acquisition dimensions.

    ``blocks`` has shape ``(blocks, coils, samples)``: one block per stored
    readout with all receivers one after another.
    """

    kind: str
    blocks: np.ndarray
    acq_size: tuple[int, ...]
    dim_desc: tuple[str, ...]
    matrix: tuple[int, ...]
    ni: int = 1
    nr: int = 1
    slices: int = 1
    echoes: int = 1
    enc_steps: tuple[tuple[int, ...], ...] = ()
    kblock: bool = True
    job0: Optional[np.ndarray] = None

    @property
    def readout(self) -> int:
        return int(self.blocks.shape[-1])

    @property
    def block_count(self) -> int:
        return int(self.blocks.shape[0])


def _linear_steps(count: int) -> tuple[int, ...]:
    return tuple(int(value - count // 2) for value in range(int(count)))


def _frame_grid(frame_indices) -> Optional[tuple[tuple[int, int, int], list]]:
    """Map frame outer indices to a full ``(slice, echo, cycle)`` grid."""

    array = np.asarray(frame_indices, dtype=int).reshape(len(frame_indices), -1)
    if array.shape[1] < 3:
        return None
    slices = np.unique(array[:, 0], return_inverse=True)[1].reshape(-1)
    echoes = np.unique(array[:, 1], return_inverse=True)[1].reshape(-1)
    cycles = np.unique(array[:, 2:], axis=0, return_inverse=True)[1].reshape(-1)
    counts = (int(slices.max()) + 1, int(echoes.max()) + 1, int(cycles.max()) + 1)
    positions = list(zip(slices.tolist(), echoes.tolist(), cycles.tolist()))
    if len(set(positions)) != len(positions) or len(positions) != int(np.prod(counts)):
        return None
    return counts, positions


def _cartesian_layout(scan: "_Scan") -> Optional[_RawLayout]:
    sources = scan.sources
    signal = scan.signal
    total = signal.shape[-1]
    if sources.frames is not None:
        acquisitions = tuple(sources.frames.acquisitions)
        samples = tuple(sources.frames.sample_indices)
        indices = tuple(sources.frames.frame_indices)
    elif sources.acquisition is not None and sources.acquisition.num_samples == total:
        acquisitions = (sources.acquisition,)
        samples = (tuple(range(total)),)
        indices = ((0, 0, 0, 0, 0),)
    else:
        return None
    read, phase = acquisitions[0].read_matrix, acquisitions[0].phase_matrix
    if any(
        (item.read_matrix, item.phase_matrix) != (read, phase) for item in acquisitions
    ):
        scan.warn("Cartesian frames with different matrices are stored as spectroscopy")
        return None
    grid = _frame_grid(indices)
    if grid is None:
        scan.warn("Cartesian frames do not form a full slice/echo/repetition grid")
        return None
    (n_slices, n_echoes, n_cycles), positions = grid
    coils = scan.coils
    ni = n_slices * n_echoes
    is_epi = scan.base_method.upper() == "EPI"
    orders = {tuple(item.phase_indices) for item in acquisitions}
    chronological = len(orders) == 1 and not is_epi
    if len(orders) > 1:
        scan.warn(
            "phase-encoding order differs between frames; lines are stored in k-space order"
        )
    if not is_epi and any(
        d < 0 for item in acquisitions for d in item.readout_directions
    ):
        scan.warn("reversed readout lines are stored in forward direction")
    if is_epi:
        blocks = np.zeros((n_cycles, ni, coils, phase * read), dtype=np.complex128)
    else:
        blocks = np.zeros((n_cycles, phase, ni, coils, read), dtype=np.complex128)
    for frame, (slice_index, echo_index, cycle) in enumerate(positions):
        item = acquisitions[frame]
        raw = signal[:, list(samples[frame])].reshape(coils, phase, read)
        object_index = slice_index * n_echoes + echo_index
        kspace = np.zeros((coils, phase, read), dtype=np.complex128)
        for line in range(phase):
            values = raw[:, line, :]
            if item.readout_directions[line] < 0:
                values = values[:, ::-1]
            if is_epi:
                kspace[:, item.phase_indices[line], :] = values
            else:
                row = line if chronological else item.phase_indices[line]
                blocks[cycle, row, object_index] = values
        if is_epi:
            # brukerapi and ParaVision mirror every odd k-space row of EPI data.
            kspace[:, 1::2, :] = kspace[:, 1::2, ::-1]
            blocks[cycle, object_index] = kspace.reshape(coils, phase * read)
    steps = (
        tuple(int(value - phase // 2) for value in acquisitions[0].phase_indices)
        if chronological
        else _linear_steps(phase)
    )
    if is_epi:
        return _RawLayout(
            kind="epi",
            blocks=blocks.reshape(n_cycles * ni, coils, phase * read),
            acq_size=(2 * read * phase, 1),
            dim_desc=("Spatial", "Spatial"),
            matrix=(read, phase),
            ni=ni,
            nr=n_cycles,
            slices=n_slices,
            echoes=n_echoes,
            enc_steps=(_linear_steps(phase),),
            kblock=False,
        )
    return _RawLayout(
        kind="cartesian_2d",
        blocks=blocks.reshape(n_cycles * phase * ni, coils, read),
        acq_size=(2 * read, phase),
        dim_desc=("Spatial", "Spatial"),
        matrix=(read, phase),
        ni=ni,
        nr=n_cycles,
        slices=n_slices,
        echoes=n_echoes,
        enc_steps=(steps,),
    )


def _volume_layout(scan: "_Scan") -> Optional[_RawLayout]:
    volumes = scan.sources.volumes
    if volumes is None:
        return None
    frames = volumes.frames
    read, phase = volumes.read_matrix, volumes.phase_matrix
    partitions = int(volumes.partition_matrix)
    coils = scan.coils
    count = volumes.num_volumes
    blocks = np.zeros((count, partitions, phase, coils, read), dtype=np.complex128)
    if any(d < 0 for item in frames.acquisitions for d in item.readout_directions):
        scan.warn("reversed readout lines are stored in forward direction")
    for volume in range(count):
        for z_index, frame in enumerate(volumes.volume_frame_indices[volume]):
            item = frames.acquisitions[frame]
            raw = scan.signal[:, list(frames.sample_indices[frame])].reshape(
                coils, phase, read
            )
            for line in range(phase):
                values = raw[:, line, :]
                if item.readout_directions[line] < 0:
                    values = values[:, ::-1]
                blocks[volume, z_index, item.phase_indices[line]] = values
    return _RawLayout(
        kind="cartesian_3d",
        blocks=blocks.reshape(count * partitions * phase, coils, read),
        acq_size=(2 * read, phase, partitions),
        dim_desc=("Spatial", "Spatial", "Spatial"),
        matrix=(read, phase, partitions),
        nr=count,
        enc_steps=(_linear_steps(phase), _linear_steps(partitions)),
    )


def _spiral_layout(scan: "_Scan") -> Optional[_RawLayout]:
    spiral = scan.sources.spiral
    if spiral is None:
        return None
    points = int(spiral.samples_per_frame)
    grid = _frame_grid(spiral.frame_indices)
    if grid is None:
        counts, positions = (1, 1, spiral.num_frames), [
            (0, 0, f) for f in range(spiral.num_frames)
        ]
    else:
        counts, positions = grid
    n_slices, n_echoes, n_cycles = counts
    ni = n_slices * n_echoes
    blocks = np.zeros((n_cycles, ni, scan.coils, points), dtype=np.complex128)
    for frame, (slice_index, echo_index, cycle) in enumerate(positions):
        blocks[cycle, slice_index * n_echoes + echo_index] = scan.signal[
            :, list(spiral.sample_indices[frame])
        ]
    return _RawLayout(
        kind="spiral",
        blocks=blocks.reshape(n_cycles * ni, scan.coils, points),
        acq_size=(2 * points, 1),
        dim_desc=("Spatial", "Spatial"),
        matrix=tuple(int(value) for value in spiral.matrix),
        ni=ni,
        nr=n_cycles,
        slices=n_slices,
        echoes=n_echoes,
    )


def _csi_layout(scan: "_Scan") -> Optional[_RawLayout]:
    spec = scan.sources.spectroscopy
    if spec is None:
        return None
    points = int(spec.spectral_points)
    n_x, n_y = scan.matrix
    chronological = scan.signal.reshape(scan.coils, spec.num_encodings, points)
    blocks = np.ascontiguousarray(np.moveaxis(chronological, 0, 1))
    if scan.is_luca_csi4:
        job0 = blocks.reshape(-1)
    else:
        grid = spec.reshape_signal(scan.signal)
        job0 = np.asarray(np.moveaxis(grid, 0, -2)).reshape(-1)
    return _RawLayout(
        kind="csi",
        blocks=blocks,
        acq_size=(2 * points, n_x, n_y),
        dim_desc=("Spectroscopic", "Spatial", "Spatial"),
        matrix=(n_x, n_y),
        nr=spec.num_repetitions,
        enc_steps=(_linear_steps(n_x), _linear_steps(n_y)),
        job0=job0,
    )


def _generic_layout(scan: "_Scan") -> _RawLayout:
    total = scan.signal.shape[-1]
    counts = tuple(int(value) for value in scan.adc_counts)
    if sum(counts) != total or not counts:
        counts = (total,)
    longest = max(counts)
    blocks = np.zeros((len(counts), scan.coils, longest), dtype=np.complex128)
    start = 0
    for index, count in enumerate(counts):
        blocks[index, :, :count] = scan.signal[:, start : start + count]
        start += count
    if len(set(counts)) > 1:
        scan.warn("ADC events of different length are zero-padded to the longest one")
    return _RawLayout(
        kind="generic",
        blocks=blocks,
        acq_size=(2 * longest,),
        dim_desc=("Spectroscopic",),
        matrix=(longest,),
        nr=len(counts),
    )


# ---------------------------------------------------------------------------
# Scan context


class _Scan:
    """Everything the parameter writers need about one simulated scan."""

    def __init__(
        self,
        result,
        *,
        program,
        phantom,
        acquisition,
        acquisition_frames,
        options: BrukerExportOptions,
        template: Optional[BrukerTemplateMetadata],
        scale: Optional[float],
        warnings: list[str],
    ):
        self.result = result
        self.program = program
        self.phantom = phantom
        self.options = options
        self.template = template
        self.warnings = warnings
        self.profile: ParaVisionProfile = get_paravision_profile(
            options.paravision_version
        )
        self.signal = _as_coil_signal(result.signal)
        self.coils = int(self.signal.shape[0])
        self.scale = _signal_scale(self.signal, scale)
        self.metadata = dict(getattr(result, "metadata", None) or {})
        program_metadata = dict(getattr(program, "metadata", None) or {})
        definitions: dict = {}
        for source in (
            program_metadata.get("definitions"),
            self.metadata.get("sequence_definitions"),
        ):
            for key, value in dict(source or {}).items():
                definitions.setdefault(str(key).lower(), value)
        self.definitions = definitions
        self.sources = _AcquisitionSources.collect(
            result, acquisition, acquisition_frames
        )
        method = options.method_name or self.sources.suggested_method()
        self.method = self.profile.method_name(method)
        self.base_method = self.method.split(":")[-1]
        self.is_luca_csi4 = (
            self.sources.spectroscopy is not None
            and self.base_method.lower() == "lucacsi4"
        )
        self.source = str(
            getattr(program, "source", None)
            or self.metadata.get("program_source")
            or "simulated"
        )
        self.adc_counts = self._adc_event_counts()
        self.dwell_s = self._dwell_s()
        self.matrix = self._matrix_hint()
        self.layout = self._build_layout()
        if options.matrix is None:
            self.matrix = tuple(int(value) for value in self.layout.matrix[:2])
            if len(self.matrix) == 1:
                self.matrix = (self.matrix[0], self.layout.block_count)
        self.fov_m = self._fov_m()
        self.nucleus = normalize_nucleus(self.metadata.get("nucleus") or "1H")
        self.field_strength_t = _finite(self.metadata.get("field_strength_t"))
        self._frequencies()
        scan_name = options.scan_name
        if scan_name is None:
            scan_name = (
                f"{GENERATOR} {Path(self.source).name}"
                if self.source != "simulated"
                else f"{GENERATOR} Sequence Simulation"
            )
        self.scan_name = scan_name
        self.protocol_name = re.sub(r"\s+", "_", scan_name)[: self.profile.name_limit]
        self.repetition_time_ms = self._repetition_time_ms()
        self.echo_times_ms = self._echo_times_ms()
        self.echo_time_ms = self.echo_times_ms[0]
        self.flip_angle_deg = (
            definition_float(self.definitions, "FlipAngleDeg", "FlipAngle") or 0.0
        )
        self.duration_ms = self._duration_ms()
        self.series = self._image_series()
        self.geometry = _Geometry.build(self)

    # -- helpers ----------------------------------------------------------

    def warn(self, text: str) -> None:
        if text not in self.warnings:
            self.warnings.append(text)

    def definition(self, *names) -> Optional[float]:
        return definition_float(self.definitions, *names)

    def _adc_event_counts(self) -> tuple[int, ...]:
        events = tuple(getattr(self.program, "adc_events", ()) or ())
        if events:
            return tuple(int(event.num_samples) for event in events)
        dimensions = _try(lambda: self.result.acquisition_dimensions)
        if dimensions is not None:
            return tuple(int(value) for value in dimensions.adc_event_sample_counts)
        return (int(self.signal.shape[-1]),)

    def _dwell_s(self) -> float:
        sources = self.sources
        layouts = sources.cartesian_layouts()
        if layouts:
            return float(layouts[0].dwell_s)
        for item in (sources.spectroscopy, sources.spiral):
            if item is not None:
                return float(item.dwell_s)
        if sources.volumes is not None:
            return float(sources.volumes.frames.acquisitions[0].dwell_s)
        events = tuple(getattr(self.program, "adc_events", ()) or ())
        if events and getattr(events[0], "dwell_s", None):
            return float(events[0].dwell_s)
        times = np.asarray(getattr(self.result, "adc_times_s", ()), dtype=float)
        diffs = np.diff(times) if times.size > 1 else np.empty(0)
        diffs = diffs[diffs > 0]
        return float(np.median(diffs)) if diffs.size else 1.0

    def _matrix_hint(self) -> tuple[int, ...]:
        if self.options.matrix is not None:
            return tuple(self.options.matrix)
        spec = self.sources.spectroscopy
        if spec is not None:
            return tuple(int(value) for value in spec.matrix)
        return ()

    def _build_layout(self) -> _RawLayout:
        if self.sources.spectroscopy is not None:
            return _csi_layout(self)
        for builder in (_volume_layout, _cartesian_layout, _spiral_layout):
            layout = builder(self)
            if layout is not None:
                return layout
        if self.base_method.upper() not in (
            "NSPECT",
            "PRESS",
            "STEAM",
            "ISIS",
            "SINGLEPULSE",
        ):
            self.warn(
                f"{self.method} raw data are stored as one spectroscopic block per ADC "
                "event because the simulation has no Cartesian, spiral or CSI layout"
            )
        return _generic_layout(self)

    def _fov_m(self) -> tuple[float, ...]:
        if self.options.fov_m is not None:
            return tuple(self.options.fov_m)
        sources = self.sources
        if sources.volumes is not None:
            return tuple(float(value) for value in sources.volumes.fov_m)
        layouts = sources.cartesian_layouts()
        if layouts:
            return tuple(float(value) for value in layouts[0].fov_m)
        for item in (sources.spectroscopy, sources.spiral):
            if item is not None:
                return tuple(float(value) for value in item.fov_m)
        fov = getattr(self.phantom, "fov", None)
        if fov is not None:
            values = np.atleast_1d(np.asarray(fov, dtype=float)).reshape(-1)
            if values.size == 1:
                return (float(values[0]), float(values[0]))
            if values.size >= 2:
                return (float(values[0]), float(values[1]))
        return (1.0, 1.0)

    def _frequencies(self) -> None:
        gamma = gyromagnetic_ratio_hz_per_t(self.nucleus)
        computed = None
        if self.field_strength_t and gamma:
            computed = self.field_strength_t * gamma / 1e6
        template_mhz = None
        if self.template is not None and self.options.use_template_frequency:
            template_mhz = self.template.reference_frequency_mhz(self.nucleus)
        if computed is not None:
            if template_mhz and abs(template_mhz - computed) <= 0.05 * computed:
                reference = template_mhz
            else:
                reference = computed
                if template_mhz:
                    self.warn(
                        f"template {self.nucleus} frequency {template_mhz:g} MHz does not "
                        f"match {self.field_strength_t:g} T; using {computed:.6f} MHz"
                    )
        elif template_mhz:
            reference = template_mhz
        else:
            reference = DEFAULT_FIELD_STRENGTH_T * (gamma or 42_577_478.518) / 1e6
            self.warn(
                f"no field strength in the simulation; assuming {DEFAULT_FIELD_STRENGTH_T:g} T"
            )
        self.reference_ppm = 4.7 if self.nucleus == "1H" else 0.0
        self.work_ppm = self._work_ppm()
        self.offset_ppm = self.work_ppm - self.reference_ppm
        self.reference_frequency_mhz = float(reference)
        self.working_frequency_mhz = reference * (1.0 + self.offset_ppm * 1e-6)
        self.offset_hz = self.offset_ppm * reference
        self.spectral_width_hz = 1.0 / self.dwell_s if self.dwell_s > 0 else 1.0
        self.spectral_width_ppm = self.spectral_width_hz / self.working_frequency_mhz

    def _work_ppm(self) -> float:
        for key in ("sequence_reference_ppm", "spectral_reference_ppm"):
            value = _finite(self.metadata.get(key))
            if value is not None:
                return value
        phantom_metadata = getattr(self.phantom, "metadata", None) or {}
        value = _finite(phantom_metadata.get("spectral_reference_ppm"))
        if value is None:
            value = _finite(getattr(self.phantom, "spectral_reference_ppm", None))
        return value if value is not None else (4.7 if self.nucleus == "1H" else 0.0)

    def _duration_ms(self) -> float:
        duration = _finite(getattr(self.program, "duration_s", None))
        if duration is None:
            duration = self.definition("TotalDuration")
        if duration is not None and duration > 0:
            return duration * 1e3
        times = np.asarray(getattr(self.result, "adc_times_s", ()), dtype=float)
        if times.size:
            return float((times.max() - times.min() + self.dwell_s) * 1e3)
        return 0.0

    def _repetition_time_ms(self) -> float:
        value = self.definition("TR", "RepetitionTime")
        if value is not None and value > 0:
            return value * 1e3
        duration = self._duration_ms()
        spec = self.sources.spectroscopy
        if spec is not None and spec.num_encodings:
            return duration / spec.num_encodings
        return duration

    def _echo_times_ms(self) -> tuple[float, ...]:
        values = definition_values(self.definitions, "EchoTimes")
        values = values[np.isfinite(values)] if values.size else values
        if values.size >= self.layout.echoes and values.size:
            return tuple(
                float(value) * 1e3 for value in values[: max(1, self.layout.echoes)]
            )
        echo = self.definition("TE")
        return (max(0.0, echo * 1e3) if echo is not None else 0.0,)

    def _image_series(self) -> list:
        if self.layout.kind == "generic":
            return []
        try:
            return list(
                extract_image_series(
                    self.result, include_pools=self.options.include_pools
                )
            )
        except Exception as error:  # pragma: no cover - defensive
            self.warn(f"images could not be reconstructed: {error}")
            return []

    @property
    def slice_thickness_mm(self) -> float:
        if self.options.slice_thickness_mm is not None:
            return float(self.options.slice_thickness_mm)
        if self.series:
            return float(self.series[0].slice_thickness_m) * 1e3
        value = self.definition("SliceThickness")
        if value is not None and value > 0:
            return value * 1e3
        return 1.0

    @property
    def fov_mm(self) -> tuple[float, ...]:
        return tuple(value * 1e3 for value in self.fov_m)


def _finite(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


@dataclass(frozen=True)
class _Geometry:
    """Slice orientation (rows read, phase, slice) and per-slice offsets."""

    orientation: np.ndarray
    centres_mm: np.ndarray

    @classmethod
    def build(cls, scan: _Scan) -> "_Geometry":
        count = max(1, scan.layout.slices)
        series = scan.series[0] if scan.series else None
        if series is None:
            return cls(np.eye(3), np.zeros((count, 3)))
        row = np.asarray(series.row_direction, dtype=float)
        column = np.asarray(series.column_direction, dtype=float)
        row_spacing, column_spacing = (value * 1e3 for value in series.pixel_spacing_m)
        centres: dict[int, np.ndarray] = {}
        for frame in series.frames:
            first = np.asarray(frame.position_m, dtype=float) * 1e3
            centre = (
                first
                + 0.5 * (series.columns - 1) * column_spacing * row
                + 0.5 * (series.rows - 1) * row_spacing * column
            )
            centres.setdefault(frame.slice_index, centre)
        values = np.asarray([centres[key] for key in sorted(centres)])
        if len(values) != count:
            values = np.repeat(values.mean(axis=0, keepdims=True), count, axis=0)
        return cls(np.asarray(series.orientation_matrix, dtype=float), values)

    def offsets(self) -> np.ndarray:
        """Return ``(slices, 3)`` read, phase and slice offsets in mm."""

        return (self.centres_mm @ self.orientation.T) * np.array([-1.0, -1.0, 1.0])

    def slice_orient(self) -> str:
        normal = np.abs(self.orientation[2])
        return ("sagittal", "coronal", "axial")[int(np.argmax(normal))]

    def read_orient(self) -> str:
        read = np.abs(self.orientation[0])
        return ("L_R", "A_P", "H_F")[int(np.argmax(read))]

    def slice_distance_mm(self, thickness_mm: float) -> float:
        if len(self.centres_mm) > 1:
            steps = np.linalg.norm(np.diff(self.centres_mm, axis=0), axis=1)
            return float(np.median(steps))
        return float(thickness_mm)


# ---------------------------------------------------------------------------
# Raw data


def _as_coil_signal(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.complex128)
    if values.ndim == 1:
        return np.ascontiguousarray(values.reshape(1, -1))
    if values.ndim == 2:
        return np.ascontiguousarray(values)
    raise ValueError("Bruker raw export requires signal shape (adc,) or (coil, adc)")


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
    values = np.asarray(values)
    scaled = np.clip(
        np.rint(np.stack((values.real, values.imag), axis=-1) * scale),
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
    )
    return np.ascontiguousarray(scaled.astype("<i4", copy=False).reshape(-1))


def _standard_kblock_complex_size(readout_samples: int, num_coils: int) -> int:
    bytes_per_int = 4
    acq_size = 2 * int(readout_samples)
    block_ints = np.ceil(acq_size * int(num_coils) * bytes_per_int / 1024.0)
    return int(block_ints * 1024.0 / bytes_per_int / 2)


def _uses_kblock(scan: _Scan) -> bool:
    return scan.layout.kblock and scan.profile.go_block_size == "Standard_KBlock_Format"


def _raw_blocks(scan: _Scan) -> np.ndarray:
    blocks = scan.layout.blocks
    return np.conj(blocks) if scan.options.conjugate_raw_data else blocks


def _fid_int32(scan: _Scan) -> np.ndarray:
    blocks = _raw_blocks(scan).reshape(scan.layout.block_count, -1)
    if _uses_kblock(scan):
        size = _standard_kblock_complex_size(scan.layout.readout, scan.coils)
        padded = np.zeros((blocks.shape[0], size), dtype=np.complex128)
        padded[:, : blocks.shape[1]] = blocks
        blocks = padded
    return _interleaved_int32(blocks, scan.scale)


def _job0_int32(scan: _Scan) -> np.ndarray:
    values = scan.layout.job0
    if values is None:
        values = scan.layout.blocks.reshape(-1)
    if scan.options.conjugate_raw_data:
        values = np.conj(values)
    return _interleaved_int32(values, scan.scale)


# ---------------------------------------------------------------------------
# Parameter files


def _write_scan(
    scan_dir: Path, scan: _Scan, context: BrukerFileContext, identity: ScanIdentity
) -> list[Path]:
    scan_dir.mkdir(parents=True, exist_ok=True)
    expno = context.expno
    files: list[Path] = []
    raw = scan.options.raw_data_files
    if raw in ("fid", "both"):
        path = scan_dir / "fid"
        _fid_int32(scan).tofile(path)
        files.append(path)
    if raw in ("rawdata.job0", "both"):
        path = scan_dir / "rawdata.job0"
        _job0_int32(scan).tofile(path)
        files.append(path)
    files.append(
        _write_acqp(scan_dir / "acqp", scan, context, identity, f"{expno}/acqp")
    )
    files.append(_write_method(scan_dir / "method", scan, context, f"{expno}/method"))
    protocol = _protocol_info(scan, context)
    reconstructions = _reconstructions(scan)
    instance_uids = [generate_uid() for _ in reconstructions]
    for procno, frames in enumerate(reconstructions, start=1):
        files.extend(
            write_pdata(
                scan_dir / "pdata" / str(procno),
                context,
                identity,
                protocol,
                frames,
                procno=procno,
                coils=scan.coils,
                conjugated=scan.options.conjugate_raw_data,
                instance_uid=instance_uids[procno - 1],
            )
        )
    if reconstructions and scan.profile.scan_visu_pars:
        stored, slope = scale_to_int16(reconstructions[0].values)
        files.append(
            write_visu_pars(
                scan_dir / "visu_pars",
                context,
                f"{expno}/visu_pars",
                identity,
                protocol,
                reconstructions[0],
                stored=stored,
                slope=slope,
                procno=1,
                instance_uid=instance_uids[0],
                scan_level=True,
            )
        )
    files.extend(_write_scan_extras(scan_dir, scan, context))
    files.append(_write_provenance(scan_dir, scan, context))
    return files


def _ns_array(values) -> list:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def _channel_array(value: float, count: int = 8) -> list:
    values = [0.0] * count
    values[0] = float(value)
    return values


def _write_acqp(
    path, scan: _Scan, context, identity: ScanIdentity, relative: str
) -> Path:
    profile = scan.profile
    layout = scan.layout
    writer = context.writer(relative, "acq")
    base = scan.base_method
    ni = layout.ni
    geometry = scan.geometry
    offsets = geometry.offsets()
    per_object = np.asarray(
        [offsets[min(index // layout.echoes, len(offsets) - 1)] for index in range(ni)]
    )
    writer.string("PULPROG", f"{base}.ppg", 32)
    writer.string("GRDPROG", "", 16)
    writer.scalar("ACQ_experiment_mode", "SingleExperiment")
    writer.scalar("ACQ_DS_enabled", True)
    scan_size = int(layout.acq_size[0]) * scan.coils
    write_job0 = scan.options.raw_data_files in ("rawdata.job0", "both")
    # JOB_DESCRIPTION_TYPE: scanSize, transactionBlocks, dummyScans,
    # nTotalScans, receiverGain, swh, scanShift, nStoredScans.  PV360 drops
    # scanShift, appends the channel number and the job title and counts
    # scanSize per receive channel.
    blocks = layout.block_count
    if profile.job_style == "pv6":
        writer.scalar("ACQ_jobs_size", 1)
        job = FID_ONLY_JOB
        if write_job0:
            job = (scan_size, 1, 0, blocks, 64, scan.spectral_width_hz, 0, blocks)
        writer.structs("ACQ_jobs", [job])
    elif profile.job_style == "pv360":
        writer.scalar("ACQ_jobs_size", 1)
        writer.structs(
            "ACQ_jobs",
            [
                (
                    int(layout.acq_size[0]),
                    1,
                    0,
                    blocks,
                    64,
                    scan.spectral_width_hz,
                    blocks,
                    1,
                    "job0",
                )
            ],
        )
    writer.scalar("ACQ_dim", len(layout.acq_size))
    writer.enums("ACQ_dim_desc", layout.dim_desc)
    writer.array("ACQ_size", layout.acq_size)
    writer.scalar("ACQ_ns_list_size", 1)
    writer.scalar("ACQ_ns", 1)
    writer.array("ACQ_ns_list", [1])
    writer.scalar("ACQ_phase_factor", 1)
    writer.scalar("ACQ_scan_size", "One_scan")
    writer.scalar("NI", ni)
    writer.scalar("NA", 1)
    writer.scalar("NAE", 1)
    writer.scalar("NR", layout.nr)
    writer.scalar("DS", 0)
    writer.scalar("SW_h", scan.spectral_width_hz)
    writer.scalar("SW", scan.spectral_width_ppm)
    writer.scalar("FW", scan.spectral_width_hz)
    writer.scalar("RG", 1)
    writer.scalar("AQ_mod", "qdig")
    writer.scalar("DR", 21)
    writer.scalar("PAPS", "QP")
    writer.scalar("PH_ref", 0)
    writer.scalar("BF1", scan.reference_frequency_mhz)
    writer.scalar("SFO1", scan.working_frequency_mhz)
    writer.scalar("O1", scan.offset_hz)
    writer.scalar("ACQ_O1_list_size", ni)
    writer.array("ACQ_O1_list", np.zeros(ni))
    writer.array("ACQ_O1B_list", np.zeros(ni))
    writer.string("NUC1", scan.nucleus, 8)
    writer.string("NUCLEUS", scan.nucleus, 8)
    writer.scalar("ACQ_Routing", "RouteDefinedByCoil")
    writer.string("ACQ_protocol_name", scan.protocol_name, 64)
    writer.string("ACQ_scan_name", f"{scan.scan_name} (E{context.expno})", 64)
    writer.string("ACQ_method", scan.method, 20 if profile.classic_values else 40)
    writer.scalar("ACQ_completed", True)
    writer.scalar("ACQ_pipe_status", "Wrapup")
    writer.scalar("ACQ_scans_completed", layout.block_count)
    writer.scalar("ACQ_nr_completed", layout.nr)
    writer.scalar("ACQ_total_completed", layout.block_count)
    writer.scalar("ACQ_word_size", "_32_BIT")
    writer.scalar("NECHOES", layout.echoes)
    writer.scalar("ACQ_n_echo_images", layout.echoes)
    writer.array("ACQ_fov", [value / 10.0 for value in scan.fov_mm])
    writer.scalar("ACQ_read_ext", 1)
    writer.array("ACQ_slice_angle", np.zeros(ni))
    writer.scalar("ACQ_slice_orient", "Arbitrary_Oblique")
    writer.scalar("ACQ_patient_pos", identity.patient_position)
    writer.array("ACQ_read_offset", per_object[:, 0])
    writer.array("ACQ_phase1_offset", per_object[:, 1])
    writer.array("ACQ_phase2_offset", np.zeros(ni))
    writer.array("ACQ_slice_offset", per_object[:, 2])
    distance = geometry.slice_distance_mm(scan.slice_thickness_mm)
    writer.array("ACQ_slice_sepn", [distance] * max(1, layout.slices - 1))
    writer.scalar("ACQ_slice_sepn_mode", "Equidistant")
    writer.scalar("ACQ_slice_thick", scan.slice_thickness_mm)
    writer.array("ACQ_obj_order", np.arange(ni))
    writer.scalar("ACQ_flip_angle", scan.flip_angle_deg)
    writer.array("ACQ_echo_time", scan.echo_times_ms)
    if len(scan.echo_times_ms) > 1:
        writer.array(
            "ACQ_inter_echo_time", [float(np.mean(np.diff(scan.echo_times_ms)))]
        )
    else:
        # ParaVision 7 FLASH writes the echo time here for single-echo scans.
        writer.array("ACQ_inter_echo_time", [scan.echo_time_ms])
    writer.array(
        "ACQ_recov_time", [max(0.0, scan.repetition_time_ms - scan.echo_time_ms)]
    )
    writer.array("ACQ_repetition_time", [scan.repetition_time_ms])
    writer.scalar("ACQ_scan_time", scan.duration_ms)
    write_date(writer, profile, "ACQ_time", context.created, 24)
    writer.array("ACQ_time_points", [0])
    write_abs_time(writer, profile, "ACQ_abs_time", context.created)
    writer.string("ACQ_operator", identity.operator, 16)
    writer.string("ACQ_institution", identity.institution, 48)
    writer.string("ACQ_station", identity.station, 64)
    writer.string("ACQ_system_order_number", identity.system_order_number, 24)
    if profile.software_version_char_array:
        writer.string("ACQ_sw_version", profile.software_version, 65)
    else:
        writer.date("ACQ_sw_version", profile.software_version)
    writer.scalar("BYTORDA", "little")
    writer.string("INSTRUM", identity.station, 64)
    if profile.go_parameters:
        writer.scalar("GO_init_files", False)
        writer.scalar("GO_data_save", True)
        writer.scalar(
            "GO_block_size",
            "Standard_KBlock_Format" if _uses_kblock(scan) else "continuous",
        )
        writer.scalar("GO_raw_data_format", "GO_32BIT_SGN_INT")
        writer.scalar("GO_disp_update", "Each_Accum")
        writer.scalar("GO_online_reco", True)
        writer.scalar("GO_reco_display", True)
    writer.scalar("ACQ_grad_matrix_size", ni)
    matrix = np.repeat(geometry.orientation[None, :, :], ni, axis=0)
    writer.array("ACQ_grad_matrix", matrix, shape=matrix.shape)
    writer.scalar("NSLICES", layout.slices)
    writer.scalar("ACQ_rare_factor", 1)
    if layout.kind == "csi":
        encoding = _csi_encoding_metadata(scan)
        writer.enums("ACQ_phase_encoding_mode", ["User_Defined_Encoding"] * 2)
        writer.array("ACQ_phase_enc_start", [-1, -1])
        writer.scalar("ACQ_spatial_size_0", len(encoding["indices"]))
        writer.array("ACQ_spatial_phase_0", encoding["chronological_values_x"])
        writer.scalar("ACQ_spatial_size_1", len(encoding["indices"]))
        writer.array("ACQ_spatial_phase_1", encoding["chronological_values_y"])
    elif layout.kind in ("cartesian_2d", "cartesian_3d", "epi"):
        dims = len(layout.enc_steps)
        # One entry per ACQ_dim, the first one for the readout (as in PV7).
        writer.enums(
            "ACQ_phase_encoding_mode", ["Read"] + ["User_Defined_Encoding"] * dims
        )
        writer.array("ACQ_phase_enc_start", [-1] * (dims + 1))
        for axis, steps in enumerate(layout.enc_steps, start=1):
            count = layout.matrix[axis]
            writer.scalar(f"ACQ_spatial_size_{axis}", len(steps))
            writer.array(
                f"ACQ_spatial_phase_{axis}", [2.0 * step / count for step in steps]
            )
    return writer.write(path)


def _scan_time_text(milliseconds: float) -> str:
    total = int(round(max(0.0, milliseconds)))
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours}h{minutes}m{seconds}s{millis}ms"


def _write_method(path, scan: _Scan, context, relative: str) -> Path:
    layout = scan.layout
    writer = context.writer(relative, "acq")
    base = scan.base_method.upper()
    spectroscopic = layout.kind in ("generic", "csi") and not scan.is_luca_csi4
    spatial_dims = (
        0 if layout.kind == "generic" else (3 if layout.kind == "cartesian_3d" else 2)
    )
    matrix = tuple(int(value) for value in scan.matrix)
    if layout.kind == "cartesian_3d" and len(matrix) == 2:
        matrix = matrix + (int(layout.matrix[2]),)
    fov_mm = tuple(scan.fov_mm)
    if len(fov_mm) < len(matrix):
        fov_mm = fov_mm + (scan.slice_thickness_mm,) * (len(matrix) - len(fov_mm))
    fov_mm = fov_mm[: len(matrix)]
    thickness = scan.slice_thickness_mm
    geometry = scan.geometry
    offsets = geometry.offsets()
    slices = max(1, layout.slices)
    distance = geometry.slice_distance_mm(thickness)

    write_label(writer, scan.profile, "Method", scan.method)
    writer.scalar("PVM_EchoTime", scan.echo_time_ms)
    writer.scalar("PVM_RepetitionTime", scan.repetition_time_ms)
    writer.scalar("PVM_NEchoImages", layout.echoes)
    writer.scalar("PVM_NAverages", 1)
    writer.scalar("PVM_NRepetitions", layout.nr)
    writer.string("PVM_ScanTimeStr", _scan_time_text(scan.duration_ms), 16)
    writer.scalar("PVM_ScanTime", scan.duration_ms)
    writer.scalar("PVM_DeriveGains", True)
    write_label(writer, scan.profile, "PVM_Nucleus1Enum", scan.nucleus)
    writer.string("PVM_Nucleus1", scan.nucleus, 8)
    writer.array("PVM_FrqRef", _channel_array(scan.reference_frequency_mhz))
    writer.array("PVM_FrqWorkOffset", _channel_array(scan.offset_hz))
    writer.array("PVM_FrqWork", _channel_array(scan.working_frequency_mhz))
    writer.array("PVM_FrqRefPpm", _channel_array(scan.reference_ppm))
    writer.array("PVM_FrqWorkOffsetPpm", _channel_array(scan.offset_ppm))
    writer.array("PVM_FrqWorkPpm", _channel_array(scan.work_ppm))
    if spatial_dims:
        write_label(writer, scan.profile, "PVM_SpatDimEnum", f"{spatial_dims}D")
        writer.scalar("PVM_IsotropicFovRes", "Isot_None")
    writer.array("PVM_Matrix", matrix)
    writer.array("PVM_AntiAlias", np.ones(len(matrix)))
    writer.scalar("PVM_EncSpectroscopy", spectroscopic)
    writer.scalar("PVM_EncUseMultiRec", scan.coils > 1)
    writer.enums("PVM_EncActReceivers", ["On"] * scan.coils)
    enc_matrix = matrix if layout.kind not in ("generic",) else (layout.readout,)
    writer.array("PVM_EncZf", np.ones(len(enc_matrix)))
    writer.array("PVM_EncPft", np.ones(len(enc_matrix)))
    writer.array("PVM_EncPpi", np.ones(len(enc_matrix)))
    writer.enums("PVM_EncOrder", ["LINEAR_ENC"] * max(1, len(enc_matrix)))
    writer.array("PVM_EncStart", [-1] * len(enc_matrix))
    writer.scalar("PVM_EncNReceivers", scan.coils)
    writer.scalar("PVM_EncAvailReceivers", max(scan.coils, 1))
    writer.array("PVM_EncChanScaling", np.ones(scan.coils))
    writer.array("PVM_EncMatrix", enc_matrix)
    writer.scalar("PVM_EncTotalAccel", 1)
    if layout.kind in ("cartesian_2d", "cartesian_3d", "epi"):
        for axis, steps in enumerate(layout.enc_steps, start=1):
            count = layout.matrix[axis]
            writer.array(f"PVM_EncSteps{axis}", steps)
        for axis, steps in enumerate(layout.enc_steps, start=1):
            writer.scalar(f"PVM_EncCentralStep{axis}", 1)
            writer.array(
                f"PVM_EncValues{axis}",
                [
                    2.0 * step / layout.matrix[axis]
                    for step in _linear_steps(layout.matrix[axis])
                ],
            )
    writer.scalar("PVM_EffSWh", scan.spectral_width_hz)
    writer.scalar("PVM_EchoPosition", 50)
    readout = layout.readout
    if layout.kind == "csi":
        readout = int(scan.sources.spectroscopy.spectral_points)
    writer.scalar("PVM_AcquisitionTime", readout * scan.dwell_s * 1e3)
    writer.scalar("PVM_DigAutSet", True)
    writer.scalar("PVM_DigQuad", True)
    write_label(writer, scan.profile, "PVM_DigFilter", "Digital_Medium")
    writer.scalar("PVM_DigDw", scan.dwell_s * 1e3)
    writer.scalar("PVM_DigSw", scan.spectral_width_hz)
    writer.scalar("PVM_DigNp", readout)
    # Simulated samples carry no digital-filter group delay.
    writer.scalar("PVM_DigShift", 0)
    writer.scalar("PVM_DigShiftDbl", 0.0)
    writer.scalar("PVM_DigDur", readout * scan.dwell_s * 1e3)
    writer.scalar("PVM_Isotropic", "Isotropic_None")
    writer.array("PVM_Fov", fov_mm)
    writer.array("PVM_FovCm", [value / 10.0 for value in fov_mm])
    writer.array("PVM_SpatResol", [fov / count for fov, count in zip(fov_mm, matrix)])
    writer.scalar("PVM_SliceThick", thickness)
    writer.scalar("PVM_ObjOrderScheme", "Sequential")
    writer.array("PVM_ObjOrderList", np.arange(slices))
    writer.scalar("PVM_NSPacks", 1)
    writer.array("PVM_SPackArrNSlices", [slices])
    writer.scalar("PVM_MajSliceOri", False)
    writer.enums(
        "PVM_SPackArrSliceOrient",
        [scan.options.slice_orientation or geometry.slice_orient()],
    )
    writer.enums(
        "PVM_SPackArrReadOrient",
        [scan.options.read_orientation or geometry.read_orient()],
    )
    mean = offsets.mean(axis=0)
    writer.array("PVM_SPackArrReadOffset", [mean[0]])
    writer.array("PVM_SPackArrPhase1Offset", [mean[1]])
    writer.array("PVM_SPackArrPhase2Offset", [0.0])
    writer.array("PVM_SPackArrSliceOffset", [mean[2]])
    writer.enums("PVM_SPackArrSliceGapMode", ["non_contiguous"])
    writer.array("PVM_SPackArrSliceGap", [max(0.0, distance - thickness)])
    writer.array("PVM_SPackArrSliceDistance", [distance])
    writer.array("PVM_SPackArrGradOrient", geometry.orientation[None], shape=(1, 3, 3))
    for name, column in (("Read", 0), ("Phase1", 1), ("Slice", 2)):
        writer.array(f"PVM_Eff{name}Offset", offsets[:, column])
    for name, column in (("Read", 0), ("Phase1", 1), ("Slice", 2)):
        writer.array(f"PVM_{name}Offset", offsets[:, column])
    writer.scalar("PVM_DummyScans", 0)
    writer.scalar("PVM_NMovieFrames", 1)
    if layout.echoes > 1:
        writer.array("EffectiveTE", scan.echo_times_ms)
    if base == "EPI":
        writer.scalar("NSegments", 1)
        writer.scalar("PVM_EpiNShots", 1)
    if layout.kind == "spiral":
        writer.scalar("PVM_SpiralNbOfInterleaves", 1)
    if layout.kind == "generic":
        _write_spectroscopy_parameters(writer, scan, layout.readout)
    if layout.kind == "csi":
        _write_csi_parameters(writer, scan)
    return writer.write(path)


def _write_spectroscopy_parameters(
    writer, scan: _Scan, points: int, *, luca: bool = False
):
    bandwidth = scan.spectral_width_hz
    write_label(writer, scan.profile, "PVM_SpecDimEnum", "1D")
    writer.array("PVM_SpecMatrix", [points])
    writer.array("PVM_SpecSWH", [bandwidth])
    writer.array("PVM_SpecSW", [scan.spectral_width_ppm])
    writer.array("PVM_SpecDwellTime", [scan.dwell_s * (5e5 if luca else 1e6)])
    writer.scalar("PVM_SpecAcquisitionTime", points * scan.dwell_s * 1e3)
    writer.array("PVM_SpecNomRes", [bandwidth / (2.0 * points)])
    writer.scalar("PVM_SpecOffsetHz", 0.0)
    writer.scalar("PVM_SpecOffsetppm", 0.0)


def _write_csi_parameters(writer, scan: _Scan) -> None:
    spec = scan.sources.spectroscopy
    encoding = _csi_encoding_metadata(scan)
    n_x, n_y = scan.matrix
    _write_spectroscopy_parameters(
        writer, scan, int(spec.spectral_points), luca=scan.is_luca_csi4
    )
    writer.array("PVM_EncSteps0", encoding["steps_x"])
    writer.array("PVM_EncSteps1", encoding["steps_y"])
    writer.array("PVM_EncValues0", encoding["values_x"])
    writer.array("PVM_EncValues1", encoding["values_y"])
    writer.scalar("PVM_EncCentralStep0", 1)
    writer.scalar("PVM_EncCentralStep1", 1)
    if not scan.is_luca_csi4:
        return
    count = spec.encodings_per_repetition
    writer.scalar("CentricEncOrder_OnOff", "On")
    writer.array("CentricEncOrderMatrixx", encoding["signed_x"])
    writer.array("CentricEncOrderMatrixy", encoding["signed_y"])
    writer.array("Distance2kspacecenter", encoding["distances"])
    writer.array("Angles", np.zeros(count))
    writer.scalar("SpiralEncOrder_OnOff", "Off")
    writer.array("SpiralEncOrderMatrixx", np.zeros(count, dtype=int))
    writer.array("SpiralEncOrderMatrixy", np.zeros(count, dtype=int))
    writer.scalar("PhaseEncGrad_OnOff", "On")
    writer.scalar("KFiltering", "On")
    writer.array("Reco_x", encoding["reco_x"])
    writer.array("Reco_y", encoding["reco_y"])


def _csi_encoding_metadata(scan: _Scan) -> dict[str, tuple]:
    spec = scan.sources.spectroscopy
    if spec is None:
        raise ValueError("CSI encoding metadata requires a spectroscopic acquisition")
    n_x, n_y = (int(value) for value in scan.matrix[:2])
    if (n_x, n_y) != tuple(int(value) for value in spec.matrix):
        raise ValueError(
            "CSI export requires PVM_Matrix to match the simulated CSI grid"
        )
    chronological = tuple(
        index
        for index, repetition in zip(spec.encoding_indices, spec.repetition_indices)
        if repetition == 0
    )
    if len(chronological) != n_x * n_y:
        raise ValueError("CSI export requires one complete first repetition")
    for repetition in range(1, spec.num_repetitions):
        order = tuple(
            index
            for index, value in zip(spec.encoding_indices, spec.repetition_indices)
            if value == repetition
        )
        if order != chronological:
            raise ValueError(
                "Bruker CSI export requires the same phase-encoding order in every repetition"
            )
    steps_x = _linear_steps(n_x)
    steps_y = _linear_steps(n_y)
    signed_x = tuple(int(x - n_x // 2) for x, _ in chronological)
    signed_y = tuple(int(y - n_y // 2) for _, y in chronological)
    values_x = tuple(2.0 * value / n_x for value in steps_x)
    values_y = tuple(2.0 * value / n_y for value in steps_y)
    fov_mm = tuple(value * 1000.0 for value in scan.fov_m)
    distances = tuple(
        float(np.hypot(x / fov_mm[0], y / fov_mm[1]))
        for x, y in zip(signed_x, signed_y)
    )
    if scan.is_luca_csi4 and np.any(np.diff(distances) < -1e-12):
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
        "chronological_values_x": tuple(values_x[x] for x, _ in chronological),
        "chronological_values_y": tuple(values_y[y] for _, y in chronological),
        "distances": distances,
        "reco_x": tuple(int(x) for x, _ in chronological),
        "reco_y": tuple(int(y) for _, y in chronological),
    }


# ---------------------------------------------------------------------------
# Reconstructions


def _protocol_info(scan: _Scan, context) -> ProtocolInfo:
    layout = scan.layout
    grad = {
        "cartesian_3d": ("read_enc", "phase_enc", "phase_enc"),
        "generic": ("read_enc",),
        "csi": ("phase_enc", "phase_enc"),
    }.get(layout.kind, ("read_enc", "phase_enc"))
    readout = layout.readout
    trajectory = {
        "spiral": "SpiralTraversal",
        "epi": "RectilinearTraversal",
    }.get(layout.kind, "RectilinearTraversal")
    return ProtocolInfo(
        method=scan.method,
        protocol=scan.protocol_name,
        scan_name=scan.scan_name,
        acquired=context.created,
        repetition_time_ms=scan.repetition_time_ms,
        echo_times_ms=scan.echo_times_ms,
        flip_angle_deg=scan.flip_angle_deg,
        frequency_mhz=scan.working_frequency_mhz,
        nucleus=scan.nucleus,
        phase_encoding_steps=(
            int(np.prod(layout.matrix[1:])) if len(layout.matrix) > 1 else 1
        ),
        acquisition_size=layout.acq_size,
        pixel_bandwidth_hz=scan.spectral_width_hz / max(1, readout),
        scan_time_ms=scan.duration_ms,
        echo_train_length=layout.matrix[1] if layout.kind == "epi" else 1,
        grad_encoding=grad,
        is_epi=layout.kind == "epi",
        trajectory=trajectory,
    )


def _reconstructions(scan: _Scan) -> list[PdataFrames]:
    layout = scan.layout
    frames: list[PdataFrames] = []
    if layout.kind == "csi":
        frames.append(_csi_spectra(scan))
    elif layout.kind == "generic":
        spectra = np.fft.fftshift(np.fft.fft(layout.blocks, axis=-1), axes=-1)
        magnitude = np.sqrt(np.sum(np.abs(spectra) ** 2, axis=1))
        frames.append(
            frames_from_spectra(
                magnitude,
                spectral_width_ppm=scan.spectral_width_ppm,
                frequency_mhz=scan.working_frequency_mhz,
                offset_ppm=scan.work_ppm,
                thickness_mm=scan.slice_thickness_mm,
            )
        )
    for series in scan.series:
        frames.append(frames_from_image_series(series))
    if not frames:
        fallback = _fallback_images(scan)
        if fallback is not None:
            frames.append(fallback)
    return frames


def _csi_spectra(scan: _Scan) -> PdataFrames:
    spec = scan.sources.spectroscopy
    spectra = np.asarray(spec.reconstruct_spectra(scan.signal))
    magnitude = np.sqrt(np.sum(np.abs(spectra) ** 2, axis=0))
    if spec.num_repetitions == 1:
        magnitude = magnitude[None, ...]
    return frames_from_spectra(
        magnitude,
        spectral_width_ppm=scan.spectral_width_ppm,
        frequency_mhz=scan.working_frequency_mhz,
        offset_ppm=scan.work_ppm,
        fov_mm=scan.fov_mm[:2],
        thickness_mm=scan.slice_thickness_mm,
        description="CSI spectra",
    )


def _fallback_images(scan: _Scan) -> Optional[PdataFrames]:
    """Reconstruct Cartesian frames without geometry (identity orientation)."""

    sources = scan.sources
    combine = "rss" if scan.coils > 1 else None
    images = []
    try:
        if sources.frames is not None:
            for frame in range(sources.frames.num_frames):
                images.append(
                    _drop_coil_axis(
                        sources.frames.reconstruct(
                            scan.result, frame, coil_combine=combine
                        )
                    )
                )
        elif (
            sources.acquisition is not None
            and sources.acquisition.num_samples == scan.signal.shape[-1]
        ):
            images.append(
                _drop_coil_axis(
                    scan.result.reconstruct_cartesian(
                        sources.acquisition, coil_combine=combine
                    )
                )
            )
    except Exception as error:
        scan.warn(f"images could not be reconstructed: {error}")
        return None
    if not images:
        return None
    stack = np.abs(np.stack(images, axis=0))
    rows, columns = stack.shape[1:]
    fov = scan.fov_mm
    from .bruker_pdata import FrameGroup

    groups = (FrameGroup(len(images), "FG_CYCLE"),) if len(images) > 1 else ()
    return PdataFrames(
        values=stack,
        core_size=(int(columns), int(rows)),
        dim_desc=("spatial", "spatial"),
        extent=(float(fov[0]), float(fov[1] if len(fov) > 1 else fov[0])),
        units=("mm", "mm"),
        frame_groups=groups,
        orientation=np.eye(3).reshape(1, 9),
        position_mm=np.asarray(
            [[-fov[0] / 2.0, -(fov[1] if len(fov) > 1 else fov[0]) / 2.0, 0.0]]
        ),
        frame_thickness_mm=scan.slice_thickness_mm,
        slice_distance_mm=scan.slice_thickness_mm,
        description=scan.scan_name,
    )


def _drop_coil_axis(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 3:
        return np.sqrt(np.sum(np.abs(values) ** 2, axis=0))
    return values


# ---------------------------------------------------------------------------
# Remaining scan files


def _write_scan_extras(scan_dir: Path, scan: _Scan, context) -> list[Path]:
    profile = scan.profile
    expno = context.expno
    files: list[Path] = []
    pulseprogram = scan_dir / profile.pulseprogram_file
    pulseprogram.write_text(
        f"; {scan.base_method}.ppg generated by {GENERATOR}\n"
        f"; simulated data, no executable pulse program\n"
        "1 ze\n  aq\nexit\n",
        encoding="latin-1",
    )
    files.append(pulseprogram)
    if "AdjStatePerScan" in profile.scan_files:
        adjust = context.writer(f"{expno}/AdjStatePerScan", "pipe")
        adjust.scalar("AdjScanStateCompleted", False)
        files.append(adjust.write(scan_dir / "AdjStatePerScan"))
    if "specpar" in profile.scan_files:
        specpar = context.writer(f"{expno}/specpar", "parx")
        specpar.scalar("BF1", scan.reference_frequency_mhz)
        specpar.scalar("SFO1", scan.working_frequency_mhz)
        specpar.string("NUC1", scan.nucleus, 8)
        files.append(specpar.write(scan_dir / "specpar"))
    if "configscan" in profile.scan_files:
        config = context.writer(f"{expno}/configscan", "parx")
        config.scalar("CONFIG_SCAN_simulated", True)
        files.append(config.write(scan_dir / "configscan"))
    if "uxnmr.par" in profile.scan_files:
        uxnmr = context.writer(f"{expno}/uxnmr.par", "parx")
        uxnmr.scalar("SW_h", scan.spectral_width_hz)
        uxnmr.scalar("BF1", scan.reference_frequency_mhz)
        uxnmr.scalar("SFO1", scan.working_frequency_mhz)
        files.append(uxnmr.write(scan_dir / "uxnmr.par"))
    if "uxnmr.info" in profile.scan_files:
        info = scan_dir / "uxnmr.info"
        info.write_text(
            f"Simulated by {GENERATOR}; no spectrometer information.\n",
            encoding="latin-1",
        )
        files.append(info)
    return files


def _write_provenance(scan_dir: Path, scan: _Scan, context) -> Path:
    path = scan_dir / "blochsim_provenance.json"
    template = scan.template
    payload = {
        "generator": GENERATOR,
        "created": context.created.isoformat(),
        "paravision_version": scan.profile.key,
        "paravision_experimental": scan.profile.experimental,
        "method": scan.method,
        "raw_layout": scan.layout.kind,
        "raw_data_files": scan.options.raw_data_files,
        "signal_scale": scan.scale,
        "signal_recovery": "complex_int / signal_scale",
        "raw_data_conjugated": bool(scan.options.conjugate_raw_data),
        "source": scan.source,
        "template_source": template.source_path if template is not None else None,
        "anonymized": bool(scan.options.anonymize),
        "notes": list(scan.warnings),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
