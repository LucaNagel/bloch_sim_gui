"""Processed-data (``pdata``) writers of the Bruker dataset exporter.

A ParaVision scan folder holds the raw data and acquisition parameters
(``fid``, ``acqp``, ``method``) and one ``pdata/<procno>`` folder per
reconstruction.  Each reconstruction folder has the image values (``2dseq``)
and their description (``visu_pars``, ``reco``, ``procs``, ``id``,
``methreco``).  The parameter order follows measured ParaVision 7.0.0
datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .bruker_jcamp import JcampEnum, JcampWriter
from .paravision import ParaVisionProfile

INT16_LIMIT = 32767.0


@dataclass(frozen=True)
class BrukerFileContext:
    """Where and by whom the exported dataset was "acquired"."""

    profile: ParaVisionProfile
    owner: str
    user_host: str
    data_root: str
    study_folder: str
    expno: int
    created: datetime

    def dataset_path(self, relative: str = "") -> str:
        parts = [self.data_root.rstrip("/"), self.study_folder]
        if relative:
            parts.append(relative.strip("/"))
        return "/".join(parts)

    def writer(
        self, relative: str, kind: str = "parx", footer: bool = True
    ) -> JcampWriter:
        """Return a writer with the ParaVision header and footer comments.

        ``relative`` is the path of the file inside the study folder (for
        example ``"17/pdata/1/visu_pars"``).  ``kind`` selects the
        ``$$ process`` line (see :meth:`ParaVisionProfile.process_path`);
        ``"none"`` omits it.  ParaVision 5.1 files name only the owner and
        have neither a process line nor a footer.
        """

        profile = self.profile
        timestamp = profile.format_timestamp(self.created)
        comments = []
        options = profile.write_options(relative)
        if options:
            comments.append(f"Write Options: {options}")
        if profile.comment_style == "pv5":
            comments += [f"{timestamp}  {self.owner}", self.dataset_path(relative)]
            footer = False
        else:
            comments += [f"{timestamp}  {self.user_host}", self.dataset_path(relative)]
            if kind != "none":
                comments.append(f"process {profile.process_path(kind)}")
        return JcampWriter(
            title=profile.title,
            origin=profile.origin,
            owner=self.owner,
            comments=comments,
            footer=f"File finished by PARX at {timestamp}" if footer else None,
            omit=profile.omitted_parameters,
        )


@dataclass(frozen=True)
class ScanIdentity:
    """Scanner, subject and study identity written into every file."""

    owner: str
    user_host: str
    operator: str
    institution: str
    station: str
    system_order_number: str
    manufacturer: str
    subject_id: str
    subject_name: str
    subject_uid: str
    subject_type: str
    subject_sex: str
    subject_birth_date: str
    subject_weight_kg: float
    subject_created: datetime
    subject_entry: str
    subject_position: str
    study_name: str
    study_uid: str
    study_number: int
    study_date: datetime
    referring_physician: str
    coil_receive_name: str
    coil_transmit_name: str
    hardware_parameters: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    @property
    def patient_position(self) -> str:
        entry = "Head" if "head" in self.subject_entry.lower() else "Foot"
        position = self.subject_position.replace("SUBJ_POS_", "") or "Supine"
        return f"{entry}_{position}"

    @property
    def visu_sex(self) -> str:
        text = self.subject_sex.strip().lower()
        if text in {"m", "male"}:
            return "MALE"
        if text in {"f", "female"}:
            return "FEMALE"
        return "UNDEFINED"


@dataclass(frozen=True)
class ProtocolInfo:
    """Acquisition parameters repeated in ``visu_pars``."""

    method: str
    protocol: str
    scan_name: str
    acquired: datetime
    repetition_time_ms: float
    echo_times_ms: tuple[float, ...]
    flip_angle_deg: float
    frequency_mhz: float
    nucleus: str
    phase_encoding_steps: int
    acquisition_size: tuple[int, ...]
    pixel_bandwidth_hz: float
    scan_time_ms: float
    averages: int = 1
    echo_train_length: int = 1
    grad_encoding: tuple[str, ...] = ("read_enc", "phase_enc")
    is_epi: bool = False
    echo_sequence_type: str = "GradientEcho"
    trajectory: str = "RectilinearTraversal"


@dataclass(frozen=True)
class FrameGroup:
    count: int
    name: str
    dependent: tuple[str, ...] = ()


@dataclass(frozen=True)
class PdataFrames:
    """Real-valued frames of one reconstruction plus their geometry.

    ``values`` has shape ``(frames, *reversed(core_size))`` so that a C-order
    write puts the first ``core_size`` axis fastest, as in ``2dseq``.
    ``orientation`` (``(n, 9)``) and ``position_mm`` (``(n, 3)``) hold one
    entry per slice when a ``FG_SLICE`` group carries them, else one entry.
    """

    values: np.ndarray
    core_size: tuple[int, ...]
    dim_desc: tuple[str, ...]
    extent: tuple[float, ...]
    units: tuple[str, ...]
    frame_groups: tuple[FrameGroup, ...]
    orientation: np.ndarray
    position_mm: np.ndarray
    frame_thickness_mm: float
    slice_distance_mm: float
    description: str
    frame_type: str = "MAGNITUDE_IMAGE"
    echo_times_ms: tuple[float, ...] = ()
    spectral_frequency_mhz: Optional[float] = None
    spectral_offset_ppm: Optional[float] = None

    @property
    def frame_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def slice_count(self) -> int:
        for group in self.frame_groups:
            if group.name == "FG_SLICE":
                return group.count
        return 1


def scale_to_int16(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(stored, slope)`` with ``values ≈ stored * slope``."""

    data = np.nan_to_num(np.asarray(values, dtype=float))
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    slope = peak / INT16_LIMIT if peak > 0 else 1.0
    stored = np.clip(np.rint(data / slope), -INT16_LIMIT, INT16_LIMIT)
    return stored.astype("<i2"), slope


def frames_from_image_series(series, *, description: str = "") -> PdataFrames:
    """Convert an :class:`~.image_export.ImageSeries` to ``PdataFrames``."""

    stack = series.magnitude_stack()
    rows, columns = stack.shape[1:]
    row_spacing_mm, column_spacing_mm = (
        value * 1e3 for value in series.pixel_spacing_m
    )
    orientation = series.orientation_matrix.reshape(1, 9)
    row_direction = np.asarray(series.row_direction, dtype=float)
    column_direction = np.asarray(series.column_direction, dtype=float)
    positions = {}
    for frame in series.frames:
        corner = (
            np.asarray(frame.position_m, dtype=float) * 1e3
            - 0.5 * column_spacing_mm * row_direction
            - 0.5 * row_spacing_mm * column_direction
        )
        positions.setdefault(frame.slice_index, corner)
    slice_keys = sorted(positions)
    groups = [
        FrameGroup(
            len(slice_keys), "FG_SLICE", ("VisuCoreOrientation", "VisuCorePosition")
        )
    ]
    echo_times = []
    if series.echo_count > 1:
        groups.append(FrameGroup(series.echo_count, "FG_ECHO", ("VisuAcqEchoTime",)))
        by_echo = {}
        for frame in series.frames:
            if frame.echo_time_s is not None:
                by_echo.setdefault(frame.echo_index, frame.echo_time_s * 1e3)
        echo_times = [by_echo.get(index, 0.0) for index in sorted(by_echo)]
    if series.repetition_count > 1:
        groups.append(FrameGroup(series.repetition_count, "FG_CYCLE"))
    spacing = series.slice_spacing_m
    thickness_mm = series.slice_thickness_m * 1e3
    return PdataFrames(
        values=stack,
        core_size=(int(columns), int(rows)),
        dim_desc=("spatial", "spatial"),
        extent=(columns * column_spacing_mm, rows * row_spacing_mm),
        units=("mm", "mm"),
        frame_groups=tuple(groups),
        orientation=np.repeat(orientation, len(slice_keys), axis=0),
        position_mm=np.asarray([positions[key] for key in slice_keys]),
        frame_thickness_mm=thickness_mm,
        slice_distance_mm=spacing * 1e3 if spacing else thickness_mm,
        description=description or series.description or series.name,
        echo_times_ms=tuple(echo_times),
    )


def frames_from_spectra(
    spectra: np.ndarray,
    *,
    spectral_width_ppm: float,
    frequency_mhz: float,
    offset_ppm: float,
    fov_mm: Sequence[float] = (),
    thickness_mm: float = 1.0,
    description: str = "Spectra",
) -> PdataFrames:
    """Wrap magnitude spectra ``(frames, [ny, nx,] points)`` as ``PdataFrames``.

    The spectral axis runs fastest (``VisuCoreSize[0]``), followed by the
    spatial CSI axes when ``fov_mm`` is given.
    """

    data = np.abs(np.asarray(spectra))
    points = int(data.shape[-1])
    if fov_mm:
        ny, nx = data.shape[-3], data.shape[-2]
        core = (points, int(nx), int(ny))
        desc = ("spectroscopic", "spatial", "spatial")
        extent = (float(spectral_width_ppm), float(fov_mm[0]), float(fov_mm[1]))
        units = ("ppm", "mm", "mm")
        # 2dseq order: spectral fastest, then x, then y → (frames, y, x, spec).
        values = data.reshape((-1, ny, nx, points))
        position = np.asarray([[-fov_mm[0] / 2.0, -fov_mm[1] / 2.0, 0.0]])
    else:
        core = (points,)
        desc = ("spectroscopic",)
        extent = (float(spectral_width_ppm),)
        units = ("ppm",)
        values = data.reshape((-1, points))
        position = np.zeros((1, 3))
    frames = int(values.shape[0])
    groups = (FrameGroup(frames, "FG_CYCLE"),) if frames > 1 else ()
    return PdataFrames(
        values=values,
        core_size=core,
        dim_desc=desc,
        extent=extent,
        units=units,
        frame_groups=groups,
        orientation=np.eye(3).reshape(1, 9),
        position_mm=position,
        frame_thickness_mm=float(thickness_mm),
        slice_distance_mm=float(thickness_mm),
        description=description,
        spectral_frequency_mhz=float(frequency_mhz),
        spectral_offset_ppm=float(offset_ppm),
    )


def _abs_time(moment: datetime) -> tuple[int, int, int]:
    offset = moment.utcoffset()
    minutes = int(offset.total_seconds() // 60) if offset is not None else 0
    return (int(moment.timestamp()), moment.microsecond // 1000, minutes)


def _char_array(writer: JcampWriter, profile, name: str, text: str, size: int):
    if profile.software_version_char_array:
        writer.string(name, text, size)
    else:
        writer.date(name, text)


def write_date(
    writer: JcampWriter,
    profile: ParaVisionProfile,
    name: str,
    moment: datetime,
    size: int = 0,
) -> None:
    """Write a date: ``<...>`` from PV6 on, a ``( size )`` char array in PV5."""

    text = profile.format_date(moment)
    if profile.classic_values:
        writer.string(name, text, size)
    else:
        writer.date(name, text)


def write_abs_time(
    writer: JcampWriter, profile: ParaVisionProfile, name: str, moment: datetime
) -> None:
    """Write epoch seconds; PV6 and later add milliseconds and the UTC offset."""

    if profile.classic_values:
        writer.scalar(name, int(moment.timestamp()))
    else:
        writer.struct(name, _abs_time(moment))


def write_label(
    writer: JcampWriter, profile: ParaVisionProfile, name: str, text: str
) -> None:
    """Write an enum that PV6 and later put in angle brackets (``<1H>``)."""

    if profile.classic_values:
        writer.enum(name, text)
    else:
        writer.date(name, text)


def write_visu_pars(
    path: Path,
    context: BrukerFileContext,
    relative: str,
    identity: ScanIdentity,
    protocol: ProtocolInfo,
    frames: PdataFrames,
    *,
    stored: np.ndarray,
    slope: float,
    procno: int,
    instance_uid: str,
    scan_level: bool = False,
) -> Path:
    profile = context.profile
    writer = context.writer(relative, "acq" if scan_level else "parx")
    frame_count = frames.frame_count
    flat = stored.reshape(frame_count, -1)
    writer.scalar("VisuVersion", profile.visu_version)
    writer.string("VisuUid", instance_uid, 65)
    writer.string("VisuCreator", "ParaVision", 65)
    writer.string(
        "VisuCreatorVersion",
        profile.visu_creator_version,
        profile.visu_creator_version_size,
    )
    write_date(writer, profile, "VisuCreationDate", context.created)
    writer.scalar("VisuInstanceType", "STANDARD_INSTANCE")
    writer.string("VisuInstanceModality", "MR", 65)
    writer.scalar("VisuCoreFrameCount", frame_count)
    writer.scalar("VisuCoreDim", len(frames.core_size))
    writer.array("VisuCoreSize", frames.core_size)
    writer.enums("VisuCoreDimDesc", frames.dim_desc)
    writer.array("VisuCoreExtent", frames.extent)
    writer.array("VisuCoreFrameThickness", [frames.frame_thickness_mm])
    writer.strings("VisuCoreUnits", frames.units, 65)
    writer.array(
        "VisuCoreOrientation", frames.orientation, shape=frames.orientation.shape
    )
    writer.array("VisuCorePosition", frames.position_mm, shape=frames.position_mm.shape)
    writer.array("VisuCoreDataMin", flat.min(axis=1).astype(float))
    writer.array("VisuCoreDataMax", flat.max(axis=1).astype(float))
    writer.array("VisuCoreDataOffs", np.zeros(frame_count))
    writer.array("VisuCoreDataSlope", np.full(frame_count, slope))
    if profile.classic_values:
        writer.scalar("VisuCoreFrameType", frames.frame_type)
    else:
        writer.enums("VisuCoreFrameType", [frames.frame_type])
    if frames.slice_count > 1 or any(g.name == "FG_SLICE" for g in frames.frame_groups):
        writer.struct("VisuCoreSlicePacksDef", (0, 1))
        writer.structs("VisuCoreSlicePacksSlices", [(0, frames.slice_count)])
        writer.array("VisuCoreSlicePacksSliceDist", [frames.slice_distance_mm])
    writer.scalar("VisuCoreWordType", "_16BIT_SGN_INT")
    writer.scalar("VisuCoreByteOrder", "littleEndian")
    if frames.spectral_frequency_mhz is not None:
        writer.array("VisuCoreAxisFrequency", [frames.spectral_frequency_mhz])
    if frames.frame_groups:
        descriptions = []
        dependents = []
        for group in frames.frame_groups:
            start = len(dependents) if group.dependent else 0
            descriptions.append(
                (group.count, group.name, "", start, len(group.dependent))
            )
            dependents.extend((name, 0) for name in group.dependent)
        writer.scalar("VisuFGOrderDescDim", len(descriptions))
        writer.structs("VisuFGOrderDesc", descriptions)
        if dependents:
            writer.structs("VisuGroupDepVals", dependents)
    writer.string("VisuSubjectName", identity.subject_name, 65)
    writer.string("VisuSubjectId", identity.subject_id, 65)
    writer.string("VisuSubjectUid", identity.subject_uid, 65)
    writer.struct(
        "VisuSubjectInstanceCreationDate", _abs_time(identity.subject_created)
    )
    if identity.subject_birth_date:
        writer.string("VisuSubjectBirthDate", identity.subject_birth_date, 9)
    writer.scalar("VisuSubjectSex", identity.visu_sex)
    writer.scalar("VisuSubjectType", identity.subject_type or "Other")
    writer.string("VisuStudyUid", identity.study_uid, 65)
    write_date(writer, profile, "VisuStudyDate", identity.study_date)
    writer.string("VisuStudyId", identity.study_name, 65)
    writer.scalar("VisuStudyNumber", identity.study_number)
    writer.scalar("VisuSubjectWeight", identity.subject_weight_kg)
    writer.string("VisuStudyReferringPhysician", identity.referring_physician, 65)
    description_size = 65 if profile.classic_values else 2048
    writer.string("VisuStudyDescription", identity.study_name, description_size)
    writer.scalar("VisuExperimentNumber", context.expno)
    writer.scalar("VisuProcessingNumber", procno)
    write_date(writer, profile, "VisuSeriesDate", protocol.acquired)
    writer.scalar("VisuSubjectPosition", identity.patient_position)
    writer.string("VisuSeriesTypeId", "ACQ_BRUKER_PVM", 65)
    writer.string("VisuSeriesComment", frames.description or protocol.scan_name, 65)
    for prefix, name in (
        ("VisuCoilReceive", identity.coil_receive_name),
        ("VisuCoilTransmit", identity.coil_transmit_name),
    ):
        writer.string(f"{prefix}Name", name, 65)
        writer.string(f"{prefix}Manufacturer", identity.manufacturer, 65)
        writer.scalar(f"{prefix}Type", "VOLUME_COIL")
        writer.scalar(f"{prefix}IsQuadrature", True)
    writer.string("VisuManufacturer", identity.manufacturer, 65)
    writer.string("VisuAcqSoftwareVersion", profile.visu_acq_software_version, 65)
    writer.string("VisuInstitution", identity.institution, 65)
    writer.string("VisuStation", identity.station, 65)
    writer.string("VisuSystemOrderNumber", identity.system_order_number, 65)
    write_date(writer, profile, "VisuAcqDate", protocol.acquired)
    writer.scalar("VisuAcqEchoTrainLength", protocol.echo_train_length)
    writer.string("VisuAcqSequenceName", protocol.method, 40)
    writer.scalar("VisuAcqNumberOfAverages", protocol.averages)
    writer.scalar("VisuAcqImagingFrequency", protocol.frequency_mhz)
    writer.string("VisuAcqImagedNucleus", protocol.nucleus, 8)
    writer.array("VisuAcqRepetitionTime", [protocol.repetition_time_ms])
    writer.scalar("VisuAcqPhaseEncSteps", protocol.phase_encoding_steps)
    writer.scalar("VisuAcqPixelBandwidth", protocol.pixel_bandwidth_hz)
    writer.scalar("VisuAcqFlipAngle", protocol.flip_angle_deg)
    writer.array("VisuAcqSize", protocol.acquisition_size)
    writer.scalar("VisuAcqImageSizeAccellerated", False)
    writer.enums("VisuAcqGradEncoding", protocol.grad_encoding)
    echo_times = frames.echo_times_ms or protocol.echo_times_ms or (0.0,)
    writer.array("VisuAcqEchoTime", echo_times)
    writer.string("VisuAcquisitionProtocol", protocol.protocol, 65)
    writer.scalar("VisuAcqScanTime", protocol.scan_time_ms)
    writer.array("VisuAcqAntiAlias", [1] * len(protocol.grad_encoding))
    writer.scalar("VisuAcqEchoSequenceType", protocol.echo_sequence_type)
    writer.scalar("VisuAcqSpinsVelocityEncoded", False)
    writer.scalar("VisuAcqHasTimeOfFlightContrast", False)
    writer.scalar("VisuAcqIsEpiSequence", protocol.is_epi)
    writer.scalar("VisuAcqSpectralSuppression", "NoSuppression")
    writer.scalar("VisuAcqKSpaceTraversal", protocol.trajectory)
    writer.enums(
        "VisuAcqEncodingOrder", ["LinearEncoding"] * len(protocol.grad_encoding)
    )
    writer.scalar("VisuAcqKSpaceTrajectoryCnt", 1)
    writer.enums("VisuAcqKSpaceFiltering", ["NO_WDW"] * len(protocol.grad_encoding))
    writer.scalar("VisuAcqFlowCompensation", "FlowNone")
    writer.scalar("VisuAcqSpoiling", "GradientSpoiled")
    writer.array("VisuAcqPartialFourier", [1] * len(protocol.grad_encoding))
    writer.array("VisuAcqParallelReductionFactor", [1] * len(protocol.grad_encoding))
    writer.scalar("VisuAcqMagnetizationTransfer", False)
    writer.scalar("VisuAcqBloodSignalNulling", False)
    writer.scalar("VisuCardiacSynchUsed", False)
    writer.scalar("VisuRespSynchUsed", False)
    return writer.write(path)


def _write_reco(
    path: Path,
    context: BrukerFileContext,
    relative: str,
    frames: PdataFrames,
    *,
    stored: np.ndarray,
    slope: float,
    coils: int,
    conjugated: bool,
    instance_uid: str,
) -> Path:
    writer = context.writer(relative, "parx")
    dims = len(frames.core_size)
    frame_count = frames.frame_count
    flat = stored.reshape(frame_count, -1)
    objects = frames.slice_count
    reps = max(1, frame_count // max(1, objects))
    fov_cm = [
        value / 10.0 if unit == "mm" else value
        for value, unit in zip(frames.extent, frames.units)
    ]
    writer.scalar("RECO_mode", "USER_MODE")
    writer.scalar("RecoAutoDerive", False)
    writer.scalar("RecoDim", dims)
    writer.scalar("RecoObjectsPerRepetition", objects)
    writer.scalar("RecoObjectsPerSetupStep", objects)
    writer.scalar("RecoNumRepetitions", reps)
    writer.scalar("RecoNumOutputChan", 1)
    writer.scalar("RecoRepetitionGroupSize", 1)
    writer.scalar("RECO_inp_order", "NO_REORDERING")
    writer.array("RECO_inp_size", frames.core_size)
    writer.array("RECO_ft_size", frames.core_size)
    writer.array("RECO_fov", fov_cm)
    writer.array("RECO_size", frames.core_size)
    writer.array("RECO_offset", np.zeros((dims, objects), dtype=int))
    writer.scalar("RECO_regrid_mode", "NO_REGRID")
    writer.enums("RECO_bc_mode", ["NO_BC"] * dims)
    qopts = [
        "COMPLEX_CONJUGATE" if conjugated and axis == 0 else "NO_QOPTS"
        for axis in range(dims)
    ]
    writer.enums("RECO_qopts", qopts)
    writer.enums("RECO_wdw_mode", ["NO_WDW"] * dims)
    writer.enums("RECO_ft_mode", ["COMPLEX_FT"] * dims)
    writer.scalar("RECO_ppc_mode", "NO_PPC")
    writer.scalar("RECO_dc_elim", False)
    writer.array("RECO_transposition", np.zeros(objects, dtype=int))
    writer.scalar("RECO_image_type", frames.frame_type)
    writer.scalar("RECO_wordtype", "_16BIT_SGN_INT")
    writer.scalar("RECO_map_mode", "ABSOLUTE_MAPPING")
    writer.array("RECO_map_range", [0, 0])
    writer.array("RECO_map_percentile", [50, 50])
    writer.scalar("RECO_map_error", 0.0025)
    writer.array("RECO_minima", flat.min(axis=1).astype(float) * slope)
    writer.array("RECO_maxima", flat.max(axis=1).astype(float) * slope)
    writer.array("RECO_map_min", flat.min(axis=1).astype(float) * slope)
    writer.array("RECO_map_max", flat.max(axis=1).astype(float) * slope)
    writer.array("RECO_map_offset", np.zeros(frame_count))
    writer.array("RECO_map_slope", np.full(frame_count, 1.0 / slope))
    writer.scalar("RECO_byte_order", "littleEndian")
    write_date(writer, context.profile, "RECO_time", context.created, 24)
    write_abs_time(writer, context.profile, "RECO_abs_time", context.created)
    writer.string("RECO_base_image_uid", instance_uid, 65)
    writer.scalar("RecoUserUpdate", False)
    writer.scalar("RECO_map_user_slope", 1)
    writer.scalar("RECO_map_user_offset", 0)
    writer.scalar("RecoNumInputChan", coils)
    writer.array("RecoScaleChan", np.ones(coils))
    writer.array("RecoPhaseChan", np.zeros(coils))
    writer.scalar("RecoCombineMode", "SumOfSquares" if coils > 1 else "AddImages")
    writer.scalar("GS_reco_display", True)
    writer.scalar("GO_reco_display", True)
    return writer.write(path)


def write_pdata(
    procdir: Path,
    context: BrukerFileContext,
    identity: ScanIdentity,
    protocol: ProtocolInfo,
    frames: PdataFrames,
    *,
    procno: int,
    coils: int = 1,
    conjugated: bool = False,
    instance_uid: str,
) -> tuple[Path, ...]:
    """Write ``2dseq`` and the parameter files of one reconstruction."""

    procdir = Path(procdir)
    procdir.mkdir(parents=True, exist_ok=True)
    relative = f"{context.expno}/pdata/{procno}"
    stored, slope = scale_to_int16(frames.values)
    written = []
    seq_path = procdir / "2dseq"
    stored.tofile(seq_path)
    written.append(seq_path)
    written.append(
        write_visu_pars(
            procdir / "visu_pars",
            context,
            f"{relative}/visu_pars",
            identity,
            protocol,
            frames,
            stored=stored,
            slope=slope,
            procno=procno,
            instance_uid=instance_uid,
        )
    )
    written.append(
        _write_reco(
            procdir / "reco",
            context,
            f"{relative}/reco",
            frames,
            stored=stored,
            slope=slope,
            coils=coils,
            conjugated=conjugated,
            instance_uid=instance_uid,
        )
    )
    files = context.profile.pdata_files
    if "procs" in files:
        procs = context.writer(f"{relative}/procs", "parx", footer=False)
        values = stored.astype(float) * slope
        if frames.spectral_frequency_mhz is not None:
            width = float(frames.extent[0])
            offset = float(frames.spectral_offset_ppm or 0.0)
            procs.scalar("OFFSET", offset + width / 2.0)
            procs.scalar("SF", frames.spectral_frequency_mhz)
            procs.scalar("SW_p", width * frames.spectral_frequency_mhz)
            procs.scalar("SI", frames.core_size[0])
        else:
            procs.scalar("OFFSET", 0)
            procs.scalar("SF", protocol.frequency_mhz)
        procs.scalar("YMAX_p", float(values.max()) if values.size else 0.0)
        procs.scalar("YMIN_p", float(values.min()) if values.size else 0.0)
        written.append(procs.write(procdir / "procs"))
    identifier = context.writer(f"{relative}/id", "parx")
    identifier.string("DATASET_KEY", instance_uid, 65)
    written.append(identifier.write(procdir / "id"))
    if "methreco" in files:
        methreco = context.writer(f"{relative}/methreco", "acq")
        methreco.scalar("RecoMethMode", "Default")
        methreco.scalar("WeightingMode", "positive_mask")
        methreco.scalar("MaskWeighting", 4)
        methreco.scalar("GaussBroadening", 1)
        methreco.scalar("PVM_GradientLinearityCorrection", False)
        written.append(methreco.write(procdir / "methreco"))
    if "d3proc" in files:
        d3proc = context.writer(f"{relative}/d3proc", "parx")
        sizes = list(frames.core_size) + [1, 1]
        d3proc.scalar("DATTYPE", JcampEnum("ip_short"))
        d3proc.scalar("IM_SIX", sizes[0])
        d3proc.scalar("IM_SIY", sizes[1])
        d3proc.scalar("IM_SIZ", frames.frame_count)
        d3proc.scalar("IM_SIT", 1)
        written.append(d3proc.write(procdir / "d3proc"))
    return tuple(written)
