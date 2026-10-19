"""ParaVision version profiles for the Bruker dataset exporter.

Every ParaVision release writes the same JCAMP-DX parameter files with small
differences: the ``##TITLE`` line, the software-version strings, which raw
data files exist (``fid`` or ``rawdata.job0``), the ``ACQ_jobs`` layout and
the subject parameter names.  A :class:`ParaVisionProfile` collects those
differences so the exporter can write one dataset layout per version.

The ParaVision 7.0.0 and 6.0.1 profiles are checked against measured
datasets.  The ParaVision 5.1 profile is checked against the public
BrukerAPI example study (Zenodo 3899268) and the ParaVision 360 V3.6
profile against the public ``PV360_StdData`` scans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class ParaVisionProfile:
    """Per-version layout of a ParaVision dataset."""

    key: str
    label: str
    family: str
    title: str
    origin: str
    software_version: str
    software_version_char_array: bool
    visu_version: int
    visu_creator_version: str
    visu_creator_version_size: int
    visu_acq_software_version: str
    install_root: str
    data_root: str
    namespaced_methods: bool
    raw_data_files: tuple[str, ...]
    default_raw_data_files: str
    job_style: str
    go_parameters: bool
    go_block_size: str
    pdata_files: tuple[str, ...]
    scan_files: tuple[str, ...]
    subject_style: str
    subject_version_nr: int
    name_limit: int
    write_options_comment: bool
    iso_dates: bool
    acq_process: str
    dicom_naming: str
    scan_visu_pars: bool = True
    pulseprogram_file: str = "pulseprogram"
    comment_style: str = "parx"
    pipe_process: str = "prog/mod/pipe"
    classic_values: bool = False
    omitted_parameters: FrozenSet[str] = frozenset()
    experimental: bool = False

    def data_root_for(self, owner: str) -> str:
        return self.data_root.format(owner=owner or "nmrsu")

    def process_path(self, kind: str) -> str:
        """Return the ``$$ process`` path of a parameter file.

        ``kind`` is ``"acq"`` for acqp/method/methreco and the scan-level
        visu_pars, ``"pipe"`` for adjustment states and ``"parx"`` for all
        other files.
        """

        program = {
            "acq": self.acq_process,
            "pipe": self.pipe_process,
        }.get(kind, "prog/bin/parxserver")
        return f"{self.install_root}/{program}"

    def write_options(self, relative: str) -> Optional[str]:
        """Return the ``$$ Write Options`` comment of a ParaVision 360 file.

        The exporter writes no ``$$ @vis`` lines, so the ``symbol visibility``
        option of the scanner's acqp/method/visu_pars files is left out.
        """

        if not self.write_options_comment:
            return None
        if relative.rsplit("/", 1)[-1] == "id":
            return "Symbolic Enums"
        return "Symbolic Enums, RLE encoded arrays"

    def method_name(self, name: str) -> str:
        """Return a method name in this version's spelling."""

        text = str(name).strip()
        if self.namespaced_methods:
            if ":" not in text:
                return f"Bruker:{text}"
            return text
        return text.split(":", 1)[-1]

    def format_date(self, moment: datetime) -> str:
        """Return a ParaVision date value without the surrounding brackets."""

        if self.iso_dates:
            millis = f"{moment.microsecond // 1000:03d}"
            return moment.strftime(f"%Y-%m-%dT%H:%M:%S,{millis}%z")
        return f"{moment:%H:%M:%S} {moment.day:2d} {moment:%b %Y}"

    def format_timestamp(self, moment: datetime) -> str:
        """Return the timestamp of ``$$`` header and footer comments."""

        if self.comment_style == "pv5":
            offset = moment.utcoffset()
            hours = offset.total_seconds() / 3600.0 if offset is not None else 0.0
            zone = moment.tzname() or "UTC"
            return (
                f"{moment:%a %b} {moment.day:2d} {moment:%H:%M:%S %Y} "
                f"{zone} (UT{hours:+g}h)"
            )
        millis = f"{moment.microsecond // 1000:03d}"
        return moment.strftime(f"%Y-%m-%d %H:%M:%S.{millis} %z")


_PV7 = ParaVisionProfile(
    key="7.0.0",
    label="ParaVision 7.0.0",
    family="pv7",
    title="Parameter List, ParaVision 7.0.0",
    origin="Bruker BioSpin MRI GmbH",
    software_version="PV-7.0.0",
    software_version_char_array=True,
    visu_version=3,
    visu_creator_version="7.0.0",
    visu_creator_version_size=195,
    visu_acq_software_version="PV-7.0.0",
    install_root="/opt/PV-7.0.0",
    data_root="/opt/nmrdata/data",
    namespaced_methods=True,
    raw_data_files=("fid", "rawdata.job0", "both"),
    default_raw_data_files="fid",
    job_style="pv6",
    go_parameters=True,
    go_block_size="Standard_KBlock_Format",
    pdata_files=("2dseq", "reco", "procs", "methreco", "id", "visu_pars"),
    scan_files=(
        "AdjStatePerScan",
        "configscan",
        "uxnmr.info",
        "uxnmr.par",
        "specpar",
        "spnam0",
    ),
    subject_style="classic",
    subject_version_nr=1,
    name_limit=64,
    write_options_comment=False,
    iso_dates=True,
    acq_process="prog/mod/acq_out",
    dicom_naming="pv",
)

_PV6 = ParaVisionProfile(
    key="6.0.1",
    label="ParaVision 6.0.1",
    family="pv6",
    title="Parameter List, ParaVision 6.0.1",
    origin="Bruker BioSpin MRI GmbH",
    software_version="PV 6.0.1",
    software_version_char_array=True,
    visu_version=3,
    visu_creator_version="6.0.1",
    visu_creator_version_size=65,
    visu_acq_software_version="6.0.1",
    install_root="/opt/PV6.0.1",
    data_root="/opt/PV6.0.1/data/{owner}",
    namespaced_methods=True,
    raw_data_files=("fid", "rawdata.job0", "both"),
    default_raw_data_files="fid",
    job_style="pv6",
    go_parameters=True,
    go_block_size="Standard_KBlock_Format",
    pdata_files=("2dseq", "reco", "procs", "methreco", "id", "visu_pars"),
    scan_files=(
        "AdjStatePerScan",
        "configscan",
        "uxnmr.info",
        "uxnmr.par",
        "specpar",
        "spnam0",
    ),
    subject_style="classic",
    subject_version_nr=1,
    name_limit=64,
    write_options_comment=False,
    iso_dates=True,
    acq_process="prog/mod/acq_out",
    dicom_naming="pv",
)

# Parameters that ParaVision 5.1 does not know.  The exporter writes the
# PV6/PV7 parameter set and drops these names for PV5.1.
_PV5_OMITTED = frozenset(
    """
    ACQ_grad_matrix_size ACQ_spatial_phase_0 ACQ_spatial_size_0
    ACQ_system_order_number
    PVM_DummyScans PVM_EffPhase1Offset PVM_EffReadOffset PVM_EffSliceOffset
    PVM_EncCentralStep0 PVM_EncOrder PVM_EncPft PVM_EncPpi PVM_EncSpectroscopy
    PVM_EncStart PVM_EncSteps0 PVM_EncValues0 PVM_EncValues1 PVM_EncZf
    PVM_FrqRef PVM_FrqRefPpm PVM_FrqWork PVM_FrqWorkOffset PVM_FrqWorkOffsetPpm
    PVM_FrqWorkPpm PVM_IsotropicFovRes PVM_Phase1Offset PVM_ReadOffset
    PVM_ScanTime PVM_SliceOffset
    VisuAcqAntiAlias VisuAcqBloodSignalNulling VisuAcqEchoSequenceType
    VisuAcqEncodingOrder VisuAcqFlowCompensation VisuAcqGradEncoding
    VisuAcqHasTimeOfFlightContrast VisuAcqIsEpiSequence VisuAcqKSpaceFiltering
    VisuAcqKSpaceTrajectoryCnt VisuAcqKSpaceTraversal
    VisuAcqMagnetizationTransfer VisuAcqParallelReductionFactor
    VisuAcqPartialFourier VisuAcqSpectralSuppression
    VisuAcqSpinsVelocityEncoded VisuAcqSpoiling VisuCardiacSynchUsed
    VisuCoilReceiveIsQuadrature VisuCoilReceiveManufacturer VisuCoilReceiveName
    VisuCoilReceiveType VisuCoilTransmitIsQuadrature
    VisuCoilTransmitManufacturer VisuCoilTransmitName VisuCoilTransmitType
    VisuCoreAxisFrequency VisuCoreSlicePacksDef VisuCoreSlicePacksSliceDist
    VisuCoreSlicePacksSlices VisuExperimentNumber VisuInstanceModality
    VisuInstanceType VisuManufacturer VisuProcessingNumber VisuRespSynchUsed
    VisuSeriesDate VisuSubjectInstanceCreationDate VisuSubjectType
    VisuSubjectUid VisuSystemOrderNumber
    RECO_map_user_offset RECO_map_user_slope RecoDim RecoNumOutputChan
    RecoNumRepetitions RecoObjectsPerRepetition RecoObjectsPerSetupStep
    RecoPhaseChan RecoRepetitionGroupSize
    SUBJECT_comment SUBJECT_instance_creation_date SUBJECT_purpose
    SUBJECT_referral SUBJECT_sex_animal
    """.split()
)

# Parameters that ParaVision 360 V3.6 no longer writes.
_PV360_OMITTED = frozenset(
    """
    ACQ_Routing ACQ_nr_completed ACQ_pipe_status ACQ_rare_factor ACQ_scan_time
    ACQ_scans_completed ACQ_time ACQ_total_completed AQ_mod DR FW GRDPROG
    INSTRUM PAPS RG SW
    PVM_DigAutSet PVM_DigDur PVM_DigDw PVM_DigFilter PVM_DigNp PVM_DigQuad
    PVM_DigShift PVM_DigShiftDbl PVM_DigSw PVM_ScanTimeStr
    VisuAcqPhaseEncSteps VisuCoreAxisFrequency VisuStudyDescription
    GO_reco_display GS_reco_display RECO_bc_mode RECO_dc_elim RECO_mode
    RECO_regrid_mode
    """.split()
)

_PV5 = ParaVisionProfile(
    key="5.1",
    label="ParaVision 5.1",
    family="pv5",
    title="Parameter List",
    origin="Bruker BioSpin MRI GmbH",
    software_version="PV 5.1",
    software_version_char_array=True,
    visu_version=1,
    visu_creator_version="5.1",
    visu_creator_version_size=65,
    visu_acq_software_version="5.1",
    install_root="/opt/PV5.1",
    data_root="/opt/PV5.1/data/{owner}/nmr",
    namespaced_methods=False,
    raw_data_files=("fid",),
    default_raw_data_files="fid",
    job_style="none",
    go_parameters=True,
    go_block_size="Standard_KBlock_Format",
    pdata_files=("2dseq", "reco", "procs", "d3proc", "id", "visu_pars"),
    scan_files=("AdjStatePerScan", "uxnmr.par", "spnam0"),
    subject_style="pv5",
    subject_version_nr=2,
    name_limit=15,
    write_options_comment=False,
    iso_dates=False,
    acq_process="prog/mod/acq_out",
    dicom_naming="pv",
    scan_visu_pars=False,
    comment_style="pv5",
    classic_values=True,
    omitted_parameters=_PV5_OMITTED,
)

_PV360 = ParaVisionProfile(
    key="360.3.6",
    label="ParaVision 360 V3.6",
    family="pv360",
    title="Parameter List, ParaVision 360 V3.6",
    origin="Bruker BioSpin GmbH & Co. KG",
    software_version="PV-360.3.6",
    software_version_char_array=True,
    visu_version=8,
    visu_creator_version="360.3.6",
    visu_creator_version_size=195,
    visu_acq_software_version="PV-360.3.6",
    install_root="/opt/PV-360.3.6",
    data_root="/opt/nmrdata/PV-360.3.6/data/{owner}",
    namespaced_methods=True,
    raw_data_files=("rawdata.job0",),
    default_raw_data_files="rawdata.job0",
    job_style="pv360",
    go_parameters=False,
    go_block_size="",
    pdata_files=("2dseq", "reco", "methreco", "id", "visu_pars"),
    scan_files=("AdjStatePerScan", "configscan"),
    subject_style="pv360",
    subject_version_nr=4,
    name_limit=64,
    write_options_comment=True,
    iso_dates=True,
    acq_process="prog/bin/parxserver",
    dicom_naming="pv360",
    scan_visu_pars=False,
    pulseprogram_file="pulseprogram.precomp",
    pipe_process="prog/bin/parxserver",
    omitted_parameters=_PV360_OMITTED,
)

PARAVISION_PROFILES: dict[str, ParaVisionProfile] = {
    profile.key: profile for profile in (_PV7, _PV6, _PV5, _PV360)
}
DEFAULT_PARAVISION_VERSION = _PV7.key

_VERSION_ALIASES = {
    "7": "7.0.0",
    "pv7": "7.0.0",
    "7.0": "7.0.0",
    "6": "6.0.1",
    "pv6": "6.0.1",
    "6.0": "6.0.1",
    "5": "5.1",
    "pv5": "5.1",
    "5.1": "5.1",
    "360": "360.3.6",
    "pv360": "360.3.6",
}


def get_paravision_profile(version: Optional[str] = None) -> ParaVisionProfile:
    """Return the profile for ``version`` such as ``"7.0.0"`` or ``"PV360"``.

    Unknown patch releases fall back to the closest supported release of the
    same major version (e.g. ``6.0`` → ``6.0.1``, ``360.3.5`` → ``360.3.6``).
    """

    if version is None or not str(version).strip():
        return PARAVISION_PROFILES[DEFAULT_PARAVISION_VERSION]
    text = str(version).strip().lower()
    text = re.sub(r"^(paravision|pv)[\s-]*", "", text)
    text = text.replace(" v", ".").replace(" ", ".")
    if text in PARAVISION_PROFILES:
        return PARAVISION_PROFILES[text]
    if text in _VERSION_ALIASES:
        return PARAVISION_PROFILES[_VERSION_ALIASES[text]]
    if text.startswith("360"):
        return PARAVISION_PROFILES["360.3.6"]
    major = text.split(".", 1)[0]
    if major in _VERSION_ALIASES:
        return PARAVISION_PROFILES[_VERSION_ALIASES[major]]
    supported = ", ".join(PARAVISION_PROFILES)
    raise ValueError(
        f"unsupported ParaVision version {version!r}; choose one of {supported}"
    )
