"""Export one sequence simulation result to several formats at once.

This is the GUI-free part of the sequence export dialog: a
:class:`SequenceExportRequest` names the output folder, a base name and the
formats, and :func:`perform_sequence_export` writes them.

Output layout for ``directory=D`` and ``base_name=B``::

    D/B.nc, D/B.ipynb, D/B.h5, D/B.npz   result data and analysis notebook
    D/<study folder>/<expno>/             Bruker study (new study)
    <study>/<next expno>/                 Bruker scan appended to a study
    D/B_dicom/S001_image/IM00001.dcm      DICOM series
    D/B_nifti/B_image.nii.gz (+ .json)    NIfTI images and sidecars

DICOM and NIfTI share the subject, study and user metadata of the Bruker
export: when a Bruker study is written in the same run, the DICOM images
join that study (same study UID and number).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

from .bruker_export import (
    BrukerExportOptions,
    BrukerExportResult,
    build_scan_identity,
    export_bruker_dataset,
)
from .bruker_jcamp import (
    BrukerTemplateMetadata,
    anonymize_template,
    load_bruker_template,
)

FORMAT_LABELS = {
    "netcdf": "xarray NetCDF (.nc)",
    "notebook": "Jupyter notebook (.ipynb, uses the NetCDF data)",
    "hdf5": "HDF5 (.h5)",
    "npz": "NumPy archive (.npz)",
    "bruker": "Bruker ParaVision study",
    "dicom": "DICOM (one file per image)",
    "nifti": "NIfTI-1 + JSON sidecar",
}
EXPORT_FORMATS = tuple(FORMAT_LABELS)
IMAGE_FORMATS = ("dicom", "nifti")


@dataclass(frozen=True)
class SequenceExportRequest:
    """What to export, where and with which subject/study metadata.

    ``template`` (a loaded and possibly edited ParaVision dataset) and
    ``anonymize`` apply to Bruker, DICOM and NIfTI alike; ``bruker`` holds
    the remaining Bruker settings.  ``bruker_study_dir`` is the existing
    study for ``bruker.study_layout == "append"``.
    """

    directory: Path
    base_name: str = "sequence_result"
    formats: tuple[str, ...] = ("netcdf", "notebook")
    bruker: BrukerExportOptions = field(default_factory=BrukerExportOptions)
    bruker_study_dir: Optional[Path] = None
    template: Optional[BrukerTemplateMetadata] = None
    anonymize: bool = False
    patient_position: Optional[str] = None
    include_pools: bool = True
    include_phase: bool = True
    overwrite: bool = False

    def __post_init__(self):
        object.__setattr__(self, "directory", Path(self.directory))
        if self.bruker_study_dir is not None:
            object.__setattr__(self, "bruker_study_dir", Path(self.bruker_study_dir))
        unknown = [name for name in self.formats if name not in FORMAT_LABELS]
        if unknown:
            raise ValueError(f"unknown export format(s): {', '.join(unknown)}")
        if not str(self.base_name).strip():
            raise ValueError("the export base name must not be empty")

    @property
    def appends_bruker_study(self) -> bool:
        return "bruker" in self.formats and self.bruker.study_layout == "append"

    def target_paths(self) -> dict[str, Path]:
        """Planned output path per format (folders for Bruker, DICOM, NIfTI)."""

        base = self.directory / str(self.base_name).strip()
        paths = {
            "netcdf": Path(f"{base}.nc"),
            "notebook": Path(f"{base}.ipynb"),
            "hdf5": Path(f"{base}.h5"),
            "npz": Path(f"{base}.npz"),
            "dicom": Path(f"{base}_dicom"),
            "nifti": Path(f"{base}_nifti"),
            "bruker": (
                self.bruker_study_dir if self.appends_bruker_study else self.directory
            ),
        }
        return {name: paths[name] for name in self.formats}


@dataclass
class SequenceExportReport:
    """Written paths (per format) and warnings of a finished export."""

    directory: Path
    written: list[tuple[str, Path]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    bruker: Optional[BrukerExportResult] = None

    def summary_text(self) -> str:
        lines = [f"Exported to {self.directory}:"]
        for name, path in self.written:
            try:
                shown = path.relative_to(self.directory)
            except ValueError:
                shown = path
            lines.append(f"  {FORMAT_LABELS.get(name, name)}: {shown}")
        if self.warnings:
            lines.append("")
            lines.append("Notes:")
            lines.extend(f"  {text}" for text in self.warnings)
        return "\n".join(lines)


def existing_export_targets(request: SequenceExportRequest) -> list[Path]:
    """Return outputs of ``request`` that already exist and would be replaced."""

    existing = []
    for name, path in request.target_paths().items():
        if name == "bruker":
            continue
        if name == "notebook" and "netcdf" not in request.formats:
            data_path = path.with_suffix(".nc")
            if data_path.exists():
                existing.append(data_path)
        if path.is_dir():
            if any(path.iterdir()):
                existing.append(path)
        elif path.exists():
            existing.append(path)
    return existing


def perform_sequence_export(
    request: SequenceExportRequest,
    result,
    *,
    program=None,
    phantom=None,
    acquisition=None,
    acquisition_frames=None,
) -> SequenceExportReport:
    """Write ``result`` in every format of ``request``."""

    if not request.formats:
        raise ValueError("select at least one export format")
    if not request.overwrite:
        existing = existing_export_targets(request)
        if existing:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"export targets already exist: {names}")
    if request.appends_bruker_study and (
        request.bruker_study_dir is None or not request.bruker_study_dir.is_dir()
    ):
        raise ValueError("select an existing Bruker study folder to append to")

    wants_images = any(name in request.formats for name in IMAGE_FORMATS)
    series = summary = None
    if wants_images:
        from .dicom_export import require_pydicom
        from .image_export import acquisition_summary, extract_image_series
        from .nifti_export import require_nibabel

        if "dicom" in request.formats:
            require_pydicom()
        if "nifti" in request.formats:
            require_nibabel()
        series = extract_image_series(
            result,
            include_pools=request.include_pools,
            include_phase=request.include_phase,
        )
        if not series:
            raise ValueError(
                "DICOM and NIfTI need reconstructed images; this simulation has no "
                "Cartesian, spiral, radial or CSI acquisition"
            )
        summary = acquisition_summary(result)

    directory = request.directory
    directory.mkdir(parents=True, exist_ok=True)
    report = SequenceExportReport(directory=directory)
    targets = request.target_paths()

    if "netcdf" in request.formats or "notebook" in request.formats:
        data_path = targets.get("netcdf") or targets["notebook"].with_suffix(".nc")
        result.save(data_path)
        report.written.append(("netcdf", data_path))
        if "notebook" in request.formats:
            from ..notebook_exporter import export_sequence_result_notebook

            export_sequence_result_notebook(str(targets["notebook"]), str(data_path))
            report.written.append(("notebook", targets["notebook"]))
    for name in ("hdf5", "npz"):
        if name in request.formats:
            result.save(targets[name])
            report.written.append((name, targets[name]))

    created = datetime.now().astimezone()
    template = request.template
    anonymize_images = request.anonymize
    new_study = True
    if "bruker" in request.formats:
        options = replace(
            request.bruker,
            template=template,
            anonymize=request.anonymize,
            patient_position=request.patient_position
            or request.bruker.patient_position,
            include_pools=request.include_pools,
        )
        bruker = export_bruker_dataset(
            result,
            targets["bruker"],
            program=program,
            phantom=phantom,
            acquisition=acquisition,
            acquisition_frames=acquisition_frames,
            options=options,
        )
        report.bruker = bruker
        report.written.append(("bruker", bruker.scan_dir))
        report.warnings.extend(f"Bruker: {text}" for text in bruker.warnings)
        try:
            template = load_bruker_template(bruker.study_dir)
            anonymize_images = False
            new_study = False
        except (OSError, ValueError) as error:
            report.warnings.append(
                f"Bruker study metadata could not be reused: {error}"
            )
    elif template is not None and request.anonymize:
        template = anonymize_template(template)
        anonymize_images = False

    if wants_images:
        identity = build_scan_identity(
            template,
            patient_position=request.patient_position,
            anonymize=anonymize_images,
            created=created,
            new_study=new_study,
        )
        if "dicom" in request.formats:
            from .dicom_export import DicomExportOptions, export_dicom

            dicom = export_dicom(
                None,
                targets["dicom"],
                series=series,
                summary=summary,
                identity=identity,
                created=created,
                options=DicomExportOptions(
                    template=template,
                    anonymize=request.anonymize,
                    include_pools=request.include_pools,
                    include_phase=request.include_phase,
                    overwrite=request.overwrite,
                ),
            )
            report.written.append(("dicom", dicom.directory))
            report.warnings.extend(f"DICOM: {text}" for text in dicom.warnings)
        if "nifti" in request.formats:
            from .nifti_export import NiftiExportOptions, export_nifti

            nifti = export_nifti(
                None,
                targets["nifti"],
                series=series,
                summary=summary,
                identity=identity if template is not None else None,
                options=NiftiExportOptions(
                    base_name=str(request.base_name).strip(),
                    include_pools=request.include_pools,
                    include_phase=request.include_phase,
                    overwrite=request.overwrite,
                ),
            )
            report.written.append(("nifti", nifti.directory))
            report.warnings.extend(f"NIfTI: {text}" for text in nifti.warnings)
    return report
