"""DICOM, NIfTI and Bruker study exports of sequence simulation results."""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import (
    ADCEvent,
    BrukerExportOptions,
    SequenceCompiler,
    SequenceProgram,
    SequenceSimulationResult,
    build_scan_identity,
    export_bruker_dataset,
    extract_image_series,
    load_pulseq,
)
from blochsimulator.sequence.acquisition import infer_spectroscopic_acquisition
from blochsimulator.sequence.bruker_jcamp import (
    ANONYMIZED_SUBJECT,
    BrukerTemplateMetadata,
    load_bruker_template,
    parse_jcamp_value,
    read_jcamp,
)
from blochsimulator.sequence.export_bundle import (
    EXPORT_FORMATS,
    SequenceExportRequest,
    perform_sequence_export,
)
from blochsimulator.sequence.paravision import PARAVISION_PROFILES
from blochsimulator.sequence.pulseq_builders import make_pulseq_flash

DATA_DIR = Path(__file__).parent / "data"
BRUKER_DATA = Path(__file__).resolve().parents[1] / "examples" / "bruker_data"
PV7_REFERENCE_STUDY = (
    BRUKER_DATA / "2025.11.07 - lin cryo - 2M 13C Urea - B1 mapping, AHP, slice profile"
)
# Public example data: study 0.2H2 of the BrukerAPI test archive (Zenodo record
# 3899268, CC-BY-4.0) and two scans of github.com/cecilyen/PV360_StdData, both
# without their binary fid, rawdata.job0 and 2dseq files.
PV5_REFERENCE_STUDY = BRUKER_DATA / "pv5.1_brukerapi_zenodo3899268" / "0.2H2"
PV5_SCANS = {"flash": "1", "csi": "24", "nspect": "26"}
# fid sizes of these scans in the Zenodo archive.
PV5_REFERENCE_FID_BYTES = {"1": 1966080, "24": 5914624, "26": 24576}
PV360_REFERENCE = BRUKER_DATA / "pv360.3.6_PV360_StdData"
PV360_SCANS = {"flash": "T1_FLASH", "nspect": "PRESS_1H"}
# Parameters the exporter adds to mark simulated data.
MARKER_PARAMETERS = {"VisuSeriesComment", "CONFIG_SCAN_simulated"}
MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"
SUBJECT_NAME = "Mouse^Twelve"


@pytest.fixture(scope="module")
def flash_case(tmp_path_factory):
    """Two-slice, two-repetition FLASH on an 8 x 4 matrix with a textured phantom."""

    directory = tmp_path_factory.mktemp("flash")
    sequence = make_pulseq_flash(
        fov_m=(0.08, 0.06),
        matrix=(8, 4),
        n_slices=2,
        repetitions=2,
        slice_thickness_m=3e-3,
        slice_gap_m=1e-3,
    )
    sequence.write(str(directory / "flash.seq"), v141_compat=True)
    program = load_pulseq(directory / "flash.seq")
    shape = (8, 4, 2)
    phantom = Phantom(
        shape=shape,
        fov=(0.08, 0.06, 0.008),
        t1_map=np.ones(shape),
        t2_map=np.full(shape, 0.1),
        pd_map=np.random.default_rng(3).uniform(0.2, 1.0, shape),
    )
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    result.metadata.update(field_strength_t=7.0)
    return result, program, phantom


@pytest.fixture(scope="module")
def csi_result():
    """Phase-encoded 2D CSI with two spectral lines (0 and 500 Hz)."""

    program = load_pulseq(DATA_DIR / "csi_2d_centric.seq")
    compiled = SequenceCompiler().compile(program)
    acquisition = infer_spectroscopic_acquisition(program, compiled=compiled)
    dwell = 1.0 / acquisition.spectral_bandwidth_hz
    times = (np.arange(acquisition.num_samples) % acquisition.spectral_points) * dwell
    signal = np.exp(0j * times) + 0.5 * np.exp(2j * np.pi * 500.0 * times)
    return SequenceSimulationResult(
        signal=signal,
        adc_times_s=compiled.adc_times_s,
        final_magnetization=np.zeros((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        adc_gradient_moment_cyc_per_m=compiled.adc_gradient_moment_cyc_per_m,
        metadata={
            "spectroscopic_acquisition": acquisition.to_metadata(),
            "pool_frequency_offsets_hz": [0.0, 500.0],
            "field_strength_t": 7.0,
            "nucleus": "13C",
        },
    )


def scanner_template(**changes) -> BrukerTemplateMetadata:
    values = dict(
        paravision_version="7.0.0",
        owner="nmrsu",
        user_host="nmrsu@biospec",
        operator="Operator",
        institution="Test Institute",
        station="BioSpec70",
        system_order_number="SON123",
        manufacturer="Bruker BioSpin MRI GmbH",
        subject_id="M12",
        subject_name=SUBJECT_NAME,
        subject_uid="1.2.826.0.1.3680043.2.1125.1",
        subject_type="Animal",
        subject_sex="female",
        subject_birth_date="2025-01-02",
        subject_weight_kg=0.025,
        study_name="Urea phantom",
        study_uid="1.2.826.0.1.3680043.2.1125.2",
        study_number=3,
        referring_physician="Referrer",
    )
    values.update(changes)
    return BrukerTemplateMetadata(**values)


def read_dicom_series(info):
    pydicom = pytest.importorskip("pydicom")
    return [pydicom.dcmread(path) for path in info.files]


def ras_mm(position_m):
    x, y, z = (1e3 * float(value) for value in position_m)
    return np.array([-x, -y, z])


# --- DICOM ------------------------------------------------------------------


def test_dicom_series_store_geometry_and_pixel_values(flash_case, tmp_path):
    pytest.importorskip("pydicom")
    from blochsimulator.sequence.dicom_export import export_dicom

    result, _, _ = flash_case
    (image,) = extract_image_series(result, include_pools=False, include_phase=True)
    exported = export_dicom(result, tmp_path / "dicom", include_pools=False)

    magnitude, phase = exported.series
    assert magnitude.image_type.endswith("\\M")
    assert phase.image_type.endswith("\\P")
    assert len(magnitude.files) == len(phase.files) == 4  # 2 slices x 2 repetitions

    datasets = read_dicom_series(magnitude)
    sop_uids = set()
    for frame, ds in zip(image.frames, datasets):
        assert ds.SOPClassUID == MR_IMAGE_STORAGE
        assert ds.Modality == "MR"
        assert ds.StudyInstanceUID == exported.study_uid
        assert ds.SeriesInstanceUID == magnitude.series_uid
        assert (ds.Rows, ds.Columns) == frame.magnitude.shape
        np.testing.assert_allclose(
            [float(v) for v in ds.PixelSpacing], np.array(image.pixel_spacing_m) * 1e3
        )
        np.testing.assert_allclose(
            [float(v) for v in ds.ImageOrientationPatient],
            [*image.row_direction, *image.column_direction],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            [float(v) for v in ds.ImagePositionPatient],
            np.array(frame.position_m) * 1e3,
            atol=1e-4,
        )
        slope = float(ds.RescaleSlope)
        np.testing.assert_allclose(
            ds.pixel_array * slope, frame.magnitude, atol=slope, rtol=0
        )
        assert int(ds.AcquisitionNumber) == frame.repetition_index + 1
        assert float(ds.MagneticFieldStrength) == pytest.approx(7.0)
        assert float(ds.SliceThickness) == pytest.approx(image.slice_thickness_m * 1e3)
        sop_uids.add(ds.SOPInstanceUID)
    assert len(sop_uids) == len(datasets)

    phase_ds = read_dicom_series(phase)[0]
    assert phase_ds.PixelRepresentation == 1
    assert phase_ds.ComplexImageComponent == "PHASE"
    values = phase_ds.pixel_array * float(phase_ds.RescaleSlope)
    assert np.all(np.abs(values) <= np.pi + 1e-3)
    assert phase_ds.SeriesInstanceUID != magnitude.series_uid
    assert phase_ds.StudyInstanceUID == exported.study_uid


def test_dicom_uses_template_identity_and_anonymizes(flash_case, tmp_path):
    pytest.importorskip("pydicom")
    from blochsimulator.sequence.dicom_export import export_dicom

    result, _, _ = flash_case
    template = scanner_template()

    named = export_dicom(
        result, tmp_path / "named", template=template, include_phase=False
    )
    ds = read_dicom_series(named.series[0])[0]
    assert str(ds.PatientName) == SUBJECT_NAME
    assert ds.PatientID == "M12"
    assert ds.PatientSex == "F"
    assert ds.PatientBirthDate == "20250102"
    assert ds.InstitutionName == "Test Institute"
    assert ds.StationName == "BioSpec70"
    assert ds.Manufacturer == "Bruker BioSpin MRI GmbH"
    assert ds.StudyID == "4"  # a new study after the template's study 3
    assert ds.StudyInstanceUID != template.study_uid
    assert "PatientIdentityRemoved" not in ds

    anonymous = export_dicom(
        result,
        tmp_path / "anonymous",
        template=template,
        anonymize=True,
        include_phase=False,
    )
    ds = read_dicom_series(anonymous.series[0])[0]
    assert str(ds.PatientName) == ANONYMIZED_SUBJECT
    assert ds.PatientID == ANONYMIZED_SUBJECT
    assert ds.PatientBirthDate == ""
    assert ds.PatientIdentityRemoved == "YES"
    assert ds.InstitutionName == "Test Institute"
    for path in anonymous.series[0].files:
        assert SUBJECT_NAME.encode() not in path.read_bytes()


def test_dicom_csi_maps_are_metabolite_maps(csi_result, tmp_path):
    pytest.importorskip("pydicom")
    from blochsimulator.sequence.dicom_export import export_dicom

    exported = export_dicom(csi_result, tmp_path / "csi")
    maps = [info for info in exported.series if "METABOLITE_MAP" in info.image_type]
    assert maps
    ds = read_dicom_series(maps[0])[0]
    assert ds.ScanningSequence == "RM"
    assert ds.ImagedNucleus == "13C"
    assert float(ds.ImagingFrequency) == pytest.approx(7.0 * 10.7084, rel=1e-3)


def test_dicom_refuses_existing_files_and_drops_stale_images(flash_case, tmp_path):
    pytest.importorskip("pydicom")
    from blochsimulator.sequence.dicom_export import export_dicom

    result, _, _ = flash_case
    first = export_dicom(
        result, tmp_path / "dicom", include_pools=False, include_phase=False
    )
    stale = first.series[0].directory / "IM00099.dcm"
    stale.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        export_dicom(
            result, tmp_path / "dicom", include_pools=False, include_phase=False
        )
    export_dicom(
        result,
        tmp_path / "dicom",
        include_pools=False,
        include_phase=False,
        overwrite=True,
    )
    assert not stale.exists()
    assert len(list(first.series[0].directory.glob("IM*.dcm"))) == 4


def test_missing_optional_modules_name_the_export_extra(monkeypatch):
    from blochsimulator.sequence.dicom_export import require_pydicom
    from blochsimulator.sequence.nifti_export import require_nibabel

    monkeypatch.setitem(sys.modules, "pydicom", None)
    monkeypatch.setitem(sys.modules, "nibabel", None)
    with pytest.raises(ImportError, match=r"blochsimulator\[export\]"):
        require_pydicom()
    with pytest.raises(ImportError, match=r"blochsimulator\[export\]"):
        require_nibabel()


# --- NIfTI ------------------------------------------------------------------


def test_nifti_volume_matches_frames_in_ras_space(flash_case, tmp_path):
    nib = pytest.importorskip("nibabel")
    from blochsimulator.sequence.nifti_export import export_nifti

    result, _, _ = flash_case
    (image,) = extract_image_series(result, include_pools=False, include_phase=False)
    exported = export_nifti(
        result, tmp_path / "nifti", base_name="flash", include_pools=False
    )
    names = sorted(info.image_path.name for info in exported.series)
    assert names == ["flash_image.nii.gz", "flash_image_phase.nii.gz"]

    info = next(item for item in exported.series if item.component == "magnitude")
    volume = nib.load(info.image_path)
    data = volume.get_fdata()
    assert data.shape == (image.columns, image.rows, 2, 2)
    assert nib.aff2axcodes(volume.affine) == ("L", "P", "S")
    zooms = volume.header.get_zooms()
    np.testing.assert_allclose(
        zooms[:3],
        [
            image.pixel_spacing_m[1] * 1e3,
            image.pixel_spacing_m[0] * 1e3,
            image.slice_spacing_m * 1e3,
        ],
        rtol=1e-5,
    )

    inverse = np.linalg.inv(volume.affine)
    for frame in image.frames:
        voxel = inverse @ np.append(ras_mm(frame.position_m), 1.0)
        np.testing.assert_allclose(voxel[:2], 0.0, atol=1e-4)
        slice_index = int(round(voxel[2]))
        assert voxel[2] == pytest.approx(slice_index, abs=1e-4)
        np.testing.assert_allclose(
            data[:, :, slice_index, frame.repetition_index],
            frame.magnitude.T,
            rtol=1e-5,
            atol=1e-6 * float(np.max(frame.magnitude)),
        )

    sidecar = json.loads(info.sidecar_path.read_text())
    assert sidecar["MagneticFieldStrength"] == pytest.approx(7.0)
    assert sidecar["ImagedNucleus"] == "1H"
    for key in ("RepetitionTime", "EchoTime", "FlipAngle", "SliceThickness"):
        assert sidecar[key] > 0


def test_nifti_csi_maps_carry_spectral_metadata(csi_result, tmp_path):
    nib = pytest.importorskip("nibabel")
    from blochsimulator.sequence.nifti_export import export_nifti

    exported = export_nifti(
        csi_result, tmp_path / "csi", base_name="csi", include_phase=False
    )
    assert exported.series
    for info in exported.series:
        assert nib.load(info.image_path).shape[:2] == info.shape[:2]
        sidecar = json.loads(info.sidecar_path.read_text())
        assert sidecar["ImagedNucleus"] == "13C"
        assert sidecar["ScanningSequence"] == "RM"


# --- Export bundle ----------------------------------------------------------


def test_bundle_writes_every_format_into_one_study(flash_case, tmp_path):
    pytest.importorskip("pydicom")
    pytest.importorskip("nibabel")
    result, program, phantom = flash_case
    request = SequenceExportRequest(
        directory=tmp_path,
        base_name="flash",
        formats=EXPORT_FORMATS,
        include_pools=False,
    )
    report = perform_sequence_export(request, result, program=program, phantom=phantom)

    written = dict(report.written)
    assert set(written) == set(EXPORT_FORMATS)
    for name in ("netcdf", "notebook", "hdf5", "npz"):
        assert written[name].is_file()
    assert (written["bruker"] / "acqp").is_file()
    assert (report.bruker.study_dir / "subject").is_file()
    assert list((tmp_path / "flash_nifti").glob("*.nii.gz"))

    study = load_bruker_template(report.bruker.study_dir)
    ds = read_dicom_series_from(tmp_path / "flash_dicom")
    assert ds.StudyInstanceUID == study.study_uid
    assert ds.StudyID == str(study.study_number)

    with pytest.raises(FileExistsError):
        perform_sequence_export(request, result, program=program, phantom=phantom)


def read_dicom_series_from(directory):
    pydicom = pytest.importorskip("pydicom")
    return pydicom.dcmread(sorted(directory.glob("S*/IM*.dcm"))[0])


def test_bundle_appends_scan_and_images_to_existing_study(flash_case, tmp_path):
    pytest.importorskip("pydicom")
    result, program, phantom = flash_case
    first = perform_sequence_export(
        SequenceExportRequest(
            directory=tmp_path, base_name="first", formats=("bruker", "dicom")
        ),
        result,
        program=program,
        phantom=phantom,
    )
    second = perform_sequence_export(
        SequenceExportRequest(
            directory=tmp_path,
            base_name="second",
            formats=("bruker", "dicom"),
            bruker=BrukerExportOptions(study_layout="append"),
            bruker_study_dir=first.bruker.study_dir,
        ),
        result,
        program=program,
        phantom=phantom,
    )
    assert second.bruker.study_dir == first.bruker.study_dir
    assert (first.bruker.expno, second.bruker.expno) == (1, 2)
    assert (first.bruker.study_dir / "2" / "pdata" / "1" / "2dseq").is_file()
    first_ds = read_dicom_series_from(tmp_path / "first_dicom")
    second_ds = read_dicom_series_from(tmp_path / "second_dicom")
    assert first_ds.StudyInstanceUID == second_ds.StudyInstanceUID
    assert first_ds.StudyID == second_ds.StudyID


def test_bundle_anonymizes_template_in_every_format(flash_case, tmp_path):
    pytest.importorskip("pydicom")
    pytest.importorskip("nibabel")
    result, program, phantom = flash_case
    report = perform_sequence_export(
        SequenceExportRequest(
            directory=tmp_path,
            base_name="anon",
            formats=("bruker", "dicom", "nifti"),
            template=scanner_template(),
            anonymize=True,
        ),
        result,
        program=program,
        phantom=phantom,
    )
    subject = load_bruker_template(report.bruker.study_dir)
    assert subject.subject_name == ANONYMIZED_SUBJECT
    assert subject.institution == "Test Institute"
    assert subject.study_number == 4
    ds = read_dicom_series_from(tmp_path / "anon_dicom")
    assert ds.PatientIdentityRemoved == "YES"
    assert ds.StudyInstanceUID == subject.study_uid

    for path in tmp_path.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert SUBJECT_NAME.encode() not in data, path
            assert b"Mouse Twelve" not in data, path


def test_bundle_rejects_invalid_requests(flash_case, tmp_path):
    result, _, _ = flash_case
    with pytest.raises(ValueError, match="unknown export format"):
        SequenceExportRequest(directory=tmp_path, formats=("tiff",))
    with pytest.raises(ValueError, match="base name"):
        SequenceExportRequest(directory=tmp_path, base_name="  ")
    with pytest.raises(ValueError, match="at least one"):
        perform_sequence_export(
            SequenceExportRequest(directory=tmp_path, formats=()), result
        )
    with pytest.raises(ValueError, match="existing Bruker study"):
        perform_sequence_export(
            SequenceExportRequest(
                directory=tmp_path,
                formats=("bruker",),
                bruker=BrukerExportOptions(study_layout="append"),
                bruker_study_dir=tmp_path / "missing",
            ),
            result,
        )


# --- Export dialog ----------------------------------------------------------


def test_export_dialog_builds_request(tmp_path):
    from blochsimulator.ui.sequence_export_dialog import SequenceExportDialog

    dialog = SequenceExportDialog(
        None,
        directory=tmp_path,
        base_name="scan",
        formats=("netcdf",),
        suggested_method="Bruker:CSI",
        scan_name="csi_scan",
        matrix=(16, 8),
        fov_m=(0.04, 0.03),
        slice_thickness_mm=2.0,
    )
    assert dialog.selected_formats() == ("netcdf",)
    assert dialog.method_name.currentText() == "Bruker:CSI"

    dialog.format_checks["bruker"].setChecked(True)
    version_index = dialog.paravision_version.findData("360.3.6")
    dialog.paravision_version.setCurrentIndex(version_index)
    choices = [
        dialog.raw_data_files.itemText(i) for i in range(dialog.raw_data_files.count())
    ]
    assert choices == ["rawdata.job0"]
    assert dialog.conjugate.isChecked()
    dialog.paravision_version.setCurrentIndex(
        dialog.paravision_version.findData("7.0.0")
    )
    choices = [
        dialog.raw_data_files.itemText(i) for i in range(dialog.raw_data_files.count())
    ]
    assert choices == ["fid", "rawdata.job0", "both"]
    dialog.raw_data_files.setCurrentText("both")

    dialog.append_study.setChecked(True)
    assert "existing ParaVision study" in dialog.validation_error()
    study = tmp_path / "study"
    study.mkdir()
    dialog.study_dir_edit.setText(str(study))
    assert "no subject file" in dialog.validation_error()
    (study / "subject").write_text("##TITLE=Parameter List\n##END=\n")
    assert dialog.validation_error() is None

    dialog.geometry.setChecked(True)
    dialog.set_template(scanner_template(paravision_version="6.0.1"))
    assert dialog.paravision_version.currentData() == "6.0.1"
    dialog.anonymize.setChecked(True)

    request = dialog.request()
    assert request.directory == tmp_path
    assert request.base_name == "scan"
    assert request.formats == ("netcdf", "bruker")
    assert request.appends_bruker_study
    assert request.bruker_study_dir == study
    assert request.anonymize
    assert request.template.subject_name == SUBJECT_NAME
    assert request.bruker.paravision_version == "6.0.1"
    assert request.bruker.method_name == "Bruker:CSI"
    assert request.bruker.scan_name == "csi_scan"
    assert request.bruker.matrix == (16, 8)
    assert request.bruker.fov_m == pytest.approx((0.04, 0.03))
    assert request.bruker.slice_thickness_mm == pytest.approx(2.0)
    assert request.bruker.conjugate_raw_data
    dialog.deleteLater()


def test_template_dialog_updates_owner_paths():
    from blochsimulator.ui.sequence_export_dialog import TemplateMetadataDialog

    template = scanner_template(data_root="/opt/nmrdata/data/nmrsu")
    dialog = TemplateMetadataDialog(template)
    dialog.fields["owner"].setText("guest")
    dialog.fields["subject_name"].setText("Phantom")
    edited = dialog.template()
    assert edited.owner == "guest"
    assert edited.user_host == "guest@biospec"
    assert edited.data_root == "/opt/nmrdata/data/guest"
    assert edited.subject_name == "Phantom"
    assert edited.study_number == template.study_number
    dialog.deleteLater()


# --- Bruker datasets --------------------------------------------------------


@pytest.mark.parametrize("version", list(PARAVISION_PROFILES))
def test_brukerapi_reads_exported_scan(flash_case, tmp_path, version):
    brukerapi_dataset = pytest.importorskip("brukerapi.dataset")
    result, program, phantom = flash_case
    profile = PARAVISION_PROFILES[version]
    raw_files = (
        "both" if "both" in profile.raw_data_files else profile.default_raw_data_files
    )
    exported = export_bruker_dataset(
        result,
        tmp_path,
        program=program,
        phantom=phantom,
        options=BrukerExportOptions(
            paravision_version=version, raw_data_files=raw_files
        ),
    )
    provenance = json.loads(
        (exported.scan_dir / "blochsim_provenance.json").read_text()
    )
    assert provenance["paravision_version"] == version
    scale = provenance["signal_scale"]
    expected = np.sort(np.abs(result.signal)) * scale

    def load(name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(
                brukerapi_dataset.Dataset(str(exported.scan_dir / name)).data
            )

    raw_names = ["fid"] if "fid" in profile.raw_data_files else []
    # brukerapi 0.1.10 reads rawdata.job0 only up to ParaVision 6.0.1; it fails
    # on measured PV7 rawdata.job0 as well (no numpy_dtype rule for PV-7.0.0).
    if profile.family == "pv6" and "rawdata.job0" in profile.raw_data_files:
        raw_names.append("rawdata.job0")
    for name in raw_names:
        raw = load(name)
        assert raw.size == result.signal.size
        np.testing.assert_allclose(
            np.sort(np.abs(raw.ravel())),
            expected,
            rtol=1e-4,
            atol=1e-6 * expected.max(),
        )
    if "fid" in raw_names:
        assert load("fid").shape == (8, 4, 2, 2, 1)

    image = load("pdata/1/2dseq")
    assert image.shape == (8, 4, 2, 2)  # read, phase, slice, repetition
    (series,) = extract_image_series(result, include_pools=False, include_phase=False)
    for frame in series.frames:
        np.testing.assert_allclose(
            image[:, :, frame.slice_index, frame.repetition_index],
            frame.magnitude.T,
            rtol=0,
            atol=2e-4 * float(series.magnitude_stack().max()),
        )


def listing(path):
    """File names in ``path`` without macOS duplicates such as ``acqp 2``."""

    return {item.name for item in path.iterdir() if " " not in item.name}


def nspect_case(points, dwell):
    """Single-FID spectroscopy result and program with ``points`` samples."""

    times = np.arange(points) * dwell
    result = SequenceSimulationResult(
        signal=np.exp((2j * np.pi * 150.0 - 20.0) * times),
        adc_times_s=times,
        final_magnetization=np.zeros((1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.asarray([], dtype=float),
        metadata={"nucleus": "C13", "field_strength_t": 7.0},
    )
    program = SequenceProgram(
        (ADCEvent(0.0, points, dwell),), duration_s=(points + 1) * dwell
    )
    return result, program


def reference_case(kind, reference, request):
    """Result, program and phantom of ``kind``; NSPECT copies ``reference``."""

    if kind == "flash":
        return request.getfixturevalue("flash_case")
    if kind == "csi":
        program = load_pulseq(DATA_DIR / "csi_2d_centric.seq")
        return request.getfixturevalue("csi_result"), program, None
    method = read_jcamp(reference / "method").parameters
    points = np.atleast_1d(parse_jcamp_value(method["PVM_SpecMatrix"]))[0]
    swh = np.atleast_1d(parse_jcamp_value(method["PVM_SpecSWH"]))[0]
    return (*nspect_case(int(points), 1.0 / float(swh)), None)


def comment_kinds(path):
    """The ``$$`` comments of a parameter file with their values blanked out."""

    kinds = []
    for comment in read_jcamp(path).comments:
        if comment.startswith("Write Options: "):
            kinds.append("write options")
        elif re.fullmatch(
            r"\w{3} \w{3} [ \d]\d [\d:]{8} \d{4} .+ \(UT[-+][\d.]+h\)  \S+", comment
        ):
            kinds.append("ctime stamp")
        elif re.fullmatch(r"[\d-]{10} [\d:.]{12} [-+]\d{4}  \S+", comment):
            kinds.append("iso stamp")
        elif comment.startswith("/"):
            kinds.append(f"path {comment.split('/data/')[0]}")
        elif comment.startswith("process /"):
            kinds.append(f"process {comment.rsplit('/', 1)[1]}")
        elif comment.startswith("File finished by PARX at "):
            kinds.append("footer")
        else:
            kinds.append(comment)
    return kinds


def write_options(path):
    (comment,) = [
        text for text in read_jcamp(path).comments if text.startswith("Write Options: ")
    ]
    return set(comment.split(": ", 1)[1].split(", "))


def value_form(raw):
    """Rank and kind (string, struct or atom) of a raw JCAMP value."""

    text = raw.strip()
    rank = 0
    sized = re.match(r"\(\s*([\d,\s]+)\)\n(.*)", text, re.S)
    if sized:
        rank = len([item for item in sized.group(1).split(",") if item.strip()])
        text = sized.group(2).strip()
    if text.startswith("<"):
        return rank, "string"
    return rank, "struct" if text.startswith("(") else "atom"


def assert_matches_reference(ours, reference, scans, names):
    """Compare the parameter files ``names`` of an exported scan.

    Header lines and the shape of the ``$$`` comments must equal those of
    ``reference``, every parameter must occur in one of the reference ``scans``
    and parameters that ``reference`` holds must have the same value form.
    """

    for name in names:
        actual, expected = read_jcamp(ours / name), read_jcamp(reference / name)
        for key in ("TITLE", "JCAMPDX", "DATATYPE", "ORIGIN", "OWNER"):
            assert actual.header.get(key) == expected.header.get(key), (name, key)
        assert comment_kinds(ours / name) == comment_kinds(reference / name), name
        known = set().union(
            *(
                read_jcamp(scan / name).parameters
                for scan in scans
                if (scan / name).is_file()
            )
        )
        extra = set(actual.parameters) - known - MARKER_PARAMETERS
        assert not extra, (name, sorted(extra))
        reshaped = sorted(
            key
            for key, raw in actual.parameters.items()
            if key in expected.parameters
            and value_form(raw) != value_form(expected.parameters[key])
        )
        assert not reshaped, (name, reshaped)


def kblock_fid_bytes(acqp):
    """Size of a ParaVision 5 fid whose blocks are padded to 1024 bytes."""

    size = [int(item) for item in np.atleast_1d(parse_jcamp_value(acqp["ACQ_size"]))]
    blocks = np.prod(size[1:], dtype=int) * int(parse_jcamp_value(acqp["NI"]))
    blocks *= int(parse_jcamp_value(acqp["NR"]))
    return int(blocks) * -(-4 * size[0] // 1024) * 1024


@pytest.mark.skipif(
    not (PV7_REFERENCE_STUDY / "3" / "acqp").is_file(),
    reason="ParaVision 7 reference study not available",
)
def test_pv7_export_matches_reference_flash_structure(flash_case, tmp_path):
    result, program, phantom = flash_case
    reference = PV7_REFERENCE_STUDY / "3"
    template = load_bruker_template(PV7_REFERENCE_STUDY)
    assert template.paravision_version.startswith("7.0")
    exported = export_bruker_dataset(
        result,
        tmp_path,
        program=program,
        phantom=phantom,
        options=BrukerExportOptions(
            paravision_version="7.0.0", template=template, anonymize=True
        ),
    )
    ours = exported.scan_dir
    assert exported.method == "Bruker:FLASH"

    core = {
        "acqp",
        "method",
        "fid",
        "pulseprogram",
        "visu_pars",
        "configscan",
        "specpar",
        "uxnmr.info",
        "uxnmr.par",
        "AdjStatePerScan",
    }
    assert core & listing(reference) <= listing(ours)
    assert listing(reference / "pdata" / "1") <= listing(ours / "pdata" / "1")

    for name in (
        "acqp",
        "method",
        "visu_pars",
        "pdata/1/reco",
        "pdata/1/procs",
        "pdata/1/methreco",
        "pdata/1/visu_pars",
        "../subject",
    ):
        expected, actual = read_jcamp(reference / name), read_jcamp(ours / name)
        for key in ("TITLE", "JCAMPDX", "DATATYPE", "ORIGIN", "OWNER"):
            assert actual.header.get(key) == expected.header.get(key), (name, key)
        extra = (
            set(actual.parameters) - set(expected.parameters) - {"VisuSeriesComment"}
        )
        assert not extra, (name, sorted(extra))

    acqp = read_jcamp(ours / "acqp").parameters
    dims = int(parse_jcamp_value(acqp["ACQ_dim"]))
    assert not isinstance(parse_jcamp_value(acqp["ACQ_flip_angle"]), list)
    encoding = parse_jcamp_value(acqp["ACQ_phase_encoding_mode"])
    assert len(encoding) == dims and str(encoding[0]) == "Read"
    assert len(parse_jcamp_value(acqp["ACQ_phase_enc_start"])) == dims
    assert parse_jcamp_value(acqp["ACQ_inter_echo_time"])


@pytest.mark.skipif(
    not (PV7_REFERENCE_STUDY / "36" / "rawdata.job0").is_file(),
    reason="ParaVision 7 rawdata.job0 reference scan not available",
)
def test_pv7_rawdata_job0_matches_reference_nspect(tmp_path):
    reference = PV7_REFERENCE_STUDY / "36"
    ref_acqp = read_jcamp(reference / "acqp").parameters
    ref_method = read_jcamp(reference / "method").parameters
    points = int(parse_jcamp_value(ref_method["PVM_DigNp"]))
    dwell = 1.0 / float(parse_jcamp_value(ref_acqp["SW_h"]))
    result, program = nspect_case(points, dwell)
    exported = export_bruker_dataset(
        result,
        tmp_path,
        program=program,
        options=BrukerExportOptions(paravision_version="7.0.0", raw_data_files="both"),
    )
    ours = exported.scan_dir
    assert exported.method == "Bruker:NSPECT"
    acqp = read_jcamp(ours / "acqp").parameters
    method = read_jcamp(ours / "method").parameters

    # JOB_DESCRIPTION_TYPE of a single-receiver PV7 job: scanSize, blocks,
    # dummy scans, total scans, receiver gain, swh, scan shift, stored scans.
    (job,) = parse_jcamp_value(acqp["ACQ_jobs"])
    (expected_job,) = parse_jcamp_value(ref_acqp["ACQ_jobs"])
    assert job == pytest.approx(expected_job)
    for name in (
        "ACQ_jobs_size",
        "ACQ_dim",
        "ACQ_dim_desc",
        "ACQ_size",
        "ACQ_scan_size",
        "GO_raw_data_format",
        "GO_block_size",
        "BYTORDA",
    ):
        np.testing.assert_array_equal(
            parse_jcamp_value(acqp[name]),
            parse_jcamp_value(ref_acqp[name]),
            err_msg=name,
        )
    for name in ("PVM_DigNp", "PVM_SpecMatrix", "PVM_EncNReceivers"):
        np.testing.assert_array_equal(
            parse_jcamp_value(method[name]),
            parse_jcamp_value(ref_method[name]),
            err_msg=name,
        )
    assert parse_jcamp_value(method["PVM_DigShift"]) == 0

    # The measured scan stores identical fid and rawdata.job0 files.
    for name in ("fid", "rawdata.job0"):
        assert (ours / name).stat().st_size == (reference / name).stat().st_size
    assert (reference / "fid").read_bytes() == (reference / "rawdata.job0").read_bytes()
    assert (ours / "fid").read_bytes() == (ours / "rawdata.job0").read_bytes()


@pytest.mark.skipif(
    not (PV5_REFERENCE_STUDY / "subject").is_file(),
    reason="ParaVision 5.1 reference study not available",
)
@pytest.mark.parametrize("kind", list(PV5_SCANS))
def test_pv5_export_matches_reference_study(kind, request, tmp_path):
    reference = PV5_REFERENCE_STUDY / PV5_SCANS[kind]
    template = load_bruker_template(PV5_REFERENCE_STUDY)
    assert template.paravision_version == "5.1"
    result, program, phantom = reference_case(kind, reference, request)
    exported = export_bruker_dataset(
        result,
        tmp_path,
        program=program,
        phantom=phantom,
        options=BrukerExportOptions(
            paravision_version="5.1", template=template, anonymize=True
        ),
    )
    ours = exported.scan_dir

    # PV5 keeps visu_pars in pdata only and has no methreco or rawdata.job0.
    assert listing(reference) <= listing(ours)
    assert listing(ours) - listing(reference) <= {"fid", "blochsim_provenance.json"}
    pdata, ref_pdata = ours / "pdata" / "1", reference / "pdata" / "1"
    assert listing(pdata) - listing(ref_pdata) <= {"2dseq"}
    assert listing(ref_pdata) - listing(pdata) <= {"meta", "roi"}

    scans = [path.parent for path in PV5_REFERENCE_STUDY.glob("*/acqp")]
    assert len(scans) > 30
    assert_matches_reference(
        ours,
        reference,
        scans,
        (
            "acqp",
            "method",
            "AdjStatePerScan",
            "pdata/1/reco",
            "pdata/1/procs",
            "pdata/1/d3proc",
            "pdata/1/id",
            "pdata/1/visu_pars",
            "../subject",
        ),
    )

    acqp = read_jcamp(ours / "acqp").parameters
    ref_acqp = read_jcamp(reference / "acqp").parameters
    for name in ("ACQ_method", "ACQ_sw_version", "GO_block_size", "BYTORDA"):
        assert acqp[name] == ref_acqp[name], name
    method = read_jcamp(ours / "method").parameters
    assert method["Method"] == read_jcamp(reference / "method").parameters["Method"]
    # Standard_KBlock_Format pads every block to 1024 bytes; the rule
    # reproduces the archived fid size of each reference scan.
    assert kblock_fid_bytes(ref_acqp) == PV5_REFERENCE_FID_BYTES[reference.name]
    assert (ours / "fid").stat().st_size == kblock_fid_bytes(acqp)


@pytest.mark.skipif(
    not all(
        (PV360_REFERENCE / scan / "acqp").is_file() for scan in PV360_SCANS.values()
    ),
    reason="ParaVision 360 reference scans not available",
)
@pytest.mark.parametrize("kind", list(PV360_SCANS))
def test_pv360_export_matches_reference_scans(kind, request, tmp_path):
    reference = PV360_REFERENCE / PV360_SCANS[kind]
    result, program, phantom = reference_case(kind, reference, request)
    exported = export_bruker_dataset(
        result,
        tmp_path,
        program=program,
        phantom=phantom,
        options=BrukerExportOptions(paravision_version="360.3.6"),
    )
    ours = exported.scan_dir

    # PV360 stores rawdata.job0 only, the precompiled pulse program and no
    # scan-level visu_pars or procs.
    assert listing(ours) - listing(reference) <= {
        "rawdata.job0",
        "blochsim_provenance.json",
    }
    assert listing(reference) - listing(ours) <= {"acqp.out", "lists", "shimcondition"}
    pdata, ref_pdata = ours / "pdata" / "1", reference / "pdata" / "1"
    assert listing(pdata) - listing(ref_pdata) <= {"2dseq"}
    assert listing(ref_pdata) - listing(pdata) <= {"MxiAcqReco.log", "reco.out"}

    names = (
        "acqp",
        "method",
        "AdjStatePerScan",
        "configscan",
        "pdata/1/reco",
        "pdata/1/methreco",
        "pdata/1/id",
        "pdata/1/visu_pars",
    )
    scans = [PV360_REFERENCE / scan for scan in PV360_SCANS.values()]
    assert_matches_reference(ours, reference, scans, names)
    for name in names:
        # Without $$ @vis lines the export cannot claim "symbol visibility".
        assert write_options(ours / name) <= write_options(reference / name), name

    acqp = read_jcamp(ours / "acqp").parameters
    ref_acqp = read_jcamp(reference / "acqp").parameters
    assert acqp["ACQ_sw_version"] == ref_acqp["ACQ_sw_version"]
    # JOB_DESCRIPTION_TYPE of PV360: scanSize per channel, transactionBlocks,
    # dummy scans, total scans, receiver gain, swh, stored scans, channel and
    # job title.  scanSize holds two words per read point.
    points, swh = {
        "flash": ("PVM_EncMatrix", "PVM_EffSWh"),
        "nspect": ("PVM_SpecMatrix", "PVM_SpecSWH"),
    }[kind]
    for scan in (ours, reference):
        job = parse_jcamp_value(read_jcamp(scan / "acqp").parameters["ACQ_jobs"])[0]
        method = read_jcamp(scan / "method").parameters
        assert len(job) == 9 and job[7] == 1 and str(job[8]) == "job0", scan
        assert job[0] == 2 * np.atleast_1d(parse_jcamp_value(method[points]))[0]
        assert job[5] == pytest.approx(np.atleast_1d(parse_jcamp_value(method[swh]))[0])
    (job,) = parse_jcamp_value(acqp["ACQ_jobs"])
    method = read_jcamp(ours / "method").parameters
    receivers = int(parse_jcamp_value(method["PVM_EncNReceivers"]))
    assert (ours / "rawdata.job0").stat().st_size == job[0] * job[6] * 4 * receivers
    assert not (ours / "fid").exists()


def test_study_folder_name_truncates_like_paravision7():
    from dataclasses import replace
    from datetime import datetime

    from blochsimulator.sequence import bruker_export

    # Subject and study of the PV7 reference study; ParaVision named its folder
    # 20251107_160930_1M_13C_Urea_Falcon_Tube_lin_cryo_86mm_posit_2_1.
    identity = replace(
        build_scan_identity(),
        subject_id="1M 13C Urea Falcon Tube",
        study_name="lin cryo 86mm positioning",
        study_number=1,
    )
    name = bruker_export._study_folder_name(
        identity, datetime(2025, 11, 7, 16, 9, 30), PARAVISION_PROFILES["7.0.0"]
    )
    assert name == "20251107_160930_1M_13C_Urea_Falcon_Tube_lin_cryo_86mm_posit_1_1"
    assert len(name) == 63
