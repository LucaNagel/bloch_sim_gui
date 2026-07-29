import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from blochsimulator.sequence import (
    BrukerExportOptions,
    SequenceCompiler,
    SequenceSimulationResult,
    export_bruker_raw,
    infer_spectroscopic_acquisition,
    load_pulseq,
)
from blochsimulator.sequence.pulseq_builders import _phase_encoding_indices


pypulseq = pytest.importorskip("pypulseq")
CSI_PATH = Path(__file__).parent / "data" / "csi_2d_centric.seq"


def _jcamp_numeric_array(path: Path, name: str, dtype=float):
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"##\${re.escape(name)}=\([^\n]*\)\n(.*?)(?=\n##)", text, re.S)
    assert match is not None, f"missing JCAMP array {name}"
    body = "\n".join(
        line
        for line in match.group(1).splitlines()
        if not line.lstrip().startswith("$$")
    )
    return np.asarray(
        [
            dtype(value)
            for value in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", body)
        ]
    )


@pytest.fixture(scope="module")
def acquisition_and_compiled():
    program = load_pulseq(CSI_PATH)
    compiled = SequenceCompiler().compile(program)
    return infer_spectroscopic_acquisition(program, compiled=compiled), compiled


def test_reference_csi_sequence_is_inferred_as_spatial_grid_plus_fid(
    acquisition_and_compiled,
):
    acquisition, compiled = acquisition_and_compiled

    assert acquisition.matrix == (16, 16)
    assert acquisition.spectral_points == 256
    assert acquisition.spectral_bandwidth_hz == pytest.approx(4000.0)
    assert acquisition.encoding_indices[0] == (8, 8)
    acquisition.validate_gradient_moments(compiled.adc_gradient_moment_cyc_per_m)


def test_csi_reshape_reorders_centric_encoding_and_preserves_spectral_axis(
    acquisition_and_compiled,
):
    acquisition, _ = acquisition_and_compiled
    chronological = np.arange(acquisition.num_samples, dtype=float).astype(complex)
    grid = acquisition.reshape_signal(chronological)

    assert grid.shape == (16, 16, 256)
    assert np.array_equal(grid[8, 8], chronological[:256])
    assert acquisition.reconstruct_spatial(chronological).shape == grid.shape
    assert acquisition.reconstruct_spectra(chronological).shape == grid.shape


def test_csi_export_contains_sorted_kspace_fid_and_spectrum(
    tmp_path, acquisition_and_compiled
):
    acquisition, compiled = acquisition_and_compiled
    signal = np.ones(acquisition.num_samples, dtype=np.complex128)
    result = SequenceSimulationResult(
        signal=signal,
        adc_times_s=compiled.adc_times_s,
        final_magnetization=np.zeros((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        adc_gradient_moment_cyc_per_m=(compiled.adc_gradient_moment_cyc_per_m),
        metadata={"spectroscopic_acquisition": acquisition.to_metadata()},
    )

    dataset = result.to_xarray()
    assert dataset["csi_kspace"].dims == (
        "phase_y",
        "phase_x",
        "spectral_point",
    )
    assert dataset["csi_kspace"].shape == (16, 16, 256)
    assert dataset["spatial_kx_cyc_per_m"].attrs["units"] == "cycles/m"
    assert dataset["spectral_frequency_hz"].attrs["units"] == "Hz"

    output = tmp_path / "csi_result.npz"
    result.save(output)
    with np.load(output) as saved:
        assert saved["csi_kspace"].shape == (16, 16, 256)
        assert saved["csi_spatial_fid"].shape == (16, 16, 256)
        assert saved["csi_spectrum"].shape == (16, 16, 256)

    h5_output = tmp_path / "csi_result.h5"
    result.save(h5_output)
    import h5py

    with h5py.File(h5_output) as saved:
        assert saved["csi_kspace"].shape == (16, 16, 256)
        assert saved["csi_frequency_hz"].shape == (256,)

    nc_output = tmp_path / "csi_result.nc"
    result.save(nc_output)
    import xarray as xr

    with xr.open_dataset(nc_output) as saved:
        assert saved["csi_kspace_real"].shape == (16, 16, 256)
        assert saved["csi_kspace_imag"].shape == (16, 16, 256)


def test_csi_bruker_export_writes_rawdata_job0_in_linear_grid_order(
    tmp_path, acquisition_and_compiled
):
    acquisition, compiled = acquisition_and_compiled
    chronological = np.arange(
        acquisition.num_samples, dtype=np.float64
    ) + 1j * np.arange(acquisition.num_samples, dtype=np.float64)
    result = SequenceSimulationResult(
        signal=chronological,
        adc_times_s=compiled.adc_times_s,
        final_magnetization=np.zeros((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        adc_gradient_moment_cyc_per_m=compiled.adc_gradient_moment_cyc_per_m,
        metadata={"spectroscopic_acquisition": acquisition.to_metadata()},
    )
    program = load_pulseq(CSI_PATH)

    output = export_bruker_raw(
        result,
        tmp_path / "bruker_csi",
        program=program,
        options=BrukerExportOptions(
            method_name="Bruker:CSI",
            raw_data_files="both",
        ),
        scale=1.0,
    )

    rawdata = np.fromfile(output / "rawdata.job0", dtype="<i4")
    assert rawdata.size == acquisition.num_samples * 2
    expected = acquisition.reshape_signal(chronological).reshape(-1)
    assert np.array_equal(rawdata[0::2], np.rint(expected.real).astype(np.int32))
    assert np.array_equal(rawdata[1::2], np.rint(expected.imag).astype(np.int32))
    assert (output / "fid").is_file()
    assert (output / "rawdata.job0").is_file()

    method = (output / "method").read_text()
    assert "##$Method=<Bruker:CSI>" in method
    assert "##$PVM_EncSpectroscopy=Yes" in method
    assert "##$PVM_SpecMatrix=256" in method
    assert "##$PVM_EncOrder=LINEAR_ENC LINEAR_ENC" in method


def test_centric_builder_matches_luca_csi4_14_by_11_method_order():
    order = _phase_encoding_indices(14, 11, "centric", (0.032, 0.026))
    signed_x = np.asarray([x - 7 for x, _ in order], dtype="<i2")
    signed_y = np.asarray([y - 5 for _, y in order], dtype="<i2")

    # Regression digests of CentricEncOrderMatrixx/y in reference scan 22.
    assert hashlib.sha256(signed_x.tobytes()).hexdigest() == (
        "3a588d0df2055ea5f67168c4205575741053fda1e1690bf0b0cf41287c4566e5"
    )
    assert hashlib.sha256(signed_y.tobytes()).hexdigest() == (
        "9aa33979ef945070766727746f32d3798c0074b58c56411e7bbff8e51d54cb26"
    )


def test_luca_csi4_export_writes_chronological_raw_and_centric_headers(
    tmp_path, acquisition_and_compiled
):
    acquisition, compiled = acquisition_and_compiled
    chronological = np.arange(acquisition.num_samples, dtype=np.float64).astype(
        np.complex128
    )
    result = SequenceSimulationResult(
        signal=chronological,
        adc_times_s=compiled.adc_times_s,
        final_magnetization=np.zeros((1, 1, 1, 3)),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.empty(0),
        adc_gradient_moment_cyc_per_m=compiled.adc_gradient_moment_cyc_per_m,
        metadata={
            "spectroscopic_acquisition": acquisition.to_metadata(),
            "field_strength_t": 7.05,
            "nucleus": "C13",
            "spectral_reference_ppm": 175.0,
        },
    )
    program = load_pulseq(CSI_PATH)

    output = export_bruker_raw(
        result,
        tmp_path / "luca_csi4",
        program=program,
        options=BrukerExportOptions(
            method_name="User:lucaCSI4",
            slice_thickness_mm=6.0,
            raw_data_files="both",
        ),
        scale=1.0,
    )

    rawdata = np.fromfile(output / "rawdata.job0", dtype="<i4")
    assert np.array_equal(rawdata[0::2], chronological.real.astype(np.int32))
    assert np.count_nonzero(rawdata[1::2]) == 0

    first_repetition = tuple(
        index
        for index, repetition in zip(
            acquisition.encoding_indices, acquisition.repetition_indices
        )
        if repetition == 0
    )
    expected_signed_x = np.asarray(
        [x - acquisition.matrix[0] // 2 for x, _ in first_repetition]
    )
    expected_signed_y = np.asarray(
        [y - acquisition.matrix[1] // 2 for _, y in first_repetition]
    )
    assert np.array_equal(
        _jcamp_numeric_array(output / "method", "CentricEncOrderMatrixx", int),
        expected_signed_x,
    )
    assert np.array_equal(
        _jcamp_numeric_array(output / "method", "CentricEncOrderMatrixy", int),
        expected_signed_y,
    )
    assert np.array_equal(
        _jcamp_numeric_array(output / "method", "Reco_x", int),
        np.asarray([x for x, _ in first_repetition]),
    )
    assert np.array_equal(
        _jcamp_numeric_array(output / "method", "Reco_y", int),
        np.asarray([y for _, y in first_repetition]),
    )

    method = (output / "method").read_text(encoding="utf-8")
    acqp = (output / "acqp").read_text(encoding="utf-8")
    visu = (output / "visu_pars").read_text(encoding="utf-8")
    assert "##$Method=<User:lucaCSI4>" in method
    assert "##$CentricEncOrder_OnOff=On" in method
    assert "##$SpiralEncOrder_OnOff=Off" in method
    assert "##$PhaseEncGrad_OnOff=On" in method
    assert "##$KFiltering=On" in method
    assert "##$PVM_EncSpectroscopy=No" in method
    assert "##$PVM_DigDw=0.25" in method
    assert "##$PVM_DigNp=256" in method
    assert "##$PVM_FrqWorkPpm=( 8 )\n175" in method
    assert "##$ACQ_dim=3" in acqp
    assert "##$ACQ_size=( 3 )\n512 16 16" in acqp
    assert "##$VisuCoreDim=3" in visu
    assert "##$VisuCoreSize=( 3 )\n256 16 16" in visu
    assert (output / "pdata" / "1" / "2dseq").stat().st_size == (
        acquisition.spectral_points
        * acquisition.matrix[0]
        * acquisition.matrix[1]
        * np.dtype("<i2").itemsize
    )
