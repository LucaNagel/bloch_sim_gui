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


pypulseq = pytest.importorskip("pypulseq")
CSI_PATH = Path(__file__).parents[1] / "sequences" / "sequences" / "csi_2d_centric.seq"


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
