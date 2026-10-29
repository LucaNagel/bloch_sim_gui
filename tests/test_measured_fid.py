"""Measured FID and noise scans as receiver-calibration inputs."""

from pathlib import Path

import numpy as np
import pytest

from blochsimulator.sequence import (
    MeasuredScan,
    ReceiverCalibration,
    estimate_fid_amplitude,
    estimate_noise_std,
    load_measured_scan,
    measured_calibration_inputs,
)
from blochsimulator.sequence.bruker_jcamp import JcampWriter
from blochsimulator.sequence.measured_fid import check_noise_scan, simulator_nucleus
from blochsimulator.ui.receiver_noise import ReceiverNoiseWidget

SW_HZ = 2000.0
DWELL = 1 / SW_HZ
# Fractional group delay on the 1/32 fine grid of the filter model.
GRPDLY = 68 - 1 / 32
FINE = 32
UREA_STUDY = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "bruker_data"
    / "2025.11.07 - lin cryo - 2M 13C Urea - B1 mapping, AHP, slice profile"
)


def bruker_filter(signal, samples, cutoff=0.95):
    """Sample a band-limited causal signal behind a linear-phase FIR filter.

    ``signal(t)`` starts at t = 0; output sample k lies at
    (k - GRPDLY) dwell, so the filter step response rings at the start.
    """
    step = DWELL / FINE
    half = 68 * FINE
    lags = np.arange(-half, half + 1) * step
    kernel = cutoff / DWELL * np.sinc(cutoff * lags / DWELL) * np.blackman(lags.size)
    kernel *= 1 / (np.sum(kernel) * step)
    times = np.arange((samples - GRPDLY) * FINE + half) * step
    values = signal(times)
    size = values.size + kernel.size - 1
    filtered = np.fft.ifft(np.fft.fft(values, size) * np.fft.fft(kernel, size)) * step
    # Output k is the fine index (k - GRPDLY) * FINE shifted by the kernel half.
    index = np.round((np.arange(samples) - GRPDLY) * FINE).astype(int) + half
    return filtered[index]


def fid_rows(
    amplitude=5e4, t2star=0.05, offset_hz=30.0, sigma=100.0, rows=1, samples=2048
):
    rng = np.random.default_rng(7)

    def signal(t):
        return amplitude * np.exp(-t / t2star + 2j * np.pi * offset_hz * t)

    clean = bruker_filter(signal, samples)
    noise = rng.normal(0, sigma, (rows, samples, 2)) @ np.array([1, 1j])
    return clean + noise


def noise_rows(sigma=100.0, rows=4, samples=1024):
    rng = np.random.default_rng(11)
    noise = rng.normal(0, sigma, (rows, samples, 2)) @ np.array([1, 1j])
    # The filter starts empty: less noise until it has settled.
    noise[:, : int(GRPDLY)] *= 0.2
    return noise


def write_bruker_scan(
    folder,
    rows,
    *,
    na=1,
    rg=64.0,
    calibrated_rg=None,
    sw=SW_HZ,
    data_format="GO_32BIT_SGN_INT",
    byte_order="little",
    raw_name="fid",
    kblock=True,
    flip=90.0,
    acq_dim=1,
    receivers=1,
):
    folder = Path(folder)
    folder.mkdir(parents=True)
    rows = np.atleast_2d(rows)
    samples = rows.shape[1]
    acqp = JcampWriter(title="acqp", origin="test", owner="test")
    acqp.enum("GO_raw_data_format", data_format)
    acqp.enum("BYTORDA", byte_order)
    acqp.array("ACQ_size", [2 * samples] + [1] * (acq_dim - 1))
    acqp.scalar("ACQ_dim", acq_dim)
    acqp.scalar("NA", na)
    acqp.scalar("NR", rows.shape[0])
    acqp.scalar("DS", 0)
    acqp.scalar("RG", rg)
    if calibrated_rg is not None:
        acqp.array("ACQ_CalibratedRG", [calibrated_rg] + [0] * 7)
    acqp.scalar("SW_h", sw)
    acqp.scalar("GRPDLY", GRPDLY)
    acqp.string("NUC1", "13C")
    acqp.scalar("BF1", 75.5025)
    acqp.scalar("ACQ_flip_angle", flip)
    acqp.array("ACQ_repetition_time", [1000.0])
    acqp.enum("GO_block_size", "Standard_KBlock_Format" if kblock else "continuous")
    acqp.write(folder / "acqp")
    method = JcampWriter(title="method", origin="test", owner="test")
    method.scalar("PVM_RepetitionTime", 1500.0)
    method.scalar("PVM_DummyScans", 2)
    method.struct("ExcPulse1", (0.128, 10000, flip, "No", 3, 1280, 1, 1, 0, 50))
    method.scalar("DeadTime", 0.05)
    method.scalar("PVM_EncNReceivers", receivers)
    method.write(folder / "method")
    dtype = np.dtype(
        (">" if byte_order == "big" else "<")
        + {"GO_32BIT_SGN_INT": "i4", "GO_16BIT_SGN_INT": "i2", "GO_32BIT_FLOAT": "f4"}[
            data_format
        ]
    )
    interleaved = np.stack([rows.real, rows.imag], axis=-1).reshape(rows.shape[0], -1)
    if dtype.kind == "i":
        interleaved = np.round(interleaved)
    interleaved = interleaved.astype(dtype)
    row_bytes = interleaved.shape[1] * dtype.itemsize
    padded = int(np.ceil(row_bytes / 1024)) * 1024 if kblock else row_bytes
    payload = b"".join(
        row.tobytes() + b"\0" * (padded - row_bytes) for row in interleaved
    )
    (folder / raw_name).write_bytes(payload)
    return folder


def test_bruker_fid_amplitude_tail_noise_and_metadata(tmp_path):
    data = fid_rows(rows=2)
    folder = write_bruker_scan(tmp_path / "5", data, na=4)
    scan = load_measured_scan(folder / "fid")
    assert scan.data.shape == (2, 2048)
    np.testing.assert_allclose(
        scan.data, np.round(data.real) + 1j * np.round(data.imag)
    )
    assert scan.dwell_s == pytest.approx(DWELL)
    assert scan.group_delay_points == pytest.approx(GRPDLY)
    assert (scan.averages, scan.total_averages, scan.receiver_gain) == (4, 8, 64.0)
    assert scan.nucleus == "C13"
    assert scan.field_strength_t == pytest.approx(7.0525, abs=1e-3)
    assert scan.repetition_time_s == pytest.approx(1.5)
    assert scan.dummy_scans == 2
    assert scan.excitation_delay_s == pytest.approx(0.064e-3 + 0.05e-3)

    amplitude = estimate_fid_amplitude(scan)
    assert amplitude.amplitude == pytest.approx(5e4, rel=5e-3)
    assert amplitude.t2star_s == pytest.approx(0.05, rel=0.05)
    assert amplitude.start_index == 72
    # The first sample after the ringing has decayed and still rings.
    assert amplitude.first_point < 0.97 * 5e4

    noise = estimate_noise_std(scan, from_tail=True)
    assert noise.std_per_row == pytest.approx(100.0, rel=0.06)
    assert noise.std == pytest.approx(noise.std_per_row / np.sqrt(2))
    assert noise.residual_ratio == pytest.approx(1.0, abs=0.08)

    inputs = measured_calibration_inputs(scan)
    assert inputs.averages == 8
    assert inputs.warnings == ()
    assert "FID tail" in inputs.summary()
    calibration = ReceiverCalibration(
        measured_amplitude=inputs.amplitude.amplitude,
        measured_noise_std=inputs.noise.std,
        concentration_mm=2000.0,
        volume_ml=1.0,
        nucleus=inputs.nucleus,
        field_strength_t=7.05,
        dwell_s=inputs.dwell_s,
        averages=inputs.averages,
    )
    # NA x NR accumulated scans: the single-scan SNR is per accumulated scan.
    assert calibration.single_scan_snr == pytest.approx(
        5e4 / 100 / np.sqrt(4), rel=0.07
    )


def test_noise_scan_skips_filter_start_and_scales_to_the_fid_rows(tmp_path):
    fid = load_measured_scan(write_bruker_scan(tmp_path / "1", fid_rows(rows=2)))
    noise = load_measured_scan(write_bruker_scan(tmp_path / "2", noise_rows()))
    estimate = estimate_noise_std(noise, averaged_rows=fid.repetitions)
    assert estimate.method == "noise scan"
    assert estimate.std_per_row == pytest.approx(100.0, rel=0.03)
    assert estimate.std == pytest.approx(estimate.std_per_row / np.sqrt(2))
    assert estimate.samples == 4 * (1024 - int(np.ceil(2 * GRPDLY)))
    # Without the skip, the quiet filter start-up would lower the estimate.
    unskipped = estimate_noise_std(MeasuredScan(noise.data))
    assert unskipped.std_per_row < 0.985 * estimate.std_per_row

    inputs = measured_calibration_inputs(fid, noise)
    assert inputs.noise == estimate
    assert inputs.snr == pytest.approx(5e4 / (100 / np.sqrt(2)), rel=0.04)


def test_spurs_in_the_noise_scan_use_the_spectral_median(tmp_path):
    fid = load_measured_scan(write_bruker_scan(tmp_path / "1", fid_rows(rows=2)))
    rows = noise_rows()
    # A cluster of spurs near the band edge, as a digitizer at low gain shows.
    time = np.arange(rows.shape[1]) * DWELL
    for frequency in (900.0, 920.0, 950.0):
        rows += 300.0 * np.exp(2j * np.pi * frequency * time)
    noise = load_measured_scan(write_bruker_scan(tmp_path / "2", rows))
    estimate = estimate_noise_std(noise, averaged_rows=fid.repetitions)
    assert estimate.residual_ratio > 1.1
    assert estimate.method == "noise scan, spectral median"
    assert estimate.std_per_row == pytest.approx(100.0, rel=0.1)
    inputs = measured_calibration_inputs(fid, noise)
    assert any("spectral median" in text for text in inputs.warnings)


def test_calibrated_receiver_gain_and_low_gain_warning(tmp_path):
    # ParaVision may keep a changed RG; ACQ_CalibratedRG is the applied gain.
    folder = write_bruker_scan(tmp_path / "1", fid_rows(), rg=64.0, calibrated_rg=1.01)
    scan = load_measured_scan(folder)
    assert scan.receiver_gain == pytest.approx(1.01)
    inputs = measured_calibration_inputs(scan)
    assert any("Receiver gain 1.01 is low" in text for text in inputs.warnings)
    unset = write_bruker_scan(tmp_path / "2", fid_rows(), rg=64.0, calibrated_rg=0)
    assert load_measured_scan(unset).receiver_gain == 64.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"rg": 32.0}, "RG 32 vs 64"),
        ({"na": 2}, "NA 2 vs 1"),
        ({"sw": 4000.0}, "bandwidth 4000 Hz vs 2000 Hz"),
    ],
)
def test_noise_scan_must_match_averages_gain_and_bandwidth(tmp_path, kwargs, message):
    fid = load_measured_scan(write_bruker_scan(tmp_path / "1", fid_rows()))
    noise = load_measured_scan(
        write_bruker_scan(tmp_path / "2", noise_rows(), **kwargs)
    )
    with pytest.raises(ValueError, match=message):
        check_noise_scan(fid, noise)
    with pytest.raises(ValueError, match="same|averages"):
        measured_calibration_inputs(fid, noise)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data_format": "GO_16BIT_SGN_INT", "byte_order": "big"},
        {"data_format": "GO_32BIT_FLOAT"},
        {"raw_name": "rawdata.job0", "kblock": False},
    ],
)
def test_raw_formats_byte_orders_and_padding(tmp_path, kwargs):
    # 500 complex points: KBlock rows are padded to whole KBytes.
    data = fid_rows(amplitude=1e4, sigma=20.0, rows=3, samples=500)
    folder = write_bruker_scan(tmp_path / "3", data, **kwargs)
    scan = load_measured_scan(folder)
    expected = (
        data
        if kwargs.get("data_format") == "GO_32BIT_FLOAT"
        else (np.round(data.real) + 1j * np.round(data.imag))
    )
    np.testing.assert_allclose(scan.data, expected, rtol=1e-6)
    assert scan.source.endswith(kwargs.get("raw_name", "fid"))


def test_unsupported_bruker_data_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="1-D"):
        load_measured_scan(write_bruker_scan(tmp_path / "a", fid_rows(), acq_dim=2))
    with pytest.raises(ValueError, match="single-receiver"):
        load_measured_scan(write_bruker_scan(tmp_path / "b", fid_rows(), receivers=2))
    folder = write_bruker_scan(tmp_path / "c", fid_rows())
    (folder / "fid").write_bytes((folder / "fid").read_bytes()[:-8])
    with pytest.raises(ValueError, match="whole rows"):
        load_measured_scan(folder)
    with pytest.raises(ValueError, match="Unsupported"):
        load_measured_scan(tmp_path / "spectrum.jdx")


def test_complex_arrays_from_numpy_and_text(tmp_path):
    data = fid_rows(rows=3, samples=256)
    np.save(tmp_path / "fid.npy", data)
    scan = load_measured_scan(tmp_path / "fid.npy")
    np.testing.assert_array_equal(scan.data, data)
    assert scan.dwell_s is None and scan.group_delay_points == 0
    np.savetxt(tmp_path / "fid.txt", np.column_stack([data[0].real, data[0].imag]))
    np.testing.assert_allclose(load_measured_scan(tmp_path / "fid.txt").data, data[:1])
    np.savetxt(
        tmp_path / "fid.csv",
        np.column_stack([data[1].real, data[1].imag]),
        delimiter=",",
    )
    np.testing.assert_allclose(load_measured_scan(tmp_path / "fid.csv").data, data[1:2])
    np.save(tmp_path / "pairs.npy", np.stack([data.real, data.imag], axis=-1))
    np.testing.assert_allclose(load_measured_scan(tmp_path / "pairs.npy").data, data)
    np.save(tmp_path / "real.npy", data.real)
    with pytest.raises(ValueError, match="real, imaginary"):
        load_measured_scan(tmp_path / "real.npy")
    # Without a dwell time the amplitude is still extrapolated, T2* unknown.
    decay = MeasuredScan(5e4 * np.exp(-np.arange(256) / 100.0) + 0j)
    estimate = estimate_fid_amplitude(decay)
    assert estimate.amplitude == pytest.approx(5e4, rel=1e-4)
    assert estimate.start_index == 0 and estimate.t2star_s is None


def test_signal_left_in_the_tail_and_low_snr_are_reported(tmp_path):
    long_t2 = MeasuredScan(
        fid_rows(t2star=2.0), dwell_s=DWELL, group_delay_points=GRPDLY
    )
    inputs = measured_calibration_inputs(long_t2)
    assert any("Signal remains" in text for text in inputs.warnings)
    # The spectral median ignores the narrow line in the tail.
    assert inputs.noise.std == pytest.approx(100.0, rel=0.1)
    weak = MeasuredScan(
        fid_rows(amplitude=1e3), dwell_s=DWELL, group_delay_points=GRPDLY
    )
    assert any("low" in text for text in measured_calibration_inputs(weak).warnings)


def test_simulator_nucleus_names():
    assert simulator_nucleus("13C") == "C13"
    assert simulator_nucleus("1H") == "H1"
    assert simulator_nucleus("23Na") == "Na23"
    assert simulator_nucleus("C13") == "C13"


def test_receiver_panel_loads_fid_and_noise_scan(tmp_path):
    fid_dir = write_bruker_scan(tmp_path / "7", fid_rows(rows=2), na=4)
    noise_dir = write_bruker_scan(tmp_path / "8", noise_rows(), na=4)
    bad_dir = write_bruker_scan(tmp_path / "9", noise_rows(), na=4, rg=16.0)
    panel = ReceiverNoiseWidget()
    panel.mode.setCurrentIndex(panel.mode.findData("calibrated"))
    assert panel.load_fid_button.isEnabled()
    assert panel.load_noise_scan(noise_dir / "fid")
    assert "load the FID" in panel.measured_info.text()

    assert panel.load_fid(fid_dir / "fid")
    fid = load_measured_scan(fid_dir)
    noise = load_measured_scan(noise_dir)
    inputs = measured_calibration_inputs(fid, noise)
    assert float(panel.amplitude.text()) == pytest.approx(
        inputs.amplitude.amplitude, rel=1e-5
    )
    assert panel.reference_std.value() == pytest.approx(inputs.noise.std, rel=1e-5)
    assert panel.averages.value() == 8
    assert panel.bandwidth_khz.value() == pytest.approx(2.0)
    assert panel.flip_angle.value() == 90.0
    assert panel.reference_tr.value() == pytest.approx(1.5)
    assert panel.reference_dummy_scans.value() == 2
    assert panel.reference_t2star.value() == pytest.approx(
        inputs.amplitude.t2star_s * 1e3, abs=0.01
    )
    assert panel.reference_delay.value() == pytest.approx(0.114)
    text = panel.measured_info.text()
    assert "7/fid (C13 at 7.053 T)" in text and "noise scan" in text
    # The reference model stays a deliberate choice.
    assert not panel.reference_scan_box.isChecked()
    # Set calibration accepts the nominal field but no other nucleus or field.
    panel.concentration.setText("2000")
    panel.volume.setText("50")
    panel.capture_calibration(nucleus="C13", field_strength_t=7.0)
    for nucleus, field, message in (
        ("H1", 7.0, "with C13, but"),
        ("C13", 3.0, "with 7.053 T"),
    ):
        with pytest.raises(ValueError, match=message):
            panel.capture_calibration(nucleus=nucleus, field_strength_t=field)

    # A noise scan at another gain is refused and leaves the inputs alone.
    amplitude = panel.amplitude.text()
    assert not panel.load_noise_scan(bad_dir / "fid")
    assert "RG 16 vs 64" in panel.measured_info.text()
    assert panel.amplitude.text() == amplitude
    # A new FID at another gain drops the loaded noise scan for the FID tail.
    other = write_bruker_scan(tmp_path / "10", fid_rows(rows=2), na=4, rg=16.0)
    assert panel.load_fid(other)
    assert "Noise scan dropped" in panel.measured_info.text()
    assert "FID tail" in panel.measured_info.text()
    assert not panel.load_fid(tmp_path / "missing.npy")
    assert "Could not read the FID" in panel.measured_info.text()

    restored = ReceiverNoiseWidget()
    restored.restore_state(panel.get_state())
    assert restored.measured_info.text() == panel.measured_info.text()
    assert restored.amplitude.text() == panel.amplitude.text()


@pytest.mark.skipif(not UREA_STUDY.is_dir(), reason="local urea study not present")
def test_measured_urea_fids_give_consistent_calibration_inputs():
    """Two repeated 90° scans of the 2 M 13C urea phantom at 7 T."""
    first, second = (
        measured_calibration_inputs(load_measured_scan(UREA_STUDY / scan))
        for scan in ("35", "36")
    )
    for inputs in (first, second):
        assert inputs.nucleus == "C13"
        assert inputs.field_strength_t == pytest.approx(7.053, abs=1e-3)
        assert inputs.amplitude_delay_s == pytest.approx(0.114e-3)
        assert inputs.amplitude.amplitude == pytest.approx(5.71e4, rel=0.02)
        assert inputs.noise.std == pytest.approx(113.0, rel=0.05)
        assert inputs.warnings == ()
    assert first.amplitude.amplitude == pytest.approx(
        second.amplitude.amplitude, rel=0.01
    )
    assert first.snr == pytest.approx(second.snr, rel=0.05)
