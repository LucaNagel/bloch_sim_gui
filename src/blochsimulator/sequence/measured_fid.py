"""Measured FID and noise scans as inputs of a receiver calibration.

:func:`load_measured_scan` reads a ParaVision ``fid`` or ``rawdata.job0``
with its ``acqp`` and ``method``, or a complex NumPy/text array. Repetitions
(NR) are rows; ParaVision accumulates the NA averages inside every row.

:func:`estimate_fid_amplitude` extrapolates |FID| to the acquisition start
with a weighted fit of log|FID| after the digital-filter group delay and its
ringing. :func:`estimate_noise_std` returns the per-quadrature noise std of
the row-averaged data, from a scan without excitation or from the FID tail.
Amplitude and noise are measured on the same scaled data, so their ratio is
independent of whether the scanner sums or averages, as long as a separate
noise scan uses the same NA, receiver gain and bandwidth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..units import NUCLEUS_GAMMA_HZ_PER_T
from .bruker_jcamp import normalize_nucleus, read_jcamp

_BRUKER_DTYPES = {
    "GO_32BIT_SGN_INT": "i4",
    "GO_16BIT_SGN_INT": "i2",
    "GO_32BIT_FLOAT": "f4",
    "GO_64BIT_FLOAT": "f8",
}
_BRUKER_FILES = ("fid", "rawdata.job0", "acqp", "method")
_ARRAY_SUFFIXES = (".npy", ".txt", ".csv", ".dat")
# Bruker digital filters settle within about two group delays.
_FILTER_SETTLING = 2.0
_RESIDUAL_LIMIT = 1.1


@dataclass(frozen=True)
class MeasuredScan:
    """Complex raw data (repetitions, samples) with its acquisition context.

    Sample ``k`` lies at ``(k - group_delay_points) * dwell_s`` after the
    acquisition start. ``averages`` is the number of scans accumulated in
    every row; ``excitation_delay_s`` the time from the excitation centre to
    the acquisition start. Unknown values are None.
    """

    data: np.ndarray
    dwell_s: Optional[float] = None
    group_delay_points: float = 0.0
    averages: int = 1
    receiver_gain: Optional[float] = None
    nucleus: Optional[str] = None
    frequency_hz: Optional[float] = None
    flip_angle_deg: Optional[float] = None
    repetition_time_s: Optional[float] = None
    dummy_scans: Optional[int] = None
    excitation_delay_s: Optional[float] = None
    source: str = ""

    def __post_init__(self):
        data = np.asarray(self.data)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 2:
            raise ValueError("Measured data must hold at least two samples per row")
        data = data.astype(np.complex128, copy=False)
        if not np.all(np.isfinite(data)):
            raise ValueError("Measured data contain non-finite samples")
        object.__setattr__(self, "data", data)
        if self.dwell_s is not None and not float(self.dwell_s) > 0:
            raise ValueError("Dwell time must be positive")
        if int(self.averages) != self.averages or self.averages < 1:
            raise ValueError("Averages must be a positive integer")

    @property
    def repetitions(self):
        return int(self.data.shape[0])

    @property
    def samples(self):
        return int(self.data.shape[1])

    @property
    def total_averages(self):
        """Scans in the row-averaged data: NA x repetitions."""
        return int(self.averages) * self.repetitions

    @property
    def bandwidth_hz(self):
        return None if self.dwell_s is None else 1.0 / self.dwell_s

    @property
    def field_strength_t(self):
        if self.nucleus is None or self.frequency_hz is None:
            return None
        gamma = NUCLEUS_GAMMA_HZ_PER_T.get(self.nucleus)
        return None if gamma is None else self.frequency_hz / gamma

    def averaged(self):
        """Mean over the repetitions."""
        return self.data.mean(axis=0)


@dataclass(frozen=True)
class AmplitudeEstimate:
    """|FID| extrapolated to the acquisition start.

    The fit covers ``points`` samples from ``start_index`` with a polynomial
    of ``order`` in time; ``t2star_s`` is -1/slope at the acquisition start
    (None without a dwell time or for a non-decaying fit).
    """

    amplitude: float
    first_point: float
    start_index: int
    points: int
    order: int
    t2star_s: Optional[float]


@dataclass(frozen=True)
class NoiseEstimate:
    """Per-quadrature noise std of the row-averaged data.

    ``std_per_row`` is the std of one repetition, ``std`` that of the mean of
    ``averaged_rows`` repetitions. ``residual_ratio`` compares the time-domain
    std with the spectral-median std; well above 1 means signal or spurs.
    """

    std: float
    std_per_row: float
    samples: int
    method: str
    averaged_rows: int
    residual_ratio: float


def load_measured_scan(path) -> MeasuredScan:
    """Read a ParaVision scan folder or file, or a complex data array."""
    path = Path(path)
    if path.is_dir() or path.name in _BRUKER_FILES:
        return read_bruker_scan(path)
    if path.suffix.lower() in _ARRAY_SUFFIXES:
        return read_complex_array(path)
    if (path.parent / "acqp").is_file():
        return read_bruker_scan(path)
    raise ValueError(
        f"Unsupported measured data file {path.name!r}: choose a ParaVision "
        "fid/rawdata.job0 or a .npy/.txt/.csv array"
    )


def read_complex_array(path) -> MeasuredScan:
    """Read complex samples from .npy, or real/imaginary columns from text.

    A 2-D complex array holds one repetition per row. A real array whose last
    axis has two entries is read as real and imaginary parts, as is a text
    file with two columns (one FID).
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    else:
        delimiter = "," if path.suffix.lower() == ".csv" else None
        try:
            values = np.loadtxt(path, delimiter=delimiter, ndmin=2)
        except ValueError:
            values = np.loadtxt(path, delimiter=delimiter, dtype=complex, ndmin=2)
        if values.shape[1] == 1:
            values = values[:, 0]
    values = np.asarray(values)
    if not np.iscomplexobj(values):
        if values.shape[-1] != 2:
            raise ValueError(
                "Real data need a last axis of two (real, imaginary) entries"
            )
        values = values[..., 0] + 1j * values[..., 1]
    return MeasuredScan(data=values, source=str(path))


def simulator_nucleus(name):
    """Simulator nucleus name (``C13``) of a ParaVision name such as ``13C``."""
    text = normalize_nucleus(name)
    match = re.match(r"^(\d+)([A-Z][a-z]?)$", text)
    return f"{match.group(2)}{match.group(1)}" if match else text


def _bruker_folder(path):
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    if not (folder / "acqp").is_file():
        raise ValueError(f"No ParaVision acqp file next to {path}")
    return folder


def _raw_file(path, folder):
    path = Path(path)
    if path.is_file() and path.name not in ("acqp", "method"):
        return path
    for name in ("fid", "rawdata.job0"):
        if (folder / name).is_file():
            return folder / name
    raise ValueError(f"No fid or rawdata.job0 in {folder}")


def _first(value):
    if value is None:
        return None
    array = np.atleast_1d(np.asarray(value, dtype=object))
    try:
        return float(array[0])
    except (TypeError, ValueError):
        return None


def _receiver_count(acqp, method):
    count = _first(method.get("PVM_EncNReceivers")) if method else None
    selected = acqp.get("ACQ_ReceiverSelect")
    if selected is not None:
        flags = [str(v) for v in np.atleast_1d(np.asarray(selected, dtype=object))]
        count = max(count or 0, sum(flag == "Yes" for flag in flags))
    return int(count or 1)


def _pulse_length_s(method):
    pulse = method.get("ExcPulse1") if method else None
    if isinstance(pulse, (tuple, list)) and pulse:
        try:
            length = float(pulse[0])
        except (TypeError, ValueError):
            return None
        return length * 1e-3 if length > 0 else None
    return None


def read_bruker_scan(path) -> MeasuredScan:
    """Read a 1-D ParaVision acquisition (pulse-acquire, noise scan).

    ``path`` is the scan folder or one of its fid/rawdata.job0/acqp/method
    files. Multi-receiver and multi-dimensional acquisitions are rejected.
    """
    folder = _bruker_folder(path)
    acqp = read_jcamp(folder / "acqp")
    method = read_jcamp(folder / "method") if (folder / "method").is_file() else None
    raw = _raw_file(path, folder)

    dimensions = int(_first(acqp.get("ACQ_dim")) or 1)
    if dimensions != 1:
        raise ValueError(
            f"Only 1-D acquisitions are supported; {folder.name} has ACQ_dim "
            f"{dimensions}"
        )
    receivers = _receiver_count(acqp, method)
    if receivers > 1:
        raise ValueError(
            f"Only single-receiver data are supported; {folder.name} has "
            f"{receivers} receivers"
        )
    format_name = acqp.text("GO_raw_data_format", "GO_32BIT_SGN_INT")
    if format_name not in _BRUKER_DTYPES:
        raise ValueError(f"Unsupported raw data format {format_name}")
    order = ">" if acqp.text("BYTORDA", "little").lower() == "big" else "<"
    dtype = np.dtype(order + _BRUKER_DTYPES[format_name])
    size = _first(acqp.get("ACQ_size"))
    if not size or int(size) % 2:
        raise ValueError("ACQ_size must give an even number of real/imaginary values")
    complex_points = int(size) // 2
    row_bytes = complex_points * 2 * dtype.itemsize
    if raw.name == "fid" and acqp.text("GO_block_size") == "Standard_KBlock_Format":
        row_bytes = int(np.ceil(row_bytes / 1024.0)) * 1024
    payload = raw.read_bytes()
    if not payload or len(payload) % row_bytes:
        raise ValueError(
            f"{raw.name} holds {len(payload)} bytes, not whole rows of {row_bytes}"
        )
    rows = np.frombuffer(payload, dtype=dtype).reshape(
        len(payload) // row_bytes, row_bytes // dtype.itemsize
    )[:, : 2 * complex_points]
    values = rows[:, 0::2].astype(float) + 1j * rows[:, 1::2].astype(float)

    bandwidth = _first(acqp.get("SW_h"))
    nucleus = acqp.text("NUC1") or (method.text("PVM_Nucleus1") if method else "")
    frequency = _first(acqp.get("BF1"))
    repetition_ms = _first(method.get("PVM_RepetitionTime")) if method else None
    if repetition_ms is None:
        repetition_ms = _first(acqp.get("ACQ_repetition_time"))
    dummy = _first(method.get("PVM_DummyScans")) if method else None
    if dummy is None:
        dummy = _first(acqp.get("DS"))
    pulse = _pulse_length_s(method)
    dead_time = _first(method.get("DeadTime")) if method else None
    delay = None
    if pulse is not None and dead_time is not None:
        delay = 0.5 * pulse + dead_time * 1e-3
    return MeasuredScan(
        data=values,
        dwell_s=1.0 / bandwidth if bandwidth else None,
        group_delay_points=float(_first(acqp.get("GRPDLY")) or 0.0),
        averages=int(_first(acqp.get("NA")) or 1),
        receiver_gain=_first(acqp.get("RG")),
        nucleus=simulator_nucleus(nucleus) if nucleus else None,
        frequency_hz=None if frequency is None else frequency * 1e6,
        flip_angle_deg=_first(acqp.get("ACQ_flip_angle")),
        repetition_time_s=None if repetition_ms is None else repetition_ms * 1e-3,
        dummy_scans=None if dummy is None else int(dummy),
        excitation_delay_s=delay,
        source=str(raw),
    )


def _fit_start(scan):
    delay = float(scan.group_delay_points)
    return int(np.ceil(delay)) + 4 if delay > 0 else 0


def estimate_fid_amplitude(
    scan: MeasuredScan, *, max_points=256, min_points=8, drop_fraction=0.6
) -> AmplitudeEstimate:
    """Extrapolate |FID| of the row-averaged data to the acquisition start.

    The fit starts four samples after the group delay and runs until the
    smoothed magnitude drops below ``drop_fraction`` of its first value.
    Adjacent samples are averaged, which cancels the filter ringing that
    alternates at the Nyquist frequency. log|FID| is fitted with |FID|^2
    weights, quadratically from 12 points on.
    """
    signal = np.abs(scan.averaged())
    start = _fit_start(scan)
    if scan.samples - start <= min_points:
        raise ValueError("The FID is too short after the digital filter delay")
    smooth = np.convolve(signal, np.ones(4) / 4, mode="valid")
    reference = smooth[start] if start < smooth.size else signal[start]
    below = np.nonzero(smooth[start:] < drop_fraction * reference)[0]
    stop = start + (below[0] if below.size else smooth.size - start)
    stop = int(
        np.clip(stop, start + min_points, min(start + max_points, scan.samples - 1))
    )
    index = np.arange(start, stop)
    magnitude = 0.5 * (signal[index] + signal[index + 1])
    if np.any(magnitude <= 0):
        raise ValueError("The FID magnitude vanishes inside the fit window")
    # Time from the acquisition start in samples; seconds with a dwell time.
    x = index + 0.5 - float(scan.group_delay_points)
    order = 2 if index.size >= 12 else 1
    coefficients = np.polyfit(x, np.log(magnitude), order, w=magnitude)
    amplitude = float(np.exp(np.polyval(coefficients, 0.0)))
    slope = coefficients[-2]
    t2star = None
    if scan.dwell_s is not None and slope < 0:
        t2star = float(-scan.dwell_s / slope)
    return AmplitudeEstimate(
        amplitude=amplitude,
        first_point=float(signal[start]),
        start_index=start,
        points=int(index.size),
        order=order,
        t2star_s=t2star,
    )


def _spectral_std(rows):
    """Per-quadrature std from the median periodogram of complex rows.

    A Hann window keeps the leakage of a residual line out of the median.
    Every bin of white noise is complex Gaussian, so |X|^2 is exponential
    with median 2 ln 2 sigma^2 times the window energy per sample.
    """
    window = np.hanning(rows.shape[-1] + 2)[1:-1]
    power = np.abs(np.fft.fft(rows * window, axis=-1)) ** 2
    energy = np.sum(window**2)
    return float(np.sqrt(np.median(power) / (2 * np.log(2) * energy)))


def _time_std(rows):
    centred = rows - rows.mean(axis=-1, keepdims=True)
    dof = rows.size - rows.shape[0]
    return float(np.sqrt(np.sum(np.abs(centred) ** 2) / (2 * dof)))


def estimate_noise_std(
    scan: MeasuredScan,
    *,
    from_tail=False,
    tail_fraction=0.25,
    averaged_rows=None,
) -> NoiseEstimate:
    """Per-quadrature noise std of the mean of ``averaged_rows`` repetitions.

    A noise scan (no excitation) uses the time-domain std of every row after
    the filter has settled (two group delays) and its mean removed.
    ``from_tail`` instead uses the last ``tail_fraction`` of every FID row:
    its time-domain std while that is white noise, or else the robust
    spectral median, which ignores a residual line (``residual_ratio`` above
    the limit).
    ``averaged_rows`` defaults to the scan's repetitions; pass the FID's
    repetition count for a separate noise scan.
    """
    rows = averaged_rows if averaged_rows is not None else scan.repetitions
    if int(rows) != rows or rows < 1:
        raise ValueError("averaged_rows must be a positive integer")
    if from_tail:
        if not 0 < tail_fraction <= 1:
            raise ValueError("tail_fraction must be in (0, 1]")
        count = max(16, int(round(tail_fraction * scan.samples)))
        start = max(_fit_start(scan), scan.samples - count)
    else:
        start = int(np.ceil(_FILTER_SETTLING * float(scan.group_delay_points)))
    segment = scan.data[:, start:]
    if segment.shape[1] < 16:
        raise ValueError("Too few noise samples after the digital filter delay")
    spectral = _spectral_std(segment)
    temporal = _time_std(segment)
    ratio = temporal / spectral if spectral > 0 else np.inf
    per_row = spectral if from_tail and ratio > _RESIDUAL_LIMIT else temporal
    if not per_row > 0:
        raise ValueError("The noise data have zero variance")
    return NoiseEstimate(
        std=per_row / np.sqrt(rows),
        std_per_row=per_row,
        samples=int(segment.size),
        method="FID tail" if from_tail else "noise scan",
        averaged_rows=int(rows),
        residual_ratio=ratio,
    )


def _differs(first, second, rel=1e-6):
    if first is None or second is None:
        return False
    return not np.isclose(first, second, rtol=rel, atol=0.0)


def check_noise_scan(fid: MeasuredScan, noise: MeasuredScan):
    """Raise unless the noise scan shares NA, receiver gain and bandwidth."""
    problems = []
    if fid.averages != noise.averages:
        problems.append(f"NA {noise.averages} vs {fid.averages}")
    if _differs(fid.receiver_gain, noise.receiver_gain):
        problems.append(f"RG {noise.receiver_gain:g} vs {fid.receiver_gain:g}")
    if _differs(fid.dwell_s, noise.dwell_s):
        problems.append(
            f"bandwidth {noise.bandwidth_hz:.6g} Hz vs {fid.bandwidth_hz:.6g} Hz"
        )
    if problems:
        raise ValueError(
            "The noise scan must use the FID's averages, receiver gain and "
            "bandwidth: " + ", ".join(problems)
        )


@dataclass(frozen=True)
class MeasuredCalibrationInputs:
    """Receiver-calibration inputs read from a measured FID (and noise scan)."""

    amplitude: AmplitudeEstimate
    noise: NoiseEstimate
    averages: int
    dwell_s: Optional[float]
    flip_angle_deg: Optional[float]
    repetition_time_s: Optional[float]
    dummy_scans: Optional[int]
    amplitude_delay_s: Optional[float]
    nucleus: Optional[str]
    field_strength_t: Optional[float]
    warnings: Tuple[str, ...] = ()

    @property
    def snr(self):
        return self.amplitude.amplitude / self.noise.std

    def summary(self):
        parts = [
            f"A = {self.amplitude.amplitude:.4g} (fit of {self.amplitude.points} pts)",
            f"σ = {self.noise.std:.4g} ({self.noise.method})",
            f"SNR = {self.snr:.4g} with {self.averages} average(s)",
        ]
        if self.amplitude.t2star_s is not None:
            parts.append(f"T2* ≈ {self.amplitude.t2star_s * 1e3:.3g} ms")
        return "; ".join(parts) + "".join(f"\n⚠ {text}" for text in self.warnings)


def measured_calibration_inputs(
    fid: MeasuredScan, noise: Optional[MeasuredScan] = None
) -> MeasuredCalibrationInputs:
    """Amplitude, noise and acquisition settings for a receiver calibration.

    ``noise`` is a scan without excitation; without it the FID tail is used.
    """
    amplitude = estimate_fid_amplitude(fid)
    warnings = []
    if noise is not None:
        check_noise_scan(fid, noise)
        estimate = estimate_noise_std(noise, averaged_rows=fid.repetitions)
        if estimate.residual_ratio > _RESIDUAL_LIMIT:
            warnings.append(
                "The noise scan is not white noise (time/spectral std "
                f"{estimate.residual_ratio:.2f}); check for signal or spurs"
            )
    else:
        estimate = estimate_noise_std(fid, from_tail=True)
        if estimate.residual_ratio > _RESIDUAL_LIMIT:
            warnings.append(
                "Signal remains in the FID tail (time/spectral std "
                f"{estimate.residual_ratio:.2f}); a 0° noise scan is more reliable"
            )
    snr = amplitude.amplitude / estimate.std
    if snr < 20:
        warnings.append(
            f"SNR {snr:.3g} is low; the log-magnitude fit overestimates the amplitude"
        )
    return MeasuredCalibrationInputs(
        amplitude=amplitude,
        noise=estimate,
        averages=fid.total_averages,
        dwell_s=fid.dwell_s,
        flip_angle_deg=fid.flip_angle_deg,
        repetition_time_s=fid.repetition_time_s,
        dummy_scans=fid.dummy_scans,
        amplitude_delay_s=fid.excitation_delay_s,
        nucleus=fid.nucleus,
        field_strength_t=fid.field_strength_t,
        warnings=tuple(warnings),
    )


__all__ = [
    "AmplitudeEstimate",
    "MeasuredCalibrationInputs",
    "MeasuredScan",
    "NoiseEstimate",
    "check_noise_scan",
    "estimate_fid_amplitude",
    "estimate_noise_std",
    "load_measured_scan",
    "measured_calibration_inputs",
    "read_bruker_scan",
    "read_complex_array",
    "simulator_nucleus",
]
