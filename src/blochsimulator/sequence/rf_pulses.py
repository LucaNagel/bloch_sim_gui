"""Shared RF waveform design for generated 2D imaging sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np


RF_PULSE_TYPES = ("sinc", "slr", "block", "designer")
SLR_SHARPNESS_VALUES = (1, 5)


def normalize_rf_pulse_type(value: str) -> str:
    """Return the canonical RF pulse type accepted by the 2D builders."""
    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "sinc": "sinc",
        "slr": "slr",
        "block": "block",
        "block pulse": "block",
        "hard": "block",
        "hard pulse": "block",
        "rect": "block",
        "rectangle": "block",
        "rectangular": "block",
        "designer": "designer",
        "rf pulse designer": "designer",
        "custom": "designer",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(RF_PULSE_TYPES)
        raise ValueError(f"rf_pulse_type must be one of: {choices}") from exc


def _slr_waveform_path(sharpness: float) -> Path:
    if not np.isfinite(sharpness) or sharpness <= 0:
        raise ValueError("rf_slr_sharpness must be positive and finite")
    rounded = int(round(float(sharpness)))
    if not np.isclose(sharpness, rounded) or rounded not in SLR_SHARPNESS_VALUES:
        choices = " or ".join(str(value) for value in SLR_SHARPNESS_VALUES)
        raise ValueError(f"rf_slr_sharpness must be {choices}")

    filename = f"SLR_sharpness_{rounded}.txt"
    module_path = Path(__file__).resolve()
    candidates = (
        # PyInstaller places RF assets below the package directory.
        module_path.parents[1] / "rfpulses" / filename,
        # Editable/source checkout.
        module_path.parents[3] / "rfpulses" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"bundled SLR waveform {filename!r} was not found; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_slr_waveform(sharpness: float) -> np.ndarray:
    data = np.loadtxt(_slr_waveform_path(sharpness), delimiter=",")
    flat = np.asarray(data, dtype=float).reshape(-1)
    if flat.size < 2 or flat.size % 2:
        raise ValueError("SLR pulse file must contain amplitude/phase pairs")
    signal = flat[0::2] * np.exp(1j * np.deg2rad(flat[1::2]))
    if not np.any(np.abs(signal) > 0):
        raise ValueError("SLR pulse waveform is empty")
    return signal.astype(np.complex128)


def _resample_complex(signal: np.ndarray, sample_count: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if signal.size == sample_count:
        return signal.copy()
    source = (np.arange(signal.size, dtype=float) + 0.5) / signal.size
    target = (np.arange(sample_count, dtype=float) + 0.5) / sample_count
    real = np.interp(target, source, signal.real)
    imag = np.interp(target, source, signal.imag)
    return real + 1j * imag


def design_rf_envelope(
    *,
    pulse_type: str,
    duration_s: float,
    raster_s: float,
    time_bandwidth_product: float = 4.0,
    apodization: float = 0.5,
    slr_sharpness: float = 1.0,
    custom_waveform: np.ndarray | None = None,
    custom_raster_s: float | None = None,
) -> tuple[np.ndarray, float, float, str]:
    """Design a baseband envelope on an RF raster.

    The returned waveform has an arbitrary amplitude but a positive real
    integral. Consumers scale it to the requested flip angle. The third return
    value is the effective time-bandwidth product used for slice selection.
    """
    pulse_type = normalize_rf_pulse_type(pulse_type)
    for name, value in {
        "duration_s": duration_s,
        "raster_s": raster_s,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not np.isfinite(time_bandwidth_product) or time_bandwidth_product <= 0:
        raise ValueError("rf_time_bandwidth_product must be positive and finite")
    if not np.isfinite(apodization) or not 0 <= apodization <= 1:
        raise ValueError("rf_apodization must be between 0 and 1")

    sample_count = int(round(float(duration_s) / float(raster_s)))
    if sample_count <= 0:
        raise ValueError("rf_duration_s must span at least one RF raster interval")
    actual_duration_s = sample_count * float(raster_s)
    effective_tbw = 1.0 if pulse_type == "block" else float(time_bandwidth_product)

    if pulse_type == "block":
        signal = np.ones(sample_count, dtype=np.complex128)
    elif pulse_type == "sinc":
        time_s = (np.arange(sample_count, dtype=float) + 0.5) * float(raster_s)
        centered_s = time_s - actual_duration_s / 2.0
        window = (
            1.0
            - float(apodization)
            + float(apodization) * np.cos(2.0 * np.pi * centered_s / actual_duration_s)
        )
        signal = window * np.sinc(effective_tbw * centered_s / actual_duration_s)
        signal = signal.astype(np.complex128)
    elif pulse_type == "slr":
        signal = _resample_complex(_load_slr_waveform(slr_sharpness), sample_count)
    else:
        if custom_waveform is None:
            raise ValueError(
                "rf_custom_waveform_hz is required when rf_pulse_type is 'designer'"
            )
        if custom_raster_s is None or not np.isfinite(custom_raster_s):
            raise ValueError(
                "rf_custom_raster_s must be positive and finite for a designer pulse"
            )
        if custom_raster_s <= 0:
            raise ValueError(
                "rf_custom_raster_s must be positive and finite for a designer pulse"
            )
        custom = np.asarray(custom_waveform, dtype=np.complex128).reshape(-1)
        if custom.size == 0 or not np.all(np.isfinite(custom)):
            raise ValueError("rf_custom_waveform_hz must be non-empty and finite")
        source_duration_s = custom.size * float(custom_raster_s)
        if not np.isclose(
            source_duration_s,
            duration_s,
            rtol=0.0,
            atol=max(float(raster_s), float(custom_raster_s)) / 2.0,
        ):
            raise ValueError(
                "rf_duration_s must match the RF Pulse Designer waveform duration"
            )
        signal = _resample_complex(custom, sample_count)

    integral = np.sum(signal)
    if not np.isfinite(integral):
        raise ValueError("RF waveform has a non-finite integral")
    if pulse_type != "designer" and abs(integral) < 1e-12:
        raise ValueError("RF waveform has zero or non-finite integral")
    if pulse_type != "designer":
        signal *= np.exp(-1j * np.angle(integral))
    signal.setflags(write=False)
    return signal, actual_duration_s, effective_tbw, pulse_type


def scale_rf_envelope_to_flip(
    envelope: np.ndarray,
    *,
    flip_angle_deg: float,
    raster_s: float,
    reference_flip_angle_deg: float | None = None,
) -> np.ndarray:
    """Scale a baseband envelope to nutation-frequency samples in hertz."""
    if not np.isfinite(flip_angle_deg):
        raise ValueError("flip_angle_deg must be finite")
    signal = np.asarray(envelope, dtype=np.complex128)
    if reference_flip_angle_deg is not None:
        if not np.isfinite(reference_flip_angle_deg) or reference_flip_angle_deg <= 0:
            raise ValueError("rf_custom_flip_angle_deg must be positive and finite")
        samples_hz = signal * (float(flip_angle_deg) / reference_flip_angle_deg)
    else:
        integral_s = np.sum(signal) * float(raster_s)
        if abs(integral_s) < 1e-15:
            raise ValueError("RF waveform has zero integral")
        target_cycles = float(flip_angle_deg) / 360.0
        samples_hz = signal * (target_cycles / abs(integral_s))
    samples_hz.setflags(write=False)
    return samples_hz
