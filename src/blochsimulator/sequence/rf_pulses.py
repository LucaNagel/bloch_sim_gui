"""Shared RF waveform design used by Free Mode and every sequence builder.

This module is deliberately independent of the desktop UI.  It is the single
place where analytic RF envelopes, including SLR beta polynomials, are
generated.  Pulseq builders and the Free Mode designer only scale or package
the resulting baseband waveform.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import remez
from scipy.signal.windows import tukey


RF_PULSE_TYPES = ("sinc", "slr", "gaussian", "block", "designer")
RF_PULSE_TYPE_LABELS = (
    "Sinc",
    "SLR",
    "Gaussian",
    "Block",
    "RF Pulse Designer",
)
DEFAULT_ANALYTIC_RF_SHAPE_PARAMETER = 4.0


def normalize_rf_pulse_type(value: str) -> str:
    """Return the canonical RF pulse type accepted by the 2D builders."""
    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "sinc": "sinc",
        "slr": "slr",
        "gauss": "gaussian",
        "gaussian": "gaussian",
        "gaussian pulse": "gaussian",
        "block": "block",
        "block pulse": "block",
        "hard": "block",
        "hard pulse": "block",
        "rect": "block",
        "rectangle": "block",
        "rectangular": "block",
        "designer": "designer",
        "rf pulse designer": "designer",
        "rf pulse designer / loaded file": "designer",
        "loaded file": "designer",
        "custom": "designer",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(RF_PULSE_TYPES)
        raise ValueError(f"rf_pulse_type must be one of: {choices}") from exc


def rf_envelope_integration_factor(envelope: np.ndarray) -> float:
    """Return the coherent normalized area of an RF envelope.

    The result is invariant to pulse amplitude, constant phase, duration, and
    RF raster. A block pulse therefore has an integration factor of one.
    """
    signal = np.asarray(envelope, dtype=np.complex128).reshape(-1)
    if signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("RF envelope must be non-empty and finite")
    peak = float(np.max(np.abs(signal)))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("RF envelope must have a positive finite peak")
    factor = float(abs(np.sum(signal / peak)) / signal.size)
    if not np.isfinite(factor) or factor <= np.finfo(float).eps:
        raise ValueError("RF envelope has no finite coherent integral")
    return factor


def rf_time_bandwidth_product_from_envelope(envelope: np.ndarray) -> float:
    """Estimate the shape-intrinsic TBW as inverse integration factor."""
    return 1.0 / rf_envelope_integration_factor(envelope)


def analytic_rf_shape_parameter(pulse_type: str, sinc_lobes: int = 3) -> float:
    """Return the non-editable construction parameter for an analytic shape."""
    normalized = normalize_rf_pulse_type(pulse_type)
    if normalized == "sinc":
        return float(max(1, int(sinc_lobes)) + 1)
    if normalized in {"slr", "gaussian"}:
        return DEFAULT_ANALYTIC_RF_SHAPE_PARAMETER
    return 1.0


def _validate_slr_sharpness(sharpness: float) -> float:
    if not np.isfinite(sharpness) or sharpness <= 0:
        raise ValueError("rf_slr_sharpness must be positive and finite")
    return float(sharpness)


@lru_cache(maxsize=64)
def _design_slr_waveform(
    sample_count: int, time_bandwidth_product: float, sharpness: float
) -> np.ndarray:
    """Design a linear-phase, small-tip SLR beta polynomial.

    ``sharpness`` continuously controls the relative transition width and the
    stop-band weighting.  The result is cached because sequence previews often
    request the same RF shape for several flip angles and slices.
    """
    sharpness = _validate_slr_sharpness(sharpness)
    if sample_count < 8:
        raise ValueError("an SLR pulse requires at least 8 RF samples")
    tbw = float(time_bandwidth_product)
    if tbw >= sample_count:
        raise ValueError(
            "rf_time_bandwidth_product must be smaller than the SLR sample count"
        )

    # ``sharpness`` historically selected bundled SLR waveforms whose number
    # of temporal lobes increased with the setting.  The first dynamic design
    # accidentally omitted that factor from the beta-polynomial bandwidth, so
    # only the ripple weighting changed and every setting looked nearly the
    # same.  Scaling the beta pass band restores the intended lobe progression
    # while the progressively narrower transition retains the sharper slice
    # profile expected from the control.
    designed_tbw = tbw * sharpness
    if designed_tbw >= sample_count:
        raise ValueError(
            "rf_time_bandwidth_product * rf_slr_sharpness must be smaller "
            "than the SLR sample count"
        )
    band_center = designed_tbw / (2.0 * sample_count)
    transition_fraction = 1.0 / (sharpness + 1.0)
    passband_edge = band_center * (1.0 - transition_fraction)
    stopband_edge = band_center * (1.0 + transition_fraction)
    if passband_edge <= 0 or stopband_edge >= 0.5:
        raise ValueError("SLR pulse parameters leave no valid transition band")

    try:
        beta = remez(
            sample_count,
            (0.0, passband_edge, stopband_edge, 0.5),
            (1.0, 0.0),
            weight=(1.0, sharpness),
            fs=1.0,
            maxiter=100,
        )
    except ValueError as exc:
        raise ValueError(
            "SLR filter design did not converge; use a longer duration, lower "
            "time-bandwidth product, or lower sharpness"
        ) from exc
    # Very narrow equiripple designs can develop numerically amplified first
    # and last coefficients when evaluated directly at a fine scanner raster.
    # RF hardware also benefits from a waveform that starts and ends at zero,
    # so apply a symmetric cosine edge taper before flip-angle normalization.
    signal = np.asarray(beta * tukey(sample_count, alpha=0.2), dtype=np.complex128)
    if not np.all(np.isfinite(signal)) or not np.any(np.abs(signal) > 0):
        raise ValueError("SLR filter design produced an invalid waveform")
    signal.setflags(write=False)
    return signal


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
    integral. Consumers scale it to the requested flip angle. For analytic
    pulses, the input ``time_bandwidth_product`` is retained as a construction
    parameter for backward compatibility; the third return value is calculated
    from the completed normalized pulse shape and is used for slice selection.
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
    shape_parameter = 1.0 if pulse_type == "block" else float(time_bandwidth_product)

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
        signal = window * np.sinc(shape_parameter * centered_s / actual_duration_s)
        signal = signal.astype(np.complex128)
    elif pulse_type == "slr":
        signal = _design_slr_waveform(
            sample_count, shape_parameter, slr_sharpness
        ).copy()
    elif pulse_type == "gaussian":
        time_s = (np.arange(sample_count, dtype=float) + 0.5) * float(raster_s)
        centered_s = time_s - actual_duration_s / 2.0
        # The fixed construction parameter defines the normalized Gaussian
        # shape. Stretching it in time changes bandwidth without changing TBW.
        spectral_fwhm_hz = shape_parameter / actual_duration_s
        sigma_s = np.sqrt(2.0 * np.log(2.0)) / (np.pi * spectral_fwhm_hz)
        signal = np.exp(-0.5 * (centered_s / sigma_s) ** 2).astype(np.complex128)
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
        effective_tbw = rf_time_bandwidth_product_from_envelope(signal)
    else:
        # Loaded formats may provide an independently measured bandwidth
        # factor. Callers pass that value through this compatibility parameter.
        effective_tbw = float(time_bandwidth_product)
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


def make_pulseq_rf_events(
    pp,
    system,
    *,
    flip_angles_deg,
    pulse_type: str,
    duration_s: float,
    time_bandwidth_product: float = 4.0,
    apodization: float = 0.5,
    slr_sharpness: float = 1.0,
    custom_waveform_hz=None,
    custom_raster_s: float | None = None,
    custom_flip_angle_deg: float | None = None,
    slice_thickness_m: float | None = None,
    frequency_offset_hz: float = 0.0,
    use: str = "excitation",
    center_s: float | None = None,
):
    """Create Pulseq RF events from the shared envelope designer.

    When ``slice_thickness_m`` is supplied, each result is the
    ``(rf, slice_gradient)`` tuple returned by Pulseq.  Otherwise
    each result is a non-selective RF event.  Loaded/designer waveforms retain
    their complex phase and are rescaled from their reference flip angle.
    """
    angles = tuple(float(value) for value in flip_angles_deg)
    if not angles or not np.all(np.isfinite(angles)):
        raise ValueError("flip_angles_deg must contain finite values")
    if not np.isfinite(frequency_offset_hz):
        raise ValueError("rf_frequency_offset_hz must be finite")
    if slice_thickness_m is not None and (
        not np.isfinite(slice_thickness_m) or slice_thickness_m <= 0
    ):
        raise ValueError("slice_thickness_m must be positive and finite")

    envelope, actual_duration_s, effective_tbw, normalized_type = design_rf_envelope(
        pulse_type=pulse_type,
        duration_s=duration_s,
        raster_s=system.rf_raster_time,
        time_bandwidth_product=time_bandwidth_product,
        apodization=apodization,
        slr_sharpness=slr_sharpness,
        custom_waveform=custom_waveform_hz,
        custom_raster_s=custom_raster_s,
    )
    loaded = normalized_type == "designer"
    results = []
    for angle_deg in angles:
        signal = (
            scale_rf_envelope_to_flip(
                envelope,
                flip_angle_deg=angle_deg,
                raster_s=system.rf_raster_time,
                reference_flip_angle_deg=custom_flip_angle_deg,
            )
            if loaded
            else envelope
        )
        kwargs = {
            "signal": signal,
            "flip_angle": np.deg2rad(angle_deg),
            "dwell": system.rf_raster_time,
            "bandwidth": effective_tbw / actual_duration_s,
            "time_bw_product": effective_tbw,
            "no_signal_scaling": loaded,
            "delay": system.rf_dead_time,
            "system": system,
            "use": use,
        }
        if center_s is not None:
            if not np.isfinite(center_s) or center_s < 0:
                raise ValueError("rf center_s must be non-negative and finite")
            kwargs["center"] = float(center_s)
        if slice_thickness_m is not None:
            kwargs.update(
                slice_thickness=slice_thickness_m,
                return_gz=True,
            )
        event = pp.make_arbitrary_rf(**kwargs)
        rf_event = event[0] if slice_thickness_m is not None else event
        rf_event.freq_offset = float(frequency_offset_hz)
        results.append(event)
    return tuple(results), actual_duration_s, effective_tbw, normalized_type


def set_rf_definitions(
    sequence,
    *,
    pulse_type: str,
    requested_duration_s: float,
    actual_duration_s: float,
    time_bandwidth_product: float,
    apodization: float,
    slr_sharpness: float,
    custom_name: str | None,
    custom_flip_angle_deg: float | None,
    frequency_offset_hz: float,
    prefix: str = "",
) -> None:
    """Persist one consistent set of reproducible RF definitions."""
    key = str(prefix)
    sequence.set_definition(f"{key}RFPulseType", pulse_type)
    sequence.set_definition(f"{key}RFDuration", actual_duration_s)
    sequence.set_definition(f"{key}RequestedRFDuration", requested_duration_s)
    sequence.set_definition(f"{key}RFTimeBandwidthProduct", time_bandwidth_product)
    sequence.set_definition(
        f"{key}RFBandwidth", time_bandwidth_product / actual_duration_s
    )
    sequence.set_definition(f"{key}RFFrequencyOffset", frequency_offset_hz)
    if pulse_type == "sinc":
        sequence.set_definition(f"{key}RFApodization", apodization)
    if pulse_type == "slr":
        sequence.set_definition(f"{key}RFSLRSharpness", slr_sharpness)
        # Retain the historical spelling used by spectral sequence readers.
        if key == "Spectral":
            sequence.set_definition("SpectralSLRSharpness", slr_sharpness)
    if pulse_type == "designer":
        sequence.set_definition(f"{key}RFDesignerPulseName", custom_name or "custom")
        if custom_flip_angle_deg is not None:
            sequence.set_definition(
                f"{key}RFDesignerFlipAngleDeg", custom_flip_angle_deg
            )
