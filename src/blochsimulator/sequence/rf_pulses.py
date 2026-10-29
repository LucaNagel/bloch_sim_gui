"""Shared RF waveform design used by Free Mode and every sequence builder.

This module is deliberately independent of the desktop UI.  It is the single
place where analytic RF envelopes, including SLR beta polynomials, are
generated.  Pulseq builders and the Free Mode designer only scale or package
the resulting baseband waveform.
"""

from __future__ import annotations

from copy import copy
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import firls

from ..units import NUCLEUS_GAMMA_HZ_PER_T
from .spectral_spatial import SpectralSpatialPulse


RF_PULSE_TYPES = (
    "sinc",
    "slr",
    "gaussian",
    "block",
    "designer",
    "spectral_spatial",
)
RF_PULSE_TYPE_LABELS = (
    "Sinc",
    "SLR",
    "Gaussian",
    "Block",
    "RF Pulse Designer",
    "Spectral-spatial Designer",
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
        "spectral spatial": "spectral_spatial",
        "spectral spatial designer": "spectral_spatial",
        "spsp": "spectral_spatial",
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


def _validate_slr_sharpness(sharpness: int) -> int:
    if (
        isinstance(sharpness, (bool, np.bool_))
        or not np.isfinite(sharpness)
        or float(sharpness) <= 0
        or not float(sharpness).is_integer()
    ):
        raise ValueError("rf_slr_sharpness must be a positive integer")
    return int(sharpness)


# Passband and stopband ripple of the SLR beta polynomial. One percent is the
# customary choice for excitation pulses (Pauly et al., IEEE TMI 1991).
SLR_PASSBAND_RIPPLE = 0.01
SLR_STOPBAND_RIPPLE = 0.01
# Odd tap count of the reference design. The design is completed on this
# fixed grid and resampled to the RF raster, so even and odd raster sample
# counts describe the same continuous pulse.
_SLR_MIN_DESIGN_SAMPLES = 511


def slr_time_bandwidth_product(sharpness: int) -> float:
    """Return the time-bandwidth product designed for an SLR sharpness.

    Sharpness one is TBW 2, a single positive lobe. Each further step adds one
    unit of TBW, so the slice profile becomes sharper and the pulse gains
    temporal side lobes.
    """
    return float(_validate_slr_sharpness(sharpness) + 1)


def _slr_transition_width(passband_ripple: float, stopband_ripple: float) -> float:
    """Return Pauly's D_inf, the transition width times the TBW."""
    l1 = np.log10(passband_ripple)
    l2 = np.log10(stopband_ripple)
    a1, a2, a3 = 5.309e-3, 7.114e-2, -4.761e-1
    a4, a5, a6 = -2.66e-3, -5.941e-1, -4.278e-1
    return float((a1 * l1 * l1 + a2 * l1 + a3) * l2 + (a4 * l1 * l1 + a5 * l1 + a6))


def _slr_beta(tap_count: int, time_bandwidth_product: float) -> np.ndarray:
    """Design the linear-phase least-squares SLR beta polynomial."""
    tbw = float(time_bandwidth_product)
    transition = min(
        _slr_transition_width(SLR_PASSBAND_RIPPLE, SLR_STOPBAND_RIPPLE) / tbw, 0.9
    )
    half = tap_count / 2.0
    edges = np.array(
        (0.0, (1.0 - transition) * tbw / 2.0, (1.0 + transition) * tbw / 2.0, half)
    )
    return firls(
        tap_count,
        edges / half,
        (1.0, 1.0, 0.0, 0.0),
        weight=(1.0, SLR_PASSBAND_RIPPLE / SLR_STOPBAND_RIPPLE),
    )


def _slr_alpha(beta: np.ndarray) -> np.ndarray:
    """Return the minimum-phase alpha polynomial matching ``beta``.

    |A|^2 + |B|^2 = 1 fixes the magnitude of A; its minimum phase follows from
    the folded cepstrum on a zero-padded frequency grid.
    """
    pad = 16 * beta.size
    beta_f = np.fft.fft(beta, pad)
    peak = np.max(np.abs(beta_f))
    if peak >= 1.0:
        beta_f /= peak * (1.0 + 1e-9)
    log_magnitude = np.log(np.maximum(np.sqrt(1.0 - np.abs(beta_f) ** 2), 1e-300))
    cepstrum = np.fft.fft(log_magnitude)
    cepstrum[1 : pad // 2] *= 2.0
    cepstrum[pad // 2 + 1 :] = 0.0
    alpha = np.fft.fft(np.exp(np.fft.ifft(cepstrum)))[: beta.size] / pad
    return alpha[::-1]


def _inverse_slr(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return hard-pulse rotation angles (rad) from the SLR polynomials."""
    alpha = np.asarray(alpha, dtype=np.complex128)
    beta = np.asarray(beta, dtype=np.complex128)
    rotations = np.zeros(beta.size, dtype=np.complex128)
    for index in range(beta.size - 1, -1, -1):
        ratio = beta[index] / alpha[index]
        cos_half = np.sqrt(1.0 / (1.0 + np.abs(ratio) ** 2))
        sin_half = np.conj(cos_half * ratio)
        rotations[index] = (
            2.0
            * np.arctan2(np.abs(sin_half), cos_half)
            * np.exp(1j * np.angle(sin_half))
        )
        if index:
            alpha, beta = (
                (cos_half * alpha + sin_half * beta)[1 : index + 1],
                (-np.conj(sin_half) * alpha + cos_half * beta)[:index],
            )
    return rotations


@lru_cache(maxsize=64)
def _design_slr_waveform(
    sample_count: int, sharpness: int, flip_angle_deg: float | None
) -> np.ndarray:
    """Design an SLR pulse on ``sample_count`` RF raster samples.

    A least-squares linear-phase beta polynomial with time-bandwidth product
    :func:`slr_time_bandwidth_product` is scaled to ``sin(flip / 2)``, paired
    with its minimum-phase alpha and converted to RF by the inverse SLR
    transform. Without a flip angle the small-tip limit, the beta polynomial
    itself, is returned. The result has arbitrary amplitude; on resonance its
    area is the flip angle, so consumers scale it by area. It is cached because
    sequence previews often request the same RF shape for several slices.
    """
    tbw = slr_time_bandwidth_product(sharpness)
    if sample_count < 8:
        raise ValueError("an SLR pulse requires at least 8 RF samples")
    if 2.0 * tbw >= sample_count:
        raise ValueError(
            "rf_slr_sharpness is too high for the RF duration; use a longer "
            "pulse or a lower sharpness"
        )

    tap_count = max(_SLR_MIN_DESIGN_SAMPLES, 32 * int(np.ceil(tbw)) + 1)
    beta = _slr_beta(tap_count, tbw)
    flip_rad = 0.0 if flip_angle_deg is None else np.deg2rad(abs(flip_angle_deg))
    if flip_rad <= np.deg2rad(1e-3):
        design = beta
    else:
        # Beyond 180 degrees the beta amplitude cannot grow further.
        beta = beta * (np.sin(min(flip_rad, np.pi) / 2.0) / np.sum(beta))
        design = _inverse_slr(_slr_alpha(beta), beta).real

    # Interpolate the smooth reference design at the centres of the raster
    # intervals. Both grids span the same pulse duration.
    source = (np.arange(tap_count, dtype=float) + 0.5) / tap_count
    target = (np.arange(sample_count, dtype=float) + 0.5) / sample_count
    signal = CubicSpline(source, design)(target).astype(np.complex128)
    if not np.all(np.isfinite(signal)) or not np.any(np.abs(signal) > 0):
        raise ValueError("SLR design produced an invalid waveform")
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
    slr_sharpness: int = 1,
    custom_waveform: np.ndarray | None = None,
    custom_raster_s: float | None = None,
    flip_angle_deg: float | None = None,
) -> tuple[np.ndarray, float, float, str]:
    """Design a baseband envelope on an RF raster.

    The returned waveform has an arbitrary amplitude but a positive real
    integral. Consumers scale it to the requested flip angle. For analytic
    pulses, the input ``time_bandwidth_product`` is retained as a construction
    parameter for backward compatibility; the third return value is calculated
    from the completed normalized pulse shape and is used for slice selection.

    SLR pulses ignore ``time_bandwidth_product``: their TBW follows from
    ``slr_sharpness`` and is returned as designed. Their shape depends on the
    flip angle, so ``flip_angle_deg`` selects the large-tip design; ``None``
    gives the small-tip shape.
    """
    pulse_type = normalize_rf_pulse_type(pulse_type)
    if pulse_type == "spectral_spatial":
        raise ValueError(
            "spectral-spatial pulses require design_spectral_spatial_pulse and "
            "cannot be represented by an RF envelope alone"
        )
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
        # Progress from an unapodized Sinc through the historical Hann default
        # at 0.5 to a Hann-squared window at 1.0. This keeps the window
        # non-negative throughout the documented range. The previous raw
        # cosine became negative above 0.5 and could therefore amplify or
        # invert the outer side lobes.
        hann = 0.5 * (1.0 + np.cos(2.0 * np.pi * centered_s / actual_duration_s))
        apodization = float(apodization)
        if apodization <= 0.5:
            # Preserve the historical 0.5 default exactly (a Hann window),
            # while interpolating from a rectangular window below it.
            window = 1.0 - 2.0 * apodization + 2.0 * apodization * hann
        else:
            # Continue from Hann to Hann squared. Unlike the former raw
            # cosine, this stronger half of the range remains non-negative.
            strength = 2.0 * apodization - 1.0
            window = hann * (1.0 - strength + strength * hann)
        signal = window * np.sinc(shape_parameter * centered_s / actual_duration_s)
        signal = signal.astype(np.complex128)
    elif pulse_type == "slr":
        if flip_angle_deg is not None and not np.isfinite(flip_angle_deg):
            raise ValueError("flip_angle_deg must be finite")
        signal = _design_slr_waveform(
            sample_count,
            _validate_slr_sharpness(slr_sharpness),
            None if flip_angle_deg is None else float(flip_angle_deg),
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
        effective_tbw = (
            slr_time_bandwidth_product(slr_sharpness)
            if pulse_type == "slr"
            else rf_time_bandwidth_product_from_envelope(signal)
        )
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


def design_hyperbolic_secant_inversion(
    *,
    duration_s: float,
    raster_s: float,
    mu: float = 4.9,
    beta_duration_product: float = 8.0,
    adiabaticity: float = 4.0,
) -> np.ndarray:
    """Return an adiabatic hyperbolic-secant inversion waveform in hertz.

    The Silver-Hoult pulse is ``A sech(beta t) exp(i phi(t))`` with the
    frequency sweep ``-mu beta tanh(beta t)`` (Baum et al., Phys Rev A 32,
    3435). ``beta`` follows from ``beta_duration_product / duration_s`` so
    the default truncates the envelope at sech(4). The peak amplitude
    satisfies ``(2 pi A)^2 = adiabaticity * mu * beta^2``, the same scaling as
    PyPulseq's ``make_adiabatic_pulse``. The sweep spans ``mu beta / pi`` Hz
    and the phase is zero at the pulse centre.
    """
    for name, value in {
        "duration_s": duration_s,
        "raster_s": raster_s,
        "mu": mu,
        "beta_duration_product": beta_duration_product,
        "adiabaticity": adiabaticity,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    sample_count = int(round(float(duration_s) / float(raster_s)))
    if sample_count < 2:
        raise ValueError("the inversion pulse must span at least two RF samples")
    actual_duration_s = sample_count * float(raster_s)
    beta = float(beta_duration_product) / actual_duration_s
    time_s = (np.arange(sample_count, dtype=float) + 0.5) * float(
        raster_s
    ) - actual_duration_s / 2.0
    frequency_rad_s = -float(mu) * beta * np.tanh(beta * time_s)
    phase = np.cumsum(frequency_rad_s) * float(raster_s)
    phase -= np.interp(0.0, time_s, phase)
    amplitude_hz = np.sqrt(float(adiabaticity) * float(mu) * beta**2) / (2.0 * np.pi)
    samples = amplitude_hz / np.cosh(beta * time_s) * np.exp(1j * phase)
    samples.setflags(write=False)
    return samples


def hyperbolic_secant_bandwidth_hz(
    duration_s: float,
    *,
    mu: float = 4.9,
    beta_duration_product: float = 8.0,
) -> float:
    """Return the frequency sweep ``mu beta / pi`` of the HS inversion.

    The adiabatic pulse inverts this band, so a slice-selective HS inversion
    of thickness ``d`` needs the gradient ``bandwidth / d`` in Hz/m.
    """
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive and finite")
    return float(mu) * float(beta_duration_product) / (np.pi * float(duration_s))


def make_pulseq_rf_events(
    pp,
    system,
    *,
    flip_angles_deg,
    pulse_type: str,
    duration_s: float,
    time_bandwidth_product: float = 4.0,
    apodization: float = 0.5,
    slr_sharpness: int = 1,
    custom_waveform_hz=None,
    custom_raster_s: float | None = None,
    custom_flip_angle_deg: float | None = None,
    slice_thickness_m: float | None = None,
    frequency_offset_hz: float = 0.0,
    use: str = "excitation",
    center_s: float | None = None,
    spectral_spatial_pulse=None,
    spatial_offset_m: float = 0.0,
    nucleus: str | None = None,
    include_slice_gradient: bool = True,
):
    """Create Pulseq RF events from the shared envelope designer.

    When ``slice_thickness_m`` is supplied, each result is the
    ``(rf, slice_gradient)`` tuple returned by Pulseq.  Otherwise
    each result is a non-selective RF event.  Loaded/designer waveforms retain
    their complex phase and are rescaled from their reference flip angle.
    ``include_slice_gradient=False`` also omits the coupled gradient of an
    SPSP pulse, returning its RF waveform alone without gradient limit checks.
    """
    if not include_slice_gradient:
        slice_thickness_m = None
        spatial_offset_m = 0.0
    angles = tuple(float(value) for value in flip_angles_deg)
    if not angles or not np.all(np.isfinite(angles)):
        raise ValueError("flip_angles_deg must contain finite values")
    if not np.isfinite(frequency_offset_hz):
        raise ValueError("rf_frequency_offset_hz must be finite")
    if slice_thickness_m is not None and (
        not np.isfinite(slice_thickness_m) or slice_thickness_m <= 0
    ):
        raise ValueError("slice_thickness_m must be positive and finite")

    normalized_type = normalize_rf_pulse_type(pulse_type)
    if normalized_type == "spectral_spatial":
        pulse = SpectralSpatialPulse.from_mapping(spectral_spatial_pulse)
        if nucleus is not None and str(nucleus) != pulse.nucleus:
            raise ValueError(
                f"spectral-spatial pulse was designed for {pulse.nucleus}, not {nucleus}"
            )
        target_thickness_m = (
            pulse.reference_thickness_m
            if slice_thickness_m is None
            else float(slice_thickness_m)
        )
        results = []
        for angle_deg in angles:
            rf_samples, gradient_samples = pulse.scaled_waveforms(
                flip_angle_deg=angle_deg,
                thickness_m=target_thickness_m,
                spatial_offset_m=spatial_offset_m,
            )
            target_rf_count = int(round(pulse.duration_s / system.rf_raster_time))
            target_gradient_count = int(
                round(pulse.duration_s / system.grad_raster_time)
            )
            rf_samples = _resample_complex(rf_samples, target_rf_count)
            if gradient_samples.size != target_gradient_count:
                source = (
                    np.arange(gradient_samples.size, dtype=float) + 0.5
                ) / gradient_samples.size
                target = (
                    np.arange(target_gradient_count, dtype=float) + 0.5
                ) / target_gradient_count
                gradient_samples = np.interp(target, source, gradient_samples)
            if include_slice_gradient:
                gamma_hz_per_t = NUCLEUS_GAMMA_HZ_PER_T[pulse.nucleus]
                physical_gradient = gradient_samples / gamma_hz_per_t
                physical_limit = float(system.max_grad) / float(system.gamma)
                if np.max(np.abs(physical_gradient)) > physical_limit * (1 + 1e-9):
                    raise ValueError(
                        "target slice/slab thickness makes the spectral-spatial "
                        "gradient exceed the scanner amplitude limit"
                    )
                physical_slew = (
                    np.concatenate(
                        (
                            [2.0 * physical_gradient[0]],
                            np.diff(physical_gradient),
                            [-2.0 * physical_gradient[-1]],
                        )
                    )
                    / system.grad_raster_time
                )
                physical_slew_limit = float(system.max_slew) / float(system.gamma)
                if np.max(np.abs(physical_slew)) > physical_slew_limit * (1 + 1e-9):
                    raise ValueError(
                        "target slice/slab thickness makes the spectral-spatial "
                        "gradient exceed the scanner slew limit"
                    )
            rf_event = pp.make_arbitrary_rf(
                signal=rf_samples,
                flip_angle=np.deg2rad(angle_deg),
                dwell=system.rf_raster_time,
                no_signal_scaling=True,
                delay=system.rf_dead_time,
                system=system,
                use=use,
                center=(
                    pulse.excitation_center_s if center_s is None else float(center_s)
                ),
            )
            rf_event.freq_offset = float(frequency_offset_hz)
            if not include_slice_gradient:
                results.append(rf_event)
                continue
            gradient_event = pp.make_arbitrary_grad(
                channel="z",
                waveform=np.asarray(gradient_samples, dtype=float),
                first=0.0,
                last=0.0,
                delay=system.rf_dead_time,
                system=system,
            )
            results.append((rf_event, gradient_event))
        return (
            tuple(results),
            pulse.duration_s,
            pulse.spatial_time_bandwidth_product,
            normalized_type,
        )

    design_kwargs = {
        "pulse_type": pulse_type,
        "duration_s": duration_s,
        "raster_s": system.rf_raster_time,
        "time_bandwidth_product": time_bandwidth_product,
        "apodization": apodization,
        "slr_sharpness": slr_sharpness,
        "custom_waveform": custom_waveform_hz,
        "custom_raster_s": custom_raster_s,
    }
    envelope, actual_duration_s, effective_tbw, normalized_type = design_rf_envelope(
        **design_kwargs
    )
    loaded = normalized_type == "designer"
    results = []
    for angle_deg in angles:
        if normalized_type == "slr":
            # The SLR shape depends on the flip angle it is designed for.
            envelope = design_rf_envelope(**design_kwargs, flip_angle_deg=angle_deg)[0]
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
            _check_slice_gradient(
                system,
                bandwidth_hz=kwargs["bandwidth"],
                slice_thickness_m=slice_thickness_m,
            )
            kwargs.update(
                slice_thickness=slice_thickness_m,
                return_gz=True,
            )
        event = pp.make_arbitrary_rf(**kwargs)
        rf_event = event[0] if slice_thickness_m is not None else event
        rf_event.freq_offset = float(frequency_offset_hz)
        results.append(event)
    return tuple(results), actual_duration_s, effective_tbw, normalized_type


def _check_slice_gradient(system, *, bandwidth_hz: float, slice_thickness_m: float):
    """Raise a readable error when the slice-select gradient exceeds the limit.

    ``system`` converts its limits with the gyromagnetic ratio of the imaged
    nucleus, so the message reports the physical gradient of that nucleus.
    """
    required = float(bandwidth_hz) / float(slice_thickness_m)
    limit = float(system.max_grad)
    if required <= limit * (1.0 + 1e-9):
        return
    gamma = float(system.gamma)
    raise ValueError(
        "the slice-selective RF pulse needs "
        f"{required / gamma * 1e3:.3g} mT/m ({float(bandwidth_hz):.4g} Hz over "
        f"{float(slice_thickness_m) * 1e3:.3g} mm), above the "
        f"{limit / gamma * 1e3:.3g} mT/m limit; lengthen the RF pulse, lower "
        "its time-bandwidth product or increase the slice thickness"
    )


def shifted_spectral_spatial_rf_event(
    rf_event,
    gradient_event,
    *,
    spatial_offset_m: float,
    rf_raster_s: float,
    gradient_raster_s: float,
):
    """Copy an SPSP RF event and translate its spatial excitation profile."""
    spatial_offset_m = float(spatial_offset_m)
    shifted = copy(rf_event)
    shifted.signal = np.array(rf_event.signal, dtype=np.complex128, copy=True)
    if not spatial_offset_m:
        return shifted
    gradient = np.asarray(gradient_event.waveform, dtype=float).reshape(-1)
    rf_times = (np.arange(shifted.signal.size, dtype=float) + 0.5) * rf_raster_s
    indices = np.floor(rf_times / gradient_raster_s).astype(int)
    gradient_on_rf = gradient[np.clip(indices, 0, gradient.size - 1)]
    k_future = np.cumsum(gradient_on_rf[::-1])[::-1] * rf_raster_s
    shifted.signal *= np.exp(-2j * np.pi * k_future * spatial_offset_m)
    return shifted


def spectral_spatial_refocus_area(
    rf_event,
    gradient_event,
    *,
    gradient_raster_s: float,
) -> float:
    """Return the gradient area that refocuses an SPSP excitation.

    Each subpulse leaves the slice wound to the gradient moment that follows
    it. The |B1|-weighted mean of that remaining moment is the effective
    excitation k-space position, so the rephaser must add its negative. For
    flyback designs this is half a positive lobe, while the pulse's own net
    area is zero. The result is in the gradient event's channel convention.
    """
    gradient = np.asarray(gradient_event.waveform, dtype=float).reshape(-1)
    signal = np.abs(np.asarray(rf_event.signal, dtype=np.complex128).reshape(-1))
    if gradient.size == 0 or not np.any(signal > 0):
        return 0.0
    gradient_delay_s = float(getattr(gradient_event, "delay", 0.0) or 0.0)
    edges_s = gradient_delay_s + np.arange(gradient.size + 1) * gradient_raster_s
    cumulative = np.concatenate(([0.0], np.cumsum(gradient) * gradient_raster_s))
    rf_times_s = float(getattr(rf_event, "delay", 0.0) or 0.0) + np.asarray(
        rf_event.t, dtype=float
    ).reshape(-1)
    k_future = cumulative[-1] - np.interp(rf_times_s, edges_s, cumulative)
    return -float(np.sum(signal * k_future) / np.sum(signal))


def set_rf_definitions(
    sequence,
    *,
    pulse_type: str,
    requested_duration_s: float,
    actual_duration_s: float,
    time_bandwidth_product: float,
    apodization: float,
    slr_sharpness: int,
    custom_name: str | None,
    custom_flip_angle_deg: float | None,
    frequency_offset_hz: float,
    prefix: str = "",
    spectral_spatial_pulse=None,
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
        slr_sharpness = _validate_slr_sharpness(slr_sharpness)
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
    if pulse_type == "spectral_spatial":
        pulse = SpectralSpatialPulse.from_mapping(spectral_spatial_pulse)
        sequence.set_definition(f"{key}RFBandwidth", pulse.spectral_alias_bandwidth_hz)
        sequence.set_definition(f"{key}RFSpectralSpatialName", pulse.name)
        sequence.set_definition(f"{key}RFSpectralSpatialMethod", pulse.method)
        sequence.set_definition(
            f"{key}RFSpectralSpatialPhaseType", pulse.spectral_phase_type
        )
        sequence.set_definition(
            f"{key}RFSpectralSpatialImplementation",
            pulse.design_metadata.get("implementation", "legacy"),
        )
        sequence.set_definition(f"{key}RFSpectralSpatialNucleus", pulse.nucleus)
        sequence.set_definition(
            f"{key}RFSpectralSpatialReferenceThickness",
            pulse.reference_thickness_m,
        )
        sequence.set_definition(
            f"{key}RFSpectralSpatialReferenceCenter",
            pulse.reference_spatial_center_m,
        )
        sequence.set_definition(
            f"{key}RFSpectralSpatialReferenceSpatialTransition",
            pulse.reference_spatial_transition_m,
        )
        sequence.set_definition(
            f"{key}RFSpectralSpatialReferenceFlipAngleDeg",
            pulse.reference_flip_angle_deg,
        )
        sequence.set_definition(
            f"{key}RFSpectralSpatialTBW",
            pulse.spatial_time_bandwidth_product,
        )
        sequence.set_definition(
            f"{key}RFSpectralSpatialSubpulseDuration",
            pulse.subpulse_duration_s,
        )
        sequence.set_definition(
            f"{key}RFSpectralSpatialSubpulsePeriod", pulse.subpulse_period_s
        )
        sequence.set_definition(f"{key}RFSpectralSpatialPeakB1T", pulse.peak_b1_t)
        sequence.set_definition(
            f"{key}RFSpectralSpatialAliasBandwidth",
            pulse.spectral_alias_bandwidth_hz,
        )
        sequence.set_definition(
            f"{key}RFSpectralSpatialBandCenters",
            [band.center_hz for band in pulse.bands],
        )
        sequence.set_definition(
            f"{key}RFSpectralSpatialBandWidths",
            [band.width_hz for band in pulse.bands],
        )
        sequence.set_definition(
            f"{key}RFSpectralSpatialBandTransitionWidths",
            [band.transition_width_hz for band in pulse.bands],
        )
        sequence.set_definition(
            f"{key}RFSpectralSpatialBandAmplitudes",
            [band.amplitude for band in pulse.bands],
        )
        sequence.set_definition(
            f"{key}RFSpectralSpatialBandPhasesDeg",
            [band.phase_deg for band in pulse.bands],
        )
