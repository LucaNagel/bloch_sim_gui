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
from scipy.signal import remez
from scipy.signal.windows import tukey

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


@lru_cache(maxsize=64)
def _design_slr_waveform(
    sample_count: int, time_bandwidth_product: float, sharpness: int
) -> np.ndarray:
    """Design a linear-phase, small-tip SLR beta polynomial.

    ``sharpness`` is the integer order of the design. Order one is a single
    positive central lobe; increasing the order progressively adds temporal
    side lobes and narrows the transition. The result is cached because
    sequence previews often request the same RF shape for several flip angles
    and slices.
    """
    sharpness = _validate_slr_sharpness(sharpness)
    if sample_count < 8:
        raise ValueError("an SLR pulse requires at least 8 RF samples")
    tbw = float(time_bandwidth_product)
    if tbw >= sample_count:
        raise ValueError(
            "rf_time_bandwidth_product must be smaller than the SLR sample count"
        )

    if sharpness == 1:
        # The first member of the historical SLR family is deliberately a
        # single, positive cosine lobe. Construct it analytically so numerical
        # equiripple noise cannot introduce tiny false side lobes.
        positions = np.linspace(-0.5, 0.5, sample_count, dtype=float)
        signal = np.cos(np.pi * positions).astype(np.complex128)
        signal.setflags(write=False)
        return signal

    # Sharpness historically selected an integer SLR family: order one had a
    # single temporal lobe, while higher orders added side lobes. Do not
    # multiply by the construction TBW here: doing so makes order one inherit
    # several lobes from the hidden TBW input. The completed waveform still
    # reports its measured TBW to consumers below.
    designed_tbw = float(sharpness)
    if designed_tbw >= sample_count:
        raise ValueError("rf_slr_sharpness must be smaller than the SLR sample count")
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
    slr_sharpness: int = 1,
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
        slr_sharpness = _validate_slr_sharpness(slr_sharpness)
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
