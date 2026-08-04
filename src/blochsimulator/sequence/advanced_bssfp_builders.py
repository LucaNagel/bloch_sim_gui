"""Advanced spectrally selective and radial multi-echo bSSFP builders."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .encoding import (
    EncodingFrame,
    logical_gradient_area,
    make_role_trapezoid,
    resolve_encoding_frame,
    set_pulseq_encoding_definitions,
)
from .rf_pulses import design_rf_envelope
from .scanner import ScannerParameters


SKINNER_REFERENCE_DOI = "10.1002/mrm.29676"
WANG_REFERENCE_DOI = "10.1002/mrm.30614"
PICCINI_REFERENCE_DOI = "10.1002/mrm.22898"
GAUBATZ_REFERENCE_TITLE = (
    "Implementation of a Multi-Echo bSSFP Pulse Sequence for "
    "Hyperpolarized 13C MRI at 7T"
)
GOLDEN_ANGLE_DEG = 137.50776405003785


def _pypulseq():
    try:
        import pypulseq as pp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Pulseq sequence generation requires the optional dependency: "
            "pip install 'blochsimulator[pulseq]'"
        ) from exc
    return pp


def _positive_integer(value: int, name: str) -> int:
    if int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_values(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not np.all(np.isfinite(result)) or min(result) <= 0:
        raise ValueError(f"{name} values must be positive and finite")
    return result


def _finite_values(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _make_system(
    pp,
    scanner_parameters: ScannerParameters | Mapping[str, float] | None,
    *,
    legacy_max_grad_mtm: float,
    legacy_max_slew_tms: float,
):
    if scanner_parameters is None:
        return pp.Opts(
            max_grad=legacy_max_grad_mtm,
            grad_unit="mT/m",
            max_slew=legacy_max_slew_tms,
            slew_unit="T/m/s",
            rf_ringdown_time=20e-6,
            rf_dead_time=100e-6,
            adc_dead_time=20e-6,
        )
    profile = ScannerParameters.from_mapping(scanner_parameters)
    return pp.Opts(**profile.to_pypulseq_kwargs())


def _raise_for_timing_errors(sequence, name: str) -> None:
    ok, errors = sequence.check_timing()
    if not ok:
        details = "; ".join(str(error) for error in errors[:8])
        raise ValueError(f"{name} timing check failed: {details}")


def _adc_dwell_for_bandwidth(
    *,
    sample_count: int,
    sampling_bandwidth_hz: float,
    adc_raster_s: float,
    grad_raster_s: float,
) -> float:
    if not np.isfinite(sampling_bandwidth_hz) or sampling_bandwidth_hz <= 0:
        raise ValueError("sampling_bandwidth_hz must be positive and finite")
    requested = 1.0 / float(sampling_bandwidth_hz)
    requested_steps = requested / adc_raster_s
    ratio = int(round(grad_raster_s / adc_raster_s))
    if ratio <= 0 or not np.isclose(
        ratio * adc_raster_s, grad_raster_s, rtol=0.0, atol=1e-15
    ):
        raise ValueError("gradient and ADC rasters must be commensurate")
    centre = max(1, int(round(requested_steps)))
    best_steps = None
    best_error = np.inf
    for offset in range(ratio + 1):
        for candidate in {centre - offset, centre + offset}:
            if candidate <= 0 or (sample_count * candidate) % ratio:
                continue
            error = abs(candidate - requested_steps)
            if error < best_error:
                best_steps = candidate
                best_error = error
        if best_steps is not None and offset > best_error:
            break
    if best_steps is None:
        raise ValueError("sampling bandwidth is incompatible with scanner rasters")
    return float(best_steps * adc_raster_s)


def _rf_event(
    pp,
    system,
    envelope: np.ndarray,
    *,
    flip_angle_deg: float,
    duration_s: float,
    bandwidth_hz: float,
):
    return pp.make_arbitrary_rf(
        signal=envelope,
        flip_angle=np.deg2rad(flip_angle_deg),
        dwell=system.rf_raster_time,
        bandwidth=bandwidth_hz,
        time_bw_product=bandwidth_hz * duration_s,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )


def make_pulseq_spectral_selective_bssfp(
    *,
    fov_m: Sequence[float] = (56e-3, 28e-3, 21e-3),
    matrix: Sequence[int] = (32, 16, 12),
    target_frequency_offsets_hz: Sequence[float] = (1655.0, -245.0),
    receiver_frequency_offsets_hz: Sequence[float] | None = (925.44725, 0.0),
    target_metabolite_names: Sequence[str] = ("Lac", "Py"),
    flip_angle_deg: float | Sequence[float] = (90.0, 4.0),
    spectral_rf_duration_s: float = 2.33e-3,
    spectral_rf_bandwidth_hz: float | None = None,
    spectral_rf_bandwidth_factor_hz_ms: float = 2100.0,
    spectral_rf_fwhm_hz: float = 900.0,
    spectral_rf_pulse_type: str = "slr",
    spectral_rf_apodization: float = 0.0,
    spectral_rf_slr_sharpness: float = 1.0,
    sampling_bandwidth_hz: float = 10_000.0,
    encoding_duration_s: float | None = None,
    repetition_time_s: float | None = 6.29e-3,
    rf_phase_start_deg: float = 0.0,
    rf_phase_increment_deg: float = 0.0,
    dummy_repetitions: int = 0,
    repetitions: int = 2,
    use_alpha_half: bool = True,
    alpha_half_center_spacing_s: float = 4.31e-3,
    end_image_spoiler_cycles_per_fov: float = 4.0,
    end_image_spoiler_duration_s: float = 1e-3,
    field_strength_t: float = 7.0,
    nucleus: str = "C13",
    encoding_axes: Sequence[str] | EncodingFrame = ("+x", "+y", "+z"),
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
):
    """Build alternating-frequency Cartesian 3D SS-bSSFP.

    The published Skinner et al. acquisition is represented by the defaults.
    Each complete 3D volume uses one RF/receiver frequency pair; target pairs
    repeat cyclically when ``repetitions`` exceeds their count.

    ``encoding_duration_s=None`` selects the shortest common pre-/rephasing
    lobe duration that can produce the required read, phase, and partition
    moments with the configured scanner limits.  The read prephaser moment is
    derived from the actual readout gradient, so changing the read matrix or
    sampling bandwidth also updates the encoding-lobe duration.
    """
    pp = _pypulseq()
    encoding_frame = resolve_encoding_frame(encoding_axes)
    fov = _positive_values(fov_m, "FOV")
    if len(fov) != 3:
        raise ValueError("fov_m must contain three values")
    matrix_values = tuple(_positive_integer(value, "matrix size") for value in matrix)
    if len(matrix_values) != 3:
        raise ValueError("matrix must contain three values")
    n_read, n_phase, n_partition = matrix_values
    target_offsets = _finite_values(
        target_frequency_offsets_hz, "target_frequency_offsets_hz"
    )
    if receiver_frequency_offsets_hz is None:
        receiver_offsets = target_offsets
    else:
        receiver_offsets = _finite_values(
            receiver_frequency_offsets_hz, "receiver_frequency_offsets_hz"
        )
    if len(receiver_offsets) != len(target_offsets):
        raise ValueError("receiver offsets must match target offsets")
    names = tuple(str(value).strip() for value in target_metabolite_names)
    if len(names) != len(target_offsets) or any(not name for name in names):
        raise ValueError("target metabolite names must match target offsets")
    if isinstance(flip_angle_deg, (int, float, np.integer, np.floating)):
        flip_angles = (float(flip_angle_deg),) * len(target_offsets)
    else:
        flip_angles = _positive_values(flip_angle_deg, "flip_angle_deg")
        if len(flip_angles) == 1:
            flip_angles *= len(target_offsets)
    if len(flip_angles) != len(target_offsets):
        raise ValueError("flip angles must be scalar or match target offsets")
    if min(flip_angles) <= 0:
        raise ValueError("flip angles must be positive")
    repetitions = _positive_integer(repetitions, "repetitions")
    dummy_repetitions = int(dummy_repetitions)
    if dummy_repetitions < 0:
        raise ValueError("dummy_repetitions must be non-negative")

    if sampling_bandwidth_hz <= 0:
        raise ValueError("sampling_bandwidth_hz must be positive")
    positive_parameters = {
        "spectral_rf_duration_s": spectral_rf_duration_s,
        "spectral_rf_bandwidth_factor_hz_ms": spectral_rf_bandwidth_factor_hz_ms,
        "spectral_rf_fwhm_hz": spectral_rf_fwhm_hz,
        "sampling_bandwidth_hz": sampling_bandwidth_hz,
        "alpha_half_center_spacing_s": alpha_half_center_spacing_s,
        "end_image_spoiler_duration_s": end_image_spoiler_duration_s,
        "field_strength_t": field_strength_t,
    }
    for name, value in positive_parameters.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if encoding_duration_s is not None and (
        not np.isfinite(encoding_duration_s) or encoding_duration_s <= 0
    ):
        raise ValueError("encoding_duration_s must be positive and finite or None")
    if not np.isfinite(end_image_spoiler_cycles_per_fov) or (
        end_image_spoiler_cycles_per_fov < 0
    ):
        raise ValueError("end-image spoiler cycles must be non-negative and finite")
    nucleus = str(nucleus).strip()
    if not nucleus:
        raise ValueError("nucleus must not be empty")
    if spectral_rf_bandwidth_hz is None:
        spectral_rf_bandwidth_hz = spectral_rf_bandwidth_factor_hz_ms / (
            spectral_rf_duration_s * 1e3
        )
    if not np.isfinite(spectral_rf_bandwidth_hz) or spectral_rf_bandwidth_hz <= 0:
        raise ValueError("spectral_rf_bandwidth_hz must be positive and finite")

    system = _make_system(
        pp,
        scanner_parameters,
        legacy_max_grad_mtm=100.0,
        legacy_max_slew_tms=1000.0,
    )
    sequence = pp.Sequence(system)
    raster = system.block_duration_raster
    rf_tbw = float(spectral_rf_bandwidth_hz) * float(spectral_rf_duration_s)
    envelope, actual_rf_duration, _, pulse_type = design_rf_envelope(
        pulse_type=spectral_rf_pulse_type,
        duration_s=spectral_rf_duration_s,
        raster_s=system.rf_raster_time,
        time_bandwidth_product=rf_tbw,
        apodization=spectral_rf_apodization,
        slr_sharpness=spectral_rf_slr_sharpness,
    )
    rfs = tuple(
        _rf_event(
            pp,
            system,
            envelope,
            flip_angle_deg=angle,
            duration_s=actual_rf_duration,
            bandwidth_hz=spectral_rf_bandwidth_hz,
        )
        for angle in flip_angles
    )

    dwell = _adc_dwell_for_bandwidth(
        sample_count=n_read,
        sampling_bandwidth_hz=sampling_bandwidth_hz,
        adc_raster_s=system.adc_raster_time,
        grad_raster_s=system.grad_raster_time,
    )
    readout_duration = n_read * dwell
    readout_amplitude = 1.0 / (fov[0] * dwell)
    readout_rise = max(
        system.adc_dead_time,
        np.ceil(abs(readout_amplitude) / system.max_slew / system.grad_raster_time)
        * system.grad_raster_time,
    )
    gx = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        flat_area=n_read / fov[0],
        flat_time=readout_duration,
        rise_time=readout_rise,
        system=system,
    )
    adc = pp.make_adc(
        num_samples=n_read,
        dwell=dwell,
        delay=gx.rise_time,
        system=system,
    )
    readout_area = logical_gradient_area(gx, encoding_frame, "read")
    ky_areas = (np.arange(n_phase) - n_phase // 2) / fov[1]
    kz_areas = (np.arange(n_partition) - n_partition // 2) / fov[2]

    # All three spatial encoders are played in one block and must therefore
    # share a duration.  Ask Pulseq for the shortest hardware-valid lobe for
    # every maximum required moment, then use the longest of those minima.
    # In particular, the read prephaser uses half the *actual* read-gradient
    # area, including its ramps; it therefore follows both read points and BW.
    encoding_areas = (
        abs(readout_area) / 2,
        float(np.max(np.abs(ky_areas))),
        float(np.max(np.abs(kz_areas))),
    )
    minimum_encoding_duration = max(
        pp.calc_duration(
            make_role_trapezoid(
                pp,
                encoding_frame,
                role,
                area=area,
                system=system,
            )
        )
        for role, area in zip(("read", "phase", "partition"), encoding_areas)
        if area > 0
    )
    minimum_encoding_duration = (
        np.ceil(minimum_encoding_duration / system.grad_raster_time - 1e-9)
        * system.grad_raster_time
    )
    requested_encoding_duration = encoding_duration_s
    if encoding_duration_s is None:
        encoding_duration_s = minimum_encoding_duration
    else:
        encoding_duration_s = max(float(encoding_duration_s), minimum_encoding_duration)
        encoding_duration_s = (
            np.ceil(encoding_duration_s / system.grad_raster_time - 1e-9)
            * system.grad_raster_time
        )
    gx_pre = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        area=-readout_area / 2,
        duration=encoding_duration_s,
        system=system,
    )
    reference_rf = rfs[0]
    rf_center, _ = pp.calc_rf_center(reference_rf)
    rf_center_from_start = reference_rf.delay + rf_center
    rf_block_duration = np.ceil(pp.calc_duration(reference_rf) / raster) * raster
    rf_block_padding = pp.make_delay(rf_block_duration)
    read_block_duration = max(pp.calc_duration(gx), pp.calc_duration(adc))
    adc_center_from_start = adc.delay + adc.num_samples * adc.dwell / 2
    pre_duration = pp.calc_duration(gx_pre)
    minimum_explicit_tr = max(
        2
        * (
            rf_block_duration
            + pre_duration
            + adc_center_from_start
            - rf_center_from_start
        ),
        2
        * (
            rf_center_from_start
            + pre_duration
            + read_block_duration
            - adc_center_from_start
        ),
    )
    minimum_explicit_tr = np.ceil(minimum_explicit_tr / raster - 1e-9) * raster
    if repetition_time_s is None:
        rf_balance_delay_value = (
            2 * rf_center_from_start
            + read_block_duration
            - 2 * adc_center_from_start
            - rf_block_duration
        )
    else:
        rf_balance_delay_value = (
            repetition_time_s / 2
            + rf_center_from_start
            - rf_block_duration
            - pre_duration
            - adc_center_from_start
        )
    rf_balance_delay_value = round(rf_balance_delay_value / raster) * raster
    if rf_balance_delay_value < 0:
        if repetition_time_s is None:
            raise ValueError("RF timing cannot be centered with a non-negative delay")
        raise ValueError(
            f"TR {repetition_time_s * 1e3:.3f} ms is too short to center "
            f"the ADC at TR/2; the minimum is {minimum_explicit_tr * 1e3:.3f} ms. "
            f"Timing components: RF block {rf_block_duration * 1e3:.3f} ms, "
            f"ADC acquisition {readout_duration * 1e3:.3f} ms, readout block "
            f"{read_block_duration * 1e3:.3f} ms, and automatic encoding lobe "
            f"{pre_duration * 1e3:.3f} ms."
        )
    minimum_tr = (
        rf_block_duration
        + rf_balance_delay_value
        + 2 * pre_duration
        + read_block_duration
    )
    if repetition_time_s is None:
        actual_tr = minimum_tr
        tr_fill = 0.0
    else:
        if not np.isfinite(repetition_time_s) or repetition_time_s <= 0:
            raise ValueError("repetition_time_s must be positive and finite")
        actual_tr = round(repetition_time_s / raster) * raster
        tr_fill = round((actual_tr - minimum_tr) / raster) * raster
        if tr_fill < -raster / 2:
            raise ValueError(
                "repetition_time_s is too short; use at least "
                f"{minimum_explicit_tr:.9g} s with the current RF, readout, "
                "encoding duration, and scanner limits"
            )
        tr_fill = max(0.0, tr_fill)

    spoilers = ()
    if end_image_spoiler_cycles_per_fov > 0:
        spoilers = tuple(
            make_role_trapezoid(
                pp,
                encoding_frame,
                role,
                area=end_image_spoiler_cycles_per_fov / axis_fov,
                duration=end_image_spoiler_duration_s,
                system=system,
            )
            for role, axis_fov in zip(("read", "phase", "partition"), fov)
        )
    spoiler_end_times: list[float] = []

    for repetition in range(repetitions):
        target_index = repetition % len(target_offsets)
        rf = rfs[target_index]
        target_offset = target_offsets[target_index]
        receiver_offset = receiver_offsets[target_index]
        frame_rf_center, _ = pp.calc_rf_center(rf)
        if use_alpha_half:
            alpha_half = _rf_event(
                pp,
                system,
                envelope,
                flip_angle_deg=flip_angles[target_index] / 2,
                duration_s=actual_rf_duration,
                bandwidth_hz=spectral_rf_bandwidth_hz,
            )
            alpha_center, _ = pp.calc_rf_center(alpha_half)
            alpha_half.freq_offset = target_offset
            alpha_half.phase_offset = (
                np.deg2rad(rf_phase_start_deg)
                - 2 * np.pi * target_offset * alpha_center
            )
            alpha_block_duration = (
                np.ceil(pp.calc_duration(alpha_half) / raster) * raster
            )
            spacing_delay = (
                alpha_half_center_spacing_s
                - alpha_block_duration
                + alpha_half.delay
                + alpha_center
                - rf.delay
                - frame_rf_center
            )
            spacing_delay = round(spacing_delay / raster) * raster
            if spacing_delay < 0:
                raise ValueError("alpha/2 center spacing is too short")
            sequence.add_block(alpha_half, pp.make_delay(alpha_block_duration))
            if spacing_delay:
                sequence.add_block(pp.make_delay(spacing_delay))

        rf_phase = float(rf_phase_start_deg)
        for _ in range(dummy_repetitions):
            rf.freq_offset = target_offset
            rf.phase_offset = (
                np.deg2rad(rf_phase) - 2 * np.pi * target_offset * frame_rf_center
            )
            rf_phase = (rf_phase + rf_phase_increment_deg) % 360.0
            sequence.add_block(rf, rf_block_padding)
            if rf_balance_delay_value:
                sequence.add_block(pp.make_delay(rf_balance_delay_value))
            sequence.add_block(gx_pre)
            sequence.add_block(gx, pp.make_delay(read_block_duration))
            sequence.add_block(gx_pre)
            if tr_fill:
                sequence.add_block(pp.make_delay(tr_fill))

        for partition, kz in enumerate(kz_areas):
            for line, ky in enumerate(ky_areas):
                rf.freq_offset = target_offset
                rf.phase_offset = (
                    np.deg2rad(rf_phase) - 2 * np.pi * target_offset * frame_rf_center
                )
                adc.freq_offset = receiver_offset
                adc.phase_offset = np.deg2rad(rf_phase)
                rf_phase = (rf_phase + rf_phase_increment_deg) % 360.0
                gy_pre = make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "phase",
                    area=float(ky),
                    duration=encoding_duration_s,
                    system=system,
                )
                gz_pre = make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "partition",
                    area=float(kz),
                    duration=encoding_duration_s,
                    system=system,
                )
                gy_rephase = make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "phase",
                    area=-float(ky),
                    duration=encoding_duration_s,
                    system=system,
                )
                gz_rephase = make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "partition",
                    area=-float(kz),
                    duration=encoding_duration_s,
                    system=system,
                )
                sequence.add_block(rf, rf_block_padding)
                if rf_balance_delay_value:
                    sequence.add_block(pp.make_delay(rf_balance_delay_value))
                sequence.add_block(gx_pre, gy_pre, gz_pre)
                sequence.add_block(
                    gx,
                    adc,
                    pp.make_label("LIN", "SET", line),
                    pp.make_label("PAR", "SET", partition),
                    pp.make_label("REP", "SET", repetition),
                )
                sequence.add_block(gx_pre, gy_rephase, gz_rephase)
                if tr_fill:
                    sequence.add_block(pp.make_delay(tr_fill))
        if spoilers:
            sequence.add_block(*spoilers)
            spoiler_end_times.append(float(sequence.duration()[0]))

    _raise_for_timing_errors(sequence, "spectrally selective bSSFP")
    sequence.set_definition("Name", "spectral_selective_bssfp_3d")
    sequence.set_definition("FOV", list(fov))
    sequence.set_definition("MatrixSize", list(matrix_values))
    set_pulseq_encoding_definitions(
        sequence,
        encoding_frame,
        fov_m=fov,
        matrix=matrix_values,
    )
    sequence.set_definition("FieldStrengthT", float(field_strength_t))
    sequence.set_definition("Nucleus", nucleus)
    sequence.set_definition("DynamicFrames", repetitions)
    sequence.set_definition("Repetitions", repetitions)
    sequence.set_definition("SpectralTargetOffsetsHz", list(target_offsets))
    sequence.set_definition("SpectralReceiverOffsetsHz", list(receiver_offsets))
    sequence.set_definition("SpectralTargetNames", list(names))
    sequence.set_definition("FlipAngleDeg", list(flip_angles))
    sequence.set_definition("SpectralRFPulseType", pulse_type)
    sequence.set_definition("SpectralRFDuration", actual_rf_duration)
    sequence.set_definition("SpectralRFBandwidthHz", spectral_rf_bandwidth_hz)
    sequence.set_definition(
        "SpectralRFBandwidthFactorHzMs", spectral_rf_bandwidth_factor_hz_ms
    )
    sequence.set_definition("SpectralRFFWHM", spectral_rf_fwhm_hz)
    if pulse_type == "slr":
        sequence.set_definition("SpectralSLRSharpness", spectral_rf_slr_sharpness)
    sequence.set_definition("SamplingBandwidth", 1.0 / dwell)
    sequence.set_definition("ReadoutBandwidthHz", 1.0 / dwell)
    sequence.set_definition("ADCDwell", dwell)
    sequence.set_definition("ReadoutDuration", readout_duration)
    sequence.set_definition("ReadoutBlockDuration", read_block_duration)
    sequence.set_definition("EncodingLobeDuration", encoding_duration_s)
    sequence.set_definition(
        "EncodingLobeDurationMode",
        "automatic" if requested_encoding_duration is None else "requested-minimum",
    )
    sequence.set_definition("MinimumEncodingLobeDuration", minimum_encoding_duration)
    sequence.set_definition("MinimumTR", minimum_explicit_tr)
    sequence.set_definition(
        "RequestedTR", actual_tr if repetition_time_s is None else repetition_time_s
    )
    sequence.set_definition("TR", actual_tr)
    sequence.set_definition("TE", actual_tr / 2)
    sequence.set_definition("RFPhaseStartDeg", float(rf_phase_start_deg))
    sequence.set_definition("RFPhaseIncrementDeg", float(rf_phase_increment_deg))
    sequence.set_definition("DummyRepetitions", dummy_repetitions)
    sequence.set_definition("UseAlphaHalf", bool(use_alpha_half))
    sequence.set_definition(
        "AlphaHalfCenterSpacing",
        alpha_half_center_spacing_s if use_alpha_half else 0.0,
    )
    sequence.set_definition(
        "EndImageSpoilerCyclesPerFOV", end_image_spoiler_cycles_per_fov
    )
    sequence.set_definition("EndImageSpoilerDuration", end_image_spoiler_duration_s)
    sequence.set_definition("EndImageSpoilerAxes", "xyz" if spoilers else "none")
    sequence.set_definition("EndImageSpoilerEndTimes", spoiler_end_times)
    sequence.set_definition("IdealSpoilerEndTimes", spoiler_end_times)
    sequence.set_definition("ReferenceDOI", SKINNER_REFERENCE_DOI)
    return sequence


def _make_me_bssfp_rf(
    pp,
    system,
    *,
    pulse_type: str,
    flip_angle_deg: float,
    duration_s: float,
    bandwidth_hz: float,
):
    normalized = str(pulse_type).strip().lower().replace("-", "_")
    if normalized in {"gauss", "gaussian"}:
        return (
            pp.make_gauss_pulse(
                flip_angle=np.deg2rad(flip_angle_deg),
                duration=duration_s,
                bandwidth=bandwidth_hz,
                apodization=0.0,
                delay=system.rf_dead_time,
                system=system,
                use="excitation",
            ),
            "gaussian",
        )
    if normalized in {"block", "hard", "rectangular"}:
        return (
            pp.make_block_pulse(
                flip_angle=np.deg2rad(flip_angle_deg),
                duration=duration_s,
                delay=system.rf_dead_time,
                system=system,
                use="excitation",
            ),
            "block",
        )
    raise ValueError("rf_pulse_type must be 'gaussian' or 'block'")


def make_pulseq_me_bssfp(
    *,
    fov_m: Sequence[float] = (56e-3, 28e-3, 24.5e-3),
    matrix: Sequence[int] = (32, 16, 14),
    echoes: int = 5,
    echo_spacing_s: float = 1.32e-3,
    readout_strategy: str = "flyback",
    sampling_bandwidth_hz: float = 39_682.5,
    flip_angle_deg: float = 3.5,
    rf_pulse_type: str = "gaussian",
    rf_duration_s: float = 0.5e-3,
    rf_bandwidth_hz: float = 5480.0,
    rf_frequency_offset_hz: float = 0.0,
    receiver_frequency_offset_hz: float = -460.0,
    encoding_duration_s: float = 0.5e-3,
    repetition_time_s: float = 8.696e-3,
    rf_phase_start_deg: float = 0.0,
    rf_phase_increment_deg: float = 180.0,
    dummy_repetitions: int = 0,
    repetitions: int = 1,
    use_alpha_half: bool = True,
    field_strength_t: float = 7.0,
    nucleus: str = "C13",
    encoding_axes: Sequence[str] | EncodingFrame = ("+x", "+y", "+z"),
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
):
    """Build Cartesian 3D multi-echo bSSFP with flyback or bipolar readout.

    The defaults reproduce the short-TR in-vivo parameter set described by
    Gaubatz. Echoes are centered between consecutive RF pulses, and every TR
    is fully balanced in read, phase, and partition directions.
    """
    pp = _pypulseq()
    encoding_frame = resolve_encoding_frame(encoding_axes)
    fov = _positive_values(fov_m, "FOV")
    if len(fov) != 3:
        raise ValueError("fov_m must contain three values")
    matrix_values = tuple(_positive_integer(value, "matrix size") for value in matrix)
    if len(matrix_values) != 3:
        raise ValueError("matrix must contain three values")
    n_read, n_phase, n_partition = matrix_values
    echoes = _positive_integer(echoes, "echoes")
    if echoes % 2 == 0:
        raise ValueError("echoes must be odd for a balanced multi-echo train")
    repetitions = _positive_integer(repetitions, "repetitions")
    dummy_repetitions = int(dummy_repetitions)
    if dummy_repetitions < 0:
        raise ValueError("dummy_repetitions must be non-negative")
    strategy = str(readout_strategy).strip().lower().replace("-", "_")
    strategy_aliases = {
        "flyback": "flyback",
        "monopolar": "flyback",
        "symmetric": "symmetric",
        "bipolar": "symmetric",
    }
    try:
        strategy = strategy_aliases[strategy]
    except KeyError as exc:
        raise ValueError("readout_strategy must be 'flyback' or 'symmetric'") from exc
    for name, value in {
        "echo_spacing_s": echo_spacing_s,
        "sampling_bandwidth_hz": sampling_bandwidth_hz,
        "flip_angle_deg": flip_angle_deg,
        "rf_duration_s": rf_duration_s,
        "rf_bandwidth_hz": rf_bandwidth_hz,
        "encoding_duration_s": encoding_duration_s,
        "repetition_time_s": repetition_time_s,
        "field_strength_t": field_strength_t,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value in {
        "rf_frequency_offset_hz": rf_frequency_offset_hz,
        "receiver_frequency_offset_hz": receiver_frequency_offset_hz,
        "rf_phase_start_deg": rf_phase_start_deg,
        "rf_phase_increment_deg": rf_phase_increment_deg,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    nucleus = str(nucleus).strip()
    if not nucleus:
        raise ValueError("nucleus must not be empty")

    system = _make_system(
        pp,
        scanner_parameters,
        legacy_max_grad_mtm=100.0,
        legacy_max_slew_tms=1000.0,
    )
    sequence = pp.Sequence(system)
    raster = system.block_duration_raster
    dwell = _adc_dwell_for_bandwidth(
        sample_count=n_read,
        sampling_bandwidth_hz=sampling_bandwidth_hz,
        adc_raster_s=system.adc_raster_time,
        grad_raster_s=system.grad_raster_time,
    )
    readout_flat_time = n_read * dwell
    readout_amplitude = 1.0 / (fov[0] * dwell)
    readout_rise = max(
        system.adc_dead_time,
        np.ceil(abs(readout_amplitude) / system.max_slew / system.grad_raster_time)
        * system.grad_raster_time,
    )
    gx_positive = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        flat_area=n_read / fov[0],
        flat_time=readout_flat_time,
        rise_time=readout_rise,
        system=system,
    )
    gx_negative = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        flat_area=-n_read / fov[0],
        flat_time=readout_flat_time,
        rise_time=readout_rise,
        system=system,
    )
    readout_total_area = logical_gradient_area(gx_positive, encoding_frame, "read")
    readout_block_duration = pp.calc_duration(gx_positive)
    adc_center_from_readout_start = readout_rise + readout_flat_time / 2
    actual_echo_spacing = round(echo_spacing_s / raster) * raster
    between_echo_duration = actual_echo_spacing - readout_block_duration
    if between_echo_duration < -raster / 2:
        raise ValueError(
            "echo_spacing_s is shorter than the readout gradient duration; "
            f"use at least {readout_block_duration:.9g} s"
        )
    between_echo_duration = max(0.0, between_echo_duration)
    if strategy == "flyback":
        try:
            between_echo_event = make_role_trapezoid(
                pp,
                encoding_frame,
                "read",
                area=-readout_total_area,
                duration=between_echo_duration,
                system=system,
            )
        except Exception as exc:
            raise ValueError(
                "echo_spacing_s leaves too little time for the flyback gradient; "
                "increase echo spacing or scanner gradient performance"
            ) from exc
    else:
        between_echo_event = (
            pp.make_delay(between_echo_duration) if between_echo_duration else None
        )

    rf, normalized_rf_type = _make_me_bssfp_rf(
        pp,
        system,
        pulse_type=rf_pulse_type,
        flip_angle_deg=flip_angle_deg,
        duration_s=rf_duration_s,
        bandwidth_hz=rf_bandwidth_hz,
    )
    alpha_half, _ = _make_me_bssfp_rf(
        pp,
        system,
        pulse_type=rf_pulse_type,
        flip_angle_deg=flip_angle_deg / 2,
        duration_s=rf_duration_s,
        bandwidth_hz=rf_bandwidth_hz,
    )
    rf_center, _ = pp.calc_rf_center(rf)
    rf_center_from_start = rf.delay + rf_center
    rf_block_duration = np.ceil(pp.calc_duration(rf) / raster - 1e-9) * raster
    rf_block_padding = pp.make_delay(rf_block_duration)
    actual_tr = round(repetition_time_s / raster) * raster
    first_echo_time = (actual_tr - (echoes - 1) * actual_echo_spacing) / 2
    delay_before_encoding = (
        rf_center_from_start
        + first_echo_time
        - rf_block_duration
        - encoding_duration_s
        - adc_center_from_readout_start
    )
    delay_before_encoding = round(delay_before_encoding / raster) * raster
    train_duration = (
        echoes * readout_block_duration + (echoes - 1) * between_echo_duration
    )
    occupied_tr = (
        rf_block_duration
        + delay_before_encoding
        + encoding_duration_s
        + train_duration
        + encoding_duration_s
    )
    trailing_delay = round((actual_tr - occupied_tr) / raster) * raster
    if delay_before_encoding < 0 or trailing_delay < 0:
        minimum_tr = max(
            (echoes - 1) * actual_echo_spacing
            + 2
            * (
                rf_block_duration
                + encoding_duration_s
                + adc_center_from_readout_start
                - rf_center_from_start
            ),
            (echoes - 1) * actual_echo_spacing
            + 2
            * (
                rf_center_from_start
                + encoding_duration_s
                + readout_block_duration
                - adc_center_from_readout_start
            ),
        )
        minimum_tr = np.ceil(minimum_tr / raster - 1e-9) * raster
        raise ValueError(
            "repetition_time_s is too short for the centered echo train; "
            f"use at least {minimum_tr:.9g} s"
        )

    if use_alpha_half:
        alpha_center, _ = pp.calc_rf_center(alpha_half)
        alpha_center_from_start = alpha_half.delay + alpha_center
        alpha_block_duration = (
            np.ceil(pp.calc_duration(alpha_half) / raster - 1e-9) * raster
        )
        preparation_delay = (
            actual_tr / 2
            - alpha_block_duration
            + alpha_center_from_start
            - rf_center_from_start
        )
        preparation_delay = round(preparation_delay / raster) * raster
        if preparation_delay < 0:
            raise ValueError("TR/2 is too short for alpha/2 preparation")
        alpha_half.freq_offset = float(rf_frequency_offset_hz)
        alpha_half.phase_offset = (
            np.deg2rad(rf_phase_start_deg)
            - 2 * np.pi * rf_frequency_offset_hz * alpha_center
        )
        sequence.add_block(alpha_half, pp.make_delay(alpha_block_duration))
        if preparation_delay:
            sequence.add_block(pp.make_delay(preparation_delay))

    gx_pre = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        area=-0.5 * readout_total_area,
        duration=encoding_duration_s,
        system=system,
    )
    ky_areas = (np.arange(n_phase) - n_phase // 2) / fov[1]
    kz_areas = (np.arange(n_partition) - n_partition // 2) / fov[2]
    rf_phase = float(rf_phase_start_deg)

    def add_tr(
        ky: float,
        kz: float,
        *,
        acquire: bool,
        line_index: int = 0,
        partition_index: int = 0,
        volume_index: int = 0,
    ) -> None:
        nonlocal rf_phase
        current_phase = rf_phase
        rf.phase_offset = (
            np.deg2rad(current_phase) - 2 * np.pi * rf_frequency_offset_hz * rf_center
        )
        rf.freq_offset = float(rf_frequency_offset_hz)
        rf_phase = (rf_phase + rf_phase_increment_deg) % 360.0
        gy_pre = make_role_trapezoid(
            pp,
            encoding_frame,
            "phase",
            area=ky,
            duration=encoding_duration_s,
            system=system,
        )
        gz_pre = make_role_trapezoid(
            pp,
            encoding_frame,
            "partition",
            area=kz,
            duration=encoding_duration_s,
            system=system,
        )
        gy_rephase = make_role_trapezoid(
            pp,
            encoding_frame,
            "phase",
            area=-ky,
            duration=encoding_duration_s,
            system=system,
        )
        gz_rephase = make_role_trapezoid(
            pp,
            encoding_frame,
            "partition",
            area=-kz,
            duration=encoding_duration_s,
            system=system,
        )
        sequence.add_block(rf, rf_block_padding)
        if delay_before_encoding:
            sequence.add_block(pp.make_delay(delay_before_encoding))
        sequence.add_block(gx_pre, gy_pre, gz_pre)
        for echo in range(echoes):
            readout = (
                gx_positive if strategy == "flyback" or echo % 2 == 0 else gx_negative
            )
            if acquire:
                adc = pp.make_adc(
                    num_samples=n_read,
                    dwell=dwell,
                    delay=readout_rise,
                    freq_offset=float(receiver_frequency_offset_hz),
                    phase_offset=np.deg2rad(current_phase),
                    system=system,
                )
                sequence.add_block(
                    readout,
                    adc,
                    pp.make_label("LIN", "SET", line_index),
                    pp.make_label("PAR", "SET", partition_index),
                    pp.make_label("ECO", "SET", echo),
                    pp.make_label("REP", "SET", volume_index),
                )
            else:
                sequence.add_block(readout)
            if echo < echoes - 1 and between_echo_event is not None:
                sequence.add_block(between_echo_event)
        sequence.add_block(gx_pre, gy_rephase, gz_rephase)
        if trailing_delay:
            sequence.add_block(pp.make_delay(trailing_delay))

    for _ in range(dummy_repetitions):
        add_tr(0.0, 0.0, acquire=False)
    for volume_index in range(repetitions):
        for partition_index, kz in enumerate(kz_areas):
            for line_index, ky in enumerate(ky_areas):
                add_tr(
                    float(ky),
                    float(kz),
                    acquire=True,
                    line_index=line_index,
                    partition_index=partition_index,
                    volume_index=volume_index,
                )

    _raise_for_timing_errors(sequence, "Cartesian multi-echo bSSFP")
    echo_times = [
        first_echo_time + echo * actual_echo_spacing for echo in range(echoes)
    ]
    sequence.set_definition("Name", "me_bssfp_3d")
    sequence.set_definition("TrajectoryType", "cartesian_3d_multi_echo")
    sequence.set_definition("FOV", list(fov))
    sequence.set_definition("MatrixSize", list(matrix_values))
    set_pulseq_encoding_definitions(
        sequence,
        encoding_frame,
        fov_m=fov,
        matrix=matrix_values,
    )
    sequence.set_definition("Echoes", echoes)
    sequence.set_definition("EchoTimes", echo_times)
    sequence.set_definition("FirstEchoTime", first_echo_time)
    sequence.set_definition("EchoSpacing", actual_echo_spacing)
    sequence.set_definition("RequestedEchoSpacing", float(echo_spacing_s))
    sequence.set_definition("ReadoutStrategy", strategy)
    sequence.set_definition(
        "ReadoutPolarity", "monopolar" if strategy == "flyback" else "bipolar"
    )
    sequence.set_definition("GradientBalancing", "per_tr_xyz")
    sequence.set_definition("SamplingBandwidth", 1.0 / dwell)
    sequence.set_definition("RequestedSamplingBandwidth", sampling_bandwidth_hz)
    sequence.set_definition("ADCDwell", dwell)
    sequence.set_definition("FlipAngleDeg", float(flip_angle_deg))
    sequence.set_definition("RFPulseType", normalized_rf_type)
    sequence.set_definition("RFDuration", float(rf_duration_s))
    sequence.set_definition("RFBandwidthHz", float(rf_bandwidth_hz))
    sequence.set_definition("RFFrequencyOffsetHz", float(rf_frequency_offset_hz))
    sequence.set_definition(
        "ReceiverFrequencyOffsetHz", float(receiver_frequency_offset_hz)
    )
    sequence.set_definition("TR", actual_tr)
    sequence.set_definition("RequestedTR", float(repetition_time_s))
    sequence.set_definition("RFPhaseStartDeg", float(rf_phase_start_deg))
    sequence.set_definition("RFPhaseIncrementDeg", float(rf_phase_increment_deg))
    sequence.set_definition("DummyRepetitions", dummy_repetitions)
    sequence.set_definition("Repetitions", repetitions)
    sequence.set_definition("DynamicFrames", repetitions)
    sequence.set_definition("UseAlphaHalf", bool(use_alpha_half))
    sequence.set_definition(
        "AcquisitionTimePerVolume", n_phase * n_partition * actual_tr
    )
    sequence.set_definition(
        "PreparedFirstVolumeAcquisitionTime",
        n_phase * n_partition * actual_tr + (actual_tr / 2 if use_alpha_half else 0.0),
    )
    sequence.set_definition("FieldStrengthT", float(field_strength_t))
    sequence.set_definition("Nucleus", nucleus)
    sequence.set_definition("ReferenceTitle", GAUBATZ_REFERENCE_TITLE)
    sequence.set_definition("ReferenceYear", 2023)
    return sequence


def spiral_phyllotaxis_directions(
    spokes_per_measurement: int,
    measurements: int = 1,
    *,
    inter_measurement_rotation_deg: float = GOLDEN_ANGLE_DEG,
) -> np.ndarray:
    """Return the Piccini spiral-phyllotaxis center-through directions.

    Only one hemisphere is required because a center-through radial line also
    samples the antipodal half. The polar angle follows
    ``theta(n) = pi/2 * sqrt(n/N)`` and the azimuth advances by the golden
    angle, as defined for the conventional spiral-phyllotaxis trajectory.
    """
    spokes = _positive_integer(spokes_per_measurement, "spokes_per_measurement")
    measurement_count = _positive_integer(measurements, "measurements")
    if not np.isfinite(inter_measurement_rotation_deg):
        raise ValueError("inter-measurement rotation must be finite")
    index = np.arange(spokes, dtype=float)
    theta = 0.5 * np.pi * np.sqrt(index / spokes)
    z = np.cos(theta)
    radius = np.sin(theta)
    base_phi = np.deg2rad(GOLDEN_ANGLE_DEG) * index
    result = np.empty((measurement_count, spokes, 3), dtype=float)
    for measurement in range(measurement_count):
        phi = base_phi + np.deg2rad(inter_measurement_rotation_deg * measurement)
        result[measurement, :, 0] = radius * np.cos(phi)
        result[measurement, :, 1] = radius * np.sin(phi)
        result[measurement, :, 2] = z
    result.setflags(write=False)
    return result


def _vector_trapezoids(
    pp,
    system,
    area_vector: np.ndarray,
    *,
    duration_s: float | None = None,
    flat_time_s: float | None = None,
    rise_time_s: float | None = None,
):
    events = []
    for axis, area in zip("xyz", np.asarray(area_vector, dtype=float)):
        if abs(area) < 1e-12:
            continue
        if flat_time_s is None:
            event = pp.make_trapezoid(
                axis, area=float(area), duration=duration_s, system=system
            )
        else:
            event = pp.make_trapezoid(
                axis,
                flat_area=float(area),
                flat_time=flat_time_s,
                rise_time=rise_time_s,
                system=system,
            )
        events.append(event)
    return tuple(events)


def make_pulseq_radial_me_bssfp(
    *,
    fov_m: float = 356e-3,
    base_resolution: int = 32,
    readout_oversampling: int = 2,
    spokes_per_measurement: int = 300,
    measurements: int = 4,
    echoes: int = 5,
    echo_spacing_s: float = 2e-3,
    pixel_bandwidth_hz: float = 1000.0,
    flip_angle_deg: float = 10.0,
    rf_duration_s: float = 0.5e-3,
    repetition_time_s: float = 16e-3,
    rf_phase_start_deg: float = 0.0,
    rf_phase_increment_deg: float = 180.0,
    use_alpha_half: bool = True,
    use_tip_back: bool = True,
    prephaser_duration_s: float = 0.5e-3,
    inter_measurement_rotation_deg: float = GOLDEN_ANGLE_DEG,
    field_strength_t: float = 3.0,
    nucleus: str = "C13",
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
):
    """Build the 3D radial five-echo bSSFP acquisition from Wang et al.

    Readouts are monopolar and center-through. A rewinder follows every echo,
    and the initial/final half-area lobes make the complete TR balanced.
    Measurements reuse one spherical spiral-phyllotaxis pattern with the
    published golden-angle z rotation between consecutive measurements.
    """
    pp = _pypulseq()
    if not np.isfinite(fov_m) or fov_m <= 0:
        raise ValueError("fov_m must be positive and finite")
    base_resolution = _positive_integer(base_resolution, "base_resolution")
    readout_oversampling = _positive_integer(
        readout_oversampling, "readout_oversampling"
    )
    spokes_per_measurement = _positive_integer(
        spokes_per_measurement, "spokes_per_measurement"
    )
    measurements = _positive_integer(measurements, "measurements")
    echoes = _positive_integer(echoes, "echoes")
    if echoes % 2 == 0:
        raise ValueError("echoes must be odd so the echo train can be centered at TR/2")
    positive_parameters = {
        "echo_spacing_s": echo_spacing_s,
        "pixel_bandwidth_hz": pixel_bandwidth_hz,
        "flip_angle_deg": flip_angle_deg,
        "rf_duration_s": rf_duration_s,
        "repetition_time_s": repetition_time_s,
        "prephaser_duration_s": prephaser_duration_s,
        "field_strength_t": field_strength_t,
    }
    for name, value in positive_parameters.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    nucleus = str(nucleus).strip()
    if not nucleus:
        raise ValueError("nucleus must not be empty")

    system = _make_system(
        pp,
        scanner_parameters,
        legacy_max_grad_mtm=32.0,
        legacy_max_slew_tms=130.0,
    )
    sequence = pp.Sequence(system)
    sample_count = base_resolution * readout_oversampling
    total_sampling_bandwidth = pixel_bandwidth_hz * sample_count
    dwell = _adc_dwell_for_bandwidth(
        sample_count=sample_count,
        sampling_bandwidth_hz=total_sampling_bandwidth,
        adc_raster_s=system.adc_raster_time,
        grad_raster_s=system.grad_raster_time,
    )
    readout_flat_time = sample_count * dwell
    readout_area = sample_count / float(fov_m)
    readout_amplitude = readout_area / readout_flat_time
    readout_rise = max(
        system.adc_dead_time,
        np.ceil(readout_amplitude / system.max_slew / system.grad_raster_time)
        * system.grad_raster_time,
    )
    # ``flat_area`` excludes the two ramps. Balancing uses the full trapezoid
    # moment; the half-area prephaser then places the ADC flat top exactly at
    # -kmax .. +kmax after the first half-ramp has been traversed.
    readout_total_area = readout_area * (1.0 + readout_rise / readout_flat_time)
    readout_block_duration = readout_flat_time + 2 * readout_rise
    flyback_duration = echo_spacing_s - readout_block_duration
    flyback_duration = round(flyback_duration / system.grad_raster_time) * (
        system.grad_raster_time
    )
    if flyback_duration <= 0:
        raise ValueError(
            "echo_spacing_s is too short for the requested readout bandwidth"
        )

    rf = pp.make_block_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=rf_duration_s,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )
    alpha_half = pp.make_block_pulse(
        flip_angle=np.deg2rad(flip_angle_deg / 2),
        duration=rf_duration_s,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )
    tip_back = pp.make_block_pulse(
        flip_angle=np.deg2rad(flip_angle_deg / 2),
        duration=rf_duration_s,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )
    rf_center, _ = pp.calc_rf_center(rf)
    rf_center_from_start = rf.delay + rf_center
    rf_block_duration = pp.calc_duration(rf)
    alpha_center, _ = pp.calc_rf_center(alpha_half)
    alpha_center_from_start = alpha_half.delay + alpha_center
    raster = system.block_duration_raster

    middle_echo = echoes // 2
    adc_center_from_readout_start = readout_rise + readout_flat_time / 2
    delay_before_prephaser = (
        repetition_time_s / 2
        + rf_center_from_start
        - rf_block_duration
        - prephaser_duration_s
        - adc_center_from_readout_start
        - middle_echo * echo_spacing_s
    )
    delay_before_prephaser = round(delay_before_prephaser / raster) * raster
    if delay_before_prephaser < 0:
        raise ValueError("TR is too short to center the echo train at TR/2")
    train_duration = echoes * readout_block_duration + (echoes - 1) * flyback_duration
    occupied_tr = (
        rf_block_duration
        + delay_before_prephaser
        + prephaser_duration_s
        + train_duration
        + prephaser_duration_s
    )
    trailing_delay = round((repetition_time_s - occupied_tr) / raster) * raster
    if trailing_delay < 0:
        raise ValueError(
            f"repetition_time_s is too short; minimum is {occupied_tr:.9g} s"
        )
    actual_tr = occupied_tr + trailing_delay
    actual_echo_spacing = readout_block_duration + flyback_duration

    if use_alpha_half:
        alpha_half.phase_offset = np.deg2rad(rf_phase_start_deg)
        sequence.add_block(alpha_half)
        preparation_delay = (
            actual_tr / 2
            - pp.calc_duration(alpha_half)
            + alpha_center_from_start
            - rf_center_from_start
        )
        preparation_delay = round(preparation_delay / raster) * raster
        if preparation_delay < 0:
            raise ValueError("TR/2 is too short for alpha/2 preparation")
        if preparation_delay:
            sequence.add_block(pp.make_delay(preparation_delay))

    directions = spiral_phyllotaxis_directions(
        spokes_per_measurement,
        measurements,
        inter_measurement_rotation_deg=inter_measurement_rotation_deg,
    )
    rf_phase = float(rf_phase_start_deg)
    for measurement in range(measurements):
        for spoke in range(spokes_per_measurement):
            direction = directions[measurement, spoke]
            rf.phase_offset = np.deg2rad(rf_phase)
            rf_phase = (rf_phase + rf_phase_increment_deg) % 360.0
            sequence.add_block(rf)
            if delay_before_prephaser:
                sequence.add_block(pp.make_delay(delay_before_prephaser))
            prephasers = _vector_trapezoids(
                pp,
                system,
                -0.5 * readout_total_area * direction,
                duration_s=prephaser_duration_s,
            )
            sequence.add_block(*prephasers)
            for echo in range(echoes):
                readouts = _vector_trapezoids(
                    pp,
                    system,
                    readout_area * direction,
                    flat_time_s=readout_flat_time,
                    rise_time_s=readout_rise,
                )
                adc = pp.make_adc(
                    num_samples=sample_count,
                    dwell=dwell,
                    delay=readout_rise,
                    phase_offset=np.deg2rad(rf_phase - rf_phase_increment_deg),
                    system=system,
                )
                sequence.add_block(
                    *readouts,
                    adc,
                    pp.make_label("LIN", "SET", spoke),
                    pp.make_label("ECO", "SET", echo),
                    pp.make_label("REP", "SET", measurement),
                )
                if echo < echoes - 1:
                    flybacks = _vector_trapezoids(
                        pp,
                        system,
                        -readout_total_area * direction,
                        duration_s=flyback_duration,
                    )
                    sequence.add_block(*flybacks)
            postphasers = _vector_trapezoids(
                pp,
                system,
                -0.5 * readout_total_area * direction,
                duration_s=prephaser_duration_s,
            )
            sequence.add_block(*postphasers)
            if trailing_delay:
                sequence.add_block(pp.make_delay(trailing_delay))

    if use_tip_back:
        tip_back.phase_offset = np.deg2rad(rf_phase + 180.0)
        sequence.add_block(tip_back)

    _raise_for_timing_errors(sequence, "radial multi-echo bSSFP")
    echo_times = [
        actual_tr / 2 + (echo - middle_echo) * actual_echo_spacing
        for echo in range(echoes)
    ]
    sequence.set_definition("Name", "radial_me_bssfp_3d")
    sequence.set_definition("TrajectoryType", "radial_3d_spiral_phyllotaxis")
    sequence.set_definition("FOV", [float(fov_m)] * 3)
    sequence.set_definition("MatrixSize", [sample_count] * 3)
    sequence.set_definition("BaseResolution", base_resolution)
    sequence.set_definition("ReadoutOversampling", readout_oversampling)
    sequence.set_definition("ReadoutSamples", sample_count)
    sequence.set_definition("SpokesPerMeasurement", spokes_per_measurement)
    sequence.set_definition("Measurements", measurements)
    sequence.set_definition("DynamicFrames", measurements)
    sequence.set_definition("Echoes", echoes)
    sequence.set_definition("EchoTimes", echo_times)
    sequence.set_definition("EchoSpacing", actual_echo_spacing)
    actual_pixel_bandwidth_hz = 1.0 / (dwell * sample_count)
    sequence.set_definition("PixelBandwidthHz", actual_pixel_bandwidth_hz)
    sequence.set_definition("RequestedPixelBandwidthHz", pixel_bandwidth_hz)
    sequence.set_definition("SamplingBandwidth", 1.0 / dwell)
    sequence.set_definition("ADCDwell", dwell)
    sequence.set_definition("FlipAngleDeg", float(flip_angle_deg))
    sequence.set_definition("TR", actual_tr)
    sequence.set_definition(
        "AcquisitionTimePerMeasurement", spokes_per_measurement * actual_tr
    )
    sequence.set_definition(
        "RadialAcquisitionTime",
        measurements * spokes_per_measurement * actual_tr,
    )
    sequence.set_definition("RFPhaseStartDeg", float(rf_phase_start_deg))
    sequence.set_definition("RFPhaseIncrementDeg", float(rf_phase_increment_deg))
    sequence.set_definition("UseAlphaHalf", bool(use_alpha_half))
    sequence.set_definition("UseTipBack", bool(use_tip_back))
    sequence.set_definition("GradientBalancing", "per_tr_xyz")
    sequence.set_definition("ReadoutPolarity", "monopolar")
    sequence.set_definition("CenterThroughReadout", True)
    sequence.set_definition("PhyllotaxisGoldenAngleDeg", GOLDEN_ANGLE_DEG)
    sequence.set_definition("PhyllotaxisReferenceDOI", PICCINI_REFERENCE_DOI)
    sequence.set_definition(
        "InterMeasurementRotationDeg", inter_measurement_rotation_deg
    )
    sequence.set_definition("FieldStrengthT", float(field_strength_t))
    sequence.set_definition("Nucleus", nucleus)
    sequence.set_definition("ReferenceDOI", WANG_REFERENCE_DOI)
    return sequence
