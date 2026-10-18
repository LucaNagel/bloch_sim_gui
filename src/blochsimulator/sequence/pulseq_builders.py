"""Configurable Pulseq sequence builders used by the desktop simulator."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import json
from typing import Mapping, Sequence

import numpy as np

from ..units import NUCLEUS_GAMMA_HZ_PER_T

from .flip_angles import VFA_REFERENCE_DOI, variable_flip_angle_schedule
from .encoding import (
    EncodingFrame,
    logical_gradient_area,
    make_role_trapezoid,
    resolve_encoding_frame,
    set_pulseq_encoding_definitions,
)
from .rf_pulses import (
    make_pulseq_rf_events,
    normalize_rf_pulse_type,
    set_rf_definitions,
    shifted_spectral_spatial_rf_event,
    spectral_spatial_refocus_area,
)
from .scanner import ScannerParameters
from .spectral_spatial import (
    SpectralSpatialPulseLibrary,
    SpectralSpatialSchedule,
    migrate_spectral_spatial_state,
    validate_spectral_spatial_schedule,
)
from .bssfp_phase import (
    advance_bssfp_phase_deg,
    pulseq_phase_offset_rad,
    wrap_phase_deg,
)


def _pypulseq():
    try:
        import pypulseq as pp
    except ImportError as exc:  # pragma: no cover - depends on optional install
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


def _ceil_to_raster(value: float, raster: float) -> float:
    """Round a non-negative duration up without changing exact multiples."""
    return float(np.ceil((float(value) - 1e-12) / raster) * raster)


def _centered_adc_dwell(requested_dwell: float, sample_count: int, system) -> float:
    """Choose the closest no-faster dwell that permits an exactly centred ADC.

    ADC dwell uses the fine ADC raster, while the ADC start uses the RF raster.
    For bipolar Cartesian readouts the complete sample window must also remain
    centred on the gradient plateau so forward and reverse lines share one
    k-space grid.  Increasing dwell in ADC-raster steps lowers bandwidth and is
    therefore the conservative direction when selecting a compatible value.
    """
    first_step = max(
        1,
        int(np.ceil((float(requested_dwell) - 1e-15) / system.adc_raster_time)),
    )
    for dwell_steps in range(first_step, first_step + 100_000):
        dwell = dwell_steps * system.adc_raster_time
        adc_duration = int(sample_count) * dwell
        flat_time = _ceil_to_raster(adc_duration, system.grad_raster_time)
        centered_offset = (flat_time - adc_duration) / 2.0
        raster_steps = centered_offset / system.rf_raster_time
        if abs(raster_steps - round(raster_steps)) < 1e-6:
            return float(dwell)
    raise ValueError(
        "sampling bandwidth cannot be aligned to the ADC and RF timing rasters"
    )


def _sequence_duration_s(sequence) -> float:
    return 0.0 if not sequence.block_events else float(sequence.duration()[0])


def _rf_spoiling_phase_deg(excitation_index: int, increment_deg: float) -> float:
    """Return the quadratic RF-spoiling phase for one excitation."""
    return (
        float(increment_deg * excitation_index * (excitation_index + 1) / 2.0) % 360.0
    )


def _uses_rf_spoiling(enabled: bool, increment_deg: float) -> bool:
    """Return whether the configured phase increment produces RF spoiling."""
    return bool(enabled) and not np.isclose(float(increment_deg) % 360.0, 0.0)


def _finish_acquisition_interval(
    pp,
    sequence,
    *,
    acquisition_start_s: float,
    requested_interval_s: float | None,
    raster_s: float,
    acquisition_name: str,
) -> tuple[float, float]:
    """Pad one complete acquisition to a requested start-to-start interval."""
    elapsed_s = _sequence_duration_s(sequence) - float(acquisition_start_s)
    if requested_interval_s is None:
        return elapsed_s, 0.0
    if not np.isfinite(requested_interval_s) or requested_interval_s <= 0:
        raise ValueError("acquisition_interval_s must be positive and finite or None")
    if requested_interval_s < elapsed_s - 1e-12:
        raise ValueError(
            f"acquisition_interval_s is too short for one {acquisition_name}; "
            f"minimum is {elapsed_s:.9g} s"
        )
    delay_s = _ceil_to_raster(max(0.0, requested_interval_s - elapsed_s), raster_s)
    if delay_s:
        sequence.add_block(pp.make_delay(delay_s))
    return elapsed_s + delay_s, delay_s


def _set_acquisition_interval_definitions(
    sequence,
    *,
    requested_interval_s: float | None,
    actual_intervals_s: Sequence[float],
    minimum_intervals_s: Sequence[float],
    start_times_s: Sequence[float],
) -> None:
    """Store explicit full-acquisition start-to-start timing metadata."""
    actual = tuple(float(value) for value in actual_intervals_s)
    minimum = tuple(float(value) for value in minimum_intervals_s)
    starts = tuple(float(value) for value in start_times_s)
    if not actual or len(actual) != len(minimum) or len(actual) != len(starts):
        raise ValueError(
            "acquisition interval metadata must have equal non-empty lengths"
        )
    sequence.set_definition("AcquisitionIntervalReference", "start-to-start")
    sequence.set_definition("AcquisitionStartTimes", list(starts))
    sequence.set_definition("AcquisitionInterval", max(actual))
    sequence.set_definition("MinimumAcquisitionInterval", max(minimum))
    if requested_interval_s is not None:
        sequence.set_definition(
            "RequestedAcquisitionInterval", float(requested_interval_s)
        )
    if len(actual) > 1 and not np.allclose(actual, actual[0], rtol=0.0, atol=1e-12):
        sequence.set_definition("AcquisitionIntervals", list(actual))
    if len(minimum) > 1 and not np.allclose(minimum, minimum[0], rtol=0.0, atol=1e-12):
        sequence.set_definition("MinimumAcquisitionIntervals", list(minimum))


def _make_scanner_system(
    pp,
    scanner_parameters: ScannerParameters | Mapping[str, float] | None,
    *,
    legacy_kwargs: Mapping[str, float | str],
):
    """Create Pulseq limits while preserving legacy standalone defaults."""
    if scanner_parameters is None:
        return pp.Opts(**dict(legacy_kwargs))
    profile = ScannerParameters.from_mapping(scanner_parameters)
    return pp.Opts(**profile.to_pypulseq_kwargs())


def _make_slice_selective_rf_events(
    pp,
    system,
    *,
    flip_angle_schedule_deg: Sequence[float],
    slice_thickness_m: float,
    rf_pulse_type: str,
    rf_duration_s: float,
    rf_time_bandwidth_product: float,
    rf_apodization: float,
    rf_slr_sharpness: int,
    rf_custom_waveform_hz: Sequence[complex] | None,
    rf_custom_raster_s: float | None,
    rf_custom_flip_angle_deg: float | None,
    rf_frequency_offset_hz: float,
    rf_spectral_spatial_pulse=None,
    spatial_offset_m: float = 0.0,
    nucleus: str | None = None,
):
    """Return slice-selective RF/gradient pairs built from one shared shape."""
    return make_pulseq_rf_events(
        pp,
        system,
        flip_angles_deg=flip_angle_schedule_deg,
        pulse_type=rf_pulse_type,
        duration_s=rf_duration_s,
        time_bandwidth_product=rf_time_bandwidth_product,
        apodization=rf_apodization,
        slr_sharpness=rf_slr_sharpness,
        custom_waveform_hz=rf_custom_waveform_hz,
        custom_raster_s=rf_custom_raster_s,
        custom_flip_angle_deg=rf_custom_flip_angle_deg,
        slice_thickness_m=slice_thickness_m,
        frequency_offset_hz=rf_frequency_offset_hz,
        spectral_spatial_pulse=rf_spectral_spatial_pulse,
        spatial_offset_m=spatial_offset_m,
        nucleus=nucleus,
    )


def _remap_gradient_event(event, frame: EncodingFrame, role: str):
    """Copy one Pulseq gradient onto the physical axis for an encoding role."""
    mapped = copy(event)
    axis, sign = frame.axis_and_sign(role)
    mapped.channel = axis
    for attribute in ("amplitude", "area", "flat_area", "first", "last"):
        if hasattr(mapped, attribute):
            value = getattr(mapped, attribute)
            if value is not None:
                setattr(mapped, attribute, sign * value)
    if hasattr(mapped, "waveform"):
        mapped.waveform = sign * np.asarray(mapped.waveform)
    return mapped


def _slice_rephase_area(
    rf_event,
    gradient_event,
    frame: EncodingFrame,
    *,
    rf_pulse_type: str,
    gradient_raster_s: float,
) -> float:
    """Return the logical partition area that refocuses the excited slice."""
    if rf_pulse_type == "spectral_spatial":
        _, sign = frame.axis_and_sign("partition")
        return sign * spectral_spatial_refocus_area(
            rf_event, gradient_event, gradient_raster_s=gradient_raster_s
        )
    return -logical_gradient_area(gradient_event, frame, "partition") / 2.0


def _set_rf_definitions(
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
    spectral_spatial_pulse=None,
) -> None:
    """Persist the RF design inputs needed to reproduce a sequence."""
    set_rf_definitions(
        sequence,
        pulse_type=pulse_type,
        requested_duration_s=requested_duration_s,
        actual_duration_s=actual_duration_s,
        time_bandwidth_product=time_bandwidth_product,
        apodization=apodization,
        slr_sharpness=slr_sharpness,
        custom_name=custom_name,
        custom_flip_angle_deg=custom_flip_angle_deg,
        frequency_offset_hz=frequency_offset_hz,
        spectral_spatial_pulse=spectral_spatial_pulse,
    )


def _phase_encoding_indices(
    n_x: int,
    n_y: int,
    order: str,
    fov_m: tuple[float, float],
) -> list[tuple[int, int]]:
    if order == "linear":
        return [(x, y) for y in range(n_y) for x in range(n_x)]
    if order == "spiral":
        center_x, center_y = n_x // 2, n_y // 2
        result = [(center_x, center_y)]
        x, y = center_x, center_y
        step_length = 1
        directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        direction_index = 0
        while len(result) < n_x * n_y:
            for _ in range(2):
                dx, dy = directions[direction_index % 4]
                direction_index += 1
                for _ in range(step_length):
                    x += dx
                    y += dy
                    if 0 <= x < n_x and 0 <= y < n_y:
                        result.append((x, y))
            step_length += 1
        return result
    if order != "centric":
        raise ValueError("phase encoding order must be linear, spiral, or centric")

    center_x, center_y = n_x // 2, n_y // 2
    fov_x, fov_y = fov_m

    def key(index: tuple[int, int]) -> tuple[float, float]:
        x, y = index
        k_x = (x - center_x) / fov_x
        k_y = (y - center_y) / fov_y
        return (
            float(np.hypot(k_x, k_y)),
            float((np.arctan2(k_y, k_x) + np.pi) % (2 * np.pi)),
        )

    return sorted(
        ((x, y) for y in range(n_y) for x in range(n_x)),
        key=key,
    )


def make_pulseq_slice_selection(
    *,
    flip_angle_deg: float = 90.0,
    rf_phase_deg: float = 0.0,
    rf_pulse_type: str = "sinc",
    rf_duration_s: float = 5e-3,
    rf_time_bandwidth_product: float = 4.0,
    rf_apodization: float = 0.5,
    rf_slr_sharpness: int = 1,
    rf_custom_waveform_hz: Sequence[complex] | None = None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    rf_custom_name: str | None = None,
    rf_spectral_spatial_pulse=None,
    rf_frequency_offset_hz: float = 0.0,
    slice_thickness_m: float = 3e-3,
    slice_offset_m: float = 0.0,
    slice_axis: str = "+z",
    slice_gradient_enabled: bool = True,
    rephase: bool = False,
    rephase_duration_s: float = 3e-3,
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
    nucleus: str = "H1",
):
    """Build one RF excitation with optional slice selection and rephasing.

    There is no ADC or spoiler, and no delay after the final gradient. The
    declared magnetization checkpoints retain the state after slice selection
    and, when enabled, after rephasing. Offsets follow the signed slice axis.
    With ``slice_gradient_enabled=False``, only RF is applied; slice position
    and rephasing are ignored, and the final state is taken after the RF block.
    """
    for name, value in (
        ("flip_angle_deg", flip_angle_deg),
        ("rf_phase_deg", rf_phase_deg),
        ("slice_offset_m", slice_offset_m),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if flip_angle_deg <= 0:
        raise ValueError("flip_angle_deg must be positive")
    axis_code = str(slice_axis).lower()
    if axis_code not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ValueError("slice_axis must be +x, -x, +y, -y, +z, or -z")
    rephase = bool(rephase and slice_gradient_enabled)
    if rephase and (not np.isfinite(rephase_duration_s) or rephase_duration_s <= 0):
        raise ValueError("rephase_duration_s must be positive and finite")
    pp = _pypulseq()
    if nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
        raise ValueError(f"unknown nucleus: {nucleus}")
    system = pp.Opts(
        **ScannerParameters.from_mapping(scanner_parameters).to_pypulseq_kwargs(),
        gamma=NUCLEUS_GAMMA_HZ_PER_T[nucleus],
    )
    sequence = pp.Sequence(system)
    raw, actual_duration, tbw, pulse_type = make_pulseq_rf_events(
        pp,
        system,
        flip_angles_deg=(flip_angle_deg,),
        slice_thickness_m=slice_thickness_m if slice_gradient_enabled else None,
        pulse_type=rf_pulse_type,
        duration_s=rf_duration_s,
        time_bandwidth_product=rf_time_bandwidth_product,
        apodization=rf_apodization,
        slr_sharpness=rf_slr_sharpness,
        custom_waveform_hz=rf_custom_waveform_hz,
        custom_raster_s=rf_custom_raster_s,
        custom_flip_angle_deg=rf_custom_flip_angle_deg,
        frequency_offset_hz=rf_frequency_offset_hz,
        spectral_spatial_pulse=rf_spectral_spatial_pulse,
        spatial_offset_m=slice_offset_m,
        nucleus=nucleus,
        include_slice_gradient=slice_gradient_enabled,
    )
    rf, logical_gradient = raw[0] if slice_gradient_enabled else (raw[0], None)
    rf = copy(rf)
    if slice_gradient_enabled and pulse_type != "spectral_spatial":
        rf.freq_offset += logical_gradient.amplitude * slice_offset_m
    rf.phase_offset = pulseq_phase_offset_rad(
        rf_phase_deg,
        frequency_offset_hz=rf.freq_offset,
        event_center_s=pp.calc_rf_center(rf)[0],
    )
    sign = -1.0 if axis_code[0] == "-" else 1.0
    if slice_gradient_enabled:
        gradient = copy(logical_gradient)
        gradient.channel = axis_code[-1]
        for attribute in (
            "amplitude",
            "area",
            "flat_area",
            "first",
            "last",
            "waveform",
        ):
            value = getattr(gradient, attribute, None)
            if value is not None:
                setattr(gradient, attribute, sign * value)
        sequence.add_block(rf, gradient)
    else:
        block_duration = _ceil_to_raster(
            pp.calc_duration(rf), system.block_duration_raster
        )
        sequence.add_block(rf, pp.make_delay(block_duration))
    selection_end = _sequence_duration_s(sequence)
    sequence.set_definition(
        "SliceSelectionEndTime" if slice_gradient_enabled else "RFExcitationEndTime",
        selection_end,
    )
    checkpoints = []
    if rephase:
        rf_center = rf.delay + pp.calc_rf_center(rf)[0]
        # Rewind the moment after the RF centre, including the falling ramp.
        # For symmetric RF and slice gradients this is half the total area.
        t = max(0.0, rf_center - logical_gradient.delay)
        if logical_gradient.type == "trap":
            rise, flat, fall = (
                logical_gradient.rise_time,
                logical_gradient.flat_time,
                logical_gradient.fall_time,
            )
            before = 0.5 * min(t, rise) ** 2 / rise if rise else 0.0
            before += np.clip(t - rise, 0.0, flat)
            tail = np.clip(t - rise - flat, 0.0, fall)
            before += tail - 0.5 * tail**2 / fall if fall else 0.0
            moment = logical_gradient.area - logical_gradient.amplitude * before
        elif pulse_type == "spectral_spatial":
            # SPSP subpulses each sit on their own lobe, so the moment after
            # the overall RF centre does not describe the slice phase.
            moment = -spectral_spatial_refocus_area(
                rf, logical_gradient, gradient_raster_s=system.grad_raster_time
            )
        else:
            waveform = np.asarray(logical_gradient.waveform)
            starts = np.arange(waveform.size) * system.grad_raster_time
            weights = np.clip(
                starts + system.grad_raster_time - t, 0.0, system.grad_raster_time
            )
            moment = float(np.sum(waveform * weights))
        if abs(moment) > 1e-12:
            sequence.add_block(
                pp.make_trapezoid(
                    channel=gradient.channel,
                    area=-sign * moment,
                    duration=rephase_duration_s,
                    system=system,
                )
            )
        rephasing_end = _sequence_duration_s(sequence)
        sequence.set_definition("RephasingEndTime", rephasing_end)
        if rephasing_end > selection_end:
            checkpoints.append(selection_end)
    sequence.set_definition(
        "Name", "rf_slice_selection" if slice_gradient_enabled else "rf_excitation"
    )
    sequence.set_definition("MagnetizationOnly", 1)
    sequence.set_definition("SliceGradientEnabled", int(bool(slice_gradient_enabled)))
    if checkpoints:
        sequence.set_definition("MagnetizationCheckpointTimes", checkpoints)
    sequence.set_definition("FlipAngleDeg", flip_angle_deg)
    sequence.set_definition("RFPhaseDeg", rf_phase_deg)
    sequence.set_definition("SliceThickness", slice_thickness_m)
    sequence.set_definition("SliceOffset", slice_offset_m)
    sequence.set_definition("SliceGradientAxis", axis_code)
    sequence.set_definition("RephasingEnabled", int(bool(rephase)))
    sequence.set_definition("RephasingDuration", rephase_duration_s if rephase else 0.0)
    sequence.set_definition("Nucleus", str(nucleus))
    _set_rf_definitions(
        sequence,
        pulse_type=pulse_type,
        requested_duration_s=rf_duration_s,
        actual_duration_s=actual_duration,
        time_bandwidth_product=tbw,
        apodization=rf_apodization,
        slr_sharpness=rf_slr_sharpness,
        custom_name=rf_custom_name,
        custom_flip_angle_deg=rf_custom_flip_angle_deg,
        frequency_offset_hz=rf_frequency_offset_hz,
        spectral_spatial_pulse=rf_spectral_spatial_pulse,
    )
    _raise_for_timing_errors(sequence, "RF + Slice Selection")
    return sequence


def make_pulseq_csi(
    *,
    fov_m: Sequence[float] = (0.21, 0.21),
    matrix: Sequence[int] = (8, 8),
    slice_thickness_m: float = 10e-3,
    spectral_bandwidth_hz: float = 4000.0,
    spectral_points: int = 128,
    phase_encoding_order: str = "linear",
    flip_angle_deg: float = 15.0,
    variable_flip_angle: bool = False,
    vfa_final_flip_angle_deg: float = 90.0,
    rf_pulse_type: str = "sinc",
    rf_duration_s: float = 3e-3,
    rf_time_bandwidth_product: float = 4.0,
    rf_apodization: float = 0.5,
    rf_slr_sharpness: int = 1,
    rf_custom_waveform_hz: Sequence[complex] | None = None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    rf_custom_name: str | None = None,
    rf_spectral_spatial_pulse=None,
    rf_frequency_offset_hz: float = 0.0,
    encoding_duration_s: float = 2e-3,
    echo_time_s: float = 6e-3,
    repetition_time_s: float = 0.1,
    repetitions: int = 1,
    acquisition_interval_s: float | None = None,
    rf_spoiling: bool = False,
    rf_spoiling_increment_deg: float = 117.0,
    slice_offset_m: float = 0.0,
    encoding_axes: Sequence[str] | EncodingFrame = ("+x", "+y", "+z"),
    spoil_after_readout: bool = True,
    spoiler_cycles_per_slice: float = 4.0,
    spoiler_cycles_per_voxel: float = 0.0,
    spoiler_duration_s: float = 2e-3,
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
    nucleus: str = "H1",
):
    """Build a repeated slice-selective Cartesian 2D CSI Pulseq sequence.

    Variable flip angles advance with the chronological phase-encoding order
    and restart at the beginning of every complete CSI repetition.
    """
    pp = _pypulseq()
    encoding_frame = resolve_encoding_frame(encoding_axes)
    fov_x, fov_y = _positive_values(fov_m, "FOV")
    if len(tuple(fov_m)) != 2:
        raise ValueError("fov_m must contain two values")
    n_x, n_y = (_positive_integer(value, "matrix size") for value in matrix)
    if len(tuple(matrix)) != 2:
        raise ValueError("matrix must contain two values")
    spectral_points = _positive_integer(spectral_points, "spectral_points")
    repetitions = _positive_integer(repetitions, "repetitions")
    positive_parameters = {
        "slice_thickness_m": slice_thickness_m,
        "spectral_bandwidth_hz": spectral_bandwidth_hz,
        "flip_angle_deg": flip_angle_deg,
        "rf_duration_s": rf_duration_s,
        "encoding_duration_s": encoding_duration_s,
        "echo_time_s": echo_time_s,
        "repetition_time_s": repetition_time_s,
        "spoiler_duration_s": spoiler_duration_s,
    }
    for name, value in positive_parameters.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value in {
        "spoiler_cycles_per_slice": spoiler_cycles_per_slice,
        "spoiler_cycles_per_voxel": spoiler_cycles_per_voxel,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    if not np.isfinite(slice_offset_m):
        raise ValueError("slice_offset_m must be finite")
    if not np.isfinite(rf_frequency_offset_hz):
        raise ValueError("rf_frequency_offset_hz must be finite")
    if not np.isfinite(rf_spoiling_increment_deg):
        raise ValueError("rf_spoiling_increment_deg must be finite")
    rf_spoiling = _uses_rf_spoiling(rf_spoiling, rf_spoiling_increment_deg)

    order = _phase_encoding_indices(n_x, n_y, phase_encoding_order, (fov_x, fov_y))
    if variable_flip_angle:
        flip_angle_schedule_deg = variable_flip_angle_schedule(
            len(order),
            final_flip_angle_deg=vfa_final_flip_angle_deg,
        )
    else:
        flip_angle_schedule_deg = np.asarray([flip_angle_deg], dtype=float)
    system = _make_scanner_system(
        pp,
        scanner_parameters,
        legacy_kwargs={
            "max_grad": 32,
            "grad_unit": "mT/m",
            "max_slew": 130,
            "slew_unit": "T/m/s",
            "grad_raster_time": 10e-6,
            "rf_ringdown_time": 30e-6,
            "rf_dead_time": 100e-6,
            "adc_dead_time": 20e-6,
        },
    )
    sequence = pp.Sequence(system)
    requested_dwell = 1.0 / float(spectral_bandwidth_hz)
    dwell = round(requested_dwell / system.adc_raster_time) * system.adc_raster_time
    if dwell <= 0:
        raise ValueError("spectral bandwidth exceeds the ADC raster capability")
    actual_bandwidth = 1.0 / dwell

    raw_rf_events, actual_rf_duration_s, effective_rf_tbw, rf_pulse_type = (
        _make_slice_selective_rf_events(
            pp,
            system,
            flip_angle_schedule_deg=flip_angle_schedule_deg,
            slice_thickness_m=slice_thickness_m,
            rf_pulse_type=rf_pulse_type,
            rf_duration_s=rf_duration_s,
            rf_time_bandwidth_product=rf_time_bandwidth_product,
            rf_apodization=rf_apodization,
            rf_slr_sharpness=rf_slr_sharpness,
            rf_custom_waveform_hz=rf_custom_waveform_hz,
            rf_custom_raster_s=rf_custom_raster_s,
            rf_custom_flip_angle_deg=rf_custom_flip_angle_deg,
            rf_frequency_offset_hz=rf_frequency_offset_hz,
            rf_spectral_spatial_pulse=rf_spectral_spatial_pulse,
            spatial_offset_m=slice_offset_m,
            nucleus=nucleus,
        )
    )
    rf_events = [
        (rf, _remap_gradient_event(gz, encoding_frame, "partition"))
        for rf, gz in raw_rf_events
    ]
    rf, gz = rf_events[0]
    adc = pp.make_adc(
        num_samples=spectral_points,
        dwell=dwell,
        delay=system.adc_dead_time,
        system=system,
    )
    spoiler_events = []
    if spoil_after_readout:
        if spoiler_cycles_per_voxel > 0:
            for role, voxel_size in zip(("read", "phase"), (fov_x / n_x, fov_y / n_y)):
                spoiler_events.append(
                    make_role_trapezoid(
                        pp,
                        encoding_frame,
                        role,
                        area=spoiler_cycles_per_voxel / voxel_size,
                        duration=spoiler_duration_s,
                        system=system,
                    )
                )
        if spoiler_cycles_per_slice > 0:
            spoiler_events.append(
                make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "partition",
                    area=spoiler_cycles_per_slice / slice_thickness_m,
                    duration=spoiler_duration_s,
                    system=system,
                )
            )

    rf_center, _ = pp.calc_rf_center(rf)
    _, slice_sign = encoding_frame.axis_and_sign("partition")
    for rf_event, slice_gradient in rf_events:
        if rf_pulse_type == "spectral_spatial":
            frequency_offset_hz = rf_frequency_offset_hz
        else:
            logical_slice_amplitude = float(slice_gradient.amplitude) * slice_sign
            frequency_offset_hz = (
                rf_frequency_offset_hz + logical_slice_amplitude * float(slice_offset_m)
            )
        rf_event.freq_offset = frequency_offset_hz
        rf_event.phase_offset = -2.0 * np.pi * frequency_offset_hz * rf_center
    rf_block_duration = pp.calc_duration(rf, gz)
    first_sample_from_adc_start = adc.delay + adc.dwell / 2
    te_delay_value = (
        echo_time_s
        - (rf_block_duration - (rf.delay + rf_center))
        - encoding_duration_s
        - first_sample_from_adc_start
    )
    if te_delay_value < 0:
        minimum_te = echo_time_s - te_delay_value
        raise ValueError(f"echo_time_s is too short; minimum is {minimum_te:.9g} s")
    raster = system.block_duration_raster
    te_delay_value = _ceil_to_raster(te_delay_value, raster)
    repetition_without_delay = (
        rf_block_duration
        + encoding_duration_s
        + te_delay_value
        + pp.calc_duration(adc)
        + (max((pp.calc_duration(e) for e in spoiler_events), default=0.0))
    )
    tr_delay_value = repetition_time_s - repetition_without_delay
    if tr_delay_value < 0:
        raise ValueError(
            "repetition_time_s is too short; minimum is "
            f"{repetition_without_delay:.9g} s"
        )
    tr_delay_value = _ceil_to_raster(tr_delay_value, raster)
    actual_tr = repetition_without_delay + tr_delay_value
    actual_te = echo_time_s + max(
        0.0,
        te_delay_value
        - (
            echo_time_s
            - (rf_block_duration - (rf.delay + rf_center))
            - encoding_duration_s
            - first_sample_from_adc_start
        ),
    )

    x_areas = (np.arange(n_x) - n_x // 2) / fov_x
    y_areas = (np.arange(n_y) - n_y // 2) / fov_y
    spoiler_end_times = []
    acquisition_start_times = []
    acquisition_intervals = []
    minimum_acquisition_intervals = []
    excitation_index = 0
    for repetition in range(repetitions):
        acquisition_start = _sequence_duration_s(sequence)
        acquisition_start_times.append(acquisition_start)
        for encoding_index, (x_index, y_index) in enumerate(order):
            rf, gz = rf_events[encoding_index if variable_flip_angle else 0]
            rf_phase_deg = (
                _rf_spoiling_phase_deg(excitation_index, rf_spoiling_increment_deg)
                if rf_spoiling
                else 0.0
            )
            total_frequency_offset_hz = float(rf.freq_offset)
            rf.phase_offset = (
                np.deg2rad(rf_phase_deg)
                - 2.0 * np.pi * total_frequency_offset_hz * rf_center
            )
            adc.phase_offset = np.deg2rad(rf_phase_deg)
            gx_phase = make_role_trapezoid(
                pp,
                encoding_frame,
                "read",
                area=float(x_areas[x_index]),
                duration=encoding_duration_s,
                system=system,
            )
            gy_phase = make_role_trapezoid(
                pp,
                encoding_frame,
                "phase",
                area=float(y_areas[y_index]),
                duration=encoding_duration_s,
                system=system,
            )
            gz_rephase = make_role_trapezoid(
                pp,
                encoding_frame,
                "partition",
                area=_slice_rephase_area(
                    rf,
                    gz,
                    encoding_frame,
                    rf_pulse_type=rf_pulse_type,
                    gradient_raster_s=system.grad_raster_time,
                ),
                duration=encoding_duration_s,
                system=system,
            )
            sequence.add_block(rf, gz)
            sequence.add_block(gx_phase, gy_phase, gz_rephase)
            if te_delay_value:
                sequence.add_block(pp.make_delay(te_delay_value))
            sequence.add_block(
                adc,
                pp.make_label("LIN", "SET", x_index),
                pp.make_label("PAR", "SET", y_index),
                pp.make_label("REP", "SET", repetition),
            )
            if spoiler_events:
                sequence.add_block(*spoiler_events)
                spoiler_end_times.append(float(sequence.duration()[0]))
            if tr_delay_value:
                sequence.add_block(pp.make_delay(tr_delay_value))
            excitation_index += 1
        minimum_interval = _sequence_duration_s(sequence) - acquisition_start
        actual_interval, _ = _finish_acquisition_interval(
            pp,
            sequence,
            acquisition_start_s=acquisition_start,
            requested_interval_s=acquisition_interval_s,
            raster_s=system.block_duration_raster,
            acquisition_name="CSI image",
        )
        minimum_acquisition_intervals.append(minimum_interval)
        acquisition_intervals.append(actual_interval)

    _raise_for_timing_errors(sequence, "CSI")
    sequence.set_definition("Name", "csi_2d")
    sequence.set_definition("FOV", [fov_x, fov_y, slice_thickness_m])
    sequence.set_definition("MatrixSize", [n_x, n_y, spectral_points])
    set_pulseq_encoding_definitions(
        sequence,
        encoding_frame,
        fov_m=(fov_x, fov_y, slice_thickness_m),
        matrix=(n_x, n_y, 1),
    )
    sequence.set_definition("SpectralBandwidth", actual_bandwidth)
    sequence.set_definition("SpectralPoints", spectral_points)
    sequence.set_definition("SpectralResolution", actual_bandwidth / spectral_points)
    sequence.set_definition("PhaseEncodingOrder", phase_encoding_order)
    sequence.set_definition(
        "FlipAngleDeg",
        float(vfa_final_flip_angle_deg if variable_flip_angle else flip_angle_deg),
    )
    sequence.set_definition("VariableFlipAngle", bool(variable_flip_angle))
    if variable_flip_angle:
        sequence.set_definition("VariableFlipAngleDimension", "phase_encode")
        sequence.set_definition(
            "VariableFlipAngleFinalDeg", float(vfa_final_flip_angle_deg)
        )
        sequence.set_definition(
            "FlipAngleScheduleDeg",
            [float(value) for value in flip_angle_schedule_deg],
        )
        sequence.set_definition("VariableFlipAngleReferenceDOI", VFA_REFERENCE_DOI)
    sequence.set_definition("TR", actual_tr)
    sequence.set_definition("TE", actual_te)
    sequence.set_definition("Repetitions", repetitions)
    sequence.set_definition("VolumeInterval", max(acquisition_intervals))
    _set_acquisition_interval_definitions(
        sequence,
        requested_interval_s=acquisition_interval_s,
        actual_intervals_s=acquisition_intervals,
        minimum_intervals_s=minimum_acquisition_intervals,
        start_times_s=acquisition_start_times,
    )
    sequence.set_definition("SliceThickness", slice_thickness_m)
    sequence.set_definition("SliceOffset", float(slice_offset_m))
    sequence.set_definition("RFSpoiling", bool(rf_spoiling))
    sequence.set_definition("RFSpoilingIncrementDeg", rf_spoiling_increment_deg)
    _set_rf_definitions(
        sequence,
        pulse_type=rf_pulse_type,
        requested_duration_s=rf_duration_s,
        actual_duration_s=actual_rf_duration_s,
        time_bandwidth_product=effective_rf_tbw,
        apodization=rf_apodization,
        slr_sharpness=rf_slr_sharpness,
        custom_name=rf_custom_name,
        custom_flip_angle_deg=rf_custom_flip_angle_deg,
        frequency_offset_hz=rf_frequency_offset_hz,
        spectral_spatial_pulse=rf_spectral_spatial_pulse,
    )
    sequence.set_definition("SpoilAfterReadout", bool(spoil_after_readout))
    sequence.set_definition("SpoilerCyclesPerSlice", spoiler_cycles_per_slice)
    sequence.set_definition("SpoilerCyclesPerVoxel", spoiler_cycles_per_voxel)
    sequence.set_definition("SpoilerDuration", spoiler_duration_s)
    sequence.set_definition(
        "SpoilerAxes", "".join(event.channel for event in spoiler_events) or "none"
    )
    sequence.set_definition("SpoilerEndTimes", spoiler_end_times)
    sequence.set_definition("IdealSpoilerEndTimes", spoiler_end_times)
    return sequence


def make_pulseq_bssfp(
    *,
    fov_m: Sequence[float] = (0.22, 0.22, 0.16),
    matrix: Sequence[int] = (8, 8, 4),
    flip_angle_deg: float = 15.0,
    rf_pulse_type: str = "block",
    rf_duration_s: float = 1e-3,
    rf_time_bandwidth_product: float = 4.0,
    rf_apodization: float = 0.5,
    rf_slr_sharpness: int = 1,
    rf_custom_waveform_hz: Sequence[complex] | None = None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    rf_custom_name: str | None = None,
    rf_spectral_spatial_pulse=None,
    rf_frequency_offset_hz: float = 0.0,
    sampling_bandwidth_hz: float = 10_000.0,
    encoding_duration_s: float = 1e-3,
    repetition_time_s: float | None = 10e-3,
    rf_phase_start_deg: float = 180.0,
    rf_phase_increment_deg: float = 180.0,
    alpha_half_phase_deg: float | None = None,
    alpha_half_use_ratios: bool = True,
    alpha_half_tr_ratio: float = 0.5,
    alpha_half_flip_ratio: float = 0.5,
    alpha_half_center_spacing_s: float | None = None,
    alpha_half_flip_angle_deg: float | None = None,
    dummy_repetitions: int = 1,
    repetitions: int = 1,
    acquisition_interval_s: float | None = None,
    use_alpha_half: bool = True,
    encoding_axes: Sequence[str] | EncodingFrame = ("+x", "+y", "+z"),
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
    nucleus: str = "H1",
):
    """Build a fully balanced, non-selective Cartesian 3D bSSFP sequence.

    If ``alpha_half_phase_deg`` is omitted, the preparation pulse uses the
    phase-cycle member immediately preceding ``rf_phase_start_deg``. Startup
    spacing is measured between the preparation and first regular RF centers.
    Ratio mode preserves the conventional ``alpha/2 - TR/2`` preparation.
    """
    pp = _pypulseq()
    encoding_frame = resolve_encoding_frame(encoding_axes)
    fov_x, fov_y, fov_z = _positive_values(fov_m, "FOV")
    if len(tuple(fov_m)) != 3:
        raise ValueError("fov_m must contain three values")
    n_read, n_phase, n_partition = (
        _positive_integer(value, "matrix size") for value in matrix
    )
    if len(tuple(matrix)) != 3:
        raise ValueError("matrix must contain three values")
    dummy_repetitions = int(dummy_repetitions)
    if dummy_repetitions < 0:
        raise ValueError("dummy_repetitions must be non-negative")
    repetitions = _positive_integer(repetitions, "repetitions")
    for name, value in {
        "flip_angle_deg": flip_angle_deg,
        "rf_duration_s": rf_duration_s,
        "sampling_bandwidth_hz": sampling_bandwidth_hz,
        "encoding_duration_s": encoding_duration_s,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not np.isfinite(rf_frequency_offset_hz):
        raise ValueError("rf_frequency_offset_hz must be finite")
    if alpha_half_phase_deg is None:
        resolved_alpha_half_phase_deg = wrap_phase_deg(
            rf_phase_start_deg - rf_phase_increment_deg
        )
    else:
        resolved_alpha_half_phase_deg = wrap_phase_deg(alpha_half_phase_deg)
    if alpha_half_use_ratios:
        for name, value in {
            "alpha_half_tr_ratio": alpha_half_tr_ratio,
            "alpha_half_flip_ratio": alpha_half_flip_ratio,
        }.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        resolved_alpha_half_flip_angle_deg = float(flip_angle_deg) * float(
            alpha_half_flip_ratio
        )
    else:
        resolved_alpha_half_flip_angle_deg = (
            float(flip_angle_deg) / 2
            if alpha_half_flip_angle_deg is None
            else float(alpha_half_flip_angle_deg)
        )
        if (
            not np.isfinite(resolved_alpha_half_flip_angle_deg)
            or resolved_alpha_half_flip_angle_deg <= 0
        ):
            raise ValueError("alpha_half_flip_angle_deg must be positive and finite")
        if alpha_half_center_spacing_s is not None and (
            not np.isfinite(alpha_half_center_spacing_s)
            or alpha_half_center_spacing_s <= 0
        ):
            raise ValueError("alpha_half_center_spacing_s must be positive and finite")

    system = _make_scanner_system(
        pp,
        scanner_parameters,
        legacy_kwargs={
            "max_grad": 28,
            "grad_unit": "mT/m",
            "max_slew": 150,
            "slew_unit": "T/m/s",
            "rf_ringdown_time": 20e-6,
            "rf_dead_time": 100e-6,
            "adc_dead_time": 20e-6,
        },
    )
    sequence = pp.Sequence(system)
    requested_dwell = 1.0 / float(sampling_bandwidth_hz)
    dwell = round(requested_dwell / system.adc_raster_time) * system.adc_raster_time
    if dwell <= 0:
        raise ValueError("sampling bandwidth exceeds the ADC raster capability")

    rf_events, actual_rf_duration_s, effective_rf_tbw, rf_pulse_type = (
        make_pulseq_rf_events(
            pp,
            system,
            flip_angles_deg=(flip_angle_deg, resolved_alpha_half_flip_angle_deg),
            pulse_type=rf_pulse_type,
            duration_s=rf_duration_s,
            time_bandwidth_product=rf_time_bandwidth_product,
            apodization=rf_apodization,
            slr_sharpness=rf_slr_sharpness,
            custom_waveform_hz=rf_custom_waveform_hz,
            custom_raster_s=rf_custom_raster_s,
            custom_flip_angle_deg=rf_custom_flip_angle_deg,
            frequency_offset_hz=rf_frequency_offset_hz,
            spectral_spatial_pulse=rf_spectral_spatial_pulse,
            slice_thickness_m=(
                fov_z
                if normalize_rf_pulse_type(rf_pulse_type) == "spectral_spatial"
                else None
            ),
            nucleus=nucleus,
        )
    )
    if rf_pulse_type == "spectral_spatial":
        (rf, rf_select), (rf_alpha_half, rf_alpha_half_select) = rf_events
        rf_select = _remap_gradient_event(rf_select, encoding_frame, "partition")
        rf_alpha_half_select = _remap_gradient_event(
            rf_alpha_half_select, encoding_frame, "partition"
        )
    else:
        rf, rf_alpha_half = rf_events
        rf_select = None
        rf_alpha_half_select = None
    readout_duration = n_read * dwell
    readout_amplitude = 1.0 / (fov_x * dwell)
    readout_rise_time = max(
        system.adc_dead_time,
        np.ceil(abs(readout_amplitude) / system.max_slew / system.grad_raster_time)
        * system.grad_raster_time,
    )
    gx = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        flat_area=n_read / fov_x,
        flat_time=readout_duration,
        rise_time=readout_rise_time,
        system=system,
    )
    adc = pp.make_adc(
        num_samples=n_read,
        dwell=dwell,
        delay=gx.rise_time,
        system=system,
    )
    readout_area = logical_gradient_area(gx, encoding_frame, "read")
    gx_pre = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        area=-readout_area / 2,
        duration=encoding_duration_s,
        system=system,
    )

    rf_center, _ = pp.calc_rf_center(rf)
    rf_center_from_start = rf.delay + rf_center
    rf_block_duration = pp.calc_duration(
        *([rf, rf_select] if rf_select is not None else [rf])
    )
    read_block_duration = max(pp.calc_duration(gx), pp.calc_duration(adc))
    adc_center_from_start = adc.delay + adc.num_samples * adc.dwell / 2
    rf_balance_delay_value = (
        2 * rf_center_from_start
        + read_block_duration
        - 2 * adc_center_from_start
        - rf_block_duration
    )
    raster = system.block_duration_raster
    rf_balance_delay_value = round(rf_balance_delay_value / raster) * raster
    if rf_balance_delay_value < 0:
        raise ValueError("RF timing cannot be centered with a non-negative delay")
    minimum_tr = (
        rf_block_duration
        + rf_balance_delay_value
        + 2 * pp.calc_duration(gx_pre)
        + read_block_duration
    )
    if repetition_time_s is None:
        actual_tr = minimum_tr
    else:
        if not np.isfinite(repetition_time_s) or repetition_time_s <= 0:
            raise ValueError("repetition_time_s must be positive and finite")
        if repetition_time_s < minimum_tr - 1e-12:
            raise ValueError(
                f"repetition_time_s is too short; minimum is {minimum_tr:.9g} s"
            )
        extra_half = _ceil_to_raster((repetition_time_s - minimum_tr) / 2, raster)
        actual_tr = minimum_tr + 2 * extra_half
    extra_half = (actual_tr - minimum_tr) / 2

    if alpha_half_use_ratios:
        resolved_alpha_half_center_spacing_s = actual_tr * float(alpha_half_tr_ratio)
    else:
        resolved_alpha_half_center_spacing_s = (
            actual_tr / 2
            if alpha_half_center_spacing_s is None
            else float(alpha_half_center_spacing_s)
        )

    if use_alpha_half:
        rf_alpha_half_center, _ = pp.calc_rf_center(rf_alpha_half)
        rf_alpha_half.phase_offset = pulseq_phase_offset_rad(
            resolved_alpha_half_phase_deg,
            frequency_offset_hz=rf_frequency_offset_hz,
            event_center_s=rf_alpha_half_center,
        )
        rf_alpha_half_center_from_start = rf_alpha_half.delay + rf_alpha_half_center
        preparation_delay = (
            resolved_alpha_half_center_spacing_s
            - pp.calc_duration(
                *(
                    [rf_alpha_half, rf_alpha_half_select]
                    if rf_alpha_half_select is not None
                    else [rf_alpha_half]
                )
            )
            - rf_center_from_start
            + rf_alpha_half_center_from_start
        )
        preparation_delay = round(preparation_delay / raster) * raster
        if preparation_delay < 0:
            raise ValueError(
                "First TR is too short for the configured preparation pulse"
            )
        sequence.add_block(
            *(
                [rf_alpha_half, rf_alpha_half_select]
                if rf_alpha_half_select is not None
                else [rf_alpha_half]
            )
        )
        if preparation_delay:
            sequence.add_block(pp.make_delay(preparation_delay))

    ky_areas = (np.arange(n_phase) - n_phase // 2) / fov_y
    kz_areas = (np.arange(n_partition) - n_partition // 2) / fov_z
    # The SPSP slab is refocused inside the balanced partition gradients.
    slab_refocus_area = (
        _slice_rephase_area(
            rf,
            rf_select,
            encoding_frame,
            rf_pulse_type=rf_pulse_type,
            gradient_raster_s=system.grad_raster_time,
        )
        if rf_select is not None
        else 0.0
    )
    rf_phase = wrap_phase_deg(rf_phase_start_deg)

    def add_repetition(
        ky: float,
        kz: float,
        *,
        acquire: bool,
        line_index: int = 0,
        partition_index: int = 0,
        volume_index: int = 0,
    ) -> None:
        nonlocal rf_phase
        rf.phase_offset = pulseq_phase_offset_rad(
            rf_phase,
            frequency_offset_hz=rf_frequency_offset_hz,
            event_center_s=rf_center,
        )
        adc.phase_offset = pulseq_phase_offset_rad(
            rf_phase,
            frequency_offset_hz=0.0,
            event_center_s=adc_center_from_start,
        )
        rf_phase = advance_bssfp_phase_deg(
            rf_phase,
            elapsed_s=actual_tr,
            frequency_offset_hz=rf_frequency_offset_hz,
            phase_increment_deg=rf_phase_increment_deg,
        )
        gy_pre = make_role_trapezoid(
            pp,
            encoding_frame,
            "phase",
            area=ky,
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
        gz_pre = make_role_trapezoid(
            pp,
            encoding_frame,
            "partition",
            area=kz + slab_refocus_area,
            duration=encoding_duration_s,
            system=system,
        )
        gz_rephase = make_role_trapezoid(
            pp,
            encoding_frame,
            "partition",
            area=-kz - slab_refocus_area,
            duration=encoding_duration_s,
            system=system,
        )
        sequence.add_block(*([rf, rf_select] if rf_select is not None else [rf]))
        if rf_balance_delay_value:
            sequence.add_block(pp.make_delay(rf_balance_delay_value))
        if extra_half:
            sequence.add_block(pp.make_delay(extra_half))
        sequence.add_block(gx_pre, gy_pre, gz_pre)
        if acquire:
            sequence.add_block(
                gx,
                adc,
                pp.make_label("LIN", "SET", line_index),
                pp.make_label("PAR", "SET", partition_index),
                pp.make_label("REP", "SET", volume_index),
            )
        else:
            sequence.add_block(gx, pp.make_delay(read_block_duration))
        sequence.add_block(gx_pre, gy_rephase, gz_rephase)
        if extra_half:
            sequence.add_block(pp.make_delay(extra_half))

    for _ in range(dummy_repetitions):
        add_repetition(0.0, 0.0, acquire=False)
    acquisition_start_times = []
    acquisition_intervals = []
    minimum_acquisition_intervals = []
    for volume_index in range(repetitions):
        acquisition_start = _sequence_duration_s(sequence)
        acquisition_start_times.append(acquisition_start)
        for partition_index, kz in enumerate(kz_areas):
            for line_index, ky in enumerate(ky_areas):
                add_repetition(
                    float(ky),
                    float(kz),
                    acquire=True,
                    line_index=line_index,
                    partition_index=partition_index,
                    volume_index=volume_index,
                )
        minimum_interval = _sequence_duration_s(sequence) - acquisition_start
        actual_interval, _ = _finish_acquisition_interval(
            pp,
            sequence,
            acquisition_start_s=acquisition_start,
            requested_interval_s=acquisition_interval_s,
            raster_s=raster,
            acquisition_name="3D bSSFP volume",
        )
        minimum_acquisition_intervals.append(minimum_interval)
        acquisition_intervals.append(actual_interval)

    _raise_for_timing_errors(sequence, "bSSFP")
    sequence.set_definition("Name", "bssfp_3d")
    sequence.set_definition("FOV", [fov_x, fov_y, fov_z])
    sequence.set_definition("MatrixSize", [n_read, n_phase, n_partition])
    set_pulseq_encoding_definitions(
        sequence,
        encoding_frame,
        fov_m=(fov_x, fov_y, fov_z),
        matrix=(n_read, n_phase, n_partition),
    )
    sequence.set_definition("FlipAngleDeg", float(flip_angle_deg))
    sequence.set_definition("SamplingBandwidth", 1.0 / dwell)
    sequence.set_definition("TR", actual_tr)
    sequence.set_definition("TE", actual_tr / 2)
    sequence.set_definition("RFPhaseStartDeg", float(rf_phase_start_deg))
    sequence.set_definition("RFPhaseIncrementDeg", float(rf_phase_increment_deg))
    sequence.set_definition("AlphaHalfPhaseDeg", float(resolved_alpha_half_phase_deg))
    sequence.set_definition(
        "AlphaHalfFlipAngleDeg", float(resolved_alpha_half_flip_angle_deg)
    )
    sequence.set_definition(
        "AlphaHalfCenterSpacing", float(resolved_alpha_half_center_spacing_s)
    )
    sequence.set_definition("AlphaHalfUsesRatios", bool(alpha_half_use_ratios))
    sequence.set_definition(
        "AlphaHalfTRRatio", float(resolved_alpha_half_center_spacing_s / actual_tr)
    )
    sequence.set_definition(
        "AlphaHalfFlipRatio",
        float(resolved_alpha_half_flip_angle_deg / float(flip_angle_deg)),
    )
    sequence.set_definition("FrequencyOffsetPhaseCoherent", True)
    sequence.set_definition("DummyRepetitions", dummy_repetitions)
    sequence.set_definition("Repetitions", repetitions)
    sequence.set_definition("VolumeInterval", max(acquisition_intervals))
    _set_acquisition_interval_definitions(
        sequence,
        requested_interval_s=acquisition_interval_s,
        actual_intervals_s=acquisition_intervals,
        minimum_intervals_s=minimum_acquisition_intervals,
        start_times_s=acquisition_start_times,
    )
    sequence.set_definition("UseAlphaHalf", bool(use_alpha_half))
    _set_rf_definitions(
        sequence,
        pulse_type=rf_pulse_type,
        requested_duration_s=rf_duration_s,
        actual_duration_s=actual_rf_duration_s,
        time_bandwidth_product=effective_rf_tbw,
        apodization=rf_apodization,
        slr_sharpness=rf_slr_sharpness,
        custom_name=rf_custom_name,
        custom_flip_angle_deg=rf_custom_flip_angle_deg,
        frequency_offset_hz=rf_frequency_offset_hz,
        spectral_spatial_pulse=rf_spectral_spatial_pulse,
    )
    return sequence


def make_pulseq_epi(
    *,
    fov_m: Sequence[float] = (0.22, 0.22),
    matrix: Sequence[int] = (16, 16),
    sampling_bandwidth_hz: float = 50_000.0,
    readout_strategy: str = "bipolar",
    flip_angle_deg: float = 90.0,
    variable_flip_angle: bool = False,
    vfa_final_flip_angle_deg: float = 90.0,
    rf_pulse_type: str = "sinc",
    rf_duration_s: float = 3e-3,
    rf_time_bandwidth_product: float = 4.0,
    rf_apodization: float = 0.5,
    rf_slr_sharpness: int = 1,
    rf_custom_waveform_hz: Sequence[complex] | None = None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    rf_custom_name: str | None = None,
    rf_spectral_spatial_pulse=None,
    rf_frequency_offset_hz: float = 0.0,
    slice_thickness_m: float = 3e-3,
    slice_gap_m: float = 0.0,
    n_slices: int = 1,
    repetitions: int = 1,
    echo_time_s: float = 20e-3,
    repetition_time_s: float | None = 1.0,
    rf_spoiling: bool = False,
    rf_spoiling_increment_deg: float = 117.0,
    slice_offset_m: float = 0.0,
    encoding_axes: Sequence[str] | EncodingFrame = ("+x", "+y", "+z"),
    spoil_after_slice: bool = True,
    spoiler_cycles_per_slice: float = 8.0,
    spoiler_cycles_per_voxel: float = 0.0,
    spoiler_duration_s: float = 4e-3,
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
    nucleus: str = "H1",
    timing_only: bool = False,
):
    """Build a Cartesian single-shot EPI Pulseq sequence for export.

    Variable flip angles advance once per repetition and are shared by every
    slice acquired within that repetition. ``readout_strategy="bipolar"``
    alternates the read gradient direction, while ``"flyback"`` acquires every
    line in the positive direction and rewinds kx between lines.
    """
    pp = _pypulseq()
    encoding_frame = resolve_encoding_frame(encoding_axes)
    fov_x, fov_y = _positive_values(fov_m, "FOV")
    if len(tuple(fov_m)) != 2:
        raise ValueError("fov_m must contain two values")
    n_x, n_y = (_positive_integer(value, "matrix size") for value in matrix)
    if len(tuple(matrix)) != 2:
        raise ValueError("matrix must contain two values")
    n_slices = _positive_integer(n_slices, "n_slices")
    repetitions = _positive_integer(repetitions, "repetitions")
    strategy = str(readout_strategy).strip().lower().replace("-", "_")
    strategy_aliases = {
        "bipolar": "bipolar",
        "alternating": "bipolar",
        "standard": "bipolar",
        "flyback": "flyback",
        "monopolar": "flyback",
        "unipolar": "flyback",
    }
    try:
        strategy = strategy_aliases[strategy]
    except KeyError as exc:
        raise ValueError("readout_strategy must be 'bipolar' or 'flyback'") from exc
    if variable_flip_angle:
        flip_angle_schedule_deg = variable_flip_angle_schedule(
            repetitions,
            final_flip_angle_deg=vfa_final_flip_angle_deg,
        )
    else:
        flip_angle_schedule_deg = np.asarray([flip_angle_deg], dtype=float)
    for name, value in {
        "sampling_bandwidth_hz": sampling_bandwidth_hz,
        "flip_angle_deg": flip_angle_deg,
        "slice_thickness_m": slice_thickness_m,
        "echo_time_s": echo_time_s,
        "spoiler_duration_s": spoiler_duration_s,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not np.isfinite(slice_gap_m) or slice_gap_m < 0:
        raise ValueError("slice_gap_m must be non-negative and finite")
    if not np.isfinite(rf_frequency_offset_hz):
        raise ValueError("rf_frequency_offset_hz must be finite")
    if not np.isfinite(slice_offset_m):
        raise ValueError("slice_offset_m must be finite")
    if not np.isfinite(rf_spoiling_increment_deg):
        raise ValueError("rf_spoiling_increment_deg must be finite")
    rf_spoiling = _uses_rf_spoiling(rf_spoiling, rf_spoiling_increment_deg)
    for name, value in {
        "spoiler_cycles_per_slice": spoiler_cycles_per_slice,
        "spoiler_cycles_per_voxel": spoiler_cycles_per_voxel,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")

    system = _make_scanner_system(
        pp,
        scanner_parameters,
        legacy_kwargs={
            "max_grad": 32,
            "grad_unit": "mT/m",
            "max_slew": 130,
            "slew_unit": "T/m/s",
            "rf_ringdown_time": 30e-6,
            "rf_dead_time": 100e-6,
            "adc_dead_time": 20e-6,
        },
    )
    sequence = pp.Sequence(system)
    dwell = _centered_adc_dwell(
        1.0 / sampling_bandwidth_hz,
        n_x,
        system,
    )
    rf_events, actual_rf_duration_s, effective_rf_tbw, rf_pulse_type = (
        _make_slice_selective_rf_events(
            pp,
            system,
            flip_angle_schedule_deg=flip_angle_schedule_deg,
            slice_thickness_m=slice_thickness_m,
            rf_pulse_type=rf_pulse_type,
            rf_duration_s=rf_duration_s,
            rf_time_bandwidth_product=rf_time_bandwidth_product,
            rf_apodization=rf_apodization,
            rf_slr_sharpness=rf_slr_sharpness,
            rf_custom_waveform_hz=rf_custom_waveform_hz,
            rf_custom_raster_s=rf_custom_raster_s,
            rf_custom_flip_angle_deg=rf_custom_flip_angle_deg,
            rf_frequency_offset_hz=rf_frequency_offset_hz,
            rf_spectral_spatial_pulse=rf_spectral_spatial_pulse,
            spatial_offset_m=0.0,
            nucleus=nucleus,
        )
    )
    spsp_logical_gz = rf_events[0][1]
    rf_events = [
        (rf, _remap_gradient_event(gz, encoding_frame, "partition"))
        for rf, gz in rf_events
    ]
    rf, gz = rf_events[0]
    delta_kx, delta_ky = 1.0 / fov_x, 1.0 / fov_y
    adc_duration = n_x * dwell
    flat_time = _ceil_to_raster(adc_duration, system.grad_raster_time)
    gx = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        amplitude=(n_x * delta_kx) / adc_duration,
        flat_time=flat_time,
        system=system,
    )
    gx_reverse = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        amplitude=-(n_x * delta_kx) / adc_duration,
        flat_time=flat_time,
        system=system,
    )
    adc = pp.make_adc(
        num_samples=n_x,
        dwell=dwell,
        delay=gx.rise_time + flat_time / 2 - adc_duration / 2,
        system=system,
    )
    pre_time = 0.8e-3
    gx_pre = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        area=-logical_gradient_area(gx, encoding_frame, "read") / 2,
        duration=pre_time,
        system=system,
    )
    gy_pre = make_role_trapezoid(
        pp,
        encoding_frame,
        "phase",
        area=-n_y / 2 * delta_ky,
        duration=pre_time,
        system=system,
    )
    gz_rephase = make_role_trapezoid(
        pp,
        encoding_frame,
        "partition",
        area=_slice_rephase_area(
            rf,
            gz,
            encoding_frame,
            rf_pulse_type=rf_pulse_type,
            gradient_raster_s=system.grad_raster_time,
        ),
        duration=pre_time,
        system=system,
    )
    logical_read_area = logical_gradient_area(gx, encoding_frame, "read")
    relative_x_end = (
        logical_read_area / 2
        if strategy == "flyback"
        else -logical_read_area / 2 + (logical_read_area if n_y % 2 else 0.0)
    )
    relative_y_end = (-n_y / 2 + max(n_y - 1, 0)) * delta_ky
    gx_post = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        area=-relative_x_end,
        duration=pre_time,
        system=system,
    )
    gy_post = make_role_trapezoid(
        pp,
        encoding_frame,
        "phase",
        area=-relative_y_end,
        duration=pre_time,
        system=system,
    )
    gy_blip = make_role_trapezoid(
        pp, encoding_frame, "phase", area=delta_ky, system=system
    )
    gx_flyback = None
    if strategy == "flyback" and n_y > 1:
        # Use one shared duration so the read rewinder and phase blip can be
        # played simultaneously without changing the intended gradient areas.
        minimum_connector_duration = max(
            pp.calc_duration(gy_blip),
            pp.calc_duration(
                make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "read",
                    area=-logical_read_area,
                    system=system,
                )
            ),
        )
        gx_flyback = make_role_trapezoid(
            pp,
            encoding_frame,
            "read",
            area=-logical_read_area,
            duration=minimum_connector_duration,
            system=system,
        )
        gy_blip = make_role_trapezoid(
            pp,
            encoding_frame,
            "phase",
            area=delta_ky,
            duration=minimum_connector_duration,
            system=system,
        )
    spoilers = []
    if spoil_after_slice:
        if spoiler_cycles_per_voxel > 0:
            for role, voxel_size in zip(("read", "phase"), (fov_x / n_x, fov_y / n_y)):
                spoilers.append(
                    make_role_trapezoid(
                        pp,
                        encoding_frame,
                        role,
                        area=spoiler_cycles_per_voxel / voxel_size,
                        duration=spoiler_duration_s,
                        system=system,
                    )
                )
        if spoiler_cycles_per_slice > 0:
            spoilers.append(
                make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "partition",
                    area=spoiler_cycles_per_slice / slice_thickness_m,
                    duration=spoiler_duration_s,
                    system=system,
                )
            )

    slice_positions = float(slice_offset_m) + (
        np.arange(n_slices, dtype=float) - (n_slices - 1) / 2
    ) * (slice_thickness_m + slice_gap_m)
    rf_center, _ = pp.calc_rf_center(rf)
    rf_block_duration = pp.calc_duration(rf, gz)
    readout_block_duration = pp.calc_duration(gx, adc)
    blip_duration = (
        pp.calc_duration(gx_flyback, gy_blip)
        if gx_flyback is not None
        else pp.calc_duration(gy_blip)
    )
    center_line = n_y // 2
    echo_without_delay = (
        rf_block_duration
        - (rf.delay + rf_center)
        + pre_time
        + center_line * (readout_block_duration + blip_duration)
        + adc.delay
        + adc_duration / 2.0
    )
    requested_te_delay = float(echo_time_s) - echo_without_delay
    if requested_te_delay < -1e-12 and not timing_only:
        raise ValueError(
            f"echo_time_s is too short; minimum is {echo_without_delay:.9g} s"
        )
    te_delay_value = _ceil_to_raster(
        max(0.0, requested_te_delay), system.block_duration_raster
    )
    actual_te = echo_without_delay + te_delay_value
    if timing_only:
        slice_duration = (
            rf_block_duration
            + pp.calc_duration(gx_pre, gy_pre, gz_rephase)
            + te_delay_value
            + n_y * readout_block_duration
            + (n_y - 1) * blip_duration
            + pp.calc_duration(gx_post, gy_post)
            + (pp.calc_duration(*spoilers) if spoilers else 0.0)
        )
        return EpiTimingPlan(
            sampling_bandwidth_hz=1.0 / dwell,
            dwell_s=float(dwell),
            minimum_echo_time_s=float(echo_without_delay),
            minimum_repetition_time_s=float(n_slices * slice_duration),
        )
    _, slice_sign = encoding_frame.axis_and_sign("partition")
    package_duration = None
    actual_repetition_time = None
    spoiler_end_times = []
    acquisition_start_times = []
    acquisition_intervals = []
    minimum_acquisition_intervals = []
    excitation_index = 0
    for repetition in range(repetitions):
        rf, gz = rf_events[repetition if variable_flip_angle else 0]
        repetition_start = (
            0.0 if not sequence.block_events else float(sequence.duration()[0])
        )
        acquisition_start_times.append(repetition_start)
        for slice_index, position in enumerate(slice_positions):
            rf_phase_deg = (
                _rf_spoiling_phase_deg(excitation_index, rf_spoiling_increment_deg)
                if rf_spoiling
                else 0.0
            )
            if rf_pulse_type == "spectral_spatial":
                rf_for_slice = shifted_spectral_spatial_rf_event(
                    rf,
                    spsp_logical_gz,
                    spatial_offset_m=position,
                    rf_raster_s=system.rf_raster_time,
                    gradient_raster_s=system.grad_raster_time,
                )
                rf_for_slice.freq_offset = rf_frequency_offset_hz
                rf_for_slice.phase_offset = (
                    np.deg2rad(rf_phase_deg)
                    - 2 * np.pi * rf_frequency_offset_hz * rf_center
                )
            else:
                logical_slice_amplitude = float(gz.amplitude) * slice_sign
                slice_frequency_offset_hz = logical_slice_amplitude * position
                rf_for_slice = rf
                rf_for_slice.freq_offset = (
                    rf_frequency_offset_hz + slice_frequency_offset_hz
                )
                rf_for_slice.phase_offset = (
                    np.deg2rad(rf_phase_deg)
                    - 2 * np.pi * slice_frequency_offset_hz * rf_center
                )
            adc.phase_offset = np.deg2rad(rf_phase_deg)
            sequence.add_block(
                rf_for_slice,
                gz,
                pp.make_label("SLC", "SET", slice_index),
                pp.make_label("REP", "SET", repetition),
            )
            sequence.add_block(gx_pre, gy_pre, gz_rephase)
            if te_delay_value:
                sequence.add_block(pp.make_delay(te_delay_value))
            for line in range(n_y):
                readout = gx if strategy == "flyback" or line % 2 == 0 else gx_reverse
                sequence.add_block(readout, adc)
                if line < n_y - 1:
                    if gx_flyback is None:
                        sequence.add_block(gy_blip)
                    else:
                        sequence.add_block(gx_flyback, gy_blip)
            sequence.add_block(gx_post, gy_post)
            if spoilers:
                sequence.add_block(*spoilers)
                spoiler_end_times.append(float(sequence.duration()[0]))
            excitation_index += 1
        acquired = float(sequence.duration()[0]) - repetition_start
        if package_duration is None:
            package_duration = acquired
            actual_repetition_time = (
                acquired if repetition_time_s is None else float(repetition_time_s)
            )
            if actual_repetition_time < acquired - 1e-12:
                raise ValueError(
                    f"repetition_time_s is too short; minimum is {acquired:.9g} s"
                )
        delay = actual_repetition_time - acquired
        if delay > 1e-12:
            sequence.add_block(pp.make_delay(delay))
        minimum_acquisition_intervals.append(acquired)
        acquisition_intervals.append(float(sequence.duration()[0]) - repetition_start)

    _raise_for_timing_errors(sequence, "EPI")
    sequence.set_definition("Name", "epi_2d")
    sequence.set_definition("TrajectoryType", "cartesian_epi")
    sequence.set_definition("ReadoutStrategy", strategy)
    slice_extent = n_slices * slice_thickness_m + (n_slices - 1) * slice_gap_m
    sequence.set_definition("FOV", [fov_x, fov_y, slice_extent])
    sequence.set_definition("MatrixSize", [n_x, n_y])
    set_pulseq_encoding_definitions(
        sequence,
        encoding_frame,
        fov_m=(fov_x, fov_y, slice_extent),
        matrix=(n_x, n_y, n_slices),
    )
    sequence.set_definition("SamplingBandwidth", 1.0 / dwell)
    sequence.set_definition("RequestedSamplingBandwidth", sampling_bandwidth_hz)
    sequence.set_definition("TE", actual_te)
    sequence.set_definition(
        "FlipAngleDeg",
        float(vfa_final_flip_angle_deg if variable_flip_angle else flip_angle_deg),
    )
    _set_rf_definitions(
        sequence,
        pulse_type=rf_pulse_type,
        requested_duration_s=rf_duration_s,
        actual_duration_s=actual_rf_duration_s,
        time_bandwidth_product=effective_rf_tbw,
        apodization=rf_apodization,
        slr_sharpness=rf_slr_sharpness,
        custom_name=rf_custom_name,
        custom_flip_angle_deg=rf_custom_flip_angle_deg,
        frequency_offset_hz=rf_frequency_offset_hz,
        spectral_spatial_pulse=rf_spectral_spatial_pulse,
    )
    sequence.set_definition("VariableFlipAngle", bool(variable_flip_angle))
    if variable_flip_angle:
        sequence.set_definition("VariableFlipAngleDimension", "repetition")
        sequence.set_definition(
            "VariableFlipAngleFinalDeg", float(vfa_final_flip_angle_deg)
        )
        sequence.set_definition(
            "FlipAngleScheduleDeg",
            [float(value) for value in flip_angle_schedule_deg],
        )
        sequence.set_definition("VariableFlipAngleReferenceDOI", VFA_REFERENCE_DOI)
    sequence.set_definition("SliceThickness", slice_thickness_m)
    sequence.set_definition("SliceGap", slice_gap_m)
    sequence.set_definition("SliceSpacing", slice_thickness_m + slice_gap_m)
    sequence.set_definition("SliceOffset", float(slice_offset_m))
    sequence.set_definition(
        "SlicePositions", [float(value) for value in slice_positions]
    )
    sequence.set_definition("Repetitions", repetitions)
    sequence.set_definition("RepetitionTime", actual_repetition_time)
    sequence.set_definition("MinimumRepetitionTime", package_duration)
    sequence.set_definition("RFSpoiling", bool(rf_spoiling))
    sequence.set_definition("RFSpoilingIncrementDeg", rf_spoiling_increment_deg)
    sequence.set_definition("VolumeInterval", max(acquisition_intervals))
    _set_acquisition_interval_definitions(
        sequence,
        requested_interval_s=repetition_time_s,
        actual_intervals_s=acquisition_intervals,
        minimum_intervals_s=minimum_acquisition_intervals,
        start_times_s=acquisition_start_times,
    )
    sequence.set_definition("SpoilAfterSlice", bool(spoil_after_slice))
    sequence.set_definition("SpoilerCyclesPerSlice", spoiler_cycles_per_slice)
    sequence.set_definition("SpoilerCyclesPerVoxel", spoiler_cycles_per_voxel)
    sequence.set_definition("SpoilerDuration", spoiler_duration_s)
    sequence.set_definition(
        "SpoilerAxes", "".join(event.channel for event in spoilers) or "none"
    )
    sequence.set_definition("SpoilerEndTimes", spoiler_end_times)
    sequence.set_definition("IdealSpoilerEndTimes", spoiler_end_times)
    return sequence


@dataclass(frozen=True)
class EpiTimingPlan:
    """Shortest feasible single-shot EPI timing for the requested parameters."""

    sampling_bandwidth_hz: float
    dwell_s: float
    minimum_echo_time_s: float
    minimum_repetition_time_s: float


@dataclass(frozen=True)
class FlashTimingPlan:
    """Shortest feasible FLASH timing for the requested parameters."""

    sampling_bandwidth_hz: float
    dwell_s: float
    minimum_echo_time_s: float
    minimum_repetition_time_s: float
    minimum_acquisition_interval_s: float


@dataclass(frozen=True)
class EpsiMgeTimingPlan:
    """Hardware-compatible timing limits of an EPSI / MGE parameter set."""

    sampling_bandwidth_hz: float
    dwell_s: float
    readout_block_duration_s: float
    minimum_echo_spacing_s: float
    minimum_echo_time_s: float
    minimum_repetition_time_s: float
    minimum_acquisition_interval_s: float


def make_pulseq_epsi_mge(
    *,
    fov_m: Sequence[float] = (0.22, 0.22),
    matrix: Sequence[int] = (16, 16),
    echoes: int = 8,
    echo_spacing_s: float = 2e-3,
    readout_strategy: str = "bipolar",
    sampling_bandwidth_hz: float = 50_000.0,
    flip_angle_deg: float = 15.0,
    rf_pulse_type: str = "sinc",
    rf_duration_s: float = 3e-3,
    rf_time_bandwidth_product: float = 4.0,
    rf_apodization: float = 0.5,
    rf_slr_sharpness: float = 1.0,
    rf_custom_waveform_hz: Sequence[complex] | None = None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    rf_custom_name: str | None = None,
    rf_spectral_spatial_pulse=None,
    rf_frequency_offset_hz: float = 0.0,
    slice_thickness_m: float = 3e-3,
    slice_gap_m: float = 0.0,
    n_slices: int = 1,
    slice_offset_m: float = 0.0,
    echo_time_s: float = 6e-3,
    repetition_time_s: float = 50e-3,
    repetitions: int = 1,
    acquisition_interval_s: float | None = None,
    rf_spoiling: bool = True,
    rf_spoiling_increment_deg: float = 117.0,
    spoil_after_readout: bool = True,
    spoiler_cycles_per_slice: float = 4.0,
    spoiler_cycles_per_voxel: float = 0.0,
    spoiler_duration_s: float = 2e-3,
    encoding_duration_s: float = 0.8e-3,
    encoding_axes: Sequence[str] | EncodingFrame = ("+x", "+y", "+z"),
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
    nucleus: str = "H1",
    timing_only: bool = False,
):
    """Build a phase-encoded 2D EPSI / multi-gradient-echo sequence.

    Each RF excitation acquires one phase-encoding line at every echo time.
    The resulting labelled ADC stream can be reconstructed as an MGE echo
    series, or Fourier transformed along the uniformly spaced echo dimension
    for EPSI processing. Bipolar readout alternates the spatial traversal;
    flyback readout acquires every echo in the same direction.

    With ``timing_only=True`` no sequence is assembled.  The function instead
    returns an :class:`EpsiMgeTimingPlan` holding the hardware-compatible
    sampling bandwidth and the shortest echo spacing, first echo time,
    repetition time and acquisition interval for the given parameters.  Requested
    timings below these minima are reported rather than rejected.
    """
    pp = _pypulseq()
    encoding_frame = resolve_encoding_frame(encoding_axes)
    fov_x, fov_y = _positive_values(fov_m, "FOV")
    if len(tuple(fov_m)) != 2:
        raise ValueError("fov_m must contain two values")
    n_x, n_y = (_positive_integer(value, "matrix size") for value in matrix)
    if len(tuple(matrix)) != 2:
        raise ValueError("matrix must contain two values")
    echoes = _positive_integer(echoes, "echoes")
    n_slices = _positive_integer(n_slices, "n_slices")
    repetitions = _positive_integer(repetitions, "repetitions")
    strategy = str(readout_strategy).strip().lower().replace("-", "_")
    strategy_aliases = {
        "bipolar": "bipolar",
        "alternating": "bipolar",
        "symmetric": "bipolar",
        "flyback": "flyback",
        "monopolar": "flyback",
        "unipolar": "flyback",
    }
    try:
        strategy = strategy_aliases[strategy]
    except KeyError as exc:
        raise ValueError("readout_strategy must be 'bipolar' or 'flyback'") from exc
    for name, value in {
        "echo_spacing_s": echo_spacing_s,
        "sampling_bandwidth_hz": sampling_bandwidth_hz,
        "flip_angle_deg": flip_angle_deg,
        "slice_thickness_m": slice_thickness_m,
        "echo_time_s": echo_time_s,
        "repetition_time_s": repetition_time_s,
        "spoiler_duration_s": spoiler_duration_s,
        "encoding_duration_s": encoding_duration_s,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value in {
        "slice_gap_m": slice_gap_m,
        "spoiler_cycles_per_slice": spoiler_cycles_per_slice,
        "spoiler_cycles_per_voxel": spoiler_cycles_per_voxel,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    for name, value in {
        "rf_frequency_offset_hz": rf_frequency_offset_hz,
        "slice_offset_m": slice_offset_m,
        "rf_spoiling_increment_deg": rf_spoiling_increment_deg,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if acquisition_interval_s is not None and (
        not np.isfinite(acquisition_interval_s) or acquisition_interval_s <= 0
    ):
        raise ValueError("acquisition_interval_s must be positive and finite or None")
    rf_spoiling = _uses_rf_spoiling(rf_spoiling, rf_spoiling_increment_deg)

    system = _make_scanner_system(
        pp,
        scanner_parameters,
        legacy_kwargs={
            "max_grad": 32,
            "grad_unit": "mT/m",
            "max_slew": 130,
            "slew_unit": "T/m/s",
            "rf_ringdown_time": 30e-6,
            "rf_dead_time": 100e-6,
            "adc_dead_time": 20e-6,
        },
    )
    sequence = pp.Sequence(system)
    dwell = _centered_adc_dwell(1.0 / sampling_bandwidth_hz, n_x, system)

    rf_events, actual_rf_duration_s, effective_rf_tbw, rf_pulse_type = (
        _make_slice_selective_rf_events(
            pp,
            system,
            flip_angle_schedule_deg=(flip_angle_deg,),
            slice_thickness_m=slice_thickness_m,
            rf_pulse_type=rf_pulse_type,
            rf_duration_s=rf_duration_s,
            rf_time_bandwidth_product=rf_time_bandwidth_product,
            rf_apodization=rf_apodization,
            rf_slr_sharpness=rf_slr_sharpness,
            rf_custom_waveform_hz=rf_custom_waveform_hz,
            rf_custom_raster_s=rf_custom_raster_s,
            rf_custom_flip_angle_deg=rf_custom_flip_angle_deg,
            rf_frequency_offset_hz=rf_frequency_offset_hz,
            rf_spectral_spatial_pulse=rf_spectral_spatial_pulse,
            spatial_offset_m=0.0,
            nucleus=nucleus,
        )
    )
    rf, gz = rf_events[0]
    spsp_logical_gz = gz
    gz = _remap_gradient_event(gz, encoding_frame, "partition")
    rf_center, _ = pp.calc_rf_center(rf)
    rf_block_duration = pp.calc_duration(rf, gz)

    delta_kx, delta_ky = 1.0 / fov_x, 1.0 / fov_y
    adc_duration = n_x * dwell
    flat_time = _ceil_to_raster(adc_duration, system.grad_raster_time)
    gx_positive = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        amplitude=(n_x * delta_kx) / adc_duration,
        flat_time=flat_time,
        system=system,
    )
    gx_negative = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        amplitude=-(n_x * delta_kx) / adc_duration,
        flat_time=flat_time,
        system=system,
    )
    adc = pp.make_adc(
        num_samples=n_x,
        dwell=dwell,
        delay=gx_positive.rise_time + flat_time / 2 - adc_duration / 2,
        system=system,
    )
    readout_area = logical_gradient_area(gx_positive, encoding_frame, "read")
    readout_block_duration = pp.calc_duration(gx_positive, adc)
    block_raster = system.block_duration_raster
    actual_echo_spacing = _ceil_to_raster(echo_spacing_s, block_raster)
    minimum_echo_spacing = _ceil_to_raster(readout_block_duration, block_raster)
    if strategy == "flyback" and echoes > 1:
        # The flyback gradient needs extra time beyond the bare readout, so find
        # the shortest spacing at which it fits the scanner's gradient limits.
        for _ in range(5000):
            try:
                make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "read",
                    area=-readout_area,
                    # Test slightly short so float noise at an exact boundary
                    # cannot make the reported minimum unbuildable.
                    duration=minimum_echo_spacing - readout_block_duration - 1e-9,
                    system=system,
                )
                break
            except Exception:
                minimum_echo_spacing += block_raster
        else:
            raise ValueError(
                "no echo spacing leaves enough time for the flyback gradient; "
                "reduce the readout area or increase scanner gradient performance"
            )
    if timing_only:
        actual_echo_spacing = max(actual_echo_spacing, minimum_echo_spacing)
    connector_duration = actual_echo_spacing - readout_block_duration
    if actual_echo_spacing < minimum_echo_spacing - block_raster / 2:
        if strategy == "flyback" and echoes > 1:
            raise ValueError(
                "echo_spacing_s leaves too little time for the flyback gradient; "
                f"use at least {minimum_echo_spacing:.9g} s"
            )
        raise ValueError(
            "echo_spacing_s is shorter than the readout gradient duration; "
            f"use at least {readout_block_duration:.9g} s"
        )
    connector_duration = max(0.0, connector_duration)
    between_echo_event = None
    if strategy == "flyback" and echoes > 1:
        between_echo_event = make_role_trapezoid(
            pp,
            encoding_frame,
            "read",
            area=-readout_area,
            duration=connector_duration,
            system=system,
        )
    elif connector_duration:
        between_echo_event = pp.make_delay(connector_duration)

    gx_pre = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        area=-readout_area / 2.0,
        duration=encoding_duration_s,
        system=system,
    )
    gz_rephase = make_role_trapezoid(
        pp,
        encoding_frame,
        "partition",
        area=_slice_rephase_area(
            rf,
            gz,
            encoding_frame,
            rf_pulse_type=rf_pulse_type,
            gradient_raster_s=system.grad_raster_time,
        ),
        duration=encoding_duration_s,
        system=system,
    )
    final_read_position = (
        readout_area / 2.0
        if strategy == "flyback" or echoes % 2
        else -readout_area / 2.0
    )
    gx_post = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        area=-final_read_position,
        duration=encoding_duration_s,
        system=system,
    )

    spoilers = []
    if spoil_after_readout:
        if spoiler_cycles_per_voxel > 0:
            for role, voxel_size in zip(("read", "phase"), (fov_x / n_x, fov_y / n_y)):
                spoilers.append(
                    make_role_trapezoid(
                        pp,
                        encoding_frame,
                        role,
                        area=spoiler_cycles_per_voxel / voxel_size,
                        duration=spoiler_duration_s,
                        system=system,
                    )
                )
        if spoiler_cycles_per_slice > 0:
            spoilers.append(
                make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "partition",
                    area=spoiler_cycles_per_slice / slice_thickness_m,
                    duration=spoiler_duration_s,
                    system=system,
                )
            )

    adc_center_from_readout_start = adc.delay + adc_duration / 2.0
    echo_without_delay = (
        rf_block_duration
        - (rf.delay + rf_center)
        + encoding_duration_s
        + adc_center_from_readout_start
    )
    requested_te_delay = float(echo_time_s) - echo_without_delay
    if requested_te_delay < -1e-12 and not timing_only:
        raise ValueError(
            f"echo_time_s is too short; minimum is {echo_without_delay:.9g} s"
        )
    te_delay = _ceil_to_raster(
        max(0.0, requested_te_delay), system.block_duration_raster
    )
    actual_first_te = echo_without_delay + te_delay
    train_duration = echoes * readout_block_duration + max(echoes - 1, 0) * (
        actual_echo_spacing - readout_block_duration
    )
    minimum_tr = (
        rf_block_duration
        + encoding_duration_s
        + te_delay
        + train_duration
        + encoding_duration_s
        + (spoiler_duration_s if spoilers else 0.0)
    )
    actual_tr = _ceil_to_raster(repetition_time_s, system.block_duration_raster)
    if timing_only:
        actual_tr = max(actual_tr, _ceil_to_raster(minimum_tr, block_raster))
        return EpsiMgeTimingPlan(
            sampling_bandwidth_hz=1.0 / dwell,
            dwell_s=dwell,
            readout_block_duration_s=readout_block_duration,
            minimum_echo_spacing_s=minimum_echo_spacing,
            minimum_echo_time_s=float(echo_without_delay),
            minimum_repetition_time_s=minimum_tr,
            minimum_acquisition_interval_s=n_slices * n_y * actual_tr,
        )
    if actual_tr < minimum_tr - 1e-12:
        raise ValueError(
            f"repetition_time_s is too short; minimum is {minimum_tr:.9g} s"
        )
    trailing_delay = max(0.0, actual_tr - minimum_tr)

    slice_spacing = slice_thickness_m + slice_gap_m
    slice_positions = (
        float(slice_offset_m)
        + (np.arange(n_slices, dtype=float) - (n_slices - 1) / 2.0) * slice_spacing
    )
    ky_areas = (np.arange(n_y, dtype=float) - n_y // 2) * delta_ky
    _, slice_sign = encoding_frame.axis_and_sign("partition")
    excitation_index = 0
    spoiler_end_times = []
    acquisition_start_times = []
    acquisition_intervals = []
    minimum_acquisition_intervals = []

    for repetition in range(repetitions):
        acquisition_start = _sequence_duration_s(sequence)
        acquisition_start_times.append(acquisition_start)
        for slice_index, position in enumerate(slice_positions):
            for line_index, ky_area in enumerate(ky_areas):
                rf_phase_deg = (
                    _rf_spoiling_phase_deg(excitation_index, rf_spoiling_increment_deg)
                    if rf_spoiling
                    else 0.0
                )
                if rf_pulse_type == "spectral_spatial":
                    rf_for_slice = shifted_spectral_spatial_rf_event(
                        rf,
                        spsp_logical_gz,
                        spatial_offset_m=position,
                        rf_raster_s=system.rf_raster_time,
                        gradient_raster_s=system.grad_raster_time,
                    )
                    rf_for_slice.freq_offset = rf_frequency_offset_hz
                    rf_for_slice.phase_offset = (
                        np.deg2rad(rf_phase_deg)
                        - 2.0 * np.pi * rf_frequency_offset_hz * rf_center
                    )
                else:
                    logical_slice_amplitude = float(gz.amplitude) * slice_sign
                    slice_frequency_offset_hz = logical_slice_amplitude * position
                    rf_for_slice = rf
                    rf_for_slice.freq_offset = (
                        rf_frequency_offset_hz + slice_frequency_offset_hz
                    )
                    rf_for_slice.phase_offset = (
                        np.deg2rad(rf_phase_deg)
                        - 2.0 * np.pi * slice_frequency_offset_hz * rf_center
                    )
                adc.phase_offset = np.deg2rad(rf_phase_deg)
                sequence.add_block(rf_for_slice, gz)
                gy_pre = make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "phase",
                    area=float(ky_area),
                    duration=encoding_duration_s,
                    system=system,
                )
                sequence.add_block(gx_pre, gy_pre, gz_rephase)
                if te_delay:
                    sequence.add_block(pp.make_delay(te_delay))
                for echo in range(echoes):
                    readout = (
                        gx_positive
                        if strategy == "flyback" or echo % 2 == 0
                        else gx_negative
                    )
                    sequence.add_block(
                        readout,
                        adc,
                        pp.make_label("LIN", "SET", line_index),
                        pp.make_label("SLC", "SET", slice_index),
                        pp.make_label("ECO", "SET", echo),
                        pp.make_label("REP", "SET", repetition),
                    )
                    if echo < echoes - 1 and between_echo_event is not None:
                        sequence.add_block(between_echo_event)
                gy_post = make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "phase",
                    area=-float(ky_area),
                    duration=encoding_duration_s,
                    system=system,
                )
                sequence.add_block(gx_post, gy_post)
                if spoilers:
                    sequence.add_block(*spoilers)
                    spoiler_end_times.append(_sequence_duration_s(sequence))
                if trailing_delay:
                    sequence.add_block(pp.make_delay(trailing_delay))
                excitation_index += 1
        minimum_interval = _sequence_duration_s(sequence) - acquisition_start
        actual_interval, _ = _finish_acquisition_interval(
            pp,
            sequence,
            acquisition_start_s=acquisition_start,
            requested_interval_s=acquisition_interval_s,
            raster_s=system.block_duration_raster,
            acquisition_name="EPSI / MGE measurement",
        )
        minimum_acquisition_intervals.append(minimum_interval)
        acquisition_intervals.append(actual_interval)

    _raise_for_timing_errors(sequence, "EPSI / MGE")
    echo_times = [
        actual_first_te + echo * actual_echo_spacing for echo in range(echoes)
    ]
    slice_extent = n_slices * slice_thickness_m + (n_slices - 1) * slice_gap_m
    sequence.set_definition("Name", "epsi_mge_2d")
    sequence.set_definition("TrajectoryType", "cartesian_2d_multi_echo")
    sequence.set_definition("FOV", [fov_x, fov_y, slice_extent])
    sequence.set_definition("MatrixSize", [n_x, n_y])
    set_pulseq_encoding_definitions(
        sequence,
        encoding_frame,
        fov_m=(fov_x, fov_y, slice_extent),
        matrix=(n_x, n_y, n_slices),
    )
    sequence.set_definition("SamplingBandwidth", 1.0 / dwell)
    sequence.set_definition("RequestedSamplingBandwidth", sampling_bandwidth_hz)
    sequence.set_definition("Echoes", echoes)
    sequence.set_definition("EchoTimes", echo_times)
    sequence.set_definition("TE", actual_first_te)
    sequence.set_definition("EchoSpacing", actual_echo_spacing)
    sequence.set_definition("RequestedEchoSpacing", float(echo_spacing_s))
    sequence.set_definition("EchoSpectralBandwidth", 1.0 / actual_echo_spacing)
    sequence.set_definition(
        "EchoSpectralResolution", 1.0 / (echoes * actual_echo_spacing)
    )
    sequence.set_definition("ReadoutStrategy", strategy)
    sequence.set_definition("FlipAngleDeg", float(flip_angle_deg))
    _set_rf_definitions(
        sequence,
        pulse_type=rf_pulse_type,
        requested_duration_s=rf_duration_s,
        actual_duration_s=actual_rf_duration_s,
        time_bandwidth_product=effective_rf_tbw,
        apodization=rf_apodization,
        slr_sharpness=rf_slr_sharpness,
        custom_name=rf_custom_name,
        custom_flip_angle_deg=rf_custom_flip_angle_deg,
        frequency_offset_hz=rf_frequency_offset_hz,
        spectral_spatial_pulse=rf_spectral_spatial_pulse,
    )
    sequence.set_definition("SliceThickness", slice_thickness_m)
    sequence.set_definition("SliceGap", slice_gap_m)
    sequence.set_definition("SliceSpacing", slice_spacing)
    sequence.set_definition("SliceOffset", float(slice_offset_m))
    sequence.set_definition(
        "SlicePositions", [float(value) for value in slice_positions]
    )
    sequence.set_definition("Repetitions", repetitions)
    sequence.set_definition("RepetitionTime", actual_tr)
    sequence.set_definition("MinimumRepetitionTime", minimum_tr)
    sequence.set_definition("RFSpoiling", bool(rf_spoiling))
    sequence.set_definition("RFSpoilingIncrementDeg", rf_spoiling_increment_deg)
    sequence.set_definition("VolumeInterval", max(acquisition_intervals))
    _set_acquisition_interval_definitions(
        sequence,
        requested_interval_s=acquisition_interval_s,
        actual_intervals_s=acquisition_intervals,
        minimum_intervals_s=minimum_acquisition_intervals,
        start_times_s=acquisition_start_times,
    )
    sequence.set_definition("SpoilAfterReadout", bool(spoil_after_readout))
    sequence.set_definition("SpoilerCyclesPerSlice", spoiler_cycles_per_slice)
    sequence.set_definition("SpoilerCyclesPerVoxel", spoiler_cycles_per_voxel)
    sequence.set_definition("SpoilerDuration", spoiler_duration_s)
    sequence.set_definition(
        "SpoilerAxes", "".join(event.channel for event in spoilers) or "none"
    )
    sequence.set_definition("SpoilerEndTimes", spoiler_end_times)
    sequence.set_definition("IdealSpoilerEndTimes", spoiler_end_times)
    return sequence


def make_pulseq_flash(
    *,
    fov_m: Sequence[float] = (0.22, 0.22),
    matrix: Sequence[int] = (64, 64),
    sampling_bandwidth_hz: float = 100_000.0,
    flip_angle_deg: float = 15.0,
    rf_pulse_type: str = "sinc",
    rf_duration_s: float = 1e-3,
    rf_time_bandwidth_product: float = 4.0,
    rf_apodization: float = 0.5,
    rf_slr_sharpness: int = 1,
    rf_custom_waveform_hz: Sequence[complex] | None = None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    rf_custom_name: str | None = None,
    rf_spectral_spatial_pulse=None,
    rf_spectral_spatial_library=None,
    rf_spectral_spatial_schedule=None,
    rf_frequency_offset_hz: float = 0.0,
    slice_thickness_m: float = 3e-3,
    slice_gap_m: float = 0.0,
    n_slices: int = 1,
    slice_offset_m: float = 0.0,
    echo_time_s: float = 5e-3,
    repetition_time_s: float = 15e-3,
    repetitions: int = 1,
    acquisition_interval_s: float | None = None,
    rf_spoiling: bool = True,
    rf_spoiling_increment_deg: float = 117.0,
    spoiler_cycles_per_slice: float = 4.0,
    spoiler_cycles_per_voxel: float = 0.0,
    spoiler_duration_s: float = 2e-3,
    encoding_duration_s: float = 1e-3,
    encoding_axes: Sequence[str] | EncodingFrame = ("+x", "+y", "+z"),
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
    nucleus: str = "H1",
    timing_only: bool = False,
):
    """Build a slice-selective 2D spoiled gradient-echo (FLASH) sequence.

    A spectral-spatial schedule advances once per complete image by default.
    Consequently every phase-encoding line and slice in one image uses the
    same pulse. ``selection_level='tr'`` is supported explicitly for advanced
    experiments and is labelled with Pulseq ``SET`` values, but it mixes
    spectral contrasts within k-space and does not imply separated images.
    """
    pp = _pypulseq()
    encoding_frame = resolve_encoding_frame(encoding_axes)
    fov_read, fov_phase = _positive_values(fov_m, "FOV")
    if len(tuple(fov_m)) != 2:
        raise ValueError("fov_m must contain two values")
    n_read, n_phase = (_positive_integer(value, "matrix size") for value in matrix)
    if len(tuple(matrix)) != 2:
        raise ValueError("matrix must contain two values")
    n_slices = _positive_integer(n_slices, "n_slices")
    repetitions = _positive_integer(repetitions, "repetitions")
    for name, value in {
        "sampling_bandwidth_hz": sampling_bandwidth_hz,
        "flip_angle_deg": flip_angle_deg,
        "rf_duration_s": rf_duration_s,
        "rf_time_bandwidth_product": rf_time_bandwidth_product,
        "slice_thickness_m": slice_thickness_m,
        "echo_time_s": echo_time_s,
        "repetition_time_s": repetition_time_s,
        "spoiler_duration_s": spoiler_duration_s,
        "encoding_duration_s": encoding_duration_s,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value in {
        "slice_gap_m": slice_gap_m,
        "spoiler_cycles_per_slice": spoiler_cycles_per_slice,
        "spoiler_cycles_per_voxel": spoiler_cycles_per_voxel,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    if not np.isfinite(slice_offset_m):
        raise ValueError("slice_offset_m must be finite")
    if not np.isfinite(rf_spoiling_increment_deg):
        raise ValueError("rf_spoiling_increment_deg must be finite")
    rf_spoiling = _uses_rf_spoiling(rf_spoiling, rf_spoiling_increment_deg)
    if not np.isfinite(rf_frequency_offset_hz):
        raise ValueError("rf_frequency_offset_hz must be finite")

    system = _make_scanner_system(
        pp,
        scanner_parameters,
        legacy_kwargs={
            "max_grad": 32,
            "grad_unit": "mT/m",
            "max_slew": 130,
            "slew_unit": "T/m/s",
            "grad_raster_time": 10e-6,
            "rf_ringdown_time": 30e-6,
            "rf_dead_time": 100e-6,
            "adc_dead_time": 20e-6,
        },
    )
    sequence = pp.Sequence(system)
    dwell = _centered_adc_dwell(
        1.0 / sampling_bandwidth_hz,
        n_read,
        system,
    )

    schedule_active = (
        rf_spectral_spatial_library is not None
        or rf_spectral_spatial_schedule is not None
    )
    pulse_library = SpectralSpatialPulseLibrary()
    pulse_schedule = SpectralSpatialSchedule()
    schedule_warnings = ()
    rf_variants = []
    if schedule_active:
        if normalize_rf_pulse_type(rf_pulse_type) != "spectral_spatial":
            raise ValueError(
                "an SPSP pulse library/schedule requires rf_pulse_type='spectral_spatial'"
            )
        pulse_library, pulse_schedule, _ = migrate_spectral_spatial_state(
            library=rf_spectral_spatial_library,
            schedule=rf_spectral_spatial_schedule,
            legacy_pulse=rf_spectral_spatial_pulse,
        )
        schedule_warnings = validate_spectral_spatial_schedule(
            pulse_library,
            pulse_schedule,
            nucleus=nucleus,
            thickness_m=slice_thickness_m,
            scanner_parameters=(
                scanner_parameters
                if scanner_parameters is not None
                else ScannerParameters()
            ),
        )
        expanded_schedule = pulse_schedule.expanded_entries()
        for schedule_index, entry in enumerate(expanded_schedule):
            pulse = pulse_library.get(entry.pulse_id)
            entry_flip_angle_deg = entry.effective_flip_angle_deg(pulse)
            events, duration_s, effective_tbw, normalized_type = (
                _make_slice_selective_rf_events(
                    pp,
                    system,
                    flip_angle_schedule_deg=(entry_flip_angle_deg,),
                    slice_thickness_m=slice_thickness_m,
                    rf_pulse_type="spectral_spatial",
                    rf_duration_s=pulse.duration_s,
                    rf_time_bandwidth_product=pulse.spatial_time_bandwidth_product,
                    rf_apodization=rf_apodization,
                    rf_slr_sharpness=rf_slr_sharpness,
                    rf_custom_waveform_hz=None,
                    rf_custom_raster_s=None,
                    rf_custom_flip_angle_deg=None,
                    rf_frequency_offset_hz=rf_frequency_offset_hz,
                    rf_spectral_spatial_pulse=pulse,
                    spatial_offset_m=0.0,
                    nucleus=nucleus,
                )
            )
            rf_event, logical_gz = events[0]
            rf_variants.append(
                {
                    "schedule_index": schedule_index,
                    "entry": entry,
                    "pulse": pulse,
                    "pulse_id": entry.pulse_id,
                    "rf": rf_event,
                    "logical_gz": logical_gz,
                    "gz": _remap_gradient_event(
                        logical_gz, encoding_frame, "partition"
                    ),
                    "duration_s": duration_s,
                    "tbw": effective_tbw,
                    "flip_angle_deg": entry_flip_angle_deg,
                }
            )
        rf_pulse_type = "spectral_spatial"
        actual_rf_duration_s = max(item["duration_s"] for item in rf_variants)
        effective_rf_tbw = rf_variants[0]["tbw"]
        rf_spectral_spatial_pulse = rf_variants[0]["pulse"]
    else:
        raw_rf_events, actual_rf_duration_s, effective_rf_tbw, rf_pulse_type = (
            _make_slice_selective_rf_events(
                pp,
                system,
                flip_angle_schedule_deg=(flip_angle_deg,),
                slice_thickness_m=slice_thickness_m,
                rf_pulse_type=rf_pulse_type,
                rf_duration_s=rf_duration_s,
                rf_time_bandwidth_product=rf_time_bandwidth_product,
                rf_apodization=rf_apodization,
                rf_slr_sharpness=rf_slr_sharpness,
                rf_custom_waveform_hz=rf_custom_waveform_hz,
                rf_custom_raster_s=rf_custom_raster_s,
                rf_custom_flip_angle_deg=rf_custom_flip_angle_deg,
                rf_frequency_offset_hz=rf_frequency_offset_hz,
                rf_spectral_spatial_pulse=rf_spectral_spatial_pulse,
                spatial_offset_m=0.0,
                nucleus=nucleus,
            )
        )
        rf_event, logical_gz = raw_rf_events[0]
        rf_variants.append(
            {
                "schedule_index": 0,
                "entry": None,
                "pulse": None,
                "pulse_id": "",
                "rf": rf_event,
                "logical_gz": logical_gz,
                "gz": _remap_gradient_event(logical_gz, encoding_frame, "partition"),
                "duration_s": actual_rf_duration_s,
                "tbw": effective_rf_tbw,
                "flip_angle_deg": float(flip_angle_deg),
            }
        )

    adc_duration = n_read * dwell
    flat_time = _ceil_to_raster(adc_duration, system.grad_raster_time)
    gx_read = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        amplitude=(n_read / fov_read) / adc_duration,
        flat_time=flat_time,
        system=system,
    )
    adc = pp.make_adc(
        num_samples=n_read,
        dwell=dwell,
        delay=gx_read.rise_time + flat_time / 2.0 - adc_duration / 2.0,
        system=system,
    )
    logical_read_area = logical_gradient_area(gx_read, encoding_frame, "read")
    gx_pre = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        area=-logical_read_area / 2.0,
        duration=encoding_duration_s,
        system=system,
    )
    readout_duration = pp.calc_duration(gx_read, adc)
    minimum_echo_time = 0.0
    minimum_repetition_time = 0.0
    for variant in rf_variants:
        rf = variant["rf"]
        gz = variant["gz"]
        rf_center, _ = pp.calc_rf_center(rf)
        rf_block_duration = pp.calc_duration(rf, gz)
        logical_slice_area = logical_gradient_area(gz, encoding_frame, "partition")
        slice_rephase_area = _slice_rephase_area(
            rf,
            gz,
            encoding_frame,
            rf_pulse_type=rf_pulse_type,
            gradient_raster_s=system.grad_raster_time,
        )
        gz_rephase = make_role_trapezoid(
            pp,
            encoding_frame,
            "partition",
            area=slice_rephase_area,
            duration=encoding_duration_s,
            system=system,
        )
        echo_without_delay = (
            rf_block_duration
            - (rf.delay + rf_center)
            + encoding_duration_s
            + adc.delay
            + adc_duration / 2.0
        )
        requested_te_delay = echo_time_s - echo_without_delay
        if timing_only:
            minimum_echo_time = max(minimum_echo_time, echo_without_delay)
        if requested_te_delay < -1e-12 and not timing_only:
            label = (
                variant["entry"].display_name or variant["pulse"].name
                if variant["entry"] is not None
                else "RF pulse"
            )
            raise ValueError(
                f"echo_time_s is too short for {label}; minimum is "
                f"{echo_without_delay:.9g} s"
            )
        te_delay_value = _ceil_to_raster(
            max(0.0, requested_te_delay), system.block_duration_raster
        )
        actual_te_variant = echo_without_delay + te_delay_value
        line_without_tr_delay = (
            rf_block_duration
            + encoding_duration_s
            + te_delay_value
            + readout_duration
            + spoiler_duration_s
        )
        requested_tr_delay = repetition_time_s - line_without_tr_delay
        if timing_only:
            minimum_repetition_time = max(
                minimum_repetition_time, line_without_tr_delay
            )
        if requested_tr_delay < -1e-12 and not timing_only:
            label = (
                variant["entry"].display_name or variant["pulse"].name
                if variant["entry"] is not None
                else "RF pulse"
            )
            raise ValueError(
                f"repetition_time_s is too short for {label}; minimum is "
                f"{line_without_tr_delay:.9g} s"
            )
        tr_delay_value = _ceil_to_raster(
            max(0.0, requested_tr_delay), system.block_duration_raster
        )
        variant.update(
            rf_center=rf_center,
            rf_block_duration=rf_block_duration,
            logical_slice_area=logical_slice_area,
            slice_rephase_area=slice_rephase_area,
            gz_rephase=gz_rephase,
            te_delay=te_delay_value,
            actual_te=actual_te_variant,
            minimum_tr=line_without_tr_delay,
            tr_delay=tr_delay_value,
            actual_tr=line_without_tr_delay + tr_delay_value,
        )
    actual_te = max(variant["actual_te"] for variant in rf_variants)
    actual_tr = max(variant["actual_tr"] for variant in rf_variants)
    if timing_only:
        actual_tr = max(actual_tr, minimum_repetition_time)
        return FlashTimingPlan(
            sampling_bandwidth_hz=1.0 / dwell,
            dwell_s=float(dwell),
            minimum_echo_time_s=float(minimum_echo_time),
            minimum_repetition_time_s=float(minimum_repetition_time),
            minimum_acquisition_interval_s=float(n_slices * n_phase * actual_tr),
        )

    phase_areas = (np.arange(n_phase) - n_phase // 2) / fov_phase
    slice_positions = float(slice_offset_m) + (
        np.arange(n_slices, dtype=float) - (n_slices - 1) / 2.0
    ) * (slice_thickness_m + slice_gap_m)
    _, slice_sign = encoding_frame.axis_and_sign("partition")
    excitation_index = 0
    spoiler_end_times = []
    acquisition_start_times = []
    acquisition_intervals = []
    minimum_acquisition_intervals = []
    used_frame_schedule_indices = []
    used_frame_pulse_ids = []
    for repetition in range(repetitions):
        acquisition_start = _sequence_duration_s(sequence)
        acquisition_start_times.append(acquisition_start)
        if schedule_active and pulse_schedule.selection_level == "frame":
            frame_schedule_index, _ = pulse_schedule.entry_for_index(repetition)
        else:
            frame_schedule_index = 0
        frame_variant = rf_variants[frame_schedule_index]
        used_frame_schedule_indices.append(int(frame_schedule_index))
        used_frame_pulse_ids.append(str(frame_variant["pulse_id"]))
        for slice_index, position in enumerate(slice_positions):
            for line, phase_area in enumerate(phase_areas):
                if schedule_active and pulse_schedule.selection_level == "tr":
                    schedule_index, _ = pulse_schedule.entry_for_index(excitation_index)
                    variant = rf_variants[schedule_index]
                else:
                    schedule_index = frame_schedule_index
                    variant = frame_variant
                rf = variant["rf"]
                gz = variant["gz"]
                spsp_logical_gz = variant["logical_gz"]
                rf_center = variant["rf_center"]
                logical_slice_area = variant["logical_slice_area"]
                slice_rephase_area = variant["slice_rephase_area"]
                gz_rephase = variant["gz_rephase"]
                te_delay_value = variant["te_delay"]
                tr_delay_value = variant["tr_delay"]
                rf_phase_deg = (
                    _rf_spoiling_phase_deg(excitation_index, rf_spoiling_increment_deg)
                    if rf_spoiling
                    else 0.0
                )
                if rf_pulse_type == "spectral_spatial":
                    rf_for_slice = shifted_spectral_spatial_rf_event(
                        rf,
                        spsp_logical_gz,
                        spatial_offset_m=position,
                        rf_raster_s=system.rf_raster_time,
                        gradient_raster_s=system.grad_raster_time,
                    )
                    total_rf_frequency_offset_hz = rf_frequency_offset_hz
                else:
                    rf_for_slice = rf
                    logical_slice_amplitude = float(gz.amplitude) * slice_sign
                    total_rf_frequency_offset_hz = (
                        rf_frequency_offset_hz + logical_slice_amplitude * position
                    )
                rf_for_slice.freq_offset = total_rf_frequency_offset_hz
                rf_for_slice.phase_offset = (
                    np.deg2rad(rf_phase_deg)
                    - 2.0 * np.pi * total_rf_frequency_offset_hz * rf_center
                )
                adc.phase_offset = np.deg2rad(rf_phase_deg)
                gy_phase = make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "phase",
                    area=float(phase_area),
                    duration=encoding_duration_s,
                    system=system,
                )
                gx_spoil = make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "read",
                    area=(
                        -logical_read_area / 2.0
                        + spoiler_cycles_per_voxel / (fov_read / n_read)
                    ),
                    duration=spoiler_duration_s,
                    system=system,
                )
                gy_spoil = make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "phase",
                    area=(
                        -float(phase_area)
                        + spoiler_cycles_per_voxel / (fov_phase / n_phase)
                    ),
                    duration=spoiler_duration_s,
                    system=system,
                )
                gz_spoil = make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "partition",
                    # Keep the net moment per TR equal to the spoiler moment.
                    area=(
                        -(logical_slice_area + slice_rephase_area)
                        + spoiler_cycles_per_slice / slice_thickness_m
                    ),
                    duration=spoiler_duration_s,
                    system=system,
                )
                rf_block_events = [
                    rf_for_slice,
                    gz,
                    pp.make_label("LIN", "SET", line),
                    pp.make_label("SLC", "SET", slice_index),
                    pp.make_label("REP", "SET", repetition),
                ]
                if schedule_active:
                    rf_block_events.append(
                        pp.make_label("SET", "SET", int(schedule_index))
                    )
                sequence.add_block(*rf_block_events)
                sequence.add_block(gx_pre, gy_phase, gz_rephase)
                if te_delay_value:
                    sequence.add_block(pp.make_delay(te_delay_value))
                sequence.add_block(gx_read, adc)
                sequence.add_block(gx_spoil, gy_spoil, gz_spoil)
                spoiler_end_times.append(float(sequence.duration()[0]))
                if tr_delay_value:
                    sequence.add_block(pp.make_delay(tr_delay_value))
                excitation_index += 1
        minimum_interval = _sequence_duration_s(sequence) - acquisition_start
        actual_interval, _ = _finish_acquisition_interval(
            pp,
            sequence,
            acquisition_start_s=acquisition_start,
            requested_interval_s=acquisition_interval_s,
            raster_s=system.block_duration_raster,
            acquisition_name="FLASH image",
        )
        minimum_acquisition_intervals.append(minimum_interval)
        acquisition_intervals.append(actual_interval)

    _raise_for_timing_errors(sequence, "FLASH")
    slice_extent = n_slices * slice_thickness_m + (n_slices - 1) * slice_gap_m
    sequence.set_definition("Name", "flash_2d")
    sequence.set_definition("FOV", [fov_read, fov_phase, slice_extent])
    sequence.set_definition("MatrixSize", [n_read, n_phase])
    set_pulseq_encoding_definitions(
        sequence,
        encoding_frame,
        fov_m=(fov_read, fov_phase, slice_extent),
        matrix=(n_read, n_phase, n_slices),
    )
    sequence.set_definition("SamplingBandwidth", 1.0 / dwell)
    sequence.set_definition("RequestedSamplingBandwidth", sampling_bandwidth_hz)
    sequence.set_definition("FlipAngleDeg", float(flip_angle_deg))
    sequence.set_definition("TE", actual_te)
    sequence.set_definition("TR", actual_tr)
    sequence.set_definition("Repetitions", repetitions)
    sequence.set_definition("VolumeInterval", max(acquisition_intervals))
    _set_acquisition_interval_definitions(
        sequence,
        requested_interval_s=acquisition_interval_s,
        actual_intervals_s=acquisition_intervals,
        minimum_intervals_s=minimum_acquisition_intervals,
        start_times_s=acquisition_start_times,
    )
    sequence.set_definition("SliceThickness", slice_thickness_m)
    sequence.set_definition("SliceGap", slice_gap_m)
    sequence.set_definition("SliceOffset", float(slice_offset_m))
    sequence.set_definition(
        "SlicePositions", [float(value) for value in slice_positions]
    )
    sequence.set_definition("RFSpoiling", bool(rf_spoiling))
    sequence.set_definition("RFSpoilingIncrementDeg", rf_spoiling_increment_deg)
    sequence.set_definition("SpoilerCyclesPerSlice", spoiler_cycles_per_slice)
    sequence.set_definition("SpoilerCyclesPerVoxel", spoiler_cycles_per_voxel)
    sequence.set_definition("SpoilerDuration", spoiler_duration_s)
    sequence.set_definition(
        "SpoilerAxes", "".join(code[-1] for code in encoding_frame.axis_codes)
    )
    sequence.set_definition("SpoilerEndTimes", spoiler_end_times)
    sequence.set_definition("IdealSpoilerEndTimes", spoiler_end_times)
    _set_rf_definitions(
        sequence,
        pulse_type=rf_pulse_type,
        requested_duration_s=rf_duration_s,
        actual_duration_s=actual_rf_duration_s,
        time_bandwidth_product=effective_rf_tbw,
        apodization=rf_apodization,
        slr_sharpness=rf_slr_sharpness,
        custom_name=rf_custom_name,
        custom_flip_angle_deg=rf_custom_flip_angle_deg,
        frequency_offset_hz=rf_frequency_offset_hz,
        spectral_spatial_pulse=rf_spectral_spatial_pulse,
    )
    if schedule_active:
        expanded = pulse_schedule.expanded_entries()
        sequence.set_definition("SPSPScheduleVersion", 1)
        sequence.set_definition(
            "SPSPScheduleSelectionLevel", pulse_schedule.selection_level
        )
        sequence.set_definition("SPSPScheduleCyclic", bool(pulse_schedule.cyclic))
        sequence.set_definition("SPSPScheduleStepCount", len(expanded))
        for definition_name, values in (
            ("SPSPScheduleEntryIDs", [entry.entry_id for entry in expanded]),
            ("SPSPSchedulePulseIDs", [entry.pulse_id for entry in expanded]),
            (
                "SPSPScheduleDisplayNames",
                [
                    entry.display_name or pulse_library.get(entry.pulse_id).name
                    for entry in expanded
                ],
            ),
            (
                "SPSPScheduleTargetNames",
                [entry.target_metabolite for entry in expanded],
            ),
            ("SPSPScheduleScaleModes", [entry.scale_mode for entry in expanded]),
        ):
            sequence.set_definition(
                definition_name, json.dumps(values, separators=(",", ":"))
            )
        sequence.set_definition(
            "SPSPScheduleAmplitudeScales",
            [
                entry.effective_amplitude_scale(pulse_library.get(entry.pulse_id))
                for entry in expanded
            ],
        )
        sequence.set_definition(
            "SPSPScheduleFlipAnglesDeg",
            [
                entry.effective_flip_angle_deg(pulse_library.get(entry.pulse_id))
                for entry in expanded
            ],
        )
        sequence.set_definition(
            "SPSPSchedulePulseDurations",
            [pulse_library.get(entry.pulse_id).duration_s for entry in expanded],
        )
        sequence.set_definition("SPSPScheduleFrameIndices", used_frame_schedule_indices)
        sequence.set_definition(
            "SPSPScheduleFramePulseIDs",
            json.dumps(used_frame_pulse_ids, separators=(",", ":")),
        )
        sequence.set_definition(
            "SPSPScheduleWarnings",
            json.dumps(list(schedule_warnings), separators=(",", ":")),
        )
        sequence.set_definition(
            "SPSPScheduleContrastWarning",
            (
                "Schedule changes within phase-encoding lines; SET ADC labels are required for interpretation."
                if pulse_schedule.selection_level == "tr"
                else "Each schedule step applies to one complete logical FLASH frame."
            ),
        )
    return sequence


def _spiral_waveforms(
    *,
    matrix: tuple[int, int],
    fov_m: tuple[float, float],
    requested_dwell_s: float,
    turns: float,
    system,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return hardware-limited, raster-aligned spiral gradient waveforms."""
    n_x, n_y = matrix
    sample_count = n_x * n_y
    adc_raster = float(system.adc_raster_time)
    grad_raster = float(system.grad_raster_time)
    raster_ratio = int(round(grad_raster / adc_raster))
    if raster_ratio <= 0 or not np.isclose(
        raster_ratio * adc_raster, grad_raster, rtol=0.0, atol=1e-15
    ):
        raise ValueError("gradient and ADC rasters are not commensurate")

    requested_steps = max(1, int(np.ceil(requested_dwell_s / adc_raster - 1e-12)))

    def aligned_dwell_steps(minimum_steps: int) -> int:
        steps = max(1, int(minimum_steps))
        while (sample_count * steps) % raster_ratio:
            steps += 1
        return steps

    dwell_steps = aligned_dwell_steps(requested_steps)
    max_grad = float(system.max_grad)
    max_slew = float(system.max_slew)
    for _ in range(20):
        dwell = dwell_steps * adc_raster
        duration = sample_count * dwell
        gradient_samples = int(round(duration / grad_raster))
        edges = np.linspace(0.0, 1.0, gradient_samples + 1)
        # Smoothstep makes dk/dt zero at both ends, allowing the arbitrary
        # gradient to connect to zero without an instantaneous slew jump.
        progress = edges * edges * (3.0 - 2.0 * edges)
        angle = 2.0 * np.pi * float(turns) * progress
        kx_edges = n_x / (2.0 * fov_m[0]) * progress * np.cos(angle)
        ky_edges = n_y / (2.0 * fov_m[1]) * progress * np.sin(angle)
        gx = np.diff(kx_edges) / grad_raster
        gy = np.diff(ky_edges) / grad_raster
        amplitude_ratio = max(
            float(np.max(np.abs(gx))) / max_grad,
            float(np.max(np.abs(gy))) / max_grad,
        )
        slew_x = np.concatenate(([-2.0 * gx[0]], np.diff(gx), [-2.0 * gx[-1]]))
        slew_y = np.concatenate(([-2.0 * gy[0]], np.diff(gy), [-2.0 * gy[-1]]))
        slew_ratio = max(
            float(np.max(np.abs(slew_x))) / (grad_raster * max_slew),
            float(np.max(np.abs(slew_y))) / (grad_raster * max_slew),
        )
        limit_ratio = max(amplitude_ratio, np.sqrt(slew_ratio))
        if limit_ratio <= 1.0 + 1e-10:
            return gx, gy, dwell
        dwell_steps = aligned_dwell_steps(
            int(np.ceil(dwell_steps * limit_ratio * 1.01))
        )
    raise ValueError("unable to design a spiral readout within the system limits")


def make_pulseq_spiral(
    *,
    fov_m: Sequence[float] = (0.22, 0.22),
    matrix: Sequence[int] = (16, 16),
    sampling_bandwidth_hz: float = 50_000.0,
    spiral_turns: float | None = None,
    flip_angle_deg: float = 90.0,
    variable_flip_angle: bool = False,
    vfa_final_flip_angle_deg: float = 90.0,
    rf_pulse_type: str = "sinc",
    rf_duration_s: float = 3e-3,
    rf_time_bandwidth_product: float = 4.0,
    rf_apodization: float = 0.5,
    rf_slr_sharpness: int = 1,
    rf_custom_waveform_hz: Sequence[complex] | None = None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    rf_custom_name: str | None = None,
    rf_spectral_spatial_pulse=None,
    rf_frequency_offset_hz: float = 0.0,
    slice_thickness_m: float = 3e-3,
    slice_gap_m: float = 0.0,
    n_slices: int = 1,
    repetitions: int = 1,
    echo_time_s: float = 20e-3,
    repetition_time_s: float | None = 1.0,
    rf_spoiling: bool = False,
    rf_spoiling_increment_deg: float = 117.0,
    slice_offset_m: float = 0.0,
    encoding_axes: Sequence[str] | EncodingFrame = ("+x", "+y", "+z"),
    spoil_after_slice: bool = True,
    spoiler_cycles_per_slice: float = 8.0,
    spoiler_cycles_per_voxel: float = 0.0,
    spoiler_duration_s: float = 4e-3,
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
    nucleus: str = "H1",
):
    """Build a single-shot, centre-out 2D spiral Pulseq acquisition.

    One continuous spiral interleaf with ``matrix[0] * matrix[1]`` ADC samples
    is acquired for every slice and repetition. The requested receiver
    bandwidth is reduced automatically when necessary to satisfy the gradient
    amplitude and slew-rate limits.
    """
    pp = _pypulseq()
    encoding_frame = resolve_encoding_frame(encoding_axes)
    fov_x, fov_y = _positive_values(fov_m, "FOV")
    if len(tuple(fov_m)) != 2:
        raise ValueError("fov_m must contain two values")
    n_x, n_y = (_positive_integer(value, "matrix size") for value in matrix)
    if len(tuple(matrix)) != 2:
        raise ValueError("matrix must contain two values")
    n_slices = _positive_integer(n_slices, "n_slices")
    repetitions = _positive_integer(repetitions, "repetitions")
    for name, value in {
        "sampling_bandwidth_hz": sampling_bandwidth_hz,
        "flip_angle_deg": flip_angle_deg,
        "slice_thickness_m": slice_thickness_m,
        "echo_time_s": echo_time_s,
        "spoiler_duration_s": spoiler_duration_s,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value in {
        "slice_gap_m": slice_gap_m,
        "spoiler_cycles_per_slice": spoiler_cycles_per_slice,
        "spoiler_cycles_per_voxel": spoiler_cycles_per_voxel,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    if spiral_turns is None:
        spiral_turns = max(1.0, min(n_x, n_y) / 2.0)
    spiral_turns = float(spiral_turns)
    if not np.isfinite(spiral_turns) or spiral_turns <= 0:
        raise ValueError("spiral_turns must be positive and finite")
    if repetition_time_s is not None and (
        not np.isfinite(repetition_time_s) or repetition_time_s <= 0
    ):
        raise ValueError("repetition_time_s must be positive and finite")
    if not np.isfinite(rf_frequency_offset_hz):
        raise ValueError("rf_frequency_offset_hz must be finite")
    if not np.isfinite(slice_offset_m):
        raise ValueError("slice_offset_m must be finite")
    if not np.isfinite(rf_spoiling_increment_deg):
        raise ValueError("rf_spoiling_increment_deg must be finite")
    rf_spoiling = _uses_rf_spoiling(rf_spoiling, rf_spoiling_increment_deg)

    if variable_flip_angle:
        flip_angle_schedule_deg = variable_flip_angle_schedule(
            repetitions,
            final_flip_angle_deg=vfa_final_flip_angle_deg,
        )
    else:
        flip_angle_schedule_deg = np.asarray([flip_angle_deg], dtype=float)

    system = _make_scanner_system(
        pp,
        scanner_parameters,
        legacy_kwargs={
            "max_grad": 32,
            "grad_unit": "mT/m",
            "max_slew": 130,
            "slew_unit": "T/m/s",
            "grad_raster_time": 10e-6,
            "rf_ringdown_time": 30e-6,
            "rf_dead_time": 100e-6,
            "adc_dead_time": 20e-6,
        },
    )
    sequence = pp.Sequence(system)
    gx_waveform, gy_waveform, dwell = _spiral_waveforms(
        matrix=(n_x, n_y),
        fov_m=(fov_x, fov_y),
        requested_dwell_s=1.0 / float(sampling_bandwidth_hz),
        turns=spiral_turns,
        system=system,
    )
    gradient_delay = system.adc_dead_time
    read_axis, read_sign = encoding_frame.axis_and_sign("read")
    phase_axis, phase_sign = encoding_frame.axis_and_sign("phase")
    gx = pp.make_arbitrary_grad(
        read_axis,
        read_sign * gx_waveform,
        first=0.0,
        last=0.0,
        delay=gradient_delay,
        system=system,
    )
    gy = pp.make_arbitrary_grad(
        phase_axis,
        phase_sign * gy_waveform,
        first=0.0,
        last=0.0,
        delay=gradient_delay,
        system=system,
    )
    adc = pp.make_adc(
        num_samples=n_x * n_y,
        dwell=dwell,
        delay=gradient_delay,
        system=system,
    )

    rf_events, actual_rf_duration_s, effective_rf_tbw, rf_pulse_type = (
        _make_slice_selective_rf_events(
            pp,
            system,
            flip_angle_schedule_deg=flip_angle_schedule_deg,
            slice_thickness_m=slice_thickness_m,
            rf_pulse_type=rf_pulse_type,
            rf_duration_s=rf_duration_s,
            rf_time_bandwidth_product=rf_time_bandwidth_product,
            rf_apodization=rf_apodization,
            rf_slr_sharpness=rf_slr_sharpness,
            rf_custom_waveform_hz=rf_custom_waveform_hz,
            rf_custom_raster_s=rf_custom_raster_s,
            rf_custom_flip_angle_deg=rf_custom_flip_angle_deg,
            rf_frequency_offset_hz=rf_frequency_offset_hz,
            rf_spectral_spatial_pulse=rf_spectral_spatial_pulse,
            spatial_offset_m=0.0,
            nucleus=nucleus,
        )
    )
    spsp_logical_gz = rf_events[0][1]
    rf_events = [
        (rf, _remap_gradient_event(gz, encoding_frame, "partition"))
        for rf, gz in rf_events
    ]
    rf, gz = rf_events[0]
    rephase_time = 0.8e-3
    gz_rephase = make_role_trapezoid(
        pp,
        encoding_frame,
        "partition",
        area=_slice_rephase_area(
            rf,
            gz,
            encoding_frame,
            rf_pulse_type=rf_pulse_type,
            gradient_raster_s=system.grad_raster_time,
        ),
        duration=rephase_time,
        system=system,
    )
    gx_rewind = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        area=-logical_gradient_area(gx, encoding_frame, "read"),
        system=system,
    )
    gy_rewind = make_role_trapezoid(
        pp,
        encoding_frame,
        "phase",
        area=-logical_gradient_area(gy, encoding_frame, "phase"),
        system=system,
    )
    rewind_duration = max(pp.calc_duration(gx_rewind), pp.calc_duration(gy_rewind))
    gx_rewind = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        area=-logical_gradient_area(gx, encoding_frame, "read"),
        duration=rewind_duration,
        system=system,
    )
    gy_rewind = make_role_trapezoid(
        pp,
        encoding_frame,
        "phase",
        area=-logical_gradient_area(gy, encoding_frame, "phase"),
        duration=rewind_duration,
        system=system,
    )

    spoilers = []
    if spoil_after_slice:
        if spoiler_cycles_per_voxel > 0:
            for role, voxel_size in zip(("read", "phase"), (fov_x / n_x, fov_y / n_y)):
                spoilers.append(
                    make_role_trapezoid(
                        pp,
                        encoding_frame,
                        role,
                        area=spoiler_cycles_per_voxel / voxel_size,
                        duration=spoiler_duration_s,
                        system=system,
                    )
                )
        if spoiler_cycles_per_slice > 0:
            spoilers.append(
                make_role_trapezoid(
                    pp,
                    encoding_frame,
                    "partition",
                    area=spoiler_cycles_per_slice / slice_thickness_m,
                    duration=spoiler_duration_s,
                    system=system,
                )
            )

    slice_spacing = slice_thickness_m + slice_gap_m
    slice_positions = (
        float(slice_offset_m)
        + (np.arange(n_slices, dtype=float) - (n_slices - 1) / 2.0) * slice_spacing
    )
    rf_center, _ = pp.calc_rf_center(rf)
    rf_block_duration = pp.calc_duration(rf, gz)
    first_sample_from_readout_start = adc.delay + adc.dwell / 2.0
    echo_without_delay = (
        rf_block_duration
        - (rf.delay + rf_center)
        + pp.calc_duration(gz_rephase)
        + first_sample_from_readout_start
    )
    requested_te_delay = float(echo_time_s) - echo_without_delay
    if requested_te_delay < -1e-12:
        raise ValueError(
            f"echo_time_s is too short; minimum is {echo_without_delay:.9g} s"
        )
    te_delay_value = _ceil_to_raster(
        max(0.0, requested_te_delay), system.block_duration_raster
    )
    actual_te = echo_without_delay + te_delay_value
    _, slice_sign = encoding_frame.axis_and_sign("partition")
    minimum_repetition_time = None
    actual_repetition_time = None
    spoiler_end_times = []
    acquisition_start_times = []
    acquisition_intervals = []
    minimum_acquisition_intervals = []
    excitation_index = 0
    for repetition in range(repetitions):
        rf, gz = rf_events[repetition if variable_flip_angle else 0]
        repetition_start = (
            0.0 if not sequence.block_events else float(sequence.duration()[0])
        )
        acquisition_start_times.append(repetition_start)
        for slice_index, position in enumerate(slice_positions):
            rf_phase_deg = (
                _rf_spoiling_phase_deg(excitation_index, rf_spoiling_increment_deg)
                if rf_spoiling
                else 0.0
            )
            if rf_pulse_type == "spectral_spatial":
                rf_for_slice = shifted_spectral_spatial_rf_event(
                    rf,
                    spsp_logical_gz,
                    spatial_offset_m=position,
                    rf_raster_s=system.rf_raster_time,
                    gradient_raster_s=system.grad_raster_time,
                )
                rf_for_slice.freq_offset = rf_frequency_offset_hz
                rf_for_slice.phase_offset = (
                    np.deg2rad(rf_phase_deg)
                    - 2.0 * np.pi * rf_frequency_offset_hz * rf_center
                )
            else:
                logical_slice_amplitude = float(gz.amplitude) * slice_sign
                slice_frequency_offset_hz = logical_slice_amplitude * position
                rf_for_slice = rf
                rf_for_slice.freq_offset = (
                    rf_frequency_offset_hz + slice_frequency_offset_hz
                )
                rf_for_slice.phase_offset = (
                    np.deg2rad(rf_phase_deg)
                    - 2.0 * np.pi * slice_frequency_offset_hz * rf_center
                )
            adc.phase_offset = np.deg2rad(rf_phase_deg)
            sequence.add_block(
                rf_for_slice,
                gz,
                pp.make_label("SLC", "SET", slice_index),
                pp.make_label("REP", "SET", repetition),
            )
            sequence.add_block(gz_rephase)
            if te_delay_value:
                sequence.add_block(pp.make_delay(te_delay_value))
            sequence.add_block(gx, gy, adc)
            sequence.add_block(gx_rewind, gy_rewind)
            if spoilers:
                sequence.add_block(*spoilers)
                spoiler_end_times.append(float(sequence.duration()[0]))
            excitation_index += 1
        acquired = float(sequence.duration()[0]) - repetition_start
        if minimum_repetition_time is None:
            minimum_repetition_time = acquired
            actual_repetition_time = (
                acquired if repetition_time_s is None else float(repetition_time_s)
            )
            if actual_repetition_time < acquired - 1e-12:
                raise ValueError(
                    f"repetition_time_s is too short; minimum is {acquired:.9g} s"
                )
        delay = actual_repetition_time - acquired
        if delay > 1e-12:
            sequence.add_block(pp.make_delay(delay))
        minimum_acquisition_intervals.append(acquired)
        acquisition_intervals.append(float(sequence.duration()[0]) - repetition_start)

    _raise_for_timing_errors(sequence, "spiral")
    slice_extent = n_slices * slice_thickness_m + (n_slices - 1) * slice_gap_m
    sequence.set_definition("Name", "spiral_2d")
    sequence.set_definition("Trajectory", "spiral")
    sequence.set_definition("FOV", [fov_x, fov_y, slice_extent])
    sequence.set_definition("MatrixSize", [n_x, n_y])
    set_pulseq_encoding_definitions(
        sequence,
        encoding_frame,
        fov_m=(fov_x, fov_y, slice_extent),
        matrix=(n_x, n_y, n_slices),
    )
    sequence.set_definition("SamplingBandwidth", 1.0 / dwell)
    sequence.set_definition("RequestedSamplingBandwidth", sampling_bandwidth_hz)
    sequence.set_definition("ReadoutDuration", n_x * n_y * dwell)
    sequence.set_definition("SpiralTurns", spiral_turns)
    sequence.set_definition("SpiralInterleaves", 1)
    sequence.set_definition("TE", actual_te)
    sequence.set_definition(
        "FlipAngleDeg",
        float(vfa_final_flip_angle_deg if variable_flip_angle else flip_angle_deg),
    )
    _set_rf_definitions(
        sequence,
        pulse_type=rf_pulse_type,
        requested_duration_s=rf_duration_s,
        actual_duration_s=actual_rf_duration_s,
        time_bandwidth_product=effective_rf_tbw,
        apodization=rf_apodization,
        slr_sharpness=rf_slr_sharpness,
        custom_name=rf_custom_name,
        custom_flip_angle_deg=rf_custom_flip_angle_deg,
        frequency_offset_hz=rf_frequency_offset_hz,
        spectral_spatial_pulse=rf_spectral_spatial_pulse,
    )
    sequence.set_definition("VariableFlipAngle", bool(variable_flip_angle))
    if variable_flip_angle:
        sequence.set_definition("VariableFlipAngleDimension", "repetition")
        sequence.set_definition(
            "VariableFlipAngleFinalDeg", float(vfa_final_flip_angle_deg)
        )
        sequence.set_definition(
            "FlipAngleScheduleDeg",
            [float(value) for value in flip_angle_schedule_deg],
        )
        sequence.set_definition("VariableFlipAngleReferenceDOI", VFA_REFERENCE_DOI)
    sequence.set_definition("SliceThickness", slice_thickness_m)
    sequence.set_definition("SliceGap", slice_gap_m)
    sequence.set_definition("SliceSpacing", slice_spacing)
    sequence.set_definition("SliceOffset", float(slice_offset_m))
    sequence.set_definition(
        "SlicePositions", [float(value) for value in slice_positions]
    )
    sequence.set_definition("Repetitions", repetitions)
    sequence.set_definition("RepetitionTime", actual_repetition_time)
    sequence.set_definition("MinimumRepetitionTime", minimum_repetition_time)
    sequence.set_definition("RFSpoiling", bool(rf_spoiling))
    sequence.set_definition("RFSpoilingIncrementDeg", rf_spoiling_increment_deg)
    sequence.set_definition("VolumeInterval", max(acquisition_intervals))
    _set_acquisition_interval_definitions(
        sequence,
        requested_interval_s=repetition_time_s,
        actual_intervals_s=acquisition_intervals,
        minimum_intervals_s=minimum_acquisition_intervals,
        start_times_s=acquisition_start_times,
    )
    sequence.set_definition("SpoilAfterSlice", bool(spoil_after_slice))
    sequence.set_definition("SpoilerCyclesPerSlice", spoiler_cycles_per_slice)
    sequence.set_definition("SpoilerCyclesPerVoxel", spoiler_cycles_per_voxel)
    sequence.set_definition("SpoilerDuration", spoiler_duration_s)
    sequence.set_definition(
        "SpoilerAxes", "".join(event.channel for event in spoilers) or "none"
    )
    sequence.set_definition("SpoilerEndTimes", spoiler_end_times)
    sequence.set_definition("IdealSpoilerEndTimes", spoiler_end_times)
    return sequence


def _raise_for_timing_errors(sequence, name: str) -> None:
    ok, errors = sequence.check_timing()
    if not ok:
        details = "\n".join(str(error) for error in errors)
        raise RuntimeError(f"{name} sequence timing check failed:\n{details}")
