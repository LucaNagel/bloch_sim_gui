"""Configurable Pulseq sequence builders used by the desktop simulator."""

from __future__ import annotations

from copy import copy
from typing import Mapping, Sequence

import numpy as np

from .flip_angles import VFA_REFERENCE_DOI, variable_flip_angle_schedule
from .encoding import (
    EncodingFrame,
    logical_gradient_area,
    make_role_trapezoid,
    resolve_encoding_frame,
    set_pulseq_encoding_definitions,
)
from .rf_pulses import make_pulseq_rf_events, set_rf_definitions
from .scanner import ScannerParameters
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


def _sequence_duration_s(sequence) -> float:
    return 0.0 if not sequence.block_events else float(sequence.duration()[0])


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
    rf_slr_sharpness: float,
    rf_custom_waveform_hz: Sequence[complex] | None,
    rf_custom_raster_s: float | None,
    rf_custom_flip_angle_deg: float | None,
    rf_frequency_offset_hz: float,
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


def _set_rf_definitions(
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
    rf_slr_sharpness: float = 1.0,
    rf_custom_waveform_hz: Sequence[complex] | None = None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    rf_custom_name: str | None = None,
    rf_frequency_offset_hz: float = 0.0,
    encoding_duration_s: float = 2e-3,
    echo_time_s: float = 6e-3,
    repetition_time_s: float = 0.1,
    repetitions: int = 1,
    acquisition_interval_s: float | None = None,
    slice_offset_m: float = 0.0,
    encoding_axes: Sequence[str] | EncodingFrame = ("+x", "+y", "+z"),
    spoil_after_readout: bool = True,
    spoiler_cycles_per_slice: float = 4.0,
    spoiler_cycles_per_voxel: float = 0.0,
    spoiler_duration_s: float = 2e-3,
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
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
        logical_slice_amplitude = float(slice_gradient.amplitude) * slice_sign
        frequency_offset_hz = rf_frequency_offset_hz + logical_slice_amplitude * float(
            slice_offset_m
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
    for repetition in range(repetitions):
        acquisition_start = _sequence_duration_s(sequence)
        acquisition_start_times.append(acquisition_start)
        for encoding_index, (x_index, y_index) in enumerate(order):
            rf, gz = rf_events[encoding_index if variable_flip_angle else 0]
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
                area=-logical_gradient_area(gz, encoding_frame, "partition") / 2,
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
    rf_slr_sharpness: float = 1.0,
    rf_custom_waveform_hz: Sequence[complex] | None = None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    rf_custom_name: str | None = None,
    rf_frequency_offset_hz: float = 0.0,
    sampling_bandwidth_hz: float = 10_000.0,
    encoding_duration_s: float = 1e-3,
    repetition_time_s: float | None = 10e-3,
    rf_phase_start_deg: float = 180.0,
    rf_phase_increment_deg: float = 180.0,
    dummy_repetitions: int = 1,
    repetitions: int = 1,
    acquisition_interval_s: float | None = None,
    use_alpha_half: bool = True,
    encoding_axes: Sequence[str] | EncodingFrame = ("+x", "+y", "+z"),
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
):
    """Build a fully balanced, non-selective Cartesian 3D bSSFP sequence."""
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
            flip_angles_deg=(flip_angle_deg, flip_angle_deg / 2),
            pulse_type=rf_pulse_type,
            duration_s=rf_duration_s,
            time_bandwidth_product=rf_time_bandwidth_product,
            apodization=rf_apodization,
            slr_sharpness=rf_slr_sharpness,
            custom_waveform_hz=rf_custom_waveform_hz,
            custom_raster_s=rf_custom_raster_s,
            custom_flip_angle_deg=rf_custom_flip_angle_deg,
            frequency_offset_hz=rf_frequency_offset_hz,
        )
    )
    rf, rf_alpha_half = rf_events
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
    rf_block_duration = pp.calc_duration(rf)
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

    if use_alpha_half:
        rf_alpha_half.phase_offset = pulseq_phase_offset_rad(
            rf_phase_start_deg,
            frequency_offset_hz=rf_frequency_offset_hz,
            event_center_s=pp.calc_rf_center(rf_alpha_half)[0],
        )
        preparation_delay = actual_tr / 2 - pp.calc_duration(rf_alpha_half)
        preparation_delay = round(preparation_delay / raster) * raster
        if preparation_delay < 0:
            raise ValueError("TR is too short for alpha/2 preparation")
        sequence.add_block(rf_alpha_half)
        if preparation_delay:
            sequence.add_block(pp.make_delay(preparation_delay))

    ky_areas = (np.arange(n_phase) - n_phase // 2) / fov_y
    kz_areas = (np.arange(n_partition) - n_partition // 2) / fov_z
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
            area=kz,
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
        sequence.add_block(rf)
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
    )
    return sequence


def make_pulseq_epi(
    *,
    fov_m: Sequence[float] = (0.22, 0.22),
    matrix: Sequence[int] = (16, 16),
    sampling_bandwidth_hz: float = 50_000.0,
    flip_angle_deg: float = 90.0,
    variable_flip_angle: bool = False,
    vfa_final_flip_angle_deg: float = 90.0,
    rf_pulse_type: str = "sinc",
    rf_duration_s: float = 3e-3,
    rf_time_bandwidth_product: float = 4.0,
    rf_apodization: float = 0.5,
    rf_slr_sharpness: float = 1.0,
    rf_custom_waveform_hz: Sequence[complex] | None = None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    rf_custom_name: str | None = None,
    rf_frequency_offset_hz: float = 0.0,
    slice_thickness_m: float = 3e-3,
    slice_gap_m: float = 0.0,
    n_slices: int = 1,
    repetitions: int = 1,
    echo_time_s: float = 20e-3,
    repetition_time_s: float | None = 1.0,
    slice_offset_m: float = 0.0,
    encoding_axes: Sequence[str] | EncodingFrame = ("+x", "+y", "+z"),
    spoil_after_slice: bool = True,
    spoiler_cycles_per_slice: float = 8.0,
    spoiler_cycles_per_voxel: float = 0.0,
    spoiler_duration_s: float = 4e-3,
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
):
    """Build a Cartesian single-shot EPI Pulseq sequence for export.

    Variable flip angles advance once per repetition and are shared by every
    slice acquired within that repetition.
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
    dwell = (
        round((1.0 / sampling_bandwidth_hz) / system.adc_raster_time)
        * system.adc_raster_time
    )
    if dwell <= 0:
        raise ValueError("sampling bandwidth exceeds the ADC raster capability")
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
        )
    )
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
        area=-logical_gradient_area(gz, encoding_frame, "partition") / 2,
        duration=pre_time,
        system=system,
    )
    logical_read_area = logical_gradient_area(gx, encoding_frame, "read")
    relative_x_end = -logical_read_area / 2 + (logical_read_area if n_y % 2 else 0.0)
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
    blip_duration = pp.calc_duration(gy_blip)
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
    if requested_te_delay < -1e-12:
        raise ValueError(
            f"echo_time_s is too short; minimum is {echo_without_delay:.9g} s"
        )
    te_delay_value = _ceil_to_raster(
        max(0.0, requested_te_delay), system.block_duration_raster
    )
    actual_te = echo_without_delay + te_delay_value
    _, slice_sign = encoding_frame.axis_and_sign("partition")
    package_duration = None
    actual_repetition_time = None
    spoiler_end_times = []
    acquisition_start_times = []
    acquisition_intervals = []
    minimum_acquisition_intervals = []
    for repetition in range(repetitions):
        rf, gz = rf_events[repetition if variable_flip_angle else 0]
        repetition_start = (
            0.0 if not sequence.block_events else float(sequence.duration()[0])
        )
        acquisition_start_times.append(repetition_start)
        for slice_index, position in enumerate(slice_positions):
            logical_slice_amplitude = float(gz.amplitude) * slice_sign
            slice_frequency_offset_hz = logical_slice_amplitude * position
            rf.freq_offset = rf_frequency_offset_hz + slice_frequency_offset_hz
            rf.phase_offset = -2 * np.pi * slice_frequency_offset_hz * rf_center
            sequence.add_block(
                rf,
                gz,
                pp.make_label("SLC", "SET", slice_index),
                pp.make_label("REP", "SET", repetition),
            )
            sequence.add_block(gx_pre, gy_pre, gz_rephase)
            if te_delay_value:
                sequence.add_block(pp.make_delay(te_delay_value))
            for line in range(n_y):
                sequence.add_block(gx if line % 2 == 0 else gx_reverse, adc)
                if line < n_y - 1:
                    sequence.add_block(gy_blip)
            sequence.add_block(gx_post, gy_post)
            if spoilers:
                sequence.add_block(*spoilers)
                spoiler_end_times.append(float(sequence.duration()[0]))
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
    rf_slr_sharpness: float = 1.0,
    rf_custom_waveform_hz: Sequence[complex] | None = None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    rf_custom_name: str | None = None,
    rf_frequency_offset_hz: float = 0.0,
    slice_thickness_m: float = 3e-3,
    slice_gap_m: float = 0.0,
    n_slices: int = 1,
    slice_offset_m: float = 0.0,
    echo_time_s: float = 5e-3,
    repetition_time_s: float = 15e-3,
    repetitions: int = 1,
    acquisition_interval_s: float | None = None,
    rf_spoiling_increment_deg: float = 117.0,
    spoiler_cycles_per_slice: float = 4.0,
    spoiler_cycles_per_voxel: float = 0.0,
    spoiler_duration_s: float = 2e-3,
    encoding_duration_s: float = 1e-3,
    encoding_axes: Sequence[str] | EncodingFrame = ("+x", "+y", "+z"),
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
):
    """Build a slice-selective 2D spoiled gradient-echo (FLASH) sequence."""
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
    dwell = (
        round((1.0 / sampling_bandwidth_hz) / system.adc_raster_time)
        * system.adc_raster_time
    )
    if dwell <= 0:
        raise ValueError("sampling bandwidth exceeds the ADC raster capability")

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
        )
    )
    rf, gz = raw_rf_events[0]
    gz = _remap_gradient_event(gz, encoding_frame, "partition")
    rf_center, _ = pp.calc_rf_center(rf)
    rf_block_duration = pp.calc_duration(rf, gz)

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
    logical_slice_area = logical_gradient_area(gz, encoding_frame, "partition")
    gz_rephase = make_role_trapezoid(
        pp,
        encoding_frame,
        "partition",
        area=-logical_slice_area / 2.0,
        duration=encoding_duration_s,
        system=system,
    )

    readout_duration = pp.calc_duration(gx_read, adc)
    echo_without_delay = (
        rf_block_duration
        - (rf.delay + rf_center)
        + encoding_duration_s
        + adc.delay
        + adc_duration / 2.0
    )
    requested_te_delay = echo_time_s - echo_without_delay
    if requested_te_delay < -1e-12:
        raise ValueError(
            f"echo_time_s is too short; minimum is {echo_without_delay:.9g} s"
        )
    te_delay_value = _ceil_to_raster(
        max(0.0, requested_te_delay), system.block_duration_raster
    )
    actual_te = echo_without_delay + te_delay_value
    line_without_tr_delay = (
        rf_block_duration
        + encoding_duration_s
        + te_delay_value
        + readout_duration
        + spoiler_duration_s
    )
    requested_tr_delay = repetition_time_s - line_without_tr_delay
    if requested_tr_delay < -1e-12:
        raise ValueError(
            "repetition_time_s is too short; minimum is "
            f"{line_without_tr_delay:.9g} s"
        )
    tr_delay_value = _ceil_to_raster(
        max(0.0, requested_tr_delay), system.block_duration_raster
    )
    actual_tr = line_without_tr_delay + tr_delay_value

    phase_areas = (np.arange(n_phase) - n_phase // 2) / fov_phase
    slice_positions = float(slice_offset_m) + (
        np.arange(n_slices, dtype=float) - (n_slices - 1) / 2.0
    ) * (slice_thickness_m + slice_gap_m)
    _, slice_sign = encoding_frame.axis_and_sign("partition")
    logical_slice_amplitude = float(gz.amplitude) * slice_sign
    excitation_index = 0
    spoiler_end_times = []
    acquisition_start_times = []
    acquisition_intervals = []
    minimum_acquisition_intervals = []
    for repetition in range(repetitions):
        acquisition_start = _sequence_duration_s(sequence)
        acquisition_start_times.append(acquisition_start)
        for slice_index, position in enumerate(slice_positions):
            slice_frequency_offset_hz = logical_slice_amplitude * position
            for line, phase_area in enumerate(phase_areas):
                rf_phase_deg = (
                    rf_spoiling_increment_deg
                    * excitation_index
                    * (excitation_index + 1)
                    / 2.0
                ) % 360.0
                total_rf_frequency_offset_hz = (
                    rf_frequency_offset_hz + slice_frequency_offset_hz
                )
                rf.freq_offset = total_rf_frequency_offset_hz
                rf.phase_offset = (
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
                    area=(
                        -logical_slice_area / 2.0
                        + spoiler_cycles_per_slice / slice_thickness_m
                    ),
                    duration=spoiler_duration_s,
                    system=system,
                )
                sequence.add_block(
                    rf,
                    gz,
                    pp.make_label("LIN", "SET", line),
                    pp.make_label("SLC", "SET", slice_index),
                    pp.make_label("REP", "SET", repetition),
                )
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
    sequence.set_definition("RFSpoilingIncrementDeg", rf_spoiling_increment_deg)
    sequence.set_definition("SpoilerCyclesPerSlice", spoiler_cycles_per_slice)
    sequence.set_definition("SpoilerCyclesPerVoxel", spoiler_cycles_per_voxel)
    sequence.set_definition("SpoilerDuration", spoiler_duration_s)
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
    rf_slr_sharpness: float = 1.0,
    rf_custom_waveform_hz: Sequence[complex] | None = None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    rf_custom_name: str | None = None,
    rf_frequency_offset_hz: float = 0.0,
    slice_thickness_m: float = 3e-3,
    slice_gap_m: float = 0.0,
    n_slices: int = 1,
    repetitions: int = 1,
    echo_time_s: float = 20e-3,
    repetition_time_s: float | None = 1.0,
    slice_offset_m: float = 0.0,
    encoding_axes: Sequence[str] | EncodingFrame = ("+x", "+y", "+z"),
    spoil_after_slice: bool = True,
    spoiler_cycles_per_slice: float = 8.0,
    spoiler_cycles_per_voxel: float = 0.0,
    spoiler_duration_s: float = 4e-3,
    scanner_parameters: ScannerParameters | Mapping[str, float] | None = None,
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
        )
    )
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
        area=-logical_gradient_area(gz, encoding_frame, "partition") / 2.0,
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
    for repetition in range(repetitions):
        rf, gz = rf_events[repetition if variable_flip_angle else 0]
        repetition_start = (
            0.0 if not sequence.block_events else float(sequence.duration()[0])
        )
        acquisition_start_times.append(repetition_start)
        for slice_index, position in enumerate(slice_positions):
            logical_slice_amplitude = float(gz.amplitude) * slice_sign
            slice_frequency_offset_hz = logical_slice_amplitude * position
            rf.freq_offset = rf_frequency_offset_hz + slice_frequency_offset_hz
            rf.phase_offset = -2.0 * np.pi * slice_frequency_offset_hz * rf_center
            sequence.add_block(
                rf,
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
