"""Configurable Pulseq sequence builders used by the desktop simulator."""

from __future__ import annotations

from typing import Sequence

import numpy as np


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
    rf_duration_s: float = 3e-3,
    encoding_duration_s: float = 2e-3,
    echo_time_s: float = 6e-3,
    repetition_time_s: float = 0.1,
    spoil_after_readout: bool = True,
    spoiler_cycles_per_slice: float = 4.0,
    spoiler_cycles_per_voxel: float = 0.0,
    spoiler_duration_s: float = 2e-3,
):
    """Build a slice-selective Cartesian 2D CSI Pulseq sequence."""
    pp = _pypulseq()
    fov_x, fov_y = _positive_values(fov_m, "FOV")
    if len(tuple(fov_m)) != 2:
        raise ValueError("fov_m must contain two values")
    n_x, n_y = (_positive_integer(value, "matrix size") for value in matrix)
    if len(tuple(matrix)) != 2:
        raise ValueError("matrix must contain two values")
    spectral_points = _positive_integer(spectral_points, "spectral_points")
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

    order = _phase_encoding_indices(n_x, n_y, phase_encoding_order, (fov_x, fov_y))
    system = pp.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        grad_raster_time=10e-6,
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
        adc_dead_time=20e-6,
    )
    sequence = pp.Sequence(system)
    requested_dwell = 1.0 / float(spectral_bandwidth_hz)
    dwell = round(requested_dwell / system.adc_raster_time) * system.adc_raster_time
    if dwell <= 0:
        raise ValueError("spectral bandwidth exceeds the ADC raster capability")
    actual_bandwidth = 1.0 / dwell

    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=rf_duration_s,
        slice_thickness=slice_thickness_m,
        apodization=0.5,
        time_bw_product=4.0,
        delay=system.rf_dead_time,
        system=system,
        return_gz=True,
        use="excitation",
    )
    adc = pp.make_adc(
        num_samples=spectral_points,
        dwell=dwell,
        delay=system.adc_dead_time,
        system=system,
    )
    spoiler_events = []
    if spoil_after_readout:
        if spoiler_cycles_per_voxel > 0:
            for axis, voxel_size in zip("xy", (fov_x / n_x, fov_y / n_y)):
                spoiler_events.append(
                    pp.make_trapezoid(
                        channel=axis,
                        area=spoiler_cycles_per_voxel / voxel_size,
                        duration=spoiler_duration_s,
                        system=system,
                    )
                )
        if spoiler_cycles_per_slice > 0:
            spoiler_events.append(
                pp.make_trapezoid(
                    channel="z",
                    area=spoiler_cycles_per_slice / slice_thickness_m,
                    duration=spoiler_duration_s,
                    system=system,
                )
            )

    rf_center, _ = pp.calc_rf_center(rf)
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
    for acquisition_index, (x_index, y_index) in enumerate(order):
        gx_phase = pp.make_trapezoid(
            "x",
            area=float(x_areas[x_index]),
            duration=encoding_duration_s,
            system=system,
        )
        gy_phase = pp.make_trapezoid(
            "y",
            area=float(y_areas[y_index]),
            duration=encoding_duration_s,
            system=system,
        )
        gz_rephase = pp.make_trapezoid(
            "z", area=-gz.area / 2, duration=encoding_duration_s, system=system
        )
        sequence.add_block(rf, gz)
        sequence.add_block(gx_phase, gy_phase, gz_rephase)
        if te_delay_value:
            sequence.add_block(pp.make_delay(te_delay_value))
        sequence.add_block(
            adc,
            pp.make_label("LIN", "SET", x_index),
            pp.make_label("PAR", "SET", y_index),
            pp.make_label("REP", "SET", acquisition_index),
        )
        if spoiler_events:
            sequence.add_block(*spoiler_events)
            spoiler_end_times.append(float(sequence.duration()[0]))
        if tr_delay_value:
            sequence.add_block(pp.make_delay(tr_delay_value))

    _raise_for_timing_errors(sequence, "CSI")
    sequence.set_definition("Name", "csi_2d")
    sequence.set_definition("FOV", [fov_x, fov_y, slice_thickness_m])
    sequence.set_definition("MatrixSize", [n_x, n_y, spectral_points])
    sequence.set_definition("SpectralBandwidth", actual_bandwidth)
    sequence.set_definition("SpectralPoints", spectral_points)
    sequence.set_definition("SpectralResolution", actual_bandwidth / spectral_points)
    sequence.set_definition("PhaseEncodingOrder", phase_encoding_order)
    sequence.set_definition("FlipAngleDeg", float(flip_angle_deg))
    sequence.set_definition("TR", actual_tr)
    sequence.set_definition("TE", actual_te)
    sequence.set_definition("SpoilAfterReadout", bool(spoil_after_readout))
    sequence.set_definition("SpoilerCyclesPerSlice", spoiler_cycles_per_slice)
    sequence.set_definition("SpoilerCyclesPerVoxel", spoiler_cycles_per_voxel)
    sequence.set_definition("SpoilerDuration", spoiler_duration_s)
    sequence.set_definition(
        "SpoilerAxes", "".join(event.channel for event in spoiler_events) or "none"
    )
    sequence.set_definition("SpoilerEndTimes", spoiler_end_times)
    return sequence


def make_pulseq_bssfp(
    *,
    fov_m: Sequence[float] = (0.22, 0.22, 0.16),
    matrix: Sequence[int] = (8, 8, 4),
    flip_angle_deg: float = 15.0,
    rf_duration_s: float = 1e-3,
    sampling_bandwidth_hz: float = 10_000.0,
    encoding_duration_s: float = 1e-3,
    repetition_time_s: float | None = 10e-3,
    rf_phase_start_deg: float = 180.0,
    rf_phase_increment_deg: float = 180.0,
    dummy_repetitions: int = 1,
    repetitions: int = 1,
    use_alpha_half: bool = True,
):
    """Build a fully balanced, non-selective Cartesian 3D bSSFP sequence."""
    pp = _pypulseq()
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

    system = pp.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=20e-6,
    )
    sequence = pp.Sequence(system)
    requested_dwell = 1.0 / float(sampling_bandwidth_hz)
    dwell = round(requested_dwell / system.adc_raster_time) * system.adc_raster_time
    if dwell <= 0:
        raise ValueError("sampling bandwidth exceeds the ADC raster capability")

    rf = pp.make_block_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=rf_duration_s,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )
    rf_alpha_half = pp.make_block_pulse(
        flip_angle=np.deg2rad(flip_angle_deg / 2),
        duration=rf_duration_s,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )
    readout_duration = n_read * dwell
    readout_amplitude = 1.0 / (fov_x * dwell)
    readout_rise_time = max(
        system.adc_dead_time,
        np.ceil(abs(readout_amplitude) / system.max_slew / system.grad_raster_time)
        * system.grad_raster_time,
    )
    gx = pp.make_trapezoid(
        "x",
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
    gx_pre = pp.make_trapezoid(
        "x", area=-gx.area / 2, duration=encoding_duration_s, system=system
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
        preparation_delay = actual_tr / 2 - pp.calc_duration(rf_alpha_half)
        preparation_delay = round(preparation_delay / raster) * raster
        if preparation_delay < 0:
            raise ValueError("TR is too short for alpha/2 preparation")
        sequence.add_block(rf_alpha_half)
        if preparation_delay:
            sequence.add_block(pp.make_delay(preparation_delay))

    ky_areas = (np.arange(n_phase) - n_phase // 2) / fov_y
    kz_areas = (np.arange(n_partition) - n_partition // 2) / fov_z
    rf_phase = float(rf_phase_start_deg)

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
        rf.phase_offset = np.deg2rad(rf_phase)
        adc.phase_offset = np.deg2rad(rf_phase)
        rf_phase = np.mod(rf_phase + rf_phase_increment_deg, 360.0)
        gy_pre = pp.make_trapezoid(
            "y", area=ky, duration=encoding_duration_s, system=system
        )
        gy_rephase = pp.make_trapezoid(
            "y", area=-ky, duration=encoding_duration_s, system=system
        )
        gz_pre = pp.make_trapezoid(
            "z", area=kz, duration=encoding_duration_s, system=system
        )
        gz_rephase = pp.make_trapezoid(
            "z", area=-kz, duration=encoding_duration_s, system=system
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
    for volume_index in range(repetitions):
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

    _raise_for_timing_errors(sequence, "bSSFP")
    sequence.set_definition("Name", "bssfp_3d")
    sequence.set_definition("FOV", [fov_x, fov_y, fov_z])
    sequence.set_definition("MatrixSize", [n_read, n_phase, n_partition])
    sequence.set_definition("FlipAngleDeg", float(flip_angle_deg))
    sequence.set_definition("SamplingBandwidth", 1.0 / dwell)
    sequence.set_definition("TR", actual_tr)
    sequence.set_definition("TE", actual_tr / 2)
    sequence.set_definition("RFPhaseStartDeg", float(rf_phase_start_deg))
    sequence.set_definition("RFPhaseIncrementDeg", float(rf_phase_increment_deg))
    sequence.set_definition("DummyRepetitions", dummy_repetitions)
    sequence.set_definition("Repetitions", repetitions)
    sequence.set_definition("UseAlphaHalf", bool(use_alpha_half))
    return sequence


def make_pulseq_epi(
    *,
    fov_m: Sequence[float] = (0.22, 0.22),
    matrix: Sequence[int] = (16, 16),
    sampling_bandwidth_hz: float = 50_000.0,
    flip_angle_deg: float = 90.0,
    slice_thickness_m: float = 3e-3,
    n_slices: int = 1,
    repetitions: int = 1,
    repetition_time_s: float | None = 1.0,
    spoil_after_slice: bool = True,
    spoiler_cycles_per_slice: float = 8.0,
    spoiler_cycles_per_voxel: float = 0.0,
    spoiler_duration_s: float = 4e-3,
):
    """Build a Cartesian single-shot EPI Pulseq sequence for export."""
    pp = _pypulseq()
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
        "spoiler_duration_s": spoiler_duration_s,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value in {
        "spoiler_cycles_per_slice": spoiler_cycles_per_slice,
        "spoiler_cycles_per_voxel": spoiler_cycles_per_voxel,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")

    system = pp.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
        adc_dead_time=20e-6,
    )
    sequence = pp.Sequence(system)
    dwell = (
        round((1.0 / sampling_bandwidth_hz) / system.adc_raster_time)
        * system.adc_raster_time
    )
    if dwell <= 0:
        raise ValueError("sampling bandwidth exceeds the ADC raster capability")
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=3e-3,
        slice_thickness=slice_thickness_m,
        apodization=0.5,
        time_bw_product=4.0,
        return_gz=True,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )
    delta_kx, delta_ky = 1.0 / fov_x, 1.0 / fov_y
    adc_duration = n_x * dwell
    flat_time = _ceil_to_raster(adc_duration, system.grad_raster_time)
    gx = pp.make_trapezoid(
        "x",
        amplitude=(n_x * delta_kx) / adc_duration,
        flat_time=flat_time,
        system=system,
    )
    gx_reverse = pp.make_trapezoid(
        "x",
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
    gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=pre_time, system=system)
    gy_pre = pp.make_trapezoid(
        "y", area=-n_y / 2 * delta_ky, duration=pre_time, system=system
    )
    gz_rephase = pp.make_trapezoid(
        "z", area=-gz.area / 2, duration=pre_time, system=system
    )
    relative_x_end = -gx.area / 2 + (gx.area if n_y % 2 else 0.0)
    relative_y_end = (-n_y / 2 + max(n_y - 1, 0)) * delta_ky
    gx_post = pp.make_trapezoid(
        "x", area=-relative_x_end, duration=pre_time, system=system
    )
    gy_post = pp.make_trapezoid(
        "y", area=-relative_y_end, duration=pre_time, system=system
    )
    gy_blip = pp.make_trapezoid("y", area=delta_ky, system=system)
    spoilers = []
    if spoil_after_slice:
        if spoiler_cycles_per_voxel > 0:
            for axis, voxel_size in zip("xy", (fov_x / n_x, fov_y / n_y)):
                spoilers.append(
                    pp.make_trapezoid(
                        axis,
                        area=spoiler_cycles_per_voxel / voxel_size,
                        duration=spoiler_duration_s,
                        system=system,
                    )
                )
        if spoiler_cycles_per_slice > 0:
            spoilers.append(
                pp.make_trapezoid(
                    "z",
                    area=spoiler_cycles_per_slice / slice_thickness_m,
                    duration=spoiler_duration_s,
                    system=system,
                )
            )

    slice_positions = (
        np.arange(n_slices, dtype=float) - (n_slices - 1) / 2
    ) * slice_thickness_m
    rf_center, _ = pp.calc_rf_center(rf)
    package_duration = None
    actual_repetition_time = None
    spoiler_end_times = []
    for repetition in range(repetitions):
        repetition_start = (
            0.0 if not sequence.block_events else float(sequence.duration()[0])
        )
        for slice_index, position in enumerate(slice_positions):
            rf.freq_offset = gz.amplitude * position
            rf.phase_offset = -2 * np.pi * rf.freq_offset * rf_center
            sequence.add_block(
                rf,
                gz,
                pp.make_label("SLC", "SET", slice_index),
                pp.make_label("REP", "SET", repetition),
            )
            sequence.add_block(gx_pre, gy_pre, gz_rephase)
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

    _raise_for_timing_errors(sequence, "EPI")
    sequence.set_definition("Name", "epi_2d")
    sequence.set_definition("FOV", [fov_x, fov_y, n_slices * slice_thickness_m])
    sequence.set_definition("MatrixSize", [n_x, n_y])
    sequence.set_definition("SamplingBandwidth", 1.0 / dwell)
    sequence.set_definition("FlipAngleDeg", float(flip_angle_deg))
    sequence.set_definition("SliceThickness", slice_thickness_m)
    sequence.set_definition("Repetitions", repetitions)
    sequence.set_definition("RepetitionTime", actual_repetition_time)
    sequence.set_definition("MinimumRepetitionTime", package_duration)
    sequence.set_definition("SpoilAfterSlice", bool(spoil_after_slice))
    sequence.set_definition("SpoilerCyclesPerSlice", spoiler_cycles_per_slice)
    sequence.set_definition("SpoilerCyclesPerVoxel", spoiler_cycles_per_voxel)
    sequence.set_definition("SpoilerDuration", spoiler_duration_s)
    sequence.set_definition(
        "SpoilerAxes", "".join(event.channel for event in spoilers) or "none"
    )
    sequence.set_definition("SpoilerEndTimes", spoiler_end_times)
    return sequence


def _raise_for_timing_errors(sequence, name: str) -> None:
    ok, errors = sequence.check_timing()
    if not ok:
        details = "\n".join(str(error) for error in errors)
        raise RuntimeError(f"{name} sequence timing check failed:\n{details}")
