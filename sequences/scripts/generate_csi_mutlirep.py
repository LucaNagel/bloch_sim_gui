"""Generate a slice-selective Cartesian 2D chemical-shift-imaging sequence.

Both spatial dimensions are phase encoded.  The ADC therefore samples an FID
without a readout gradient; its dwell time defines the spectral bandwidth.
"""

from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp
from blochsimulator.sequence.rf_pulses import (
    make_pulseq_rf_events,
    set_rf_definitions,
)


PhaseEncodingOrder = Literal["linear", "spiral", "centric"]


def _ceil_to_raster(value: float, raster: float) -> float:
    """Round a non-negative duration up without overshooting exact multiples."""
    return float(np.ceil((value - 1e-12) / raster) * raster)


def _as_2d_fov(fov: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(fov, (int, float)):
        values = (float(fov), float(fov))
    else:
        if len(fov) != 2:
            raise ValueError("fov must be a scalar or a two-element tuple")
        values = tuple(float(value) for value in fov)
    if any(value <= 0 for value in values):
        raise ValueError("all FOV values must be positive")
    return values


def _matrix_size(value: int, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def phase_encoding_indices(
    n_x: int,
    n_y: int,
    order: PhaseEncodingOrder = "linear",
    *,
    fov: float | tuple[float, float] = 160e-3,
) -> list[tuple[int, int]]:
    """Return ``(x_index, y_index)`` pairs in acquisition order.

    ``linear`` traverses complete x-lines starting in a k-space corner.
    ``spiral`` starts at k=0 and follows an outward square spiral. ``centric``
    follows increasing physical distance from k=0; equal-radius samples are
    ordered by polar angle, as in ``calcEquiDistPhaseEncOrder`` from the
    ParaVision reference implementation.
    """
    n_x = _matrix_size(n_x, "n_x")
    n_y = _matrix_size(n_y, "n_y")
    fov_x, fov_y = _as_2d_fov(fov)
    if order not in {"linear", "spiral", "centric"}:
        raise ValueError("phase_encoding_order must be linear, spiral, or centric")

    if order == "linear":
        return [(x, y) for y in range(n_y) for x in range(n_x)]

    center_x, center_y = n_x // 2, n_y // 2
    if order == "spiral":
        result = [(center_x, center_y)]
        x, y = center_x, center_y
        step_length = 1
        # Right, up, left, down gives contiguous square shells around k=0.
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

    indices = [(x, y) for y in range(n_y) for x in range(n_x)]

    def radius_and_angle(index: tuple[int, int]) -> tuple[float, float]:
        x, y = index
        k_x = (x - center_x) / fov_x
        k_y = (y - center_y) / fov_y
        radius = np.hypot(k_x, k_y)
        angle = (np.arctan2(k_y, k_x) + np.pi) % (2 * np.pi)
        return float(radius), float(angle)

    # Match the reference's distance shells and rotate each angularly sorted
    # shell to continue near the final angle of the preceding shell.
    radius_sorted = sorted(indices, key=lambda index: radius_and_angle(index)[0])
    result: list[tuple[int, int]] = []
    last_angle = 0.0
    group_start = 0
    while group_start < len(radius_sorted):
        radius = radius_and_angle(radius_sorted[group_start])[0]
        group_end = group_start + 1
        while group_end < len(radius_sorted) and np.isclose(
            radius_and_angle(radius_sorted[group_end])[0],
            radius,
            rtol=1e-12,
            atol=1e-12,
        ):
            group_end += 1
        shell = sorted(
            radius_sorted[group_start:group_end],
            key=lambda index: radius_and_angle(index)[1],
        )
        angles = [radius_and_angle(index)[1] for index in shell]
        split = next(
            (index for index, angle in enumerate(angles) if angle >= last_angle), 0
        )
        shell = shell[split:] + shell[:split]
        result.extend(shell)
        last_angle = radius_and_angle(shell[-1])[1]
        group_start = group_end
    return result


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "csi_2d.seq",
    *,
    fov: float | tuple[float, float] = (210e-3, 210e-3),
    n_x: int = 16,
    n_y: int = 16,
    slice_thickness: float = 10e-3,
    spectral_bandwidth_hz: float = 4000.0,
    n_spectral_points: int = 1024,
    spectral_resolution_hz: float | None = None,
    phase_encoding_order: PhaseEncodingOrder = "linear",
    flip_angle_deg: float = 15.0,
    rf_pulse_type: str = "sinc",
    rf_duration: float = 3e-3,
    rf_time_bandwidth_product: float = 4.0,
    rf_apodization: float = 0.5,
    rf_slr_sharpness: float = 1.0,
    rf_custom_waveform_hz=None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    encoding_duration: float = 2e-3,
    te: float = 5e-3,
    tr: float = 1.0,
    n_repetitions: int = 1,
    spoil_after_readout: bool = True,
    spoiler_duration: float = 2e-3,
    spoiler_cycles: float = 4.0,
    spoiler_cycles_per_voxel: float = 0.0,
):
    """Create a 2D CSI sequence using SI units.

    ``spectral_bandwidth_hz`` and ``n_spectral_points`` determine ADC dwell and
    FID duration.  Alternatively, set ``spectral_resolution_hz``; then the
    point count is chosen as ``ceil(bandwidth / resolution)``.  The exact
    hardware-raster bandwidth and resulting resolution are stored in the
    sequence definitions. ``n_repetitions`` repeats the complete spatial CSI
    encoding grid and writes its zero-based index to the Pulseq ``REP`` label.
    The optional spoiler is played after every FID. Its through-slice moment is
    given by ``spoiler_cycles``; an additional x/y moment can be requested in
    cycles across one CSI voxel.
    """
    fov_x, fov_y = _as_2d_fov(fov)
    n_x = _matrix_size(n_x, "n_x")
    n_y = _matrix_size(n_y, "n_y")
    n_repetitions = _matrix_size(n_repetitions, "n_repetitions")
    if slice_thickness <= 0:
        raise ValueError("slice_thickness must be positive")
    if spectral_bandwidth_hz <= 0:
        raise ValueError("spectral_bandwidth_hz must be positive")
    if spectral_resolution_hz is not None:
        if spectral_resolution_hz <= 0:
            raise ValueError("spectral_resolution_hz must be positive")
    else:
        n_spectral_points = _matrix_size(n_spectral_points, "n_spectral_points")
    if flip_angle_deg <= 0 or rf_duration <= 0 or encoding_duration <= 0:
        raise ValueError("flip angle and event durations must be positive")
    if te <= 0 or tr <= 0:
        raise ValueError("TE and TR must be positive")
    if not np.isfinite(spoiler_duration) or spoiler_duration <= 0:
        raise ValueError("spoiler_duration must be positive and finite")
    if not np.isfinite(spoiler_cycles) or spoiler_cycles < 0:
        raise ValueError("spoiler_cycles must be non-negative and finite")
    if not np.isfinite(spoiler_cycles_per_voxel) or spoiler_cycles_per_voxel < 0:
        raise ValueError("spoiler_cycles_per_voxel must be non-negative and finite")

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
    seq = pp.Sequence(system)

    requested_dwell = 1.0 / spectral_bandwidth_hz
    adc_dwell = round(requested_dwell / system.adc_raster_time) * system.adc_raster_time
    if adc_dwell <= 0:
        raise ValueError("spectral bandwidth exceeds the ADC raster capability")
    actual_bandwidth = 1.0 / adc_dwell
    if spectral_resolution_hz is not None:
        n_spectral_points = int(np.ceil(actual_bandwidth / spectral_resolution_hz))
    actual_resolution = actual_bandwidth / n_spectral_points

    rf_events, actual_rf_duration, effective_rf_tbw, rf_pulse_type = (
        make_pulseq_rf_events(
            pp,
            system,
            flip_angles_deg=(flip_angle_deg,),
            pulse_type=rf_pulse_type,
            duration_s=rf_duration,
            time_bandwidth_product=rf_time_bandwidth_product,
            apodization=rf_apodization,
            slr_sharpness=rf_slr_sharpness,
            custom_waveform_hz=rf_custom_waveform_hz,
            custom_raster_s=rf_custom_raster_s,
            custom_flip_angle_deg=rf_custom_flip_angle_deg,
            slice_thickness_m=slice_thickness,
        )
    )
    rf, gz = rf_events[0]
    adc = pp.make_adc(
        num_samples=n_spectral_points,
        dwell=adc_dwell,
        delay=system.adc_dead_time,
        system=system,
    )
    spoiler_events = []
    if spoil_after_readout:
        if spoiler_cycles_per_voxel > 0:
            spoiler_events.extend(
                pp.make_trapezoid(
                    channel=axis,
                    area=spoiler_cycles_per_voxel / voxel_size,
                    duration=spoiler_duration,
                    system=system,
                )
                for axis, voxel_size in zip("xy", (fov_x / n_x, fov_y / n_y))
            )
        if spoiler_cycles > 0:
            spoiler_events.append(
                pp.make_trapezoid(
                    channel="z",
                    area=spoiler_cycles / slice_thickness,
                    duration=spoiler_duration,
                    system=system,
                )
            )
    spoiler_block_duration = (
        max(pp.calc_duration(event) for event in spoiler_events)
        if spoiler_events
        else 0.0
    )

    rf_center, _ = pp.calc_rf_center(rf)
    rf_center_from_start = rf.delay + rf_center
    rf_block_duration = pp.calc_duration(rf, gz)
    first_sample_from_adc_start = adc.delay + adc.dwell / 2
    te_delay_value = (
        te
        - (rf_block_duration - rf_center_from_start)
        - encoding_duration
        - first_sample_from_adc_start
    )
    raster = system.block_duration_raster
    if te_delay_value < 0:
        minimum_te = (
            rf_block_duration
            - rf_center_from_start
            + encoding_duration
            + first_sample_from_adc_start
        )
        raise ValueError(f"TE is too short; minimum TE is {minimum_te * 1e3:.3f} ms")
    te_delay_value = _ceil_to_raster(te_delay_value, raster)
    te_delay = pp.make_delay(te_delay_value) if te_delay_value else None

    adc_block_duration = pp.calc_duration(adc)
    repetition_without_tr_delay = (
        rf_block_duration
        + encoding_duration
        + te_delay_value
        + adc_block_duration
        + spoiler_block_duration
    )
    tr_delay_value = tr - repetition_without_tr_delay
    if tr_delay_value < 0:
        raise ValueError(
            "TR is too short; minimum TR is "
            f"{repetition_without_tr_delay * 1e3:.3f} ms"
        )
    tr_delay_value = _ceil_to_raster(tr_delay_value, raster)
    tr_delay = pp.make_delay(tr_delay_value) if tr_delay_value else None
    actual_tr = max(tr, repetition_without_tr_delay + tr_delay_value)
    actual_te = max(
        te,
        rf_block_duration
        - rf_center_from_start
        + encoding_duration
        + te_delay_value
        + first_sample_from_adc_start,
    )

    x_areas = (np.arange(n_x) - n_x // 2) / fov_x
    y_areas = (np.arange(n_y) - n_y // 2) / fov_y
    order = phase_encoding_indices(n_x, n_y, phase_encoding_order, fov=(fov_x, fov_y))
    spoiler_end_times = []
    for repetition in range(n_repetitions):
        for x_index, y_index in order:
            gx_phase = pp.make_trapezoid(
                "x",
                area=float(x_areas[x_index]),
                duration=encoding_duration,
                system=system,
            )
            gy_phase = pp.make_trapezoid(
                "y",
                area=float(y_areas[y_index]),
                duration=encoding_duration,
                system=system,
            )
            gz_rephase = pp.make_trapezoid(
                "z", area=-gz.area / 2, duration=encoding_duration, system=system
            )
            seq.add_block(rf, gz)
            seq.add_block(gx_phase, gy_phase, gz_rephase)
            if te_delay is not None:
                seq.add_block(te_delay)
            seq.add_block(
                adc,
                pp.make_label("LIN", "SET", x_index),
                pp.make_label("PAR", "SET", y_index),
                pp.make_label("REP", "SET", repetition),
            )
            if spoiler_events:
                seq.add_block(*spoiler_events)
                spoiler_end_times.append(float(seq.duration()[0]))
            if tr_delay is not None:
                seq.add_block(tr_delay)

    ok, error_report = seq.check_timing()
    if not ok:
        details = "\n".join(str(error) for error in error_report)
        raise RuntimeError(f"CSI sequence timing check failed:\n{details}")

    print(
        f"CSI: {n_x} x {n_y}, FOV: {fov_x * 1e3:.1f} x {fov_y * 1e3:.1f} mm, "
        f"{n_spectral_points} spectral points, "
        f"bandwidth = {actual_bandwidth:.6g} Hz, "
        f"resolution = {actual_resolution:.6g} Hz/point"
        f"Using {phase_encoding_order} phase encoding, "
        f"repetitions = {n_repetitions}, "
        f"TR = {actual_tr * 1e3:.3f} ms, TE = {actual_te * 1e3:.3f} ms, "
        f"spoiler after readout = {spoil_after_readout}, "
    )
    print(f"TR = {actual_tr * 1e3:.3f} ms, TE = {actual_te * 1e3:.3f} ms")

    seq.set_definition("Name", "csi_2d")
    seq.set_definition("FOV", [fov_x, fov_y, slice_thickness])
    seq.set_definition("MatrixSize", [n_x, n_y, n_spectral_points])
    seq.set_definition("SpectralBandwidth", actual_bandwidth)
    seq.set_definition("SpectralPoints", n_spectral_points)
    seq.set_definition("SpectralResolution", actual_resolution)
    seq.set_definition("PhaseEncodingOrder", phase_encoding_order)
    seq.set_definition("TR", actual_tr)
    seq.set_definition("TE", actual_te)
    seq.set_definition("Repetitions", n_repetitions)
    seq.set_definition("SpoilAfterReadout", bool(spoil_after_readout))
    seq.set_definition("SpoilerCyclesPerSlice", spoiler_cycles)
    seq.set_definition("SpoilerCyclesPerVoxel", spoiler_cycles_per_voxel)
    seq.set_definition("SpoilerDuration", spoiler_duration)
    seq.set_definition(
        "SpoilerAxes",
        "".join(event.channel for event in spoiler_events) or "none",
    )
    seq.set_definition("SpoilerEndTimes", spoiler_end_times)
    seq.set_definition("IdealSpoilerEndTimes", spoiler_end_times)
    set_rf_definitions(
        seq,
        pulse_type=rf_pulse_type,
        requested_duration_s=rf_duration,
        actual_duration_s=actual_rf_duration,
        time_bandwidth_product=effective_rf_tbw,
        apodization=rf_apodization,
        slr_sharpness=rf_slr_sharpness,
        custom_name=None,
        custom_flip_angle_deg=rf_custom_flip_angle_deg,
        frequency_offset_hz=0.0,
    )

    if test_report:
        print(seq.test_report())
    if plot:
        seq.plot(time_range=(0, min(actual_tr * 2, seq.duration()[0])))
        plt.show()
    if write_seq:
        if seq_filename is not None:
            seq_filename = Path(seq_filename)
            seq_filename = seq_filename.with_name(
                f"{seq_filename.stem}_{phase_encoding_order}.seq"
            )

        output_path = (
            Path(__file__).resolve().parent.parent / "sequences" / seq_filename
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seq.write(str(output_path), v141_compat=True)
        print(f"Sequence written to {output_path}")

    return seq


if __name__ == "__main__":
    main(
        write_seq=True,
        seq_filename="csi_2d_linear.seq",
        phase_encoding_order="centric",
        plot=False,
        n_spectral_points=128,
        flip_angle_deg=8.0,
        n_x=12,
        n_y=12,
        fov=(21e-3, 21e-3),
        tr=0.1,
        n_repetitions=10,
        spoil_after_readout=True,
    )
