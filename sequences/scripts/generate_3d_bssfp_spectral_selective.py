"""Generate a spectrally selective Cartesian 3D balanced-SSFP sequence.

Each dynamic frame uses a narrow-band RF pulse and receiver demodulation at one
target frequency offset.  This is useful for hyperpolarized 13C simulations
where separate pyruvate/lactate images are acquired by alternating the RF
carrier.  The read, phase, and partition gradients are balanced within every
TR, so the sequence remains a true 3D bSSFP readout rather than a slice-select
2D acquisition.  A separate end-of-image spoiler is applied after each 3D
volume; it is deliberately outside the balanced readout train.

Offsets are specified in Hz relative to the sequence centre frequency.  For
example, pass two offsets for pyruvate and lactate and set ``n_repetition=2``
to acquire one 3D volume for each target.
"""

import copy
from pathlib import Path
import warnings

import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp

from blochsimulator.sequence.encoding import (
    EncodingFrame,
    logical_gradient_area,
    make_role_trapezoid,
    resolve_encoding_frame,
    set_pulseq_encoding_definitions,
)
from blochsimulator.sequence.bssfp_phase import (
    advance_bssfp_phase_deg,
    pulseq_phase_offset_rad,
    wrap_phase_deg,
)


def _as_3d_fov(fov: float | tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(fov, (int, float)):
        values = (float(fov),) * 3
    else:
        if len(fov) != 3:
            raise ValueError("fov must be a scalar or a three-element tuple")
        values = tuple(float(value) for value in fov)
    if any(value <= 0 or not np.isfinite(value) for value in values):
        raise ValueError("all FOV values must be positive and finite")
    return values


def _encoding_areas(matrix_size: int, delta_k: float) -> np.ndarray:
    """Return Cartesian encoding moments with a sample at k=0."""
    if not isinstance(matrix_size, (int, np.integer)) or matrix_size <= 0:
        raise ValueError("matrix sizes must be positive integers")
    return (np.arange(matrix_size) - matrix_size // 2) * delta_k


def _target_offsets(values: tuple[float, ...]) -> tuple[float, ...]:
    offsets = tuple(float(value) for value in values)
    if not offsets or not np.all(np.isfinite(offsets)):
        raise ValueError("target_frequency_offsets_hz must contain finite values")
    return offsets


def _receiver_offsets(
    values: tuple[float, ...] | None,
    *,
    n_targets: int,
) -> tuple[float, ...]:
    if values is None:
        return ()
    offsets = tuple(float(value) for value in values)
    if len(offsets) != n_targets or not np.all(np.isfinite(offsets)):
        raise ValueError(
            "receiver_frequency_offsets_hz must contain one finite value per target"
        )
    return offsets


def _target_flip_angles(
    values: float | tuple[float, ...],
    *,
    n_targets: int,
) -> tuple[float, ...]:
    if isinstance(values, (int, float)):
        angles = (float(values),) * n_targets
    else:
        angles = tuple(float(value) for value in values)
        if len(angles) == 1:
            angles = angles * n_targets
    if len(angles) != n_targets:
        raise ValueError(
            "flip_angle_deg must be a scalar, a single-element tuple, or match "
            "target_frequency_offsets_hz"
        )
    if any(value < 0 or not np.isfinite(value) for value in angles):
        raise ValueError("flip_angle_deg values must be non-negative and finite")
    return angles


def _target_names(values: tuple[str, ...], *, n_targets: int) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if len(names) != n_targets or any(not name for name in names):
        raise ValueError(
            "target_metabolite_names must match target_frequency_offsets_hz"
        )
    return names


def _readout_dwell(
    *,
    n_read: int,
    readout_bandwidth_hz: float | None,
    adc_dwell: float | None,
    adc_raster_time: float,
    grad_raster_time: float,
) -> float:
    if readout_bandwidth_hz is not None:
        readout_bandwidth_hz = float(readout_bandwidth_hz)
        if readout_bandwidth_hz <= 0 or not np.isfinite(readout_bandwidth_hz):
            raise ValueError("readout_bandwidth_hz must be positive and finite")
        adc_dwell = 1.0 / readout_bandwidth_hz
    elif adc_dwell is not None:
        adc_dwell = float(adc_dwell)
        if adc_dwell <= 0 or not np.isfinite(adc_dwell):
            raise ValueError("adc_dwell must be positive and finite")
    else:
        raise ValueError("either readout_bandwidth_hz or adc_dwell must be provided")

    dwell_steps = max(1, int(np.round(adc_dwell / adc_raster_time)))
    best_dwell = None
    best_error = np.inf
    for offset in range(0, 10000):
        candidates = (dwell_steps + offset,)
        if offset:
            candidates += (dwell_steps - offset,)
        for candidate_steps in candidates:
            if candidate_steps <= 0:
                continue
            candidate_dwell = candidate_steps * adc_raster_time
            readout_duration = n_read * candidate_dwell
            raster_error = abs(
                readout_duration / grad_raster_time
                - np.round(readout_duration / grad_raster_time)
            )
            if raster_error > 1e-9:
                continue
            error = abs(candidate_dwell - adc_dwell)
            if error < best_error:
                best_dwell = candidate_dwell
                best_error = error
        if best_dwell is not None:
            break
    if best_dwell is None or best_dwell <= 0:
        raise ValueError("readout bandwidth is too high for the ADC raster time")
    return float(best_dwell)


def _default_slr_pulse_path(sharpness: float = 1.0) -> Path:
    if not np.isfinite(sharpness) or sharpness <= 0:
        raise ValueError("spectral_slr_sharpness must be positive and finite")
    rounded = round(float(sharpness))
    if not np.isclose(sharpness, rounded):
        raise ValueError("bundled SLR pulses require integer sharpness values")
    return (
        Path(__file__).resolve().parents[2]
        / "rfpulses"
        / f"SLR_sharpness_{int(rounded)}.txt"
    )


def _load_amp_phase_waveform(path: str | Path) -> np.ndarray:
    data = np.loadtxt(Path(path), delimiter=",")
    flat = np.asarray(data, dtype=float).reshape(-1)
    if flat.size < 2 or flat.size % 2:
        raise ValueError("SLR pulse file must contain amp, phase pairs")

    amplitudes = flat[0::2]
    phases_rad = np.deg2rad(flat[1::2])
    signal = amplitudes * np.exp(1j * phases_rad)
    if not np.any(np.abs(signal) > 0):
        raise ValueError("SLR pulse waveform is empty")
    return signal.astype(np.complex128)


def _resample_waveform_to_raster(
    signal: np.ndarray,
    *,
    duration: float,
    raster: float,
) -> tuple[np.ndarray, float]:
    n_samples = int(np.round(duration / raster))
    if n_samples <= 0:
        raise ValueError("spectral_rf_duration must be at least one RF raster interval")
    actual_duration = n_samples * raster
    if signal.size == n_samples:
        return signal, actual_duration

    source = np.linspace(0.0, 1.0, signal.size, endpoint=True)
    target = np.linspace(0.0, 1.0, n_samples, endpoint=True)
    resampled = np.interp(target, source, signal.real) + 1j * np.interp(
        target, source, signal.imag
    )
    return resampled.astype(np.complex128), actual_duration


def _make_spectral_rf(
    *,
    pulse_type: str,
    flip_angle_rad: float,
    duration: float,
    bandwidth_hz: float,
    apodization: float,
    system,
    slr_pulse_path: str | Path | None,
    slr_sharpness: float,
):
    pulse_type = pulse_type.lower()
    if pulse_type == "gauss":
        return pp.make_gauss_pulse(
            flip_angle=flip_angle_rad,
            duration=duration,
            bandwidth=bandwidth_hz,
            apodization=apodization,
            delay=system.rf_dead_time,
            system=system,
            use="excitation",
        )
    if pulse_type == "sinc":
        return pp.make_sinc_pulse(
            flip_angle=flip_angle_rad,
            duration=duration,
            time_bw_product=duration * bandwidth_hz,
            apodization=apodization,
            delay=system.rf_dead_time,
            system=system,
            use="excitation",
        )
    if pulse_type == "block":
        return pp.make_block_pulse(
            flip_angle=flip_angle_rad,
            duration=duration,
            delay=system.rf_dead_time,
            system=system,
            use="excitation",
        )
    if pulse_type == "slr":
        signal = _load_amp_phase_waveform(
            _default_slr_pulse_path(slr_sharpness)
            if slr_pulse_path is None
            else slr_pulse_path
        )
        signal, _ = _resample_waveform_to_raster(
            signal,
            duration=duration,
            raster=system.rf_raster_time,
        )
        return pp.make_arbitrary_rf(
            signal=signal,
            flip_angle=flip_angle_rad,
            dwell=system.rf_raster_time,
            delay=system.rf_dead_time,
            system=system,
            use="excitation",
        )
    raise ValueError("spectral_pulse_type must be 'slr', 'gauss', 'sinc', or 'block'")


def _zero_amplitude_rf(reference_rf):
    event = copy.deepcopy(reference_rf)
    event.signal = np.zeros_like(event.signal)
    return event


def _add_rf_block(sequence, rf_event, *events):
    if np.any(np.abs(rf_event.signal) > 0):
        sequence.add_block(rf_event, *events)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sequence.add_block(rf_event, *events)


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "bssfp_3d_spectral_selective.seq",
    *,
    fov: float | tuple[float, float, float] = (56e-3, 28e-3, 21e-3),
    n_read: int = 32,
    n_phase: int = 16,
    n_partition: int = 12,
    n_repetition: int = 2,
    target_frequency_offsets_hz: tuple[float, ...] = (1655.0, -245.0),
    receiver_frequency_offsets_hz: tuple[float, ...] | None = (925.44725, 0.0),
    target_metabolite_names: tuple[str, ...] = ("Lac", "Py"),
    flip_angle_deg: float | tuple[float, ...] = (90.0, 4.0),
    spectral_rf_duration: float = 2.33e-3,
    spectral_rf_bandwidth_hz: float | None = None,
    spectral_rf_bandwidth_factor_hz_ms: float = 2100.0,
    spectral_rf_fwhm_hz: float = 900.0,
    spectral_pulse_type: str = "slr",
    spectral_rf_apodization: float = 0.0,
    spectral_slr_pulse_path: str | Path | None = None,
    spectral_slr_sharpness: float = 1.0,
    target_tr: float | None = 6.29e-3,
    readout_bandwidth_hz: float | None = 10e3,
    adc_dwell: float | None = None,
    encoding_duration: float = 0.2e-3,
    rf_phase_start: float = 0.0,
    rf_phase_increment: float = 0.0,
    dummy_repetitions: int = 0,
    use_alpha_half: bool = True,
    alpha_half_center_spacing: float = 4.31e-3,
    end_image_spoiler_cycles_per_fov: float = 4.0,
    end_image_spoiler_cycles_per_voxel: float = 0.0,
    end_image_spoiler_voxel_size_m: tuple[float, float, float] | None = None,
    end_image_spoiler_duration: float = 1.0e-3,
    use_labels: bool = True,
    v141_compat: bool = True,
    max_grad_mtm: float = 100.0,
    max_slew_tms: float = 1000.0,
    field_strength_t: float = 7.0,
    nucleus: str = "C13",
    encoding_axes: tuple[str, str, str] | EncodingFrame = ("+x", "+y", "+z"),
):
    """Create a spectrally selective Cartesian 3D bSSFP sequence.

    Parameters are in SI units. ``target_frequency_offsets_hz`` contains the
    off-resonant RF centre frequencies used for spectral selection.
    ``receiver_frequency_offsets_hz`` contains the ADC demodulation frequencies;
    set it to ``None`` to use the RF centre frequencies for legacy behaviour.
    The offsets are cycled over ``n_repetition`` dynamic volumes.
    ``flip_angle_deg`` may be a scalar or a tuple matching
    ``target_frequency_offsets_hz`` for metabolite-specific nominal flip
    angles. Defaults follow Skinner et al. 2023 for alternating
    lactate/pyruvate 3D bSSFP: TR 6.29 ms, TE TR/2, FOV 56 x 28 x 21 mm3,
    matrix 32 x 16 x 12, 10 kHz readout bandwidth, 2.33 ms SLR RF with
    bandwidth factor 2100 Hz ms, FWHM 900 Hz, and nominal flip angles
    alpha_Lac=90 degrees and alpha_Py=4 degrees. The default receiver offsets
    demodulate lactate/pyruvate at 183.35/171.0 ppm for 7 T 13C. The bundled
    default SLR waveform is loaded from ``rfpulses/SLR_sharpness_1.txt`` unless
    an explicit ``spectral_slr_pulse_path`` is provided.  The default
    ``alpha_half_center_spacing`` reproduces the reported 4.31 ms separation
    between the preparation-pulse and first readout-pulse centres.  The
    end-of-image spoiler can combine cycles across each FOV with cycles across
    each actual simulation voxel. ``end_image_spoiler_voxel_size_m`` is in
    physical scanner X/Y/Z order; reconstructed image voxel sizes are used
    when it is omitted.
    """
    fov_x, fov_y, fov_z = _as_3d_fov(fov)
    encoding_frame = resolve_encoding_frame(encoding_axes)
    _encoding_areas(n_read, 1.0)  # Validate readout matrix size.
    ky_areas = _encoding_areas(n_phase, 1 / fov_y)
    kz_areas = _encoding_areas(n_partition, 1 / fov_z)
    target_offsets = _target_offsets(target_frequency_offsets_hz)
    receiver_offsets = _receiver_offsets(
        receiver_frequency_offsets_hz,
        n_targets=len(target_offsets),
    )
    if not receiver_offsets:
        receiver_offsets = target_offsets
    target_names = _target_names(target_metabolite_names, n_targets=len(target_offsets))
    target_flip_angles = _target_flip_angles(
        flip_angle_deg,
        n_targets=len(target_offsets),
    )

    if not isinstance(n_repetition, (int, np.integer)) or n_repetition <= 0:
        raise ValueError("n_repetition must be a positive integer")
    if spectral_rf_duration <= 0:
        raise ValueError("spectral_rf_duration must be positive")
    if spectral_rf_bandwidth_hz is None:
        spectral_rf_bandwidth_hz = spectral_rf_bandwidth_factor_hz_ms / (
            spectral_rf_duration * 1e3
        )
    if (
        spectral_rf_duration <= 0
        or spectral_rf_bandwidth_hz <= 0
        or spectral_rf_bandwidth_factor_hz_ms <= 0
        or spectral_rf_fwhm_hz <= 0
        or encoding_duration <= 0
    ):
        raise ValueError("event durations and RF bandwidth parameters must be positive")
    if target_tr is not None and (target_tr <= 0 or not np.isfinite(target_tr)):
        raise ValueError("target_tr must be positive and finite")
    if not isinstance(dummy_repetitions, (int, np.integer)) or dummy_repetitions < 0:
        raise ValueError("dummy_repetitions must be a non-negative integer")
    if alpha_half_center_spacing <= 0 or not np.isfinite(alpha_half_center_spacing):
        raise ValueError("alpha_half_center_spacing must be positive and finite")
    if end_image_spoiler_cycles_per_fov < 0 or not np.isfinite(
        end_image_spoiler_cycles_per_fov
    ):
        raise ValueError(
            "end_image_spoiler_cycles_per_fov must be non-negative and finite"
        )
    if end_image_spoiler_cycles_per_voxel < 0 or not np.isfinite(
        end_image_spoiler_cycles_per_voxel
    ):
        raise ValueError(
            "end_image_spoiler_cycles_per_voxel must be non-negative and finite"
        )
    if end_image_spoiler_voxel_size_m is None:
        logical_voxel_sizes = np.asarray(
            (fov_x / n_read, fov_y / n_phase, fov_z / n_partition), dtype=float
        )
        physical_voxel_sizes = np.zeros(3, dtype=float)
        for role, voxel_size in zip(
            ("read", "phase", "partition"), logical_voxel_sizes
        ):
            axis, _ = encoding_frame.axis_and_sign(role)
            physical_voxel_sizes["xyz".index(axis)] = voxel_size
    else:
        physical_voxel_sizes = np.asarray(
            tuple(float(value) for value in end_image_spoiler_voxel_size_m),
            dtype=float,
        )
        if (
            physical_voxel_sizes.shape != (3,)
            or not np.all(np.isfinite(physical_voxel_sizes))
            or np.any(physical_voxel_sizes <= 0)
        ):
            raise ValueError(
                "end_image_spoiler_voxel_size_m must contain positive finite "
                "X, Y, and Z sizes"
            )
    if end_image_spoiler_duration <= 0 or not np.isfinite(end_image_spoiler_duration):
        raise ValueError("end_image_spoiler_duration must be positive and finite")
    if max_grad_mtm <= 0 or max_slew_tms <= 0:
        raise ValueError("gradient limits must be positive")
    if field_strength_t <= 0 or not np.isfinite(field_strength_t):
        raise ValueError("field_strength_t must be positive and finite")
    nucleus = str(nucleus).strip()
    if not nucleus:
        raise ValueError("nucleus must not be empty")

    system = pp.Opts(
        max_grad=max_grad_mtm,
        grad_unit="mT/m",
        max_slew=max_slew_tms,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=20e-6,
    )
    seq = pp.Sequence(system)

    timing_rf = _make_spectral_rf(
        pulse_type=spectral_pulse_type,
        flip_angle_rad=np.deg2rad(1.0),
        duration=spectral_rf_duration,
        bandwidth_hz=spectral_rf_bandwidth_hz,
        apodization=spectral_rf_apodization,
        system=system,
        slr_pulse_path=spectral_slr_pulse_path,
        slr_sharpness=spectral_slr_sharpness,
    )
    rfs = tuple(
        (
            _make_spectral_rf(
                pulse_type=spectral_pulse_type,
                flip_angle_rad=np.deg2rad(flip_angle),
                duration=spectral_rf_duration,
                bandwidth_hz=spectral_rf_bandwidth_hz,
                apodization=spectral_rf_apodization,
                system=system,
                slr_pulse_path=spectral_slr_pulse_path,
                slr_sharpness=spectral_slr_sharpness,
            )
            if flip_angle > 0
            else _zero_amplitude_rf(timing_rf)
        )
        for flip_angle in target_flip_angles
    )
    rf = next((event for event in rfs if event is not None), timing_rf)
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.plot(np.real(rf.signal))
    if readout_bandwidth_hz is None and adc_dwell is None:
        readout_bandwidth_hz = n_read / 3.8e-3
    adc_dwell = _readout_dwell(
        n_read=n_read,
        readout_bandwidth_hz=readout_bandwidth_hz,
        adc_dwell=adc_dwell,
        adc_raster_time=system.adc_raster_time,
        grad_raster_time=system.grad_raster_time,
    )
    readout_duration = n_read * adc_dwell
    readout_amplitude = 1 / (fov_x * adc_dwell)
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
        duration=readout_duration,
        delay=gx.rise_time,
        system=system,
    )
    readout_area = logical_gradient_area(gx, encoding_frame, "read")
    gx_pre = make_role_trapezoid(
        pp,
        encoding_frame,
        "read",
        area=-readout_area / 2,
        duration=encoding_duration,
        system=system,
    )

    rf_center, _ = pp.calc_rf_center(rf)
    rf_center_from_block_start = rf.delay + rf_center
    rf_block_duration = pp.calc_duration(rf)
    read_block_duration = max(pp.calc_duration(gx), pp.calc_duration(adc))
    adc_center_from_block_start = adc.delay + adc.num_samples * adc.dwell / 2
    raster = system.block_duration_raster
    pre_duration = pp.calc_duration(gx_pre)
    if target_tr is None:
        rf_balance_delay_value = (
            2 * rf_center_from_block_start
            + read_block_duration
            - 2 * adc_center_from_block_start
            - rf_block_duration
        )
    else:
        rf_balance_delay_value = (
            target_tr / 2
            + rf_center_from_block_start
            - rf_block_duration
            - pre_duration
            - adc_center_from_block_start
        )
    rf_balance_delay_value = np.round(rf_balance_delay_value / raster) * raster
    if rf_balance_delay_value < 0:
        raise ValueError("RF timing cannot be centered with a non-negative delay")
    rf_balance_delay = (
        pp.make_delay(rf_balance_delay_value) if rf_balance_delay_value > 0 else None
    )

    minimum_tr = (
        rf_block_duration
        + rf_balance_delay_value
        + 2 * pre_duration
        + read_block_duration
    )
    print(f"Minimum TR = {minimum_tr * 1e3:.3f} ms")
    if target_tr is None:
        tr = minimum_tr
        tr_fill_delay_value = 0.0
    else:
        tr = np.round(target_tr / raster) * raster
        tr_fill_delay_value = np.round((tr - minimum_tr) / raster) * raster
        if tr_fill_delay_value < -raster / 2:
            raise ValueError(
                f"target_tr {target_tr * 1e3:.3f} ms is shorter than the minimum "
                f"TR {minimum_tr * 1e3:.3f} ms for these RF/readout settings"
            )
        tr_fill_delay_value = max(0.0, tr_fill_delay_value)
    tr_fill_delay = (
        pp.make_delay(tr_fill_delay_value) if tr_fill_delay_value > 0 else None
    )
    te = tr / 2

    spoiler_role_areas = []
    for role, axis_fov in zip(("read", "phase", "partition"), (fov_x, fov_y, fov_z)):
        axis, _ = encoding_frame.axis_and_sign(role)
        voxel_size = physical_voxel_sizes["xyz".index(axis)]
        spoiler_role_areas.append(
            end_image_spoiler_cycles_per_fov / axis_fov
            + end_image_spoiler_cycles_per_voxel / voxel_size
        )
    end_image_spoilers = ()
    if any(area > 0 for area in spoiler_role_areas):
        end_image_spoilers = tuple(
            make_role_trapezoid(
                pp,
                encoding_frame,
                role,
                area=area,
                duration=end_image_spoiler_duration,
                system=system,
            )
            for role, area in zip(("read", "phase", "partition"), spoiler_role_areas)
        )

    spoiler_end_times = []

    def set_rf_and_adc_offsets(
        rf_event,
        rf_center_value: float,
        common_phase_deg: float,
        target_frequency_hz: float,
        receiver_frequency_hz: float,
    ):
        if rf_event is not None:
            rf_event.freq_offset = target_frequency_hz
            rf_event.phase_offset = pulseq_phase_offset_rad(
                common_phase_deg,
                frequency_offset_hz=target_frequency_hz,
                event_center_s=rf_center_value,
            )
        adc.freq_offset = receiver_frequency_hz
        adc_phase = advance_bssfp_phase_deg(
            common_phase_deg,
            elapsed_s=tr / 2,
            frequency_offset_hz=receiver_frequency_hz,
        )
        adc.phase_offset = pulseq_phase_offset_rad(
            adc_phase,
            frequency_offset_hz=receiver_frequency_hz,
            event_center_s=adc_center_from_block_start,
        )

    for rep in range(n_repetition):
        target_index = rep % len(target_offsets)
        target_frequency_hz = target_offsets[target_index]
        receiver_frequency_hz = receiver_offsets[target_index]
        target_name = target_names[target_index]
        target_flip_angle = target_flip_angles[target_index]
        rf_frame = rfs[target_index]
        rf_frame_timing = rf_frame
        rf_frame_center, _ = pp.calc_rf_center(rf_frame_timing)
        print(
            f"Spectral frame {rep + 1}/{n_repetition}: "
            f"{target_name}, RF offset {target_frequency_hz:.3f} Hz, "
            f"receiver offset {receiver_frequency_hz:.3f} Hz, "
            f"flip {target_flip_angle:.3f} deg"
        )

        if use_alpha_half:
            rf_alpha_half = (
                _make_spectral_rf(
                    pulse_type=spectral_pulse_type,
                    flip_angle_rad=np.deg2rad(target_flip_angle / 2),
                    duration=spectral_rf_duration,
                    bandwidth_hz=spectral_rf_bandwidth_hz,
                    apodization=spectral_rf_apodization,
                    system=system,
                    slr_pulse_path=spectral_slr_pulse_path,
                    slr_sharpness=spectral_slr_sharpness,
                )
                if target_flip_angle > 0
                else _zero_amplitude_rf(rf)
            )
            rf_alpha_timing = rf_alpha_half
            rf_alpha_half_center, _ = pp.calc_rf_center(rf_alpha_timing)
            rf_alpha_half_center_from_block_start = (
                rf_alpha_timing.delay + rf_alpha_half_center
            )
            rf_frame_center_from_block_start = rf_frame_timing.delay + rf_frame_center
            rf_alpha_half.freq_offset = target_frequency_hz
            rf_alpha_half.phase_offset = pulseq_phase_offset_rad(
                wrap_phase_deg(rf_phase_start + rf_phase_increment),
                frequency_offset_hz=target_frequency_hz,
                event_center_s=rf_alpha_half_center,
            )
            alpha_half_delay_value = (
                alpha_half_center_spacing
                - pp.calc_duration(rf_alpha_timing)
                + rf_alpha_half_center_from_block_start
                - rf_frame_center_from_block_start
            )
            alpha_half_delay_value = np.round(alpha_half_delay_value / raster) * raster
            if alpha_half_delay_value < 0:
                raise ValueError(
                    "alpha_half_center_spacing is shorter than the minimum "
                    "non-overlapping RF-pulse center spacing"
                )
            _add_rf_block(seq, rf_alpha_half)
            if alpha_half_delay_value > 0:
                seq.add_block(pp.make_delay(alpha_half_delay_value))

        # Pulseq frequency offsets are local to each event. Continue the target
        # RF oscillator explicitly and use that phase to lock the receiver, so
        # the off-resonant pulse train is physical without writing the RF/RX
        # carrier difference into successive Cartesian lines.
        common_phase = wrap_phase_deg(rf_phase_start)
        if use_alpha_half:
            common_phase = advance_bssfp_phase_deg(
                common_phase,
                elapsed_s=alpha_half_center_spacing,
                frequency_offset_hz=target_frequency_hz,
            )

        def add_repetition(
            ky: float,
            kz: float,
            acquire: bool,
            partition_index: int | None = None,
        ) -> None:
            nonlocal common_phase

            set_rf_and_adc_offsets(
                rf_frame,
                rf_frame_center,
                common_phase,
                target_frequency_hz,
                receiver_frequency_hz,
            )
            common_phase = advance_bssfp_phase_deg(
                common_phase,
                elapsed_s=tr,
                frequency_offset_hz=target_frequency_hz,
                phase_increment_deg=rf_phase_increment,
            )

            gy_pre = make_role_trapezoid(
                pp,
                encoding_frame,
                "phase",
                area=ky,
                duration=encoding_duration,
                system=system,
            )
            gy_reph = make_role_trapezoid(
                pp,
                encoding_frame,
                "phase",
                area=-ky,
                duration=encoding_duration,
                system=system,
            )
            gz_pre = make_role_trapezoid(
                pp,
                encoding_frame,
                "partition",
                area=kz,
                duration=encoding_duration,
                system=system,
            )
            gz_reph = make_role_trapezoid(
                pp,
                encoding_frame,
                "partition",
                area=-kz,
                duration=encoding_duration,
                system=system,
            )

            _add_rf_block(seq, rf_frame)
            if rf_balance_delay is not None:
                seq.add_block(rf_balance_delay)
            seq.add_block(gx_pre, gy_pre, gz_pre)
            if acquire:
                if partition_index is None:
                    raise ValueError("acquired repetitions require a partition index")
                if use_labels:
                    partition_label = pp.make_label("PAR", "SET", partition_index)
                    repetition_label = pp.make_label("REP", "SET", rep)
                    seq.add_block(gx, adc, partition_label, repetition_label)
                else:
                    seq.add_block(gx, adc)
            else:
                seq.add_block(gx, pp.make_delay(read_block_duration))
            seq.add_block(gx_pre, gy_reph, gz_reph)
            if tr_fill_delay is not None:
                seq.add_block(tr_fill_delay)

        for _ in range(dummy_repetitions):
            add_repetition(ky=0.0, kz=0.0, acquire=False)

        for partition_index, kz in enumerate(kz_areas):
            for ky in ky_areas:
                add_repetition(
                    ky=float(ky),
                    kz=float(kz),
                    acquire=True,
                    partition_index=partition_index,
                )

        if end_image_spoilers:
            seq.add_block(*end_image_spoilers)
            spoiler_end_times.append(float(seq.duration()[0]))

    ok, error_report = seq.check_timing()
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed. Error listing follows:")
        for error in error_report:
            print(error)

    print(f"TR = {tr * 1e3:.3f} ms, TE = {te * 1e3:.3f} ms")
    print("Single-metabolite acquisition time = " f"{n_phase * n_partition * tr:.3f} s")
    print(
        f"Spectral RF: {spectral_pulse_type}, duration "
        f"{spectral_rf_duration * 1e3:.3f} ms, bandwidth "
        f"{spectral_rf_bandwidth_hz:.3f} Hz"
    )
    print(f"Readout bandwidth = {1 / adc_dwell:.3f} Hz")

    if test_report:
        print(seq.test_report())

    if plot:
        preparation_duration = (
            alpha_half_center_spacing + dummy_repetitions * tr
            if use_alpha_half
            else dummy_repetitions * tr
        )
        seq.plot(time_range=(preparation_duration, preparation_duration + 2 * tr))

        waveforms = seq.waveforms_and_times()[0]
        plt.figure()
        plt.plot(
            waveforms[0][0],
            waveforms[0][1],
            waveforms[1][0],
            waveforms[1][1],
            waveforms[2][0],
            waveforms[2][1],
        )
        plt.show()

    seq.set_definition(key="FOV", value=[fov_x, fov_y, fov_z])
    seq.set_definition(key="MatrixSize", value=[n_read, n_phase, n_partition])
    set_pulseq_encoding_definitions(
        seq,
        encoding_frame,
        fov_m=(fov_x, fov_y, fov_z),
        matrix=(n_read, n_phase, n_partition),
    )
    seq.set_definition(key="FieldStrengthT", value=field_strength_t)
    seq.set_definition(key="Nucleus", value=nucleus)
    seq.set_definition(key="DynamicFrames", value=int(n_repetition))
    seq.set_definition(key="SpectralTargetOffsetsHz", value=list(target_offsets))
    seq.set_definition(key="SpectralReceiverOffsetsHz", value=list(receiver_offsets))
    seq.set_definition(key="SpectralTargetNames", value=list(target_names))
    seq.set_definition(key="FlipAngleDeg", value=list(target_flip_angles))
    seq.set_definition(key="SpectralRFBandwidthHz", value=spectral_rf_bandwidth_hz)
    seq.set_definition(
        key="SpectralRFBandwidthFactorHzMs",
        value=spectral_rf_bandwidth_factor_hz_ms,
    )
    seq.set_definition(key="SpectralRFFWHM", value=spectral_rf_fwhm_hz)
    seq.set_definition(key="SpectralRFDuration", value=spectral_rf_duration)
    seq.set_definition(
        key="AlphaHalfCenterSpacing",
        value=alpha_half_center_spacing if use_alpha_half else 0.0,
    )
    seq.set_definition(
        key="AlphaHalfPhaseDeg",
        value=(
            wrap_phase_deg(rf_phase_start + rf_phase_increment)
            if use_alpha_half
            else 0.0
        ),
    )
    seq.set_definition(
        key="EndImageSpoilerCyclesPerFOV",
        value=end_image_spoiler_cycles_per_fov,
    )
    seq.set_definition(
        key="EndImageSpoilerCyclesPerVoxel",
        value=end_image_spoiler_cycles_per_voxel,
    )
    seq.set_definition(
        key="EndImageSpoilerVoxelSizeM",
        value=physical_voxel_sizes.tolist(),
    )
    seq.set_definition(
        key="EndImageSpoilerDuration",
        value=end_image_spoiler_duration,
    )
    seq.set_definition(
        key="EndImageSpoilerAxes",
        value="".join(event.channel for event in end_image_spoilers) or "none",
    )
    seq.set_definition(key="EndImageSpoilerEndTimes", value=spoiler_end_times)
    seq.set_definition(key="IdealSpoilerEndTimes", value=spoiler_end_times)
    seq.set_definition(
        key="SingleMetaboliteAcquisitionTime",
        value=n_phase * n_partition * tr,
    )
    seq.set_definition(key="ReadoutBandwidthHz", value=1 / adc_dwell)
    seq.set_definition(key="ADCDwell", value=adc_dwell)
    seq.set_definition(key="ReadoutDuration", value=readout_duration)
    if target_tr is not None:
        seq.set_definition(key="TargetTR", value=target_tr)
    seq.set_definition(
        key="SpectralRFPulseType", value=str(spectral_pulse_type).lower()
    )
    if str(spectral_pulse_type).lower() == "slr":
        seq.set_definition(key="SpectralSLRSharpness", value=spectral_slr_sharpness)
        seq.set_definition(
            key="SpectralRFPulseFile",
            value=str(
                _default_slr_pulse_path(spectral_slr_sharpness)
                if spectral_slr_pulse_path is None
                else spectral_slr_pulse_path
            ),
        )
    seq.set_definition(key="TR", value=tr)
    seq.set_definition(key="TE", value=te)
    seq.set_definition(key="RFPhaseStartDeg", value=rf_phase_start)
    seq.set_definition(key="RFPhaseIncrementDeg", value=rf_phase_increment)
    seq.set_definition(key="FrequencyOffsetPhaseCoherent", value=True)
    seq.set_definition(key="PhaseReference", value="rf-target-locked")

    if write_seq:
        script_dir = Path(__file__).resolve().parent
        output_path = script_dir.parent / "sequences" / seq_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seq.write(str(output_path), v141_compat=v141_compat)
        print(f"Sequence written to {output_path}")

    return seq


if __name__ == "__main__":
    main(
        spectral_pulse_type="slr",
        alpha_half_center_spacing=6.29e-3,
        spectral_slr_sharpness=1.0,
        plot=False,
        write_seq=True,
        n_repetition=20,
        rf_phase_increment=0.0,  # same-phase RF train used by Skinner et al.
        seq_filename="bssfp_3d_spectral_selective_skinner.seq",
    )
