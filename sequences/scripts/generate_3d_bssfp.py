"""Generate a non-selective Cartesian 3D balanced-SSFP sequence.

The read, phase-encoding, and partition-encoding gradient moments are rewound
within every TR.  A non-selective RF block pulse is intentional: in 3D imaging
the excited volume is encoded along z rather than selected as an individual
2D slice.  Restricting the excited volume to a slab would require an additional
fully balanced slab-select gradient.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp

from blochsimulator.sequence.bssfp_phase import (
    advance_bssfp_phase_deg,
    pulseq_phase_offset_rad,
    wrap_phase_deg,
)
from blochsimulator.sequence.rf_pulses import (
    make_pulseq_rf_events,
    set_rf_definitions,
)


def _as_3d_fov(fov: float | tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(fov, (int, float)):
        values = (float(fov),) * 3
    else:
        if len(fov) != 3:
            raise ValueError("fov must be a scalar or a three-element tuple")
        values = tuple(float(value) for value in fov)

    if any(value <= 0 for value in values):
        raise ValueError("all FOV values must be positive")
    return values


def _encoding_areas(matrix_size: int, delta_k: float) -> np.ndarray:
    """Return Cartesian encoding moments with a sample at k=0."""
    if not isinstance(matrix_size, (int, np.integer)) or matrix_size <= 0:
        raise ValueError("matrix sizes must be positive integers")
    return (np.arange(matrix_size) - matrix_size // 2) * delta_k


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "bssfp_3d.seq",
    *,
    fov: float | tuple[float, float, float] = (220e-3, 220e-3, 160e-3),
    n_read: int = 64,
    n_phase: int = 64,
    n_partition: int = 32,
    flip_angle_deg: float = 15,
    rf_pulse_type: str = "block",
    rf_duration: float = 1e-3,
    rf_time_bandwidth_product: float = 4.0,
    rf_apodization: float = 0.5,
    rf_slr_sharpness: float = 1.0,
    rf_custom_waveform_hz=None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    adc_dwell: float = 100e-6,
    encoding_duration: float = 1e-3,
    rf_phase_start: float = 180,
    rf_phase_increment: float = 180,
    dummy_repetitions: int = 1,
    use_alpha_half: bool = True,
):
    """Create a Cartesian 3D bSSFP sequence.

    Parameters are expressed in SI units. ``fov`` is ordered as
    ``(fov_x, fov_y, fov_z)`` and the acquired data are ordered as partition,
    phase-encode, readout. RF phase cycling is continuous through dummy and
    acquired repetitions.
    """
    fov_x, fov_y, fov_z = _as_3d_fov(fov)
    _encoding_areas(n_read, 1.0)  # Validate the readout matrix size as well.
    ky_areas = _encoding_areas(n_phase, 1 / fov_y)
    kz_areas = _encoding_areas(n_partition, 1 / fov_z)

    if flip_angle_deg <= 0:
        raise ValueError("flip_angle_deg must be positive")
    if rf_duration <= 0 or adc_dwell <= 0 or encoding_duration <= 0:
        raise ValueError("event durations must be positive")
    if not isinstance(dummy_repetitions, (int, np.integer)) or dummy_repetitions < 0:
        raise ValueError("dummy_repetitions must be a non-negative integer")

    system = pp.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=20e-6,
    )
    seq = pp.Sequence(system)

    rf_events, actual_rf_duration, effective_rf_tbw, rf_pulse_type = (
        make_pulseq_rf_events(
            pp,
            system,
            flip_angles_deg=(flip_angle_deg, flip_angle_deg / 2),
            pulse_type=rf_pulse_type,
            duration_s=rf_duration,
            time_bandwidth_product=rf_time_bandwidth_product,
            apodization=rf_apodization,
            slr_sharpness=rf_slr_sharpness,
            custom_waveform_hz=rf_custom_waveform_hz,
            custom_raster_s=rf_custom_raster_s,
            custom_flip_angle_deg=rf_custom_flip_angle_deg,
        )
    )
    rf, rf_alpha_half = rf_events

    readout_duration = n_read * adc_dwell
    readout_amplitude = 1 / (fov_x * adc_dwell)
    readout_rise_time = max(
        system.adc_dead_time,
        np.ceil(abs(readout_amplitude) / system.max_slew / system.grad_raster_time)
        * system.grad_raster_time,
    )
    gx = pp.make_trapezoid(
        channel="x",
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
    gx_pre = pp.make_trapezoid(
        channel="x",
        area=-gx.area / 2,
        duration=encoding_duration,
        system=system,
    )

    # RF dead time occurs before the RF envelope and ringdown after it. Add the
    # difference after RF so that the ADC center lies halfway between RF centers.
    rf_center, _ = pp.calc_rf_center(rf)
    rf_center_from_block_start = rf.delay + rf_center
    rf_block_duration = pp.calc_duration(rf)
    read_block_duration = max(pp.calc_duration(gx), pp.calc_duration(adc))
    adc_center_from_block_start = adc.delay + adc.num_samples * adc.dwell / 2
    rf_balance_delay_value = (
        2 * rf_center_from_block_start
        + read_block_duration
        - 2 * adc_center_from_block_start
        - rf_block_duration
    )
    raster = system.block_duration_raster
    rf_balance_delay_value = np.round(rf_balance_delay_value / raster) * raster
    if rf_balance_delay_value < 0:
        raise ValueError("RF timing cannot be centered with a non-negative delay")
    rf_balance_delay = (
        pp.make_delay(rf_balance_delay_value) if rf_balance_delay_value > 0 else None
    )

    pre_duration = pp.calc_duration(gx_pre)
    tr = (
        rf_block_duration
        + rf_balance_delay_value
        + 2 * pre_duration
        + read_block_duration
    )
    te = tr / 2

    # An alpha/2 pulse one half-TR before the first full pulse provides the
    # standard catalyzation used by the Pulseq TrueFISP reference sequence.
    if use_alpha_half:
        rf_alpha_half.phase_offset = pulseq_phase_offset_rad(
            rf_phase_start,
            frequency_offset_hz=0.0,
            event_center_s=pp.calc_rf_center(rf_alpha_half)[0],
        )
        alpha_half_delay_value = tr / 2 - pp.calc_duration(rf_alpha_half)
        alpha_half_delay_value = np.round(alpha_half_delay_value / raster) * raster
        if alpha_half_delay_value < 0:
            raise ValueError("TR is too short for alpha/2 preparation")
        seq.add_block(rf_alpha_half)
        if alpha_half_delay_value > 0:
            seq.add_block(pp.make_delay(alpha_half_delay_value))

    rf_phase = wrap_phase_deg(rf_phase_start)

    def add_repetition(
        ky: float,
        kz: float,
        acquire: bool,
        partition_index: int | None = None,
    ) -> None:
        nonlocal rf_phase

        rf.phase_offset = pulseq_phase_offset_rad(
            rf_phase,
            frequency_offset_hz=0.0,
            event_center_s=rf_center,
        )
        adc.phase_offset = pulseq_phase_offset_rad(
            rf_phase,
            frequency_offset_hz=0.0,
            event_center_s=adc_center_from_block_start,
        )
        rf_phase = advance_bssfp_phase_deg(
            rf_phase,
            elapsed_s=tr,
            phase_increment_deg=rf_phase_increment,
        )

        gy_pre = pp.make_trapezoid(
            channel="y", area=ky, duration=encoding_duration, system=system
        )
        gy_reph = pp.make_trapezoid(
            channel="y", area=-ky, duration=encoding_duration, system=system
        )
        gz_pre = pp.make_trapezoid(
            channel="z", area=kz, duration=encoding_duration, system=system
        )
        gz_reph = pp.make_trapezoid(
            channel="z", area=-kz, duration=encoding_duration, system=system
        )

        seq.add_block(rf)
        if rf_balance_delay is not None:
            seq.add_block(rf_balance_delay)
        seq.add_block(gx_pre, gy_pre, gz_pre)
        if acquire:
            if partition_index is None:
                raise ValueError("acquired repetitions require a partition index")
            partition_label = pp.make_label(
                label="PAR",
                type="SET",
                value=partition_index,
            )
            seq.add_block(gx, adc, partition_label)
        else:
            # ADC dead time can make the acquired readout block longer than
            # the gradient alone. Preserve an identical TR during dummy scans.
            seq.add_block(gx, pp.make_delay(read_block_duration))
        seq.add_block(gx_pre, gy_reph, gz_reph)

    for _ in range(dummy_repetitions):
        add_repetition(ky=0.0, kz=0.0, acquire=False)

    # Linear partition-major ordering. The RF/ADC phase is deliberately not
    # reset at partition boundaries, preserving the continuous bSSFP pulse train.
    for partition_index, kz in enumerate(kz_areas):
        for ky in ky_areas:
            add_repetition(
                ky=float(ky),
                kz=float(kz),
                acquire=True,
                partition_index=partition_index,
            )

    ok, error_report = seq.check_timing()
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed. Error listing follows:")
        for error in error_report:
            print(error)

    print(f"TR = {tr * 1e3:.3f} ms, TE = {te * 1e3:.3f} ms")

    if test_report:
        print(seq.test_report())

    if plot:
        preparation_duration = (
            tr / 2 + dummy_repetitions * tr
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
    seq.set_definition(key="Name", value="bssfp_3d")
    seq.set_definition(key="MatrixSize", value=[n_read, n_phase, n_partition])
    seq.set_definition(key="TR", value=tr)
    seq.set_definition(key="TE", value=te)
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

    if write_seq:
        script_dir = Path(__file__).resolve().parent
        output_path = script_dir.parent / "sequences" / seq_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the generated file readable by Pulseq 1.4.x consumers as well
        # as PyPulseq 1.5. The sequence does not require any 1.5-only events.
        seq.write(str(output_path), v141_compat=True)
        print(f"Sequence written to {output_path}")

    return seq


if __name__ == "__main__":
    main(
        plot=True,
        write_seq=True,
        n_read=16,
        n_phase=16,
        n_partition=16,
    )
