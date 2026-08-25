"""
Basic 2D Cartesian bSSFP-like sequence.
"""

from pathlib import Path

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


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "bssfp_2d.seq",
    *,
    fov: float | tuple[float, float] = 1000e-3,
    slice_thickness: float = 8e-3,
    n_read: int = 64,
    n_phase: int = 64,
    flip_angle_deg: float = 15,
    rf_pulse_type: str = "sinc",
    rf_duration: float = 1e-3,
    rf_time_bandwidth_product: float = 4.0,
    rf_apodization: float = 0.5,
    rf_slr_sharpness: float = 1.0,
    rf_custom_waveform_hz=None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    rf_phase_start: float = 180,
    rf_phase_increment: float = 180,
):
    """Create a simple 2D Cartesian bSSFP-like sequence."""

    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

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

    dwell = 10 * system.grad_raster_time

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

    gx = pp.make_trapezoid(
        channel="x",
        flat_area=n_read,
        flat_time=n_read * dwell,
        system=system,
    )

    adc = pp.make_adc(
        num_samples=n_read,
        duration=n_read * dwell,
        phase_offset=0,
        delay=gx.rise_time,
        system=system,
    )

    gx_pre = pp.make_trapezoid(
        channel="x",
        area=-gx.area / 2,
        duration=1e-3,
        system=system,
    )

    rf_phase = wrap_phase_deg(rf_phase_start)

    rf_alpha_half.phase_offset = pulseq_phase_offset_rad(
        rf_phase_start,
        frequency_offset_hz=0.0,
        event_center_s=pp.calc_rf_center(rf_alpha_half)[0],
    )
    seq.add_block(rf_alpha_half)
    seq.add_block(pp.make_delay(pp.calc_duration(gx_pre) + pp.calc_duration(gx) / 2))

    for phase_idx in range(-n_phase // 2, n_phase // 2):
        rf.phase_offset = pulseq_phase_offset_rad(
            rf_phase,
            frequency_offset_hz=0.0,
            event_center_s=pp.calc_rf_center(rf)[0],
        )
        adc.phase_offset = pulseq_phase_offset_rad(
            rf_phase,
            frequency_offset_hz=0.0,
            event_center_s=adc.delay + adc.num_samples * adc.dwell / 2,
        )

        rf_phase = advance_bssfp_phase_deg(
            rf_phase,
            elapsed_s=(
                pp.calc_duration(rf)
                + 2 * pp.calc_duration(gx_pre)
                + pp.calc_duration(gx)
            ),
            phase_increment_deg=rf_phase_increment,
        )

        gy_pre = pp.make_trapezoid(
            channel="y",
            area=phase_idx,
            duration=1e-3,
            system=system,
        )

        gy_reph = pp.make_trapezoid(
            channel="y",
            area=-phase_idx,
            duration=1e-3,
            system=system,
        )

        seq.add_block(rf)
        seq.add_block(gx_pre, gy_pre)
        seq.add_block(adc, gx)
        seq.add_block(gx_pre, gy_reph)

    ok, error_report = seq.check_timing()
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed. Error listing follows:")
        [print(e) for e in error_report]

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

        gw = seq.waveforms_and_times()[0]
        plt.figure()
        plt.plot(
            gw[0][0],
            gw[0][1],
            gw[1][0],
            gw[1][1],
            gw[2][0],
            gw[2][1],
        )
        plt.show()

    seq.set_definition(key="FOV", value=[fov_x, fov_y, slice_thickness])
    seq.set_definition(key="Name", value="bssfp_2d")
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

        seq.write(str(output_path))
        print(f"Sequence written to {output_path}")

    return seq


if __name__ == "__main__":
    main(plot=True, write_seq=True)
