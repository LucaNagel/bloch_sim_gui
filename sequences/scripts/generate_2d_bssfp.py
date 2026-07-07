"""
Basic 2D Cartesian bSSFP-like sequence.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp


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
    rf_duration: float = 1e-3,
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

    rf, _, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=rf_duration,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        return_gz=True,
    )

    rf_alpha_half, _, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_angle_deg / 2),
        duration=rf_duration,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        return_gz=True,
    )

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

    rf_phase = rf_phase_start

    seq.add_block(rf_alpha_half)
    seq.add_block(pp.make_delay(pp.calc_duration(gx_pre) + pp.calc_duration(gx) / 2))

    for phase_idx in range(-n_phase // 2, n_phase // 2):
        rf.phase_offset = np.deg2rad(rf_phase)
        adc.phase_offset = np.deg2rad(rf_phase)

        rf_phase = np.mod(rf_phase + rf_phase_increment, 360.0)

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

    if write_seq:
        script_dir = Path(__file__).resolve().parent
        output_path = script_dir.parent / "sequences" / seq_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        seq.write(str(output_path))
        print(f"Sequence written to {output_path}")

    return seq


if __name__ == "__main__":
    main(plot=True, write_seq=True)
