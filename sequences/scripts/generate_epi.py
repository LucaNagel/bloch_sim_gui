import numpy as np
import pypulseq as pp


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "epi_pypulseq_multi_rep.seq",
    *,
    v141_compat: bool = True,
    flip_angle_deg: float = 90.0,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 64,
    n_y: int = 64,
    slice_thickness: float = 3e-3,
    slice_gap: float = 0.0,
    n_slices: int = 3,
    n_repetitions: int = 1,
    repetition_time: float | None = None,
    use_slice_labels: bool = False,
    spoil_after_slice: bool = True,
    spoiler_cycles_per_slice: float = 8.0,
    spoiler_cycles_per_voxel: float = 0.0,
    spoiler_duration: float = 4e-3,
):
    """Create a basic EPI sequence without ramp-sampling.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is
        'epi_pypulseq_multi_rep.seq'.
    v141_compat : bool, optional
        Write Pulseq 1.4.1-compatible syntax when ``write_seq`` is True. This
        avoids Pulseq 1.5 RF/ADC columns such as RF ``use`` strings that older
        GUI readers parse as unmatched numeric data. Default is True.
    flip_angle_deg : float, optional
        Nominal excitation flip angle in degrees. Default is 90.
    fov : float or tuple of float, optional
        Field of view in meters. If a single value, it is used for both x and y.
        If a tuple, it is (fov_x, fov_y). Default is 220e-3.
    n_x : int, optional
        Number of readout samples. Default is 64.
    n_y : int, optional
        Number of phase encoding steps. Default is 64.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 3e-3.
    slice_gap : float, optional
        Edge-to-edge gap between adjacent slices in meters. The slice-center
        spacing is ``slice_thickness + slice_gap``. Default is 0.
    n_slices : int, optional
        Number of slices. Default is 3.
    n_repetitions : int, optional
        Number of complete multi-slice repetitions. Default is 1.
    repetition_time : float, optional
        Time in seconds between excitations of the first slice in consecutive
        repetitions. It must be at least as long as all slice readouts in one
        repetition. If omitted, repetitions are played back-to-back at the
        shortest possible repetition time.
    use_slice_labels : bool, optional
        Add Pulseq ``SLC`` and ``REP`` label extensions for every acquired
        frame. Disabled by default because some Pulseq readers reject label
        extensions with string label names. When disabled, the simulator still
        infers slice and repetition frames from the RF excitations.
    spoil_after_slice : bool, optional
        Add a through-slice crusher after each slice readout to suppress
        residual transverse magnetization before the next slice excitation.
        Default is True.
    spoiler_cycles_per_slice : float, optional
        Through-slice spoiler moment in cycles across one slice thickness.
        Default is 8.
    spoiler_cycles_per_voxel : float, optional
        Optional extra in-plane spoiler moment in cycles across one acquired
        voxel. Keep at zero for repeatable 2D slice frames; if enabled, use a
        non-integer value to avoid refocusing exactly on the voxel grid.
        Default is 0.
    spoiler_duration : float, optional
        Duration of the spoiler block in seconds. Default is 4 ms.

    Returns
    -------
    seq : pypulseq.Sequence
        The EPI sequence object.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov
    if len((fov_x, fov_y)) != 2 or not np.all(np.isfinite((fov_x, fov_y))):
        raise ValueError("fov must contain two finite values")
    if min(fov_x, fov_y) <= 0:
        raise ValueError("fov values must be positive")
    if not np.isfinite(flip_angle_deg) or flip_angle_deg <= 0:
        raise ValueError("flip_angle_deg must be positive and finite")
    if not np.isfinite(slice_thickness) or slice_thickness <= 0:
        raise ValueError("slice_thickness must be positive and finite")
    if not np.isfinite(slice_gap) or slice_gap < 0:
        raise ValueError("slice_gap must be finite and non-negative")
    if not isinstance(n_slices, (int, np.integer)) or n_slices <= 0:
        raise ValueError("n_slices must be a positive integer")
    if not isinstance(n_repetitions, (int, np.integer)) or n_repetitions <= 0:
        raise ValueError("n_repetitions must be a positive integer")
    if repetition_time is not None and (
        not np.isfinite(repetition_time) or repetition_time <= 0
    ):
        raise ValueError("repetition_time must be positive and finite")
    if not np.isfinite(spoiler_cycles_per_slice) or spoiler_cycles_per_slice < 0:
        raise ValueError("spoiler_cycles_per_slice must be finite and non-negative")
    if not np.isfinite(spoiler_cycles_per_voxel) or spoiler_cycles_per_voxel < 0:
        raise ValueError("spoiler_cycles_per_voxel must be finite and non-negative")
    if not np.isfinite(spoiler_duration) or spoiler_duration <= 0:
        raise ValueError("spoiler_duration must be positive and finite")
    slice_spacing = slice_thickness + slice_gap
    slice_positions = (
        np.arange(n_slices, dtype=float) - (n_slices - 1) / 2.0
    ) * slice_spacing

    # Set system limits
    system = pp.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
    )

    seq = pp.Sequence(system)

    # Create the slice-selection pulse and gradient
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        system=system,
        duration=3e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        return_gz=True,
        delay=system.rf_dead_time,
        use="excitation",
    )

    # Define other gradients and ADC events
    delta_kx = 1 / fov_x
    delta_ky = 1 / fov_y
    k_width = n_x * delta_kx
    adc_dwell = 4e-6
    adc_duration = n_x * adc_dwell
    gx_flat_time = adc_duration
    gx_flat_time = np.ceil(gx_flat_time * 1e5) * 1e-5  # Round-up to the gradient raster
    gx = pp.make_trapezoid(
        channel="x",
        system=system,
        amplitude=k_width / adc_duration,
        flat_time=gx_flat_time,
    )
    gx_reverse = pp.make_trapezoid(
        channel="x",
        system=system,
        amplitude=-k_width / adc_duration,
        flat_time=gx_flat_time,
    )
    adc = pp.make_adc(
        num_samples=n_x,
        duration=adc_duration,
        # Pulseq ADC samples are dwell-centred. Centre the complete aperture on
        # the gradient flat top; subtracting only the centre-to-centre span
        # shifts alternating EPI lines onto different kx grids.
        delay=gx.rise_time + gx_flat_time / 2 - adc_duration / 2,
    )

    # Pre-phasing gradients
    pre_time = 8e-4
    gx_pre = pp.make_trapezoid(
        channel="x", system=system, area=-gx.area / 2, duration=pre_time
    )
    gz_reph = pp.make_trapezoid(
        channel="z", system=system, area=-gz.area / 2, duration=pre_time
    )
    gy_pre = pp.make_trapezoid(
        channel="y", system=system, area=-n_y / 2 * delta_ky, duration=pre_time
    )
    relative_x_end_area = -gx.area / 2 + (gx.area if n_y % 2 else 0.0)
    relative_y_end_area = (-n_y / 2 + max(n_y - 1, 0)) * delta_ky
    gx_post = pp.make_trapezoid(
        channel="x", system=system, area=-relative_x_end_area, duration=pre_time
    )
    gy_post = pp.make_trapezoid(
        channel="y", system=system, area=-relative_y_end_area, duration=pre_time
    )

    # Phase blip in the shortest possible time
    gy_blip_duration = 2 * np.sqrt(delta_ky / system.max_slew)
    gy_blip_duration = np.ceil(gy_blip_duration / 10e-6) * 10e-6
    gy = pp.make_trapezoid(
        channel="y", system=system, area=delta_ky, duration=gy_blip_duration
    )
    gx_spoil = gy_spoil = gz_spoil = None
    if spoil_after_slice and (
        spoiler_cycles_per_slice > 0 or spoiler_cycles_per_voxel > 0
    ):
        if spoiler_cycles_per_voxel > 0:
            gx_spoil = pp.make_trapezoid(
                channel="x",
                system=system,
                area=spoiler_cycles_per_voxel / (fov_x / n_x),
                duration=spoiler_duration,
            )
            gy_spoil = pp.make_trapezoid(
                channel="y",
                system=system,
                area=spoiler_cycles_per_voxel / (fov_y / n_y),
                duration=spoiler_duration,
            )
        if spoiler_cycles_per_slice > 0:
            gz_spoil = pp.make_trapezoid(
                channel="z",
                system=system,
                area=spoiler_cycles_per_slice / slice_thickness,
                duration=spoiler_duration,
            )

    rf_center_time, _ = pp.calc_rf_center(rf)
    spoiler_end_times = []

    # Loop over complete multi-slice repetitions. TR is measured from the first
    # slice excitation of one repetition to the first slice of the next.
    minimum_repetition_time = None
    actual_repetition_time = None
    for i_repetition in range(n_repetitions):
        repetition_start = 0.0 if not seq.block_events else float(seq.duration()[0])
        for i_slice in range(n_slices):
            rf.freq_offset = gz.amplitude * slice_positions[i_slice]
            rf.phase_offset = -2 * np.pi * rf.freq_offset * rf_center_time
            if use_slice_labels:
                seq.add_block(
                    rf,
                    gz,
                    pp.make_label("SLC", "SET", i_slice),
                    pp.make_label("REP", "SET", i_repetition),
                )
            else:
                seq.add_block(rf, gz)
            seq.add_block(gx_pre, gy_pre, gz_reph)
            for i_line in range(n_y):
                readout_gradient = gx if i_line % 2 == 0 else gx_reverse
                seq.add_block(readout_gradient, adc)  # Read one line of k-space
                if i_line < n_y - 1:
                    seq.add_block(gy)  # Phase blip
            seq.add_block(gx_post, gy_post)
            spoiler_events = [
                event for event in (gx_spoil, gy_spoil, gz_spoil) if event is not None
            ]
            if spoiler_events:
                seq.add_block(*spoiler_events)
                spoiler_end_times.append(float(seq.duration()[0]))

        acquisition_end = float(seq.duration()[0])
        package_duration = acquisition_end - repetition_start
        if minimum_repetition_time is None:
            minimum_repetition_time = package_duration
            actual_repetition_time = (
                package_duration if repetition_time is None else repetition_time
            )
            tolerance = max(1e-12, package_duration * 1e-10)
            if actual_repetition_time < package_duration - tolerance:
                raise ValueError(
                    "repetition_time is shorter than the minimum multi-slice "
                    f"acquisition time ({package_duration:.9g} s)"
                )
        repetition_delay = actual_repetition_time - package_duration
        if repetition_delay > max(1e-12, actual_repetition_time * 1e-12):
            seq.add_block(pp.make_delay(repetition_delay))

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

    slice_package_extent = n_slices * slice_thickness + (n_slices - 1) * slice_gap
    seq.set_definition(key="FOV", value=[fov_x, fov_y, slice_package_extent])
    seq.set_definition(key="SliceThickness", value=slice_thickness)
    seq.set_definition(key="SliceGap", value=slice_gap)
    seq.set_definition(key="SliceSpacing", value=slice_spacing)
    seq.set_definition(key="SlicePositions", value=slice_positions.tolist())
    seq.set_definition(key="FlipAngleDeg", value=flip_angle_deg)
    seq.set_definition(key="Repetitions", value=n_repetitions)
    seq.set_definition(key="RepetitionTime", value=actual_repetition_time)
    seq.set_definition(key="MinimumRepetitionTime", value=minimum_repetition_time)
    seq.set_definition(key="UseSliceLabels", value=int(bool(use_slice_labels)))
    seq.set_definition(key="SpoilAfterSlice", value=bool(spoil_after_slice))
    seq.set_definition(key="SpoilerCyclesPerSlice", value=spoiler_cycles_per_slice)
    seq.set_definition(key="SpoilerCyclesPerVoxel", value=spoiler_cycles_per_voxel)
    seq.set_definition(key="SpoilerDuration", value=spoiler_duration)
    seq.set_definition(
        key="SpoilerAxes",
        value="".join(
            axis
            for axis, event in zip("xyz", (gx_spoil, gy_spoil, gz_spoil))
            if event is not None
        )
        or "none",
    )
    seq.set_definition(key="SpoilerEndTimes", value=spoiler_end_times)
    seq.set_definition(key="IdealSpoilerEndTimes", value=spoiler_end_times)

    if write_seq:
        from pathlib import Path

        script_dir = Path(__file__).resolve().parent
        output_path = script_dir.parent / "sequences" / seq_filename

        seq.write(output_path, v141_compat=v141_compat)
        print(f"Sequence written to {output_path}")

    return seq


if __name__ == "__main__":
    main(
        plot=False,
        write_seq=True,
        n_x=64,
        n_y=64,
        n_slices=1,
        slice_gap=2e-3,
        slice_thickness=3e-3,
        n_repetitions=10,
        repetition_time=2,
        flip_angle_deg=30,
    )
