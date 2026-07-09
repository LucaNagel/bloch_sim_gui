import numpy as np
import pypulseq as pp


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "epi_pypulseq.seq",
    *,
    v141_compat: bool = True,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 64,
    n_y: int = 64,
    slice_thickness: float = 3e-3,
    slice_gap: float = 0.0,
    n_slices: int = 3,
    use_slice_labels: bool = False,
    spoil_after_slice: bool = True,
    spoiler_cycles_per_slice: float = 8.0,
    spoiler_cycles_per_voxel: float = 0.0,
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
        Output filename for the .seq file. Default is 'epi_pypulseq.seq'.
    v141_compat : bool, optional
        Write Pulseq 1.4.1-compatible syntax when ``write_seq`` is True. This
        avoids Pulseq 1.5 RF/ADC columns such as RF ``use`` strings that older
        GUI readers parse as unmatched numeric data. Default is True.
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
    use_slice_labels : bool, optional
        Add Pulseq ``SLC`` label extensions for each slice. Disabled by default
        because some Pulseq readers reject label extensions with string label
        names. When disabled, the simulator still infers slice frames from the
        RF frequency offsets.
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
    if not np.isfinite(slice_thickness) or slice_thickness <= 0:
        raise ValueError("slice_thickness must be positive and finite")
    if not np.isfinite(slice_gap) or slice_gap < 0:
        raise ValueError("slice_gap must be finite and non-negative")
    if not isinstance(n_slices, (int, np.integer)) or n_slices <= 0:
        raise ValueError("n_slices must be a positive integer")
    if not np.isfinite(spoiler_cycles_per_slice) or spoiler_cycles_per_slice < 0:
        raise ValueError("spoiler_cycles_per_slice must be finite and non-negative")
    if not np.isfinite(spoiler_cycles_per_voxel) or spoiler_cycles_per_voxel < 0:
        raise ValueError("spoiler_cycles_per_voxel must be finite and non-negative")
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

    # Create 90 degree slice selection pulse and gradient
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.pi / 2,
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
    if spoil_after_slice and spoiler_cycles_per_slice > 0:
        spoiler_duration = 4e-3
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
        gz_spoil = pp.make_trapezoid(
            channel="z",
            system=system,
            area=spoiler_cycles_per_slice / slice_thickness,
            duration=spoiler_duration,
        )

    rf_center_time, _ = pp.calc_rf_center(rf)

    # Loop over slices
    for i_slice in range(n_slices):
        rf.freq_offset = gz.amplitude * slice_positions[i_slice]
        rf.phase_offset = -2 * np.pi * rf.freq_offset * rf_center_time
        if use_slice_labels:
            seq.add_block(rf, gz, pp.make_label("SLC", "SET", i_slice))
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
    seq.set_definition(key="UseSliceLabels", value=int(bool(use_slice_labels)))
    seq.set_definition(key="SpoilAfterSlice", value=bool(spoil_after_slice))
    seq.set_definition(key="SpoilerCyclesPerSlice", value=spoiler_cycles_per_slice)
    seq.set_definition(key="SpoilerCyclesPerVoxel", value=spoiler_cycles_per_voxel)

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
        n_slices=3,
        slice_gap=2e-3,
        slice_thickness=3e-3,
    )
