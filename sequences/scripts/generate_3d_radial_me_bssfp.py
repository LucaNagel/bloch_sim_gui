"""Generate the 3D radial multi-echo bSSFP sequence from Wang et al. 2025.

The Pulseq sequence uses five monopolar center-through echoes per TR, a
golden-angle spiral-phyllotaxis distribution of 3D spokes, and an alpha/2
preparation. Defaults reproduce the acquisition parameters reported in
doi:10.1002/mrm.30614 for in-vivo hyperpolarized 13C imaging.
"""

from pathlib import Path

from blochsimulator.sequence import make_pulseq_radial_me_bssfp


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "radial_me_bssfp_3d_wang.seq",
    *,
    fov_m: float = 356e-3,
    base_resolution: int = 32,
    readout_oversampling: int = 2,
    spokes_per_measurement: int = 300,
    measurements: int = 4,
    echoes: int = 5,
    echo_spacing_s: float = 2e-3,
    pixel_bandwidth_hz: float = 1000.0,
    flip_angle_deg: float = 10.0,
    rf_pulse_type: str = "block",
    rf_duration_s: float = 0.5e-3,
    rf_time_bandwidth_product: float = 4.0,
    rf_apodization: float = 0.5,
    rf_slr_sharpness: float = 1.0,
    rf_custom_waveform_hz=None,
    rf_custom_raster_s: float | None = None,
    rf_custom_flip_angle_deg: float | None = None,
    repetition_time_s: float = 16e-3,
    rf_phase_start_deg: float = 0.0,
    rf_phase_increment_deg: float = 180.0,
    use_alpha_half: bool = True,
    use_tip_back: bool = True,
    prephaser_duration_s: float = 0.5e-3,
    inter_measurement_rotation_deg: float = 137.50776405003785,
    field_strength_t: float = 3.0,
    nucleus: str = "C13",
    scanner_parameters=None,
    v141_compat: bool = True,
):
    """Create, optionally inspect, and optionally write radial ME-bSSFP."""
    sequence = make_pulseq_radial_me_bssfp(
        fov_m=fov_m,
        base_resolution=base_resolution,
        readout_oversampling=readout_oversampling,
        spokes_per_measurement=spokes_per_measurement,
        measurements=measurements,
        echoes=echoes,
        echo_spacing_s=echo_spacing_s,
        pixel_bandwidth_hz=pixel_bandwidth_hz,
        flip_angle_deg=flip_angle_deg,
        rf_pulse_type=rf_pulse_type,
        rf_duration_s=rf_duration_s,
        rf_time_bandwidth_product=rf_time_bandwidth_product,
        rf_apodization=rf_apodization,
        rf_slr_sharpness=rf_slr_sharpness,
        rf_custom_waveform_hz=rf_custom_waveform_hz,
        rf_custom_raster_s=rf_custom_raster_s,
        rf_custom_flip_angle_deg=rf_custom_flip_angle_deg,
        repetition_time_s=repetition_time_s,
        rf_phase_start_deg=rf_phase_start_deg,
        rf_phase_increment_deg=rf_phase_increment_deg,
        use_alpha_half=use_alpha_half,
        use_tip_back=use_tip_back,
        prephaser_duration_s=prephaser_duration_s,
        inter_measurement_rotation_deg=inter_measurement_rotation_deg,
        field_strength_t=field_strength_t,
        nucleus=nucleus,
        scanner_parameters=scanner_parameters,
    )
    if test_report:
        print(sequence.test_report())
    if plot:
        sequence.plot(time_range=(0.0, 2.5 * repetition_time_s))
    if write_seq:
        script_dir = Path(__file__).resolve().parent
        output_path = script_dir.parent / "sequences" / seq_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sequence.write(str(output_path), v141_compat=v141_compat)
        print(f"Sequence written to {output_path}")
    return sequence


if __name__ == "__main__":
    main(write_seq=True)
