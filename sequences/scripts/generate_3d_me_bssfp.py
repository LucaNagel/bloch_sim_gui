"""Generate the Cartesian 3D multi-echo bSSFP sequence from Gaubatz 2023.

The default acquisition uses a five-echo monopolar flyback train centered
between consecutive RF pulses. Parameters reproduce the short-TR in-vivo
hyperpolarized 13C acquisition described in the thesis.
"""

from pathlib import Path

from blochsimulator.sequence import make_pulseq_me_bssfp


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "me_bssfp_3d_gaubatz.seq",
    *,
    fov_m=(56e-3, 28e-3, 24.5e-3),
    matrix=(32, 16, 14),
    echoes: int = 5,
    echo_spacing_s: float = 1.32e-3,
    readout_strategy: str = "flyback",
    sampling_bandwidth_hz: float = 39_682.5,
    flip_angle_deg: float = 3.5,
    rf_pulse_type: str = "gaussian",
    rf_duration_s: float = 0.5e-3,
    rf_bandwidth_hz: float = 5480.0,
    rf_frequency_offset_hz: float = 0.0,
    receiver_frequency_offset_hz: float = -460.0,
    encoding_duration_s: float = 0.5e-3,
    repetition_time_s: float = 8.696e-3,
    rf_phase_start_deg: float = 0.0,
    rf_phase_increment_deg: float = 180.0,
    dummy_repetitions: int = 0,
    repetitions: int = 1,
    use_alpha_half: bool = True,
    field_strength_t: float = 7.0,
    nucleus: str = "C13",
    encoding_axes=("+x", "+y", "+z"),
    scanner_parameters=None,
    v141_compat: bool = True,
):
    """Create, optionally inspect, and optionally write Cartesian ME-bSSFP."""
    sequence = make_pulseq_me_bssfp(
        fov_m=fov_m,
        matrix=matrix,
        echoes=echoes,
        echo_spacing_s=echo_spacing_s,
        readout_strategy=readout_strategy,
        sampling_bandwidth_hz=sampling_bandwidth_hz,
        flip_angle_deg=flip_angle_deg,
        rf_pulse_type=rf_pulse_type,
        rf_duration_s=rf_duration_s,
        rf_bandwidth_hz=rf_bandwidth_hz,
        rf_frequency_offset_hz=rf_frequency_offset_hz,
        receiver_frequency_offset_hz=receiver_frequency_offset_hz,
        encoding_duration_s=encoding_duration_s,
        repetition_time_s=repetition_time_s,
        rf_phase_start_deg=rf_phase_start_deg,
        rf_phase_increment_deg=rf_phase_increment_deg,
        dummy_repetitions=dummy_repetitions,
        repetitions=repetitions,
        use_alpha_half=use_alpha_half,
        field_strength_t=field_strength_t,
        nucleus=nucleus,
        encoding_axes=encoding_axes,
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
