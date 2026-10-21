import numpy as np
import pytest

from blochsimulator import BlochSimulator
from PyQt5.QtCore import Qt

from blochsimulator.sequence import (
    ScannerParameters,
    SpectralBand,
    SpectralStopBand,
    SpectralSpatialPulse,
    SequenceCompiler,
    check_spectral_alias_compatibility,
    design_spectral_spatial_pulse,
    find_compatible_spectral_sampling_rate,
    load_pulseq,
    make_pulseq_bssfp,
    make_pulseq_csi,
    make_pulseq_epi,
    make_pulseq_epsi_mge,
    make_pulseq_flash,
    make_pulseq_me_bssfp,
    make_pulseq_radial_me_bssfp,
    make_pulseq_slice_selection,
    make_pulseq_spectral_selective_bssfp,
    make_pulseq_spiral,
    simulate_spectral_spatial_profile,
)
from blochsimulator.ui.spectral_spatial_designer import (
    SpectralSpatialPulseDesignerDialog,
)
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget
from blochsimulator.units import NUCLEUS_GAMMA_HZ_PER_T, ppm_to_hz


def _test_pulse():
    return design_spectral_spatial_pulse(
        bands=(
            SpectralBand(0.0, 100.0, name="Reference"),
            SpectralBand(150.0, 60.0, amplitude=0.4, name="Second peak"),
        ),
        flip_angle_deg=30.0,
        slice_thickness_m=30e-3,
        spatial_time_bandwidth_product=2.0,
        subpulse_duration_s=0.2e-3,
        subpulse_count=3,
        nucleus="H1",
        max_gradient_t_per_m=0.032,
        max_slew_t_per_m_s=130.0,
        name="Test SPSP",
    )


def test_spectral_spatial_design_is_balanced_serializable_and_simulatable():
    pulse = _test_pulse()

    assert pulse.duration_s > 0.0
    assert pulse.max_gradient_t_per_m <= 0.032
    assert pulse.max_slew_t_per_m_s <= 130.0
    assert np.sum(pulse.gradient_hz_per_m) * pulse.gradient_raster_s == pytest.approx(
        0.0, abs=1e-8
    )

    restored = SpectralSpatialPulse.from_mapping(pulse.to_dict())
    assert restored.name == pulse.name
    assert restored.bands == pulse.bands
    assert np.array_equal(restored.rf_hz, pulse.rf_hz)
    assert np.array_equal(restored.gradient_hz_per_m, pulse.gradient_hz_per_m)

    profile = simulate_spectral_spatial_profile(
        restored,
        positions_m=(-15e-3, 0.0, 15e-3),
        frequency_offsets_hz=(0.0, 150.0),
    )
    assert profile["flip_angle_deg"].shape == (3, 2)
    assert profile["flip_angle_deg"][1, 0] == pytest.approx(30.0, abs=0.2)
    assert profile["flip_angle_deg"][1, 1] > 3.0

    translated = simulate_spectral_spatial_profile(
        restored,
        positions_m=(-15e-3, 0.0, 15e-3),
        frequency_offsets_hz=(0.0,),
        spatial_offset_m=15e-3,
    )["flip_angle_deg"][:, 0]
    assert int(np.argmax(translated)) == 2


def test_rf_only_spectral_spatial_pulse_omits_gradient_and_its_hardware_limits(
    tmp_path,
):
    pulse = _test_pulse().to_dict()
    shared = dict(
        rf_pulse_type="spectral_spatial",
        rf_spectral_spatial_pulse=pulse,
        slice_thickness_m=30e-3,
        nucleus="H1",
        flip_angle_deg=30,
    )
    selected = make_pulseq_slice_selection(**shared)
    rf_only = make_pulseq_slice_selection(
        **shared,
        slice_gradient_enabled=False,
        rephase=True,
        scanner_parameters=ScannerParameters(max_grad_mtm=0.001, max_slew_tms=0.01),
    )
    programs = []
    for name, sequence in (("selected", selected), ("rf_only", rf_only)):
        assert sequence.check_timing()[0]
        path = tmp_path / f"{name}.seq"
        sequence.write(str(path), v141_compat=True)
        programs.append(load_pulseq(path))
    assert not programs[1].gradient_events
    np.testing.assert_array_equal(
        programs[0].rf_events[0].samples_hz, programs[1].rf_events[0].samples_hz
    )


def test_every_generated_sequence_builder_accepts_spectral_spatial_rf():
    pulse = _test_pulse().to_dict()
    shared = {
        "rf_pulse_type": "spectral_spatial",
        "rf_spectral_spatial_pulse": pulse,
        "nucleus": "H1",
    }
    cases = (
        (
            make_pulseq_slice_selection,
            {"slice_thickness_m": 30e-3, "rephase": True, **shared},
            "RFPulseType",
        ),
        (
            make_pulseq_csi,
            {"matrix": (2, 2), "slice_thickness_m": 30e-3, **shared},
            "RFPulseType",
        ),
        (
            make_pulseq_epi,
            {"matrix": (4, 4), "slice_thickness_m": 30e-3, **shared},
            "RFPulseType",
        ),
        (
            make_pulseq_epsi_mge,
            {
                "matrix": (4, 4),
                "echoes": 3,
                "slice_thickness_m": 30e-3,
                **shared,
            },
            "RFPulseType",
        ),
        (
            make_pulseq_flash,
            {"matrix": (4, 4), "slice_thickness_m": 30e-3, **shared},
            "RFPulseType",
        ),
        (
            make_pulseq_spiral,
            {"matrix": (4, 4), "slice_thickness_m": 30e-3, **shared},
            "RFPulseType",
        ),
        (
            make_pulseq_bssfp,
            {
                "fov_m": (0.1, 0.1, 30e-3),
                "matrix": (2, 2, 2),
                "repetition_time_s": 20e-3,
                **shared,
            },
            "RFPulseType",
        ),
        (
            make_pulseq_spectral_selective_bssfp,
            {
                "fov_m": (0.1, 0.1, 30e-3),
                "matrix": (2, 2, 2),
                "target_frequency_offsets_hz": (0.0,),
                "receiver_frequency_offsets_hz": (0.0,),
                "target_metabolite_names": ("Target",),
                "flip_angle_deg": (10.0,),
                "repetition_time_s": 20e-3,
                "use_alpha_half": False,
                "spectral_rf_pulse_type": "spectral_spatial",
                "spectral_rf_spectral_spatial_pulse": pulse,
                "nucleus": "H1",
            },
            "SpectralRFPulseType",
        ),
        (
            make_pulseq_radial_me_bssfp,
            {
                "fov_m": 30e-3,
                "base_resolution": 4,
                "readout_oversampling": 1,
                "spokes_per_measurement": 1,
                "measurements": 1,
                "echoes": 3,
                "repetition_time_s": 30e-3,
                "use_alpha_half": False,
                "use_tip_back": False,
                **shared,
            },
            "RFPulseType",
        ),
        (
            make_pulseq_me_bssfp,
            {
                "fov_m": (0.1, 0.1, 30e-3),
                "matrix": (2, 2, 2),
                "echoes": 3,
                "repetition_time_s": 30e-3,
                "use_alpha_half": False,
                **shared,
            },
            "RFPulseType",
        ),
    )

    for builder, parameters, definition_name in cases:
        sequence = builder(**parameters)
        assert sequence.check_timing()[0]
        assert sequence.definitions[definition_name] == "spectral_spatial"


def test_spectral_spatial_rescaling_rechecks_scanner_limits():
    pulse = _test_pulse().to_dict()

    with pytest.raises(ValueError, match="gradient exceed the scanner amplitude"):
        make_pulseq_epi(
            matrix=(4, 4),
            slice_thickness_m=1e-3,
            rf_pulse_type="spectral_spatial",
            rf_spectral_spatial_pulse=pulse,
            nucleus="H1",
        )


def test_spectral_spatial_pulseq_round_trip_compiles(tmp_path):
    pulse = _test_pulse().to_dict()
    sequence = make_pulseq_epi(
        matrix=(4, 4),
        slice_thickness_m=30e-3,
        rf_pulse_type="spectral_spatial",
        rf_spectral_spatial_pulse=pulse,
        nucleus="H1",
    )
    path = tmp_path / "spectral_spatial_epi.seq"

    sequence.write(path)
    program = load_pulseq(path)
    compiled = SequenceCompiler().compile(program)

    assert program.rf_events
    assert program.gradient_events
    assert compiled.n_intervals > 0
    assert program.metadata["definitions"]["RFPulseType"] == "spectral_spatial"
    assert program.metadata["definitions"]["RFSpectralSpatialPhaseType"] == "linear"
    assert (
        program.metadata["definitions"]["RFSpectralSpatialImplementation"]
        == "native-python-larson-kerr-flyback-v2"
    )


def _gradient_samples(event, times_s):
    if event is None:
        return np.zeros_like(times_s)
    if event.type == "trap":
        corners = event.delay + np.cumsum(
            [0.0, event.rise_time, event.flat_time, event.fall_time]
        )
        amplitudes = [0.0, event.amplitude, event.amplitude, 0.0]
    else:
        corners = event.delay + np.concatenate(([0.0], event.tt, [event.shape_dur]))
        amplitudes = np.concatenate(([event.first], event.waveform, [event.last]))
    return np.interp(times_s, corners, amplitudes, left=0.0, right=0.0)


def _slice_profile_at_first_echo(sequence, positions_m, raster_s=1e-6):
    """Rotate spins along the slice axis from the first RF to the first echo.

    Only the slice-axis gradient is played, so the phase across the slice is
    the excitation phase left after slice refocusing. The first ADC centre
    lies after every rephaser; for radial spokes along the slab axis it is
    also where the readout prephaser has been undone. Sequences without an
    ADC are followed to their end.
    """
    block_ids = list(sequence.block_events)
    blocks = [sequence.get_block(block_id) for block_id in block_ids]
    first = next(
        index
        for index, block in enumerate(blocks)
        if block.rf is not None and np.any(np.abs(block.rf.signal) > 0)
    )
    axis = next(
        channel
        for channel in "xyz"
        if getattr(blocks[first], f"g{channel}") is not None
    )
    magnetization = np.zeros((positions_m.size, 3))
    magnetization[:, 2] = 1.0
    for block in blocks[first:]:
        duration_s = float(block.block_duration)
        if block.adc is not None:
            duration_s = block.adc.delay + block.adc.num_samples * block.adc.dwell / 2
        times_s = (np.arange(int(round(duration_s / raster_s))) + 0.5) * raster_s
        b1_hz = np.zeros(times_s.size, dtype=complex)
        if block.rf is not None:
            rf_times_s = block.rf.delay + block.rf.t
            b1_hz = np.interp(
                times_s, rf_times_s, block.rf.signal.real, left=0.0, right=0.0
            ) + 1j * np.interp(
                times_s, rf_times_s, block.rf.signal.imag, left=0.0, right=0.0
            )
        gradient_hz_per_m = _gradient_samples(getattr(block, f"g{axis}"), times_s)
        for b1, gradient in zip(b1_hz, gradient_hz_per_m):
            field = (
                2.0
                * np.pi
                * np.column_stack(
                    (
                        np.full(positions_m.size, b1.real),
                        np.full(positions_m.size, b1.imag),
                        gradient * positions_m,
                    )
                )
            )
            norm = np.linalg.norm(field, axis=1, keepdims=True)
            angle = norm * raster_s
            unit = np.divide(field, norm, out=np.zeros_like(field), where=norm > 0)
            magnetization = (
                magnetization * np.cos(angle)
                + np.cross(magnetization, unit) * np.sin(angle)
                + unit
                * np.sum(unit * magnetization, axis=1, keepdims=True)
                * (1.0 - np.cos(angle))
            )
        if block.adc is not None:
            break
    return magnetization[:, 0] + 1j * magnetization[:, 1]


def _assert_flat_slice_phase(transverse, tolerance_rad=0.05):
    selected = np.abs(transverse) > 0.5 * np.max(np.abs(transverse))
    assert np.count_nonzero(selected) > 10
    phase = np.unwrap(np.angle(transverse[selected]))
    assert np.ptp(phase) < tolerance_rad, f"slice phase spans {np.ptp(phase):.3f} rad"


def test_spectral_spatial_slice_rephaser_flattens_simulated_slice_phase(tmp_path):
    from blochsimulator import BlochSimulator
    from blochsimulator.phantom import Phantom

    pulse = _test_pulse()
    sequence = make_pulseq_slice_selection(
        rf_pulse_type="spectral_spatial",
        rf_spectral_spatial_pulse=pulse.to_dict(),
        slice_thickness_m=30e-3,
        flip_angle_deg=30.0,
        rephase=True,
        nucleus="H1",
    )
    path = tmp_path / "spsp_rephased.seq"
    sequence.write(str(path), v141_compat=True)
    shape = (1, 1, 121)
    phantom = Phantom(
        shape, (0.01, 0.01, 0.06), np.full(shape, 1e6), np.full(shape, 1e6)
    )
    result = BlochSimulator(use_parallel=False).simulate_sequence(
        load_pulseq(path), phantom, simulation_timestep_s=2e-6
    )

    # The flyback lobes leave half a lobe of moment after every subpulse,
    # although the pulse's net gradient area is zero.
    metrics = pulse.design_metadata["metrics"]
    lobe = pulse.gradient_hz_per_m[: int(metrics["positive_lobe_samples"])]
    assert sequence.get_block(2).gz.area == pytest.approx(
        np.sum(lobe) * pulse.gradient_raster_s / 2.0, rel=1e-2
    )
    _assert_flat_slice_phase(result.mx[0, 0] + 1j * result.my[0, 0])


@pytest.mark.parametrize(
    "builder,parameters",
    [
        (make_pulseq_slice_selection, {"slice_thickness_m": 30e-3, "rephase": True}),
        (make_pulseq_csi, {"matrix": (1, 1), "slice_thickness_m": 30e-3}),
        (make_pulseq_epi, {"matrix": (4, 4), "slice_thickness_m": 30e-3}),
        (
            make_pulseq_epsi_mge,
            {"matrix": (4, 4), "echoes": 3, "slice_thickness_m": 30e-3},
        ),
        (make_pulseq_flash, {"matrix": (4, 4), "slice_thickness_m": 30e-3}),
        (make_pulseq_spiral, {"matrix": (4, 4), "slice_thickness_m": 30e-3}),
        (
            make_pulseq_bssfp,
            {
                "fov_m": (0.1, 0.1, 30e-3),
                "matrix": (2, 2, 1),
                "repetition_time_s": 20e-3,
                "dummy_repetitions": 0,
                "use_alpha_half": False,
            },
        ),
        (
            make_pulseq_spectral_selective_bssfp,
            {
                "fov_m": (0.1, 0.1, 30e-3),
                "matrix": (2, 2, 1),
                "target_frequency_offsets_hz": (0.0,),
                "receiver_frequency_offsets_hz": (0.0,),
                "target_metabolite_names": ("Target",),
                "flip_angle_deg": (30.0,),
                "repetition_time_s": 20e-3,
                "use_alpha_half": False,
            },
        ),
        (
            make_pulseq_me_bssfp,
            {
                "fov_m": (0.1, 0.1, 30e-3),
                "matrix": (2, 2, 1),
                "echoes": 3,
                "repetition_time_s": 30e-3,
                "encoding_duration_s": 1e-3,
                "use_alpha_half": False,
            },
        ),
        (
            make_pulseq_radial_me_bssfp,
            {
                "fov_m": 30e-3,
                "base_resolution": 4,
                "readout_oversampling": 1,
                "spokes_per_measurement": 1,
                "measurements": 1,
                "echoes": 3,
                "repetition_time_s": 30e-3,
                "prephaser_duration_s": 1e-3,
                "use_alpha_half": False,
                "use_tip_back": False,
            },
        ),
    ],
    ids=lambda value: getattr(value, "__name__", ""),
)
def test_every_builder_refocuses_spectral_spatial_slice_phase(builder, parameters):
    pulse = _test_pulse().to_dict()
    if builder is make_pulseq_spectral_selective_bssfp:
        rf = {
            "spectral_rf_pulse_type": "spectral_spatial",
            "spectral_rf_spectral_spatial_pulse": pulse,
        }
    else:
        rf = {
            "rf_pulse_type": "spectral_spatial",
            "rf_spectral_spatial_pulse": pulse,
            "flip_angle_deg": 30.0,
        }
    sequence = builder(**parameters, **rf, nucleus="H1")
    assert sequence.check_timing()[0]

    transverse = _slice_profile_at_first_echo(sequence, np.linspace(-0.03, 0.03, 121))

    _assert_flat_slice_phase(transverse)


def test_metabolite_profiles_run_for_a_predefined_peak_set(qt_application):
    dialog = SpectralSpatialPulseDesignerDialog(
        scanner_parameters=ScannerParameters(),
        nucleus="H1",
        field_strength_t=3.0,
        target_thickness_m=30e-3,
    )

    dialog.subpulse_count.setValue(3)
    dialog.sim_position_points.setValue(9)
    dialog.sim_frequency_points.setValue(17)
    dialog._design()
    assert dialog.pulse() is not None

    dialog._simulate_quick_object()
    assert len(dialog.object_plot.listDataItems()) >= 3
    assert "metabolite profiles" in dialog.status.text()

    dialog.close()
    dialog.deleteLater()
    qt_application.processEvents()


def test_spsp_designer_exposes_band_parameters_and_continuous_map(qt_application):
    dialog = SpectralSpatialPulseDesignerDialog(
        scanner_parameters=ScannerParameters(),
        nucleus="C13",
        field_strength_t=3.0,
        target_thickness_m=10e-3,
    )

    assert dialog.tabs.tabText(0) == "Excitation map"
    assert [dialog.bands.columnWidth(column) for column in range(6)] == [
        110,
        80,
        80,
        110,
        75,
        75,
    ]
    assert "Pyruvate" in dialog.object_description.text()
    assert "Lactate" in dialog.object_description.text()
    assert "continuous frequency sweep" in dialog.object_description.text()
    assert (
        dialog.open_larson_reference_button.text() == "Open LarsonLab reference toolbox"
    )
    assert "LarsonLab/Spectral-Spatial-RF-Pulse-Design" in (
        dialog.open_larson_reference_button.toolTip()
    )
    assert dialog.sim_position_points.value() == 101
    assert dialog.sim_position_points.maximum() == 2001
    assert dialog.sim_frequency_min_hz.value() == pytest.approx(-1500.0)
    assert dialog.sim_frequency_max_hz.value() == pytest.approx(1500.0)
    assert dialog.sim_frequency_points.value() == 513
    assert dialog.sim_frequency_points.maximum() == 8193
    assert dialog.auto_simulation_range.isChecked()
    assert not any(control.isEnabled() for control in dialog.simulation_range_controls)
    assert not any(
        label.isEnabled() for label in dialog.simulation_range_control_labels
    )

    dialog.subpulse_count.setValue(3)
    dialog.sim_position_points.setValue(33)
    dialog.sim_frequency_points.setValue(81)
    dialog._design()
    dialog.wait_for_response_simulation()
    assert dialog.response_image.image.shape == (33, 81)
    assert "33 positions" in dialog.response_description.text()
    assert "81 frequency offsets" in dialog.response_description.text()
    assert "pass-band RMS error" in dialog.validation_metrics.text()

    dialog.auto_simulation_range.setChecked(False)
    assert all(control.isEnabled() for control in dialog.simulation_range_controls)
    assert all(label.isEnabled() for label in dialog.simulation_range_control_labels)
    dialog.sim_position_min_mm.setValue(-8.0)
    dialog.sim_position_max_mm.setValue(12.0)
    dialog.sim_position_points.setValue(17)
    dialog.sim_frequency_min_hz.setValue(-120.0)
    dialog.sim_frequency_max_hz.setValue(180.0)
    dialog.sim_frequency_points.setValue(31)
    dialog._simulate_requested_range()
    dialog.wait_for_response_simulation()
    assert dialog.response_image.image.shape == (17, 31)

    dialog.close()
    dialog.deleteLater()
    qt_application.processEvents()


def test_spsp_designer_reopens_without_blocking_and_reuses_cached_map(
    qt_application, monkeypatch
):
    import blochsimulator.ui.spectral_spatial_designer as designer

    dialog = SpectralSpatialPulseDesignerDialog(
        scanner_parameters=ScannerParameters(),
        nucleus="C13",
        field_strength_t=3.0,
        target_thickness_m=10e-3,
    )
    dialog.subpulse_count.setValue(3)
    dialog._design()
    dialog.wait_for_response_simulation()
    pulse = dialog.pulse()
    assert dialog.response_image.image.shape == (101, 513)
    dialog.close()

    calls = []
    original = designer.simulate_spectral_spatial_profile

    def counting(pulse, positions, frequencies, **kwargs):
        calls.append(np.size(positions) * np.size(frequencies))
        return original(pulse, positions, frequencies, **kwargs)

    monkeypatch.setattr(designer, "simulate_spectral_spatial_profile", counting)
    reopened = SpectralSpatialPulseDesignerDialog(
        scanner_parameters=ScannerParameters(),
        nucleus="C13",
        field_strength_t=3.0,
        target_thickness_m=10e-3,
        initial_pulse=pulse,
    )
    assert calls == []
    assert reopened._response_thread is None
    assert reopened.response_image.image.shape == (101, 513)
    assert len(reopened.object_plot.listDataItems()) >= 3
    reopened.close()


def test_spsp_designer_map_runs_in_background_and_can_be_cancelled(qt_application):
    dialog = SpectralSpatialPulseDesignerDialog(
        scanner_parameters=ScannerParameters(),
        nucleus="C13",
        field_strength_t=3.0,
        target_thickness_m=10e-3,
    )
    dialog.subpulse_count.setValue(3)
    dialog.sim_position_points.setValue(301)
    dialog.sim_frequency_points.setValue(4001)
    dialog._design()
    assert dialog._response_thread is not None
    assert dialog.response_progress_row.isVisibleTo(dialog)
    dialog._cancel_response_simulation()
    dialog.wait_for_response_simulation()
    assert dialog._response_thread is None
    assert dialog._last_response is None
    assert "cancelled" in dialog.response_description.text()
    dialog.close()


def test_spsp_automatic_range_covers_every_active_peak(qt_application):
    dialog = SpectralSpatialPulseDesignerDialog(
        scanner_parameters=ScannerParameters(),
        nucleus="C13",
        field_strength_t=3.0,
        sequence_reference_ppm=171.076,
        target_thickness_m=10e-3,
    )
    dialog.subpulse_count.setValue(3)
    dialog.sim_position_points.setValue(9)
    dialog.sim_frequency_points.setValue(17)
    dialog._design()
    dialog.wait_for_response_simulation()
    alias = dialog.pulse().spectral_alias_bandwidth_hz
    offsets = [peak.carrier_offset_hz for peak in dialog._resolved_peaks()]
    assert max(offsets) - min(offsets) > alias
    assert dialog.sim_frequency_min_hz.value() < min(min(offsets), -alias / 2)
    assert dialog.sim_frequency_max_hz.value() > max(max(offsets), alias / 2)
    dialog.close()


def test_spsp_metabolite_table_supports_c13_presets_and_editable_peaks(
    qt_application,
):
    dialog = SpectralSpatialPulseDesignerDialog(
        scanner_parameters=ScannerParameters(),
        nucleus="C13",
        field_strength_t=3.0,
        sequence_reference_ppm=171.076,
        carrier_offset_hz=10.0,
        target_thickness_m=30e-3,
    )

    labels = [
        dialog.object_preset.itemText(i) for i in range(dialog.object_preset.count())
    ]
    assert any("Pyruvate metabolism" in label for label in labels)
    assert any("Fumarate / malate" in label for label in labels)
    assert any("Common C13 metabolites" in label for label in labels)
    assert [dialog.peak_table.horizontalHeaderItem(i).text() for i in range(6)] == [
        "Use",
        "Peak",
        "ppm",
        "Seq. (Hz)",
        "Carrier (Hz)",
        "Rel. flip",
    ]

    peaks = dialog._resolved_peaks()
    assert peaks[0].name == "Pyruvate"
    assert peaks[0].sequence_offset_hz == pytest.approx(0.0)
    assert peaks[0].carrier_offset_hz == pytest.approx(-10.0)
    assert any(peak.name == "Lactate" for peak in peaks)

    dialog.peak_table.item(0, 2).setText("172.076")
    edited = dialog._resolved_peaks()[0]
    assert edited.sequence_offset_hz == pytest.approx(float(ppm_to_hz(1.0, 3.0, "C13")))

    dialog.close()
    dialog.deleteLater()
    qt_application.processEvents()


def test_peak_linked_spsp_is_retargeted_after_sequence_reference_change(
    qt_application,
):
    original = SpectralSpatialPulseDesignerDialog(
        scanner_parameters=ScannerParameters(),
        nucleus="C13",
        field_strength_t=3.0,
        sequence_reference_ppm=171.076,
        carrier_offset_hz=0.0,
        target_thickness_m=30e-3,
    )
    for row in range(1, original.peak_table.rowCount()):
        original.peak_table.item(row, 0).setCheckState(Qt.Unchecked)
    original._replace_bands_with_preset()
    original.subpulse_count.setValue(3)
    original.sim_position_points.setValue(9)
    original.sim_frequency_points.setValue(17)
    original._design()
    pulse = original.pulse()
    assert pulse is not None
    assert pulse.design_metadata["frequency_prescription"]["bands_linked_to_peak_table"]

    retargeted = SpectralSpatialPulseDesignerDialog(
        scanner_parameters=ScannerParameters(),
        nucleus="C13",
        field_strength_t=3.0,
        sequence_reference_ppm=170.076,
        carrier_offset_hz=10.0,
        target_thickness_m=30e-3,
        initial_pulse=pulse,
    )
    band = retargeted._band_values()[0]
    expected_center = float(ppm_to_hz(1.0, 3.0, "C13")) - 10.0
    assert band.center_hz == pytest.approx(expected_center)
    assert retargeted._design_dirty
    assert "recalculated" in retargeted.status.text()

    original.close()
    original.deleteLater()
    retargeted.close()
    retargeted.deleteLater()
    qt_application.processEvents()


def test_sequence_nucleus_is_always_available_beside_sequence_reference(
    qt_application,
):
    widget = SequenceSimulationWidget()
    widget._update_frequency_reference_info()

    assert widget.nucleus.objectName() == "sequence_nucleus"
    assert widget.nucleus.isEnabled()
    assert not widget.nucleus.isHidden()
    assert not widget.nucleus_label.isHidden()

    widget.close()
    widget.deleteLater()
    qt_application.processEvents()


def test_spsp_target_center_and_transition_width_round_trip():
    pulse = design_spectral_spatial_pulse(
        bands=(
            SpectralBand(
                0.0,
                80.0,
                name="Target",
                transition_width_hz=25.0,
            ),
        ),
        flip_angle_deg=20.0,
        slice_thickness_m=30e-3,
        spatial_center_m=12e-3,
        spatial_transition_m=4e-3,
        spatial_time_bandwidth_product=2.0,
        subpulse_duration_s=0.2e-3,
        subpulse_count=3,
        nucleus="H1",
        max_gradient_t_per_m=0.032,
        max_slew_t_per_m_s=130.0,
    )

    restored = SpectralSpatialPulse.from_mapping(pulse.to_dict())
    assert restored.reference_spatial_center_m == pytest.approx(12e-3)
    assert restored.reference_spatial_transition_m == pytest.approx(4e-3)
    assert restored.bands[0].transition_width_hz == pytest.approx(25.0)
    profile = simulate_spectral_spatial_profile(
        restored,
        positions_m=(-12e-3, 0.0, 12e-3),
        frequency_offsets_hz=(0.0,),
    )["flip_angle_deg"][:, 0]
    assert int(np.argmax(profile)) == 2


def test_alias_conflicts_are_reported_and_sampling_rate_search_is_quantized():
    conflicting = (
        SpectralBand(0.0, 40.0, amplitude=1.0, transition_width_hz=10.0),
        SpectralBand(1000.0, 40.0, amplitude=0.5, transition_width_hz=10.0),
    )
    result = check_spectral_alias_compatibility(conflicting, 1000.0)
    assert not result.compatible
    assert "different pass target" in result.reason

    stop_conflict = check_spectral_alias_compatibility(
        (conflicting[0],),
        1000.0,
        stopbands=(SpectralStopBand(950.0, 1050.0),),
    )
    assert not stop_conflict.compatible
    assert "stop band" in stop_conflict.reason

    rate, compatible, samples = find_compatible_spectral_sampling_rate(
        conflicting,
        minimum_period_samples=90,
        maximum_period_samples=110,
        raster_s=10e-6,
    )
    assert compatible.compatible
    assert rate == pytest.approx(1.0 / (samples * 10e-6))
    assert 90 <= samples <= 110


def test_flyback_rewinder_is_independent_balanced_and_metadata_is_reproducible():
    pulse = _test_pulse()
    metrics = pulse.design_metadata["metrics"]
    positive_count = metrics["positive_lobe_samples"]
    rewinder_count = metrics["rewinder_samples"]
    period_count = metrics["period_samples"]
    gradient = pulse.gradient_hz_per_m[:period_count]

    assert positive_count != rewinder_count
    assert np.all(gradient[:positive_count] > 0)
    assert np.all(gradient[positive_count : positive_count + rewinder_count] < 0)
    assert np.sum(gradient) * pulse.gradient_raster_s == pytest.approx(0.0, abs=1e-9)
    assert pulse.design_metadata["implementation"].endswith("flyback-v2")
    assert pulse.design_metadata["matlab_reference_validated"] is False
    assert len(pulse.design_metadata["spectral_coefficients"]) == 3
    assert pulse.design_metadata["inputs"]["nucleus"] == "H1"


def test_linear_and_minimum_phase_filters_have_distinct_energy_delay():
    common = dict(
        bands=(
            SpectralBand(
                0.0,
                100.0,
                transition_width_hz=50.0,
                passband_ripple=0.05,
                stopband_ripple=0.05,
            ),
        ),
        flip_angle_deg=5.0,
        slice_thickness_m=30e-3,
        spatial_transition_m=6e-3,
        spatial_time_bandwidth_product=2.0,
        subpulse_duration_s=0.3e-3,
        subpulse_count=31,
        nucleus="H1",
        max_gradient_t_per_m=0.032,
        max_slew_t_per_m_s=130.0,
        max_duration_s=0.1,
    )
    linear = design_spectral_spatial_pulse(**common, spectral_phase_type="linear")
    minimum = design_spectral_spatial_pulse(**common, spectral_phase_type="minimum")

    def coefficient_centroid(pulse):
        coefficients = np.asarray(
            [
                value["real"] + 1j * value["imag"]
                for value in pulse.design_metadata["spectral_coefficients"]
            ]
        )
        energy = np.abs(coefficients) ** 2
        return float(np.sum(np.arange(energy.size) * energy) / np.sum(energy))

    assert coefficient_centroid(minimum) < coefficient_centroid(linear)
    assert linear.spectral_phase_type == "linear"
    assert minimum.spectral_phase_type == "minimum"
    assert not np.array_equal(linear.rf_hz, minimum.rf_hz)


def test_complex_multiband_targets_preserve_relative_amplitude_and_phase():
    pulse = design_spectral_spatial_pulse(
        bands=(
            SpectralBand(
                0.0,
                80.0,
                amplitude=1.0,
                phase_deg=0.0,
                transition_width_hz=40.0,
                passband_ripple=0.05,
                stopband_ripple=0.05,
            ),
            SpectralBand(
                300.0,
                80.0,
                amplitude=0.5,
                phase_deg=90.0,
                transition_width_hz=40.0,
                passband_ripple=0.05,
                stopband_ripple=0.05,
            ),
        ),
        flip_angle_deg=5.0,
        slice_thickness_m=30e-3,
        spatial_transition_m=6e-3,
        spatial_time_bandwidth_product=2.0,
        subpulse_duration_s=0.3e-3,
        subpulse_count=31,
        nucleus="H1",
        max_gradient_t_per_m=0.032,
        max_slew_t_per_m_s=130.0,
        max_duration_s=0.1,
    )
    response = simulate_spectral_spatial_profile(
        pulse, positions_m=(0.0,), frequency_offsets_hz=(0.0, 300.0)
    )
    flips = response["flip_angle_deg"][0]
    phases = np.rad2deg(np.angle(response["mxy"][0]))
    relative_phase = (phases[1] - phases[0] + 180.0) % 360.0 - 180.0

    assert flips[1] / flips[0] == pytest.approx(0.5, abs=0.03)
    assert relative_phase == pytest.approx(90.0, abs=8.0)


def test_spatial_transition_and_hardware_constraints_are_effective():
    common = dict(
        bands=(SpectralBand(0.0, 100.0, transition_width_hz=30.0),),
        flip_angle_deg=20.0,
        slice_thickness_m=30e-3,
        spatial_time_bandwidth_product=2.0,
        subpulse_duration_s=0.3e-3,
        subpulse_count=9,
        nucleus="H1",
        max_gradient_t_per_m=0.032,
        max_slew_t_per_m_s=130.0,
    )
    sharp = design_spectral_spatial_pulse(**common, spatial_transition_m=2e-3)
    broad = design_spectral_spatial_pulse(**common, spatial_transition_m=8e-3)
    assert not np.allclose(sharp.rf_hz, broad.rf_hz)

    with pytest.raises(ValueError, match="B1 limit"):
        design_spectral_spatial_pulse(
            **common, spatial_transition_m=6e-3, max_b1_t=1e-9
        )
    with pytest.raises(ValueError, match="minimum duration"):
        design_spectral_spatial_pulse(
            **common, spatial_transition_m=6e-3, max_duration_s=0.1e-3
        )
    with pytest.raises(NotImplementedError, match="echo_planar"):
        design_spectral_spatial_pulse(
            **common, spatial_transition_m=6e-3, method="echo_planar"
        )


def test_bloch_validation_covers_edges_stopband_alias_and_flip_scaling():
    pulse = design_spectral_spatial_pulse(
        bands=(
            SpectralBand(
                0.0,
                100.0,
                transition_width_hz=100.0,
                passband_ripple=0.05,
                stopband_ripple=0.05,
            ),
        ),
        flip_angle_deg=10.0,
        slice_thickness_m=30e-3,
        spatial_transition_m=6e-3,
        spatial_time_bandwidth_product=8.0,
        spatial_passband_ripple=0.05,
        spatial_stopband_ripple=0.05,
        subpulse_duration_s=0.6e-3,
        subpulse_count=31,
        nucleus="H1",
        max_gradient_t_per_m=0.032,
        max_slew_t_per_m_s=130.0,
        max_duration_s=50e-3,
    )
    alias_hz = pulse.spectral_alias_bandwidth_hz
    response = simulate_spectral_spatial_profile(
        pulse,
        positions_m=(0.0, 15e-3, 21e-3),
        frequency_offsets_hz=(0.0, 50.0, 200.0, alias_hz),
    )["flip_angle_deg"]

    assert pulse.design_metadata["metrics"]["all_filter_specifications_met"]
    assert response[0, 0] == pytest.approx(10.0, abs=0.1)
    assert response[0, 1] > 9.5  # spectral pass-band edge
    assert response[0, 2] < 0.1  # spectral stop band
    assert response[0, 3] > 9.5  # first periodic alias band
    assert response[1, 0] > 9.0  # spatial pass-band edge
    assert response[2, 0] < 0.5  # beyond the spatial transition

    for requested_flip in (5.0, 20.0):
        achieved = simulate_spectral_spatial_profile(
            pulse,
            positions_m=(0.0,),
            frequency_offsets_hz=(0.0,),
            flip_angle_deg=requested_flip,
        )["flip_angle_deg"][0, 0]
        assert achieved == pytest.approx(requested_flip, abs=0.1)


def test_adapted_larson_flyback_example_specifications_regress_constraints():
    # Adapted from demo_general.m: 1.5 T H1 water pass band and fat stop band.
    hz_per_ppm = 15000.0 * 4258.0 * 1e-6
    water_fat = design_spectral_spatial_pulse(
        bands=(
            SpectralBand(
                0.0,
                2.0 * 0.5 * hz_per_ppm,
                transition_width_hz=15.0,
                passband_ripple=0.02,
                stopband_ripple=0.02,
            ),
        ),
        stopbands=(
            SpectralStopBand(
                (0.9 - 0.5 - 4.7) * hz_per_ppm,
                (1.3 + 0.5 - 4.7) * hz_per_ppm,
                maximum_amplitude=0.02,
                name="Fat",
            ),
        ),
        flip_angle_deg=30.0,
        slice_thickness_m=10e-3,
        spatial_time_bandwidth_product=4.0,
        spatial_transition_m=3e-3,
        subpulse_duration_s=0.3e-3,
        subpulse_count=11,
        nucleus="H1",
        max_gradient_t_per_m=0.04,
        max_slew_t_per_m_s=150.0,
        max_duration_s=16e-3,
        max_b1_t=500e-6,
        spectral_phase_type="minimum",
    )

    # Adapted from demo_mband_phs.m: centered C13 urea/pyruvate/alanine
    # bands, 6 cm slice, TBW 4, linear-phase whole flyback, 24 ms limit.
    carbon_hz_per_half_ppm = 0.5e-6 * 3.0 * NUCLEUS_GAMMA_HZ_PER_T["C13"]
    reference_centers = (-465.0, -230.0, -45.0)
    reference_midpoint = (reference_centers[0] + reference_centers[-1]) / 2.0
    phase_demo = design_spectral_spatial_pulse(
        bands=tuple(
            SpectralBand(
                center - reference_midpoint,
                2.0 * multiplier * carbon_hz_per_half_ppm,
                amplitude=amplitude,
                transition_width_hz=10.0,
                passband_ripple=0.05,
                stopband_ripple=0.05,
            )
            for center, multiplier, amplitude in (
                (-465.0, 2.0, 0.5),
                (-230.0, 2.0, 0.5),
                (-45.0, 1.5, 1.0),
            )
        ),
        flip_angle_deg=12.0,
        slice_thickness_m=60e-3,
        spatial_time_bandwidth_product=4.0,
        spatial_transition_m=10e-3,
        subpulse_duration_s=0.3e-3,
        subpulse_count=21,
        nucleus="C13",
        max_gradient_t_per_m=0.04,
        max_slew_t_per_m_s=150.0,
        max_duration_s=24e-3,
        max_b1_t=500e-6,
        spectral_phase_type="linear",
    )

    for pulse, duration_limit in ((water_fat, 16e-3), (phase_demo, 24e-3)):
        assert pulse.duration_s <= duration_limit
        assert pulse.max_gradient_t_per_m <= 0.04
        assert pulse.max_slew_t_per_m_s <= 150.0
        assert pulse.peak_b1_t <= 500e-6
        assert pulse.design_metadata["matlab_reference_validated"] is False


def test_sequence_mode_restores_and_exposes_spectral_spatial_design(qt_application):
    widget = SequenceSimulationWidget()
    pulse = _test_pulse()

    assert widget.restore_spectral_spatial_pulse_state(pulse.to_dict())
    for prefix in (
        "epi",
        "epsi",
        "csi",
        "flash",
        "bssfp",
        "ss_bssfp",
        "radial_me",
        "me_bssfp",
    ):
        getattr(widget, f"{prefix}_rf_pulse_type").setCurrentText(
            "Spectral-spatial Designer"
        )
        parameters = widget._shared_rf_parameters(prefix)
        assert parameters["rf_pulse_type"] == "spectral_spatial"
        assert parameters["rf_spectral_spatial_pulse"]["name"] == "Test SPSP"
        assert not getattr(widget, f"{prefix}_rf_duration_ms").isEnabled()

    state = widget.spectral_spatial_pulse_state()
    assert state["name"] == "Test SPSP"
    assert np.array_equal(state["rf_hz"], pulse.rf_hz)

    widget.close()
    widget.deleteLater()
    qt_application.processEvents()


def test_radial_me_bssfp_spectral_spatial_tip_back_is_slab_refocused(tmp_path):
    pulse = _test_pulse().to_dict()
    sequence = make_pulseq_radial_me_bssfp(
        fov_m=30e-3,
        base_resolution=4,
        readout_oversampling=1,
        spokes_per_measurement=2,
        measurements=1,
        echoes=3,
        repetition_time_s=30e-3,
        use_alpha_half=True,
        use_tip_back=True,
        rf_pulse_type="spectral_spatial",
        rf_spectral_spatial_pulse=pulse,
        nucleus="H1",
    )
    assert sequence.check_timing()[0]
    path = tmp_path / "radial_spsp_tip_back.seq"
    sequence.write(path)
    program = load_pulseq(path)
    repetition_time = float(program.metadata["definitions"]["TR"])
    rf_events = program.rf_events
    centres = np.array(
        [
            event.start_s + 0.5 * event.samples_hz.size * event.raster_s
            for event in rf_events
        ]
    )

    # alpha/2, two acquired spokes, the extra alpha and the alpha/2 tip-back.
    assert len(rf_events) == 5
    assert centres[-1] - centres[-2] == pytest.approx(repetition_time / 2, abs=2e-6)
    assert np.max(np.abs(rf_events[-1].samples_hz)) == pytest.approx(
        0.5 * np.max(np.abs(rf_events[-2].samples_hz)), rel=1e-3
    )
    # The tip-back slab gradient is refocused like every other SPSP pulse, so
    # spins across the slab are stored back along z alike.
    positions = np.zeros((3, 3))
    positions[:, 2] = (-10e-3, 0.0, 10e-3)
    result = BlochSimulator().simulate_sequence_probes(
        program,
        positions,
        np.array([0.0]),
        checkpoints_s=[program.duration_s],
        t1_s=50.0,
        t2_s=10.0,
    )
    final = result.magnetization[-1, :, 0]
    stored_fraction = final[:, 2] / np.linalg.norm(final, axis=-1)
    assert np.all(stored_fraction > 0.999)
