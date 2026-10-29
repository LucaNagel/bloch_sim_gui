from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtWidgets import QPushButton

from blochsimulator.phantom import Phantom
from blochsimulator.sequence import (
    GradientEvent,
    RFEvent,
    SequenceCompiler,
    SequenceProgram,
    SpectralBand,
    SpectralSpatialPulseLibrary,
    SpectralSpatialSchedule,
    SpectralSpatialScheduleEntry,
    design_spectral_spatial_pulse,
    load_pulseq,
    make_pulseq_epi,
    make_pulseq_flash,
    migrate_spectral_spatial_state,
    simulate_spectral_spatial_profile,
    simulate_spectral_spatial_schedule_preview,
    spectral_spatial_peaks,
    spectral_spatial_schedule_warnings,
    validate_spectral_spatial_schedule,
)
from blochsimulator.simulator import BlochSimulator
from blochsimulator.spectral_phantom import ChemicalSpecies, SpectralPhantom
from blochsimulator.project_io import load_project, save_project
from blochsimulator.sequence import ScannerParameters
from blochsimulator.ui.spectral_spatial_designer import (
    SpectralSpatialScheduleDialog,
    SpectralSpatialSchedulePreviewDialog,
)
from blochsimulator.units import ppm_to_hz


def _pulse(name="Pyruvate", center_hz=0.0, count=3, max_b1_t=500e-6):
    return design_spectral_spatial_pulse(
        bands=(SpectralBand(center_hz, 100.0, name=name),),
        flip_angle_deg=20.0,
        slice_thickness_m=30e-3,
        spatial_time_bandwidth_product=2.0,
        subpulse_duration_s=0.2e-3,
        subpulse_count=count,
        nucleus="C13",
        max_gradient_t_per_m=0.032,
        max_slew_t_per_m_s=130.0,
        max_b1_t=max_b1_t,
        name=name,
    )


def _library_and_schedule():
    library = SpectralSpatialPulseLibrary()
    pyruvate = library.add(_pulse(), pulse_id="pyruvate-pulse")
    lactate = library.add(
        _pulse("Lactate", center_hz=395.0, count=5), pulse_id="lactate-pulse"
    )
    schedule = SpectralSpatialSchedule(
        [
            SpectralSpatialScheduleEntry(
                pyruvate,
                display_name="Pyr 10 deg",
                target_metabolite="Pyruvate",
                flip_angle_deg=10.0,
            ),
            SpectralSpatialScheduleEntry(
                lactate,
                display_name="Lac half",
                target_metabolite="Lactate",
                amplitude_scale=0.5,
                repeat_count=2,
            ),
        ]
    )
    return library, schedule


def test_library_schedule_serialization_and_legacy_migration():
    library, schedule = _library_and_schedule()
    restored_library = SpectralSpatialPulseLibrary.from_mapping(library.to_dict())
    restored_schedule = SpectralSpatialSchedule.from_mapping(schedule.to_dict())

    assert tuple(restored_library.pulses) == ("pyruvate-pulse", "lactate-pulse")
    assert len(restored_schedule.expanded_entries()) == 3
    assert restored_schedule.entries[0].flip_angle_deg == 10.0

    migrated_library, migrated_schedule, migrated = migrate_spectral_spatial_state(
        legacy_pulse=_pulse().to_dict()
    )
    assert migrated
    assert tuple(migrated_library.pulses) == ("legacy-single-pulse",)
    assert migrated_schedule.entries[0].pulse_id == "legacy-single-pulse"


def test_project_archive_round_trips_library_and_schedule_arrays(tmp_path):
    library, schedule = _library_and_schedule()
    path = tmp_path / "scheduled_spsp.blochproj"
    save_project(
        path,
        {
            "sequence_spectral_spatial_pulse_library": library.to_dict(),
            "sequence_spectral_spatial_schedule": schedule.to_dict(),
            "sequence_spectral_spatial_pulse": library.get("pyruvate-pulse").to_dict(),
        },
    )
    state = load_project(path)["state"]
    restored = SpectralSpatialPulseLibrary.from_mapping(
        state["sequence_spectral_spatial_pulse_library"]
    )
    restored_schedule = SpectralSpatialSchedule.from_mapping(
        state["sequence_spectral_spatial_schedule"]
    )
    np.testing.assert_array_equal(
        restored.get("lactate-pulse").rf_hz,
        library.get("lactate-pulse").rf_hz,
    )
    assert restored_schedule.entries[1].repeat_count == 2


def test_library_and_schedule_edit_operations_preserve_stable_references():
    library, schedule = _library_and_schedule()
    copied_pulse_id = library.duplicate("pyruvate-pulse")
    copied_entry_id = schedule.duplicate(0)
    schedule.move(1, 2)
    removed = schedule.remove(2)
    removed_pulse = library.remove(copied_pulse_id)

    assert copied_entry_id != schedule.entries[0].entry_id
    assert removed.entry_id == copied_entry_id
    assert removed_pulse.name.endswith("copy")
    assert schedule.entries[0].pulse_id == "pyruvate-pulse"


def test_schedule_scaling_is_exact_and_rechecks_b1():
    library, schedule = _library_and_schedule()
    first, second, _ = schedule.expanded_entries()
    assert first.effective_flip_angle_deg(library.get(first.pulse_id)) == 10.0
    assert second.effective_flip_angle_deg(library.get(second.pulse_id)) == 10.0
    validate_spectral_spatial_schedule(
        library,
        schedule,
        nucleus="C13",
        thickness_m=30e-3,
    )

    limited = SpectralSpatialPulseLibrary({"limited": _pulse(max_b1_t=60e-6)})
    excessive = SpectralSpatialSchedule(
        [SpectralSpatialScheduleEntry("limited", flip_angle_deg=10000.0)]
    )
    with pytest.raises(ValueError, match="B1"):
        validate_spectral_spatial_schedule(
            limited, excessive, nucleus="C13", thickness_m=30e-3
        )


def test_phantom_peak_frequencies_use_absolute_sequence_and_carrier_references():
    phantom = SimpleNamespace(
        spectral_reference_ppm=171.076,
        species=(
            ChemicalSpecies("Pyruvate", 0.0, 25.0, 0.3),
            ChemicalSpecies("Lactate", 183.35 - 171.076, 25.0, 0.3),
        ),
        metadata={"sequence_reference_ppm": 170.0},
    )
    peaks = spectral_spatial_peaks(
        phantom=phantom,
        field_strength_t=3.0,
        nucleus="C13",
        sequence_reference_ppm=171.076,
        rf_carrier_offset_hz=100.0,
    )

    assert peaks[0].absolute_ppm == pytest.approx(171.076)
    assert peaks[0].sequence_offset_hz == pytest.approx(0.0)
    assert peaks[0].carrier_offset_hz == pytest.approx(-100.0)
    assert peaks[1].absolute_ppm == pytest.approx(183.35)
    assert all(peak.source == "phantom" for peak in peaks)


def test_schedule_warnings_report_missing_target_and_reference_mismatch():
    library, schedule = _library_and_schedule()
    schedule.entries[0] = SpectralSpatialScheduleEntry(
        "pyruvate-pulse", target_metabolite="Not in phantom"
    )
    peaks = spectral_spatial_peaks(
        presets=(ChemicalSpecies("Pyruvate", 171.076, 25.0, 0.3),),
        field_strength_t=3.0,
        nucleus="C13",
        sequence_reference_ppm=171.076,
    )
    warnings = spectral_spatial_schedule_warnings(
        library,
        schedule,
        peaks,
        phantom_sequence_reference_ppm=170.0,
        sequence_reference_ppm=171.076,
    )
    assert any("not present" in warning for warning in warnings)
    assert any("reference differs" in warning for warning in warnings)


def test_cyclic_schedule_selection_is_deterministic():
    _, schedule = _library_and_schedule()
    assert [schedule.entry_for_index(index)[0] for index in range(7)] == [
        0,
        1,
        2,
        0,
        1,
        2,
        0,
    ]


def test_flash_schedule_is_frame_level_labelled_and_round_trips(tmp_path):
    library, schedule = _library_and_schedule()
    sequence = make_pulseq_flash(
        matrix=(4, 4),
        slice_thickness_m=30e-3,
        echo_time_s=8e-3,
        repetition_time_s=20e-3,
        repetitions=4,
        rf_pulse_type="spectral_spatial",
        rf_spectral_spatial_library=library.to_dict(),
        rf_spectral_spatial_schedule=schedule.to_dict(),
        nucleus="C13",
        scanner_parameters=ScannerParameters(),
    )
    assert sequence.check_timing()[0]
    assert sequence.definitions["SPSPScheduleSelectionLevel"] == "frame"
    assert sequence.definitions["SPSPScheduleFrameIndices"] == [0, 1, 2, 0]

    path = Path(tmp_path) / "scheduled_flash.seq"
    sequence.write(path, v141_compat=True)
    program = load_pulseq(path)
    compiled = SequenceCompiler().compile(program)
    assert compiled.n_intervals > 0
    assert program.metadata["definitions"]["SPSPSchedulePulseIDs"]
    set_labels = program.metadata["adc_label_values"]["SET"]
    assert tuple(set_labels[:4]) == (0, 0, 0, 0)
    assert tuple(set_labels[4:8]) == (1, 1, 1, 1)


def _scheduled_epi(schedule, library, **overrides):
    parameters = dict(
        matrix=(4, 4),
        fov_m=(0.08, 0.08),
        slice_thickness_m=30e-3,
        slice_gap_m=5e-3,
        n_slices=2,
        repetitions=4,
        echo_time_s=20e-3,
        repetition_time_s=None,
        rf_pulse_type="spectral_spatial",
        rf_spectral_spatial_library=library.to_dict(),
        rf_spectral_spatial_schedule=schedule.to_dict(),
        nucleus="C13",
        scanner_parameters=ScannerParameters(),
    )
    parameters.update(overrides)
    return make_pulseq_epi(**parameters)


def test_epi_frame_schedule_shares_one_pulse_across_slices(tmp_path):
    library, schedule = _library_and_schedule()
    sequence = _scheduled_epi(schedule, library)

    assert sequence.check_timing()[0]
    definitions = sequence.definitions
    assert definitions["SPSPScheduleSelectionLevel"] == "frame"
    assert definitions["SPSPScheduleFrameIndices"] == [0, 1, 2, 0]
    assert "all slices" in definitions["SPSPScheduleContrastWarning"]

    path = Path(tmp_path) / "scheduled_epi.seq"
    sequence.write(path, v141_compat=True)
    program = load_pulseq(path)
    set_labels = tuple(program.metadata["adc_label_values"]["SET"])
    # Two slices of four EPI lines per repetition.
    assert set_labels == (0,) * 8 + (1,) * 8 + (2,) * 8 + (0,) * 8
    rf_events = [event for event in program.events if isinstance(event, RFEvent)]
    durations = [round(event.end_s - event.start_s, 6) for event in rf_events]
    pyruvate, lactate = (
        round(library.get(pulse_id).duration_s, 6)
        for pulse_id in ("pyruvate-pulse", "lactate-pulse")
    )
    assert durations == [pyruvate] * 2 + [lactate] * 4 + [pyruvate] * 2
    assert SequenceCompiler().compile(program).n_intervals > 0


def test_epi_tr_schedule_advances_per_slice_and_keeps_echo_time(tmp_path):
    library, schedule = _library_and_schedule()
    tr_schedule = SpectralSpatialSchedule(
        schedule.entries, cyclic=True, selection_level="tr"
    )
    sequence = _scheduled_epi(tr_schedule, library)

    assert sequence.check_timing()[0]
    assert sequence.definitions["SPSPScheduleSelectionLevel"] == "tr"
    assert "slice image" in sequence.definitions["SPSPScheduleContrastWarning"]
    path = Path(tmp_path) / "scheduled_tr_epi.seq"
    sequence.write(path, v141_compat=True)
    program = load_pulseq(path)
    set_labels = program.metadata["adc_label_values"]["SET"]
    assert tuple(set_labels[::4]) == (0, 1, 2, 0, 1, 2, 0, 1)

    # Pulses of different length keep the same TE through per-pulse delays.
    compiled = SequenceCompiler().compile_acquisition(program)
    rf_events = program.rf_events
    samples_per_slice = compiled.adc_times_s.size // len(rf_events)
    k_radius = np.linalg.norm(compiled.adc_gradient_moment_cyc_per_m[:, :2], axis=1)
    pulses = [library.get(f"{name}-pulse") for name in ("pyruvate", "lactate")]
    echo_times = []
    for slice_number, (rf_event, pulse_index) in enumerate(
        zip(rf_events, (0, 1, 1, 0, 1, 1, 0, 1))
    ):
        assert rf_event.end_s - rf_event.start_s == pytest.approx(
            pulses[pulse_index].duration_s, abs=2e-6
        )
        window = slice(
            slice_number * samples_per_slice, (slice_number + 1) * samples_per_slice
        )
        echo_sample = int(np.argmin(k_radius[window]))
        echo_times.append(
            compiled.adc_times_s[window][echo_sample]
            - rf_event.start_s
            - pulses[pulse_index].excitation_center_s
        )
    dwell = float(np.diff(compiled.adc_times_s[:2])[0])
    assert np.ptp(echo_times) <= dwell + 2e-6
    assert np.mean(echo_times) == pytest.approx(20e-3, abs=dwell)


def test_epi_schedule_rejects_variable_flip_angles():
    library, schedule = _library_and_schedule()
    with pytest.raises(ValueError, match="variable flip angles"):
        _scheduled_epi(schedule, library, variable_flip_angle=True)


def test_fast_preview_matches_single_pulse_profile_and_propagates_complex_state():
    pulse = _pulse()
    library = SpectralSpatialPulseLibrary({"p": pulse})
    schedule = SpectralSpatialSchedule(
        [SpectralSpatialScheduleEntry("p", flip_angle_deg=10.0, repeat_count=2)]
    )
    peaks = spectral_spatial_peaks(
        presets=(ChemicalSpecies("Pyruvate", 171.076, 25.0, 0.3),),
        field_strength_t=3.0,
        nucleus="C13",
        sequence_reference_ppm=171.076,
    )
    positions = np.asarray([-5e-3, 0.0, 5e-3])
    preview = simulate_spectral_spatial_schedule_preview(
        library,
        schedule,
        peaks,
        positions,
        pulse_spacing_s=20e-3,
        spoiling="none",
    )
    direct = simulate_spectral_spatial_profile(
        pulse,
        positions,
        (0.0,),
        flip_angle_deg=10.0,
    )
    np.testing.assert_allclose(preview["isolated_mxy"][0], direct["mxy"], atol=1e-12)
    np.testing.assert_allclose(preview["isolated_mz"][0], direct["mz"], atol=1e-12)
    assert not np.allclose(preview["cumulative_mxy"][1], preview["isolated_mxy"][1])


def test_fast_preview_matches_full_event_based_bloch_simulation():
    pulse = _pulse(count=3)
    library = SpectralSpatialPulseLibrary({"p": pulse})
    schedule = SpectralSpatialSchedule(
        [SpectralSpatialScheduleEntry("p", flip_angle_deg=12.0)]
    )
    peaks = (
        {
            "name": "on resonance",
            "absolute_ppm": 0.0,
            "sequence_offset_hz": 0.0,
            "carrier_offset_hz": 0.0,
        },
        {
            "name": "off resonance",
            "absolute_ppm": 0.0,
            "sequence_offset_hz": 35.0,
            "carrier_offset_hz": 35.0,
        },
    )
    positions = np.asarray([-2e-3, 0.0, 2e-3])
    preview = simulate_spectral_spatial_schedule_preview(
        library,
        schedule,
        peaks,
        positions,
        pulse_spacing_s=0.0,
        spoiling="none",
    )

    rf, gradient = pulse.scaled_waveforms(flip_angle_deg=12.0)
    program = SequenceProgram(
        (
            RFEvent(0.0, rf, pulse.rf_raster_s),
            GradientEvent("z", 0.0, gradient, pulse.gradient_raster_s),
        ),
        duration_s=pulse.duration_s,
    )
    probes = BlochSimulator(use_parallel=False).simulate_sequence_probes(
        program,
        np.column_stack(
            (np.zeros(positions.size), np.zeros(positions.size), positions)
        ),
        np.asarray([0.0, 35.0]),
        checkpoints_s=(pulse.duration_s,),
        t1_s=1e12,
        t2_s=1e12,
        simulation_timestep_s=pulse.rf_raster_s,
        sequence_kernel="strict_reference",
    )
    full = probes.magnetization[0]
    np.testing.assert_allclose(
        preview["cumulative_mxy"][0], full[..., 0] + 1j * full[..., 1], atol=2e-10
    )
    np.testing.assert_allclose(preview["cumulative_mz"][0], full[..., 2], atol=2e-10)


def test_simulation_result_retains_schedule_definitions_and_adc_labels():
    program = SequenceProgram(
        (),
        duration_s=1e-3,
        metadata={
            "definitions": {
                "SPSPSchedulePulseIDs": '["pyruvate-pulse"]',
                "SPSPScheduleTargetNames": '["Pyruvate"]',
                "SPSPScheduleAmplitudeScales": [0.5],
            },
            "adc_label_values": {"SET": (0,)},
        },
    )
    phantom = Phantom(
        shape=(1,),
        fov=(0.02,),
        t1_map=np.ones(1),
        t2_map=np.ones(1),
        pd_map=np.ones(1),
    )
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)

    assert result.metadata["adc_label_values"] == {"SET": (0,)}
    assert (
        result.metadata["spectral_spatial_schedule"] == program.metadata["definitions"]
    )


def test_schedule_editor_and_shared_preview_expose_complete_state(qt_application):
    library, schedule = _library_and_schedule()
    dialog = SpectralSpatialScheduleDialog(
        library=library,
        schedule=schedule,
        scanner_parameters=ScannerParameters(),
        nucleus="C13",
        field_strength_t=3.0,
        sequence_reference_ppm=171.076,
        target_thickness_m=30e-3,
    )
    assert dialog.library_table.rowCount() == 2
    assert dialog.schedule_table.rowCount() == 2
    dialog.schedule_table.selectRow(0)
    dialog._duplicate_schedule_entry()
    dialog._move_schedule_entry(1)
    assert len(dialog.pulse_schedule().entries) == 3

    preview = SpectralSpatialSchedulePreviewDialog(
        library=library,
        schedule=schedule,
        nucleus="C13",
        field_strength_t=3.0,
        sequence_reference_ppm=171.076,
        target_thickness_m=30e-3,
        pulse_spacing_s=20e-3,
    )
    assert preview.step.maximum() == 3
    assert preview.overview_table.rowCount() == 3
    assert "preset peaks" in preview.warning_label.text()
    preview._show_step(1)
    previous = next(
        button
        for button in preview.findChildren(QPushButton)
        if button.text() == "Previous"
    )
    previous.click()
    assert preview.step.value() == 1
    preview.close()
    preview.deleteLater()
    dialog.close()
    dialog.deleteLater()
    qt_application.processEvents()


def test_schedule_preview_rejects_a_schedule_without_enabled_steps(qt_application):
    library, _ = _library_and_schedule()
    schedule = SpectralSpatialSchedule()
    with pytest.raises(ValueError, match="no enabled pulse steps"):
        SpectralSpatialSchedulePreviewDialog(
            library=library,
            schedule=schedule,
            nucleus="C13",
            field_strength_t=3.0,
            sequence_reference_ppm=171.076,
        )


def test_c13_four_metabolite_schedule_is_target_dominant_on_a_common_scale():
    names_ppm = (
        ("Pyruvate", 171.076),
        ("Lactate", 183.35),
        ("Alanine", 176.5),
        ("Pyruvate hydrate", 179.5),
    )
    sequence_reference_ppm = 171.076
    field_strength_t = 3.0
    library = SpectralSpatialPulseLibrary()
    entries = []
    species = []
    for name, absolute_ppm in names_ppm:
        center_hz = float(
            ppm_to_hz(
                absolute_ppm - sequence_reference_ppm,
                field_strength_t,
                "C13",
            )
        )
        pulse = design_spectral_spatial_pulse(
            bands=(
                SpectralBand(
                    center_hz,
                    30.0,
                    name=name,
                    transition_width_hz=20.0,
                    passband_ripple=0.05,
                    stopband_ripple=0.05,
                ),
            ),
            flip_angle_deg=10.0,
            slice_thickness_m=30e-3,
            spatial_transition_m=6e-3,
            spatial_time_bandwidth_product=4.0,
            subpulse_duration_s=0.5e-3,
            subpulse_count=31,
            nucleus="C13",
            max_gradient_t_per_m=0.032,
            max_slew_t_per_m_s=130.0,
            max_b1_t=500e-6,
            max_duration_s=0.1,
            name=name,
        )
        pulse_id = library.add(pulse, pulse_id=name.lower().replace(" ", "-"))
        entries.append(
            SpectralSpatialScheduleEntry(
                pulse_id,
                display_name=name,
                target_metabolite=name,
                flip_angle_deg=10.0,
            )
        )
        species.append(
            ChemicalSpecies(
                name,
                absolute_ppm - sequence_reference_ppm,
                t1=25.0,
                t2=0.3,
                t2_star=0.05,
            )
        )
    phantom = SpectralPhantom(
        shape=(1,),
        fov=(30e-3,),
        species=species,
        concentration_maps={item.name: np.ones(1) for item in species},
        field_strength=field_strength_t,
        nucleus="C13",
        spectral_reference_ppm=sequence_reference_ppm,
        metadata={"sequence_reference_ppm": sequence_reference_ppm},
    )
    schedule = SpectralSpatialSchedule(entries)
    peaks = spectral_spatial_peaks(
        phantom=phantom,
        field_strength_t=field_strength_t,
        nucleus="C13",
        sequence_reference_ppm=sequence_reference_ppm,
    )
    assert all(peak.source == "phantom" for peak in peaks)
    preview = simulate_spectral_spatial_schedule_preview(
        library,
        schedule,
        peaks,
        positions_m=(0.0,),
        pulse_spacing_s=0.0,
        spoiling="ideal",
    )
    response = np.abs(preview["isolated_mxy"][:, 0, :])

    assert np.all(np.argmax(response, axis=1) == np.arange(4))
    assert np.all(np.diag(response) > 0.17)
    off_diagonal = response[~np.eye(4, dtype=bool)]
    assert np.max(off_diagonal) < 0.003
    warnings = spectral_spatial_schedule_warnings(
        library,
        schedule,
        peaks,
        sequence_reference_ppm=sequence_reference_ppm,
    )
    assert not any("not present" in warning for warning in warnings)
