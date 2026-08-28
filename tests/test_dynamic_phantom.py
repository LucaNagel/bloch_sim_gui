import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from unittest.mock import MagicMock

from blochsimulator import BlochSimulator
from blochsimulator import dynamic_metal_backend as metal_backend_module
from blochsimulator.dynamic_phantom import (
    DynamicB0,
    DynamicSpectralPhantom,
    KineticRegionDefinition,
    PyruvateInflow,
    TimeCurve,
    rasterize_kpl_regions,
    simulate_two_pool_kinetics,
)
from blochsimulator.dynamic_metal_backend import (
    _PairwiseComplexAccumulator,
    _build_interval_plan,
    _hybrid_signal_correction,
    _hybrid_subvoxel_partition,
    _pairwise_sum_complex128,
    metal_capability,
    run_metal_hybrid_probe,
    run_metal_hybrid_sequence,
    run_metal_precision_probe,
)
from blochsimulator.phantom_design import (
    PhantomDesign,
    ShapeDefinition,
    SpectralPeakDefinition,
)
from blochsimulator.sequence import (
    ADCEvent,
    GradientEvent,
    RFEvent,
    SequenceCompiler,
    SequenceProgram,
    SpinSampling,
)
from blochsimulator.spectral_phantom import ChemicalSpecies
from blochsimulator.ui.phantom_designer import SpectralPhantomDesignerDialog
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationThread
from blochsimulator.ui.volume_viewer import PhantomInspectorWidget


def _dynamic_phantom(kpl=(0.0, 0.1)):
    shape = (2, 1, 1)
    return DynamicSpectralPhantom(
        shape=shape,
        fov=(0.02, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.asarray(kpl, dtype=float).reshape(shape),
        nucleus="C13",
    )


def test_private_metal_probe_capability_is_safe_and_machine_readable():
    capability = metal_capability()

    assert {
        "available",
        "supported_platform",
        "device_name",
        "apple_gpu_family",
        "recommended_max_working_set_bytes",
        "reason",
        "probe_extension_available",
    } <= set(capability)
    assert isinstance(capability["available"], bool)
    assert isinstance(capability["supported_platform"], bool)
    if capability["available"]:
        assert capability["device_name"]
        assert capability["recommended_max_working_set_bytes"] > 0
        assert capability["reason"] is None
    else:
        assert capability["reason"]


def test_metal_probe_plan_preserves_compiled_boundaries_and_adc_indices():
    phantom = _dynamic_phantom()
    phantom.pyruvate_inflow = PyruvateInflow(
        TimeCurve((0.0, 7.5e-4, 2e-3), (0.0, 1.0, 0.0)),
        np.ones(phantom.shape),
        TimeCurve((0.0, 2e-3), (0.2, 0.3), outside="hold"),
    )
    program = SequenceProgram(
        (
            RFEvent(0.0, np.asarray([100.0, 200.0]), 5e-4),
            ADCEvent(1e-3, 3, 2.5e-4),
        ),
        duration_s=2e-3,
    )
    compiled = SequenceCompiler().compile(
        program,
        extra_boundaries_s=phantom.dynamic_breakpoints_s(program.duration_s),
        simulation_timestep_s=2e-4,
    )

    plan = _build_interval_plan(compiled, phantom)

    assert plan.dtype == np.float32
    assert plan.shape == (compiled.n_intervals, 16)
    assert plan[:, 3] == pytest.approx(compiled.dt_s.astype(np.float32))
    assert np.cumsum(plan[:, 3], dtype=np.float64) == pytest.approx(
        compiled.interval_end_s, abs=5e-10
    )
    assert np.array_equal(
        compiled.adc_state_indices,
        np.searchsorted(compiled.interval_end_s, compiled.adc_times_s) + 1,
    )
    compiled_boundaries = np.concatenate(([0.0], compiled.interval_end_s))
    assert all(
        np.any(np.isclose(compiled_boundaries, breakpoint))
        for breakpoint in phantom.dynamic_breakpoints_s(program.duration_s)
    )


def test_metal_probe_pairwise_reduction_has_fixed_float64_tree():
    values = np.asarray(
        [
            [1.0e16 + 0.0j, 1.0 + 2.0j],
            [-1.0e16 + 0.0j, 3.0 + 4.0j],
            [3.0 + 0.0j, 5.0 + 6.0j],
        ],
        dtype=np.complex128,
    )

    first = _pairwise_sum_complex128(values)
    second = _pairwise_sum_complex128(values.copy())

    assert np.array_equal(first, second)
    assert first == pytest.approx([3.0 + 0.0j, 9.0 + 12.0j])

    accumulator = _PairwiseComplexAccumulator()
    for value in values:
        accumulator.add(value)
    assert np.array_equal(accumulator.finish(), first)


def test_partial_subvoxel_grids_preserve_full_grid_quadrature_weights():
    sampling = SpinSampling((3, 1, 1))
    offsets, weights = sampling.normalized_offsets_and_weights()
    left = sampling.select((0,))
    middle_and_right = sampling.select((1, 2))

    left_offsets, left_weights = left.normalized_offsets_and_weights()
    remainder_offsets, remainder_weights = (
        middle_and_right.normalized_offsets_and_weights()
    )

    assert left.enabled is True
    assert left.spins_per_voxel == 1
    assert left.grid_spins_per_voxel == 3
    assert np.array_equal(left_offsets, offsets[[0]])
    assert np.array_equal(remainder_offsets, offsets[[1, 2]])
    assert np.sum(left_weights) == pytest.approx(1.0 / 3.0)
    assert np.sum(remainder_weights) == pytest.approx(2.0 / 3.0)
    assert np.sum(left_weights) + np.sum(remainder_weights) == pytest.approx(
        np.sum(weights)
    )


def test_partial_float64_subvoxel_results_add_to_the_complete_grid():
    phantom = _dynamic_phantom(kpl=(0.02, 0.1))
    program = SequenceProgram(
        (
            RFEvent(0.0, np.asarray([180.0]), 5e-4),
            GradientEvent(
                axis="x",
                start_s=5e-4,
                samples_hz_per_m=np.asarray([300.0]),
                raster_s=1e-3,
            ),
            ADCEvent(7.5e-4, 3, 2.5e-4),
        ),
        duration_s=1.5e-3,
    )
    sampling = SpinSampling((3, 1, 1))
    simulator = BlochSimulator(use_parallel=False)
    full = simulator.simulate_dynamic_sequence(
        program,
        phantom,
        simulation_timestep_s=1e-4,
        spin_sampling=sampling,
        spoiler_mode="gradient",
    )
    partial = [
        simulator.simulate_dynamic_sequence(
            program,
            phantom,
            simulation_timestep_s=1e-4,
            spin_sampling=sampling.select((index,)),
            spoiler_mode="gradient",
        )
        for index in range(3)
    ]

    assert np.sum(
        [result.species_signal for result in partial], axis=0
    ) == pytest.approx(full.species_signal, abs=2e-15)
    assert np.sum(
        [result.final_pool_magnetization for result in partial], axis=0
    ) == pytest.approx(full.final_pool_magnetization, abs=2e-15)


def test_hybrid_partition_is_deterministic_disjoint_and_leaves_gpu_only_spins():
    sampling = SpinSampling((3, 3, 1))

    first = _hybrid_subvoxel_partition(sampling, 0.2, 0.2)
    second = _hybrid_subvoxel_partition(sampling, 0.2, 0.2)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[0].size >= 2
    assert first[1].size >= 2
    assert np.intersect1d(*first).size == 0
    assert first[0].size + first[1].size < sampling.grid_spins_per_voxel


def test_hybrid_correction_recovers_shared_gpu_gain_and_checks_held_out_spins():
    true_signal = np.asarray([[1.0 + 0.5j, 2.0 - 0.25j], [0.2 + 0.1j, 0.4 - 0.05j]])
    gain = np.asarray(
        [[1.002 + 0.01j, 0.998 - 0.02j], [1.001 + 0.005j, 1.003 - 0.004j]]
    )
    gpu_signal = true_signal / gain
    calibration_fraction = 0.25
    validation_fraction = 0.125

    corrected, metrics = _hybrid_signal_correction(
        gpu_signal,
        calibration_fraction * gpu_signal,
        calibration_fraction * true_signal,
        validation_fraction * gpu_signal,
        validation_fraction * true_signal,
        calibration_weight_fraction=calibration_fraction,
        validation_weight_fraction=validation_fraction,
    )

    assert corrected == pytest.approx(true_signal)
    assert metrics["estimated_full_total_signal_nrmse"] < 1e-15
    assert max(metrics["estimated_full_species_signal_nrmse"]) < 1e-15


def test_hybrid_correction_detects_a_nonrepresentative_calibration_sample():
    full = np.ones((2, 4), dtype=np.complex128)
    calibration_fraction = 0.25
    validation_fraction = 0.25

    _, metrics = _hybrid_signal_correction(
        full,
        calibration_fraction * full,
        calibration_fraction * full + 0.01,
        validation_fraction * full,
        validation_fraction * full,
        calibration_weight_fraction=calibration_fraction,
        validation_weight_fraction=validation_fraction,
    )

    assert metrics["estimated_full_total_signal_nrmse"] > 1e-3


def test_private_metal_probe_matches_small_float64_oracle_when_available():
    capability = metal_capability()
    if not capability["available"]:
        pytest.skip(capability["reason"])
    phantom = _dynamic_phantom(kpl=(0.0, 0.1))
    program = SequenceProgram(
        (
            RFEvent(0.0, np.asarray([250.0]), 1e-3),
            ADCEvent(1e-3, 2, 5e-4),
        ),
        duration_s=2e-3,
    )
    reference = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom, simulation_timestep_s=1e-4
    )

    candidate = run_metal_precision_probe(program, phantom, simulation_timestep_s=1e-4)

    signal_error = np.linalg.norm(candidate["signal"] - reference.signal)
    assert signal_error / np.linalg.norm(reference.signal) < 1e-3
    assert candidate["adc_times_s"] == pytest.approx(reference.adc_times_s)
    assert candidate["metadata"]["probe_only"] is True


def test_private_metal_probe_enforces_its_memory_budget_before_gpu_allocation(
    monkeypatch,
):
    monkeypatch.setattr(
        metal_backend_module,
        "metal_capability",
        lambda: {
            "available": True,
            "supported_platform": True,
            "device_name": "test device",
            "apple_gpu_family": 9,
            "recommended_max_working_set_bytes": 1024**3,
            "reason": None,
            "probe_extension_available": True,
        },
    )
    phantom = _dynamic_phantom()
    program = SequenceProgram(
        (ADCEvent(0.0, 2, 5e-4),),
        duration_s=1e-3,
    )

    with pytest.raises(MemoryError, match="memory limit exceeded"):
        run_metal_precision_probe(
            program,
            phantom,
            simulation_timestep_s=1e-4,
            memory_budget_bytes=1,
        )


def test_private_metal_probe_chunks_outputs_and_retains_only_requested_spins(
    monkeypatch,
):
    monkeypatch.setattr(
        metal_backend_module,
        "metal_capability",
        lambda: {
            "available": True,
            "supported_platform": True,
            "device_name": "test device",
            "apple_gpu_family": 9,
            "recommended_max_working_set_bytes": 1024**3,
            "reason": None,
            "probe_extension_available": True,
        },
    )
    calls = []

    def fake_run_probe(
        _source,
        _plan,
        adc_states,
        _demodulation,
        _crushers,
        initial,
        _spatial,
        _kinetic,
        _constants,
        _precision_mode,
    ):
        calls.append(initial.shape[0])
        return {
            "final_pool_state": np.array(initial, copy=True),
            "per_spin_species_signal": np.zeros(
                (initial.shape[0], 2, adc_states.size, 2), dtype=np.float32
            ),
            "pipeline_compile_seconds": 0.0,
            "simulation_seconds": 0.001,
        }

    import blochsimulator

    monkeypatch.setattr(
        blochsimulator,
        "_dynamic_metal_probe",
        MagicMock(run_probe=fake_run_probe),
        raising=False,
    )
    phantom = _dynamic_phantom()
    program = SequenceProgram(
        (ADCEvent(0.0, 2, 5e-4),),
        duration_s=1e-3,
    )

    result = run_metal_precision_probe(
        program,
        phantom,
        simulation_timestep_s=1e-4,
        spin_sampling=SpinSampling((2, 1, 1)),
        spin_chunk_size=2,
        capture_spin_indices=(0, 3),
        capture_spin_groups=((0, 2), (1, 3)),
    )

    assert calls == [2, 2]
    assert result["metadata"]["spin_chunk_count"] == 2
    assert result["metadata"]["effective_spin_chunk_size"] == 2
    assert np.array_equal(result["captured_spin_indices"], [0, 3])
    assert result["captured_spin_species_signal"].shape == (2, 2, 2)
    assert result["captured_group_species_signal"].shape == (2, 2, 2)
    assert result["captured_group_final_pool_magnetization"].shape == (
        2,
        2,
    ) + phantom.shape + (3,)


def test_hybrid_probe_returns_float64_fallback_when_held_out_sample_fails(
    monkeypatch,
):
    phantom = _dynamic_phantom()
    program = SequenceProgram(
        (ADCEvent(0.0, 2, 5e-4),),
        duration_s=1e-3,
    )
    sampling = SpinSampling((2, 2, 2))
    calibration, validation = _hybrid_subvoxel_partition(sampling, 0.1, 0.05)
    true_species = np.asarray([[1.0 + 0.1j, 0.8 - 0.2j], [0.2 + 0.05j, 0.3 - 0.04j]])
    true_final = np.zeros((2,) + phantom.shape + (3,), dtype=np.float64)
    true_final[..., 2] = 1.0
    monkeypatch.setattr(
        metal_backend_module,
        "metal_capability",
        lambda: {
            "available": True,
            "supported_platform": True,
            "device_name": "test device",
            "apple_gpu_family": 9,
            "recommended_max_working_set_bytes": 1024**3,
            "reason": None,
            "probe_extension_available": True,
        },
    )

    def fake_cpu_sample(*_args, sampling, **_kwargs):
        fraction = sampling.spins_per_voxel / sampling.grid_spins_per_voxel
        species = true_species * fraction
        if tuple(sampling.selected_indices or ()) == tuple(validation):
            species = species + 0.02
        result = MagicMock()
        result.species_signal = species
        result.signal = species.sum(axis=0)
        result.final_pool_magnetization = true_final * fraction
        result.final_magnetization = result.final_pool_magnetization.sum(axis=0)
        result.adc_times_s = np.asarray([0.0, 5e-4])
        return result, 0.01

    def fake_metal(*_args, capture_spin_groups, **_kwargs):
        group_fractions = np.asarray(
            [
                len(group) / (phantom.n_active * sampling.grid_spins_per_voxel)
                for group in capture_spin_groups
            ]
        )
        return {
            "signal": true_species.sum(axis=0),
            "species_signal": true_species,
            "final_pool_magnetization": true_final,
            "final_magnetization": true_final.sum(axis=0),
            "adc_times_s": np.asarray([0.0, 5e-4]),
            "captured_group_species_signal": (
                group_fractions[:, None, None] * true_species[None]
            ),
            "captured_group_final_pool_magnetization": (
                group_fractions[:, None, None, None, None, None] * true_final[None]
            ),
            "metadata": {"probe_only": True},
        }

    monkeypatch.setattr(
        metal_backend_module, "_run_cpu_float64_sample", fake_cpu_sample
    )
    monkeypatch.setattr(metal_backend_module, "run_metal_precision_probe", fake_metal)

    result = run_metal_hybrid_probe(
        program,
        phantom,
        simulation_timestep_s=1e-4,
        spin_sampling=sampling,
        run_concurrently=False,
    )

    assert tuple(calibration) != tuple(validation)
    assert result["metadata"]["hybrid_validation_passed"] is False
    assert result["metadata"]["hybrid_fallback_used"] is True
    assert result["metadata"]["actual_backend"] == "cpu_float64_fallback"
    assert result["species_signal"] == pytest.approx(true_species)


def test_hybrid_sequence_wraps_checked_arrays_as_a_regular_result(monkeypatch):
    phantom = _dynamic_phantom()
    program = SequenceProgram(
        (ADCEvent(0.0, 2, 5e-4),),
        duration_s=1e-3,
    )
    sampling = SpinSampling((2, 2, 2))
    template = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program,
        phantom,
        simulation_timestep_s=1e-4,
        spin_sampling=sampling.select((0, 7)),
    )
    checked_species = np.asarray(template.species_signal) * 4.0
    checked_final_pool = np.asarray(template.final_pool_magnetization) * 4.0

    monkeypatch.setattr(
        metal_backend_module,
        "run_metal_hybrid_probe",
        lambda *_args, **_kwargs: {
            "signal": checked_species.sum(axis=0),
            "species_signal": checked_species,
            "final_pool_magnetization": checked_final_pool,
            "final_magnetization": checked_final_pool.sum(axis=0),
            "adc_times_s": template.adc_times_s,
            "metadata": {
                "actual_backend": "metal_cpu_subvoxel_hybrid_probe",
                "hybrid_validation_passed": True,
                "hybrid_fallback_used": False,
            },
            "_sequence_result": template,
        },
    )

    result = run_metal_hybrid_sequence(
        program,
        phantom,
        simulation_timestep_s=1e-4,
        spin_sampling=sampling,
    )

    assert result.signal == pytest.approx(checked_species.sum(axis=0))
    assert result.species_signal == pytest.approx(checked_species)
    assert result.final_pool_magnetization == pytest.approx(checked_final_pool)
    assert result.metadata["requested_sequence_kernel"] == "metal_hybrid"
    assert result.metadata["sequence_kernel"] == "metal_hybrid"
    assert result.metadata["probe_only"] is False
    assert result.metadata["subvoxel_spin_counts_xyz"] == (2, 2, 2)


def test_hybrid_sequence_uses_exact_cpu_when_gpu_is_unavailable(monkeypatch):
    phantom = _dynamic_phantom()
    program = SequenceProgram(
        (ADCEvent(0.0, 2, 5e-4),),
        duration_s=1e-3,
    )
    sampling = SpinSampling((2, 2, 2))
    monkeypatch.setattr(
        metal_backend_module,
        "run_metal_hybrid_probe",
        MagicMock(side_effect=RuntimeError("Metal unavailable for test")),
    )

    result = BlochSimulator(
        use_parallel=False,
        dynamic_sequence_kernel="metal_hybrid",
    ).simulate_dynamic_sequence(
        program,
        phantom,
        simulation_timestep_s=1e-4,
        spin_sampling=sampling,
    )
    reference = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program,
        phantom,
        simulation_timestep_s=1e-4,
        spin_sampling=sampling,
    )

    assert np.array_equal(result.signal, reference.signal)
    assert np.array_equal(result.species_signal, reference.species_signal)
    assert result.metadata["requested_sequence_kernel"] == "metal_hybrid"
    assert result.metadata["actual_backend"] == "cpu_float64_fallback"
    assert result.metadata["hybrid_fallback_used"] is True
    assert "Metal unavailable for test" in result.metadata["hybrid_fallback_reason"]


def test_kinetic_regions_rasterize_with_later_region_priority():
    whole = KineticRegionDefinition(
        "whole", "box", (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), 0.02
    )
    right = KineticRegionDefinition(
        "right", "box", (0.75, 0.5, 0.5), (0.5, 1.0, 1.0), 0.08
    )
    result = rasterize_kpl_regions((4, 1, 1), (whole, right))

    assert np.array_equal(result[:, 0, 0], [0.02, 0.02, 0.08, 0.08])


def test_dynamic_phantom_exposes_initial_spectrum_to_inspector():
    app = QApplication.instance() or QApplication([])
    phantom = _dynamic_phantom()
    phantom.initial_concentration_maps["Lactate"][:] = 0.5
    phantom.spectral_reference_ppm = 175.0
    phantom.spectral_window_center_ppm = 177.5
    phantom.spectral_bandwidth_ppm = 30.0

    frequency_ppm, spectrum = phantom.spectrum_at_ppm(
        (0, 0, 0), frequency_ppm=np.asarray([0.0, 12.0]), absolute=False
    )

    assert frequency_ppm == pytest.approx([0.0, 12.0])
    assert spectrum[0] == pytest.approx(1.0, rel=2e-3)
    assert spectrum[1] == pytest.approx(0.5, rel=2e-3)

    default_axis, _ = phantom.spectrum_at_ppm((0, 0, 0))
    assert default_axis[[0, -1]] == pytest.approx([162.5, 192.5])

    inspector = PhantomInspectorWidget()
    inspector.set_phantom(phantom)
    plotted = inspector.spectrum_plot.listDataItems()

    assert len(plotted) == 1
    assert np.max(plotted[0].getData()[1]) > 0.0
    assert "2 Lorentzian components" in inspector.spectrum_info.text()
    inspector.close()
    app.processEvents()


def test_no_rf_dynamic_sequence_matches_irreversible_analytic_solution():
    phantom = _dynamic_phantom()
    duration = 2.0
    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        SequenceProgram((), duration_s=duration), phantom
    )
    final = result.final_pool_magnetization[..., 2]

    for voxel, kpl in enumerate((0.0, 0.1)):
        a = 1.0 / 30.0 + kpl
        b = 1.0 / 25.0
        expected_pyruvate = np.exp(-a * duration)
        expected_lactate = (
            kpl * (np.exp(-b * duration) - np.exp(-a * duration)) / (a - b)
            if kpl
            else 0.0
        )
        assert final[0, voxel, 0, 0] == pytest.approx(expected_pyruvate)
        assert final[1, voxel, 0, 0] == pytest.approx(expected_lactate)


def test_dynamic_sequence_applies_declared_ideal_spoiler_to_both_pools():
    phantom = _dynamic_phantom(kpl=(0.0, 0.0))
    program = SequenceProgram(
        (RFEvent(0.0, np.array([250.0]), 1e-3),),
        duration_s=2e-3,
        metadata={"definitions": {"IdealSpoilerEndTimes": [1.5e-3]}},
    )

    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program,
        phantom,
        checkpoints_s=(1.5e-3,),
        sequence_kernel="optimized",
    )

    assert result.final_pool_magnetization[..., :2] == pytest.approx(0.0, abs=1e-12)
    assert result.checkpoint_pool_magnetization[0, ..., :2] == pytest.approx(
        0.0, abs=1e-12
    )
    assert result.metadata["ideal_spoiling_applied"] is True
    assert result.metadata["ideal_spoiler_end_times_s"] == pytest.approx([1.5e-3])


def test_dynamic_gradient_waveform_matches_ideal_crusher_spoiling():
    phantom = _dynamic_phantom(kpl=(0.0, 0.0))
    rf_duration_s = 1e-3
    spoiler_duration_s = 10e-3
    spoiler_end_s = rf_duration_s + spoiler_duration_s
    voxel_width_m = phantom.fov[0] / phantom.shape[0]
    gradient_hz_per_m = 1.0 / voxel_width_m / spoiler_duration_s
    program = SequenceProgram(
        (
            RFEvent(0.0, np.array([250.0]), rf_duration_s),
            GradientEvent(
                "x",
                rf_duration_s,
                np.array([gradient_hz_per_m]),
                spoiler_duration_s,
            ),
            ADCEvent(spoiler_end_s, 1, 1e-3),
        ),
        duration_s=spoiler_end_s + 1e-3,
        metadata={"definitions": {"IdealSpoilerEndTimes": [spoiler_end_s]}},
    )
    simulator = BlochSimulator(use_parallel=False)

    ideal = simulator.simulate_dynamic_sequence(program, phantom, spoiler_mode="ideal")
    physical = simulator.simulate_dynamic_sequence(
        program,
        phantom,
        spin_sampling=SpinSampling((9, 1, 1)),
        spoiler_mode="gradient",
    )

    assert ideal.signal[0] == pytest.approx(0.0j, abs=1e-12)
    assert physical.signal[0] == pytest.approx(ideal.signal[0], abs=1e-11)
    assert physical.metadata["subvoxel_spins_per_voxel"] == 9
    assert physical.metadata["ideal_spoiling_applied"] is False


def test_dynamic_sequence_reports_intermediate_live_previews():
    phantom = _dynamic_phantom()
    program = SequenceProgram(
        (ADCEvent(0.0, 4, 1e-3),),
        duration_s=4e-3,
    )
    previews = []

    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program,
        phantom,
        preview_callback=lambda fraction, signal: previews.append(
            (fraction, np.array(signal, copy=True))
        ),
    )

    assert len(previews) > 1
    assert 0.0 < previews[0][0] < 1.0
    assert previews[-1][0] == pytest.approx(1.0)
    assert previews[-1][1] == pytest.approx(result.signal)


def test_time_curve_integrates_linear_step_and_outside_regions():
    linear = TimeCurve((1.0, 3.0), (0.0, 2.0), "linear", "zero")
    step = TimeCurve((1.0, 3.0), (2.0, 4.0), "step", "zero")

    assert linear.integral(0.0, 4.0) == pytest.approx(2.0)
    assert linear.interval_values(0.0, 1.0) == (0.0, 0.0)
    assert step.integral(0.0, 4.0) == pytest.approx(4.0)


def test_time_curve_shift_preserves_values_and_shape():
    curve = TimeCurve((0.0, 2.0), (0.25, 1.0), "linear", "zero")

    shifted = curve.shifted(-1.5)

    assert shifted.times_s == (-1.5, 0.5)
    assert shifted.values == curve.values
    assert shifted.interpolation == curve.interpolation
    assert shifted.outside == curve.outside


def test_two_pool_kinetics_preview_uses_inflow_and_conversion_solver():
    times_s = np.linspace(0.0, 2.0, 101)
    inflow = TimeCurve((0.0, 2.0), (1.0, 1.0), "linear", "zero")

    unconverted = simulate_two_pool_kinetics(
        times_s, (0.0, 0.0), (1e15, 1e15), 0.0, inflow
    )
    converted = simulate_two_pool_kinetics(
        times_s, (0.0, 0.0), (1e15, 1e15), 0.2, inflow
    )

    assert unconverted[0, -1] == pytest.approx(2.0)
    assert np.all(unconverted[1] == 0.0)
    assert converted[0, -1] < unconverted[0, -1]
    assert converted[1, -1] > 0.0
    assert converted[:, -1].sum() == pytest.approx(2.0, rel=1e-12)


def test_concentration_inflow_has_independent_polarization_and_relaxes_to_one():
    times = np.asarray((0.0, 5.0, 6.0, 11.0, 31.0))
    concentration_rate = TimeCurve(
        (0.0, 5.0, 6.0),
        (0.0, 10.0, 0.0),
        interpolation="step",
        outside="zero",
    )
    inflow_polarization = TimeCurve(
        (0.0, 5.0, 6.0),
        (1.0, 10000.0, 1.0),
        interpolation="step",
        outside="hold",
    )

    magnetization, concentration = simulate_two_pool_kinetics(
        times,
        initial_mz=(0.0, 0.0),
        initial_concentration=(0.0, 0.0),
        t1_s=(5.0, 5.0),
        kpl_s_inv=0.0,
        inflow_curve=concentration_rate,
        inflow_polarization_curve=inflow_polarization,
        equilibrium_polarization=1.0,
        return_concentration=True,
    )
    pyruvate_polarization = np.divide(
        magnetization[0],
        concentration[0],
        out=np.zeros_like(magnetization[0]),
        where=concentration[0] > 0,
    )

    assert concentration[0, 1] == pytest.approx(0.0)
    assert concentration[0, 2] == pytest.approx(10.0)
    assert pyruvate_polarization[2] > 8000.0
    assert pyruvate_polarization[3] > 1.0
    assert pyruvate_polarization[4] > 1.0
    assert pyruvate_polarization[4] < pyruvate_polarization[3]
    expected_late = 1.0 + (pyruvate_polarization[2] - 1.0) * np.exp(-25.0 / 5.0)
    assert pyruvate_polarization[4] == pytest.approx(expected_late, rel=1e-10)


def test_designed_empty_pool_receives_highly_polarized_concentration_bolus():
    design = PhantomDesign(
        shape=(1, 1, 1),
        fov_m=(0.01, 0.01, 0.01),
        dynamic_enabled=True,
        pyruvate_inflow_curve=TimeCurve(
            (0.0, 5.0, 6.0), (0.0, 10.0, 0.0), "step", "zero"
        ),
        pyruvate_inflow_polarization_curve=TimeCurve(
            (0.0, 5.0, 6.0), (1.0, 10000.0, 1.0), "step", "hold"
        ),
        shapes=[
            ShapeDefinition(
                "Injection region",
                kind="box",
                size=(1.0, 1.0, 1.0),
                peaks=[
                    SpectralPeakDefinition(
                        "Pyruvate",
                        amplitude=0.0,
                        t1_s=5.0,
                        initial_polarization=1.0,
                    ),
                    SpectralPeakDefinition(
                        "Lactate",
                        amplitude=0.0,
                        t1_s=5.0,
                        initial_polarization=1.0,
                    ),
                ],
            )
        ],
    )
    phantom = design.build()
    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        SequenceProgram((), duration_s=11.0),
        phantom,
        checkpoints_s=(5.0, 6.0, 11.0),
    )
    pyruvate_mz = result.checkpoint_pool_magnetization[:, 0, 0, 0, 0, 2]

    assert phantom.equilibrium_polarization == pytest.approx(1.0)
    assert phantom.initial_spin_density_maps["Pyruvate"][0, 0, 0] == 0.0
    assert pyruvate_mz[0] == pytest.approx(0.0)
    assert pyruvate_mz[1] > 80000.0
    assert 10.0 < pyruvate_mz[2] < pyruvate_mz[1]


def test_negative_inflow_and_conversion_start_set_sequence_start_distribution():
    times_s = np.asarray([-2.0, -1.0, 0.0])
    inflow = TimeCurve((-2.0, 0.0), (1.0, 1.0), "linear", "zero")
    preview = simulate_two_pool_kinetics(
        times_s,
        (0.0, 0.0),
        (1e15, 1e15),
        0.2,
        inflow,
        conversion_start_s=-1.0,
        initial_time_s=-2.0,
    )
    shape = (1, 1, 1)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.01, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 1e15, 1.0),
            ChemicalSpecies("Lactate", 12.0, 1e15, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.zeros(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.full(shape, 0.2),
        pyruvate_inflow=PyruvateInflow(inflow, np.ones(shape)),
        conversion_start_s=-1.0,
        nucleus="C13",
    )

    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        SequenceProgram((), duration_s=0.0),
        phantom,
        checkpoints_s=(0.0,),
    )

    expected_pyruvate = np.exp(-0.2) + (1.0 - np.exp(-0.2)) / 0.2
    expected_lactate = 2.0 - expected_pyruvate
    assert preview[:, -1] == pytest.approx(
        [expected_pyruvate, expected_lactate],
        rel=1e-12,
    )
    assert result.final_pool_magnetization[..., 2].reshape(2) == pytest.approx(
        preview[:, -1],
        rel=1e-12,
    )
    assert result.metadata["kinetic_preroll_start_s"] == -2.0
    assert result.metadata["conversion_start_s"] == -1.0


@pytest.mark.parametrize("sequence_kernel", ["reference", "optimized"])
def test_global_kinetics_offset_shifts_inflow_and_conversion_together(
    sequence_kernel,
):
    shape = (1, 1, 1)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.01, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 1e15, 1.0),
            ChemicalSpecies("Lactate", 12.0, 1e15, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.zeros(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.ones(shape),
        pyruvate_inflow=PyruvateInflow(
            TimeCurve((0.0, 2.0), (1.0, 1.0), "linear", "zero"),
            np.ones(shape),
        ),
        conversion_start_s=1.0,
        kinetics_time_offset_s=1.0,
        nucleus="C13",
    )

    assert phantom.inflow_curve_on_sequence_timeline.times_s == (-1.0, 1.0)
    assert phantom.conversion_start_on_sequence_timeline_s == 0.0
    assert phantom.dynamic_breakpoints_s(1.0) == (0.0, 1.0)

    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        SequenceProgram((), duration_s=1.0),
        phantom,
        checkpoints_s=(0.0,),
        sequence_kernel=sequence_kernel,
    )

    assert result.checkpoint_pool_magnetization[0, ..., 2].reshape(2) == pytest.approx(
        [1.0, 0.0], rel=1e-12, abs=1e-12
    )
    assert result.final_pool_magnetization[..., 2].reshape(2) == pytest.approx(
        [1.0, 1.0], rel=1e-12, abs=1e-12
    )
    assert result.metadata["kinetics_time_offset_s"] == 1.0
    assert result.metadata["conversion_start_s"] == 1.0
    assert result.metadata["sequence_conversion_start_s"] == 0.0
    assert result.metadata["kinetic_preroll_start_s"] == -1.0


@pytest.mark.parametrize("sequence_kernel", ["reference", "optimized"])
def test_positive_conversion_start_delays_kpl_during_sequence(sequence_kernel):
    shape = (1, 1, 1)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.01, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 1e15, 1.0),
            ChemicalSpecies("Lactate", 12.0, 1e15, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.full(shape, 0.2),
        conversion_start_s=1.0,
        nucleus="C13",
    )

    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        SequenceProgram((), duration_s=2.0),
        phantom,
        sequence_kernel=sequence_kernel,
    )

    expected_pyruvate = np.exp(-0.2)
    assert result.final_pool_magnetization[..., 2].reshape(2) == pytest.approx(
        [expected_pyruvate, 1.0 - expected_pyruvate],
        rel=1e-12,
    )


def test_inflow_populates_initially_empty_voxel_with_analytic_t1_decay():
    shape = (1, 1, 1)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.01, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.zeros(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.zeros(shape),
        pyruvate_inflow=PyruvateInflow(
            TimeCurve((0.0, 2.0), (1.0, 1.0), "linear", "zero"),
            np.ones(shape),
        ),
        nucleus="C13",
    )

    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        SequenceProgram((), duration_s=2.0), phantom
    )

    expected = 30.0 * (1.0 - np.exp(-2.0 / 30.0))
    assert phantom.n_active == 1
    assert result.final_pool_magnetization[0, 0, 0, 0, 2] == pytest.approx(expected)
    assert result.final_pool_magnetization[1, 0, 0, 0, 2] == pytest.approx(0.0)


def test_linear_inflow_and_conversion_conserve_added_mass_without_relaxation():
    shape = (1, 1, 1)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.01, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 1e15, 1.0),
            ChemicalSpecies("Lactate", 12.0, 1e15, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.zeros(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.full(shape, 0.2),
        pyruvate_inflow=PyruvateInflow(
            TimeCurve((0.0, 2.0), (0.0, 2.0), "linear", "zero"),
            np.ones(shape),
        ),
        nucleus="C13",
    )

    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        SequenceProgram((), duration_s=2.0), phantom
    )

    final_mass = result.final_pool_magnetization[..., 2].sum()
    assert final_mass == pytest.approx(2.0, rel=1e-12)


def test_dynamic_b0_uses_integrated_frequency_phase():
    shape = (1, 1, 1)
    kwargs = dict(
        shape=shape,
        fov=(0.01, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 10.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 10.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.zeros(shape),
        nucleus="C13",
    )
    reference = DynamicSpectralPhantom(**kwargs)
    dynamic = DynamicSpectralPhantom(
        **kwargs,
        dynamic_b0=DynamicB0(
            TimeCurve((0.0, 1.0), (0.0, 7.0), "linear", "hold"),
            np.ones(shape),
        ),
    )
    program = SequenceProgram(
        (RFEvent(0.0, np.asarray([250000.0]), 1e-6), ADCEvent(1.0, 1, 1e-3)),
        duration_s=1.001,
    )
    simulator = BlochSimulator(use_parallel=False)

    reference_signal = simulator.simulate_dynamic_sequence(program, reference).signal[0]
    dynamic_signal = simulator.simulate_dynamic_sequence(program, dynamic).signal[0]
    phase_integral = dynamic.dynamic_b0.offset_curve_hz.integral(0.5e-6, 1.0)

    assert dynamic_signal / reference_signal == pytest.approx(
        np.exp(-2j * np.pi * phase_integral), rel=1e-10, abs=1e-10
    )


def test_dynamic_sequence_returns_pool_resolved_adc_and_xarray():
    phantom = _dynamic_phantom()
    program = SequenceProgram(
        (
            RFEvent(0.0, np.asarray([250.0]), 1e-3),
            ADCEvent(2e-3, 2, 1e-3),
        ),
        duration_s=4e-3,
    )
    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom
    )

    assert result.species_signal.shape == (2, 2)
    assert np.allclose(result.signal, result.species_signal.sum(axis=0))
    dataset = result.to_xarray()
    assert dataset.species_signal.dims == ("pool", "adc")
    assert list(dataset.pool.values) == ["Pyruvate", "Lactate"]


@pytest.mark.parametrize("with_drivers", [False, True])
@pytest.mark.parametrize(
    "kpl",
    [
        (0.0, 0.1),
        (1.0 / 25.0 - 1.0 / 30.0, 0.1),
        (1.0 / 25.0 - 1.0 / 30.0,) * 2,
    ],
    ids=("regular", "mixed-rates", "equal-rates"),
)
def test_optimized_dynamic_kernel_is_bitwise_equal_to_reference(with_drivers, kpl):
    phantom = _dynamic_phantom(kpl)
    raster = 20e-6
    if with_drivers:
        phantom.pyruvate_inflow = PyruvateInflow(
            TimeCurve(
                (0.0, 6 * raster, 12 * raster),
                (0.0, 0.2, 0.1),
                "linear",
                "hold",
            ),
            np.ones(phantom.shape),
        )
        phantom.dynamic_b0 = DynamicB0(
            TimeCurve(
                (0.0, 6 * raster, 12 * raster),
                (-3.0, 5.0, 2.0),
                "linear",
                "hold",
            ),
            np.linspace(0.5, 1.0, phantom.nvoxels).reshape(phantom.shape),
            (1.0, 0.8),
        )
    program = SequenceProgram(
        (
            RFEvent(
                2 * raster,
                np.asarray([80.0 + 20.0j, 120.0 - 10.0j, 60.0 + 5.0j, 0.0j]),
                raster,
            ),
            GradientEvent(
                "x",
                0.0,
                np.linspace(-150.0, 200.0, 12),
                raster,
            ),
            ADCEvent(0.0, 6, 2 * raster, phase_offset_rad=0.37),
        ),
        duration_s=12 * raster,
    )
    simulator = BlochSimulator(use_parallel=True, num_threads=2)
    kwargs = {
        "checkpoints_s": (0.0, 7 * raster, 12 * raster),
        "simulation_timestep_s": 5e-6,
    }

    reference = simulator.simulate_dynamic_sequence(
        program,
        phantom,
        sequence_kernel="reference",
        **kwargs,
    )
    optimized = simulator.simulate_dynamic_sequence(
        program,
        phantom,
        sequence_kernel="optimized",
        **kwargs,
    )

    assert optimized.metadata["sequence_kernel"] == "optimized"
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
        "checkpoint_magnetization",
        "checkpoint_pool_magnetization",
    ):
        assert np.array_equal(getattr(optimized, name), getattr(reference, name))


def test_float32_dynamic_shadow_path_reports_dtypes_and_tracks_float64():
    phantom = _dynamic_phantom()
    raster = 20e-6
    phantom.pyruvate_inflow = PyruvateInflow(
        TimeCurve(
            (0.0, 6 * raster, 12 * raster),
            (0.0, 0.2, 0.1),
            "linear",
            "hold",
        ),
        np.ones(phantom.shape),
    )
    phantom.dynamic_b0 = DynamicB0(
        TimeCurve(
            (0.0, 6 * raster, 12 * raster),
            (-3.0, 5.0, 2.0),
            "linear",
            "hold",
        ),
        np.linspace(0.5, 1.0, phantom.nvoxels).reshape(phantom.shape),
        (1.0, 0.8),
    )
    program = SequenceProgram(
        (
            RFEvent(
                2 * raster,
                np.asarray([80.0 + 20.0j, 120.0 - 10.0j, 60.0 + 5.0j, 0.0j]),
                raster,
            ),
            GradientEvent("x", 0.0, np.linspace(-150.0, 200.0, 12), raster),
            ADCEvent(0.0, 6, 2 * raster, phase_offset_rad=0.37),
        ),
        duration_s=12 * raster,
    )
    kwargs = {
        "checkpoints_s": (0.0, 7 * raster, 12 * raster),
        "simulation_timestep_s": 5e-6,
    }
    float64 = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom, **kwargs
    )
    float32 = BlochSimulator(
        use_parallel=False,
        dynamic_sequence_precision="float32",
    ).simulate_dynamic_sequence(program, phantom, **kwargs)

    assert float32.metadata["simulation_precision"] == "float32"
    assert float32.metadata["state_dtype"] == "float32"
    assert float32.metadata["signal_dtype"] == "complex64"
    assert float32.metadata["coefficient_precompute_dtype"] == "float64"
    assert float32.signal.dtype == np.complex64
    assert float32.species_signal.dtype == np.complex64
    assert float32.final_magnetization.dtype == np.float32
    assert float32.final_pool_magnetization.dtype == np.float32
    assert float32.checkpoint_magnetization.dtype == np.float32
    assert float32.checkpoint_pool_magnetization.dtype == np.float32
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
        "checkpoint_magnetization",
        "checkpoint_pool_magnetization",
    ):
        actual = getattr(float32, name)
        expected = getattr(float64, name)
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_dynamic_animation_can_reduce_only_checkpoint_storage_precision():
    phantom = _dynamic_phantom()
    program = SequenceProgram((), duration_s=1e-3)
    simulator = BlochSimulator(use_parallel=False)

    baseline = simulator.simulate_dynamic_sequence(
        program, phantom, checkpoints_s=(0.0, 1e-3)
    )
    reduced = simulator.simulate_dynamic_sequence(
        program,
        phantom,
        checkpoints_s=(0.0, 1e-3),
        checkpoint_dtype="float32",
    )

    assert reduced.final_magnetization.dtype == np.float64
    assert reduced.checkpoint_magnetization.dtype == np.float32
    assert reduced.checkpoint_pool_magnetization.dtype == np.float32
    np.testing.assert_array_equal(
        reduced.final_magnetization, baseline.final_magnetization
    )
    np.testing.assert_allclose(
        reduced.checkpoint_pool_magnetization,
        baseline.checkpoint_pool_magnetization,
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("kernel", ["reference", "native_serial", "native_parallel"])
def test_float32_dynamic_shadow_path_rejects_nonoptimized_kernels(kernel):
    with pytest.raises(ValueError, match="requires sequence_kernel='optimized'"):
        BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
            SequenceProgram((), duration_s=1e-3),
            _dynamic_phantom(),
            sequence_kernel=kernel,
            simulation_precision="float32",
        )


def test_bloch_simulator_rejects_native_kernel_with_float32_precision():
    with pytest.raises(ValueError, match="requires the optimized kernel"):
        BlochSimulator(
            use_parallel=False,
            dynamic_sequence_kernel="native_serial",
            dynamic_sequence_precision="float32",
        )


@pytest.mark.parametrize(
    "kpl",
    [
        (0.0, 0.1),
        (1.0 / 25.0 - 1.0 / 30.0, 0.1),
        (1.0 / 25.0 - 1.0 / 30.0,) * 2,
    ],
    ids=("regular", "mixed-rates", "equal-rates"),
)
def test_native_serial_dynamic_pilot_is_bitwise_equal(kpl):
    phantom = _dynamic_phantom(kpl)
    raster = 20e-6
    program = SequenceProgram(
        (
            RFEvent(
                2 * raster,
                np.asarray([80.0 + 20.0j, 120.0 - 10.0j, 0.0j]),
                raster,
            ),
            GradientEvent("x", 0.0, np.linspace(-150.0, 200.0, 12), raster),
            ADCEvent(0.0, 6, 2 * raster, phase_offset_rad=0.37),
        ),
        duration_s=12 * raster,
    )
    simulator = BlochSimulator(use_parallel=False)
    kwargs = {
        "checkpoints_s": (0.0, 7 * raster, 12 * raster),
        "simulation_timestep_s": 5e-6,
    }
    reference = simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel="reference", **kwargs
    )
    optimized = simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel="optimized", **kwargs
    )
    native = simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel="native_serial", **kwargs
    )

    assert native.metadata["sequence_kernel"] == "native_serial"
    assert native.metadata["native_fallback_reason"] is None
    assert native.metadata["native_rf_block_enabled"] is True
    assert native.metadata["native_rf_threads"] == 1
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
        "checkpoint_magnetization",
        "checkpoint_pool_magnetization",
    ):
        assert np.array_equal(getattr(native, name), getattr(reference, name))
        assert np.array_equal(getattr(native, name), getattr(optimized, name))


def test_configured_dynamic_kernel_is_used_without_call_override():
    result = BlochSimulator(
        use_parallel=False, dynamic_sequence_kernel="native_serial"
    ).simulate_dynamic_sequence(
        SequenceProgram((), duration_s=1e-3),
        _dynamic_phantom(),
    )

    assert result.metadata["requested_sequence_kernel"] == "native_serial"
    assert result.metadata["sequence_kernel"] == "native_serial"


@pytest.mark.parametrize("num_threads", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "rf_hz",
    [123.4 + 56.7j, -987.6 + 1.2j, 250.0j],
    ids=("complex-axis", "negative-x", "y-axis"),
)
def test_native_rf_voxel_block_is_bitwise_equal_to_numpy(num_threads, rf_hz):
    from blochsimulator.dynamic_bloch_cy import (
        apply_rf_rotation_transverse_block,
    )
    from blochsimulator.dynamic_phantom import _prepare_rf_rotation, _rf_rotate

    rng = np.random.default_rng(20260722)
    state = rng.standard_normal((2, 1728, 3))
    expected = state.copy()
    transverse = state[:, :, 0] + 1j * state[:, :, 1]
    duration_s = 10e-6

    _rf_rotate(expected, rf_hz, duration_s)
    prepared = _prepare_rf_rotation(rf_hz, duration_s)
    apply_rf_rotation_transverse_block(
        state,
        transverse,
        prepared[0],
        prepared[1],
        prepared[2],
        prepared[3],
        prepared[4],
        num_threads,
    )
    state[:, :, 0] = transverse.real
    state[:, :, 1] = transverse.imag

    assert np.array_equal(state, expected)


@pytest.mark.parametrize("kernel", ["native_serial", "native_parallel"])
@pytest.mark.parametrize(
    "driver",
    ["inflow", "dynamic_b0", "delayed_conversion", "concentration_tracking"],
)
def test_native_dynamic_kernel_keeps_rf_fast_path_with_driver(driver, kernel):
    phantom = _dynamic_phantom()
    curve = TimeCurve((0.0, 1e-3), (0.0, 1.0), "linear", "hold")
    if driver == "inflow":
        phantom.pyruvate_inflow = PyruvateInflow(curve, np.ones(phantom.shape))
    elif driver == "dynamic_b0":
        phantom.dynamic_b0 = DynamicB0(curve, np.ones(phantom.shape))
    elif driver == "delayed_conversion":
        phantom.conversion_start_s = 0.5e-3
    else:
        phantom.initial_spin_density_maps = {
            "Pyruvate": np.ones(phantom.shape),
            "Lactate": np.full(phantom.shape, 0.1),
        }
        phantom.equilibrium_polarization = 0.02
    program = SequenceProgram(
        (RFEvent(0.0, np.asarray([80.0 + 20.0j, 0.0j]), 20e-6),),
        duration_s=1e-3,
    )
    status_messages = []

    native = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program,
        phantom,
        sequence_kernel=kernel,
        status_callback=status_messages.append,
        checkpoints_s=(0.5e-3,),
    )
    optimized = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program,
        phantom,
        sequence_kernel="optimized",
        checkpoints_s=(0.5e-3,),
    )

    assert native.metadata["requested_sequence_kernel"] == kernel
    assert native.metadata["sequence_kernel"] == kernel
    assert native.metadata["native_fallback_reason"] is None
    assert native.metadata["native_rf_block_enabled"] is True
    if driver != "dynamic_b0":
        assert native.metadata["native_hybrid"] is True
        assert native.metadata["native_longitudinal_step_enabled"] is False
        reason_fragment = {
            "inflow": "pyruvate inflow",
            "delayed_conversion": "delayed conversion",
            "concentration_tracking": "concentration tracking",
        }[driver]
        assert reason_fragment in native.metadata["native_longitudinal_fallback_reason"]
        assert any("hybrid dynamic kernel" in message for message in status_messages)
    else:
        assert native.metadata["native_hybrid"] is False
        assert native.metadata["native_longitudinal_step_enabled"] is True
        assert native.metadata["native_longitudinal_fallback_reason"] is None
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
        "checkpoint_magnetization",
        "checkpoint_pool_magnetization",
    ):
        assert np.array_equal(getattr(native, name), getattr(optimized, name))


def test_native_inflow_with_spatial_tx_reports_complete_optimized_fallback():
    phantom = _dynamic_phantom()
    phantom.pyruvate_inflow = PyruvateInflow(
        TimeCurve((0.0, 1e-3), (0.0, 1.0), "linear", "hold"),
        np.ones(phantom.shape),
    )
    phantom.tx_sensitivity_map = np.asarray([1.0, 0.8]).reshape(phantom.shape)
    program = SequenceProgram(
        (RFEvent(0.0, np.asarray([80.0 + 20.0j, 0.0j]), 20e-6),),
        duration_s=1e-3,
    )

    native = BlochSimulator(use_parallel=True, num_threads=2).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="native_parallel"
    )
    optimized = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom, sequence_kernel="optimized"
    )

    assert native.metadata["requested_sequence_kernel"] == "native_parallel"
    assert native.metadata["sequence_kernel"] == "optimized"
    assert native.metadata["native_hybrid"] is False
    assert native.metadata["native_rf_block_enabled"] is False
    assert "spatial transmit sensitivity" in native.metadata["native_fallback_reason"]
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
    ):
        assert np.array_equal(getattr(native, name), getattr(optimized, name))


@pytest.mark.parametrize("num_threads", [1, 2, 4, 8])
def test_native_parallel_dynamic_kernel_is_bitwise_equal(num_threads):
    shape = (12, 12, 12)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.12, 0.12, 0.12),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.linspace(0.0, 0.1, np.prod(shape)).reshape(shape),
        b0_map=np.linspace(-8.0, 8.0, np.prod(shape)).reshape(shape),
        nucleus="C13",
    )
    raster = 20e-6
    program = SequenceProgram(
        (
            RFEvent(2 * raster, np.asarray([80.0 + 20.0j, 0.0j]), raster),
            GradientEvent("x", 0.0, np.linspace(-150.0, 200.0, 32), raster),
            ADCEvent(0.0, 16, 2 * raster, phase_offset_rad=0.37),
        ),
        duration_s=32 * raster,
    )
    kwargs = {
        "checkpoints_s": (0.0, 17 * raster, 32 * raster),
        "simulation_timestep_s": 5e-6,
    }
    simulator = BlochSimulator(use_parallel=True, num_threads=num_threads)
    reference = simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel="reference", **kwargs
    )
    native = simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel="native_parallel", **kwargs
    )

    assert native.metadata["sequence_kernel"] == "native_parallel"
    assert native.metadata["native_parallel_threads"] == num_threads
    assert native.metadata["native_rf_block_enabled"] is True
    assert native.metadata["native_rf_threads"] == num_threads
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
        "checkpoint_magnetization",
        "checkpoint_pool_magnetization",
    ):
        assert np.array_equal(getattr(native, name), getattr(reference, name))


@pytest.mark.parametrize("num_threads", [1, 4])
def test_native_fused_rf_block_tracks_coupled_concentration_inflow(num_threads):
    phantom = _dynamic_phantom(kpl=(0.03, 0.1))
    phantom.initial_spin_density_maps = {
        "Pyruvate": np.ones(phantom.shape),
        "Lactate": np.full(phantom.shape, 0.05),
    }
    phantom.equilibrium_polarization = 0.7
    phantom.pyruvate_inflow = PyruvateInflow(
        TimeCurve((0.0, 1e-3), (0.2, 0.5), "linear", "hold"),
        np.asarray([0.8, 1.1]).reshape(phantom.shape),
        TimeCurve((0.0, 1e-3), (0.4, 0.25), "linear", "hold"),
    )
    raster = 20e-6
    program = SequenceProgram(
        (
            RFEvent(
                0.0,
                np.linspace(40.0 + 15.0j, 110.0 - 10.0j, 12),
                raster,
            ),
            GradientEvent("z", 0.0, np.full(12, 120.0), raster),
            ADCEvent(12 * raster, 1, raster),
        ),
        duration_s=13 * raster,
    )
    simulator = BlochSimulator(use_parallel=True, num_threads=num_threads)
    optimized = simulator.simulate_dynamic_sequence(
        program,
        phantom,
        sequence_kernel="optimized",
        simulation_timestep_s=5e-6,
    )
    native = simulator.simulate_dynamic_sequence(
        program,
        phantom,
        sequence_kernel="native_parallel",
        simulation_timestep_s=5e-6,
    )

    assert native.metadata["native_concentration_inflow_step_enabled"] is True
    assert native.metadata["native_rf_fused_block_enabled"] is True
    assert native.metadata["native_rf_fused_blocks"] == 1
    assert native.metadata["native_rf_fused_intervals"] >= 8
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
    ):
        np.testing.assert_allclose(
            getattr(native, name),
            getattr(optimized, name),
            rtol=2e-13,
            atol=2e-13,
        )


def test_native_serial_fused_rf_block_keeps_one_worker(monkeypatch):
    import blochsimulator.dynamic_bloch_cy as dynamic_bloch_cy

    base = _dynamic_phantom()
    shape = (1024, 1, 1)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.02, 0.01, 0.01),
        pools=base.pools,
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.full(shape, 0.1),
        initial_spin_density_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.full(shape, 0.05),
        },
        equilibrium_polarization=0.7,
        pyruvate_inflow=PyruvateInflow(
            TimeCurve((0.0, 1e-3), (0.2, 0.5), "linear", "hold"),
            np.ones(shape),
            TimeCurve((0.0, 1e-3), (0.4, 0.25), "linear", "hold"),
        ),
    )
    raster = 20e-6
    program = SequenceProgram(
        (
            RFEvent(0.0, np.full(12, 80.0 + 10.0j), raster),
            ADCEvent(12 * raster, 1, raster),
        ),
        duration_s=13 * raster,
    )
    native_block = dynamic_bloch_cy.apply_dynamic_rf_block_with_concentration_inflow
    worker_counts = []

    def record_worker_count(*args):
        worker_counts.append(args[-1])
        return native_block(*args)

    monkeypatch.setattr(
        dynamic_bloch_cy,
        "apply_dynamic_rf_block_with_concentration_inflow",
        record_worker_count,
    )
    result = BlochSimulator(use_parallel=True, num_threads=4).simulate_dynamic_sequence(
        program,
        phantom,
        sequence_kernel="native_serial",
        simulation_timestep_s=5e-6,
    )

    assert worker_counts
    assert set(worker_counts) == {1}
    assert result.metadata["native_rf_threads"] == 1
    assert result.metadata["native_longitudinal_threads"] == 1


def test_native_parallel_dynamic_kernel_respects_tiny_block_budget():
    phantom = _dynamic_phantom()
    program = SequenceProgram((), duration_s=1e-3)
    simulator = BlochSimulator(use_parallel=True, num_threads=8)
    optimized = simulator.simulate_dynamic_sequence(
        program, phantom, sequence_kernel="optimized", checkpoints_s=(5e-4,)
    )
    native = simulator.simulate_dynamic_sequence(
        program,
        phantom,
        sequence_kernel="native_parallel",
        checkpoints_s=(5e-4,),
        memory_budget_bytes=1,
    )

    assert native.metadata["native_parallel_memory_limited"] is True
    assert native.metadata["native_parallel_threads"] == 1
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
        "checkpoint_magnetization",
        "checkpoint_pool_magnetization",
    ):
        assert np.array_equal(getattr(native, name), getattr(optimized, name))


@pytest.mark.parametrize("suffix", [".npz", ".h5", ".nc"])
def test_dynamic_phantom_round_trip(tmp_path, suffix):
    phantom = _dynamic_phantom()
    phantom.spectral_reference_ppm = 175.0
    phantom.spectral_window_center_ppm = 177.5
    path = phantom.save(tmp_path / f"dynamic{suffix}")
    loaded = DynamicSpectralPhantom.load(path)

    assert loaded.shape == phantom.shape
    assert [pool.name for pool in loaded.pools] == ["Pyruvate", "Lactate"]
    assert np.array_equal(loaded.kpl_map_s_inv, phantom.kpl_map_s_inv)
    assert loaded.spectral_window_center_ppm == pytest.approx(177.5)
    assert loaded.coordinate_system == "object_xyz"
    assert np.array_equal(loaded.affine_ijk_to_xyz_m, phantom.affine_ijk_to_xyz_m)


@pytest.mark.parametrize("suffix", [".npz", ".h5", ".nc"])
def test_dynamic_driver_round_trip(tmp_path, suffix):
    phantom = _dynamic_phantom()
    phantom.conversion_start_s = -1.5
    phantom.kinetics_time_offset_s = 0.75
    phantom.initial_spin_density_maps = {
        pool.name: np.full(phantom.shape, index + 1.0)
        for index, pool in enumerate(phantom.pools)
    }
    phantom.equilibrium_polarization = 1.0
    phantom.pyruvate_inflow = PyruvateInflow(
        TimeCurve((-2.0, 2.0), (0.0, 0.4), "linear", "zero"),
        np.ones(phantom.shape),
        polarization_curve=TimeCurve((-2.0, 2.0), (10000.0, 5000.0), "linear", "hold"),
    )
    phantom.dynamic_b0 = DynamicB0(
        TimeCurve((0.0, 2.0), (-3.0, 4.0), "linear", "hold"),
        np.linspace(0.5, 1.0, phantom.nvoxels).reshape(phantom.shape),
    )

    loaded = DynamicSpectralPhantom.load(phantom.save(tmp_path / f"drivers{suffix}"))

    assert (
        loaded.pyruvate_inflow.rate_curve_s_inv
        == phantom.pyruvate_inflow.rate_curve_s_inv
    )
    assert (
        loaded.pyruvate_inflow.polarization_curve
        == phantom.pyruvate_inflow.polarization_curve
    )
    assert loaded.equilibrium_polarization == pytest.approx(1.0)
    for pool in phantom.pools:
        assert np.array_equal(
            loaded.initial_spin_density_maps[pool.name],
            phantom.initial_spin_density_maps[pool.name],
        )
    assert np.array_equal(
        loaded.pyruvate_inflow.delivery_map, phantom.pyruvate_inflow.delivery_map
    )
    assert loaded.dynamic_b0.offset_curve_hz == phantom.dynamic_b0.offset_curve_hz
    assert loaded.conversion_start_s == phantom.conversion_start_s
    assert loaded.kinetics_time_offset_s == phantom.kinetics_time_offset_s
    assert np.array_equal(
        loaded.dynamic_b0.spatial_scale_map, phantom.dynamic_b0.spatial_scale_map
    )


def test_dynamic_phantom_xarray_dataset_labels_pools_and_coordinates():
    phantom = _dynamic_phantom()
    dataset = phantom.to_xarray()

    assert dataset["initial_concentration"].dims == ("species", "x", "y", "z")
    assert dataset["kpl_map_s_inv"].dims == ("x", "y", "z")
    assert list(dataset.coords["species"].values) == ["Pyruvate", "Lactate"]
    assert dataset.attrs["coordinate_system"] == "object_xyz"
    assert np.asarray(dataset.coords["x"]) == pytest.approx([-0.005, 0.005])


def test_phantom_design_builds_dynamic_pool_maps_and_kpl_regions():
    design = PhantomDesign(
        shape=(4, 2, 1),
        fov_m=(0.04, 0.02, 0.01),
        dynamic_enabled=True,
        shapes=[
            ShapeDefinition(
                "Object",
                kind="box",
                size=(1.0, 1.0, 1.0),
                t1_s=30.0,
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 1.0, 0.0, 1.0),
                    SpectralPeakDefinition("Lactate", 0.0, 12.0, 1.0),
                ],
            )
        ],
        kinetic_regions=[
            KineticRegionDefinition(
                "Tumor", "box", (0.75, 0.5, 0.5), (0.5, 1.0, 1.0), 0.08
            )
        ],
    )
    phantom = design.build()

    assert isinstance(phantom, DynamicSpectralPhantom)
    assert phantom.nucleus == "C13"
    assert np.array_equal(phantom.kpl_map_s_inv[:, 0, 0], [0.0, 0.0, 0.08, 0.08])
    assert PhantomDesign.from_phantom(phantom).dynamic_enabled


def test_phantom_design_uses_metabolite_specific_t1_for_dynamic_pools():
    design = PhantomDesign(
        shape=(1, 1, 1),
        fov_m=(0.01, 0.01, 0.01),
        dynamic_enabled=True,
        shapes=[
            ShapeDefinition(
                "Object",
                kind="box",
                size=(1.0, 1.0, 1.0),
                t1_s=99.0,
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 1.0, 0.0, 1.0, t1_s=30.0),
                    SpectralPeakDefinition("Lactate", 0.0, 12.0, 1.0, t1_s=25.0),
                ],
            )
        ],
    )

    phantom = design.build()

    assert [pool.t1 for pool in phantom.pools] == pytest.approx([30.0, 25.0])


def test_phantom_design_builds_inflow_and_dynamic_b0_drivers():
    design = PhantomDesign(
        shape=(2, 2, 1),
        fov_m=(0.02, 0.02, 0.01),
        dynamic_enabled=True,
        pyruvate_inflow_curve=TimeCurve((-2.0, 2.0), (0.0, 0.2), "linear", "zero"),
        conversion_start_s=-1.0,
        kinetics_time_offset_s=0.5,
        dynamic_b0_curve=TimeCurve((0.0, 2.0), (0.0, 5.0), "linear", "hold"),
        shapes=[
            ShapeDefinition(
                "Object",
                kind="box",
                size=(1.0, 1.0, 1.0),
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 0.0, 0.0, 1.0),
                    SpectralPeakDefinition("Lactate", 0.0, 12.0, 1.0),
                ],
            )
        ],
    )

    phantom = design.build()
    restored = PhantomDesign.from_phantom(phantom)

    assert phantom.pyruvate_inflow is not None
    assert np.all(phantom.pyruvate_inflow.delivery_map == 1.0)
    assert phantom.n_active == phantom.nvoxels
    assert phantom.dynamic_b0 is not None
    assert phantom.conversion_start_s == -1.0
    assert phantom.kinetics_time_offset_s == 0.5
    assert restored.pyruvate_inflow_curve == design.pyruvate_inflow_curve
    assert restored.conversion_start_s == design.conversion_start_s
    assert restored.kinetics_time_offset_s == design.kinetics_time_offset_s
    assert restored.dynamic_b0_curve == design.dynamic_b0_curve


def test_phantom_designer_exposes_kinetic_regions_and_preview():
    app = QApplication.instance() or QApplication([])
    design = PhantomDesign(
        shape=(4, 2, 1),
        fov_m=(0.04, 0.02, 0.01),
        dynamic_enabled=True,
        shapes=[
            ShapeDefinition(
                "Object",
                kind="box",
                size=(1.0, 1.0, 1.0),
                t1_s=30.0,
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 1.0, 0.0, 1.0),
                    SpectralPeakDefinition("Lactate", 0.0, 12.0, 1.0),
                ],
            )
        ],
    )
    dialog = SpectralPhantomDesignerDialog(design=design)
    dialog._add_kinetic_region("ellipsoid")
    dialog.kinetics_shape_preview_3d.set_shapes = MagicMock()
    dialog._update_kinetics_spatial_preview()
    spatial_call = dialog.kinetics_shape_preview_3d.set_shapes.call_args
    assert spatial_call.kwargs["selected_row"] == 0
    assert spatial_call.kwargs["highlighted_region"].kind == "ellipsoid"
    assert "kPL region 1" in dialog.kinetics_spatial_preview_info.text()
    dialog.inflow_enabled.setChecked(True)
    dialog.dynamic_b0_enabled.setChecked(True)
    dialog._preview()

    assert isinstance(dialog.phantom, DynamicSpectralPhantom)
    assert dialog.inspector.map_combo.currentText() == "kPL"
    assert dialog.phantom.kinetic_regions[0].kpl_s_inv == pytest.approx(0.05)
    assert dialog.phantom.pyruvate_inflow is not None
    assert dialog.phantom.dynamic_b0 is not None
    dialog.close()
    app.processEvents()


def test_phantom_designer_sorts_new_inflow_point_by_edited_time():
    app = QApplication.instance() or QApplication([])
    dialog = SpectralPhantomDesignerDialog()
    assert dialog.kinetics_controls_scroll.widgetResizable()
    assert dialog.kinetics_controls_scroll.minimumWidth() == 620
    assert dialog.kinetics_splitter.childrenCollapsible() is False
    table = dialog.inflow_curve_table

    dialog._add_curve_point(table)
    table.item(3, 1).setText("0.25")
    table.item(3, 0).setText("3.0")

    assert [float(table.item(row, 0).text()) for row in range(4)] == [
        0.0,
        3.0,
        5.0,
        6.0,
    ]
    assert [float(table.item(row, 1).text()) for row in range(4)] == [
        0.0,
        0.25,
        10.0,
        0.0,
    ]
    assert table.currentRow() == 1
    curve, polarization_curve = dialog._read_inflow_curves()
    assert curve.times_s == (0.0, 3.0, 5.0, 6.0)
    assert curve.values == (0.0, 0.25, 10.0, 0.0)
    assert polarization_curve.values == (1.0, 1.0, 10000.0, 1.0)

    dialog._add_curve_point(table)
    table.item(4, 1).setText("0.5")
    table.item(4, 0).setText("-2.0")
    curve, polarization_curve = dialog._read_inflow_curves()
    assert curve.times_s == (-2.0, 0.0, 3.0, 5.0, 6.0)
    assert curve.values[0] == 0.5
    assert polarization_curve.values[0] == pytest.approx(1.0)
    assert table.currentRow() == 0

    dialog.close()
    app.processEvents()


def test_phantom_designer_previews_negative_kinetic_preroll():
    app = QApplication.instance() or QApplication([])
    design = PhantomDesign(
        shape=(1, 1, 1),
        fov_m=(0.01, 0.01, 0.01),
        dynamic_enabled=True,
        default_kpl_s_inv=0.2,
        pyruvate_inflow_curve=TimeCurve(
            (-2.0, 0.0),
            (1.0, 1.0),
            "linear",
            "zero",
        ),
        conversion_start_s=-1.0,
        shapes=[
            ShapeDefinition(
                "Object",
                kind="box",
                size=(1.0, 1.0, 1.0),
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 0.0, 0.0, 1.0, t1_s=1e15),
                    SpectralPeakDefinition("Lactate", 0.0, 12.0, 1.0, t1_s=1e15),
                ],
            )
        ],
    )

    dialog = SpectralPhantomDesignerDialog(design=design)
    times_s, pyruvate = dialog.pyruvate_preview_curve.getData()
    _, lactate = dialog.lactate_preview_curve.getData()
    zero_index = int(np.searchsorted(times_s, 0.0))

    assert times_s[0] == -2.0
    assert dialog.conversion_start_s.value() == -1.0
    assert pyruvate[zero_index] + lactate[zero_index] == pytest.approx(2.0)
    assert lactate[zero_index] > 0.0
    assert "sequence-start Mz at t=0" in dialog.kinetics_preview_info.text()

    dialog.close()
    app.processEvents()


def test_phantom_designer_shifts_entire_kinetics_timeline_with_offset():
    app = QApplication.instance() or QApplication([])
    design = PhantomDesign(
        shape=(1, 1, 1),
        fov_m=(0.01, 0.01, 0.01),
        dynamic_enabled=True,
        default_kpl_s_inv=1.0,
        pyruvate_inflow_curve=TimeCurve(
            (0.0, 2.0),
            (1.0, 1.0),
            "linear",
            "zero",
        ),
        conversion_start_s=1.0,
        kinetics_time_offset_s=1.0,
        shapes=[
            ShapeDefinition(
                "Object",
                kind="box",
                size=(1.0, 1.0, 1.0),
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 0.0, 0.0, 1.0, t1_s=1e15),
                    SpectralPeakDefinition("Lactate", 0.0, 12.0, 1.0, t1_s=1e15),
                ],
            )
        ],
    )

    dialog = SpectralPhantomDesignerDialog(design=design)
    times_s, inflow = dialog.inflow_preview_curve.getData()
    _, pyruvate = dialog.pyruvate_preview_curve.getData()
    _, lactate = dialog.lactate_preview_curve.getData()
    zero_index = int(np.searchsorted(times_s, 0.0))

    assert times_s[0] == -1.0
    assert inflow[zero_index] == pytest.approx(1.0)
    assert pyruvate[zero_index] == pytest.approx(1.0)
    assert lactate[zero_index] == pytest.approx(0.0)
    assert dialog.conversion_start_line.value() == pytest.approx(0.0)
    assert "kinetics t at sequence t=0: 1 s" in dialog.kinetics_preview_info.text()
    assert [
        float(dialog.inflow_curve_table.item(row, 0).text())
        for row in range(dialog.inflow_curve_table.rowCount())
    ] == [0.0, 2.0]

    dialog.kinetics_time_offset_s.setValue(-1.0)
    times_s, inflow = dialog.inflow_preview_curve.getData()
    zero_index = int(np.searchsorted(times_s, 0.0))
    first_knot_index = int(np.searchsorted(times_s, 1.0))

    assert times_s[0] == 0.0
    assert inflow[zero_index] == pytest.approx(0.0)
    assert inflow[first_knot_index] == pytest.approx(1.0)
    assert dialog.conversion_start_line.value() == pytest.approx(2.0)
    dialog._sync_global()
    assert dialog.design.kinetics_time_offset_s == -1.0

    dialog.close()
    app.processEvents()


def test_phantom_designer_updates_conversion_plot_live_when_kpl_changes():
    app = QApplication.instance() or QApplication([])
    design = PhantomDesign(
        shape=(1, 1, 1),
        fov_m=(0.01, 0.01, 0.01),
        dynamic_enabled=True,
        default_kpl_s_inv=0.02,
        pyruvate_inflow_curve=TimeCurve(
            (0.0, 2.0, 4.0), (0.0, 1.0, 0.0), "linear", "zero"
        ),
        shapes=[
            ShapeDefinition(
                "Object",
                kind="box",
                size=(1.0, 1.0, 1.0),
                initial_mz=1.0,
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 1.0, 0.0, 1.0, t1_s=30.0),
                    SpectralPeakDefinition("Lactate", 0.0, 12.0, 1.0, t1_s=25.0),
                ],
            )
        ],
    )
    dialog = SpectralPhantomDesignerDialog(design=design)
    dialog.kinetics_preview_duration.setValue(10.0)
    _, lactate_low = dialog.lactate_preview_curve.getData()

    dialog.default_kpl.setValue(0.2)
    _, lactate_high = dialog.lactate_preview_curve.getData()
    _, inflow_values = dialog.inflow_preview_curve.getData()

    assert lactate_high[-1] > lactate_low[-1]
    assert np.max(inflow_values) == pytest.approx(1.0)
    assert "kPL=0.2" in dialog.kinetics_preview_info.text()

    dialog._add_kinetic_region("ellipsoid")
    dialog.kinetic_table.item(0, 8).setText("0.4")
    assert dialog.kinetics_preview_region.currentData() == 0
    assert "kPL=0.4" in dialog.kinetics_preview_info.text()
    dialog.close()
    app.processEvents()


def test_phantom_designer_explains_overlapping_pool_curves_and_shape_context():
    app = QApplication.instance() or QApplication([])
    design = PhantomDesign(
        shape=(1, 1, 1),
        fov_m=(0.01, 0.01, 0.01),
        dynamic_enabled=True,
        default_kpl_s_inv=0.0,
        shapes=[
            ShapeDefinition(
                "Identical pools",
                initial_mz=1.0,
                t1_s=1.0,
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 1.0, 0.0, 1.0),
                    SpectralPeakDefinition("Lactate", 1.0, 12.0, 1.0),
                ],
            )
        ],
    )

    dialog = SpectralPhantomDesignerDialog(design=design)
    _, pyruvate = dialog.pyruvate_preview_curve.getData()
    _, lactate = dialog.lactate_preview_curve.getData()
    info = dialog.kinetics_preview_info.text()

    assert np.allclose(pyruvate, lactate)
    assert dialog.kinetics_preview_shape.currentText() == "Identical pools"
    assert dialog.kinetics_preview_region.currentText() == "Default kPL"
    assert "kPL=0: no P→L conversion" in info
    assert "T1 relaxation toward polarization 1" in info
    assert "identical and overlap" in info
    assert "initialized, not created by conversion" in info

    dialog.zero_lactate_button.click()
    _, lactate_zero = dialog.lactate_preview_curve.getData()
    assert design.shapes[0].peaks[1].amplitude == 1.0
    assert design.shapes[0].peaks[1].initial_polarization == 0.0
    assert lactate_zero[0] == pytest.approx(0.0)
    assert lactate_zero[-1] == pytest.approx(1.0)
    assert "recovers thermally" in dialog.kinetics_preview_info.text()

    dialog.default_kpl.setValue(0.2)
    _, lactate_converted = dialog.lactate_preview_curve.getData()
    assert lactate_converted[0] == 0.0
    assert lactate_converted[-1] > 0.0
    assert "alongside thermal recovery" in dialog.kinetics_preview_info.text()
    dialog.close()
    app.processEvents()


def test_phantom_designer_does_not_show_hp_decay_when_hp_model_is_disabled():
    app = QApplication.instance() or QApplication([])
    design = PhantomDesign(
        shape=(1, 1, 1),
        fov_m=(0.01, 0.01, 0.01),
        dynamic_enabled=False,
        shapes=[
            ShapeDefinition(
                "Conventional pools",
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 1.0, 0.0, 1.0),
                    SpectralPeakDefinition("Lactate", 1.0, 12.0, 1.0),
                ],
            )
        ],
    )

    dialog = SpectralPhantomDesignerDialog(design=design)
    _, pyruvate = dialog.pyruvate_preview_curve.getData()
    _, lactate = dialog.lactate_preview_curve.getData()

    assert pyruvate is None
    assert lactate is None
    assert "Hyperpolarized preview inactive" in dialog.kinetics_preview_info.text()

    dialog.dynamic_enabled.setChecked(True)
    _, pyruvate = dialog.pyruvate_preview_curve.getData()
    assert pyruvate[0] == pytest.approx(1.0)
    assert pyruvate[-1] == pytest.approx(1.0)
    dialog.close()
    app.processEvents()


def test_phantom_designer_selects_shape_for_representative_kinetics_preview():
    app = QApplication.instance() or QApplication([])
    pool_peaks = [
        SpectralPeakDefinition("Pyruvate", 1.0, 0.0, 1.0, t1_s=30.0),
        SpectralPeakDefinition("Lactate", 0.0, 12.0, 1.0, t1_s=25.0),
    ]
    design = PhantomDesign(
        dynamic_enabled=True,
        shapes=[
            ShapeDefinition("Low Mz", initial_mz=1.0, peaks=list(pool_peaks)),
            ShapeDefinition("High Mz", initial_mz=3.0, peaks=list(pool_peaks)),
        ],
    )
    dialog = SpectralPhantomDesignerDialog(design=design)

    dialog.kinetics_preview_shape.setCurrentIndex(
        dialog.kinetics_preview_shape.findData(1)
    )
    _, pyruvate = dialog.pyruvate_preview_curve.getData()

    assert dialog.shape_list.currentRow() == 1
    assert pyruvate[0] == pytest.approx(3.0)
    assert "Representative voxel in High Mz" in dialog.kinetics_preview_info.text()
    dialog.close()
    app.processEvents()


def test_sequence_worker_dispatches_dynamic_phantom_to_dynamic_solver():
    app = QApplication.instance() or QApplication([])
    phantom = _dynamic_phantom()
    expected = object()
    simulator = MagicMock()
    simulator.simulate_dynamic_sequence.return_value = expected
    worker = SequenceSimulationThread(
        simulator,
        SequenceProgram((), duration_s=1e-3),
        phantom,
        (),
        live_preview=False,
    )
    results = []
    worker.result_ready.connect(results.append)

    worker.run()

    simulator.simulate_dynamic_sequence.assert_called_once()
    assert results == [expected]
    app.processEvents()
