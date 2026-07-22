import pytest

from blochsimulator.sequence import (
    ConvergenceCriteria,
    SequenceConvergenceCase,
    make_adiabatic_inversion_case,
    make_default_sequence_convergence_cases,
    make_hard_pulse_case,
    run_sequence_convergence_suite,
)


def test_default_suite_covers_supported_sequence_families():
    cases = make_default_sequence_convergence_cases()

    assert [case.name for case in cases] == [
        "hard_pulse_fid_gre",
        "slice_selective_epi_ute_csi",
        "spin_echo_refocusing",
        "bssfp_repeated_train",
        "mprage_inversion_readout_train",
        "spectral_selective_phase_modulated",
        "adiabatic_frequency_sweep",
    ]
    assert {case.family for case in cases} == {
        "FID/GRE hard pulse",
        "Slice-selective imaging",
        "Spin echo",
        "bSSFP",
        "MPRAGE/inversion recovery",
        "Spectral-selective RF",
        "Adiabatic inversion",
    }
    assert all(case.probes.n_spins >= 27 for case in cases)
    assert all(case.program.metadata["convergence_case"] == case.name for case in cases)


def test_suite_reports_global_limit_and_limiting_sequence_class():
    raster = 50e-6
    cases = (
        make_hard_pulse_case(raster),
        make_adiabatic_inversion_case(raster),
    )

    result = run_sequence_convergence_suite(
        cases,
        timesteps_s=(None, raster, 2 * raster),
        criteria=ConvergenceCriteria(
            max_vector_error=1e-3,
            rms_vector_error=1e-3,
        ),
    )

    native, native_raster, coarse = result.summary_records()
    assert native["all_passed"]
    assert native_raster["all_passed"]
    assert not coarse["all_passed"]
    assert coarse["limiting_case"] == "adiabatic_frequency_sweep"
    assert coarse["max_vector_error"] > 0.1
    assert result.coarsest_passing_timestep_s == pytest.approx(raster)
    assert len(result.to_records()) == len(cases) * 3


def test_suite_rejects_duplicate_case_names():
    case = make_hard_pulse_case()
    duplicate = SequenceConvergenceCase(
        name=case.name,
        family="duplicate",
        description="duplicate name",
        program=case.program,
        probes=case.probes,
    )

    with pytest.raises(ValueError, match="names must be unique"):
        run_sequence_convergence_suite((case, duplicate), timesteps_s=(None,))
