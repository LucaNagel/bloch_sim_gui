import numpy as np
import pytest

from blochsimulator.sequence import load_pulseq
from blochsimulator.sequence.kspace_trajectory import calculate_kspace
from blochsimulator.sequence.pulseq_builders import make_pulseq_spiral
from blochsimulator.sequence.spiral_design import design_spiral


def test_design_reproduces_the_spiral_sequence_trajectory(tmp_path):
    parameters = dict(fov_m=(0.08, 0.08), matrix=(16, 16), sampling_bandwidth_hz=50e3)
    path = tmp_path / "spiral.seq"
    make_pulseq_spiral(**parameters).write(str(path), v141_compat=True)
    k = calculate_kspace(load_pulseq(path)).adc_k_cyc_per_m[:256, :2]

    design = design_spiral(**parameters)

    assert design.k_cyc_per_m.shape == (256, 2)
    k_max = 16 / (2 * 0.08)
    assert np.abs(design.k_cyc_per_m - k).max() < 1e-3 * k_max
    assert design.peak_gradient_mtm <= design.max_grad_mtm * (1 + 1e-9)
    assert design.peak_slew_tms <= design.max_slew_tms * (1 + 1e-6)
    assert design.readout_duration_s == pytest.approx(256 * design.dwell_s)


def test_carbon_limits_apply_to_the_physical_gradient():
    parameters = dict(fov_m=(0.12, 0.12), matrix=(32, 32), sampling_bandwidth_hz=50e3)
    proton = design_spiral(**parameters, nucleus="H1")
    carbon = design_spiral(**parameters, nucleus="C13")

    assert not proton.bandwidth_limited
    assert carbon.bandwidth_limited
    assert carbon.sampling_bandwidth_hz < 0.75 * proton.sampling_bandwidth_hz
    assert carbon.peak_gradient_mtm <= carbon.max_grad_mtm * (1 + 1e-9)
    assert carbon.peak_slew_tms <= carbon.max_slew_tms * (1 + 1e-6)
    # Same k-space path, slower: the physical gradient scales with 1/gamma.
    assert np.abs(carbon.k_cyc_per_m).max() == pytest.approx(
        np.abs(proton.k_cyc_per_m).max(), rel=1e-2
    )


def test_undersampled_spiral_aliases_and_t2_star_broadens_the_psf():
    parameters = dict(fov_m=(0.12, 0.12), matrix=(32, 32), sampling_bandwidth_hz=50e3)
    nyquist = design_spiral(**parameters, spiral_turns=16)
    sparse = design_spiral(**parameters, spiral_turns=6)

    assert nyquist.turn_spacing == pytest.approx(1.0)
    assert sparse.turn_spacing == pytest.approx(16 / 6)
    assert sparse.alias_free_fov_m == pytest.approx((0.045, 0.045))
    # Within one FOV only the undersampled spiral shows its aliasing ring.
    ideal = nyquist.sampling_psf()
    aliased = sparse.sampling_psf()
    assert aliased.peak_sidelobe_ratio > 2 * ideal.peak_sidelobe_ratio
    assert ideal.fwhm_voxels["x"] == pytest.approx(1.4, abs=0.2)
    assert ideal.peak_offset_voxels["x"] == pytest.approx(0.0, abs=1e-6)

    decayed = nyquist.sampling_psf(t2_star_s=0.2 * nyquist.readout_duration_s)
    assert decayed.fwhm_voxels["x"] > 1.5 * ideal.fwhm_voxels["x"]
    assert decayed.peak < ideal.peak


def test_design_rejects_invalid_input_and_exports_si_arrays():
    with pytest.raises(ValueError, match="nucleus"):
        design_spiral(
            fov_m=(0.1, 0.1), matrix=(8, 8), sampling_bandwidth_hz=1e4, nucleus="X"
        )
    with pytest.raises(ValueError, match="matrix"):
        design_spiral(fov_m=(0.1,), matrix=(8,), sampling_bandwidth_hz=1e4)
    design = design_spiral(fov_m=(0.1, 0.1), matrix=(8, 8), sampling_bandwidth_hz=1e4)
    exported = design.to_dict()
    assert exported["k_cyc_per_m"].shape == (64, 2)
    assert exported["sample_times_s"][0] == pytest.approx(0.5 * design.dwell_s)
