from argparse import Namespace

import numpy as np
import pytest

pypulseq = pytest.importorskip("pypulseq")

from benchmarks.common import (
    DEFAULT_PHANTOMS,
    SS_BSSFP_ENCODING_AXES,
    array_similarity,
    build_sequence,
    load_phantom,
    print_benchmark_header,
    print_phantom_summary,
    print_runtime_row,
    print_sequence_summary,
    print_simulation_start,
    result_similarity,
    write_and_load_sequence,
)
from benchmarks.benchmark_crushers import build_parser as build_crusher_parser
from benchmarks.benchmark_kernels import (
    DEFAULT_KERNEL_FLASH_MATRIX,
    DEFAULT_KERNEL_PHANTOM_MATRIX,
    DEFAULT_KERNEL_SUBVOXEL_SPINS,
    DYNAMIC_KERNELS,
    SEQUENCE_KERNELS,
    build_parser as build_kernel_parser,
    print_kernel_descriptions,
)
from benchmarks.benchmark_resolution import (
    DEFAULT_FLASH_MATRIX,
    DEFAULT_RESOLUTION_SCALES,
    _resolution_cases,
    build_parser as build_resolution_parser,
    build_resolution_flash_sequence,
    conservative_resample_2d,
    make_resolution_slice,
    make_resolution_volume,
)
from benchmarks.benchmark_sequences import build_parser as build_sequence_parser
from benchmarks.run_benchmarks import build_parser as build_runner_parser
from blochsimulator.dynamic_phantom import DynamicSpectralPhantom
from blochsimulator.spectral_phantom import SpectralPhantom


@pytest.mark.parametrize(
    ("parser_builder", "goal_text"),
    [
        (build_runner_parser, "one reproducible entry point"),
        (build_sequence_parser, "which sequence features"),
        (build_crusher_parser, "how many subvoxel spins"),
        (build_resolution_parser, "phantom discretization"),
        (build_kernel_parser, "speed and numerical agreement"),
    ],
)
def test_benchmark_help_explains_goal_usage_and_results(parser_builder, goal_text):
    help_text = parser_builder().format_help()

    assert "Goal" in help_text
    assert "Typical use" in help_text
    assert "Reading the output" in help_text
    assert goal_text in help_text
    assert "python -m benchmarks." in help_text


@pytest.mark.parametrize("alias", ["static", "dynamic"])
def test_ss_bssfp_benchmark_uses_read_z_for_supplied_phantoms(tmp_path, alias):
    phantom = load_phantom(DEFAULT_PHANTOMS[alias])
    sequence = build_sequence("ss_bssfp", phantom, profile="quick")
    program = write_and_load_sequence(sequence, tmp_path / f"{alias}.seq")
    definitions = program.metadata["definitions"]

    assert SS_BSSFP_ENCODING_AXES == ("+z", "+x", "+y")
    assert definitions["ReadoutAxis"] == "+z"
    assert definitions["PhaseEncodingAxis"] == "+x"
    assert definitions["PartitionEncodingAxis"] == "+y"
    assert np.asarray(definitions["IdealSpoilerEndTimes"]).size == int(
        definitions["Repetitions"]
    )


def test_array_similarity_has_clear_error_and_similarity_endpoints():
    reference = np.asarray([1 + 1j, 2 - 1j])
    identical = array_similarity(reference, reference, prefix="signal")
    zero = array_similarity(reference, np.zeros_like(reference), prefix="signal")

    assert identical["signal_relative_l2_error"] == pytest.approx(0.0)
    assert identical["signal_l2_similarity"] == pytest.approx(1.0)
    assert identical["signal_complex_correlation"] == pytest.approx(1.0)
    assert zero["signal_relative_l2_error"] == pytest.approx(1.0)
    assert zero["signal_l2_similarity"] == pytest.approx(0.5)
    assert zero["signal_complex_correlation"] == pytest.approx(0.0)

    zero_reference = array_similarity(
        np.zeros_like(reference), reference, prefix="signal"
    )
    assert zero_reference["signal_relative_l2_error"] == pytest.approx(1.0)
    assert zero_reference["signal_l2_similarity"] == pytest.approx(0.0)


def test_result_similarity_reports_post_crusher_samples():
    class Result:
        signal = np.asarray([1 + 0j, 2 + 0j, 3 + 0j])
        adc_times_s = np.asarray([0.5, 1.5, 2.5])
        final_magnetization = np.ones((1, 1, 1, 3))
        species_signal = None
        metadata = {"declared_ideal_spoiler_end_times_s": [1.0]}

    metrics = result_similarity(Result(), Result())

    assert metrics["post_crusher_signal_relative_l2_error"] == 0.0
    assert metrics["post_crusher_signal_l2_similarity"] == 1.0


def test_conservative_2d_resampling_preserves_constant_and_integral():
    source = np.arange(24, dtype=float).reshape(4, 6)

    for shape in ((2, 3), (3, 5), (8, 12)):
        resampled = conservative_resample_2d(source, shape)
        assert resampled.shape == shape
        assert np.mean(resampled) == pytest.approx(np.mean(source))
    assert conservative_resample_2d(np.ones((4, 6)), (7, 9)) == pytest.approx(1.0)


def test_resolution_cases_support_phantoms_finer_than_source_and_acquisition():
    source = load_phantom(DEFAULT_PHANTOMS["static"])
    args = build_resolution_parser().parse_args(
        ["--flash-matrix", "16", "12", "--phantom-matrices", "20x24", "84X112"]
    )

    cases = _resolution_cases(
        source,
        scales=args.resolution_scales,
        phantom_matrices=args.phantom_matrices,
    )

    assert args.resolution_scales is None
    assert [case["shape"] for case in cases] == [(20, 24), (84, 112)]
    assert cases[-1]["scale_x"] == pytest.approx(2.0)
    assert cases[-1]["scale_y"] == pytest.approx(2.0)
    assert cases[-1]["shape"][0] > args.flash_matrix[0]
    assert cases[-1]["shape"][1] > args.flash_matrix[1]


def test_default_resolution_sweep_extends_beyond_source_grid():
    source = load_phantom(DEFAULT_PHANTOMS["static"])
    cases = _resolution_cases(source)

    assert DEFAULT_RESOLUTION_SCALES == (0.25, 0.5, 1.0, 2.0)
    assert cases[0]["shape"] == (10, 14)
    assert cases[-1]["shape"] == (84, 112)


@pytest.mark.parametrize(
    ("alias", "expected_type"),
    [("static", SpectralPhantom), ("dynamic", DynamicSpectralPhantom)],
)
def test_resolution_slice_retains_supplied_phantom_model(alias, expected_type):
    source = load_phantom(DEFAULT_PHANTOMS[alias])
    slice_index = source.shape[2] // 2
    phantom = make_resolution_slice(source, (10, 14), slice_index)

    assert isinstance(phantom, expected_type)
    assert phantom.shape == (10, 14, 1)
    assert phantom.fov[:2] == source.fov[:2]
    assert phantom.fov[2] == pytest.approx(source.fov[2] / source.shape[2])
    if isinstance(phantom, DynamicSpectralPhantom):
        assert phantom.pyruvate_inflow is not None
        assert (
            phantom.pyruvate_inflow.rate_curve_s_inv
            == source.pyruvate_inflow.rate_curve_s_inv
        )
        assert phantom.kpl_map_s_inv == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("alias", "expected_type"),
    [("static", SpectralPhantom), ("dynamic", DynamicSpectralPhantom)],
)
def test_resolution_volume_retains_every_z_plane(alias, expected_type):
    source = load_phantom(DEFAULT_PHANTOMS[alias])
    phantom = make_resolution_volume(source, (10, 14))

    assert isinstance(phantom, expected_type)
    assert phantom.shape == (10, 14, source.shape[2])
    assert phantom.fov == source.fov
    assert phantom.fov[2] / phantom.shape[2] == pytest.approx(
        source.fov[2] / source.shape[2]
    )
    assert phantom.spectral_reference_ppm == source.spectral_reference_ppm
    assert phantom.spectral_window_center_ppm == source.spectral_window_center_ppm
    assert phantom.spectral_bandwidth_ppm == source.spectral_bandwidth_ppm
    assert phantom.spectral_points == source.spectral_points
    assert phantom.metadata["resolution_benchmark_volume_mode"] == "full"

    source_center = np.asarray(source.affine_ijk_to_xyz_m) @ np.append(
        (np.asarray(source.shape) - 1.0) / 2.0, 1.0
    )
    phantom_center = np.asarray(phantom.affine_ijk_to_xyz_m) @ np.append(
        (np.asarray(phantom.shape) - 1.0) / 2.0, 1.0
    )
    assert phantom_center == pytest.approx(source_center)

    if isinstance(phantom, SpectralPhantom):
        for species in source.species:
            assert np.mean(
                phantom.concentration_maps[species.name], axis=(0, 1)
            ) == pytest.approx(
                np.mean(source.concentration_maps[species.name], axis=(0, 1))
            )
    else:
        assert phantom.pyruvate_inflow is not None
        assert phantom.pyruvate_inflow.delivery_map.shape == phantom.shape
        assert phantom.kpl_map_s_inv.shape == phantom.shape


def test_dynamic_resolution_flash_contains_temporal_frames():
    source = load_phantom(DEFAULT_PHANTOMS["dynamic"])
    args = build_resolution_parser().parse_args([])
    slice_index = source.shape[2] // 2
    sequence = build_resolution_flash_sequence(source, args, slice_index)

    assert sequence.definitions["DynamicFrames"] == 4
    assert sequence.definitions["DynamicFrameInterval"] == pytest.approx(1.0)
    assert sequence.definitions["Repetitions"] == 4
    assert DEFAULT_FLASH_MATRIX == (16, 16)
    assert tuple(sequence.definitions["MatrixSize"]) == DEFAULT_FLASH_MATRIX
    assert tuple(sequence.definitions["EncodingMatrixSize"][:2]) == DEFAULT_FLASH_MATRIX
    assert sequence.definitions["AcquisitionStartTimes"] == pytest.approx(
        [0.0, 1.0, 2.0, 3.0]
    )


def test_benchmark_progress_output_describes_case_before_and_after_simulation(
    tmp_path, capsys
):
    phantom = load_phantom(DEFAULT_PHANTOMS["dynamic"])
    sequence = build_sequence("ss_bssfp", phantom, profile="quick")
    sequence_path = tmp_path / "progress.seq"
    program = write_and_load_sequence(sequence, sequence_path)
    args = Namespace(
        threads=0,
        profile="quick",
        repeats=2,
        timestep_us=20.0,
        sequence_kernel="optimized",
        dynamic_kernel="optimized",
    )

    print(39 * "x-")
    print("Benchmark progress output test")
    print_benchmark_header(
        "Sequence comparison",
        args,
        tmp_path,
        selection={"Sequences": ["ss_bssfp"], "Phantoms": ["dynamic"]},
    )
    print_phantom_summary("dynamic", DEFAULT_PHANTOMS["dynamic"], phantom)
    print_sequence_summary("ss_bssfp", sequence_path, program, 0.125)
    print_simulation_start(
        repeat=1,
        repeats=2,
        spoiler_mode="gradient",
        spin_sampling=(1, 1, 5),
        timestep_us=20.0,
        details={"phantom": "dynamic", "sequence": "ss_bssfp"},
    )
    print_runtime_row(
        {
            "simulation_time_s": 1.25,
            "generation_time_s": 0.125,
            "adc_samples": 64,
            "compiled_intervals": 32,
            "active_voxels": 10,
            "simulated_spins": 50,
            "crusher": "gradient_z5",
            "post_crusher_signal_l2_similarity": 0.987654,
        }
    )

    output = capsys.readouterr().out
    assert "Benchmark: Sequence comparison" in output
    assert "Phantom: dynamic" in output
    assert "dynamic_spectral" in output
    assert "pyruvate inflow=yes" in output
    assert "Pools/metabolites: 2 components" in output
    assert "Metabolite centre offsets:" in output
    assert "Spectral display grid (not FLASH ADC):" in output
    assert "points=1024" in output
    assert "Sequence: ss_bssfp" in output
    assert "Axes: read=+z, phase=+x, partition=+y" in output
    assert "flip=[20, 10] deg" in output
    assert "Starting simulation 1/2" in output
    assert "crusher=gradient" in output
    assert "subvoxel spins=1x1x5" in output
    assert "Completed: Simulation 1.250 s" in output
    assert "post-crusher similarity 0.987654" in output


def test_kernel_benchmark_defaults_cover_both_kernel_families():
    args = build_kernel_parser().parse_args([])

    assert tuple(args.sequence_kernels) == SEQUENCE_KERNELS
    assert tuple(args.dynamic_kernels) == DYNAMIC_KERNELS
    assert args.kernel_phantom_matrix == DEFAULT_KERNEL_PHANTOM_MATRIX == (24, 24)
    assert tuple(args.kernel_flash_matrix) == DEFAULT_KERNEL_FLASH_MATRIX == (16, 16)
    assert (
        tuple(args.kernel_subvoxel_spins) == DEFAULT_KERNEL_SUBVOXEL_SPINS == (1, 1, 9)
    )


def test_kernel_benchmark_prints_physical_and_implementation_differences(capsys):
    print_kernel_descriptions(SEQUENCE_KERNELS, DYNAMIC_KERNELS)

    output = capsys.readouterr().out
    assert "ordinary Bloch propagation" in output
    assert "runs each metabolite independently" in output
    assert "coupled pyruvate and lactate" in output
    assert "kPL conversion" in output
    assert "reference: Direct allocation-heavy NumPy" in output
    assert "native_parallel: Native dynamic blocks with OpenMP" in output
    assert "metal_hybrid: Experimental Apple-GPU" in output
    assert "Metal capability:" in output
