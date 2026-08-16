#!/usr/bin/env python3
"""Diagnose SS-bSSFP voxel spoiling and optional metabolite crosstalk.

The quick analysis reads a ``.blochproj`` file, measures the actual end-volume
gradient moment, converts it to phase cycles across one phantom voxel, and
compares the continuous rectangular-voxel result with midpoint subvoxel grids.

Use ``--run-species`` for the slower, component-resolved image simulation.  It
prints the signal from each spectral component in every reconstructed volume
for physical-gradient and ideal-crusher modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from blochsimulator.project_io import load_project
from blochsimulator.simulator import BlochSimulator


DEFAULT_PROJECT = (
    Path(__file__).resolve().parents[1]
    / "exports"
    / "debug"
    / "bssfp_no_pyr_fa.blochproj"
)


def phantom_voxel_extents_xyz_m(phantom) -> np.ndarray:
    """Return the physical X/Y/Z extent of one phantom cell."""
    affine = np.asarray(phantom.affine_ijk_to_xyz_m, dtype=float)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("phantom affine must be a finite 4x4 matrix")
    extents = np.sum(np.abs(affine[:3, :3]), axis=1)
    if np.any(extents <= 0):
        raise ValueError("the project must contain a three-dimensional phantom")
    return extents


def midpoint_grid_coherence(cycles_xyz, counts_xyz) -> float:
    """Coherent magnitude retained by a regular midpoint subvoxel grid."""
    value = 1.0
    for cycles, count in zip(cycles_xyz, counts_xyz):
        offsets = (np.arange(int(count), dtype=float) + 0.5) / int(count) - 0.5
        value *= abs(np.mean(np.exp(2j * np.pi * float(cycles) * offsets)))
    return float(value)


def continuous_voxel_coherence(cycles_xyz) -> float:
    """Exact result for a uniformly filled rectangular voxel."""
    return float(np.prod(np.abs(np.sinc(np.asarray(cycles_xyz, dtype=float)))))


def flash_through_slice_spoiler_report(
    *,
    cycles_per_slice: float = 4.0,
    slice_thickness_m: float = 3e-3,
    phantom_voxel_size_m: float = 0.5e-3,
    subvoxel_count: int = 4,
) -> dict:
    """Predict retained FLASH coherence for a through-slice spoiler."""
    values = np.asarray(
        (cycles_per_slice, slice_thickness_m, phantom_voxel_size_m), dtype=float
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("FLASH spoiler values must be positive and finite")
    subvoxel_count = int(subvoxel_count)
    if subvoxel_count < 1:
        raise ValueError("subvoxel_count must be positive")
    cycles_per_phantom_voxel = float(
        cycles_per_slice * phantom_voxel_size_m / slice_thickness_m
    )
    cycles_xyz = (0.0, 0.0, cycles_per_phantom_voxel)
    return {
        "cycles_per_slice": float(cycles_per_slice),
        "slice_thickness_m": float(slice_thickness_m),
        "phantom_voxel_size_m": float(phantom_voxel_size_m),
        "cycles_per_phantom_voxel": cycles_per_phantom_voxel,
        "continuous_retained_coherence": continuous_voxel_coherence(cycles_xyz),
        "grid_retained_coherence": midpoint_grid_coherence(
            cycles_xyz, (1, 1, subvoxel_count)
        ),
        "subvoxel_count": subvoxel_count,
        "cycles_per_slice_for_one_cycle_per_voxel": float(
            slice_thickness_m / phantom_voxel_size_m
        ),
    }


def first_end_spoiler_moment_xyz(program) -> np.ndarray:
    """Measure the first declared end-volume spoiler in cycles/metre."""
    definitions = dict(program.metadata.get("definitions", {}))
    end_times = np.asarray(
        definitions.get("EndImageSpoilerEndTimes", ()), dtype=float
    ).reshape(-1)
    if end_times.size == 0:
        return np.zeros(3, dtype=float)
    end_time = float(end_times[0])
    moments = np.zeros(3, dtype=float)
    for event in program.gradient_events:
        if np.isclose(event.end_s, end_time, rtol=0.0, atol=2e-8):
            moments["xyz".index(event.axis)] += (
                np.sum(event.samples_hz_per_m) * event.raster_s
            )
    return moments


def _metabolite_key(name: str) -> str:
    normalized = "".join(
        character for character in str(name).lower() if character.isalnum()
    )
    if "lactate" in normalized or normalized.startswith("lac"):
        return "lactate"
    if "pyruvate" in normalized or normalized.startswith("pyr") or normalized == "py":
        return "pyruvate"
    return normalized


def target_frequency_comparison(project) -> list[dict]:
    """Compare named RF carriers with the corresponding phantom peaks."""
    phantom = project["phantom"]
    program = project["program"]
    definitions = dict(program.metadata.get("definitions", {}))
    names_value = definitions.get("SpectralTargetNames", ())
    if isinstance(names_value, str):
        target_names = tuple(names_value.replace(",", " ").split())
    else:
        target_names = tuple(str(value) for value in names_value)
    target_offsets = np.asarray(
        definitions.get("SpectralTargetOffsetsHz", ()), dtype=float
    ).reshape(-1)
    receiver_offsets = np.asarray(
        definitions.get("SpectralReceiverOffsetsHz", ()), dtype=float
    ).reshape(-1)
    components = getattr(phantom, "species", getattr(phantom, "pools", ()))
    peak_offsets = {
        _metabolite_key(component.name): float(
            phantom.get_frequency_offset(
                component.name, phantom.field_strength, phantom.nucleus
            )
        )
        for component in components
    }
    rows = []
    for index, name in enumerate(target_names):
        key = _metabolite_key(name)
        matching = [
            value
            for component_key, value in peak_offsets.items()
            if key == component_key or key in component_key or component_key in key
        ]
        peak = matching[0] if len(matching) == 1 else np.nan
        rf = target_offsets[index] if index < target_offsets.size else np.nan
        receiver = receiver_offsets[index] if index < receiver_offsets.size else np.nan
        rows.append(
            {
                "target": name,
                "phantom_peak_hz": float(peak),
                "rf_carrier_hz": float(rf),
                "receiver_hz": float(receiver),
                "rf_minus_peak_hz": float(rf - peak),
            }
        )
    return rows


def analyze_project(project_path, grids: Iterable[int] = (2, 4, 5, 8)) -> dict:
    """Return the fast spoiler and frequency analysis as plain Python data."""
    project_path = Path(project_path).expanduser().resolve()
    project = load_project(project_path)
    phantom = project["phantom"]
    program = project["program"]
    voxel_sizes = phantom_voxel_extents_xyz_m(phantom)
    moment = first_end_spoiler_moment_xyz(program)
    cycles = np.abs(moment) * voxel_sizes
    continuous = continuous_voxel_coherence(cycles)
    grid_rows = [
        {
            "grid": f"{count}x{count}x{count}",
            "retained_coherence": midpoint_grid_coherence(cycles, (count,) * 3),
        }
        for count in grids
    ]
    recommended_cycles = np.ones(3, dtype=float)
    return {
        "project": str(project_path),
        "phantom_shape": tuple(int(value) for value in phantom.shape),
        "phantom_fov_m": tuple(float(value) for value in phantom.fov),
        "phantom_voxel_size_m_xyz": tuple(float(value) for value in voxel_sizes),
        "spoiler_moment_cycles_per_m_xyz": tuple(float(value) for value in moment),
        "spoiler_cycles_per_voxel_xyz": tuple(float(value) for value in cycles),
        "continuous_retained_coherence": continuous,
        "grid_results": grid_rows,
        "recommended_cycles_per_voxel_xyz": tuple(recommended_cycles),
        "recommended_continuous_retained_coherence": continuous_voxel_coherence(
            recommended_cycles
        ),
        "frequency_comparison": target_frequency_comparison(project),
    }


def component_image_norms(
    project_path,
    *,
    timestep_s: float = 20e-6,
    spin_sampling=(1, 1, 1),
) -> list[dict]:
    """Run the slower component-resolved physical/ideal crusher comparison."""
    project = load_project(Path(project_path).expanduser().resolve())
    phantom = project["phantom"]
    program = project["program"]
    simulator = BlochSimulator(use_parallel=True)
    rows = []
    for component_name, component in phantom.to_component_phantoms(
        phantom.field_strength, phantom.nucleus
    ):
        for spoiler_mode in ("gradient", "ideal"):
            result = simulator.simulate_sequence(
                program,
                component,
                simulation_timestep_s=timestep_s,
                spin_sampling=spin_sampling,
                spoiler_mode=spoiler_mode,
                sequence_kernel="optimized",
            )
            images = np.abs(result.reconstruct_cartesian_3d())
            for volume, image in enumerate(images):
                rows.append(
                    {
                        "component": component_name,
                        "spoiler_mode": spoiler_mode,
                        "volume": int(volume),
                        "image_l2_norm": float(np.linalg.norm(image)),
                        "image_max": float(np.max(image)),
                    }
                )
    return rows


def plot_spoiler_response(output_path, current_cycles_xyz=None):
    """Plot exact and midpoint-grid coherence for equal XYZ spoiler cycles."""
    from matplotlib import pyplot as plt

    cycles = np.linspace(0.0, 6.0, 1201)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(
        cycles,
        np.abs(np.sinc(cycles)) ** 3,
        color="black",
        linewidth=2,
        label="continuous voxel",
    )
    for count in (2, 4, 5, 8):
        values = [
            midpoint_grid_coherence((value,) * 3, (count,) * 3) for value in cycles
        ]
        axis.plot(cycles, values, label=f"{count}x{count}x{count} midpoint grid")
    axis.axvline(
        1.0, color="#15803d", linestyle="--", label="recommended: 1 cycle/voxel"
    )
    if current_cycles_xyz is not None:
        current = float(np.mean(np.asarray(current_cycles_xyz, dtype=float)))
        axis.axvline(current, color="#b45309", linestyle=":", label="current mean")
    axis.set(
        xlabel="spoiler cycles per voxel on X, Y and Z",
        ylabel="retained coherent signal",
        ylim=(-0.02, 1.02),
        title="SS-bSSFP end-volume spoiler",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _print_report(analysis: dict) -> None:
    voxel_mm = 1e3 * np.asarray(analysis["phantom_voxel_size_m_xyz"])
    cycles = np.asarray(analysis["spoiler_cycles_per_voxel_xyz"])
    print("Project:", analysis["project"])
    print("Phantom voxel XYZ [mm]:", ", ".join(f"{value:.6g}" for value in voxel_mm))
    print("Spoiler cycles/voxel XYZ:", ", ".join(f"{value:.6g}" for value in cycles))
    print(
        "Continuous retained coherence:",
        f"{100 * analysis['continuous_retained_coherence']:.6g}%",
    )
    for row in analysis["grid_results"]:
        print(
            f"{row['grid']} retained coherence:",
            f"{100 * row['retained_coherence']:.6g}%",
        )
    print("\nRF/phantom frequencies:")
    for row in analysis["frequency_comparison"]:
        print(
            f"  {row['target']}: peak {row['phantom_peak_hz']:.6g} Hz, "
            f"RF {row['rf_carrier_hz']:.6g} Hz, receiver "
            f"{row['receiver_hz']:.6g} Hz, RF-peak "
            f"{row['rf_minus_peak_hz']:+.6g} Hz"
        )


def _print_flash_report(report: dict) -> None:
    print("\nFLASH through-slice example:")
    print(
        f"  {report['cycles_per_slice']:.6g} cycles across "
        f"{1e3 * report['slice_thickness_m']:.6g} mm correspond to "
        f"{report['cycles_per_phantom_voxel']:.6g} cycles per "
        f"{1e3 * report['phantom_voxel_size_m']:.6g} mm phantom voxel."
    )
    print(
        "  Retained coherence: continuous "
        f"{100 * report['continuous_retained_coherence']:.6g}%, "
        f"{report['subvoxel_count']}-point grid "
        f"{100 * report['grid_retained_coherence']:.6g}%."
    )
    print(
        "  One cycle per phantom voxel requires "
        f"{report['cycles_per_slice_for_one_cycle_per_voxel']:.6g} cycles/slice."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", nargs="?", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--run-species", action="store_true")
    parser.add_argument("--timestep-us", type=float, default=20.0)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--flash-example",
        action="store_true",
        help="also calculate the 4 cycles/3 mm/0.5 mm FLASH example",
    )
    parser.add_argument("--flash-cycles-per-slice", type=float, default=4.0)
    parser.add_argument("--flash-slice-mm", type=float, default=3.0)
    parser.add_argument("--flash-voxel-mm", type=float, default=0.5)
    parser.add_argument("--flash-subvoxels", type=int, default=4)
    args = parser.parse_args(argv)

    analysis = analyze_project(args.project)
    _print_report(analysis)
    if args.flash_example:
        analysis["flash_example"] = flash_through_slice_spoiler_report(
            cycles_per_slice=args.flash_cycles_per_slice,
            slice_thickness_m=args.flash_slice_mm * 1e-3,
            phantom_voxel_size_m=args.flash_voxel_mm * 1e-3,
            subvoxel_count=args.flash_subvoxels,
        )
        _print_flash_report(analysis["flash_example"])
    if args.run_species:
        analysis["component_image_norms"] = component_image_norms(
            args.project, timestep_s=args.timestep_us * 1e-6
        )
        print("\nComponent-resolved reconstructed image norms:")
        for row in analysis["component_image_norms"]:
            print(
                f"  {row['component']} | {row['spoiler_mode']} | "
                f"volume {row['volume']}: L2={row['image_l2_norm']:.6g}, "
                f"max={row['image_max']:.6g}"
            )
    if args.plot:
        output = plot_spoiler_response(
            args.plot, analysis["spoiler_cycles_per_voxel_xyz"]
        )
        print("\nPlot:", output)
    if args.json:
        output = args.json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print("JSON:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
