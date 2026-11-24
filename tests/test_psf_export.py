"""PSF results archive: data, metadata and an example notebook that runs."""

import json
import os
from pathlib import Path
from unittest import mock
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from blochsimulator import BlochSimulator  # noqa: E402
from blochsimulator.psf_export import (  # noqa: E402
    psf_result_arrays,
    write_psf_results,
)
from blochsimulator.sequence import load_pulseq, make_pulseq_flash  # noqa: E402
from blochsimulator.sequence.acquisition import (  # noqa: E402
    CartesianAcquisition,
    make_cartesian_epi,
)
from blochsimulator.sequence.advanced_bssfp_builders import (  # noqa: E402
    make_pulseq_spectral_selective_bssfp,
)
from blochsimulator.sequence.pulseq_builders import make_pulseq_spiral  # noqa: E402
from blochsimulator.sequence.psf import (  # noqa: E402
    PointSource,
    estimate_psf,
    estimate_slice_profile,
)


def _simulator():
    return BlochSimulator(use_parallel=False)


def _round_trip(sequence, path: Path):
    sequence.write(str(path), v141_compat=True)
    return load_pulseq(path)


def _epi_psf():
    acquisition = CartesianAcquisition.epi(16, 16, (0.08, 0.08), 10e-6)
    program = make_cartesian_epi(
        acquisition,
        rf_duration_s=100e-6,
        prephaser_duration_s=100e-6,
        blip_duration_s=50e-6,
    )
    source = PointSource(position_m=(0.005, 0.0, 0.0), t2_star_s=3e-3)
    return estimate_psf(program, source, simulator=_simulator())


def _run_notebook(archive_path: Path, workdir: Path):
    """Extract the archive and execute every code cell of its notebook."""
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(archive.namelist())
        archive.extractall(workdir)
    notebook_name = next(name for name in names if name.endswith(".ipynb"))
    notebook = json.loads((workdir / notebook_name).read_text())
    assert notebook["nbformat"] == 4
    namespace = {}
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                # Agg cannot show figures; the cells only have to run.
                with mock.patch.object(plt, "show"):
                    exec(
                        compile("".join(cell["source"]), notebook_name, "exec"),
                        namespace,
                    )
    finally:
        os.chdir(previous)
        plt.close("all")
    return names, namespace


def test_psf_archive_holds_data_and_a_notebook_that_runs(tmp_path):
    psf = _epi_psf()
    path = write_psf_results(
        tmp_path / "epi psf", psf, title="EPI 16x16", settings={"scale": "linear"}
    )

    assert path.name == "epi psf.zip"
    names, namespace = _run_notebook(path, tmp_path / "unpacked")
    assert names == ["epi psf.ipynb", "epi psf.npz"]

    with np.load(tmp_path / "unpacked" / "epi psf.npz") as data:
        np.testing.assert_array_equal(data["psf"], psf.psf)
        np.testing.assert_array_equal(data["modulation"], psf.modulation)
        for role, offsets in zip(psf.roles, psf.offsets_m):
            np.testing.assert_array_equal(data[f"offsets_m_{role}"], offsets)
            np.testing.assert_array_equal(
                data[f"profile_{role}"], psf.profiles[role][1]
            )
        meta = json.loads(str(data["metadata_json"]))
    assert meta["title"] == "EPI 16x16"
    assert meta["roles"] == list(psf.roles)
    assert meta["settings"] == {"scale": "linear"}
    assert meta["source"]["t2_star_s"] == pytest.approx(3e-3)
    phase = meta["axis_metrics"]["phase"]
    assert phase["fwhm_voxels"] == pytest.approx(psf.axis_metrics["phase"].fwhm_voxels)
    assert namespace["meta"]["label"] == psf.label


def test_psf_arrays_load_without_pickles(tmp_path):
    psf = _epi_psf()
    path = tmp_path / "plain.npz"
    np.savez(path, **psf_result_arrays(psf, role_labels=["Phase (y)", "Read (x)"]))

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
    assert meta["role_labels"] == ["Phase (y)", "Read (x)"]


def test_notebook_runs_for_3d_spiral_and_slice_profile_results(tmp_path):
    bssfp = make_pulseq_spectral_selective_bssfp(
        fov_m=(56e-3, 28e-3, 21e-3),
        matrix=(8, 4, 4),
        target_frequency_offsets_hz=(0.0,),
        receiver_frequency_offsets_hz=(0.0,),
        target_metabolite_names=("A",),
        flip_angle_deg=(30.0,),
        repetitions=1,
        rf_phase_increment_deg=180.0,
    )
    volume = estimate_psf(
        _round_trip(bssfp, tmp_path / "bssfp.seq"),
        PointSource(position_m=(7e-3, 7e-3, 5.25e-3)),
        simulator=_simulator(),
    )
    spiral = estimate_psf(
        _round_trip(
            make_pulseq_spiral(
                fov_m=(0.08, 0.08),
                matrix=(16, 16),
                sampling_bandwidth_hz=50e3,
                read_offset_m=0.0,
            ),
            tmp_path / "spiral.seq",
        ),
        PointSource(position_m=(0.01, -0.005, 0.0)),
        simulator=_simulator(),
    )
    flash_program = _round_trip(
        make_pulseq_flash(
            fov_m=(0.08, 0.06),
            matrix=(8, 4),
            echo_time_s=5e-3,
            repetition_time_s=15e-3,
            flip_angle_deg=10.0,
            slice_thickness_m=3e-3,
        ),
        tmp_path / "flash.seq",
    )
    source = PointSource(position_m=(0.01, 0.0, 0.0))
    flash = estimate_psf(flash_program, source, simulator=_simulator())
    profile = estimate_slice_profile(flash_program, source, simulator=_simulator())

    cases = {
        "volume": (volume, None),
        "spiral": (spiral, None),
        "flash": (flash, profile),
    }
    for name, (psf, slice_profile) in cases.items():
        path = write_psf_results(
            tmp_path / f"{name}.zip", psf, slice_profile=slice_profile
        )
        _, namespace = _run_notebook(path, tmp_path / name)
        assert namespace["meta"]["kind"] == psf.kind
    assert volume.psf.ndim == 3
    assert spiral.kind == "spiral"
    flash_meta = namespace["meta"]
    assert len(flash_meta["slice_metrics"]) == len(profile.frame_labels)
