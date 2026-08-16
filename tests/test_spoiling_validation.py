import runpy
from pathlib import Path

import numpy as np
import pytest


HELPERS = runpy.run_path(
    str(Path(__file__).parents[1] / "examples" / "validate_ss_bssfp_spoiling.py")
)


def test_flash_example_reproduces_partial_through_slice_spoiling():
    report = HELPERS["flash_through_slice_spoiler_report"](
        cycles_per_slice=4.0,
        slice_thickness_m=3e-3,
        phantom_voxel_size_m=0.5e-3,
        subvoxel_count=4,
    )

    assert report["cycles_per_phantom_voxel"] == pytest.approx(2.0 / 3.0)
    assert report["continuous_retained_coherence"] == pytest.approx(
        abs(np.sinc(2.0 / 3.0))
    )
    assert report["grid_retained_coherence"] == pytest.approx(0.4330127018922194)
    assert report["cycles_per_slice_for_one_cycle_per_voxel"] == pytest.approx(6.0)


def test_regular_grid_aliases_integer_spoiler_cycles():
    continuous = HELPERS["continuous_voxel_coherence"]((4.0, 0.0, 0.0))
    sampled = HELPERS["midpoint_grid_coherence"]((4.0, 0.0, 0.0), (4, 1, 1))

    assert continuous == pytest.approx(0.0, abs=1e-15)
    assert sampled == pytest.approx(1.0)
