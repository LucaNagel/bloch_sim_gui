import warnings

import numpy as np
import pyqtgraph as pg
import pytest

import blochsimulator.ui  # noqa: F401  (installs the guard)


@pytest.mark.parametrize("levels", [(0, 5e-324), (0, 1e-310), (float("nan"), 1)])
def test_degenerate_levels_render_without_warning(levels):
    item = pg.ImageItem()
    item.setImage(np.random.rand(8, 8) * 1e-310, levels=levels, autoLevels=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        item.render()
