"""Keep pyqtgraph from overflowing on degenerate image levels."""

import numpy as np
import pyqtgraph as pg

# 255 / rng must stay finite in float64; anything narrower is visually flat.
_MIN_RANGE = 1e-200


def _safe_levels(levels):
    """Return levels with a finite, non-degenerate span (or the input)."""
    if levels is None:
        return levels
    try:
        arr = np.asarray(levels, dtype=float)
    except (TypeError, ValueError):
        return levels
    if arr.ndim == 1 and arr.size == 2:
        lo, hi = arr
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return (0.0, 1.0)
        if abs(hi - lo) < _MIN_RANGE:
            return (lo, lo + 1.0)
    return levels


def install_image_levels_guard():
    """Sanitize ImageItem levels right before rendering (idempotent)."""
    if getattr(pg.ImageItem, "_levels_guard_installed", False):
        return
    original = pg.ImageItem.render

    def render(self):
        safe = _safe_levels(self.levels)
        if safe is not self.levels:
            self.levels = safe
        return original(self)

    pg.ImageItem.render = render
    pg.ImageItem._levels_guard_installed = True
