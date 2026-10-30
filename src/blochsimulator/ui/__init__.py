"""Shared desktop UI behavior."""

from .image_levels_guard import install_image_levels_guard
from .plot_interaction import (
    install_axis_constrained_zoom,
    install_data_range_zoom_limit,
)


install_axis_constrained_zoom()
install_data_range_zoom_limit()
install_image_levels_guard()
