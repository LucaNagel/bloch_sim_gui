"""One "Export plot…" action for every plot of the GUI.

A target is a pyqtgraph widget (``PlotWidget``, ``GraphicsLayoutWidget``,
``ImageView``), a matplotlib ``Figure`` or canvas, or any ``QWidget`` page
that holds several plots. :func:`export_plot` writes it as

* PNG and SVG; single pyqtgraph plots go through
  :class:`~blochsimulator.visualization.ImageExporter` (``pyqtgraph.exporters``),
  matplotlib figures through ``savefig`` and pages of several plots through
  ``QWidget.grab`` and ``QSvgGenerator``,
* PDF, through ``savefig`` or ``QPdfWriter``,
* CSV and NPZ with the plotted data: every visible curve (x and y) and image
  of the target, plus the arrays of ``data_provider`` when one is given. CSV
  holds the 1D arrays as columns; NPZ holds everything.

:func:`prompt_plot_export` asks for the file with one save dialog whose
filter chooses the format; :func:`make_export_button` and
:func:`install_plot_context_export` offer it as a button and as an entry of
the right-click menu of each plot.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QMarginsF, QRectF, QSize, QSizeF
from PyQt5.QtGui import QPainter, QPdfWriter, QPageSize
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QToolButton, QWidget

IMAGE_FORMATS = ("png", "svg", "pdf")
DATA_FORMATS = ("csv", "npz")
FILE_FILTERS = {
    "png": "PNG image (*.png)",
    "svg": "SVG vector image (*.svg)",
    "pdf": "PDF document (*.pdf)",
    "csv": "Plot data as CSV (*.csv)",
    "npz": "Plot data as NumPy archive (*.npz)",
}
EXPORT_ACTION_TEXT = "Export plot…"

DataProvider = Callable[[], Dict[str, object]]


# Targets ---------------------------------------------------------------------


def _is_matplotlib(target) -> bool:
    try:
        from matplotlib.figure import Figure
    except ImportError:  # pragma: no cover - matplotlib is a dependency
        return False
    return isinstance(target, Figure) or isinstance(
        getattr(target, "figure", None), Figure
    )


def _single_pyqtgraph_plot(target):
    """Return the widget that ``ImageExporter`` exports, or ``None``."""
    if isinstance(target, pg.ImageView):
        return None
    if isinstance(target, (pg.PlotWidget, pg.GraphicsLayoutWidget)):
        return target
    return None


def _graphics_views(target) -> List[pg.GraphicsView]:
    if isinstance(target, pg.GraphicsView):
        return [target]
    if isinstance(target, pg.ImageView):
        return [target.ui.graphicsView]
    if isinstance(target, QWidget):
        return [
            view
            for view in target.findChildren(pg.GraphicsView)
            if view.isVisibleTo(target)
        ]
    return []


# Data --------------------------------------------------------------------------


def _slug(text: str, fallback: str) -> str:
    text = re.sub(r"<[^>]+>", "", str(text or ""))
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return text or fallback


def _owner_title(item) -> str:
    parent = item.parentItem()
    while parent is not None:
        if isinstance(parent, pg.PlotItem):
            label = getattr(parent, "titleLabel", None)
            return getattr(label, "text", "") or ""
        parent = parent.parentItem()
    return ""


def collect_plot_data(target) -> Dict[str, np.ndarray]:
    """Return the visible curves and images of ``target`` as named arrays.

    Curves give ``<plot>__<curve>__x`` and ``__y`` entries, images one
    ``<plot>__image`` entry each.
    """
    if _is_matplotlib(target):
        return _collect_matplotlib_data(target)
    data: Dict[str, np.ndarray] = {}
    for view_index, view in enumerate(_graphics_views(target)):
        scene = view.scene()
        if scene is None:
            continue
        for item_index, item in enumerate(scene.items()):
            if not item.isVisible():
                continue
            if isinstance(item, pg.PlotDataItem):
                x, y = item.getData()
                kind = "curve"
            elif isinstance(
                item, (pg.PlotCurveItem, pg.ScatterPlotItem)
            ) and not isinstance(item.parentItem(), pg.PlotDataItem):
                x, y = item.getData()
                kind = "points" if isinstance(item, pg.ScatterPlotItem) else "curve"
            elif isinstance(item, pg.ImageItem):
                image = item.image
                if image is None:
                    continue
                key = f"{_slug(_owner_title(item), f'plot{view_index}')}__image"
                data[_unique(data, key)] = np.asarray(image)
                continue
            else:
                continue
            if x is None or y is None or len(x) == 0:
                continue
            plot = _slug(_owner_title(item), f"plot{view_index}")
            name = item.name() if callable(getattr(item, "name", None)) else None
            curve = _slug(name, f"{kind}{item_index}")
            key = _unique(data, f"{plot}__{curve}")
            data[f"{key}__x"] = np.asarray(x)
            data[f"{key}__y"] = np.asarray(y)
    return data


def _unique(data: Dict[str, np.ndarray], key: str) -> str:
    if key not in data and f"{key}__x" not in data:
        return key
    index = 2
    while f"{key}_{index}" in data or f"{key}_{index}__x" in data:
        index += 1
    return f"{key}_{index}"


def _collect_matplotlib_data(target) -> Dict[str, np.ndarray]:
    figure = getattr(target, "figure", target)
    data: Dict[str, np.ndarray] = {}
    for axes_index, axes in enumerate(figure.axes):
        plot = _slug(axes.get_title(), f"axes{axes_index}")
        for line_index, line in enumerate(axes.get_lines()):
            label = line.get_label()
            curve = _slug(
                "" if str(label).startswith("_") else label, f"curve{line_index}"
            )
            key = _unique(data, f"{plot}__{curve}")
            data[f"{key}__x"] = np.asarray(line.get_xdata())
            data[f"{key}__y"] = np.asarray(line.get_ydata())
        for image in axes.get_images():
            data[_unique(data, f"{plot}__image")] = np.asarray(image.get_array())
    return data


def _write_npz(path: Path, data: Dict[str, object]) -> None:
    arrays = {key: np.asarray(value) for key, value in data.items()}
    np.savez_compressed(path, **arrays)


def _write_csv(path: Path, data: Dict[str, object]) -> None:
    arrays = {key: np.asarray(value) for key, value in data.items()}
    columns = {key: value for key, value in arrays.items() if value.ndim <= 1}
    if not columns:
        images = [value for value in arrays.values() if value.ndim == 2]
        if not images:
            raise ValueError(
                "The plot holds no curves or 2D images that fit a CSV file; "
                "use NPZ instead."
            )
        image = images[0]
        if np.iscomplexobj(image):
            image = np.abs(image)
        np.savetxt(path, image, delimiter=",")
        return
    expanded: Dict[str, np.ndarray] = {}
    for key, value in columns.items():
        value = np.atleast_1d(value)
        if np.iscomplexobj(value):
            expanded[f"{key}_real"] = value.real
            expanded[f"{key}_imag"] = value.imag
        else:
            expanded[key] = value
    length = max(len(value) for value in expanded.values())
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(expanded))
        for row in range(length):
            writer.writerow(
                [
                    repr(value[row].item()) if row < len(value) else ""
                    for value in expanded.values()
                ]
            )


# Images ------------------------------------------------------------------------


def _render_widget_vector(widget: QWidget, path: Path, fmt: str) -> None:
    size = widget.size()
    if size.width() < 2 or size.height() < 2:
        size = widget.sizeHint().expandedTo(QSize(640, 480))
        widget.resize(size)
    if fmt == "svg":
        from PyQt5.QtSvg import QSvgGenerator

        generator = QSvgGenerator()
        generator.setFileName(str(path))
        generator.setSize(size)
        generator.setViewBox(QRectF(0, 0, size.width(), size.height()))
        generator.setTitle(widget.windowTitle() or path.stem)
        painter = QPainter(generator)
    else:
        # One widget pixel per PDF point keeps the on-screen proportions.
        writer = QPdfWriter(str(path))
        writer.setResolution(72)
        writer.setPageSize(
            QPageSize(QSizeF(size.width(), size.height()), QPageSize.Point)
        )
        writer.setPageMargins(QMarginsF(0, 0, 0, 0))
        painter = QPainter(writer)
        scale = min(
            writer.width() / max(size.width(), 1),
            writer.height() / max(size.height(), 1),
        )
        painter.scale(scale, scale)
    try:
        widget.render(painter)
    finally:
        painter.end()


def _export_image(target, path: Path, fmt: str, width: Optional[int]) -> None:
    if _is_matplotlib(target):
        figure = getattr(target, "figure", target)
        figure.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        return
    plot = _single_pyqtgraph_plot(target)
    if plot is not None and fmt in ("png", "svg"):
        from ..visualization import ImageExporter

        written = ImageExporter().export_pyqtgraph_plot(
            plot, str(path), format=fmt, width=width
        )
        if not written:
            raise RuntimeError("pyqtgraph could not export the plot")
        return
    widget = target.ui.graphicsView if isinstance(target, pg.ImageView) else target
    if not isinstance(widget, QWidget):
        raise TypeError(f"Cannot export {type(target).__name__} as an image")
    if fmt == "png":
        pixmap = widget.grab()
        if pixmap.isNull() or not pixmap.save(str(path), "PNG"):
            raise RuntimeError("Qt could not write the PNG image")
    else:
        _render_widget_vector(widget, path, fmt)


def export_plot(
    target,
    path: Union[str, Path],
    fmt: Optional[str] = None,
    *,
    data_provider: Optional[DataProvider] = None,
    width: Optional[int] = None,
) -> Path:
    """Write ``target`` to ``path`` and return the written path.

    ``fmt`` is one of :data:`IMAGE_FORMATS` or :data:`DATA_FORMATS`; without
    it the file suffix decides. The suffix is corrected to the format.
    ``data_provider`` returns extra named arrays for CSV and NPZ, such as the
    full result behind the plotted curves.
    """
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".") or "png").lower()
    if fmt not in IMAGE_FORMATS + DATA_FORMATS:
        raise ValueError(f"Unsupported plot export format: {fmt}")
    if path.suffix.lower() != f".{fmt}":
        path = path.with_suffix(f".{fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt in DATA_FORMATS:
        data = collect_plot_data(target)
        if data_provider is not None:
            data.update(data_provider() or {})
        if not data:
            raise ValueError("The plot holds no data to export yet.")
        (_write_csv if fmt == "csv" else _write_npz)(path, data)
    else:
        _export_image(target, path, fmt, width)
    if not path.exists():
        raise RuntimeError(f"Export did not write {path}")
    return path


# Dialog and actions -------------------------------------------------------------


def _default_directory(parent) -> Path:
    window = parent.window() if isinstance(parent, QWidget) else None
    provider = getattr(window, "_get_export_directory", None)
    if callable(provider):
        try:
            return Path(provider())
        except Exception:
            pass
    return Path.cwd()


def request_export_path(
    parent, title: str, default_path: Path, formats: Tuple[str, ...]
) -> Tuple[Optional[Path], Optional[str]]:
    """Ask for the export file; return ``(path, format)`` or ``(None, None)``.

    Tests replace this function to skip the dialog.
    """
    filters = [FILE_FILTERS[fmt] for fmt in formats]
    filename, selected = QFileDialog.getSaveFileName(
        parent, title, str(default_path), ";;".join(filters)
    )
    if not filename:
        return None, None
    path = Path(filename)
    suffix = path.suffix.lower().lstrip(".")
    fmt = next(
        (fmt for fmt in formats if FILE_FILTERS[fmt] == selected),
        suffix if suffix in formats else formats[0],
    )
    if suffix in formats and suffix != fmt:
        # A typed suffix wins over the filter left at its default.
        fmt = suffix
    return path, fmt


def prompt_plot_export(
    parent: QWidget,
    target,
    default_name: str,
    *,
    data_provider: Optional[DataProvider] = None,
    title: str = "Export plot",
    include_data: bool = True,
) -> Optional[Path]:
    """Ask for a file and export ``target``; report errors in a message box.

    ``target`` may be a callable that returns the widget to export, so a
    button can follow the page currently shown.
    """
    if callable(target) and not isinstance(target, QWidget):
        target = target()
    if target is None:
        return None
    formats = IMAGE_FORMATS + (DATA_FORMATS if include_data else ())
    default_path = _default_directory(parent) / f"{_slug(default_name, 'plot')}.png"
    path, fmt = request_export_path(parent, title, default_path, formats)
    if path is None:
        return None
    try:
        written = export_plot(target, path, fmt, data_provider=data_provider)
    except Exception as exc:
        QMessageBox.warning(
            parent, "Export failed", f"Could not export the plot:\n{exc}"
        )
        return None
    logger = getattr(parent.window(), "log_message", None)
    if callable(logger):
        try:
            logger(f"Exported plot to: {written}")
        except Exception:
            pass
    return written


def make_export_button(
    parent: QWidget,
    target,
    default_name: str,
    *,
    data_provider: Optional[DataProvider] = None,
    tooltip: Optional[str] = None,
    object_name: Optional[str] = None,
) -> QToolButton:
    """Return an "Export plot…" button that exports ``target``."""
    button = QToolButton(parent)
    button.setText(EXPORT_ACTION_TEXT)
    button.setToolTip(
        tooltip
        or "Save the plots shown here as PNG, SVG or PDF, or their data as CSV or NPZ"
    )
    if object_name:
        button.setObjectName(object_name)
    button.clicked.connect(
        lambda: prompt_plot_export(
            parent, target, default_name, data_provider=data_provider
        )
    )
    return button


def install_plot_context_export(root, default_name: Optional[str] = None) -> int:
    """Add "Export plot…" to the right-click menu of each plot below ``root``.

    ``root`` is a plot widget or a page; each pyqtgraph ``PlotWidget`` found
    exports itself. Returns the number of plots that gained the entry.
    """
    if isinstance(root, pg.PlotWidget):
        plots = [root]
    elif isinstance(root, QWidget):
        plots = root.findChildren(pg.PlotWidget)
    else:
        plots = []
    count = 0
    for plot in plots:
        view_box = plot.getViewBox()
        menu = getattr(view_box, "menu", None)
        # Plots without a menu keep it off; their page has a button instead.
        if (
            menu is None
            or not view_box.menuEnabled()
            or getattr(plot, "_plot_export_installed", False)
        ):
            continue
        name = default_name or _slug(
            getattr(plot.plotItem.titleLabel, "text", ""), "plot"
        )
        menu.addSeparator()
        action = menu.addAction(EXPORT_ACTION_TEXT)
        action.triggered.connect(
            lambda _=False, widget=plot, stem=name: prompt_plot_export(
                widget, widget, stem
            )
        )
        plot._plot_export_installed = True
        plot._plot_export_action = action
        count += 1
    return count
