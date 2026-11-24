"""The "Export plot…" action of the GUI plots writes images and data files."""

import csv

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication, QMessageBox, QVBoxLayout, QWidget

from blochsimulator import BlochSimulator
from blochsimulator.sequence import load_pulseq, make_pulseq_flash
from blochsimulator.ui import plot_export
from blochsimulator.ui.kspace_view import KSpaceContext, KSpaceView
from blochsimulator.ui.psf_analysis_view import PSFAnalysisContext, PSFAnalysisView
from blochsimulator.ui.reconstruction_explorer import SequenceReconstructionExplorer
from blochsimulator.ui.sequence_geometry_viewer import SequenceGeometryViewer
from blochsimulator.ui.spoiling_analysis_view import SpoilingAnalysisView

ALL_FORMATS = plot_export.IMAGE_FORMATS + plot_export.DATA_FORMATS


@pytest.fixture
def export_to(monkeypatch, tmp_path):
    """Skip the save dialog: each call answers with the next queued format."""
    requests = []

    def answer(parent, title, default_path, formats):
        fmt = requests.pop(0)
        assert fmt in formats
        return tmp_path / f"{default_path.stem}_{fmt}.{fmt}", fmt

    def warn(parent, title, text):
        raise AssertionError(text)

    monkeypatch.setattr(plot_export, "request_export_path", answer)
    monkeypatch.setattr(QMessageBox, "warning", warn)

    def queue(*formats):
        requests.extend(formats)

    return queue


def _assert_written(path, fmt):
    assert path is not None and path.exists() and path.stat().st_size > 0
    assert path.suffix == f".{fmt}"
    head = path.read_bytes()[:8]
    if fmt == "png":
        assert head.startswith(b"\x89PNG")
    elif fmt == "pdf":
        assert head.startswith(b"%PDF")
    elif fmt == "svg":
        assert b"<svg" in path.read_bytes()
    elif fmt == "npz":
        assert np.load(path).files


def _click_export(export_to, button, formats=ALL_FORMATS):
    written = []
    original = plot_export.export_plot

    def recording(*args, **kwargs):
        path = original(*args, **kwargs)
        written.append(path)
        return path

    plot_export.export_plot = recording
    try:
        for fmt in formats:
            export_to(fmt)
            button.click()
            QApplication.processEvents()
    finally:
        plot_export.export_plot = original
    assert len(written) == len(formats)
    for path, fmt in zip(written, formats):
        _assert_written(path, fmt)
    return written


def _flash(tmp_path):
    sequence = make_pulseq_flash(
        fov_m=(0.08, 0.06),
        matrix=(8, 4),
        echo_time_s=5e-3,
        repetition_time_s=15e-3,
        flip_angle_deg=10.0,
        slice_thickness_m=3e-3,
    )
    path = tmp_path / "flash.seq"
    sequence.write(str(path), v141_compat=True)
    return load_pulseq(path)


# Helper -----------------------------------------------------------------------


def test_export_plot_writes_every_format_of_a_plot_widget(qt_application, tmp_path):
    plot = pg.PlotWidget(title="Signal")
    plot.addLegend()
    plot.plot([0.0, 1.0, 2.0], [1.0, 4.0, 9.0], name="Pyruvate")
    plot.resize(400, 300)

    for fmt in ALL_FORMATS:
        path = plot_export.export_plot(plot, tmp_path / "signal", fmt)
        _assert_written(path, fmt)

    rows = list(csv.reader((tmp_path / "signal.csv").open()))
    assert rows[0] == ["Signal__Pyruvate__x", "Signal__Pyruvate__y"]
    assert [float(value) for value in rows[2]] == [1.0, 4.0]
    archive = np.load(tmp_path / "signal.npz")
    np.testing.assert_allclose(archive["Signal__Pyruvate__y"], [1.0, 4.0, 9.0])


def test_export_plot_handles_pages_images_and_matplotlib(qt_application, tmp_path):
    page = QWidget()
    layout = QVBoxLayout(page)
    curve_plot = pg.PlotWidget(title="Profile")
    curve_plot.plot([0.0, 1.0], [2.0, 3.0])
    image_plot = pg.PlotWidget(title="Image")
    image_plot.addItem(pg.ImageItem(np.arange(12.0).reshape(3, 4)))
    layout.addWidget(curve_plot)
    layout.addWidget(image_plot)
    page.resize(500, 600)

    for fmt in ALL_FORMATS:
        _assert_written(plot_export.export_plot(page, tmp_path / "page", fmt), fmt)
    archive = np.load(tmp_path / "page.npz")
    np.testing.assert_array_equal(
        archive["Image__image"], np.arange(12.0).reshape(3, 4)
    )

    view = pg.ImageView()
    view.setImage(np.random.default_rng(0).random((5, 6)))
    for fmt in ("png", "svg", "pdf", "csv"):
        _assert_written(plot_export.export_plot(view, tmp_path / "view", fmt), fmt)

    from matplotlib.figure import Figure

    figure = Figure()
    axes = figure.add_subplot()
    axes.set_title("Decay")
    axes.plot([0, 1, 2], [3, 2, 1], label="Lactate")
    for fmt in ALL_FORMATS:
        _assert_written(plot_export.export_plot(figure, tmp_path / "mpl", fmt), fmt)
    np.testing.assert_array_equal(
        np.load(tmp_path / "mpl.npz")["Decay__Lactate__y"], [3, 2, 1]
    )


def test_export_plot_rejects_empty_data_and_unknown_formats(qt_application, tmp_path):
    with pytest.raises(ValueError, match="no data"):
        plot_export.export_plot(pg.PlotWidget(), tmp_path / "empty.csv")
    with pytest.raises(ValueError, match="Unsupported"):
        plot_export.export_plot(pg.PlotWidget(), tmp_path / "plot.tiff")


def test_context_menu_entry_exports_the_plot(qt_application, tmp_path, export_to):
    page = QWidget()
    layout = QVBoxLayout(page)
    plot = pg.PlotWidget(title="Spectrum")
    plot.plot([0.0, 1.0], [1.0, 0.0])
    silent = pg.PlotWidget()
    silent.setMenuEnabled(False)
    layout.addWidget(plot)
    layout.addWidget(silent)

    assert plot_export.install_plot_context_export(page) == 1
    assert plot_export.install_plot_context_export(page) == 0
    assert plot._plot_export_action.text() == plot_export.EXPORT_ACTION_TEXT

    export_to("svg")
    plot._plot_export_action.trigger()
    _assert_written(tmp_path / "Spectrum_svg.svg", "svg")


# Tabs -------------------------------------------------------------------------


def test_psf_tab_exports_images_and_psf_arrays(qt_application, tmp_path, export_to):
    program = _flash(tmp_path)
    view = PSFAnalysisView()
    view.context_provider = lambda: PSFAnalysisContext(
        program=program,
        simulator=BlochSimulator(use_parallel=False),
        simulation_timestep_s=None,
        title="Test sequence",
        token=("revision", 0),
    )
    view.token_provider = lambda: ("revision", 0)
    view.slice_profile_enabled.setChecked(False)
    view.resize(900, 900)
    view.show()
    view.compute()
    view.worker.wait()
    QApplication.processEvents()
    assert view.psf is not None, view.status_label.text()

    written = _click_export(export_to, view.export_button)
    archive = np.load(written[-1])
    np.testing.assert_array_equal(archive["psf"], view.psf.psf)
    assert any(key.endswith("__image") for key in archive.files), archive.files


def test_kspace_tab_exports_trajectory(qt_application, tmp_path, export_to):
    program = _flash(tmp_path)
    view = KSpaceView()
    view.context_provider = lambda: KSpaceContext(
        program=program, title="Test", token=("revision", 0), nucleus="H1"
    )
    view.token_provider = lambda: ("revision", 0)
    view.recalculate()
    view.resize(900, 900)

    written = _click_export(export_to, view.export_button)
    archive = np.load(written[-1])
    np.testing.assert_array_equal(
        archive["adc_k_cyc_per_m"], view.kspace.adc_k_cyc_per_m
    )


def test_setup_and_spoiling_tabs_export(qt_application, tmp_path, export_to):
    geometry = SequenceGeometryViewer()
    geometry.resize(800, 700)
    _click_export(export_to, geometry.export_button, ("png", "svg", "pdf"))

    spoiling = SpoilingAnalysisView()
    spoiling.resize(800, 700)
    _click_export(export_to, spoiling.export_button, ("png", "svg", "pdf"))


def test_reconstruction_explorer_exports_shown_page(
    qt_application, tmp_path, export_to
):
    from test_reconstruction_explorer import _framed_cartesian_dataset

    explorer = SequenceReconstructionExplorer()
    explorer.set_dataset(_framed_cartesian_dataset(), source="example.nc")
    explorer.resize(1000, 800)

    written = _click_export(export_to, explorer.export_plot_button)
    archive = np.load(written[-1])
    np.testing.assert_array_equal(archive["current_view"], explorer._current_display)


def test_sequence_workspace_and_designer_plots_export(
    qt_application, tmp_path, export_to
):
    from blochsimulator.ui.main_window import BlochSimulatorGUI

    window = BlochSimulatorGUI()
    window.app_settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    window.set_workspace_mode("sequence")
    window._ensure_sequence_simulation_workspace(window.sequence_simulation_tab_index)
    widget = window.sequence_simulation_widget
    widget.resize(1400, 900)
    widget._generate_selected_sequence()
    QApplication.processEvents()

    written = _click_export(export_to, widget.sequence_export_plot_button)
    archive = np.load(written[-1])
    assert any("RF" in key for key in archive.files), archive.files
    _click_export(export_to, widget.signal_export_plot_button, ("png", "svg", "pdf"))
    _click_export(export_to, widget.probe_export_plot_button, ("png", "svg", "pdf"))
    assert widget.rf_plot._plot_export_installed
    assert widget.signal_plot._plot_export_installed

    designer = window.sequence_designer
    designer.update_diagram()
    _click_export(export_to, designer.diagram_export_button)
