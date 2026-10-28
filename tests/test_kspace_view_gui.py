import sys

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator.sequence import load_pulseq, make_pulseq_flash
from blochsimulator.sequence.kspace_trajectory import calculate_kspace
from blochsimulator.sequence.pulseq_builders import make_pulseq_spiral
from blochsimulator.sequence.result import SequenceSimulationResult
from blochsimulator.ui.kspace_view import (
    KSpaceContext,
    KSpaceView,
    PLANES,
    peak_slew_t_per_m_per_s,
    readout_layout,
    sampling_psf,
)
from blochsimulator.ui.main_window import BlochSimulatorGUI
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget


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


def _spiral(tmp_path):
    sequence = make_pulseq_spiral(
        fov_m=(0.08, 0.08),
        matrix=(8, 8),
        sampling_bandwidth_hz=50_000.0,
        spiral_turns=4.0,
        repetitions=2,
    )
    path = tmp_path / "spiral.seq"
    sequence.write(str(path), v141_compat=True)
    return load_pulseq(path)


def _view(program, nucleus="H1"):
    view = KSpaceView()
    state = {"token": ("revision", 0)}
    view.context_provider = lambda: KSpaceContext(
        program=program,
        title="Test sequence",
        token=state["token"],
        nucleus=nucleus,
    )
    view.token_provider = lambda: state["token"]
    view.recalculate()
    return view, state


def test_readout_layout_groups_cartesian_and_spiral_frames(tmp_path):
    flash = _flash(tmp_path)
    layout = readout_layout(flash, calculate_kspace(flash))
    assert layout.kind == "cartesian"
    assert layout.matrix == (8, 4)
    assert layout.fov_m == pytest.approx((0.08, 0.06))
    assert len(layout.frames) == 1
    assert layout.frames[0][1] == tuple(range(4))

    spiral = _spiral(tmp_path)
    layout = readout_layout(spiral, calculate_kspace(spiral))
    assert layout.kind == "spiral"
    assert layout.matrix == (8, 8)
    assert [events for _, events in layout.frames] == [(0,), (1,)]


def test_sampling_psf_of_full_cartesian_grid_is_a_unit_peak():
    fov = (0.08, 0.08)
    matrix = (8, 8)
    ky, kx = np.meshgrid(
        (np.arange(8) - 4) / fov[1], (np.arange(8) - 4) / fov[0], indexing="ij"
    )
    psf = sampling_psf(np.column_stack((kx.ravel(), ky.ravel())), fov, matrix)
    assert psf.image.shape == (8, 8)
    assert psf.peak == pytest.approx(1.0, rel=0.05)
    # Cell centres of a full grid lie sqrt(2)/2 cells from the samples.
    assert psf.sampling_gap == pytest.approx(np.sqrt(0.5), rel=0.05)


def test_kspace_view_shows_frames_imperfections_and_cursor(tmp_path):
    view, state = _view(_flash(tmp_path))
    assert view.kspace is view.ideal_kspace
    assert view.readout_selector.maximum() == 4
    assert view.show_mode.currentText() == "Frame"
    assert view.selected_events() == (0, 1, 2, 3)
    view.render()
    assert view.psf is not None
    assert view.psf.matrix == (8, 4)
    names = [
        view.info_table.item(row, 0).text() for row in range(view.info_table.rowCount())
    ]
    assert "Detected acquisition" in names
    assert "Peak slew in window" in names

    view.delay_us[0].setValue(20.0)
    assert view.kspace is not view.ideal_kspace
    assert view.imperfections().delay_s[0] == pytest.approx(20e-6)
    shifted = view.kspace.event_k(0)[:, 0]
    ideal = view.ideal_kspace.event_k(0)[:, 0]
    assert not np.allclose(shifted, ideal)
    view.render()
    names = [
        view.info_table.item(row, 0).text() for row in range(view.info_table.rowCount())
    ]
    assert "Deviation from ideal" in names
    view.reset_imperfections()
    assert view.kspace is view.ideal_kspace

    view.show_mode.setCurrentIndex(0)
    view.readout_selector.setValue(3)
    assert view.selected_events() == (2,)
    start, stop = view.window_s
    assert start < view.kspace.adc_times_s[view.kspace.adc_event_slice(2).start] < stop
    view.render()
    before = view.cursor_s
    view._advance_animation()
    assert view.cursor_s > before
    view.set_cursor(stop + 1.0)
    assert view.cursor_s == pytest.approx(stop)
    assert "after excitation" in view.time_label.text()

    state["token"] = ("revision", 1)
    assert view._stale()
    view.recalculate()
    assert not view._stale()
    view.deleteLater()


def test_kspace_view_reports_missing_sequence():
    view = KSpaceView()

    def missing():
        raise ValueError("Choose a sequence first.")

    view.context_provider = missing
    view.recalculate()
    assert view.status_label.text() == "Choose a sequence first."
    assert view.kspace is None
    view.deleteLater()


def test_kspace_view_uses_nucleus_gamma_for_gradient_units(tmp_path):
    program = _spiral(tmp_path)
    proton, _ = _view(program, "H1")
    carbon, _ = _view(program, "C13")
    proton.render()
    carbon.render()
    _, proton_values = proton._waveform
    _, carbon_values = carbon._waveform
    assert np.allclose(proton_values, carbon_values)
    assert carbon.gamma_hz_per_t < proton.gamma_hz_per_t
    assert proton.layout_info.kind == "spiral"
    assert proton.show_mode.currentText() == "Frame"
    assert proton.frame_selector.count() == 2
    proton.deleteLater()
    carbon.deleteLater()


def test_kspace_tab_follows_sequence_mode_and_link(qt_application):
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    index = window.kspace_tab_index
    tabs = window.tab_widget
    assert tabs.tabText(index) == "k-Space Trajectory"
    assert not tabs.isTabVisible(index)

    window.set_workspace_mode("sequence")
    assert tabs.isTabVisible(index)
    sequence = window.sequence_simulation_widget
    view = sequence.kspace_trajectory_view
    assert not window.kspace_page.isAncestorOf(view)
    sequence.psf_analysis_link.linkActivated.emit("kspace")
    assert tabs.currentIndex() == index
    assert window.kspace_page.isAncestorOf(view)

    sequence.sequence_source.setCurrentIndex(sequence.FLASH_SOURCE)
    context = sequence._kspace_context()
    assert context.program is sequence.program
    assert context.nucleus == sequence.nucleus.currentText()
    assert context.token == sequence._psf_analysis_token()

    view.summary_label.linkActivated.emit("psf-analysis")
    assert tabs.currentIndex() == window.psf_analysis_tab_index

    window.set_workspace_mode("free")
    assert not tabs.isTabVisible(index)
    window.close()
    window.deleteLater()
    app.processEvents()


def test_peak_slew_ignores_split_raster_cells():
    raster = 10e-6
    edges = np.asarray([0.0, raster, raster + 1e-12, 2 * raster, 3 * raster])
    values = np.zeros((4, 3))
    values[:, 0] = [0.0, 1e6, 1e6, 2e6]
    slew = peak_slew_t_per_m_per_s(edges, values, 1e6)
    assert slew[0] == pytest.approx(1.0 / raster, rel=1e-6)
    assert slew[1] == 0.0


def test_large_sampling_psf_waits_for_request(tmp_path, monkeypatch):
    from blochsimulator.ui import kspace_view

    monkeypatch.setattr(kspace_view, "AUTO_PSF_SAMPLES", 10)
    view, _ = _view(_flash(tmp_path))
    view.render()
    assert view.psf is None
    assert not view.compute_psf_button.isHidden()
    view.compute_psf_button.click()
    assert view.psf is not None
    assert view.compute_psf_button.isHidden()

    # A small selection computes again after a deferred large one.
    view.delay_us[0].setValue(1.0)
    view.show_mode.setCurrentIndex(0)
    view.render()
    assert view.psf is not None
    view.show_mode.setCurrentIndex(1)
    view.render()
    assert view.psf is None
    view.show_mode.setCurrentIndex(0)
    view.render()
    assert view.psf is not None
    view.deleteLater()


def test_sampling_psf_survives_switching_to_unsupported_plane_and_back(tmp_path):
    view, _ = _view(_flash(tmp_path))
    view.render()
    assert view.psf is not None
    plane = view.plane_selector.currentText()
    other = next(
        name
        for name in PLANES
        if PLANES[name] != PLANES[plane] and PLANES[name] != (0, 1)
    )
    view.plane_selector.setCurrentText(other)
    view.render()
    assert view.psf is None
    view.plane_selector.setCurrentText(plane)
    view.render()
    assert view.psf is not None
    view.deleteLater()


def test_saved_result_without_layout_gets_spiral_from_program(tmp_path):
    widget = SequenceSimulationWidget()
    widget.program = _spiral(tmp_path)
    widget._infer_current_acquisition()
    assert widget.spiral_acquisition is not None
    samples = widget.spiral_acquisition.num_samples

    def result(samples, **metadata):
        return SequenceSimulationResult(
            signal=np.zeros(samples, dtype=complex),
            adc_times_s=np.arange(samples) * 1e-5,
            final_magnetization=np.zeros((1, 3)),
            checkpoint_magnetization=None,
            checkpoint_times_s=np.zeros(0),
            metadata={"spiral_acquisition": None, **metadata},
        )

    # Results simulated before spirals could be inferred.
    old = result(samples)
    assert widget.complete_result_acquisition(old)
    assert old.spiral_acquisition.num_frames == 2
    # A stored layout and a foreign ADC stream stay untouched.
    stored = result(samples, cartesian_acquisition={"type": "cartesian_2d"})
    assert not widget.complete_result_acquisition(stored)
    assert stored.metadata["spiral_acquisition"] is None
    foreign = result(samples + 1)
    assert not widget.complete_result_acquisition(foreign)
    assert foreign.metadata["spiral_acquisition"] is None
    widget.deleteLater()
