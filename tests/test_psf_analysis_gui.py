import sys

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator import BlochSimulator
from blochsimulator.sequence import load_pulseq, make_pulseq_flash
from blochsimulator.sequence.advanced_bssfp_builders import (
    make_pulseq_spectral_selective_bssfp,
)
from blochsimulator.ui.main_window import BlochSimulatorGUI
from blochsimulator.ui.psf_analysis_view import (
    PRESETS,
    PSFAnalysisContext,
    PSFAnalysisView,
)


def _flash(tmp_path):
    sequence = make_pulseq_flash(
        fov_m=(0.08, 0.06),
        matrix=(8, 4),
        echo_time_s=5e-3,
        repetition_time_s=15e-3,
        flip_angle_deg=10.0,
        slice_thickness_m=3e-3,
        slice_gap_m=1e-3,
        n_slices=2,
        slice_offset_m=4e-3,
    )
    path = tmp_path / "flash.seq"
    sequence.write(str(path), v141_compat=True)
    return load_pulseq(path)


def _bssfp_3d(tmp_path):
    sequence = make_pulseq_spectral_selective_bssfp(
        fov_m=(56e-3, 28e-3, 21e-3),
        matrix=(8, 4, 4),
        target_frequency_offsets_hz=(0.0,),
        receiver_frequency_offsets_hz=(0.0,),
        target_metabolite_names=("A",),
        flip_angle_deg=(30.0,),
        repetitions=1,
        rf_phase_increment_deg=180.0,
    )
    path = tmp_path / "bssfp_3d.seq"
    sequence.write(str(path), v141_compat=True)
    return load_pulseq(path)


def _view(program, token=("revision", 0)):
    view = PSFAnalysisView()
    state = {"token": token}

    def context():
        return PSFAnalysisContext(
            program=program,
            simulator=BlochSimulator(use_parallel=False),
            simulation_timestep_s=None,
            title="Test sequence",
            token=state["token"],
        )

    view.context_provider = context
    view.token_provider = lambda: state["token"]
    return view, state


def _compute(view):
    view.compute()
    assert view.worker is not None
    view.worker.wait()
    QApplication.processEvents()
    assert view.compute_button.isEnabled(), view.status_label.text()


def test_psf_view_computes_2d_psf_and_slice_profile(qt_application, tmp_path):
    view, state = _view(_flash(tmp_path))
    view.position_mm[0].setValue(10.0)
    view.position_mm[2].setValue(2.0)
    view.show()
    _compute(view)

    computation = view.computation
    assert computation is not None
    psf = view.psf
    assert psf.roles == ("phase", "read")
    assert psf.peak_efficiency == pytest.approx(1.0, abs=0.05)
    assert view.frame_selector.count() == 2
    assert view.frame_selector.isEnabled()
    assert not view.plane_selector.isEnabled()
    assert view.title_label.text().startswith("Test sequence · 2D frame")
    assert "(10, 0, 2) mm" in view.summary_label.text()

    profile = computation.slice_profile
    assert profile is not None and profile.profiles.shape[0] == 2
    table = view.metrics_table
    assert [table.item(row, 0).text() for row in range(table.rowCount())] == [
        "Read (+x)",
        "Phase (+y)",
        "Slice",
    ]
    assert float(table.item(0, 3).text()) == pytest.approx(1.2136, abs=0.01)
    assert float(table.item(2, 1).text()) == pytest.approx(3.0)
    assert float(table.item(2, 3).text()) == pytest.approx(1.0, abs=0.05)
    assert "outside the nominal slice" in view.metrics_summary.text()
    assert len(view.slice_plot.listDataItems()) == 2

    # Another frame reconstructs from the stored simulation.
    simulation = computation.simulation
    view.frame_selector.setCurrentIndex(1)
    assert view.psf.frame == 1
    assert view.computation.simulation is simulation
    view.image_scale.setCurrentText("dB")
    view.modulation_part.setCurrentText("Phase")
    image = view.image_item.image
    assert image.max() <= 0.5 and image.min() >= -60.0

    # A changed sequence marks the result as stale.
    state["token"] = ("revision", 1)
    view.hide()
    view.show()
    assert "sequence changed" in view.summary_label.text()
    view.close()


def test_psf_view_shows_3d_planes_without_slice_sweep(qt_application, tmp_path):
    view, _ = _view(_bssfp_3d(tmp_path))
    view.show()
    _compute(view)

    psf = view.psf
    assert psf.roles == ("partition", "phase", "read")
    assert view.computation.slice_profile is None
    assert "3D acquisitions" in view.computation.slice_message
    assert view.plane_selector.isEnabled()
    for plane, shape in (
        ("Read × phase", (16, 32)),
        ("Read × partition", (16, 32)),
        ("Phase × partition", (16, 16)),
    ):
        view.plane_selector.setCurrentText(plane)
        assert view.image_item.image.shape == shape
        assert view.image_item.image.max() == pytest.approx(1.0, abs=1e-3)
    assert view.metrics_table.rowCount() == 3
    view.close()


def test_psf_view_source_controls_and_nucleus_preset(qt_application):
    view = PSFAnalysisView()
    source = view.source()
    assert source.t1_s == 1e6 and source.t2_s == 1e6
    assert source.t2_star_s is None and source.equilibrium_mz == 1.0

    view.nucleus_provider = lambda: "C13"
    view.show()
    assert view.preset_selector.currentText() == PRESETS[2].label
    assert view.hyperpolarized.isChecked()
    source = view.source()
    assert source.t1_s == pytest.approx(30.0)
    assert source.t2_star_s == pytest.approx(0.02)
    assert source.equilibrium_mz == pytest.approx(1e-4)

    # A preset the user chose survives the nucleus.
    view.preset_selector.setCurrentIndex(0)
    view._preset_activated(0)
    view.hide()
    view.show()
    assert view.preset_selector.currentIndex() == 0

    view.t2_ms.setValue(10.0)
    view.t2_star_ms.setValue(20.0)
    view.context_provider = lambda: pytest.fail("context requested")
    view.compute()
    assert view.worker is None
    assert "T2* must not exceed T2" in view.status_label.text()

    def unavailable():
        raise ValueError("Generate a valid sequence first.")

    view.t2_star_ms.setValue(0.0)
    view.context_provider = unavailable
    view.compute()
    assert view.status_label.text() == "Generate a valid sequence first."
    view.close()


def test_psf_analysis_tab_follows_sequence_mode_and_link(qt_application):
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    index = window.psf_analysis_tab_index
    tabs = window.tab_widget
    assert tabs.tabText(index) == "PSF Analysis"
    assert not tabs.isTabVisible(index)

    window.set_workspace_mode("sequence")
    assert tabs.isTabVisible(index)
    sequence = window.sequence_simulation_widget
    view = sequence.psf_analysis_view
    assert not window.psf_analysis_page.isAncestorOf(view)
    sequence.psf_analysis_link.linkActivated.emit("psf-analysis")
    assert tabs.currentIndex() == index
    assert window.psf_analysis_page.isAncestorOf(view)

    sequence.sequence_source.setCurrentIndex(sequence.FLASH_SOURCE)
    context = sequence._psf_analysis_context()
    assert context.program is sequence.program
    assert context.simulator is not sequence.simulator
    assert context.simulation_timestep_s == pytest.approx(
        sequence.simulation_timestep_us.value() * 1e-6
    )
    assert context.token == sequence._psf_analysis_token()
    sequence._mark_sequence_generation_pending()
    assert sequence._psf_analysis_token() != context.token

    window.set_workspace_mode("free")
    assert not tabs.isTabVisible(index)
    window.close()
    window.deleteLater()
    app.processEvents()
