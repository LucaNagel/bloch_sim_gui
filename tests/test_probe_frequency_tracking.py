import sys

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from blochsimulator.sequence import SequenceProbeResult
from blochsimulator.ui.probe_viewers import SequenceProbeSpectrumViewer


def _probe_result():
    magnetization = np.zeros((3, 1, 3, 3), dtype=float)
    magnetization[:, 0, :, 0] = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ]
    )
    return SequenceProbeResult(
        time_s=np.array([0.0, 0.001, 0.002]),
        positions_m=np.zeros((1, 3)),
        frequency_offsets_hz=np.array([-100.0, 0.0, 100.0]),
        magnetization=magnetization,
    )


def test_frequency_tracking_supports_multiple_selection_modes():
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = SequenceProbeSpectrumViewer()
    viewer.set_result(_probe_result())

    viewer.selection_mode.setCurrentText("Single frequency")
    viewer.selection_center.setValue(0.0)
    viewer.add_frequency_selection()

    viewer.selection_mode.setCurrentText("Frequency range")
    viewer.selection_center.setValue(0.0)
    viewer.selection_width.setValue(200.0)
    viewer.add_frequency_selection()

    viewer.selection_mode.setCurrentText("Lorentzian")
    viewer.width_kind.setCurrentText("FWHM")
    viewer.selection_center.setValue(0.0)
    viewer.selection_width.setValue(200.0)
    viewer.add_frequency_selection()

    assert viewer.selection_list.count() == 3
    assert len(viewer._selection_markers) == 3
    curves = viewer.trace_plot.listDataItems()
    assert len(curves) == 3
    assert curves[0].yData == pytest.approx([2.0, 4.0, 6.0])
    assert curves[1].yData == pytest.approx([2.0, 4.0, 6.0])
    assert curves[2].yData == pytest.approx([2.0, 4.0, 6.0])

    viewer.selection_center.setValue(50.0)
    viewer.selection_width.setValue(100.0)
    viewer.update_frequency_selection()

    updated = viewer.frequency_selections[2]
    assert updated["center_hz"] == pytest.approx(50.0)
    assert updated["width"] == pytest.approx(100.0)
    assert "50 Hz" in viewer.selection_list.item(2).text()
    assert viewer.trace_plot.listDataItems()[2].yData[0] > 2.0

    viewer.close()
    viewer.deleteLater()
    app.processEvents()


def test_frequency_evolution_panel_is_large_and_scrollable():
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = SequenceProbeSpectrumViewer()
    viewer.set_result(_probe_result())
    viewer.resize(800, 500)
    viewer.show()
    app.processEvents()

    assert viewer.trace_plot.minimumHeight() >= 400
    assert viewer.scroll_area.verticalScrollBar().maximum() > 0

    viewer.close()
    viewer.deleteLater()
    app.processEvents()


def test_lorentzian_t2_is_converted_to_fwhm():
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = SequenceProbeSpectrumViewer()
    viewer.set_result(_probe_result())
    selection = {
        "mode": "Lorentzian",
        "center_hz": 0.0,
        "width": 100.0,
        "width_kind": "T2",
    }

    assert viewer._selection_fwhm_hz(selection) == pytest.approx(1.0 / (np.pi * 0.1))
    assert np.sum(viewer._frequency_weights(selection)) == pytest.approx(1.0)

    viewer.close()
    viewer.deleteLater()
    app.processEvents()
