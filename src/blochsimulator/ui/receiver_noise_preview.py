"""SNR budget beside the receiver controls in the Receiver / Noise tab."""

import math

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..sequence.receiver import receiver_noise_samples
from ..sequence.snr_budget import snr_budget
from ..sequence.signal_units import normalize_signal_unit

#: Illustrative decay of the reference FID plot; no budget value uses it.
REFERENCE_T2STAR_S = 0.02


def reference_fid(amplitude, noise_std, dwell_s, seed=0, t2star_s=REFERENCE_T2STAR_S):
    """Real channel of an on-resonance FID with per-quadrature noise ``noise_std``."""
    count = int(min(max(math.ceil(5 * t2star_s / dwell_s), 64), 8192))
    time_s = np.arange(count) * dwell_s
    ideal = amplitude * np.exp(-time_s / t2star_s)
    noisy = ideal + receiver_noise_samples((count,), seed, noise_std).real
    return time_s, ideal, noisy


class SNRBudgetPanel(QWidget):
    """Explain the receiver setting as an SNR chain for the current object.

    ``context`` returns the sequence workspace state (phantom, signal weighting,
    program, spatial grid, result, nucleus, field strength); every refresh
    reads it again, so the budget follows the object and the last run.
    """

    def __init__(self, controls, context=None, parent=None):
        super().__init__(parent)
        self.controls = controls
        self.context = context or dict
        self.sections = ()
        self.reference = None
        layout = QVBoxLayout(self)
        title = QLabel("SNR budget")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        self.description = QLabel(
            "From the receiver setting to the expected voxel SNR. Rows before "
            "“Last run” are ceilings for one scan: ideal 90° excitation without "
            "relaxation or flip-angle consumption. “Last run” uses the simulated "
            "noiseless signal, so it contains everything the simulation models."
        )
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Quantity", "Value", "Note"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.NoSelection)
        self.tree.setFocusPolicy(Qt.NoFocus)
        self.tree.setWordWrap(True)
        self.tree.setUniformRowHeights(False)
        header = self.tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # The table gets the space; the reference FID can be enlarged by hand.
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.tree)
        reference = QWidget()
        reference_layout = QVBoxLayout(reference)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(reference)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([3000, 1000])
        layout.addWidget(splitter, 1)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(140)
        self.plot.setLabel("bottom", "Time", units="ms")
        self.plot.setLabel("left", "Real channel")
        self.ideal_curve = self.plot.plot(pen=pg.mkPen("#a0a0a0", width=2))
        self.noisy_curve = self.plot.plot(pen=pg.mkPen("#00b7ff", width=1))
        dashed = pg.mkPen("#ffb000", width=1, style=Qt.DashLine)
        self.sigma_lines = (
            pg.InfiniteLine(angle=0, pen=dashed),
            pg.InfiniteLine(angle=0, pen=dashed),
        )
        for line in self.sigma_lines:
            self.plot.addItem(line)
        reference_layout.addWidget(self.plot, 1)
        self.plot_info = QLabel()
        self.plot_info.setWordWrap(True)
        reference_layout.addWidget(self.plot_info)
        controls.preview_changed.connect(self.refresh)
        self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def refresh(self):
        context = dict(self.context() or {})
        try:
            noise, preview = self.controls.budget_settings(
                nucleus=context.get("nucleus"),
                field_strength_t=context.get("field_strength_t"),
            )
        except (TypeError, ValueError) as exc:
            self.sections = ()
            self._show_message(f"{exc}.")
        else:
            self.sections = snr_budget(
                noise,
                calibration_preview=preview,
                phantom=context.get("phantom"),
                signal_weighting=context.get("signal_weighting", "voxel"),
                program=context.get("program"),
                grid=context.get("grid"),
                result=context.get("result"),
                nucleus=context.get("nucleus"),
                field_strength_t=context.get("field_strength_t"),
            )
            self._show_sections()
        self._refresh_reference()

    def _show_message(self, text):
        self.tree.clear()
        item = QTreeWidgetItem([text])
        self.tree.addTopLevelItem(item)
        item.setFirstColumnSpanned(True)

    def _show_sections(self):
        self.tree.clear()
        for section in self.sections:
            top = QTreeWidgetItem([section.title, section.status, ""])
            font = top.font(0)
            font.setBold(True)
            top.setFont(0, font)
            if section.status:
                status_font = top.font(1)
                status_font.setItalic(True)
                top.setFont(1, status_font)
            self.tree.addTopLevelItem(top)
            for row in section.rows:
                item = QTreeWidgetItem([row.label, row.value, row.note])
                item.setToolTip(2, row.note)
                item.setTextAlignment(1, int(Qt.AlignRight | Qt.AlignVCenter))
                top.addChild(item)
                if not row.value and not row.note:
                    item.setFirstColumnSpanned(True)
            top.setExpanded(True)

    def _reference_parameters(self):
        """Amplitude, noise std and dwell of the reference FID, or a message."""
        controls = self.controls
        mode = controls.mode.currentData()
        if mode == "off":
            return None, "Receiver noise is off."
        dwell = controls._reference_dwell_s()
        if mode == "relative":
            try:
                snr = controls._relative_snr()
            except ValueError as exc:
                return None, f"{exc}."
            return (1.0, 1.0 / snr, dwell), (
                f"Reference FID normalized to 1 with SNR {snr:.6g} at "
                f"{1e-3 / dwell:g} kHz"
            )
        try:
            amplitude = float(controls.amplitude.text())
            std = float(controls.reference_std.text())
        except ValueError:
            amplitude = std = math.nan
        if not (math.isfinite(amplitude) and amplitude > 0 and std > 0):
            return None, "Enter a positive measured FID amplitude and noise std."
        return (amplitude, std, dwell), (
            f"Measured reference FID: amplitude {amplitude:g}, noise std {std:g} "
            f"(SNR {amplitude / std:.6g}) at {1e-3 / dwell:g} kHz"
        )

    def _refresh_reference(self):
        snr_units = (
            normalize_signal_unit((self.context() or {}).get("signal_unit")) == "snr"
        )
        self.plot.setLabel("left", "Signal / σ" if snr_units else "Real channel")
        parameters, text = self._reference_parameters()
        self.reference = parameters
        if parameters is None:
            self.ideal_curve.setData([], [])
            self.noisy_curve.setData([], [])
            for line in self.sigma_lines:
                line.hide()
            self.plot_info.setText(text)
            return
        amplitude, std, dwell = parameters
        time_s, ideal, noisy = reference_fid(
            amplitude, std, dwell, seed=self.controls.seed.value()
        )
        if snr_units:
            ideal, noisy = ideal / std, noisy / std
            std = 1.0
        self.ideal_curve.setData(time_s * 1e3, ideal)
        self.noisy_curve.setData(time_s * 1e3, noisy)
        for line, value in zip(self.sigma_lines, (std, -std)):
            line.setValue(value)
            line.show()
        self.plot_info.setText(
            f"{text}; dashed: ±σ per quadrature. Decay T2* "
            f"{REFERENCE_T2STAR_S * 1e3:g} ms is illustrative only."
        )


class ReceiverNoiseWorkspace(QWidget):
    """Keep controls outside the sequence sidebar, beside the SNR budget."""

    def __init__(self, controls, parent=None, context=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        control_host = QWidget()
        control_layout = QVBoxLayout(control_host)
        control_layout.setContentsMargins(0, 0, 0, 0)
        controls.setParent(control_host)
        for form in (controls.layout(), controls.calibration_box.layout()):
            form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        controls.show()
        control_layout.addWidget(controls)
        control_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setMinimumWidth(280)
        scroll.setWidget(control_host)
        self.preview = SNRBudgetPanel(controls, context)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(scroll)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 850])
        layout.addWidget(splitter)
