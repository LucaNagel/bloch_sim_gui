"""Capture and restore the plain values of a widget's direct Qt controls.

Project files and sequence presets both store controls by attribute name, so
a state survives reordered layouts and newly added controls.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSlider,
    QSpinBox,
)


def capture_widget_state(owner):
    """Capture direct, user-editable Qt attributes without serializing Qt itself."""
    state = {}
    for name, widget in vars(owner).items():
        if isinstance(widget, QComboBox):
            state[name] = {
                "type": "combo",
                "index": widget.currentIndex(),
                "text": widget.currentText(),
            }
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
            state[name] = {"type": "value", "value": widget.value()}
        elif isinstance(widget, QCheckBox):
            state[name] = {"type": "checked", "value": widget.isChecked()}
        elif isinstance(widget, QLineEdit):
            state[name] = {"type": "text", "value": widget.text()}
    return state


def restore_widget_state(owner, state, *, block_signals=False):
    """Apply a state from :func:`capture_widget_state`, skipping unknown names."""
    for name, saved in state.items():
        widget = getattr(owner, name, None)
        if widget is None:
            continue
        previous_blocked = widget.blockSignals(True) if block_signals else None
        try:
            kind = saved.get("type")
            if kind == "combo" and isinstance(widget, QComboBox):
                text = str(saved.get("text", ""))
                index = widget.findText(text)
                if index < 0 and text:
                    # Older project files may use equivalent numeric formatting,
                    # e.g. "7T" while the current item is "7.0T".
                    try:
                        saved_number = float(text.removesuffix("T").strip())
                    except ValueError:
                        saved_number = None
                    if saved_number is not None:
                        for candidate in range(widget.count()):
                            candidate_text = widget.itemText(candidate)
                            try:
                                if np.isclose(
                                    float(candidate_text.removesuffix("T").strip()),
                                    saved_number,
                                ):
                                    index = candidate
                                    break
                            except ValueError:
                                continue
                if index < 0:
                    saved_index = int(saved.get("index", -1))
                    index = (
                        saved_index
                        if 0 <= saved_index < widget.count()
                        else widget.currentIndex()
                    )
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif kind == "value" and isinstance(
                widget, (QSpinBox, QDoubleSpinBox, QSlider)
            ):
                value = saved["value"]
                if isinstance(widget, (QSpinBox, QSlider)):
                    value = int(round(float(value)))
                widget.setValue(value)
            elif kind == "checked" and isinstance(widget, QCheckBox):
                widget.setChecked(bool(saved["value"]))
            elif kind == "text" and isinstance(widget, QLineEdit):
                widget.setText(str(saved.get("value", "")))
        finally:
            if block_signals:
                widget.blockSignals(previous_blocked)
