"""Typing 0 into a spin box picks the smallest value the form accepts.

Qt rejects text below a spin box's minimum and restores the previous value.
For fields such as TR or TE that is unhelpful: the shortest feasible value
depends on every other parameter. :class:`ZeroToMinimum` intercepts a
committed "0" (Return, Enter or leaving the field) and sets the smallest value
for which ``feasible(spin)`` holds, leaving every other field unchanged.
"""

import math
import re

from PyQt5 import sip
from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox

_MINIMUM_PATTERN = re.compile(
    r"minimum (?:is|of)\s+(?:the\s+)?([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    r"\s*(ms|s)\b"
)


def minimum_ms_from_message(message):
    """Millisecond minimum named in a builder error, or None.

    Builders phrase limits as "... minimum is 6.730 ms" or
    "... minimum is 0.00783 s".
    """
    match = _MINIMUM_PATTERN.search(str(message))
    if match is None:
        return None
    value = float(match.group(1))
    return value * 1000.0 if match.group(2) == "s" else value


class ZeroToMinimum(QObject):
    """Event filter turning a typed 0 into the smallest feasible value.

    Parameters
    ----------
    feasible : callable
        ``feasible(spin)`` says whether the form works with the spin box's
        current value; exceptions count as infeasible.
    hint : callable, optional
        ``hint(spin)`` may return a candidate minimum, checked before the
        bisection runs.
    """

    def __init__(self, feasible, hint=None, parent=None):
        super().__init__(parent)
        self._feasible = feasible
        self._hint = hint
        self._spins = {}

    def install(self, spins):
        """Watch ``spins`` whose minimum is above 0, so 0 cannot mean itself."""
        for spin in spins:
            if spin in self._spins.values() or spin.minimum() <= 0:
                continue
            for target in (spin, spin.lineEdit()):
                target.installEventFilter(self)
                self._spins[target] = spin

    def eventFilter(self, watched, event):
        commit = event.type() == QEvent.FocusOut or (
            event.type() == QEvent.KeyPress
            and event.key() in (Qt.Key_Return, Qt.Key_Enter)
        )
        spin = self._spins.get(watched) if commit else None
        if spin is None or sip.isdeleted(spin):
            return False
        if spin.isEnabled() and not spin.isReadOnly() and self._typed_zero(spin):
            self.apply(spin)
        return False

    @staticmethod
    def _typed_zero(spin):
        text = spin.cleanText().strip()
        if not text:
            return False
        value, ok = spin.locale().toDouble(text)
        return ok and value == 0.0 and spin.minimum() > 0

    def apply(self, spin):
        """Set ``spin`` to its smallest feasible value; return it or None."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            value = self.smallest_feasible(spin)
        finally:
            QApplication.restoreOverrideCursor()
        if value is None:
            return None
        spin.setValue(value)
        return spin.value()

    def smallest_feasible(self, spin):
        """Smallest value on the spin box's grid that ``feasible`` accepts.

        Assumes larger values stay feasible up to the current one. Returns
        None when not even the current value works.
        """
        step = 10.0 ** -spin.decimals() if isinstance(spin, QDoubleSpinBox) else 1
        original = spin.value()

        def value_of(index):
            return round(index * step, 12) if step != 1 else int(index)

        def accepts(index):
            previous = spin.blockSignals(True)
            spin.setValue(value_of(index))
            spin.blockSignals(previous)
            try:
                return bool(self._feasible(spin))
            except Exception:
                return False

        low = math.ceil(spin.minimum() / step - 1e-6)
        high = math.floor(original / step + 1e-6)
        try:
            if accepts(low):
                return value_of(low)
            hint = None if self._hint is None else self._hint(spin)
            if hint is not None and spin.minimum() < hint <= spin.maximum():
                candidate = math.ceil(hint / step - 1e-6)
                if accepts(candidate):
                    if not accepts(candidate - 1):
                        return value_of(candidate)
                    high = min(high, candidate)
                else:
                    low = max(low, candidate)
            if high <= low or not accepts(high):
                return None
            # ``low`` is infeasible and ``high`` feasible throughout.
            while high - low > 1:
                middle = (low + high) // 2
                if accepts(middle):
                    high = middle
                else:
                    low = middle
            return value_of(high)
        finally:
            previous = spin.blockSignals(True)
            spin.setValue(original)
            spin.blockSignals(previous)


def spin_boxes(widget):
    """Every numeric spin box below ``widget``."""
    return [*widget.findChildren(QSpinBox), *widget.findChildren(QDoubleSpinBox)]
