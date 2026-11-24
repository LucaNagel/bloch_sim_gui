"""Inputs for fundamental values that ask before a user changes them.

B0, the nucleus and the sequence reference frequency set the ppm-to-Hz
conversion, the Larmor frequency and the RF/gradient units of everything
else, so a slip of the mouse wheel or a stray keystroke must not change them.
The widgets below ask for confirmation whenever the *user* changes the value
(arrow buttons/keys, typed text, a popup choice). Programmatic changes, e.g.
loading a project or a phantom, are not asked about. The mouse wheel is
passed on to the surrounding scroll area instead of changing the value.
"""

from __future__ import annotations

from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QMouseEvent, QValidator
from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QMessageBox

B0_REASON = (
    "B0 sets the Larmor frequency and converts every ppm value (peaks, "
    "B0 offsets, the sequence reference) to Hz."
)
NUCLEUS_REASON = (
    "The nucleus sets the gyromagnetic ratio, i.e. the ppm-to-Hz conversion "
    "and the physical RF and gradient units."
)
REFERENCE_REASON = (
    "The reference is the absolute frequency taken as 0 Hz; all RF and "
    "receiver carrier offsets are relative to it."
)


def confirm_fundamental_change(parent, label: str, old: str, new: str, reason: str):
    """Ask whether ``label`` may change from ``old`` to ``new``.

    Tests replace this function to answer without a dialog.
    """
    answer = QMessageBox.question(
        parent,
        f"Change {label}?",
        f"{reason}\n\nChange {label} from {old} to {new}?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    return answer == QMessageBox.Yes


def _ask(widget, old: str, new: str) -> bool:
    if widget._asking:
        return False
    widget._asking = True
    try:
        # Looked up at call time so tests can replace the module function.
        return bool(
            globals()["confirm_fundamental_change"](
                widget, widget.confirm_label, old, new, widget.confirm_reason
            )
        )
    finally:
        widget._asking = False


class ConfirmedDoubleSpinBox(QDoubleSpinBox):
    """Spin box that asks before a user change; see the module docstring."""

    def __init__(self, label: str, reason: str, parent=None):
        super().__init__(parent)
        self.confirm_label = label
        self.confirm_reason = reason
        self._asking = False
        # Typed text is committed (and asked about) on Enter or focus loss.
        self.setKeyboardTracking(False)
        self.setFocusPolicy(Qt.StrongFocus)

    def _display(self, value: float) -> str:
        return f"{self.prefix()}{self.textFromValue(value)}{self.suffix()}".strip()

    def _differs(self, value: float) -> bool:
        return abs(value - self.value()) > 0.5 * 10.0 ** (-self.decimals())

    _STEP_KEYS = (Qt.Key_Up, Qt.Key_Down, Qt.Key_PageUp, Qt.Key_PageDown)

    def _stop_auto_repeat(self):
        """End the arrow button's press-and-hold repeat before a dialog.

        The modal question takes the mouse release, so Qt would keep
        stepping (and asking) after the dialog closes.
        """
        release = QMouseEvent(
            QEvent.MouseButtonRelease,
            QPointF(-1.0, -1.0),
            Qt.LeftButton,
            Qt.NoButton,
            Qt.NoModifier,
        )
        super().mouseReleaseEvent(release)

    def stepBy(self, steps):
        target = min(
            max(self.value() + steps * self.singleStep(), self.minimum()),
            self.maximum(),
        )
        if not self._differs(target):
            return
        self._stop_auto_repeat()
        if _ask(self, self._display(self.value()), self._display(target)):
            super().stepBy(steps)

    def wheelEvent(self, event):
        event.ignore()

    def _check_typed_value(self):
        if self._asking:
            return
        text = self.text()
        if self.validate(text, len(text))[0] != QValidator.Acceptable:
            return
        target = min(
            max(self.valueFromText(self.cleanText()), self.minimum()),
            self.maximum(),
        )
        if self._differs(target) and not _ask(
            self, self._display(self.value()), self._display(target)
        ):
            self.lineEdit().setText(
                f"{self.prefix()}{self.textFromValue(self.value())}{self.suffix()}"
            )

    def keyPressEvent(self, event):
        if event.key() in self._STEP_KEYS and event.isAutoRepeat():
            # A held arrow key asks once, not once per repeat.
            event.accept()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._check_typed_value()
        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        self._check_typed_value()
        super().focusOutEvent(event)


class ConfirmedComboBox(QComboBox):
    """Combo box that asks before a user choice; see the module docstring.

    Choices are only made in the popup; keys that would step through the
    items open the popup instead.
    """

    _STEP_KEYS = (
        Qt.Key_Up,
        Qt.Key_Down,
        Qt.Key_PageUp,
        Qt.Key_PageDown,
        Qt.Key_Home,
        Qt.Key_End,
    )

    def __init__(self, label: str, reason: str, parent=None):
        super().__init__(parent)
        self.confirm_label = label
        self.confirm_reason = reason
        self._asking = False
        self.setFocusPolicy(Qt.StrongFocus)
        # Installed after the popup container's own filters, so it runs first.
        self.view().installEventFilter(self)
        self.view().viewport().installEventFilter(self)

    def wheelEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        if event.key() in self._STEP_KEYS:
            self.showPopup()
            return
        if event.text() and event.text().isprintable() and event.key() != Qt.Key_Space:
            # Keyboard search would change the item without asking.
            event.ignore()
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event):
        view = self.view()
        if watched is view.viewport() and event.type() == QEvent.MouseButtonRelease:
            return self._choose(view.indexAt(event.pos()))
        if (
            watched is view
            and event.type() == QEvent.KeyPress
            and event.key() in (Qt.Key_Return, Qt.Key_Enter)
        ):
            return self._choose(view.currentIndex())
        return super().eventFilter(watched, event)

    def choose(self, row: int) -> bool:
        """Select ``row`` as the user would; return whether it was accepted."""
        if row == self.currentIndex() or not 0 <= row < self.count():
            return False
        if not _ask(self, self.currentText(), self.itemText(row)):
            return False
        self.setCurrentIndex(row)
        self.activated[int].emit(row)
        return True

    def _choose(self, index) -> bool:
        if not index.isValid() or index.row() == self.currentIndex():
            return False
        if not index.flags() & Qt.ItemIsEnabled:
            return False
        self.hidePopup()
        self.choose(index.row())
        return True
