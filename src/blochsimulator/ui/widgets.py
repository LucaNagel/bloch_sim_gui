from PyQt5.QtCore import QEvent, QObject, QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QWidget,
)
import pyqtgraph as pg


IMAGE_HISTOGRAM_WIDTH = 48
IMAGE_CANVAS_BACKGROUND = (18, 18, 20)
IMAGE_FOV_BORDER = (62, 62, 68)


class SpinBoxWheelGuard(QObject):
    """Reserve unmodified wheel events for scrolling surrounding forms."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._forwarding_to_spin_box = False

    @staticmethod
    def _spin_box_ancestor(watched):
        widget = watched if isinstance(watched, QWidget) else None
        while widget is not None:
            if isinstance(widget, QAbstractSpinBox):
                return widget
            widget = widget.parentWidget()
        return None

    @staticmethod
    def _scroll_area_ancestor(widget):
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QAbstractScrollArea):
                return parent
            parent = parent.parentWidget()
        return None

    @staticmethod
    def _wheel_event_for_receiver(event, receiver, modifiers):
        local_position = QPointF(receiver.mapFromGlobal(event.globalPos()))
        return QWheelEvent(
            local_position,
            event.globalPosF(),
            event.pixelDelta(),
            event.angleDelta(),
            event.buttons(),
            modifiers,
            event.phase(),
            event.inverted(),
            event.source(),
        )

    def eventFilter(self, watched, event):
        if self._forwarding_to_spin_box or event.type() != QEvent.Wheel:
            return False

        spin_box = self._spin_box_ancestor(watched)
        if spin_box is None:
            return False

        modifiers = event.modifiers()
        if modifiers & Qt.ControlModifier:
            # Qt normally interprets Ctrl as its accelerated (10x) step
            # modifier. Strip only Ctrl so the explicit opt-in still changes
            # the value by the spin box's regular single step.
            forwarded = self._wheel_event_for_receiver(
                event,
                spin_box,
                modifiers & ~Qt.ControlModifier,
            )
            self._forwarding_to_spin_box = True
            try:
                QApplication.sendEvent(spin_box, forwarded)
            finally:
                self._forwarding_to_spin_box = False
            return True

        scroll_area = self._scroll_area_ancestor(spin_box)
        if scroll_area is not None:
            forwarded = self._wheel_event_for_receiver(
                event,
                scroll_area.viewport(),
                modifiers,
            )
            QApplication.sendEvent(scroll_area.viewport(), forwarded)

        # Outside a scroll area, an unmodified wheel event intentionally does
        # nothing rather than changing a value accidentally.
        return True


def install_spin_box_wheel_guard(app=None):
    """Install the application-wide spin-box wheel policy once."""
    app = app or QApplication.instance()
    if app is None:
        raise RuntimeError("A QApplication is required")

    attribute = "_bloch_spin_box_wheel_guard"
    guard = getattr(app, attribute, None)
    if guard is None:
        guard = SpinBoxWheelGuard(app)
        app.installEventFilter(guard)
        setattr(app, attribute, guard)
    return guard


def style_image_item(image_item):
    """Draw a subtle frame around the rectangular image/FOV extent."""
    image_item.setBorder(pg.mkPen(IMAGE_FOV_BORDER, width=1))
    return image_item


def style_image_view(view):
    """Distinguish a black image/FOV from its slightly lighter canvas."""
    view.ui.graphicsView.setBackground(IMAGE_CANVAS_BACKGROUND)
    style_image_item(view.getImageItem())
    return view


def compact_image_histogram(view, width: int = IMAGE_HISTOGRAM_WIDTH):
    """Keep an ImageView LUT compact and make its image/FOV extent visible."""
    style_image_view(view)
    histogram = view.ui.histogram
    histogram.setFixedWidth(int(width))
    item = histogram.item
    item.axis.setStyle(showValues=False, tickLength=3)
    item.axis.setWidth(6)
    item.gradient.setMaximumWidth(30)
    item.vb.setMaximumWidth(12)
    return histogram


class CheckableComboBox(QComboBox):
    """A combo box with checkable items for multi-selection."""

    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.closeOnLineEditClick = False
        self.lineEdit().installEventFilter(self)
        self.model().dataChanged.connect(self._on_model_data_changed)

    def _on_model_data_changed(self, top_left, bottom_right, roles):
        if Qt.CheckStateRole in roles:
            self.update_display_text()
            self.selection_changed.emit()

    def eventFilter(self, obj, event):
        if obj == self.lineEdit() and event.type() == event.MouseButtonRelease:
            if self.closeOnLineEditClick:
                self.hidePopup()
            else:
                self.showPopup()
            return True
        return super().eventFilter(obj, event)

    def showPopup(self):
        super().showPopup()
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        self.closeOnLineEditClick = False

    def add_items(self, items):
        for text in items:
            self.addItem(text)
            item = self.model().item(self.count() - 1)
            item.setCheckState(Qt.Unchecked)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)

    def get_selected_items(self):
        selected = []
        for i in range(self.count()):
            item = self.model().item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected

    def set_selected_items(self, items):
        self.model().blockSignals(True)
        for i in range(self.count()):
            item = self.model().item(i)
            if item.text() in items:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
        self.model().blockSignals(False)
        self.update_display_text()

    def update_display_text(self):
        selected = self.get_selected_items()
        text = ", ".join(selected) if selected else "None"
        self.lineEdit().setText(text)

    def currentText(self):
        return self.lineEdit().text()
