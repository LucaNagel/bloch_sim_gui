from PyQt5.QtWidgets import QComboBox
from PyQt5.QtCore import Qt, pyqtSignal
import pyqtgraph as pg


IMAGE_HISTOGRAM_WIDTH = 48
IMAGE_CANVAS_BACKGROUND = (18, 18, 20)
IMAGE_FOV_BORDER = (62, 62, 68)


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
