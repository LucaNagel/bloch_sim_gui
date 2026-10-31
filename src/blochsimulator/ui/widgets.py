from PyQt5.QtCore import QEvent, QObject, QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QWheelEvent
from PyQt5.QtWidgets import (
    QAbstractButton,
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg


IMAGE_HISTOGRAM_WIDTH = 48
IMAGE_CANVAS_BACKGROUND = (18, 18, 20)
IMAGE_FOV_BORDER = (62, 62, 68)


class PolarizationPercentSpinBox(QDoubleSpinBox):
    """Precise percent values with compact, locale-aware fixed-point text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(12)
        self.setRange(0.0, 100.0)
        self.setSuffix(" %")
        self.setSingleStep(1.0)

    def textFromValue(self, value):
        text = self.locale().toString(value, "f", self.decimals())
        return text.rstrip("0").rstrip(self.locale().decimalPoint())


class ExplicitAcceptDialog(QDialog):
    """Keep Enter available to editors; accept only through an explicit action."""

    def showEvent(self, event):
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        super().showEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)


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


FOCUS_ACCENT = QColor(47, 128, 237)
DISABLED_TEXT_ROLES = (
    QPalette.WindowText,
    QPalette.Text,
    QPalette.ButtonText,
    QPalette.HighlightedText,
)


def blend_colors(base, accent, amount):
    """Mix ``amount`` (0..1) of ``accent`` into ``base``."""
    amount = min(max(float(amount), 0.0), 1.0)
    return QColor(
        round(base.red() + (accent.red() - base.red()) * amount),
        round(base.green() + (accent.green() - base.green()) * amount),
        round(base.blue() + (accent.blue() - base.blue()) * amount),
    )


def _over(color, background):
    """``color`` composited over an opaque ``background``."""
    return blend_colors(background, QColor(color.rgb()), color.alphaF())


def _lightness(color):
    return 0.2126 * color.redF() + 0.7152 * color.greenF() + 0.0722 * color.blueF()


def palette_with_grey_disabled_text(palette, min_contrast=0.2):
    """Return ``palette`` with disabled text visibly greyed.

    The macOS platform palette draws disabled WindowText/Text as black at 76 %
    instead of 85 % opacity, so under Fusion disabled labels, check boxes and
    spin boxes look enabled.
    """
    palette = QPalette(palette)
    window = palette.color(QPalette.Active, QPalette.Window)
    for role in DISABLED_TEXT_ROLES:
        active = _over(palette.color(QPalette.Active, role), window)
        disabled = _over(palette.color(QPalette.Disabled, role), window)
        if abs(_lightness(active) - _lightness(disabled)) < min_contrast:
            palette.setColor(
                QPalette.Disabled, role, blend_colors(active, window, 0.55)
            )
    return palette


class InputFeedbackFilter(QObject):
    """Application-wide feedback for editable fields and disabled controls.

    * A focused text or number field gets a tinted background and a strong
      focus frame, so it is obvious that typing goes into that field.
    * Disabled text is greyed even where the platform palette does not.
    * A label whose buddy is disabled is greyed along with it.
    * Typing a decimal point in front of an existing fraction replaces that
      fraction, so selecting "32" in "32.00 mm" and typing "64.3" works.
    """

    FIELD_TINT = 0.13
    FRAME_TINT = 0.75
    LABELLED_TYPES = (
        QAbstractSpinBox,
        QComboBox,
        QLineEdit,
        QAbstractButton,
        QAbstractSlider,
    )

    def __init__(self, app):
        super().__init__(app)
        self._app = app
        self._fixing_palette = False
        self.refresh_palette()

    def refresh_palette(self):
        if self._fixing_palette:
            return
        fixed = palette_with_grey_disabled_text(self._app.palette())
        if fixed == self._app.palette():
            return
        self._fixing_palette = True
        try:
            self._app.setPalette(fixed)
        finally:
            self._fixing_palette = False

    @staticmethod
    def _editable_field(widget):
        if isinstance(widget, QAbstractSpinBox):
            return None if widget.isReadOnly() else widget
        if isinstance(widget, QLineEdit) and not widget.isReadOnly():
            # A spin box line edit forwards focus feedback to its spin box.
            parent = widget.parentWidget()
            if isinstance(parent, (QAbstractSpinBox, QComboBox)):
                return None
            return widget
        return None

    def _highlight(self, field, focused):
        saved = field.property("_input_feedback_palette")
        if focused:
            if saved is None:
                field.setProperty("_input_feedback_palette", QPalette(field.palette()))
                palette = QPalette(field.palette())
            else:
                palette = QPalette(saved)
            base = palette.color(QPalette.Active, QPalette.Base)
            for group in (QPalette.Active, QPalette.Inactive):
                palette.setColor(
                    group,
                    QPalette.Base,
                    blend_colors(base, FOCUS_ACCENT, self.FIELD_TINT),
                )
                palette.setColor(
                    group,
                    QPalette.Highlight,
                    blend_colors(base, FOCUS_ACCENT, self.FRAME_TINT),
                )
                palette.setColor(group, QPalette.HighlightedText, QColor(255, 255, 255))
            field.setPalette(palette)
        elif saved is not None:
            field.setProperty("_input_feedback_palette", None)
            field.setPalette(saved)

    @staticmethod
    def _sync_buddy_labels(widget):
        parent = widget.parentWidget()
        if parent is None:
            return
        for label in parent.findChildren(QLabel, "", Qt.FindDirectChildrenOnly):
            if label.buddy() is widget:
                label.setEnabled(widget.isEnabled())

    @staticmethod
    def _type_decimal_point(spin_box, event):
        """Move the decimal point to the cursor, dropping the old fraction.

        Returns True when the key press was handled here.
        """
        if event.text() not in (".", ","):
            return False
        if spin_box.isReadOnly() or spin_box.decimals() <= 0:
            return False
        edit = spin_box.lineEdit()
        point = spin_box.locale().decimalPoint()
        if edit.hasSelectedText():
            edit.del_()
        text = edit.text()
        cursor = edit.cursorPosition()
        existing = text.find(point, len(spin_box.prefix()))
        if 0 <= existing < cursor:
            # The fraction is already in front of the cursor.
            return False
        if existing >= 0:
            end = existing + 1
            while end < len(text) and text[end].isdigit():
                end += 1
            edit.setSelection(existing, end - existing)
            edit.del_()
            edit.setCursorPosition(cursor)
        edit.insert(point)
        return True

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.KeyPress and isinstance(watched, QDoubleSpinBox):
            return self._type_decimal_point(watched, event)
        if kind in (QEvent.FocusIn, QEvent.FocusOut):
            field = self._editable_field(watched)
            if field is not None:
                self._highlight(field, kind == QEvent.FocusIn)
        elif kind == QEvent.EnabledChange:
            if isinstance(watched, self.LABELLED_TYPES):
                self._sync_buddy_labels(watched)
        elif kind == QEvent.ApplicationPaletteChange and watched is self._app:
            self.refresh_palette()
        return False


def install_input_feedback(app=None):
    """Install focus tint, grey disabled text and buddy-label greying once."""
    app = app or QApplication.instance()
    if app is None:
        raise RuntimeError("A QApplication is required")

    attribute = "_bloch_input_feedback"
    feedback = getattr(app, attribute, None)
    if feedback is None:
        feedback = InputFeedbackFilter(app)
        app.installEventFilter(feedback)
        setattr(app, attribute, feedback)
    else:
        feedback.refresh_palette()
    return feedback


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


class GroupedComboBox(QComboBox):
    """A combo box whose popup lists its items under group headings.

    The items keep their indices, which projects store, while the popup
    orders them by group and can end a group with extra actions such as
    "Load file…".
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._groups = []
        self.popup_menu = None

    def set_groups(self, groups):
        """Set ``[(title, [index or (text, callback), ...]), ...]``."""
        self._groups = [(title, list(entries)) for title, entries in groups]

    def showPopup(self):
        if not self._groups:
            super().showPopup()
            return
        menu = QMenu(self)
        menu.setObjectName(f"{self.objectName() or 'grouped_combo'}_popup")
        menu.setMinimumWidth(self.width())
        for title, entries in self._groups:
            menu.addSection(title)
            for entry in entries:
                if isinstance(entry, int):
                    action = menu.addAction(self.itemText(entry))
                    if entry == self.currentIndex():
                        font = action.font()
                        font.setBold(True)
                        action.setFont(font)
                        menu.setActiveAction(action)
                    action.setToolTip(self.itemData(entry, Qt.ToolTipRole) or "")
                    action.triggered.connect(
                        lambda _checked=False, index=entry: self.setCurrentIndex(index)
                    )
                else:
                    text, callback = entry
                    menu.addAction(text).triggered.connect(
                        lambda _checked=False, run=callback: run()
                    )
        menu.setToolTipsVisible(True)
        self.popup_menu = menu
        menu.aboutToHide.connect(menu.deleteLater)
        menu.popup(self.mapToGlobal(self.rect().bottomLeft()))


class _TabBarWheelScroller(QObject):
    """Slide a tab bar's overflowing tabs with the wheel or a two-finger swipe.

    Qt's default would switch the current tab; here the gesture only moves the
    visible tabs, as the scroll arrows do.
    """

    STEP = 60  # accumulated angle delta (1/8 degree) per scrolled tab

    def __init__(self, tab_bar):
        super().__init__(tab_bar)
        self._tab_bar = tab_bar
        self._accumulated = 0
        tab_bar.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self._tab_bar and event.type() == QEvent.Wheel:
            angle = event.angleDelta()
            delta = angle.y() if abs(angle.y()) >= abs(angle.x()) else angle.x()
            self._accumulated += delta
            steps = int(self._accumulated / self.STEP)
            if steps:
                self._accumulated -= steps * self.STEP
                self.scroll(-steps)
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def scroll(self, steps):
        """Scroll ``steps`` tabs to the right (negative: to the left)."""
        buttons = sorted(
            (
                button
                for button in self._tab_bar.findChildren(QToolButton)
                if button.isVisibleTo(self._tab_bar)
            ),
            key=lambda button: button.x(),
        )
        if len(buttons) < 2:
            return
        button = buttons[-1] if steps > 0 else buttons[0]
        for _ in range(abs(steps)):
            if not button.isEnabled():
                break
            button.click()


def install_tab_bar_wheel_scroll(tab_widget: QTabWidget) -> _TabBarWheelScroller:
    """Let the wheel/trackpad slide ``tab_widget``'s tabs without switching."""
    return _TabBarWheelScroller(tab_widget.tabBar())


class SegmentedButtons(QWidget):
    """A row of exclusive, flat buttons; one key is current at a time.

    The application style sheet draws the joined buttons from the
    ``segmented`` property of the row and the ``segment`` position of each
    button; ``size="large"`` gives the row more padding.
    """

    current_changed = pyqtSignal(str)

    def __init__(self, items, parent=None, *, size="normal"):
        """``items`` is a sequence of ``(key, text)`` pairs."""
        super().__init__(parent)
        self.setObjectName("segmented_buttons")
        self.setProperty("segmented", True)
        self.setProperty("size", str(size))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 8, 2)
        layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons = {}
        for position, (key, text) in enumerate(items):
            button = QToolButton()
            button.setObjectName(f"segment_{key}")
            button.setText(text)
            button.setCheckable(True)
            button.setAutoRaise(False)
            if position == 0:
                button.setProperty("segment", "first")
            elif position == len(items) - 1:
                button.setProperty("segment", "last")
            else:
                button.setProperty("segment", "middle")
            self._group.addButton(button)
            self.buttons[key] = button
            layout.addWidget(button)
            button.clicked.connect(
                lambda _checked=False, key=key: self.current_changed.emit(key)
            )

    def current(self):
        for key, button in self.buttons.items():
            if button.isChecked():
                return key
        return None

    def set_current(self, key):
        """Check ``key`` without emitting :attr:`current_changed`."""
        button = self.buttons.get(key)
        if button is not None:
            button.setChecked(True)


class CollapsibleSection(QWidget):
    """A section whose body folds away when its header is clicked.

    A titled QGroupBox body keeps its frame and hands its title to the
    header. With ``settings`` and ``key`` the expanded state is remembered.
    """

    toggled = pyqtSignal(bool)
    SETTINGS_GROUP = "view/expanded_sections"
    # Marks the folding body so visibility checks can look through it.
    BODY_PROPERTY = "collapsible_section_body"

    def __init__(
        self, body, title="", *, expanded=True, settings=None, key="", parent=None
    ):
        super().__init__(parent)
        if isinstance(body, QGroupBox) and body.title():
            title = title or body.title()
            body.setTitle("")
        self._settings = settings
        self._key = key
        if settings is not None and key:
            stored = settings.value(f"{self.SETTINGS_GROUP}/{key}", expanded)
            expanded = str(stored).lower() not in ("false", "0")
        self.body = body
        body.setProperty(self.BODY_PROPERTY, True)
        self.header = QToolButton()
        self.header.setObjectName("collapsible_section_header")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setAutoRaise(True)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.header.setCursor(Qt.PointingHandCursor)
        # The group-title style sheet resets nested widgets to normal weight.
        self.header.setStyleSheet(
            "QToolButton { font-weight: bold; border: none; padding: 2px 0px;"
            " background: transparent; }"
        )
        self.header.setToolTip(f"Show or hide {title}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.header)
        layout.addWidget(body)
        body.setVisible(expanded)
        self.header.toggled.connect(self.set_expanded)

    def is_expanded(self):
        return self.header.isChecked()

    def set_expanded(self, expanded):
        expanded = bool(expanded)
        if self.header.isChecked() != expanded:
            self.header.setChecked(expanded)
            return  # toggled re-enters with the new state
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.body.setVisible(expanded)
        if self._settings is not None and self._key:
            self._settings.setValue(f"{self.SETTINGS_GROUP}/{self._key}", expanded)
        self.toggled.emit(expanded)
