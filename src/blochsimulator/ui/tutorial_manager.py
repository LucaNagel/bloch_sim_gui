import json
import os
from PyQt5.QtWidgets import (
    QPushButton,
    QTabBar,
    QComboBox,
    QAction,
    QApplication,
    QWidget,
    QTabWidget,
    QCheckBox,
    QRadioButton,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QToolButton,
    QMenu,
    QMenuBar,
    QToolBar,
    QRubberBand,
    QAbstractItemView,
)
from PyQt5.QtCore import QObject, QEvent, pyqtSignal, Qt, QRect, QPoint


class HighlightOverlay(QRubberBand):
    """A translucent highlight overlay for specific UI elements."""

    def __init__(self, parent=None):
        super().__init__(QRubberBand.Rectangle, parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            """
            QRubberBand {
                border: 2px solid #149071;
                background-color: rgba(152, 241, 219, 80);
            }
        """
        )


class TutorialManager(QObject):
    """
    Handles recording and playback of GUI tutorials by highlighting widgets.
    """

    step_reached = pyqtSignal(int, int)  # current, total
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal(int)
    playback_started = pyqtSignal()
    playback_finished = pyqtSignal()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.is_recording = False
        self.is_playing = False
        self.steps = []
        self.current_step_idx = -1

        # Styles for highlighting
        self.highlight_style = """
            border: 2px solid #149071;
            background-color: #98F1DB;
        """
        self.original_styles = {}  # Store old styles to revert them
        self.overlays = []  # List of HighlightOverlay instances

    def start_recording(self):
        self.steps = []
        self.is_recording = True
        self.is_playing = False
        # Install event filter on the application to catch all clicks
        QApplication.instance().installEventFilter(self)
        # Connect to all combo boxes to catch selections specifically
        self._connect_combos(self.main_window)
        self.recording_started.emit()

    def _connect_combos(self, parent):
        for combo in parent.findChildren(QComboBox):
            try:
                combo.activated.connect(self._on_combo_activated)
            except Exception:
                pass

    def _on_combo_activated(self, index):
        if not self.is_recording:
            return
        combo = self.sender()
        name = combo.objectName()
        if name:
            text = combo.itemText(index)
            # Avoid duplicates if click also recorded
            if (
                self.steps
                and self.steps[-1].get("name") == name
                and self.steps[-1].get("type") == "click"
            ):
                self.steps[-1] = {
                    "name": name,
                    "type": "select",
                    "index": index,
                    "text": text,
                }
            else:
                self.steps.append(
                    {"name": name, "type": "select", "index": index, "text": text}
                )
            print(f"[Tutorial] Recorded selection: {name} -> {text}")

    def stop_recording(self):
        self.is_recording = False
        QApplication.instance().removeEventFilter(self)
        # Disconnect combos
        self._disconnect_combos(self.main_window)
        self.recording_stopped.emit(len(self.steps))
        return self.steps

    def _disconnect_combos(self, parent):
        for combo in parent.findChildren(QComboBox):
            try:
                combo.activated.disconnect(self._on_combo_activated)
            except Exception:
                pass

    def save_tutorial(self, name):
        folder = "tutorials"
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, f"{name}.json")
        with open(path, "w") as f:
            json.dump(self.steps, f, indent=4)
        return path

    def load_tutorial(self, name):
        path = os.path.join("tutorials", f"{name}.json")
        if not os.path.exists(path):
            return False
        with open(path, "r") as f:
            self.steps = json.load(f)
        return True

    def start_playback(self):
        if not self.steps:
            return
        self.is_playing = True
        self.is_recording = False
        self.current_step_idx = 0
        QApplication.instance().installEventFilter(self)
        # Also listen for combo activations during playback to auto-advance
        self._connect_combos_playback(self.main_window)
        self.playback_started.emit()
        self._highlight_current_step()

    def _connect_combos_playback(self, parent):
        for combo in parent.findChildren(QComboBox):
            try:
                combo.activated.connect(self._on_combo_activated_playback)
            except Exception:
                pass

    def _on_combo_activated_playback(self, index):
        if not self.is_playing:
            return
        target = self.steps[self.current_step_idx]
        combo = self.sender()
        if combo.objectName() == target["name"] and target.get("type") == "select":
            if index == target["index"]:
                self.next_step()

    def stop_playback(self):
        self._clear_highlights()
        self.is_playing = False
        QApplication.instance().removeEventFilter(self)
        self._disconnect_combos_playback(self.main_window)
        self.playback_finished.emit()

    def _disconnect_combos_playback(self, parent):
        for combo in parent.findChildren(QComboBox):
            try:
                combo.activated.disconnect(self._on_combo_activated_playback)
            except Exception:
                pass

    def eventFilter(self, obj, event):
        if self.is_recording and event.type() == QEvent.MouseButtonRelease:
            action_name = None

            # Check for QAction via Menu/Toolbar/MenuBar
            if isinstance(obj, (QMenu, QMenuBar, QToolBar)):
                action = obj.actionAt(event.pos())
                if action and action.objectName():
                    action_name = action.objectName()
                    # Record action immediately
                    if not self.steps or self.steps[-1].get("name") != action_name:
                        self.steps.append(
                            {
                                "name": action_name,
                                "type": "action",
                                "class": "QAction",
                                "text": action.text().replace("&", ""),
                            }
                        )
                        print(f"[Tutorial] Recorded action: {action_name}")
                    return False  # Don't consume

            # If the widget itself has no name, crawl up to find first named parent
            curr = obj
            name = curr.objectName()
            while curr and not name:
                curr = curr.parent()
                if curr:
                    name = curr.objectName()

            if curr:
                obj = curr

            if not name:
                return super().eventFilter(obj, event)

            supported_classes = (
                QPushButton,
                QComboBox,
                QCheckBox,
                QRadioButton,
                QSpinBox,
                QDoubleSpinBox,
                QSlider,
                QToolButton,
            )

            if isinstance(obj, supported_classes):
                # Avoid duplicate steps if multiple events fire for one click
                if not self.steps or self.steps[-1].get("name") != name:
                    self.steps.append(
                        {"name": name, "type": "click", "class": obj.__class__.__name__}
                    )
                    print(
                        f"[Tutorial] Recorded click on: {name} ({obj.__class__.__name__})"
                    )

            elif isinstance(obj, QTabBar):
                # For tabs, we record the index
                idx = obj.tabAt(event.pos())
                if idx != -1:
                    # Find the parent tab widget's name
                    tw = obj.parent()
                    while tw and not isinstance(tw, QTabWidget):
                        tw = tw.parent()

                    if tw and tw.objectName():
                        tab_name = tw.objectName()
                        # Avoid duplicates
                        if (
                            not self.steps
                            or self.steps[-1].get("name") != tab_name
                            or self.steps[-1].get("index") != idx
                        ):
                            self.steps.append(
                                {"name": tab_name, "type": "tab", "index": idx}
                            )
                            print(
                                f"[Tutorial] Recorded tab switch: {tab_name} -> index {idx}"
                            )

        elif self.is_playing:
            if event.type() == QEvent.MouseButtonRelease:
                # Check if the clicked object is the one we are highlighting
                if self.current_step_idx >= len(self.steps):
                    self.stop_playback()
                    return super().eventFilter(obj, event)

                target = self.steps[self.current_step_idx]
                obj_name = obj.objectName()
                is_correct = False

                # 1. Check QAction via Menu/Toolbar
                if target.get("type") == "action":
                    if isinstance(obj, (QMenu, QMenuBar, QToolBar)):
                        action = obj.actionAt(event.pos())
                        if action and action.objectName() == target["name"]:
                            is_correct = True

                # 2. Check direct match (Widgets)
                elif obj_name == target["name"]:
                    is_correct = True

                # 3. Check tab match
                elif target["type"] == "tab" and isinstance(obj, QTabBar):
                    # We need to check if this tab bar belongs to the named QTabWidget
                    parent = obj.parent()
                    while parent and not isinstance(parent, QTabWidget):
                        parent = parent.parent()

                    if parent and parent.objectName() == target["name"]:
                        idx = obj.tabAt(event.pos())
                        if idx == target["index"]:
                            is_correct = True

                if is_correct:
                    # Move to next step shortly after to allow the event to process
                    self.next_step()

            elif event.type() == QEvent.Show:
                # Special handling for QComboBox popups to highlight the target item
                if isinstance(obj, QAbstractItemView):
                    parent = obj.parentWidget()
                    while parent and not isinstance(parent, QComboBox):
                        parent = parent.parentWidget()

                    if parent:
                        target = self.steps[self.current_step_idx]
                        if (
                            parent.objectName() == target.get("name")
                            and target.get("type") == "select"
                        ):
                            idx = target.get("index")
                            if idx is not None:
                                # We highlight the item in the list view
                                # This is best done by setting the current index
                                obj.setCurrentIndex(obj.model().index(idx, 0))
                                # Also scroll to it
                                obj.scrollTo(obj.model().index(idx, 0))

        return super().eventFilter(obj, event)

    def next_step(self):
        """Advance to the next tutorial step."""
        self._clear_highlights()
        self.current_step_idx += 1
        if self.current_step_idx < len(self.steps):
            self._highlight_current_step()
            self.step_reached.emit(self.current_step_idx, len(self.steps))
        else:
            self.stop_playback()

    def prev_step(self):
        """Go back to the previous tutorial step."""
        if self.current_step_idx > 0:
            self._clear_highlights()
            self.current_step_idx -= 1
            self._highlight_current_step()
            self.step_reached.emit(self.current_step_idx, len(self.steps))

    def get_current_instruction(self):
        """Get text instruction for the current step."""
        if 0 <= self.current_step_idx < len(self.steps):
            step = self.steps[self.current_step_idx]
            name = step.get("name", "Unknown Widget")
            if step.get("type") == "action":
                text = step.get("text", name)
                return f"Click '{text}' in menu or toolbar"

            if step.get("type") == "select":
                text = step.get("text", f"index {step.get('index')}")
                return f"Select '{text}' in '{name}'"

            action = "Click" if step.get("type") == "click" else "Select"
            if step.get("type") == "tab":
                return f"{action} tab index {step.get('index')} on '{name}'"
            return f"{action} '{name}'"
        return "Tutorial finished"

    def get_current_description(self):
        """Get optional description for the current step."""
        if 0 <= self.current_step_idx < len(self.steps):
            return self.steps[self.current_step_idx].get("description")
        return None

    def _highlight_current_step(self):
        target = self.steps[self.current_step_idx]

        # Handle Actions
        if target.get("type") == "action":
            # Try to find a widget associated with this action (e.g. Toolbar button)
            action = self.main_window.findChild(QAction, target["name"])
            if action:
                for toolbar in self.main_window.findChildren(QToolBar):
                    widget = toolbar.widgetForAction(action)
                    if widget and widget.isVisible():
                        self.original_styles[widget] = widget.styleSheet()
                        widget.setStyleSheet(self.highlight_style)
                        return
            return

        # Handle Widgets
        widget = self.main_window.findChild(QWidget, target["name"])

        if widget:
            if target.get("type") == "tab" and isinstance(widget, QTabWidget):
                # For tabs, we highlight ONLY the specific tab in the bar
                bar = widget.tabBar()
                idx = target.get("index")
                if idx is not None and 0 <= idx < bar.count():
                    rect = bar.tabRect(idx)
                    # Create a highlight overlay over the tab
                    overlay = HighlightOverlay(bar)
                    overlay.setGeometry(rect)
                    overlay.show()
                    self.overlays.append(overlay)

                    # Also scroll to the tab if it's hidden
                    bar.setCurrentIndex(idx)
            else:
                self.original_styles[widget] = widget.styleSheet()
                widget.setStyleSheet(self.highlight_style)

    def _clear_highlights(self):
        for widget, style in self.original_styles.items():
            try:
                widget.setStyleSheet(style)
            except RuntimeError:
                pass
        self.original_styles = {}

        for overlay in self.overlays:
            try:
                overlay.hide()
                overlay.deleteLater()
            except RuntimeError:
                pass
        self.overlays = []
