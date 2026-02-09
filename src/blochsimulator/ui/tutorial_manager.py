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
)
from PyQt5.QtCore import QObject, QEvent, pyqtSignal, Qt


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
            border: 3px solid #FFD700;
            background-color: #FFF8DC;
            border-radius: 4px;
        """
        self.original_styles = {}  # Store old styles to revert them

    def start_recording(self):
        self.steps = []
        self.is_recording = True
        self.is_playing = False
        # Install event filter on the application to catch all clicks
        QApplication.instance().installEventFilter(self)
        self.recording_started.emit()

    def stop_recording(self):
        self.is_recording = False
        QApplication.instance().removeEventFilter(self)
        self.recording_stopped.emit(len(self.steps))
        return self.steps

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
        self.playback_started.emit()
        self._highlight_current_step()

    def stop_playback(self):
        self._clear_highlights()
        self.is_playing = False
        QApplication.instance().removeEventFilter(self)
        self.playback_finished.emit()

    def eventFilter(self, obj, event):
        if self.is_recording and event.type() == QEvent.MouseButtonRelease:
            # Record buttons, tabs, and combo boxes
            name = obj.objectName()

            # If the widget itself has no name, check its immediate parent
            # (sometimes clicks land on sub-elements of complex widgets like SpinBox)
            if not name and obj.parent():
                parent_name = obj.parent().objectName()
                if parent_name:
                    obj = obj.parent()
                    name = parent_name

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

        elif self.is_playing and event.type() == QEvent.MouseButtonRelease:
            # Check if the clicked object is the one we are highlighting
            if self.current_step_idx >= len(self.steps):
                self.stop_playback()
                return super().eventFilter(obj, event)

            target = self.steps[self.current_step_idx]
            obj_name = obj.objectName()

            is_correct = False

            # 1. Check direct match
            if obj_name == target["name"]:
                is_correct = True

            # 2. Check tab match
            if target["type"] == "tab" and isinstance(obj, QTabBar):
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
                # We can't delay in eventFilter, so we just call next_step
                self._next_step()

        return super().eventFilter(obj, event)

    def _next_step(self):
        self._clear_highlights()
        self.current_step_idx += 1
        if self.current_step_idx < len(self.steps):
            self._highlight_current_step()
            self.step_reached.emit(self.current_step_idx, len(self.steps))
        else:
            self.stop_playback()

    def _highlight_current_step(self):
        target = self.steps[self.current_step_idx]
        widget = self.main_window.findChild(QWidget, target["name"])

        if widget:
            self.original_styles[widget] = widget.styleSheet()
            # Append highlight style to existing style (or replace if empty)
            new_style = self.highlight_style
            widget.setStyleSheet(new_style)

    def _clear_highlights(self):
        for widget, style in self.original_styles.items():
            try:
                widget.setStyleSheet(style)
            except RuntimeError:
                # Widget might have been deleted
                pass
        self.original_styles = {}
