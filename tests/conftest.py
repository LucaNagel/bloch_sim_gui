"""Shared pytest fixtures for GUI tests."""

import gc

import pyqtgraph as pg
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def qt_application():
    """Keep one QApplication alive for the complete test session.

    Recreating QApplication between tests while pyqtgraph objects are waiting
    for deferred deletion can segfault inside Qt during Python garbage
    collection.  A session-owned reference gives all GUI tests the same Qt
    application and context-sharing policy.
    """
    # pyqtgraph ViewBoxes contain Python/Qt reference cycles whose destruction
    # callbacks can enter already-deleted C++ widgets when automatic cyclic GC
    # runs between tests. Reference counting and the explicit Qt cleanup below
    # remain active; the short-lived pytest process releases everything at exit.
    gc.disable()
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication([])
    yield app
    for widget in QApplication.topLevelWidgets():
        widget.close()
    app.processEvents()
    # Run pyqtgraph's ViewBox signal disconnection while QApplication and all
    # Qt wrappers are still alive. Relying on atexit is too late on macOS and
    # can otherwise produce exit code 139 after every test has passed.
    pg.cleanup()


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qt_application):
    """Close top-level widgets while keeping pyqtgraph's exit registry intact."""
    yield

    for widget in QApplication.topLevelWidgets():
        widget.close()
    qt_application.processEvents()
