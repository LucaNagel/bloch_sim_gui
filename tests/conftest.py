"""Shared pytest fixtures for GUI tests."""

import gc
import os

import pyqtgraph as pg
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication


BATCHED_PYTEST_ENV = "BLOCHSIMULATOR_BATCHED_PYTEST"
_QT_APPLICATION = None


@pytest.fixture(scope="session", autouse=True)
def qt_application():
    """Keep one QApplication alive for the complete test session.

    Recreating QApplication between tests while pyqtgraph objects are waiting
    for deferred deletion can segfault inside Qt during Python garbage
    collection.  A process-owned reference gives all GUI tests the same Qt
    application and keeps it alive until the batch runner exits.
    """
    global _QT_APPLICATION

    # pyqtgraph ViewBoxes contain Python/Qt reference cycles whose destruction
    # callbacks can enter already-deleted C++ widgets when automatic cyclic GC
    # runs between tests. Normal pytest runs use the explicit cleanup below;
    # disposable batch workers leave the cycles alive until their direct exit.
    gc.disable()
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    _QT_APPLICATION = QApplication.instance() or QApplication([])
    app = _QT_APPLICATION
    yield app
    if os.environ.get(BATCHED_PYTEST_ENV):
        # The batch runner exits with os._exit as soon as pytest has reported
        # its result.  Do not enter Qt/PyQtGraph session teardown first: that
        # is the native code path which can segfault after otherwise-passing
        # headless OpenGL tests.  The disposable worker process owns all of
        # these resources, so the operating system reclaims them on exit.
        return

    for widget in QApplication.topLevelWidgets():
        widget.close()
    app.processEvents()
    # Normal pytest processes still need pyqtgraph's ViewBox signal
    # disconnection while QApplication and all Qt wrappers are alive.
    pg.cleanup()


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qt_application):
    """Close top-level widgets while keeping pyqtgraph's exit registry intact."""
    yield

    for widget in QApplication.topLevelWidgets():
        widget.close()
    qt_application.processEvents()
