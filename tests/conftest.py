"""Shared pytest fixtures for GUI tests."""

import gc
import os

# GUI tests must not attempt to connect to the native window server in headless
# environments. Keep an explicitly selected platform intact for local/Xvfb
# runs, but provide a safe default before importing Qt or pyqtgraph.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from pyqtgraph.Qt import isQObjectAlive
import pytest
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QApplication

from blochsimulator.memory import get_default_memory_policy, set_default_memory_policy


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
        if widget.isVisible():
            widget.close()
    app.processEvents()
    # Normal pytest processes still need pyqtgraph's ViewBox signal
    # disconnection while QApplication and all Qt wrappers are alive.
    pg.cleanup()


def _is_application_widget(widget):
    return type(widget).__module__.startswith("blochsimulator.")


def _forget_deleted_viewboxes():
    """Drop ViewBoxes whose Qt object is gone from pyqtgraph's global registry.

    With automatic GC disabled, their Python wrappers stay referenced by
    cycles, so the WeakKeyDictionary never forgets them on its own.
    """
    for view in list(pg.ViewBox.AllViews):
        if not isQObjectAlive(view):
            pg.ViewBox.AllViews.pop(view, None)


def _close_and_delete_top_level_widgets(app):
    """Close visible windows and schedule parentless app widgets for deletion.

    Kept apart from the fixture so no local variable still references a
    window when the reference cycles are collected. Returns whether any
    top-level widget existed.
    """
    top_level_widgets = QApplication.topLevelWidgets()
    for widget in top_level_widgets:
        if widget.isVisible():
            widget.close()
    app.processEvents()
    for widget in QApplication.topLevelWidgets():
        if widget.parentWidget() is None and _is_application_widget(widget):
            widget.deleteLater()
    return bool(top_level_widgets)


@pytest.fixture(autouse=True)
def inline_sequence_generation(monkeypatch):
    """Generate sequences inline so tests can check the result right away.

    Tests of the background generation switch it back on themselves.
    """
    from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget

    monkeypatch.setattr(
        SequenceSimulationWidget, "background_sequence_generation", False
    )


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qt_application):
    """Close and destroy top-level widgets between tests.

    ``processEvents`` never delivers ``DeferredDelete`` outside a running
    event loop, so ``deleteLater`` alone left every test's windows alive.
    Each new named pyqtgraph ViewBox then refreshed the context menus of all
    ViewBoxes ever created, making every test slower than the one before.
    Parentless application widgets a test did not delete itself are deleted
    too. Third-party top-level widgets are not deleted directly: some are
    owned elsewhere (e.g. QWidgetAction default widgets in pyqtgraph menus),
    and deleting them separately crashes when their owner is destroyed.
    Only visible windows are closed: every ``close()`` scans all top-level
    widgets.

    The parentless pyqtgraph menus belong to Python wrappers in reference
    cycles, which the disabled automatic GC never frees. PyQt also keeps
    every lambda slot, and with it the window it captures, until the
    ``destroyed`` call its deleted sender queued has been delivered. So the
    queued calls are delivered, the slot proxies deleted and the cycles
    collected here, at a point where no Qt code is running.
    """
    memory_policy = get_default_memory_policy()
    yield

    set_default_memory_policy(memory_policy)
    had_widgets = _close_and_delete_top_level_widgets(qt_application)
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.sendPostedEvents()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    _forget_deleted_viewboxes()
    if had_widgets:
        # Tests without widgets skip the ~35 ms full collection.
        gc.collect()
