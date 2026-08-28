from types import SimpleNamespace

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QInputDialog, QMessageBox

from blochsimulator.ui.simulation_explorer import (
    SessionSimulationExplorer,
    SessionSimulationRun,
)
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget


def _run(run_id="run-1"):
    return SessionSimulationRun(
        run_id=run_id,
        created_at_utc="2026-08-28T10:20:30+00:00",
        sequence_name="internal-flash-2d",
        phantom_name="Kidney phantom",
        phantom_shape=(8, 9, 10),
        sequence_duration_s=0.125,
        adc_samples=72,
        runtime_s=1.25,
        kernel="optimized",
        result=SimpleNamespace(signal=np.zeros((2, 72))),
        state={"program": object(), "phantom": object()},
    )


def test_session_simulation_explorer_adds_opens_and_deletes_runs(
    qt_application, monkeypatch
):
    explorer = SessionSimulationExplorer()
    run = explorer.add_run(_run())

    assert run.display_name == "Run 1"
    assert explorer.run_tree.topLevelItemCount() == 1
    assert explorer.run_tree.topLevelItem(0).text(1) == "internal-flash-2d"
    assert explorer.run_tree.topLevelItem(0).text(2) == "Kidney phantom"
    detail_names = {
        explorer.details_table.item(row, 0).text()
        for row in range(explorer.details_table.rowCount())
    }
    assert {"Sequence", "Phantom", "Phantom matrix", "Simulation runtime"} <= (
        detail_names
    )

    opened = []
    explorer.run_open_requested.connect(opened.append)
    explorer._open_selected()
    assert opened == [run]

    deleted = []
    explorer.run_deleted.connect(deleted.append)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes
    )
    explorer._delete_selected()
    assert deleted == [run]
    assert explorer.run_tree.topLevelItemCount() == 0
    assert explorer.runs == ()


def test_session_simulation_explorer_keeps_newest_run_first(qt_application):
    explorer = SessionSimulationExplorer()
    first = explorer.add_run(_run("first"))
    second = explorer.add_run(_run("second"))

    assert first.display_name == "Run 1"
    assert second.display_name == "Run 2"
    assert explorer.run_tree.topLevelItem(0).data(0, Qt.UserRole) == "second"

    first_item = explorer.run_tree.topLevelItem(1)
    first_item.setSelected(True)
    exported = []
    explorer.runs_export_requested.connect(exported.append)
    explorer._export_selected()
    assert len(exported) == 1
    assert {run.run_id for run in exported[0]} == {"first", "second"}


def test_session_simulation_explorer_renames_one_run(qt_application, monkeypatch):
    explorer = SessionSimulationExplorer()
    run = explorer.add_run(_run())
    renamed = []
    explorer.run_renamed.connect(renamed.append)
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        lambda *args, **kwargs: ("Renal baseline", True),
    )

    explorer._rename_selected()

    assert run.display_name == "Renal baseline"
    assert run.custom_name
    assert explorer.run_tree.topLevelItem(0).text(0) == "Renal baseline"
    assert explorer.details_table.item(0, 1).text() == "Renal baseline"
    assert renamed == [run]


def test_sequence_workspace_restores_and_forgets_a_session_run(qt_application):
    widget = SequenceSimulationWidget()
    widget.object_source.setCurrentIndex(1)
    widget.matrix_size.setValue(2)
    widget.z_matrix_size.setValue(2)
    widget._build_phantom()
    result = widget.simulator.simulate_sequence(widget.program, widget.phantom)
    run = widget._register_session_simulation_run(result)
    run.display_name = "Run 1"

    widget.result = None
    assert widget.restore_session_simulation_run(run)
    assert widget.result is result
    assert widget._active_session_run_id == run.run_id
    assert widget.status.text() == "Showing Run 1 from this session"

    widget.forget_session_simulation_run(run)
    assert widget.result is None
    assert widget._active_session_run_id is None
    assert widget.progress.format() == "Run deleted"

    widget.close()
    widget.deleteLater()
    qt_application.processEvents()
