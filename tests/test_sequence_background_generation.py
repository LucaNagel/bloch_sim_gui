import sys
import threading
import time

import numpy as np
import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox

from blochsimulator.sequence import SpinSampling
from blochsimulator.sequence.spoiling import (
    _phase_train_error_function,
    _repeated_train_error_function,
    analyze_phase_cycle_train,
    analyze_repeated_spoiler_train,
)
from blochsimulator.ui import sequence_simulation_widget
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget


class _JobRecorder:
    """Record generation jobs and optionally hold the first worker job."""

    def __init__(self, monkeypatch, *, hold=False, error=None):
        self.threads = []
        self.parameters = []
        self.release = threading.Event()
        self.started = threading.Event()
        if not hold:
            self.release.set()
        original = sequence_simulation_widget._GeneratedSequenceJob.run
        recorder = self

        def run(job, status_callback=None):
            recorder.threads.append(threading.current_thread())
            recorder.parameters.append(dict(job.parameters))
            if threading.current_thread() is not threading.main_thread():
                recorder.started.set()
                assert recorder.release.wait(30.0)
            if error is not None:
                raise error
            return original(job, status_callback=status_callback)

        monkeypatch.setattr(
            sequence_simulation_widget._GeneratedSequenceJob, "run", run
        )


def _background_widget(monkeypatch):
    monkeypatch.setattr(
        SequenceSimulationWidget, "background_sequence_generation", True
    )
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SequenceSimulationWidget()
    widget.sequence_source.setCurrentIndex(widget.FLASH_SOURCE)
    widget.flash_read_matrix.setValue(8)
    widget.flash_phase_matrix.setValue(4)
    widget.flash_repetitions.setValue(1)
    return app, widget


def _wait_until(app, condition, timeout_s=60.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "condition was not reached in time"
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()


def _wait_for_generation(app, widget):
    _wait_until(
        app,
        lambda: widget._sequence_generation is None
        and not widget._sequence_generation_threads,
    )


def _close(app, widget):
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_generate_builds_the_sequence_in_a_worker_thread(monkeypatch):
    app, widget = _background_widget(monkeypatch)
    recorder = _JobRecorder(monkeypatch, hold=True)
    previous = widget.program

    widget.generate_sequence_button.click()
    assert recorder.started.wait(30.0)
    app.processEvents()

    # The window keeps handling events while the job runs.
    assert widget.program is previous
    assert not widget.generate_sequence_button.isEnabled()
    assert widget.generate_sequence_button.text() == "Generating sequence…"
    assert widget.sequence_info.text().startswith("Generating sequence…")

    recorder.release.set()
    _wait_for_generation(app, widget)

    assert recorder.threads[0] is not threading.main_thread()
    assert widget.program.source == "internal-flash-2d"
    assert widget.acquisition.read_matrix == 8
    assert not widget._generated_sequence_needs_reload()
    assert widget.generate_sequence_button.isEnabled()
    assert widget.generate_sequence_button.text() == "Generate sequence"
    _close(app, widget)


def test_settings_changed_during_generation_generate_again(monkeypatch):
    app, widget = _background_widget(monkeypatch)
    recorder = _JobRecorder(monkeypatch, hold=True)

    widget.generate_sequence_button.click()
    assert recorder.started.wait(30.0)
    widget.flash_read_matrix.setValue(16)
    app.processEvents()

    assert widget.generate_sequence_button.text() == "Generating sequence…"
    assert "generated again with the new values" in widget.sequence_info.text()

    recorder.release.set()
    _wait_for_generation(app, widget)

    assert len(recorder.parameters) == 2
    assert widget.acquisition.read_matrix == 16
    assert not widget._generated_sequence_needs_reload()
    _close(app, widget)


def test_outdated_live_preview_result_is_discarded_without_restart(monkeypatch):
    app, widget = _background_widget(monkeypatch)
    recorder = _JobRecorder(monkeypatch, hold=True)
    previous = widget.program
    widget.sequence_live_preview.setChecked(True)
    assert recorder.started.wait(30.0)

    # Without live preview and without a waiting caller, nothing regenerates.
    widget.sequence_live_preview.setChecked(False)
    widget.flash_read_matrix.setValue(16)
    assert widget.generate_sequence_button.isEnabled()
    assert "Click Generate sequence" in widget.sequence_info.text()

    recorder.release.set()
    _wait_for_generation(app, widget)

    assert len(recorder.parameters) == 1
    assert widget.program is previous
    assert widget._generated_sequence_needs_reload()
    _close(app, widget)


def test_run_waits_for_the_generation_and_then_starts(monkeypatch):
    app, widget = _background_widget(monkeypatch)
    recorder = _JobRecorder(monkeypatch, hold=True)
    started = []
    monkeypatch.setattr(
        widget,
        "_start_sequence_run",
        lambda: started.append(widget.program) or True,
    )

    assert widget._run()
    assert recorder.started.wait(30.0)
    app.processEvents()

    assert started == []
    assert widget._run_waiting_for_generation
    assert not widget.run_button.isEnabled()
    assert widget.cancel_button.isEnabled()
    assert widget.progress.format() == "Generating sequence…"

    recorder.release.set()
    _wait_for_generation(app, widget)

    assert not widget._run_waiting_for_generation
    assert len(started) == 1
    assert started[0] is widget.program
    assert widget.program.source == "internal-flash-2d"
    _close(app, widget)


def test_cancel_while_waiting_drops_the_run_but_keeps_the_sequence(monkeypatch):
    app, widget = _background_widget(monkeypatch)
    recorder = _JobRecorder(monkeypatch, hold=True)
    started = []
    monkeypatch.setattr(
        widget, "_start_sequence_run", lambda: started.append(True) or True
    )

    widget._run()
    assert recorder.started.wait(30.0)
    widget.cancel_button.click()

    assert not widget._run_waiting_for_generation
    assert not widget.cancel_button.isEnabled()
    assert widget.run_button.isEnabled()
    assert widget.progress.format() == "Stopped"
    assert widget.status.text() == "Run cancelled before the sequence was ready"

    recorder.release.set()
    _wait_for_generation(app, widget)

    assert started == []
    assert widget.program.source == "internal-flash-2d"
    _close(app, widget)


def test_failed_generation_reports_to_the_waiting_run(monkeypatch):
    app, widget = _background_widget(monkeypatch)
    _JobRecorder(monkeypatch, error=ValueError("raster check failed"))
    previous = widget.program
    started = []
    monkeypatch.setattr(
        widget, "_start_sequence_run", lambda: started.append(True) or True
    )
    dialogs = []
    monkeypatch.setattr(
        QMessageBox, "exec_", lambda dialog: dialogs.append(dialog.detailedText())
    )

    widget._run()
    _wait_for_generation(app, widget)

    assert started == []
    assert widget.program is previous
    assert widget._generated_sequence_needs_reload()
    assert widget.status.text() == "Sequence generation failed"
    assert widget.progress.format() == "Not started"
    assert widget.run_button.isEnabled()
    assert len(dialogs) == 1
    assert "raster check failed" in dialogs[0]
    _close(app, widget)


def test_inline_reload_supersedes_the_background_generation(monkeypatch):
    app, widget = _background_widget(monkeypatch)
    recorder = _JobRecorder(monkeypatch, hold=True)
    outcomes = []

    widget._generate_selected_sequence(outcomes.append)
    assert recorder.started.wait(30.0)
    widget.flash_read_matrix.setValue(16)

    # Callers that need the sequence right away generate it inline.
    assert widget._ensure_current_generated_sequence()
    assert outcomes == [None]
    assert recorder.threads[-1] is threading.main_thread()
    assert widget.acquisition.read_matrix == 16
    program = widget.program

    recorder.release.set()
    _wait_until(app, lambda: not widget._sequence_generation_threads)

    assert widget.program is program
    assert widget.generate_sequence_button.isEnabled()
    _close(app, widget)


def _select_loop_parameter(widget, key, start, stop, steps):
    combo = widget.parameter_loop_parameter
    index = next(
        index for index in range(combo.count()) if combo.itemData(index).key == key
    )
    combo.setCurrentIndex(index)
    widget.parameter_loop_start.setValue(start)
    widget.parameter_loop_stop.setValue(stop)
    widget.parameter_loop_steps.setValue(steps)


def test_parameter_loop_waits_for_each_background_generation(monkeypatch):
    app, widget = _background_widget(monkeypatch)
    recorder = _JobRecorder(monkeypatch)
    _select_loop_parameter(widget, "flash_flip_angle_deg", 10.0, 30.0, 3)
    started = []

    def start_run():
        started.append(
            (
                recorder.parameters[-1]["flip_angle_deg"],
                recorder.threads[-1],
                widget.program,
            )
        )
        QTimer.singleShot(0, widget._parameter_loop_iteration_finished)
        return True

    monkeypatch.setattr(widget, "_start_sequence_run", start_run)

    widget._start_parameter_loop()
    _wait_until(app, lambda: widget._parameter_loop_state is None)

    assert [flip for flip, _thread, _program in started] == [10.0, 20.0, 30.0]
    assert all(thread is not threading.main_thread() for _, thread, _ in started)
    assert len({id(program) for _, _, program in started}) == 3
    assert widget.loop_progress.format().startswith("Loop complete · 3/3 runs")
    _close(app, widget)


def test_cancelling_a_loop_while_its_sequence_is_generated(monkeypatch):
    app, widget = _background_widget(monkeypatch)
    recorder = _JobRecorder(monkeypatch, hold=True)
    _select_loop_parameter(widget, "flash_flip_angle_deg", 10.0, 30.0, 3)
    started = []
    monkeypatch.setattr(
        widget, "_start_sequence_run", lambda: started.append(True) or True
    )

    widget._start_parameter_loop()
    _wait_until(app, recorder.started.is_set)
    widget.cancel_button.click()

    assert widget._parameter_loop_state is None
    assert not widget._run_waiting_for_generation
    assert widget.loop_progress.format().startswith("Loop cancelled after 0/3 runs")
    assert not widget.cancel_button.isEnabled()

    recorder.release.set()
    _wait_for_generation(app, widget)

    assert started == []
    assert widget.run_button.isEnabled()
    _close(app, widget)


@pytest.mark.parametrize(
    "cycles, excitations",
    [((1.0, 1.0, 1.0), 64), ((0.5, 2.0, 0.0), 32), ((3.0, 0.25, 1.5), 1)],
)
@pytest.mark.parametrize("counts", [(1, 1, 1), (4, 8, 2), (7, 3, 5)])
def test_cached_midpoint_error_matches_the_full_train_analysis(
    cycles, excitations, counts
):
    cycles = np.asarray(cycles)
    expected = analyze_repeated_spoiler_train(
        cycles, SpinSampling(counts, method="midpoint"), excitations
    ).maximum_sampling_error
    error = _repeated_train_error_function(cycles, "midpoint", excitations)

    assert error(counts) == pytest.approx(expected, rel=1e-12, abs=1e-14)

    phases = np.array([[0.5, 1.0, 0.0], [1.5, 0.25, 2.0], [3.0, 0.75, 1.0]])
    expected = analyze_phase_cycle_train(
        phases, SpinSampling(counts, method="midpoint")
    ).maximum_sampling_error
    assert _phase_train_error_function(phases, "midpoint")(counts) == pytest.approx(
        expected, rel=1e-12, abs=1e-14
    )
