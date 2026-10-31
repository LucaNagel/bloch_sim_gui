"""Setup / Results / Analysis tab groups and folded "Advanced" rows."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QFormLayout

from blochsimulator.ui.main_window import BlochSimulatorGUI


def _visible_tabs(window):
    tabs = window.tab_widget
    return [index for index in range(tabs.count()) if tabs.isTabVisible(index)]


def test_group_bar_shows_only_the_tabs_of_the_selected_group():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    bar = window.main_tab_group_bar

    assert bar.current() == "results"
    assert window.magnetization_tab_index in _visible_tabs(window)
    for index in _visible_tabs(window):
        assert window._main_tab_group_of.get(index) == "results"

    window.tab_widget.setCurrentIndex(window.signal_tab_index)
    bar.buttons["analysis"].click()
    app.processEvents()
    assert bar.current() == "analysis"
    assert window.tab_widget.currentIndex() in _visible_tabs(window)
    for index in _visible_tabs(window):
        assert window._main_tab_group_of.get(index) == "analysis"
    assert window.signal_tab_index not in _visible_tabs(window)

    # Returning to a group reopens the tab that was last shown there.
    bar.buttons["results"].click()
    app.processEvents()
    assert window.tab_widget.currentIndex() == window.signal_tab_index
    window.close()


def test_sequence_mode_starts_in_setup_and_follows_tab_requests():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()

    window.set_workspace_mode("sequence")
    app.processEvents()
    bar = window.main_tab_group_bar
    assert bar.current() == "setup"
    assert window.tab_widget.currentIndex() == window.sequence_simulation_tab_index
    assert window.tab_widget.isTabVisible(window.phantom_tab_index)
    assert window.main_tab_available(window.simulation_explorer_tab_index)
    assert not window.tab_widget.isTabVisible(window.simulation_explorer_tab_index)

    # Selecting a tab of another group, e.g. from a "Show …" link, switches
    # the group with it.
    window.tab_widget.setCurrentIndex(window.simulation_explorer_tab_index)
    app.processEvents()
    assert bar.current() == "results"
    assert window.tab_widget.isTabVisible(window.simulation_explorer_tab_index)
    assert not window.tab_widget.isTabVisible(window.phantom_tab_index)
    window.close()


def test_rarely_changed_sequence_rows_fold_into_advanced():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    window.set_workspace_mode("sequence")
    app.processEvents()
    widget = window.sequence_simulation_widget

    section = widget.advanced_sections["flash_group"]
    body_form = section.body.layout()
    assert isinstance(body_form, QFormLayout)
    assert body_form.getWidgetPosition(widget.flash_read_gradient_axis)[0] >= 0
    assert body_form.getWidgetPosition(widget.flash_spoiler_duration_ms)[0] >= 0
    main_form = widget.flash_group.layout()
    assert main_form.getWidgetPosition(widget.flash_spoiler_duration_ms)[0] < 0
    assert main_form.getWidgetPosition(widget.flash_read_matrix)[0] >= 0
    assert body_form.labelForField(widget.flash_spoiler_duration_ms).text() == (
        "Spoiler duration"
    )

    # Folded rows remain parameter-loop targets.
    expanded = section.is_expanded()
    try:
        section.set_expanded(False)
        widget.sequence_source.setCurrentIndex(widget.FLASH_SOURCE)
        labels = [parameter.label for parameter in widget._sequence_loop_parameters()]
        assert "Spoiler duration" in labels
    finally:
        section.set_expanded(expanded)
    window.close()


def test_free_mode_parameter_panel_keeps_its_width_when_the_window_grows():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BlochSimulatorGUI()
    for width in (1280, 1920):
        window.resize(width, 800)
        window.show()
        app.processEvents()
        # The plots, and with them the tab row, take the extra width.
        assert window.main_splitter.sizes()[0] == 420
    window.close()
