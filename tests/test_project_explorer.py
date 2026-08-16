from PyQt5.QtCore import QSettings

from blochsimulator.project_io import save_project
from blochsimulator.ui.project_explorer import ProjectExplorerDialog


def test_project_explorer_indexes_filters_and_opens_projects(tmp_path, qt_application):
    projects = tmp_path / "projects"
    projects.mkdir()
    path = projects / "sequence-study.blochproj"
    save_project(
        path,
        {
            "application_version": "3.0",
            "workspace_mode": "sequence",
        },
        legacy_result={"signal": [1.0, 2.0]},
    )
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    dialog = ProjectExplorerDialog(
        settings=settings,
        default_folders=(projects,),
    )
    qt_application.processEvents()

    assert dialog.folder_list.count() == 1
    assert dialog.project_tree.topLevelItemCount() == 1
    assert dialog.selected_project_path() == str(path.resolve())
    assert "Free-mode result" in dialog.project_tree.topLevelItem(0).text(1)
    assert "sequence" in dialog.details.toPlainText()

    dialog.filter_edit.setText("does-not-exist")
    assert dialog.project_tree.topLevelItem(0).isHidden()
    assert dialog.status_label.text() == "0 of 1 projects"
    dialog.filter_edit.clear()

    opened = []
    dialog.project_open_requested.connect(opened.append)
    dialog._open_selected()
    assert opened == [str(path.resolve())]

    reopened = ProjectExplorerDialog(settings=settings)
    assert reopened.folder_list.item(0).text() == str(projects)
