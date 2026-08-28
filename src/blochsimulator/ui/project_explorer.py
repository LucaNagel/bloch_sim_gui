"""Metadata-only browser for saved Bloch Simulator projects."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QSettings, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..project_io import scan_project_folders


SETTINGS_FOLDERS = "project_explorer/folders"
SETTINGS_RECURSIVE = "project_explorer/recursive"


def _shape_text(shape):
    if not shape:
        return ""
    return "×".join(str(value) for value in shape)


def _size_text(byte_count):
    value = float(byte_count or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return ""


def _date_text(value):
    if not value:
        return "—"
    try:
        return (
            datetime.fromisoformat(str(value)).astimezone().strftime("%Y-%m-%d %H:%M")
        )
    except (TypeError, ValueError):
        return str(value)


def _content_labels(metadata):
    contents = metadata.get("contents", {})
    labels = []
    if contents.get("phantom"):
        labels.append("Phantom")
    if contents.get("tx_field"):
        labels.append("Tx B1")
    if contents.get("rx_field"):
        labels.append("Rx B1")
    if contents.get("sequence"):
        labels.append("Sequence")
    if contents.get("free_mode_result"):
        labels.append("Free-mode result")
    if contents.get("sequence_result"):
        labels.append("Sequence result")
    return labels


def _detail_html(metadata):
    def escape(value):
        return html.escape(str(value))

    if metadata.get("error"):
        return (
            f"<h3>{escape(metadata.get('name', 'Invalid project'))}</h3>"
            f"<p><b>This project could not be indexed.</b><br>"
            f"{escape(metadata['error'])}</p>"
            f"<p>{escape(metadata.get('path', ''))}</p>"
        )

    contents = metadata.get("contents", {})
    lines = [f"<h3>{escape(metadata.get('name', 'Project'))}</h3>"]
    lines.append(f"<p>{escape(metadata.get('path', ''))}</p>")
    general = []
    if metadata.get("application_version"):
        general.append(f"Application {escape(metadata['application_version'])}")
    if metadata.get("workspace_mode"):
        general.append(f"Workspace: {escape(metadata['workspace_mode'])}")
    general.append(
        f"Saved: {escape(_date_text(metadata.get('saved_at') or metadata.get('modified_at')))}"
    )
    general.append(f"File size: {escape(_size_text(metadata.get('file_size')))}")
    lines.append("<p>" + "<br>".join(general) + "</p>")

    entries = []
    phantom = contents.get("phantom")
    if phantom:
        description = escape(phantom.get("name", "Phantom"))
        shape = _shape_text(phantom.get("shape"))
        if shape:
            description += f" — {escape(shape)} voxels"
        components = phantom.get("components", [])
        if components:
            description += " — " + escape(", ".join(str(item) for item in components))
        if phantom.get("nucleus"):
            description += f" — {escape(phantom['nucleus'])}"
        entries.append(("Phantom", description))

    for key, label in (("tx_field", "Transmit B1"), ("rx_field", "Receive B1")):
        field = contents.get(key)
        if field:
            description = escape(field.get("name") or label)
            shape = _shape_text(field.get("shape"))
            if shape:
                description += f" — {escape(shape)}"
            entries.append((label, description))

    sequence = contents.get("sequence")
    if sequence:
        parts = []
        if sequence.get("source"):
            parts.append(str(sequence["source"]))
        if sequence.get("duration_s") is not None:
            parts.append(f"{float(sequence['duration_s']) * 1e3:.3g} ms")
        if sequence.get("event_count") is not None:
            parts.append(f"{int(sequence['event_count'])} events")
        event_types = sequence.get("event_types", {})
        event_text = ", ".join(
            f"{event_types.get(key, 0)} {key.upper()}"
            for key in ("rf", "gradient", "adc")
        )
        if event_text:
            parts.append(event_text)
        entries.append(("Sequence", escape(" · ".join(parts))))

    free_result = contents.get("free_mode_result")
    if free_result:
        arrays = free_result.get("array_shapes", {})
        description = (
            ", ".join(f"{key} {_shape_text(shape)}" for key, shape in arrays.items())
            or "Stored simulation data"
        )
        entries.append(("Free-mode result", escape(description)))

    sequence_result = contents.get("sequence_result")
    if sequence_result:
        parts = []
        if sequence_result.get("kind") == "spin-probe":
            parts.extend(
                [
                    f"{int(sequence_result.get('positions', 0))} positions",
                    f"{int(sequence_result.get('frequencies', 0))} frequencies",
                    f"{int(sequence_result.get('time_samples', 0))} time samples",
                ]
            )
        signal_shape = _shape_text(sequence_result.get("signal_shape"))
        magnetization_shape = _shape_text(
            sequence_result.get("final_magnetization_shape")
        )
        if signal_shape:
            parts.append(f"signal {signal_shape}")
        if magnetization_shape:
            parts.append(f"magnetization {magnetization_shape}")
        if sequence_result.get("adc_samples") is not None:
            parts.append(f"{int(sequence_result['adc_samples'])} ADC samples")
        if sequence_result.get("checkpoint_count"):
            parts.append(f"{int(sequence_result['checkpoint_count'])} checkpoints")
        entries.append(
            (
                (
                    "Spin-probe result"
                    if sequence_result.get("kind") == "spin-probe"
                    else "Sequence result"
                ),
                escape(" · ".join(parts) or "Stored result"),
            )
        )

    if entries:
        lines.append("<h4>Contents</h4><ul>")
        lines.extend(
            f"<li><b>{escape(label)}:</b> {description}</li>"
            for label, description in entries
        )
        lines.append("</ul>")
    else:
        lines.append("<p>No stored data objects were found in the manifest.</p>")
    return "".join(lines)


class ProjectExplorerDialog(QDialog):
    """Browse projects from persistent folders without loading their arrays."""

    project_open_requested = pyqtSignal(str)

    def __init__(self, parent=None, *, settings=None, default_folders=()):
        super().__init__(parent)
        self.settings = (
            settings
            if settings is not None
            else QSettings("BlochSimulator", "BlochSimulator")
        )
        self.default_folders = [
            str(Path(folder).expanduser()) for folder in default_folders
        ]
        self.projects = []
        self.setWindowTitle("Project Explorer")
        self.setObjectName("project_explorer")
        self.resize(1100, 680)

        root = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter projects, folders, or contents…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        controls.addWidget(self.filter_edit, 1)
        self.recursive_checkbox = QCheckBox("Include subfolders")
        self.recursive_checkbox.setChecked(
            self.settings.value(SETTINGS_RECURSIVE, True, type=bool)
        )
        self.recursive_checkbox.toggled.connect(self._recursive_changed)
        controls.addWidget(self.recursive_checkbox)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh)
        controls.addWidget(refresh_button)
        root.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal)
        folder_panel = QWidget()
        folder_layout = QVBoxLayout(folder_panel)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.addWidget(QLabel("Project folders"))
        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        folder_layout.addWidget(self.folder_list, 1)
        folder_buttons = QHBoxLayout()
        add_button = QPushButton("Add…")
        add_button.clicked.connect(self._add_folder)
        folder_buttons.addWidget(add_button)
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self._remove_folders)
        folder_buttons.addWidget(remove_button)
        folder_layout.addLayout(folder_buttons)
        splitter.addWidget(folder_panel)

        project_panel = QWidget()
        project_layout = QVBoxLayout(project_panel)
        project_layout.setContentsMargins(0, 0, 0, 0)
        self.project_tree = QTreeWidget()
        self.project_tree.setObjectName("project_explorer_tree")
        self.project_tree.setHeaderLabels(
            ["Project", "Contents", "Workspace", "Modified", "Size", "Folder"]
        )
        self.project_tree.setRootIsDecorated(False)
        self.project_tree.setAlternatingRowColors(True)
        self.project_tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.project_tree.setSortingEnabled(True)
        self.project_tree.itemSelectionChanged.connect(self._selection_changed)
        self.project_tree.itemDoubleClicked.connect(lambda *_: self._open_selected())
        header = self.project_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        project_layout.addWidget(self.project_tree, 2)
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(False)
        self.details.setPlaceholderText("Select a project to inspect its contents.")
        project_layout.addWidget(self.details, 1)
        splitter.addWidget(project_panel)
        splitter.setSizes([250, 850])
        root.addWidget(splitter, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel()
        footer.addWidget(self.status_label, 1)
        self.open_button = QPushButton("Open selected project")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_selected)
        footer.addWidget(self.open_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        footer.addWidget(close_button)
        root.addLayout(footer)

        self._load_folders()
        self.refresh()

    def _load_folders(self):
        stored = self.settings.value(SETTINGS_FOLDERS, [])
        if isinstance(stored, str):
            stored = [stored] if stored else []
        folders = [str(Path(folder).expanduser()) for folder in stored or []]
        if not folders:
            folders = self.default_folders
        for folder in dict.fromkeys(folders):
            self.folder_list.addItem(folder)
        self._save_folders()

    def _folder_paths(self):
        return [
            self.folder_list.item(row).text() for row in range(self.folder_list.count())
        ]

    def _save_folders(self):
        self.settings.setValue(SETTINGS_FOLDERS, self._folder_paths())

    def _add_folder(self):
        start = self._folder_paths()[0] if self._folder_paths() else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Add project folder", start)
        if not folder or folder in self._folder_paths():
            return
        self.folder_list.addItem(folder)
        self._save_folders()
        self.refresh()

    def _remove_folders(self):
        for item in self.folder_list.selectedItems():
            self.folder_list.takeItem(self.folder_list.row(item))
        self._save_folders()
        self.refresh()

    def _recursive_changed(self, checked):
        self.settings.setValue(SETTINGS_RECURSIVE, bool(checked))
        self.refresh()

    def refresh(self):
        selected = self.selected_project_path()
        self.projects = scan_project_folders(
            self._folder_paths(), recursive=self.recursive_checkbox.isChecked()
        )
        self.project_tree.setSortingEnabled(False)
        self.project_tree.clear()
        selected_item = None
        for metadata in self.projects:
            labels = _content_labels(metadata)
            item = QTreeWidgetItem(
                [
                    str(metadata.get("name", "")),
                    (
                        " · ".join(labels)
                        if labels
                        else ("Unreadable" if metadata.get("error") else "Empty")
                    ),
                    str(metadata.get("workspace_mode", "—") or "—"),
                    _date_text(metadata.get("modified_at")),
                    _size_text(metadata.get("file_size")),
                    str(metadata.get("folder", "")),
                ]
            )
            item.setData(0, Qt.UserRole, metadata)
            item.setToolTip(0, str(metadata.get("path", "")))
            self.project_tree.addTopLevelItem(item)
            if metadata.get("path") == selected:
                selected_item = item
        self.project_tree.setSortingEnabled(True)
        self.project_tree.sortItems(3, Qt.DescendingOrder)
        if selected_item is not None:
            self.project_tree.setCurrentItem(selected_item)
        elif self.project_tree.topLevelItemCount():
            self.project_tree.setCurrentItem(self.project_tree.topLevelItem(0))
        else:
            self.details.clear()
        self._apply_filter(self.filter_edit.text())

    def _apply_filter(self, text):
        words = str(text).casefold().split()
        visible = 0
        for row in range(self.project_tree.topLevelItemCount()):
            item = self.project_tree.topLevelItem(row)
            metadata = item.data(0, Qt.UserRole) or {}
            haystack = " ".join(
                [item.text(column) for column in range(self.project_tree.columnCount())]
                + [json.dumps(metadata, sort_keys=True)]
            ).casefold()
            hidden = not all(word in haystack for word in words)
            item.setHidden(hidden)
            visible += not hidden
        current = self.project_tree.currentItem()
        if current is not None and current.isHidden():
            replacement = next(
                (
                    self.project_tree.topLevelItem(row)
                    for row in range(self.project_tree.topLevelItemCount())
                    if not self.project_tree.topLevelItem(row).isHidden()
                ),
                None,
            )
            if replacement is not None:
                self.project_tree.setCurrentItem(replacement)
            else:
                self.project_tree.clearSelection()
                self.project_tree.setCurrentItem(None)
                self.details.clear()
                self.open_button.setEnabled(False)
        elif current is None and visible:
            for row in range(self.project_tree.topLevelItemCount()):
                item = self.project_tree.topLevelItem(row)
                if not item.isHidden():
                    self.project_tree.setCurrentItem(item)
                    break
        self.status_label.setText(
            f"{visible} of {len(self.projects)} projects"
            if words
            else f"{len(self.projects)} projects"
        )

    def _selection_changed(self):
        item = self.project_tree.currentItem()
        metadata = item.data(0, Qt.UserRole) if item is not None else None
        (
            self.details.setHtml(_detail_html(metadata))
            if metadata
            else self.details.clear()
        )
        self.open_button.setEnabled(bool(metadata and not metadata.get("error")))

    def selected_project_path(self):
        item = self.project_tree.currentItem()
        metadata = item.data(0, Qt.UserRole) if item is not None else None
        return str(metadata.get("path", "")) if metadata else ""

    def _open_selected(self):
        path = self.selected_project_path()
        item = self.project_tree.currentItem()
        metadata = item.data(0, Qt.UserRole) if item is not None else None
        if path and metadata and not metadata.get("error"):
            self.project_open_requested.emit(path)
