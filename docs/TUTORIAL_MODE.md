# Tutorial Mode Implementation

This document outlines the design and implementation of the interactive Tutorial Mode in the Bloch Simulator GUI.

## 1. Objective
The goal is to provide a "Guided User Experience" where experts can record a sequence of actions (clicks, tab switches, parameter changes) and learners can follow them. Unlike a static video, this system is interactive: it highlights the actual UI elements and allows the user to either perform the action themselves or skip through steps using navigation controls.

## 2. Architecture

### A. Tutorial Manager (`tutorial_manager.py`)
The core logic resides in the `TutorialManager` class. It uses a **Qt Event Filter** to monitor the application state without interfering with the existing signal/slot logic.

- **Recording:** Intercepts `MouseButtonRelease` events. It "crawls up" the widget tree from the click point to find the nearest parent with a unique `objectName`. It stores the widget ID, action type (click/tab), and metadata (like tab index).
- **Playback:** Iterates through the recorded steps. It locates widgets by name using `findChild` and applies a high-visibility CSS stylesheet (Gold border/background).
- **State Machine:** Tracks the `current_step_idx`. It automatically advances when the correct widget is clicked or when the user clicks "Next" on the overlay.

### B. Tutorial Overlay (`tutorial_overlay.py`)
A floating, translucent window that stays on top of the main application.

- **Purpose:** Provides context and navigation.
- **Controls:**
    - **Step Label:** Shows progress (e.g., "Step 3/10").
    - **Instruction:** Dynamic text explaining what to do (e.g., "Click 'Run Simulation'").
    - **Prev/Next:** Allows manual navigation through the tutorial.
    - **Stop:** Exits the tutorial mode and clears all highlights.

### C. Widget Identification
For the system to be robust, every interactive element must have a unique ID.
- **Implementation:** Systematically assigned `objectName` to all buttons, combo boxes, sliders, and spin boxes across the codebase (`MainWindow`, `RFPulseDesigner`, `SequenceDesigner`, etc.).
- **Consistency:** Used prefixes like `rf_tab_` vs `rf_compact_` to differentiate between similar widgets in different views.

## 3. Key Features

- **Robust Recording:** Handles clicks on sub-elements (like the arrows of a `QSpinBox`) by identifying the logical parent widget.
- **Tab Support:** Specifically handles `QTabWidget` by recording the tab index and highlighting the tab bar.
- **Menu Support:** Can record actions triggered via `QAction` (top menus).
- **Non-Destructive Styling:** Stores the original stylesheet of a widget before highlighting it and restores it exactly when the step is finished or the tutorial stops.
- **JSON Persistence:** Tutorials are saved as human-readable JSON files in the `tutorials/` directory.

## 4. Usage

### Recording a Tutorial
1. Go to **Tutorials -> Record New Tutorial...**
2. Enter a name (e.g., "Basic Spin Echo").
3. Perform the actions in the GUI.
4. Go to **Tutorials -> Stop Recording**. The file is saved to `tutorials/<name>.json`.

### Playing a Tutorial
1. Go to **Tutorials -> Load Tutorial...**
2. Select the JSON file.
3. The **Tutorial Overlay** will appear, and the first widget will be highlighted in gold.
4. Either perform the action or click **Next** to skip.

## 5. Files Created/Modified
- `src/blochsimulator/ui/tutorial_manager.py`: Core logic.
- `src/blochsimulator/ui/tutorial_overlay.py`: Navigation UI.
- `src/blochsimulator/ui/main_window.py`: Integration, menu, and widget naming.
- `src/blochsimulator/ui/rf_pulse_designer.py`: Widget naming.
- `src/blochsimulator/ui/sequence_designer.py`: Widget naming.
- `src/blochsimulator/ui/tissue_parameters.py`: Widget naming.
- `src/blochsimulator/ui/controls.py`: Widget naming.
- `src/blochsimulator/ui/magnetization_viewer.py`: Widget naming.
- `src/blochsimulator/ui/dialogs.py`: Widget naming.
- `tests/test_gui_smoke.py`: Verification test.
