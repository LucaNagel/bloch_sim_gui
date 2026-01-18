# Developer Guide

This guide explains how to build, package, and release the BlochSimulator application, as well as how to extend it with new pulse sequences and GUI options.

## 1. Environment Setup

The build process uses a dedicated virtual environment (`.venv-packaging`) to ensure a clean build with specific versions of dependencies (like PyInstaller).

To set up the environment (handled automatically by the build script, but good to know):
1.  Ensure you have Python 3 installed.
2.  The build script will create `.venv-packaging` and install dependencies from `requirements.txt`.

**Note:** The `.venv-packaging` directory contains many files but is configured to be ignored by `.gitignore`. You should **not** commit it to the repository.

## 2. Release Workflow (Recommended)

Follow these steps to publish a new version of BlochSimulator. This process is highly automated via GitHub Actions.

### Step 1: Bump Version
Use the included helper script to update version numbers across all files (`pyproject.toml`, `setup.py`, etc.).

```bash
# Replace 1.0.6 with your new version number
python bump_version.py 1.0.6
```

This script will:
*   Update version strings in `pyproject.toml`, `setup.py`, `src/blochsimulator/simulator.py`, `src/blochsimulator/gui.py`, `docs/conf.py`, and `src/blochsimulator/__init__.py`.
*   Print the exact git commands you need to run to commit and tag the release.

### Step 2: Commit and Tag
Run the commands suggested by the `bump_version.py` script.

```bash
git add .
git commit -m "Bump version to 1.0.6"
git tag v1.0.6
```

### Step 3: Push and Automate
Pushing the tag triggers the CI/CD pipelines to automatically build the PyPI package and the standalone applications for macOS, Windows, and Linux.

```bash
git push origin main v1.0.6
```

**What happens automatically:**
1.  **PyPI:** Binary wheels and source distributions are built and uploaded to PyPI.
2.  **GitHub Release:** Standalone executables for all platforms are built.
3.  **Draft Release:** A new "Draft" release is created on GitHub with all binaries attached.

### Step 4: Finalize Release on GitHub
1.  Go to your GitHub Repository page and click on **Releases**.
2.  Find the new **Draft** release.
3.  Edit the release: Add a title (e.g., "v1.0.6") and describe the changes.
4.  Click **Publish release** to make it public.

---

## 3. Manual Build and Packaging (Optional)

If you need to build the application locally for testing, use these scripts.

### Building the Application
```bash
./scripts/build_pyinstaller.sh
```
**What this does:** Creates the virtual environment, installs dependencies, builds Cython extensions, and runs PyInstaller. Outputs to `dist/BlochSimulator.app` (macOS) or `dist/BlochSimulator` (Windows/Linux).

### Packaging for Release
```bash
./scripts/package_for_release.sh
```
**What this does:** Compresses the app into a `.zip` or `.tar.gz`, automatically naming it based on the current Git tag and architecture.

---

## 4. User Installation Instructions

Since the app is not signed with an Apple Developer ID, users will see a "damaged" or "unidentified developer" warning.

**Instructions for macOS:**
1. Download and unzip the application.
2. Move `BlochSimulator.app` to your Applications folder.
3. **Important:** Run this command in Terminal to allow the app to run:
   ```bash
   xattr -cr /Applications/BlochSimulator.app
   ```
4. Double-click to open.

---

## 5. Automated CI/CD (GitHub Actions)

The repository uses two main workflows:

### A. PyPI Release (`publish.yml`)
Triggered on push to `main` (test only) and on tags `v*` (publish).
- **`test`**: Runs `pytest` on Linux.
- **`build_wheels`**: Builds binary wheels for all OSs.
- **`publish`**: Uploads to PyPI (only on tags).

### B. Standalone App Build (`build_standalone.yml`)
Triggered on tags `v*` or manual dispatch.
- **`build`**: Builds standalone executables on macOS, Windows, and Linux.
- **`release`**: Creates a **Draft Release** on GitHub with the binaries.

### Pre-flight Check
To test your code before tagging:
1.  Push to `main` without a tag.
2.  Check the **Actions** tab on GitHub.
3.  If `test`, `build_wheels`, and `build_sdist` pass, it is safe to proceed with the release.

---

## 6. Package Configuration (MANIFEST.in)

The `MANIFEST.in` file tells `setuptools` which non-Python files to include in the **Source Distribution (`sdist`)**. This is critical for users installing from source who need the C/Cython files and RF pulse data.

Ensure any new asset directories are added here to be bundled with the library.

---

## 7. Extending the Simulator

### How to Add a New Pulse Sequence

Adding a new sequence involves updates to both the core simulator logic and the GUI.

**1. Define the Sequence Class (Optional but Recommended)**
In `src/blochsimulator/simulator.py`:
Create a new class inheriting from `PulseSequence`. Implement the `compile()` method to return `(b1, gradients, time)`.

```python
class MyNewSequence(PulseSequence):
    def __init__(self, param1, param2, ...):
        # Initialize parameters
        pass

    def compile(self, dt=1e-6):
        # Generate b1 (complex), gradients (N,3), and time arrays
        return b1, gradients, time
```

**2. Register in GUI (`src/blochsimulator/gui.py`)**

*   **Add to List:** In `SequenceDesigner.init_ui()`, add your sequence name to the `self.sequence_type` ComboBox.
    ```python
    self.sequence_type.addItems([..., "My New Sequence"])
    ```

*   **Define Default Parameters:** In `SequenceDesigner.get_sequence_preset_params()`, add a dictionary for your sequence. This sets default TE, TR, and other values when the user selects your sequence.
    ```python
    "My New Sequence": {
        "te_ms": 15,
        "tr_ms": 100,
        "flip_angle": 45,
    },
    ```

*   **Implement Generation Logic:** In `SequenceDesigner.get_sequence()`, handle the new sequence type instantiation.
    ```python
    elif seq_type == "My New Sequence":
        return MyNewSequence(
            te=te,
            tr=tr,
            param1=...,
        )
    ```

### How to Hide/Show Options per Sequence

You can customize which widgets (e.g., TI spinbox, Echo Count, custom checkboxes) are visible for each sequence type.

**1. Create a Container for New Options (if needed)**
In `SequenceDesigner.init_ui()`, create a `QWidget` or `QGroupBox` to hold your specific controls. Add it to `self.options_container`.

```python
# Create widget
self.my_seq_opts = QWidget()
layout = QHBoxLayout()
self.my_param_spin = QSpinBox()
layout.addWidget(QLabel("My Param:"))
layout.addWidget(self.my_param_spin)
self.my_seq_opts.setLayout(layout)

# Add to main container
self.options_container.addWidget(self.my_seq_opts)

# Hide by default
self.my_seq_opts.setVisible(False)
```

**2. Update Visibility Logic**
In `SequenceDesigner._update_sequence_options()`, add your logic to show or hide the widget based on `seq_type`.

```python
def _update_sequence_options(self):
    seq_type = self.sequence_type.currentText()

    # Toggle visibility
    self.spin_echo_opts.setVisible(seq_type in ("Spin Echo", ...))
    self.my_seq_opts.setVisible(seq_type == "My New Sequence")

    # Update pulse list roles if needed (e.g., Excitation only, or Excitation + Refocusing)
    if seq_type == "My New Sequence":
        roles = ["Excitation", "MyPulse"]
    ...
```

This ensures users only see relevant controls for the active sequence.

---

## 8. Web Simulation Extensions

The web version of the Bloch Simulator runs via Pyodide (WASM) and interacts with the DOM. To extend the web simulation options, you need to modify three main components: the HTML view, the JavaScript controller, and the Python simulation logic.

### 1. Update the HTML View
File: `web/partials/rf_explorer.html` (or create a new partial for a new view)

Add the new input control (e.g., a slider or input box). Ensure it has:
*   An `id` (e.g., `id="my_new_param"`).
*   The class `sim-input` (this automatically triggers the `triggerSimulation` event listener).

```html
<div class="control-group">
    <label for="my_new_param">My Parameter</label>
    <input type="number" id="my_new_param" value="1.0" step="0.1" class="sim-input">
</div>
```

### 2. Update the JavaScript Controller
File: `web/static/js/app.js`

You need to update two functions:
*   **`triggerSimulation()`**: Read the value from your new HTML input and pass it to the Python function.
    ```javascript
    const vals = {
        // ... existing params ...
        myParam: parseFloat(document.getElementById("my_new_param").value)
    };

    // Pass to Python
    pyFunc(vals.t1, vals.t2, vals.duration, vals.freq, vals.type, vals.myParam);
    ```

### 3. Update the Python Logic
File: `web/static/js/app.js` (inside the `runPythonAsync` block)

Update the `update_simulation` Python function signature to accept the new argument and use it in the simulation.

```python
def update_simulation(t1_ms, t2_ms, duration_ms, freq_offset_hz, pulse_type, my_param):
    # Use my_param in your simulation logic
    # ...
```

### Testing Changes
Run the local dev server to test your changes without deploying:
```bash
python scripts/dev_server.py
```
This builds the site to `_dev/` and serves it at `http://localhost:8000`.

### Running Real Physics Locally (Docker)
By default, the dev server runs in "Mock Mode" because the C-extension (`bloch_core_modified.c`) isn't compiled for the browser. To run the **real physics engine** locally, you can use Docker to compile the WebAssembly wheel.

1.  **Prerequisites**: Install Docker Desktop.
2.  **Build the WASM Wheel**: Run this command from the project root:
    ```bash
    docker run --rm -v $(pwd):/src -w /src python:3.11 /bin/bash -c "
      pip install pyodide-build &&
      export EMSCRIPTEN=1 &&
      export CFLAGS='-g0 -O3' &&
      pyodide build
    "
    ```
    *Note: This pulls a standard Python image and installs the build tools. The first run takes a few minutes.*

3.  **Start the Dev Server**:
    ```bash
    python scripts/dev_server.py
    ```
    The script will automatically detect the new `.whl` file in `dist/` and switch to "Real Physics Mode". You will see "Installing bloch_simulator... Ready" in the status bar instead of "Dev mode".
