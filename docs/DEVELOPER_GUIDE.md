# Developer Guide

This guide explains how to build, package, and release the BlochSimulator application, as well as how to extend it with new pulse sequences and GUI options.

## 1. Environment Setup

The source GUI and PyInstaller build share the Python 3.12 runtime declared in
`.python-version` and the dedicated `.venv-packaging` environment. This keeps
their Python ABI and GUI dependencies aligned.

To set up the environment (handled automatically by the build script, but good to know):
1.  Ensure Python 3.12 is installed.
2.  Run `./scripts/run_gui.sh` for source development or
    `./scripts/build_pyinstaller.sh` for an app build. Both commands create or
    reuse `.venv-packaging`, install the current repository in editable mode,
    and reject an environment from another Python minor version.
3.  If necessary, select the interpreter explicitly with
    `BLOCH_PYTHON=/path/to/python3.12`.

**Note:** The `.venv-packaging` directory contains many files but is configured to be ignored by `.gitignore`. You should **not** commit it to the repository.

## 2. Release Workflow (Recommended)

Follow these steps to publish a new version of BlochSimulator. This process is highly automated via GitHub Actions.

### Step 1: Bump Version
Use the included helper script to update version numbers across all files (`pyproject.toml`, `setup.py`, etc.).

```bash
# Replace 2.6.5 with your new version number
python bump_version.py 2.6.5
```

This script will:
*   Update version strings in `pyproject.toml`, `setup.py`, `src/blochsimulator/simulator.py`, `src/blochsimulator/gui.py`, `docs/conf.py`, and `src/blochsimulator/__init__.py`.
*   Print the exact git commands you need to run to commit and tag the release.

### Step 2: Commit and Push the Version Bump

Commit the updated version files and push the commit to the appropriate release
branch. For a stable release, use `main`:

```bash
git add .
git commit -m "Bump version to 2.6.5"
git push origin main
```

### Step 3: Create and Push the Release Tag

Wait for the push-triggered `Tests` workflow to succeed on the exact release
commit. Only then create the tag on that commit and push it explicitly:

```bash
git tag v2.6.5
git push origin v2.6.5
```

Pushing the branch does **not** push a newly created tag automatically. Verify
that `git status` shows the intended branch and that `git show v2.6.5` points to
the version-bump commit before pushing the tag. If you are preparing a
pre-release, use the corresponding development tag (for example,
`v2.6.5.dev1`) and the `pre-release` branch instead.

### Release Paths

#### Pre-release validation path

Use this path to exercise the complete GitHub Actions release workflow before
merging to `main`. Development tags create a GitHub pre-release with the built
wheel and application artifacts, but are never uploaded to PyPI.

1. Start from the `pre-release` branch and update the project to the intended
   final version (for example, `2.3.0`).
   ```bash
   git switch pre-release
   git pull --ff-only origin pre-release
   python bump_version.py 2.3.0
   git add pyproject.toml setup.py src/blochsimulator docs/conf.py web index.html
   git commit -m "Bump version to 2.3.0"
   git push origin pre-release
   ```
2. Wait for the `Tests` workflow triggered by this push to finish successfully.
   The successful `Full test suite` job must belong to this exact commit.
3. Create and push a development tag such as `v2.3.0.dev1`.
   ```bash
   git tag v2.3.0.dev1
   git push origin v2.3.0.dev1
   ```
   The release workflow requires this tagged commit to be reachable from
   `pre-release` and to have a successful push-triggered `Tests` run on that
   branch. It temporarily sets the package version to `2.3.0.dev1` while
   building, creates a GitHub pre-release, and does not publish to PyPI.
4. Inspect the GitHub Actions run and the generated GitHub pre-release assets.
   If another validation build is needed, push a later commit, wait for its
   tests to pass, and use the next development tag (for example,
   `v2.3.0.dev2`). Do not reuse a published tag.

#### Final stable-release path

1. Merge the tested `pre-release` commit into `main`, then wait for the
   `Tests` workflow triggered by the push to `main` to finish successfully.
2. Tag that exact tested `main` commit with the final version and push the tag.
   ```bash
   git switch main
   git pull --ff-only origin main
   git tag v2.3.0
   git push origin v2.3.0
   ```
   A stable tag is accepted only when its commit is reachable from `main` and
   has a successful push-triggered `Tests` workflow on `main`.
3. The workflow builds the PyPI distributions and application artifacts,
   uploads the Python distributions to PyPI, and creates the stable GitHub
   Release with the application artifacts.

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

### Choosing Subvoxel Sampling for Imported Pulseq Sequences

Imported `.seq` files must use gradient-waveform spoiling because they do not
contain the simulator's explicit ideal-crusher markers. Choose the subvoxel
counts with a convergence study over the complete RF/ADC gradient-moment train:
increase the X, Y, and Z counts only on axes with intravoxel phase dispersion
until the signal change or the sampled-versus-continuous coherence error is
below the chosen tolerance (the GUI recommendation uses 1%). The cost scales
with the product $N_x N_y N_z$, so the optimal grid is the smallest converged
one, not the largest affordable grid. Prefer the regular midpoint grid when its
complete train has been checked, because it is symmetric and usually reaches a
given quadrature accuracy with fewer spins. Use deterministic stratified points
when the full train cannot be checked or a regular grid shows artificial exact
rephasing; they reproducibly break short grid recurrences, but generally need
more spins and still require a convergence check.

### Mouse Perfusion: Current Transport Semantics

The mouse perfusion phantom currently implements a spatially delayed source,
not advection of existing spins. Its concentration source is

$$
u(t,\mathbf r)=D(\mathbf r)q\left(t-\tau(\mathbf r)\right),
$$

where $q(t)$ is the dose-normalized injection curve, $D(\mathbf r)$ is the
delivery map, and $\tau(\mathbf r)$ is the voxelwise arrival delay. The arrival
map follows the directed route from the tail vein through the right heart,
lungs, left heart, arteries, and organ beds. This makes the displayed bolus and
the simulated source appear at different positions at different times.

The source is more than a post-hoc signal-amplitude scaling: it adds local
concentration and longitudinal magnetization during sequence simulation. The
inflow concentration and its polarization are tracked separately. Once added,
however, a spin state remains assigned to that voxel. It experiences the local
RF field, gradient phase, off-resonance, $T_1/T_2$ relaxation, and configured
chemical conversion, but it is not transferred to a downstream voxel.

For example, consider a slice-selective RF pulse through the center of the
mouse:

1. Pyruvate already present in that slice is rotated or saturated by the pulse.
2. Its resulting $(M_x,M_y,M_z)$ state remains in the same simulation voxel.
3. Later inflow can add fresh polarized Pyruvate to that voxel and partially
   replenish its signal.
4. A downstream organ receives its own delayed source. It does not receive the
   RF history of the Pyruvate that was excited in the central slice.

The renal $k_{PL}$ and hepatic $k_{PA}$ maps are active reaction terms. They
convert local Pyruvate into Lactate and Alanine, respectively, and precursor
loss includes both product channels where both are present. These are local
reactions and do not imply spatial transport.

Keep the animation model separate from sequence physics. The animated
**Injected perfusion** concentration convolves the delayed source with a local
mono-exponential clearance to make accumulation and wash-out visible. This
clearance is currently preview-only. The Bloch sequence solver has no
perfusion-dependent outflow, vascular exchange, recirculation, or transport of
previously excited magnetization. Breathing is likewise stored as a preview
displacement field and is not yet coupled to gradient phase.

The relevant implementation boundaries are:

- `src/blochsimulator/mouse_phantom.py` builds the anatomy, vascular transit
  map, delayed delivery, preview concentration, and organ curves.
- `src/blochsimulator/dynamic_phantom.py` integrates the local inflow, Bloch
  evolution, relaxation, and reaction terms.
- `arrival_delay_map_s` changes when source material is created in a voxel; it
  does not move a previously created magnetization state.

### Mouse Perfusion: Recommended Transport Roadmap

The next solver should transport concentration and the complete magnetization
state rather than only scheduling independent local sources. A staged
implementation keeps validation and computational cost manageable.

#### Stage 1: Directed vascular-graph transport

Start with the existing directed vessel graph instead of immediately solving a
full 3D flow field.

1. Divide every vessel edge into one-dimensional transport cells with explicit
   volume, velocity, flow rate, and transit-time dispersion.
2. Store concentration and $(M_x,M_y,M_z)$ for every compound in every
   transport cell.
3. Make downstream inflow equal upstream outflow, including the transported RF
   and relaxation history. Do not recreate downstream spins from the original
   injection curve.
4. Add blood and tissue compartments at organ nodes. Exchange should conserve
   compound amount, while $k_{PL}$ and $k_{PA}$ remain reaction terms within
   the selected compartments.
5. Add explicit clearance and venous return before introducing recirculation.
6. Apply RF and gradient evolution according to the current spatial position
   of each transport cell. A saturated bolus passing through a selected slice
   must therefore remain saturated downstream, subject to relaxation and new
   mixing.

This stage can use operator splitting: perform local Bloch/reaction evolution,
then a conservative transport/exchange step. It should expose a transport-model
interface while retaining the existing delayed-source model for backwards
compatibility and fast previews.

#### Stage 2: Spatial advection-reaction Bloch model

For capillary or tissue-scale transport, evolve each pool according to an
advection-reaction Bloch equation such as

$$
\frac{\partial \mathbf M_p}{\partial t}
+\nabla\cdot\left(\mathbf v_p\mathbf M_p\right)
=
\mathcal B_p(\mathbf M)
+\mathcal R_p(\mathbf M,C)
+\mathbf S_p
-\mathbf W_p,
$$

where $\mathcal B_p$ contains RF, gradient phase, off-resonance, and
relaxation; $\mathcal R_p$ contains metabolic exchange; $\mathbf S_p$ is true
external injection; and $\mathbf W_p$ represents clearance or compartment
exchange. Concentration must obey the corresponding conservative transport and
reaction equation.

A finite-volume method is preferable when strict mass conservation is the
priority. A semi-Lagrangian method is easier to stabilize for large time steps
but needs an explicit conservation correction. A Lagrangian moving-spin model
is another useful option for resolved vessels and naturally carries RF and
phase history, but it requires careful particle-to-voxel interpolation and
noise control. Whichever representation is selected, the transport time step
must be validated independently of the Bloch integration time step.

#### Stage 3: Motion and higher-order physiology

After transport is validated, add respiratory deformation of anatomy,
velocity, and magnetization together. Gradient phase must use the time-varying
physical spin position. Later extensions can add pulsatility, portal-hepatic
circulation, renal filtration, recirculation, heterogeneous capillary transit,
and compound-specific permeability or relaxivity.

#### Required validation tests

At minimum, the transport implementation should include:

- dose and concentration conservation with relaxation and reactions disabled;
- a known plug-flow or one-dimensional advection solution;
- a transit-time test on every vascular branch;
- an RF-tagging test in which a saturated slice produces a delayed downstream
  signal reduction;
- recovery of the current local model when velocity and exchange are disabled;
- non-negative concentrations under the supported time-step limits;
- precursor/product balance for $k_{PL}$ and $k_{PA}$;
- agreement between reference and optimized kernels.

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
