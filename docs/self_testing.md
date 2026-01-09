# Testing the Bloch Simulator Package (from PyPI)

Follow these steps to test the package installation directly from PyPI in a clean environment. This verifies exactly what a new user experiences.

## 1. Create a Fresh Virtual Environment

Open your terminal in the project root directory (or anywhere else) and run:

```bash
# Create a virtual environment named 'venv_test_pypi'
python3 -m venv venv_test_pypi
```

## 2. Activate the Environment

**macOS / Linux:**
```bash
source venv_test_pypi/bin/activate
```

**Windows (Command Prompt):**
```cmd
venv_test_pypi\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.\venv_test_pypi\Scripts\Activate.ps1
```

*(You will see `(venv_test_pypi)` appear at the start of your command prompt)*

## 3. Install from PyPI

Install the package directly from the Python Package Index:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install blochsimulator
pip install blochsimulator
```

*Note: If you just uploaded it, it might take a minute to become available. If pip can't find it, wait a moment and try again.*

## 4. Verification

### A. Test Import
Verify that the package imports correctly. This checks if the pre-built wheels (if you uploaded them) or the source distribution works on your machine.

```bash
python -c "from blochsimulator import BlochSimulator; print('✓ Import successful!')"
```

### B. Run the GUI
Launch the GUI using the installed command:

```bash
bloch-gui
```

### C. Run an Example Script
If you are in the project directory, you can run the examples using the installed package:

```bash
python examples/run_bloch_sim.py --flip 45 --duration_ms 2.0
```

## 5. Cleanup

When you are done testing:

1.  **Deactivate** the virtual environment:
    ```bash
    deactivate
    ```

2.  **Delete** the environment folder:
    ```bash
    rm -rf venv_test_pypi
    ```
