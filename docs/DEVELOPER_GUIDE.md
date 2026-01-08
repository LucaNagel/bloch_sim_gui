# Developer Guide

This guide explains how to build, package, and release the BlochSimulator application.

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