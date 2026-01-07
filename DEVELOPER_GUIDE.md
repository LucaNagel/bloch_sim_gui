# Developer Guide

This guide explains how to build, package, and release the BlochSimulator application for macOS.

## 1. Environment Setup

The build process uses a dedicated virtual environment (`.venv-packaging`) to ensure a clean build with specific versions of dependencies (like PyInstaller).

To set up the environment (handled automatically by the build script, but good to know):
1.  Ensure you have Python 3 installed.
2.  The build script will create `.venv-packaging` and install dependencies from `requirements.txt`.

**Note:** The `.venv-packaging` directory contains many files but is configured to be ignored by `.gitignore`. You should **not** commit it to the repository.

## 2. Building the Application

To build the standalone `.app` bundle, run the build script:

```bash
./scripts/build_pyinstaller.sh
```

**What this does:**
1.  Creates/activates the `.venv-packaging` virtual environment.
2.  Installs project requirements and `pyinstaller`.
3.  Builds the Cython extensions in-place.
4.  Runs PyInstaller using `bloch_gui.spec`.
5.  Outputs the application to `dist/BlochSimulator.app`.

## 3. Packaging for Release

We distribute the application as a zipped file via GitHub Releases. This keeps the git repository small by not committing large binary files.

To package the built application, run:

```bash
./scripts/package_for_release.sh
```

**What this does:**
1.  Checks if `dist/BlochSimulator.app` exists.
2.  **Versioning:** It automatically detects the current Git tag (e.g., `v1.0.0`) to name the file. If no tag is present, it uses the commit hash.
3.  Creates a zip file named `BlochSimulator_macOS_arm64_<version>.zip` in the `dist/` directory.
4.  **Symbolic Links:** It uses `zip -y` to preserve symbolic links inside the `.app` bundle, which is critical for macOS applications to function correctly.

## 4. Release Workflow

Follow these steps to publish a new version of BlochSimulator.

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
Run the commands suggested by the `bump_version.py` script. For example:

```bash
git add .
git commit -m "Bump version to 1.0.6"
git tag v1.0.6
git push origin main v1.0.6
```

### Step 3: Build and Package
Run the build and package scripts:

```bash
# 1. Build the app
./scripts/build_pyinstaller.sh

# 2. Package into a zip
./scripts/package_for_release.sh
```

You should now see a file like `dist/BlochSimulator_macOS_arm64_v1.0.6.zip`.

### Step 4: Upload to GitHub
1.  Go to your GitHub Repository page.
2.  Click on **Releases** in the sidebar.
3.  Click **Draft a new release**.
4.  Select the tag you just pushed (`v1.0.6`).
5.  Title the release (e.g., "v1.0.6").
6.  Add a description of the changes.
7.  **Attach Binaries**: Drag and drop the `.zip` file from your `dist/` folder into the "Attach binaries" box.
8.  Click **Publish release**.

## 5. User Installation Instructions

Since the app is not signed with an Apple Developer ID, users will see a "damaged" or "unidentified developer" warning. You must provide these instructions to them (e.g., in the Release description or README):

> **Installation Instructions for macOS:**
>
> 1. Download and unzip the application.
> 2. Move `BlochSimulator.app` to your Applications folder.
> 3. **Important:** Run the following command in Terminal to allow the app to run (fixes the "damaged" error):
>    ```bash
>    xattr -cr /Applications/BlochSimulator.app
>    ```
> 4. Double-click to open.

## 6. Automated CI/CD & PyPI Release

The repository is configured with GitHub Actions (`.github/workflows/publish.yml`) to automatically test, build, and publish the Python package to PyPI.

### Workflow Structure
The workflow consists of four main jobs:
1.  **`test`**: Runs unit tests (`pytest`) on Ubuntu.
2.  **`build_wheels`**: Builds binary wheels for macOS, Windows, and Linux.
3.  **`build_sdist`**: Builds the source distribution (`.tar.gz`).
4.  **`publish`**: Uploads the artifacts to PyPI. **This job only runs on tags.**

### How to Test Before Releasing (Pre-flight Check)

To ensure your code passes all tests and builds successfully **before** you assign a new version number (and potentially "burn" it on a failed build), follow this workflow:

1.  **Commit and Push to `main`**:
    Simply push your changes to the `main` branch *without* a tag.
    ```bash
    git add .
    git commit -m "Fix bug in simulation core"
    git push origin main
    ```

2.  **Check GitHub Actions**:
    *   Go to the **Actions** tab in your GitHub repository.
    *   You will see a workflow run for your latest commit.
    *   Watch the `test`, `build_wheels`, and `build_sdist` jobs.

3.  **Verify Success**:
    *   If these jobs **Pass (Green)**: Your code is safe to release. The `publish` job will be marked as "Skipped" (Grey) because there was no tag.
    *   If any job **Fails (Red)**: Fix the issue, push new commits, and wait for the check to pass. You haven't used up a version number yet!

4.  **Proceed to Release**:
    Only *after* the build on `main` is green, proceed with the [Release Workflow](#4-release-workflow) (Bump Version -> Tag -> Push Tag).
    *   When you push the tag, a *new* workflow run will start.
    *   Since the code is identical to what just passed (plus the version bump), it is highly likely to pass again and successfully run the `publish` step.