# -*- mode: python -*-
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# project_root is the directory containing this spec file
project_root = Path(os.getcwd())

# Add src to pathex so PyInstaller can find the package
pathex = [str(project_root), str(project_root / "src")]

# Data files bundled into the app
datas = []
# Collect data files for rfc3987_syntax (needed by jsonschema/nbformat)
datas += collect_data_files('rfc3987_syntax')

rf_dir = project_root / "rfpulses"
if rf_dir.exists():
    # Place rfpulses inside blochsimulator package so pulse_loader.py finds it relative to __file__
    datas.append((str(rf_dir), "blochsimulator/rfpulses"))

# Native binaries
binaries = []
# Look for compiled extension in src/blochsimulator
# We place it in 'blochsimulator' folder in the bundle to match package structure
for ext in (project_root / "src" / "blochsimulator").glob("blochsimulator_cy.*"):
    binaries.append((str(ext), "blochsimulator"))

# Common OpenMP runtime locations (macOS and Linux)
omp_candidates = [
    Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
    Path("/usr/local/opt/libomp/lib/libomp.dylib"),
    Path("/usr/lib/libgomp.so.1"),
    Path("/usr/local/lib/libgomp.so.1"),
]
for omp in omp_candidates:
    if omp.exists():
        binaries.append((str(omp), "."))

# Hidden imports needed by PyQt5, pyqtgraph OpenGL, and image exporters
hiddenimports = [
    "PyQt5.sip",
    "pyqtgraph.opengl",
    "imageio_ffmpeg",
    "blochsimulator.blochsimulator_cy", # Ensure Cython module is found
]
hiddenimports += collect_submodules("OpenGL.platform")

a = Analysis(
    ["bloch_gui.py"],
    pathex=pathex,
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="BlochSimulator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name='BlochSimulator.app',
    icon=None,
    bundle_identifier='com.lucanagel.blochsimulator',
)
        