# -*- mode: python -*-
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules


block_cipher = None

project_root = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# Data files bundled into the app
datas = []
rf_dir = project_root / "rfpulses"
if rf_dir.exists():
    datas.append((str(rf_dir), "rfpulses"))

# Native binaries to keep alongside the app
binaries = []
for ext in project_root.glob("bloch_simulator_cy.*"):
    binaries.append((str(ext), "."))

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
]
hiddenimports += collect_submodules("OpenGL.platform")

a = Analysis(
    ["bloch_gui.py"],
    pathex=[str(project_root)],
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
