"""
test_dependencies.py - Verify all critical dependencies are installed

This module performs a comprehensive check of the runtime environment.
It ensures that all libraries required for the full feature set (GUI, export,
simulation) are present and importable.
"""

import unittest
import importlib

# List of critical modules that must be present.
# Format: (import_name, description)
# Note: import_name may differ from pip package name (e.g. 'OpenGL' vs 'PyOpenGL')
REQUIRED_MODULES = [
    ("numpy", "Core numerical library"),
    ("scipy", "Scientific algorithms"),
    ("matplotlib", "Plotting support"),
    ("PyQt5", "GUI framework"),
    ("pyqtgraph", "Fast plotting library"),
    ("OpenGL", "3D visualization (PyOpenGL)"),
    ("imageio", "Animation export"),
    ("imageio_ffmpeg", "MP4 export support"),
    ("h5py", "HDF5 data export/import"),
    ("nbformat", "Jupyter notebook export"),
    ("cython", "C-extension compilation support"),
]


class TestDependencies(unittest.TestCase):
    """Test that the environment is correctly set up with all dependencies."""

    def test_required_modules(self):
        """
        Iterate through all critical modules and verify they can be imported.

        This prevents runtime crashes or silent failures (like missing export options)
        due to incomplete environments.
        """
        missing_modules = []

        for module_name, description in REQUIRED_MODULES:
            with self.subTest(module=module_name):
                try:
                    importlib.import_module(module_name)
                except ImportError:
                    missing_modules.append(f"{module_name} ({description})")

        if missing_modules:
            failure_msg = (
                f"\n\nMissing {len(missing_modules)} required dependencies:\n"
                + "\n".join(f"  - {m}" for m in missing_modules)
                + "\nPlease run: pip install -e ."
            )
            self.fail(failure_msg)


if __name__ == "__main__":
    unittest.main()
