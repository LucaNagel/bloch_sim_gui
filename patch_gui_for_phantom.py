#!/usr/bin/env python
"""
patch_gui_for_phantom.py - Automatically patch bloch_gui.py to add phantom support

Usage:
    python patch_gui_for_phantom.py [path_to_bloch_gui.py]

This script will:
1. Add phantom imports
2. Add the PhantomWidget tab to the GUI
3. Create a backup of the original file

Author: Luca Nagel
Date: 2024/2025
"""

import sys
import re
import shutil
from pathlib import Path


def patch_gui_file(gui_path: str) -> bool:
    """
    Patch bloch_gui.py to add phantom simulation support.
    
    Parameters
    ----------
    gui_path : str
        Path to bloch_gui.py
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    gui_path = Path(gui_path)
    
    if not gui_path.exists():
        print(f"Error: File not found: {gui_path}")
        return False
    
    # Read the file
    print(f"Reading {gui_path}...")
    content = gui_path.read_text(encoding='utf-8')
    
    # Check if already patched
    if 'PHANTOM_AVAILABLE' in content:
        print("File appears to already be patched (PHANTOM_AVAILABLE found)")
        return True
    
    # Create backup
    backup_path = gui_path.with_suffix('.py.backup')
    print(f"Creating backup: {backup_path}")
    shutil.copy(gui_path, backup_path)
    
    # Patch 1: Add imports after visualization_export imports
    import_pattern = r"(from visualization_export import \([^)]+\))"
    import_replacement = r'''\1

# Import phantom module for 2D/3D phantom simulation
try:
    from phantom import Phantom, PhantomFactory
    from phantom_widget import PhantomWidget
    PHANTOM_AVAILABLE = True
except ImportError:
    PHANTOM_AVAILABLE = False
    print("Phantom module not available - phantom tab will be disabled")'''
    
    content, n1 = re.subn(import_pattern, import_replacement, content, count=1, flags=re.DOTALL)
    if n1 == 0:
        print("Warning: Could not find visualization_export import block")
        # Try alternative insertion point
        alt_pattern = r'(import pyqtgraph\.opengl as gl\n)'
        alt_replacement = r'''\1
# Import phantom module for 2D/3D phantom simulation
try:
    from phantom import Phantom, PhantomFactory
    from phantom_widget import PhantomWidget
    PHANTOM_AVAILABLE = True
except ImportError:
    PHANTOM_AVAILABLE = False
    print("Phantom module not available - phantom tab will be disabled")

'''
        content, n1 = re.subn(alt_pattern, alt_replacement, content, count=1)
        if n1 == 0:
            print("Error: Could not find suitable import insertion point")
            return False
    
    print(f"  ✓ Added phantom imports")
    
    # Patch 2: Add PhantomWidget tab after Spatial tab
    # Look for: self.tab_widget.addTab(spatial_container, "Spatial")
    spatial_tab_pattern = r'(self\.tab_widget\.addTab\(spatial_container,\s*["\']Spatial["\']\))'
    spatial_tab_replacement = r'''\1

        # === PHANTOM TAB (2D/3D Imaging) ===
        if PHANTOM_AVAILABLE:
            self.phantom_widget = PhantomWidget(
                parent=self,
                log_callback=self.log_message
            )
            self.tab_widget.addTab(self.phantom_widget, "Phantom")
        else:
            self.phantom_widget = None'''
    
    content, n2 = re.subn(spatial_tab_pattern, spatial_tab_replacement, content, count=1)
    if n2 == 0:
        print("Warning: Could not find Spatial tab insertion point")
        # Try to find any tab_widget.addTab and add after the last one before right_layout
        alt_pattern2 = r'(self\.tab_widget\.addTab\([^)]+\))\n(\s*\n\s*# Share spatial time lines|\s*\n\s*right_layout\.addWidget\(self\.tab_widget\))'
        alt_replacement2 = r'''\1

        # === PHANTOM TAB (2D/3D Imaging) ===
        if PHANTOM_AVAILABLE:
            self.phantom_widget = PhantomWidget(
                parent=self,
                log_callback=self.log_message
            )
            self.tab_widget.addTab(self.phantom_widget, "Phantom")
        else:
            self.phantom_widget = None
\2'''
        content, n2 = re.subn(alt_pattern2, alt_replacement2, content, count=1)
        if n2 == 0:
            print("Error: Could not find tab widget insertion point")
            print("  You may need to manually add the phantom tab code")
            # Still continue - the imports are useful
    
    if n2 > 0:
        print(f"  ✓ Added PhantomWidget tab")
    
    # Write the patched file
    print(f"Writing patched file: {gui_path}")
    gui_path.write_text(content, encoding='utf-8')
    
    print("\n✓ Patching complete!")
    print(f"\nBackup saved as: {backup_path}")
    print("\nMake sure you have these files in the same directory:")
    print("  - phantom.py")
    print("  - phantom_widget.py")
    print("  - bloch_simulator.py (with simulate_phantom method)")
    print("  - bloch_simulator_cy.so (compiled Cython extension)")
    
    return True


def verify_dependencies():
    """Check if required files exist."""
    required = ['phantom.py', 'phantom_widget.py']
    missing = []
    for f in required:
        if not Path(f).exists():
            missing.append(f)
    
    if missing:
        print(f"\nWarning: Missing files: {', '.join(missing)}")
        print("The GUI will load but the Phantom tab will be disabled.")
        return False
    return True


def main():
    if len(sys.argv) < 2:
        # Look for bloch_gui.py in current directory
        default_path = Path("bloch_gui.py")
        if default_path.exists():
            gui_path = str(default_path)
        else:
            print("Usage: python patch_gui_for_phantom.py [path_to_bloch_gui.py]")
            print("\nNo bloch_gui.py found in current directory.")
            sys.exit(1)
    else:
        gui_path = sys.argv[1]
    
    print("=" * 60)
    print("Phantom GUI Integration Patcher")
    print("=" * 60)
    print()
    
    verify_dependencies()
    
    success = patch_gui_file(gui_path)
    
    if success:
        print("\n" + "=" * 60)
        print("To test, run: python bloch_gui.py")
        print("Look for the '🔬 Phantom' tab in the visualization area")
        print("=" * 60)
    else:
        print("\nPatching failed. Please apply changes manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
