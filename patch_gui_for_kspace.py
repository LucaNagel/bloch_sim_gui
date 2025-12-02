"""
patch_gui_for_kspace.py - Patch to add K-Space tab to Bloch Simulator GUI

This script automatically patches your bloch_gui.py to add the K-Space simulation tab.

Usage:
    python patch_gui_for_kspace.py

Or manually apply the changes shown below.

Author: Luca Nagel
Date: 2025
"""

import re
from pathlib import Path


def patch_bloch_gui(gui_path: str = "bloch_gui.py", backup: bool = True):
    """
    Patch bloch_gui.py to add K-Space simulation tab.
    
    Parameters
    ----------
    gui_path : str
        Path to bloch_gui.py
    backup : bool
        Create backup before patching
    """
    gui_path = Path(gui_path)
    
    if not gui_path.exists():
        print(f"Error: {gui_path} not found")
        return False
    
    # Read the file
    content = gui_path.read_text()
    
    # Create backup
    if backup:
        backup_path = gui_path.with_suffix('.py.bak')
        backup_path.write_text(content)
        print(f"Backup created: {backup_path}")
    
    # Check if already patched
    if 'kspace_widget' in content.lower() or 'KSpaceWidget' in content:
        print("File appears to already be patched for K-Space")
        return False
    
    # =========================================================================
    # PATCH 1: Add import statement
    # =========================================================================
    # Find the phantom import block and add k-space import after it
    
    phantom_import = """# Import phantom module for 2D/3D phantom simulation
try:
    from phantom import Phantom, PhantomFactory
    from phantom_widget import PhantomWidget
    PHANTOM_AVAILABLE = True
except ImportError:
    PHANTOM_AVAILABLE = False
    print("Phantom module not available - phantom tab will be disabled")"""
    
    kspace_import = """# Import phantom module for 2D/3D phantom simulation
try:
    from phantom import Phantom, PhantomFactory
    from phantom_widget import PhantomWidget
    PHANTOM_AVAILABLE = True
except ImportError:
    PHANTOM_AVAILABLE = False
    print("Phantom module not available - phantom tab will be disabled")

# Import k-space simulation widget
try:
    from kspace_widget import KSpaceWidget
    from kspace_simulator import KSpaceSimulator, EddyCurrentModel, EPIParameters, CSIParameters
    KSPACE_AVAILABLE = True
except ImportError:
    KSPACE_AVAILABLE = False
    print("K-Space module not available - k-space tab will be disabled")"""
    
    if phantom_import in content:
        content = content.replace(phantom_import, kspace_import)
        print("✓ Added K-Space import statements")
    else:
        print("⚠ Could not find phantom import block - adding import at top")
        # Add after other imports
        import_line = "\n# Import k-space simulation widget\ntry:\n    from kspace_widget import KSpaceWidget\n    KSPACE_AVAILABLE = True\nexcept ImportError:\n    KSPACE_AVAILABLE = False\n\n"
        
        # Find a good place to insert (after phantom import or other imports)
        insert_pos = content.find("PHANTOM_AVAILABLE")
        if insert_pos != -1:
            # Find end of that line
            end_pos = content.find("\n\n", insert_pos)
            if end_pos != -1:
                content = content[:end_pos+2] + import_line + content[end_pos+2:]
    
    # =========================================================================
    # PATCH 2: Add K-Space tab creation
    # =========================================================================
    # Find the phantom tab creation and add k-space tab after it
    
    phantom_tab_code = '''        # === PHANTOM TAB (2D/3D Imaging) ===
        if PHANTOM_AVAILABLE:
            self.phantom_widget = PhantomWidget(self)
            self.tab_widget.addTab(self.phantom_widget, "🔬 Phantom")
        else:
            self.phantom_widget = None'''
    
    kspace_tab_code = '''        # === PHANTOM TAB (2D/3D Imaging) ===
        if PHANTOM_AVAILABLE:
            self.phantom_widget = PhantomWidget(self)
            self.tab_widget.addTab(self.phantom_widget, "🔬 Phantom")
        else:
            self.phantom_widget = None

        # === K-SPACE TAB (Signal-based simulation) ===
        if KSPACE_AVAILABLE:
            def get_phantom_for_kspace():
                """Get current phantom from PhantomWidget."""
                if self.phantom_widget is not None:
                    # PhantomWidget stores phantom in current_phantom
                    return getattr(self.phantom_widget, 'current_phantom', None)
                return None
            
            def get_magnetization_for_kspace():
                """Get magnetization from last Bloch simulation."""
                if self.last_result is not None:
                    return {
                        'mx': self.last_result.get('mx'),
                        'my': self.last_result.get('my'),
                        'mz': self.last_result.get('mz'),
                    }
                return None
            
            self.kspace_widget = KSpaceWidget(
                self,
                get_phantom_callback=get_phantom_for_kspace,
                get_magnetization_callback=get_magnetization_for_kspace
            )
            self.tab_widget.addTab(self.kspace_widget, "📡 K-Space")
        else:
            self.kspace_widget = None'''
    
    if phantom_tab_code in content:
        content = content.replace(phantom_tab_code, kspace_tab_code)
        print("✓ Added K-Space tab creation")
    else:
        print("⚠ Could not find exact phantom tab code - trying alternative pattern")
        
        # Try to find a simpler pattern
        pattern = r'(self\.tab_widget\.addTab\(self\.phantom_widget.*?\".*?Phantom\"\))'
        match = re.search(pattern, content)
        
        if match:
            kspace_insert = '''
        
        # === K-SPACE TAB (Signal-based simulation) ===
        if KSPACE_AVAILABLE:
            def get_phantom_for_kspace():
                if self.phantom_widget is not None:
                    return getattr(self.phantom_widget, 'current_phantom', None)
                return None
            
            self.kspace_widget = KSpaceWidget(
                self,
                get_phantom_callback=get_phantom_for_kspace
            )
            self.tab_widget.addTab(self.kspace_widget, "📡 K-Space")
        else:
            self.kspace_widget = None'''
            
            insert_pos = match.end()
            content = content[:insert_pos] + kspace_insert + content[insert_pos:]
            print("✓ Added K-Space tab (alternative method)")
        else:
            print("✗ Could not find phantom tab - please add K-Space tab manually")
            print("  See the manual instructions below")
    
    # Write patched file
    gui_path.write_text(content)
    print(f"\n✓ Patched file saved: {gui_path}")
    
    return True


# =============================================================================
# MANUAL PATCH INSTRUCTIONS
# =============================================================================

MANUAL_INSTRUCTIONS = """
================================================================================
MANUAL PATCH INSTRUCTIONS FOR bloch_gui.py
================================================================================

If the automatic patching doesn't work, follow these manual steps:

STEP 1: Add imports at the top of bloch_gui.py (after the phantom imports)
------------------------------------------------------------------------

# Import k-space simulation widget
try:
    from kspace_widget import KSpaceWidget
    from kspace_simulator import KSpaceSimulator, EddyCurrentModel, EPIParameters, CSIParameters
    KSPACE_AVAILABLE = True
except ImportError:
    KSPACE_AVAILABLE = False
    print("K-Space module not available - k-space tab will be disabled")


STEP 2: Add K-Space tab creation in init_ui() method
----------------------------------------------------
Find this line (around line 2433):
    self.tab_widget.addTab(self.phantom_widget, "🔬 Phantom")

Add the following code AFTER the phantom tab block (around line 2436):

        # === K-SPACE TAB (Signal-based simulation) ===
        if KSPACE_AVAILABLE:
            def get_phantom_for_kspace():
                \"\"\"Get current phantom from PhantomWidget.\"\"\"
                if self.phantom_widget is not None:
                    return getattr(self.phantom_widget, 'current_phantom', None)
                return None
            
            def get_magnetization_for_kspace():
                \"\"\"Get magnetization from last Bloch simulation.\"\"\"
                if self.last_result is not None:
                    return {
                        'mx': self.last_result.get('mx'),
                        'my': self.last_result.get('my'),
                        'mz': self.last_result.get('mz'),
                    }
                return None
            
            self.kspace_widget = KSpaceWidget(
                self,
                get_phantom_callback=get_phantom_for_kspace,
                get_magnetization_callback=get_magnetization_for_kspace
            )
            self.tab_widget.addTab(self.kspace_widget, "📡 K-Space")
        else:
            self.kspace_widget = None


STEP 3: Required files
----------------------
Make sure these files are in your project directory:
1. kspace_widget.py      - The GUI widget
2. kspace_simulator.py   - The simulation engine
3. spectral_phantom.py   - Spectral phantom support for CSI

================================================================================
"""


if __name__ == "__main__":
    import sys
    
    print("K-Space Integration Patcher for Bloch Simulator GUI")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        gui_path = sys.argv[1]
    else:
        gui_path = "bloch_gui.py"
    
    print(f"\nLooking for: {gui_path}")
    
    if Path(gui_path).exists():
        print(f"Found {gui_path}")
        
        response = input("\nApply patch? (y/n): ").strip().lower()
        if response == 'y':
            success = patch_bloch_gui(gui_path)
            if success:
                print("\n✓ Patching complete!")
                print("\nMake sure you have these files in your project directory:")
                print("  - kspace_widget.py")
                print("  - kspace_simulator.py")
                print("  - spectral_phantom.py")
            else:
                print("\n⚠ Patching had issues - check the output above")
        else:
            print("\nPatch cancelled")
    else:
        print(f"\n{gui_path} not found")
        print("\nManual instructions:")
    
    print(MANUAL_INSTRUCTIONS)
