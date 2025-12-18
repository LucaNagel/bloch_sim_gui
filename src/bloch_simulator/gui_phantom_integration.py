"""
bloch_gui_phantom_integration.py

This file contains the code needed to integrate the PhantomWidget into bloch_gui.py.

INSTRUCTIONS FOR INTEGRATION:
=============================

1. Copy phantom.py and phantom_widget.py to your project directory

2. Add these imports at the top of bloch_gui.py (around line 36, after existing imports):

   # Import phantom module for 2D/3D phantom simulation
   try:
       from phantom import Phantom, PhantomFactory
       from phantom_widget import PhantomWidget
       PHANTOM_AVAILABLE = True
   except ImportError:
       PHANTOM_AVAILABLE = False
       print("Phantom module not available - phantom tab will be disabled")

3. In the BlochSimulatorGUI.init_ui() method, after the "Spatial" tab is added 
   (around line 2397, after: self.tab_widget.addTab(spatial_container, "Spatial")),
   add the following code:

        # === PHANTOM TAB (2D/3D Imaging) ===
        if PHANTOM_AVAILABLE:
            self.phantom_widget = PhantomWidget(self)
            self.tab_widget.addTab(self.phantom_widget, "🔬 Phantom")
        else:
            self.phantom_widget = None

4. That's it! The phantom tab will appear in the main visualization tabs.

ALTERNATIVE: Standalone Phantom Window
======================================
If you prefer to have the phantom simulation in a separate window,
add this method to BlochSimulatorGUI class and a menu item to trigger it:

    def open_phantom_window(self):
        \"\"\"Open phantom simulation in a separate window.\"\"\"
        if not PHANTOM_AVAILABLE:
            QMessageBox.warning(self, "Not Available", 
                "Phantom module is not installed. Please add phantom.py and phantom_widget.py")
            return
        
        self.phantom_window = QMainWindow()
        self.phantom_window.setWindowTitle("Phantom Simulation")
        self.phantom_window.resize(1200, 800)
        self.phantom_window.setCentralWidget(PhantomWidget())
        self.phantom_window.show()

Then add to create_menu():
    phantom_action = file_menu.addAction("Phantom Simulation...")
    phantom_action.triggered.connect(self.open_phantom_window)

"""

# ============================================================================
# COMPLETE INTEGRATION CODE (can be copy-pasted directly)
# ============================================================================

# Add this import block after line 36 in bloch_gui.py:
IMPORT_BLOCK = '''
# Import phantom module for 2D/3D phantom simulation
try:
    from phantom import Phantom, PhantomFactory
    from phantom_widget import PhantomWidget
    PHANTOM_AVAILABLE = True
except ImportError:
    PHANTOM_AVAILABLE = False
    print("Phantom module not available - phantom tab will be disabled")
'''

# Add this code block after the "Spatial" tab is created (around line 2397):
TAB_INTEGRATION_BLOCK = '''
        # === PHANTOM TAB (2D/3D Imaging) ===
        if PHANTOM_AVAILABLE:
            self.phantom_widget = PhantomWidget(self)
            self.tab_widget.addTab(self.phantom_widget, "🔬 Phantom")
        else:
            self.phantom_widget = None
'''

# ============================================================================
# TESTING SCRIPT
# ============================================================================

def test_phantom_widget_standalone():
    """Test the phantom widget in a standalone window."""
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Check if phantom module is available
    try:
        from .phantom import PhantomFactory
        from .phantom_widget import PhantomWidget
    except ImportError as e:
        print(f"Error importing phantom modules: {e}")
        print("Make sure phantom.py and phantom_widget.py are in the same directory")
        sys.exit(1)
    
    window = QMainWindow()
    window.setWindowTitle("Phantom Simulation - Standalone Test")
    window.resize(1200, 800)
    window.setCentralWidget(PhantomWidget())
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    test_phantom_widget_standalone()
