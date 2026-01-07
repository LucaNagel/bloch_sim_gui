#!/usr/bin/env python3
"""
Wrapper script for launching the Bloch Simulator GUI.
This is used by PyInstaller and for development testing.
"""
import sys
import os

# Ensure the src directory is in the python path
# This allows running from source without installation
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from blochsimulator.gui import main

if __name__ == "__main__":
    main()
