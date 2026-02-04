#!/usr/bin/env python3
"""
bloch_gui.py - Interactive GUI for Bloch equation simulator

This module provides a graphical user interface for designing pulse sequences,
setting parameters, running simulations, and visualizing results.

This file is now a wrapper around the refactored `ui` package.
It re-exports key classes for backward compatibility.
"""

from .ui.main_window import main, BlochSimulatorGUI
from .ui.sequence_designer import SequenceDesigner
from .ui.rf_pulse_designer import RFPulseDesigner
from .ui.tissue_parameters import TissueParameterWidget
from .ui.magnetization_viewer import MagnetizationViewer
from .ui.parameter_sweep import ParameterSweepWidget
from .ui.controls import UniversalTimeControl

if __name__ == "__main__":
    main()
