#!/usr/bin/env python
"""
Simple test script to verify PyQtGraph export functionality.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter, SVGExporter
from pathlib import Path
import pytest

def test_export_simple():
    # Create QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create a simple plot
    plot_widget = pg.PlotWidget()
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plot_widget.plot(x, y, pen='r')
    plot_widget.setLabel('left', 'Amplitude')
    plot_widget.setLabel('bottom', 'Time', 's')

    # Show the widget (important - some exporters need rendered content)
    plot_widget.show()
    app.processEvents()  # Process events to ensure rendering

    # Test PNG export
    print("Testing PNG export...")
    exporter = ImageExporter(plot_widget.plotItem)
    exporter.parameters()['width'] = 800
    output_file_png = 'test_export.png'
    exporter.export(output_file_png)

    assert Path(output_file_png).exists(), "PNG export FAILED: File not created"
    print(f"✓ PNG export SUCCESS: {output_file_png}")
    print(f"  File size: {Path(output_file_png).stat().st_size} bytes")

    # Test SVG export
    print("\nTesting SVG export...")
    exporter = SVGExporter(plot_widget.plotItem)
    output_file_svg = 'test_export.svg'
    exporter.export(output_file_svg)

    assert Path(output_file_svg).exists(), "SVG export FAILED: File not created"
    print(f"✓ SVG export SUCCESS: {output_file_svg}")
    print(f"  File size: {Path(output_file_svg).stat().st_size} bytes")
    
    # Cleanup
    plot_widget.close()

if __name__ == "__main__":
    test_export_simple()
