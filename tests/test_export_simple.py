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

# Create QApplication
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
try:
    exporter = ImageExporter(plot_widget.plotItem)
    exporter.parameters()['width'] = 800
    output_file = 'test_export.png'
    exporter.export(output_file)

    if Path(output_file).exists():
        print(f"✓ PNG export SUCCESS: {output_file}")
        print(f"  File size: {Path(output_file).stat().st_size} bytes")
    else:
        print(f"✗ PNG export FAILED: File not created")
except Exception as e:
    print(f"✗ PNG export ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test SVG export
print("\nTesting SVG export...")
try:
    exporter = SVGExporter(plot_widget.plotItem)
    output_file = 'test_export.svg'
    exporter.export(output_file)

    if Path(output_file).exists():
        print(f"✓ SVG export SUCCESS: {output_file}")
        print(f"  File size: {Path(output_file).stat().st_size} bytes")
    else:
        print(f"✗ SVG export FAILED: File not created")
except Exception as e:
    print(f"✗ SVG export ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete. Check for test_export.png and test_export.svg in current directory.")
sys.exit(0)
