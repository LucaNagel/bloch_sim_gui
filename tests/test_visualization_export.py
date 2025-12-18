"""
test_visualization_export.py - Tests for visualization export functionality

This module tests the export capabilities for static images from GUI visualizations.

Author: Luca Nagel
Date: 2024
"""

import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg

# Import the export classes
from blochsimulator.visualization import ImageExporter, ExportImageDialog, AnimationExporter, DatasetExporter

try:
    import imageio  # noqa: F401
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False


# Create QApplication instance for tests (required for PyQt widgets)
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


class TestImageExporter(unittest.TestCase):
    """Test cases for ImageExporter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.exporter = ImageExporter()
        self.temp_dir = tempfile.mkdtemp()

    def test_exporter_initialization(self):
        """Test that ImageExporter initializes correctly."""
        self.assertIsNotNone(self.exporter)
        self.assertEqual(self.exporter.default_dpi, 300)

    def test_png_export(self):
        """Test PNG export from a simple plot."""
        # Create a simple plot
        plot_widget = pg.PlotWidget()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plot_widget.plot(x, y)

        # Export to PNG
        output_path = Path(self.temp_dir) / "test_plot.png"
        result = self.exporter.export_pyqtgraph_plot(
            plot_widget,
            str(output_path),
            format='png',
            width=800
        )

        # Check that file was created
        self.assertTrue(result)
        self.assertTrue(Path(result).exists())
        self.assertEqual(Path(result).suffix, '.png')

    def test_svg_export(self):
        """Test SVG export from a simple plot."""
        # Create a simple plot
        plot_widget = pg.PlotWidget()
        x = np.linspace(0, 10, 100)
        y = np.cos(x)
        plot_widget.plot(x, y)

        # Export to SVG
        output_path = Path(self.temp_dir) / "test_plot.svg"
        result = self.exporter.export_pyqtgraph_plot(
            plot_widget,
            str(output_path),
            format='svg'
        )

        # Check that file was created
        self.assertTrue(result)
        self.assertTrue(Path(result).exists())
        self.assertEqual(Path(result).suffix, '.svg')

    def test_widget_screenshot(self):
        """Test screenshot export of a widget."""
        # Create a simple widget
        plot_widget = pg.PlotWidget()
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * np.cos(x)
        plot_widget.plot(x, y)

        # Export screenshot
        output_path = Path(self.temp_dir) / "test_screenshot.png"
        result = self.exporter.export_widget_screenshot(
            plot_widget,
            str(output_path),
            format='png'
        )

        # Check that file was created
        self.assertTrue(result)
        self.assertTrue(Path(result).exists())

    def test_format_validation(self):
        """Test that unsupported formats raise an error."""
        plot_widget = pg.PlotWidget()
        output_path = Path(self.temp_dir) / "test_plot.xyz"

        result = self.exporter.export_pyqtgraph_plot(
            plot_widget,
            str(output_path),
            format='xyz'
        )
        self.assertFalse(result, "Expected export to fail for unsupported format")

    def test_file_extension_correction(self):
        """Test that file extensions are corrected automatically."""
        plot_widget = pg.PlotWidget()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plot_widget.plot(x, y)

        # Request PNG but provide wrong extension
        output_path = Path(self.temp_dir) / "test_plot.txt"
        result = self.exporter.export_pyqtgraph_plot(
            plot_widget,
            str(output_path),
            format='png'
        )

        # Check that file has correct extension
        self.assertTrue(result)
        self.assertEqual(Path(result).suffix, '.png')


class TestExportImageDialog(unittest.TestCase):
    """Test cases for ExportImageDialog class."""

    def test_dialog_initialization(self):
        """Test that ExportImageDialog initializes correctly."""
        dialog = ExportImageDialog(default_filename="test_export")
        self.assertIsNotNone(dialog)
        self.assertEqual(dialog.default_filename, "test_export")
        self.assertEqual(dialog.selected_format, 'png')

    def test_format_options(self):
        """Test that all format options are available."""
        dialog = ExportImageDialog()
        formats = ['PNG (Raster)', 'SVG (Vector)', 'PDF (Vector)']

        self.assertEqual(dialog.format_combo.count(), len(formats))
        for i, format_name in enumerate(formats):
            self.assertEqual(dialog.format_combo.itemText(i), format_name)

    def test_default_width(self):
        """Test that default width is set correctly."""
        dialog = ExportImageDialog()
        self.assertEqual(dialog.width_spin.value(), 2400)


class TestExportIntegration(unittest.TestCase):
    """Integration tests for export workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_complete_export_workflow_png(self):
        """Test complete export workflow for PNG."""
        # Create plot with multiple traces
        plot_widget = pg.PlotWidget()
        x = np.linspace(0, 10, 100)
        plot_widget.plot(x, np.sin(x), pen='r', name='Sin')
        plot_widget.plot(x, np.cos(x), pen='b', name='Cos')
        plot_widget.setLabel('left', 'Amplitude')
        plot_widget.setLabel('bottom', 'Time', 's')

        # Export
        exporter = ImageExporter()
        output_path = Path(self.temp_dir) / "multi_trace.png"
        result = exporter.export_pyqtgraph_plot(
            plot_widget,
            str(output_path),
            format='png',
            width=1200
        )

        # Verify
        self.assertTrue(result)
        self.assertTrue(Path(result).exists())
        # Check file size is reasonable (not empty)
        self.assertGreater(Path(result).stat().st_size, 1000)

    def test_complete_export_workflow_svg(self):
        """Test complete export workflow for SVG."""
        # Create plot
        plot_widget = pg.PlotWidget()
        x = np.linspace(0, 2*np.pi, 100)
        plot_widget.plot(x, np.sin(x)**2, pen='g')
        plot_widget.setLabel('left', 'Power')
        plot_widget.setLabel('bottom', 'Phase', 'rad')

        # Export
        exporter = ImageExporter()
        output_path = Path(self.temp_dir) / "power_plot.svg"
        result = exporter.export_pyqtgraph_plot(
            plot_widget,
            str(output_path),
            format='svg'
        )

        # Verify
        self.assertTrue(result)
        self.assertTrue(Path(result).exists())
        # SVG files should contain XML content
        with open(result, 'r') as f:
            content = f.read()
            self.assertIn('<?xml', content)
            self.assertIn('<svg', content)

class TestDatasetExporter(unittest.TestCase):
    """Basic dataset export tests."""

    def setUp(self):
        self.exporter = DatasetExporter()
        self.temp_dir = tempfile.mkdtemp()

    def test_magnetization_csv_export(self):
        time_s = np.linspace(0, 1e-3, 4)
        mx = np.ones((4, 1, 1)) * 0.1
        my = np.ones((4, 1, 1)) * 0.2
        mz = np.ones((4, 1, 1)) * 0.3
        outfile = Path(self.temp_dir) / "mag.csv"
        result = self.exporter.export_magnetization(time_s, mx, my, mz, None, None, str(outfile), format='csv')
        self.assertTrue(result)
        self.assertTrue(Path(result).exists())

    def test_signal_npy_export(self):
        time_s = np.linspace(0, 2e-3, 5)
        signal = (np.ones((5, 1, 1)) * (1 + 1j)).astype(np.complex128)
        outfile = Path(self.temp_dir) / "signal.npy"
        result = self.exporter.export_signal(time_s, signal, str(outfile), format='npy')
        arr = np.load(result)
        self.assertIn('time_ms', arr.dtype.names)
        self.assertIn('signal_real_p0_f0', arr.dtype.names)
        self.assertEqual(arr.shape[0], 5)


class TestAnimationExporter(unittest.TestCase):
    """Basic tests for AnimationExporter."""

    @unittest.skipUnless(_HAS_IMAGEIO, "imageio not installed")
    def test_gif_export(self):
        """Ensure GIF export creates a file."""
        exporter = AnimationExporter()
        temp_dir = tempfile.mkdtemp()
        time = np.linspace(0, 1, 30)
        trace = np.sin(2 * np.pi * time)
        groups = [
            {
                'title': 'Test Trace',
                'ylabel': 'Value',
                'series': [{'data': trace, 'label': 'sin', 'color': 'r'}]
            }
        ]
        output_path = Path(temp_dir) / "anim.gif"
        result = exporter.export_time_series_animation(
            time,
            groups,
            str(output_path),
            fps=10,
            max_frames=10,
            width=480,
            height=360
        )
        self.assertTrue(result)
        self.assertTrue(Path(result).exists())
        self.assertGreater(Path(result).stat().st_size, 500)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestImageExporter))
    suite.addTests(loader.loadTestsFromTestCase(TestExportImageDialog))
    suite.addTests(loader.loadTestsFromTestCase(TestExportIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetExporter))
    suite.addTests(loader.loadTestsFromTestCase(TestAnimationExporter))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
