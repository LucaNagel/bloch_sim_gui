from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QRadioButton,
    QPushButton,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QApplication,
    QDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import numpy as np
import time
import json
from typing import Optional
from pathlib import Path
from ..visualization import ParameterSweepExportDialog
from ..notebook_exporter import export_notebook


class ParameterSweepWidget(QWidget):
    """Widget for running parameter sweeps (multiple simulations with varying parameters)."""

    sweep_finished = pyqtSignal(dict)  # Emits results when sweep completes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.sweep_running = False
        self.sweep_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Parameter Sweep")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title_label)

        # Info label
        info_label = QLabel(
            "Run multiple simulations by sweeping a parameter over a range."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Parameter selection
        param_group = QGroupBox("Sweep Parameter")
        param_layout = QVBoxLayout()

        param_sel_layout = QHBoxLayout()
        param_sel_layout.addWidget(QLabel("Parameter:"))
        self.param_combo = QComboBox()
        self.param_combo.addItems(
            [
                "Flip Angle (deg)",
                "TE (ms)",
                "TR (ms)",
                "TI (ms)",
                "B1 Scale Factor",
                "B1 Amplitude (G)",
                "T1 (ms)",
                "T2 (ms)",
                "Frequency Offset (Hz)",
            ]
        )
        self.param_combo.currentTextChanged.connect(self._update_range_limits)
        param_sel_layout.addWidget(self.param_combo)
        param_layout.addLayout(param_sel_layout)

        # Range controls
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Start:"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(-1e6, 1e6)
        self.start_spin.setValue(30)
        self.start_spin.setDecimals(2)
        range_layout.addWidget(self.start_spin)

        range_layout.addWidget(QLabel("End:"))
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(-1e6, 1e6)
        self.end_spin.setValue(90)
        self.end_spin.setDecimals(2)
        range_layout.addWidget(self.end_spin)

        range_layout.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(2, 500)
        self.steps_spin.setValue(13)
        range_layout.addWidget(self.steps_spin)
        param_layout.addLayout(range_layout)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Data Collection Mode
        mode_group = QGroupBox("Data Collection Mode")
        mode_layout = QVBoxLayout()

        self.mode_final = QRadioButton("Final State Only")
        self.mode_final.setChecked(True)
        self.mode_final.setToolTip(
            "Collect only the final magnetization/signal state (smaller memory usage)."
        )

        self.mode_dynamic = QRadioButton("Dynamic (Time-Resolved)")
        self.mode_dynamic.setToolTip(
            "Collect full time evolution (larger memory usage)."
        )

        mode_layout.addWidget(self.mode_final)
        mode_layout.addWidget(self.mode_dynamic)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Control buttons
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Sweep")
        self.run_button.clicked.connect(self.run_sweep)
        button_layout.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_sweep)
        button_layout.addWidget(self.stop_button)

        self.export_button = QPushButton("Export Results")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results)
        button_layout.addWidget(self.export_button)

        layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Results plot
        self.results_plot = pg.PlotWidget()
        self.results_plot.setLabel("left", "Metric Value")
        self.results_plot.setLabel("bottom", "Parameter Value")
        self.results_plot.setMinimumHeight(300)
        self.results_plot.addLegend()
        layout.addWidget(self.results_plot)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setMaximumHeight(200)
        layout.addWidget(self.results_table)

        layout.addStretch()
        self.setLayout(layout)

        # Initialize range limits
        self._update_range_limits()

        # Storage for results
        self.last_sweep_results = None

    def _update_range_limits(self):
        """Update spin box ranges based on selected parameter."""
        param = self.param_combo.currentText()

        if "Flip Angle" in param:
            self.start_spin.setRange(0, 9998)
            self.start_spin.setValue(30)
            self.end_spin.setRange(0, 9999)
            self.end_spin.setValue(90)
        elif "TE" in param or "TR" in param or "TI" in param:
            self.start_spin.setRange(0.1, 1000000)
            self.start_spin.setValue(10)
            self.end_spin.setRange(0.1, 1000000)
            self.end_spin.setValue(100)
        elif "B1 Scale" in param:
            self.start_spin.setRange(0, 5)
            self.start_spin.setValue(0.5)
            self.start_spin.setDecimals(3)
            self.end_spin.setRange(0, 5)
            self.end_spin.setValue(1.5)
            self.end_spin.setDecimals(3)
        elif "B1 Amplitude" in param:
            self.start_spin.setRange(0, 100)
            self.start_spin.setValue(0.0)
            self.start_spin.setDecimals(4)
            self.end_spin.setRange(0, 100)
            self.end_spin.setValue(1.0)
            self.end_spin.setDecimals(4)
        elif "T1" in param:
            self.start_spin.setRange(1, 180000)
            self.start_spin.setValue(500)
            self.end_spin.setRange(1, 180000)
            self.end_spin.setValue(2000)
        elif "T2" in param:
            self.start_spin.setRange(1, 20000)
            self.start_spin.setValue(20)
            self.end_spin.setRange(1, 20000)
            self.end_spin.setValue(100)
        elif "Frequency" in param:
            self.start_spin.setRange(-10000, 10000)
            self.start_spin.setValue(-500)
            self.end_spin.setRange(-10000, 10000)
            self.end_spin.setValue(500)

    def run_sweep(self):
        """Run parameter sweep."""
        if not self.parent_gui:
            QMessageBox.warning(self, "Error", "No parent GUI available.")
            return

        if hasattr(self.parent_gui, "set_sweep_mode"):
            self.parent_gui.set_sweep_mode(True)

        self.sweep_running = True
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)

        # Get sweep parameters
        param_name = self.param_combo.currentText()
        start_val = self.start_spin.value()
        end_val = self.end_spin.value()
        n_steps = self.steps_spin.value()

        # Create parameter array
        param_values = np.linspace(start_val, end_val, n_steps)

        # Determine mode
        save_full = self.mode_dynamic.isChecked()

        # Enforce simulation mode
        original_sim_mode = self.parent_gui.mode_combo.currentText()
        target_sim_mode = "Time-resolved" if save_full else "Endpoint"
        self.parent_gui.mode_combo.setCurrentText(target_sim_mode)

        # Standard metrics to collect for every sweep
        selected_metrics = ["Mx", "My", "Mz", "Signal", "Mean Signal"]

        # Capture constant parameters
        constant_params = {}
        try:
            # Tissue
            if hasattr(self.parent_gui, "tissue_widget"):
                tw = self.parent_gui.tissue_widget
                params = tw.get_parameters()
                constant_params["t1"] = float(params.t1)
                constant_params["t2"] = float(params.t2)
                constant_params["tissue_name"] = str(params.name)
                constant_params["m0"] = float(tw.get_initial_mz())

            # Sequence (basic params)
            if hasattr(self.parent_gui, "sequence_designer"):
                sd = self.parent_gui.sequence_designer
                constant_params["te"] = float(sd.te_spin.value() / 1000.0)
                constant_params["tr"] = float(sd.tr_spin.value() / 1000.0)
                constant_params["ti"] = float(sd.ti_spin.value() / 1000.0)
                constant_params["sequence_type"] = str(sd.sequence_type.currentText())
                constant_params["echo_count"] = int(sd.spin_echo_echoes.value())
                constant_params["slice_thickness"] = float(
                    sd.slice_thickness_spin.value()
                )
                constant_params["slice_gradient"] = float(
                    sd.slice_gradient_spin.value()
                )
                constant_params["ssfp_repeats"] = int(sd.ssfp_repeats.value())

            # RF
            if hasattr(self.parent_gui, "rf_designer"):
                rd = self.parent_gui.rf_designer
                constant_params["flip_angle"] = float(rd.flip_angle.value())
                constant_params["pulse_duration"] = float(rd.duration.value() / 1000.0)
                constant_params["pulse_type"] = str(rd.pulse_type.currentText())
                constant_params["rf_phase"] = float(rd.phase.value())
                constant_params["rf_freq_offset"] = float(rd.freq_offset.value())
                constant_params["rf_time_bw_product"] = float(rd.tbw.value())

            # Simulation
            if hasattr(self.parent_gui, "pos_spin"):
                constant_params["num_positions"] = int(self.parent_gui.pos_spin.value())
                constant_params["position_range_cm"] = float(
                    self.parent_gui.pos_range.value()
                )
            if hasattr(self.parent_gui, "freq_spin"):
                constant_params["num_frequencies"] = int(
                    self.parent_gui.freq_spin.value()
                )
                constant_params["frequency_range_hz"] = float(
                    self.parent_gui.freq_range.value()
                )
            if hasattr(self.parent_gui, "time_step_spin"):
                constant_params["time_step"] = float(
                    self.parent_gui.time_step_spin.value() * 1e-6
                )

        except Exception as e:
            msg = f"Warning: Could not capture constant parameters: {e}"
            if self.parent_gui:
                self.parent_gui.log_message(msg)
            print(msg)

        if constant_params and self.parent_gui:
            self.parent_gui.log_message(
                f"Captured {len(constant_params)} constant parameters for export."
            )

        # Run sweep
        results = {
            "parameter_name": param_name,
            "parameter_values": param_values,
            "metrics": {metric: [] for metric in selected_metrics},
            "mode": "dynamic" if save_full else "final",
            "constant_params": constant_params,
        }

        # Store initial values for parameters that need them (like B1 scale)
        initial_flip_angle = self.parent_gui.rf_designer.flip_angle.value()

        # Capture initial state of the parameter to restore it later
        initial_param_val = None
        initial_freq_range = None
        if "Flip Angle" in param_name:
            initial_param_val = self.parent_gui.rf_designer.flip_angle.value()
        elif "TE" in param_name:
            initial_param_val = self.parent_gui.sequence_designer.te_spin.value()
        elif "TR" in param_name:
            initial_param_val = self.parent_gui.sequence_designer.tr_spin.value()
        elif "TI" in param_name:
            initial_param_val = self.parent_gui.sequence_designer.ti_spin.value()
        elif "T1" in param_name:
            initial_param_val = self.parent_gui.tissue_widget.t1_spin.value()
        elif "T2" in param_name:
            initial_param_val = self.parent_gui.tissue_widget.t2_spin.value()
        elif "Frequency" in param_name:
            initial_param_val = (
                self.parent_gui.freq_center.value()
                if hasattr(self.parent_gui, "freq_center")
                else 0
            )
            initial_freq_range = (
                self.parent_gui.freq_range.value()
                if hasattr(self.parent_gui, "freq_range")
                else 0
            )
        elif "B1 Scale" in param_name:
            initial_param_val = 1.0  # Scale factor is 1.0 relative to initial
        elif "B1 Amplitude" in param_name:
            initial_param_val = self.parent_gui.rf_designer.b1_amplitude.value()

        try:
            for i, param_val in enumerate(param_values):
                if not self.sweep_running:
                    break

                # Update parameter
                self._apply_parameter_value(param_name, param_val, initial_flip_angle)

                # Run simulation
                try:
                    self.parent_gui.run_simulation()
                    sim_thread = getattr(self.parent_gui, "simulation_thread", None)

                    step_result_container = []
                    step_error_container = []

                    def on_step_finished(res):
                        step_result_container.append(res)

                    def on_step_error(err):
                        step_error_container.append(err)

                    if sim_thread:
                        sim_thread.finished.connect(on_step_finished)
                        sim_thread.error.connect(on_step_error)

                        # Wait for completion
                        while not step_result_container and not step_error_container:
                            QApplication.processEvents()
                            if not self.sweep_running:
                                if hasattr(sim_thread, "request_cancel"):
                                    sim_thread.request_cancel()
                                break

                            if (
                                not sim_thread.isRunning()
                                and not step_result_container
                                and not step_error_container
                            ):
                                # Thread stopped but no signals? Wait a brief moment for pending signals
                                start_wait = time.time()
                                while (
                                    time.time() - start_wait < 0.2
                                ):  # 200ms grace period
                                    QApplication.processEvents()
                                    if step_result_container or step_error_container:
                                        break
                                    time.sleep(0.01)
                                break

                            time.sleep(0.005)

                    if step_error_container:
                        raise RuntimeError(
                            f"Simulation failed: {step_error_container[0]}"
                        )

                    # Extract metrics from results
                    if step_result_container:
                        result = step_result_container[0]
                        # Capture time vector if available and relevant
                        if (
                            save_full
                            and "time" not in results
                            and result.get("time") is not None
                        ):
                            results["time"] = result["time"]
                            # Also capture positions and frequencies for full data exports
                            if (
                                "positions" not in results
                                and result.get("mx") is not None
                            ):
                                # result is from SimulationThread, which might have them
                                # or we get them from main_window
                                results["positions"] = self.parent_gui.last_positions
                                results["frequencies"] = (
                                    self.parent_gui.last_frequencies
                                )

                        for metric in selected_metrics:
                            value = self._extract_metric(metric, result, save_full)
                            results["metrics"][metric].append(value)
                    else:
                        # No result - append NaN to maintain array length
                        for metric in selected_metrics:
                            results["metrics"][metric].append(float("nan"))
                        if self.sweep_running:
                            self.parent_gui.log_message(
                                f"Warning: No result for {param_name}={param_val:.2f}"
                            )
                except Exception as e:
                    import traceback

                    error_msg = f"Error at {param_name}={param_val:.2f}: {str(e)}\n{traceback.format_exc()}"
                    self.parent_gui.log_message(error_msg)
                    break

                # Update progress
                self.progress_bar.setValue(int((i + 1) / n_steps * 100))

            # Store and display results
            self.last_sweep_results = results
            self._display_results(results)
        finally:
            # Restore initial parameter value
            if initial_param_val is not None:
                self._apply_parameter_value(
                    param_name, initial_param_val, initial_flip_angle
                )
            if initial_freq_range is not None and hasattr(
                self.parent_gui, "freq_range"
            ):
                self.parent_gui.freq_range.setValue(initial_freq_range)

            # Restore simulation mode
            if hasattr(self, "parent_gui") and original_sim_mode:
                self.parent_gui.mode_combo.setCurrentText(original_sim_mode)

            self.sweep_running = False
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.export_button.setEnabled(True)
            if hasattr(self.parent_gui, "set_sweep_mode"):
                self.parent_gui.set_sweep_mode(False)

    def stop_sweep(self):
        """Stop the parameter sweep."""
        self.sweep_running = False
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if hasattr(self.parent_gui, "set_sweep_mode"):
            self.parent_gui.set_sweep_mode(False)

    def _apply_parameter_value(self, param_name, value, initial_flip_angle=None):
        """Apply parameter value to the GUI controls."""
        if "Flip Angle" in param_name:
            self.parent_gui.rf_designer.flip_angle.setValue(value)
        elif "TE" in param_name:
            self.parent_gui.sequence_designer.te_spin.setValue(value)
        elif "TR" in param_name:
            self.parent_gui.sequence_designer.tr_spin.setValue(value)
        elif "TI" in param_name:
            self.parent_gui.sequence_designer.ti_spin.setValue(value)
        elif "B1 Scale" in param_name:
            # Scale the initial flip angle (not current, to avoid cumulative scaling)
            if initial_flip_angle is not None:
                self.parent_gui.rf_designer.flip_angle.setValue(
                    initial_flip_angle * value
                )
            else:
                # Fallback if not provided
                self.parent_gui.rf_designer.flip_angle.setValue(90 * value)
        elif "B1 Amplitude" in param_name:
            self.parent_gui.rf_designer.b1_amplitude.setValue(value)
        elif "T1" in param_name:
            self.parent_gui.tissue_widget.t1_spin.setValue(value)
        elif "T2" in param_name:
            self.parent_gui.tissue_widget.t2_spin.setValue(value)
        elif "Frequency" in param_name:
            # Set single frequency offset
            self.parent_gui.freq_range.setValue(0)
            self.parent_gui.freq_center.setValue(value)

    def _extract_metric(self, metric_name, result, save_full):
        """Extract metric value from simulation result based on mode."""
        try:
            if metric_name == "Mean Signal":
                # Scalar for plotting (mean absolute transverse magnetization)
                mx = result.get("mx")
                my = result.get("my")
                if mx is not None and my is not None:
                    mxy = np.sqrt(mx**2 + my**2)
                    if save_full:
                        # For time-resolved, take mean over spatial dimensions (axis 1 onwards)
                        # Result should be 1D array of time points
                        if mxy.ndim > 1:
                            return np.mean(mxy, axis=tuple(range(1, mxy.ndim)))
                        return mxy  # If 1D, it's already a single spin time course
                    else:
                        # For final state, mean over all spins
                        return float(np.mean(mxy))
                return 0.0

            # Raw Data Extraction
            key = metric_name.lower()
            val = result.get(key)
            if val is not None:
                if save_full:
                    return val
                return val

        except Exception as e:
            if self.parent_gui:
                self.parent_gui.log_message(
                    f"Error extracting metric '{metric_name}': {str(e)}"
                )
            return float("nan")
        return 0.0

    def _display_results(self, results):
        """Display sweep results in plot and table."""
        self.results_plot.clear()

        param_values = results["parameter_values"]
        param_name = results["parameter_name"]

        # Plot each metric
        colors = ["r", "g", "b", "y", "m"]
        for i, (metric, values) in enumerate(results["metrics"].items()):
            if len(values) != len(param_values):
                continue
            # Only plot scalar metrics
            scalar_vals = []
            all_scalar = True
            for v in values:
                try:
                    scalar_vals.append(float(v))
                except Exception:
                    all_scalar = False
                    break
            if not all_scalar:
                continue
            color = colors[i % len(colors)]
            self.results_plot.plot(
                param_values,
                scalar_vals,
                pen=pg.mkPen(color, width=2),
                symbol="o",
                symbolBrush=color,
                name=metric,
            )

        self.results_plot.setLabel("bottom", param_name)

        # Update table
        metrics = list(results["metrics"].keys())
        self.results_table.setRowCount(len(param_values))
        self.results_table.setColumnCount(1 + len(metrics))
        self.results_table.setHorizontalHeaderLabels([param_name] + metrics)

        for row, param_val in enumerate(param_values):
            self.results_table.setItem(row, 0, QTableWidgetItem(f"{param_val:.3f}"))
            for col, metric in enumerate(metrics):
                if row < len(results["metrics"][metric]):
                    value = results["metrics"][metric][row]
                    if isinstance(value, np.ndarray):
                        display_text = f"array{value.shape}"
                    else:
                        try:
                            display_text = f"{float(value):.6f}"
                        except Exception:
                            display_text = str(value)
                    self.results_table.setItem(
                        row, col + 1, QTableWidgetItem(display_text)
                    )

        self.results_table.resizeColumnsToContents()

    def _process_results_for_export(self, results, save_full):
        """Filter results based on export options and data availability."""
        mode = results.get("mode", "final")

        # If data is already reduced, or user wants full data (and it exists), return as is
        if mode == "final" or save_full:
            return results

        # Mode is dynamic, but user requested final state -> Slice time dimension
        processed = results.copy()
        processed["metrics"] = {}
        for metric, vals in results["metrics"].items():
            new_vals = []
            for v in vals:
                # Assume 3D arrays are (ntime, npos, nfreq) -> take last time point
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    new_vals.append(v[-1, :, :])
                # Assume 1D arrays matching time length -> take last point
                elif isinstance(v, np.ndarray) and v.ndim == 1 and len(v) > 1:
                    new_vals.append(v[-1])
                else:
                    new_vals.append(v)
            processed["metrics"][metric] = new_vals
        return processed

    def export_results(self):
        """Export sweep results using the unified dialog."""
        if not self.last_sweep_results:
            QMessageBox.warning(self, "No Results", "No sweep results to export.")
            return

        dialog = ParameterSweepExportDialog(
            self,
            default_filename="sweep_results",
            default_directory=self._default_export_dir(),
        )

        # Configure dialog based on available data
        mode = self.last_sweep_results.get("mode", "final")
        if mode == "final":
            dialog.radio_full.setEnabled(False)
            dialog.radio_full.setText(dialog.radio_full.text() + " (Not available)")
            dialog.radio_full.setToolTip("Sweep was run in 'Final State Only' mode.")
            dialog.radio_final.setChecked(True)

        if dialog.exec_() != QDialog.Accepted:
            return

        options = dialog.get_export_options()
        base_path = Path(options["base_path"])
        save_full = options.get("save_full_time_course", True)

        export_data = self._process_results_for_export(
            self.last_sweep_results, save_full
        )
        exported_files = []
        primary_data_file = None

        try:
            # Export CSV
            if options["csv"]:
                csv_path = base_path.with_suffix(".csv")
                self._save_sweep_results_csv(csv_path, export_data)
                exported_files.append(str(csv_path.name))
                if not primary_data_file:
                    primary_data_file = csv_path

            # Export NPZ (mapped from HDF5 checkbox)
            if options["hdf5"]:
                npz_path = base_path.with_suffix(".npz")
                self._save_sweep_results_npz(npz_path, export_data)
                exported_files.append(str(npz_path.name))
                primary_data_file = npz_path  # Prefer NPZ as primary for notebook

            # Export Analysis Notebook
            if options["notebook_analysis"]:
                # Ensure we have a data file to load
                if not primary_data_file:
                    # User asked for notebook only? Save NPZ automatically
                    npz_path = base_path.with_suffix(".npz")
                    self._save_sweep_results_npz(npz_path, export_data)
                    exported_files.append(str(npz_path.name) + " (auto-generated)")
                    primary_data_file = npz_path

                nb_path = base_path.with_suffix(".ipynb")

                # Determine dynamic flag from data
                is_dynamic = export_data.get("mode", "final") == "dynamic"

                export_notebook(
                    mode="sweep",
                    filename=str(nb_path),
                    data_filename=primary_data_file.name,  # Use relative path
                    param_name=export_data["parameter_name"],
                    metrics=list(export_data["metrics"].keys()),
                    is_dynamic=is_dynamic,
                )
                exported_files.append(str(nb_path.name))

            if exported_files:
                QMessageBox.information(
                    self, "Export Complete", f"Exported:\n" + "\n".join(exported_files)
                )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _default_export_dir(self) -> Path:
        """Resolve a writable export directory for sweep results."""
        if getattr(self, "parent_gui", None) and hasattr(
            self.parent_gui, "_get_export_directory"
        ):
            try:
                return Path(self.parent_gui._get_export_directory())
            except Exception:
                pass
        return Path.cwd()

    def _save_sweep_results_csv(self, path: Path, results: dict) -> Optional[Path]:
        """Save sweep results to CSV and return any auxiliary array path."""
        # results passed in
        param_name = results["parameter_name"]
        metrics = list(results["metrics"].keys())
        array_metrics = {m: [] for m in metrics}
        constant_params = results.get("constant_params", {})

        with open(path, "w") as f:
            # Header with metadata
            f.write(f"# BlochSimulator Parameter Sweep\n")
            f.write(f"# Parameter: {param_name}\n")
            f.write(f"# Constant Parameters: {json.dumps(constant_params)}\n")

            f.write(f"{param_name}," + ",".join(metrics) + "\n")

            for i, param_val in enumerate(results["parameter_values"]):
                row = [f"{param_val:.6f}"]
                for metric in metrics:
                    if i < len(results["metrics"][metric]):
                        value = results["metrics"][metric][i]
                        if isinstance(value, np.ndarray):
                            row.append(f"array{value.shape}")
                            array_metrics[metric].append(value)
                        else:
                            try:
                                row.append(f"{float(value):.6f}")
                            except Exception:
                                row.append(str(value))
                    else:
                        row.append("")
                f.write(",".join(row) + "\n")

        stacked_arrays = {}
        for metric, vals in array_metrics.items():
            if not vals:
                continue
            try:
                stacked_arrays[metric] = np.stack(vals)
            except Exception:
                stacked_arrays[metric] = np.array(vals, dtype=object)

        # Include time vector in sidecar if available
        if "time" in results:
            stacked_arrays["time"] = results["time"]
        if "positions" in results:
            stacked_arrays["positions"] = results["positions"]
        if "frequencies" in results:
            stacked_arrays["frequencies"] = results["frequencies"]

        if stacked_arrays:
            array_path = path.with_name(path.stem + "_arrays.npz")
            np.savez(
                array_path,
                parameter_name=results["parameter_name"],
                parameter_values=np.asarray(results["parameter_values"]),
                constant_params=str(json.dumps(constant_params)),
                **stacked_arrays,
            )
            return array_path
        return None

    def _save_sweep_results_npz(self, path: Path, results: dict):
        """Save sweep results into a single NPZ archive."""
        # results passed in
        payload = {
            "parameter_name": results["parameter_name"],
            "parameter_values": np.asarray(results["parameter_values"]),
            "constant_params": json.dumps(results.get("constant_params", {})),
        }
        if "time" in results:
            payload["time"] = results["time"]
        if "positions" in results:
            payload["positions"] = results["positions"]
        if "frequencies" in results:
            payload["frequencies"] = results["frequencies"]

        for metric, values in results["metrics"].items():
            payload[metric] = self._stack_metric_values(values)
        np.savez(path, **payload)

    def _save_sweep_results_npy(self, path: Path):
        """Save sweep results as a NumPy binary with a dictionary payload."""
        results = self.last_sweep_results
        payload = {
            "parameter_name": results["parameter_name"],
            "parameter_values": np.asarray(results["parameter_values"]),
            "metrics": {
                metric: self._stack_metric_values(vals)
                for metric, vals in results["metrics"].items()
            },
        }
        np.save(path, payload, allow_pickle=True)

    def _stack_metric_values(self, values):
        """Best-effort stacking for metric values with mixed scalar/array content."""
        try:
            return np.stack([np.asarray(v) for v in values])
        except Exception:
            try:
                return np.asarray(values, dtype=float)
            except Exception:
                return np.asarray(values, dtype=object)
