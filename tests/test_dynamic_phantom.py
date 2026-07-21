import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from unittest.mock import MagicMock

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import (
    DynamicB0,
    DynamicSpectralPhantom,
    KineticRegionDefinition,
    PyruvateInflow,
    TimeCurve,
    rasterize_kpl_regions,
    simulate_two_pool_kinetics,
)
from blochsimulator.phantom_design import (
    PhantomDesign,
    ShapeDefinition,
    SpectralPeakDefinition,
)
from blochsimulator.sequence import ADCEvent, GradientEvent, RFEvent, SequenceProgram
from blochsimulator.spectral_phantom import ChemicalSpecies
from blochsimulator.ui.phantom_designer import SpectralPhantomDesignerDialog
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationThread


def _dynamic_phantom(kpl=(0.0, 0.1)):
    shape = (2, 1, 1)
    return DynamicSpectralPhantom(
        shape=shape,
        fov=(0.02, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.asarray(kpl, dtype=float).reshape(shape),
        nucleus="C13",
    )


def test_kinetic_regions_rasterize_with_later_region_priority():
    whole = KineticRegionDefinition(
        "whole", "box", (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), 0.02
    )
    right = KineticRegionDefinition(
        "right", "box", (0.75, 0.5, 0.5), (0.5, 1.0, 1.0), 0.08
    )
    result = rasterize_kpl_regions((4, 1, 1), (whole, right))

    assert np.array_equal(result[:, 0, 0], [0.02, 0.02, 0.08, 0.08])


def test_no_rf_dynamic_sequence_matches_irreversible_analytic_solution():
    phantom = _dynamic_phantom()
    duration = 2.0
    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        SequenceProgram((), duration_s=duration), phantom
    )
    final = result.final_pool_magnetization[..., 2]

    for voxel, kpl in enumerate((0.0, 0.1)):
        a = 1.0 / 30.0 + kpl
        b = 1.0 / 25.0
        expected_pyruvate = np.exp(-a * duration)
        expected_lactate = (
            kpl * (np.exp(-b * duration) - np.exp(-a * duration)) / (a - b)
            if kpl
            else 0.0
        )
        assert final[0, voxel, 0, 0] == pytest.approx(expected_pyruvate)
        assert final[1, voxel, 0, 0] == pytest.approx(expected_lactate)


def test_time_curve_integrates_linear_step_and_outside_regions():
    linear = TimeCurve((1.0, 3.0), (0.0, 2.0), "linear", "zero")
    step = TimeCurve((1.0, 3.0), (2.0, 4.0), "step", "zero")

    assert linear.integral(0.0, 4.0) == pytest.approx(2.0)
    assert linear.interval_values(0.0, 1.0) == (0.0, 0.0)
    assert step.integral(0.0, 4.0) == pytest.approx(4.0)


def test_two_pool_kinetics_preview_uses_inflow_and_conversion_solver():
    times_s = np.linspace(0.0, 2.0, 101)
    inflow = TimeCurve((0.0, 2.0), (1.0, 1.0), "linear", "zero")

    unconverted = simulate_two_pool_kinetics(
        times_s, (0.0, 0.0), (1e15, 1e15), 0.0, inflow
    )
    converted = simulate_two_pool_kinetics(
        times_s, (0.0, 0.0), (1e15, 1e15), 0.2, inflow
    )

    assert unconverted[0, -1] == pytest.approx(2.0)
    assert np.all(unconverted[1] == 0.0)
    assert converted[0, -1] < unconverted[0, -1]
    assert converted[1, -1] > 0.0
    assert converted[:, -1].sum() == pytest.approx(2.0, rel=1e-12)


def test_inflow_populates_initially_empty_voxel_with_analytic_t1_decay():
    shape = (1, 1, 1)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.01, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 1.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.zeros(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.zeros(shape),
        pyruvate_inflow=PyruvateInflow(
            TimeCurve((0.0, 2.0), (1.0, 1.0), "linear", "zero"),
            np.ones(shape),
        ),
        nucleus="C13",
    )

    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        SequenceProgram((), duration_s=2.0), phantom
    )

    expected = 30.0 * (1.0 - np.exp(-2.0 / 30.0))
    assert phantom.n_active == 1
    assert result.final_pool_magnetization[0, 0, 0, 0, 2] == pytest.approx(expected)
    assert result.final_pool_magnetization[1, 0, 0, 0, 2] == pytest.approx(0.0)


def test_linear_inflow_and_conversion_conserve_added_mass_without_relaxation():
    shape = (1, 1, 1)
    phantom = DynamicSpectralPhantom(
        shape=shape,
        fov=(0.01, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 1e15, 1.0),
            ChemicalSpecies("Lactate", 12.0, 1e15, 1.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.zeros(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.full(shape, 0.2),
        pyruvate_inflow=PyruvateInflow(
            TimeCurve((0.0, 2.0), (0.0, 2.0), "linear", "zero"),
            np.ones(shape),
        ),
        nucleus="C13",
    )

    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        SequenceProgram((), duration_s=2.0), phantom
    )

    final_mass = result.final_pool_magnetization[..., 2].sum()
    assert final_mass == pytest.approx(2.0, rel=1e-12)


def test_dynamic_b0_uses_integrated_frequency_phase():
    shape = (1, 1, 1)
    kwargs = dict(
        shape=shape,
        fov=(0.01, 0.01, 0.01),
        pools=(
            ChemicalSpecies("Pyruvate", 0.0, 30.0, 10.0),
            ChemicalSpecies("Lactate", 12.0, 25.0, 10.0),
        ),
        initial_concentration_maps={
            "Pyruvate": np.ones(shape),
            "Lactate": np.zeros(shape),
        },
        kpl_map_s_inv=np.zeros(shape),
        nucleus="C13",
    )
    reference = DynamicSpectralPhantom(**kwargs)
    dynamic = DynamicSpectralPhantom(
        **kwargs,
        dynamic_b0=DynamicB0(
            TimeCurve((0.0, 1.0), (0.0, 7.0), "linear", "hold"),
            np.ones(shape),
        ),
    )
    program = SequenceProgram(
        (RFEvent(0.0, np.asarray([250000.0]), 1e-6), ADCEvent(1.0, 1, 1e-3)),
        duration_s=1.001,
    )
    simulator = BlochSimulator(use_parallel=False)

    reference_signal = simulator.simulate_dynamic_sequence(program, reference).signal[0]
    dynamic_signal = simulator.simulate_dynamic_sequence(program, dynamic).signal[0]
    phase_integral = dynamic.dynamic_b0.offset_curve_hz.integral(0.5e-6, 1.0)

    assert dynamic_signal / reference_signal == pytest.approx(
        np.exp(-2j * np.pi * phase_integral), rel=1e-10, abs=1e-10
    )


def test_dynamic_sequence_returns_pool_resolved_adc_and_xarray():
    phantom = _dynamic_phantom()
    program = SequenceProgram(
        (
            RFEvent(0.0, np.asarray([250.0]), 1e-3),
            ADCEvent(2e-3, 2, 1e-3),
        ),
        duration_s=4e-3,
    )
    result = BlochSimulator(use_parallel=False).simulate_dynamic_sequence(
        program, phantom
    )

    assert result.species_signal.shape == (2, 2)
    assert np.allclose(result.signal, result.species_signal.sum(axis=0))
    dataset = result.to_xarray()
    assert dataset.species_signal.dims == ("pool", "adc")
    assert list(dataset.pool.values) == ["Pyruvate", "Lactate"]


@pytest.mark.parametrize("with_drivers", [False, True])
@pytest.mark.parametrize(
    "kpl",
    [
        (0.0, 0.1),
        (1.0 / 25.0 - 1.0 / 30.0, 0.1),
        (1.0 / 25.0 - 1.0 / 30.0,) * 2,
    ],
    ids=("regular", "mixed-rates", "equal-rates"),
)
def test_optimized_dynamic_kernel_is_bitwise_equal_to_reference(with_drivers, kpl):
    phantom = _dynamic_phantom(kpl)
    raster = 20e-6
    if with_drivers:
        phantom.pyruvate_inflow = PyruvateInflow(
            TimeCurve(
                (0.0, 6 * raster, 12 * raster),
                (0.0, 0.2, 0.1),
                "linear",
                "hold",
            ),
            np.ones(phantom.shape),
        )
        phantom.dynamic_b0 = DynamicB0(
            TimeCurve(
                (0.0, 6 * raster, 12 * raster),
                (-3.0, 5.0, 2.0),
                "linear",
                "hold",
            ),
            np.linspace(0.5, 1.0, phantom.nvoxels).reshape(phantom.shape),
            (1.0, 0.8),
        )
    program = SequenceProgram(
        (
            RFEvent(
                2 * raster,
                np.asarray([80.0 + 20.0j, 120.0 - 10.0j, 60.0 + 5.0j, 0.0j]),
                raster,
            ),
            GradientEvent(
                "x",
                0.0,
                np.linspace(-150.0, 200.0, 12),
                raster,
            ),
            ADCEvent(0.0, 6, 2 * raster, phase_offset_rad=0.37),
        ),
        duration_s=12 * raster,
    )
    simulator = BlochSimulator(use_parallel=True, num_threads=2)
    kwargs = {
        "checkpoints_s": (0.0, 7 * raster, 12 * raster),
        "simulation_timestep_s": 5e-6,
    }

    reference = simulator.simulate_dynamic_sequence(
        program,
        phantom,
        sequence_kernel="reference",
        **kwargs,
    )
    optimized = simulator.simulate_dynamic_sequence(
        program,
        phantom,
        sequence_kernel="optimized",
        **kwargs,
    )

    assert optimized.metadata["sequence_kernel"] == "optimized"
    for name in (
        "signal",
        "species_signal",
        "final_magnetization",
        "final_pool_magnetization",
        "checkpoint_magnetization",
        "checkpoint_pool_magnetization",
    ):
        assert np.array_equal(getattr(optimized, name), getattr(reference, name))


@pytest.mark.parametrize("suffix", [".npz", ".h5", ".nc"])
def test_dynamic_phantom_round_trip(tmp_path, suffix):
    phantom = _dynamic_phantom()
    path = phantom.save(tmp_path / f"dynamic{suffix}")
    loaded = DynamicSpectralPhantom.load(path)

    assert loaded.shape == phantom.shape
    assert [pool.name for pool in loaded.pools] == ["Pyruvate", "Lactate"]
    assert np.array_equal(loaded.kpl_map_s_inv, phantom.kpl_map_s_inv)
    assert loaded.coordinate_system == "object_xyz"
    assert np.array_equal(loaded.affine_ijk_to_xyz_m, phantom.affine_ijk_to_xyz_m)


@pytest.mark.parametrize("suffix", [".npz", ".h5", ".nc"])
def test_dynamic_driver_round_trip(tmp_path, suffix):
    phantom = _dynamic_phantom()
    phantom.pyruvate_inflow = PyruvateInflow(
        TimeCurve((0.0, 2.0), (0.0, 0.4), "linear", "zero"),
        np.ones(phantom.shape),
    )
    phantom.dynamic_b0 = DynamicB0(
        TimeCurve((0.0, 2.0), (-3.0, 4.0), "linear", "hold"),
        np.linspace(0.5, 1.0, phantom.nvoxels).reshape(phantom.shape),
    )

    loaded = DynamicSpectralPhantom.load(phantom.save(tmp_path / f"drivers{suffix}"))

    assert (
        loaded.pyruvate_inflow.rate_curve_s_inv
        == phantom.pyruvate_inflow.rate_curve_s_inv
    )
    assert np.array_equal(
        loaded.pyruvate_inflow.delivery_map, phantom.pyruvate_inflow.delivery_map
    )
    assert loaded.dynamic_b0.offset_curve_hz == phantom.dynamic_b0.offset_curve_hz
    assert np.array_equal(
        loaded.dynamic_b0.spatial_scale_map, phantom.dynamic_b0.spatial_scale_map
    )


def test_dynamic_phantom_xarray_dataset_labels_pools_and_coordinates():
    phantom = _dynamic_phantom()
    dataset = phantom.to_xarray()

    assert dataset["initial_concentration"].dims == ("species", "x", "y", "z")
    assert dataset["kpl_map_s_inv"].dims == ("x", "y", "z")
    assert list(dataset.coords["species"].values) == ["Pyruvate", "Lactate"]
    assert dataset.attrs["coordinate_system"] == "object_xyz"
    assert np.asarray(dataset.coords["x"]) == pytest.approx([-0.005, 0.005])


def test_phantom_design_builds_dynamic_pool_maps_and_kpl_regions():
    design = PhantomDesign(
        shape=(4, 2, 1),
        fov_m=(0.04, 0.02, 0.01),
        dynamic_enabled=True,
        shapes=[
            ShapeDefinition(
                "Object",
                kind="box",
                size=(1.0, 1.0, 1.0),
                t1_s=30.0,
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 1.0, 0.0, 1.0),
                    SpectralPeakDefinition("Lactate", 0.0, 12.0, 1.0),
                ],
            )
        ],
        kinetic_regions=[
            KineticRegionDefinition(
                "Tumor", "box", (0.75, 0.5, 0.5), (0.5, 1.0, 1.0), 0.08
            )
        ],
    )
    phantom = design.build()

    assert isinstance(phantom, DynamicSpectralPhantom)
    assert np.array_equal(phantom.kpl_map_s_inv[:, 0, 0], [0.0, 0.0, 0.08, 0.08])
    assert PhantomDesign.from_phantom(phantom).dynamic_enabled


def test_phantom_design_uses_metabolite_specific_t1_for_dynamic_pools():
    design = PhantomDesign(
        shape=(1, 1, 1),
        fov_m=(0.01, 0.01, 0.01),
        dynamic_enabled=True,
        shapes=[
            ShapeDefinition(
                "Object",
                kind="box",
                size=(1.0, 1.0, 1.0),
                t1_s=99.0,
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 1.0, 0.0, 1.0, t1_s=30.0),
                    SpectralPeakDefinition("Lactate", 0.0, 12.0, 1.0, t1_s=25.0),
                ],
            )
        ],
    )

    phantom = design.build()

    assert [pool.t1 for pool in phantom.pools] == pytest.approx([30.0, 25.0])


def test_phantom_design_builds_inflow_and_dynamic_b0_drivers():
    design = PhantomDesign(
        shape=(2, 2, 1),
        fov_m=(0.02, 0.02, 0.01),
        dynamic_enabled=True,
        pyruvate_inflow_curve=TimeCurve((0.0, 2.0), (0.0, 0.2), "linear", "zero"),
        dynamic_b0_curve=TimeCurve((0.0, 2.0), (0.0, 5.0), "linear", "hold"),
        shapes=[
            ShapeDefinition(
                "Object",
                kind="box",
                size=(1.0, 1.0, 1.0),
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 0.0, 0.0, 1.0),
                    SpectralPeakDefinition("Lactate", 0.0, 12.0, 1.0),
                ],
            )
        ],
    )

    phantom = design.build()
    restored = PhantomDesign.from_phantom(phantom)

    assert phantom.pyruvate_inflow is not None
    assert np.all(phantom.pyruvate_inflow.delivery_map == 1.0)
    assert phantom.n_active == phantom.nvoxels
    assert phantom.dynamic_b0 is not None
    assert restored.pyruvate_inflow_curve == design.pyruvate_inflow_curve
    assert restored.dynamic_b0_curve == design.dynamic_b0_curve


def test_phantom_designer_exposes_kinetic_regions_and_preview():
    app = QApplication.instance() or QApplication([])
    design = PhantomDesign(
        shape=(4, 2, 1),
        fov_m=(0.04, 0.02, 0.01),
        dynamic_enabled=True,
        shapes=[
            ShapeDefinition(
                "Object",
                kind="box",
                size=(1.0, 1.0, 1.0),
                t1_s=30.0,
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 1.0, 0.0, 1.0),
                    SpectralPeakDefinition("Lactate", 0.0, 12.0, 1.0),
                ],
            )
        ],
    )
    dialog = SpectralPhantomDesignerDialog(design=design)
    dialog._add_kinetic_region("ellipsoid")
    dialog.inflow_enabled.setChecked(True)
    dialog.dynamic_b0_enabled.setChecked(True)
    dialog._preview()

    assert isinstance(dialog.phantom, DynamicSpectralPhantom)
    assert dialog.inspector.map_combo.currentText() == "kPL"
    assert dialog.phantom.kinetic_regions[0].kpl_s_inv == pytest.approx(0.05)
    assert dialog.phantom.pyruvate_inflow is not None
    assert dialog.phantom.dynamic_b0 is not None
    dialog.close()
    app.processEvents()


def test_phantom_designer_updates_conversion_plot_live_when_kpl_changes():
    app = QApplication.instance() or QApplication([])
    design = PhantomDesign(
        shape=(1, 1, 1),
        fov_m=(0.01, 0.01, 0.01),
        dynamic_enabled=True,
        default_kpl_s_inv=0.02,
        pyruvate_inflow_curve=TimeCurve(
            (0.0, 2.0, 4.0), (0.0, 1.0, 0.0), "linear", "zero"
        ),
        shapes=[
            ShapeDefinition(
                "Object",
                kind="box",
                size=(1.0, 1.0, 1.0),
                initial_mz=1.0,
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 1.0, 0.0, 1.0, t1_s=30.0),
                    SpectralPeakDefinition("Lactate", 0.0, 12.0, 1.0, t1_s=25.0),
                ],
            )
        ],
    )
    dialog = SpectralPhantomDesignerDialog(design=design)
    dialog.kinetics_preview_duration.setValue(10.0)
    _, lactate_low = dialog.lactate_preview_curve.getData()

    dialog.default_kpl.setValue(0.2)
    _, lactate_high = dialog.lactate_preview_curve.getData()
    _, inflow_values = dialog.inflow_preview_curve.getData()

    assert lactate_high[-1] > lactate_low[-1]
    assert np.max(inflow_values) == pytest.approx(1.0)
    assert "kPL=0.2" in dialog.kinetics_preview_info.text()

    dialog._add_kinetic_region("ellipsoid")
    dialog.kinetic_table.item(0, 8).setText("0.4")
    assert dialog.kinetics_preview_region.currentData() == 0
    assert "kPL=0.4" in dialog.kinetics_preview_info.text()
    dialog.close()
    app.processEvents()


def test_phantom_designer_explains_overlapping_pool_curves_and_shape_context():
    app = QApplication.instance() or QApplication([])
    design = PhantomDesign(
        shape=(1, 1, 1),
        fov_m=(0.01, 0.01, 0.01),
        dynamic_enabled=True,
        default_kpl_s_inv=0.0,
        shapes=[
            ShapeDefinition(
                "Identical pools",
                initial_mz=1.0,
                t1_s=1.0,
                peaks=[
                    SpectralPeakDefinition("Pyruvate", 1.0, 0.0, 1.0),
                    SpectralPeakDefinition("Lactate", 1.0, 12.0, 1.0),
                ],
            )
        ],
    )

    dialog = SpectralPhantomDesignerDialog(design=design)
    _, pyruvate = dialog.pyruvate_preview_curve.getData()
    _, lactate = dialog.lactate_preview_curve.getData()
    info = dialog.kinetics_preview_info.text()

    assert np.allclose(pyruvate, lactate)
    assert dialog.kinetics_preview_shape.currentText() == "Identical pools"
    assert dialog.kinetics_preview_region.currentText() == "Default kPL"
    assert "kPL=0: no P→L conversion" in info
    assert "HP Mz=1 is an initial normalization" in info
    assert "identical and overlap" in info
    assert "initialized, not created by conversion" in info

    dialog.zero_lactate_button.click()
    _, lactate_zero = dialog.lactate_preview_curve.getData()
    assert design.shapes[0].peaks[1].amplitude == 0.0
    assert np.all(lactate_zero == 0.0)
    assert "Lz starts at zero and remains zero" in dialog.kinetics_preview_info.text()

    dialog.default_kpl.setValue(0.2)
    _, lactate_converted = dialog.lactate_preview_curve.getData()
    assert lactate_converted[0] == 0.0
    assert lactate_converted[-1] > 0.0
    assert "all subsequent Lactate is created" in dialog.kinetics_preview_info.text()
    dialog.close()
    app.processEvents()


def test_phantom_designer_selects_shape_for_representative_kinetics_preview():
    app = QApplication.instance() or QApplication([])
    pool_peaks = [
        SpectralPeakDefinition("Pyruvate", 1.0, 0.0, 1.0, t1_s=30.0),
        SpectralPeakDefinition("Lactate", 0.0, 12.0, 1.0, t1_s=25.0),
    ]
    design = PhantomDesign(
        dynamic_enabled=True,
        shapes=[
            ShapeDefinition("Low Mz", initial_mz=1.0, peaks=list(pool_peaks)),
            ShapeDefinition("High Mz", initial_mz=3.0, peaks=list(pool_peaks)),
        ],
    )
    dialog = SpectralPhantomDesignerDialog(design=design)

    dialog.kinetics_preview_shape.setCurrentIndex(
        dialog.kinetics_preview_shape.findData(1)
    )
    _, pyruvate = dialog.pyruvate_preview_curve.getData()

    assert dialog.shape_list.currentRow() == 1
    assert pyruvate[0] == pytest.approx(3.0)
    assert "Representative voxel in High Mz" in dialog.kinetics_preview_info.text()
    dialog.close()
    app.processEvents()


def test_sequence_worker_dispatches_dynamic_phantom_to_dynamic_solver():
    app = QApplication.instance() or QApplication([])
    phantom = _dynamic_phantom()
    expected = object()
    simulator = MagicMock()
    simulator.simulate_dynamic_sequence.return_value = expected
    worker = SequenceSimulationThread(
        simulator,
        SequenceProgram((), duration_s=1e-3),
        phantom,
        (),
        live_preview=False,
    )
    results = []
    worker.result_ready.connect(results.append)

    worker.run()

    simulator.simulate_dynamic_sequence.assert_called_once()
    assert results == [expected]
    app.processEvents()
