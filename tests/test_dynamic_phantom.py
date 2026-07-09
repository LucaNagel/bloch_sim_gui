import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from unittest.mock import MagicMock

from blochsimulator import BlochSimulator
from blochsimulator.dynamic_phantom import (
    DynamicSpectralPhantom,
    KineticRegionDefinition,
    rasterize_kpl_regions,
)
from blochsimulator.phantom_design import (
    PhantomDesign,
    ShapeDefinition,
    SpectralPeakDefinition,
)
from blochsimulator.sequence import ADCEvent, RFEvent, SequenceProgram
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
    dialog._preview()

    assert isinstance(dialog.phantom, DynamicSpectralPhantom)
    assert dialog.inspector.map_combo.currentText() == "kPL"
    assert dialog.phantom.kinetic_regions[0].kpl_s_inv == pytest.approx(0.05)
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
