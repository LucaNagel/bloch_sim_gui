from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QWidget

from blochsimulator import BlochSimulator
from blochsimulator.phantom_design import (
    PhantomDesign,
    ShapeDefinition,
    SpectralPeakDefinition,
)
from blochsimulator.sequence import ADCEvent, RFEvent, SequenceProgram
from blochsimulator.spectral_phantom import SpectralPhantom
from blochsimulator.ui.phantom_designer import SpectralPhantomDesignerDialog
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget
from blochsimulator.ui.volume_viewer import VolumeViewerWidget


def _spectral_design():
    return PhantomDesign(
        name="Two peak phantom",
        shape=(6, 6, 4),
        fov_m=(0.06, 0.06, 0.004),
        shapes=[
            ShapeDefinition(
                name="Object",
                kind="box",
                center=(0.5, 0.5, 0.5),
                size=(0.6, 0.6, 1.0),
                t1_s=1.2,
                b0_hz=10.0,
                peaks=[
                    SpectralPeakDefinition("Water", 1.0, -100.0, 0.020),
                    SpectralPeakDefinition("Metabolite", 0.25, 80.0, 0.010),
                ],
            )
        ],
    )


def test_shape_design_builds_lorentzian_components():
    phantom = _spectral_design().build()
    assert phantom.shape == (6, 6, 4)
    assert phantom.n_species == 2
    assert phantom.n_active > 0
    assert np.allclose(phantom.positions.mean(axis=0), 0.0)

    species = phantom.species[0]
    centre = species.frequency_offset_hz + 10.0
    half_width = 1.0 / (2 * np.pi * species.t2_star)
    frequency = np.asarray([centre, centre + half_width])
    _, spectrum = phantom.spectrum_at((3, 3, 2), frequency_hz=frequency)
    # The distant second peak contributes slightly at both points.
    assert spectrum[0] == pytest.approx(1.0, rel=2e-3)
    assert spectrum[1] == pytest.approx(0.5, rel=1e-2)


@pytest.mark.parametrize("suffix", [".npz", ".h5"])
def test_spectral_phantom_round_trip_preserves_design(tmp_path, suffix):
    phantom = _spectral_design().build()
    path = tmp_path / f"spectral{suffix}"
    phantom.save(path)
    loaded = SpectralPhantom.load(path)

    assert loaded.shape == phantom.shape
    assert loaded.fov == phantom.fov
    assert [item.name for item in loaded.species] == [
        item.name for item in phantom.species
    ]
    assert PhantomDesign.from_phantom(loaded).to_dict() == _spectral_design().to_dict()
    for name in phantom.concentration_maps:
        assert np.array_equal(
            loaded.concentration_maps[name], phantom.concentration_maps[name]
        )


def test_spectral_sequence_signal_is_sum_of_independent_components():
    phantom = _spectral_design().build()
    program = SequenceProgram(
        events=(
            RFEvent(0.0, np.asarray([250.0]), 1e-3),
            ADCEvent(1.05e-3, 5, 100e-6),
        ),
        duration_s=1.5e-3,
    )
    simulator = BlochSimulator(use_parallel=False)
    result = simulator.simulate_spectral_sequence(program, phantom)
    component_results = [
        simulator.simulate_sequence(program, component)
        for _, component in phantom.to_component_phantoms()
    ]

    assert np.allclose(result.signal, sum(item.signal for item in component_results))
    assert result.final_magnetization.shape == phantom.shape + (3,)
    assert result.metadata["spectral_component_count"] == 2


def test_designer_and_sequence_workspace_accept_spectral_phantom():
    app = QApplication.instance() or QApplication([])
    dialog = SpectralPhantomDesignerDialog(design=_spectral_design())
    dialog._preview()
    assert dialog.phantom.n_species == 2
    assert dialog.inspector.volume.data.shape == dialog.phantom.shape

    host = QWidget()
    host.phantom_widget = SimpleNamespace(current_phantom=dialog.phantom)
    widget = SequenceSimulationWidget(host)
    widget._build_phantom()
    assert widget.phantom is dialog.phantom
    dialog.close()
    host.close()
    app.processEvents()


def test_volume_viewer_normalizes_2d_mask_and_resets_stale_indices():
    app = QApplication.instance() or QApplication([])
    viewer = VolumeViewerWidget()

    viewer.set_volume(np.ones((64, 8, 4)), mask=np.ones((64, 8, 4), dtype=bool))
    viewer.sliders[0].setValue(48)

    data = np.arange(64.0)[None, :]
    viewer.set_volume(data, mask=np.ones(data.shape, dtype=bool))

    assert viewer.data.shape == (1, 64, 1)
    assert viewer.mask.shape == viewer.data.shape
    assert viewer.indices == (0, 32, 0)
    viewer._indices_updated()
    viewer.close()
    app.processEvents()
