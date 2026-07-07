import numpy as np
import pytest

from blochsimulator import BlochSimulator, TissueParameters
from blochsimulator.notebook_exporter import export_sequence_result_notebook
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import ADCEvent, GradientEvent, RFEvent, SequenceProgram


def _phantom(
    shape=(1,),
    *,
    t1=1.0,
    t2=0.2,
    pd=1.0,
    b0=0.0,
    chemical_shift=0.0,
    m0=(0.0, 0.0, 1.0),
    tx_sensitivity=None,
    rx_sensitivities=None,
):
    m0_map = np.empty(shape + (3,), dtype=float)
    m0_map[...] = m0
    return Phantom(
        shape=shape,
        fov=tuple(0.02 for _ in shape),
        t1_map=np.full(shape, t1),
        t2_map=np.full(shape, t2),
        pd_map=np.full(shape, pd),
        b0_map=np.full(shape, b0),
        chemical_shift_map=np.full(shape, chemical_shift),
        m0_map=m0_map,
        tx_sensitivity_map=tx_sensitivity,
        rx_sensitivity_maps=rx_sensitivities,
    )


def test_free_relaxation_final_and_checkpoints():
    phantom = _phantom(m0=(1.0, 0.0, 0.0))
    program = SequenceProgram((), duration_s=0.1)
    result = BlochSimulator(use_parallel=False).simulate_sequence(
        program, phantom, checkpoints_s=(0.05,)
    )
    assert result.mx.item() == pytest.approx(np.exp(-0.1 / 0.2), rel=1e-10)
    assert result.my.item() == pytest.approx(0.0, abs=1e-12)
    assert result.mz.item() == pytest.approx(1 - np.exp(-0.1 / 1.0), rel=1e-10)
    checkpoint = result.checkpoint_magnetization[0, 0]
    assert checkpoint[0] == pytest.approx(np.exp(-0.05 / 0.2), rel=1e-10)
    assert checkpoint[2] == pytest.approx(1 - np.exp(-0.05), rel=1e-10)


def test_hard_90_degree_pulse_creates_transverse_magnetization():
    phantom = _phantom(t1=1e9, t2=1e9)
    program = SequenceProgram(
        (RFEvent(0.0, np.array([250.0]), 1e-3),),
        duration_s=1e-3,
    )
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    assert np.hypot(result.mx.item(), result.my.item()) == pytest.approx(1.0, abs=1e-8)
    assert result.mz.item() == pytest.approx(0.0, abs=1e-8)


def test_adc_signal_b0_and_chemical_shift_phase():
    phantom = _phantom(
        t1=1e9,
        t2=1e9,
        b0=7.0,
        chemical_shift=3.0,
        m0=(1.0, 0.0, 0.0),
    )
    program = SequenceProgram(
        (ADCEvent(0.0, 2, 0.025),),
        duration_s=0.05,
    )
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    assert result.signal[0] == pytest.approx(1 + 0j, abs=1e-10)
    assert result.signal[1] == pytest.approx(
        np.exp(-1j * 2 * np.pi * 10.0 * 0.025), abs=1e-8
    )


def test_receiver_demodulation_phase():
    phantom = _phantom(t1=1e9, t2=1e9, m0=(1.0, 0.0, 0.0))
    program = SequenceProgram(
        (ADCEvent(0.0, 1, 1e-3, phase_offset_rad=np.pi / 2),),
        duration_s=1e-3,
    )
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    assert result.signal[0] == pytest.approx(-1j, abs=1e-10)


def test_sequence_preview_reports_cumulative_signal_and_finishes_at_one():
    phantom = _phantom(shape=(4,), m0=(1.0, 0.0, 0.0))
    program = SequenceProgram((ADCEvent(0.0, 2, 0.001),), duration_s=0.002)
    previews = []

    result = BlochSimulator(use_parallel=False).simulate_sequence(
        program,
        phantom,
        chunk_voxels=2,
        preview_callback=lambda fraction, signal: previews.append(
            (fraction, np.array(signal, copy=True))
        ),
    )

    assert [fraction for fraction, _ in previews] == pytest.approx([0.5, 1.0])
    assert previews[-1][1] == pytest.approx(result.signal)


def test_single_tx_map_scales_local_rf_amplitude_and_phase():
    phantom = _phantom(
        t1=1e9,
        t2=1e9,
        tx_sensitivity=np.array([0.0 + 0.5j]),
    )
    program = SequenceProgram(
        (RFEvent(0.0, np.array([250.0]), 1e-3),),
        duration_s=1e-3,
    )
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    assert result.mx.item() == pytest.approx(np.sin(np.pi / 4), abs=1e-8)
    assert result.my.item() == pytest.approx(0.0, abs=1e-8)
    assert result.mz.item() == pytest.approx(np.cos(np.pi / 4), abs=1e-8)


def test_multi_rx_maps_return_independent_coil_signals():
    phantom = _phantom(
        t1=1e9,
        t2=1e9,
        m0=(1.0, 0.0, 0.0),
        rx_sensitivities=np.array([[1.0 + 0j], [0.0 + 2.0j]]),
    )
    program = SequenceProgram((ADCEvent(0.0, 1, 1e-3),), duration_s=1e-3)
    simulator = BlochSimulator(use_parallel=False)
    result = simulator.simulate_sequence(program, phantom)
    assert result.signal.shape == (2, 1)
    assert result.signal[:, 0] == pytest.approx([1.0 + 0j, 0.0 + 2.0j])
    dataset = result.to_xarray()
    assert dataset["signal"].dims == ("coil", "adc")
    assert dataset.sizes["coil"] == 2
    simulator_dataset = simulator.get_results_as_xarray()
    assert simulator_dataset["signal"].dims == ("coil", "adc")


def test_gradient_phase_is_generated_once_by_bloch_kernel():
    phantom = _phantom(t1=1e9, t2=1e9, m0=(1.0, 0.0, 0.0))
    position = phantom.positions[0, 0]
    gradient = 100.0
    duration = 0.01
    program = SequenceProgram(
        (
            GradientEvent("x", 0.0, np.array([gradient]), duration),
            ADCEvent(duration, 1, 1e-3),
        ),
        duration_s=duration + 1e-3,
    )
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    expected = np.exp(-1j * 2 * np.pi * gradient * position * duration)
    assert result.signal[0] == pytest.approx(expected, abs=1e-8)


def test_chunk_size_does_not_change_signal_or_state():
    phantom = _phantom(shape=(2, 2, 2), t1=1e9, t2=1e9, m0=(1.0, 0.0, 0.0))
    program = SequenceProgram(
        (ADCEvent(0.0, 3, 1e-3),),
        duration_s=3e-3,
    )
    simulator = BlochSimulator(use_parallel=True, num_threads=2)
    one = simulator.simulate_sequence(program, phantom, chunk_voxels=1)
    all_at_once = simulator.simulate_sequence(program, phantom, chunk_voxels=8)
    assert np.allclose(one.signal, all_at_once.signal)
    assert np.allclose(one.final_magnetization, all_at_once.final_magnetization)


def test_sequence_simulation_reports_compile_and_chunk_status():
    phantom = _phantom(shape=(2,), t1=1e9, t2=1e9)
    program = SequenceProgram(
        (ADCEvent(0.0, 2, 1e-3),),
        duration_s=2e-3,
    )
    messages = []

    result = BlochSimulator(use_parallel=False).simulate_sequence(
        program,
        phantom,
        chunk_voxels=1,
        simulation_timestep_s=5e-6,
        status_callback=messages.append,
    )

    assert result.metadata["simulation_timestep_s"] == pytest.approx(5e-6)
    assert any("Compiled" in message for message in messages)
    assert any("spin-interval updates" in message for message in messages)
    assert any("Simulating chunk 1/2" in message for message in messages)
    assert any("Simulating chunk 2/2" in message for message in messages)


def test_multi_rx_chunking_and_openmp_do_not_change_signal():
    shape = (2, 2, 2)
    rx = np.stack(
        (
            np.ones(shape, dtype=complex),
            np.full(shape, 0.5 + 0.25j, dtype=complex),
        )
    )
    phantom = _phantom(
        shape=shape,
        t1=1e9,
        t2=1e9,
        m0=(1.0, 0.0, 0.0),
        rx_sensitivities=rx,
    )
    program = SequenceProgram(
        (ADCEvent(0.0, 3, 1e-3),),
        duration_s=3e-3,
    )
    serial = BlochSimulator(use_parallel=False).simulate_sequence(
        program, phantom, chunk_voxels=1
    )
    parallel = BlochSimulator(use_parallel=True, num_threads=2).simulate_sequence(
        program, phantom, chunk_voxels=8
    )
    assert serial.signal.shape == (2, 3)
    assert np.allclose(serial.signal, parallel.signal)
    assert np.allclose(serial.final_magnetization, parallel.final_magnetization)


def test_invalid_active_relaxation_rejected():
    phantom = _phantom(t1=0.0)
    with pytest.raises(ValueError, match="T1 > 0"):
        BlochSimulator(use_parallel=False).simulate_sequence(
            SequenceProgram((), duration_s=0.0), phantom
        )


def test_sparse_result_xarray_and_hdf5_export(tmp_path):
    phantom = _phantom(t1=1e9, t2=1e9, m0=(1.0, 0.0, 0.0))
    program = SequenceProgram(
        (ADCEvent(0.0, 2, 1e-3),),
        duration_s=2e-3,
    )
    simulator = BlochSimulator(use_parallel=False)
    result = simulator.simulate_sequence(program, phantom, checkpoints_s=(1e-3,))
    dataset = result.to_xarray()
    assert dataset.sizes["adc"] == 2
    assert dataset.sizes["checkpoint"] == 1
    assert dataset["adc_gradient_moment_cyc_per_m"].shape == (2, 3)
    assert np.array_equal(dataset["t"], result.adc_times_s)
    assert np.array_equal(dataset["kx"], [0.0, 0.0])
    assert np.array_equal(dataset["ky"], [0.0, 0.0])
    assert np.array_equal(dataset["kz"], [0.0, 0.0])
    assert dataset["kx"].attrs["units"] == "cycles/m"
    simulator_dataset = simulator.get_results_as_xarray()
    assert simulator_dataset.sizes["adc"] == 2

    filename = tmp_path / "sequence_result.h5"
    simulator.save_results(filename)
    import h5py

    with h5py.File(filename, "r") as handle:
        assert handle["signal"].shape == (2,)
        assert handle["adc_gradient_moment_cyc_per_m"].shape == (2, 3)
        assert handle["final_magnetization"].shape == (1, 3)
        assert handle["checkpoint_magnetization"].shape == (1, 1, 3)

    loaded = BlochSimulator(use_parallel=False)
    loaded.load_results(filename)
    assert np.array_equal(loaded.last_result["signal"], result.signal)
    assert np.array_equal(
        loaded.last_result["adc_gradient_moment_cyc_per_m"],
        result.adc_gradient_moment_cyc_per_m,
    )


def test_outer_acquisition_indices_are_exposed_per_adc_sample():
    phantom = _phantom(t1=1e9, t2=1e9, m0=(1.0, 0.0, 0.0))
    program = SequenceProgram(
        (ADCEvent(0.0, 2, 1e-3), ADCEvent(3e-3, 1, 1e-3)),
        duration_s=4e-3,
        metadata={
            "adc_label_values": {
                "SLC": (2, 3),
                "ECO": (0, 1),
                "REP": (4, 4),
            }
        },
    )

    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    dimensions = result.acquisition_dimensions
    assert dimensions.source == "pulseq_labels"
    assert dimensions.varying_axes == ("slice", "echo")
    dataset = result.to_xarray()
    assert np.array_equal(dataset["slice_index"], [2, 2, 3])
    assert np.array_equal(dataset["echo_index"], [0, 0, 1])
    assert np.array_equal(dataset["repetition_index"], [4, 4, 4])
    assert np.array_equal(dataset["adc_event_index"], [0, 0, 1])
    assert np.array_equal(dataset["readout_sample_index"], [0, 1, 0])


@pytest.mark.parametrize("suffix", [".npz", ".h5", ".nc"])
def test_sequence_result_export_formats(tmp_path, suffix):
    phantom = _phantom(t1=1e9, t2=1e9, m0=(1.0, 0.0, 0.0))
    program = SequenceProgram((ADCEvent(0.0, 2, 1e-3),), duration_s=2e-3)
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    path = result.save(tmp_path / f"result{suffix}")

    assert path.is_file()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            assert np.array_equal(data["signal"], result.signal)
    elif suffix == ".h5":
        import h5py

        with h5py.File(path, "r") as handle:
            assert np.array_equal(handle["signal"][...], result.signal)
    else:
        import xarray as xr

        with xr.open_dataset(path) as dataset:
            signal = dataset.signal_real + 1j * dataset.signal_imag
            assert np.array_equal(signal, result.signal)
            assert np.array_equal(dataset.t, result.adc_times_s)
            assert all(axis in dataset.coords for axis in ("kx", "ky", "kz"))


def test_sequence_result_notebook_uses_xarray_dataset(tmp_path):
    phantom = _phantom(t1=1e9, t2=1e9, m0=(1.0, 0.0, 0.0))
    program = SequenceProgram((ADCEvent(0.0, 1, 1e-3),), duration_s=1e-3)
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    data_path = result.save(tmp_path / "result.nc")
    notebook_path = export_sequence_result_notebook(
        str(tmp_path / "analysis.ipynb"), str(data_path)
    )

    text = notebook_path.read_text(encoding="utf-8")
    assert "xr.open_dataset" in text
    assert "result.nc" in text


def test_phantom_split_maps_round_trip_npz_and_hdf5(tmp_path):
    phantom = _phantom(
        b0=11.0,
        chemical_shift=-4.0,
        tx_sensitivity=np.array([0.8 + 0.1j]),
        rx_sensitivities=np.array([[1.0 + 0j], [0.5 - 0.25j]]),
    )
    for suffix in (".npz", ".h5"):
        filename = tmp_path / f"phantom{suffix}"
        phantom.save(filename)
        loaded = Phantom.load(filename)
        assert np.array_equal(loaded.b0_map, phantom.b0_map)
        assert np.array_equal(loaded.chemical_shift_map, phantom.chemical_shift_map)
        assert np.array_equal(loaded.effective_df_map, phantom.effective_df_map)
        assert np.array_equal(loaded.tx_sensitivity_map, phantom.tx_sensitivity_map)
        assert np.array_equal(loaded.rx_sensitivity_maps, phantom.rx_sensitivity_maps)


def test_legacy_adapter_matches_existing_endpoint_solver():
    dt = 10e-6
    time = np.arange(20) * dt
    b1 = np.zeros(20, dtype=complex)
    b1[:10] = 0.01 + 0.002j
    gradients = np.zeros((20, 3))
    gradients[10:, 2] = 0.1
    tissue = TissueParameters("reference", t1=1.0, t2=0.1)
    position = np.array([[0.0, 0.0, 0.003]])
    legacy = BlochSimulator(use_parallel=False).simulate(
        (b1, gradients, time),
        tissue,
        positions=position,
        frequencies=np.array([17.0]),
        mode=0,
    )
    phantom = Phantom(
        shape=(1,),
        fov=(0.01,),
        t1_map=np.array([tissue.t1]),
        t2_map=np.array([tissue.t2]),
        b0_map=np.array([17.0]),
    )
    # Use the identical point coordinate as the legacy reference.
    phantom.positions[0] = position[0]
    program = SequenceProgram.from_legacy(b1, gradients, time)
    streamed = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    assert streamed.mx.item() == pytest.approx(legacy["mx"].item(), abs=1e-10)
    assert streamed.my.item() == pytest.approx(legacy["my"].item(), abs=1e-10)
    assert streamed.mz.item() == pytest.approx(legacy["mz"].item(), abs=1e-10)
