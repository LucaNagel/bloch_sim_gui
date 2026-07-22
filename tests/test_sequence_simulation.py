import numpy as np
import pytest

from blochsimulator import BlochSimulator, TissueParameters
from blochsimulator.notebook_exporter import export_sequence_result_notebook
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import (
    ADCEvent,
    BrukerExportOptions,
    GradientEvent,
    RFEvent,
    SequenceProgram,
    SequenceSimulationResult,
    export_bruker_raw,
)
from blochsimulator.simulator import resolve_num_threads


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


def test_automatic_thread_count_uses_available_logical_processors(monkeypatch):
    monkeypatch.setattr("blochsimulator.simulator.os.cpu_count", lambda: 12)

    assert resolve_num_threads(None) == 12
    assert resolve_num_threads(0) == 12
    assert resolve_num_threads(3) == 3
    assert BlochSimulator().num_threads == 12
    with pytest.raises(ValueError, match="num_threads"):
        resolve_num_threads(-1)


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


def test_sequence_spectral_probe_tracks_frequency_axis():
    program = SequenceProgram((), duration_s=0.01)
    frequencies = np.array([-100.0, 0.0, 100.0])
    checkpoints = np.array([0.0, 0.005, 0.01])

    result = BlochSimulator(use_parallel=False).simulate_sequence_probes(
        program,
        positions_m=np.array([[0.0, 0.0, 0.0]]),
        frequency_offsets_hz=frequencies,
        checkpoints_s=checkpoints,
        t1_s=1e9,
        t2_s=1e9,
        initial_magnetization=(1.0, 0.0, 0.0),
    )

    assert result.magnetization.shape == (3, 1, 3, 3)
    assert np.allclose(result.positions_m, [[0.0, 0.0, 0.0]])
    assert result.frequency_offsets_hz == pytest.approx(frequencies)
    transverse = result.mx[:, 0, :] + 1j * result.my[:, 0, :]
    expected = np.exp(-1j * 2 * np.pi * checkpoints[:, None] * frequencies[None, :])
    assert transverse == pytest.approx(expected, abs=1e-8)
    assert result.metadata["probe_type"] == "spectral"
    dataset = result.to_xarray()
    assert dataset["magnetization"].dims == (
        "time",
        "position",
        "frequency",
        "component",
    )


def test_sequence_probe_accepts_large_initial_magnetization():
    program = SequenceProgram((), duration_s=0.0)
    initial_mz = 1e7

    result = BlochSimulator(use_parallel=False).simulate_sequence_probes(
        program,
        positions_m=np.array([[0.0, 0.0, 0.0]]),
        frequency_offsets_hz=np.array([0.0]),
        checkpoints_s=(0.0,),
        t1_s=1e9,
        t2_s=1e9,
        initial_magnetization=(0.0, 0.0, initial_mz),
    )

    assert result.mz[0, 0, 0] == pytest.approx(initial_mz)
    assert result.metadata["initial_magnetization"] == pytest.approx(
        [0.0, 0.0, initial_mz]
    )


def test_sequence_geometry_probe_preserves_explicit_positions():
    program = SequenceProgram(
        (GradientEvent("x", 0.0, np.array([100.0]), 0.01),),
        duration_s=0.01,
    )
    positions = np.array([[-0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])

    result = BlochSimulator(use_parallel=False).simulate_sequence_probes(
        program,
        positions_m=positions,
        frequency_offsets_hz=np.array([0.0]),
        checkpoints_s=(0.01,),
        t1_s=1e9,
        t2_s=1e9,
        initial_magnetization=(1.0, 0.0, 0.0),
    )

    assert result.magnetization.shape == (1, 3, 1, 3)
    assert np.allclose(result.positions_m, positions)
    transverse = result.mx[0, :, 0] + 1j * result.my[0, :, 0]
    expected = np.exp(-1j * 2 * np.pi * 100.0 * positions[:, 0] * 0.01)
    assert transverse == pytest.approx(expected, abs=1e-8)
    assert result.metadata["probe_type"] == "geometry"
    assert result.coherent_mxy[0, 0] == pytest.approx(np.mean(expected), abs=1e-8)
    assert result.coherent_mxy_magnitude[0, 0] == pytest.approx(
        abs(np.mean(expected)), abs=1e-8
    )
    dataset = result.to_xarray()
    assert dataset["coherent_mxy_magnitude"].dims == ("time", "frequency")


@pytest.mark.parametrize("relaxation_model", ["uniform", "discrete", "continuous"])
def test_optimized_sequence_kernel_matches_reference(relaxation_model):
    """Keep the optimized native path tied to the reference implementation."""
    rng = np.random.default_rng(20260708)
    shape = (5, 4, 4)
    t1 = np.full(shape, 1.1)
    t2 = np.full(shape, 0.09)
    if relaxation_model == "discrete":
        labels = np.arange(np.prod(shape)).reshape(shape) % 4
        t1 = np.choose(labels, [0.7, 0.9, 1.2, 1.6])
        t2 = np.choose(labels, [0.05, 0.07, 0.1, 0.14])
    elif relaxation_model == "continuous":
        t1 += rng.uniform(-0.2, 0.2, shape)
        t2 += rng.uniform(-0.01, 0.01, shape)
    m0 = rng.normal(size=shape + (3,))
    m0 /= np.linalg.norm(m0, axis=-1, keepdims=True)
    rx = rng.normal(size=(2,) + shape) + 1j * rng.normal(size=(2,) + shape)
    phantom = Phantom(
        shape=shape,
        fov=(0.04, 0.03, 0.02),
        t1_map=t1,
        t2_map=t2,
        pd_map=rng.uniform(0.3, 1.0, shape),
        b0_map=rng.uniform(-40.0, 40.0, shape),
        chemical_shift_map=rng.uniform(-10.0, 10.0, shape),
        m0_map=m0,
        tx_sensitivity_map=(
            rng.normal(1.0, 0.05, shape) + 1j * rng.normal(0.0, 0.03, shape)
        ),
        rx_sensitivity_maps=rx,
    )
    raster = 20e-6
    interval_count = 24
    program = SequenceProgram(
        (
            RFEvent(
                2 * raster,
                np.array([80 + 20j, 120 - 10j, 60 + 5j, 0j]),
                raster,
            ),
            GradientEvent("x", 0.0, np.linspace(-150.0, 200.0, interval_count), raster),
            GradientEvent("z", 0.0, np.linspace(80.0, -60.0, interval_count), raster),
            ADCEvent(0.0, 12, 2 * raster, phase_offset_rad=0.37),
        ),
        duration_s=interval_count * raster,
    )
    simulator = BlochSimulator(use_parallel=True, num_threads=2)
    reference = simulator.simulate_sequence(
        program,
        phantom,
        checkpoints_s=(0.0, 6 * raster, 13 * raster, interval_count * raster),
        chunk_voxels=np.prod(shape),
        sequence_kernel="reference",
    )
    optimized = simulator.simulate_sequence(
        program,
        phantom,
        checkpoints_s=(0.0, 6 * raster, 13 * raster, interval_count * raster),
        chunk_voxels=np.prod(shape),
        sequence_kernel="optimized",
    )

    assert optimized.metadata["sequence_kernel"] == "optimized"
    assert np.allclose(optimized.signal, reference.signal, rtol=5e-13, atol=5e-13)
    assert np.allclose(
        optimized.final_magnetization,
        reference.final_magnetization,
        rtol=5e-13,
        atol=5e-13,
    )
    assert np.allclose(
        optimized.checkpoint_magnetization,
        reference.checkpoint_magnetization,
        rtol=5e-13,
        atol=5e-13,
    )


def test_invalid_sequence_kernel_is_rejected():
    with pytest.raises(ValueError, match="sequence_kernel"):
        BlochSimulator(sequence_kernel="unknown")

    with pytest.raises(ValueError, match="dynamic_sequence_kernel"):
        BlochSimulator(dynamic_sequence_kernel="unknown")


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


def test_sequence_result_bruker_raw_export_writes_interleaved_fid(tmp_path):
    result = SequenceSimulationResult(
        signal=np.asarray([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128),
        adc_times_s=np.asarray([0.0, 1e-3]),
        final_magnetization=np.zeros((1, 3), dtype=float),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.asarray([], dtype=float),
        metadata={"field_strength_t": 7.0, "nucleus": "H1"},
    )
    program = SequenceProgram((ADCEvent(0.0, 2, 1e-3),), duration_s=2e-3)

    output = export_bruker_raw(
        result, tmp_path / "bruker" / "1", program=program, scale=1000
    )

    assert output.is_dir()
    assert (output / "pdata").is_dir()
    assert not (output / "rawdata.job0").exists()
    fid = np.fromfile(output / "fid", dtype="<i4")
    assert fid.size == 256
    assert np.array_equal(
        fid[:4], np.asarray([1000, 2000, -3000, 4000], dtype=np.int32)
    )
    assert np.count_nonzero(fid[4:]) == 0
    acqp = (output / "acqp").read_text()
    method = (output / "method").read_text()
    assert "##$ACQ_scan_name=<BlochSimulator " in acqp
    assert "##$GO_raw_data_format=GO_32BIT_SGN_INT" in acqp
    assert "##$BYTORDA=little" in acqp
    assert "##$BLOCHSIM_signal_scale=1000" in acqp
    assert "##$PVM_EncNReceivers=1" in method
    assert "##$PVM_EncSpectroscopy=No" in method
    assert "##$Method=<Bruker:RARE>" in method


def test_bruker_export_accepts_method_and_spatial_metadata_overrides(tmp_path):
    result = SequenceSimulationResult(
        signal=np.ones(4, dtype=np.complex128),
        adc_times_s=np.arange(4, dtype=float) * 1e-3,
        final_magnetization=np.zeros((1, 3), dtype=float),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.asarray([], dtype=float),
    )
    program = SequenceProgram((ADCEvent(0.0, 4, 1e-3),), duration_s=4e-3)
    options = BrukerExportOptions(
        method_name="Bruker:FLASH",
        scan_name="custom scan",
        matrix=(64, 32),
        fov_m=(0.06, 0.03),
        slice_thickness_mm=2.5,
    )

    output = export_bruker_raw(
        result, tmp_path / "bruker_custom", program=program, options=options
    )

    acqp = (output / "acqp").read_text()
    method = (output / "method").read_text()
    assert "##$ACQ_scan_name=<custom scan>" in acqp
    assert "##$ACQ_method=<Bruker:FLASH>" in acqp
    assert "##$Method=<Bruker:FLASH>" in method
    assert "##$PVM_Matrix=( 2 )\n64 32" in method
    assert "##$PVM_Fov=( 2 )\n60 30" in method
    assert "##$PVM_SpatResol=( 2 )\n0.9375 0.9375" in method
    assert "##$PVM_SliceThick=2.5" in method


def test_bruker_export_can_write_rawdata_job0_or_both(tmp_path):
    result = SequenceSimulationResult(
        signal=np.asarray([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128),
        adc_times_s=np.asarray([0.0, 1e-3]),
        final_magnetization=np.zeros((1, 3), dtype=float),
        checkpoint_magnetization=None,
        checkpoint_times_s=np.asarray([], dtype=float),
    )
    program = SequenceProgram((ADCEvent(0.0, 2, 1e-3),), duration_s=2e-3)

    raw_only = export_bruker_raw(
        result,
        tmp_path / "raw_only",
        program=program,
        scale=1000,
        options=BrukerExportOptions(raw_data_files="rawdata.job0"),
    )

    assert not (raw_only / "fid").exists()
    assert np.array_equal(
        np.fromfile(raw_only / "rawdata.job0", dtype="<i4"),
        np.asarray([1000, 2000, -3000, 4000], dtype=np.int32),
    )

    both = export_bruker_raw(
        result,
        tmp_path / "both",
        program=program,
        scale=1000,
        options=BrukerExportOptions(raw_data_files="both"),
    )

    assert (both / "fid").is_file()
    assert (both / "rawdata.job0").is_file()


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
    assert "adc_event_index" in text
    assert "cartesian_kspace" in text
    assert "cartesian_image_magnitude" in text
    assert "import ipywidgets as widgets" in text
    assert "x_slider = _index_slider('x', x_dim)" in text
    assert "y_slider = _index_slider('y', y_dim)" in text
    assert "z_slider = _index_slider('z', z_dim)" in text
    assert "repetition_slider = _index_slider('Repetition'" in text
    assert "spectral_point_slider = _index_slider(" in text
    assert "widgets.interactive_output" in text
    assert "continuous_update=True" in text
    import nbformat

    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(notebook_path), "exec")


@pytest.mark.parametrize("suffix", [".npz", ".h5", ".nc"])
def test_phantom_split_maps_round_trip_npz_hdf5_and_xarray(tmp_path, suffix):
    phantom = _phantom(
        shape=(2, 2, 2),
        b0=11.0,
        chemical_shift=-4.0,
        tx_sensitivity=np.full((2, 2, 2), 0.8 + 0.1j),
        rx_sensitivities=np.stack(
            [
                np.ones((2, 2, 2), dtype=np.complex128),
                np.full((2, 2, 2), 0.5 - 0.25j),
            ]
        ),
    )
    filename = tmp_path / f"phantom{suffix}"
    phantom.save(filename)
    loaded = Phantom.load(filename)

    assert np.array_equal(loaded.b0_map, phantom.b0_map)
    assert np.array_equal(loaded.chemical_shift_map, phantom.chemical_shift_map)
    assert np.array_equal(loaded.effective_df_map, phantom.effective_df_map)
    assert np.array_equal(loaded.tx_sensitivity_map, phantom.tx_sensitivity_map)
    assert np.array_equal(loaded.rx_sensitivity_maps, phantom.rx_sensitivity_maps)
    assert loaded.coordinate_system == "object_xyz"
    assert np.array_equal(loaded.affine_ijk_to_xyz_m, phantom.affine_ijk_to_xyz_m)


def test_phantom_xarray_dataset_exposes_physical_coordinates():
    phantom = _phantom(shape=(2, 3, 4))
    dataset = phantom.to_xarray()

    assert dataset["t1"].dims == ("x", "y", "z")
    assert dataset["m0"].dims == ("x", "y", "z", "component")
    assert dataset["rx_sensitivity_real"].dims == ("coil", "x", "y", "z")
    assert dataset.attrs["coordinate_system"] == "object_xyz"
    assert dataset.attrs["affine_ijk_to_xyz_m"].shape == (16,)
    assert np.asarray(dataset.coords["x"]) == pytest.approx([-0.005, 0.005])
    assert np.asarray(dataset.coords["z"]) == pytest.approx(
        [-0.0075, -0.0025, 0.0025, 0.0075]
    )


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
