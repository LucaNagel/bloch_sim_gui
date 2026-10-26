from pathlib import Path

import numpy as np
import pytest

from blochsimulator import BlochSimulator
from blochsimulator.phantom import Phantom
from blochsimulator.sequence import (
    infer_cartesian_acquisition,
    load_pulseq,
    make_pulseq_bssfp,
    make_pulseq_csi,
    make_pulseq_epsi_mge,
    make_pulseq_flash,
    make_pulseq_me_bssfp,
    make_pulseq_spectral_selective_bssfp,
)
from blochsimulator.sequence.encoding_order import (
    PAIRED_ORDERS,
    axis_encoding_order,
    normalized_two_axis_order,
    two_axis_encoding_order,
    volume_encoding_order,
)
from blochsimulator.sequence.pulseq_builders import _phase_encoding_indices

pypulseq = pytest.importorskip("pypulseq")


def _write_and_load(sequence, path: Path):
    sequence.write(str(path), v141_compat=True)
    return load_pulseq(path)


def _adc_labels(sequence, name):
    """Return the value of one Pulseq label at every ADC, in time order."""
    values, current = [], 0
    for block_id in sequence.block_events:
        block = sequence.get_block(block_id)
        for item in (getattr(block, "label", None) or {}).values():
            if item.label == name:
                current = (
                    current + item.value if item.type == "labelinc" else item.value
                )
        if block.adc is not None:
            values.append(int(current))
    return values


# --- order helper ---------------------------------------------------------


def test_linear_and_centric_follow_the_requested_k_space_order():
    # Index i encodes k = i - n // 2.
    assert [i - 2 for i in axis_encoding_order(5, "linear")] == [-2, -1, 0, 1, 2]
    assert [i - 2 for i in axis_encoding_order(5, "centric")] == [0, -1, 1, -2, 2]
    assert [i - 2 for i in axis_encoding_order(4, "centric")] == [0, -1, 1, -2]
    assert axis_encoding_order(1, "centric") == (0,)


@pytest.mark.parametrize("order", PAIRED_ORDERS)
@pytest.mark.parametrize("inner_axis", (0, 1))
def test_paired_orders_nest_the_inner_axis(order, inner_axis):
    mode_a, mode_b = order.split("-")
    pairs = two_axis_encoding_order((4, 3), order, inner_axis=inner_axis)
    assert sorted(pairs) == [(a, b) for a in range(4) for b in range(3)]
    if inner_axis == 0:
        assert [a for a, _ in pairs[:4]] == list(axis_encoding_order(4, mode_a))
        assert [b for _, b in pairs[::4]] == list(axis_encoding_order(3, mode_b))
    else:
        assert [b for _, b in pairs[:3]] == list(axis_encoding_order(3, mode_b))
        assert [a for a, _ in pairs[::3]] == list(axis_encoding_order(4, mode_a))


def test_elliptical_centric_sorts_by_physical_k_space_radius():
    fov = (0.08, 0.04)
    pairs = two_axis_encoding_order((6, 5), "elliptical_centric", fov_m=fov)
    assert sorted(pairs) == [(a, b) for a in range(6) for b in range(5)]
    radii = [np.hypot((a - 3) / fov[0], (b - 2) / fov[1]) for a, b in pairs]
    assert pairs[0] == (3, 2)
    assert np.all(np.diff(radii) >= -1e-12)


def test_order_names_accept_aliases_and_reject_unknown_orders():
    assert normalized_two_axis_order("centric") == "centric-centric"
    assert normalized_two_axis_order(("Linear", "centric")) == "linear-centric"
    assert normalized_two_axis_order("Elliptical-Centric") == "elliptical_centric"
    with pytest.raises(ValueError):
        normalized_two_axis_order("random")
    with pytest.raises(ValueError):
        volume_encoding_order((4, 4), "spiral", fov_m=(1, 1))
    with pytest.raises(ValueError):
        volume_encoding_order((4, 4), "linear-linear", "kx", fov_m=(1, 1))


def test_csi_centric_keeps_its_elliptical_meaning():
    fov = (0.08, 0.06)
    assert _phase_encoding_indices(5, 4, "centric", fov) == list(
        two_axis_encoding_order((5, 4), "elliptical_centric", fov_m=fov)
    )
    sequence = make_pulseq_csi(
        fov_m=fov, matrix=(3, 2), spectral_points=8, phase_encoding_order="centric"
    )
    assert sequence.definitions["PhaseEncodingOrder"] == "elliptical_centric"


@pytest.mark.parametrize("order", PAIRED_ORDERS)
def test_csi_paired_orders_and_inner_axis_reach_the_labels(order):
    sequence = make_pulseq_csi(
        fov_m=(0.08, 0.06),
        matrix=(3, 4),
        spectral_points=8,
        phase_encoding_order=order,
        phase_encoding_inner_axis="ky",
    )
    expected = two_axis_encoding_order((3, 4), order, inner_axis=1)
    pairs = list(zip(_adc_labels(sequence, "LIN"), _adc_labels(sequence, "PAR")))
    assert sequence.definitions["PhaseEncodingInnerAxis"] == "ky"
    assert sorted(set(pairs)) == sorted(expected)
    # Each acquisition's (kx, ky) step appears in the chosen order.
    first_seen = list(dict.fromkeys(pairs))
    assert first_seen in (list(expected), [(b, a) for a, b in expected])


# --- 2D readouts ----------------------------------------------------------


def _flash_like(builder, order, **overrides):
    parameters = dict(
        fov_m=(0.064, 0.064),
        matrix=(8, 8),
        sampling_bandwidth_hz=20e3,
        phase_encoding_order=order,
    )
    parameters.update(overrides)
    return builder(**parameters)


@pytest.mark.parametrize("builder", (make_pulseq_flash, make_pulseq_epsi_mge))
def test_2d_line_labels_follow_the_order(builder):
    centric = _flash_like(builder, "centric")
    linear = _flash_like(builder, "linear")
    assert centric.definitions["PhaseEncodingOrder"] == "centric"
    centric_lines = list(dict.fromkeys(_adc_labels(centric, "LIN")))
    linear_lines = list(dict.fromkeys(_adc_labels(linear, "LIN")))
    assert centric_lines == list(axis_encoding_order(8, "centric"))
    assert linear_lines == list(range(8))
    with pytest.raises(ValueError):
        _flash_like(builder, "linear-centric")


def _disc_phantom(shape, fov, relaxation_s):
    positions = [
        (np.arange(n) - (n - 1) / 2) * (extent / n) for n, extent in zip(shape, fov)
    ]
    grids = np.meshgrid(*positions, indexing="ij")
    centre = (0.008, -0.006, 0.0)[: len(shape)]
    radius = 0.4 * min(fov[:2])
    distance = sum((grid - c) ** 2 for grid, c in zip(grids, centre))
    pd_map = (distance < radius**2).astype(float)
    return Phantom(
        shape=shape,
        fov=fov,
        t1_map=np.full(shape, relaxation_s),
        t2_map=np.full(shape, relaxation_s),
        pd_map=pd_map,
    )


def _image(program, phantom):
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    image = result.to_xarray()["cartesian_image_magnitude"]
    return np.asarray(image.sum([dim for dim in image.dims[:-2]]))


def test_flash_centric_reconstructs_the_linear_image(tmp_path):
    # T1 = T2 well below TR: every excitation sees fully relaxed magnetisation,
    # so the image cannot depend on the line order.
    phantom = _disc_phantom((16, 16, 1), (0.064, 0.064, 3e-3), 2e-3)
    images = {}
    for order in ("linear", "centric"):
        sequence = _flash_like(
            make_pulseq_flash,
            order,
            matrix=(16, 16),
            echo_time_s=2.5e-3,
            repetition_time_s=30e-3,
        )
        program = _write_and_load(sequence, tmp_path / f"{order}.seq")
        acquisition = infer_cartesian_acquisition(program)
        assert [acquisition.phase_indices[i] for i in range(4)] == (
            list(axis_encoding_order(16, order)[:4])
        )
        images[order] = _image(program, phantom)
    np.testing.assert_allclose(
        images["centric"], images["linear"], atol=1e-6 * images["linear"].max()
    )


def test_ir_flash_centric_times_the_inversion_to_the_first_line():
    sequence = make_pulseq_flash(
        matrix=(4, 4),
        inversion_times_s=0.2,
        repetition_time_s=10e-3,
        echo_time_s=4e-3,
        phase_encoding_order="centric",
    )
    rf_blocks = [b for b in sequence.block_events if sequence.get_block(b).rf]
    block_ids = list(sequence.block_durations)
    starts = np.cumsum([0.0] + [float(d) for d in sequence.block_durations.values()])

    def centre(block_id):
        rf = sequence.get_block(block_id).rf
        return starts[block_ids.index(block_id)] + rf.delay + float(rf.shape_dur) / 2

    # Centric acquires the k-space centre first, right after the inversion.
    assert centre(rf_blocks[1]) - centre(rf_blocks[0]) == pytest.approx(0.2, abs=2e-5)
    assert _adc_labels(sequence, "LIN")[0] == 2


# --- 3D readouts ----------------------------------------------------------

VOLUME_BUILDERS = (
    (make_pulseq_bssfp, {}),
    (make_pulseq_me_bssfp, {"echoes": 3}),
    (make_pulseq_spectral_selective_bssfp, {}),
)


@pytest.mark.parametrize("builder, options", VOLUME_BUILDERS)
@pytest.mark.parametrize(
    "order, inner",
    (
        ("centric-linear", "ky"),
        ("linear-centric", "kz"),
        ("elliptical_centric", "ky"),
    ),
)
def test_3d_labels_follow_the_order(builder, options, order, inner):
    fov = (0.032, 0.032, 0.024)
    sequence = builder(
        fov_m=fov,
        matrix=(4, 4, 3),
        dummy_repetitions=0,
        phase_encoding_order=order,
        phase_encoding_inner_axis=inner,
        **options,
    )
    _, _, expected = volume_encoding_order((4, 3), order, inner, fov_m=(fov[1], fov[2]))
    pairs = list(zip(_adc_labels(sequence, "LIN"), _adc_labels(sequence, "PAR")))
    assert list(dict.fromkeys(pairs)) == list(expected)
    assert sequence.definitions["PhaseEncodingOrder"] == order
    assert sequence.definitions["PhaseEncodingInnerAxis"] == inner


@pytest.mark.parametrize(
    "order, inner",
    (
        ("centric-centric", "ky"),
        ("linear-linear", "kz"),
        ("centric-linear", "kz"),
        ("elliptical_centric", "ky"),
    ),
)
def test_3d_bssfp_orders_reconstruct_the_linear_volume(tmp_path, order, inner):
    # Without relaxation and on resonance the alpha/2 preparation keeps the
    # magnetisation close to steady state; a misplaced k-space line would
    # change the volume at order one, the residual transient only at 1e-4.
    shape, fov = (8, 8, 4), (0.032, 0.032, 0.024)
    phantom = _disc_phantom(shape, fov, 1e6)

    def volume(order, inner, name):
        sequence = make_pulseq_bssfp(
            fov_m=fov,
            matrix=shape,
            dummy_repetitions=0,
            phase_encoding_order=order,
            phase_encoding_inner_axis=inner,
        )
        program = _write_and_load(sequence, tmp_path / f"{name}.seq")
        result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
        data = result.to_xarray()
        name = "cartesian_3d_image_magnitude"
        assert name in data
        return np.asarray(data[name])

    reference = volume("linear-linear", "ky", "reference")
    candidate = volume(order, inner, "candidate")
    assert candidate.shape == reference.shape
    np.testing.assert_allclose(candidate, reference, atol=1e-3 * reference.max())


# --- Bruker export --------------------------------------------------------


def _export_method(tmp_path, sequence, phantom, name):
    from blochsimulator.sequence import export_bruker_dataset
    from blochsimulator.sequence.bruker_jcamp import parse_jcamp_value, read_jcamp

    program = _write_and_load(sequence, tmp_path / f"{name}.seq")
    result = BlochSimulator(use_parallel=False).simulate_sequence(program, phantom)
    result.metadata.update(field_strength_t=7.0)
    exported = export_bruker_dataset(
        result, tmp_path / name, program=program, phantom=phantom
    )
    parameters = read_jcamp(exported.scan_dir / "method").parameters
    method = {
        key: parse_jcamp_value(parameters[key])
        for key in ("PVM_EncOrder", "PVM_EncSteps1", "PVM_EncSteps2")
        if key in parameters
    }
    return result, exported, method


def test_bruker_flash_export_stores_centric_steps(tmp_path):
    phantom = _disc_phantom((8, 8, 1), (0.064, 0.064, 3e-3), 0.1)
    sequence = _flash_like(make_pulseq_flash, "centric")
    _, exported, method = _export_method(tmp_path, sequence, phantom, "flash")
    assert list(method["PVM_EncOrder"]) == ["LINEAR_ENC", "CENTRIC_ENC"]
    assert list(method["PVM_EncSteps1"]) == [0, -1, 1, -2, 2, -3, 3, -4]
    assert not any("order" in text for text in exported.warnings)


@pytest.mark.parametrize(
    "order, inner, steps, enc_order",
    (
        ("centric-centric", "ky", ([0, -1, 1, -2], [0, -1, 1]), "CENTRIC_ENC"),
        ("linear-centric", "kz", ([-2, -1, 0, 1], [-1, 0, 1]), "LINEAR_ENC"),
    ),
)
def test_bruker_3d_export_stores_ky_inner_orders_chronologically(
    tmp_path, order, inner, steps, enc_order
):
    import json
    import warnings

    brukerapi_dataset = pytest.importorskip("brukerapi.dataset")
    shape, fov = (4, 4, 3), (0.032, 0.032, 0.024)
    phantom = Phantom(
        shape=shape,
        fov=fov,
        t1_map=np.ones(shape),
        t2_map=np.ones(shape),
        pd_map=np.random.default_rng(1).uniform(0.2, 1.0, shape),
    )
    sequence = make_pulseq_bssfp(
        fov_m=fov,
        matrix=shape,
        dummy_repetitions=0,
        phase_encoding_order=order,
        phase_encoding_inner_axis=inner,
    )
    result, exported, method = _export_method(tmp_path, sequence, phantom, "volume")
    assert [list(method["PVM_EncSteps1"]), list(method["PVM_EncSteps2"])] == [
        list(item) for item in steps
    ]
    assert list(method["PVM_EncOrder"]) == ["LINEAR_ENC", enc_order, enc_order]
    # Only a ky-inner loop fits Bruker's raw layout; others are reordered.
    reordered = any("ky-inner" in text for text in exported.warnings)
    assert reordered == (inner == "kz")

    scale = json.loads((exported.scan_dir / "blochsim_provenance.json").read_text())[
        "signal_scale"
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = np.asarray(brukerapi_dataset.Dataset(str(exported.scan_dir / "fid")).data)
    # brukerapi sorts the lines by PVM_EncSteps1 only; sort the partitions here.
    kspace = np.squeeze(raw)[:, :, np.argsort(method["PVM_EncSteps2"])] / scale
    expected = np.asarray(result.to_xarray()["cartesian_3d_kspace"]).transpose(2, 1, 0)
    np.testing.assert_allclose(kspace, expected, atol=1e-6 * np.abs(expected).max())


# --- GUI ------------------------------------------------------------------


def _enabled(widget):
    # Form groups of inactive sequences are disabled as a whole.
    return widget.isEnabledTo(widget.parentWidget())


def _select(combo, name):
    index = combo.findData(name)
    assert index >= 0, name
    combo.setCurrentIndex(index)


def test_gui_forms_pass_the_phase_encoding_order(qt_application):
    from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget

    widget = SequenceSimulationWidget()
    try:
        for prefix in ("flash", "epsi"):
            _select(getattr(widget, f"{prefix}_phase_encoding_order"), "centric")
        assert widget._flash_pulseq_parameters()["phase_encoding_order"] == "centric"
        parameters = widget._epsi_mge_pulseq_parameters()
        assert parameters["phase_encoding_order"] == "centric"

        widget.ir_readout.setCurrentIndex(widget.ir_readout.findData("gre"))
        _select(widget.ir_phase_encoding_order, "centric")
        assert _enabled(widget.ir_phase_encoding_order)
        assert widget._ir_pulseq_parameters()["phase_encoding_order"] == "centric"
        widget.ir_readout.setCurrentIndex(widget.ir_readout.findData("epi"))
        assert "phase_encoding_order" not in widget._ir_pulseq_parameters()

        for prefix, method in (
            ("bssfp", widget._bssfp_pulseq_parameters),
            ("ss_bssfp", widget._ss_bssfp_pulseq_parameters),
            ("me_bssfp", widget._me_bssfp_pulseq_parameters),
        ):
            order = getattr(widget, f"{prefix}_phase_encoding_order")
            inner = getattr(widget, f"{prefix}_phase_encoding_inner_axis")
            _select(order, "linear-centric")
            inner.setCurrentText("kz")
            assert _enabled(inner)
            assert (
                method()["phase_encoding_order"],
                method()["phase_encoding_inner_axis"],
            ) == ("linear-centric", "kz")
            _select(order, "elliptical_centric")
            assert not _enabled(inner)

        _select(widget.csi_encoding_order, "centric-linear")
        widget.csi_encoding_inner_axis.setCurrentText("ky")
        parameters = widget._csi_pulseq_parameters()
        assert parameters["phase_encoding_order"] == "centric-linear"
        assert parameters["phase_encoding_inner_axis"] == "ky"
    finally:
        widget.close()
        widget.deleteLater()
        qt_application.processEvents()


def test_saved_csi_centric_order_restores_as_elliptical(qt_application):
    from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget
    from blochsimulator.ui.widget_state import restore_widget_state

    widget = SequenceSimulationWidget()
    try:
        for text, expected in (
            ("centric", "elliptical_centric"),
            ("linear", "linear-linear"),
            ("spiral", "spiral"),
        ):
            restore_widget_state(
                widget,
                {"csi_encoding_order": {"type": "combo", "index": 0, "text": text}},
            )
            assert widget.csi_encoding_order.currentData() == expected
    finally:
        widget.close()
        widget.deleteLater()
        qt_application.processEvents()
