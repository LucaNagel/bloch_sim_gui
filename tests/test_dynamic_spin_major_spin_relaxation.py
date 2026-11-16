"""Constant relaxation groups and per-spin equilibrium in the spin-major executor.

Every chunk of a real run is executed three more ways from the same start
state: unchanged, with spins split into relaxation groups and equilibria
(``spin % 3`` selects group -1, 0 or 1), and once per partition with the
partition's constants as pool constants. The grouped run must match the
partition runs on each partition's spins; without new inputs, or with inputs
equal to the pool constants, it must match the plain call bitwise.

A partition equilibrium ``q`` differing from the scalar ``e`` enters the
partition reference as ``equilibrium=q`` with the global excess sources
shifted by ``(e - q)`` times the concentration sources, which is how the
executor applies the per-spin difference to source rows and segment weights
computed with ``e``.
"""

import numpy as np
import pytest

import test_dynamic_spin_major_multi_pool as multi_pool
import test_dynamic_spin_major_rf_cache_multi_pool as rf_cache
from blochsimulator import dynamic_phantom as dynamic_module

spin_major = pytest.importorskip("blochsimulator.dynamic_spin_major_cy")

_STATE = ("transverse_state", "state", "concentration_state", "checkpoint_states")
# Spin axis of every per-spin output.
_SPIN_AXIS = {
    "transverse_state": 1,
    "state": 1,
    "concentration_state": 1,
    "checkpoint_states": 2,
}
_PARTITIONS = (-1, 0, 1)


def _group_table(arguments):
    """Per-pool (R1, T2) of groups 0 and 1, distinct from the pool constants."""
    r1 = np.asarray(arguments["r1"], dtype=np.float64)
    t2 = np.asarray(arguments["t2_s"], dtype=np.float64)
    return np.ascontiguousarray(
        np.stack(
            (
                np.stack((1.7 * r1 + 0.05, 0.6 * t2), axis=-1),
                np.stack((0.8 * r1 + 0.11, 1.3 * t2), axis=-1),
            )
        )
    )


class _RelaxationShadow:
    """Compare grouped chunks against one plain call per partition."""

    def __init__(self, kernel, *, equilibria, identity=False):
        self.kernel = kernel
        self.equilibria = equilibria
        self.identity = identity
        self.calls = 0
        self.propagated = 0
        self.changed = 0
        self.shifted = 0
        self.buffers = {}

    def _propagators(self, arguments, variant):
        entries = arguments.get("propagators")
        if entries is None:
            return None
        key = (id(entries), variant)
        if key not in self.buffers:
            self.buffers[key] = (np.zeros_like(entries), entries)
        return self.buffers[key][0]

    def _copy(self, arguments, start, variant):
        copy = dict(arguments)
        for name in _STATE:
            if start.get(name) is not None:
                copy[name] = start[name].copy()
        copy["adc_signal"] = np.zeros_like(arguments["adc_signal"])
        copy["propagators"] = self._propagators(arguments, variant)
        return copy

    def _spin_equilibrium(self, arguments, partition):
        equilibrium = arguments["equilibrium"]
        values = np.empty(arguments["conversion"].shape[1])
        for part, scale in zip(_PARTITIONS, self.equilibria):
            values[partition == part] = scale * equilibrium
        return values

    def __call__(self, **arguments):
        start = {
            name: None if arguments.get(name) is None else arguments[name].copy()
            for name in _STATE
        }
        adc_start = arguments["adc_signal"].copy()
        returned = self.kernel(**arguments)
        self.calls += 1
        self.propagated += arguments.get("propagators") is not None
        adc_plain = arguments["adc_signal"] - adc_start

        nspins = arguments["conversion"].shape[1]
        partition = np.arange(nspins) % 3 - 1
        tracked = arguments["concentration_state"] is not None
        table = _group_table(arguments)
        grouped = self._copy(arguments, start, "grouped")
        if self.identity:
            grouped["relaxation_group"] = np.zeros(nspins, dtype=np.int32)
            grouped["relaxation_groups"] = np.ascontiguousarray(
                np.stack((arguments["r1"], arguments["t2_s"]), axis=-1)[None]
            )
            if tracked:
                grouped["spin_equilibrium"] = np.full(
                    nspins, float(arguments["equilibrium"])
                )
            self.kernel(**grouped)
            for name in _STATE:
                if arguments.get(name) is not None:
                    np.testing.assert_array_equal(
                        grouped[name], arguments[name], err_msg=name
                    )
            np.testing.assert_array_equal(grouped["adc_signal"], adc_plain)
            return returned

        grouped["relaxation_group"] = partition.astype(np.int32)
        grouped["relaxation_groups"] = table
        if tracked:
            grouped["spin_equilibrium"] = self._spin_equilibrium(arguments, partition)
        self.kernel(**grouped)
        self.changed += not np.array_equal(grouped["state"], arguments["state"])

        adc_sum = np.zeros_like(adc_plain)
        for part, scale in zip(_PARTITIONS, self.equilibria):
            reference = self._copy(arguments, start, part)
            if part >= 0:
                reference["r1"] = table[part, :, 0].copy()
                reference["t2_s"] = table[part, :, 1].copy()
            mask = partition == part
            reference["signal_weights"] = np.where(
                mask, arguments["signal_weights"], 0.0
            )
            shift = 0.0
            if tracked:
                shift = (1.0 - scale) * arguments["equilibrium"]
                reference["equilibrium"] = scale * arguments["equilibrium"]
                self.shifted += bool(shift)
            if shift and arguments.get("source_table") is not None:
                rows = np.array(arguments["source_table"], dtype=np.float64)
                rows[:, (0, 1, 4, 5)] += shift * rows[:, (2, 3, 6, 7)]
                reference["source_table"] = rows
            if shift and arguments.get("segment_sources") is not None:
                weights = np.array(arguments["segment_sources"], dtype=np.float64)
                weights[:, :3] += shift * weights[:, 3:]
                reference["segment_sources"] = weights
            self.kernel(**reference)
            adc_sum += reference["adc_signal"]
            for name, axis in _SPIN_AXIS.items():
                if grouped.get(name) is None:
                    continue
                actual = np.compress(mask, grouped[name], axis=axis)
                expected = np.compress(mask, reference[name], axis=axis)
                if shift:
                    scale_ = max(np.max(np.abs(expected)), 1e-300)
                    assert np.max(np.abs(actual - expected)) <= 1e-12 * scale_, name
                else:
                    np.testing.assert_array_equal(actual, expected, err_msg=name)
        scale_ = max(np.max(np.abs(adc_sum)), 1e-300)
        assert np.max(np.abs(grouped["adc_signal"] - adc_sum)) <= 1e-12 * scale_
        return returned


def _shadowed_run(monkeypatch, module, phantom, program, kernel, shadow, **kwargs):
    original, tile_spins = dynamic_module._spin_major_kernel()
    shadow.kernel = original
    monkeypatch.setattr(
        dynamic_module, "_spin_major_kernel", lambda: (shadow, tile_spins)
    )
    result = module._run(phantom, program, kernel, **kwargs)
    assert result.metadata["native_spin_major_enabled"]
    assert shadow.calls > 0
    if not shadow.identity:
        assert shadow.changed > 0
    return shadow


def test_spin_relaxation_executor_api():
    assert spin_major.API_VERSION == 9
    assert spin_major.SPIN_RELAXATION_SUPPORTED


_PLAIN_CASES = [
    (2, None, False),
    (3, "direct", False),
    (3, "polarized", True),
    (4, None, True),
]


@pytest.mark.parametrize("kernel", ["native_serial", "native_parallel"])
@pytest.mark.parametrize("n_pools, inflow, tracked", _PLAIN_CASES)
def test_inputs_equal_to_pool_constants_are_bitwise_neutral(
    monkeypatch, kernel, n_pools, inflow, tracked
):
    phantom = multi_pool._phantom(n_pools, (40, 3, 1), inflow=inflow, tracked=tracked)
    shadow = _RelaxationShadow(None, equilibria=(1.0,) * 3, identity=True)
    _shadowed_run(
        monkeypatch,
        multi_pool,
        phantom,
        multi_pool._program(),
        kernel,
        shadow,
        use_parallel=kernel == "native_parallel",
        num_threads=3,
    )


@pytest.mark.parametrize("kernel", ["native_serial", "native_parallel"])
@pytest.mark.parametrize("n_pools, inflow, tracked", _PLAIN_CASES)
@pytest.mark.parametrize("max_delay_s", [None, 6 * multi_pool.RASTER])
def test_groups_match_partition_runs(
    monkeypatch, kernel, n_pools, inflow, tracked, max_delay_s
):
    if max_delay_s is not None and inflow is None:
        pytest.skip("arrival delays need inflow")
    phantom = multi_pool._phantom(
        n_pools,
        (40, 3, 1),
        inflow=inflow,
        tracked=tracked,
        max_delay_s=max_delay_s,
    )
    # Partition 0 keeps the scalar equilibrium: groups alone stay bitwise.
    shadow = _RelaxationShadow(None, equilibria=(1.0, 1.0, 2.5))
    _shadowed_run(
        monkeypatch,
        multi_pool,
        phantom,
        multi_pool._program(),
        kernel,
        shadow,
        use_parallel=kernel == "native_parallel",
        num_threads=3,
    )
    assert (shadow.shifted > 0) == tracked


@pytest.mark.parametrize("max_delay_s", [None, 40 * rf_cache.RASTER])
@pytest.mark.parametrize("identity", [True, False])
def test_groups_with_rf_cache(monkeypatch, max_delay_s, identity):
    phantom = rf_cache._phantom(3, max_delay_s=max_delay_s)
    shadow = _RelaxationShadow(None, equilibria=(1.0, 1.0, 2.5), identity=identity)
    _shadowed_run(
        monkeypatch,
        rf_cache,
        phantom,
        rf_cache._program(),
        "native_parallel",
        shadow,
        use_parallel=True,
        num_threads=2,
    )
    assert shadow.propagated > 0


class _Capture:
    def __init__(self, kernel):
        self.kernel = kernel
        self.arguments = None

    def __call__(self, **arguments):
        if self.arguments is None:
            self.arguments = {
                name: value.copy() if isinstance(value, np.ndarray) else value
                for name, value in arguments.items()
            }
        return self.kernel(**arguments)


def _captured(monkeypatch, phantom):
    original, tile_spins = dynamic_module._spin_major_kernel()
    capture = _Capture(original)
    monkeypatch.setattr(
        dynamic_module, "_spin_major_kernel", lambda: (capture, tile_spins)
    )
    multi_pool._run(phantom, multi_pool._program(), "native_serial")
    assert capture.arguments is not None
    return original, capture.arguments


def test_spin_equilibrium_needs_tracked_concentration(monkeypatch):
    kernel, arguments = _captured(monkeypatch, multi_pool._phantom(2, (12, 1, 1)))
    assert arguments["concentration_state"] is None
    arguments["spin_equilibrium"] = np.full(12, 0.02)
    with pytest.raises(ValueError):
        kernel(**arguments)


@pytest.mark.parametrize("field", ["groups", "equilibrium"])
def test_spin_relaxation_rejects_washout_sources(monkeypatch, field):
    phantom = multi_pool._with_washout(
        multi_pool._phantom(2, (12, 1, 1), inflow="polarized", tracked=True)
    )
    kernel, arguments = _captured(monkeypatch, phantom)
    assert arguments["washout_compartment"] is not None
    if field == "groups":
        arguments["relaxation_group"] = np.zeros(12, dtype=np.int32)
        arguments["relaxation_groups"] = _group_table(arguments)[:1]
    else:
        arguments["spin_equilibrium"] = np.full(12, 0.02)
    with pytest.raises(ValueError):
        kernel(**arguments)


@pytest.mark.parametrize(
    "group, table",
    [
        (5, None),
        (-2, None),
        (0, "nan"),
        (0, "t2"),
    ],
)
def test_invalid_relaxation_groups_raise(monkeypatch, group, table):
    kernel, arguments = _captured(monkeypatch, multi_pool._phantom(2, (12, 1, 1)))
    rows = _group_table(arguments)
    if table == "nan":
        rows[0, 1, 0] = np.nan
    elif table == "t2":
        rows[1, 0, 1] = 0.0
    arguments["relaxation_group"] = np.full(12, group, dtype=np.int32)
    arguments["relaxation_groups"] = rows
    with pytest.raises(ValueError):
        kernel(**arguments)
