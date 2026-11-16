"""One pool in the spin-major executor.

A single pool is a lone precursor: no products and a conversion array of
shape ``(0, nspins)``. Without conversion pool 0 evolves independently of the
products, so every chunk of a two-pool plan whose second pool is empty and
unconverted, rerun with only pool 0, must give pool 0 bitwise unchanged.
DynamicSpectralPhantom still needs two pools, so the one-pool calls are made
directly on the extension, alongside the two-pool chunks of a real run.
"""

import numpy as np
import pytest

import test_dynamic_spin_major_multi_pool as multi_pool
import test_dynamic_spin_major_rf_cache_multi_pool as rf_cache
from blochsimulator import dynamic_phantom as dynamic_module

spin_major = pytest.importorskip("blochsimulator.dynamic_spin_major_cy")

# Arrays with the pool axis first: (npools, ...).
_POOL_FIRST = ("transverse_state", "state", "concentration_state", "pool_b0_hz")
_POOL_VECTORS = ("r1", "t2_s", "pool_offset_hz")
# Arrays written by the kernel; pool 0 of each is compared after the call.
_OUTPUTS = (
    ("transverse_state", 0),
    ("state", 0),
    ("concentration_state", 0),
    ("checkpoint_states", 1),
    ("adc_signal", 0),
)


def _first_pool(array, axis):
    return np.ascontiguousarray(np.take(array, [0], axis=axis))


class _SinglePoolShadow:
    """Run every two-pool chunk once more with pool 0 alone and compare."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.calls = 0
        self.propagated = 0
        self.propagators = {}

    def __call__(self, **arguments):
        single = dict(arguments)
        assert arguments["conversion"].shape[0] == 1
        assert not np.any(arguments["conversion"])
        single["conversion"] = np.empty((0, arguments["conversion"].shape[1]))
        for name in _POOL_FIRST:
            if single.get(name) is not None:
                single[name] = _first_pool(single[name], 0)
        for name in _POOL_VECTORS:
            single[name] = np.asarray(single[name], dtype=np.float64)[:1].copy()
        single["checkpoint_states"] = _first_pool(arguments["checkpoint_states"], 1)
        single["adc_signal"] = _first_pool(arguments["adc_signal"], 0)
        if single.get("washout_table") is not None:
            single["washout_table"] = _first_pool(arguments["washout_table"], 3)
        if arguments.get("propagators") is not None:
            # Entries persist across chunks; keep a one-pool buffer per buffer.
            entries = arguments["propagators"]
            buffer = self.propagators.get(id(entries))
            if buffer is None:
                values = spin_major.propagator_values(
                    1,
                    arguments["concentration_state"] is not None,
                    arguments["delivery"] is not None,
                    bool(arguments["concentration_sources"]),
                )
                buffer = np.zeros(
                    (entries.shape[0], entries.shape[1], values, entries.shape[3])
                )
                self.propagators[id(entries)] = (buffer, entries)
            else:
                buffer = buffer[0]
            single["propagators"] = buffer
            self.propagated += 1

        returned = self.kernel(**arguments)
        self.kernel(**single)
        self.calls += 1
        for name, axis in _OUTPUTS:
            if arguments.get(name) is None:
                continue
            expected = _first_pool(arguments[name], axis)
            np.testing.assert_array_equal(single[name], expected, err_msg=name)
        return returned


def _empty_second_pool(phantom):
    """Remove the second pool's spins and its conversion."""
    product = phantom.pools[1].name
    phantom.initial_concentration_maps[product] = np.zeros(phantom.shape)
    if phantom.initial_spin_density_maps is not None:
        phantom.initial_spin_density_maps[product] = np.zeros(phantom.shape)
    phantom.kpl_map_s_inv = np.zeros(phantom.shape)
    return phantom


def _shadowed_run(monkeypatch, module, phantom, program, kernel, **kwargs):
    original, tile_spins = dynamic_module._spin_major_kernel()
    shadow = _SinglePoolShadow(original)
    monkeypatch.setattr(
        dynamic_module, "_spin_major_kernel", lambda: (shadow, tile_spins)
    )
    result = module._run(phantom, program, kernel, **kwargs)
    assert result.metadata["native_spin_major_enabled"]
    assert shadow.calls > 0
    return shadow


def test_single_pool_executor_api():
    assert spin_major.API_VERSION == 9
    assert spin_major.SINGLE_POOL_SUPPORTED
    assert spin_major.propagator_values(1, False, False, False) > 0
    with pytest.raises(ValueError, match="1 to"):
        spin_major.propagator_values(0, False, False, False)


@pytest.mark.parametrize("kernel", ["native_serial", "native_parallel"])
@pytest.mark.parametrize(
    "inflow, tracked, washout",
    [
        (None, False, False),
        ("direct", False, False),
        ("polarized", True, False),
        ("direct", True, True),
    ],
)
def test_single_pool_matches_first_pool_of_two(
    monkeypatch, kernel, inflow, tracked, washout
):
    phantom = multi_pool._phantom(2, (40, 3, 1), inflow=inflow, tracked=tracked)
    if washout:
        phantom = multi_pool._with_washout(phantom)
    phantom = _empty_second_pool(phantom)
    kwargs = dict(use_parallel=kernel == "native_parallel", num_threads=3)
    _shadowed_run(
        monkeypatch, multi_pool, phantom, multi_pool._program(), kernel, **kwargs
    )


@pytest.mark.parametrize("max_delay_s", [None, 40 * rf_cache.RASTER])
def test_single_pool_matches_first_pool_of_two_with_rf_cache(monkeypatch, max_delay_s):
    phantom = _empty_second_pool(rf_cache._phantom(2, max_delay_s=max_delay_s))
    shadow = _shadowed_run(
        monkeypatch,
        rf_cache,
        phantom,
        rf_cache._program(),
        "native_parallel",
        use_parallel=True,
        num_threads=2,
    )
    assert shadow.propagated > 0
