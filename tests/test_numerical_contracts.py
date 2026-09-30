import numpy as np


def test_ordinary_bloch_builds_have_distinct_numerical_contracts():
    import blochsimulator.blochsimulator_cy as legacy
    import blochsimulator.blochsimulator_strict_cy as strict

    assert legacy.__file__ != strict.__file__
    assert legacy.NUMERICAL_CONTRACT == "legacy-fast-math"
    assert strict.NUMERICAL_CONTRACT == "strict-float64"


def test_strict_and_legacy_ordinary_bloch_builds_execute_same_fixture():
    import blochsimulator.blochsimulator_cy as legacy
    import blochsimulator.blochsimulator_strict_cy as strict

    interval_count = 96
    duration_s = np.linspace(1e-6, 20e-6, interval_count)
    rf = np.linspace(-0.04, 0.07, interval_count) + 1j * np.linspace(
        0.03, -0.02, interval_count
    )
    gradients = np.column_stack(
        (
            np.linspace(-0.2, 0.3, interval_count),
            np.linspace(0.1, -0.15, interval_count),
            np.linspace(-0.05, 0.08, interval_count),
        )
    )
    frequencies = np.asarray([-125.0, 0.0, 73.0])
    positions = np.asarray(
        [
            [-0.7, 0.2, 0.4],
            [0.0, 0.0, 0.0],
            [0.6, -0.3, -0.5],
        ]
    )
    arguments = (
        rf,
        gradients,
        duration_s,
        1.2,
        0.18,
        frequencies,
        positions,
    )

    legacy_result = legacy.simulate_bloch(*arguments, num_threads=1)
    strict_result = strict.simulate_bloch(*arguments, num_threads=1)

    for legacy_value, strict_value in zip(legacy_result, strict_result):
        np.testing.assert_allclose(strict_value, legacy_value, rtol=2e-14, atol=2e-14)
