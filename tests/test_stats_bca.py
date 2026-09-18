import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from holdout.stats.bootstrap import bootstrap_ci


def _median(sample: NDArray[np.float64]) -> float:
    return float(np.median(sample))


def _unique_count(sample: NDArray[np.float64]) -> float:
    return float(np.unique(sample).size)


def _as_floats(arr: NDArray[np.float64]) -> list[float]:
    return [float(v) for v in arr]


def test_bca_matches_scipy_bca_on_continuous_normal_data() -> None:
    rng = np.random.default_rng(42)
    data: NDArray[np.float64] = rng.normal(loc=0.7, scale=1.0, size=100)
    scale = float(np.std(data))

    est = bootstrap_ci(_as_floats(data), n_resamples=20_000, seed=123, method="bca")
    assert est.method == "bootstrap-bca"
    assert est.value == pytest.approx(float(np.mean(data)))

    res = scipy_stats.bootstrap(
        (data,),
        np.mean,
        confidence_level=0.95,
        n_resamples=20_000,
        method="BCa",
        rng=np.random.default_rng(7),
    )
    scipy_low = float(res.confidence_interval.low)
    scipy_high = float(res.confidence_interval.high)

    assert est.ci_low == pytest.approx(scipy_low, abs=0.03 * scale)
    assert est.ci_high == pytest.approx(scipy_high, abs=0.03 * scale)


def test_bca_shifts_interval_right_for_right_skewed_mean() -> None:
    rng = np.random.default_rng(8)
    vals = _as_floats(rng.lognormal(mean=0.0, sigma=1.0, size=80))

    bca = bootstrap_ci(vals, n_resamples=4_000, seed=17, method="bca")
    pct = bootstrap_ci(vals, n_resamples=4_000, seed=17, method="percentile")

    assert bca.method == "bootstrap-bca"
    assert (bca.ci_low, bca.ci_high) != (pct.ci_low, pct.ci_high)
    assert bca.ci_high > pct.ci_high
    assert bca.ci_low > pct.ci_low
    bca_mid = (bca.ci_low + bca.ci_high) / 2.0
    pct_mid = (pct.ci_low + pct.ci_high) / 2.0
    assert bca_mid > pct_mid


def test_z0_undefined_falls_back_to_percentile_and_discloses() -> None:
    vals = [float(i) for i in range(20)]
    est = bootstrap_ci(vals, statistic=_unique_count, n_resamples=500, seed=2, method="bca")

    assert "bca z0 undefined" in est.method
    assert est.method.startswith("bootstrap-percentile")
    assert est.value == pytest.approx(20.0)
    assert math.isfinite(est.ci_low)
    assert math.isfinite(est.ci_high)
    assert est.ci_low <= est.ci_high
    assert est.ci_high < est.value


def test_discrete_binary_data_never_crashes_or_returns_nan() -> None:
    vals = [1.0] * 7 + [0.0] * 3
    est = bootstrap_ci(vals, n_resamples=2_000, seed=5, method="bca")

    assert est.method in {"bootstrap-bca", "bootstrap-percentile (bca z0 undefined)"}
    assert math.isfinite(est.ci_low)
    assert math.isfinite(est.ci_high)
    assert 0.0 <= est.ci_low <= est.ci_high <= 1.0
    assert est.value == pytest.approx(0.7)


def test_symmetric_data_bca_is_close_to_percentile() -> None:
    vals = [-2.0, -1.0, 0.0, 1.0, 2.0] * 6
    bca = bootstrap_ci(vals, n_resamples=4_000, seed=3, method="bca")
    pct = bootstrap_ci(vals, n_resamples=4_000, seed=3, method="percentile")

    assert bca.method == "bootstrap-bca"
    assert bca.ci_low == pytest.approx(pct.ci_low, abs=0.06)
    assert bca.ci_high == pytest.approx(pct.ci_high, abs=0.06)
    assert bca.value == pytest.approx(0.0)


def test_median_statistic_goes_through_jackknife_loop_and_brackets_median() -> None:
    rng = np.random.default_rng(13)
    data: NDArray[np.float64] = rng.normal(loc=5.0, scale=1.0, size=101)
    vals = _as_floats(data)
    sample_median = float(np.median(data))

    est = bootstrap_ci(vals, statistic=_median, n_resamples=800, seed=9, method="bca")

    assert est.method == "bootstrap-bca"
    assert est.value == pytest.approx(sample_median)
    assert est.ci_low <= sample_median <= est.ci_high
    assert math.isfinite(est.ci_low)
    assert math.isfinite(est.ci_high)
    assert 0.0 < est.width < 2.0


def test_same_seed_gives_identical_bca_estimate() -> None:
    rng = np.random.default_rng(0)
    vals = _as_floats(rng.normal(size=60))
    a = bootstrap_ci(vals, n_resamples=2_000, seed=123, method="bca")
    b = bootstrap_ci(vals, n_resamples=2_000, seed=123, method="bca")
    assert a == b
    assert a.method == "bootstrap-bca"


def test_nan_values_raise() -> None:
    with pytest.raises(ValueError, match="NaN"):
        bootstrap_ci([1.0, float("nan"), 2.0], method="bca")


def test_two_dimensional_input_raises() -> None:
    grid: NDArray[np.float64] = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="one-dimensional"):
        bootstrap_ci(grid, method="bca")


def test_n_equals_one_returns_degenerate_method() -> None:
    est = bootstrap_ci([2.5], n_resamples=500, seed=0, method="bca")
    assert est.method == "degenerate (n=1)"
    assert est.value == est.ci_low == est.ci_high == pytest.approx(2.5)
    assert est.n == 1
