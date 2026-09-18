from collections.abc import Callable, Sequence
from statistics import NormalDist
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from holdout.stats.estimate import Estimate

Statistic = Callable[[NDArray[np.float64]], float]

_NORMAL = NormalDist()


def _validate_sample(values: "Sequence[float] | NDArray[np.float64]") -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not np.isfinite(arr).all():
        raise ValueError("values contain NaN or infinity")
    return arr


def _resampled_statistics(
    arr: NDArray[np.float64],
    statistic: Statistic | None,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, NDArray[np.float64]]:
    indices = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    resamples = arr[indices]
    if statistic is None:
        return float(arr.mean()), resamples.mean(axis=1)
    point = float(statistic(arr))
    boot = np.asarray([statistic(row) for row in resamples], dtype=np.float64)
    return point, boot


def _jackknife_statistics(
    arr: NDArray[np.float64], statistic: Statistic | None
) -> NDArray[np.float64]:
    n = arr.size
    if statistic is None:
        return np.asarray((arr.sum() - arr) / (n - 1), dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = statistic(np.delete(arr, i))
    return out


def _bca_adjusted_level(q: float, z0: float, accel: float) -> float:
    z = _NORMAL.inv_cdf(q)
    num = z0 + z
    d = 1.0 - accel * num
    if d <= 0.0:
        return 1.0 if num > 0 else 0.0
    return _NORMAL.cdf(z0 + num / d)


def bootstrap_ci(
    values: "Sequence[float] | NDArray[np.float64]",
    *,
    statistic: Statistic | None = None,
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
    method: Literal["bca", "percentile"] = "bca",
) -> Estimate:
    arr = _validate_sample(values)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    if arr.size == 1:
        v = float(arr[0]) if statistic is None else float(statistic(arr))
        return Estimate(value=v, ci_low=v, ci_high=v, n=1, level=level, method="degenerate (n=1)")

    rng = np.random.default_rng(seed)
    point, boot = _resampled_statistics(arr, statistic, n_resamples, rng)
    alpha = 1.0 - level
    n = int(arr.size)

    if method == "percentile":
        lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
        return Estimate(
            value=point,
            ci_low=float(lo),
            ci_high=float(hi),
            n=n,
            level=level,
            method="bootstrap-percentile",
        )

    below = float((boot < point).sum())
    ties = float((boot == point).sum())
    prop = (below + 0.5 * ties) / float(boot.size)
    if prop <= 0.0 or prop >= 1.0:
        lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
        return Estimate(
            value=point,
            ci_low=float(lo),
            ci_high=float(hi),
            n=n,
            level=level,
            method="bootstrap-percentile (bca z0 undefined)",
        )
    z0 = _NORMAL.inv_cdf(prop)

    jack = _jackknife_statistics(arr, statistic)
    diffs = jack.mean() - jack
    denom = float((diffs**2).sum()) ** 1.5
    accel = 0.0 if denom == 0.0 else float((diffs**3).sum()) / (6.0 * denom)

    a1 = _bca_adjusted_level(alpha / 2.0, z0, accel)
    a2 = _bca_adjusted_level(1.0 - alpha / 2.0, z0, accel)
    lo = np.quantile(boot, min(a1, a2))
    hi = np.quantile(boot, max(a1, a2))
    return Estimate(
        value=point,
        ci_low=float(lo),
        ci_high=float(hi),
        n=n,
        level=level,
        method="bootstrap-bca",
    )
