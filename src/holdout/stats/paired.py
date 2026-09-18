import math
from collections.abc import Sequence
from itertools import product

import numpy as np
from numpy.typing import NDArray

from holdout.stats.bootstrap import bootstrap_ci
from holdout.stats.result import TestResult


def paired_diffs(scores_a: Sequence[float], scores_b: Sequence[float]) -> NDArray[np.float64]:
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("paired scores must be one-dimensional sequences")
    if a.size != b.size:
        raise ValueError(f"paired scores must have equal length, got {a.size} and {b.size}")
    if a.size < 2:
        raise ValueError(f"paired tests need at least 2 pairs, got {a.size}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("scores contain NaN or infinity")
    return b - a


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> TestResult:
    d = paired_diffs(scores_a, scores_b)
    n = int(d.size)
    effect = float(d.mean())
    ci = bootstrap_ci(d, level=level, n_resamples=n_resamples, seed=seed)

    rng = np.random.default_rng([seed, 0x01])
    centered = d - d.mean()
    indices = rng.integers(0, n, size=(n_resamples, n))
    null_means = centered[indices].mean(axis=1)
    extreme = int((np.abs(null_means) >= abs(effect)).sum())
    p = (extreme + 1) / (n_resamples + 1)

    return TestResult(
        test="paired-bootstrap",
        p_value=min(1.0, float(p)),
        effect=effect,
        ci=ci,
        n=n,
        detail="H0: mean(b - a) = 0, two-sided, shifted-null bootstrap",
    )


def _binom_cdf_half(k: int, m: int) -> float:
    if k < 0:
        return 0.0
    if k >= m:
        return 1.0
    log_half_m = m * math.log(0.5)
    total = 0.0
    for i in range(k + 1):
        log_term = math.lgamma(m + 1) - math.lgamma(i + 1) - math.lgamma(m - i + 1) + log_half_m
        total += math.exp(log_term)
    return min(1.0, total)


def mcnemar_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> TestResult:
    d = paired_diffs(scores_a, scores_b)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    for name, arr in (("scores_a", a), ("scores_b", b)):
        if not np.isin(arr, (0.0, 1.0)).all():
            raise ValueError(f"mcnemar_test requires binary scores; {name} has other values")

    n01 = int(((a == 0.0) & (b == 1.0)).sum())
    n10 = int(((a == 1.0) & (b == 0.0)).sum())
    m = n01 + n10
    p = 1.0 if m == 0 else min(1.0, 2.0 * _binom_cdf_half(min(n01, n10), m))

    ci = bootstrap_ci(d, level=level, n_resamples=n_resamples, seed=seed)
    return TestResult(
        test="mcnemar-exact",
        p_value=p,
        effect=float(d.mean()),
        ci=ci,
        n=int(d.size),
        detail=f"discordant pairs: improved={n01}, regressed={n10}",
    )


def permutation_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> TestResult:
    d = paired_diffs(scores_a, scores_b)
    n = int(d.size)
    effect = float(d.mean())
    ci = bootstrap_ci(d, level=level, n_resamples=n_resamples, seed=seed)
    tol = 1e-12 * max(1.0, abs(effect))

    if 2**n <= n_resamples:
        signs = np.asarray(list(product((1.0, -1.0), repeat=n)), dtype=np.float64)
        perm_means = (signs * d).mean(axis=1)
        p = float((np.abs(perm_means) >= abs(effect) - tol).mean())
        return TestResult(
            test="permutation-exact",
            p_value=min(1.0, p),
            effect=effect,
            ci=ci,
            n=n,
            detail=f"exact enumeration of 2^{n} sign assignments",
        )

    rng = np.random.default_rng([seed, 0x02])
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n_resamples, n))
    perm_means = (signs * d).mean(axis=1)
    extreme = int((np.abs(perm_means) >= abs(effect) - tol).sum())
    p = (extreme + 1) / (n_resamples + 1)
    return TestResult(
        test="permutation-mc",
        p_value=min(1.0, float(p)),
        effect=effect,
        ci=ci,
        n=n,
        detail=f"{n_resamples} Monte-Carlo sign flips",
    )
