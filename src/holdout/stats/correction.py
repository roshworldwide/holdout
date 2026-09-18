from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def _validate_p_values(p_values: Sequence[float]) -> NDArray[np.float64]:
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"p_values must be one-dimensional, got shape {p.shape}")
    if p.size and (np.isnan(p).any() or (p < 0.0).any() or (p > 1.0).any()):
        raise ValueError("p_values must all be in [0, 1]")
    return p


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    p = _validate_p_values(p_values)
    m = int(p.size)
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    q = ranked * (m / np.arange(1, m + 1, dtype=np.float64))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(m, dtype=np.float64)
    out[order] = q
    return [float(x) for x in out]


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    p = _validate_p_values(p_values)
    m = int(p.size)
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    q = (m - np.arange(m, dtype=np.float64)) * ranked
    q = np.maximum.accumulate(q)
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(m, dtype=np.float64)
    out[order] = q
    return [float(x) for x in out]
