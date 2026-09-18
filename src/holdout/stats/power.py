import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

from holdout.stats.paired import paired_diffs

_NORMAL = NormalDist()


@dataclass(frozen=True, slots=True)
class PowerAnalysis:
    n: int
    mde: float
    sd_diff: float
    alpha: float
    power: float

    def __str__(self) -> str:
        return (
            f"n={self.n} pairs detects |Δ| >= {self.mde:.4f} at alpha={self.alpha:g} "
            f"with power {self.power:g} (sd_diff={self.sd_diff:.4f})"
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "mde": self.mde,
            "sd_diff": self.sd_diff,
            "alpha": self.alpha,
            "power": self.power,
        }


def _validate_design(sd_diff: float, alpha: float, power: float) -> tuple[float, float]:
    if sd_diff <= 0.0:
        raise ValueError(f"sd_diff must be > 0, got {sd_diff}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    return _NORMAL.inv_cdf(1.0 - alpha / 2.0), _NORMAL.inv_cdf(power)


def minimum_detectable_effect(
    n: int, sd_diff: float, *, alpha: float = 0.05, power: float = 0.80
) -> PowerAnalysis:
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    z_alpha, z_power = _validate_design(sd_diff, alpha, power)
    mde = (z_alpha + z_power) * sd_diff / math.sqrt(n)
    return PowerAnalysis(n=n, mde=mde, sd_diff=sd_diff, alpha=alpha, power=power)


def required_sample_size(
    mde: float, sd_diff: float, *, alpha: float = 0.05, power: float = 0.80
) -> PowerAnalysis:
    if mde <= 0.0:
        raise ValueError(f"mde must be > 0, got {mde}")
    z_alpha, z_power = _validate_design(sd_diff, alpha, power)
    n = max(2, math.ceil(((z_alpha + z_power) * sd_diff / mde) ** 2))
    return PowerAnalysis(n=n, mde=mde, sd_diff=sd_diff, alpha=alpha, power=power)


def paired_binary_sd(p01: float, p10: float) -> float:
    if p01 < 0.0 or p10 < 0.0 or p01 + p10 > 1.0:
        raise ValueError(f"p01 and p10 must be >= 0 with p01 + p10 <= 1, got {p01} and {p10}")
    var = p01 + p10 - (p01 - p10) ** 2
    return math.sqrt(max(var, 0.0))


def sd_diff_from_scores(scores_a: Sequence[float], scores_b: Sequence[float]) -> float:
    d = paired_diffs(scores_a, scores_b)
    return float(d.std(ddof=1))
