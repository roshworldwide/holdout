from dataclasses import dataclass

from holdout.stats.estimate import Estimate


@dataclass(frozen=True, slots=True)
class TestResult:
    test: str
    p_value: float
    effect: float
    ci: Estimate
    n: int
    detail: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"p_value must be in [0, 1], got {self.p_value}")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")

    def __str__(self) -> str:
        pct = f"{self.ci.level * 100:g}"
        return (
            f"Δ={self.effect:+.3f} [{pct}% CI {self.ci.ci_low:+.3f}, "
            f"{self.ci.ci_high:+.3f}], p={self.p_value:.4g} ({self.test}, n={self.n})"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "test": self.test,
            "p_value": self.p_value,
            "effect": self.effect,
            "ci": self.ci.to_dict(),
            "n": self.n,
            "detail": self.detail,
        }
