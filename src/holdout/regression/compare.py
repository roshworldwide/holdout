from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from holdout.core.run import Run
from holdout.stats.bootstrap import bootstrap_ci
from holdout.stats.correction import benjamini_hochberg, holm_bonferroni
from holdout.stats.estimate import Estimate
from holdout.stats.paired import mcnemar_test, paired_bootstrap_test, permutation_test
from holdout.stats.result import TestResult

Verdict = Literal["improved", "regressed", "no_significant_change", "insufficient_data"]
Correction = Literal["benjamini-hochberg", "holm", "none"]
PairedTest = Literal["auto", "paired-bootstrap", "mcnemar", "permutation"]

_DISPLAY = {
    "improved": "IMPROVED",
    "regressed": "REGRESSED",
    "no_significant_change": "no sig. Δ",
    "insufficient_data": "insufficient data",
}


@dataclass(frozen=True, slots=True)
class MetricComparison:
    metric: str
    verdict: Verdict
    n_pairs: int
    result: TestResult | None = None
    p_adjusted: float | None = None
    baseline: Estimate | None = None
    candidate: Estimate | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "verdict": self.verdict,
            "n_pairs": self.n_pairs,
            "result": self.result.to_dict() if self.result else None,
            "p_adjusted": self.p_adjusted,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class RunComparison:
    eval_name: str
    baseline_run_id: str
    candidate_run_id: str
    baseline_target: str
    candidate_target: str
    alpha: float
    correction: Correction
    comparisons: tuple[MetricComparison, ...]
    warnings: tuple[str, ...]

    @property
    def verdict(self) -> Verdict:
        verdicts = {c.verdict for c in self.comparisons}
        if "regressed" in verdicts:
            return "regressed"
        if "improved" in verdicts:
            return "improved"
        if "no_significant_change" in verdicts:
            return "no_significant_change"
        return "insufficient_data"

    @property
    def regressed(self) -> tuple[MetricComparison, ...]:
        return tuple(c for c in self.comparisons if c.verdict == "regressed")

    @property
    def improved(self) -> tuple[MetricComparison, ...]:
        return tuple(c for c in self.comparisons if c.verdict == "improved")

    def summary(self) -> str:
        head = (
            f"{self.eval_name}: {self.baseline_target} ({self.baseline_run_id[:12]}) vs "
            f"{self.candidate_target} ({self.candidate_run_id[:12]})"
        )
        lines = [head]
        width = max((len(c.metric) for c in self.comparisons), default=6)
        for c in self.comparisons:
            tag = _DISPLAY[c.verdict]
            if c.result is None:
                lines.append(f"  {c.metric:<{width}}  {tag} ({c.note})")
                continue
            r = c.result
            p_part = (
                f"p={r.p_value:.4g}"
                if c.p_adjusted is None
                else f"p={c.p_adjusted:.4g} ({self.correction}-adjusted)"
            )
            lines.append(
                f"  {c.metric:<{width}}  {tag:<17}  Δ={r.effect:+.3f}  "
                f"[{r.ci.level * 100:g}% CI {r.ci.ci_low:+.3f}, {r.ci.ci_high:+.3f}]  "
                f"{r.test}  {p_part}  n={c.n_pairs}"
            )
        lines.append(f"verdict: {_DISPLAY[self.verdict]} (alpha={self.alpha:g})")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "eval_name": self.eval_name,
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "baseline_target": self.baseline_target,
            "candidate_target": self.candidate_target,
            "alpha": self.alpha,
            "correction": self.correction,
            "verdict": self.verdict,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "warnings": list(self.warnings),
        }


def _aligned_scores(
    baseline: Run, candidate: Run, metric: str
) -> tuple[list[str], list[float], list[float], int, int]:
    a_scores = baseline.case_scores(metric)
    b_scores = candidate.case_scores(metric)
    ids = sorted(set(a_scores) & set(b_scores))
    return (
        ids,
        [a_scores[i] for i in ids],
        [b_scores[i] for i in ids],
        len(a_scores) - len(ids),
        len(b_scores) - len(ids),
    )


def compare(
    baseline: Run,
    candidate: Run,
    *,
    alpha: float = 0.05,
    correction: Correction = "benjamini-hochberg",
    test: PairedTest = "auto",
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> RunComparison:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    warnings: list[str] = []
    if baseline.eval_fingerprint != candidate.eval_fingerprint:
        warnings.append(
            "the two runs were made on different eval datasets (fingerprint mismatch); "
            "comparison uses only case ids present in both, and assumes equal ids mean "
            "identical inputs"
        )

    shared = [m for m in baseline.scorer_names if m in candidate.scorer_names]
    skipped = sorted(set(baseline.scorer_names).symmetric_difference(candidate.scorer_names))
    if skipped:
        warnings.append(f"metrics present in only one run were skipped: {skipped}")
    if not shared:
        raise ValueError(
            f"runs share no metrics (baseline has {list(baseline.scorer_names)}, "
            f"candidate has {list(candidate.scorer_names)})"
        )

    tested: list[tuple[str, list[float], list[float], TestResult]] = []
    untestable: dict[str, MetricComparison] = {}
    for metric in shared:
        ids, a, b, dropped_a, dropped_b = _aligned_scores(baseline, candidate, metric)
        if dropped_a or dropped_b:
            warnings.append(
                f"{metric}: dropped {dropped_a} baseline / {dropped_b} candidate "
                "unpaired case(s) (errors or dataset mismatch)"
            )
        if len(ids) < 2:
            untestable[metric] = MetricComparison(
                metric=metric,
                verdict="insufficient_data",
                n_pairs=len(ids),
                note=f"only {len(ids)} paired case(s); a paired test needs at least 2",
            )
            continue
        both_binary = baseline.score_kind(metric) == "binary" and (
            candidate.score_kind(metric) == "binary"
        )
        chosen = test
        if chosen == "auto":
            chosen = "mcnemar" if both_binary else "paired-bootstrap"
        if chosen == "mcnemar":
            result = mcnemar_test(a, b, level=level, n_resamples=n_resamples, seed=seed)
        elif chosen == "permutation":
            result = permutation_test(a, b, level=level, n_resamples=n_resamples, seed=seed)
        else:
            result = paired_bootstrap_test(a, b, level=level, n_resamples=n_resamples, seed=seed)
        tested.append((metric, a, b, result))

    p_raw = [r.p_value for _, _, _, r in tested]
    if correction == "benjamini-hochberg":
        p_adj: Sequence[float | None] = benjamini_hochberg(p_raw)
    elif correction == "holm":
        p_adj = holm_bonferroni(p_raw)
    elif correction == "none":
        p_adj = list(p_raw)
    else:
        raise ValueError(f"unknown correction {correction!r}")

    comparisons: list[MetricComparison] = []
    for (metric, a, b, result), adj in zip(tested, p_adj, strict=True):
        assert adj is not None
        if adj <= alpha:
            verdict: Verdict = "improved" if result.effect > 0 else "regressed"
        else:
            verdict = "no_significant_change"
        comparisons.append(
            MetricComparison(
                metric=metric,
                verdict=verdict,
                n_pairs=result.n,
                result=result,
                p_adjusted=float(adj),
                baseline=bootstrap_ci(a, level=level, n_resamples=n_resamples, seed=seed),
                candidate=bootstrap_ci(b, level=level, n_resamples=n_resamples, seed=seed),
            )
        )
    ordered = [
        next(c for c in comparisons if c.metric == m) if m not in untestable else untestable[m]
        for m in shared
    ]

    return RunComparison(
        eval_name=baseline.eval_name,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_target=baseline.target_name,
        candidate_target=candidate.target_name,
        alpha=alpha,
        correction=correction,
        comparisons=tuple(ordered),
        warnings=tuple(warnings),
    )
