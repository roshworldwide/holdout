import functools
import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

from holdout.core.evalset import Eval
from holdout.core.run import Run
from holdout.core.runner import run as _run_eval
from holdout.core.target import Target
from holdout.regression.compare import Correction, PairedTest, RunComparison, compare
from holdout.stats.power import PowerAnalysis, required_sample_size, sd_diff_from_scores

if TYPE_CHECKING:
    from holdout.store.run_store import RunStore

P = ParamSpec("P")
R = TypeVar("R")


def assert_no_regression(
    baseline: Run,
    candidate: Run,
    *,
    alpha: float = 0.05,
    correction: Correction = "benjamini-hochberg",
    test: PairedTest = "auto",
    n_resamples: int = 10_000,
    seed: int = 0,
) -> RunComparison:
    cmp = compare(
        baseline,
        candidate,
        alpha=alpha,
        correction=correction,
        test=test,
        n_resamples=n_resamples,
        seed=seed,
    )
    if cmp.verdict == "regressed":
        names = ", ".join(c.metric for c in cmp.regressed)
        raise AssertionError(f"regression detected on: {names}\n{cmp.summary()}")
    if cmp.verdict == "insufficient_data":
        raise AssertionError(
            "refusing to certify 'no regression': no metric had enough paired data "
            f"to test\n{cmp.summary()}"
        )
    return cmp


def assert_significant_improvement(
    baseline: Run,
    candidate: Run,
    *,
    metric: str | None = None,
    alpha: float = 0.05,
    correction: Correction = "benjamini-hochberg",
    test: PairedTest = "auto",
    n_resamples: int = 10_000,
    seed: int = 0,
) -> RunComparison:
    cmp = compare(
        baseline,
        candidate,
        alpha=alpha,
        correction=correction,
        test=test,
        n_resamples=n_resamples,
        seed=seed,
    )
    if cmp.regressed:
        names = ", ".join(c.metric for c in cmp.regressed)
        raise AssertionError(
            f"candidate regressed on {names} — not a clean improvement\n{cmp.summary()}"
        )
    if metric is not None:
        match = next((c for c in cmp.comparisons if c.metric == metric), None)
        if match is None:
            known = [c.metric for c in cmp.comparisons]
            raise AssertionError(f"metric {metric!r} not in comparison (has {known})")
        if match.verdict != "improved":
            raise AssertionError(
                f"{metric} did not significantly improve (verdict: {match.verdict})\n"
                f"{cmp.summary()}"
            )
    elif not cmp.improved:
        raise AssertionError(
            f"no metric significantly improved (verdict: {cmp.verdict})\n{cmp.summary()}"
        )
    return cmp


def assert_adequately_powered(
    baseline: Run,
    candidate: Run,
    *,
    mde: float,
    metric: str | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, PowerAnalysis]:
    metrics = (
        [metric]
        if metric is not None
        else [m for m in baseline.scorer_names if m in candidate.scorer_names]
    )
    if not metrics:
        raise AssertionError("runs share no metrics to power-check")

    analyses: dict[str, PowerAnalysis] = {}
    underpowered: list[str] = []
    for m in metrics:
        a_scores = baseline.case_scores(m)
        b_scores = candidate.case_scores(m)
        ids = sorted(set(a_scores) & set(b_scores))
        if len(ids) < 2:
            underpowered.append(f"{m}: only {len(ids)} paired case(s)")
            continue
        a = [a_scores[i] for i in ids]
        b = [b_scores[i] for i in ids]
        sd = sd_diff_from_scores(a, b)
        if sd == 0.0:
            continue
        analysis = required_sample_size(mde, sd, alpha=alpha, power=power)
        analyses[m] = analysis
        if len(ids) < analysis.n:
            underpowered.append(
                f"{m}: have {len(ids)} pairs, need {analysis.n} to detect "
                f"|Δ| >= {mde:g} (sd_diff={sd:.4f}, alpha={alpha:g}, power={power:g})"
            )
    if underpowered:
        raise AssertionError(
            "comparison is underpowered — a null result here would be meaningless:\n  "
            + "\n  ".join(underpowered)
        )
    return analyses


def assert_no_leakage(
    ev: Eval,
    corpus: "str | Sequence[str] | Target",
    *,
    ngram_size: int = 5,
    threshold: float = 0.5,
    duplicate_threshold: float | None = 0.8,
) -> None:
    from holdout.leakage.contamination import check_contamination
    from holdout.leakage.duplicates import find_near_duplicates

    texts: str | Sequence[str]
    if isinstance(corpus, str | Sequence):
        texts = corpus
    else:
        system = getattr(corpus, "system", None)
        if not isinstance(system, str) or not system:
            raise ValueError(
                f"target {corpus.name!r} exposes no system prompt to audit; pass the "
                "prompt/few-shot text explicitly"
            )
        texts = system

    problems: list[str] = []
    report = check_contamination(ev, texts, ngram_size=ngram_size, threshold=threshold)
    if not report.clean:
        problems.append(report.summary())
    if duplicate_threshold is not None:
        dupes = find_near_duplicates(ev, threshold=duplicate_threshold)
        if dupes:
            listing = "\n".join(f"  {d}" for d in dupes)
            problems.append(
                f"near-duplicate cases inflate the effective sample size "
                f"(n={len(ev.cases)} is overstated):\n{listing}"
            )
    if problems:
        raise AssertionError("eval leakage detected:\n" + "\n".join(problems))


def llm_eval(
    ev: Eval,
    *,
    target: Target,
    seed: int = 0,
    store: "RunStore | str | None" = None,
    max_concurrency: int = 8,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    def decorate(fn: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            result = _run_eval(ev, target=target, seed=seed, max_concurrency=max_concurrency)
            if store is not None:
                from holdout.store.run_store import RunStore

                (store if isinstance(store, RunStore) else RunStore(store)).save(result)
            return fn(*args, run=result, **kwargs)

        sig = inspect.signature(fn)
        params = [p for name, p in sig.parameters.items() if name != "run"]
        wrapper.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
        try:
            import pytest

            return cast("Callable[..., R]", pytest.mark.llm_eval(wrapper))
        except ImportError:  # pragma: no cover - pytest is present in every dev env
            return wrapper

    return decorate
