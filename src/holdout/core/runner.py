import asyncio
from datetime import UTC, datetime
from time import perf_counter

import holdout
from holdout.core.case import Case
from holdout.core.evalset import Eval
from holdout.core.run import CaseResult, Run
from holdout.core.scoring import Score, Scorer
from holdout.core.target import Target


async def _run_case(
    case: Case,
    target: Target,
    scorers: tuple[Scorer, ...],
    seed: int | None,
    sem: asyncio.Semaphore,
) -> CaseResult:
    assert case.id is not None
    async with sem:
        t0 = perf_counter()
        try:
            completion = await target.generate(case.input, seed=seed)
        except Exception as exc:
            return CaseResult(
                case_id=case.id,
                output=None,
                error=f"generation failed: {type(exc).__name__}: {exc}",
                latency_s=perf_counter() - t0,
            )
        latency = perf_counter() - t0

        scores: dict[str, Score] = {}
        errors: list[str] = []
        for scorer in scorers:
            try:
                scores[scorer.name] = await scorer.score(case, completion.text)
            except Exception as exc:
                errors.append(f"scorer {scorer.name!r} failed: {type(exc).__name__}: {exc}")
        return CaseResult(
            case_id=case.id,
            output=completion.text,
            scores=scores,
            error="; ".join(errors) if errors else None,
            latency_s=latency,
        )


async def arun(
    ev: Eval,
    *,
    target: Target,
    seed: int | None = None,
    max_concurrency: int = 8,
) -> Run:
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
    sem = asyncio.Semaphore(max_concurrency)
    results = await asyncio.gather(
        *(_run_case(case, target, ev.scorers, seed, sem) for case in ev.cases)
    )
    return Run(
        eval_name=ev.name,
        eval_fingerprint=ev.fingerprint,
        target_name=target.name,
        target_fingerprint=target.fingerprint,
        scorer_names=tuple(s.name for s in ev.scorers),
        scorer_fingerprints=tuple(s.fingerprint for s in ev.scorers),
        seed=seed,
        results=tuple(results),
        created_at=datetime.now(UTC).isoformat(),
        holdout_version=holdout.__version__,
    )


def run(
    ev: Eval,
    *,
    target: Target,
    seed: int | None = None,
    max_concurrency: int = 8,
) -> Run:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(arun(ev, target=target, seed=seed, max_concurrency=max_concurrency))
    raise RuntimeError("run() cannot be called from a running event loop; use arun() instead")
