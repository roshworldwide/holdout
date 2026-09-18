from holdout import Case, Eval, Run, run
from holdout.providers import StaticTarget
from holdout.scorers import ExactMatch
from holdout.testing import (
    assert_adequately_powered,
    assert_no_leakage,
    assert_no_regression,
    llm_eval,
)

N = 50
SYSTEM_PROMPT = "You are a terse arithmetic assistant. Answer with the number only."

cases = [Case(input=f"{i} + {i} = ?", reference=str(2 * i), id=f"add{i:02d}") for i in range(N)]
arithmetic = Eval("arithmetic", cases, [ExactMatch()])

prompt_v1 = StaticTarget({f"{i} + {i} = ?": str(2 * i) for i in range(N)}, name="prompt-v1")
prompt_v2 = StaticTarget(
    {f"{i} + {i} = ?": str(2 * i) for i in range(N)},
    name="prompt-v2",
)


def test_eval_is_not_contaminated_by_the_prompt() -> None:
    assert_no_leakage(arithmetic, SYSTEM_PROMPT)


def test_prompt_v2_does_not_regress() -> None:
    baseline = run(arithmetic, target=prompt_v1, seed=7)
    candidate = run(arithmetic, target=prompt_v2, seed=7)
    comparison = assert_no_regression(baseline, candidate, alpha=0.05, seed=7)
    assert comparison.verdict in ("no_significant_change", "improved")


def test_comparison_can_detect_what_we_care_about() -> None:
    baseline = run(arithmetic, target=prompt_v1, seed=7)
    candidate = run(arithmetic, target=prompt_v2, seed=7)
    assert_adequately_powered(baseline, candidate, mde=0.15)


@llm_eval(arithmetic, target=prompt_v1, seed=7)
def test_baseline_quality_floor(run: Run) -> None:
    est = run.metrics()["exact_match"]
    assert est.ci_low >= 0.9, f"quality floor breached: {est}"
