from holdout import Case, Eval, run
from holdout.providers import StaticTarget
from holdout.regression import compare
from holdout.scorers import ExactMatch
from holdout.testing import assert_no_regression

N = 60
cases = [Case(input=f"task {i}", reference="done", id=f"t{i:02d}") for i in range(N)]
ev = Eval("worker-tasks", cases, [ExactMatch()])


def target(wrong: set[int], name: str) -> StaticTarget:
    return StaticTarget(
        {f"task {i}": ("failed" if i in wrong else "done") for i in range(N)}, name=name
    )


baseline = run(ev, target=target(set(), "prompt-v1"), seed=7)

wobble = run(ev, target=target({3, 41}, "prompt-v2-wobble"), seed=7)
verdict = compare(baseline, wobble, seed=7)
print(verdict.summary())
print()
assert_no_regression(baseline, wobble, seed=7)
print("scenario 1: naked score dropped 0.967 -> 0.933, but the gate correctly stays green\n")

broken = run(ev, target=target(set(range(12)), "prompt-v2-broken"), seed=7)
print(compare(baseline, broken, seed=7).summary())
print()
try:
    assert_no_regression(baseline, broken, seed=7)
except AssertionError as exc:
    print(f"scenario 2: the gate fails, with evidence:\n{exc}")
