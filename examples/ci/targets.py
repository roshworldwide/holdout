from holdout.providers.static import StaticTarget

candidate = StaticTarget(
    {f"What is {i} + {i}?": str(i + i) for i in range(40)},
    name="examples-ci-candidate",
)
