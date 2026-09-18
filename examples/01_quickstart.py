from holdout import Case, Eval, run
from holdout.providers import StaticTarget
from holdout.scorers import ExactMatch

cases = [
    Case(input="What is the capital of France?", reference="Paris"),
    Case(input="What is 12 * 12?", reference="144"),
    Case(input="Boiling point of water at sea level, in Celsius?", reference="100"),
    Case(input="Chemical symbol for gold?", reference="Au"),
    Case(input="How many continents are there?", reference="7"),
    Case(input="What year did the Berlin Wall fall?", reference="1989"),
    Case(input="Largest planet in the solar system?", reference="Jupiter"),
    Case(input="Square root of 256?", reference="16"),
]

qa = Eval("general-qa", cases, [ExactMatch()])

model = StaticTarget(
    {
        "What is the capital of France?": "Paris",
        "What is 12 * 12?": "144",
        "Boiling point of water at sea level, in Celsius?": "100",
        "Chemical symbol for gold?": "Ag",
        "How many continents are there?": "7",
        "What year did the Berlin Wall fall?": "1989",
        "Largest planet in the solar system?": "Saturn",
        "Square root of 256?": "16",
    },
    name="demo-model",
)

result = run(qa, target=model, seed=7)
print(result.summary())
