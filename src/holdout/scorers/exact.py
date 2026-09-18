import re
from collections.abc import Mapping

from holdout.core.case import Case
from holdout.core.scoring import Score, Scorer

_WHITESPACE = re.compile(r"\s+")


class ExactMatch(Scorer):
    requires_reference = True

    def __init__(self, *, normalize: bool = True) -> None:
        self._normalize = normalize

    @property
    def name(self) -> str:
        return "exact_match"

    def config(self) -> Mapping[str, object]:
        return {"normalize": self._normalize}

    def _canon(self, text: str) -> str:
        if not self._normalize:
            return text
        return _WHITESPACE.sub(" ", text.strip().casefold())

    async def score(self, case: Case, output: str) -> Score:
        assert case.reference is not None
        matched = self._canon(output) == self._canon(case.reference)
        return Score(value=1.0 if matched else 0.0, kind="binary")
