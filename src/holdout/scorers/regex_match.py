import re
from collections.abc import Mapping

from holdout.core.case import Case
from holdout.core.scoring import Score, Scorer


class RegexMatch(Scorer):
    def __init__(self, pattern: str, *, ignore_case: bool = False) -> None:
        flags = re.IGNORECASE if ignore_case else 0
        self._pattern = re.compile(pattern, flags)
        self._ignore_case = ignore_case

    @property
    def name(self) -> str:
        return "regex_match"

    def config(self) -> Mapping[str, object]:
        return {"pattern": self._pattern.pattern, "ignore_case": self._ignore_case}

    async def score(self, case: Case, output: str) -> Score:
        del case
        matched = self._pattern.search(output) is not None
        return Score(
            value=1.0 if matched else 0.0,
            kind="binary",
            detail=f"pattern={self._pattern.pattern!r}",
        )
