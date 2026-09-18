import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Self

from holdout.core.case import Case
from holdout.core.hashing import fingerprint
from holdout.core.scoring import Scorer


class Eval:
    def __init__(self, name: str, cases: Sequence[Case], scorers: Sequence[Scorer]) -> None:
        if not name:
            raise ValueError("eval name must be non-empty")
        if not cases:
            raise ValueError(f"eval {name!r} has no cases")
        if not scorers:
            raise ValueError(f"eval {name!r} has no scorers")

        normalized = tuple(c if c.id is not None else replace(c, id=c.content_id()) for c in cases)

        ids = [c.id for c in normalized]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1 and i is not None})
            raise ValueError(
                f"eval {name!r} has duplicate case ids: {dupes[:5]}. Identical duplicate "
                "cases add no statistical information and break paired comparison; "
                "deduplicate them or assign explicit distinct ids."
            )

        scorer_names = [s.name for s in scorers]
        if len(set(scorer_names)) != len(scorer_names):
            raise ValueError(f"eval {name!r} has duplicate scorer names: {scorer_names}")

        for scorer in scorers:
            if scorer.requires_reference:
                missing = [c.id for c in normalized if c.reference is None]
                if missing:
                    raise ValueError(
                        f"scorer {scorer.name!r} requires a reference on every case, but "
                        f"{len(missing)} case(s) have none (first: {missing[0]!r})"
                    )

        self._name = name
        self._cases = normalized
        self._scorers = tuple(scorers)

    @property
    def name(self) -> str:
        return self._name

    @property
    def cases(self) -> tuple[Case, ...]:
        return self._cases

    @property
    def scorers(self) -> tuple[Scorer, ...]:
        return self._scorers

    @property
    def fingerprint(self) -> str:
        return fingerprint({"name": self._name, "cases": [c.to_dict() for c in self._cases]})

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        scorers: Sequence[Scorer],
        name: str | None = None,
    ) -> Self:
        p = Path(path)
        cases: list[Case] = []
        with p.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
                if "input" not in obj:
                    raise ValueError(f"{p}:{lineno}: missing required field 'input'")
                cases.append(
                    Case(
                        input=obj["input"],
                        reference=obj.get("reference"),
                        id=obj.get("id"),
                        metadata=obj.get("metadata", {}),
                    )
                )
        return cls(name=name or p.stem, cases=cases, scorers=scorers)

    def __len__(self) -> int:
        return len(self._cases)

    def __repr__(self) -> str:
        return (
            f"Eval(name={self._name!r}, cases={len(self._cases)}, "
            f"scorers={[s.name for s in self._scorers]!r})"
        )
