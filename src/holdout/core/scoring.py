from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from holdout.core.hashing import fingerprint

if TYPE_CHECKING:
    from holdout.core.case import Case

ScoreKind = Literal["binary", "continuous"]


@dataclass(frozen=True, slots=True)
class Score:
    value: float
    kind: ScoreKind
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "binary" and self.value not in (0.0, 1.0):
            raise ValueError(f"binary scores must be 0.0 or 1.0, got {self.value}")

    def to_dict(self) -> dict[str, object]:
        return {"value": self.value, "kind": self.kind, "detail": self.detail}


class Scorer(ABC):
    requires_reference: ClassVar[bool] = False

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def config(self) -> Mapping[str, object]:
        return {}

    @property
    def fingerprint(self) -> str:
        return fingerprint({"scorer": self.name, "config": dict(self.config())})

    @abstractmethod
    async def score(self, case: "Case", output: str) -> Score:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self.config())!r})"
