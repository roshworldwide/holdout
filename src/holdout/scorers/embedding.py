import math
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from holdout.core.case import Case
from holdout.core.scoring import Score, Scorer


@runtime_checkable
class EmbeddingBackend(Protocol):
    @property
    def name(self) -> str:
        ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector dimensions differ: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingSimilarity(Scorer):
    requires_reference = True

    def __init__(self, backend: EmbeddingBackend, *, threshold: float | None = None) -> None:
        if threshold is not None and not -1.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [-1, 1], got {threshold}")
        self._backend = backend
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "embedding_similarity"

    def config(self) -> Mapping[str, object]:
        return {"backend": self._backend.name, "threshold": self._threshold}

    async def score(self, case: Case, output: str) -> Score:
        assert case.reference is not None
        vec_out, vec_ref = await self._backend.embed([output, case.reference])
        sim = cosine_similarity(vec_out, vec_ref)
        if self._threshold is None:
            return Score(value=sim, kind="continuous", detail=f"backend={self._backend.name}")
        return Score(
            value=1.0 if sim >= self._threshold else 0.0,
            kind="binary",
            detail=f"cosine={sim:.4f}, threshold={self._threshold}",
        )
