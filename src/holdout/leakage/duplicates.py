from dataclasses import dataclass

from holdout.core.evalset import Eval
from holdout.leakage.ngram import jaccard, normalize, word_ngrams


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    case_a: str
    case_b: str
    similarity: float

    def __str__(self) -> str:
        return f"{self.case_a} ~ {self.case_b} (similarity={self.similarity:.3f})"


def find_near_duplicates(
    ev: Eval, *, ngram_size: int = 3, threshold: float = 0.8
) -> list[DuplicatePair]:
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    cases = ev.cases
    norms = [normalize(c.input) for c in cases]
    grams = [word_ngrams(c.input, ngram_size) for c in cases]

    pairs: list[DuplicatePair] = []
    for i in range(len(cases)):
        id_i = cases[i].id
        assert id_i is not None
        for j in range(i + 1, len(cases)):
            id_j = cases[j].id
            assert id_j is not None
            if norms[i] and norms[i] == norms[j]:
                pairs.append(DuplicatePair(id_i, id_j, 1.0))
                continue
            sim = jaccard(grams[i], grams[j])
            if sim >= threshold:
                pairs.append(DuplicatePair(id_i, id_j, sim))
    pairs.sort(key=lambda p: -p.similarity)
    return pairs
