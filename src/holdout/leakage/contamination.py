from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from holdout.core.evalset import Eval
from holdout.leakage.ngram import containment, tokens, word_ngrams
from holdout.scorers.embedding import EmbeddingBackend, cosine_similarity

ContaminationKind = Literal["exact-substring", "ngram-overlap", "embedding-similarity"]


@dataclass(frozen=True, slots=True)
class ContaminationFinding:
    case_id: str
    field: Literal["input", "reference"]
    kind: ContaminationKind
    score: float
    detail: str

    def __str__(self) -> str:
        return f"{self.case_id}.{self.field}: {self.kind} (score={self.score:.3f}) — {self.detail}"


@dataclass(frozen=True, slots=True)
class ContaminationReport:
    findings: tuple[ContaminationFinding, ...]
    n_cases: int
    method: str

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def contaminated_case_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for f in self.findings:
            seen.setdefault(f.case_id, None)
        return tuple(seen)

    def summary(self) -> str:
        head = (
            f"contamination check ({self.method}): "
            f"{len(self.contaminated_case_ids)}/{self.n_cases} case(s) flagged"
        )
        lines = [head]
        lines.extend(f"  {f}" for f in self.findings)
        if self.clean:
            lines.append("  no contamination detected")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_cases": self.n_cases,
            "method": self.method,
            "clean": self.clean,
            "findings": [
                {
                    "case_id": f.case_id,
                    "field": f.field,
                    "kind": f.kind,
                    "score": f.score,
                    "detail": f.detail,
                }
                for f in self.findings
            ],
        }


def _as_corpus(corpus: str | Sequence[str]) -> list[str]:
    return [corpus] if isinstance(corpus, str) else list(corpus)


def check_contamination(
    ev: Eval,
    corpus: str | Sequence[str],
    *,
    ngram_size: int = 5,
    threshold: float = 0.5,
    min_tokens: int = 3,
) -> ContaminationReport:
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    texts = _as_corpus(corpus)
    padded = [" " + " ".join(tokens(t)) + " " for t in texts]
    corpus_grams = [word_ngrams(t, ngram_size) for t in texts]

    findings: list[ContaminationFinding] = []
    for case in ev.cases:
        assert case.id is not None
        fields: list[tuple[Literal["input", "reference"], str]] = [("input", case.input)]
        if case.reference is not None:
            fields.append(("reference", case.reference))
        for field_name, text in fields:
            toks = tokens(text)
            if not toks:
                continue
            needle = " " + " ".join(toks) + " "
            hit = next((i for i, t in enumerate(padded) if needle in t), None)
            if hit is not None:
                findings.append(
                    ContaminationFinding(
                        case_id=case.id,
                        field=field_name,
                        kind="exact-substring",
                        score=1.0,
                        detail=f"appears verbatim in corpus text #{hit}",
                    )
                )
                continue
            grams = word_ngrams(text, ngram_size)
            if len(toks) < min_tokens or not grams:
                continue
            best, best_idx = 0.0, -1
            for i, cg in enumerate(corpus_grams):
                c = containment(grams, cg)
                if c > best:
                    best, best_idx = c, i
            if best >= threshold:
                findings.append(
                    ContaminationFinding(
                        case_id=case.id,
                        field=field_name,
                        kind="ngram-overlap",
                        score=best,
                        detail=(
                            f"{best:.0%} of its {ngram_size}-grams appear in corpus "
                            f"text #{best_idx}"
                        ),
                    )
                )
    findings.sort(key=lambda f: -f.score)
    return ContaminationReport(
        findings=tuple(findings),
        n_cases=len(ev.cases),
        method=f"exact-substring + {ngram_size}-gram containment >= {threshold:g}",
    )


async def check_contamination_embeddings(
    ev: Eval,
    corpus: str | Sequence[str],
    backend: EmbeddingBackend,
    *,
    threshold: float = 0.9,
) -> ContaminationReport:
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    texts = _as_corpus(corpus)
    if not texts:
        raise ValueError("corpus is empty")

    fields: list[tuple[str, Literal["input", "reference"], str]] = []
    for case in ev.cases:
        assert case.id is not None
        fields.append((case.id, "input", case.input))
        if case.reference is not None:
            fields.append((case.id, "reference", case.reference))

    vectors = await backend.embed([text for _, _, text in fields] + texts)
    field_vecs = vectors[: len(fields)]
    corpus_vecs = vectors[len(fields) :]

    findings: list[ContaminationFinding] = []
    for (case_id, field_name, _), vec in zip(fields, field_vecs, strict=True):
        best, best_idx = -1.0, -1
        for i, cv in enumerate(corpus_vecs):
            sim = cosine_similarity(vec, cv)
            if sim > best:
                best, best_idx = sim, i
        if best >= threshold:
            findings.append(
                ContaminationFinding(
                    case_id=case_id,
                    field=field_name,
                    kind="embedding-similarity",
                    score=best,
                    detail=f"cosine {best:.3f} to corpus text #{best_idx} ({backend.name})",
                )
            )
    findings.sort(key=lambda f: -f.score)
    return ContaminationReport(
        findings=tuple(findings),
        n_cases=len(ev.cases),
        method=f"embedding cosine >= {threshold:g} ({backend.name})",
    )
