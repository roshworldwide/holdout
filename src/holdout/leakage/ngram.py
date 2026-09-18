import re

_TOKEN = re.compile(r"\w+")


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    toks = tokens(text)
    if not toks:
        return set()
    if len(toks) < n:
        return {tuple(toks)}
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def containment(needle: set[tuple[str, ...]], corpus: set[tuple[str, ...]]) -> float:
    if not needle:
        return 0.0
    return len(needle & corpus) / len(needle)


def jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
