from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@runtime_checkable
class Target(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def fingerprint(self) -> str:
        ...

    async def generate(self, prompt: str, *, seed: int | None = None) -> Completion:
        ...
