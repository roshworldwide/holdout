from collections.abc import Mapping
from dataclasses import dataclass, field

from holdout.core.hashing import fingerprint


@dataclass(frozen=True)
class Case:
    input: str
    reference: str | None = None
    id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def content_id(self) -> str:
        payload = {
            "input": self.input,
            "reference": self.reference,
            "metadata": dict(self.metadata),
        }
        return "c" + fingerprint(payload)[:11]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "input": self.input,
            "reference": self.reference,
            "metadata": dict(self.metadata),
        }
