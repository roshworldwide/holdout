from collections.abc import Mapping

from holdout.core.hashing import fingerprint
from holdout.core.target import Completion


class StaticTarget:
    def __init__(
        self,
        responses: Mapping[str, str],
        *,
        name: str = "static",
        default: str | None = None,
    ) -> None:
        self._responses = dict(responses)
        self._name = name
        self._default = default

    @property
    def name(self) -> str:
        return self._name

    @property
    def fingerprint(self) -> str:
        return fingerprint(
            {"static": self._name, "responses": self._responses, "default": self._default}
        )

    async def generate(self, prompt: str, *, seed: int | None = None) -> Completion:
        del seed
        if prompt in self._responses:
            return Completion(text=self._responses[prompt], model=self._name)
        if self._default is not None:
            return Completion(text=self._default, model=self._name)
        raise KeyError(f"no static response for input: {prompt!r}")
