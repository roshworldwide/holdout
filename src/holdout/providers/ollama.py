from collections.abc import Mapping

import httpx

from holdout.core.target import Completion
from holdout.providers.base import ModelProvider


class Ollama(ModelProvider):
    provider_id = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
        )
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, transport=transport
        )

    def _extra_config(self) -> Mapping[str, object]:
        return {"base_url": self.base_url}

    async def _generate_once(self, prompt: str, *, seed: int | None) -> Completion:
        messages: list[dict[str, str]] = []
        if self.system is not None:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, object] = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        if seed is not None:
            options["seed"] = seed

        resp = await self._client.post(
            "/api/chat",
            json={"model": self.model, "messages": messages, "stream": False, "options": options},
        )
        resp.raise_for_status()
        data = resp.json()
        return Completion(
            text=data["message"]["content"],
            model=data.get("model", self.model),
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
