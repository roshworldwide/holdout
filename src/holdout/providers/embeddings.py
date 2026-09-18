from collections.abc import Sequence

import httpx

from holdout.exceptions import MissingDependencyError


class OllamaEmbeddings:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, transport=transport
        )

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        resp = await self._client.post(
            "/api/embed", json={"model": self.model, "input": list(texts)}
        )
        resp.raise_for_status()
        embeddings = resp.json()["embeddings"]
        return [[float(x) for x in vec] for vec in embeddings]

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenAIEmbeddings:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise MissingDependencyError("openai", "openai") from exc
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self.model, input=list(texts))
        return [[float(x) for x in item.embedding] for item in resp.data]
