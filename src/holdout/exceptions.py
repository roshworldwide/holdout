class HoldoutError(Exception):
    pass


class MissingDependencyError(HoldoutError):
    def __init__(self, package: str, extra: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"{package!r} is required for this feature but is not installed. "
            f"Install it with: pip install 'holdout[{extra}]'"
        )


class ProviderError(HoldoutError):
    def __init__(self, provider: str, attempts: int, cause: Exception) -> None:
        self.provider = provider
        self.attempts = attempts
        self.cause = cause
        super().__init__(
            f"provider {provider!r} failed after {attempts} attempt(s): "
            f"{type(cause).__name__}: {cause}"
        )
