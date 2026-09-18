from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from holdout.store.run_store import RunStore


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("holdout", "holdout LLM evaluation")
    group.addoption(
        "--holdout-store",
        default=".holdout",
        help="run store directory (default: .holdout)",
    )
    group.addoption(
        "--holdout-seed",
        type=int,
        default=0,
        help="seed for eval runs and resampling (default: 0)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "llm_eval: marks a test as an LLM evaluation (may call real model targets)",
    )


@pytest.fixture(scope="session")
def holdout_store(request: pytest.FixtureRequest) -> "RunStore":
    from holdout.store.run_store import RunStore

    return RunStore(str(request.config.getoption("--holdout-store")))


@pytest.fixture(scope="session")
def holdout_seed(request: pytest.FixtureRequest) -> int:
    return int(str(request.config.getoption("--holdout-seed")))
