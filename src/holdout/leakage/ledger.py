import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

DisciplineLevel = Literal["ok", "caution", "overfit-risk"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS uses (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_fingerprint TEXT NOT NULL,
    eval_name        TEXT NOT NULL,
    kind             TEXT NOT NULL,
    context          TEXT,
    used_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_uses_fp ON uses (eval_fingerprint);
"""


@dataclass(frozen=True, slots=True)
class DisciplineReport:
    eval_name: str
    eval_fingerprint: str
    uses: int
    budget: int
    level: DisciplineLevel

    def __str__(self) -> str:
        msg = (
            f"eval {self.eval_name!r} has been used {self.uses} time(s) "
            f"of a budget of {self.budget} [{self.level}]"
        )
        if self.level == "overfit-risk":
            msg += (
                " — results on this eval now reflect tuning-to-the-test as much as "
                "quality; cut a fresh holdout set (Dwork et al. 2015)"
            )
        elif self.level == "caution":
            msg += " — plan a fresh holdout set before the budget runs out"
        return msg

    def to_dict(self) -> dict[str, object]:
        return {
            "eval_name": self.eval_name,
            "eval_fingerprint": self.eval_fingerprint,
            "uses": self.uses,
            "budget": self.budget,
            "level": self.level,
        }


class HoldoutLedger:
    def __init__(self, root: str | Path = ".holdout") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._db_path = self.root / "ledger.sqlite3"
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def record_use(
        self,
        eval_fingerprint: str,
        eval_name: str,
        *,
        kind: str = "compare",
        context: str | None = None,
    ) -> int:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO uses (eval_fingerprint, eval_name, kind, context, used_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    eval_fingerprint,
                    eval_name,
                    kind,
                    context,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return self.uses(eval_fingerprint)

    def uses(self, eval_fingerprint: str) -> int:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM uses WHERE eval_fingerprint = ?", (eval_fingerprint,)
            ).fetchone()
        return int(row[0])

    def check(self, eval_fingerprint: str, eval_name: str, *, budget: int = 20) -> DisciplineReport:
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        n = self.uses(eval_fingerprint)
        level: DisciplineLevel
        if n >= budget:
            level = "overfit-risk"
        elif n * 2 >= budget:
            level = "caution"
        else:
            level = "ok"
        return DisciplineReport(
            eval_name=eval_name,
            eval_fingerprint=eval_fingerprint,
            uses=n,
            budget=budget,
            level=level,
        )
