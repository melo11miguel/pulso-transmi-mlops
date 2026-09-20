"""Bitácora de ejecuciones de los workflows en `pipeline_runs`."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from .supa import Supabase

log = logging.getLogger("pulso.runlog")


class RunContext:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.skipped = False

    def skip(self, reason: str) -> None:
        """Marca la ejecución como omitida (p. ej. no hay ciclo abierto): no es un error."""
        self.skipped = True
        self.summary["skipped_reason"] = reason


def _now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def pipeline_run(db: Supabase, job: str) -> Iterator[RunContext]:
    """Registra inicio y fin de un job. Relanza cualquier excepción para que el workflow falle."""
    row = db.insert("pipeline_runs", {
        "job": job,
        "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "git_commit": os.getenv("GITHUB_SHA"),
    })
    ctx = RunContext()
    try:
        yield ctx
    except Exception as exc:
        _finish(db, row["id"], "error", ctx.summary, f"{type(exc).__name__}: {exc}")
        raise
    else:
        _finish(db, row["id"], "skipped" if ctx.skipped else "success", ctx.summary, None)


def _finish(db: Supabase, run_id: int, status: str, summary: dict, error: str | None) -> None:
    try:
        db.update("pipeline_runs",
                  {"status": status, "finished_at": _now(), "summary": summary,
                   "error": error[:2000] if error else None},
                  {"id": f"eq.{run_id}"})
    except Exception:  # noqa: BLE001 - no debe tapar el resultado real del job
        log.exception("No se pudo cerrar la ejecución %s", run_id)
