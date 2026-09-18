"""Persistencia del collector: la interfaz `Store`, su versión Supabase y una en memoria."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from .supa import Supabase


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Store(Protocol):
    def known_stations(self) -> set[str]: ...

    def get_cursor(self, resource: str) -> str | None: ...

    def start_run(self, source: str, cursor_before: str | None) -> int: ...

    def finish_run(self, run_id: int, *, status: str, rows_received: int, rows_upserted: int,
                   cursor_after: str | None, error: str | None) -> None: ...

    def ingest(self, rows: list[dict], resource: str, cursor_after: str | None) -> dict[str, int]:
        """Upsert del lote y avance del cursor de forma atómica."""
        ...


class SupabaseStore:
    def __init__(self, db: Supabase) -> None:
        self.db = db

    def known_stations(self) -> set[str]:
        rows = self.db.select("stations", columns="station_id")
        return {r["station_id"] for r in rows}

    def get_cursor(self, resource: str) -> str | None:
        rows = self.db.select("collector_state", columns="cursor",
                              filters={"resource": f"eq.{resource}"})
        return rows[0]["cursor"] if rows else None

    def start_run(self, source: str, cursor_before: str | None) -> int:
        row = self.db.insert("ingestion_runs", {"source": source, "cursor_before": cursor_before})
        return row["id"]

    def finish_run(self, run_id: int, *, status: str, rows_received: int, rows_upserted: int,
                   cursor_after: str | None, error: str | None) -> None:
        self.db.update(
            "ingestion_runs",
            {"status": status, "finished_at": _now(), "rows_received": rows_received,
             "rows_upserted": rows_upserted, "cursor_after": cursor_after,
             "error": error[:2000] if error else None},
            {"id": f"eq.{run_id}"},
        )

    def ingest(self, rows: list[dict], resource: str, cursor_after: str | None) -> dict[str, int]:
        return self.db.rpc(
            "ingest_observations",
            {"p_rows": rows, "p_cursor_resource": resource, "p_cursor_after": cursor_after},
        )


class MemoryStore:
    """Implementación en memoria con la misma semántica (para pruebas y ensayos locales)."""

    def __init__(self, stations: set[str], cursor: str | None = None) -> None:
        self.stations = set(stations)
        self.cursors: dict[str, str | None] = {}
        if cursor is not None:
            self.cursors["stream_observations"] = cursor
        self.observations: dict[tuple[str, str], dict[str, Any]] = {}
        self.runs: dict[int, dict[str, Any]] = {}
        self.fail_on_call: int | None = None  # nº (1-based) de la llamada a ingest que falla
        self.ingest_calls = 0

    def known_stations(self) -> set[str]:
        return set(self.stations)

    def get_cursor(self, resource: str) -> str | None:
        return self.cursors.get(resource)

    def start_run(self, source: str, cursor_before: str | None) -> int:
        run_id = len(self.runs) + 1
        self.runs[run_id] = {"source": source, "cursor_before": cursor_before, "status": "running"}
        return run_id

    def finish_run(self, run_id: int, *, status: str, rows_received: int, rows_upserted: int,
                   cursor_after: str | None, error: str | None) -> None:
        self.runs[run_id].update(status=status, rows_received=rows_received,
                                 rows_upserted=rows_upserted, cursor_after=cursor_after,
                                 error=error)

    def ingest(self, rows: list[dict], resource: str, cursor_after: str | None) -> dict[str, int]:
        self.ingest_calls += 1
        if self.fail_on_call == self.ingest_calls:  # transacción que falla: no cambia nada
            raise RuntimeError("fallo simulado de la base")
        inserted = updated = 0
        staged = dict(self.observations)
        for row in rows:
            key = (row["station_id"], row["observed_at"])
            if key in staged:
                updated += 1
            else:
                inserted += 1
            staged[key] = {**staged.get(key, {}), **row}
        self.observations = staged
        self.cursors[resource] = cursor_after
        return {"inserted": inserted, "updated": updated}
