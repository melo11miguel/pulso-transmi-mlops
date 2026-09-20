"""Utilidades de prueba: un servidor simulado del stream y de la API con cursores opacos."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from pulso.api import PulsoApi
from pulso.store import MemoryStore

TZ = timezone(timedelta(hours=-5))
STATIONS = ["02300", "03000", "05000"]


def make_rows(n: int, start: datetime | None = None) -> list[dict]:
    """n filas del stream, en orden de liberación (3 estaciones por instante)."""
    start = start or datetime(2026, 9, 16, 10, 0, tzinfo=TZ)
    rows = []
    for i in range(n):
        ts = start + timedelta(minutes=15 * (i // len(STATIONS)))
        rows.append({
            "station_id": STATIONS[i % len(STATIONS)],
            "observed_at": ts.isoformat(),
            "demand": 100 + i,
            "released_at": (ts + timedelta(seconds=3)).isoformat(),
        })
    return rows


class FakeServer:
    """Imita /v1/stream/observations, /v1/clock y el ciclo. Cursor = 'c<índice>'."""

    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = rows or []
        self.calls: list[httpx.Request] = []
        self.clock_state = "waiting"
        self.cycle: dict | None = None
        self.fail_with: list[httpx.Response] = []  # respuestas a servir antes de las normales
        self.reject_cursors = False
        self.loop_cursor = False
        self.submissions: dict[str, tuple[int, dict]] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.fail_with:
            return self.fail_with.pop(0)
        path = request.url.path
        if path == "/v1/stream/observations":
            return self._stream(request)
        if path == "/v1/clock":
            return httpx.Response(200, json={"state": self.clock_state})
        if path == "/v1/forecast-cycles/current":
            if self.cycle is None:
                return httpx.Response(404, json={"detail": {"code": "no_open_cycle",
                                                            "message": "sin ciclo"}})
            return httpx.Response(200, json=self.cycle)
        if path == "/v1/submissions" and request.method == "POST":
            key = request.headers["idempotency-key"]
            body = json.loads(request.content)
            if key in self.submissions:
                status, receipt = self.submissions[key]
                return httpx.Response(200, json=receipt)
            receipt = {"submission_id": f"sub_{len(self.submissions) + 1}", "status": "accepted",
                       "attempt": len(self.submissions) + 1, "is_official": True,
                       "predictions_received": len(body["predictions"]),
                       "expected_predictions": len(body["predictions"])}
            self.submissions[key] = (201, receipt)
            return httpx.Response(201, json=receipt, headers={"x-request-id": "req-1"})
        return httpx.Response(404, json={"detail": "Not Found"})

    def _stream(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        limit = int(params.get("limit", 1000))
        cursor = params.get("cursor")
        if self.reject_cursors and cursor:
            return httpx.Response(400, json={"detail": {"code": "invalid_cursor",
                                                        "message": "cursor inválido"}})
        start = int(cursor[1:]) if cursor else 0
        chunk = self.rows[start:start + limit]
        nxt = f"c{start + limit}" if start + limit < len(self.rows) else None
        if self.loop_cursor:
            nxt = cursor or "c0"
        return httpx.Response(200, json={"data": chunk, "count": len(chunk), "next_cursor": nxt})


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def api(server: FakeServer, sleeps: list[float]) -> PulsoApi:
    return PulsoApi("https://api.test", "SECRETKEY", transport=httpx.MockTransport(server.handler),
                    sleep=sleeps.append)


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(set(STATIONS))


# ----------------------------------------------------------------------------------------------
# Base de datos falsa en memoria con la interfaz de `pulso.supa.Supabase`
# ----------------------------------------------------------------------------------------------
import pandas as pd  # noqa: E402

from pulso.supa import SupabaseError  # noqa: E402

IDENTITY_TABLES = {"pipeline_runs", "submissions", "ingestion_runs", "retrain_decisions",
                   "metric_snapshots", "drift_signals"}
UNIQUE = {"submissions": ["idempotency_key"], "model_versions": ["version"]}
TIME_COLUMNS = {"observed_at", "target_at", "data_cutoff", "origin_at", "closes_at", "opens_at"}


def _cmp_value(column: str, value):
    if column in TIME_COLUMNS and value is not None:
        return pd.Timestamp(value).tz_convert("UTC")
    return value


class FakeDb:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict]] = {}
        self.storage: dict[tuple[str, str], bytes] = {}
        self._ids: dict[str, int] = {}
        self.writes = 0

    def close(self) -> None:
        pass

    def _rows(self, table: str) -> list[dict]:
        return self.tables.setdefault(table, [])

    @staticmethod
    def _match(row: dict, filters: dict[str, str] | None) -> bool:
        for column, condition in (filters or {}).items():
            if column in ("select", "order", "limit", "offset"):
                continue
            op, _, raw = condition.partition(".")
            left, right = _cmp_value(column, row.get(column)), _cmp_value(column, raw)
            if op == "eq":
                if isinstance(left, bool):
                    left = str(left).lower()
                if str(left) != str(right) and left != right:
                    return False
            elif op == "lte":
                if not left <= right:
                    return False
            elif op == "gte":
                if not left >= right:
                    return False
            else:
                raise NotImplementedError(op)
        return True

    def _prediction_results(self) -> list[dict]:
        """Réplica de la vista public.prediction_results (predicciones + valor real)."""
        subs = {s["id"]: s for s in self._rows("submissions")}
        actual = {(o["station_id"], pd.Timestamp(o["observed_at"]).tz_convert("UTC")): o["demand"]
                  for o in self._rows("observations")}
        out = []
        for p in self._rows("predictions"):
            sub = subs[p["submission_row_id"]]
            key = (p["station_id"], pd.Timestamp(p["target_at"]).tz_convert("UTC"))
            out.append({"submission_row_id": sub["id"], "cycle_id": sub["cycle_id"],
                        "model_version": sub["model_version"], "is_official": sub.get("is_official"),
                        "status": sub["status"], "station_id": p["station_id"],
                        "target_at": p["target_at"], "horizon_minutes": p["horizon_minutes"],
                        "predicted": p["value"], "actual": actual.get(key)})
        return out

    def select(self, table, *, columns="*", filters=None, order=None, limit=None, offset=None):
        source = self._prediction_results() if table == "prediction_results" else self._rows(table)
        rows = [dict(r) for r in source if self._match(r, filters)]
        if order:
            for spec in reversed(order.split(",")):  # orden estable por la clave menos significativa
                key, _, direction = spec.partition(".")
                rows.sort(key=lambda r, key=key: _cmp_value(key, r.get(key)),
                          reverse=direction == "desc")
        start = offset or 0
        rows = rows[start:start + limit] if limit is not None else rows[start:]
        if columns != "*":
            wanted = columns.split(",")
            rows = [{k: r.get(k) for k in wanted} for r in rows]
        return rows

    def select_pages(self, table, *, columns="*", filters=None, order, page=1000):
        offset = 0
        while True:
            rows = self.select(table, columns=columns, filters=filters, order=order,
                               limit=page, offset=offset)
            if rows:
                yield rows
            if len(rows) < page:
                return
            offset += page

    def _check_unique(self, table: str, row: dict) -> None:
        for column in UNIQUE.get(table, []):
            if any(r.get(column) == row.get(column) for r in self._rows(table)):
                raise SupabaseError(409, f"insert {table}", f"duplicate key {column}")

    def insert(self, table, row):
        self.writes += 1
        row = dict(row)
        self._check_unique(table, row)
        if table in IDENTITY_TABLES or table == "submissions":
            self._ids[table] = self._ids.get(table, 0) + 1
            row["id"] = self._ids[table]
        row.setdefault("status", "candidate" if table == "model_versions" else row.get("status"))
        stamp = (pd.Timestamp("2026-09-20", tz="UTC") + pd.Timedelta(seconds=self.writes)).isoformat()
        for column in {"drift_signals": ["computed_at"], "metric_snapshots": ["computed_at"],
                       "retrain_decisions": ["decided_at"], "pipeline_runs": ["started_at"],
                       "ingestion_runs": ["started_at"]}.get(table, []):
            row.setdefault(column, stamp)
        self._rows(table).append(row)
        return dict(row)

    def upsert(self, table, rows, on_conflict, chunk=2000):
        self.writes += 1
        keys = on_conflict.split(",")
        for new in rows:
            existing = next((r for r in self._rows(table)
                             if all(str(r.get(k)) == str(new.get(k)) for k in keys)), None)
            if existing:
                existing.update(new)
            else:
                self._rows(table).append(dict(new))

    def update(self, table, values, filters):
        self.writes += 1
        for row in self._rows(table):
            if self._match(row, filters):
                row.update(values)

    def rpc(self, function, args, *, retries=4):
        if function != "promote_model":
            raise NotImplementedError(function)
        rows = self._rows("model_versions")
        target = next((r for r in rows if r["version"] == args["p_version"]), None)
        if target is None:
            raise SupabaseError(400, "rpc promote_model", "model version does not exist")
        previous = next((r["version"] for r in rows if r["status"] == "champion"), None)
        for r in rows:
            if r["status"] == "champion":
                r["status"] = "archived"
        target["status"] = "champion"
        target["reason"] = args.get("p_reason") or target.get("reason")
        return {"promoted": args["p_version"], "previous": previous}

    def upload(self, bucket, path, data, content_type="application/octet-stream"):
        if (bucket, path) in self.storage:
            raise SupabaseError(409, f"upload {bucket}/{path}", "Duplicate")
        self.storage[(bucket, path)] = data

    def download(self, bucket, path):
        return self.storage[(bucket, path)]

    def load_observations(self, wide: pd.DataFrame) -> None:
        stack = wide.stack().reset_index()
        stack.columns = ["observed_at", "station_id", "demand"]
        self.tables["stations"] = [{"station_id": s} for s in wide.columns]
        self.tables["observations"] = [
            {"observed_at": r.observed_at.tz_convert("UTC").isoformat(), "station_id": r.station_id,
             "demand": int(r.demand)} for r in stack.itertuples()
        ]


@pytest.fixture
def fake_db() -> FakeDb:
    return FakeDb()
