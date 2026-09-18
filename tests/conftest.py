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
