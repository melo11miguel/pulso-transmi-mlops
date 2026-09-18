"""Pruebas de la capa de acceso: calidad, HTTP con reintentos, API de Pulso y Supabase."""

import json

import httpx
import pandas as pd
import pytest

from pulso.api import PulsoApiError
from pulso.http import send
from pulso.quality import DataQualityError, quality_report, validate_observations
from pulso.supa import Supabase, SupabaseError

KNOWN = {"02300", "03000"}
GOOD = {"station_id": "02300", "observed_at": "2026-09-16T10:15:00-05:00", "demand": 10}


# ------------------------------------------------------------------ calidad
def test_validate_accepts_good_row_and_keeps_released_at():
    row = {**GOOD, "released_at": "2026-09-16T10:30:03-05:00"}
    out = validate_observations([row], KNOWN)
    assert out == [{**GOOD, "released_at": "2026-09-16T10:30:03-05:00"}]


@pytest.mark.parametrize("patch, message", [
    ({"station_id": "99999"}, "desconocida"),
    ({"station_id": 2300}, "desconocida"),
    ({"observed_at": "2026-09-16T10:15:00"}, "sin zona horaria"),
    ({"observed_at": "2026-09-16T10:10:00-05:00"}, "rejilla"),
    ({"demand": -1}, "fuera de rango"),
    ({"demand": 100_001}, "fuera de rango"),
    ({"demand": float("nan")}, "no finita"),
    ({"demand": float("inf")}, "no finita"),
    ({"demand": "10"}, "no numérica"),
    ({"demand": True}, "no numérica"),
    ({"demand": 10.5}, "no entera"),
])
def test_validate_rejects_bad_rows(patch, message):
    with pytest.raises(DataQualityError, match=message):
        validate_observations([{**GOOD, **patch}], KNOWN)


def test_validate_rejects_duplicates_in_batch():
    with pytest.raises(DataQualityError, match="duplicado"):
        validate_observations([GOOD, dict(GOOD)], KNOWN)


def test_validate_accepts_integer_valued_float():
    assert validate_observations([{**GOOD, "demand": 10.0}], KNOWN)[0]["demand"] == 10


def test_quality_report_detects_gaps_and_duplicates():
    ts = pd.date_range("2026-09-16 00:00", periods=4, freq="15min", tz="America/Bogota")
    obs = pd.DataFrame({"station_id": "02300", "observed_at": ts, "demand": [1, 2, 3, 4]})
    assert quality_report(obs).ok
    broken = pd.concat([obs.drop(index=1), obs.iloc[[3]]])
    report = quality_report(broken)
    assert report.duplicates == 1 and report.missing_periods == 1 and not report.ok


# ------------------------------------------------------------------ http.send
def _client(handler):
    return httpx.Client(base_url="https://x.test", transport=httpx.MockTransport(handler))


def test_send_retries_transport_errors_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("boom")
        return httpx.Response(200)

    sleeps = []
    assert send(_client(handler), "GET", "/", sleep=sleeps.append).status_code == 200
    assert calls["n"] == 3 and sleeps == [1.0, 2.0]


def test_send_raises_when_network_never_recovers():
    def handler(request):
        raise httpx.ConnectError("boom")

    with pytest.raises(httpx.ConnectError):
        send(_client(handler), "GET", "/", retries=2, sleep=lambda s: None)


def test_send_does_not_retry_when_retries_zero():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503)

    assert send(_client(handler), "GET", "/", retries=0).status_code == 503
    assert calls["n"] == 1


def test_send_does_not_retry_client_errors():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(422)

    assert send(_client(handler), "GET", "/").status_code == 422 and calls["n"] == 1


# ------------------------------------------------------------------ API de Pulso
def test_current_cycle_returns_none_when_no_open_cycle(api, server):
    assert api.current_cycle() is None
    server.cycle = {"cycle_id": "cyc_1", "targets": []}
    assert api.current_cycle()["cycle_id"] == "cyc_1"


def test_current_cycle_other_errors_are_raised(api, server):
    server.fail_with = [httpx.Response(404, json={"detail": {"code": "cycle_not_found"}})]
    with pytest.raises(PulsoApiError) as info:
        api.current_cycle()
    assert info.value.code == "cycle_not_found"


def test_submit_sends_idempotency_key_and_repeat_gets_same_receipt(api, server):
    payload = {"predictions": [{"station_id": "02300", "target_at": "t", "value": 1.0}]}
    status1, r1 = api.submit(payload, "gha-1-1-cyc")
    status2, r2 = api.submit(payload, "gha-1-1-cyc")
    assert (status1, status2) == (201, 200)
    assert r1["submission_id"] == r2["submission_id"]
    assert r1["_request_id"] == "req-1"
    sent = [c for c in server.calls if c.method == "POST"]
    assert all(c.headers["idempotency-key"] == "gha-1-1-cyc" for c in sent)
    assert all(c.headers["authorization"] == "Bearer SECRETKEY" for c in sent)


def test_submit_conflict_raises_with_code(api, server):
    server.fail_with = [httpx.Response(409, json={"detail": {"code": "idempotency_conflict",
                                                             "message": "x"}},
                                       headers={"x-request-id": "r9"})]
    with pytest.raises(PulsoApiError) as info:
        api.submit({"predictions": []}, "k")
    assert (info.value.status, info.value.code, info.value.request_id) == (409, "idempotency_conflict", "r9")
    assert "SECRETKEY" not in str(info.value)


# ------------------------------------------------------------------ Supabase
def _supabase(handler, key="eyJ.service.jwt"):
    return Supabase("https://p.supabase.co", key, transport=httpx.MockTransport(handler),
                    sleep=lambda s: None)


def test_supabase_headers_depend_on_key_type():
    seen = []

    def handler(request):
        seen.append(dict(request.headers))
        return httpx.Response(200, json=[])

    _supabase(handler, "eyJ.legacy.jwt").select("stations")
    _supabase(handler, "sb_secret_abc").select("stations")
    assert seen[0]["apikey"] == "eyJ.legacy.jwt" and seen[0]["authorization"].startswith("Bearer ")
    assert seen[1]["apikey"] == "sb_secret_abc" and "authorization" not in seen[1]


def test_supabase_upsert_uses_on_conflict_and_chunks():
    calls = []

    def handler(request):
        calls.append((str(request.url), json.loads(request.content), request.headers["prefer"]))
        return httpx.Response(201)

    _supabase(handler).upsert("observations", [{"a": i} for i in range(5)], "station_id,observed_at",
                              chunk=2)
    assert len(calls) == 3
    assert "on_conflict=station_id%2Cobserved_at" in calls[0][0]
    assert "merge-duplicates" in calls[0][2]


def test_supabase_select_pages_walks_whole_table():
    data = [{"i": i} for i in range(7)]

    def handler(request):
        p = request.url.params
        off, lim = int(p["offset"]), int(p["limit"])
        return httpx.Response(200, json=data[off:off + lim])

    pages = list(_supabase(handler).select_pages("t", order="i", page=3))
    assert [len(p) for p in pages] == [3, 3, 1]


def test_supabase_select_pages_exact_multiple_terminates():
    data = [{"i": i} for i in range(6)]

    def handler(request):
        p = request.url.params
        return httpx.Response(200, json=data[int(p["offset"]):int(p["offset"]) + int(p["limit"])])

    assert sum(len(p) for p in _supabase(handler).select_pages("t", order="i", page=3)) == 6


def test_supabase_insert_is_not_retried_and_errors_hide_key():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="down")

    with pytest.raises(SupabaseError) as info:
        _supabase(handler, "eyJ.secretvalue.jwt").insert("ingestion_runs", {"source": "x"})
    assert calls["n"] == 1 and "secretvalue" not in str(info.value)


def test_supabase_rpc_and_upload():
    seen = []

    def handler(request):
        seen.append((request.url.path, request.headers.get("x-upsert"), request.content))
        return httpx.Response(200, json={"inserted": 1})

    db = _supabase(handler)
    assert db.rpc("ingest_observations", {"p_rows": []})["inserted"] == 1
    db.upload("models", "v1/model.joblib", b"abc")
    assert seen[0][0] == "/rest/v1/rpc/ingest_observations"
    assert seen[1][0] == "/storage/v1/object/models/v1/model.joblib" and seen[1][1] == "false"
