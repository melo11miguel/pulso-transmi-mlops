import httpx
import pytest
from conftest import make_rows

from pulso.ingest import CollectorError, run_collector
from pulso.quality import DataQualityError


def test_empty_stream_leaves_evidence_and_no_cursor(api, store):
    result = run_collector(api, store)
    assert result.received == 0 and result.pages == 1
    assert result.clock_state == "waiting"
    assert store.get_cursor("stream_observations") is None
    assert store.runs[1]["status"] == "success"
    assert store.runs[1]["rows_received"] == 0


def test_multi_page_ingest_keeps_tail_cursor(api, server, store):
    server.rows = make_rows(25)
    result = run_collector(api, store, page_size=10)
    assert result.pages == 3 and result.inserted == 25 and result.updated == 0
    assert len(store.observations) == 25
    # la última página se pidió con c20; ese es el cursor de cola
    assert store.get_cursor("stream_observations") == "c20"
    assert store.runs[1]["cursor_after"] == "c20"


def test_rerun_without_news_does_not_duplicate(api, server, store):
    server.rows = make_rows(25)
    run_collector(api, store, page_size=10)
    again = run_collector(api, store, page_size=10)
    assert len(store.observations) == 25
    assert again.inserted == 0 and again.updated == 5  # releyó solo la página de cola
    assert store.get_cursor("stream_observations") == "c20"


def test_rerun_picks_up_only_new_rows(api, server, store):
    server.rows = make_rows(25)
    run_collector(api, store, page_size=10)
    server.rows = make_rows(40)
    result = run_collector(api, store, page_size=10)
    assert result.inserted == 15
    assert len(store.observations) == 40
    assert store.get_cursor("stream_observations") == "c30"


def test_released_at_is_kept(api, server, store):
    server.rows = make_rows(3)
    run_collector(api, store)
    assert all("released_at" in row for row in store.observations.values())


def test_invalid_batch_is_rejected_and_cursor_does_not_move(api, server, store):
    rows = make_rows(20)
    rows[15]["station_id"] = "99999"  # cae en la segunda página
    server.rows = rows
    with pytest.raises(DataQualityError):
        run_collector(api, store, page_size=10)
    assert len(store.observations) == 10  # la página buena se confirmó, la mala no
    assert store.get_cursor("stream_observations") == "c10"
    assert store.runs[1]["status"] == "error"
    assert "DataQualityError" in store.runs[1]["error"]


def test_database_failure_keeps_last_confirmed_cursor_and_propagates(api, server, store):
    server.rows = make_rows(25)
    store.fail_on_call = 2  # falla al confirmar la segunda página
    with pytest.raises(RuntimeError, match="fallo simulado"):
        run_collector(api, store, page_size=10)
    assert store.get_cursor("stream_observations") == "c10"  # no avanzó a c20
    assert len(store.observations) == 10
    assert store.runs[1]["status"] == "error"
    assert store.runs[1]["cursor_after"] == "c10"


def test_rejected_cursor_replays_from_start(api, server, store):
    server.rows = make_rows(12)
    store.cursors["stream_observations"] = "c999"
    server.reject_cursors = True
    result = run_collector(api, store, page_size=100)
    assert result.replayed_from_start and result.inserted == 12
    assert store.get_cursor("stream_observations") is None


def test_repeated_cursor_is_an_error(api, server, store):
    server.rows = make_rows(5)
    server.loop_cursor = True
    with pytest.raises(CollectorError, match="repetido"):
        run_collector(api, store, page_size=2)
    assert store.runs[1]["status"] == "error"


def test_max_pages_truncates_and_next_run_continues(api, server, store):
    server.rows = make_rows(30)
    first = run_collector(api, store, page_size=10, max_pages=2)
    assert first.truncated and len(store.observations) == 20
    assert store.get_cursor("stream_observations") == "c20"
    second = run_collector(api, store, page_size=10, max_pages=2)
    assert not second.truncated and len(store.observations) == 30


def test_transient_errors_are_retried_with_backoff(api, server, store, sleeps):
    server.rows = make_rows(3)
    server.fail_with = [httpx.Response(503), httpx.Response(429, headers={"retry-after": "7"})]
    # el primer GET es /v1/clock (informativo); los 503/429 se consumen ahí y en el stream
    result = run_collector(api, store)
    assert result.inserted == 3
    assert sleeps and 7.0 in sleeps


def test_clock_failure_does_not_break_ingest(api, server, store):
    server.rows = make_rows(3)
    server.fail_with = [httpx.Response(500)] * 5  # agota los reintentos del reloj
    result = run_collector(api, store)
    assert result.clock_state is None and result.inserted == 3


def test_api_error_never_leaks_key(api, server, store):
    server.fail_with = [httpx.Response(401, json={"detail": {"code": "invalid_api_key",
                                                             "message": "no"}})] * 2
    with pytest.raises(Exception) as info:
        run_collector(api, store)
    assert "SECRETKEY" not in str(info.value)
    assert "SECRETKEY" not in str(store.runs[1]["error"])
