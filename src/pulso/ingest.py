"""Collector incremental del stream de observaciones.

Garantías:
- Idempotente: repetir una página o una ejecución completa no duplica datos (upsert por
  `(station_id, observed_at)`).
- El cursor solo avanza cuando el lote se confirmó en la base (misma transacción).
- Deja evidencia en `ingestion_runs` en cada ejecución, incluso sin novedades o con error.
- Un lote inválido se rechaza completo: el cursor no se mueve y la ejecución falla.

Cursor: la API entrega `next_cursor = null` en la última página. Se guarda entonces el cursor
con el que se pidió esa página (la «cola»); la siguiente ejecución la vuelve a leer y recibe
también lo nuevo. Repite como máximo una página, y el upsert lo hace inofensivo.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass

import httpx

from .api import PulsoApi, PulsoApiError
from .config import Settings
from .quality import validate_observations
from .store import Store, SupabaseStore
from .supa import Supabase

log = logging.getLogger("pulso.ingest")

STREAM_RESOURCE = "stream_observations"


class CollectorError(RuntimeError):
    pass


@dataclass
class CollectorResult:
    pages: int = 0
    received: int = 0
    inserted: int = 0
    updated: int = 0
    cursor_before: str | None = None
    cursor_after: str | None = None
    clock_state: str | None = None
    replayed_from_start: bool = False
    truncated: bool = False


def _clock_state(api: PulsoApi) -> str | None:
    try:
        return api.clock().get("state")
    except (PulsoApiError, httpx.HTTPError):  # informativo: no debe tumbar la ingesta
        return None


def run_collector(api: PulsoApi, store: Store, *, page_size: int = 1000, max_pages: int = 500,
                  resource: str = STREAM_RESOURCE) -> CollectorResult:
    committed = store.get_cursor(resource)
    result = CollectorResult(cursor_before=committed, cursor_after=committed)
    run_id = store.start_run("stream", committed)
    try:
        result.clock_state = _clock_state(api)
        known = store.known_stations()
        cursor = committed
        seen: set[str] = {cursor} if cursor else set()

        while True:
            if result.pages >= max_pages:
                result.truncated = True  # la próxima ejecución continúa desde el cursor confirmado
                break
            try:
                page = api.stream_page(cursor, page_size)
            except PulsoApiError as exc:
                retryable = (exc.code == "invalid_cursor" and cursor is not None
                             and not result.replayed_from_start)
                if retryable:
                    log.warning("Cursor rechazado por la API; se relee desde el inicio")
                    result.replayed_from_start = True
                    cursor, seen = None, set()
                    continue
                raise

            rows = validate_observations(page.get("data", []), known)
            next_cursor = page.get("next_cursor")
            after = next_cursor if next_cursor is not None else cursor
            result.pages += 1
            result.received += len(rows)

            if rows or after != committed:
                counts = store.ingest(rows, resource, after)
                committed = result.cursor_after = after
                result.inserted += counts.get("inserted", 0)
                result.updated += counts.get("updated", 0)

            if next_cursor is None:
                break
            if next_cursor in seen:
                raise CollectorError("La API devolvió un cursor repetido")
            seen.add(next_cursor)
            cursor = next_cursor

        store.finish_run(run_id, status="success", rows_received=result.received,
                         rows_upserted=result.inserted + result.updated,
                         cursor_after=result.cursor_after, error=None)
        return result
    except Exception as exc:
        # Se deja constancia del error y se relanza para que el workflow falle a la vista.
        try:
            store.finish_run(run_id, status="error", rows_received=result.received,
                             rows_upserted=result.inserted + result.updated,
                             cursor_after=result.cursor_after,
                             error=f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001 - no debe tapar el error original
            log.exception("No se pudo registrar el error de la ejecución")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Descarga observaciones nuevas del stream")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=500)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = Settings.from_env()
    url, key = settings.require_supabase()
    db = Supabase(url, key)
    try:
        with PulsoApi(settings.api_url, settings.api_key) as api:
            result = run_collector(api, SupabaseStore(db), page_size=args.page_size,
                                   max_pages=args.max_pages)
    finally:
        db.close()
    log.info("Collector OK: %s", asdict(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
