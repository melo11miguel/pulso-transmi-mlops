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
RONDA_RESOURCE = "ronda_vista"   # guarda el codigo de la ronda en la misma tabla del cursor


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
    # Filas del esquema 2 sin valor usable (`quality` distinto de "observed"). Se omiten en vez de
    # guardarse como cero, y se cuentan aqui para que no desaparezcan sin dejar rastro.
    omitidas: int = 0
    ronda: str | None = None


def _reloj(api: PulsoApi) -> dict:
    """El reloj, una sola vez por ejecucion. Un fallo no debe tumbar la ingesta.

    Se lee una vez y de ahi salen el estado y el codigo de ronda. Pedirlo dos veces no solo
    gastaba una peticion: cambiaba cuantas respuestas consume el colector, que es justo lo que
    detecto la prueba de que un 401 no filtra la API key.
    """
    try:
        return api.clock()
    except (PulsoApiError, httpx.HTTPError):
        return {}


def revisar_ronda(reloj: dict, store: Store) -> str | None:
    """Comprueba que la ronda sea la misma de siempre, y para en seco si cambio.

    `observations` tiene clave primaria `(station_id, observed_at)` y nada que identifique la
    ronda. Si el reto abre una ronda nueva que reusa fechas virtuales, cosa probable porque el
    historico de arranque siempre termina el 2026-09-08, el upsert escribiria los datos nuevos
    ENCIMA de los viejos sin error ninguno, y el modelo entrenaria sobre una serie pegada de dos
    procesos generadores distintos. Un numero plausible y mal, que es la peor clase de fallo.

    Esto no lo arregla, lo hace visible: para antes de escribir y dice que decidir. Arreglarlo de
    verdad pide una columna de ronda en la clave primaria, y no vale cambiar la clave de una tabla
    viva por una ronda que todavia no existe; cuando exista, este error es el que la pide.

    Un reloj ilegible o entre rondas no dispara nada: solo un codigo distinto del guardado.
    """
    actual = reloj.get("code")
    if actual is None:
        return None
    visto = store.get_cursor(RONDA_RESOURCE)
    if visto is None:
        store.ingest([], RONDA_RESOURCE, actual)   # primera vez: se registra sin avisar
        log.info("Ronda registrada: %s", actual)
        return actual
    if visto != actual:
        raise CollectorError(
            f"La ronda cambio de {visto!r} a {actual!r} y `observations` no distingue rondas: "
            f"ingerir ahora sobrescribiria el historico de la ronda anterior donde las fechas "
            f"virtuales coincidan. Hay que decidir primero (columna de ronda en la clave "
            f"primaria, o archivar la serie vieja) y recien despues reanudar la ingesta."
        )
    return actual


def run_collector(api: PulsoApi, store: Store, *, page_size: int = 1000, max_pages: int = 500,
                  resource: str = STREAM_RESOURCE) -> CollectorResult:
    committed = store.get_cursor(resource)
    result = CollectorResult(cursor_before=committed, cursor_after=committed)
    run_id = store.start_run("stream", committed)
    try:
        reloj = _reloj(api)
        result.clock_state = reloj.get("state")
        result.ronda = revisar_ronda(reloj, store)
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

            omitidas: list[dict] = []
            rows = validate_observations(page.get("data", []), known, omitidas=omitidas)
            if omitidas:
                result.omitidas += len(omitidas)
                log.warning("%d observaciones sin valor usable, omitidas (ejemplo: %s)",
                            len(omitidas), omitidas[0])
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
