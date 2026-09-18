"""Petición HTTP con reintentos y backoff. Los errores nunca incluyen cabeceras."""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}


def send(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    retries: int = 4,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
) -> httpx.Response:
    """Envía la petición y reintenta fallas transitorias (red, 429, 5xx).

    Se debe usar `retries=0` con operaciones que no sean idempotentes. Devuelve la
    última respuesta (aunque sea de error) para que el llamador decida; solo lanza
    si todos los intentos fallan por red.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.request(method, url, **kwargs)
        except httpx.TransportError as exc:  # timeouts, DNS, conexión rechazada
            last_exc = exc
        else:
            if response.status_code not in RETRY_STATUS or attempt == retries:
                return response
            retry_after = response.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else None
            if delay is not None:
                sleep(min(delay, 60.0))
                continue
        if attempt < retries:
            sleep(min(backoff * 2**attempt, 30.0))
    assert last_exc is not None
    raise last_exc
