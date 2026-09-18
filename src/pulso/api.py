"""Cliente de la API Pulso TransMi (contrato 0.4.1).

Reglas del contrato que este cliente respeta:
- El ciclo vigente se descubre en la API; nunca se infiere de la hora local.
- La API key va solo en la cabecera Authorization y nunca se registra.
- Los envíos llevan `Idempotency-Key`; un reintento reutiliza la misma llave.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

from .http import send


class PulsoApiError(RuntimeError):
    """Respuesta de error de la API, con lo necesario para diagnosticar sin filtrar secretos."""

    def __init__(self, status: int, code: str | None, message: str, request_id: str | None,
                 path: str):
        super().__init__(f"{path} -> HTTP {status} {code or ''} {message} [req {request_id}]")
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id
        self.path = path


def _error_from(response: httpx.Response, path: str) -> PulsoApiError:
    code = message = None
    try:
        detail = response.json().get("detail")
        if isinstance(detail, dict):
            code, message = detail.get("code"), detail.get("message")
        elif detail is not None:
            message = str(detail)[:300]
    except ValueError:
        message = response.text[:200]
    return PulsoApiError(
        response.status_code, code, message or "", response.headers.get("x-request-id"), path
    )


class PulsoApi:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        headers = {"User-Agent": "pulso-transmi-mlops/0.1"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout,
            transport=transport, follow_redirects=True,
        )
        self._sleep = sleep

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PulsoApi:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, *, retries: int = 4, **kwargs) -> httpx.Response:
        return send(self._client, method, path, retries=retries, sleep=self._sleep, **kwargs)

    def _json(self, path: str, **params: Any) -> dict[str, Any]:
        params = {k: v for k, v in params.items() if v is not None}
        response = self._request("GET", path, params=params)
        if response.status_code != 200:
            raise _error_from(response, path)
        return response.json()

    # ---- datos --------------------------------------------------------------------------
    def clock(self) -> dict[str, Any]:
        return self._json("/v1/clock")

    def stream_page(self, cursor: str | None = None, limit: int = 1000) -> dict[str, Any]:
        return self._json("/v1/stream/observations", cursor=cursor, limit=limit)

    # ---- ciclo --------------------------------------------------------------------------
    def current_cycle(self) -> dict[str, Any] | None:
        """Ciclo abierto o None si no hay ventana activa (404 no_open_cycle)."""
        path = "/v1/forecast-cycles/current"
        response = self._request("GET", path)
        if response.status_code == 200:
            return response.json()
        error = _error_from(response, path)
        if error.status == 404 and error.code == "no_open_cycle":
            return None
        raise error

    # ---- identidad y entregas -----------------------------------------------------------
    def me(self) -> dict[str, Any]:
        return self._json("/v1/me")

    def submit(self, payload: dict[str, Any], idempotency_key: str) -> tuple[int, dict[str, Any]]:
        """POST /v1/submissions. Devuelve (status, recibo). 201 nuevo, 200 repetido idempotente.

        Reintenta solo fallas transitorias y siempre con la MISMA llave, así que es seguro.
        Lanza PulsoApiError en cualquier respuesta que no sea 200/201.
        """
        path = "/v1/submissions"
        response = self._request(
            "POST", path, json=payload, headers={"Idempotency-Key": idempotency_key}
        )
        if response.status_code not in (200, 201):
            raise _error_from(response, path)
        receipt = response.json()
        receipt.setdefault("_request_id", response.headers.get("x-request-id"))
        return response.status_code, receipt

    def receipt(self, submission_id: str) -> dict[str, Any]:
        return self._json(f"/v1/submissions/{submission_id}")

    def leaderboard(self, window: str = "cumulative") -> dict[str, Any]:
        return self._json("/v1/leaderboard", window=window)
