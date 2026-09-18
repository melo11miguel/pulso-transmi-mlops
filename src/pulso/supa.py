"""Cliente mínimo de Supabase por REST (PostgREST + Storage) con la service role.

Se evita supabase-py para mantener pocas dependencias y controlar reintentos y errores.
Las claves nunca aparecen en mensajes de error.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from .http import send

PAGE = 1000  # tope por defecto de filas por respuesta en PostgREST


class SupabaseError(RuntimeError):
    def __init__(self, status: int, what: str, detail: str):
        super().__init__(f"Supabase {what} -> HTTP {status}: {detail[:300]}")
        self.status = status
        self.detail = detail


class Supabase:
    def __init__(
        self,
        url: str,
        key: str,
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        headers = {"apikey": key}
        # Las claves nuevas (sb_secret_...) no son JWT: solo van en `apikey`.
        if key.startswith("eyJ"):
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.Client(
            base_url=url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )
        self._sleep = sleep

    def close(self) -> None:
        self._client.close()

    def _do(self, method: str, path: str, what: str, *, retries: int = 4, **kwargs):
        response = send(self._client, method, path, retries=retries, sleep=self._sleep, **kwargs)
        if response.status_code >= 400:
            raise SupabaseError(response.status_code, what, response.text)
        return response

    # ---- tablas -------------------------------------------------------------------------
    def select(self, table: str, *, columns: str = "*", filters: dict[str, str] | None = None,
               order: str | None = None, limit: int | None = None,
               offset: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"select": columns, **(filters or {})}
        if order:
            params["order"] = order
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._do("GET", f"/rest/v1/{table}", f"select {table}", params=params).json()

    def select_pages(self, table: str, *, columns: str = "*", filters: dict[str, str] | None = None,
                     order: str, page: int = PAGE) -> Iterator[list[dict[str, Any]]]:
        """Recorre toda una tabla por páginas. `order` debe ser determinista (p. ej. la PK)."""
        offset = 0
        while True:
            rows = self.select(table, columns=columns, filters=filters, order=order,
                               limit=page, offset=offset)
            if rows:
                yield rows
            if len(rows) < page:
                return
            offset += page

    def insert(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        """Inserta una fila y la devuelve. No se reintenta: no es idempotente."""
        response = self._do(
            "POST", f"/rest/v1/{table}", f"insert {table}", retries=0, json=row,
            headers={"Prefer": "return=representation"},
        )
        return response.json()[0]

    def upsert(self, table: str, rows: list[dict[str, Any]], on_conflict: str,
               chunk: int = 2000) -> None:
        for i in range(0, len(rows), chunk):
            self._do(
                "POST", f"/rest/v1/{table}", f"upsert {table}",
                params={"on_conflict": on_conflict}, json=rows[i:i + chunk],
                headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            )

    def update(self, table: str, values: dict[str, Any], filters: dict[str, str]) -> None:
        self._do("PATCH", f"/rest/v1/{table}", f"update {table}", params=filters, json=values,
                 headers={"Prefer": "return=minimal"})

    def rpc(self, function: str, args: dict[str, Any], *, retries: int = 4) -> Any:
        return self._do("POST", f"/rest/v1/rpc/{function}", f"rpc {function}", retries=retries,
                        json=args).json()

    # ---- storage ------------------------------------------------------------------------
    def upload(self, bucket: str, path: str, data: bytes,
               content_type: str = "application/octet-stream") -> None:
        """Sube un objeto NUEVO. Falla si ya existe: las versiones nunca se sobrescriben."""
        self._do("POST", f"/storage/v1/object/{bucket}/{path}", f"upload {bucket}/{path}",
                 content=data, headers={"Content-Type": content_type, "x-upsert": "false"})

    def download(self, bucket: str, path: str) -> bytes:
        return self._do("GET", f"/storage/v1/object/{bucket}/{path}",
                        f"download {bucket}/{path}").content
