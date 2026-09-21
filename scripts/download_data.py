"""Descarga el corte inicial oficial a data/ y verifica su SHA-256 contra /v1/meta.

Uso:  python scripts/download_data.py
Los CSV no se versionan: la fuente de verdad es la API (y Supabase para el pipeline).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from pulso.config import Settings  # noqa: E402

FILES = ("stations.csv", "observations.csv", "context.csv")


def main() -> int:
    base = Settings.from_env().api_url
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    with httpx.Client(base_url=base, timeout=60, follow_redirects=True) as client:
        meta = client.get("/v1/meta").raise_for_status().json()
        for name in FILES:
            content = client.get(f"/v1/downloads/{name}").raise_for_status().content
            expected = meta["dataset"]["files"][name]["sha256"]
            if hashlib.sha256(content).hexdigest() != expected:
                print(f"{name}: el hash NO coincide con /v1/meta; no se guarda")
                return 1
            (out / name).write_bytes(content)
            print(f"{name}: {len(content):,} bytes, hash verificado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
