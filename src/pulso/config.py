"""Configuración desde variables de entorno (y un .env local opcional).

En GitHub Actions los valores llegan como Secrets. Nunca se imprimen ni se guardan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_API_URL = "https://pulso-transmi.72-60-245-2.sslip.io"


def load_dotenv(path: Path | None = None) -> None:
    """Carga KEY=VALUE de un .env sin sobrescribir variables ya definidas."""
    path = path or Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class MissingConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    api_url: str = DEFAULT_API_URL
    api_key: str | None = field(default=None, repr=False)
    supabase_url: str | None = None
    supabase_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        return cls(
            api_url=(os.getenv("PULSO_API_URL") or DEFAULT_API_URL).rstrip("/"),
            api_key=os.getenv("PULSO_API_KEY") or None,
            supabase_url=(os.getenv("SUPABASE_URL") or "").rstrip("/") or None,
            supabase_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY") or None,
        )

    def require_supabase(self) -> tuple[str, str]:
        if not self.supabase_url or not self.supabase_key:
            raise MissingConfigError("Faltan SUPABASE_URL y/o SUPABASE_SERVICE_ROLE_KEY")
        return self.supabase_url, self.supabase_key

    def require_api_key(self) -> str:
        if not self.api_key:
            raise MissingConfigError("Falta PULSO_API_KEY")
        return self.api_key
