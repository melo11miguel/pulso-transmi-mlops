"""Lectura de datos desde Supabase hacia la rejilla de demanda."""

from __future__ import annotations

import pandas as pd

from .features import to_wide
from .supa import Supabase


def load_station_ids(db: Supabase) -> list[str]:
    return sorted(r["station_id"] for r in db.select("stations", columns="station_id"))


def load_wide(db: Supabase, until: str | None = None, page: int = 1000) -> pd.DataFrame:
    """Toda la demanda (opcionalmente hasta `until`, ISO 8601, inclusive) como rejilla de 15 min.

    Solo se lee lo que existe en la base: nunca se rellena ni se inventa un dato.
    """
    filters = {"observed_at": f"lte.{until}"} if until else None
    rows: list[dict] = []
    for chunk in db.select_pages("observations", columns="observed_at,station_id,demand",
                                 filters=filters, order="observed_at.asc,station_id.asc",
                                 page=page):
        rows.extend(chunk)
    if not rows:
        raise ValueError("No hay observaciones en la base")
    frame = pd.DataFrame(rows)
    return to_wide(frame, stations=load_station_ids(db))
