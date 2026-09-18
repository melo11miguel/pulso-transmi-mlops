"""Controles de calidad de datos.

Se usan en dos lugares: el collector (rechaza lotes inválidos antes de tocar la
base) y el análisis exploratorio (reporta la calidad del histórico).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

MAX_DEMAND = 100_000  # límite del contrato de la API
STEP = timedelta(minutes=15)


class DataQualityError(ValueError):
    """El lote no cumple el contrato y no debe persistirse."""


def _parse_ts(value: object) -> datetime:
    ts = datetime.fromisoformat(str(value))
    if ts.tzinfo is None:
        raise DataQualityError(f"timestamp sin zona horaria: {value!r}")
    return ts


def validate_observations(rows: list[dict], known_stations: set[str]) -> list[dict]:
    """Valida y normaliza filas del stream. Lanza DataQualityError si algo falla.

    Devuelve filas con `station_id` texto, `observed_at` ISO 8601 con zona y
    `demand` entero. No elimina filas en silencio: un lote con una fila inválida
    se rechaza completo para que el error sea visible.
    """
    seen: set[tuple[str, datetime]] = set()
    clean: list[dict] = []
    for row in rows:
        station = row.get("station_id")
        if not isinstance(station, str) or station not in known_stations:
            raise DataQualityError(f"estación desconocida: {station!r}")
        ts = _parse_ts(row.get("observed_at"))
        if ts.minute % 15 or ts.second or ts.microsecond:
            raise DataQualityError(f"timestamp fuera de la rejilla de 15 min: {ts.isoformat()}")
        demand = row.get("demand")
        if isinstance(demand, bool) or not isinstance(demand, (int, float)):
            raise DataQualityError(f"demanda no numérica: {demand!r}")
        if demand != demand or demand in (float("inf"), float("-inf")):
            raise DataQualityError("demanda no finita")
        if demand < 0 or demand > MAX_DEMAND:
            raise DataQualityError(f"demanda fuera de rango [0, {MAX_DEMAND}]: {demand}")
        if int(demand) != demand:
            raise DataQualityError(f"demanda no entera: {demand}")
        key = (station, ts)
        if key in seen:
            raise DataQualityError(f"duplicado en el lote: {station} {ts.isoformat()}")
        seen.add(key)
        clean.append(
            {"station_id": station, "observed_at": ts.isoformat(), "demand": int(demand)}
        )
    return clean


@dataclass
class QualityReport:
    rows: int
    stations: int
    start: pd.Timestamp
    end: pd.Timestamp
    duplicates: int
    missing_periods: int
    nulls: int
    negatives: int
    out_of_grid: int
    per_station: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (
            self.duplicates or self.missing_periods or self.nulls or self.negatives
            or self.out_of_grid
        )


def quality_report(obs: pd.DataFrame) -> QualityReport:
    """Resume continuidad, duplicados y validez de un DataFrame de observaciones.

    Espera columnas `station_id`, `observed_at` (datetime con zona) y `demand`.
    """
    duplicates = int(obs.duplicated(["station_id", "observed_at"]).sum())
    start, end = obs["observed_at"].min(), obs["observed_at"].max()
    grid = pd.date_range(start, end, freq="15min")
    per_station = obs.groupby("station_id")["observed_at"].nunique().to_dict()
    missing = sum(len(grid) - n for n in per_station.values())
    ts = obs["observed_at"]
    out_of_grid = int(((ts.dt.minute % 15 != 0) | (ts.dt.second != 0)).sum())
    return QualityReport(
        rows=len(obs),
        stations=obs["station_id"].nunique(),
        start=start,
        end=end,
        duplicates=duplicates,
        missing_periods=int(missing),
        nulls=int(obs[["station_id", "observed_at", "demand"]].isna().sum().sum()),
        negatives=int((obs["demand"] < 0).sum()),
        out_of_grid=out_of_grid,
        per_station={k: int(v) for k, v in per_station.items()},
    )
