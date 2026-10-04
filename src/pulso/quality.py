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
UNIDAD = "passengers"      # la unidad del esquema 2; otra cambiaria el significado del numero
CALIDAD_USABLE = "observed"

# Marca una fila que la API declara sin valor usable, distinta de un valor que simplemente es None.
# La diferencia importa: en el esquema 2 la ausencia viene DECLARADA en `quality`, asi que omitirla
# es seguir el contrato; en el esquema 1 no hay anotacion, asi que un None es una violacion y debe
# seguir fallando fuerte. Sin esta distincion el arreglo del esquema 2 habria aflojado el esquema 1.
AUSENTE = object()


class DataQualityError(ValueError):
    """El lote no cumple el contrato y no debe persistirse."""


def _parse_ts(value: object) -> datetime:
    ts = datetime.fromisoformat(str(value))
    if ts.tzinfo is None:
        raise DataQualityError(f"timestamp sin zona horaria: {value!r}")
    return ts


def _demanda(row: dict) -> object:
    """Demanda de una fila del stream, en los dos esquemas que publica la API.

    Esquema 1 (API hasta 0.8.0): `demand` entero plano.
    Esquema 2 (API 0.9.0): `measurement` con `value` como CADENA decimal ("546.00"), mas `unit`
    y `quality`. Devuelve AUSENTE cuando la API declara que la fila no trae valor usable, y el
    llamador la omite; cualquier otra cosa sigue el camino normal de validacion.

    Las dos ramas siguen vivas a proposito: el stream mezcla los dos esquemas en la misma pagina
    segun donde este el cursor, asi que tratarlo como una migracion limpia perderia historia.

    La distincion entre `unit` y `quality` no es cosmetica. `unit` cambia el SIGNIFICADO del
    numero, asi que una unidad inesperada es un error duro: ingerirla seria reescalar la demanda
    en silencio. `quality` describe la fiabilidad de UNA fila, asi que un valor nuevo solo hace
    que esa fila se omita, contada y registrada. Esa asimetria es la que evita repetir el fallo
    que motivo este codigo: el esquema 2 tumbo la ingesta entera durante horas y con ella la
    cobertura, que es lo unico que no se recupera.
    """
    if "measurement" in row:
        medida = row["measurement"]
        if not isinstance(medida, dict):
            raise DataQualityError(f"measurement no es un objeto: {medida!r}")
        unidad = medida.get("unit")
        if unidad != UNIDAD:
            raise DataQualityError(f"unidad inesperada: {unidad!r} (se esperaba {UNIDAD!r})")
        if medida.get("quality") != CALIDAD_USABLE:
            return AUSENTE         # "missing" y cualquier calidad futura: ausente, nunca cero
        valor = medida.get("value")
        if valor is None:
            return AUSENTE
        try:
            return float(valor)    # llega como cadena decimal, no como numero
        except (TypeError, ValueError) as exc:
            raise DataQualityError(f"valor no numérico: {valor!r}") from exc
    return row.get("demand")


def validate_observations(rows: list[dict], known_stations: set[str], *,
                          omitidas: list[dict] | None = None) -> list[dict]:
    """Valida y normaliza filas del stream. Lanza DataQualityError si algo falla.

    Devuelve filas con `station_id` texto, `observed_at` ISO 8601 con zona y
    `demand` entero. No elimina filas en silencio: un lote con una fila inválida
    se rechaza completo para que el error sea visible.

    La unica excepcion son las filas sin valor usable (esquema 2 con `quality` distinto de
    "observed"), que se omiten y, si se pasa `omitidas`, se acumulan ahi para que el colector las
    cuente. Omitirlas es lo correcto y no un atajo: una observacion ausente es DESCONOCIDA, no un
    cero. La rejilla de `to_wide` ya deja NaN en los instantes que faltan y el modelo lo maneja;
    guardar un cero, en cambio, hundiria el perfil estacional de esa franja.
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
        demand = _demanda(row)
        if demand is AUSENTE:
            if omitidas is not None:
                omitidas.append({"station_id": station, "observed_at": ts.isoformat(),
                                 "quality": (row.get("measurement") or {}).get("quality")})
            continue
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
        item = {"station_id": station, "observed_at": ts.isoformat(), "demand": int(demand)}
        if row.get("released_at") is not None:
            item["released_at"] = _parse_ts(row["released_at"]).isoformat()
        clean.append(item)
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
