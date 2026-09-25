"""Monitoreo: accuracy móvil, cambio de nivel por estación y persistencia de las señales.

Funciones puras sobre DataFrames; el pipeline (`monitor_job.py`) las alimenta con datos de Supabase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import Profile
from .metrics import official_accuracy, station_accuracy


def window_mask(target_at: pd.Series, now: pd.Timestamp, hours: float) -> pd.Series:
    """Targets ya resueltos en (now - hours, now]."""
    return (target_at > now - pd.Timedelta(hours=hours)) & (target_at <= now)


def rolling_accuracy(scored: pd.DataFrame, now: pd.Timestamp, hours: float = 24.0,
                     min_stations: int = 6) -> tuple[float, int]:
    """Accuracy oficial sobre los targets resueltos en las últimas `hours` horas.

    `scored`: station_id, target_at (con zona), actual, prediction. Devuelve (accuracy, n).
    Con menos de `min_stations` estaciones con datos devuelve NaN: un promedio de pocas
    estaciones no es comparable con la referencia de 12.
    """
    part = scored[window_mask(scored["target_at"], now, hours)].dropna(subset=["actual"])
    if part["station_id"].nunique() < min_stations:
        return float("nan"), len(part)
    return official_accuracy(part["actual"], part["prediction"], part["station_id"]), len(part)


def accuracy_by_station(scored: pd.DataFrame, now: pd.Timestamp, hours: float = 24.0) -> dict:
    part = scored[window_mask(scored["target_at"], now, hours)].dropna(subset=["actual"])
    if part.empty:
        return {}
    return station_accuracy(part["actual"], part["prediction"], part["station_id"]).round(2).to_dict()


def station_level_shift(profile: Profile, wide: pd.DataFrame, now: pd.Timestamp,
                        hours: float = 24.0) -> pd.Series:
    """Media del residuo logarítmico (real vs perfil del modelo) por estación en las últimas horas.

    No depende de las predicciones enviadas: detecta cambios de nivel aunque se pierda un ciclo.
    +0.10 ≈ la demanda real está ~10 % por encima de lo que espera el perfil.
    """
    window = wide.loc[(wide.index > now - pd.Timedelta(hours=hours)) & (wide.index <= now)]
    if window.empty:
        return pd.Series(np.nan, index=wide.columns)
    demand = window.to_numpy(dtype=float)
    log_demand = np.log(np.where(demand > 0, demand, np.nan))
    residual = log_demand - profile.matrix(window.index, log_demand)
    with np.errstate(all="ignore"):
        shift = np.nanmean(residual, axis=0)
    return pd.Series(shift, index=wide.columns)


def level_noise(profile: Profile, wide: pd.DataFrame, hours: float = 24.0) -> dict[str, float]:
    """Ruido de fondo del desplazamiento de nivel por estación, medido en la ventana de entrenamiento.

    Es el máximo de |media móvil de `hours` del residuo (leave-one-out)|. Incluye el efecto de la
    lluvia y de los eventos que ya ocurrieron: una estación sensible a eventos tiene más ruido y
    por tanto un umbral más alto. Así un evento transitorio no se confunde con un cambio de nivel.
    """
    demand = wide.to_numpy(dtype=float)
    log_demand = np.log(np.where(demand > 0, demand, np.nan))
    residual = pd.DataFrame(log_demand - profile.matrix(wide.index, log_demand),
                            index=wide.index, columns=[str(c) for c in wide.columns])
    window = int(hours * 4)
    rolled = residual.rolling(window, min_periods=int(window * 0.75)).mean()
    return {c: float(rolled[c].abs().max()) for c in rolled.columns}


def level_thresholds(noise: dict[str, float], floor: float, multiplier: float) -> pd.Series:
    """Umbral por estación: max(piso, multiplicador × ruido de fondo propio)."""
    return pd.Series({s: max(floor, multiplier * (v if np.isfinite(v) else 0.0))
                      for s, v in noise.items()})


def update_streaks(previous: dict[str, int], shifts: pd.Series,
                   thresholds: pd.Series | float) -> dict[str, int]:
    """Evaluaciones consecutivas con |desplazamiento| >= umbral de la estación."""
    streaks = {}
    for station, value in shifts.items():
        limit = float(thresholds[station]) if isinstance(thresholds, pd.Series) else float(thresholds)
        hit = bool(np.isfinite(value) and abs(value) >= limit)
        streaks[str(station)] = previous.get(str(station), 0) + 1 if hit else 0
    return streaks


def baseline_accuracy(profile: Profile, wide: pd.DataFrame, scored: pd.DataFrame,
                      now: pd.Timestamp, hours: float = 24.0) -> float:
    """Accuracy del perfil puro sobre EXACTAMENTE los mismos targets: ¿el modelo aporta algo?"""
    part = scored[window_mask(scored["target_at"], now, hours)].dropna(subset=["actual"]).copy()
    if part.empty:
        return float("nan")
    columns = {s: i for i, s in enumerate(wide.columns)}
    # A la zona de la rejilla antes de evaluar el perfil: es funcion de la hora de la semana, asi
    # que con las horas en UTC queda desplazado cinco horas y devuelve un disparate. Estuvo asi
    # todo el proyecto y el baseline almacenado marcaba 12,15 donde el perfil real saca ~87.
    times = pd.DatetimeIndex(part["target_at"])
    times = times.tz_localize("UTC") if times.tz is None else times
    times = times.tz_convert(wide.index.tz)
    base = np.exp(profile.matrix(times))
    part["prediction"] = base[np.arange(len(part)), part["station_id"].map(columns).to_numpy()]
    return official_accuracy(part["actual"], part["prediction"], part["station_id"])
