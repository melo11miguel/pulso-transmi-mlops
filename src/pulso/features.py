"""Rejilla de demanda, perfil estacional y construcción de variables.

Regla de oro (probada en tests/test_features.py): las variables de una fila con origen `o`
solo dependen de observaciones con índice <= o. Nada del futuro entra al modelo.

El perfil estacional «estación × hora de la semana» se calcula con leave-one-out dentro de la
ventana de entrenamiento: el valor de un instante nunca forma parte de su propio perfil. Así el
residuo que ve el modelo al entrenar se parece al que verá al predecir.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import TZ

STEP = pd.Timedelta("15min")
SLOTS_PER_DAY = 96
HOURS_OF_WEEK = 7 * SLOTS_PER_DAY  # 672 periodos de 15 min por semana
HORIZONS = (1, 2, 3, 4)  # en periodos de 15 min: +15, +30, +45, +60


def to_wide(obs: pd.DataFrame, stations: list[str] | None = None) -> pd.DataFrame:
    """Demanda como matriz tiempo × estación en rejilla completa de 15 min (hora de Bogotá)."""
    frame = obs[["observed_at", "station_id", "demand"]].copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True).dt.tz_convert(TZ)
    frame["station_id"] = frame["station_id"].astype(str)
    wide = frame.groupby(["observed_at", "station_id"])["demand"].last().unstack("station_id")
    if stations is not None:
        wide = wide.reindex(columns=stations)
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq=STEP)
    wide = wide.reindex(grid).astype(float)
    wide.index.name = "observed_at"
    return wide


def how_index(times: pd.DatetimeIndex) -> np.ndarray:
    """Hora de la semana 0..671 (lunes 00:00 = 0)."""
    return (times.dayofweek * SLOTS_PER_DAY + times.hour * 4 + times.minute // 15).to_numpy()


def extend_grid(wide: pd.DataFrame, steps: int) -> pd.DataFrame:
    """Agrega `steps` filas futuras vacías (NaN) para poder referenciar los targets."""
    future = pd.date_range(wide.index[-1] + STEP, periods=steps, freq=STEP)
    return pd.concat([wide, pd.DataFrame(np.nan, index=future, columns=wide.columns)])


class Profile:
    """Media logarítmica por (hora de la semana, estación), con decaimiento opcional por edad."""

    def __init__(self, half_life_days: float | None = None) -> None:
        self.half_life_days = half_life_days
        self.sums: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.station_mean: np.ndarray | None = None
        self.fit_start: pd.Timestamp | None = None
        self.fit_end: pd.Timestamp | None = None

    def _w(self, times: pd.DatetimeIndex) -> np.ndarray:
        if not self.half_life_days:
            return np.ones(len(times))
        age_days = (self.fit_end - times) / pd.Timedelta(days=1)
        return np.exp2(-np.clip(np.asarray(age_days, dtype=float), 0.0, None) / self.half_life_days)

    def fit(self, times: pd.DatetimeIndex, log_demand: np.ndarray) -> Profile:
        self.fit_start, self.fit_end = times[0], times[-1]
        w = self._w(times)[:, None]
        valid = ~np.isnan(log_demand)
        n_stations = log_demand.shape[1]
        self.sums = np.zeros((HOURS_OF_WEEK, n_stations))
        self.weights = np.zeros((HOURS_OF_WEEK, n_stations))
        how = how_index(times)
        np.add.at(self.sums, how, np.where(valid, log_demand * w, 0.0))
        np.add.at(self.weights, how, np.where(valid, w, 0.0))
        self.station_mean = np.nanmean(log_demand, axis=0)
        return self

    def matrix(self, times: pd.DatetimeIndex, log_demand: np.ndarray | None = None) -> np.ndarray:
        """Perfil (tiempo × estación). Con `log_demand`, los instantes dentro de la ventana de
        ajuste usan leave-one-out; el resto (futuro) usa la media simple."""
        how = how_index(times)
        with np.errstate(invalid="ignore", divide="ignore"):
            plain = self.sums[how] / self.weights[how]
        plain = np.where(self.weights[how] > 0, plain, self.station_mean[None, :])
        if log_demand is None:
            return plain
        inside = ((times >= self.fit_start) & (times <= self.fit_end))[:, None]
        w = self._w(times)[:, None]
        numerator = self.sums[how] - w * np.nan_to_num(log_demand)
        denominator = self.weights[how] - w
        usable = inside & ~np.isnan(log_demand) & (denominator > 1e-9)
        with np.errstate(invalid="ignore", divide="ignore"):
            loo = numerator / denominator
        return np.where(usable, loo, plain)


@dataclass(frozen=True)
class FeatureSet:
    """Filas de features para un horizonte, con metadatos para reconstruir la predicción."""

    X: pd.DataFrame
    base: np.ndarray  # log-perfil en el instante objetivo
    target: np.ndarray  # log(demanda real) − perfil (NaN si aún se desconoce)
    meta: pd.DataFrame  # origin_idx, station_idx, horizon, target_at


FEATURE_COLUMNS = [
    "station", "horizon", "slot", "dow", "how", "slot_sin", "slot_cos",
    "p_target", "p_origin", "p_delta", "p_curv",
    "r0", "r1", "r2", "r3", "m4", "m16", "m96", "c0", "c4",
]
CATEGORICAL = ["station"]


def build_features(profile: Profile, times: pd.DatetimeIndex, demand: np.ndarray,
                   origins: np.ndarray, horizon: int) -> FeatureSet:
    """Variables para todos los `origins` (índices) y estaciones, a `horizon` periodos.

    `times`/`demand` deben incluir la fila del instante objetivo (NaN si es futuro).
    Solo se usan filas con índice <= origen para las variables; el objetivo usa la fila origen+h.
    """
    n_time, n_stations = demand.shape
    log_demand = np.log(np.where(demand > 0, demand, np.nan))
    p = profile.matrix(times, log_demand)  # perfil por instante (LOO dentro de la ventana)
    resid = log_demand - p
    resid_frame = pd.DataFrame(resid)
    with warnings.catch_warnings():  # filas sin ninguna estación observada -> NaN, sin aviso
        warnings.simplefilter("ignore", category=RuntimeWarning)
        city = pd.Series(np.nanmean(resid, axis=1))
    roll = {w: resid_frame.rolling(w, min_periods=max(1, w // 2)).mean().to_numpy()
            for w in (4, 16, 96)}
    city4 = city.rolling(4, min_periods=1).mean().to_numpy()

    origins = np.asarray(origins)
    target_idx = origins + horizon
    if target_idx.max() >= n_time:
        raise ValueError("La rejilla no incluye el instante objetivo; use extend_grid()")

    def at(matrix: np.ndarray, idx: np.ndarray) -> np.ndarray:
        safe = np.clip(idx, 0, n_time - 1)
        out = matrix[safe]
        out[idx < 0] = np.nan
        return out

    target_times = times[target_idx]
    slot = (target_times.hour * 4 + target_times.minute // 15).to_numpy()
    dow = target_times.dayofweek.to_numpy()
    how = how_index(target_times)
    o_len = len(origins)

    def tile(vec: np.ndarray) -> np.ndarray:  # (O,) -> (O*S,) repitiendo por estación
        return np.repeat(vec, n_stations)

    def flat(matrix: np.ndarray) -> np.ndarray:  # (O,S) -> (O*S,) fila-mayor
        return matrix.reshape(-1)

    p_target = flat(p[target_idx])
    p_origin = flat(p[origins])
    # Curvatura del perfil en el instante objetivo (segunda diferencia). El perfil es escalonado
    # por hora de la semana, así que su segunda diferencia marca dónde está a punto de quebrarse:
    # es la variable que le falta al modelo para acertar en las rampas (amanecer y caída de la
    # tarde), que eran sus peores horas. Se calcula con el mismo perfil que `p_target`, para que
    # entrenamiento y producción vean lo mismo. `extend_grid` reserva la fila objetivo+1; en el
    # borde del histórico se recorta contra la última fila, que es una sola de cada lote.
    p_prev = flat(at(p, target_idx - 1))
    p_next = flat(at(p, np.minimum(target_idx + 1, n_time - 1)))
    y_target = log_demand[target_idx]
    data = {
        "station": np.tile(np.arange(n_stations), o_len),
        "horizon": np.full(o_len * n_stations, horizon),
        "slot": tile(slot), "dow": tile(dow), "how": tile(how),
        "slot_sin": tile(np.sin(2 * np.pi * slot / SLOTS_PER_DAY)),
        "slot_cos": tile(np.cos(2 * np.pi * slot / SLOTS_PER_DAY)),
        "p_target": p_target, "p_origin": p_origin, "p_delta": p_target - p_origin,
        "p_curv": p_next - 2 * p_target + p_prev,
        "r0": flat(at(resid, origins)), "r1": flat(at(resid, origins - 1)),
        "r2": flat(at(resid, origins - 2)), "r3": flat(at(resid, origins - 3)),
        "m4": flat(roll[4][origins]), "m16": flat(roll[16][origins]), "m96": flat(roll[96][origins]),
        "c0": tile(city[origins].to_numpy()), "c4": tile(city4[origins]),
    }
    X = pd.DataFrame(data)[FEATURE_COLUMNS]
    meta = pd.DataFrame({
        "origin_idx": np.repeat(origins, n_stations),
        "station_idx": np.tile(np.arange(n_stations), o_len),
        "horizon": horizon,
        "target_at": np.repeat(target_times.to_numpy(), n_stations),
    })
    return FeatureSet(X=X, base=p_target, target=flat(y_target) - p_target, meta=meta)
