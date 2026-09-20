"""Baselines contra los que se juzga cualquier modelo (todos causales)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .features import FEATURE_COLUMNS, HORIZONS, SLOTS_PER_DAY, Profile, build_features, how_index
from .model import Forecaster, _frame


class LastValue(Forecaster):
    """Repite la última observación."""

    name = "last_value"

    def fit(self, wide: pd.DataFrame) -> LastValue:
        return self

    def predict_batch(self, wide_ext, origins, horizons=HORIZONS):
        demand = wide_ext.to_numpy(dtype=float)
        return pd.concat(
            [_frame(wide_ext, origins, h, demand[origins].reshape(-1)) for h in horizons],
            ignore_index=True,
        )


class SeasonalNaive(Forecaster):
    """Mismo instante de hace `period` periodos (96 = ayer, 672 = semana pasada)."""

    def __init__(self, period: int) -> None:
        self.period = period
        self.name = {SLOTS_PER_DAY: "same_time_yesterday", 7 * SLOTS_PER_DAY: "same_time_last_week"}.get(
            period, f"seasonal_naive_{period}")

    def fit(self, wide: pd.DataFrame) -> SeasonalNaive:
        return self

    def predict_batch(self, wide_ext, origins, horizons=HORIZONS):
        demand = wide_ext.to_numpy(dtype=float)
        frames = []
        for h in horizons:
            idx = np.asarray(origins) + h - self.period
            if (idx > np.asarray(origins)).any():
                raise ValueError("period demasiado corto: usaría información futura")
            values = np.where((idx >= 0)[:, None], demand[np.clip(idx, 0, None)], np.nan)
            frames.append(_frame(wide_ext, origins, h, values.reshape(-1)))
        return pd.concat(frames, ignore_index=True)


class ProfileMedian(Forecaster):
    """Mediana de la demanda en la misma hora de la semana durante la ventana de entrenamiento."""

    name = "profile_median"

    def __init__(self) -> None:
        self.table: np.ndarray | None = None

    def fit(self, wide: pd.DataFrame) -> ProfileMedian:
        how = how_index(wide.index)
        demand = wide.to_numpy(dtype=float)
        table = np.full((7 * SLOTS_PER_DAY, demand.shape[1]), np.nan)
        for k in range(table.shape[0]):
            table[k] = np.nanmedian(demand[how == k], axis=0)
        self.table = table
        return self

    def predict_batch(self, wide_ext, origins, horizons=HORIZONS):
        frames = []
        for h in horizons:
            how = how_index(wide_ext.index[np.asarray(origins) + h])
            frames.append(_frame(wide_ext, origins, h, self.table[how].reshape(-1)))
        return pd.concat(frames, ignore_index=True)


class ProfileMean(Forecaster):
    """exp(media logarítmica) por hora de la semana: el perfil puro del modelo, sin corrección."""

    name = "profile_only"

    def __init__(self, half_life_days: float | None = None) -> None:
        self.profile = Profile(half_life_days)

    def fit(self, wide: pd.DataFrame) -> ProfileMean:
        demand = wide.to_numpy(dtype=float)
        self.profile.fit(wide.index, np.log(np.where(demand > 0, demand, np.nan)))
        return self

    def predict_batch(self, wide_ext, origins, horizons=HORIZONS):
        frames = []
        for h in horizons:
            base = self.profile.matrix(wide_ext.index[np.asarray(origins) + h])
            frames.append(_frame(wide_ext, origins, h, np.exp(base).reshape(-1)))
        return pd.concat(frames, ignore_index=True)


class RidgeResidual(Forecaster):
    """Perfil + corrección lineal con las mismas variables recientes (baseline fuerte y simple)."""

    name = "ridge_residual"
    LINEAR = ["r0", "r1", "r2", "r3", "m4", "m16", "m96", "c0", "c4", "p_delta"]

    def __init__(self, half_life_days: float | None = None, alpha: float = 10.0) -> None:
        self.profile = Profile(half_life_days)
        self.alpha = alpha
        self.models: dict[int, Ridge] = {}

    def fit(self, wide: pd.DataFrame) -> RidgeResidual:
        times, demand = wide.index, wide.to_numpy(dtype=float)
        self.profile.fit(times, np.log(np.where(demand > 0, demand, np.nan)))
        for h in HORIZONS:
            origins = np.arange(SLOTS_PER_DAY, len(times) - h)
            fs = build_features(self.profile, times, demand, origins, h)
            x = fs.X[self.LINEAR].fillna(0.0)
            ok = np.isfinite(fs.target)
            self.models[h] = Ridge(alpha=self.alpha).fit(x[ok], fs.target[ok])
        return self

    def predict_batch(self, wide_ext, origins, horizons=HORIZONS):
        times, demand = wide_ext.index, wide_ext.to_numpy(dtype=float)
        frames = []
        for h in horizons:
            fs = build_features(self.profile, times, demand, np.asarray(origins), h)
            residual = self.models[h].predict(fs.X[self.LINEAR].fillna(0.0))
            frames.append(_frame(wide_ext, origins, h, np.exp(fs.base + residual)))
        return pd.concat(frames, ignore_index=True)


assert set(RidgeResidual.LINEAR) <= set(FEATURE_COLUMNS)
