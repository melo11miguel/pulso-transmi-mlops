"""Modelo de pronóstico: perfil estacional + corrección por gradient boosting.

Predice, para cada estación y horizonte (+15…+60 min), la desviación logarítmica respecto al
perfil «estación × hora de la semana». Pérdida L1: minimiza el error absoluto, que es lo que
mide el WAPE. La interfaz `Forecaster` (fit / predict_batch) la comparten los baselines.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import (
    CATEGORICAL,
    FEATURE_COLUMNS,
    HORIZONS,
    STEP,
    Profile,
    build_features,
    extend_grid,
)
from .monitor import level_noise

PRED_COLUMNS = ["origin_idx", "station_idx", "horizon", "target_at", "prediction"]


@dataclass(frozen=True)
class ModelConfig:
    half_life_days: float | None = None  # None = todas las semanas pesan igual
    max_iter: int = 250
    learning_rate: float = 0.06
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 100
    l2_regularization: float = 1.0
    min_origin: int = 96  # descarta el primer día: sin historia para los rezagos largos
    seed: int = 42
    drop_features: tuple[str, ...] = ()  # para ablaciones; vacío en producción


class Forecaster:
    """Interfaz común. `predict_batch` recibe la rejilla con filas futuras vacías al final."""

    name = "forecaster"

    def fit(self, wide: pd.DataFrame) -> Forecaster:
        raise NotImplementedError

    def predict_batch(self, wide_ext: pd.DataFrame, origins: np.ndarray,
                      horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
        raise NotImplementedError


def _frame(wide_ext: pd.DataFrame, origins: np.ndarray, h: int, values: np.ndarray) -> pd.DataFrame:
    n_stations = wide_ext.shape[1]
    target_times = wide_ext.index[np.asarray(origins) + h]
    return pd.DataFrame({
        "origin_idx": np.repeat(origins, n_stations),
        "station_idx": np.tile(np.arange(n_stations), len(origins)),
        "horizon": h,
        "target_at": np.repeat(target_times.to_numpy(), n_stations),
        "prediction": values,
    })


class GbmResidualModel(Forecaster):
    name = "gbm_residual"

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.feature_columns = [c for c in FEATURE_COLUMNS if c not in self.config.drop_features]
        self.profile = Profile(self.config.half_life_days)
        self.gbm: HistGradientBoostingRegressor | None = None
        self.stations: list[str] = []
        self.train_start: pd.Timestamp | None = None
        self.train_end: pd.Timestamp | None = None
        self.n_train_rows = 0
        self.level_noise: dict[str, float] = {}

    # ---- entrenamiento --------------------------------------------------------------------
    def fit(self, wide: pd.DataFrame) -> GbmResidualModel:
        cfg = self.config
        self.stations = [str(c) for c in wide.columns]
        times, demand = wide.index, wide.to_numpy(dtype=float)
        self.train_start, self.train_end = times[0], times[-1]
        self.profile.fit(times, np.log(np.where(demand > 0, demand, np.nan)))
        self.level_noise = level_noise(self.profile, wide)

        parts_x, parts_y = [], []
        for h in HORIZONS:
            origins = np.arange(cfg.min_origin, len(times) - h)
            fs = build_features(self.profile, times, demand, origins, h)
            ok = np.isfinite(fs.target) & np.isfinite(fs.X["p_target"].to_numpy())
            parts_x.append(fs.X[ok])
            parts_y.append(fs.target[ok])
        x = pd.concat(parts_x, ignore_index=True)[self.feature_columns]
        y = np.concatenate(parts_y)
        self.n_train_rows = len(y)
        self.gbm = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=cfg.max_iter, learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.max_leaf_nodes, min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=cfg.l2_regularization, categorical_features=CATEGORICAL,
            early_stopping=False, random_state=cfg.seed,
        ).fit(x, y)
        return self

    # ---- predicción -----------------------------------------------------------------------
    def predict_batch(self, wide_ext: pd.DataFrame, origins: np.ndarray,
                      horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
        if self.gbm is None:
            raise RuntimeError("El modelo no está entrenado")
        if [str(c) for c in wide_ext.columns] != self.stations:
            raise ValueError("Las estaciones de la rejilla no coinciden con las del modelo")
        times, demand = wide_ext.index, wide_ext.to_numpy(dtype=float)
        origins = np.asarray(origins)
        frames = []
        for h in horizons:
            fs = build_features(self.profile, times, demand, origins, h)
            residual = self.gbm.predict(fs.X[self.feature_columns])
            # Sin observación en el origen no hay señal reciente: se cae al perfil puro.
            residual = np.where(np.isnan(fs.X["r0"].to_numpy()), 0.0, residual)
            frames.append(_frame(wide_ext, origins, h, np.exp(fs.base + residual)))
        return pd.concat(frames, ignore_index=True)

    def predict_next(self, history: pd.DataFrame) -> pd.DataFrame:
        """Pronostica +15…+60 min desde la última fila de `history` (rejilla hasta el corte).

        Devuelve station_id, target_at, horizon_minutes, value. Solo usa filas de `history`.
        """
        history = history.reindex(columns=self.stations)
        max_h = max(HORIZONS)
        # Un paso más que el horizonte máximo: `p_curv` mira el perfil en objetivo+1 y sin esa
        # fila la curvatura del horizonte de 60 min saldría recortada.
        wide_ext = extend_grid(history, max_h + 1)
        origin = len(history) - 1
        out = self.predict_batch(wide_ext, np.array([origin]))
        out["station_id"] = [self.stations[i] for i in out["station_idx"]]
        out["horizon_minutes"] = out["horizon"] * int(STEP / pd.Timedelta(minutes=1))
        out["value"] = out["prediction"].clip(lower=0.0)
        return out[["station_id", "target_at", "horizon_minutes", "value"]]

    # ---- persistencia ---------------------------------------------------------------------
    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        joblib.dump({"model": self, "sklearn": sklearn.__version__, "numpy": np.__version__},
                    buffer, compress=3)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> GbmResidualModel:
        payload = joblib.load(io.BytesIO(data))
        if payload["sklearn"] != sklearn.__version__:
            raise RuntimeError(
                f"Artefacto entrenado con scikit-learn {payload['sklearn']}, "
                f"instalado {sklearn.__version__}: reentrene o fije la versión"
            )
        return payload["model"]

    def describe(self) -> dict:
        return {"config": asdict(self.config), "stations": self.stations,
                "train_start": self.train_start.isoformat(), "train_end": self.train_end.isoformat(),
                "n_train_rows": self.n_train_rows, "features": self.feature_columns,
                "level_noise": self.level_noise,
                "sklearn": sklearn.__version__}
