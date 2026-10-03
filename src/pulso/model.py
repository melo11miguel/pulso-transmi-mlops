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


class IncompatibleArtifactError(RuntimeError):
    """El artefacto del champion no se puede usar con este código.

    Base comun de los fallos permanentes de compatibilidad. Importa que sean un tipo propio: el
    bucle de la sesion cuenta errores seguidos y a los 5 se mata, asi que un fallo que reintentar
    no arregla tiene que degradar al perfil en vez de gastar reintentos. Las subclases distinguen
    la causa, pero el camino de inferencia las trata igual.
    """


class UnusableArtifactError(IncompatibleArtifactError):
    """El artefacto no se pudo descargar, deserializar o usar con este código.

    Es el caso general del anterior: no es que falte una variable, es que el objeto mismo no sirve
    (hash que no cuadra, pickle de una version mas nueva, perfil con una clave que este codigo no
    conoce). Antes esto reventaba en `registry.load_champion()`, fuera del guard, y se comia los
    5 reintentos del bucle hasta matar la sesion.
    """


class FeatureMismatchError(IncompatibleArtifactError):
    """El artefacto pide variables que este código no sabe construir.

    Pasa cuando se promueve un modelo con una variable nueva mientras hay procesos en vuelo con
    el código anterior: el registro les entrega el champion nuevo y `build_features` no produce
    esa columna. Es un error permanente, no transitorio: reintentarlo no lo arregla, así que se
    distingue por tipo para que la inferencia degrade al perfil en vez de reintentar en vano.
    """

    def __init__(self, missing: list[str]) -> None:
        self.missing = list(missing)
        super().__init__(
            f"El modelo requiere variables que este código no genera: {', '.join(self.missing)}. "
            "El artefacto es más nuevo que el código desplegado; actualice el despliegue."
        )


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
    # Potencia del peso de entrenamiento: w = y^p normalizado dentro de cada estación.
    #
    # La L1 sobre el residual LOG no es WAPE, aunque el README lo decía. Para errores pequeños
    # |log y - log ŷ| ≈ |y - ŷ| / y, así que la L1 en log pondera cada error absoluto por 1/y,
    # mientras WAPE los pondera igual. Consecuencia medida: un target de 50 pasajeros de
    # madrugada pesaba en el ajuste lo mismo que uno de 700 en hora pico, y el pico se
    # subpredecía un 7 %.
    #
    # p=0 es la pérdida vieja, p=1 la métrica exacta, p=0.5 el punto medio. Medido en la ventana
    # de competencia sobre cuatro particiones temporales, p=0.5 gana las cuatro (+0,88, +0,84,
    # +0,72, +0,10; media +0,635) y le gana a p=1 en tres de las cuatro: la métrica exacta
    # sobrecorrige y se pasa cuando la base ya viene sin sesgo.
    #
    # El valor por defecto es 0 A PROPÓSITO: al reconstruir la receta de un champion antiguo hay
    # que darle la suya, no la nueva, o la puerta compararía dos modelos idénticos.
    peso_metrica: float = 0.0
    # Dias de historia que se usan para entrenar. None = toda la disponible.
    #
    # El reto inyecta eventos que reorganizan la demanda entera (Ricaurte +107 %, El Dorado
    # +102 %, Universidad Nacional -33 %). Con 54 dias de historia el regimen nuevo queda ahogado:
    # medido sobre los ciclos posteriores al evento, entrenar con todo da 42,72 y con 21 dias
    # 47,10. Con 14 da 47,87 pero es fragil: con 7 se desploma a 17,54 porque el perfil por hora de
    # la semana se queda con celdas casi vacias. 21 dias deja tres observaciones por celda.
    #
    # Por defecto None, para que `receta_del_champion` reconstruya fielmente a un champion antiguo.
    dias_de_historia: int | None = None
    # Clave del perfil: "semana" son 672 celdas (franja de la semana) y necesita una semana entera
    # de datos para llenarse; "dia" son 96 (franja del dia) y se llena con un dia, a costa de no
    # distinguir martes de domingo. Con un regimen de dos dias de vida esa diferencia lo es todo:
    # medida en 3 bloques alrededor del evento del 09-18, la clave de dia con 7 dias de historia
    # gana +10,24 contra la receta vigente, ganando los 3 bloques. Ver `dow_index` en features.
    #
    # Por defecto "semana" para que `receta_del_champion` reconstruya fielmente a uno antiguo.
    grano_perfil: str = "semana"


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
        self.profile = Profile(self.config.half_life_days, self.config.grano_perfil)
        self.gbm: HistGradientBoostingRegressor | None = None
        self.stations: list[str] = []
        self.train_start: pd.Timestamp | None = None
        self.train_end: pd.Timestamp | None = None
        self.n_train_rows = 0
        self.level_noise: dict[str, float] = {}

    # ---- entrenamiento --------------------------------------------------------------------
    def fit(self, wide: pd.DataFrame) -> GbmResidualModel:
        cfg = self.config
        if cfg.dias_de_historia:
            desde = wide.index[-1] - pd.Timedelta(days=cfg.dias_de_historia)
            recortada = wide.loc[desde:]
            # El minimo depende de la clave del perfil, no es un numero suelto: el perfil semanal
            # reparte la historia en 672 celdas y con menos de diez dias se queda sin observaciones
            # en muchas; el de franja del dia usa 96 y le basta un par de dias. Antes esta guarda
            # era un 10 fijo, que habria anulado en silencio la ventana de tres dias.
            if len(recortada) >= 96 * (10 if cfg.grano_perfil == "semana" else 2):
                wide = recortada
        self.stations = [str(c) for c in wide.columns]
        times, demand = wide.index, wide.to_numpy(dtype=float)
        self.train_start, self.train_end = times[0], times[-1]
        self.profile.fit(times, np.log(np.where(demand > 0, demand, np.nan)))
        self.level_noise = level_noise(self.profile, wide)

        parts_x, parts_y, parts_w, parts_s = [], [], [], []
        for h in HORIZONS:
            origins = np.arange(cfg.min_origin, len(times) - h)
            fs = build_features(self.profile, times, demand, origins, h)
            ok = np.isfinite(fs.target) & np.isfinite(fs.X["p_target"].to_numpy())
            parts_x.append(fs.X[ok])
            parts_y.append(fs.target[ok])
            # y real reconstruida exactamente: exp(perfil + residual) = exp(log y) = y
            parts_w.append(np.exp(fs.base[ok] + fs.target[ok]))
            parts_s.append(fs.meta["station_idx"].to_numpy()[ok])
        x = pd.concat(parts_x, ignore_index=True)[self.feature_columns]
        y = np.concatenate(parts_y)
        self.n_train_rows = len(y)
        peso = self._pesos(np.concatenate(parts_w), np.concatenate(parts_s)) if cfg.peso_metrica else None
        self.gbm = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=cfg.max_iter, learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.max_leaf_nodes, min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=cfg.l2_regularization, categorical_features=CATEGORICAL,
            early_stopping=False, random_state=cfg.seed,
        ).fit(x, y, sample_weight=peso)
        return self

    def _pesos(self, y_real: np.ndarray, station: np.ndarray) -> np.ndarray:
        """w = y^p normalizado DENTRO de cada estación, con media 1.

        La normalización por estación es la que respeta el promedio sin ponderar de la métrica:
        con peso global, la hora pico de las estaciones grandes se comería el ajuste entero.
        """
        w = np.power(np.maximum(y_real, 0.0), self.config.peso_metrica)
        total = pd.Series(w).groupby(station).transform("sum").to_numpy()
        w = w / np.where(total > 0, total, 1.0)
        return w * (len(w) / w.sum())

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
        # `build_features` produce exactamente FEATURE_COLUMNS. Si el artefacto pide alguna que no
        # esté ahí, el código es más viejo que el modelo: se corta con un error propio y claro en
        # vez de un KeyError de pandas a mitad del lote.
        faltan = [c for c in self.feature_columns if c not in FEATURE_COLUMNS]
        if faltan:
            raise FeatureMismatchError(faltan)
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
