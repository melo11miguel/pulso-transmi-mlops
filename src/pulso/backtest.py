"""Backtesting temporal: entrena con el pasado y valida en semanas posteriores.

Cada fold entrena con datos <= train_end y evalúa en la ventana siguiente con orígenes cada
hora en punto (como los ciclos reales), a +15/+30/+45/+60 min. Nunca hay particiones aleatorias.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import HORIZONS, STEP
from .metrics import official_accuracy, station_accuracy, wape
from .model import Forecaster

VAL_DAYS = 7


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: pd.Timestamp  # último instante de entrenamiento (inclusive)
    val_start: pd.Timestamp  # primer origen posible de validación
    val_end: pd.Timestamp  # último instante objetivo (inclusive)


def make_folds(times: pd.DatetimeIndex, n_folds: int = 3, val_days: int = VAL_DAYS) -> list[Fold]:
    """Ventanas semanales consecutivas que terminan en el último dato; expansivas hacia atrás."""
    last = times[-1]
    folds = []
    for k in range(n_folds - 1, -1, -1):
        val_end = last - pd.Timedelta(days=val_days * k)
        val_start = val_end - pd.Timedelta(days=val_days) + STEP
        folds.append(Fold(f"fold{n_folds - k}", val_start - STEP, val_start, val_end))
    return folds


def hourly_origins(times: pd.DatetimeIndex, fold: Fold) -> np.ndarray:
    """Índices de origen: cada hora en punto, con los 4 horizontes dentro de la ventana."""
    last_origin = fold.val_end - max(HORIZONS) * STEP
    mask = (times >= fold.val_start) & (times <= last_origin) & (times.minute == 0)
    return np.flatnonzero(mask)


def score_predictions(wide: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Une predicción con valor real (fila origen+h) y devuelve una fila por target."""
    demand = wide.to_numpy(dtype=float)
    target_idx = preds["origin_idx"].to_numpy() + preds["horizon"].to_numpy()
    out = preds.copy()
    out["actual"] = demand[target_idx, preds["station_idx"].to_numpy()]
    out["station_id"] = wide.columns.to_numpy()[preds["station_idx"].to_numpy()]
    return out.dropna(subset=["actual"])


def summarize(scored: pd.DataFrame) -> dict:
    per_h = {
        int(h) * 15: round(official_accuracy(g["actual"], g["prediction"], g["station_id"]), 2)
        for h, g in scored.groupby("horizon")
    }
    return {
        "accuracy": official_accuracy(scored["actual"], scored["prediction"], scored["station_id"]),
        "wape": wape(scored["actual"], scored["prediction"]),
        "n_targets": len(scored),
        "by_horizon": per_h,
        "by_station": station_accuracy(scored["actual"], scored["prediction"],
                                       scored["station_id"]).round(2).to_dict(),
    }


def run_backtest(factories: dict[str, Callable[[], Forecaster]], wide: pd.DataFrame,
                 folds: list[Fold]) -> tuple[pd.DataFrame, dict[tuple[str, str], dict]]:
    """Devuelve (tabla modelo × fold, detalle por (modelo, fold))."""
    rows, detail = [], {}
    for fold in folds:
        train = wide.loc[:fold.train_end]
        origins = hourly_origins(wide.index, fold)
        for name, factory in factories.items():
            model = factory().fit(train)
            scored = score_predictions(wide, model.predict_batch(wide, origins))
            summary = summarize(scored)
            detail[(name, fold.name)] = summary
            rows.append({"model": name, "fold": fold.name, "train_days": len(train) / 96,
                         "accuracy": summary["accuracy"], "wape": summary["wape"],
                         "n_targets": summary["n_targets"]})
    return pd.DataFrame(rows), detail
