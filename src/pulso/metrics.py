"""Métrica oficial del reto.

WAPE por estación = sum|real - pred| / sum(real);  accuracy = 100 * max(0, 1 - WAPE);
accuracy oficial = promedio NO ponderado de las accuracies por estación.
Un target sin predicción cuenta como predicción cero (y baja la cobertura).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def station_accuracy(actual: pd.Series, predicted: pd.Series, station: pd.Series) -> pd.Series:
    """Accuracy (0-100) por estación. `predicted` NaN = ausente = 0."""
    frame = pd.DataFrame({
        "station_id": station.to_numpy(),
        "actual": actual.to_numpy(dtype=float),
        "predicted": predicted.fillna(0.0).to_numpy(dtype=float),
    })
    frame["abs_err"] = (frame["actual"] - frame["predicted"]).abs()
    grouped = frame.groupby("station_id")
    denominator = grouped["actual"].sum()
    wape = grouped["abs_err"].sum() / denominator.where(denominator > 0)
    return 100.0 * (1.0 - wape).clip(lower=0.0, upper=1.0)


def official_accuracy(actual: pd.Series, predicted: pd.Series, station: pd.Series) -> float:
    per_station = station_accuracy(actual, predicted, station)
    return float(per_station.mean()) if len(per_station) else float("nan")


def wape(actual: pd.Series, predicted: pd.Series) -> float:
    total = float(np.sum(actual))
    if not total:
        return float("nan")
    return float(np.sum(np.abs(np.asarray(actual) - np.asarray(predicted))) / total)
