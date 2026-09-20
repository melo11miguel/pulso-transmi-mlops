"""Simulador de drift para pruebas de estrés (no forma parte del pipeline de producción).

El histórico oficial no tiene drift, así que no hay forma de saber cómo reaccionará un modelo.
Estas funciones aplican cambios sintéticos que imitan los tipos de drift publicados
(`level_shift`, `peak_shift`, `trend_change`, `closure`) sobre la rejilla de demanda.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ramp(index: pd.DatetimeIndex, start: pd.Timestamp, ramp_hours: float) -> np.ndarray:
    """0 antes de `start`, sigmoide hasta 1 en `ramp_hours` (transición gradual)."""
    if ramp_hours <= 0:
        return (index >= start).astype(float)
    x = (index - start) / pd.Timedelta(hours=ramp_hours)
    x = np.asarray(x, dtype=float)
    return np.where(x <= 0, 0.0, 1.0 / (1.0 + np.exp(-12.0 * (np.clip(x, 0, 1) - 0.5))))


def apply_drift(wide: pd.DataFrame, kind: str, stations: list[str], start: pd.Timestamp,
                magnitude: float = 0.25, ramp_hours: float = 12.0) -> pd.DataFrame:
    """Devuelve una copia de `wide` con drift a partir de `start` en `stations`.

    - level_shift: demanda × (1 + magnitude)
    - trend_change: demanda × (1 + magnitude × días desde start)  (pendiente diaria)
    - peak_shift: el patrón se desplaza `magnitude` horas más tarde (entero de periodos de 15 min)
    - closure: la estación cae a (1 - magnitude) de su nivel
    """
    out = wide.copy()
    w = _ramp(wide.index, start, ramp_hours)
    for station in stations:
        series = wide[station].to_numpy(dtype=float)
        if kind == "level_shift":
            new = series * (1.0 + magnitude)
        elif kind == "closure":
            new = series * (1.0 - magnitude)
        elif kind == "trend_change":
            days = np.clip((wide.index - start) / pd.Timedelta(days=1), 0, None)
            new = series * (1.0 + magnitude * np.asarray(days, dtype=float))
        elif kind == "peak_shift":
            steps = int(round(magnitude * 4))
            new = np.roll(series, steps)
            new[:steps] = series[:steps]
        else:
            raise ValueError(f"drift desconocido: {kind}")
        out[station] = np.round((1 - w) * series + w * new)
    return out


SCENARIOS = {
    "nivel +25 % (4 estaciones)": dict(kind="level_shift", magnitude=0.25,
                                        stations=["02300", "05100", "07111", "10009"]),
    "pico +1 h (todas las de grupo B)": dict(kind="peak_shift", magnitude=1.0,
                                             stations=["03000", "05000", "05100", "06000", "07111", "09000"]),
    "cierre parcial −60 % (1 estación)": dict(kind="closure", magnitude=0.6, stations=["07111"]),
    "tendencia +8 %/día (3 estaciones)": dict(kind="trend_change", magnitude=0.08,
                                              stations=["03000", "07105", "09122"]),
}
