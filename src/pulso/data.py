"""Carga de datos locales (CSV oficiales) con tipos correctos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TZ = "America/Bogota"  # UTC-5 todo el año, sin horario de verano
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_stations(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "stations.csv", dtype={"station_id": "string"})


def load_observations(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Demanda por estación y periodo de 15 min, ordenada, con hora local."""
    frame = pd.read_csv(data_dir / "observations.csv", dtype={"station_id": "string"})
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True).dt.tz_convert(TZ)
    return frame.sort_values(["station_id", "observed_at"], ignore_index=True)


def load_context(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / "context.csv")
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True).dt.tz_convert(TZ)
    return frame.sort_values("observed_at", ignore_index=True)
