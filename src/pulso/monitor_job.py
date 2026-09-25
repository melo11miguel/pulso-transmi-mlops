"""Trabajo de monitoreo (Fase 5): evalúa, detecta degradación y decide qué hacer.

Cada ejecución (cada 30 min en producción):
1. une las predicciones oficiales con los valores reales que ya recolectó el collector;
2. calcula accuracy móvil de 24 h y acumulada, junto con la del baseline sobre los mismos targets;
3. mide el desplazamiento de nivel por estación y su persistencia;
4. junta las señales operacionales (retraso del collector, cobertura, ejecuciones fallidas);
5. aplica `policy.decide_retrain`, guarda la decisión con su evidencia y, si corresponde, avisa
   al workflow para que dispare el entrenamiento.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from .api import PulsoApi, PulsoApiError
from .config import Settings
from .dbdata import load_wide
from .monitor import (
    accuracy_by_station,
    baseline_accuracy,
    level_thresholds,
    rolling_accuracy,
    station_level_shift,
    update_streaks,
)
from .policy import MonitorState, RetrainRules, decide_retrain
from .registry import ModelRegistry
from .runlog import pipeline_run
from .supa import Supabase

log = logging.getLogger("pulso.monitor")

ALL_HOURS = 24.0 * 365 * 3  # «toda la historia» para la ventana acumulada


def _num(value: float, digits: int | None = None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits) if digits is not None else float(value)


def load_scored(db: Supabase, model_version: str | None = None) -> pd.DataFrame:
    """Predicciones oficiales con su valor real (cuando ya existe), listas para evaluar."""
    rows: list[dict] = []
    for chunk in db.select_pages(
        "prediction_results",
        columns="cycle_id,model_version,station_id,target_at,horizon_minutes,predicted,actual",
        filters={"is_official": "eq.true"}, order="target_at.asc,station_id.asc",
    ):
        rows.extend(chunk)
    frame = pd.DataFrame(rows, columns=["cycle_id", "model_version", "station_id", "target_at",
                                        "horizon_minutes", "predicted", "actual"])
    if frame.empty:
        return frame.assign(prediction=pd.Series(dtype=float))
    frame["target_at"] = pd.to_datetime(frame["target_at"], utc=True)
    frame["station_id"] = frame["station_id"].astype(str)
    frame["prediction"] = frame["predicted"].astype(float)
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    if model_version:
        frame = frame[frame["model_version"] == model_version]
    return frame


def _previous_streaks(db: Supabase, stations: list[str]) -> dict[str, int]:
    rows = db.select("drift_signals", columns="station_id,details,computed_at",
                     filters={"name": "eq.level_shift"}, order="computed_at.desc",
                     limit=max(60, len(stations) * 5))
    streaks: dict[str, int] = {}
    for row in rows:
        station = row["station_id"]
        if station and station not in streaks:
            streaks[station] = int((row.get("details") or {}).get("streak", 0))
    return streaks


def _recent_accuracies(db: Supabase, version: str, n: int) -> list[float]:
    rows = db.select("metric_snapshots", columns="accuracy,computed_at",
                     filters={"window_name": "eq.rolling_24h", "model_version": f"eq.{version}"},
                     order="computed_at.desc", limit=n)
    return [float(r["accuracy"]) for r in reversed(rows) if r["accuracy"] is not None]


def _operational(db: Supabase, api: PulsoApi | None, data_now: pd.Timestamp,
                 now: datetime) -> dict[str, Any]:
    """Señales de salud del pipeline: retraso del collector, cobertura de entregas y fallos."""
    lag = None
    if api is not None:
        try:
            clock = api.clock()
            if clock.get("state") not in (None, "waiting"):  # sin competencia no aplica
                reference = pd.Timestamp(clock.get("virtual_now") or clock["server_time"])
                lag = (reference - data_now) / pd.Timedelta(minutes=1)
        except (PulsoApiError, KeyError, ValueError):
            lag = None
    cycles = db.select("forecast_cycles", columns="outcome,closes_at")
    closed = [c for c in cycles if pd.Timestamp(c["closes_at"]) <= pd.Timestamp(now)]
    coverage = sum(c["outcome"] == "submitted" for c in closed) / len(closed) if closed else None
    day_ago = (pd.Timestamp(now) - pd.Timedelta(hours=24)).isoformat()
    failed = (
        len(db.select("pipeline_runs", columns="status", filters={
            "status": "eq.error", "started_at": f"gte.{day_ago}"}))
        + len(db.select("ingestion_runs", columns="status", filters={
            "status": "eq.error", "started_at": f"gte.{day_ago}"}))
    )
    return {"collector_lag_minutes": lag, "coverage": coverage, "failed_runs_24h": failed}


def capture_leaderboard(db: Supabase, api: PulsoApi) -> dict[str, Any]:
    """Guarda nuestra posición en `leaderboard_snapshots` para que el dashboard no use la API key.

    Solo se persiste lo nuestro y el mejor accuracy de la cohorte como referencia numérica: los
    nombres de los demás participantes no son nuestros para publicarlos.
    """
    try:
        yo = api.me().get("display_name")
    except (PulsoApiError, KeyError):
        return {}
    capturado = {}
    for window in ("cumulative", "rolling_24h"):
        try:
            rows = api.leaderboard(window).get("data") or []
        except PulsoApiError:
            continue
        mio = next((r for r in rows if r.get("display_name") == yo), None)
        mejor = rows[0] if rows else None
        if mio is None:
            continue
        db.insert("leaderboard_snapshots", {
            "window_name": window, "rank": mio.get("rank"), "participants": len(rows),
            "accuracy": mio.get("accuracy"), "coverage": mio.get("coverage"),
            "best_accuracy": (mejor or {}).get("accuracy"),
            "best_coverage": (mejor or {}).get("coverage"),
        })
        capturado[window] = {"puesto": mio.get("rank"), "accuracy": mio.get("accuracy")}
    return capturado


def run_monitor(db: Supabase, registry: ModelRegistry, api: PulsoApi | None = None, *,
                now: datetime | None = None, rules: RetrainRules = RetrainRules()) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    champion_row, model = registry.load_champion()
    version = champion_row["version"]
    wide = load_wide(db)
    data_now = wide.index[-1]
    thresholds = level_thresholds(model.level_noise, rules.level_drift_floor,
                                  rules.level_noise_multiplier)

    ops = _operational(db, api, data_now, now)

    # ---- desempeño: predicciones oficiales vs realidad
    scored = load_scored(db)
    resolved = scored.dropna(subset=["actual"]) if not scored.empty else scored
    accuracy24 = baseline24 = float("nan")
    if not resolved.empty:
        window_end = resolved["target_at"].max()
        windows = {
            "rolling_24h": (window_end - pd.Timedelta(hours=24), 24.0),
            "cumulative": (resolved["target_at"].min(), ALL_HOURS),
        }
        by_horizon = {}
        for horizon, part in resolved.groupby("horizon_minutes"):
            value, _ = rolling_accuracy(part, window_end, hours=ALL_HOURS, min_stations=1)
            by_horizon[str(int(horizon))] = _num(value, 2)
        for name, (start_at, hours) in windows.items():
            acc, n = rolling_accuracy(resolved, window_end, hours=hours,
                                      min_stations=6 if name == "rolling_24h" else 1)
            base = baseline_accuracy(model.profile, wide, resolved, window_end, hours=hours)
            if name == "rolling_24h":
                accuracy24, baseline24 = acc, base
            db.insert("metric_snapshots", {
                "window_name": name, "model_version": version,
                "window_start": start_at.isoformat(), "window_end": window_end.isoformat(),
                "accuracy": _num(acc), "wape": None if np.isnan(acc) else float(1 - acc / 100),
                "coverage": ops["coverage"], "n_targets": int(len(scored)), "n_resolved": int(n),
                "baseline_accuracy": _num(base),
                "by_station": accuracy_by_station(resolved, window_end, hours),
                "by_horizon": by_horizon,
            })

    # ---- datos: desplazamiento de nivel por estación y persistencia
    shifts = station_level_shift(model.profile, wide, data_now, hours=24)
    stations = [str(s) for s in wide.columns]
    streaks = update_streaks(_previous_streaks(db, stations), shifts, thresholds)
    for station in stations:
        shift = shifts[station]
        db.insert("drift_signals", {
            "kind": "data", "name": "level_shift", "station_id": station,
            "value": None if not np.isfinite(shift) else float(shift),
            "threshold": float(thresholds[station]),
            "triggered": streaks[station] >= rules.data_persistence,
            "details": {"streak": streaks[station], "window_hours": 24},
        })

    # ---- decisión
    reference = (champion_row.get("validation") or {}).get("accuracy")
    history = _recent_accuracies(db, version, rules.persistence)
    # El enfriamiento debe medirse desde el ULTIMO INTENTO de entrenamiento, no desde el
    # entrenamiento del champion. Con la puerta honesta, un refresco de pura cadencia da ganancia
    # ~0 y se rechaza, asi que el champion no cambia; anclando el contador a el, nunca se reinicia
    # y el enfriamiento no entra jamas. Medido: 22 entrenamientos en 10 horas, uno cada vuelta del
    # monitor, todos rechazados por la misma razon.
    ultimo_intento = db.select("model_versions", columns="created_at",
                               order="created_at.desc", limit=1)
    ancla = (pd.Timestamp(ultimo_intento[0]["created_at"]) if ultimo_intento
             else pd.Timestamp(champion_row["trained_at"]))
    state = MonitorState(
        reference_accuracy=None if reference is None else float(reference),
        rolling_accuracies=history,
        level_drift_streaks=streaks,
        hours_since_training=(pd.Timestamp(now) - ancla) / pd.Timedelta(hours=1),
        new_data_hours=max(0.0, (data_now - pd.Timestamp(champion_row["training_data_end"])
                                 ) / pd.Timedelta(hours=1)),
        **ops,
    )
    leaderboard = capture_leaderboard(db, api) if api is not None else {}

    decision = decide_retrain(state, rules)
    reference_acc = state.reference_accuracy
    limit = None if reference_acc is None else reference_acc - rules.performance_drop_pts
    db.insert("drift_signals", {
        "kind": "performance", "name": "rolling_24h_accuracy", "station_id": None,
        "value": _num(accuracy24), "threshold": limit,
        "triggered": decision.signals["performance"],
        "details": {"recent": history[-rules.persistence:], "reference": state.reference_accuracy},
    })
    db.insert("drift_signals", {
        "kind": "operational", "name": "pipeline_health", "station_id": None,
        "value": None, "threshold": None, "triggered": decision.signals["operational"],
        "details": {k: (None if v is None else float(v)) for k, v in ops.items()},
    })
    evidence = {
        "signals": decision.signals, "rolling_24h_accuracy": _num(accuracy24, 2),
        "baseline_24h_accuracy": _num(baseline24, 2),
        "reference_accuracy": state.reference_accuracy,
        "hours_since_training": round(state.hours_since_training, 1),
        "new_data_hours": round(state.new_data_hours, 1),
        "level_streaks": {s: n for s, n in streaks.items() if n},
        **{k: (None if v is None else round(float(v), 3)) for k, v in ops.items()},
    }
    db.insert("retrain_decisions", {
        "decision": decision.decision, "reason": decision.reason, "evidence": evidence,
        "model_version": version, "github_run_id": os.getenv("GITHUB_RUN_ID"),
    })
    return {"decision": decision.decision, "reason": decision.reason, "signals": decision.signals,
            "model_version": version, "data_now": data_now.isoformat(),
            "leaderboard": leaderboard, **evidence}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Evalúa el desempeño y decide si reentrenar").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings.from_env()
    url, key = settings.require_supabase()
    db = Supabase(url, key)
    try:
        with pipeline_run(db, "monitor") as run, PulsoApi(settings.api_url, settings.api_key) as api:
            result = run_monitor(db, ModelRegistry(db), api)
            run.summary.update({k: result[k] for k in ("decision", "reason", "signals", "model_version")})
            log.info("Monitoreo: %s", result)
            output = os.getenv("GITHUB_OUTPUT")
            if output:  # el workflow dispara el entrenamiento si la decisión es reentrenar
                with open(output, "a", encoding="utf-8") as fh:
                    fh.write(f"decision={result['decision']}\n")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
