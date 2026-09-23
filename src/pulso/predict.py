"""Inferencia y entrega: descubre el ciclo, predice exactamente sus targets y envía.

Reglas del contrato que se cumplen aquí:
- El ciclo, su `data_cutoff` y sus targets vienen de la API; no se calculan a partir de la hora.
- Solo se usan observaciones con `observed_at <= data_cutoff`.
- Se envían todos los targets pedidos y solo esos, finitos y no negativos.
- La llave de idempotencia es estable dentro de una ejecución; los reintentos la reutilizan.
- Se deja evidencia (entrega + predicciones) ANTES de enviar y se guarda el recibo después.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from .api import PulsoApi, PulsoApiError
from .config import Settings
from .dbdata import load_wide
from .features import STEP
from .ingest import run_collector
from .model import GbmResidualModel
from .registry import ModelRegistry, RegistryError, current_git_commit
from .runlog import pipeline_run
from .store import SupabaseStore
from .supa import Supabase

log = logging.getLogger("pulso.predict")

# La API acepta 3 intentos VÁLIDOS por ciclo; los rechazados no cuentan. Como aquí se omite el ciclo
# en cuanto hay uno aceptado, este tope solo frena un bucle de fallos (la API limita a 10 req/min).
MAX_ROWS_PER_CYCLE = 10


class PredictionError(RuntimeError):
    pass


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise PredictionError(f"timestamp sin zona horaria en el ciclo: {value!r}")
    return ts.tz_convert("UTC")


@dataclass
class Submission:
    payload: dict[str, Any]
    predictions: pd.DataFrame  # station_id, target_at (texto del ciclo), horizon_minutes, value
    fallback_targets: int  # targets fuera de +15…+60 resueltos con el perfil puro


def build_predictions(model: GbmResidualModel, history: pd.DataFrame,
                      cycle: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    """Valor para cada target del ciclo, usando solo `history` (ya recortada al corte)."""
    cutoff = _utc(cycle["data_cutoff"])
    if history.empty or _utc(history.index[-1]) > cutoff:
        raise PredictionError("La historia incluye datos posteriores al data_cutoff")
    if _utc(history.index[-1]) < cutoff:
        # Falta la observación del propio corte: predict_next lo maneja (cae al perfil),
        # pero los pasos deben contarse desde el corte, no desde la última fila.
        pad = pd.date_range(history.index[-1] + STEP, cutoff.tz_convert(history.index.tz), freq=STEP)
        history = pd.concat([history, pd.DataFrame(np.nan, index=pad, columns=history.columns)])
    model_out = model.predict_next(history)
    by_key = {(r.station_id, _utc(r.target_at)): float(r.value) for r in model_out.itertuples()}

    rows, fallback = [], 0
    columns = {s: i for i, s in enumerate(model.stations)}
    for target in cycle["targets"]:
        station, target_at = str(target["station_id"]), _utc(target["target_at"])
        if station not in columns:
            raise PredictionError(f"El modelo no conoce la estación {station}")
        value = by_key.get((station, target_at))
        steps = int((target_at - cutoff) / STEP)
        if value is None:
            # Horizonte fuera de +15…+60: se cubre con el perfil (mejor que perder cobertura).
            if steps < 1:
                raise PredictionError(f"target anterior al corte: {target}")
            local = pd.DatetimeIndex([target_at]).tz_convert(history.index.tz)
            value = float(np.exp(model.profile.matrix(local)[0, columns[station]]))
            fallback += 1
        rows.append({"station_id": station, "target_at": target["target_at"],
                     "horizon_minutes": int(target.get("horizon_minutes",
                                                       steps * int(STEP.total_seconds() // 60))),
                     "value": round(max(0.0, value), 3)})
    frame = pd.DataFrame(rows)
    validate_submission(frame, cycle)
    return frame, fallback


def validate_submission(predictions: pd.DataFrame, cycle: dict[str, Any]) -> None:
    """Conjunto exacto de targets, sin duplicados, extras, NaN, infinitos ni negativos."""
    expected = {(str(t["station_id"]), _utc(t["target_at"])) for t in cycle["targets"]}
    got = [(r.station_id, _utc(r.target_at)) for r in predictions.itertuples()]
    if len(got) != len(set(got)):
        raise PredictionError("Hay targets duplicados")
    if set(got) != expected:
        missing, extra = expected - set(got), set(got) - expected
        raise PredictionError(f"Conjunto de targets distinto: faltan {len(missing)}, sobran {len(extra)}")
    values = predictions["value"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise PredictionError("Hay predicciones no finitas o negativas")
    if len(predictions) != cycle.get("expected_predictions", len(predictions)):
        raise PredictionError("La cantidad de predicciones no coincide con expected_predictions")


def build_payload(cycle: dict[str, Any], predictions: pd.DataFrame, *, model_version: str,
                  trained_at: str, training_data_end: str, git_commit: str | None,
                  client_run_id: str) -> dict[str, Any]:
    if _utc(training_data_end) > _utc(cycle["data_cutoff"]):
        raise PredictionError(
            "El modelo se entrenó con datos posteriores al data_cutoff del ciclo: "
            "la API rechazaría la entrega (training_data_end <= data_cutoff)")
    model_info: dict[str, Any] = {"version": model_version, "trained_at": trained_at,
                                  "training_data_end": training_data_end}
    if git_commit:
        model_info["git_commit"] = git_commit
    return {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": client_run_id,
        "data_cutoff": cycle["data_cutoff"],  # idéntico al del ciclo, sin reformatear
        "model": model_info,
        "predictions": [{"station_id": r.station_id, "target_at": r.target_at, "value": r.value}
                        for r in predictions.itertuples()],
    }


def payload_hash(payload: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def idempotency_key(cycle_id: str, model_version: str, predictions: pd.DataFrame) -> str:
    """Estable para el mismo ciclo, modelo y contenido (guía operativa v2.0).

    Antes dependía de GITHUB_RUN_ID, así que dos ejecuciones que entregaban lo MISMO creaban dos
    intentos válidos (gastando 2 de los 3 del ciclo) en vez de recibir el mismo recibo. Derivarla
    del contenido hace que un reintento —aunque sea de otra ejecución— devuelva 200 con el recibo
    original. Si cambia el modelo o cambian los valores, la llave cambia: es una entrega distinta.
    """
    body = json.dumps([[r.station_id, r.target_at, r.value] for r in predictions.itertuples()],
                      sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    return f"ptm-{cycle_id}-{model_version}-{digest}"[:128]


def client_run_id(now: datetime) -> str:
    run_id = os.getenv("GITHUB_RUN_ID")
    if run_id:
        return f"github-{run_id}-{os.getenv('GITHUB_RUN_ATTEMPT', '1')}"
    return f"manual-{now:%Y%m%dT%H%M%S}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def run_inference(api: PulsoApi, db: Supabase, registry: ModelRegistry, *,
                  dry_run: bool = False, now: datetime | None = None,
                  sync: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Un ciclo de inferencia. Devuelve un resumen; lanza si algo falla (el workflow debe fallar).

    `sync` (opcional) se ejecuta justo antes de leer la historia, una sola vez y solo cuando el
    ciclo está abierto y falta entregarlo: recolecta las observaciones más recientes para que el
    modelo prediga con los rezagos del corte y no con datos viejos.
    """
    cycle = api.current_cycle()
    if cycle is None:
        return {"status": "skipped", "reason": "no_open_cycle"}
    cycle_id = cycle["cycle_id"]
    now = now or datetime.now(UTC)
    if not dry_run:
        db.upsert("forecast_cycles", [{
            "cycle_id": cycle_id, "state": cycle.get("state", "open"),
            "origin_at": cycle["origin_at"], "data_cutoff": cycle["data_cutoff"],
            "opens_at": cycle.get("opens_at"), "closes_at": cycle["closes_at"],
            "expected_predictions": cycle["expected_predictions"], "targets": cycle["targets"],
        }], "cycle_id")
        previous = db.select("submissions", columns="id,status,submission_id",
                             filters={"cycle_id": f"eq.{cycle_id}"})
        if any(p["status"] == "accepted" for p in previous):
            return {"status": "skipped", "reason": "already_submitted", "cycle_id": cycle_id}
        if len(previous) >= MAX_ROWS_PER_CYCLE:
            return {"status": "skipped", "reason": "attempt_limit", "cycle_id": cycle_id}

    closes = _utc(cycle["closes_at"])
    if pd.Timestamp(now) >= closes:
        if not dry_run:
            db.update("forecast_cycles", {"outcome": "missed"}, {"cycle_id": f"eq.{cycle_id}"})
        return {"status": "skipped", "reason": "cycle_closed", "cycle_id": cycle_id}

    if sync is not None:
        sync()
    champion_row, model = registry.load_champion()
    history = load_wide(db, until=cycle["data_cutoff"])
    predictions, fallback = build_predictions(model, history, cycle)
    payload = build_payload(
        cycle, predictions, model_version=champion_row["version"],
        trained_at=champion_row["trained_at"], training_data_end=champion_row["training_data_end"],
        git_commit=champion_row.get("git_commit") or current_git_commit(),
        client_run_id=client_run_id(now),
    )
    summary = {"cycle_id": cycle_id, "model_version": champion_row["version"],
               "n_predictions": len(predictions), "fallback_targets": fallback,
               "payload_hash": payload_hash(payload)}
    if dry_run:
        return {"status": "dry_run", **summary}

    key = idempotency_key(cycle_id, champion_row["version"], predictions)
    fields = {
        "cycle_id": cycle_id, "client_run_id": payload["client_run_id"],
        "model_version": champion_row["version"], "status": "pending",
        "payload_hash": summary["payload_hash"], "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "git_commit": payload["model"].get("git_commit"), "updated_at": _now(),
    }
    # La llave es única en la base: si este contenido ya se intentó antes (p. ej. una ejecución que
    # murió tras el POST), se reutiliza esa fila en vez de duplicarla.
    previous_row = db.select("submissions", columns="id",
                             filters={"idempotency_key": f"eq.{key}"})
    if previous_row:
        row = previous_row[0]
        db.update("submissions", fields, {"id": f"eq.{row['id']}"})
    else:
        row = db.insert("submissions", {"idempotency_key": key, **fields})
    db.upsert("predictions", [
        {"submission_row_id": row["id"], "cycle_id": cycle_id, "station_id": r.station_id,
         "target_at": r.target_at, "horizon_minutes": int(r.horizon_minutes), "value": r.value}
        for r in predictions.itertuples()
    ], "submission_row_id,station_id,target_at")
    try:
        status, receipt = api.submit(payload, key)
    except PulsoApiError as exc:
        closed = exc.code == "cycle_closed"
        db.update("submissions", {"status": "error" if exc.status >= 500 else "rejected",
                                  "http_status": exc.status, "request_id": exc.request_id,
                                  "error": f"{exc.code}: {exc.message}"[:1000], "updated_at": _now()},
                  {"id": f"eq.{row['id']}"})
        db.update("forecast_cycles", {"outcome": "missed" if closed else "error"},
                  {"cycle_id": f"eq.{cycle_id}"})
        raise
    except Exception as exc:
        db.update("submissions", {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:1000],
                                  "updated_at": _now()}, {"id": f"eq.{row['id']}"})
        db.update("forecast_cycles", {"outcome": "error"}, {"cycle_id": f"eq.{cycle_id}"})
        raise

    db.update("submissions", {
        "status": "accepted", "attempt": receipt.get("attempt"),
        "submission_id": receipt.get("submission_id"), "is_official": receipt.get("is_official"),
        "http_status": status, "request_id": receipt.pop("_request_id", None),
        "receipt": receipt, "updated_at": _now(),
    }, {"id": f"eq.{row['id']}"})
    db.update("forecast_cycles", {"outcome": "submitted"}, {"cycle_id": f"eq.{cycle_id}"})
    return {"status": "submitted", "submission_id": receipt.get("submission_id"),
            "attempt": receipt.get("attempt"), "http": status, **summary}


def run_inference_waiting(api: PulsoApi, db: Supabase, registry: ModelRegistry, *,
                          wait_seconds: float, poll_seconds: float = 60.0,
                          dry_run: bool = False, sleep: Callable[[float], None] = time.sleep,
                          clock: Callable[[], float] = time.monotonic,
                          now_fn: Callable[[], datetime] | None = None,
                          sync: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Como `run_inference`, pero si no hay ciclo abierto espera a que abra uno.

    Los cron de GitHub Actions se retrasan y a menudo se saltan ejecuciones, así que no se puede
    depender de que un disparo caiga dentro de la ventana de 25 minutos. Esperando dentro del job,
    basta con que GitHub arranque UNA ejecución en la hora previa para cubrir el ciclo.

    Solo espera ante `no_open_cycle`: cualquier otro motivo (ya entregado, ventana cerrada) termina
    de inmediato, y un error se propaga como siempre.
    """
    now_fn = now_fn or (lambda: datetime.now(UTC))
    deadline = clock() + wait_seconds
    attempts = 0
    while True:
        # La hora se relee en cada intento: mientras se espera, la ventana puede abrir o cerrar.
        result = run_inference(api, db, registry, dry_run=dry_run, now=now_fn(), sync=sync)
        attempts += 1
        if result.get("reason") != "no_open_cycle" or clock() >= deadline:
            if attempts > 1:
                result["waited_polls"] = attempts
            return result
        sleep(min(poll_seconds, max(0.0, deadline - clock())))


def run_session(api: PulsoApi, db: Supabase, registry: ModelRegistry, *,
                duration_seconds: float, poll_seconds: float = 60.0, dry_run: bool = False,
                sync: Callable[[], Any] | None = None, max_consecutive_errors: int = 5,
                sleep: Callable[[float], None] = time.sleep,
                clock: Callable[[], float] = time.monotonic,
                now_fn: Callable[[], datetime] | None = None) -> dict[str, Any]:
    """Cubre TODOS los ciclos que abran durante `duration_seconds`, no solo el primero.

    GitHub llegó a estar 5 h sin disparar ninguna ejecución programada, así que esperar un único
    ciclo no basta: un job cubre varias horas y va entregando cada ciclo que aparece. Los errores
    transitorios no cortan la sesión (se reintenta en el siguiente sondeo), pero varios seguidos sí
    la terminan para que el workflow falle a la vista.
    """
    deadline = clock() + duration_seconds
    submitted: list[str] = []
    errors = 0
    last_error: Exception | None = None
    while clock() < deadline:
        try:
            result = run_inference_waiting(
                api, db, registry, wait_seconds=deadline - clock(), poll_seconds=poll_seconds,
                dry_run=dry_run, sync=sync, sleep=sleep, clock=clock, now_fn=now_fn,
            )
            errors = 0
            if result["status"] in ("submitted", "dry_run"):
                submitted.append(result["cycle_id"])
                log.info("Ciclo entregado: %s", result)
        except Exception as exc:  # noqa: BLE001 - una falla puntual no debe cortar horas de cobertura
            errors += 1
            last_error = exc
            log.warning("Fallo en la sesión (%d seguidos): %s", errors, exc)
            if errors >= max_consecutive_errors:
                raise
        if clock() < deadline:  # espera a que abra el ciclo siguiente
            sleep(min(poll_seconds, max(0.0, deadline - clock())))
    summary = {"status": "session", "cycles_submitted": len(submitted), "cycles": submitted}
    if last_error is not None:
        summary["last_error"] = f"{type(last_error).__name__}: {last_error}"
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predice el ciclo abierto y envía la entrega")
    parser.add_argument("--dry-run", action="store_true", help="construye y valida sin enviar")
    parser.add_argument("--wait-for-cycle", type=float, default=0.0, metavar="MINUTOS",
                        help="si no hay ciclo abierto, espera hasta N minutos a que abra uno")
    parser.add_argument("--run-for", type=float, default=0.0, metavar="MINUTOS",
                        help="mantiene la sesión N minutos entregando cada ciclo que abra")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = Settings.from_env()
    url, key = settings.require_supabase()
    db = Supabase(url, key)
    try:
        with pipeline_run(db, "predict") as run, \
                PulsoApi(settings.api_url, settings.require_api_key()) as api:
            store = SupabaseStore(db)
            # Sincroniza en el último momento: entre despertar y entregar pueden pasar 30+
            # minutos de espera y la API libera datos nuevos cada 30 min.
            sync = lambda: run_collector(api, store)  # noqa: E731
            registry_ = ModelRegistry(db)
            if args.run_for > 0:
                result = run_session(api, db, registry_, duration_seconds=args.run_for * 60,
                                     dry_run=args.dry_run, sync=sync)
            else:
                result = run_inference_waiting(
                    api, db, registry_, wait_seconds=args.wait_for_cycle * 60,
                    dry_run=args.dry_run, sync=sync,
                )
            run.summary.update(result)
            if result["status"] == "skipped":
                run.skip(result["reason"])
            log.info("Inferencia: %s", result)
    except RegistryError as exc:
        log.error("%s", exc)
        return 1
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
