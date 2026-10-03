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
from .features import STEP, Profile
from .ingest import run_collector
from .model import (
    GbmResidualModel,
    IncompatibleArtifactError,
    UnusableArtifactError,
)
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


# Correccion de nivel. El GBM no absorbe del todo el desplazamiento reciente del nivel: el
# residual medio de la ultima hora contra el perfil predice el sesgo del ciclo siguiente con
# pendiente +0,401 (SE robusto por ciclo 0,048, t=8,3, IC95 [0,300, 0,501], G=20 conglomerados).
#
# Se aplica de forma ADITIVA EN LOG, asi que la prediccion corregida sigue siendo la mediana
# posterior predictiva, que es lo que minimiza WAPE.
#
# Decisiones tomadas con validacion temporal, ajustando en los ciclos anteriores y evaluando en
# los posteriores:
#   - un coeficiente unico le gana a uno por horizonte (+0,965 contra +0,882), pese a que la
#     persistencia si decae con el horizonte (+0,519 a +15 min, +0,251 a +60);
#   - la componente de ciudad no aporta nada (coeficiente -0,024); la senal es de estacion;
#   - aplicarla solo en las franjas donde gana es sobreajuste: la compuerta decidida en el pasado
#     elige mal y hunde la ganancia de +0,965 a +0,057.
# APAGADA. Medida tres veces en el camino real y resta las tres: -1,0 con modelo fresco en el
# regimen de septiembre, -2,4 aplicando la fase, y -3,0 en el regimen nuevo. El GBM ya absorbe el
# nivel reciente con sus variables de residual y esta correccion lo cuenta dos veces. Se conserva
# el codigo porque con un modelo MUY rancio si ayudo (+3,61), y es la red de seguridad si el
# refresco se rompe: por eso el coeficiente es 0 y no se borra el mecanismo.
COEF_NIVEL = 0.0
# La correccion solo sirve cuando el modelo NO tiene el nivel reciente en sus datos. Medido en el
# camino real: con el champion rancio (3 dias virtuales) suma +3,61; con un modelo fresco RESTA
# entre 1,0 y 1,7, porque cuenta dos veces un nivel que el modelo ya aprendio. Por eso se escala
# por rancidez: nula recien entrenado, plena a partir de HORAS_RANCIO_PLENO.
HORAS_RANCIO_SIN = 6.0     # por debajo de esto el modelo esta al dia y corregir resta
HORAS_RANCIO_PLENO = 24.0  # a partir de aqui la rancidez ya cuesta puntos de verdad
TOPE_NIVEL = 0.25   # ~28 % de correccion maxima, por si una rafaga se lee mal
VENTANA_NIVEL = 4   # ultimas 4 filas observadas = 1 hora


# Desfase de fase. El evento del 09-13 no solo movio el nivel: en cuatro estaciones corrio el PICO
# 45 minutos manteniendo la forma del dia intacta (correlaciones de 0,98 a 0,995 entre el perfil
# reciente y el historico). Una correccion de nivel no puede arreglar un pico que se movio en el
# tiempo, por mucho coeficiente que se le ponga.
#
# Se DETECTA y se registra en el log, pero NO se corrige. Medido en el camino real con champion
# honesto: aplicar el desplazamiento completo del perfil sobre la salida del modelo da -2,44,
# porque el GBM ya absorbe parte del desfase con sus variables de residual reciente y la
# correccion lo cuenta dos veces. Es el mismo patron que la correccion de nivel sobre un modelo
# fresco. El fenomeno es real (corregir el PERFIL da +3,42), pero corregir la SALIDA no lo es.
DESFASES = range(-4, 5)          # de -1 h a +1 h
DIAS_FASE = 3                    # ventana reciente para estimar la fase
CORRELACION_MINIMA = 0.90        # sin buen ajuste de forma, no se mueve nada
DESFASE_MINIMO = 2               # menos de 30 min es ruido


def desfase_por_estacion(model: GbmResidualModel, history: pd.DataFrame) -> dict[str, int]:
    """Desfase en franjas de 15 min por estacion, por correlacion cruzada del perfil diario.

    Solo mira `history`, que ya viene recortada al data_cutoff del ciclo.
    """
    if len(history) < 96 * (DIAS_FASE + 7):
        return {}
    corte = history.index[-1] - pd.Timedelta(days=DIAS_FASE)
    reciente, viejo = history.loc[corte:], history.loc[:corte]
    if len(reciente) < 96 or len(viejo) < 96 * 7:
        return {}

    def perfil_dia(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(df.index.hour * 4 + df.index.minute // 15).mean()

    pv, pr = perfil_dia(viejo), perfil_dia(reciente)
    out: dict[str, int] = {}
    for col in history.columns:
        x, y = pv[col].to_numpy(dtype=float), pr[col].to_numpy(dtype=float)
        if len(x) != 96 or len(y) != 96 or not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        xn = (x - x.mean()) / (x.std() + 1e-9)
        yn = (y - y.mean()) / (y.std() + 1e-9)
        mejor, r_max = 0, -9.0
        for k in DESFASES:
            r = float(np.corrcoef(xn, np.roll(yn, k))[0, 1])
            if r > r_max:
                mejor, r_max = k, r
        if r_max >= CORRELACION_MINIMA and abs(mejor) >= DESFASE_MINIMO:
            out[str(col)] = mejor
    return out


def ajuste_de_fase(model: GbmResidualModel, station: str, cuando: pd.Timestamp,
                   desfases: dict[str, int], tz) -> float:
    """Cuanto multiplicar la prediccion para corregir el desfase de esa estacion."""
    k = desfases.get(station, 0)
    if k == 0:
        return 1.0
    j = model.stations.index(station)
    local = pd.DatetimeIndex([cuando]).tz_convert(tz)
    corrido = local + k * STEP
    p0 = model.profile.matrix(local)[0, j]
    p1 = model.profile.matrix(corrido)[0, j]
    if not (np.isfinite(p0) and np.isfinite(p1)):
        return 1.0
    return float(np.exp(np.clip(p1 - p0, -TOPE_NIVEL * 3, TOPE_NIVEL * 3)))


def factor_rancidez(model: GbmResidualModel, corte: pd.Timestamp) -> float:
    """0 si el modelo se entreno con datos hasta el corte, 1 si lleva HORAS_RANCIO_PLENO atras."""
    fin = getattr(model, "train_end", None)
    if fin is None:
        return 1.0
    horas = (pd.Timestamp(corte) - pd.Timestamp(fin)) / pd.Timedelta(hours=1)
    # Rampa que arranca a las HORAS_RANCIO_SIN: medido, hasta las 12 h el modelo rinde igual que
    # recien entrenado (83,4 y 83,2 contra 83,1), asi que corregir ahi solo mete ruido. El
    # despeñadero esta en las 24 h (80,4) y en las 48 (75,1).
    tramo = max(HORAS_RANCIO_PLENO - HORAS_RANCIO_SIN, 1e-9)
    return float(np.clip((horas - HORAS_RANCIO_SIN) / tramo, 0.0, 1.0))


def nivel_reciente(model: GbmResidualModel, history: pd.DataFrame,
                   ventana: int = VENTANA_NIVEL) -> dict[str, float]:
    """Residual medio en log contra el perfil, por estacion, en la ultima hora CON datos.

    Se descartan las filas de relleno del final: cuando falta la observacion del corte,
    `build_predictions` rellena con NaN y la cola quedaria vacia.
    """
    cola = history.dropna(how="all").tail(ventana)
    if cola.empty:
        return {}
    perfil = model.profile.matrix(cola.index)
    real = cola.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        residual = np.log(np.where(real > 0, real, np.nan)) - perfil
    with np.errstate(all="ignore"):
        medio = np.nanmean(residual, axis=0)
    return {str(c): float(v) for c, v in zip(cola.columns, medio, strict=False)
            if np.isfinite(v)}


class PerfilDeEmergencia:
    """Pronosticador de ultimo recurso, ajustado aqui con el codigo que corre ahora mismo.

    Existe por una regla del reto: un target que no se entrega cuenta como prediccion CERO, asi
    que perder un ciclo cuesta mucho mas que entregarlo con el modelo peor. Cuando el artefacto del
    champion no se puede usar, el respaldo del perfil que ya tiene `build_predictions` tampoco
    sirve, porque lee `model.profile`, que es parte del artefacto roto. Este objeto reconstruye un
    perfil estacional desde la historia que acabamos de leer, sin tocar el artefacto, y expone lo
    justo que `build_predictions` necesita: `stations`, `profile` y un `predict_next` que declara
    la incompatibilidad para que el camino entre degradado.

    Vale aproximadamente lo que el baseline de solo perfil, unos 51 de accuracy frente a los ~57
    del modelo, contra 0 de un ciclo perdido.
    """

    def __init__(self, history: pd.DataFrame, motivo: str) -> None:
        self.motivo = motivo
        self.stations = [str(c) for c in history.columns]
        valores = history.to_numpy(dtype=float)
        self.profile = Profile().fit(history.index, np.log(np.where(valores > 0, valores, np.nan)))
        self.train_end = history.index[-1]

    def predict_next(self, history: pd.DataFrame) -> pd.DataFrame:
        raise UnusableArtifactError(self.motivo)


def cargar_champion(registry: ModelRegistry, history: pd.DataFrame) -> tuple[dict, Any]:
    """Fila del champion y su modelo, cayendo al perfil de emergencia si el artefacto no sirve.

    La fila se lee de la base (barato y sin pickle) y solo la descarga del artefacto puede fallar.
    Separarlas es el punto: antes `load_champion` hacia las dos cosas y su excepcion escapaba del
    guard de degradacion, se contaba como error de sesion y a los 5 seguidos mataba la corrida.
    """
    row = registry.champion()
    if row is None:
        raise RegistryError("No hay modelo champion: ejecute el workflow de entrenamiento")
    try:
        return row, registry.load(row["version"])
    except Exception as exc:  # noqa: BLE001 - cualquier fallo del artefacto degrada, no mata
        log.error("El artefacto del champion %s no se puede usar con este codigo (%s: %s); "
                  "se entrega el perfil estacional reconstruido aqui", row["version"],
                  type(exc).__name__, exc)
        return row, PerfilDeEmergencia(history, f"{type(exc).__name__}: {exc}")


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
    try:
        model_out = model.predict_next(history)
    except IncompatibleArtifactError as exc:
        # El artefacto es más nuevo que este proceso. Reintentar no lo arregla, y quedarse sin
        # entregar es lo peor que puede pasar: un target ausente cuenta como predicción cero y
        # hunde la cobertura. Se deja `by_key` vacío a propósito para que cada target caiga por el
        # respaldo de abajo, que usa el perfil estacional del propio artefacto y no necesita
        # ninguna variable construida. Queda contado en `fallback_targets`, que ya vigila el monitor.
        faltan = ", ".join(getattr(exc, "missing", None) or [str(exc)])
        log.error("Modelo incompatible con este código (%s): se entrega el perfil estacional", faltan)
        model_out = None
    degradado = model_out is None
    by_key = ({} if degradado
              else {(r.station_id, _utc(r.target_at)): float(r.value)
                    for r in model_out.itertuples()})
    niveles = {} if degradado else nivel_reciente(model, history)
    rancidez = 0.0 if degradado else factor_rancidez(model, cutoff)
    fases = {} if degradado else desfase_por_estacion(model, history)
    if fases:
        log.info('Desfase de fase detectado: %s', fases)
    if niveles and rancidez < 1.0:
        log.info("Correccion de nivel al %.0f %% (el modelo tiene datos hasta %s)",
                 100 * rancidez, getattr(model, "train_end", "?"))

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
        # Solo se corrige lo que salio del modelo: los targets que cayeron al perfil por estar
        # fuera de +15..+60 ya son el perfil, y corregirlos seria aplicarle el residual a si mismo.
        if not degradado and by_key.get((station, target_at)) is not None:
            if station in niveles:
                # Escalada por rancidez: con el modelo al dia la correccion resta, porque cuenta
                # dos veces un nivel que el ya aprendio.
                value = value * float(np.exp(np.clip(COEF_NIVEL * rancidez * niveles[station],
                                                     -TOPE_NIVEL, TOPE_NIVEL)))

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
    history = load_wide(db, until=cycle["data_cutoff"])
    champion_row, model = cargar_champion(registry, history)
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
