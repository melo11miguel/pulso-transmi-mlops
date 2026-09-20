"""Entrenamiento de un candidato y decisión de promoción.

Flujo: entrena con todos los datos → valida con un «gemelo» entrenado hasta 7 días antes y
evaluado en esos 7 días (validación temporal) → compara con el baseline de perfil y con el champion
sobre la MISMA ventana → prueba de humo de inferencia → registra → promueve o rechaza.

Nota de justicia: el champion puede haber visto parte de la ventana de validación (si se entrenó
después de su inicio). Eso lo favorece, así que la regla es conservadora con los candidatos.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .backtest import Fold, hourly_origins, score_predictions, summarize
from .baselines import ProfileMean
from .config import Settings
from .dbdata import load_wide
from .features import HORIZONS, STEP
from .model import GbmResidualModel, ModelConfig
from .policy import PromotionRules, decide_promotion
from .registry import ModelRegistry, current_git_commit
from .runlog import pipeline_run
from .supa import Supabase

log = logging.getLogger("pulso.train")

HOLDOUT_DAYS = 7
MIN_DAYS = 21  # con menos historia el perfil semanal no es fiable


class TrainingError(RuntimeError):
    pass


@dataclass
class TrainResult:
    version: str
    promoted: bool
    reason: str
    candidate_accuracy: float
    baseline_accuracy: float
    champion_accuracy: float | None
    holdout_start: str
    holdout_end: str
    n_targets: int


def holdout_fold(wide: pd.DataFrame, days: int = HOLDOUT_DAYS) -> Fold:
    end = wide.index[-1]
    start = end - pd.Timedelta(days=days) + STEP
    return Fold("holdout", start - STEP, start, end)


def evaluate(model, wide: pd.DataFrame, fold: Fold) -> dict:
    origins = hourly_origins(wide.index, fold)
    if len(origins) == 0:
        raise TrainingError("La ventana de validación no tiene orígenes")
    return summarize(score_predictions(wide, model.predict_batch(wide, origins)))


def smoke_test(model: GbmResidualModel, wide: pd.DataFrame) -> tuple[bool, str]:
    """El modelo debe producir un pronóstico completo, finito y no negativo con la historia real."""
    try:
        out = model.predict_next(wide)
    except Exception as exc:  # noqa: BLE001 - cualquier falla cuenta como prueba fallida
        return False, f"{type(exc).__name__}: {exc}"
    expected = len(model.stations) * len(HORIZONS)
    if len(out) != expected:
        return False, f"esperaba {expected} predicciones y obtuvo {len(out)}"
    values = out["value"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        return False, "predicciones no finitas o negativas"
    return True, "ok"


def train_and_register(wide: pd.DataFrame, registry: ModelRegistry, *,
                       config: ModelConfig | None = None, reason: str = "manual",
                       git_commit: str | None = None,
                       rules: PromotionRules = PromotionRules()) -> TrainResult:
    config = config or ModelConfig()
    n_days = len(wide) / 96
    if n_days < MIN_DAYS:
        raise TrainingError(f"Solo hay {n_days:.1f} días de datos (mínimo {MIN_DAYS})")

    fold = holdout_fold(wide)
    train_part = wide.loc[:fold.train_end]
    twin = GbmResidualModel(config).fit(train_part)  # mismo config, sin ver la ventana
    candidate_val = evaluate(twin, wide, fold)
    baseline_val = evaluate(ProfileMean().fit(train_part), wide, fold)

    champion_row = registry.champion()
    champion_acc, parent = None, None
    if champion_row:
        parent = champion_row["version"]
        try:
            champion = registry.load(parent)
            champion_acc = evaluate(champion, wide, fold)["accuracy"]
        except Exception:  # noqa: BLE001 - un champion ilegible no debe bloquear el reemplazo
            log.exception("No se pudo evaluar al champion %s; se trata como ausente", parent)

    final = GbmResidualModel(config).fit(wide)
    smoke_ok, smoke_msg = smoke_test(final, wide)
    promote, why = decide_promotion(candidate_val["accuracy"], champion_acc,
                                    baseline_val["accuracy"], smoke_ok, rules)
    validation = {
        "holdout_start": fold.val_start.isoformat(), "holdout_end": fold.val_end.isoformat(),
        "accuracy": candidate_val["accuracy"], "wape": candidate_val["wape"],
        "by_horizon": candidate_val["by_horizon"], "by_station": candidate_val["by_station"],
        "n_targets": candidate_val["n_targets"],
        "baseline_profile_only_accuracy": baseline_val["accuracy"],
        "champion_accuracy_same_window": champion_acc, "smoke_test": smoke_msg,
        "decision": "promote" if promote else "reject", "decision_reason": why,
    }
    row = registry.register_candidate(final, validation=validation, git_commit=git_commit,
                                      reason=f"{reason} | {why}", parent_version=parent)
    if promote:
        registry.promote(row["version"], why)
    else:
        registry.reject(row["version"], why)
    return TrainResult(row["version"], promote, why, candidate_val["accuracy"],
                       baseline_val["accuracy"], champion_acc, fold.val_start.isoformat(),
                       fold.val_end.isoformat(), candidate_val["n_targets"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entrena un candidato y decide su promoción")
    parser.add_argument("--reason", default=os.getenv("TRAIN_REASON", "manual"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    url, key = Settings.from_env().require_supabase()
    db = Supabase(url, key)
    try:
        with pipeline_run(db, "train") as run:
            wide = load_wide(db)
            log.info("Datos: %s filas, %s → %s", len(wide), wide.index[0], wide.index[-1])
            result = train_and_register(wide, ModelRegistry(db), reason=args.reason,
                                        git_commit=current_git_commit())
            run.summary.update(asdict(result))
            log.info("Entrenamiento: %s", asdict(result))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
