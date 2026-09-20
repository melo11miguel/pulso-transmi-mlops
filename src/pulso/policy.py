"""Reglas de decisión del sistema: promover un modelo y reentrenar.

Son funciones puras (sin red ni base de datos) para poder probarlas y justificarlas. Los
umbrales viven en dataclasses con su razón; se calibraron con el backtesting y el estrés de
drift (reports/backtest.md, reports/drift_stress.md), no se copiaron de una cifra universal.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------- promoción
@dataclass(frozen=True)
class PromotionRules:
    # Ruido entre semanas del backtest: ~±0.3 pts. Exigir 0.2 evita promover por azar sin
    # bloquear mejoras reales (p. ej. un reentrenamiento tras drift gana varios puntos).
    min_gain_over_champion: float = 0.2
    # El modelo debe superar al perfil puro (baseline sin aprendizaje reciente).
    min_gain_over_baseline: float = 0.0


def decide_promotion(candidate_acc: float, champion_acc: float | None, baseline_acc: float,
                     smoke_ok: bool, rules: PromotionRules = PromotionRules()) -> tuple[bool, str]:
    """Devuelve (promover, razón). Novedad por sí sola no es mejora."""
    if not smoke_ok:
        return False, "falló la inferencia de prueba"
    if candidate_acc != candidate_acc:  # NaN
        return False, "la validación no produjo métrica"
    if candidate_acc < baseline_acc + rules.min_gain_over_baseline:
        return False, (f"no supera al baseline de perfil ({candidate_acc:.2f} vs {baseline_acc:.2f})")
    if champion_acc is None:
        return True, f"primer champion: {candidate_acc:.2f} vs baseline {baseline_acc:.2f}"
    gain = candidate_acc - champion_acc
    if gain >= rules.min_gain_over_champion:
        return True, f"mejora {gain:+.2f} pts sobre el champion ({champion_acc:.2f})"
    return False, f"mejora insuficiente ({gain:+.2f} pts < {rules.min_gain_over_champion})"


# ---------------------------------------------------------------------------- reentrenamiento
@dataclass(frozen=True)
class RetrainRules:
    # Caída del accuracy rolling 24 h frente a la referencia de validación que cuenta como
    # degradación. El backtest semanal varía ±0.5; una caída de 3 pts ya es 6σ.
    performance_drop_pts: float = 3.0
    # Cambio de nivel por estación: |media del residuo log de 24 h| >= max(piso, k × ruido de fondo
    # propio). El ruido de fondo (eventos, lluvia) se mide al entrenar (monitor.level_noise): con un
    # umbral fijo de 0.10 un evento de tarde-noche disparó una falsa alarma en la prueba de estrés.
    level_drift_floor: float = 0.10
    level_noise_multiplier: float = 1.25
    # Evaluaciones consecutivas (una cada 30 min) que deben cumplirse: evita reaccionar a un
    # único periodo difícil. El desempeño es una señal fuerte (4 = 2 h); el nivel se mide sobre
    # ventanas de 24 h casi solapadas, por lo que exige más (12 = 6 h).
    persistence: int = 4
    data_persistence: int = 12
    # Datos nuevos mínimos desde el último entrenamiento para que reentrenar aporte algo.
    min_new_hours: float = 12.0
    # Enfriamiento: no reentrenar de nuevo antes de este tiempo.
    cooldown_hours: float = 6.0
    # Cobertura mínima de entregas (elegibilidad prevista: 95 %).
    min_coverage: float = 0.95
    # Retraso máximo tolerado entre el reloj de la API y el último dato recolectado.
    max_collector_lag_minutes: float = 90.0
    # Ejecuciones fallidas (workflows + ingestas) en 24 h a partir de las cuales se investiga la
    # operación. Un fallo aislado de red no debe congelar la respuesta a un drift.
    max_failed_runs_24h: int = 3


@dataclass
class MonitorState:
    """Lo que el monitor sabe en este momento."""

    reference_accuracy: float | None  # accuracy de validación del champion
    rolling_accuracies: list[float] = field(default_factory=list)  # más reciente al final
    level_drift_streaks: dict[str, int] = field(default_factory=dict)  # estación -> evaluaciones seguidas
    hours_since_training: float | None = None
    new_data_hours: float = 0.0
    coverage: float | None = None
    collector_lag_minutes: float | None = None
    failed_runs_24h: int = 0


@dataclass
class Decision:
    decision: str  # keep | investigate | retrain
    reason: str
    signals: dict[str, bool] = field(default_factory=dict)


def performance_signal(state: MonitorState, rules: RetrainRules) -> bool:
    """True si las últimas `persistence` evaluaciones están todas por debajo del umbral."""
    if state.reference_accuracy is None or len(state.rolling_accuracies) < rules.persistence:
        return False
    limit = state.reference_accuracy - rules.performance_drop_pts
    return all(a < limit for a in state.rolling_accuracies[-rules.persistence:])


def data_signal(state: MonitorState, rules: RetrainRules) -> bool:
    return any(streak >= rules.data_persistence for streak in state.level_drift_streaks.values())


def operational_signal(state: MonitorState, rules: RetrainRules) -> bool:
    lagging = (state.collector_lag_minutes is not None
               and state.collector_lag_minutes > rules.max_collector_lag_minutes)
    low_coverage = state.coverage is not None and state.coverage < rules.min_coverage
    return lagging or low_coverage or state.failed_runs_24h >= rules.max_failed_runs_24h


def decide_retrain(state: MonitorState, rules: RetrainRules = RetrainRules()) -> Decision:
    perf, data, ops = (performance_signal(state, rules), data_signal(state, rules),
                       operational_signal(state, rules))
    signals = {"performance": perf, "data": data, "operational": ops}

    if ops:
        # Primero se arregla la operación: reentrenar no repara un pipeline que no entrega.
        return Decision("investigate", "falla operacional (collector, cobertura o ejecuciones "
                        "fallidas): corregir antes de culpar al modelo", signals)
    if not (perf or data):
        return Decision("keep", "sin degradación persistente ni cambio de nivel", signals)

    cause = "caída persistente de accuracy" if perf else "cambio de nivel persistente"
    if perf and data:
        cause = "caída persistente de accuracy y cambio de nivel"
    if state.hours_since_training is not None and state.hours_since_training < rules.cooldown_hours:
        return Decision("investigate", f"{cause}, pero el último entrenamiento fue hace "
                        f"{state.hours_since_training:.1f} h (enfriamiento {rules.cooldown_hours} h)",
                        signals)
    if state.new_data_hours < rules.min_new_hours:
        return Decision("investigate", f"{cause}, pero solo hay {state.new_data_hours:.1f} h de "
                        f"datos nuevos (mínimo {rules.min_new_hours} h)", signals)
    return Decision("retrain", f"{cause}; {state.new_data_hours:.1f} h de datos nuevos desde el "
                    "último entrenamiento", signals)
