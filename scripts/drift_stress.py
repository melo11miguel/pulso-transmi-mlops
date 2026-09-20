"""Prueba de estrés con drift sintético (Fase 5): impacto, recuperación y detección.

Uso:  python scripts/drift_stress.py
Escribe reports/drift_stress.md y reports/figures/06_drift_estres.png.

El histórico oficial no tiene drift; se simula (src/pulso/stress.py) para responder tres preguntas:
1. ¿Cuánto se degrada un modelo congelado? 2. ¿Reentrenar lo recupera? 3. ¿Las reglas de
monitoreo (policy.py) lo detectan a tiempo y sin falsas alarmas?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402
import plotstyle  # noqa: E402

from pulso.backtest import Fold, hourly_origins, score_predictions  # noqa: E402
from pulso.baselines import ProfileMean, RidgeResidual  # noqa: E402
from pulso.data import TZ, load_observations  # noqa: E402
from pulso.features import to_wide  # noqa: E402
from pulso.metrics import official_accuracy  # noqa: E402
from pulso.model import GbmResidualModel, ModelConfig  # noqa: E402
from pulso.monitor import (  # noqa: E402
    level_thresholds,
    rolling_accuracy,
    station_level_shift,
    update_streaks,
)
from pulso.policy import MonitorState, RetrainRules, decide_retrain  # noqa: E402
from pulso.stress import SCENARIOS, apply_drift  # noqa: E402

plotstyle.apply()
RULES = RetrainRules()


def T(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=TZ)


wide0 = to_wide(load_observations())
END = wide0.index[-1]
DRIFT_START = T("2026-09-04 00:00")
TRAIN_END = T("2026-09-01 23:45")
RETRAIN_END = T("2026-09-05 23:45")  # 2 días de datos con drift
EVAL_FROZEN = T("2026-09-04 12:00")  # rampa completada
EVAL_RETRAIN = T("2026-09-06 00:00")


def score(model, wide, t0, t1=END):
    fold = Fold("x", t0 - pd.Timedelta(minutes=15), t0, t1)
    origins = hourly_origins(wide.index, fold)
    return score_predictions(wide, model.predict_batch(wide, origins))


def acc(scored, stations=None):
    if stations is not None:
        scored = scored[scored["station_id"].isin(stations)]
    return official_accuracy(scored["actual"], scored["prediction"], scored["station_id"])


# ---- modelos congelados (entrenados hasta 09-01) y referencia de validación honesta
frozen = {
    "profile_only": ProfileMean().fit(wide0.loc[:TRAIN_END]),
    "ridge_residual": RidgeResidual().fit(wide0.loc[:TRAIN_END]),
    "gbm_residual": GbmResidualModel(ModelConfig()).fit(wide0.loc[:TRAIN_END]),
}
champion = frozen["gbm_residual"]
THRESHOLDS = level_thresholds(champion.level_noise, RULES.level_drift_floor, RULES.level_noise_multiplier)
# Referencia = accuracy de validación del champion en la semana previa (fold 2), no la de la semana evaluada
prev = GbmResidualModel(ModelConfig()).fit(wide0.loc[:T("2026-08-25 23:45")])
REFERENCE = acc(score(prev, wide0, T("2026-08-26 00:00"), T("2026-09-01 23:45")))
print("referencia de validación:", round(REFERENCE, 2), flush=True)

scenarios = {"sin drift": dict(kind=None, stations=[])} | SCENARIOS
wides = {name: (wide0 if spec["kind"] is None else
                apply_drift(wide0, spec["kind"], spec["stations"], DRIFT_START, spec["magnitude"]))
         for name, spec in scenarios.items()}

impact_rows, recovery_rows, detection_rows, timelines = [], [], [], {}
for name, spec in scenarios.items():
    wd, aff = wides[name], spec["stations"] or None
    for mname, model in frozen.items():
        s_frozen = score(model, wd, EVAL_FROZEN)
        s_late = score(model, wd, EVAL_RETRAIN)
        impact_rows.append({"escenario": name, "modelo": mname, "acc_total": acc(s_frozen),
                            "acc_afectadas": acc(s_frozen, aff) if aff else np.nan,
                            "acc_total_desde_09-06": acc(s_late)})
    retrained = GbmResidualModel(ModelConfig()).fit(wd.loc[:RETRAIN_END])
    s_re = score(retrained, wd, EVAL_RETRAIN)
    s_fr = score(champion, wd, EVAL_RETRAIN)
    recovery_rows.append({
        "escenario": name, "congelado": acc(s_fr), "reentrenado": acc(s_re),
        "congelado_afectadas": acc(s_fr, aff) if aff else np.nan,
        "reentrenado_afectadas": acc(s_re, aff) if aff else np.nan,
    })

    # ---- detección: evaluaciones cada 30 min con el champion congelado
    val_origins = hourly_origins(wd.index, Fold("v", T("2026-09-01 23:45"), T("2026-09-02 00:00"), END))
    scored = score_predictions(wd, champion.predict_batch(wd, val_origins))
    scored["target_at"] = pd.to_datetime(scored["target_at"])
    evals = pd.date_range(T("2026-09-03 00:00"), END, freq="30min")
    streaks: dict[str, int] = {}
    accs, perf_first, data_first, retrain_first, max_shift, max_ratio = [], None, None, None, 0.0, 0.0
    for t in evals:
        a, _ = rolling_accuracy(scored, t)
        accs.append(a)
        shifts = station_level_shift(champion.profile, wd, t)
        streaks = update_streaks(streaks, shifts, THRESHOLDS)
        max_shift = max(max_shift, float(np.nanmax(np.abs(shifts.to_numpy()))))
        max_ratio = max(max_ratio, float(np.nanmax(np.abs(shifts / THRESHOLDS).to_numpy())))
        state = MonitorState(
            reference_accuracy=REFERENCE, rolling_accuracies=[x for x in accs if x == x],
            level_drift_streaks=streaks, hours_since_training=(t - TRAIN_END) / pd.Timedelta(hours=1),
            new_data_hours=(t - TRAIN_END) / pd.Timedelta(hours=1))
        decision = decide_retrain(state, RULES)
        if perf_first is None and decision.signals["performance"]:
            perf_first = t
        if data_first is None and decision.signals["data"]:
            data_first = t
        if retrain_first is None and decision.decision == "retrain":
            retrain_first = t
    timelines[name] = pd.Series(accs, index=evals)
    lat = lambda t: (t - DRIFT_START) / pd.Timedelta(hours=1) if t is not None else np.nan  # noqa: E731
    detection_rows.append({
        "escenario": name, "acc_rolling_min": float(np.nanmin(accs)),
        "max_|desplaz|": max_shift, "max_desplaz/umbral": max_ratio,
        "señal_desempeño_h": lat(perf_first),
        "señal_datos_h": lat(data_first), "reentrenar_h": lat(retrain_first),
    })
    print(name, "listo", flush=True)

impact = pd.DataFrame(impact_rows)
recovery = pd.DataFrame(recovery_rows)
detection = pd.DataFrame(detection_rows)


def md(frame: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("—" if v != v else fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---- figura
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
drift_names = [n for n in scenarios if n != "sin drift"]
x = np.arange(len(drift_names))
w = 0.26
fz_prof = [impact[(impact.escenario == n) & (impact.modelo == "profile_only")].acc_afectadas.iloc[0] for n in drift_names]
fz_gbm = [impact[(impact.escenario == n) & (impact.modelo == "gbm_residual")].acc_afectadas.iloc[0] for n in drift_names]
re_gbm = [recovery[recovery.escenario == n].reentrenado_afectadas.iloc[0] for n in drift_names]
a1.bar(x - w, fz_prof, w * 0.92, color="#a9a8a2", label="perfil puro (congelado)")
a1.bar(x, fz_gbm, w * 0.92, color=plotstyle.ORANGE, label="modelo (congelado)")
a1.bar(x + w, re_gbm, w * 0.92, color=plotstyle.BLUE, label="modelo reentrenado con 2 días")
a1.set_xticks(x, [n.split(" (")[0] for n in drift_names], fontsize=8)
a1.set_ylabel("accuracy en estaciones afectadas (%)")
a1.set_title("Un modelo congelado se degrada; reentrenar lo recupera")
a1.legend(fontsize=8, loc="lower left")
for name, color in [("sin drift", "#a9a8a2"), (drift_names[1], plotstyle.ORANGE), (drift_names[0], plotstyle.BLUE)]:
    series = timelines[name]
    a2.plot(series.index, series.values, color=color, lw=2, label=name.split(" (")[0])
a2.axhline(REFERENCE - RULES.performance_drop_pts, color=plotstyle.INK2, lw=1, ls="--")
a2.text(timelines["sin drift"].index[2], REFERENCE - RULES.performance_drop_pts + 0.4,
        f"umbral = referencia − {RULES.performance_drop_pts:.0f} pts", fontsize=7.5, color=plotstyle.INK2)
a2.axvline(DRIFT_START, color=plotstyle.INK2, lw=1)
a2.text(DRIFT_START, a2.get_ylim()[0] + 0.5, " inicio del drift", fontsize=7.5, color=plotstyle.INK2)
a2.set_title("Accuracy móvil de 24 h")
a2.set_ylabel("%")
a2.legend(fontsize=8, loc="lower left")
a2.tick_params(axis="x", labelsize=7.5)
fig.autofmt_xdate(rotation=0, ha="center")
fig.tight_layout()
fig.savefig(ROOT / "reports" / "figures" / "06_drift_estres.png")

nod = detection[detection.escenario == "sin drift"].iloc[0]
alarm = bool(pd.notna(nod["señal_desempeño_h"]) or pd.notna(nod["señal_datos_h"]))
report = f"""# Prueba de estrés con drift sintético

> Generado por `scripts/drift_stress.py`. El histórico oficial no tiene drift; aquí se **simula**
> (`src/pulso/stress.py`) imitando los tipos publicados: `level_shift`, `peak_shift`,
> `closure` y `trend_change`. Son escenarios propios, no los de la competencia.

**Montaje:** champion congelado (entrenado hasta 2026-09-01), drift desde {DRIFT_START:%Y-%m-%d %H:%M}
con transición gradual de 12 h, y evaluación hasta {END:%Y-%m-%d}. Referencia de validación del
champion: **{REFERENCE:.2f} %** (accuracy de la semana previa, no la evaluada).

## 1. ¿Cuánto se degrada un modelo congelado?

Accuracy desde {EVAL_FROZEN:%m-%d %H:%M} (rampa completada). `acc_afectadas` = solo las estaciones con drift.

{md(impact[['escenario', 'modelo', 'acc_total', 'acc_afectadas']])}

- El modelo con variables recientes **amortigua** los cambios de nivel y de tendencia mucho mejor que
  el perfil puro (compárense `gbm_residual` y `profile_only` en las estaciones afectadas).
- Un **cambio de pico** o un **cierre** rompen cualquier modelo congelado: el patrón aprendido ya no existe.
  En el cierre, la accuracy de la estación afectada llega a ~0 (WAPE > 1: se predice mucho más de lo real).

## 2. ¿Reentrenar lo recupera?

Reentrenar con los datos hasta {RETRAIN_END:%m-%d} (2 días con drift) y evaluar desde {EVAL_RETRAIN:%m-%d}:

{md(recovery)}

Con solo 2 días de datos nuevos el reentrenamiento recupera casi todo en nivel, tendencia y cierre.
El cambio de pico se recupera parcialmente: el perfil semanal mezcla 5 semanas del patrón viejo con
2 días del nuevo; mejora conforme llegan más días (reentrenamientos programados diarios).

**Descartado:** dar más peso a lo reciente (vida media de 5 días) empeoró tanto con drift como sin él
en las pruebas exploratorias: con pocas muestras por celda semanal, la varianza pesa más que el sesgo.

## 3. ¿Las reglas de monitoreo lo detectan?

Reglas (`policy.RetrainRules`): señal de **desempeño** = accuracy móvil 24 h por debajo de
referencia − {RULES.performance_drop_pts:.0f} pts durante {RULES.persistence} evaluaciones seguidas (cada 30 min);
señal de **datos** = |desplazamiento de nivel de 24 h| ≥ umbral propio de la estación durante
{RULES.data_persistence} evaluaciones seguidas (6 h). El umbral es `max({RULES.level_drift_floor}, {RULES.level_noise_multiplier} × ruido de fondo de la estación)`,
donde el ruido de fondo es el mayor desplazamiento de 24 h visto en el entrenamiento (eventos, lluvia).
Las horas se cuentan desde el inicio del drift.

Umbrales por estación: {", ".join(f"{k} {v:.2f}" for k, v in THRESHOLDS.items())}.

{md(detection, "{:.2f}")}

- **Falsas alarmas:** en el escenario sin drift, el accuracy móvil mínimo fue {nod['acc_rolling_min']:.2f} % (umbral
  {REFERENCE - RULES.performance_drop_pts:.2f}) y el desplazamiento de nivel máximo llegó al {nod['max_desplaz/umbral'] * 100:.0f} % de su umbral.
  {'**Hubo falsa alarma.**' if alarm else 'Ninguna señal se disparó.'}
- **Lección de la calibración:** con un umbral fijo de 0.10 y persistencia de 2 h, un evento de tarde-noche
  (+23 % en Museo Nacional y Movistar Arena) disparó una falsa alarma a las 24 h. Por eso el umbral pasó a
  ser propio de cada estación (las sensibles a eventos tienen más ruido) y la persistencia del nivel a 6 h.
  Sigue siendo una calibración con 5 eventos históricos: un evento mayor podría disparar una alarma, cuyo
  costo es bajo (un reentrenamiento que la regla de promoción rechaza si no mejora).
- Las dos señales se **complementan**: el cambio de nivel y la tendencia casi no mueven el accuracy
  total (afectan a pocas estaciones) pero sí el desplazamiento de nivel; el cambio de pico no mueve el
  nivel medio (sube y baja) pero sí hunde el accuracy.
- «reentrenar_h» es cuándo `decide_retrain` decidiría reentrenar (tras persistencia, con ≥ {RULES.min_new_hours:.0f} h de
  datos nuevos y fuera del enfriamiento de {RULES.cooldown_hours:.0f} h).

![Estrés](figures/06_drift_estres.png)

## 4. Conclusiones

1. Un modelo estático **no basta**: pico y cierre lo rompen; hace falta reentrenar.
2. Reentrenar con datos recientes es la respuesta correcta y funciona con pocos días.
3. La política combina persistencia, volumen de datos nuevos y enfriamiento para no reaccionar a un
   único periodo difícil, y prioriza arreglar fallas operacionales antes de reentrenar.
4. **Límite:** estos drifts son propios y simples. Los de la competencia (con `weather_change`,
   `variance_shift`, redistribución espacial en cierres) pueden comportarse distinto; los umbrales
   deben revisarse con los datos reales de los primeros ciclos.
"""
(ROOT / "reports" / "drift_stress.md").write_text(report, encoding="utf-8")
print("reports/drift_stress.md escrito")
