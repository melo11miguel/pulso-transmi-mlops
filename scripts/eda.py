"""Análisis exploratorio del histórico inicial (Fase 1).

Uso:  python scripts/eda.py
Lee data/*.csv y escribe reports/eda.md + reports/figures/*.png.
Todas las cifras del reporte se calculan aquí; nada se copia a mano.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulso.data import load_context, load_observations, load_stations  # noqa: E402
from pulso.quality import quality_report  # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Paleta de referencia (tokens de la guía de visualización), modo claro.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # slots categóricos 1-3 (validados all-pairs)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "font.size": 9, "legend.frameon": False, "figure.dpi": 130,
    }
)

obs = load_observations()
ctx = load_context()
stations = load_stations().set_index("station_id")
name = stations["station_name"].to_dict()

# ---------------------------------------------------------------- calidad
qr = quality_report(obs)
ctx_dups = int(ctx["observed_at"].duplicated().sum())
ctx_nulls = int(ctx.isna().sum().sum())

# ---------------------------------------------------------------- preparación
df = obs.merge(ctx, on="observed_at", how="left", validate="many_to_one")
df["hour"] = df["observed_at"].dt.hour
df["dow"] = df["observed_at"].dt.dayofweek
df["weekend"] = df["dow"] >= 5
df["slot"] = df["observed_at"].dt.hour * 4 + df["observed_at"].dt.minute // 15
df["how"] = df["dow"] * 96 + df["slot"]  # hora de la semana (0-671)
df["log"] = np.log(df["demand"])
df["seasonal"] = df.groupby(["station_id", "how"])["log"].transform("mean")
df["resid"] = df["log"] - df["seasonal"]

# R² del perfil estación × hora-de-la-semana sobre log(demanda), dentro de cada estación
centered = df["log"] - df.groupby("station_id")["log"].transform("mean")
r2_seasonal = 1 - df["resid"].var() / centered.var()

# ---------------------------------------------------------------- grupos por razón fin de semana / semana
means = df.groupby(["station_id", "weekend"])["demand"].mean().unstack()
ratio = (means[True] / means[False]).rename("we_ratio")


def group_of(r: float) -> str:
    return "A" if r < 0.65 else ("B" if r < 1.0 else "C")


groups = ratio.map(group_of).rename("group")
GROUP_LABEL = {
    "A": "Grupo A · entre semana ≫ fin de semana",
    "B": "Grupo B · fin de semana algo menor",
    "C": "Grupo C · fin de semana mayor",
}
GROUP_COLOR = {"A": C1, "B": C2, "C": C3}
df["group"] = df["station_id"].map(groups)

# ---------------------------------------------------------------- Fig 1: perfiles horarios
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
hours = np.arange(24)
station_mean = df.groupby("station_id")["demand"].transform("mean")
df["rel"] = df["demand"] / station_mean  # siempre respecto a la media global de la estación
for ax, g in zip(axes, "ABC", strict=True):
    for wknd, ls, lab in [(False, "-", "entre semana"), (True, "--", "fin de semana")]:
        sub = df[(df["group"] == g) & (df["weekend"] == wknd)]
        prof = sub.groupby("hour")["rel"].mean().reindex(hours)
        ax.plot(hours, prof.values, ls, color=GROUP_COLOR[g], lw=2, label=lab)
    members = ", ".join(sorted(groups[groups == g].index))
    ax.set_title(GROUP_LABEL[g], fontsize=9.5)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel(f"hora local\nestaciones: {members}")
    ax.set_ylim(0, 3.3)
    ax.legend(loc="upper center", fontsize=8)
axes[0].set_ylabel("demanda / media de la estación")
fig.suptitle("Perfil diario: tres comportamientos distintos", x=0.01, ha="left",
             fontweight="bold", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG / "01_perfiles_horarios.png")
plt.close(fig)

# ---------------------------------------------------------------- Fig 2: total diario
daily = df.groupby(df["observed_at"].dt.date)["demand"].sum()
daily.index = pd.to_datetime(daily.index)
we = daily.index.dayofweek >= 5
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.plot(daily.index, daily.values / 1e3, color=INK2, lw=1.2, zorder=1)
ax.scatter(daily.index[~we], daily.values[~we] / 1e3, s=22, color=C1, label="día hábil",
           zorder=3, edgecolor=SURFACE, linewidth=1)
ax.scatter(daily.index[we], daily.values[we] / 1e3, s=22, color=C2, label="fin de semana",
           zorder=3, edgecolor=SURFACE, linewidth=1)
ax.set_ylabel("pasajeros por día (miles, 12 estaciones)")
ax.set_title("El nivel diario es estable: no hay tendencia ni drift dentro del histórico")
ax.legend(loc="lower right", ncols=2)
fig.tight_layout()
fig.savefig(FIG / "02_total_diario.png")
plt.close(fig)

# ---------------------------------------------------------------- clima y eventos
bins = [-0.001, 0.05, 0.5, 1.5, 100]
labels = ["<0,05", "0,05–0,5", "0,5–1,5", ">1,5"]
df["rain_bin"] = pd.cut(df["rain_mm"], bins=bins, labels=labels)
rain_eff = (
    df.groupby(["group", "rain_bin"], observed=True)["resid"].mean().unstack("group")
    .pipe(lambda t: np.exp(t) - 1) * 100
)
EVENT_MIN = 0.3
df["event"] = df["event_intensity"] > EVENT_MIN
ev_eff = (df.groupby(["station_id", "event"])["resid"].mean().unstack()
          .pipe(lambda t: (np.exp(t[True] - t[False]) - 1) * 100).sort_values())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.3]})
w = 0.26
for i, g in enumerate("ABC"):
    a1.bar(np.arange(len(labels)) + (i - 1) * w, rain_eff[g].values, w * 0.9,
           color=GROUP_COLOR[g], label=f"Grupo {g}")
a1.axhline(0, color=INK2, lw=0.8)
a1.set_xticks(range(len(labels)), labels)
a1.set_xlabel("lluvia (mm por 15 min)")
a1.set_ylabel("efecto sobre la demanda (%)")
a1.set_title("Lluvia: reduce la demanda, salvo en el grupo C")
a1.legend(loc="lower left", fontsize=8)
colors = [GROUP_COLOR[groups[s]] for s in ev_eff.index]
a2.barh([f"{s} {name[s][:20]}{chr(8230) if len(name[s]) > 20 else chr(32)}" for s in ev_eff.index], ev_eff.values, color=colors, height=0.7)
a2.set_xlabel(f"efecto durante eventos (intensidad > {EVENT_MIN}), %")
a2.set_title("Eventos: efecto mucho mayor en dos estaciones")
a2.tick_params(axis="y", labelsize=7.5)
fig.tight_layout()
fig.savefig(FIG / "03_clima_eventos.png")
plt.close(fig)

# ---------------------------------------------------------------- residuos: autocorrelación y ruido
piv = df.pivot(index="observed_at", columns="station_id", values="resid")
lags = np.arange(1, 97)
acf = np.array([np.mean([piv[s].autocorr(int(k)) for s in piv]) for k in lags])
cross = (piv.corr().values.sum() - piv.shape[1]) / (piv.shape[1] * (piv.shape[1] - 1))
cell = df.groupby(["station_id", "how"])["demand"].agg(["mean", "var"])
vmr = float((cell["var"] / cell["mean"]).median())
cv = float((np.sqrt(cell["var"]) / cell["mean"]).median())

fig, ax = plt.subplots(figsize=(11, 2.8))
ax.bar(lags, acf, color=C1, width=0.8)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("rezago (periodos de 15 min)")
ax.set_ylabel("autocorrelación")
ax.set_title("Lo que queda tras quitar la estacionalidad es casi ruido: autocorrelación baja")
fig.tight_layout()
fig.savefig(FIG / "04_autocorrelacion_residuos.png")
plt.close(fig)

# ---------------------------------------------------------------- cifras para el reporte
peak_hour = (df[~df["weekend"]].groupby(["station_id", "hour"])["demand"].mean().unstack()
             .idxmax(axis=1))
station_tbl = pd.DataFrame(
    {
        "estación": [name[s] for s in ratio.index],
        "grupo": groups.values,
        "media": df.groupby("station_id")["demand"].mean().round(0).astype(int).values,
        "máx": df.groupby("station_id")["demand"].max().values,
        "hora pico (hábil)": [f"{int(peak_hour[s]):02d}:00" for s in ratio.index],
        "fin sem./sem.": ratio.round(2).values,
    },
    index=ratio.index,
)
event_days = (ctx.loc[ctx["event_intensity"] > 0.1]
              .groupby(ctx["observed_at"].dt.date)["event_intensity"].max())
n_events = int((event_days > 0.9).sum())
rain_corr_city = float(df.groupby("observed_at")["resid"].mean()
                       .corr(ctx.set_index("observed_at")["rain_mm"]))
rain_fc_corr = float(ctx["rain_mm"].corr(ctx["rain_forecast"]))
temp_fc_corr = float(ctx["temperature_c"].corr(ctx["temperature_forecast"]))
temp_corr_city = float(df.groupby("observed_at")["resid"].mean()
                       .corr(ctx.set_index("observed_at")["temperature_c"]))
weekly = df.groupby((df["observed_at"] - df["observed_at"].min()).dt.days // 7)["demand"].mean()
weekly_change = float(weekly.iloc[-1] / weekly.iloc[0] - 1) * 100


def md_table(frame: pd.DataFrame) -> str:
    cols = [frame.index.name or "id", *frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for idx, row in frame.iterrows():
        lines.append("| " + " | ".join([str(idx), *[str(v) for v in row]]) + " |")
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{x:+.0f} %"


gtxt = {g: ", ".join(sorted(groups[groups == g].index)) for g in "ABC"}
report = f"""# Análisis exploratorio — Pulso TransMi

> Generado por `scripts/eda.py` a partir del corte inicial oficial. Todas las cifras
> se recalculan al ejecutarlo.

## 1. Calidad de los datos

| Control | Resultado |
|---|---|
| Observaciones | {qr.rows:,} ({qr.stations} estaciones × {qr.rows // qr.stations:,} periodos) |
| Rango (hora local, UTC-5) | {qr.start:%Y-%m-%d %H:%M} → {qr.end:%Y-%m-%d %H:%M} |
| Periodos faltantes en la rejilla de 15 min | {qr.missing_periods} |
| Duplicados `(station_id, observed_at)` | {qr.duplicates} |
| Valores nulos / demanda negativa / fuera de rejilla | {qr.nulls} / {qr.negatives} / {qr.out_of_grid} |
| Contexto: filas, duplicados, nulos | {len(ctx):,}, {ctx_dups}, {ctx_nulls} |
| Demanda mínima / mediana / máxima | {obs.demand.min()} / {int(obs.demand.median())} / {obs.demand.max()} |
| Ceros | {int((obs.demand == 0).sum())} |

**Conclusión:** el histórico es completo y consistente ({'sin ningún problema' if qr.ok else 'con problemas'}).
No hace falta imputar ni filtrar. Los IDs de estación son texto de 5 dígitos y se conservan como tal.

## 2. Variable objetivo y diccionario de datos

**Objetivo:** `demand` (pasajeros por estación y periodo de 15 min) en cuatro horizontes
futuros: +15, +30, +45 y +60 min desde el `data_cutoff` del ciclo.

| Campo | Tipo | Descripción |
|---|---|---|
| `station_id` | texto (5 dígitos) | Estación; conservar los ceros iniciales |
| `observed_at` | timestamptz | Inicio del periodo de 15 min; la API lo entrega en UTC-5 |
| `demand` | entero ≥ 0 | Pasajeros del periodo (conteo, binomial negativa según el generador) |
| `rain_mm`, `rain_forecast` | real | Lluvia observada y pronóstico |
| `temperature_c`, `temperature_forecast` | real | Temperatura observada y pronóstico |
| `event_intensity` | real [0, 1] | Intensidad de eventos masivos (curva suave alrededor del evento) |

## 3. Estacionalidad: domina todo

El perfil «estación × hora de la semana» (672 valores por estación) explica el
**{r2_seasonal * 100:.1f} %** de la varianza de `log(demanda)` dentro de cada estación.
El ruido restante tiene una desviación de {df.resid.std():.3f} en escala logarítmica (≈ {df.resid.std() * 100:.0f} %).

![Perfiles](figures/01_perfiles_horarios.png)

Las 12 estaciones se separan en tres comportamientos según la razón demanda de fin de
semana / entre semana:

- **Grupo A** (razón ≈ {ratio[groups == "A"].mean():.2f}): {gtxt["A"]}. Demanda laboral/universitaria: cae a la mitad el fin de semana.
- **Grupo B** (razón ≈ {ratio[groups == "B"].mean():.2f}): {gtxt["B"]}. Residencial/intercambio: picos de ida y vuelta.
- **Grupo C** (razón ≈ {ratio[groups == "C"].mean():.2f}): {gtxt["C"]}. Ocio: sube el fin de semana y de noche.

{md_table(station_tbl)}

Los grupos son una inferencia a partir de los datos; los arquetipos reales del generador son privados.

## 4. Nivel y tendencia

![Total diario](figures/02_total_diario.png)

Entre la primera y la última semana el promedio cambia {weekly_change:+.1f} %, sin tendencia
apreciable. **No hay drift dentro del histórico**: cualquier degradación que aparezca en la
competencia será un cambio nuevo, no algo que el modelo pueda aprender de estos 45 días.

## 5. Clima y eventos

![Clima y eventos](figures/03_clima_eventos.png)

- La lluvia observada y su pronóstico están muy correlacionados (r = {rain_fc_corr:.2f}); en temperatura, r = {temp_fc_corr:.2f}.
- La lluvia reduce la demanda en los grupos A y B (hasta {pct(rain_eff[["A", "B"]].min().min())} con lluvia fuerte)
  y la aumenta en el grupo C ({pct(rain_eff["C"].max())}). A escala ciudad, la correlación del residuo con la lluvia es {rain_corr_city:.2f}.
- La temperatura no tiene efecto detectable (r = {temp_corr_city:.2f} con el residuo de ciudad).
- Hay {n_events} eventos en el histórico (tarde-noche, en días distintos). Elevan la demanda
  {pct(ev_eff.median())} en la mediana de estaciones y {pct(ev_eff.max())} en la más afectada.

## 6. Estructura del ruido

![Autocorrelación](figures/04_autocorrelacion_residuos.png)

- Autocorrelación del residuo: {acf[0]:.2f} (rezago 1), {acf[3]:.2f} (rezago 4), {acf[95]:.2f} (rezago 96).
- Correlación media entre estaciones del mismo instante: {cross:.2f} (factor de ciudad débil).
- Sobredispersión: varianza/media ≈ {vmr:.1f} por celda (estación × hora de la semana);
  coeficiente de variación ≈ {cv:.2f}. Coherente con una binomial negativa.

## 7. Hipótesis (a contrastar con backtesting temporal)

1. **H1 – Calendario domina.** Un perfil por estación y hora de la semana será un baseline muy
   difícil de superar; el aporte de los rezagos recientes será pequeño (autocorrelación del residuo ≈ {acf[0]:.2f}).
2. **H2 – Hay un techo de accuracy cercano a {100 - 0.8 * cv * 100:.0f} %.** Si la variación por celda (CV ≈ {cv:.2f})
   fuera ruido gaussiano puro, el WAPE mínimo sería ≈ 0,8 × CV ≈ {0.8 * cv * 100:.0f} % (aproximación; parte de esa
   variación es clima y eventos, así que el techo real puede ser algo mayor). Los modelos deben medirse
   contra ese techo y no contra 100.
3. **H3 – El clima ayuda solo si está disponible al predecir.** Explica una parte real del residuo de
   ciudad, pero `/v1/context` es un corte estático: hay que verificar si la API lo publica durante la
   competencia antes de depender de él. El modelo principal debe funcionar sin contexto.
4. **H4 – La robustez pesa más que la complejidad.** Sin drift en el histórico, un modelo complejo no
   puede demostrar que resiste cambios. Hay que evaluar cómo degradan los modelos ante un cambio de nivel
   o de perfil, y preferir modelos que se reentrenan rápido con datos recientes.
5. **H5 – Métrica por estación.** La accuracy promedia estaciones sin ponderar; las de baja demanda
   (grupo A/B pequeñas) tienen más ruido relativo y pesan igual. Hay que reportar el error por estación.

## 8. Implicaciones para el diseño

- Validación temporal por bloques de ciclos (origen cada 30 min, 4 horizontes), nunca aleatoria.
- Features de calendario (hora de la semana), rezagos recientes y perfil histórico por estación.
- Un modelo por horizonte (o horizonte como variable) porque el peso de los rezagos cambia con h.
- Reentrenar de forma barata y frecuente con ventana deslizante para reaccionar a drift.
"""
(ROOT / "reports" / "eda.md").write_text(report, encoding="utf-8")
print("reports/eda.md escrito;", len(list(FIG.glob("*.png"))), "figuras")
print(f"R2={r2_seasonal:.4f} resid_std={df.resid.std():.3f} cv={cv:.3f} vmr={vmr:.2f} cross={cross:.3f}")
print(f"acf1={acf[0]:.3f} acf4={acf[3]:.3f} acf96={acf[95]:.3f} weekly_change={weekly_change:.2f}%")
print(f"events(>0.9)={n_events} ev_median={ev_eff.median():.1f} ev_max={ev_eff.max():.1f}")
print(rain_eff.round(1).to_string())
print(groups.to_dict())
