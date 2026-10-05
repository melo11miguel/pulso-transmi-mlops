"""Comparacion pareada de dos recetas sobre muchos cortes de entrenamiento independientes.

Existe por un error propio que costo un ciclo de trabajo. El 2026-10-05 medi unas variables
nuevas en 3 bloques contiguos de 16 ciclos y dieron +2,38 ganando 3/3, con la ganancia creciendo
monotonamente con el horizonte: parecia la firma de un mecanismo. En 5 ventanas independientes el
efecto se caia, y en la mas parecida a produccion se desplomaba -19 puntos. La prueba mas clara de
que el instrumento era el problema: la misma puerta de 7 dias dio -0,23 y +0,31 en dos corridas
separadas por UNA hora de reloj virtual.

Tres cosas estaban mal y aqui se corrigen:

1. Bloques CONTIGUOS comparten regimen, asi que ganar en los tres no dice mas que ganar en uno.
   Aqui los cortes se reparten a lo largo de toda la historia disponible y se separan lo
   suficiente para que sus ventanas de evaluacion no se toquen.

2. Un solo numero sin dispersion no se puede juzgar. Cada corte da una diferencia pareada, y de
   esas diferencias sale un error estandar: el corte es la unidad de observacion, no el target.
   Los targets de un mismo corte comparten modelo y ventana, asi que tratarlos como
   independientes infla la muestra y da significancia donde no hay.

3. `make_folds` evalua con el modelo entrenado hasta 7 dias antes, y produccion refresca el
   champion cada pocas horas. Aqui la distancia entre el corte y su ventana es explicita y por
   defecto corta, para que la medicion se parezca a lo que se despliega.

El veredicto exige las dos cosas a la vez: ganar la mayoria de los cortes Y que el intervalo del
95 % de la media no toque el cero. Una sola de las dos se cumple por azar con demasiada facilidad.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .backtest import Fold, hourly_origins, score_predictions, summarize
from .features import STEP
from .model import GbmResidualModel, ModelConfig

log = logging.getLogger("pulso.evaluacion")

CORTES = 12          # cuantos cortes de entrenamiento
HORAS_EVAL = 8       # cuanto se evalua despues de cada corte
HORAS_SALTO = 12     # separacion entre cortes; > HORAS_EVAL para que no se solapen
MARGEN_MINIMO = 0.2  # la misma regla de la puerta, en puntos de accuracy


@dataclass(frozen=True)
class Comparacion:
    """Resultado pareado. `diferencias` es candidato menos referencia, un valor por corte."""

    cortes: list[pd.Timestamp]
    diferencias: list[float]
    acc_referencia: list[float]
    acc_candidato: list[float]
    por_horizonte: dict[int, list[float]] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.diferencias)

    @property
    def gana(self) -> int:
        return sum(1 for d in self.diferencias if d > 0)

    @property
    def media(self) -> float:
        return float(np.mean(self.diferencias)) if self.diferencias else float("nan")

    @property
    def error_estandar(self) -> float:
        """Error estandar de la media sobre los CORTES, que son la unidad independiente."""
        if self.n < 2:
            return float("nan")
        return float(np.std(self.diferencias, ddof=1) / np.sqrt(self.n))

    @property
    def intervalo(self) -> tuple[float, float]:
        e = self.error_estandar
        return (self.media - 1.96 * e, self.media + 1.96 * e)

    @property
    def significativa(self) -> bool:
        """El intervalo del 95 % no toca el cero."""
        lo, hi = self.intervalo
        return np.isfinite(lo) and (lo > 0 or hi < 0)

    def veredicto(self, margen: float = MARGEN_MINIMO) -> tuple[bool, str]:
        if self.n < 3:
            return False, f"solo {self.n} cortes: no alcanza para juzgar"
        lo, hi = self.intervalo
        mayoria = self.gana > self.n / 2
        if not self.significativa:
            return False, (f"{self.media:+.2f} pts de media, pero el intervalo del 95 % "
                           f"[{lo:+.2f}, {hi:+.2f}] contiene el cero: indistinguible del ruido")
        if self.media < margen:
            return False, (f"{self.media:+.2f} pts es significativo pero no alcanza el minimo "
                           f"de {margen:+.2f}")
        if not mayoria:
            return False, (f"{self.media:+.2f} pts de media pero solo gana {self.gana}/{self.n} "
                           f"cortes: la media la cargan unos pocos")
        return True, (f"{self.media:+.2f} pts, intervalo del 95 % [{lo:+.2f}, {hi:+.2f}], "
                      f"gana {self.gana}/{self.n} cortes")

    def tabla(self) -> str:
        lineas = [f"{'corte':<22} {'referencia':>10} {'candidato':>10} {'dif':>8}"]
        for t, a, b, d in zip(self.cortes, self.acc_referencia, self.acc_candidato,
                              self.diferencias, strict=True):
            lineas.append(f"{str(t):<22} {a:10.2f} {b:10.2f} {d:+8.2f}")
        lineas.append("")
        for h, ds in sorted(self.por_horizonte.items()):
            g = sum(1 for d in ds if d > 0)
            lineas.append(f"  +{h:2} min   medio {np.mean(ds):+6.2f}   gana {g}/{len(ds)}")
        ok, por_que = self.veredicto()
        lineas.append(f"\n{'ADOPTAR' if ok else 'DESCARTAR'}: {por_que}")
        return "\n".join(lineas)


def cortes_de_entrenamiento(times: pd.DatetimeIndex, *, cortes: int = CORTES,
                            horas_eval: float = HORAS_EVAL,
                            horas_salto: float = HORAS_SALTO) -> list[pd.Timestamp]:
    """Cortes repartidos hacia atras desde el final, uno cada `horas_salto`.

    `horas_salto` debe superar `horas_eval` para que las ventanas de evaluacion no se toquen; si no
    se cumple, las diferencias comparten datos y el error estandar queda subestimado.
    """
    if horas_salto <= horas_eval:
        raise ValueError("horas_salto debe ser mayor que horas_eval: las ventanas se solaparian")
    fin = times[-1]
    salida = []
    for k in range(cortes):
        corte = fin - pd.Timedelta(hours=horas_eval + k * horas_salto)
        if corte <= times[0]:
            break
        salida.append(corte)
    return list(reversed(salida))


def _evaluar(cfg: ModelConfig, wide: pd.DataFrame, corte: pd.Timestamp,
             horas_eval: float) -> dict | None:
    entrenamiento = wide.loc[:corte]
    fold = Fold("corte", corte, corte + STEP, corte + pd.Timedelta(hours=horas_eval))
    assert entrenamiento.index[-1] <= corte, "el entrenamiento se paso del corte"
    assert entrenamiento.index[-1] < fold.val_start, "entrenamiento y evaluacion se solapan"
    if len(hourly_origins(wide.index, fold)) == 0:
        return None
    modelo = GbmResidualModel(cfg).fit(entrenamiento)
    predicciones = modelo.predict_batch(wide, hourly_origins(wide.index, fold))
    return summarize(score_predictions(wide, predicciones))


def comparar(wide: pd.DataFrame, referencia: ModelConfig, candidato: ModelConfig, *,
             cortes: int = CORTES, horas_eval: float = HORAS_EVAL,
             horas_salto: float = HORAS_SALTO) -> Comparacion:
    """Entrena las dos recetas en cada corte y devuelve las diferencias pareadas.

    Pareado a proposito: las dos recetas ven el MISMO corte y la MISMA ventana, asi que la
    dificultad de la ventana se cancela en la diferencia. Es lo que permite comparar cortes que
    por si solos dan accuracies muy distintas.
    """
    ts, difs, refs, cands = [], [], [], []
    por_h: dict[int, list[float]] = {}
    for corte in cortes_de_entrenamiento(wide.index, cortes=cortes, horas_eval=horas_eval,
                                         horas_salto=horas_salto):
        a = _evaluar(referencia, wide, corte, horas_eval)
        b = _evaluar(candidato, wide, corte, horas_eval)
        if a is None or b is None:
            continue
        ts.append(corte)
        refs.append(a["accuracy"])
        cands.append(b["accuracy"])
        difs.append(b["accuracy"] - a["accuracy"])
        for h in a["by_horizon"]:
            por_h.setdefault(int(h), []).append(b["by_horizon"][h] - a["by_horizon"][h])
        log.info("corte %s: referencia %.2f candidato %.2f (%+.2f)",
                 corte, refs[-1], cands[-1], difs[-1])
    return Comparacion(cortes=ts, diferencias=difs, acc_referencia=refs,
                       acc_candidato=cands, por_horizonte=por_h)


def comparar_varias(wide: pd.DataFrame, recetas: dict[str, ModelConfig], referencia: str, *,
                    cortes: int = CORTES, horas_eval: float = HORAS_EVAL,
                    horas_salto: float = HORAS_SALTO) -> dict[str, Comparacion]:
    """Varias recetas sobre los MISMOS cortes, cada una evaluada una sola vez.

    `comparar` reentrena la referencia en cada llamada, asi que un barrido de N recetas la
    entrenaba N veces. Aqui cada receta se evalua una vez por corte y las diferencias pareadas
    salen de esa tabla: mismo resultado, N+1 ajustes por corte en vez de 2N.

    Un corte solo entra si TODAS las recetas pudieron evaluarse en el, para que las comparaciones
    compartan exactamente los mismos cortes y sigan siendo pareadas.
    """
    if referencia not in recetas:
        raise ValueError(f"la referencia {referencia!r} no esta entre las recetas")
    acc: dict[str, dict[pd.Timestamp, dict]] = {nombre: {} for nombre in recetas}
    for corte in cortes_de_entrenamiento(wide.index, cortes=cortes, horas_eval=horas_eval,
                                         horas_salto=horas_salto):
        salidas = {n: _evaluar(cfg, wide, corte, horas_eval) for n, cfg in recetas.items()}
        if any(v is None for v in salidas.values()):
            continue
        for n, v in salidas.items():
            acc[n][corte] = v
        log.info("corte %s: " + "  ".join(f"{n} %.2f" for n in recetas), corte,
                 *[salidas[n]["accuracy"] for n in recetas])

    comunes = sorted(acc[referencia])
    salida: dict[str, Comparacion] = {}
    for nombre in recetas:
        if nombre == referencia:
            continue
        por_h: dict[int, list[float]] = {}
        for t in comunes:
            for h in acc[referencia][t]["by_horizon"]:
                por_h.setdefault(int(h), []).append(
                    acc[nombre][t]["by_horizon"][h] - acc[referencia][t]["by_horizon"][h])
        salida[nombre] = Comparacion(
            cortes=comunes,
            diferencias=[acc[nombre][t]["accuracy"] - acc[referencia][t]["accuracy"]
                         for t in comunes],
            acc_referencia=[acc[referencia][t]["accuracy"] for t in comunes],
            acc_candidato=[acc[nombre][t]["accuracy"] for t in comunes],
            por_horizonte=por_h,
        )
    return salida


def comparar_sin(wide: pd.DataFrame, base: ModelConfig, variables: tuple[str, ...],
                 **kwargs) -> Comparacion:
    """Mide que aportan `variables`: la referencia es la MISMA receta con ellas descartadas.

    Gemelo contra gemelo. La referencia se construye con `drop_features`, nunca con un artefacto
    viejo del registro: si las dos recetas difieren en algo mas que las variables en cuestion, la
    diferencia ya no mide lo que se cree.
    """
    sobran = tuple(dict.fromkeys((*base.drop_features, *variables)))
    return comparar(wide, replace(base, drop_features=sobran), base, **kwargs)
