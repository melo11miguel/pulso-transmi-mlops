"""Replica el registro de modelos de Supabase en MLflow, para el bono del reto.

QUE ES Y QUE NO ES. El registro de verdad del pipeline es la tabla `model_versions` en Supabase:
ahi se decide la promocion, de ahi lee el predictor y ahi esta la trazabilidad. Este script no
cambia eso, lo EXPORTA: lee las versiones ya registradas y las escribe en MLflow con sus params,
metricas y linaje, para tener la interfaz de MLflow sobre datos reales del proyecto. No inventa
ninguna corrida ni ninguna metrica.

Se eligio exportar en vez de instrumentar el entrenamiento por una razon operativa: el pipeline
corre en ejecutores efimeros de GitHub Actions y ya tiene su registro con verificacion de hash,
indice unico de un solo champion y linaje por `parent_version`. Meter MLflow como registro
primario pedia exponer la cadena de conexion de Postgres a Actions y duplicar la fuente de
verdad, con el riesgo de que las dos se desincronicen.

POR QUE SQLITE Y NO ARCHIVOS. El Model Registry de MLflow no funciona sobre el almacen de
archivos, solo sobre una base. De ahi el `sqlite:///mlflow.db` por defecto.

POR QUE SIN DEPENDENCIAS DEL PIPELINE. `requirements.txt` fija versiones exactas porque un
artefacto de scikit-learn solo carga con la version con que se entreno. MLflow arrastra sus
propias versiones, asi que vive en su propio entorno (.venv-mlflow) y este script lee Supabase
con la libreria estandar en vez de importar `pulso`.

PUERTO. Se usa 5055 y NO el 5000 por defecto de MLflow: en macOS el 5000 lo ocupa el receptor de
AirPlay del Centro de Control, que responde 403, asi que el navegador parece mostrar una interfaz
vacia cuando en realidad no esta hablando con MLflow.

Uso:
    .venv-mlflow/bin/python scripts/mlflow_export.py
    .venv-mlflow/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5055
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENTO = "pulso-transmi"
MODELO = "pulso-transmi-demanda"
BUCKET = "models"
PUERTO = 5055
PAGINA = 1000


def _env(nombre: str) -> str:
    valor = os.environ.get(nombre)
    if not valor:
        sys.exit(f"Falta la variable de entorno {nombre} (cargue .env antes de correr)")
    return valor


def _get(url: str, key: str) -> bytes:
    pedido = urllib.request.Request(url, headers={"apikey": key, "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(pedido, timeout=60) as respuesta:
        return respuesta.read()


def versiones(base: str, key: str) -> list[dict]:
    """Todas las filas de `model_versions`, paginadas.

    Paginado a la fuerza: Supabase corta en 1000 filas y devuelve las mas viejas, asi que un
    select suelto dejaria fuera las versiones recientes en silencio. Ya nos paso.
    """
    salida: list[dict] = []
    while True:
        params = urllib.parse.urlencode({
            "select": "*", "order": "created_at.asc",
            "limit": PAGINA, "offset": len(salida),
        })
        lote = json.loads(_get(f"{base}/rest/v1/model_versions?{params}", key))
        salida.extend(lote)
        if len(lote) < PAGINA:
            return salida


def artefacto(base: str, key: str, ruta: str) -> bytes | None:
    try:
        return _get(f"{base}/storage/v1/object/{BUCKET}/{urllib.parse.quote(ruta)}", key)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(f"  (no se pudo bajar el artefacto {ruta}: {exc})")
        return None


def _ms(iso: str | None) -> int | None:
    if not iso:
        return None
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)


def metricas(validation: dict) -> dict[str, float]:
    """Metricas planas desde el jsonb de validacion, incluidas las de por horizonte y estacion."""
    salida: dict[str, float] = {}
    for clave in ("accuracy", "wape", "baseline_profile_only_accuracy",
                  "champion_twin_accuracy", "n_targets"):
        valor = validation.get(clave)
        if isinstance(valor, (int, float)):
            salida[clave] = float(valor)
    for h, valor in (validation.get("by_horizon") or {}).items():
        if isinstance(valor, (int, float)):
            salida[f"accuracy_mas_{h}min"] = float(valor)
    for estacion, valor in (validation.get("by_station") or {}).items():
        if isinstance(valor, (int, float)):
            salida[f"accuracy_estacion_{estacion}"] = float(valor)
    # Margen sobre el gemelo del champion: es el numero que decide la promocion.
    if "accuracy" in salida and "champion_twin_accuracy" in salida:
        salida["margen_sobre_champion"] = salida["accuracy"] - salida["champion_twin_accuracy"]
    return salida


def etiquetas(fila: dict) -> dict[str, str]:
    validation = fila.get("validation") or {}
    crudas = {
        "status": fila.get("status"),
        "algorithm": fila.get("algorithm"),
        "git_commit": fila.get("git_commit"),
        "parent_version": fila.get("parent_version"),
        "artifact_path": fila.get("artifact_path"),
        "artifact_sha256": fila.get("artifact_sha256"),
        "training_data_start": fila.get("training_data_start"),
        "training_data_end": fila.get("training_data_end"),
        "promoted_at": fila.get("promoted_at"),
        "reason": fila.get("reason"),
        "decision": validation.get("decision"),
        "decision_reason": validation.get("decision_reason"),
        "modo": validation.get("modo"),
        "comparacion": validation.get("comparacion"),
        "holdout_start": validation.get("holdout_start"),
        "holdout_end": validation.get("holdout_end"),
        "smoke_test": validation.get("smoke_test"),
        "fuente": "exportado de model_versions (Supabase)",
    }
    return {k: str(v)[:500] for k, v in crudas.items() if v not in (None, "")}


def main() -> int:
    base = _env("SUPABASE_URL").rstrip("/")
    key = _env("SUPABASE_SERVICE_ROLE_KEY")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    cliente = MlflowClient()

    filas = versiones(base, key)
    print(f"{len(filas)} versiones leidas de Supabase")
    if not filas:
        return 1

    experimento = cliente.get_experiment_by_name(EXPERIMENTO)
    exp_id = experimento.experiment_id if experimento else cliente.create_experiment(EXPERIMENTO)

    ya = {r.data.tags.get("mlflow.runName")
          for r in mlflow.search_runs([exp_id], output_format="list", max_results=50000)}
    print(f"{len(ya)} corridas ya estaban exportadas; se omiten")

    champion_run = None
    nuevas = 0
    for fila in filas:
        version = fila["version"]
        if version in ya:
            if fila.get("status") == "champion":
                champion_run = next(
                    r.info.run_id for r in
                    mlflow.search_runs([exp_id], f"tags.mlflow.runName = '{version}'",
                                       output_format="list"))
            continue
        corrida = cliente.create_run(exp_id, start_time=_ms(fila.get("trained_at")),
                                     run_name=version, tags=etiquetas(fila))
        rid = corrida.info.run_id
        for nombre, valor in (fila.get("params") or {}).items():
            cliente.log_param(rid, nombre, "" if valor is None else str(valor))
        variables = fila.get("features") or []
        cliente.log_param(rid, "n_features", len(variables))
        cliente.log_param(rid, "features", ",".join(map(str, variables))[:500])
        for nombre, valor in metricas(fila.get("validation") or {}).items():
            cliente.log_metric(rid, nombre, valor)
        cliente.set_terminated(rid, "FINISHED", end_time=_ms(fila.get("created_at")))
        nuevas += 1
        if fila.get("status") == "champion":
            champion_run = rid
    print(f"{nuevas} corridas nuevas exportadas al experimento {EXPERIMENTO!r}")

    # El champion, con su artefacto real y registrado en el Model Registry.
    champion = next((f for f in filas if f.get("status") == "champion"), None)
    if champion is None or champion_run is None:
        print("Sin champion en el registro: no se registra modelo")
        return 0

    ya_subido = any(a.path == "model" for a in cliente.list_artifacts(champion_run))
    datos = (None if ya_subido or not champion.get("artifact_path")
             else artefacto(base, key, champion["artifact_path"]))
    if ya_subido:
        print("El artefacto del champion ya estaba subido")
    if datos:
        local = "model.joblib"
        with open(local, "wb") as f:
            f.write(datos)
        cliente.log_artifact(champion_run, local, artifact_path="model")
        os.remove(local)
        print(f"Artefacto del champion subido ({len(datos) / 1e6:.1f} MB)")

    try:
        cliente.create_registered_model(
            MODELO, description="Demanda de TransMilenio a +15/+30/+45/+60 min. "
                                "Perfil estacional por franja del dia mas correccion GBM "
                                "con perdida L1 sobre el residual logaritmico.")
    except Exception:
        pass  # ya existia
    # Idempotente tambien aqui: sin esta busqueda, cada ejecucion creaba una version nueva del
    # modelo apuntando al MISMO champion, y el registro se llenaba de duplicados.
    existente = next((v for v in cliente.search_model_versions(f"name = '{MODELO}'")
                      if v.run_id == champion_run), None)
    if existente is not None:
        version = existente.version
        print(f"La version {version} ya apuntaba a esta corrida: no se duplica")
    else:
        version = cliente.create_model_version(
            MODELO, source=f"runs:/{champion_run}/model", run_id=champion_run,
            description=f"Champion del pipeline. accuracy de validacion "
                        f"{(champion.get('validation') or {}).get('accuracy')}, "
                        f"datos hasta {champion.get('training_data_end')}.").version
    cliente.set_registered_model_alias(MODELO, "champion", version)
    print(f"Modelo {MODELO!r} version {version} con alias 'champion' -> {champion['version']}")
    print(f"\n{len(filas)} versiones en el experimento {EXPERIMENTO!r}. Abra la interfaz con:\n"
          f"  .venv-mlflow/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port {PUERTO}\n"
          f"y luego http://127.0.0.1:{PUERTO}\n"
          f"El puerto va explicito: en macOS el 5000 por defecto lo ocupa AirPlay y responde 403, "
          f"asi que la interfaz se ve vacia sin que el error sea evidente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
