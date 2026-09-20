"""Entrega de práctica en modo local (sin Supabase): CSV oficial + modelo entrenado al vuelo.

Uso:
    python scripts/practice_submit.py            # simulacro: construye y valida, NO envía
    python scripts/practice_submit.py --send     # envía (requiere PULSO_API_KEY en el entorno)

El ciclo, sus targets y el data_cutoff los entrega la API; el script no tiene nada quemado.
El recibo queda en artifacts/practice_receipt.json (ignorado por git; no contiene la API key).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulso.api import PulsoApi, PulsoApiError  # noqa: E402
from pulso.config import Settings  # noqa: E402
from pulso.data import load_observations  # noqa: E402
from pulso.features import to_wide  # noqa: E402
from pulso.model import GbmResidualModel, ModelConfig  # noqa: E402
from pulso.predict import (  # noqa: E402
    build_payload,
    build_predictions,
    client_run_id,
    payload_hash,
)
from pulso.registry import current_git_commit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--send", action="store_true", help="enviar de verdad a la API")
    args = parser.parse_args()

    settings = Settings.from_env()
    with PulsoApi(settings.api_url, settings.api_key) as api:
        cycle = api.current_cycle()
        if cycle is None:
            print("No hay ciclo abierto: nada que enviar.")
            return 0
        print(f"Ciclo: {cycle['cycle_id']}  cierra: {cycle['closes_at']}  "
              f"targets: {cycle['expected_predictions']}")

        wide = to_wide(load_observations())
        print(f"Historia local: {wide.index[0]} → {wide.index[-1]} ({len(wide)} periodos)")
        model = GbmResidualModel(ModelConfig()).fit(wide)
        predictions, fallback = build_predictions(model, wide, cycle)
        now = datetime.now(UTC)
        payload = build_payload(
            cycle, predictions, model_version="practice-gbm-residual-0.1",
            trained_at=now.isoformat(), training_data_end=model.train_end.isoformat(),
            git_commit=current_git_commit(), client_run_id=client_run_id(now),
        )
        print(f"Payload válido: {len(predictions)} predicciones, {fallback} con perfil de respaldo, "
              f"hash {payload_hash(payload)[:19]}…")
        print(predictions.to_string(index=False))

        if not args.send:
            print("\nSimulacro: no se envió nada. Repite con --send para entregar.")
            return 0
        settings.require_api_key()
        identity = api.me()
        print(f"Identidad: {identity.get('display_name', '?')}")
        key = f"practice-{cycle['cycle_id']}-{now:%Y%m%dT%H%M%S}"
        try:
            status, receipt = api.submit(payload, key)
        except PulsoApiError as exc:
            print(f"La API rechazó la entrega: {exc}")
            return 1
        out = ROOT / "artifacts" / "practice_receipt.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"http_status": status, "idempotency_key": key, **receipt},
                                  indent=2, default=str), encoding="utf-8")
        print(f"HTTP {status}  entrega {receipt.get('submission_id')}  estado {receipt.get('status')}  "
              f"predicciones {receipt.get('predictions_received')}/{receipt.get('expected_predictions')}")
        print(f"Recibo guardado en {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
