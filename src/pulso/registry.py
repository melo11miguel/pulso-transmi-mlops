"""Registro de modelos: versiones con linaje en `model_versions` y artefactos en Storage.

Reglas: cada versión tiene identidad (id, corte de datos, commit, features, validación,
ubicación y hash del artefacto); los artefactos nunca se sobrescriben; solo existe un champion.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import UTC, datetime
from typing import Any

from .model import GbmResidualModel
from .supa import Supabase, SupabaseError

BUCKET = "models"


class RegistryError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_version(model: GbmResidualModel, git_commit: str | None) -> str:
    cut = model.train_end.tz_convert("UTC").strftime("%Y%m%dT%H%MZ")
    return f"gbm-{cut}-{(git_commit or 'nocommit')[:7]}"


class ModelRegistry:
    def __init__(self, db: Supabase, bucket: str = BUCKET) -> None:
        self.db = db
        self.bucket = bucket

    # ---- consulta -----------------------------------------------------------------------
    def get(self, version: str) -> dict[str, Any] | None:
        rows = self.db.select("model_versions", filters={"version": f"eq.{version}"})
        return rows[0] if rows else None

    def champion(self) -> dict[str, Any] | None:
        rows = self.db.select("model_versions", filters={"status": "eq.champion"})
        return rows[0] if rows else None

    # ---- registro -----------------------------------------------------------------------
    def register_candidate(self, model: GbmResidualModel, *, validation: dict[str, Any],
                           git_commit: str | None = None, reason: str | None = None,
                           parent_version: str | None = None) -> dict[str, Any]:
        """Sube el artefacto (nuevo, sin sobrescribir) y crea la versión como `candidate`."""
        version = make_version(model, git_commit)
        if self.get(version):  # mismo corte y commit: se distingue con un sufijo incremental
            n = 2
            while self.get(f"{version}-{n}"):
                n += 1
            version = f"{version}-{n}"
        data = model.to_bytes()
        digest = sha256(data)
        path = f"{version}/model.joblib"
        try:
            self.db.upload(self.bucket, path, data)
        except SupabaseError as exc:
            if exc.status not in (400, 409):  # ya existe: solo se acepta si es idéntico
                raise
            if sha256(self.db.download(self.bucket, path)) != digest:
                raise RegistryError(f"El artefacto {path} ya existe y es distinto") from exc
        info = model.describe()
        return self.db.insert("model_versions", {
            "version": version, "status": "candidate", "algorithm": "hist_gbm_residual_v1",
            "trained_at": _now(), "training_data_start": info["train_start"],
            "training_data_end": info["train_end"], "git_commit": git_commit,
            "features": info["features"],
            "params": {**info["config"], "n_train_rows": info["n_train_rows"],
                       "sklearn": info["sklearn"]},
            "validation": validation, "artifact_path": path, "artifact_sha256": digest,
            "parent_version": parent_version, "reason": reason,
        })

    def load(self, version: str) -> GbmResidualModel:
        """Descarga y verifica el hash antes de deserializar (no se carga «el último archivo»)."""
        row = self.get(version)
        if not row or not row.get("artifact_path"):
            raise RegistryError(f"Versión {version!r} sin artefacto registrado")
        data = self.db.download(self.bucket, row["artifact_path"])
        if sha256(data) != row["artifact_sha256"]:
            raise RegistryError(f"El hash de {version} no coincide con el registrado")
        return GbmResidualModel.from_bytes(data)

    def load_champion(self) -> tuple[dict[str, Any], GbmResidualModel]:
        row = self.champion()
        if row is None:
            raise RegistryError("No hay modelo champion: ejecute el workflow de entrenamiento")
        return row, self.load(row["version"])

    # ---- estado -------------------------------------------------------------------------
    def promote(self, version: str, reason: str) -> dict[str, Any]:
        """Champion nuevo (o rollback a una versión anterior) de forma atómica en la base."""
        return self.db.rpc("promote_model", {"p_version": version, "p_reason": reason})

    def reject(self, version: str, reason: str) -> None:
        self.db.update("model_versions", {"status": "rejected", "reason": reason},
                       {"version": f"eq.{version}", "status": "eq.candidate"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


def current_git_commit() -> str | None:
    """Commit del código que produjo el modelo (en Actions: GITHUB_SHA)."""
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        return out if len(out) == 40 else None
    except (OSError, subprocess.CalledProcessError):
        return None
