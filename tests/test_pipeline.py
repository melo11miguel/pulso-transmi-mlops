"""Pruebas de política, monitoreo, estrés, registro, entrenamiento e inferencia de punta a punta."""

from datetime import timedelta

import httpx
import numpy as np
import pandas as pd
import pytest
from conftest import FakeDb, FakeServer
from test_features import FAST, synthetic_wide

from pulso.api import PulsoApi, PulsoApiError
from pulso.features import Profile
from pulso.model import GbmResidualModel
from pulso.monitor import (
    baseline_accuracy,
    level_noise,
    level_thresholds,
    rolling_accuracy,
    station_level_shift,
    update_streaks,
)
from pulso.policy import (
    MonitorState,
    PromotionRules,
    RetrainRules,
    decide_promotion,
    decide_retrain,
    performance_signal,
)
from pulso.predict import (
    PredictionError,
    build_payload,
    build_predictions,
    payload_hash,
    run_inference,
    validate_submission,
)
from pulso.registry import ModelRegistry, RegistryError, make_version, sha256
from pulso.runlog import pipeline_run
from pulso.stress import apply_drift
from pulso.train import TrainingError, holdout_fold, smoke_test, train_and_register

TZ = "America/Bogota"
LENIENT = PromotionRules(min_gain_over_champion=0.2, min_gain_over_baseline=-100.0)


# ------------------------------------------------------------------ política de promoción
@pytest.mark.parametrize("cand, champ, base, smoke, promote", [
    (88.0, None, 86.0, True, True),  # primer champion
    (88.0, 87.0, 86.0, True, True),  # +1.0 sobre el champion
    (87.15, 87.0, 86.0, True, False),  # +0.15 < 0.2: novedad no es mejora
    (87.2, 87.0, 86.0, True, True),  # justo en el umbral
    (85.9, None, 86.0, True, False),  # no supera al baseline
    (90.0, 87.0, 86.0, False, False),  # falló la prueba de humo
    (float("nan"), 87.0, 86.0, True, False),  # sin métrica
    (86.5, 88.0, 86.0, True, False),  # peor que el champion
])
def test_decide_promotion(cand, champ, base, smoke, promote):
    ok, reason = decide_promotion(cand, champ, base, smoke)
    assert ok is promote and reason


# ------------------------------------------------------------------ política de reentrenamiento
def _state(**kw):
    defaults = dict(reference_accuracy=87.0, rolling_accuracies=[87.0] * 6,
                    hours_since_training=48.0, new_data_hours=48.0, coverage=1.0,
                    collector_lag_minutes=10.0, failed_runs_24h=0)
    return MonitorState(**{**defaults, **kw})


def test_keep_when_everything_is_fine():
    assert decide_retrain(_state()).decision == "keep"


def test_single_bad_period_does_not_trigger_retrain():
    state = _state(rolling_accuracies=[87, 87, 87, 80.0])  # una sola evaluación mala
    assert not performance_signal(state, RetrainRules())
    assert decide_retrain(state).decision == "keep"


def test_persistent_degradation_triggers_retrain():
    d = decide_retrain(_state(rolling_accuracies=[87, 80.0, 80.0, 80.0, 80.0]))
    assert d.decision == "retrain" and d.signals["performance"]


def test_threshold_is_strict_below_reference_minus_drop():
    on_edge = _state(rolling_accuracies=[84.0] * 4)  # 87 - 3 = 84: no es «por debajo»
    assert decide_retrain(on_edge).decision == "keep"
    assert decide_retrain(_state(rolling_accuracies=[83.99] * 4)).decision == "retrain"


def test_level_drift_alone_triggers_retrain():
    d = decide_retrain(_state(level_drift_streaks={"02300": 12, "03000": 0}))
    assert d.decision == "retrain" and d.signals["data"] and not d.signals["performance"]


def test_short_level_streak_is_ignored():
    assert decide_retrain(_state(level_drift_streaks={"02300": 11})).decision == "keep"


def test_cooldown_blocks_retrain():
    d = decide_retrain(_state(rolling_accuracies=[70.0] * 4, hours_since_training=2.0))
    assert d.decision == "investigate" and "enfriamiento" in d.reason


def test_too_little_new_data_blocks_retrain():
    d = decide_retrain(_state(rolling_accuracies=[70.0] * 4, new_data_hours=3.0))
    assert d.decision == "investigate" and "datos nuevos" in d.reason


@pytest.mark.parametrize("field, value", [
    ("collector_lag_minutes", 200.0), ("coverage", 0.5), ("failed_runs_24h", 3),
])
def test_operational_failure_takes_priority_over_retraining(field, value):
    d = decide_retrain(_state(rolling_accuracies=[70.0] * 4, **{field: value}))
    assert d.decision == "investigate" and d.signals["operational"] and "operacional" in d.reason


def test_isolated_failure_does_not_block_the_response_to_drift():
    d = decide_retrain(_state(rolling_accuracies=[70.0] * 4, failed_runs_24h=2))
    assert d.decision == "retrain" and not d.signals["operational"]


def test_needs_reference_to_judge_performance():
    assert not performance_signal(_state(reference_accuracy=None, rolling_accuracies=[10.0] * 9),
                                  RetrainRules())


# ------------------------------------------------------------------ monitoreo
def _scored(days=3, error=0.0):
    idx = pd.date_range("2026-09-01", periods=days * 96, freq="15min", tz=TZ)
    frames = []
    for station in ("A", "B", "C", "D", "E", "F"):
        actual = np.full(len(idx), 100.0)
        frames.append(pd.DataFrame({"station_id": station, "target_at": idx, "actual": actual,
                                    "prediction": actual * (1 + error)}))
    return pd.concat(frames, ignore_index=True), idx


def test_rolling_accuracy_uses_only_the_window():
    scored, idx = _scored()
    now = idx[-1]
    scored.loc[scored["target_at"] < now - pd.Timedelta(hours=24), "prediction"] = 0.0  # pasado malo
    accuracy, n = rolling_accuracy(scored, now, hours=24)
    assert accuracy == pytest.approx(100.0) and n == 6 * 96


def test_rolling_accuracy_window_is_open_on_the_left_and_closed_on_the_right():
    scored, idx = _scored()
    now = idx[200]
    _, n = rolling_accuracy(scored, now, hours=24)
    assert n == 6 * 96  # (now-24h, now]: exactamente 96 periodos por estación


def test_rolling_accuracy_is_nan_with_too_few_stations():
    scored, idx = _scored()
    few = scored[scored["station_id"].isin(["A", "B"])]
    accuracy, _ = rolling_accuracy(few, idx[-1])
    assert np.isnan(accuracy)


def test_station_level_shift_detects_a_level_change():
    wide = synthetic_wide(days=24)
    profile = Profile().fit(wide.index[:14 * 96], np.log(wide.to_numpy()[:14 * 96]))
    drifted = wide.copy()
    drifted["03000"] = drifted["03000"] * 1.3
    now = wide.index[-1]
    shift = station_level_shift(profile, drifted, now, hours=24)
    assert shift["03000"] == pytest.approx(np.log(1.3), abs=0.06)
    assert abs(shift["02300"]) < 0.05 and abs(shift["05000"]) < 0.05


def test_update_streaks_counts_consecutive_hits_and_resets():
    shifts = pd.Series({"A": 0.2, "B": 0.05, "C": np.nan, "D": -0.15})
    first = update_streaks({}, shifts, 0.10)
    assert first == {"A": 1, "B": 0, "C": 0, "D": 1}
    second = update_streaks(first, pd.Series({"A": 0.2, "B": 0.2, "C": 0.0, "D": 0.0}), 0.10)
    assert second == {"A": 2, "B": 1, "C": 0, "D": 0}


def test_level_noise_reflects_transient_events_and_sets_higher_thresholds():
    wide = synthetic_wide(days=24)
    events = wide.copy()
    events.iloc[10 * 96 + 40:10 * 96 + 90, 0] *= 1.6  # evento de ~12 h en la primera estación
    times, log = events.index, np.log(events.to_numpy())
    noise = level_noise(Profile().fit(times, log), events)
    assert noise["02300"] > 2 * noise["03000"] and noise["02300"] > 0.15
    thresholds = level_thresholds(noise, floor=0.10, multiplier=1.25)
    assert thresholds["02300"] == pytest.approx(1.25 * noise["02300"])
    assert thresholds["03000"] == 0.10  # el piso protege a estaciones con poco ruido


def test_update_streaks_accepts_per_station_thresholds():
    shifts = pd.Series({"A": 0.15, "B": 0.15})
    thresholds = pd.Series({"A": 0.10, "B": 0.20})
    assert update_streaks({}, shifts, thresholds) == {"A": 1, "B": 0}


def test_baseline_accuracy_matches_profile_prediction():
    wide = synthetic_wide(days=24)
    profile = Profile().fit(wide.index[:20 * 96], np.log(wide.to_numpy()[:20 * 96]))
    idx = wide.index[20 * 96:]
    scored = pd.concat([pd.DataFrame({"station_id": s, "target_at": idx, "actual": wide.loc[idx, s].to_numpy(),
                                      "prediction": 1.0}) for s in wide.columns], ignore_index=True)
    acc = baseline_accuracy(profile, wide, scored, wide.index[-1], hours=48)
    assert 80 < acc <= 100  # el perfil sobre datos sintéticos con ruido de 10 %


# ------------------------------------------------------------------ estrés de drift
def test_apply_drift_leaves_the_past_and_other_stations_untouched():
    wide = synthetic_wide(days=10)
    start = wide.index[5 * 96]
    out = apply_drift(wide, "level_shift", ["03000"], start, magnitude=0.25, ramp_hours=0)
    assert out.loc[:start - pd.Timedelta(minutes=15), "03000"].equals(wide.loc[:start - pd.Timedelta(minutes=15), "03000"])
    assert out["02300"].equals(wide["02300"])
    late = wide.index[-1]
    assert out.loc[late, "03000"] == pytest.approx(wide.loc[late, "03000"] * 1.25, abs=1)


def test_apply_drift_ramp_is_gradual():
    wide = synthetic_wide(days=10)
    start = wide.index[5 * 96]
    out = apply_drift(wide, "level_shift", ["03000"], start, magnitude=0.5, ramp_hours=12)
    ratio = (out["03000"] / wide["03000"]).loc[start:start + pd.Timedelta(hours=18)]
    assert ratio.iloc[1] < 1.05 and ratio.iloc[-1] > 1.45  # arranca suave y llega al nivel final


def test_apply_drift_peak_shift_moves_the_pattern_later():
    wide = synthetic_wide(days=10)
    start = wide.index[3 * 96]
    out = apply_drift(wide, "peak_shift", ["03000"], start, magnitude=1.0, ramp_hours=0)
    t = wide.index[6 * 96 + 40]
    assert out.loc[t, "03000"] == wide["03000"].iloc[6 * 96 + 40 - 4]


def test_apply_drift_closure_and_trend():
    wide = synthetic_wide(days=10)
    start = wide.index[5 * 96]
    closed = apply_drift(wide, "closure", ["03000"], start, magnitude=0.6, ramp_hours=0)
    assert closed["03000"].iloc[-1] == pytest.approx(wide["03000"].iloc[-1] * 0.4, abs=1)
    trend = apply_drift(wide, "trend_change", ["03000"], start, magnitude=0.1, ramp_hours=0)
    assert trend["03000"].iloc[-1] > wide["03000"].iloc[-1] * 1.3


def test_apply_drift_rejects_unknown_kind():
    with pytest.raises(ValueError, match="desconocido"):
        apply_drift(synthetic_wide(days=3), "meteor", ["03000"], pd.Timestamp("2026-08-04", tz=TZ))


# ------------------------------------------------------------------ registro
@pytest.fixture(scope="module")
def trained() -> tuple[pd.DataFrame, GbmResidualModel]:
    wide = synthetic_wide(days=30)
    return wide, GbmResidualModel(FAST).fit(wide)


def test_registry_registers_uploads_and_promotes(fake_db, trained):
    wide, model = trained
    registry = ModelRegistry(fake_db)
    row = registry.register_candidate(model, validation={"accuracy": 88.0}, git_commit="a" * 40)
    assert row["status"] == "candidate" and row["version"] == make_version(model, "a" * 40)
    stored = fake_db.storage[("models", row["artifact_path"])]
    assert sha256(stored) == row["artifact_sha256"]
    registry.promote(row["version"], "primero")
    assert registry.champion()["version"] == row["version"]


def test_registry_keeps_a_single_champion_and_supports_rollback(fake_db, trained):
    _, model = trained
    registry = ModelRegistry(fake_db)
    v1 = registry.register_candidate(model, validation={}, git_commit="a" * 40)["version"]
    v2 = registry.register_candidate(model, validation={}, git_commit="a" * 40)["version"]
    assert v2 == v1 + "-2"  # mismo corte y commit: sufijo, sin sobrescribir el artefacto
    registry.promote(v1, "x")
    registry.promote(v2, "mejor")
    assert [r["status"] for r in fake_db.tables["model_versions"]] == ["archived", "champion"]
    registry.promote(v1, "rollback")
    assert registry.champion()["version"] == v1
    assert sum(r["status"] == "champion" for r in fake_db.tables["model_versions"]) == 1


def test_registry_load_verifies_hash(fake_db, trained):
    _, model = trained
    registry = ModelRegistry(fake_db)
    row = registry.register_candidate(model, validation={}, git_commit="b" * 40)
    assert isinstance(registry.load(row["version"]), GbmResidualModel)
    fake_db.storage[("models", row["artifact_path"])] += b"corrupt"
    with pytest.raises(RegistryError, match="hash"):
        registry.load(row["version"])


def test_registry_without_champion_raises(fake_db):
    with pytest.raises(RegistryError, match="champion"):
        ModelRegistry(fake_db).load_champion()


def test_registry_reject_only_touches_candidates(fake_db, trained):
    _, model = trained
    registry = ModelRegistry(fake_db)
    version = registry.register_candidate(model, validation={}, git_commit="c" * 40)["version"]
    registry.promote(version, "x")
    registry.reject(version, "no debería aplicar")  # ya es champion
    assert registry.champion()["version"] == version


# ------------------------------------------------------------------ entrenamiento
def test_train_first_run_promotes_and_second_identical_run_does_not(fake_db):
    wide = synthetic_wide(days=30)
    registry = ModelRegistry(fake_db)
    first = train_and_register(wide, registry, config=FAST, rules=LENIENT, git_commit="d" * 40)
    assert first.promoted and registry.champion()["version"] == first.version
    second = train_and_register(wide, registry, config=FAST, rules=LENIENT, git_commit="d" * 40)
    assert not second.promoted and "insuficiente" in second.reason
    assert registry.champion()["version"] == first.version  # el champion no cambió
    statuses = {r["version"]: r["status"] for r in fake_db.tables["model_versions"]}
    assert statuses[second.version] == "rejected"
    assert second.champion_accuracy is not None


def test_train_validation_metadata_is_recorded(fake_db):
    wide = synthetic_wide(days=30)
    result = train_and_register(wide, ModelRegistry(fake_db), config=FAST, rules=LENIENT)
    row = fake_db.tables["model_versions"][0]
    v = row["validation"]
    assert v["holdout_end"].startswith(str(wide.index[-1].date())) and v["n_targets"] > 0
    assert v["baseline_profile_only_accuracy"] == pytest.approx(result.baseline_accuracy)
    assert row["training_data_end"] == wide.index[-1].isoformat()
    assert row["features"] and row["artifact_sha256"]


def test_train_rejects_candidate_that_loses_to_baseline(fake_db):
    wide = synthetic_wide(days=30)
    strict = PromotionRules(min_gain_over_baseline=50.0)  # imposible
    result = train_and_register(wide, ModelRegistry(fake_db), config=FAST, rules=strict)
    assert not result.promoted and "baseline" in result.reason


def test_train_needs_enough_history(fake_db):
    with pytest.raises(TrainingError, match="días"):
        train_and_register(synthetic_wide(days=10), ModelRegistry(fake_db), config=FAST)


def test_holdout_is_the_last_seven_days():
    wide = synthetic_wide(days=30)
    fold = holdout_fold(wide)
    assert fold.val_end == wide.index[-1]
    assert fold.val_end - fold.val_start == pd.Timedelta(days=7) - pd.Timedelta(minutes=15)


def test_smoke_test_flags_broken_models(trained, monkeypatch):
    wide, model = trained
    assert smoke_test(model, wide) == (True, "ok")
    ok, message = smoke_test(GbmResidualModel(FAST), wide)  # sin entrenar
    assert not ok and "entrenado" in message

    class NanModel(GbmResidualModel):
        def predict_next(self, history):
            out = super().predict_next(history)
            out["value"] = np.nan
            return out

    nan_model = NanModel(FAST)
    nan_model.__dict__.update(model.__dict__)
    ok, message = smoke_test(nan_model, wide)
    assert not ok and "no finitas" in message


def test_missing_station_history_still_yields_a_full_forecast(trained):
    wide, model = trained
    out = model.predict_next(wide[["02300", "03000"]])  # falta 05000: cae al perfil, no pierde cobertura
    assert len(out) == 12 and np.isfinite(out["value"]).all()


# ------------------------------------------------------------------ inferencia
def _cycle(wide, horizons=(1, 2, 3, 4), cycle_id="cyc_test_1"):
    cutoff = wide.index[-1]
    utc = lambda ts: ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    targets = [{"station_id": s, "target_at": utc(cutoff + pd.Timedelta(minutes=15 * h)),
                "horizon_minutes": 15 * h} for s in wide.columns for h in horizons]
    return {"cycle_id": cycle_id, "state": "open", "origin_at": utc(cutoff),
            "data_cutoff": utc(cutoff), "opens_at": utc(cutoff),
            "closes_at": utc(cutoff + pd.Timedelta(hours=1)),
            "expected_predictions": len(targets), "targets": targets}


def test_build_predictions_covers_exactly_the_cycle_targets(trained):
    wide, model = trained
    cycle = _cycle(wide)
    predictions, fallback = build_predictions(model, wide, cycle)
    assert fallback == 0 and len(predictions) == 12
    assert set(predictions["target_at"]) == {t["target_at"] for t in cycle["targets"]}  # texto original
    assert (predictions["value"] > 0).all()


def test_build_predictions_one_horizon_only_like_the_practice_cycle(trained):
    wide, model = trained
    cycle = _cycle(wide, horizons=(1,))
    predictions, _ = build_predictions(model, wide, cycle)
    assert len(predictions) == 3 and set(predictions["horizon_minutes"]) == {15}


def test_build_predictions_falls_back_to_profile_for_unsupported_horizons(trained):
    wide, model = trained
    cycle = _cycle(wide, horizons=(1, 6))  # +90 min no lo cubre el modelo
    predictions, fallback = build_predictions(model, wide, cycle)
    assert fallback == 3 and len(predictions) == 6 and (predictions["value"] > 0).all()


def test_build_predictions_rejects_history_beyond_the_cutoff(trained):
    wide, model = trained
    cycle = _cycle(wide.iloc[:-8])
    with pytest.raises(PredictionError, match="posteriores"):
        build_predictions(model, wide, cycle)


def test_build_predictions_handles_missing_cutoff_observation(trained):
    wide, model = trained
    cycle = _cycle(wide)
    predictions, _ = build_predictions(model, wide.iloc[:-1], cycle)  # falta la fila del corte
    assert len(predictions) == 12 and np.isfinite(predictions["value"]).all()


def _predictions_for(cycle):
    return pd.DataFrame([{"station_id": t["station_id"], "target_at": t["target_at"],
                          "horizon_minutes": t["horizon_minutes"], "value": 10.0}
                         for t in cycle["targets"]])


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p.iloc[1:], "faltan"),
    (lambda p: pd.concat([p, p.iloc[[0]]]), "duplicados"),
    (lambda p: p.assign(value=p["value"].where(p.index != 0, -1.0)), "negativas"),
    (lambda p: p.assign(value=p["value"].where(p.index != 0, np.nan)), "no finitas"),
    (lambda p: p.assign(value=p["value"].where(p.index != 0, np.inf)), "no finitas"),
    (lambda p: p.assign(station_id=p["station_id"].where(p.index != 0, "99999")), "sobran"),
])
def test_validate_submission_rejects_bad_sets(trained, mutate, message):
    wide, _ = trained
    cycle = _cycle(wide)
    validate_submission(_predictions_for(cycle), cycle)  # el bueno pasa
    with pytest.raises(PredictionError, match=message):
        validate_submission(mutate(_predictions_for(cycle)), cycle)


def test_payload_follows_the_contract_exactly(trained):
    wide, _ = trained
    cycle = _cycle(wide)
    payload = build_payload(cycle, _predictions_for(cycle), model_version="v1",
                            trained_at="2026-09-18T10:00:00+00:00",
                            training_data_end=cycle["data_cutoff"], git_commit="e" * 40,
                            client_run_id="github-1-1")
    assert set(payload) == {"schema_version", "cycle_id", "client_run_id", "data_cutoff", "model",
                            "predictions"}
    assert payload["schema_version"] == "1.0" and payload["data_cutoff"] == cycle["data_cutoff"]
    assert set(payload["model"]) == {"version", "trained_at", "training_data_end", "git_commit"}
    assert all(set(p) == {"station_id", "target_at", "value"} for p in payload["predictions"])
    no_commit = build_payload(cycle, _predictions_for(cycle), model_version="v1", trained_at="t",
                              training_data_end=cycle["data_cutoff"], git_commit=None,
                              client_run_id="x")
    assert "git_commit" not in no_commit["model"]
    assert payload_hash(payload) == payload_hash(dict(reversed(list(payload.items()))))


def test_payload_refuses_a_model_trained_after_the_cutoff(trained):
    wide, _ = trained
    cycle = _cycle(wide)
    with pytest.raises(PredictionError, match="posteriores"):
        build_payload(cycle, _predictions_for(cycle), model_version="v", trained_at="t",
                      training_data_end="2099-01-01T00:00:00Z", git_commit=None, client_run_id="x")


@pytest.fixture
def world(trained):
    """Base con historia y champion, servidor de API con ciclo abierto, y cliente."""
    wide, _ = trained
    db = FakeDb()
    db.load_observations(wide)
    registry = ModelRegistry(db)
    train_and_register(wide, registry, config=FAST, rules=LENIENT, git_commit="f" * 40)
    server = FakeServer()
    server.cycle = _cycle(wide)
    api = PulsoApi("https://api.test", "SECRETKEY", transport=httpx.MockTransport(server.handler),
                   sleep=lambda s: None)
    now = pd.Timestamp(server.cycle["opens_at"]).to_pydatetime() + pd.Timedelta(minutes=7)
    return db, registry, server, api, now, wide


def test_inference_submits_and_records_evidence(world):
    db, registry, server, api, now, wide = world
    result = run_inference(api, db, registry, now=now)
    assert result["status"] == "submitted" and result["n_predictions"] == 12
    sub = db.tables["submissions"][0]
    assert sub["status"] == "accepted" and sub["submission_id"] == "sub_1" and sub["is_official"]
    assert sub["idempotency_key"] and sub["model_version"] == registry.champion()["version"]
    assert len(db.tables["predictions"]) == 12
    assert db.tables["forecast_cycles"][0]["outcome"] == "submitted"
    sent = [c for c in server.calls if c.method == "POST"]
    assert len(sent) == 1 and sent[0].headers["idempotency-key"] == sub["idempotency_key"]


def test_inference_skips_when_there_is_no_open_cycle(world):
    db, registry, server, api, now, _ = world
    server.cycle = None
    assert run_inference(api, db, registry, now=now) == {"status": "skipped", "reason": "no_open_cycle"}
    assert not db.tables.get("submissions")


def test_inference_does_not_resubmit_an_accepted_cycle(world):
    db, registry, server, api, now, _ = world
    run_inference(api, db, registry, now=now)
    again = run_inference(api, db, registry, now=now)
    assert again["reason"] == "already_submitted"
    assert len([c for c in server.calls if c.method == "POST"]) == 1


def test_inference_skips_closed_cycle_without_calling_submit(world):
    db, registry, server, api, _, _ = world
    late = pd.Timestamp(server.cycle["closes_at"]).to_pydatetime() + pd.Timedelta(minutes=1)
    assert run_inference(api, db, registry, now=late)["reason"] == "cycle_closed"
    assert not [c for c in server.calls if c.method == "POST"]
    assert db.tables["forecast_cycles"][0]["outcome"] == "missed"


def test_inference_dry_run_writes_nothing(world):
    db, registry, server, api, now, _ = world
    before = db.writes
    result = run_inference(api, db, registry, dry_run=True, now=now)
    assert result["status"] == "dry_run" and db.writes == before
    assert not [c for c in server.calls if c.method == "POST"]


def test_inference_records_api_rejection_and_raises(world):
    db, registry, server, api, now, _ = world
    server.fail_with = [httpx.Response(200, json=server.cycle),  # current_cycle
                        httpx.Response(422, json={"detail": {"code": "invalid_target_set",
                                                             "message": "faltan"}},
                                       headers={"x-request-id": "r7"})]
    with pytest.raises(PulsoApiError) as info:
        run_inference(api, db, registry, now=now)
    assert info.value.code == "invalid_target_set"
    sub = db.tables["submissions"][0]
    assert sub["status"] == "rejected" and sub["http_status"] == 422 and sub["request_id"] == "r7"
    assert db.tables["forecast_cycles"][0]["outcome"] == "error"
    assert len(db.tables["predictions"]) == 12  # la evidencia se guardó antes de enviar


def test_inference_marks_missed_when_api_says_cycle_closed(world):
    db, registry, server, api, now, _ = world
    server.fail_with = [httpx.Response(200, json=server.cycle),
                        httpx.Response(409, json={"detail": {"code": "cycle_closed", "message": "x"}})]
    with pytest.raises(PulsoApiError):
        run_inference(api, db, registry, now=now)
    assert db.tables["forecast_cycles"][0]["outcome"] == "missed"


def test_inference_retry_after_failure_uses_a_fresh_row_and_can_succeed(world, monkeypatch):
    db, registry, server, api, now, _ = world
    server.fail_with = [httpx.Response(200, json=server.cycle), httpx.Response(422, json={"detail": {"code": "x"}})]
    with pytest.raises(PulsoApiError):
        run_inference(api, db, registry, now=now)
    monkeypatch.setenv("GITHUB_RUN_ID", "999")  # otra ejecución -> otra llave
    result = run_inference(api, db, registry, now=now)
    assert result["status"] == "submitted"
    statuses = [s["status"] for s in db.tables["submissions"]]
    assert statuses == ["rejected", "accepted"]


def test_inference_refuses_a_champion_trained_after_the_cutoff(world):
    db, registry, server, api, _, wide = world
    server.cycle = _cycle(wide.iloc[:-96])  # ciclo más viejo que el entrenamiento del champion
    now = pd.Timestamp(server.cycle["opens_at"]).to_pydatetime() + pd.Timedelta(minutes=7)
    with pytest.raises(PredictionError, match="posteriores"):
        run_inference(api, db, registry, now=now)
    assert not [c for c in server.calls if c.method == "POST"]


def test_inference_without_champion_fails_clearly(world):
    db, _, server, api, now, _ = world
    db.tables["model_versions"] = []
    with pytest.raises(RegistryError, match="champion"):
        run_inference(api, db, ModelRegistry(db), now=now)


# ------------------------------------------------------------------ bitácora
def test_pipeline_run_records_success_skip_and_error(fake_db):
    with pipeline_run(fake_db, "predict") as run:
        run.summary["x"] = 1
    with pipeline_run(fake_db, "predict") as run:
        run.skip("no_open_cycle")
    with pytest.raises(ValueError), pipeline_run(fake_db, "train"):
        raise ValueError("boom")
    rows = fake_db.tables["pipeline_runs"]
    assert [r["status"] for r in rows] == ["success", "skipped", "error"]
    assert rows[0]["summary"] == {"x": 1} and "ValueError: boom" in rows[2]["error"]


# ------------------------------------------------------------------ trabajo de monitoreo
from pulso.monitor_job import load_scored, run_monitor  # noqa: E402

CHAMPION_DAYS = 30
STATIONS6 = ["02300", "03000", "05000", "05100", "06000", "06111"]


def six_station_wide(days: int) -> pd.DataFrame:
    frames = [synthetic_wide(days=days, seed=i).iloc[:, [i % 3]].set_axis([s], axis=1)
              for i, s in enumerate(STATIONS6)]
    return pd.concat(frames, axis=1)


@pytest.fixture(scope="module")
def competition():
    """Champion entrenado con 30 días y 2 días de «competencia» con predicciones oficiales."""
    full = six_station_wide(days=CHAMPION_DAYS + 2)
    history = full.iloc[:CHAMPION_DAYS * 96]
    return full, history


def _build_world(competition, *, actual_transform=None, trained_at="2026-09-01T00:00:00+00:00"):
    full, history = competition
    db = FakeDb()
    db.load_observations(history)
    registry = ModelRegistry(db)
    train_and_register(history, registry, config=FAST, rules=LENIENT, git_commit="9" * 40)
    db.tables["model_versions"][0]["trained_at"] = trained_at
    _, model = registry.load_champion()

    # observaciones nuevas (la «competencia») y predicciones oficiales para cada hora en punto
    new = full.iloc[CHAMPION_DAYS * 96:].copy()
    if actual_transform:
        new = actual_transform(new)
    combined = pd.concat([history, new])
    db.load_observations(combined)
    origins = [i for i in range(CHAMPION_DAYS * 96, len(full) - 4) if full.index[i].minute == 0]
    preds = model.predict_batch(full, np.array(origins))
    preds["station_id"] = [model.stations[i] for i in preds["station_idx"]]
    for origin, group in preds.groupby("origin_idx"):
        cycle_id = f"cyc_{origin}"
        db.tables.setdefault("forecast_cycles", []).append({
            "cycle_id": cycle_id, "outcome": "submitted",
            "closes_at": full.index[origin].tz_convert("UTC").isoformat()})
        sub = db.insert("submissions", {"cycle_id": cycle_id, "client_run_id": "r",
                                        "idempotency_key": f"k{origin}", "status": "accepted",
                                        "is_official": True, "model_version": registry.champion()["version"]})
        db.upsert("predictions", [
            {"submission_row_id": sub["id"], "cycle_id": cycle_id, "station_id": r.station_id,
             "target_at": pd.Timestamp(r.target_at).tz_convert("UTC").isoformat(),
             "horizon_minutes": int(r.horizon) * 15, "value": float(r.prediction)}
            for r in group.itertuples()], "submission_row_id,station_id,target_at")
    return db, registry, full


NOW = pd.Timestamp("2026-10-01", tz="UTC").to_pydatetime()


def test_monitor_healthy_system_keeps_and_records_metrics(competition):
    db, registry, _ = _build_world(competition)
    result = run_monitor(db, registry, now=NOW)
    assert result["decision"] == "keep" and not any(result["signals"].values())
    snaps = {s["window_name"]: s for s in db.tables["metric_snapshots"]}
    assert set(snaps) == {"rolling_24h", "cumulative"}
    assert 75 < snaps["rolling_24h"]["accuracy"] <= 100 and snaps["rolling_24h"]["baseline_accuracy"] > 0
    assert set(snaps["cumulative"]["by_horizon"]) == {"15", "30", "45", "60"}
    kinds = [(d["kind"], d["name"]) for d in db.tables["drift_signals"]]
    assert kinds.count(("data", "level_shift")) == 6 and ("performance", "rolling_24h_accuracy") in kinds
    decision = db.tables["retrain_decisions"][0]
    assert decision["decision"] == "keep" and decision["evidence"]["rolling_24h_accuracy"] is not None


def test_monitor_without_predictions_still_reports_and_does_not_crash(competition):
    db, registry, _ = _build_world(competition)
    db.tables["predictions"], db.tables["submissions"] = [], []
    result = run_monitor(db, registry, now=NOW)
    assert result["decision"] == "keep" and not db.tables.get("metric_snapshots")
    assert result["rolling_24h_accuracy"] is None


def test_monitor_persistent_degradation_triggers_retrain(competition):
    def halve_demand(new):
        new = new.copy()
        new.iloc[-2 * 96:] *= 0.4  # todo cae en el último día y medio
        return new

    db, registry, _ = _build_world(competition, actual_transform=halve_demand)
    version = registry.champion()["version"]
    for _ in range(3):  # 3 evaluaciones previas ya degradadas del mismo modelo
        db.insert("metric_snapshots", {"window_name": "rolling_24h", "model_version": version,
                                       "accuracy": 40.0})
    result = run_monitor(db, registry, now=NOW)
    assert result["decision"] == "retrain" and result["signals"]["performance"]
    assert db.tables["retrain_decisions"][-1]["decision"] == "retrain"


def test_monitor_level_shift_needs_persistence_then_triggers(competition):
    def raise_one_station(new):
        new = new.copy()
        new["03000"] = new["03000"] * 1.8
        return new

    db, registry, _ = _build_world(competition, actual_transform=raise_one_station)
    first = run_monitor(db, registry, now=NOW)
    streak1 = next(d for d in db.tables["drift_signals"]
                   if d["name"] == "level_shift" and d["station_id"] == "03000")["details"]["streak"]
    assert streak1 == 1 and not first["signals"]["data"]  # una sola evaluación: no basta
    for _ in range(10):
        run_monitor(db, registry, now=NOW)
    last = run_monitor(db, registry, now=NOW)  # 12.ª evaluación consecutiva
    assert last["signals"]["data"] and last["decision"] == "retrain"
    assert last["level_streaks"]["03000"] == 12
    assert "02300" not in last["level_streaks"]  # las demás estaciones no se ven afectadas


def test_monitor_operational_failures_take_priority(competition):
    def collapse(new):
        new = new.copy()
        new.iloc[-2 * 96:] *= 0.4
        return new

    db, registry, _ = _build_world(competition, actual_transform=collapse)
    version = registry.champion()["version"]
    for _ in range(3):
        db.insert("metric_snapshots", {"window_name": "rolling_24h", "model_version": version,
                                       "accuracy": 40.0})
    for _ in range(3):  # 3 ingestas fallidas en 24 h
        db.insert("ingestion_runs", {"source": "stream", "status": "error",
                                     "started_at": (pd.Timestamp(NOW) - pd.Timedelta(hours=2)).isoformat()})
    result = run_monitor(db, registry, now=NOW)
    assert result["decision"] == "investigate" and result["signals"]["operational"]


def test_monitor_low_coverage_is_an_operational_signal(competition):
    db, registry, _ = _build_world(competition)
    for cycle in db.tables["forecast_cycles"][:8]:
        cycle["outcome"] = "missed"
    result = run_monitor(db, registry, now=NOW)
    assert result["signals"]["operational"] and result["coverage"] < 0.95


def test_monitor_uses_only_official_predictions(competition):
    db, registry, _ = _build_world(competition)
    for sub in db.tables["submissions"]:
        sub["is_official"] = False
    assert load_scored(db).empty


def test_monitor_cooldown_after_recent_training(competition):
    def collapse(new):
        new = new.copy()
        new.iloc[-2 * 96:] *= 0.4
        return new

    recent = (pd.Timestamp(NOW) - pd.Timedelta(hours=1)).isoformat()
    db, registry, _ = _build_world(competition, actual_transform=collapse, trained_at=recent)
    version = registry.champion()["version"]
    for _ in range(3):
        db.insert("metric_snapshots", {"window_name": "rolling_24h", "model_version": version,
                                       "accuracy": 40.0})
    result = run_monitor(db, registry, now=NOW)
    assert result["decision"] == "investigate" and "enfriamiento" in result["reason"]


# ------------------------------------------------------------------ espera dentro de la ventana
from pulso.predict import run_inference_waiting  # noqa: E402


class FakeClock:
    """Reloj monótono falso: sólo avanza cuando se 'duerme'."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    def now(self) -> float:
        return self.t


def test_waiting_submits_immediately_when_a_cycle_is_open(world):
    db, registry, server, api, now, _ = world
    clock = FakeClock()
    result = run_inference_waiting(api, db, registry, wait_seconds=2700, sleep=clock.sleep,
                                   clock=clock.now, now_fn=lambda: now)
    assert result["status"] == "submitted" and clock.sleeps == []  # no esperó


def test_waiting_polls_until_the_cycle_opens_then_submits(world):
    db, registry, server, api, now, wide = world
    pending = _cycle(wide)
    server.cycle = None  # aún no abre

    clock = FakeClock()
    original_sleep = clock.sleep

    def sleep_and_open(seconds):
        original_sleep(seconds)
        if len(clock.sleeps) == 3:  # al tercer intento abre la ventana
            server.cycle = pending

    result = run_inference_waiting(api, db, registry, wait_seconds=2700, poll_seconds=60,
                                   sleep=sleep_and_open, clock=clock.now, now_fn=lambda: now)
    assert result["status"] == "submitted" and result["waited_polls"] == 4
    assert clock.sleeps == [60, 60, 60]
    assert db.tables["submissions"][0]["status"] == "accepted"


def test_waiting_gives_up_at_the_deadline_without_failing(world):
    db, registry, server, api, _, _ = world
    server.cycle = None
    clock = FakeClock()
    result = run_inference_waiting(api, db, registry, wait_seconds=300, poll_seconds=60,
                                   sleep=clock.sleep, clock=clock.now)
    assert result == {"status": "skipped", "reason": "no_open_cycle", "waited_polls": 6}
    assert sum(clock.sleeps) == 300  # no se pasa del plazo
    assert not db.tables.get("submissions")


def test_waiting_does_not_wait_when_the_cycle_was_already_submitted(world):
    db, registry, server, api, now, _ = world
    run_inference(api, db, registry, now=now)  # primera entrega
    clock = FakeClock()
    result = run_inference_waiting(api, db, registry, wait_seconds=2700, sleep=clock.sleep,
                                   clock=clock.now, now_fn=lambda: now)
    assert result["reason"] == "already_submitted" and clock.sleeps == []


def test_waiting_zero_behaves_like_a_single_attempt(world):
    db, registry, server, api, _, _ = world
    server.cycle = None
    clock = FakeClock()
    result = run_inference_waiting(api, db, registry, wait_seconds=0, sleep=clock.sleep,
                                   clock=clock.now)
    assert result == {"status": "skipped", "reason": "no_open_cycle"} and clock.sleeps == []


def test_waiting_never_sleeps_past_the_deadline(world):
    db, registry, server, api, _, _ = world
    server.cycle = None
    clock = FakeClock()
    run_inference_waiting(api, db, registry, wait_seconds=150, poll_seconds=60,
                          sleep=clock.sleep, clock=clock.now)
    assert clock.sleeps == [60, 60, 30]  # el último se recorta para no pasarse


def test_waiting_rereads_the_wall_clock_so_a_closing_window_is_noticed(world):
    """Mientras se espera, la ventana puede cerrarse: la hora debe releerse en cada intento."""
    db, registry, server, api, now, _ = world
    pending = server.cycle
    server.cycle = None
    clock = FakeClock()

    def moving_now():  # la hora de pared avanza junto con el reloj monótono
        return now + timedelta(seconds=clock.t)

    def sleep_then_open(seconds):
        clock.sleep(seconds)
        server.cycle = pending  # abre, pero para entonces su closes_at ya pasó

    result = run_inference_waiting(api, db, registry, wait_seconds=7200, poll_seconds=3600,
                                   sleep=sleep_then_open, clock=clock.now, now_fn=moving_now)
    # Con la hora congelada habría enviado; al releerla, ve la ventana cerrada.
    assert result["reason"] == "cycle_closed"
    assert not db.tables.get("submissions")


def test_sync_runs_once_just_before_building_the_prediction(world):
    """El sincronizador corre solo cuando se va a entregar, y una sola vez."""
    db, registry, server, api, now, _ = world
    calls = []
    result = run_inference(api, db, registry, now=now, sync=lambda: calls.append("sync"))
    assert result["status"] == "submitted" and calls == ["sync"]


@pytest.mark.parametrize("setup, reason", [
    (lambda server, db, registry, api, now: setattr(server, "cycle", None), "no_open_cycle"),
    (lambda server, db, registry, api, now: run_inference(api, db, registry, now=now),
     "already_submitted"),
])
def test_sync_does_not_run_when_there_is_nothing_to_submit(world, setup, reason):
    db, registry, server, api, now, _ = world
    setup(server, db, registry, api, now)
    calls = []
    result = run_inference(api, db, registry, now=now, sync=lambda: calls.append("sync"))
    assert result["reason"] == reason and calls == []


def test_sync_does_not_run_for_a_closed_window(world):
    db, registry, server, api, _, _ = world
    late = pd.Timestamp(server.cycle["closes_at"]).to_pydatetime() + timedelta(minutes=1)
    calls = []
    result = run_inference(api, db, registry, now=late, sync=lambda: calls.append("sync"))
    assert result["reason"] == "cycle_closed" and calls == []


def test_waiting_passes_the_sync_through_on_the_attempt_that_submits(world):
    db, registry, server, api, now, wide = world
    pending = server.cycle
    server.cycle = None
    clock = FakeClock()
    calls = []

    def sleep_then_open(seconds):
        clock.sleep(seconds)
        server.cycle = pending

    result = run_inference_waiting(api, db, registry, wait_seconds=600, poll_seconds=60,
                                   sleep=sleep_then_open, clock=clock.now, now_fn=lambda: now,
                                   sync=lambda: calls.append("sync"))
    assert result["status"] == "submitted"
    assert calls == ["sync"]  # no se sincroniza en los intentos sin ciclo


# ------------------------------------------------------------------ sesión de varios ciclos
from pulso.predict import run_session  # noqa: E402


def test_session_submits_every_cycle_that_opens(world):
    """Un solo job cubre varios ciclos consecutivos, no solo el primero."""
    db, registry, server, api, now, wide = world
    opened = []

    def next_cycle(n):
        c = _cycle(wide, cycle_id=f"cyc_{n}")
        opened.append(c["cycle_id"])
        return c

    server.cycle = next_cycle(1)
    clock = FakeClock()

    def sleep_and_rotate(seconds):
        clock.sleep(seconds)
        # cada 2 minutos de espera abre un ciclo nuevo
        if len(clock.sleeps) in (2, 4):
            server.cycle = next_cycle(len(clock.sleeps))

    result = run_session(api, db, registry, duration_seconds=300, poll_seconds=60,
                         sleep=sleep_and_rotate, clock=clock.now, now_fn=lambda: now)
    assert result["status"] == "session"
    assert result["cycles_submitted"] == 3
    assert result["cycles"] == ["cyc_1", "cyc_2", "cyc_4"]
    assert len(db.tables["submissions"]) == 3


def test_session_stops_at_the_deadline(world):
    db, registry, server, api, now, _ = world
    server.cycle = None
    clock = FakeClock()
    result = run_session(api, db, registry, duration_seconds=180, poll_seconds=60,
                         sleep=clock.sleep, clock=clock.now, now_fn=lambda: now)
    assert result["cycles_submitted"] == 0
    assert clock.t == 180  # no se pasa del plazo


def test_session_survives_a_transient_error_and_keeps_covering(world):
    db, registry, server, api, now, wide = world
    clock = FakeClock()
    calls = {"n": 0}
    real_cycle = server.cycle

    def flaky_current_cycle():
        calls["n"] += 1
        if calls["n"] == 1:
            raise PulsoApiError(503, None, "caída temporal", "req-x", "/v1/forecast-cycles/current")
        return real_cycle

    api.current_cycle = flaky_current_cycle
    result = run_session(api, db, registry, duration_seconds=120, poll_seconds=60,
                         sleep=clock.sleep, clock=clock.now, now_fn=lambda: now)
    assert result["cycles_submitted"] == 1  # se recuperó tras el fallo
    assert "last_error" in result and "503" in result["last_error"]


def test_session_gives_up_after_too_many_consecutive_errors(world):
    db, registry, server, api, now, _ = world
    clock = FakeClock()

    def always_failing():
        raise PulsoApiError(500, None, "roto", None, "/v1/forecast-cycles/current")

    api.current_cycle = always_failing
    with pytest.raises(PulsoApiError):
        run_session(api, db, registry, duration_seconds=3600, poll_seconds=60,
                    max_consecutive_errors=3, sleep=clock.sleep, clock=clock.now,
                    now_fn=lambda: now)
