"""Pruebas de features, métrica, backtest y modelo. La más importante: no hay fuga del futuro."""

import numpy as np
import pandas as pd
import pytest

from pulso.backtest import Fold, hourly_origins, make_folds, run_backtest, score_predictions
from pulso.baselines import LastValue, ProfileMean, SeasonalNaive
from pulso.features import (
    FEATURE_COLUMNS,
    HORIZONS,
    Profile,
    build_features,
    extend_grid,
    how_index,
    to_wide,
)
from pulso.metrics import official_accuracy, station_accuracy
from pulso.model import GbmResidualModel, ModelConfig

TZ = "America/Bogota"
FAST = ModelConfig(max_iter=15, min_samples_leaf=20)


def synthetic_wide(days: int = 24, seed: int = 0) -> pd.DataFrame:
    """Tres estaciones con patrón semanal + ruido multiplicativo."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-08-03", periods=days * 96, freq="15min", tz=TZ)  # lunes
    slot = idx.hour * 4 + idx.minute // 15
    daily = 1.2 + np.sin(2 * np.pi * (slot - 28) / 96)
    weekend = np.where(idx.dayofweek >= 5, 0.6, 1.0)
    cols = {}
    for i, station in enumerate(["02300", "03000", "05000"]):
        level = 200 * (i + 1)
        cols[station] = level * daily * weekend * rng.lognormal(0, 0.1, len(idx))
    return pd.DataFrame(cols, index=idx).round()


# ------------------------------------------------------------------ perfil
def test_profile_leave_one_out_matches_bruteforce():
    wide = synthetic_wide(days=21)
    times, log = wide.index, np.log(wide.to_numpy())
    profile = Profile().fit(times, log)
    loo = profile.matrix(times, log)
    how = how_index(times)
    for t in (5, 300, 1000, 2000):
        same = np.flatnonzero((how == how[t]) & (np.arange(len(times)) != t))
        assert np.allclose(loo[t], log[same].mean(axis=0))


def test_profile_outside_fit_window_uses_plain_mean():
    wide = synthetic_wide(days=21)
    times, log = wide.index, np.log(wide.to_numpy())
    profile = Profile().fit(times[:14 * 96], log[:14 * 96])
    matrix = profile.matrix(times, log)
    plain = profile.matrix(times[14 * 96:])
    assert np.allclose(matrix[14 * 96:], plain)


def test_profile_half_life_weights_recent_more():
    wide = synthetic_wide(days=21)
    log = np.log(wide.to_numpy())
    log[-7 * 96:] += 1.0  # la última semana sube ~e veces
    flat = Profile().fit(wide.index, log).matrix(wide.index[:1])
    recent = Profile(half_life_days=3).fit(wide.index, log).matrix(wide.index[:1])
    assert (recent > flat).all()


# ------------------------------------------------------------------ sin fuga del futuro
def _perturb_after(wide: pd.DataFrame, cut: int, seed: int = 1) -> pd.DataFrame:
    noisy = wide.copy()
    rng = np.random.default_rng(seed)
    noisy.iloc[cut + 1:] = rng.uniform(1, 5000, size=noisy.iloc[cut + 1:].shape)
    return noisy


def test_features_ignore_everything_after_the_origin():
    wide = synthetic_wide(days=24)
    train = wide.iloc[:18 * 96]
    profile = Profile().fit(train.index, np.log(train.to_numpy()))
    origins = np.array([20 * 96 + 8, 21 * 96 + 40, 22 * 96 + 60])  # fuera de la ventana de ajuste
    for h in (1, 4):
        for origin in origins:
            base = build_features(profile, wide.index, wide.to_numpy(), np.array([origin]), h)
            other = build_features(profile, wide.index, _perturb_after(wide, origin).to_numpy(),
                                   np.array([origin]), h)
            pd.testing.assert_frame_equal(base.X, other.X)
            assert np.array_equal(base.base, other.base)


@pytest.mark.parametrize("model_cls", [GbmResidualModel])
def test_predictions_ignore_everything_after_the_origin(model_cls):
    wide = synthetic_wide(days=24)
    model = model_cls(FAST).fit(wide.iloc[:18 * 96])
    origins = np.array([19 * 96, 20 * 96 + 20, 21 * 96 + 44, 22 * 96 + 88])
    reference = model.predict_batch(wide, origins)
    for cut_origin in origins:
        noisy = _perturb_after(wide, cut_origin)
        again = model.predict_batch(noisy, np.array([cut_origin]))
        expected = reference[reference["origin_idx"] == cut_origin]
        assert np.allclose(again["prediction"].to_numpy(), expected["prediction"].to_numpy())


def test_baselines_ignore_everything_after_the_origin():
    wide = synthetic_wide(days=24)
    train = wide.iloc[:18 * 96]
    origin = 20 * 96 + 8
    noisy = _perturb_after(wide, origin)
    for baseline in (LastValue(), SeasonalNaive(96), SeasonalNaive(672), ProfileMean()):
        baseline.fit(train)
        a = baseline.predict_batch(wide, np.array([origin]))["prediction"].to_numpy()
        b = baseline.predict_batch(noisy, np.array([origin]))["prediction"].to_numpy()
        assert np.allclose(a, b, equal_nan=True), baseline.name


def test_training_profile_excludes_the_row_being_predicted():
    """El perfil del instante objetivo no incluye su propio valor (leave-one-out)."""
    wide = synthetic_wide(days=21)
    origin, h = 10 * 96, 2
    tweaked = wide.copy()
    tweaked.iloc[origin + h] = 9999.0

    def features(frame):
        profile = Profile().fit(frame.index, np.log(frame.to_numpy()))  # se reajusta con esos datos
        return build_features(profile, frame.index, frame.to_numpy(), np.array([origin]), h)

    base, other = features(wide), features(tweaked)
    assert np.allclose(base.base, other.base)  # p_target no depende del valor a predecir
    assert not np.allclose(base.target, other.target)  # pero el objetivo sí cambia


def test_seasonal_naive_rejects_period_that_would_peek_into_the_future():
    wide = synthetic_wide(days=5)
    with pytest.raises(ValueError, match="futura"):
        SeasonalNaive(2).predict_batch(wide, np.array([200]))


# ------------------------------------------------------------------ modelo
def test_predict_next_shape_and_targets():
    wide = synthetic_wide(days=24)
    model = GbmResidualModel(FAST).fit(wide.iloc[:20 * 96])
    history = wide.iloc[:22 * 96]
    out = model.predict_next(history)
    assert len(out) == 3 * 4
    assert set(out["horizon_minutes"]) == {15, 30, 45, 60}
    cutoff = history.index[-1]
    assert set(out["target_at"]) == {cutoff + pd.Timedelta(minutes=m) for m in (15, 30, 45, 60)}
    assert (out["value"] >= 0).all() and np.isfinite(out["value"]).all()


def test_predict_next_matches_batch_prediction():
    wide = synthetic_wide(days=24)
    model = GbmResidualModel(FAST).fit(wide.iloc[:20 * 96])
    history = wide.iloc[:22 * 96]
    nxt = model.predict_next(history).sort_values(["station_id", "horizon_minutes"])
    # La rejilla se extiende un paso más que el horizonte máximo: `p_curv` mira el perfil en
    # objetivo+1, así que con solo max(HORIZONS) filas la última quedaría recortada.
    batch = model.predict_batch(extend_grid(history, max(HORIZONS) + 1),
                                np.array([len(history) - 1]))
    batch["station_id"] = [model.stations[i] for i in batch["station_idx"]]
    batch = batch.sort_values(["station_id", "horizon"])
    assert np.allclose(nxt["value"].to_numpy(), batch["prediction"].clip(lower=0).to_numpy())


def test_longest_horizon_needs_the_extra_grid_row():
    """Con la rejilla justa, la curvatura del horizonte más largo se recorta y cambia el valor.

    Es el error que tendría producción si `predict_next` extendiera solo `max(HORIZONS)` filas:
    los tres primeros horizontes saldrían bien y solo el de 60 min quedaría mal, que es la clase
    de fallo que no se nota a simple vista.
    """
    wide = synthetic_wide(days=24)
    model = GbmResidualModel(FAST).fit(wide.iloc[:20 * 96])
    history = wide.iloc[:22 * 96]
    origin = np.array([len(history) - 1])
    justa = model.predict_batch(extend_grid(history, max(HORIZONS)), origin)
    holgada = model.predict_batch(extend_grid(history, max(HORIZONS) + 1), origin)
    corto = justa["horizon"] < max(HORIZONS)
    assert np.allclose(justa[corto]["prediction"].to_numpy(),
                       holgada[corto]["prediction"].to_numpy())
    largo = justa["horizon"] == max(HORIZONS)
    assert not np.allclose(justa[largo]["prediction"].to_numpy(),
                           holgada[largo]["prediction"].to_numpy())


def test_missing_last_observation_falls_back_to_the_profile():
    wide = synthetic_wide(days=24)
    model = GbmResidualModel(FAST).fit(wide.iloc[:20 * 96])
    history = wide.iloc[:22 * 96].copy()
    history.iloc[-1] = np.nan  # la observación del corte aún no llegó
    out = model.predict_next(history).sort_values(["station_id", "horizon_minutes"])
    ext = extend_grid(history, 4)
    origin = len(history) - 1
    expected = []
    for station_idx in range(3):
        for h in (1, 2, 3, 4):
            p = model.profile.matrix(ext.index[[origin + h]])[0, station_idx]
            expected.append(np.exp(p))
    assert np.allclose(out["value"].to_numpy(), expected)


def test_model_roundtrip_gives_identical_predictions():
    wide = synthetic_wide(days=24)
    model = GbmResidualModel(FAST).fit(wide.iloc[:20 * 96])
    restored = GbmResidualModel.from_bytes(model.to_bytes())
    history = wide.iloc[:22 * 96]
    assert np.array_equal(model.predict_next(history)["value"], restored.predict_next(history)["value"])


def test_predict_rejects_wrong_stations():
    wide = synthetic_wide(days=24)
    model = GbmResidualModel(FAST).fit(wide.iloc[:20 * 96])
    with pytest.raises(ValueError, match="estaciones"):
        model.predict_batch(extend_grid(wide[["02300", "03000"]], 4), np.array([100]))


def test_feature_columns_are_all_present_and_numeric():
    wide = synthetic_wide(days=10)
    profile = Profile().fit(wide.index, np.log(wide.to_numpy()))
    fs = build_features(profile, wide.index, wide.to_numpy(), np.arange(96, 200), 3)
    assert list(fs.X.columns) == FEATURE_COLUMNS
    assert len(fs.X) == (200 - 96) * 3
    assert fs.X["horizon"].eq(3).all()


# ------------------------------------------------------------------ métrica oficial
def test_official_accuracy_matches_definition_and_is_unweighted():
    # Estación A (grande) casi perfecta, B (pequeña) con 50 % de error: promedio simple
    actual = pd.Series([1000.0, 1000.0, 10.0, 10.0])
    pred = pd.Series([1000.0, 1000.0, 5.0, 15.0])
    station = pd.Series(["A", "A", "B", "B"])
    per = station_accuracy(actual, pred, station)
    assert per["A"] == pytest.approx(100.0) and per["B"] == pytest.approx(50.0)
    assert official_accuracy(actual, pred, station) == pytest.approx(75.0)


def test_missing_prediction_counts_as_zero_and_accuracy_floors_at_zero():
    actual = pd.Series([10.0, 10.0])
    station = pd.Series(["A", "A"])
    assert official_accuracy(actual, pd.Series([np.nan, np.nan]), station) == 0.0
    assert official_accuracy(actual, pd.Series([100.0, 100.0]), station) == 0.0  # WAPE > 1


# ------------------------------------------------------------------ folds y backtest
def test_folds_are_temporal_disjoint_and_contiguous():
    idx = pd.date_range("2026-07-26", periods=4320, freq="15min", tz=TZ)
    folds = make_folds(idx)
    assert len(folds) == 3
    for fold in folds:
        assert fold.train_end < fold.val_start <= fold.val_end
        assert (fold.val_end - fold.val_start) == pd.Timedelta(days=7) - pd.Timedelta(minutes=15)
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.val_end < b.val_start and b.train_end == a.val_end
    assert folds[-1].val_end == idx[-1]


def test_hourly_origins_are_on_the_hour_and_targets_stay_inside_the_window():
    idx = pd.date_range("2026-07-26", periods=4320, freq="15min", tz=TZ)
    fold = make_folds(idx)[-1]
    origins = hourly_origins(idx, fold)
    assert (idx[origins].minute == 0).all()
    assert idx[origins].min() >= fold.val_start
    assert idx[origins + 4].max() <= fold.val_end
    assert len(origins) == 7 * 24 - 1  # el último origen posible deja +60 dentro de la ventana


def test_backtest_trains_only_on_the_past():
    wide = synthetic_wide(days=24)
    seen = {}

    class Spy(LastValue):
        def fit(self, w):
            seen.setdefault("ends", []).append(w.index[-1])
            return self

    folds = make_folds(wide.index, n_folds=2, val_days=4)
    table, _ = run_backtest({"spy": Spy}, wide, folds)
    assert seen["ends"] == [f.train_end for f in folds]
    assert len(table) == 2


def test_score_predictions_aligns_actual_with_target_time():
    wide = synthetic_wide(days=5)
    origin = 200
    pred = LastValue().predict_batch(wide, np.array([origin]))
    scored = score_predictions(wide, pred)
    row = scored[(scored["station_id"] == "03000") & (scored["horizon"] == 3)].iloc[0]
    assert row["actual"] == wide.iloc[origin + 3]["03000"]
    assert row["prediction"] == wide.iloc[origin]["03000"]


def test_to_wide_builds_full_grid_and_keeps_station_text():
    obs = pd.DataFrame({
        "observed_at": ["2026-09-01 00:00:00-05:00", "2026-09-01 00:30:00-05:00"] * 2,
        "station_id": ["02300", "02300", "03000", "03000"],
        "demand": [1, 2, 3, 4],
    })
    wide = to_wide(obs)
    assert list(wide.columns) == ["02300", "03000"]
    assert len(wide) == 3 and np.isnan(wide.iloc[1]["02300"])  # hueco en la rejilla -> NaN
    assert str(wide.index.tz) == TZ


def test_fold_dataclass_is_hashable():
    ts = pd.Timestamp("2026-09-01", tz=TZ)
    assert Fold("x", ts, ts, ts) in {Fold("x", ts, ts, ts)}
