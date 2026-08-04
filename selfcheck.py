"""
Self-checks on synthetic data. No download, no network.

    python selfcheck.py

The numbers these print are meaningless - the data is fake. What matters is
that the assertions hold: nothing reaches forward in time, the calendar features
are on German clocks, weather behaves the way the chosen mode says it does, the
MLP's scalers never see the test period, and every model is scored on the same
rows.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data import _from_netcdf, fake_frame, fake_temperature
from src.evaluate import (assert_same_rows, baseline_preds, mae,
                          mae_by_target_hour, predict, seasonal_naive, skill)
from src.features import (build_features, chronological_split, degree_hours,
                          usable_temperature)
from src.models import ALL_MODELS, fit_mlp

VAL_START, TEST_START = "2016-09-01", "2016-11-01"


def _poke(series, pos, amount):
    out = series.copy()
    out.iloc[pos] += amount
    return out, out.index[pos]


def _changed_rows(before, after):
    common = before.index.intersection(after.index)
    return common[(before.loc[common] != after.loc[common]).any(axis=1)]


def check_no_load_leakage(load):
    # 1) no feature may react to a load change less than 24h before it.
    # Note the >= t: a feature that used the value AT t would be the worst leak
    # of all, and the old version of this test let it through by only looking
    # strictly after t.
    Xb, _ = build_features(load)
    poked, t = _poke(load, 3000, 50000)
    Xa, _ = build_features(poked)

    changed = _changed_rows(Xb, Xa)
    too_soon = changed[(changed >= t) & (changed < t + pd.Timedelta("24h"))]
    assert len(too_soon) == 0, f"LEAKAGE: features reacted within 24h at {list(too_soon)[:3]}"
    assert (changed >= t + pd.Timedelta("24h")).any(), "lags look broken - nothing reacted"
    print("  [ok] no feature uses load data newer than 24h")


def check_local_time(load):
    # 2) calendar features follow German clocks, not UTC.
    # 23:00 UTC on 31 Dec is already New Year's Day in Germany.
    X, _ = build_features(load)
    loc = X.index.tz_convert("Europe/Berlin")

    assert (X["hour"].to_numpy() == loc.hour.to_numpy()).all(), "hour is not local"
    assert (X["dayofweek"].to_numpy() == loc.dayofweek.to_numpy()).all(), "dow is not local"
    assert (X["hour"].to_numpy() != X.index.hour.to_numpy()).any(), \
        "local and UTC hours are identical here, so this check proves nothing"

    nye = pd.Timestamp("2016-12-31 23:00", tz="UTC")
    if nye in X.index:
        assert X.loc[nye, "is_holiday"] == 1, \
            "23:00 UTC on 31 Dec is 1 Jan in Germany and should be a holiday"
        assert X.loc[nye, "hour"] == 0, "local hour should be 0"
    print("  [ok] calendar features are on Europe/Berlin, not UTC")


def check_feature_table(load):
    # 3) target not among the features, no NaNs, aligned
    X, y = build_features(load)
    assert "load_mw" not in X.columns and len(X) == len(y) and (X.index == y.index).all()
    assert not X.isna().any().any() and not y.isna().any()
    print("  [ok] feature table is clean and aligned")


def check_baseline_and_skill(frame):
    # 4) seasonal naive is a 168h shift, and the skill score has the right signs
    load = frame["load_mw"]
    assert seasonal_naive(load).iloc[168] == load.iloc[0]
    yv = pd.Series([10.0, 20.0, 30.0]); b = pd.Series([12.0, 18.0, 33.0])
    assert abs(skill(yv, yv, b) - 1.0) < 1e-9 and abs(skill(yv, b, b)) < 1e-9
    print("  [ok] seasonal naive is a 168h shift, skill score behaves")


def check_beats_naive(frame):
    # 5) a model beats naive on clean synthetic data
    load = frame["load_mw"]
    X, y = build_features(load)
    (Xtr, ytr), _, (Xte, yte) = chronological_split(X, y, VAL_START, TEST_START)
    m = HistGradientBoostingRegressor(max_iter=100, early_stopping=False,
                                      random_state=0).fit(Xtr, ytr)
    nv = seasonal_naive(load).reindex(yte.index)
    assert mae(yte, predict(m, Xte)) < mae(yte, nv)
    print("  [ok] model beats the baseline on synthetic data")


def check_mlp_scalers_and_determinism(load):
    """7) the MLP's scalers only ever see the fold it is fitted on.

    The fit functions are never handed the test set, so structurally they can't scale by
    test statistics. Asserting that is weak on its own, so this does it the hard
    way: wreck the load series inside the test period, refit everything, and
    check the model that comes out is bit-for-bit the one from the clean run.
    """
    X, y = build_features(load)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(X, y, VAL_START, TEST_START)
    Xfit, yfit = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])

    fn_a, info_a = fit_mlp(Xtr, ytr, Xva, yva, Xfit, yfit, verbose=False)
    fn_b, _ = fit_mlp(Xtr, ytr, Xva, yva, Xfit, yfit, verbose=False)
    pa = fn_a(Xte)
    assert (pa.to_numpy() == fn_b(Xte).to_numpy()).all(), "MLP is not deterministic"
    print("  [ok] MLP is bit-identical across runs (fixed seed, fixed batch order)")

    assert np.array_equal(info_a["scaler_x_mean"], Xfit.to_numpy(dtype=np.float64).mean(axis=0))
    assert float(yfit.to_numpy().mean()) == info_a["scaler_y_mean"]
    assert not np.allclose(X.to_numpy(dtype=np.float64).mean(axis=0),
                           info_a["scaler_x_mean"]), \
        "fit-fold and full-series stats are identical here, so this proves nothing"

    cut = pd.Timestamp(TEST_START, tz="UTC") + pd.Timedelta("504h")
    wrecked = load.copy()
    wrecked.loc[cut:] = wrecked.loc[cut:] * 7.5
    Xw, yw = build_features(wrecked)
    (Xtr_w, ytr_w), (Xva_w, yva_w), _ = chronological_split(Xw, yw, VAL_START, TEST_START)
    assert Xtr_w.equals(Xtr) and Xva_w.equals(Xva), "the wrecking touched the training folds"

    fn_w, info_w = fit_mlp(
        Xtr_w, ytr_w, Xva_w, yva_w,
        pd.concat([Xtr_w, Xva_w]), pd.concat([ytr_w, yva_w]), verbose=False)
    assert np.array_equal(info_a["scaler_x_mean"], info_w["scaler_x_mean"])
    assert (pa.to_numpy() == fn_w(Xte).to_numpy()).all(), \
        "test-period values changed the fitted MLP - something leaked"
    print("  [ok] wrecking the test period leaves the fitted MLP bit-identical")


def main():
    print("Self-check on synthetic data (numbers are meaningless)...\n")
    frame = fake_frame(400, seed=1)
    load = frame["load_mw"]

    check_no_load_leakage(load)
    check_local_time(load)
    check_feature_table(load)
    check_baseline_and_skill(frame)
    check_beats_naive(frame)
    check_weather_modes(load)
    check_netcdf_reader()
    check_mlp_scalers_and_determinism(load)
    check_same_rows(frame)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
