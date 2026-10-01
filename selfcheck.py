"""
Self-checks on synthetic data, no download or network needed.

    python selfcheck.py

The printed numbers mean nothing (the data is fake); the assertions are the
point: no feature sees data from after the issue time, calendar features are on
German time, the weather modes behave as described, the MLP never sees the test
period, and every model is scored on the same rows.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data import _from_netcdf, fake_frame, fake_temperature
from src.evaluate import (assert_same_rows, baseline_preds, diebold_mariano, mae,
                          mae_by_target_hour, predict, seasonal_naive, skill)
from src.features import (ISSUES, build_features, chronological_split,
                          degree_hours, holiday_share, newest_usable,
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


def _leaks(build, series, amount, issue):
    """Spike each hour of one day in turn, rebuild the features, and return the
    rows that changed even though the spike was newer than they may see.

    All 24 hours, because at 10:00 what a row may see depends on its hour."""
    before = build(series)
    start = len(series) // 2
    bad = []
    for k in range(24):
        poked, t = _poke(series, start + k, amount)
        changed = _changed_rows(before, build(poked))
        assert len(changed) > 0, "nothing reacted to the poke"
        bad += list(changed[newest_usable(changed, issue) < t])
    return bad


def check_no_load_leakage(load):
    # 10am: nothing after 09:00 on D-1; midnight: nothing within 24 hours
    for issue in ISSUES:
        bad = _leaks(lambda s: build_features(s, issue=issue)[0], load, 50000, issue)
        assert not bad, f"leakage ({issue}): features reacted early at {bad[:3]}"
    print("  [ok] no feature uses load published after the issue time (10am and midnight)")

    # the check must also catch a real leak: same hour yesterday is fine at
    # midnight but mostly not published by 10:00
    def leaky(s):
        X, _ = build_features(s)
        X["lag_24h"] = s.shift(24).reindex(X.index)
        return X
    assert _leaks(leaky, load, 50000, "10am"), "the leak check missed a 24h lag at 10am"
    print("  [ok] the leak check fails when a 24h lag is slipped into the 10am features")


def check_local_time(load):
    # 23:00 UTC on 31 Dec is already New Year's Day in Germany
    X, _ = build_features(load)
    loc = X.index.tz_convert("Europe/Berlin")

    assert (X["hour"].to_numpy() == loc.hour.to_numpy()).all(), "hour is not local"
    assert (X["dayofweek"].to_numpy() == loc.dayofweek.to_numpy()).all(), "dow is not local"
    assert (X["hour"].to_numpy() != X.index.hour.to_numpy()).any(), \
        "local and UTC hours are identical in this sample"

    nye = pd.Timestamp("2016-12-31 23:00", tz="UTC")
    if nye in X.index:
        assert X.loc[nye, "is_holiday"] == 1, "1 Jan (local) should be a holiday"
        assert X.loc[nye, "hour"] == 0, "local hour should be 0"
    print("  [ok] calendar features are on Europe/Berlin, not UTC")

    # Corpus Christi ~two thirds of the country, Christmas all of it, a normal
    # Tuesday none
    s = holiday_share(pd.DatetimeIndex(["2016-05-26", "2016-12-25", "2016-07-05"]))
    assert 0.5 < s[0] < 0.8 and s[1] == 1.0 and s[2] == 0.0, f"holiday shares look wrong: {s}"
    print("  [ok] regional holiday shares (Corpus Christi 0.64, Christmas 1, normal day 0)")


def check_feature_table(load):
    X, y = build_features(load)
    assert "load_mw" not in X.columns and len(X) == len(y) and (X.index == y.index).all()
    assert not X.isna().any().any() and not y.isna().any()
    print("  [ok] feature table is clean and aligned")


def check_baseline_and_skill(frame):
    load = frame["load_mw"]
    assert seasonal_naive(load).iloc[168] == load.iloc[0]
    yv = pd.Series([10.0, 20.0, 30.0]); b = pd.Series([12.0, 18.0, 33.0])
    assert abs(skill(yv, yv, b) - 1.0) < 1e-9 and abs(skill(yv, b, b)) < 1e-9
    print("  [ok] seasonal naive is a 168h shift, skill score behaves")

    # most of yesterday isn't published at 10:00, so no 10am baseline may use it
    idx = load.index[1000:1100]
    assert "yesterday" not in baseline_preds(frame, idx)
    assert "yesterday" in baseline_preds(frame, idx, issue="midnight")
    print("  [ok] the 10am baselines only use data out by 10:00 on D-1")

    # a forecast with a third of the error should win clearly, and swapping
    # the two should only flip the sign
    y = load.iloc[:24 * 120]
    rng = np.random.default_rng(0)
    good, poor = y + rng.normal(0, 500, len(y)), y + rng.normal(0, 1500, len(y))
    stat, p = diebold_mariano(y, good, poor)
    assert stat < 0 and p < 0.01, f"DM missed a clearly better forecast: {stat:.2f}, p={p:.3f}"
    assert abs(stat + diebold_mariano(y, poor, good)[0]) < 1e-9, "DM is not antisymmetric"
    print("  [ok] Diebold-Mariano picks the better forecast and is antisymmetric")


def check_beats_naive(frame):
    load = frame["load_mw"]
    X, y = build_features(load)
    (Xtr, ytr), _, (Xte, yte) = chronological_split(X, y, VAL_START, TEST_START)
    m = HistGradientBoostingRegressor(max_iter=100, early_stopping=False,
                                      random_state=0).fit(Xtr, ytr)
    nv = seasonal_naive(load).reindex(yte.index)
    assert mae(yte, predict(m, Xte)) < mae(yte, nv)
    print("  [ok] model beats the baseline on synthetic data")


def check_weather_modes(load):
    """Spike the temperature series: 'lagged' must not react to anything
    measured after the issue time, and 'perfect' must react at the spiked
    hour. Checking both directions stops the modes quietly becoming the same."""
    temp = fake_temperature(load.index, seed=3)

    hdh, cdh = degree_hours(pd.Series([-5.0, 18.0, 30.0]))
    assert hdh.tolist() == [20.0, 0.0, 0.0] and cdh.tolist() == [0.0, 0.0, 8.0]

    assert usable_temperature(temp, "none") is None
    assert usable_temperature(temp, "perfect").equals(temp)
    assert usable_temperature(temp, "lagged").equals(temp.shift(48))
    assert usable_temperature(temp, "lagged", issue="midnight").equals(temp.shift(24))
    fc = usable_temperature(temp, "noisy", seed=0)
    assert not fc.equals(temp), "noisy mode must not be the exact truth"
    assert usable_temperature(temp, "noisy", seed=0).equals(fc), "noisy mode not seeded"

    for issue in ISSUES:
        def lagged(s, issue=issue):
            return build_features(load, s, weather_mode="lagged", issue=issue)[0]
        bad = _leaks(lagged, temp, 25.0, issue)
        assert not bad, f"leakage ({issue}): lagged weather reacted early at {bad[:3]}"
    print("  [ok] weather_mode='lagged' uses no temperature measured after the issue time")

    poked, t = _poke(temp, 4000, 25.0)
    for issue in ISSUES:
        Xb, _ = build_features(load, temp, weather_mode="perfect", issue=issue)
        Xa, _ = build_features(load, poked, weather_mode="perfect", issue=issue)
        assert t in _changed_rows(Xb, Xa), "perfect mode should react at the poked hour"
    print("  [ok] weather_mode='perfect' does use target-hour temperature, as documented")

    for issue in ISSUES:
        n_none = build_features(load, temp, weather_mode="none", issue=issue)[0].shape[1]
        for mode in ("lagged", "noisy", "perfect"):
            n_wx = build_features(load, temp, weather_mode=mode, issue=issue)[0].shape[1]
            assert n_wx == n_none + 5, \
                f"{mode}/{issue}: expected 5 weather features, got {n_wx - n_none}"
    print(f"  [ok] weather adds exactly 5 features in every mode ({n_none} -> {n_wx})")


def check_netcdf_reader():
    """Write a small two-file NetCDF in ERA5's layout and read it back through
    _from_netcdf. Two files because the download writes one per year."""
    try:
        import xarray as xr
    except ImportError:
        raise AssertionError("xarray is a pinned dependency - install it to run this check")

    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    try:
        files, expected = [], []
        for k, yr in enumerate((2016, 2017)):
            idx = pd.date_range(f"{yr}-01-01", periods=36, freq="h")
            lats = np.arange(55.0, 53.9, -0.25)
            lons = np.arange(5.5, 6.6, 0.25)
            # every cell in hour i holds 273.15 + i + k, so the mean is i + k in C
            base = np.arange(len(idx), dtype="float32") + k + 273.15
            data = np.repeat(np.repeat(base[:, None, None], len(lats), 1), len(lons), 2)
            f = f"{tmp}/era5_t2m_{yr}.nc"
            xr.Dataset({"t2m": (("valid_time", "latitude", "longitude"), data)},
                       coords={"valid_time": idx, "latitude": lats,
                               "longitude": lons}).to_netcdf(f)
            files.append(f)
            expected.extend((np.arange(len(idx)) + k).tolist())

        s = _from_netcdf(files)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    assert len(s) == 72, f"expected 72 hours across two files, got {len(s)}"
    assert s.index.tz is not None and str(s.index.tz) == "UTC", "ERA5 index must be UTC"
    assert np.allclose(s.to_numpy(), expected), \
        "spatial mean or the Kelvin->Celsius conversion is wrong"
    assert s.index.is_monotonic_increasing, "concatenated files are out of order"
    print("  [ok] NetCDF reader handles multiple files, no dask, K->C correct")


def check_mlp_scalers_and_determinism(load):
    """The MLP's scalers only see the data it is fitted on. Tested by scaling
    up the test period, refitting, and checking the model is bit-identical."""
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
        "fit and full-series means are identical in this sample"

    cut = pd.Timestamp(TEST_START, tz="UTC") + pd.Timedelta("504h")
    wrecked = load.copy()
    wrecked.loc[cut:] = wrecked.loc[cut:] * 7.5
    Xw, yw = build_features(wrecked)
    (Xtr_w, ytr_w), (Xva_w, yva_w), _ = chronological_split(Xw, yw, VAL_START, TEST_START)
    assert Xtr_w.equals(Xtr) and Xva_w.equals(Xva), "the change touched the training data"

    fn_w, info_w = fit_mlp(
        Xtr_w, ytr_w, Xva_w, yva_w,
        pd.concat([Xtr_w, Xva_w]), pd.concat([ytr_w, yva_w]), verbose=False)
    assert np.array_equal(info_a["scaler_x_mean"], info_w["scaler_x_mean"])
    assert (pa.to_numpy() == fn_w(Xte).to_numpy()).all(), \
        "test-period values changed the fitted MLP"
    print("  [ok] wrecking the test period leaves the fitted MLP bit-identical")


def check_same_rows(frame):
    """Every model and baseline is scored on the same rows, and the check
    itself fails when they aren't."""
    load = frame["load_mw"]
    X, y = build_features(load)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(X, y, VAL_START, TEST_START)
    Xfit, yfit = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])

    preds = baseline_preds(frame, yte.index)
    assert "entsoe_benchmark" in preds, "the published benchmark should be in the baselines"
    for fit in ALL_MODELS:
        fn, info = fit(Xtr, ytr, Xva, yva, Xfit, yfit, verbose=False)
        preds[info["name"]] = fn(Xte)

    assert_same_rows(yte, preds)
    print(f"  [ok] all {len(preds)} models scored on the same {len(yte):,} rows")

    lead = mae_by_target_hour(yte, preds)
    assert list(lead.index) == list(range(24)), "local hour should run 0..23"

    broken = dict(preds)
    broken["gbm"] = broken["gbm"].iloc[:-1]
    try:
        assert_same_rows(yte, broken)
    except AssertionError:
        print("  [ok] the same-rows check actually fails when rows differ")
    else:
        raise AssertionError("assert_same_rows accepted mismatched rows")


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
