"""
Day-ahead electricity load forecasting - single file version.

Forecasts tomorrow's 24 hourly electricity demand values, and checks whether
the model actually beats a simple baseline (harder than it sounds).

Data: hourly load from Open Power System Data (ENTSO-E figures, no API key).

    pip install pandas numpy scikit-learn
    python load_forecast.py            # Germany, real data (~200MB download once)
    python load_forecast.py --country GB
    python load_forecast.py --test     # self-check on synthetic data, no download

The one rule everything hangs on: we forecast day D at midnight, so the newest
data we may use is 23:00 on day D-1. That means NO feature uses a lag shorter
than 24 hours. The --test mode includes a check that no feature reacts to a
change less than 24h before it.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CSV_URL = (
    "https://data.open-power-system-data.org/time_series/latest/"
    "time_series_60min_singleindex.csv"
)
CACHE = "opsd_60min.csv"

LAGS = [24, 48, 72, 168, 336]

# German national public holidays 2015-2020. Load drops hard on these and the
# weekday features can't see them. Swap this list if you change country.
HOLIDAYS = {
    "2015-01-01", "2015-04-03", "2015-04-06", "2015-05-01", "2015-05-14",
    "2015-05-25", "2015-10-03", "2015-12-25", "2015-12-26",
    "2016-01-01", "2016-03-25", "2016-03-28", "2016-05-01", "2016-05-05",
    "2016-05-16", "2016-10-03", "2016-12-25", "2016-12-26",
    "2017-01-01", "2017-04-14", "2017-04-17", "2017-05-01", "2017-05-25",
    "2017-06-05", "2017-10-03", "2017-10-31", "2017-12-25", "2017-12-26",
    "2018-01-01", "2018-03-30", "2018-04-02", "2018-05-01", "2018-05-10",
    "2018-05-21", "2018-10-03", "2018-12-25", "2018-12-26",
    "2019-01-01", "2019-04-19", "2019-04-22", "2019-05-01", "2019-05-30",
    "2019-06-10", "2019-10-03", "2019-12-25", "2019-12-26",
    "2020-01-01", "2020-04-10", "2020-04-13", "2020-05-01", "2020-05-21",
    "2020-06-01", "2020-10-03", "2020-12-25", "2020-12-26",
}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_series(country="DE"):
    """One country's hourly load as a Series indexed by UTC timestamp."""
    import os

    col = f"{country}_load_actual_entsoe_transparency"

    if os.path.exists(CACHE):
        df = pd.read_csv(CACHE, usecols=["utc_timestamp", col],
                         parse_dates=["utc_timestamp"])
    else:
        print("Downloading OPSD data (~200MB, one-off, give it a minute)...")
        full = pd.read_csv(CSV_URL, low_memory=False)
        full.to_csv(CACHE, index=False)
        print(f"Cached to {CACHE}")
        df = full[["utc_timestamp", col]].copy()
        df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])

    s = df.set_index("utc_timestamp")[col]
    s = s.loc[s.first_valid_index():s.last_valid_index()]

    # ensure every hour exists, then fill the small internal gaps so the fixed
    # 24h/168h lags don't silently skip rows
    s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="h"))
    missing = int(s.isna().sum())
    if missing:
        print(f"{missing} missing hours ({missing / len(s):.3%}) - interpolating")
        s = s.interpolate(limit=6).ffill().bfill()

    s.name = "load_mw"
    s.index.name = "timestamp"
    return s


def fake_series(n_days=1500, seed=0):
    """Synthetic load for the self-check. Numbers are meaningless - do NOT
    report them. It's only here so the code runs without the download."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n_days * 24, freq="h")
    hour, dow, doy = idx.hour.to_numpy(), idx.dayofweek.to_numpy(), idx.dayofyear.to_numpy()
    daily = 8000 * np.sin((hour - 3) / 24 * 2 * np.pi) + 3000 * np.sin(hour / 12 * 2 * np.pi)
    weekly = np.where(dow >= 5, -6000, 0)
    yearly = 5000 * np.cos((doy - 15) / 365 * 2 * np.pi)
    load = 50000 + daily + weekly + yearly + rng.normal(0, 900, len(idx))
    return pd.Series(load, index=idx, name="load_mw").rename_axis("timestamp")


# --------------------------------------------------------------------------
# features (nothing newer than 24h)
# --------------------------------------------------------------------------
def _cyclical(values, period):
    r = 2 * np.pi * values / period
    return np.sin(r), np.cos(r)


def build_features(load):
    df = pd.DataFrame({"load_mw": load})
    idx = df.index

    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_holiday"] = idx.normalize().strftime("%Y-%m-%d").isin(HOLIDAYS).astype(int)

    df["hour_sin"], df["hour_cos"] = _cyclical(idx.hour.to_numpy(), 24)
    df["dow_sin"], df["dow_cos"] = _cyclical(idx.dayofweek.to_numpy(), 7)
    df["doy_sin"], df["doy_cos"] = _cyclical(idx.dayofyear.to_numpy(), 365)

    for lag in LAGS:
        df[f"lag_{lag}h"] = df["load_mw"].shift(lag)

    past = df["load_mw"].shift(24)  # everything rolls off the 24h-lagged series
    df["roll_mean_24h"] = past.rolling(24).mean()
    df["roll_mean_168h"] = past.rolling(168).mean()
    df["roll_std_24h"] = past.rolling(24).std()

    df["same_hour_3wk_mean"] = (
        df["load_mw"].shift(168) + df["load_mw"].shift(336) + df["load_mw"].shift(504)
    ) / 3

    df = df.dropna()
    y = df.pop("load_mw")
    return df, y


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def seasonal_naive(load):
    """Same hour, same weekday, last week. The one to beat."""
    return load.shift(168)


def yesterday(load):
    return load.shift(24)


def mean_last_4_weeks(load):
    return sum(load.shift(168 * (w + 1)) for w in range(4)) / 4


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(a, b):
    return float(np.mean(np.abs((a - b) / a)) * 100)


def skill(y, pred, base):
    """Fraction of the baseline's error removed. 0 = no better than naive."""
    return 1.0 - mae(y, pred) / mae(y, base)


def predict(model, X):
    return pd.Series(model.predict(X), index=X.index)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def chronological_split(X, y, val_start, test_start):
    vs = pd.Timestamp(val_start, tz=X.index.tz)
    ts = pd.Timestamp(test_start, tz=X.index.tz)
    tr, va, te = X.index < vs, (X.index >= vs) & (X.index < ts), X.index >= ts
    if not (tr.any() and va.any() and te.any()):
        raise ValueError("a split came out empty - check your dates vs the data range")
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--country", default="DE")
    p.add_argument("--val-start", default="2018-01-01")
    p.add_argument("--test-start", default="2019-01-01")
    p.add_argument("--test", action="store_true", help="self-check on synthetic data")
    args = p.parse_args()

    if args.test:
        run_self_check()
        return

    print(f"Loading {args.country} hourly load...")
    load = load_series(args.country)
    print(f"{len(load):,} hours, {load.index[0]:%Y-%m-%d} to {load.index[-1]:%Y-%m-%d}\n")

    X, y = build_features(load)
    print(f"{X.shape[1]} features, {len(X):,} usable rows (first 3 weeks go to lags)\n")

    (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(
        X, y, args.val_start, args.test_start
    )
    print(f"train {ytr.index[0]:%Y-%m-%d}..{ytr.index[-1]:%Y-%m-%d}  ({len(ytr):,}h)")
    print(f"val   {yva.index[0]:%Y-%m-%d}..{yva.index[-1]:%Y-%m-%d}  ({len(yva):,}h)")
    print(f"test  {yte.index[0]:%Y-%m-%d}..{yte.index[-1]:%Y-%m-%d}  ({len(yte):,}h)\n")

    # tune ridge alpha on validation
    print("Tuning on validation (test untouched):")
    best_alpha, best = 1.0, float("inf")
    for a in [0.1, 1.0, 10.0, 100.0]:
        m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Xtr, ytr)
        v = mae(yva, predict(m, Xva))
        print(f"  ridge alpha={a:>6}  val MAE {v:8.1f}")
        if v < best:
            best_alpha, best = a, v

    # tune gbm on validation
    best_gbm, best = (0.05, 400), float("inf")
    for lr in [0.05, 0.1]:
        for it in [300, 600]:
            m = HistGradientBoostingRegressor(learning_rate=lr, max_iter=it,
                                              random_state=0).fit(Xtr, ytr)
            v = mae(yva, predict(m, Xva))
            print(f"  gbm lr={lr} iters={it:>4}  val MAE {v:8.1f}")
            if v < best:
                best_gbm, best = (lr, it), v
    print(f"  chose ridge alpha={best_alpha}, gbm lr={best_gbm[0]} iters={best_gbm[1]}\n")

    # refit on train+val, score once on test
    Xfit, yfit = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
    ridge_m = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)).fit(Xfit, yfit)
    gbm_m = HistGradientBoostingRegressor(
        learning_rate=best_gbm[0], max_iter=best_gbm[1], random_state=0
    ).fit(Xfit, yfit)

    preds = {
        "yesterday": yesterday(load).reindex(yte.index),
        "seasonal_naive": seasonal_naive(load).reindex(yte.index),
        "mean_last_4_weeks": mean_last_4_weeks(load).reindex(yte.index),
        "ridge": predict(ridge_m, Xte),
        "gbm": predict(gbm_m, Xte),
    }
    base = preds["seasonal_naive"]

    rows = {n: {"MAE_MW": mae(yte, pr), "RMSE_MW": rmse(yte, pr),
                "MAPE_%": mape(yte, pr), "skill_vs_naive": skill(yte, pr, base)}
            for n, pr in preds.items()}
    table = pd.DataFrame(rows).T.sort_values("MAE_MW")

    print("Test-set results (scored once, never tuned on):\n")
    print(table.round(3).to_string())

    naive_mae, gbm_mae = mae(yte, base), mae(yte, preds["gbm"])
    print(f"\nGBM vs seasonal naive: {naive_mae:,.0f} -> {gbm_mae:,.0f} MW "
          f"({skill(yte, preds['gbm'], base):.1%} of baseline error removed)")
    if gbm_mae >= naive_mae:
        print("The model did NOT beat the baseline. Say so in the README.")

    err = (preds["gbm"] - yte).abs()
    print("\nGBM mean error by hour of day (MW):")
    print(err.groupby(err.index.hour).mean().round(0).to_string())

    daily = err.groupby(err.index.date).mean().sort_values(ascending=False)
    print("\nWorst 5 days (usually holidays):")
    print(daily.head(5).round(0).to_string())

    table.to_csv("results_scores.csv")
    pd.DataFrame(preds).assign(actual=yte).to_csv("results_predictions.csv")
    print("\nWrote results_scores.csv and results_predictions.csv")


# --------------------------------------------------------------------------
# self-check (no download) - the leakage test lives here
# --------------------------------------------------------------------------
def run_self_check():
    print("Self-check on synthetic data (numbers are meaningless)...\n")
    load = fake_series(400, seed=1)

    # 1) no feature may react to a change less than 24h before it
    Xb, _ = build_features(load)
    poked = load.copy()
    t = poked.index[3000]
    poked.iloc[3000] += 50000
    Xa, _ = build_features(poked)
    common = Xb.index.intersection(Xa.index)
    changed = common[(Xb.loc[common] != Xa.loc[common]).any(axis=1)]
    too_soon = changed[(changed > t) & (changed < t + pd.Timedelta("24h"))]
    assert len(too_soon) == 0, f"LEAKAGE: features reacted within 24h at {list(too_soon)[:3]}"
    assert (changed >= t + pd.Timedelta("24h")).any(), "lags look broken"
    print("  [ok] no feature uses data newer than 24h (no leakage)")

    # 2) target not among features, no NaNs, aligned
    X, y = build_features(load)
    assert "load_mw" not in X.columns and len(X) == len(y) and (X.index == y.index).all()
    assert not X.isna().any().any() and not y.isna().any()
    print("  [ok] feature table is clean and aligned")

    # 3) seasonal naive is a 168h shift
    assert seasonal_naive(load).iloc[168] == load.iloc[0]
    print("  [ok] seasonal naive baseline is a 168h shift")

    # 4) skill score signs
    yv = pd.Series([10.0, 20.0, 30.0]); b = pd.Series([12.0, 18.0, 33.0])
    assert abs(skill(yv, yv, b) - 1.0) < 1e-9 and abs(skill(yv, b, b)) < 1e-9
    print("  [ok] skill score behaves (1 = perfect, 0 = matches baseline)")

    # 5) model beats naive on clean synthetic data
    X, y = build_features(load)
    (Xtr, ytr), _, (Xte, yte) = chronological_split(X, y, "2016-09-01", "2016-11-01")
    m = HistGradientBoostingRegressor(max_iter=100, random_state=0).fit(Xtr, ytr)
    nv = seasonal_naive(load).reindex(yte.index)
    assert mae(yte, predict(m, Xte)) < mae(yte, nv)
    print("  [ok] model beats the baseline on synthetic data")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
