"""
Baselines, metrics, the scoring table, the Diebold-Mariano test and the
rolling-origin backtest.

All scoring goes through this file, so every model is scored by the same code
on the same rows. Seasonal naive shows whether a model learned anything; the
published TSO forecast shows whether it is in the right range.
"""

import math

import numpy as np
import pandas as pd

TZ = "Europe/Berlin"


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def seasonal_naive(load):
    """Same hour, same weekday, one week earlier.

    168 UTC hours, so in the two weeks with a clock change it lands an hour off
    the same local hour. Left as is to keep the baseline unchanged.
    """
    return load.shift(168)


def yesterday(load):
    return load.shift(24)


def two_days_ago(load):
    """Same hour two days back: the 10:00 replacement for "yesterday", most of
    which isn't published yet at 10:00 on D-1."""
    return load.shift(48)


def mean_last_4_weeks(load):
    return sum(load.shift(168 * (w + 1)) for w in range(4)) / 4


def baseline_preds(frame, index, issue="10am"):
    """The shift-based baselines plus the published forecast, cut to the scored
    rows. `frame` is what data.load_frame returns."""
    load = frame["load_mw"]
    out = {"seasonal_naive": seasonal_naive(load).reindex(index),
           "mean_last_4_weeks": mean_last_4_weeks(load).reindex(index)}
    if issue == "midnight":
        out["yesterday"] = yesterday(load).reindex(index)
    else:
        out["two_days_ago"] = two_days_ago(load).reindex(index)
    if "benchmark_mw" in frame:
        bench = frame["benchmark_mw"].reindex(index)
        gaps = int(bench.isna().sum())
        if gaps:
            # dropped rather than filled, so it is never scored on made-up values
            print(f"  benchmark has {gaps} missing hours in the scored window "
                  f"({gaps / len(index):.2%}) - excluded from the table")
        else:
            out["entsoe_benchmark"] = bench
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(a, b):
    return float(np.mean(np.abs((a - b) / a)) * 100)


def bias(y, pred):
    """Mean signed error, positive when the forecast runs high. MAE can't show
    a systematic over- or under-forecast; this can."""
    return float(np.mean(pred - y))


def skill(y, pred, base):
    """Share of the baseline's MAE removed. 0 = no better than the baseline."""
    return 1.0 - mae(y, pred) / mae(y, base)


def predict(model, X):
    return pd.Series(model.predict(X), index=X.index)


def diebold_mariano(y, a, b, lags=7):
    """Diebold-Mariano test on absolute errors: is a better than b, or could one
    test window give this gap by chance? Negative stat means a is better.
    Returns (stat, two-sided p-value).

    Uses the daily mean of the hourly loss difference, because hours within a
    day are strongly correlated and counting them separately would overstate
    the evidence. Newey-West variance with a week of lags covers the remaining
    day-to-day correlation.
    """
    loss = (a - y).abs() - (b - y).abs()
    daily = loss.groupby(y.index.tz_convert(TZ).date).mean().to_numpy()
    n = len(daily)
    d = daily - daily.mean()
    var = d @ d / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    stat = daily.mean() / np.sqrt(var / n)
    return float(stat), float(math.erfc(abs(stat) / math.sqrt(2)))


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def assert_same_rows(y, preds):
    """Every model and baseline must be scored on the same timestamps in the
    same order, otherwise the MAEs are averages over different hours."""
    for name, pr in preds.items():
        if len(pr) != len(y):
            raise AssertionError(
                f"{name}: {len(pr)} predictions vs {len(y)} target rows")
        if not pr.index.equals(y.index):
            raise AssertionError(f"{name}: scored on a different index to the target")
        if pr.isna().any():
            raise AssertionError(f"{name}: {int(pr.isna().sum())} NaN predictions")


def score_table(y, preds, base="seasonal_naive"):
    assert_same_rows(y, preds)
    b = preds[base]
    rows = {n: {"MAE_MW": mae(y, pr), "RMSE_MW": rmse(y, pr), "MAPE_%": mape(y, pr),
                "bias_MW": bias(y, pr), "skill_vs_naive": skill(y, pr, b)}
            for n, pr in preds.items()}
    return pd.DataFrame(rows).T.sort_values("MAE_MW")


def mae_by_target_hour(y, preds):
    """MAE by the target's local clock hour.

    With one issue time per day, hour of day and forecast horizon are the same
    thing, so this shows which hours are hard but can't separate the two.
    """
    lead = pd.Index(y.index.tz_convert(TZ).hour, name="local_hour")
    return pd.DataFrame({n: pd.Series((pr - y).abs().to_numpy()).groupby(lead).mean()
                         for n, pr in preds.items()})


def worst_days(y, pred, n=5):
    err = (pred - y).abs()
    local_date = pd.Index(y.index.tz_convert(TZ).date, name="date")
    return err.groupby(local_date).mean().sort_values(ascending=False).head(n)


# --------------------------------------------------------------------------
# rolling-origin backtest
#
# A single split gives one number per model, and a gap between two models can
# just be the window. Here each fold trains on everything before its test
# block, tunes on the year before it, and scores six months; the same protocol
# as the main run, four times. Folds share training data, so a majority of
# folds is a stability check, not an independent test.
# --------------------------------------------------------------------------
def backtest_folds(index, first_test_start, block_months=6, val_months=12):
    """Expanding-window folds: (val_start, test_start, test_end) per fold."""
    start = pd.Timestamp(first_test_start, tz="UTC")
    last = index.max()
    folds = []
    while start < last:
        end = start + pd.DateOffset(months=block_months)
        val_start = start - pd.DateOffset(months=val_months)
        if end > last:
            end = last + pd.Timedelta("1h")
        if (index >= start).sum() < 24 * 30:   # skip a stub final block
            break
        folds.append((val_start, start, end))
        start = end
    return folds


def backtest_run(X, y, models, folds, verbose=True):
    """Fit every model on every fold. Returns tidy MAE per (fold, model)."""
    from .features import chronological_split

    rows = []
    for k, (val_start, test_start, test_end) in enumerate(folds, 1):
        Xk, yk = X[X.index < test_end], y[y.index < test_end]
        (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(
            Xk, yk, val_start, test_start)
        Xfit, yfit = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])

        if verbose:
            print(f"  fold {k}: train {len(ytr):,}h  val {len(yva):,}h  "
                  f"test {yte.index[0]:%Y-%m-%d}..{yte.index[-1]:%Y-%m-%d} "
                  f"({len(yte):,}h)")

        preds = {}
        for fit in models:
            fn, info = fit(Xtr, ytr, Xva, yva, Xfit, yfit, verbose=False)
            preds[info["name"]] = fn(Xte)
        assert_same_rows(yte, preds)

        for name, pr in preds.items():
            rows.append({"fold": k,
                         "test_start": yte.index[0].date(),
                         "model": name,
                         "MAE_MW": mae(yte, pr)})

    return pd.DataFrame(rows)


def backtest_summary(tidy):
    """Mean MAE per model, plus how many folds each one won."""
    wide = tidy.pivot(index="fold", columns="model", values="MAE_MW")
    wins = wide.idxmin(axis=1).value_counts()
    out = pd.DataFrame({
        "mean_MAE_MW": wide.mean(),
        "worst_fold_MAE_MW": wide.max(),
        "folds_won": wins.reindex(wide.columns).fillna(0).astype(int),
    }).sort_values("mean_MAE_MW")
    return wide, out
