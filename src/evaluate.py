"""
Baselines, metrics, the scoring table, and the rolling-origin backtest.

Everything that turns predictions into numbers lives here, so every model is
scored by exactly the same code on exactly the same rows.

The baselines now include a real published benchmark. OPSD ships a day-ahead
load forecast column derived from ENTSO-E Transparency data, in the same file as
the demand figures. Seasonal naive tells you whether the model learned anything;
the published benchmark tells you whether the answer is in the right ballpark.

Be careful how that benchmark is described. It is OPSD's aggregation, not a raw
untouched TSO series, and the file keeps target timestamps but no forecast
vintage - so there is no way to know whether a value is the first issuance or a
later revision. Its information cutoff is earlier than this model's assumed
midnight either way, so beating it is not a like-for-like win.
"""

import numpy as np
import pandas as pd

TZ = "Europe/Berlin"


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def seasonal_naive(load):
    """Same hour, same weekday, last week. The one to beat first.

    168 rows back is 168 UTC hours, which is the same LOCAL clock hour on every
    week except the two containing a daylight-saving change - there it lands an
    hour out. Two weeks a year, left as is because changing it would move the
    published baseline, but worth knowing it is not exactly "same local hour".
    """
    return load.shift(168)


def yesterday(load):
    return load.shift(24)


def mean_last_4_weeks(load):
    return sum(load.shift(168 * (w + 1)) for w in range(4)) / 4


def baseline_preds(frame, index):
    """The shift-based baselines plus the official forecast, cut to the scored rows.

    `frame` is what data.load_frame returns: load_mw and benchmark_mw.
    """
    load = frame["load_mw"]
    out = {
        "yesterday": yesterday(load).reindex(index),
        "seasonal_naive": seasonal_naive(load).reindex(index),
        "mean_last_4_weeks": mean_last_4_weeks(load).reindex(index),
    }
    if "benchmark_mw" in frame:
        bench = frame["benchmark_mw"].reindex(index)
        gaps = int(bench.isna().sum())
        if gaps:
            # never fabricate benchmark values just to keep a column in the
            # table - drop it and say why
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


def bias(pred, y):
    """Mean signed error. Positive means the forecast runs high. MAE hides this
    completely - a forecast can have a fine MAE and still be systematically
    over, which matters if you're buying generation against it."""
    return float(np.mean(pred - y))


def skill(y, pred, base):
    """Fraction of the baseline's error removed. 0 = no better than the baseline."""
    return 1.0 - mae(y, pred) / mae(y, base)


def predict(model, X):
    return pd.Series(model.predict(X), index=X.index)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def assert_same_rows(y, preds):
    """Every model and baseline must be scored on the same timestamps, in the
    same order. If one quietly dropped or reordered rows its MAE is an average
    over different hours and the table compares nothing."""
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
                "bias_MW": bias(pr, y), "skill_vs_naive": skill(y, pr, b)}
            for n, pr in preds.items()}
    return pd.DataFrame(rows).T.sort_values("MAE_MW")


def mae_by_target_hour(y, preds):
    """Error against the target's local clock hour.

    This is NOT lead-time verification, and an earlier version of this file
    claimed it was. With a single assumed forecast origin at midnight, clock
    hour and horizon are the same variable - hour 14 is always 14 hours ahead -
    so the two are perfectly confounded and you cannot tell "the forecast decays
    with horizon" from "afternoon load is harder". Real lead-time verification
    needs the same valid time forecast from several issue times, which means
    carrying issue_time, valid_time and lead_time explicitly. This project
    doesn't.

    It is doubly wrong for entsoe_benchmark, whose issue time is not midnight at
    all, so its horizon isn't the clock hour even nominally.

    What the curve does show, honestly: which hours of the day are hard.
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
# One train/val/test split gives one number per model and no way to tell whether
# a gap between two of them is real or just which window you happened to pick.
# That matters here: on the single split the GBM and the MLP finish about 21 MW
# apart, under 2%.
#
# So walk the origin forward. Each fold trains on everything up to its own
# cutoff, tunes on the year before its test block, and scores the block. Same
# protocol as the main run, four times, on four different test periods.
#
# Two things this is NOT. Not a significance test - that's a Diebold-Mariano
# test on the paired errors, accounting for serial correlation, and it isn't
# done here. And the folds aren't independent trials: they share training data
# and load is serially correlated, so winning three of four is a stability
# signal, not four coin flips. A reversal between folds can equally mean
# genuine regime-dependent performance rather than noise.
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
        assert_same_rows(yte, preds)   # same guarantee as the main run

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
