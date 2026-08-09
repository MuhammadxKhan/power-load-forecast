"""
Run ridge, gradient boosting and the MLP on identical ground, against the
seasonal-naive baseline and a published ENTSO-E-derived day-ahead benchmark.

    pip install -r requirements.txt
    python run_comparison.py                        # no weather, the old model
    python run_comparison.py --weather lagged       # yesterday's temperature only
    python run_comparison.py --weather noisy        # + synthetic temperature error (sensitivity)
    python run_comparison.py --weather perfect      # perfect prognosis upper bound
    python run_comparison.py --weather noisy --backtest
    python selfcheck.py                             # checks, no download

Identical ground means the features and the split come from src/features.py, so
every model sees the same columns and rows; every model runs the same protocol
(small grid on validation, one refit on train+val, one score on test); and every
model is scored by the same code in src/evaluate.py on the same index, which is
asserted rather than assumed.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")            # no display in a plain terminal
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_frame, load_temperature
from src.evaluate import (backtest_folds, backtest_run, backtest_summary,
                          baseline_preds, mae, mae_by_target_hour, score_table,
                          skill, worst_days)
from src.features import build_features, chronological_split
from src.models import ALL_MODELS




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weather", default="none",
                   choices=["none", "lagged", "noisy", "perfect"])
    p.add_argument("--val-start", default="2018-01-01")
    p.add_argument("--test-start", default="2019-01-01")
    p.add_argument("--backtest", action="store_true",
                   help="rolling-origin folds as well as the single split (slow)")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    print("Loading German hourly load...")
    frame = load_frame()
    load = frame["load_mw"]
    print(f"{len(load):,} hours, {load.index[0]:%Y-%m-%d} to {load.index[-1]:%Y-%m-%d}")

    temp = None
    if args.weather != "none":
        temp = load_temperature(load.index)
        print(f"temperature: {temp.min():.1f} to {temp.max():.1f} C, "
              f"mean {temp.mean():.1f} C   (mode: {args.weather})")
    print()

    X, y = build_features(load, temp, weather_mode=args.weather)
    print(f"{X.shape[1]} features, {len(X):,} usable rows (first 3 weeks go to lags)\n")

    (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(
        X, y, args.val_start, args.test_start)
    for nm, s in (("train", ytr), ("val", yva), ("test", yte)):
        print(f"{nm:<6}{s.index[0]:%Y-%m-%d}..{s.index[-1]:%Y-%m-%d}  ({len(s):,}h)")
    print()

    Xfit, yfit = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])

    print("Tuning on validation (test untouched):")
    preds, chosen = baseline_preds(frame, yte.index), []
    for fit in ALL_MODELS:
        fn, info = fit(Xtr, ytr, Xva, yva, Xfit, yfit, verbose=True)
        preds[info["name"]] = fn(Xte)
        chosen.append(info)
    print("\n  chose " + ", ".join(f"{i['name']} {i['params']}" for i in chosen) + "\n")

    table = score_table(yte, preds)
    print("Test-set results (scored once, never tuned on):\n")
    print(table.round(3).to_string())

    base, best = preds["seasonal_naive"], table.index[0]
    print(f"\nBest model: {best}  ({skill(yte, preds[best], base):.1%} of the "
          f"seasonal-naive error removed)")

    if "entsoe_benchmark" in preds:
        off, bm = mae(yte, preds["entsoe_benchmark"]), mae(yte, preds[best])
        print(f"vs the published ENTSO-E-derived benchmark: {bm:,.0f} vs "
              f"{off:,.0f} MW MAE")
        print("  NOT a like-for-like comparison. The benchmark is published at "
              "least two hours\n  before day-ahead gate closure (~10:00 on D-1 "
              "for Germany); this model assumes\n  midnight, so it has ~14 hours "
              "more demand data. Lower MAE here does not mean\n  a better "
              "forecast. See the README.")

    gbm_mae, mlp_mae = mae(yte, preds["gbm"]), mae(yte, preds["mlp"])
    gap = abs(gbm_mae - mlp_mae)
    print(f"\nGBM vs MLP: {gbm_mae:,.0f} vs {mlp_mae:,.0f} MW "
          f"({gap / min(gbm_mae, mlp_mae):.1%} apart)")
    if gap / min(gbm_mae, mlp_mae) < 0.02:
        print("  Under 2% on one test window - treat that as a tie, not a winner."
              "\n  Run with --backtest to see whether the ordering is even stable.")

    print("\nMAE by local target hour, MW (NOT lead time - see evaluate.py):")
    lead = mae_by_target_hour(yte, preds)
    print(lead[[c for c in ("gbm", "mlp", "entsoe_benchmark") if c in lead]]
          .round(0).to_string())

    print(f"\nWorst 5 days for {best}:")
    print(worst_days(yte, preds[best]).round(0).to_string())

    if args.backtest:
        print("\nRolling-origin backtest (same protocol, origin walked forward):")
        folds = backtest_folds(X.index, args.test_start, block_months=6)
        tidy = backtest_run(X, y, ALL_MODELS, folds)
        wide, summary = backtest_summary(tidy)
        print("\nMAE per fold, MW:")
        print(wide.round(0).to_string())
        print("\nAcross folds:")
        print(summary.round(1).to_string())
        tidy.to_csv("results_backtest.csv", index=False)

    if not args.no_plots:
        made = all_plots(yte, preds, temp)
        print("\nWrote " + ", ".join(made))

    table.to_csv("results_scores.csv")
    pd.DataFrame(preds).assign(actual=yte).to_csv("results_predictions.csv")
    print("Wrote results_scores.csv and results_predictions.csv")


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
OUT = "figures"
TZ = "Europe/Berlin"


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_actual_vs_forecast(y, preds, days=7, start=None):
    """One week of actual demand with the forecasts on top.

    The table says the GBM is off by ~1,200 MW on average. This says what that
    looks like: whether it's tracking the shape and sitting slightly off, or
    missing the peaks, which are very different problems.
    """
    idx = y.index.tz_convert(TZ)
    start = pd.Timestamp(start, tz=TZ) if start else idx[0]
    m = (idx >= start) & (idx < start + pd.Timedelta(days=days))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(idx[m], y[m] / 1000, color="black", lw=2, label="actual")
    for name in ("gbm", "mlp", "entsoe_benchmark"):
        if name in preds:
            ax.plot(idx[m], preds[name][m] / 1000, lw=1.2, alpha=0.85, label=name)
    ax.set_ylabel("GW")
    ax.set_xlabel(f"local time, {days} days from {start:%Y-%m-%d}")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "actual_vs_forecast.png")

