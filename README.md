# Day-ahead electricity load forecasting

Forecasting German hourly electricity demand a day in advance, and — the part
that turned out to matter more — checking whether the model actually beats a
simple baseline.

Data is hourly load from [Open Power System Data](https://data.open-power-system-data.org/time_series/)
(ENTSO-E's transparency figures, republished as one CSV, no API key). Germany,
2015–2020.

```bash
pip install numpy pandas scikit-learn
python load_forecast.py            # Germany (downloads ~200MB once, then caches)
python load_forecast.py --country GB
python load_forecast.py --test     # self-check on synthetic data, no download
```

## The setup

Standing at midnight at the start of day D, predict all 24 hourly load values
for day D. The newest observation available is 23:00 on day D−1.

That constraint drives the whole feature table: **no feature uses a lag shorter
than 24 hours**. It would be easy to slip in a 1-hour lag, watch the error
collapse, and ship something that can't actually run a day ahead. The `--test`
mode pokes a large spike into the load series and asserts that no feature row
within the next 24 hours reacts to it.

## Beating the baseline is the whole problem

The baseline is **seasonal naive**: this hour looks like the same hour, same
weekday, last week. One line of code, and it already captures the daily demand
curve, the weekend drop, and slow seasonal drift. It's much harder to beat than
it looks — the point of the project is establishing that a model beats it
*honestly*, not just reporting an impressive-sounding MAE.

Headline metric is the **skill score**, `1 − MAE_model / MAE_baseline`: the
fraction of the baseline's error removed. 0 = no better than the naive rule;
negative = worse than doing nothing clever.

## Results

Test period **2019-01-01 to 2020-09-30**, scored once. Hyperparameters were
chosen on 2018 (the validation year); the test set was untouched until the
final run.

| model | MAE (MW) | RMSE (MW) | MAPE | skill vs naive |
|---|---:|---:|---:|---:|
| **gbm** | **1,220** | 1,637 | 2.31% | **+0.495** |
| ridge | 1,837 | 2,514 | 3.48% | +0.240 |
| seasonal_naive | 2,416 | 4,184 | 4.55% | 0.000 |
| mean_last_4_weeks | 2,648 | 4,134 | 4.95% | −0.096 |
| yesterday | 4,340 | 6,620 | 8.09% | −0.796 |

The gradient-boosted model removes **~49% of the seasonal-naive baseline's
error** (2,416 → 1,220 MW MAE). Ridge on the same features gets about half that.
`yesterday` scoring worst confirms the weekly structure is what matters, which
is why seasonal naive — not a plain 24-hour lag — is the right baseline.

One honest note on tuning: ridge's error barely moves across alpha (0.1 → 100),
because with ~50k rows and 20 features there's little overfitting for
regularisation to fix. The tuning grid is deliberately small.

## Where it fails

Mean absolute error is fairly flat across the day (~1,050–1,450 MW), slightly
worse through the midday/afternoon peak. The interesting failures are the worst
individual days:

| date | MAE (MW) | what it is |
|---|---:|---|
| 2019-06-20 | 8,212 | Corpus Christi (regional holiday, not in the national list) |
| 2020-06-11 | 7,820 | Corpus Christi |
| 2019-04-21 | 4,771 | Easter Sunday |
| 2020-04-09 | 4,385 | COVID demand collapse |
| 2020-04-12 | 3,981 | Easter / COVID |

Two clear patterns. First, **regional holidays the model can't see** — Corpus
Christi is a public holiday in only some German states, so it isn't in the
national holiday list and the model treats it as an ordinary weekday. Second,
**structural breaks** — spring 2020 is the COVID demand drop, which no model
trained on 2015–2018 could anticipate. Both point at concrete fixes rather than
just "the model is noisy".

## What I'd do next

* **A proper holiday calendar**, including state-level holidays like Corpus
  Christi — this alone would remove the two largest error days.
* **Temperature.** The biggest missing feature by far. Load is strongly
  weather-driven; right now the model infers season from the calendar rather
  than actual conditions.
* **A separate model per hour**, since 03:00 and 14:00 behave very differently.
* **Prediction intervals** (e.g. quantile regression) — a point forecast with no
  sense of its own uncertainty isn't much use to anyone taking a position on it.
* **Prices, eventually.** Load is smooth; day-ahead prices are spiky, sometimes
  negative, and regime-switching. I chose load as the tractable problem on
  purpose, not the interesting one.

## Limitations, up front

* No weather data — the largest omission.
* One country, one model class, one test period; nothing here shows it
  generalises.
* Holidays are hardcoded national-only for Germany 2015–2020 (which is exactly
  why the regional-holiday days blow up).
* Missing hours (<1% — clock changes, reporting gaps) are interpolated rather
  than dropped, because dropping them would silently break the fixed 24h/168h
  lags.
* The gradient-boosting model is lightly tuned (four combinations on the
  validation year). I'd rather report an honest, lightly tuned number than
  grind the test set.

## How it's laid out

Single file, `load_forecast.py`:

* data loading (download, cache, clean the load series)
* feature building (calendar + lags, none newer than 24h)
* baselines (seasonal naive and two weaker ones)
* ridge + gradient boosting, and the chronological split
* metrics (MAE / RMSE / MAPE, skill score, error breakdowns)
* `--test` runs the self-checks, including the leakage check
