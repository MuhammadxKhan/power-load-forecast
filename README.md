# Day-ahead load forecasting for Germany, and what weather data is worth to it

A day-ahead forecast of German hourly electricity demand, built as an instrument
for measuring the value of weather information rather than as an end in itself.

Temperature enters through four switchable modes - `none`, `lagged`, `noisy`,
`perfect` - so "how much would better weather information actually buy us?" gets
a number instead of an assumption. `perfect` is the ceiling: the most any weather
model, however good, could contribute to this particular decision.

The short answer is that it depends on the season, and that the effect is small
enough that a single test window cannot resolve it. Establishing both took a
rolling-origin backtest and a ten-seed study; a single run would have reported
roughly twice the true effect, in the direction I was hoping for.

Data: [OPSD](https://open-power-system-data.org/) time series (2020-10-06
release), German hourly load 2015-2020, plus ERA5 2m temperature (NetCDF, via
the Copernicus CDS).

---

## Results

Test period 2019-01-01 to 2020-09-30. Trained on 2015-2017, validated on 2018.
The models never see the test period.

| model | MAE (MW) | RMSE (MW) | MAPE | bias (MW) | skill vs naive |
|---|---:|---:|---:|---:|---:|
| **gradient boosting** | **1,196** | 1,631 | 2.25% | +110 | **0.505** |
| MLP (PyTorch) | 1,217 | 1,674 | 2.29% | +216 | 0.496 |
| ENTSO-E published forecast | 1,762 | 2,253 | 3.22% | **-608** | 0.271 |
| ridge | 1,837 | 2,515 | 3.46% | +236 | 0.240 |
| seasonal naive (168h) | 2,416 | 4,184 | 4.55% | -55 | 0 |
| mean of last 4 weeks | 2,648 | 4,134 | 4.94% | +75 | -0.096 |
| yesterday | 4,340 | 6,620 | 8.09% | -16 | -0.796 |

Skill score is `1 - MAE_model / MAE_naive`. Gradient boosting removes just over
half the seasonal-naive baseline's error.

The ENTSO-E row is the TSOs' own published day-ahead forecast, scored on the
same rows. Note its bias: it runs 608 MW low on average, which MAE hides
entirely. See Limitations for why this is not a like-for-like comparison.

---

## Does weather help?

The more interesting question, and the answer is "depends on the season" rather
than yes or no.

Temperature enters through four switchable modes, so the value of *better*
weather information can be measured rather than assumed:

- `none` — no weather at all
- `lagged` — temperature from 24h+ ago only (honest: no future information)
- `noisy` — true temperature plus 1 degC Gaussian noise (stand-in for forecast error)
- `perfect` — exact temperature at the target hour (perfect prognosis: an upper bound, not achievable)

### Single test window, gradient boosting

| mode | MAE (MW) | vs none |
|---|---:|---:|
| none | 1,196.0 | — |
| lagged | 1,196.3 | +0.3 |
| noisy (seed 0) | 1,178.0 | -18.0 |
| perfect | 1,191.2 | -4.8 |

Perfect foreknowledge of temperature buys 0.4%. Small enough to be suspicious,
so it was checked properly.

### The single-window number is inside the noise

`noisy` across ten random seeds, gradient boosting:

| | MAE (MW) |
|---|---:|
| mean | 1,186.1 |
| std | 8.4 |
| min | 1,178.0 |
| max | 1,203.0 |
| **spread** | **25.0** |

The spread from nothing but changing the random seed (25.0 MW) is larger than
the apparent gain from adding weather (18.0 MW). And seed 0 — the default, the
one a single run reports — is joint-best of the ten. Reporting one seed would
have overstated the effect roughly twofold.

### Across rolling folds, weather does help

Rolling-origin backtest, gradient boosting, 6-month blocks:

| fold | test period | none | noisy | delta |
|---|---|---:|---:|---:|
| 1 | 2019 H1 | 1,280.5 | 1,317.2 | +36.7 |
| 2 | 2019 H2 | 1,021.9 | 930.6 | -91.3 |
| 3 | 2020 H1 | 1,324.9 | 1,281.4 | -43.5 |
| 4 | 2020 Q3 | 970.1 | 888.0 | -82.0 |

Mean -45.0 MW, helping in 3 folds of 4 — but the fold-to-fold range is 128 MW.

### The effect is seasonal

Mean absolute error by month, test period:

| month | none | with weather | delta |
|---|---:|---:|---:|
| Jan | 1,221 | 1,409 | **+15.4%** |
| Feb | 1,026 | 1,018 | -0.8% |
| Mar | 1,430 | 1,435 | +0.3% |
| Apr | 1,555 | 1,526 | -1.8% |
| May | 1,376 | 1,423 | +3.4% |
| Jun | 1,253 | 1,258 | +0.3% |
| Jul | 918 | 799 | **-13.0%** |
| Aug | 1,063 | 894 | **-15.9%** |
| Sep | 980 | 945 | -3.6% |
| Oct | 1,046 | 1,072 | +2.5% |
| Nov | 1,021 | 1,016 | -0.5% |
| Dec | 1,385 | 1,222 | -11.8% |

**Summer (Jun-Aug): -95 MW. Winter (Nov-Mar): +3 MW.**

Temperature is worth 13-16% in July and August and nothing across winter on net.
January is 15% *worse* with weather and I have no explanation for it; possibly a
cold snap the noise handled badly, possibly overfitting.

### Why weather adds so little on average

Temperature is 96% autocorrelated at 24 hours:

| lag | corr with now |
|---|---:|
| 1h | +0.996 |
| **24h** | **+0.962** |
| 168h | +0.839 |

The strongest feature in the model is `lag_24h` — yesterday's demand at the same
hour. Yesterday's demand already contains yesterday's weather, and yesterday's
weather is 96% of today's weather. So explicit temperature is largely
re-delivering information the model already has.

Measured on daily means:

```
corr(load today, temp today)      = -0.360
corr(load today, temp yesterday)  = -0.360
```

Yesterday's temperature predicts today's demand exactly as well as today's does.

In winter the calendar features carry the rest — it is cold every day, so "it is
January at 18:00" is already most of the answer. In summer the calendar cannot
tell a cool August from a hot one, so temperature earns its keep.

A further reason the effect is muted in Germany: cooling degree hours (above
22 degC) are active in only 5.5% of hours, against 72% for heating. The
demand-temperature curve is a V, but the right-hand arm is thinly populated.

---

## Where this connects to AI weather models

The four modes above are a forecast-value framework, and that is the question
asked of GraphCast, Pangu-Weather, AIFS and the rest once the headline RMSE is
in: the model verifies better against analysis, but does the downstream user's
decision actually improve?

This answers it for one downstream user - a German day-ahead load forecast -
using a stand-in for forecast error rather than a real forecast. The honest
version of the study swaps `noisy` for archived operational forecasts, IFS and
an AI model, same issue times and same valid times, and rescores the load
forecast under each. None of the machinery here changes; only the temperature
series does.

Three things the numbers already say, worth knowing before running that study:

- **The ceiling is low on average.** `perfect` - exact temperature at the target
  hour, unattainable by any model - buys 0.4% over no weather at all, and that is
  inside the seed noise. Most of the information is already in `lag_24h`, because
  temperature is 96% autocorrelated at 24 hours.
- **The ceiling is not low in summer.** July and August are 13-16%. A weather
  model that is better in a heatwave is worth more to this user than its annual
  RMSE suggests, and an annual average would hide that entirely.
- **Verification has to be paired and multi-window.** One test period and one
  seed overstated the effect roughly twofold here.

Two things this is not. The models are gradient boosting and a small MLP over
tabular features, not weather models. And ERA5 is reanalysis, so nothing here is
a statement about any operational forecast's skill.

---

## Running it

```bash
pip install -r requirements.txt

python run_comparison.py                    # no weather
python run_comparison.py --weather noisy    # with weather
python run_comparison.py --backtest         # rolling-origin folds
python selfcheck.py                         # 13 correctness checks
```

`data/era5_temp_de.csv` is committed, so the weather modes run without a
Copernicus account. `python -m src.download_era5` regenerates the raw NetCDF if
wanted; that needs a free CDS account and is not required.

```
src/          data loading, features, models, evaluation, ERA5 download
data/         committed inputs - OPSD load extract, derived ERA5 series
results/      scores, predictions, backtest, figures/
selfcheck.py  13 correctness checks, synthetic data, no network
old-models/   the original single-file version, kept for reference
```

---

## What was checked

- **No lookahead, tested rather than asserted.** `selfcheck.py` spikes one value
  in the load series, rebuilds the features and asserts nothing within the next
  24 hours moved. The comparison is `>=`, not `>` — an earlier version used
  strict inequality, which would have permitted a feature reading the value *at*
  the poked hour.
- **Both weather directions.** `lagged` must not react within 24h; `perfect`
  *must* react at the target hour. Otherwise the two modes could quietly
  collapse into each other.
- **Fitting is independent of the test set.** Wrecking the test period by 7.5x
  and refitting leaves the trained MLP bit-identical.
- **Same rows for every model**, and the check that verifies this is itself
  tested — confirmed to fail when rows genuinely differ.
- **Calendar features are Europe/Berlin, not UTC.** Getting this wrong shifts
  every hour-of-day feature by 1-2 hours.
- **Holiday table validated** against the `holidays` package: 55/55 exact match
  for German national holidays 2015-2020, including the one-off Reformation Day
  in 2017.
- **Committed results reproduce bit-for-bit**, MLP included.

---

## Limitations

- **No forecast-origin structure.** A single assumed midnight cutoff, so there is
  no real lead-time verification. That needs issue time, valid time and lead time
  carried explicitly, with the same valid hour forecast from several origins. It
  also means error-by-hour cannot separate "forecast decays with horizon" from
  "afternoon load is harder".
- **The ENTSO-E comparison is not like-for-like.** Under Regulation 543/2013 the
  first day-ahead load forecast is published at least two hours before gate
  closure, around 10:00 on D-1 for Germany. This model assumes midnight, so it
  has roughly fourteen more hours of demand data. OPSD keeps target timestamps
  but no forecast vintage, so a given value may be a later revision. Reported
  because it is the right thing to measure against, not as a win.
- **ERA5 is reanalysis, not forecast.** It is the best estimate of what the
  weather *was*, assembled after the fact. `noisy` is a crude stand-in for
  forecast error: real forecast error is autocorrelated and state-dependent,
  this noise is neither.
- **One national temperature number**, an unweighted average over a box that
  includes the North Sea and part of Poland. A land mask or population weighting
  would be better.
- **No wind or solar.** Load is only half the picture in a renewables-heavy grid.
- **Fold results are not independent.** Folds share training data and load is
  serially correlated, so 3-of-4 is a stability signal, not four coin flips. A
  Diebold-Mariano test on paired errors would be the proper check.
- **The test period contains COVID.** No model trained on 2015-2018 was going to
  handle spring 2020.
- **Reproducibility is pinned, not guaranteed.** Versions are pinned in
  `requirements.txt`; different BLAS or hardware can still shift the last digits.

---

## Bugs found and fixed after review

- **Silent early stopping.** scikit-learn's `HistGradientBoostingRegressor`
  defaults to `early_stopping="auto"`, which switches itself on above 10,000 rows
  and carves an internal 10% validation slice out of training data. With 25,800
  rows it was active without my knowing, so `max_iter=600` was really running ~95
  iterations and the grid over [300, 600] was tuning a parameter the model
  ignored. Now explicitly off, so the external validation fold is the only one.
- **UTC vs local time.** Calendar features were built on UTC timestamps. Germany
  is UTC+1/+2, so every hour-of-day and is-weekend feature was offset.
- **Leakage test off by one.** The perturbation check used `>` where it needed
  `>=`, so a feature using the value at the poked hour would have passed.
- **NetCDF multi-file read.** `xarray.open_mfdataset` needs dask, which is not a
  dependency; the single-file path masked it. Files are now opened individually
  and concatenated.
- **Two worst days are both Corpus Christi** (2019-06-20, 2020-06-11) — a public
  holiday in some German states but not all, so it is absent from the national
  list and treated as a working day. Plausible but unconfirmed; the ablation that
  would prove it has not been run.
