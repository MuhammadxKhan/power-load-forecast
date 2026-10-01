# Analysis

The workings behind [README.md](README.md). Unless a section says otherwise
everything is gradient boosting, forecast issued at 10:00 on D-1, regional
holidays on, test period 2019-01-01 to 2020-09-30. Each table says which run it
comes from; anything computed from `results/predictions.csv` can be checked
against the committed file.

---

## The TSO comparison

From `python run_comparison.py` (the headline run).

| | gradient boosting | TSO forecast |
|---|---:|---:|
| MAE (MW) | 1,258.6 | 1,761.8 |
| bias (MW) | +189 | -608 |
| MAE after removing its own monthly bias | 1,219.6 | 1,332.6 |

The TSO forecast's monthly bias runs from -3,148 MW to +1,847 MW, and is more
than 1 GW either way in 10 of the 22 test months. Removing each forecast's own
monthly average error shrinks the gap from 503 MW to 113, so about 78% of it is
level. That correction uses the test period itself, so it's a diagnostic, not
something you could do live. A correction you could do live, subtracting the
TSO forecast's average 2018 bias (-475 MW), only gets it to 1,694 MW.

By half-year, with a Diebold-Mariano test on the daily mean absolute errors
(Newey-West variance, 7 lags; negative means gradient boosting is better):

| period | gradient boosting | TSO forecast | DM stat | p |
|---|---:|---:|---:|---:|
| 2019 H1 | 1,326 | 2,438 | -6.25 | <0.001 |
| 2019 H2 | 1,084 | 1,548 | -4.86 | <0.001 |
| 2020 H1 | 1,401 | 1,683 | -4.34 | <0.001 |
| 2020 Jul-Sep | 1,193 | 1,017 | +2.16 | 0.031 |
| whole test period | 1,259 | 1,762 | -6.44 | 1.2e-10 |

Gradient boosting has the lower daily error on 64% of the 640 test days. The
TSO forecast wins the last quarter outright, when its bias is far smaller than
it was through 2019 (-562 MW against -1,334 MW).

The level/shape split, from the committed predictions:

```python
import pandas as pd
p = pd.read_csv("results/predictions.csv", parse_dates=["timestamp"], index_col="timestamp")
month = p.index.tz_convert("Europe/Berlin").strftime("%Y-%m")
for c in ("gbm", "entsoe_benchmark"):
    e = p[c] - p["actual"]
    print(c, e.abs().mean(), (e - e.groupby(month).transform("mean")).abs().mean())
```

---

## Bias by period

Mean of forecast minus actual, MW, from the headline run:

| period | gradient boosting | MLP | TSO forecast |
|---|---:|---:|---:|
| 2019 (full year) | +84 | +459 | -1,334 |
| 2020 Jan-Feb | -124 | +93 | -130 |
| 2020 Mar-Jun | +680 | +1,563 | +1,294 |
| 2020 Jul-Sep | +162 | +1,039 | -562 |

In a normal year gradient boosting is close to unbiased: +84 MW across 2019 is
0.15% of mean demand (54,729 MW). Most of its overall +189 MW is the spring 2020
lockdown, when demand fell and nothing trained on 2015-2018 could have known.
The TSO forecast over-predicted by more in the same months. The MLP runs high
all the way through and badly so from March 2020; refitting helps it a lot, see
[below](#gradient-boosting-vs-the-mlp).

---

## Does the weather gain hold up?

`noisy` draws one realisation of the synthetic error, so one run is one draw.
Ten seeds:

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do python run_comparison.py --weather noisy --seed $s --no-plots; done
```

| | MAE (MW) |
|---|---:|
| mean | 1,202.4 |
| standard deviation | 8.5 |
| range | 1,186.5 to 1,216.1 |
| spread | 29.6 |
| mean gain against `none` | 56.2 |

The gain is about twice the spread, so it survives. Seed 0, the one a single
run reports, comes 3rd of 10. `perfect` (1,198.8) falls inside the range,
which is the point made in the README: with independent hourly error, `noisy`
and `perfect` can't be told apart.

Rolling-origin backtest, six-month test blocks, each fold trained on everything
before it and tuned on the year before its test block:

```bash
python run_comparison.py --backtest --no-plots
python run_comparison.py --weather noisy --backtest --no-plots
```

Both runs are kept side by side in `results/backtest.csv`.

| fold | test period | `none` | `noisy` | change |
|---|---|---:|---:|---:|
| 1 | 2019 H1 | 1,326.8 | 1,286.4 | -40.4 |
| 2 | 2019 H2 | 1,105.5 | 980.5 | -125.1 |
| 3 | 2020 H1 | 1,312.7 | 1,260.4 | -52.4 |
| 4 | 2020 Jul-Sep | 1,071.6 | 954.3 | -117.4 |

It helps in all four, by 83.8 MW on average. The two second halves gain the
most, which is the seasonal result again. Folds share training data and load is
serially correlated, so four out of four is a stability signal, not four
independent trials.

---

## The effect is seasonal

MAE by calendar month, `none` against `noisy` (seed 0), from the two
single-split runs:

| month | `none` | `noisy` | change |
|---|---:|---:|---:|
| Jan | 1,387 | 1,531 | +10.5% |
| Feb | 1,134 | 1,114 | -1.8% |
| Mar | 1,675 | 1,520 | -9.3% |
| Apr | 1,639 | 1,477 | -9.9% |
| May | 1,309 | 1,360 | +3.9% |
| Jun | 1,020 | 979 | -4.0% |
| Jul | 965 | 850 | -11.9% |
| Aug | 1,277 | 1,062 | -16.8% |
| Sep | 1,109 | 1,060 | -4.4% |
| Oct | 962 | 1,001 | +4.1% |
| Nov | 1,070 | 948 | -11.4% |
| Dec | 1,350 | 1,238 | -8.3% |

June to August: -124 MW (-11.4%). November to March: -37 MW (-2.7%).

Each month appears only once or twice in the test period, so single months are
noisy. January getting worse with weather is the clearest oddity and I don't
have an explanation for it.

---

## Why weather adds less than you'd expect

Temperature is very persistent:

| lag | correlation with now |
|---|---:|
| 1h | +0.996 |
| 24h | +0.962 |
| 168h | +0.839 |

On daily means, yesterday's temperature predicts today's demand as well as
today's temperature does (correlation -0.362 against -0.360). So recent demand
already carries most of the recent weather, and an explicit temperature feature
partly re-delivers it. That effect is weaker at 10:00 than at midnight, because
the model sees less of yesterday:

| issue time | `none` | `noisy` (seed 0) | change |
|---|---:|---:|---:|
| midnight | 1,160.2 | 1,122.7 | -3.2% |
| 10:00 on D-1 | 1,258.6 | 1,195.9 | -5.0% |

(`--issue midnight` and `--issue midnight --weather noisy` for the first row.)

The demand curve is also lopsided. Daily means on working days, 2015-2019:

| daily mean temperature (°C) | below -5 | -5 to 0 | 0 to 5 | 5 to 10 | 10 to 15 | 15 to 18 | 18 to 21 | 21 to 24 | above 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| demand (GW) | 65.5 | 64.0 | 62.2 | 59.9 | 57.6 | 56.8 | 56.9 | 57.7 | 58.4 |
| days | 9 | 72 | 291 | 281 | 243 | 171 | 138 | 46 | 12 |

Heating is a real slope; cooling barely registers, and there are only 58 days
above 21 °C to see it in. Heating degree hours are active 72.0% of the time and
cooling degree hours 5.5%.

---

## Holidays

```bash
python run_comparison.py --holidays national --no-plots
```

| | national only | with regional |
|---|---:|---:|
| gradient boosting MAE | 1,298.3 | 1,258.6 |
| MLP | 1,451.2 | 1,334.4 |
| ridge | 1,864.0 | 1,667.8 |

Worst days with national holidays only, and the same days with regional ones
(daily MAE, MW):

| day | national only | with regional | |
|---|---:|---:|---|
| 2020-06-11 | 8,203 | 1,806 | Corpus Christi |
| 2019-06-20 | 8,117 | 912 | Corpus Christi |
| 2020-04-09 | 5,740 | 5,584 | Maundy Thursday, lockdown |
| 2019-04-21 | 5,627 | 4,093 | Easter Sunday |
| 2020-04-12 | 4,935 | 3,293 | Easter Sunday, lockdown |
| 2019-05-02 | 4,404 | 3,923 | day after May Day |
| 2019-09-30 | 3,964 | 4,150 | ordinary Monday |
| 2020-03-24 | 3,906 | 4,447 | lockdown |

The holiday share comes from the `holidays` package per state, weighted by
state population, so Corpus Christi counts as 0.64, Epiphany 0.32, All Saints'
Day 0.57. The features are the share for the target day, for 1, 2 and 7 days
before it (the days the lags land on), and a flag for 24-31 December.

30 September 2019 is the one I can't explain. The actuals look like any other
Monday that autumn (daily mean 57.7 GW against 57.0 and 57.4 on the Mondays
either side) and the model is 3-6 GW low in every hour. It's bad at midnight
too (9th worst day there), so it isn't the issue time.

---

## Midnight vs 10:00

| | midnight | 10:00 on D-1 | change |
|---|---:|---:|---:|
| gradient boosting | 1,160.2 | 1,258.6 | +98.4 |
| MLP | 1,208.2 | 1,334.4 | +126.2 |
| ridge | 1,694.6 | 1,667.8 | -26.8 |

Both with regional holidays. With national holidays only, midnight gives
1,200.6, the original published number. Ridge is the odd one out and gets
slightly better. My guess is that the single snapshot of the latest load, shared
by every hour of D, is easier for a linear model to use than the per-hour lags,
but I haven't tested that.

Error by hour of the target day, gradient boosting at 10:00: about 1,000 MW
from midnight to 04:00, rising to 1,544 MW at 16:00. With one issue time per
day the hour of day and the forecast horizon are the same thing, so this can't
say how much of that is horizon and how much is afternoon load being harder.

---

## Gradient boosting vs the MLP

On the single split gradient boosting wins (1,258.6 against 1,334.4), but the
difference isn't significant (Diebold-Mariano stat -1.72, p = 0.086). In the
rolling backtest the order flips:

| fold | test period | gradient boosting | MLP | ridge |
|---|---|---:|---:|---:|
| 1 | 2019 H1 | 1,327 | 1,204 | 1,874 |
| 2 | 2019 H2 | 1,106 | 1,011 | 1,587 |
| 3 | 2020 H1 | 1,313 | 1,419 | 1,774 |
| 4 | 2020 Jul-Sep | 1,072 | 904 | 1,197 |
| mean | | 1,204 | 1,134 | 1,608 |

The MLP wins three folds of four and loses only the lockdown half-year, and its
average across folds is lower. Trained once on 2015-2018 and left alone for 21
months it drifts high instead (+238 MW in 2019 H1, over +1,000 MW in 2020), and
refitting is what fixes it: in July-September 2020 its error is 1,317 MW from
the single split and 904 MW when refitted. Gradient boosting gains much less
from refitting (1,193 to 1,072 in the same months). So which model is "best"
depends on how often you'd refit it, and the headline table on its own would
give the wrong idea.

---

## What was checked

`python selfcheck.py` runs 17 checks on synthetic data, with no network, in CI on
every push.

- **No lookahead, at either issue time.** One load value is spiked, the features
  are rebuilt, and no row may react to a value newer than it is allowed to see:
  09:00 on D-1 for the 10:00 setup, 24 hours back for midnight. This is done for
  every hour of a whole day, because at 10:00 what a row may see depends on its
  hour. The check is itself tested: slipping a 24-hour lag into the 10:00
  features has to make it fail.
- **Both weather directions.** `lagged` must not react to temperature measured
  after the issue time; `perfect` must react at the target hour.
- **Baselines.** At 10:00 no baseline may use "same hour yesterday".
- **Regional holidays.** Corpus Christi between 0.5 and 0.8, Christmas 1, an
  ordinary Tuesday 0. The national table matches the `holidays` package 55/55
  for 2015-2020, including the one-off Reformation Day in 2017.
- **Diebold-Mariano.** A forecast with a third of the error has to win with
  p < 0.01, and swapping the two must flip the sign.
- **Fitting is independent of the test set.** Wrecking the test period by 7.5x
  and refitting leaves the trained MLP bit-identical.
- **Same rows for every model**, and that check fails when rows differ.
- **Calendar features are Europe/Berlin, not UTC.**

Outside `selfcheck.py`: `--issue midnight --holidays national` reproduces the
original published table exactly, and rerunning the headline reproduces the
committed `scores.csv` and `predictions.csv` byte for byte.

---

## Limitations

- **One issue time per day.** Real lead-time verification needs the same valid
  hour forecast from several issue times, with issue, valid and lead time
  carried explicitly.
- **The TSO values have no vintage.** OPSD keeps target timestamps but not when
  a value was published, so some may be later revisions. And the published
  forecast is a transparency obligation, not necessarily the one the TSOs run
  the grid on.
- **ERA5 is reanalysis, and `noisy` understates real forecast error.** The
  synthetic error is independent hour to hour, so the 24-hour rolling
  temperature averages most of it away (1.00 °C down to about 0.2 °C). Real
  forecast error is autocorrelated. In the earlier midnight setup, AR(1) error
  with the same size roughly doubled the spread across draws. Archived
  operational forecasts are the fix.
- **One national temperature**, an unweighted average over a box that includes
  the North Sea and parts of neighbouring countries. Population weighting would
  be better.
- **Holidays are state-level.** Corpus Christi is also a holiday in parts of
  Saxony and Thuringia, and Assumption Day in parts of Bavaria; neither is
  counted. School holidays aren't modelled.
- **No wind or solar.**
- **The test period contains the spring 2020 lockdown.**
- **Day-of-year is on a 365-day cycle**, so leap years drift by a day in the
  seasonal terms. Small, but wrong.
- **Reproducibility is pinned, not guaranteed.** Versions are pinned in
  `requirements.txt`; different hardware can still move the last digits,
  especially for the MLP.
