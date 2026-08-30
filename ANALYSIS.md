# Analysis

The workings behind the summary in [README.md](README.md). Every table here
regenerates from the code in this repo, in the pinned environment; the command
is given alongside each one.

---

## Bias is almost entirely COVID

The +258 MW annual bias on the gradient boosting model is not a standing
tendency to over-forecast. Split by period:

| period | gbm | MLP | ENTSO-E |
|---|---:|---:|---:|
| 2019 (full year) | **+94** | +109 | -1,333 |
| 2020 Jan-Feb | **-25** | +116 | -130 |
| **2020 Mar-Jun** | **+917** | +1,163 | **+1,291** |
| 2020 Jul-Sep | +217 | +373 | -562 |

Under normal conditions the model is close to unbiased - +94 MW over a full year
is 0.17% of mean demand (54,729 MW). Essentially all of the annual bias comes
from March-June 2020, when demand fell and nothing trained on 2015-2018 could
have known. The TSOs' own forecast over-predicted by more in the same window
(+1,291 MW) despite a much shorter horizon, which is worth knowing before
treating any single bias number as a property of a model rather than of a period.

---

## Does the gain survive reseeding?

`noisy` across ten random seeds, gradient boosting:

Mean 1,181.7 MW, std 4.1, range 1,174.5 to 1,185.5 - a spread of **11.0 MW**
against a mean gain of 18.9. It survives, but not by much. Seed 0, the default a
single run reports, lands 6th of ten: representative rather than flattering.

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do python run_comparison.py --weather noisy --seed $s --no-plots; done
```

The check matters more than its result: had the spread exceeded the gain, the
single-window number would have been noise reported as a finding.

---

## Across rolling folds

Rolling-origin backtest, gradient boosting, 6-month blocks
(`python run_comparison.py --backtest`):

| fold | test period | none | noisy | delta |
|---|---|---:|---:|---:|
| 1 | 2019 H1 | 1,276.2 | 1,310.5 | +34.3 |
| 2 | 2019 H2 | 1,024.1 | 979.8 | -44.4 |
| 3 | 2020 H1 | 1,301.4 | 1,302.0 | +0.6 |
| 4 | 2020 Q3 | 959.8 | 861.3 | -98.5 |

Mean -27.0 MW, but it helps clearly in only two folds of four, is neutral in a
third, and is worse in the first. The fold-to-fold range is 133 MW - five times
the mean effect. On this evidence "weather helps" is a claim about summer, not a
claim about the year.

---

## The effect is seasonal

Mean absolute error by month, test period:

| month | none | with weather | delta |
|---|---:|---:|---:|
| Jan | 1,214 | 1,367 | **+12.6%** |
| Feb | 1,031 | 1,015 | -1.5% |
| Mar | 1,428 | 1,491 | +4.4% |
| Apr | 1,578 | 1,528 | -3.2% |
| May | 1,395 | 1,403 | +0.6% |
| Jun | 1,254 | 1,222 | -2.6% |
| Jul | 922 | 802 | **-13.0%** |
| Aug | 1,061 | 887 | **-16.4%** |
| Sep | 970 | 983 | +1.4% |
| Oct | 1,042 | 1,133 | +8.7% |
| Nov | 1,065 | 1,032 | -3.1% |
| Dec | 1,379 | 1,269 | -7.9% |

**Summer (Jun-Aug): -109 MW. Winter (Nov-Mar): +11 MW.**

Temperature is worth 13-16% in July and August and nothing across winter on net.
January is 13% *worse* with weather and I have no explanation for it; possibly a
cold snap the noise handled badly, possibly overfitting.

---

## Why weather adds so little on average

Temperature is 96% autocorrelated at 24 hours:

| lag | corr with now |
|---|---:|
| 1h | +0.996 |
| **24h** | **+0.962** |
| 168h | +0.839 |

The strongest feature is `lag_24h`. Yesterday's demand already contains
yesterday's weather, and yesterday's weather is 96% of today's - so explicit
temperature largely re-delivers what the model has. On daily means:

```
corr(load today, temp today)      = -0.358
corr(load today, temp yesterday)  = -0.360
```

Yesterday's temperature predicts today's demand as well as today's does.

In winter the calendar carries the rest - it is cold every day, so "January at
18:00" is most of the answer. In summer the calendar cannot tell a cool August
from a hot one, so temperature earns its keep. Cooling degree hours (above
22 degC) are also active in only 5.5% of hours against 72% for heating, which is
the thin right arm of the V in the README's scatter plot.

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
- **ERA5 is reanalysis, and `noisy` understates its own uncertainty.** ERA5 is
  the best estimate of what the weather *was*. The synthetic error on top of it
  is independent hour to hour, so `temp_roll_mean_24h` averages it from 1.00 to
  0.21 degC (measured; 1/sqrt(24)); real forecast error is autocorrelated and
  does not. AR(1) error at phi=0.95, same marginal sigma, barely moves the mean
  gain (-16.1 MW against -18.9) but doubles the spread across draws (23.1 against
  11.0) - under a realistic error structure the spread exceeds the effect. The
  point estimate survives; the confidence in it does not. An archived
  operational forecast is the real fix.
- **One national temperature number**, an unweighted average over a box that
  includes the North Sea and part of Poland. A land mask or population weighting
  would be better.
- **No wind or solar.** Load is only half the picture in a renewables-heavy grid.
- **Fold results are not independent.** Folds share training data and load is
  serially correlated, so 2-of-4 is a weak stability signal, not four coin
  flips. A Diebold-Mariano test on paired errors would be the proper check.
- **Day-of-year is encoded on a 365-day cycle**, so leap years drift by one day
  in the seasonal sine and cosine. The effect is negligible but it is wrong.
- **The test period contains COVID.** No model trained on 2015-2018 was going to
  handle spring 2020.
- **Reproducibility is pinned, not guaranteed.** Versions are pinned in
  `requirements.txt`; different BLAS or hardware can still shift the last digits.
