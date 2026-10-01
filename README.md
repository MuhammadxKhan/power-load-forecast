# Day-ahead load forecasting for Germany

Forecasts German hourly electricity demand for the next day and measures what
better weather information would be worth to it.

The forecast for day D is made at **10:00 on D-1**, when the TSOs publish their
own day-ahead forecast and two hours before the day-ahead auction closes. The
first version forecast at midnight, which is too late for the auction and gave
the model about 14 more hours of data than the TSOs had.

Data: [OPSD](https://open-power-system-data.org/) (2020-10-06 release), German
hourly load 2015-2020, and ERA5 2m temperature. Train 2015-2017, tune 2018,
test 2019-01-01 to 2020-09-30. Detailed workings in [ANALYSIS.md](ANALYSIS.md).

## Results

No weather in this table.

| model | MAE (MW) | RMSE (MW) | MAPE | bias (MW) | skill vs naive |
|---|---:|---:|---:|---:|---:|
| **gradient boosting** | **1,259** | 1,654 | 2.34% | +189 | **0.479** |
| MLP (PyTorch) | 1,334 | 1,755 | 2.49% | +718 | 0.448 |
| ridge | 1,668 | 2,287 | 3.17% | +183 | 0.310 |
| TSO published forecast | 1,762 | 2,253 | 3.22% | -608 | 0.271 |
| seasonal naive (same hour last week) | 2,416 | 4,184 | 4.55% | -55 | 0 |
| mean of last 4 weeks | 2,648 | 4,134 | 4.94% | +75 | -0.096 |
| same hour two days ago | 7,075 | 9,354 | 13.28% | -32 | -1.928 |

Skill is `1 - MAE / MAE_naive`. "Same hour two days ago" replaces "same hour
yesterday", which isn't published yet at 10:00 (two days before a Monday is a
Saturday, hence the score).

The MLP isn't significantly worse than gradient boosting here (Diebold-Mariano
p = 0.09), and it wins three of four folds in the rolling backtest, where it
gets refitted every six months.

## Against the TSO forecast

Same issue time, same hours. Gradient boosting is lower over the test period
(1,259 vs 1,762 MW, Diebold-Mariano p < 1e-9), but about 78% of the gap is the
TSO forecast's level: it runs 608 MW low on average and its monthly bias ranges
from -3.1 to +1.8 GW. With each forecast's monthly bias removed it's 1,220 vs
1,333. In July-September 2020 the TSO forecast is better (p = 0.03).

| period | gradient boosting | TSO forecast |
|---|---:|---:|
| 2019 H1 | 1,326 | 2,438 |
| 2019 H2 | 1,084 | 1,548 |
| 2020 H1 | 1,401 | 1,683 |
| 2020 Jul-Sep | 1,193 | **1,017** |

OPSD doesn't record when each TSO value was published, so some may be later
revisions.

## Does weather help?

| mode | what the model gets | MAE (MW) | vs none |
|---|---|---:|---:|
| `none` | no weather | 1,258.6 | |
| `lagged` | temperature measured by 10:00 on D-1 | 1,230.4 | -28.2 |
| `noisy` | actual temperature + 1 °C random error | 1,195.9 | -62.7 |
| `perfect` | actual temperature at the target hour | 1,198.8 | -59.8 |

ERA5 is a reanalysis (a reconstruction of past weather), not a forecast, so
`noisy` and `perfect` are upper bounds rather than achievable results.

- Perfect temperature is worth about 5% overall (2.5% in the old midnight
  setup). By month, `noisy` gains most in July and August (12-17%).
- The gain holds over ten noise seeds (56 MW against a 30 MW spread) and in all
  four rolling folds.
- `noisy` and `perfect` can't be told apart: independent hourly noise mostly
  cancels out in the 24-hour average, which real forecast error wouldn't.

![German hourly demand against temperature, coloured by local hour](results/figures/load_vs_temperature.png)

The rise on the warm side is mostly time of day (hot hours are afternoon
hours). On daily averages, demand climbs about 8 GW from 15 °C down to below
-5 °C, and only about 1.5 GW from 20 °C up to the hottest days.

## Holidays

Corpus Christi is a holiday for about 64% of the population but not
nationally, and it was the model's worst day in both years. Each day now gets
the population share on holiday, the same for 1, 2 and 7 days earlier, and a
24-31 December flag.

| daily MAE (MW) | national holidays only | with regional |
|---|---:|---:|
| Corpus Christi 2019 | 8,117 | 912 |
| Corpus Christi 2020 | 8,203 | 1,806 |
| whole test period | 1,298 | 1,259 |

## Midnight vs 10:00

| | midnight | 10:00 on D-1 |
|---|---:|---:|
| gradient boosting | 1,160 | 1,259 |
| MLP | 1,208 | 1,334 |
| ridge | 1,695 | 1,668 |

Forecasting at 10:00 costs gradient boosting 98 MW (8.5%).
`--issue midnight --holidays national` reproduces the original table
(1,200.6 MW).

## Running it

```bash
pip install -r requirements.txt

python run_comparison.py                       # 10:00 on D-1, no weather
python run_comparison.py --weather noisy       # with weather
python run_comparison.py --backtest            # rolling-origin folds
python run_comparison.py --issue midnight      # original midnight setup
python run_comparison.py --holidays national   # without regional holidays
python selfcheck.py                            # 17 checks on synthetic data
```

The OPSD extract and the ERA5 series are committed, so no download or
Copernicus account is needed.

```
src/          data loading, features, models, evaluation, ERA5 download
data/         OPSD load extract, ERA5 temperature series
results/      scores, predictions, backtest, figures/
ANALYSIS.md   TSO comparison, bias, seeds, folds, months, holidays, limitations
selfcheck.py  leakage, calendar, weather-mode and scoring checks
old-models/   the original single-file version
```

## Bugs found and fixed

- **Forecast time.** Forecasting at midnight used 14 hours of data the TSOs
  didn't have. Now 10:00 on D-1.
- **A leak the leakage test missed.** An early `lagged` temperature for the
  10:00 setup used yesterday afternoon's temperature. The test poked one hour,
  which happened to be in the morning. It now pokes every hour of a day and is
  tested against a deliberately leaky feature.
- **Silent early stopping.** sklearn's gradient boosting turns early stopping
  on above 10,000 rows, so `max_iter` was being ignored. Now off.
- **UTC vs local time.** Calendar features were on UTC, offsetting every hour
  and weekend feature by 1-2 hours.
- **Leakage test off by one.** `>` instead of `>=` let a feature using the
  poked hour itself pass.
- **NetCDF reading.** `open_mfdataset` needs dask, which isn't installed; files
  are now read one at a time.

## Limits

One issue time per day, so no lead-time verification. TSO values may be later
revisions. ERA5 is reanalysis, not a forecast. Temperature is an unweighted
average over a box that includes the North Sea. No wind or solar. Data ends in
2020 (the last OPSD release) and the test period includes the 2020 lockdown.
[Full list](ANALYSIS.md#limitations).
