# Day-ahead electricity load forecasting

Predicting Germany's hourly electricity demand one day ahead, using data from
[Open Power System Data](https://data.open-power-system-data.org/time_series/)
(ENTSO-E figures, no API key needed). 2015–2020, hourly.

I mostly wanted to see whether I could beat the obvious baseline, and to be
careful about not accidentally cheating while doing it.

```bash
pip install numpy pandas scikit-learn
python load_forecast.py            # Germany, downloads ~200MB once then caches
python load_forecast.py --country GB
python load_forecast.py --test     # quick self-check, no download
```

## The setup

You're at midnight, forecasting all 24 hours of the coming day. So the most
recent data you're allowed to use is 23:00 the day before.

This is the whole reason the features look the way they do. Nothing uses a lag
shorter than 24 hours. If you let a 1-hour lag in, the error drops through the
floor and you feel great, but the model is useless because in reality you don't
have the last hour's demand when forecasting a full day ahead. I got this wrong
in an earlier version, which is why there's now a check for it (see `--test`):
it spikes one value in the load series and confirms nothing within the next 24
hours reacts to it.

## The baseline

Before any model, the number to beat is **seasonal naive**: assume this hour
will be the same as the same hour, same weekday, last week. It's one line and
it's annoyingly good, because electricity demand is very habitual — weekdays
look like weekdays, weekends like weekends. If a model can't beat "just copy
last week", it isn't doing anything.

So the real question the whole project answers is: can a model beat
copy-last-week, and by how much? The metric I use for that is the skill score,
`1 - MAE_model / MAE_baseline` — the share of the baseline's error that the
model removes. 0 means no better than copying last week.

## Results

Trained on 2015–2017, tuned on 2018, and tested once on 2019 to Sep 2020.

| model | MAE (MW) | RMSE (MW) | MAPE | skill vs naive |
|---|---:|---:|---:|---:|
| gbm | 1,220 | 1,637 | 2.31% | 0.495 |
| ridge | 1,837 | 2,514 | 3.48% | 0.240 |
| seasonal naive | 2,416 | 4,184 | 4.55% | 0.000 |
| mean of last 4 weeks | 2,648 | 4,134 | 4.95% | -0.096 |
| yesterday | 4,340 | 6,620 | 8.09% | -0.796 |

Gradient boosting cut the baseline's error roughly in half (2,416 -> 1,220 MW).
Ridge on the same features got about halfway there. Worth noting the "yesterday"
row is the worst of all — that's the point of using seasonal naive instead of a
plain 24-hour lag, since a Saturday looks nothing like the Friday before it.

Ridge barely changed across regularisation strengths (alpha 0.1 to 100), which
makes sense — with ~50k rows and 20 features there's not much overfitting for it
to fix. I kept the tuning small on purpose.

## Where it goes wrong

Error is fairly even across the day, a bit worse in the afternoon peak. The
interesting bit is the worst individual days:

| date | MAE (MW) | what happened |
|---|---:|---|
| 2019-06-20 | 8,212 | Corpus Christi — a holiday in some German states but not all |
| 2020-06-11 | 7,820 | Corpus Christi again |
| 2019-04-21 | 4,771 | Easter Sunday |
| 2020-04-09 | 4,385 | COVID demand drop |
| 2020-04-12 | 3,981 | Easter / COVID |

Two things break it. Regional holidays it can't see — Corpus Christi is only a
public holiday in some states, so it's not in my (national) holiday list and the
model treats it as a normal working day. And spring 2020, which is COVID, and no
model trained on 2015–2018 was going to see that coming. Both of those are
fixable-ish and I've listed them below rather than pretending the model is just
"a bit noisy".

## Things I'd add with more time

- A proper holiday calendar with state-level holidays. That alone kills the two
  worst days.
- Temperature data. This is the big missing piece — demand is heavily driven by
  heating and cooling, and right now the model only knows the season from the
  calendar, not the actual weather.
- A separate model per hour, since 3am and 2pm really don't behave the same.
- Some kind of prediction interval, not just a single number.
- Prices instead of load, eventually. Load is smooth and well-behaved; prices
  spike, go negative, and switch regimes, so that's a much harder problem. I
  picked load on purpose to keep the scope sane.

## Known limitations

- No weather data, which is the biggest gap by far.
- One country, one model, one test window — I'm not claiming this generalises.
- Holidays are hardcoded, national-only, for Germany 2015–2020, which is exactly
  why the regional-holiday days blow up.
- The <1% of missing hours are interpolated rather than dropped, because
  dropping them would quietly break the fixed 24h and 168h lags.
- The gradient boosting is only lightly tuned (four settings on the validation
  year). I'd rather report an honest lightly-tuned number than keep poking the
  test set until it looks good.

## What's in the file

It's all in `load_forecast.py`: downloading and cleaning the data, building the
features, the baselines, ridge and gradient boosting with a time-ordered
train/val/test split, the metrics, and the `--test` self-checks including the
leakage one.
