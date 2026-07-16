"""
Features and the train/val/test split.

This is the only place either of those is defined. Every model imports from
here, so no model can train on a different feature set or a different split
than the one it's compared against.

Two rules everything hangs on.

1. The load rule. We forecast day D at midnight, so the newest demand figure we
   have is 23:00 on day D-1. No feature touches the load series at a lag under
   24 hours. selfcheck.py pokes the series and checks nothing reacts too soon.

2. Weather is different, and this is the bit worth being careful about. You
   genuinely do have a weather forecast for tomorrow when you make a load
   forecast - that is how it works in practice. So temperature at the target
   hour is not automatically cheating. What IS cheating is using ERA5's
   after-the-fact reanalysis and pretending it was a forecast, because a real
   forecast has error in it. weather_mode below makes the choice explicit
   instead of burying it:

     "lagged"   only temperature from 24h+ ago. Nothing to argue about.
     "noisy"    target-hour temperature with synthetic error added. A
                SENSITIVITY TEST, not a forecast - see below.
     "perfect"  target-hour temperature exactly. Perfect prognosis - an upper
                bound on what weather can buy you, not a deployable result.
                The gap between "perfect" and "noisy" is the cost of the
                imposed error model, NOT the measured cost of real forecast
                error. Only an archived forecast answers that.
     "none"     no weather at all, i.e. the old model.

Everything calendar-related is computed in Europe/Berlin, not UTC. The index
stays UTC because that has no daylight-saving ambiguity, but German demand
follows German clocks: 23:00 UTC on 31 December is already New Year's Day in
Germany. Getting this wrong mislabels the hour on every single row and the
weekday on about 7% of them.
"""

import numpy as np
import pandas as pd

LAGS = [24, 48, 72, 168, 336]
TZ = "Europe/Berlin"

# Size of the synthetic error injected by weather_mode="noisy", degrees C.
#
# Be clear about what this is NOT. It is not a weather forecast. It starts from
# ERA5 truth and adds independent Gaussian noise, so it assumes the error is
# unbiased, uncorrelated hour to hour, the same at every lead time, the same in
# every season, and the same everywhere in the country. Real forecast error is
# none of those things. Using an archived operational forecast is the only way
# to answer this properly, and I haven't got one.
#
# So this measures the cost of temperature error UNDER THIS IMPOSED ERROR MODEL,
# and nothing more. 1.0 is a plausible order of magnitude for day-ahead 2m
# temperature, not a measured value for any particular model or region.
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

def _cyclical(values, period):
    r = 2 * np.pi * values / period
    return np.sin(r), np.cos(r)


def build_features(load, temp=None, weather_mode="none", seed=0):
    df = pd.DataFrame({"load_mw": load})
    idx = df.index
    loc = idx.tz_convert(TZ)   # German clocks, not UTC

    df["hour"] = loc.hour
    df["dayofweek"] = loc.dayofweek
    df["month"] = loc.month
    df["is_weekend"] = (loc.dayofweek >= 5).astype(int)
    df["is_holiday"] = loc.strftime("%Y-%m-%d").isin(HOLIDAYS).astype(int)

    df["hour_sin"], df["hour_cos"] = _cyclical(loc.hour.to_numpy(), 24)
    df["dow_sin"], df["dow_cos"] = _cyclical(loc.dayofweek.to_numpy(), 7)
    df["doy_sin"], df["doy_cos"] = _cyclical(loc.dayofyear.to_numpy(), 365)

    for lag in LAGS:
        df[f"lag_{lag}h"] = df["load_mw"].shift(lag)

    past = df["load_mw"].shift(24)  # everything rolls off the 24h-lagged series
    df["roll_mean_24h"] = past.rolling(24).mean()
    df["roll_mean_168h"] = past.rolling(168).mean()
    df["roll_std_24h"] = past.rolling(24).std()

    df["same_hour_3wk_mean"] = (
        df["load_mw"].shift(168) + df["load_mw"].shift(336) + df["load_mw"].shift(504)
    ) / 3

    t = usable_temperature(temp, weather_mode, seed)
    if t is not None:
        t = t.reindex(idx)
        df["temp_c"] = t
        df["hdh"], df["cdh"] = degree_hours(t)
        # buildings have thermal inertia - today's demand responds to the last
        # day of weather, not just this instant
        df["temp_roll_mean_24h"] = t.rolling(24).mean()
        # warming or cooling relative to the same hour yesterday
        df["temp_change_24h"] = t - t.shift(24)

    df = df.dropna()
    y = df.pop("load_mw")
    return df, y


def _as_utc(t):
    """Accept "2019-01-01" or an already-tz-aware Timestamp. the backtest in evaluate.py
    builds its fold boundaries by date arithmetic, so they arrive already aware."""
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")

