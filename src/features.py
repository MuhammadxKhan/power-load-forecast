"""
Features and the train/val/test split.

Defined only here, so no model can train on a different feature set or split
than the one it is compared against.

Two rules.

1. Load. We forecast day D at midnight, so the newest demand figure is 23:00 on
   D-1. No feature touches the load series at a lag under 24 hours.
   selfcheck.py pokes the series and checks nothing reacts too soon.

2. Weather. You genuinely do have a forecast for tomorrow, so target-hour
   temperature is not automatically cheating; using ERA5 reanalysis and calling
   it a forecast is. weather_mode makes the choice explicit:

     "none"     no weather at all
     "lagged"   temperature from 24h+ ago only
     "noisy"    target-hour temperature plus synthetic error - a sensitivity
                test, not a forecast
     "perfect"  target-hour temperature exactly - an upper bound, not a
                deployable result

   The perfect/noisy gap is the cost of the imposed error model, not of real
   forecast error. Only an archived forecast answers that.

Calendar features use Europe/Berlin; the index stays UTC to avoid DST
ambiguity. 23:00 UTC on 31 December is already New Year's Day in Germany, and
getting this wrong mislabels the hour on every row and the weekday on ~7%.
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
SYNTHETIC_TEMP_ERROR_C = 1.0

# German national public holidays 2015-2020, as LOCAL dates. Load drops hard on
# these and the weekday features can't see them. Regional holidays are missing
# and that's the model's worst failure - see the README.
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

WEATHER_MODES = ("none", "lagged", "noisy", "perfect")

# Hinge points for the heating/cooling terms. 15C/22C are conventional-ish for
# Europe but they are a CHOICE, not a fact, and the choice matters: a model can
# rescale a feature's magnitude but it cannot move where the hinge sits, so a
# badly placed base temperature is a badly placed kink that ridge in particular
# can't recover from. Worth a sensitivity check I haven't run.
HDD_BASE = 15.0
CDD_BASE = 22.0


def degree_hours(temp_c):
    """Heating and cooling degree HOURS - one-sided hinges on hourly temperature.

    Not degree days: those use the daily mean and accumulate over the day (see
    Eurostat). These are per-hour hinges on the instantaneous value.

    Load against temperature is V-shaped - demand rises when cold and again when
    hot - and a straight line cannot fit a V, which ridge in particular
    struggles with. Two one-sided variables turn the V into two straight arms.
    """
    return np.maximum(0.0, HDD_BASE - temp_c), np.maximum(0.0, temp_c - CDD_BASE)


def _cyclical(values, period):
    r = 2 * np.pi * values / period
    return np.sin(r), np.cos(r)


def usable_temperature(temp, weather_mode, seed=0):
    """The temperature the model is allowed to use for the target hour.

    Split out from build_features so selfcheck.py can test it on its own.
    """
    if weather_mode not in WEATHER_MODES:
        raise ValueError(f"weather_mode must be one of {WEATHER_MODES}")
    if weather_mode == "none":
        return None
    if temp is None:
        raise ValueError(f"weather_mode={weather_mode!r} needs a temperature series")

    if weather_mode == "lagged":
        return temp.shift(24)
    if weather_mode == "perfect":
        return temp
    # "noisy": truth plus synthetic error. One random realisation, seeded, so
    # the run reproduces - but one realisation is not an uncertainty estimate.
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, SYNTHETIC_TEMP_ERROR_C, len(temp))
    return temp + noise


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


def chronological_split(X, y, val_start, test_start):
    vs, ts = _as_utc(val_start), _as_utc(test_start)
    tr, va, te = X.index < vs, (X.index >= vs) & (X.index < ts), X.index >= ts
    if not (tr.any() and va.any() and te.any()):
        raise ValueError("a split came out empty - check your dates vs the data range")
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])
