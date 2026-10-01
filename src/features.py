"""
Features and the train/val/test split.

Defined only here, so no model can train on a different feature set or split
than the one it is compared against.

Two rules.

1. Load. The forecast for day D is made at 10:00 on D-1, which is when the TSOs
   publish theirs (two hours before the day-ahead auction closes at 12:00).
   ENTSO-E publishes actual load up to an hour after the hour ends, so the
   newest hour the model may see is the one starting 08:00 on D-1. Every load
   feature for day D is built from data up to that hour and nothing later.

   issue="midnight" is the original setup: forecast at midnight, every load
   lag at least 24 hours. It is too late for the day-ahead auction and gives
   the model about fourteen hours more data than the TSO forecast had, so it is
   kept for comparison only. selfcheck.py checks both.

2. Weather. You genuinely do have a forecast for tomorrow, so target-hour
   temperature is not automatically cheating; using ERA5 reanalysis and calling
   it a forecast is. weather_mode makes the choice explicit:

     "none"     no weather at all
     "lagged"   only temperature already measured at the forecast time
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

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

TZ = "Europe/Berlin"
ISSUES = ("10am", "midnight")

# 10:00 issue: same-hour lags that are always published by 10:00 on D-1, plus
# a snapshot of the latest data (see last_known).
ISSUE_LAGS = [48, 72, 168, 336]
ISSUE_HOUR = 10
LAST_KNOWN_HOUR = ISSUE_HOUR - 2   # the 08:00-09:00 value is out by 10:00

# midnight issue: the original lags
LAGS = [24, 48, 72, 168, 336]

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
# these and the weekday features can't see them.
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

# Population by state in millions (2020). Corpus Christi, All Saints' Day,
# Epiphany and the rest are holidays in some states only, and the national list
# above treats them as working days. That was the model's worst failure, so
# each date gets the share of the population that has the day off.
STATE_POPULATION = {
    "BW": 11.10, "BY": 13.14, "BE": 3.66, "BB": 2.53, "HB": 0.68, "HH": 1.85,
    "HE": 6.29, "MV": 1.61, "NI": 8.00, "NW": 17.93, "RP": 4.10, "SL": 0.98,
    "SN": 4.06, "ST": 2.18, "SH": 2.91, "TH": 2.12,
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


def last_known(index):
    """For each target hour, the start of the newest load hour that is out by
    10:00 local on the day before. All 24 hours of day D share one value."""
    day = index.tz_convert(TZ).normalize().tz_localize(None)
    lk = day - pd.Timedelta(days=1) + pd.Timedelta(hours=LAST_KNOWN_HOUR)
    return lk.tz_localize(TZ).tz_convert("UTC")


def newest_usable(index, issue="10am"):
    """The newest load timestamp each row is allowed to depend on. selfcheck.py
    pokes the series and checks nothing reacts to a value later than this."""
    if issue == "midnight":
        return index - pd.Timedelta(hours=24)
    return last_known(index)


@lru_cache(maxsize=None)
def _state_calendars(years):
    return {s: holidays.Germany(subdiv=s, years=years) for s in STATE_POPULATION}


def holiday_share(dates):
    """Share of Germany's population on a public holiday, per local date.

    1.0 on national holidays, 0.64 on Corpus Christi, 0.32 on Epiphany.
    `dates` is a naive DatetimeIndex of local dates.
    """
    cal = _state_calendars(tuple(sorted(set(dates.year))))
    total = sum(STATE_POPULATION.values())
    days = dates.unique()
    share = {d: sum(p for s, p in STATE_POPULATION.items() if d in cal[s]) / total
             for d in days}
    return pd.Series(share).reindex(dates).to_numpy()


def usable_temperature(temp, weather_mode, seed=0, issue="10am"):
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
        # same hour on the most recent day that has been measured by then
        return temp.shift(24 if issue == "midnight" else 48)
    if weather_mode == "perfect":
        return temp
    # "noisy": truth plus synthetic error. One random realisation, seeded, so
    # the run reproduces - but one realisation is not an uncertainty estimate.
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, SYNTHETIC_TEMP_ERROR_C, len(temp))
    return temp + noise


def build_features(load, temp=None, weather_mode="none", seed=0, issue="10am",
                   regional_holidays=True):
    if issue not in ISSUES:
        raise ValueError(f"issue must be one of {ISSUES}")

    df = pd.DataFrame({"load_mw": load})
    idx = df.index
    loc = idx.tz_convert(TZ)   # German clocks, not UTC

    df["hour"] = loc.hour
    df["dayofweek"] = loc.dayofweek
    df["month"] = loc.month
    df["is_weekend"] = (loc.dayofweek >= 5).astype(int)
    df["is_holiday"] = loc.strftime("%Y-%m-%d").isin(HOLIDAYS).astype(int)

    if regional_holidays:
        day = loc.normalize().tz_localize(None)
        df["holiday_share"] = holiday_share(day)
        # The lags below land on these days. A holiday yesterday or a week ago
        # drags the lag down, and the model should know why.
        for back in (1, 2, 7):
            df[f"holiday_share_d{back}"] = holiday_share(day - pd.Timedelta(days=back))
        # 24-31 December: mostly not holidays, but plenty of people are off
        df["christmas_week"] = ((loc.month == 12) & (loc.day >= 24)).astype(int)

    df["hour_sin"], df["hour_cos"] = _cyclical(loc.hour.to_numpy(), 24)
    df["dow_sin"], df["dow_cos"] = _cyclical(loc.dayofweek.to_numpy(), 7)
    df["doy_sin"], df["doy_cos"] = _cyclical(loc.dayofyear.to_numpy(), 365)

    if issue == "midnight":
        for lag in LAGS:
            df[f"lag_{lag}h"] = df["load_mw"].shift(lag)
        past = df["load_mw"].shift(24)  # everything rolls off the 24h-lagged series
        df["roll_mean_24h"] = past.rolling(24).mean()
        df["roll_mean_168h"] = past.rolling(168).mean()
        df["roll_std_24h"] = past.rolling(24).std()
    else:
        for lag in ISSUE_LAGS:
            df[f"lag_{lag}h"] = df["load_mw"].shift(lag)
        # one snapshot of what is known at 10:00 on D-1, shared by all 24
        # hours of D - the way a forecast run actually works
        lk = last_known(idx)
        df["last_known"] = df["load_mw"].reindex(lk).to_numpy()
        df["roll_mean_24h"] = df["load_mw"].rolling(24).mean().reindex(lk).to_numpy()
        df["roll_mean_168h"] = df["load_mw"].rolling(168).mean().reindex(lk).to_numpy()
        df["roll_std_24h"] = df["load_mw"].rolling(24).std().reindex(lk).to_numpy()

    df["same_hour_3wk_mean"] = (
        df["load_mw"].shift(168) + df["load_mw"].shift(336) + df["load_mw"].shift(504)
    ) / 3

    t = usable_temperature(temp, weather_mode, seed, issue)
    if t is not None:
        t = t.reindex(idx)
        df["temp_c"] = t
        df["hdh"], df["cdh"] = degree_hours(t)
        if weather_mode == "lagged" and issue == "10am":
            # t.shift(24) would reach past 10:00 for most hours, so anchor on
            # the last measured hour instead
            measured = temp.reindex(idx)
            lk = last_known(idx)
            df["temp_roll_mean_24h"] = measured.rolling(24).mean().reindex(lk).to_numpy()
            df["temp_last_known"] = measured.reindex(lk).to_numpy()
        else:
            # buildings have thermal inertia - today's demand responds to the
            # last day of weather, not just this instant
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
