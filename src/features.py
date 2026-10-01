"""
Features and the train/val/test split, defined in one place so every model
sees the same columns and rows.

Load: the forecast for day D is made at 10:00 on D-1, when the TSOs publish
theirs (two hours before the 12:00 day-ahead auction closes). ENTSO-E publishes
actual load up to an hour after the hour ends, so the newest hour available is
the one starting 08:00 on D-1. issue="midnight" is the original setup (every
lag at least 24 hours), kept for comparison.

Weather: weather_mode sets how much temperature the model gets.

  "none"     no weather
  "lagged"   only temperature already measured at the issue time
  "noisy"    target-hour temperature plus synthetic error, as a stand-in
             for a forecast
  "perfect"  target-hour temperature exactly, as an upper bound

Calendar features use Europe/Berlin time; the index stays UTC so DST changes
don't create duplicate or missing hours.
"""

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

TZ = "Europe/Berlin"
ISSUES = ("10am", "midnight")

# 10:00 issue: same-hour lags that are always published by 10:00 on D-1, plus
# a snapshot of the latest data (see last_known)
ISSUE_LAGS = [48, 72, 168, 336]
ISSUE_HOUR = 10
LAST_KNOWN_HOUR = ISSUE_HOUR - 2   # the 08:00-09:00 value is out by 10:00

# midnight issue: the original lags
LAGS = [24, 48, 72, 168, 336]

# Standard deviation of the noise in weather_mode="noisy", degrees C. Roughly
# the size of a day-ahead 2m temperature error, not a measured value. The noise
# is independent hour to hour, which real forecast error isn't.
SYNTHETIC_TEMP_ERROR_C = 1.0

# German national public holidays 2015-2020, as local dates
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

# Population by state in millions (2020). Regional holidays (Corpus Christi,
# All Saints' Day, Epiphany...) are weighted by how many people have the day
# off, rather than counted as a yes/no.
STATE_POPULATION = {
    "BW": 11.10, "BY": 13.14, "BE": 3.66, "BB": 2.53, "HB": 0.68, "HH": 1.85,
    "HE": 6.29, "MV": 1.61, "NI": 8.00, "NW": 17.93, "RP": 4.10, "SL": 0.98,
    "SN": 4.06, "ST": 2.18, "SH": 2.91, "TH": 2.12,
}

WEATHER_MODES = ("none", "lagged", "noisy", "perfect")

# Base temperatures for the heating/cooling terms. Common European values;
# not tuned.
HDD_BASE = 15.0
CDD_BASE = 22.0


def degree_hours(temp_c):
    """Heating and cooling degree hours: how far each hour is below HDD_BASE or
    above CDD_BASE.

    Demand against temperature is V-shaped, which a linear model like ridge
    can't fit from raw temperature. Two one-sided terms give it the two arms.
    """
    return np.maximum(0.0, HDD_BASE - temp_c), np.maximum(0.0, temp_c - CDD_BASE)


def _cyclical(values, period):
    # sin/cos so that 23:00 and 00:00 (or December and January) end up close
    r = 2 * np.pi * values / period
    return np.sin(r), np.cos(r)


def last_known(index):
    """For each target hour, the start of the newest load hour that is out by
    10:00 local on the day before. All 24 hours of day D share one value."""
    day = index.tz_convert(TZ).normalize().tz_localize(None)
    lk = day - pd.Timedelta(days=1) + pd.Timedelta(hours=LAST_KNOWN_HOUR)
    return lk.tz_localize(TZ).tz_convert("UTC")


def newest_usable(index, issue="10am"):
    """The newest load timestamp each row may depend on. Used by the leakage
    check in selfcheck.py."""
    if issue == "midnight":
        return index - pd.Timedelta(hours=24)
    return last_known(index)


@lru_cache(maxsize=None)
def _state_calendars(years):
    return {s: holidays.Germany(subdiv=s, years=years) for s in STATE_POPULATION}


def holiday_share(dates):
    """Share of Germany's population on a public holiday, per local date:
    1.0 on national holidays, 0.64 on Corpus Christi, 0.32 on Epiphany.
    `dates` is a naive DatetimeIndex of local dates."""
    cal = _state_calendars(tuple(sorted(set(dates.year))))
    total = sum(STATE_POPULATION.values())
    days = dates.unique()
    share = {d: sum(p for s, p in STATE_POPULATION.items() if d in cal[s]) / total
             for d in days}
    return pd.Series(share).reindex(dates).to_numpy()


def usable_temperature(temp, weather_mode, seed=0, issue="10am"):
    """The temperature the model may use for the target hour."""
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
    # "noisy": seeded so a run reproduces; compare several seeds before trusting it
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, SYNTHETIC_TEMP_ERROR_C, len(temp))
    return temp + noise


def build_features(load, temp=None, weather_mode="none", seed=0, issue="10am",
                   regional_holidays=True):
    if issue not in ISSUES:
        raise ValueError(f"issue must be one of {ISSUES}")

    df = pd.DataFrame({"load_mw": load})
    idx = df.index
    loc = idx.tz_convert(TZ)

    df["hour"] = loc.hour
    df["dayofweek"] = loc.dayofweek
    df["month"] = loc.month
    df["is_weekend"] = (loc.dayofweek >= 5).astype(int)
    df["is_holiday"] = loc.strftime("%Y-%m-%d").isin(HOLIDAYS).astype(int)

    if regional_holidays:
        day = loc.normalize().tz_localize(None)
        df["holiday_share"] = holiday_share(day)
        # the lags land on these days, so a holiday there pulls the lag down
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
        past = df["load_mw"].shift(24)
        df["roll_mean_24h"] = past.rolling(24).mean()
        df["roll_mean_168h"] = past.rolling(168).mean()
        df["roll_std_24h"] = past.rolling(24).std()
    else:
        for lag in ISSUE_LAGS:
            df[f"lag_{lag}h"] = df["load_mw"].shift(lag)
        # one snapshot of what is known at 10:00 on D-1, shared by all 24 hours
        # of D, the way a real forecast run works
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
            # t.shift(24) would reach past 10:00 for most hours, so use the
            # last measured hour instead
            measured = temp.reindex(idx)
            lk = last_known(idx)
            df["temp_roll_mean_24h"] = measured.rolling(24).mean().reindex(lk).to_numpy()
            df["temp_last_known"] = measured.reindex(lk).to_numpy()
        else:
            # buildings respond to the last day of weather, not just this hour
            df["temp_roll_mean_24h"] = t.rolling(24).mean()
            df["temp_change_24h"] = t - t.shift(24)

    df = df.dropna()
    y = df.pop("load_mw")
    return df, y


def _as_utc(t):
    """Accept "2019-01-01" or a tz-aware Timestamp (the backtest passes those)."""
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def chronological_split(X, y, val_start, test_start):
    vs, ts = _as_utc(val_start), _as_utc(test_start)
    tr, va, te = X.index < vs, (X.index >= vs) & (X.index < ts), X.index >= ts
    if not (tr.any() and va.any() and te.any()):
        raise ValueError("a split came out empty - check your dates vs the data range")
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])
