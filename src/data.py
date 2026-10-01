"""
Loads the data: German demand from OPSD, temperature from ERA5.

Two columns come out of the OPSD file:

  load_mw       actual demand (the target)
  benchmark_mw  the TSOs' published day-ahead load forecast, as aggregated by
                OPSD from ENTSO-E. It has target timestamps but no publication
                time, so some values may be later revisions. It is due about
                two hours before the 12:00 auction, i.e. ~10:00 on D-1.

Temperature is ERA5 reanalysis: ECMWF's after-the-fact estimate of what the
weather was, on a 0.25 degree grid, hourly. It stands in for a forecast here; it
is not one.
"""

import glob
import os

import numpy as np
import pandas as pd

# A dated release rather than /latest/, so the numbers keep reproducing if OPSD
# republishes.
OPSD_VERSION = "2020-10-06"
OPSD_URL = (f"https://data.open-power-system-data.org/time_series/{OPSD_VERSION}/"
            "time_series_60min_singleindex.csv")
OPSD_CACHE = "opsd_60min.csv"        # full 94MB file, gitignored
OPSD_EXTRACT = "data/de_hourly.csv"  # the three columns used, ~2MB, committed

ERA5_CACHE = "data/era5_temp_de.csv"   # national hourly series, committed
ERA5_GLOB = "data/era5_raw/*.nc"       # raw download, ~1GB, gitignored

# rough box around Germany (it includes sea and neighbouring countries)
BBOX = {"north": 55.0, "south": 47.0, "west": 5.5, "east": 15.5}

ACTUAL_COL = "DE_load_actual_entsoe_transparency"
BENCH_COL = "DE_load_forecast_entsoe_transparency"


# --------------------------------------------------------------------------
# demand
# --------------------------------------------------------------------------
def load_frame():
    """Actual German load and the published benchmark, hourly, indexed by UTC."""
    # The committed extract is enough to run everything; the full download is
    # only needed to rebuild it.
    if os.path.exists(OPSD_EXTRACT):
        df = pd.read_csv(OPSD_EXTRACT, parse_dates=["utc_timestamp"])
        df = df.set_index("utc_timestamp")
    else:
        if not os.path.exists(OPSD_CACHE):
            print(f"Downloading OPSD {OPSD_VERSION} (~94MB, one-off)...")
            pd.read_csv(OPSD_URL, low_memory=False).to_csv(OPSD_CACHE, index=False)
            print(f"Cached to {OPSD_CACHE}")
        df = pd.read_csv(OPSD_CACHE, usecols=["utc_timestamp", ACTUAL_COL, BENCH_COL],
                         parse_dates=["utc_timestamp"]).set_index("utc_timestamp")
        df = df.rename(columns={ACTUAL_COL: "load_mw", BENCH_COL: "benchmark_mw"})
        os.makedirs(os.path.dirname(OPSD_EXTRACT), exist_ok=True)
        df.to_csv(OPSD_EXTRACT)
        print(f"Wrote {OPSD_EXTRACT}")

    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")

    s = df["load_mw"]
    df = df.loc[s.first_valid_index():s.last_valid_index()]

    # every hour has to exist, otherwise shift(168) is not "168 hours back"
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="h", tz="UTC"))

    missing = int(df["load_mw"].isna().sum())
    if missing:
        print(f"{missing} missing load hours ({missing / len(df):.3%}) - "
              "filling from earlier values")
    df["load_mw"] = _fill_gaps(df["load_mw"])

    # The benchmark is not filled: that would invent forecasts nobody published.
    # evaluate.py drops it if the scored window has gaps.
    gaps = int(df["benchmark_mw"].isna().sum())
    if gaps:
        print(f"{gaps} missing benchmark hours ({gaps / len(df):.3%}) - left as NaN")

    df.index.name = "timestamp"
    return df


def _fill_gaps(s):
    """Forward-fill gaps (backward-fill only a leading gap).

    Forward fill rather than interpolate(), because interpolation uses the value
    after the gap, which a lag feature would then be reading from the future.
    The pinned OPSD release has no gaps once trimmed, so this is a safeguard.
    """
    return s.ffill().bfill()


# --------------------------------------------------------------------------
# weather
# --------------------------------------------------------------------------
def load_temperature(index=None):
    """National hourly 2m temperature in Celsius, indexed by UTC.

    Reads the committed CSV if present, otherwise builds it from the NetCDF
    files in era5_raw/. It is an unweighted mean over BBOX, so the North Sea
    counts as much as Berlin; population weighting would be better.
    """
    if os.path.exists(ERA5_CACHE):
        s = pd.read_csv(ERA5_CACHE, parse_dates=["timestamp"]).set_index("timestamp")["temp_c"]
        s.index = pd.DatetimeIndex(s.index).tz_convert("UTC")
    else:
        files = sorted(glob.glob(ERA5_GLOB))
        if not files:
            raise FileNotFoundError(
                f"no {ERA5_CACHE} and nothing matching {ERA5_GLOB}.\n"
                "Run  python -m src.download_era5  first (free Copernicus account "
                "needed), or run without --weather.")
        s = _from_netcdf(files)
        s.rename_axis("timestamp").rename("temp_c").to_csv(ERA5_CACHE)
        print(f"Wrote {ERA5_CACHE} ({len(s):,} hours)")

    s.name = "temp_c"
    if index is not None:
        s = s.reindex(index)
        gaps = int(s.isna().sum())
        if gaps:
            print(f"{gaps} hours have no temperature - filling from earlier values")
            s = _fill_gaps(s)
    return s


def _from_netcdf(files):
    """Average the ERA5 grid down to one number per hour.

    Opens the yearly files one at a time rather than with open_mfdataset, which
    needs dask. Each file is reduced to a 1-D series straight away, so memory
    stays small.
    """
    import xarray as xr

    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            if "t2m" not in ds:
                raise KeyError(f"{f}: no 't2m' variable, found {list(ds.data_vars)}")
            # CDS has used both 'time' and 'valid_time' as the time dimension
            tname = "valid_time" if "valid_time" in ds["t2m"].dims else "time"
            space = [d for d in ds["t2m"].dims if d != tname]
            parts.append(ds["t2m"].mean(dim=space).to_series())

    s = pd.concat(parts).sort_index()
    dupes = int(s.index.duplicated().sum())
    if dupes:
        # yearly files can overlap at the boundary
        print(f"{dupes} duplicate ERA5 timestamps, keeping the first of each")
        s = s[~s.index.duplicated(keep="first")]

    s = s - 273.15                            # Kelvin to Celsius
    s.index = pd.DatetimeIndex(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")  # ERA5 timestamps are UTC
    return s.sort_index()


# --------------------------------------------------------------------------
# synthetic data for selfcheck.py
# --------------------------------------------------------------------------
def fake_frame(n_days=1500, seed=0):
    """Synthetic load with daily, weekly and yearly cycles, so the checks run
    without the download. The numbers mean nothing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n_days * 24, freq="h", tz="UTC")
    loc = idx.tz_convert("Europe/Berlin")
    hour, dow, doy = loc.hour.to_numpy(), loc.dayofweek.to_numpy(), loc.dayofyear.to_numpy()

    daily = 8000 * np.sin((hour - 3) / 24 * 2 * np.pi) + 3000 * np.sin(hour / 12 * 2 * np.pi)
    weekly = np.where(dow >= 5, -6000, 0)
    yearly = 5000 * np.cos((doy - 15) / 365 * 2 * np.pi)
    load = 50000 + daily + weekly + yearly + rng.normal(0, 900, len(idx))

    # a noisy fake benchmark, so the comparison code has something to score
    return pd.DataFrame({"load_mw": load,
                         "benchmark_mw": load + rng.normal(0, 1800, len(idx))},
                        index=idx).rename_axis("timestamp")


def fake_temperature(index, seed=0):
    """Synthetic temperature: seasonal and daily cycles plus a slow random
    wander, so consecutive days are correlated like real weather."""
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(index)
    loc = idx.tz_convert("Europe/Berlin")

    seasonal = 9.5 - 9.0 * np.cos((loc.dayofyear.to_numpy() - 20) / 365 * 2 * np.pi)
    diurnal = 3.5 * np.sin((loc.hour.to_numpy() - 9) / 24 * 2 * np.pi)
    wander = (pd.Series(rng.normal(0, 1.0, len(idx)))
              .rolling(72, min_periods=1).mean() * 6.0).to_numpy()

    return pd.Series(seasonal + diurnal + wander, index=idx,
                     name="temp_c").rename_axis("timestamp")
