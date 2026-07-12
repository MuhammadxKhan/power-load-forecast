"""
Getting data in. Demand from OPSD, temperature from ERA5. No features here.

Two columns come out of the OPSD file:

  load_mw               what demand actually was (the target)
  benchmark_mw          a published day-ahead load forecast, derived by OPSD
                        from ENTSO-E Transparency data. Careful how this gets
                        described: it's OPSD's aggregation, not a raw untouched
                        TSO series, and the file keeps target timestamps but no
                        forecast vintage - so there's no telling whether a value
                        is the first issuance or a later revision. Under
                        Regulation 543/2013 the first publication is due at
                        least two hours before day-ahead gate closure (12:00 for
                        Germany, so about 10:00 on D-1). Either way its
                        information cutoff is earlier than this model's assumed
                        midnight, so beating it is not a like-for-like win.

Temperature is ERA5, ECMWF's reanalysis - their best after-the-fact estimate of
what the weather was. Native atmospheric resolution about 31 km, served on a
0.25 degree grid, hourly, as NetCDF. xarray reads it.

Only Germany. An earlier version had a --country flag that applied German
holidays to whatever you asked for, and OPSD's British column isn't even called
what that flag assumed (GB_GBN_..., not GB_...), so it never worked. Removed
rather than half-fixed: adding a country needs a column name, a timezone AND a
holiday list, all three.
"""

import glob
import os

import numpy as np
import pandas as pd

# Pinned to a dated package, not /latest/. OPSD republishes, and a moving URL
# means the README's numbers stop reproducing without anyone noticing.
OPSD_VERSION = "2020-10-06"
OPSD_URL = (f"https://data.open-power-system-data.org/time_series/{OPSD_VERSION}/"
            "time_series_60min_singleindex.csv")
OPSD_CACHE = "opsd_60min.csv"        # the full 94MB file, gitignored
OPSD_EXTRACT = "data/de_hourly.csv"  # three columns of it, ~2MB, committed

BBOX = {"north": 55.0, "south": 47.0, "west": 5.5, "east": 15.5}

ACTUAL_COL = "DE_load_actual_entsoe_transparency"
BENCH_COL = "DE_load_forecast_entsoe_transparency"


# --------------------------------------------------------------------------
# demand
# --------------------------------------------------------------------------
def load_frame():
    """Actual German load and the published benchmark, hourly, indexed by UTC."""
    # data/de_hourly.csv is three columns pulled out of the full OPSD file and
    # committed, so cloning the repo is enough to run this - no 94MB download,
    # no account, nothing. The full-file path below is what generated it and is
    # kept so the extract is reproducible rather than a magic artefact.
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
        print(f"Wrote {OPSD_EXTRACT} - commit this and nobody needs the download")

    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")

    s = df["load_mw"]
    df = df.loc[s.first_valid_index():s.last_valid_index()]

    # every hour must exist, or "168 rows back" stops meaning "168 hours back"
    # and every lag silently misaligns
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="h", tz="UTC"))

    missing = int(df["load_mw"].isna().sum())
    if missing:
        print(f"{missing} missing load hours ({missing / len(df):.3%}) - "
              "filling from earlier values")
    df["load_mw"] = _fill_gaps(df["load_mw"])

    # The benchmark is deliberately NOT filled. Filling it would invent forecast
    # values nobody ever published and then score models against them. Missing
    # stays missing; evaluate.py drops the benchmark if the scored window has
    # holes, and says so.
    gaps = int(df["benchmark_mw"].isna().sum())
    if gaps:
        print(f"{gaps} missing benchmark hours ({gaps / len(df):.3%}) - left as NaN")

    df.index.name = "timestamp"
    return df


def _fill_gaps(s):
    """Fill interior gaps from earlier values only. A LEADING gap is the one
    exception and gets filled backwards, because there is nothing earlier.

    The obvious thing is interpolate(), and that's what I had. But linear
    interpolation fills a hole from the valid values on BOTH sides, weighted by
    position, so the filled number carries information from the future - and a
    lag_24h feature a day later would then be built on data only 18 hours old,
    breaking the 24-hour rule everything here rests on.

    On the pinned OPSD package the German load series has NO missing hours once
    it's trimmed to its first and last valid reading, so on this data it changes
    nothing at all. It's here because the code shouldn't depend on the data
    happening to be clean.
    """
    return s.ffill().bfill()

