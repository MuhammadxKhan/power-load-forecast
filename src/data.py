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

