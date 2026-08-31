"""
Fetch ERA5 2m temperature over Germany, one NetCDF per year, into era5_raw/.

Setup, once:

  1. Free account at https://cds.climate.copernicus.eu
  2. Put your token from https://cds.climate.copernicus.eu/profile into
     ~/.cdsapirc (C:\\Users\\<you>\\.cdsapirc on Windows, no extension):

         url: https://cds.climate.copernicus.eu/api
         key: <your-token>

  3. Open the ERA5 hourly single-levels page once and accept the licence, or
     every request returns 403.
  4. pip install cdsapi

Then `python -m src.download_era5`. Requests queue on ECMWF's side and can sit
a while. Each year is 100-200 MB and data/era5_raw/ is gitignored; the first run
of run_comparison.py reduces it to data/era5_temp_de.csv, which is committed so
nobody else needs an account.
"""

import os

from .data import BBOX

YEARS = [str(y) for y in range(2015, 2021)]
OUTDIR = "data/era5_raw"


def main():
    import cdsapi

    os.makedirs(OUTDIR, exist_ok=True)
    client = cdsapi.Client()

    for year in YEARS:
        target = os.path.join(OUTDIR, f"era5_t2m_{year}.nc")
        if os.path.exists(target):
            print(f"{target} already here, skipping")
            continue

        print(f"requesting {year} (this queues - be patient)")
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": ["2m_temperature"],
                "year": [year],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                # north/west/south/east, the German bounding box
                "area": [BBOX["north"], BBOX["west"], BBOX["south"], BBOX["east"]],
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            target,
        )
        print(f"  -> {target}")

    print("\nDone. Now run: python run_comparison.py --weather noisy")


if __name__ == "__main__":
    main()
