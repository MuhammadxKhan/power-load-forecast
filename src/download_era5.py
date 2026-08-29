"""
Fetch ERA5 2m temperature over Germany, one NetCDF per year, into era5_raw/.

Setup, once:

  1. Make a free account at https://cds.climate.copernicus.eu
  2. Copy your personal access token from https://cds.climate.copernicus.eu/profile
  3. Put it in a file called .cdsapirc in your home directory:

         url: https://cds.climate.copernicus.eu/api
         key: <your-token>

     On Windows that's C:\\Users\\<you>\\.cdsapirc - no extension, and Notepad
     will try to add .txt, so save it as "All files".
  4. Open the ERA5 hourly single-levels dataset page once and accept the
     licence, or every request comes back 403.
  5. pip install cdsapi

Then:

  python -m src.download_era5

Requests queue on ECMWF's side and can sit there a while, so start this before
you need it. Each year is roughly 100-200 MB; data/era5_raw/ is gitignored.

You only need this once. It writes NetCDF into data/era5_raw/, and the first run of
run_comparison.py boils that down to data/era5_temp_de.csv - a small national series.
Commit that file once you have generated it, so anyone cloning the repo can
reproduce the weather results without a Copernicus account. It is not in the
repo until you run this.
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
