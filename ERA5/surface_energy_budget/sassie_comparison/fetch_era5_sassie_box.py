"""Cache ERA5 surface fluxes over the SASSIE 2022 ship-track box.

The Beaufort Sea box used here (68.5-74.0 N, 167.0-144.0 W) is a superset of the
R/V *Woldstad* track during the SASSIE fall 2022 campaign (69.20-73.52 N,
165.96-144.89 W), so every 20 min shipboard record has ERA5 grid cells on all
sides of it.

Data come from the NSF NCAR public ERA5 mirror on S3 (anonymous access) via
``era5_aws.load_region``, because the CDS ``barrow`` archive already on disk
(70-80 N, 165-150 W) misses ~42% of the ship track.

Run once; the notebook reads the netCDF it writes.

    python fetch_era5_sassie_box.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import era5_aws  # noqa: E402

# --- SASSIE ship-track bounding box, padded to whole ERA5 grid lines ---------
NORTH_DEG, WEST_DEG, SOUTH_DEG, EAST_DEG = 74.0, -167.0, 68.5, -144.0

# Shipboard MET record runs 2022-09-09T01:08Z to 2022-10-01T23:48Z. One day of
# padding on each side covers the hour-ending accumulation windows at the edges.
START_DATE, END_DATE = "2022-09-08", "2022-10-02"

# Six flux fields (all W m-2, ERA5 positive DOWNWARD) plus two surface-state
# fields used to interpret the residuals.
VARIABLES = [
    "msdwswrf",  # downwelling shortwave
    "msnswrf",   # net shortwave  (SWD - SWU)
    "msdwlwrf",  # downwelling longwave
    "msnlwrf",   # net longwave   (LWD - LWU)
    "msshf",     # sensible heat  (into surface)
    "mslhf",     # latent heat    (into surface)
    "siconc",    # sea-ice cover, 0-1
    "sst",       # sea surface temperature, K
]

OUT_PATH = HERE / "data" / "era5_sassie_box_202209_202210.nc"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds = era5_aws.load_region(
        NORTH_DEG, WEST_DEG, SOUTH_DEG, EAST_DEG,
        START_DATE, END_DATE, variables=VARIABLES, verbose=True,
    )
    ds.attrs["title"] = "ERA5 surface fluxes over the SASSIE 2022 ship-track box"
    ds.attrs["box_north_south_west_east_deg"] = [
        NORTH_DEG, SOUTH_DEG, WEST_DEG, EAST_DEG
    ]
    ds.attrs["date_range"] = f"{START_DATE}..{END_DATE}"
    ds.to_netcdf(OUT_PATH)
    print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"elapsed {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
