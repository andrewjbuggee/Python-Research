#!/usr/bin/env python3
"""Generate a fake ODB-shaped CSV so the analysis path can be exercised offline.

The MARS half of this package cannot run without an ECMWF account, which makes
the analysis half easy to leave untested until the day the real data arrives.
This script fabricates a file with the same column names, suffixes and bitfield
member layout that ``odb sql`` produces, so ``read_era5_obs_feedback.py --csv``
can be run end to end on any laptop.

The numbers are invented. Nothing produced here is data about Barrow, and the
verdict printed from it means nothing physically -- it only proves the plumbing
works. Seeded so runs are reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 20260828  # recorded in the output header line


def synthesise(n_reports: int = 400, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Two stations in the box: the airport (SYNOP + radiosonde) and a second
    # identifier with SYNOP only, to mimic the 70026/70027 ambiguity.
    specs = [
        # statid, lat, lon, varno, obstype, active_fraction, obs_error
        ("70026", 71.28, -156.79, 110, 1, 0.95, 100.0),   # ps, assimilated
        ("70026", 71.28, -156.79, 39, 1, 0.00, 2.0),      # t2m, screen level
        ("70026", 71.28, -156.79, 58, 1, 0.00, 0.05),     # rh2m, screen level
        ("70026", 71.28, -156.79, 2, 5, 0.90, 1.0),       # radiosonde T
        ("70026", 71.28, -156.79, 3, 5, 0.88, 2.0),       # radiosonde u
        ("70027", 71.32, -156.61, 110, 1, 0.60, 120.0),   # ps, partly rejected
    ]

    rows = []
    for statid, lat, lon, varno, obstype, active_fraction, obs_error in specs:
        n = n_reports
        is_active = rng.random(n) < active_fraction
        # Rejected/passive split for the non-active remainder.
        is_passive = (~is_active) & (rng.random(n) < 0.7)
        is_rejected = (~is_active) & (~is_passive)

        fg_depar = rng.normal(0.0, obs_error, n)
        # Active data get pulled toward the observation: |o-a| < |o-b|.
        shrink = np.where(is_active, 0.45, 1.0)
        an_depar = fg_depar * shrink + rng.normal(0.0, 0.1 * obs_error, n)

        dates = pd.date_range("2015-01-01", periods=n, freq="6h")
        rows.append(
            pd.DataFrame(
                {
                    "statid@hdr": statid,
                    "lat@hdr": lat,
                    "lon@hdr": lon,
                    "stalt@hdr": 12.0,
                    "date@hdr": dates.strftime("%Y%m%d").astype(int),
                    "time@hdr": dates.strftime("%H%M%S").astype(int),
                    "obstype@hdr": obstype,
                    "reportype@hdr": 16001 if obstype == 1 else 16045,
                    "varno@body": varno,
                    "obsvalue@body": rng.normal(0.0, 1.0, n),
                    "fg_depar@body": fg_depar,
                    "an_depar@body": an_depar,
                    "obs_error@errstat": obs_error,
                    "biascorr@body": 0.0,
                    "datum_status.active@body": is_active.astype(int),
                    "datum_status.passive@body": is_passive.astype(int),
                    "datum_status.rejected@body": is_rejected.astype(int),
                    "datum_status.blacklisted@body": 0,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/obs_feedback_synthetic")
    parser.add_argument("--n-reports", type=int, default=400)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "SYNTHETIC_barrow_ofb_201501.csv"
    synthesise(args.n_reports, args.seed).to_csv(out_path, index=False)
    print(f"Wrote {out_path}  (SYNTHETIC, seed={args.seed} -- not real data)")
    print(f"Test the analysis with:\n"
          f"  python read_era5_obs_feedback.py --csv --in-dir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
