#!/usr/bin/env python3
"""Read ERA5 observation-feedback ODB files and turn flags into a verdict.

The retrieval step gives you rows. This step answers the actual question, which
is not "did a report from Barrow reach ECMWF" but "did it change the analysis".
Those differ, and the ODB is the only place the difference is recorded.

THE FOUR OUTCOMES, AND WHY ONLY ONE OF THEM COUNTS
==================================================
Every datum carries a ``datum_status`` bitfield:

  active       Used in the 4D-Var minimisation. The analysis is different
               because this observation existed. THIS is assimilation.
  passive      Ingested, screened, compared to the model, departures computed
               and archived -- but given zero weight. Monitoring only. A
               passive datum is present in every diagnostic you might plot and
               influenced nothing.
  rejected     Failed a quality-control test (background departure too large,
               duplicate, gross error).
  blacklisted  Excluded a priori by station/variable/period, independent of
               this report's quality.

A station can be simultaneously active in one variable and passive in another,
which is the normal case at a land site and the single most misread result
here. See the note on screen-level variables below.

READ THE SURFACE VARIABLES WITH CARE
====================================
For a land station in ERA5:

  * ps (varno 110) is assimilated in 4D-Var. Expect active.
  * radiosonde t/u/v/q (varno 2/3/4/7) are assimilated. Expect active.
  * 2 m T and 2 m RH (varno 39/58) are NOT part of the 4D-Var atmospheric
    analysis in ERA5. They feed a separate 2D optimal-interpolation screen-level
    analysis and the soil-moisture simplified EKF (Hersbach et al. 2020, QJRMS
    146:1999-2049, section 3). So a non-active flag for varno 39/58 in this
    feedback does NOT mean the observation was unused by the system -- it means
    it was not used by the atmospheric analysis whose feedback this is. The
    summary flags these varnos explicitly rather than letting the table imply
    otherwise.
  * 10 m wind over land is generally not assimilated.

And the point that no ODB query can answer, because it is true a priori: ERA5
assimilates no surface radiation, no cloud radar, and no microwave-radiometer
LWP from ARM/NSA or NOAA-BRW. There is no observation operator for downwelling
longwave irradiance in the IFS. Barrow's radiative fluxes in ERA5 are model
output constrained only indirectly, through the thermodynamic profile.

DEPARTURE SIGN CONVENTION
=========================
``fg_depar`` = observation minus first guess (o - b).
``an_depar``  = observation minus analysis    (o - a).
For an actively assimilated datum, |an_depar| < |fg_depar| on average: that
shrinkage IS the analysis pulling toward the observation, and it is the
quantitative signature of assimilation rather than mere presence. The summary
reports the ratio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from era5_odb_config import (
    DATUM_STATUS_BITS,
    STATION_IDS_OF_INTEREST,
    SURFACE_VARNOS,
    USAGE_CATEGORIES,
    obstype_label,
    reportype_label,
    varno_label,
)

SCREEN_LEVEL_VARNOS = (39, 58, 40)  # see "READ THE SURFACE VARIABLES WITH CARE"


def _strip_odb_suffix(column: str) -> str:
    """``fg_depar@body`` -> ``fg_depar``. ODB table suffixes carry no meaning here."""
    return column.split("@", 1)[0]


def read_odb_file(path: Path) -> pd.DataFrame:
    """Read one ODB-2 file into a DataFrame with suffix-free column names.

    Prefers ``codc`` (C-backed, fast) and falls back to ``pyodc`` (pure Python).
    Both are ECMWF packages; neither is on conda-forge by default.
    """
    reader = None
    try:
        import codc as reader  # type: ignore
    except ImportError:
        try:
            import pyodc as reader  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "Neither codc nor pyodc is installed; cannot read ODB files.\n"
                "  pip install pyodc          (pure Python, no ODB C library needed)\n"
                "Alternatively convert to CSV on an ECMWF platform with\n"
                "  odb sql 'select *' -i file.odb -o file.csv --no_alignment\n"
                "and pass the CSV to this script with --csv."
            ) from exc

    frames = reader.read_odb(str(path), single=False)
    if isinstance(frames, pd.DataFrame):
        frames = [frames]
    if not frames:
        return pd.DataFrame()

    table = pd.concat(frames, ignore_index=True, sort=False)
    table.columns = [_strip_odb_suffix(c) for c in table.columns]
    return table


def read_csv_file(path: Path) -> pd.DataFrame:
    """Read an ``odb sql``-produced CSV, for the no-pyodc path."""
    table = pd.read_csv(path)
    table.columns = [_strip_odb_suffix(str(c)).strip() for c in table.columns]
    return table


def normalise_statid(table: pd.DataFrame) -> pd.DataFrame:
    """Clean ``statid`` into a plain stripped string column.

    ODB station identifiers arrive as fixed-width character data, so they may be
    bytes, and they may carry leading or trailing blanks. Comparing those
    directly is how a real station turns into a spurious "not found".
    """
    if "statid" not in table.columns:
        table["statid"] = "unknown"
        return table

    def _clean(value: object) -> str:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return str(value).strip().strip("'\"")

    table["statid"] = table["statid"].map(_clean)
    return table


def ensure_usage_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Guarantee boolean ``datum_active`` / ``datum_passive`` / ... columns.

    Two shapes can arrive:
      * the SQL-expanded shape, with ``datum_status.active`` as 0/1 columns --
        preferred, and used as-is; or
      * the packed shape, with a single integer ``datum_status`` -- decoded here
        using the bit offsets in era5_odb_config, which are the one part of this
        package worth double-checking against ``odb header`` on your own file.
    """
    expanded = [f"datum_status.{name}" for name in USAGE_CATEGORIES]
    if all(column in table.columns for column in expanded):
        for name in USAGE_CATEGORIES:
            table[f"datum_{name}"] = table[f"datum_status.{name}"].fillna(0).astype(int).astype(bool)
        table.attrs["status_source"] = "sql_expanded_members"
        return table

    if "datum_status" in table.columns:
        packed = table["datum_status"].fillna(0).astype("int64")
        for name, offset in DATUM_STATUS_BITS.items():
            table[f"datum_{name}"] = ((packed >> offset) & 1).astype(bool)
        table.attrs["status_source"] = "decoded_packed_bitfield"
        return table

    raise SystemExit(
        "No datum_status column found. The retrieval did not include the usage\n"
        "flags, so the file cannot answer whether anything was assimilated.\n"
        "Re-run the fetch step without a custom --filter."
    )


def load_feedback(paths: Sequence[Path], is_csv: bool = False) -> pd.DataFrame:
    """Load and concatenate every retrieved feedback file."""
    tables: List[pd.DataFrame] = []
    for path in paths:
        table = read_csv_file(path) if is_csv else read_odb_file(path)
        if table.empty:
            print(f"  {path.name}: 0 rows")
            continue
        table["source_file"] = path.name
        print(f"  {path.name}: {len(table):,} rows")
        tables.append(table)

    if not tables:
        return pd.DataFrame()

    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined = normalise_statid(combined)
    combined = ensure_usage_columns(combined)
    return combined


def station_census(table: pd.DataFrame) -> pd.DataFrame:
    """Every station identifier found in the box, with position and report counts.

    This is the answer to "70026 or 70027?" -- read it off the archive instead of
    assuming. Run it first; the per-variable summary is only meaningful once you
    know which identifiers actually exist here.
    """
    grouped = table.groupby("statid", dropna=False)
    census = pd.DataFrame(
        {
            "n_rows": grouped.size(),
            "lat_deg": grouped["lat"].median() if "lat" in table.columns else np.nan,
            "lon_deg": grouped["lon"].median() if "lon" in table.columns else np.nan,
            "alt_m": grouped["stalt"].median() if "stalt" in table.columns else np.nan,
            "n_active": grouped["datum_active"].sum(),
            "first_date": grouped["date"].min() if "date" in table.columns else np.nan,
            "last_date": grouped["date"].max() if "date" in table.columns else np.nan,
        }
    )
    census["frac_active"] = census["n_active"] / census["n_rows"]
    return census.sort_values("n_rows", ascending=False)


def position_history(table: pd.DataFrame, tol_deg: float = 0.02) -> pd.DataFrame:
    """Detect station relocations: one statid reporting from more than one place.

    This is not hypothetical at Utqiagvik. NWS Service Change Notice 18-89
    moved the WMO 70026 radiosonde release point on 12 February 2019, from the
    legacy site to 71.32267 N, 156.61784 W -- 4.6 miles northeast, which is the
    DOE ARM North Slope of Alaska C1 site. So a multi-year query on 70026 spans
    two physically different launch points, and the median position reported by
    ``station_census`` is a blend of both that corresponds to neither.

    Any statistic computed across the move -- departure bias especially -- mixes
    two records. Split on the move date rather than averaging through it.
    """
    if not {"lat", "lon"}.issubset(table.columns):
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for statid, group in table.groupby("statid", dropna=False):
        # Cluster positions onto a coarse grid; tol_deg sets what counts as
        # "the same place" against ordinary reporting jitter.
        lat_key = (group["lat"] / tol_deg).round() * tol_deg
        lon_key = (group["lon"] / tol_deg).round() * tol_deg
        for (lat_c, lon_c), cluster in group.groupby([lat_key, lon_key]):
            rows.append(
                {
                    "statid": statid,
                    "lat_deg": float(cluster["lat"].median()),
                    "lon_deg": float(cluster["lon"].median()),
                    "alt_m": float(cluster["stalt"].median())
                    if "stalt" in cluster.columns else np.nan,
                    "n_rows": len(cluster),
                    "first_date": cluster["date"].min() if "date" in cluster.columns else np.nan,
                    "last_date": cluster["date"].max() if "date" in cluster.columns else np.nan,
                }
            )

    history = pd.DataFrame(rows)
    if history.empty:
        return history
    return history.sort_values(["statid", "first_date"])


def usage_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per station and variable: how many data were used, monitored, rejected.

    The departure columns are the corroborating evidence. Counting active rows
    tells you the flag was set; ``depar_shrinkage`` tells you the analysis
    actually moved toward the observation, which is harder to fake.
    """
    rows: List[Dict[str, object]] = []
    group_keys = ["statid", "varno"]
    if "obstype" in table.columns:
        group_keys.insert(1, "obstype")

    for keys, group in table.groupby(group_keys, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record: Dict[str, object] = dict(zip(group_keys, keys))
        varno = int(record["varno"])
        record["variable"] = varno_label(varno)
        if "obstype" in record:
            record["obstype_name"] = obstype_label(int(record["obstype"]))

        record["n_obs"] = len(group)
        for name in USAGE_CATEGORIES:
            record[f"n_{name}"] = int(group[f"datum_{name}"].sum())
        record["frac_active"] = record["n_active"] / record["n_obs"]

        active = group[group["datum_active"]]
        for label, source in (("all", group), ("active", active)):
            for depar in ("fg_depar", "an_depar"):
                if depar in source.columns and len(source):
                    values = pd.to_numeric(source[depar], errors="coerce").dropna()
                    record[f"rms_{depar}_{label}"] = (
                        float(np.sqrt(np.mean(values**2))) if len(values) else np.nan
                    )
                    record[f"bias_{depar}_{label}"] = (
                        float(values.mean()) if len(values) else np.nan
                    )
                else:
                    record[f"rms_{depar}_{label}"] = np.nan
                    record[f"bias_{depar}_{label}"] = np.nan

        rms_fg = record.get("rms_fg_depar_active", np.nan)
        rms_an = record.get("rms_an_depar_active", np.nan)
        # < 1 means the analysis was pulled toward the observation.
        record["depar_shrinkage"] = (
            rms_an / rms_fg if (rms_fg and np.isfinite(rms_fg) and rms_fg > 0) else np.nan
        )

        if "biascorr" in group.columns:
            bias_values = pd.to_numeric(group["biascorr"], errors="coerce").dropna()
            record["mean_biascorr"] = float(bias_values.mean()) if len(bias_values) else np.nan

        record["screen_level_caveat"] = varno in SCREEN_LEVEL_VARNOS
        rows.append(record)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["statid", "n_active"], ascending=[True, False])


def unknown_code_report(table: pd.DataFrame) -> Dict[str, List[int]]:
    """List codes present in the data that the config tables do not name.

    A long list here means a stale code table in era5_odb_config.py, not a bad
    retrieval -- the numbers are still correct, only the labels are missing.
    """
    unknown: Dict[str, List[int]] = {}
    for column, labeller in (
        ("varno", varno_label),
        ("obstype", obstype_label),
        ("reportype", reportype_label),
    ):
        if column not in table.columns:
            continue
        codes = sorted({int(c) for c in table[column].dropna().unique()})
        missing = [c for c in codes if labeller(c).startswith(f"{column}_")]
        if missing:
            unknown[column] = missing
    return unknown


def print_verdict(census: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Print the plain-language answer, with its limits stated."""
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)

    if census.empty:
        print("No observations at all in the box for this period.")
        print("Before concluding 'not assimilated', rule out a bad query: verify")
        print("the MARS stream/time keys with --probe and widen the box.")
        return

    for statid, row in census.iterrows():
        label = f"station {statid}"
        if statid in STATION_IDS_OF_INTEREST:
            label += "  <- requested"
        print(f"\n{label}")
        print(f"  position   {row['lat_deg']:.3f} N, {row['lon_deg']:.3f} E")
        print(f"  rows       {int(row['n_rows']):,}   active {int(row['n_active']):,}"
              f"  ({100 * row['frac_active']:.1f}%)")

        station_rows = summary[summary["statid"] == statid]
        if station_rows.empty:
            continue
        for _, entry in station_rows.iterrows():
            flag = "ASSIMILATED" if entry["n_active"] > 0 else "not assimilated"
            note = ""
            if entry["screen_level_caveat"] and entry["n_active"] == 0:
                note = "  [screen-level: used by the separate OI/land analysis, not 4D-Var]"
            shrink = entry["depar_shrinkage"]
            shrink_text = f", rms(o-a)/rms(o-b)={shrink:.2f}" if np.isfinite(shrink) else ""
            print(f"    {entry['variable']:<28} {flag:<16} "
                  f"n={int(entry['n_obs']):<6} active={int(entry['n_active']):<6}"
                  f"{shrink_text}{note}")

    print("\nNot answerable from this archive, and true regardless of the above:")
    print("  ERA5 assimilates no surface radiation, cloud radar, or MWR LWP from")
    print("  ARM/NSA or NOAA-BRW. Those measurements are independent of ERA5 and")
    print("  can be used to evaluate it without circularity.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise ERA5 observation feedback near Barrow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--in-dir", default="data/obs_feedback",
                        help="Directory holding retrieved .odb (or .csv) files.")
    parser.add_argument("--csv", action="store_true",
                        help="Inputs are odb-sql CSV exports rather than ODB-2.")
    parser.add_argument("--out-dir", default=None,
                        help="Where the summary tables are written. Defaults to "
                             "<in-dir>/summary, so results always land beside "
                             "the inputs that produced them.")
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    pattern = "*.csv" if args.csv else "*.odb"
    paths = sorted(in_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"No {pattern} files in {in_dir}. Run the fetch step first.")

    print(f"Loading {len(paths)} file(s) from {in_dir}:")
    table = load_feedback(paths, is_csv=args.csv)
    if table.empty:
        raise SystemExit(
            "Files loaded but contain no rows.\n"
            "That is not yet evidence of non-assimilation -- an empty result is\n"
            "also what a wrong stream/time key or longitude convention produces.\n"
            "Verify the MARS keys with 'fetch_era5_obs_feedback.py --probe'."
        )

    print(f"\nusage flags from: {table.attrs.get('status_source', 'unknown')}")

    unknown = unknown_code_report(table)
    if unknown:
        print("\nCodes present in the data but not named in era5_odb_config.py:")
        for column, codes in unknown.items():
            print(f"  {column}: {codes}")
        print("  (counts are unaffected; only the labels are missing)")

    census = station_census(table)
    summary = usage_summary(table)
    history = position_history(table)

    # Output defaults to <in-dir>/summary rather than a fixed path. A fixed
    # default lets a run over the synthetic fixture write fabricated numbers
    # into the directory holding the real results, where nothing in the file
    # itself would later reveal the mix-up.
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Second belt on the same trousers: carry the fixture's marker into the
    # output filenames, so a stray synthetic table is identifiable wherever it
    # ends up, including when --out-dir was set by hand.
    prefix = "SYNTHETIC_" if any(p.name.startswith("SYNTHETIC") for p in paths) else ""
    census.to_csv(out_dir / f"{prefix}station_census.csv")
    summary.to_csv(out_dir / f"{prefix}usage_summary.csv", index=False)
    if not history.empty:
        history.to_csv(out_dir / f"{prefix}position_history.csv", index=False)

    print("\nStations found in the box:")
    print(census.to_string())

    if not history.empty:
        moved = history.groupby("statid").size()
        moved = moved[moved > 1]
        if len(moved):
            print("\nRELOCATION WARNING -- these identifiers report from more")
            print("than one position, so any statistic averaged across the whole")
            print("period mixes physically different sites:")
            print(history[history["statid"].isin(moved.index)].to_string(index=False))
            print("\nAt Utqiagvik this is expected: NWS SCN 18-89 moved WMO 70026")
            print("to 71.32267 N, 156.61784 W on 2019-02-12 -- the ARM NSA C1 site.")

    print_verdict(census, summary)
    print(f"\nTables written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
