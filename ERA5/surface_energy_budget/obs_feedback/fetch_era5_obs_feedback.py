#!/usr/bin/env python3
"""Retrieve ERA5 observation feedback (ODB) for Utqiagvik/Barrow from MARS.

This is the ONLY way to settle whether a report was assimilated. The CDS serves
ERA5 gridded output -- the analysis after the fact -- and contains no record of
which observations produced it. MARS ``type=ofb`` serves the observation
feedback archive: one row per datum, carrying the observed value, the departure
from the first guess and from the analysis, the applied bias correction, and
the QC/usage flags that say whether the datum was ACTIVE (used), PASSIVE
(monitored only), REJECTED, or BLACKLISTED.

ACCESS -- READ THIS BEFORE YOU DEBUG ANYTHING ELSE
==================================================
Full MARS is not open to a CDS account. You need an ECMWF account with MARS
access, which in practice means one of:

  (a) a login on an ECMWF platform (Atos HPC, ecgate) where the ``mars`` command
      line client exists -- easiest by far, no credentials to manage; or
  (b) an ECMWF account entitled for the web API, driving MARS remotely through
      ``ecmwf-api-client`` with a ~/.ecmwfapirc key.

Not every ECMWF account carries the licence for the ERA5 feedback archive. If a
request returns a permissions error rather than an empty table, that is an
entitlement problem, not a query problem -- ask ECMWF user support.

This module writes requests for BOTH paths. ``--emit-only`` produces .req files
you can scp to Atos and run with ``mars file.req``; without it, the request is
submitted through the web API.

WHICH MARS KEYS ARE CERTAIN AND WHICH ARE NOT
=============================================
Certain:  class=ea (ERA5), type=ofb (observation feedback), obsgroup=conv
          (conventional observations, which is where SYNOP and radiosondes
          live), expver=1, and the ODB SQL ``filter``.

Verify for your period: ``stream`` and ``time``. ERA5 runs 12-hour 4D-Var
windows (09-21 UTC and 21-09 UTC), and the archive is keyed by window, not by
synoptic hour -- so ``time`` here is a WINDOW key, not the observation time.
The defaults below (stream=oper, time=0900/2100) are my best reading of the
ERA5 feedback layout, but I have not verified them against a live archive, and
a wrong window key returns "no data found" that looks like a real negative.

Settle it in one command before you trust any result. On Atos:

    mars <<'EOF'
    list, class=ea, type=ofb, obsgroup=conv, expver=1, date=2015-01-15
    EOF

``list`` costs nothing and prints the keys actually archived for that date. Set
--mars-stream / --mars-time from what it shows. ``--probe`` writes exactly that
request for you.

WHY THE ODB SQL FILTER MATTERS
==============================
Unfiltered, one 12-hour window of global conventional feedback is a large file,
and a multi-year request is unmanageable. The ``filter`` key pushes an SQL
WHERE clause into the archive so only rows inside the Barrow box are ever
materialised. Filtering on the box rather than on ``statid`` is deliberate --
see the docstring of era5_odb_config.py for why a station-identifier equality
test is a false-negative machine.

Requests are chunked by month and skipped if the target already exists, so an
interrupted multi-year pull resumes where it stopped.
"""

from __future__ import annotations

import argparse
import calendar
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from era5_odb_config import (
    DEFAULT_BOX_HALFWIDTH_LAT_DEG,
    DEFAULT_BOX_HALFWIDTH_LON_DEG,
    ODB_BODY_COLUMNS,
    ODB_HEADER_COLUMNS,
    barrow_box,
)

# MARS keys. See the module docstring for which of these to verify.
MARS_CLASS = "ea"
MARS_TYPE = "ofb"
MARS_EXPVER = "1"
DEFAULT_MARS_STREAM = "oper"
DEFAULT_MARS_TIME = "0900/2100"
DEFAULT_OBSGROUP = "conv"


def month_chunks(start: date, end: date) -> Iterator[Tuple[date, date]]:
    """Yield (first_day, last_day) for each calendar month spanned by the range."""
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        last_day_of_month = calendar.monthrange(cursor.year, cursor.month)[1]
        chunk_start = max(cursor, start)
        chunk_end = min(date(cursor.year, cursor.month, last_day_of_month), end)
        yield chunk_start, chunk_end
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)


def build_odb_filter(
    half_lat_deg: float = DEFAULT_BOX_HALFWIDTH_LAT_DEG,
    half_lon_deg: float = DEFAULT_BOX_HALFWIDTH_LON_DEG,
    varnos: Optional[Sequence[int]] = None,
) -> str:
    """Build the ODB/SQL ``filter`` string for one request.

    Two details that are easy to get wrong:

    1. Named bitfield members are selected explicitly (``datum_status.active``
       and friends) rather than the packed ``datum_status`` integer. That makes
       the usage flags arrive as plain 0/1 columns and removes any dependence on
       bit offsets, which differ between ODB schema versions.
    2. The longitude test accepts BOTH conventions. ERA5 feedback files are not
       consistent about whether ``lon@hdr`` runs -180..180 or 0..360, and a
       query written for one convention returns zero rows under the other --
       silently, and indistinguishably from "never assimilated".
    """
    box = barrow_box(half_lat_deg, half_lon_deg)

    select_columns: List[str] = list(ODB_HEADER_COLUMNS)
    # Replace the packed status columns with their named members.
    select_columns.remove("report_status")
    select_columns += [
        "report_status.active",
        "report_status.passive",
        "report_status.rejected",
        "report_status.blacklisted",
    ]
    for column in ODB_BODY_COLUMNS:
        if column == "datum_status":
            select_columns += [
                "datum_status.active",
                "datum_status.passive",
                "datum_status.rejected",
                "datum_status.blacklisted",
            ]
        else:
            select_columns.append(column)

    lat_test = f"lat >= {box['lat_min_deg']:.4f} and lat <= {box['lat_max_deg']:.4f}"
    lon_test = (
        f"((lon >= {box['lon_min_deg']:.4f} and lon <= {box['lon_max_deg']:.4f})"
        f" or (lon >= {box['lon_min_360_deg']:.4f}"
        f" and lon <= {box['lon_max_360_deg']:.4f}))"
    )

    where_clauses = [lat_test, lon_test]
    if varnos:
        varno_list = ",".join(str(int(v)) for v in varnos)
        where_clauses.append(f"varno in ({varno_list})")

    return (
        "select " + ", ".join(select_columns) + " where " + " and ".join(where_clauses)
    )


def build_mars_request(
    chunk_start: date,
    chunk_end: date,
    target_path: Path,
    odb_filter: str,
    stream: str = DEFAULT_MARS_STREAM,
    mars_time: str = DEFAULT_MARS_TIME,
    obsgroup: str = DEFAULT_OBSGROUP,
    reportypes: Optional[Sequence[int]] = None,
) -> Dict[str, str]:
    """Assemble one MARS retrieval as a key/value dict."""
    request: Dict[str, str] = {
        "class": MARS_CLASS,
        "type": MARS_TYPE,
        "expver": MARS_EXPVER,
        "stream": stream,
        "obsgroup": obsgroup,
        "date": f"{chunk_start:%Y-%m-%d}/to/{chunk_end:%Y-%m-%d}",
        "time": mars_time,
        "filter": odb_filter,
        "target": str(target_path),
    }
    # reportype is optional. Omitting it takes every conventional subtype, which
    # is what you want for a first pass -- narrowing it prematurely is another
    # way to manufacture a false negative.
    if reportypes:
        request["reportype"] = "/".join(str(int(r)) for r in reportypes)
    return request


def format_mars_request(request: Dict[str, str], verb: str = "retrieve") -> str:
    """Render a request dict as a MARS request file for the ``mars`` CLI."""
    lines = [verb + ","]
    items = list(request.items())
    for index, (key, value) in enumerate(items):
        terminator = "" if index == len(items) - 1 else ","
        # The SQL filter must be quoted; quoting the target too keeps paths
        # with spaces from silently truncating.
        rendered = f'"{value}"' if key in ("filter", "target") else value
        lines.append(f"    {key}={rendered}{terminator}")
    return "\n".join(lines) + "\n"


def build_probe_request(probe_date: date) -> str:
    """A zero-cost ``list`` request that reveals the archived stream/time keys."""
    request = {
        "class": MARS_CLASS,
        "type": MARS_TYPE,
        "expver": MARS_EXPVER,
        "obsgroup": DEFAULT_OBSGROUP,
        "date": f"{probe_date:%Y-%m-%d}",
    }
    return format_mars_request(request, verb="list")


def submit_via_web_api(request: Dict[str, str], target_path: Path) -> None:
    """Submit one request through ecmwf-api-client.

    Kept isolated so the request-building code above stays importable and
    testable on a machine with no ECMWF credentials at all.
    """
    try:
        from ecmwfapi import ECMWFService
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "ecmwf-api-client is not installed. Either\n"
            "  pip install ecmwf-api-client\n"
            "and put your key in ~/.ecmwfapirc, or re-run with --emit-only and\n"
            "run the generated .req files with the mars CLI on an ECMWF platform."
        ) from exc

    payload = {key: value for key, value in request.items() if key != "target"}
    service = ECMWFService("mars")
    service.execute(payload, str(target_path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve ERA5 observation feedback (ODB) near Barrow, Alaska.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start", required=False, default="2015-01-01",
                        help="First date, YYYY-MM-DD.")
    parser.add_argument("--end", required=False, default="2015-12-31",
                        help="Last date, YYYY-MM-DD.")
    parser.add_argument("--out-dir", default="data/obs_feedback",
                        help="Directory for retrieved .odb files.")
    parser.add_argument("--box-lat-deg", type=float, default=None,
                        help="Half-width of the search box in latitude, degrees.")
    parser.add_argument("--box-lon-deg", type=float, default=None,
                        help="Half-width of the search box in longitude, degrees.")
    parser.add_argument("--varnos", default="",
                        help="Comma-separated varnos to keep. Empty = all, which "
                             "is the honest default for a first pass.")
    parser.add_argument("--reportypes", default="",
                        help="Comma-separated ODB reportypes. Empty = all.")
    parser.add_argument("--mars-stream", default=DEFAULT_MARS_STREAM,
                        help="Verify with --probe before trusting the default.")
    parser.add_argument("--mars-time", default=DEFAULT_MARS_TIME,
                        help="4D-Var WINDOW key, not the observation time. "
                             "Verify with --probe.")
    parser.add_argument("--obsgroup", default=DEFAULT_OBSGROUP)
    parser.add_argument("--emit-only", action="store_true",
                        help="Write .req files for the mars CLI instead of "
                             "submitting through the web API.")
    parser.add_argument("--probe", action="store_true",
                        help="Write a zero-cost MARS 'list' request that shows "
                             "which stream/time keys actually exist, then exit.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        parser.error("--end precedes --start")

    if args.probe:
        probe_path = out_dir / "probe_list.req"
        probe_path.write_text(build_probe_request(start))
        print(f"Wrote {probe_path}\n")
        print(build_probe_request(start))
        print("Run it on an ECMWF platform with:  mars " + str(probe_path))
        print("Then set --mars-stream / --mars-time from what it lists.")
        return 0

    half_lat_deg = (
        args.box_lat_deg if args.box_lat_deg is not None
        else DEFAULT_BOX_HALFWIDTH_LAT_DEG
    )
    half_lon_deg = (
        args.box_lon_deg if args.box_lon_deg is not None
        else DEFAULT_BOX_HALFWIDTH_LON_DEG
    )
    box = barrow_box(half_lat_deg, half_lon_deg)
    print(
        "Search box: "
        f"lat {box['lat_min_deg']:.2f} to {box['lat_max_deg']:.2f} deg, "
        f"lon {box['lon_min_deg']:.2f} to {box['lon_max_deg']:.2f} deg east"
    )

    varnos = [int(v) for v in args.varnos.split(",") if v.strip()]
    reportypes = [int(r) for r in args.reportypes.split(",") if r.strip()]

    odb_filter = build_odb_filter(
        half_lat_deg=half_lat_deg,
        half_lon_deg=half_lon_deg,
        varnos=varnos or None,
    )

    n_submitted = 0
    n_skipped = 0
    for chunk_start, chunk_end in month_chunks(start, end):
        target = out_dir / f"barrow_ofb_{chunk_start:%Y%m}.odb"
        if target.exists() and target.stat().st_size > 0:
            print(f"skip  {target.name}  (already present)")
            n_skipped += 1
            continue

        request = build_mars_request(
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            target_path=target,
            odb_filter=odb_filter,
            stream=args.mars_stream,
            mars_time=args.mars_time,
            obsgroup=args.obsgroup,
            reportypes=reportypes or None,
        )

        if args.emit_only:
            req_path = target.with_suffix(".req")
            req_path.write_text(format_mars_request(request))
            print(f"wrote {req_path.name}")
        else:
            print(f"submit {target.name}  ({chunk_start} to {chunk_end})")
            submit_via_web_api(request, target)
        n_submitted += 1

    print(f"\n{n_submitted} request(s) issued, {n_skipped} already present.")
    if args.emit_only:
        print(f"Run them on an ECMWF platform:  for f in {out_dir}/*.req; "
              'do mars "$f"; done')
    return 0


if __name__ == "__main__":
    sys.exit(main())
