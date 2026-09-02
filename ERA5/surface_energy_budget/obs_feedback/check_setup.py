#!/usr/bin/env python3
"""Check every prerequisite for the ERA5 observation-feedback retrieval.

Run this before anything else. Each check prints PASS, FAIL, or N/A along with
the specific next action, so you never have to guess which of four possible
missing pieces caused an error.

There are two routes to MARS and they need different things:

  ROUTE A -- run on an ECMWF platform (Atos HPC / ecgate).
             Needs: the 'mars' command. No Python packages, no API key.
             Convert to CSV there with 'odb sql' and analyse the CSV locally.

  ROUTE B -- drive MARS remotely from this laptop.
             Needs: ecmwf-api-client, ~/.ecmwfapirc with a valid key, and an
             ECMWF account entitled for the ERA5 feedback archive.
             Needs pyodc as well, to read the ODB-2 files it returns.

You need ONE of these, not both. The script reports which one is closest to
working. Nothing here can verify the entitlement itself -- that only shows up
when a real request either returns data or is refused.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

PASS = "PASS"
FAIL = "FAIL"
INFO = "----"

ECMWFAPIRC_PATH = Path.home() / ".ecmwfapirc"
EXPECTED_API_URL = "https://api.ecmwf.int/v1"


def report(status: str, label: str, detail: str = "") -> None:
    print(f"  [{status}] {label}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")


def check_core_packages() -> bool:
    """pandas and numpy -- needed on every route."""
    ok = True
    for name in ("numpy", "pandas"):
        try:
            module = __import__(name)
            report(PASS, f"{name} {getattr(module, '__version__', '?')}")
        except ImportError:
            report(FAIL, name, "pip install " + name)
            ok = False
    return ok


def check_mars_cli() -> bool:
    """Route A: the mars client, present only on ECMWF platforms."""
    path = shutil.which("mars")
    if path:
        report(PASS, "mars command line client", f"found at {path}")
        return True
    report(
        INFO,
        "mars command line client not found",
        "Expected on a laptop. Only present on Atos HPC / ecgate.\n"
        "If you have an ECMWF platform login, run the retrieval there\n"
        "instead of here -- it needs no API key and no Python packages.",
    )
    return False


def check_odb_cli() -> bool:
    """Route A companion: odb sql, for converting ODB-2 to CSV on the ECMWF side."""
    path = shutil.which("odb")
    if path:
        report(PASS, "odb tools", f"found at {path}")
        return True
    report(INFO, "odb tools not found", "Only needed on an ECMWF platform.")
    return False


def check_ecmwf_api_client() -> bool:
    """Route B: the Python client that submits MARS requests remotely."""
    try:
        import ecmwfapi  # noqa: F401
        report(PASS, "ecmwf-api-client installed")
        return True
    except ImportError:
        report(FAIL, "ecmwf-api-client not installed", "pip install ecmwf-api-client")
        return False


def check_ecmwfapirc() -> bool:
    """Route B: the credentials file.

    Validates structure only. The key is never printed and never echoed -- if
    this file is wrong you will find out from the server, not from here.
    """
    if not ECMWFAPIRC_PATH.exists():
        report(
            FAIL,
            f"{ECMWFAPIRC_PATH} missing",
            "Get your key from https://api.ecmwf.int/v1/key/ after logging in,\n"
            "then write it to that path. See the printed template below.",
        )
        return False

    try:
        content = json.loads(ECMWFAPIRC_PATH.read_text())
    except json.JSONDecodeError as exc:
        report(FAIL, f"{ECMWFAPIRC_PATH} is not valid JSON", str(exc))
        return False

    missing = [k for k in ("url", "key", "email") if k not in content]
    if missing:
        report(FAIL, f"{ECMWFAPIRC_PATH} missing keys", f"absent: {missing}")
        return False

    if content["url"].rstrip("/") != EXPECTED_API_URL:
        report(
            FAIL,
            "unexpected url in ~/.ecmwfapirc",
            f"found {content['url']!r}, expected {EXPECTED_API_URL!r}.\n"
            "A cdsapi url here is the usual mistake -- CDS and MARS are\n"
            "different services with different credentials.",
        )
        return False

    mode = ECMWFAPIRC_PATH.stat().st_mode & 0o777
    detail = f"account {content['email']}"
    if mode & 0o077:
        detail += f"\nPermissions are {mode:o}; tighten with chmod 600."
    report(PASS, "~/.ecmwfapirc present and well formed", detail)
    return True


def check_odb_reader() -> bool:
    """Route B: reading the ODB-2 files that come back."""
    for name in ("codc", "pyodc"):
        try:
            __import__(name)
            report(PASS, f"{name} available (ODB reader)")
            return True
        except ImportError:
            continue
    report(
        FAIL,
        "no ODB reader (codc or pyodc)",
        "pip install pyodc\n"
        "Not needed if you convert to CSV on the ECMWF side with 'odb sql'\n"
        "and run the analysis with --csv.",
    )
    return False


def check_cds_confusion() -> None:
    """Point out the credentials the user already has, and why they don't apply."""
    cdsapirc = Path.home() / ".cdsapirc"
    if cdsapirc.exists():
        report(
            INFO,
            "~/.cdsapirc exists (Copernicus CDS)",
            "This does NOT grant MARS access. The CDS serves ERA5 gridded\n"
            "output; the observation feedback lives only in MARS, behind a\n"
            "separate ECMWF account.",
        )


def main() -> int:
    print("\nERA5 observation-feedback setup check")
    print("=" * 74)

    print("\nCore (both routes):")
    core_ok = check_core_packages()

    print("\nRoute A -- run on an ECMWF platform:")
    mars_ok = check_mars_cli()
    check_odb_cli()

    print("\nRoute B -- drive MARS remotely from this machine:")
    api_ok = check_ecmwf_api_client()
    rc_ok = check_ecmwfapirc()
    reader_ok = check_odb_reader()

    print("\nNotes:")
    check_cds_confusion()

    print("\n" + "=" * 74)
    print("WHERE YOU STAND")
    print("=" * 74)

    if mars_ok:
        print("Route A is available. Run the retrieval here with the mars client.")
    elif core_ok and api_ok and rc_ok and reader_ok:
        print("Route B looks complete. Try the probe request next:")
        print("  python fetch_era5_obs_feedback.py --probe --start 2015-01-15")
        print("\nA permissions error at that point means your ECMWF account is not")
        print("entitled for the ERA5 feedback archive -- a licensing matter for")
        print("ECMWF user support, not something to debug in this code.")
    else:
        outstanding: List[Tuple[str, str]] = []
        if not core_ok:
            outstanding.append(("numpy/pandas", "pip install numpy pandas"))
        if not api_ok:
            outstanding.append(("ecmwf-api-client", "pip install ecmwf-api-client"))
        if not rc_ok:
            outstanding.append(("~/.ecmwfapirc", "see the template below"))
        if not reader_ok:
            outstanding.append(("ODB reader", "pip install pyodc"))
        print("Neither route is ready. Outstanding for Route B:")
        for item, action in outstanding:
            print(f"  - {item:<20} {action}")
        print("\nRoute A avoids all of these if you can get an ECMWF platform login.")

    if not rc_ok:
        print("\n~/.ecmwfapirc template (key from https://api.ecmwf.int/v1/key/):")
        print(json.dumps(
            {"url": EXPECTED_API_URL, "key": "<your key>", "email": "<your email>"},
            indent=4,
        ))
        print("Then:  chmod 600 ~/.ecmwfapirc")

    print("\nThe analysis half needs none of the above. To exercise it now:")
    print("  python make_synthetic_feedback.py")
    print("  python read_era5_obs_feedback.py --csv --in-dir data/obs_feedback_synthetic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
