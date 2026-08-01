"""Shared command-line plumbing for the pipeline's entry points.

Lives in the package rather than scripts/_plot_common.py so that non-plotting
entry points (build_sonde_library.py, download_nsa_data.py) can use it without
importing matplotlib.

The one thing here is --data-root: every script needs to agree on where the
data tree lives, and the answer changes once the archive outgrows the internal
disk. THERMOCLDPHASE alone is ~280 GiB for 2014-2026, so an external drive is
the normal case, not the exception.

Three ways to point the pipeline at a drive, in precedence order:

    1. --data-root DIR                 per-run, explicit, best for reproducibility
    2. export ARM_NSA_DATA_ROOT=DIR    set once per shell, applies to every script
    3. nothing                         <repo>/data

Layout is unchanged in all three cases: DIR/raw/<datastream>/... for downloads,
DIR/processed/... for pipeline output. Figures always stay in <repo>/figures so
they travel with the code rather than the data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config

DATA_ROOT_HELP = (
    "Root of the data tree, i.e. the directory containing raw/ and "
    "processed/. Point this at an external drive to keep the archive off the "
    "internal disk, e.g. --data-root '/Volumes/My Passport/SCRIPPS/DOE_ARM/"
    "NSA/data'. QUOTE the path if it contains spaces. Overrides the "
    "ARM_NSA_DATA_ROOT environment variable; default is <repo>/data."
)


def add_data_root_argument(parser: argparse.ArgumentParser) -> None:
    """Register the shared --data-root flag on `parser`."""
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        metavar="DIR",
        help=DATA_ROOT_HELP,
    )


def apply_data_root(
    data_root: Path | None,
    *,
    require_raw: bool = True,
    quiet: bool = False,
) -> Path:
    """Repoint the pipeline at `data_root`, checking it is actually reachable.

    Returns the resolved root, or the unchanged default when `data_root` is
    None. Set require_raw=False for entry points that create raw/ themselves
    (i.e. the downloader) rather than reading from it.

    Failing loudly here matters more than it looks: an unmounted external drive
    makes every downstream reader report "no local files, download them first",
    which sends you off re-downloading hundreds of GB that are sitting on a
    disk you forgot to plug in.
    """
    if data_root is None:
        return config.DATA_ROOT
    root = config.set_data_root(data_root)
    _check_reachable(root, require_raw=require_raw)
    if not quiet:
        print(f"Data root: {root}")
    return root


def _check_reachable(root: Path, *, require_raw: bool) -> None:
    """Raise SystemExit with an actionable message if `root` is unusable."""
    parts = root.parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        mount_point = Path(*parts[:3])  # /Volumes/<drive>
        if not mount_point.exists():
            raise SystemExit(
                f"error: {mount_point} is not mounted, so {root} cannot be "
                f"used.\n"
                f"  Connect the drive and check the name with `ls /Volumes`.\n"
                f"  If the path contains spaces it must be quoted:\n"
                f'    --data-root "{root}"'
            )
    if not root.is_dir():
        raise SystemExit(
            f"error: data root {root} does not exist (or is not a "
            f"directory).\n"
            f'  Paths with spaces must be quoted: --data-root "{root}"'
        )
    if require_raw and not (root / "raw").is_dir():
        try:
            found = sorted(p.name for p in root.iterdir())[:8]
        except OSError:
            found = []
        raise SystemExit(
            f"error: {root} has no raw/ subdirectory.\n"
            f"  --data-root expects the PARENT of raw/ and processed/, e.g.\n"
            f"    --data-root '/Volumes/My Passport/SCRIPPS/DOE_ARM/NSA/data'\n"
            f"  (that directory currently contains: {found})"
        )
