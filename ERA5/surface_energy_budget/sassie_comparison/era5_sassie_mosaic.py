"""Stitch the Barrow strip and the three SASSIE gap tiles into one ERA5 dataset.

The SASSIE ship track needs 69.00-73.75 N, 166.25-144.50 W on the 0.25 deg grid.
That box is covered by four non-overlapping archives:

        lon ->    -166.25 .. -165.25 | -165.00 .. -150.00 | -149.75 .. -144.50
    73.75 .. 70.00   gap_west        |      barrow        |     gap_east
    69.75 .. 69.00   ------------------- gap_south --------------------------

``barrow`` comes from ``../data/barrow`` and predates this analysis; the three
``sassie_gap_*`` tiles are written by ``download_era5_sassie_gap.py``. All four
are produced by ``../download_era5_seb.py``, so they share a grid, a variable
set, an hourly time axis, and the canonical ERA5 short names.

``check_tiling()`` proves the arrangement is exact - every target cell supplied
once, none twice - and ``load_sassie_box()`` returns the merged Dataset.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from download_era5_seb import days_covered_by_file  # noqa: E402
from era5_seb_variables import REGIONS, normalise_names  # noqa: E402

GRID_SPACING_DEG = 0.25

# --- The box the SASSIE comparison needs ------------------------------------
# R/V Woldstad ran 69.196-73.520 N, 165.964-144.891 W. The nearest grid cells to
# those extremes are 69.25/73.50 N and 166.00/144.75 W; one cell of margin on
# each side gives the box below.
TARGET_NORTH_DEG, TARGET_SOUTH_DEG = 73.75, 69.00
TARGET_WEST_DEG, TARGET_EAST_DEG = -166.25, -144.50

# Tile name -> directory holding its netCDF files.
TILE_DIRS: dict[str, Path] = {
    "barrow": HERE.parent / "data" / "barrow",
    "sassie_gap_south": HERE / "data" / "sassie_gap_south",
    "sassie_gap_west": HERE / "data" / "sassie_gap_west",
    "sassie_gap_east": HERE / "data" / "sassie_gap_east",
}

# Per-file coordinates that carry no information for this analysis and whose
# values can differ between archives downloaded at different times. Left in
# place they turn a merge into a coordinate conflict.
DROP_COORDS = ("number", "expver")


def _grid_axis(low_deg: float, high_deg: float) -> np.ndarray:
    """The 0.25 deg grid lines inside [low, high], ascending."""
    n = int(round((high_deg - low_deg) / GRID_SPACING_DEG))
    return low_deg + GRID_SPACING_DEG * np.arange(n + 1)


def check_tiling() -> dict[str, int]:
    """Verify the four tiles cover the target box exactly once each.

    Raises ``AssertionError`` on an overlap or a hole, so a later edit to any
    region box cannot silently produce a mosaic with a seam in it.
    """
    target = {
        (round(float(la), 3), round(float(lo), 3))
        for la in _grid_axis(TARGET_SOUTH_DEG, TARGET_NORTH_DEG)
        for lo in _grid_axis(TARGET_WEST_DEG, TARGET_EAST_DEG)
    }

    counts: dict[tuple[float, float], int] = {}
    per_tile: dict[str, int] = {}
    for name in TILE_DIRS:
        r = REGIONS[name]
        cells = {
            (round(float(la), 3), round(float(lo), 3))
            for la in _grid_axis(r.south_deg, r.north_deg)
            for lo in _grid_axis(r.west_deg, r.east_deg)
        } & target
        per_tile[name] = len(cells)
        for cell in cells:
            counts[cell] = counts.get(cell, 0) + 1

    duplicated = {c for c, n in counts.items() if n > 1}
    missing = target - set(counts)
    assert not duplicated, f"{len(duplicated)} target cells are supplied by >1 tile"
    assert not missing, f"{len(missing)} target cells are supplied by no tile"
    assert sum(per_tile.values()) == len(target)

    per_tile["TOTAL"] = len(target)
    return per_tile


def _as_date(x) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return datetime.strptime(str(x), "%Y-%m-%d").date()


def _files_in_range(tile_dir: Path, start: date, end: date) -> list[Path]:
    """Files whose encoded day range intersects [start, end].

    Selecting on the filename rather than opening everything matters for
    ``barrow``, which holds 600+ files spanning 2000-2025.
    """
    wanted: set[date] = set()
    day = start
    while day <= end:
        wanted.add(day)
        day = date.fromordinal(day.toordinal() + 1)

    hits = [p for p in sorted(tile_dir.glob("*.nc"))
            if days_covered_by_file(p) & wanted]
    if not hits:
        raise FileNotFoundError(
            f"No files in {tile_dir} cover {start}..{end}. "
            f"For a sassie_gap_* tile, run download_era5_sassie_gap.py first."
        )
    return hits


def _load_tile(tile_dir: Path, start: date, end: date,
               variables: list[str] | None) -> xr.Dataset:
    files = _files_in_range(tile_dir, start, end)
    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
    ds = normalise_names(ds)
    if variables is not None:
        missing = [v for v in variables if v not in ds.variables]
        if missing:
            raise KeyError(f"{tile_dir.name} lacks {missing}")
        ds = ds[variables]
    ds = ds.drop_vars([c for c in DROP_COORDS if c in ds.coords], errors="ignore")
    ds = ds.sel(
        latitude=slice(TARGET_NORTH_DEG, TARGET_SOUTH_DEG),   # stored descending
        longitude=slice(TARGET_WEST_DEG, TARGET_EAST_DEG),
    )
    lo = np.datetime64(f"{start}T00:00:00")
    hi = np.datetime64(f"{end}T23:00:00")
    return ds.sel(valid_time=slice(lo, hi))


def load_sassie_box(start, end, variables: list[str] | None = None,
                    verbose: bool = True) -> xr.Dataset:
    """Merged ERA5 over 69.00-73.75 N, 166.25-144.50 W for [start, end].

    Parameters
    ----------
    start, end : Inclusive dates, ``YYYY-MM-DD`` or ``date``/``datetime``.
    variables : ERA5 short names to keep. ``None`` keeps everything the tiles
        have in common.
    """
    start, end = _as_date(start), _as_date(end)
    if verbose:
        print(f"tiling check: {check_tiling()}")

    tiles = []
    for name, tile_dir in TILE_DIRS.items():
        ds = _load_tile(tile_dir, start, end, variables)
        if verbose:
            print(f"  {name:<18} {ds.sizes['latitude']:>3} lat x "
                  f"{ds.sizes['longitude']:>3} lon x {ds.sizes['valid_time']:>4} h "
                  f"from {len(_files_in_range(tile_dir, start, end))} file(s)")
        tiles.append(ds)

    # The four rectangles form a U, not an N-D hypercube, so combine_by_coords
    # cannot infer the layout. An outer-join merge aligns them onto the union
    # grid and fills each tile's own footprint; "no_conflicts" makes an
    # accidental overlap with disagreeing values an error rather than a silent
    # pick-one.
    merged = xr.merge(tiles, join="outer", compat="no_conflicts")
    merged = merged.sortby("longitude").sortby("latitude", ascending=False)
    merged = merged.load()

    expected_lat = _grid_axis(TARGET_SOUTH_DEG, TARGET_NORTH_DEG).size
    expected_lon = _grid_axis(TARGET_WEST_DEG, TARGET_EAST_DEG).size
    if (merged.sizes["latitude"], merged.sizes["longitude"]) != (expected_lat, expected_lon):
        raise ValueError(
            f"mosaic is {merged.sizes['latitude']}x{merged.sizes['longitude']}, "
            f"expected {expected_lat}x{expected_lon}"
        )

    # A hole anywhere in the mosaic would show up as an all-NaN cell in a field
    # that is defined everywhere over land and sea alike.
    probe = "msdwlwrf" if "msdwlwrf" in merged else list(merged.data_vars)[0]
    n_empty = int((~np.isfinite(merged[probe])).all("valid_time").sum())
    if n_empty:
        raise ValueError(f"{n_empty} cells are empty in the mosaic ({probe} all-NaN)")

    merged.attrs["title"] = "ERA5 mosaic over the SASSIE 2022 ship-track box"
    merged.attrs["tiles"] = ", ".join(TILE_DIRS)
    merged.attrs["box_north_south_west_east_deg"] = [
        TARGET_NORTH_DEG, TARGET_SOUTH_DEG, TARGET_WEST_DEG, TARGET_EAST_DEG
    ]
    merged.attrs["convention"] = "ERA5: surface fluxes positive downward"
    if verbose:
        print(f"  merged             {merged.sizes['latitude']} lat x "
              f"{merged.sizes['longitude']} lon x {merged.sizes['valid_time']} h, "
              f"{len(merged.data_vars)} variables, no holes")
    return merged


if __name__ == "__main__":
    print(check_tiling())
