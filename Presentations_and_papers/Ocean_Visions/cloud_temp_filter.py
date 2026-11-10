"""Cloud temperature from ERA5 pressure levels, and the supercooled-only filter.

WHAT IS HERE
============
* ``open_pressure_archive`` and ``weighted_cloud_temperature``: moved here
  unchanged from make_era5_hourly_values_csvs.py (which still imports them
  under the same names) so every product computes cloud temperature with the
  SAME code.
* ``SupercooledFilter``: removes liquid-containing (and liquid-only)
  cell-hours whose cloud temperature is not below 0 deg C. Used by every
  make_era5_*.py product when ``RunConfig.supercooled_only`` is True (see
  SUPERCOOLED_ONLY in ov_config.py: on by default for GDEX/S3, off for the
  local archive, which holds pressure levels for 2024/25 only).

CLOUD TEMPERATURE
=================
The liquid-water-path-weighted mean temperature of the column, as in
turbulent_flux_response.cloud_temperature_field:

    T_cloud = sum_l (q_l dp_l T_l) / sum_l (q_l dp_l)

q_l is the cloud liquid mixing ratio clwc (kg kg-1) on pressure level l, dp_l
the level's pressure thickness (Pa, clipped at the surface pressure by
convert_specific_to_absolute.layer_thickness_pa), T_l its temperature (K).
q_l dp_l / g is the liquid water path of level l (hydrostatic balance), so
each level counts in proportion to the liquid it holds; g cancels. A column
whose pressure-level liquid path is below CLOUD_T_MIN_LWP_G (1e-3 g m-2) has
no cloud temperature (NaN).

THE FILTER
==========
A liquid-containing cell-hour is KEPT only if T_cloud < 0 deg C. It is
dropped if T_cloud >= 0 deg C (warm) or if T_cloud is undefined (the 23
pressure levels hold < 1e-3 g m-2 of liquid although tclw is above the
0.01 g m-2 floor), because a temperature that cannot be computed cannot show
the cloud is supercooled. Both counts are reported. Liquid only is a subset of
liquid containing and is filtered the same way. Ice-only, clear-sky and
all-valid cell-hours are NOT touched: the filter concerns liquid, and the
denominators of the time fractions (all valid cell-hours) stay as they were.

A dropped cell-hour is not moved to another category: it is simply not a
supercooled liquid-containing cloud, so it leaves the liquid-containing
statistics and is counted nowhere else.

COST
====
Each product re-reads t and clwc on 23 levels for every hour that has at
least one liquid-containing cell (hours with none are skipped). On Casper the
decoded chunks come from the chunk cache after the first job that reads them
for the same domain, so later jobs are much cheaper than the first.

Nothing here is stochastic, so there is no random seed.
"""

from __future__ import annotations

import argparse

# First, before numpy: on Casper it sets thread limits and the cache location,
# and it puts the analysis modules on sys.path.
import ov_config as ov  # noqa: F401  (imported for its side effects)

import numpy as np

import plot_lwp_histogram_by_surface_class as lwph  # noqa: E402
from convert_specific_to_absolute import G_M_S2, layer_thickness_pa  # noqa: E402
from seb_analysis_common import available_regions, load_seb_data  # noqa: E402
from turbulent_flux_response import CLOUD_T_MIN_LWP_G  # noqa: E402  g m-2

FREEZING_C = 0.0              # deg C: supercooled means cloud temperature below this

# Pressure-level hours loaded at once: 23 levels x 2,501 cells x 3 fields at
# Barrow. The filter scales it down for larger domains (pl_block_hours).
PL_BLOCK_HOURS = 48
_PL_REFERENCE_CELLS = 4 * 41 * 61   # cells at which 48 h was still comfortable (Rui-region ran
                                    # 9,821 cells x 48 h at 6.2 GB peak in the hourly job)


def pl_block_hours(n_cells: int) -> int:
    """Pressure-level hours per read, so a block stays near the measured footprint."""
    return max(1, min(PL_BLOCK_HOURS, PL_BLOCK_HOURS * _PL_REFERENCE_CELLS // max(n_cells, 1)))


# ----------------------------------------------------------------------------
# Cloud temperature from the pressure-level archive
# ----------------------------------------------------------------------------
def open_pressure_archive(args: argparse.Namespace, ds, required: bool = False,
                          variables: tuple[str, ...] = ("t", "clwc", "ciwc")):
    """The pressure-level dataset for the region, or None if there is none.

    Same lookup as ``turbulent_flux_response.cloud_temperature_field``.
    ``required=True`` raises instead of returning None: the supercooled
    filter cannot run without it.
    """
    data_root = lwph.resolve_data_root(args.storage, args.data_root)
    region_pl = f"{args.region}_pressure"
    if region_pl not in available_regions(data_root):
        if required:
            raise FileNotFoundError(
                f"the supercooled-only filter needs the pressure-level archive {region_pl} "
                f"under {data_root}, which is not there. Run with --storage glade (Casper) "
                "or s3, or pass --supercooled-only off.")
        print(f"  !! no pressure-level archive {region_pl}: CloudTemp will be blank")
        return None
    pds = load_seb_data(region_pl, None, None, data_root)
    missing = sorted(set(variables) - set(pds.data_vars))
    if missing:
        raise KeyError(f"pressure archive is missing {missing}")
    # Same 0.25 deg grid in the same order, or the cell-by-cell pairing below
    # would silently mix columns.
    for c in ("latitude", "longitude"):
        if not np.array_equal(pds[c].values, ds[c].values):
            raise ValueError(f"pressure and single-level {c} axes differ")
    return pds


def weighted_cloud_temperature(pds, pos: np.ndarray, sp_pa: np.ndarray,
                               species: tuple[str, ...] = ("clwc", "ciwc")) -> tuple:
    """Water-weighted column temperature (deg C) for some hours.

    ``pos`` indexes the pressure archive's time axis; ``sp_pa`` is the
    surface pressure (Pa) of the same hours, ``(n, lat, lon)``. Returns one
    ``(n, lat, lon)`` array per entry of ``species`` -- by default
    ``(t_liq_C, t_ice_C)`` -- NaN where that species' weighting path is under
    CLOUD_T_MIN_LWP_G. ``species=("clwc",)`` reads and returns liquid only.
    """
    lv = pds["pressure_level"].values.astype(float)
    order = np.argsort(lv)                                   # ascending pressure
    p_pa = lv[order] * 100.0                                 # hPa -> Pa
    blk = pds[["t", *species]].isel(valid_time=pos).load()
    dims = [d for d in blk["t"].dims if d != "pressure_level"]

    def lev_last(name: str) -> np.ndarray:                   # (n, lat, lon, lev)
        return blk[name].transpose(*dims, "pressure_level").values.astype(
            np.float64)[..., order]

    t_k = lev_last("t")
    dp_pa = layer_thickness_pa(p_pa, sp_pa)                  # (n, lat, lon, lev)
    out = []
    for name in species:
        # Water path each level contributes, g m-2: q * dp / g * 1000.
        wp_l = np.clip(lev_last(name), 0.0, None) * dp_pa / G_M_S2 * 1000.0
        denom = wp_l.sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            t_mean = (wp_l * t_k).sum(axis=-1) / denom
        t_mean[~(denom >= CLOUD_T_MIN_LWP_G)] = np.nan
        out.append(t_mean - 273.15)                          # K -> deg C
    return tuple(out)


# ----------------------------------------------------------------------------
# The filter
# ----------------------------------------------------------------------------
class SupercooledFilter:
    """Drop liquid-containing cell-hours whose cloud temperature is not < 0 C.

    ``ds`` is the single-level dataset the caller streams. ``cell=(i, j)``
    restricts the pressure data to one grid cell (the netCDF script's UTQ
    series); arrays passed in then have shape ``(t, 1, 1)``.
    """

    def __init__(self, args: argparse.Namespace, ds, cell: tuple[int, int] | None = None):
        pds = open_pressure_archive(args, ds, required=True, variables=("t", "clwc"))
        if cell is not None:
            i, j = cell
            pds = pds.isel(latitude=[i], longitude=[j])
        self.pds = pds
        self.t_pl = np.asarray(pds["valid_time"].values)
        n_cells = pds.sizes["latitude"] * pds.sizes["longitude"]
        self.block_hours = pl_block_hours(n_cells)
        # Tallies over every call, for the log and the readME.
        self.n_checked = 0          # liquid-containing cell-hours tested
        self.n_warm = 0             # dropped: cloud temperature >= 0 C
        self.n_undefined = 0        # dropped: no cloud temperature
        self.n_hours_read = 0       # hours whose pressure levels were read

    def cloud_temperature_C(self, times: np.ndarray, sp_pa: np.ndarray,
                            need: np.ndarray | None = None) -> np.ndarray:
        """Liquid-weighted cloud temperature (deg C), ``(t, lat, lon)``.

        ``need`` (t,) bool: only these hours are read; the rest stay NaN.
        Every needed hour must exist in the pressure archive: a missing hour
        would otherwise be dropped from the statistics without a trace.
        """
        times = np.asarray(times)
        t_C = np.full(sp_pa.shape, np.nan)
        hours = np.flatnonzero(need) if need is not None else np.arange(times.size)
        if hours.size == 0:
            return t_C
        pos = np.searchsorted(self.t_pl, times[hours])
        ok = (pos < self.t_pl.size)
        ok[ok] = self.t_pl[pos[ok]] == times[hours][ok]
        if not ok.all():
            first = times[hours][~ok][0]
            raise ValueError(
                f"{int((~ok).sum())} hour(s) have no pressure-level data (first {first}); "
                "the supercooled-only filter needs every hour. Use --storage glade or s3, "
                "or --supercooled-only off.")
        for j0 in range(0, hours.size, self.block_hours):
            h = hours[j0:j0 + self.block_hours]
            (t_C[h],) = weighted_cloud_temperature(self.pds, pos[j0:j0 + self.block_hours],
                                                   sp_pa[h], species=("clwc",))
        self.n_hours_read += hours.size
        return t_C

    def apply_temperature(self, t_C: np.ndarray, masks: dict,
                          keys: tuple[str, ...] = ("liquid_containing", "liquid_only")) -> None:
        """Remove non-supercooled cell-hours from ``masks[keys]``, in place.

        ``masks["liquid_containing"]`` must be present: it is what is counted.
        """
        lc = masks["liquid_containing"]
        cold = t_C < FREEZING_C                     # NaN compares False: dropped
        self.n_checked += int(lc.sum())
        self.n_warm += int((lc & (t_C >= FREEZING_C)).sum())
        self.n_undefined += int((lc & np.isnan(t_C)).sum())
        for k in keys:
            masks[k] = masks[k] & cold

    def apply(self, times: np.ndarray, sp_pa: np.ndarray, masks: dict,
              keys: tuple[str, ...] = ("liquid_containing", "liquid_only")) -> np.ndarray:
        """Read cloud temperature where needed, filter ``masks`` in place, return it.

        Only hours holding at least one liquid-containing cell are read.
        """
        lc = masks["liquid_containing"]
        need = lc.reshape(lc.shape[0], -1).any(axis=1)
        t_C = self.cloud_temperature_C(times, sp_pa, need)
        self.apply_temperature(t_C, masks, keys)
        return t_C

    def summary(self) -> dict:
        n = max(self.n_checked, 1)
        return {"n_checked": self.n_checked, "n_warm": self.n_warm,
                "n_undefined": self.n_undefined, "n_kept": self.n_checked - self.n_warm
                - self.n_undefined, "share_kept": (self.n_checked - self.n_warm
                                                   - self.n_undefined) / n,
                "n_hours_read": self.n_hours_read}

    def report(self) -> str:
        s = self.summary()
        return (f"  supercooled filter: {s['n_kept']:,} of {s['n_checked']:,} liquid-containing "
                f"cell-hours kept ({100 * s['share_kept']:.2f}%); dropped {s['n_warm']:,} with "
                f"cloud temperature >= 0 C and {s['n_undefined']:,} with none "
                f"(pressure levels read for {s['n_hours_read']:,} hours)")


def readme_note(summary: dict | None) -> str:
    """Paragraph for a readME: what the filter did, or that it was off."""
    if summary is None:
        return ("SUPERCOOLED FILTER: off. Liquid-containing statistics include every "
                "liquid-containing cell-hour, whatever its cloud temperature.\n")
    s = summary
    return (
        "SUPERCOOLED FILTER: on. Liquid-containing and liquid-only cell-hours are kept only\n"
        "if their cloud temperature is below 0 C. Cloud temperature is the liquid-water-\n"
        "path-weighted mean temperature of the column from the ERA5 pressure levels:\n"
        "    T_cloud = sum_l (clwc_l dp_l T_l) / sum_l (clwc_l dp_l)\n"
        "(dp_l = level pressure thickness, clipped at the surface). Cell-hours whose\n"
        f"pressure-level liquid path is under {CLOUD_T_MIN_LWP_G:g} g m-2 have no cloud temperature and\n"
        "are dropped too. Ice-only, clear-sky and all-valid cell-hours are unchanged.\n"
        f"  liquid-containing cell-hours tested  {s['n_checked']:,}\n"
        f"  kept (T_cloud < 0 C)                 {s['n_kept']:,} ({100 * s['share_kept']:.2f}%)\n"
        f"  dropped, T_cloud >= 0 C              {s['n_warm']:,}\n"
        f"  dropped, no T_cloud                  {s['n_undefined']:,}\n"
    )
