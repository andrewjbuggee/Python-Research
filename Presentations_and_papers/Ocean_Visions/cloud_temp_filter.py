"""Cloud temperature from ERA5 pressure levels, and the supercooled-only filter.

WHAT IS HERE
============
* ``open_pressure_archive`` and ``weighted_cloud_temperature``: moved here
  from make_era5_hourly_values_csvs.py (which still imports them under the
  same names) so every product computes cloud temperature with the SAME code.
* ``SupercooledFilter``: removes liquid-containing (and liquid-only)
  cell-hours that are not supercooled. Used by every make_era5_*.py product
  when ``RunConfig.supercooled_only`` is True (see SUPERCOOLED_ONLY in
  ov_config.py: on by default for GDEX/S3, off for the local archive, whose
  pressure levels cover only 2024/25 and 2025/26 completely).

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
no T_cloud (NaN).

WHY SOME LIQUID HAS NO T_cloud
==============================
tclw integrates ERA5's liquid over all model levels down to the surface. The
pressure levels stop at 1000 hPa, so liquid lying between the surface and
1000 hPa (when surface pressure is above 1000 hPa, as it usually is under
Arctic winter highs) is in tclw but on no pressure level. The same happens on
Casper: GDEX holds the same pressure levels. Measured, Barrow 2024/25, every
6th hour (Oct-Mar): 16.4% of liquid-containing cell-hours had no T_cloud; in
100% of them clwc was 0 on every pressure level, and 99.5% had sp > 1000 hPa.
They are thin (tclw median 0.73 g m-2, 90th percentile 4.9 g m-2).

THE FILTER
==========
A liquid-containing cell-hour is KEPT when

  1. T_cloud is defined and T_cloud < 0 deg C; or
  2. T_cloud is undefined (the surface fallback) and BOTH the 2 m temperature
     (t2m) and the temperature at the lowest pressure level above the surface
     (normally 1000 hPa) are < 0 deg C.

Otherwise it is dropped. Rule 2 is my own choice, not from a source: the
liquid then lies between those two heights, so if both are below freezing
the layer between them almost certainly is too (it would take a warm layer
squeezed between two sub-freezing levels a few hundred metres apart). It is
not tested against an independent observation. Two checks on the same
Barrow 2024/25 sample:
  * the undefined cell-hours are far from freezing: median max(t2m, T_lowest)
    -18.3 C, 99th percentile -10.3 C; rule 2 keeps 99.95% of them;
  * applied to cell-hours that DO have a T_cloud, rule 2 calls a warm cloud
    cold in 222 of ~614,000 cases (0.04%). Its errors go mostly the other
    way (20,659 cold clouds over a warm surface called warm), which only
    removes clouds, so it is conservative for a supercooled population.

Liquid only is a subset of liquid containing and is filtered the same way.
Ice-only, clear-sky and all-valid cell-hours are NOT touched: the filter
concerns liquid, and the denominators of the time fractions (all valid
cell-hours) stay as they were. A dropped cell-hour is not moved to another
category; it leaves the liquid-containing statistics and is counted nowhere
else.

COST
====
Each product re-reads t and clwc on the pressure levels for every hour that
has at least one liquid-containing cell (hours with none are skipped). On
Casper the decoded chunks come from the chunk cache after the first job that
reads them for the same domain, so later jobs are much cheaper.

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
KELVIN_OFFSET = 273.15        # K at 0 deg C

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
                               species: tuple[str, ...] = ("clwc", "ciwc"),
                               lowest_level: bool = False) -> tuple:
    """Water-weighted column temperature (deg C) for some hours.

    ``pos`` indexes the pressure archive's time axis; ``sp_pa`` is the
    surface pressure (Pa) of the same hours, ``(n, lat, lon)``. Returns one
    ``(n, lat, lon)`` array per entry of ``species`` -- by default
    ``(t_liq_C, t_ice_C)`` -- NaN where that species' weighting path is under
    CLOUD_T_MIN_LWP_G. ``species=("clwc",)`` reads and returns liquid only.

    ``lowest_level=True`` appends one more array: the temperature (deg C) at
    the lowest pressure level ABOVE the surface (largest p_l < sp), from the
    same read, for the surface fallback. NaN if no level is above the surface.
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
        out.append(t_mean - KELVIN_OFFSET)                   # K -> deg C
    if lowest_level:
        # Levels are in ascending pressure, so the lowest level above the
        # ground is the LAST one with p_l < sp. above: (n, lat, lon, lev) bool.
        above = p_pa < np.asarray(sp_pa, dtype=float)[..., None]
        n_above = above.sum(axis=-1)                         # (n, lat, lon)
        idx = np.clip(n_above - 1, 0, None)[..., None]       # index of that level
        t_low_k = np.take_along_axis(t_k, idx, axis=-1)[..., 0]
        t_low_k[n_above == 0] = np.nan
        out.append(t_low_k - KELVIN_OFFSET)
    return tuple(out)


# ----------------------------------------------------------------------------
# The filter
# ----------------------------------------------------------------------------
def supercooled_mask(t_cloud_C: np.ndarray, t2m_C: np.ndarray,
                     t_lowest_C: np.ndarray) -> np.ndarray:
    """True where a liquid-bearing column counts as supercooled (rules 1 and 2).

    Rule 1: T_cloud defined and < 0 C. Rule 2 (surface fallback, used only
    where T_cloud is undefined): t2m < 0 C AND the lowest above-ground
    pressure level < 0 C. NaN inputs compare False, so they are not kept.
    """
    defined = np.isfinite(t_cloud_C)
    rule1 = defined & (t_cloud_C < FREEZING_C)
    rule2 = ~defined & (t2m_C < FREEZING_C) & (t_lowest_C < FREEZING_C)
    return rule1 | rule2


class SupercooledFilter:
    """Drop liquid-containing cell-hours that are not supercooled (see module doc).

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
        self.n_kept_tcloud = 0      # kept by rule 1: T_cloud < 0 C
        self.n_warm = 0             # dropped: T_cloud >= 0 C
        self.n_kept_fallback = 0    # no T_cloud; kept by rule 2 (t2m and lowest level < 0 C)
        self.n_dropped_fallback = 0  # no T_cloud; dropped by rule 2
        self.n_hours_read = 0       # hours whose pressure levels were read
        # The same tallies split by (season start year, group), plus the count of
        # valid cell-hours as the denominator, for fractions_table(). Filled only
        # when the caller passes ``tally=`` (hour_count and hourly do).
        self.group_tally: dict[tuple[int, str], np.ndarray] = {}

    def cloud_temperature_C(self, times: np.ndarray, sp_pa: np.ndarray,
                            need: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``(t_cloud_C, t_lowest_C)``, each ``(t, lat, lon)`` deg C.

        ``need`` (t,) bool: only these hours are read; the rest stay NaN.
        Every needed hour must exist in the pressure archive: a missing hour
        would otherwise be dropped from the statistics without a trace.
        """
        times = np.asarray(times)
        t_cloud_C = np.full(sp_pa.shape, np.nan)
        t_lowest_C = np.full(sp_pa.shape, np.nan)
        hours = np.flatnonzero(need) if need is not None else np.arange(times.size)
        if hours.size == 0:
            return t_cloud_C, t_lowest_C
        pos = np.searchsorted(self.t_pl, times[hours])
        ok = pos < self.t_pl.size
        ok[ok] = self.t_pl[pos[ok]] == times[hours][ok]
        if not ok.all():
            first = times[hours][~ok][0]
            raise ValueError(
                f"{int((~ok).sum())} hour(s) have no pressure-level data (first {first}); "
                "the supercooled-only filter needs every hour. Use --storage glade or s3, "
                "or --supercooled-only off.")
        for j0 in range(0, hours.size, self.block_hours):
            h = hours[j0:j0 + self.block_hours]
            t_cloud_C[h], t_lowest_C[h] = weighted_cloud_temperature(
                self.pds, pos[j0:j0 + self.block_hours], sp_pa[h],
                species=("clwc",), lowest_level=True)
        self.n_hours_read += hours.size
        return t_cloud_C, t_lowest_C

    def apply_temperature(self, t_cloud_C: np.ndarray, t2m_C: np.ndarray,
                          t_lowest_C: np.ndarray, masks: dict,
                          keys: tuple[str, ...] = ("liquid_containing", "liquid_only"),
                          tally: dict | None = None) -> None:
        """Remove non-supercooled cell-hours from ``masks[keys]``, in place.

        ``masks["liquid_containing"]`` must be present: it is what is counted.
        ``tally`` (optional) = ``{"season": (t,) season start years,
        "groups": {name: (t, lat, lon) bool}, "valid": (t, lat, lon) bool}``:
        the counts are then also kept per (season, group) for fractions_table().
        """
        lc = masks["liquid_containing"]
        keep = supercooled_mask(t_cloud_C, t2m_C, t_lowest_C)
        defined = np.isfinite(t_cloud_C)
        if tally is not None:
            self._tally_groups(tally, lc, defined, keep)
        self.n_checked += int(lc.sum())
        self.n_kept_tcloud += int((lc & defined & keep).sum())
        self.n_warm += int((lc & defined & ~keep).sum())
        self.n_kept_fallback += int((lc & ~defined & keep).sum())
        self.n_dropped_fallback += int((lc & ~defined & ~keep).sum())
        for k in keys:
            masks[k] = masks[k] & keep

    # Order of the per-group counts; fractions_table() names them.
    _TALLY_FIELDS = ("n_valid", "n_lc", "n_tcloud_below0", "n_tcloud_ge0",
                     "n_no_tcloud_kept", "n_no_tcloud_dropped")

    def _tally_groups(self, tally: dict, lc: np.ndarray, defined: np.ndarray,
                      keep: np.ndarray) -> None:
        """Add this block's counts to ``group_tally``, per (season, group)."""
        cats = (tally["valid"], lc, lc & defined & keep, lc & defined & ~keep,
                lc & ~defined & keep, lc & ~defined & ~keep)          # each (t, lat, lon)
        season = np.asarray(tally["season"])
        for s in np.unique(season):
            hs = season == s                                         # (t,) hours of season s
            for name, g in tally["groups"].items():
                gs = g[hs]
                row = self.group_tally.setdefault((int(s), name),
                                                  np.zeros(len(self._TALLY_FIELDS), np.int64))
                for k, c in enumerate(cats):
                    row[k] += int((c[hs] & gs).sum())

    def fractions_table(self, group_order: list[str] | None = None):
        """What the filter kept and dropped, per season and group, as a DataFrame.

        One row per (season, group), plus an "all" season row per group (the
        sums). Counts are cell-hours; fractions are of the stated denominator:

          frac_liquid_containing_of_valid   n_lc / n_valid (valid = finite tcc, tclw, tciw)
          frac_tcloud_below0_of_lc          kept by T_cloud < 0 C        / n_lc
          frac_tcloud_ge0_of_lc             dropped, T_cloud >= 0 C       / n_lc
          frac_no_tcloud_of_lc              no T_cloud (liquid below 1000 hPa) / n_lc
          frac_no_tcloud_kept               kept by the surface fallback  / n_no_tcloud
          frac_no_tcloud_dropped            dropped by the fallback       / n_no_tcloud
          frac_lc_kept, frac_lc_dropped     all kept / all dropped        / n_lc
        """
        import pandas as pd

        if not self.group_tally:
            return None
        seasons = sorted({s for s, _ in self.group_tally})
        groups = group_order or sorted({g for _, g in self.group_tally})
        rows = []
        for s in [*seasons, "all"]:
            for g in groups:
                if s == "all":
                    v = sum((self.group_tally[(y, g)] for y in seasons
                             if (y, g) in self.group_tally), np.zeros(6, np.int64))
                elif (s, g) in self.group_tally:
                    v = self.group_tally[(s, g)]
                else:
                    continue
                n_valid, n_lc, n_cold, n_warm, n_fb_keep, n_fb_drop = (int(x) for x in v)
                n_none = n_fb_keep + n_fb_drop
                frac = lambda a, b: a / b if b else np.nan      # NaN when the denominator is 0
                rows.append({
                    "season": s if s == "all" else f"{s}/{(s + 1) % 100:02d}", "group": g,
                    "n_valid_cell_hours": n_valid, "n_liquid_containing": n_lc,
                    "frac_liquid_containing_of_valid": frac(n_lc, n_valid),
                    "n_tcloud_below0": n_cold, "frac_tcloud_below0_of_lc": frac(n_cold, n_lc),
                    "n_tcloud_ge0": n_warm, "frac_tcloud_ge0_of_lc": frac(n_warm, n_lc),
                    "n_no_tcloud": n_none, "frac_no_tcloud_of_lc": frac(n_none, n_lc),
                    "n_no_tcloud_kept": n_fb_keep, "frac_no_tcloud_kept": frac(n_fb_keep, n_none),
                    "n_no_tcloud_dropped": n_fb_drop,
                    "frac_no_tcloud_dropped": frac(n_fb_drop, n_none),
                    "n_lc_kept": n_cold + n_fb_keep, "frac_lc_kept": frac(n_cold + n_fb_keep, n_lc),
                    "n_lc_dropped": n_warm + n_fb_drop,
                    "frac_lc_dropped": frac(n_warm + n_fb_drop, n_lc),
                })
        return pd.DataFrame(rows)

    def apply(self, times: np.ndarray, sp_pa: np.ndarray, t2m_k: np.ndarray, masks: dict,
              keys: tuple[str, ...] = ("liquid_containing", "liquid_only"),
              tally: dict | None = None) -> np.ndarray:
        """Read cloud temperature where needed, filter ``masks`` in place.

        ``t2m_k`` is the 2 m temperature (K), ``(t, lat, lon)``. Only hours
        holding at least one liquid-containing cell are read. Returns
        T_cloud (deg C), NaN where undefined or not read.
        """
        lc = masks["liquid_containing"]
        need = lc.reshape(lc.shape[0], -1).any(axis=1)
        t_cloud_C, t_lowest_C = self.cloud_temperature_C(times, sp_pa, need)
        t2m_C = np.asarray(t2m_k, dtype=float) - KELVIN_OFFSET
        self.apply_temperature(t_cloud_C, t2m_C, t_lowest_C, masks, keys, tally)
        return t_cloud_C

    def summary(self) -> dict:
        n_kept = self.n_kept_tcloud + self.n_kept_fallback
        return {"n_checked": self.n_checked, "n_kept": n_kept,
                "share_kept": n_kept / max(self.n_checked, 1),
                "n_kept_tcloud": self.n_kept_tcloud, "n_warm": self.n_warm,
                "n_kept_fallback": self.n_kept_fallback,
                "n_dropped_fallback": self.n_dropped_fallback,
                "n_hours_read": self.n_hours_read}

    def report(self) -> str:
        s = self.summary()
        return (f"  supercooled filter: {s['n_kept']:,} of {s['n_checked']:,} liquid-containing "
                f"cell-hours kept ({100 * s['share_kept']:.2f}%): {s['n_kept_tcloud']:,} with "
                f"T_cloud < 0 C, {s['n_kept_fallback']:,} by the surface fallback (no T_cloud); "
                f"dropped {s['n_warm']:,} with T_cloud >= 0 C and {s['n_dropped_fallback']:,} "
                f"failing the fallback (pressure levels read for {s['n_hours_read']:,} hours)")


def readme_note(summary: dict | None) -> str:
    """Paragraph for a readME: what the filter did, or that it was off."""
    if summary is None:
        return ("SUPERCOOLED FILTER: off. Liquid-containing statistics include every "
                "liquid-containing cell-hour, whatever its cloud temperature.\n")
    s = summary
    return (
        "SUPERCOOLED FILTER: on. Liquid-containing and liquid-only cell-hours are kept only\n"
        "if they are supercooled. Cloud temperature T_cloud is the liquid-water-path-\n"
        "weighted mean temperature of the column from the ERA5 pressure levels:\n"
        "    T_cloud = sum_l (clwc_l dp_l T_l) / sum_l (clwc_l dp_l)\n"
        "(dp_l = level pressure thickness, clipped at the surface). A cell-hour is kept if\n"
        "  1. T_cloud < 0 C, or\n"
        "  2. T_cloud cannot be computed (pressure-level liquid path under\n"
        f"     {CLOUD_T_MIN_LWP_G:g} g m-2: the liquid lies below the lowest pressure level, 1000 hPa)\n"
        "     and both the 2 m temperature and the lowest above-ground pressure level are\n"
        "     below 0 C.\n"
        "Ice-only, clear-sky and all-valid cell-hours are unchanged.\n"
        f"  liquid-containing cell-hours tested   {s['n_checked']:,}\n"
        f"  kept                                  {s['n_kept']:,} ({100 * s['share_kept']:.2f}%)\n"
        f"    by T_cloud < 0 C                    {s['n_kept_tcloud']:,}\n"
        f"    by the surface fallback (rule 2)    {s['n_kept_fallback']:,}\n"
        f"  dropped, T_cloud >= 0 C               {s['n_warm']:,}\n"
        f"  dropped, no T_cloud and fallback >= 0 {s['n_dropped_fallback']:,}\n"
    )


FRACTIONS_README = """\
ERA5_supercooledFilter_fractions_<span>.csv -- what the supercooled-only filter kept
and dropped, one row per season and group (UTQ, the five surface classes, AllCells),
plus an "all" row per group summing the seasons. Counts are cell-hours.
  n_valid_cell_hours                 cell-hours with finite tcc, tclw and tciw
  n_liquid_containing (lc)           liquid-containing cell-hours BEFORE the filter
  frac_liquid_containing_of_valid    lc / valid
  n_tcloud_below0, frac_..._of_lc    kept: cloud temperature < 0 C        (/ lc)
  n_tcloud_ge0, frac_..._of_lc       dropped: cloud temperature >= 0 C    (/ lc)
  n_no_tcloud, frac_no_tcloud_of_lc  no cloud temperature: the liquid lies below the
                                     lowest pressure level, 1000 hPa  (/ lc)
  n_no_tcloud_kept, frac_no_tcloud_kept        kept by the surface fallback (/ no_tcloud)
  n_no_tcloud_dropped, frac_no_tcloud_dropped  dropped by the fallback      (/ no_tcloud)
  n_lc_kept, frac_lc_kept / n_lc_dropped, frac_lc_dropped   overall      (/ lc)
"""
