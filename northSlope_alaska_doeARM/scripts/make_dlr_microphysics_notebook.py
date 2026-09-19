#!/usr/bin/env python3
"""Generate dlr_cloud_microphysics_barrow.ipynb (the DLR / cloud-microphysics study).

The notebook is generated from this script so the cell sources live in one
reviewable file; run with --execute to also run it with nbconvert.

    python scripts/make_dlr_microphysics_notebook.py            # write the .ipynb
    python scripts/make_dlr_microphysics_notebook.py --execute  # write + execute

Environment variables read by the notebook:
    DLR_NB_PERIODS   "YYYY-MM-DD:YYYY-MM-DD,..." (default: the two 2023/24, 2024/25 seasons)
    DLR_NB_LIDAR     mplgr (default) or hsrl
    DLR_NB_REBUILD   1 to rebuild the merged 1-min tables from the column files
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parent.parent
NB_PATH = REPO / "dlr_cloud_microphysics_barrow.ipynb"

CELLS = []


def md(src: str) -> None:
    CELLS.append(new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(new_code_cell(src.strip("\n")))


# =============================================================================
md(r'''
# Downwelling longwave and cloud microphysics at Utqiaġvik (ARM NSA C1)

**Question.** In the ERA5 regression of downwelling longwave (DLR) on liquid water path (LWP) below
(cold seasons 2024/25 and 2025/26, overcast liquid-bearing cells, Barrow domain), DLR spans
roughly 130 to 330 W m⁻² at LWP < 50 g m⁻². Which cloud properties set that spread, and do
cloud *microphysics* (droplet and ice-particle size, ice/liquid layering, precipitation) play a
measurable role? The practical motivation is mixed-phase cloud thinning (MCT; Villanueva et al.
2022): glaciating or precipitating out supercooled liquid reduces DLR, and the size of that
reduction depends on exactly the quantities examined here.

![ERA5 DLR vs LWP by surface class](figures/dlr_microphysics/era5_dlr_vs_lwp_by_surface_class_2024-2026.png)

**Hypotheses tested** (each gets a section):

| # | Hypothesis | Observable used |
|---|---|---|
| H1 | The low-LWP DLR spread is set by the **cloud (base) temperature**, i.e. the emission temperature of the lowest layer | sonde temperature at the lidar/radar cloud base (`t_cloud_base_K`) |
| H2 | It is set by the **surface–cloud temperature contrast** (inversions) | `t_cloud_base_K − t_skin_irt_K`, low-level inversion strength |
| H3 | It is set by **ice**: low-LWP columns are often mixed or ice-only, and ice water path (IWP) adds emissivity | Ze-based IWP proxy, phase mask |
| H4 | **Multi-layer clouds** add DLR above what the lowest layer gives | ARSCL layer count, upper-layer top temperature |
| H5 | The **vertical layering of liquid and ice** (liquid-topped mixed clouds, ice above liquid, precipitating ice below liquid) matters at fixed LWP | phase mask geometry |
| H6 | **Droplet / ice-particle size** changes emissivity at fixed water path | radar–MWR droplet radius (Frisch et al. 1995), MICROBASE r_e, Doppler fall speed, depolarisation |

**Period.** The two most recent complete cold seasons, **1 Sep 2023 – 30 Apr 2024** and
**1 Sep 2024 – 30 Apr 2025**. The product the study was scoped around, `microbasepi2`
("Continuous Baseline Microphysical Retrieval, Profile-Instantaneous"), exists at NSA only
for **2002-01-01 to 2011-03-22**, whereas the thermodynamic phase product `thermocldphase`
starts in Nov 2011 — the two never overlap (ARM Live archive, queried 2026-09-18). Its
successor `microbase` (nsamicrobaseC1.c1, 2011–2014 and 2020–2025) does overlap, and these two
seasons are the latest with complete phase, radiation and MICROBASE coverage (2025/26
`thermocldphase` stops on 2026-01-20 and the QCRAD pyrgeometer failed Dec 2025–Feb 2026).

**Approach.** Simple: conditional distributions, Pearson/Spearman correlations, ordinary
least squares with standardised coefficients, and matched-bin comparisons. Sample counts
are minutes; they are far from independent (the lag-1 autocorrelation of DLR is ≈ 0.99), so
every quoted uncertainty comes from a **day-block bootstrap** and effective sample sizes are
reported alongside n.
''')

md(r'''
## 0. What microphysical information actually exists at NSA (data inventory)

The honest answer to "is there useful data on ice habit, ice particle size and size distributions,
and the same for droplets" for a continuous multi-season record is: **no direct measurements,
only remote-sensing proxies with strong assumptions**. Archive coverage below is from the ARM Live
API (`arm_nsa.download.query_files`, 2026-09-18); "via Live" means downloadable programmatically.

| Product (datastream) | Coverage at NSA C1 | What it gives for microphysics | Caveat |
|---|---|---|---|
| **THERMOCLDPHASE** `nsathermocldphaseC1.c0` (c1: 2011-11 → 2014-02) | 2014-02 → 2026-01 (via Live) | pixel phase (liquid / ice / mixed / drizzle / rain / snow), per-layer phase, up to 10 ARSCL layers, KAZR Ze / Doppler velocity / spectral width / LDR, MPL backscatter + depolarisation, interpolated-sonde T/RH, MWR LWP, all on 30 s × 30 m | phase is a *classification* (Shupe 2007 rules); no sizes |
| **MICROBASE** `nsamicrobaseC1.c1` | 2011-11 → 2014-06, 2020-10 → 2025-12 (via Live), 670 MB/day | LWC, IWC, liquid r_e, ice r_e on 4 s × 30 m | liquid r_e is a function of LWC only (N = 200 cm⁻³, σ = 0.35 assumed); ice r_e is a function of temperature only (Ivanova et al. 2001); phase by temperature (−16 °C … 0 °C linear), not by lidar. Wang et al. (2025), DOE/SC-ARM-TR-095 |
| **MICROBASEPI2** `nsamicrobasepi2C1.c1` | 2002-01 → 2011-03 (via Live) | same quantities, MMCR era, 10 s × 45 m | same algorithm family; no overlap with THERMOCLDPHASE |
| **Shupe–Turner** `nsamicrobase2shupeturnC1.c1` | ≈ 2004 → 2019, *order only* (not via Live) | phase-aware LWC/IWC/r_e (Shupe et al. 2015), used by Bertrand et al. (2025) | must be ordered from the ARM Data Center |
| **ARSCL KAZR** `nsaarsclkazr1kolliasC1.c1` | 2011 →, via Live | full Ze, Doppler velocity, spectral width, LDR at 4 s | the Doppler velocity is the only *direct* particle-size-related observable (fall speed); THERMOCLDPHASE carries a 30-s subsample of all four moments, which this notebook uses |
| **MPL polarised** `nsamplpolfsC1.b1` | 2010 →, via Live | linear depolarisation ratio (phase; crude habit/orientation information) | carried in THERMOCLDPHASE as `mpl_ldr` |
| **AERI** `nsaaerich1/2C1.b1` | 2003 →, via Live (radiances only) | with a retrieval (MIXCRA, Turner 2005) gives liquid and ice r_e for optically thin clouds from the 8–13 µm window — the most radiatively relevant size information that exists here | no ARM VAP at NSA; would have to be run by us |
| **cldmicroprop (PI data)** | campaign only — Data Discovery lists it under M-PACE (Oct 2004) and ISDAC (Apr 2008); the two PI entries are the McFarquhar–Zhang aircraft in-situ products (moderately confident of the attribution; check the product page) | in-situ size distributions, habits, number concentrations | weeks, not seasons; ideal for calibrating the proxies below |
| ERA5 | continuous | none (bulk LWC/IWC, parameterised r_e) | — |

Consequences for this notebook:

* **Droplet size:** for liquid-only, non-drizzling clouds the KAZR reflectivity together with the MWR LWP
  constrains the droplet effective radius without assuming a number concentration (Frisch et al. 1995,
  *J. Atmos. Sci.* 52, 2788–2799; lognormal width assumed). That is the only semi-independent
  droplet-size estimate available continuously and it is derived in Section 8. MICROBASE's r_e is
  shown for comparison and demonstrated to be LWC^(1/3) by construction.
* **Ice size / habit:** the Doppler fall speed (`mdv_*`) and its relation to Ze are size/riming proxies;
  radar and lidar LDR are coarse habit/orientation proxies. No size-distribution parameters
  (Hansen & Travis 1974 effective variance etc.) are retrievable from these instruments.
* **Precipitation and layering:** these are the strong observables here — the 30 m phase mask gives ice
  below/above/within the liquid, snow and drizzle pixels, and multi-layer structure directly.
''')

# =============================================================================
md(r'''
## 1. Setup
''')
code(r'''
from __future__ import annotations

import os
import sys
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import stats

# Repo root: the notebook lives at the root of northSlope_alaska_doeARM.
REPO = Path.cwd() if (Path.cwd() / "arm_nsa").exists() else Path(
    "/Users/andrewbuggee/Documents/VS_CODE/Python-Research/northSlope_alaska_doeARM"
)
sys.path.insert(0, str(REPO))
from arm_nsa import config  # noqa: E402
from arm_nsa.column_features import COLUMN_CLASS, build_column_files  # noqa: E402
from arm_nsa.dlr_dataset import SIGMA_SB_W_M2_K4, build_dlr_dataset  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- reproducibility --------------------------------------------------------------
SEED = 20260918
rng = np.random.default_rng(SEED)
print(f"numpy RNG seed = {SEED}")

# --- periods / options (env overrides let the same notebook run on a test slice) ----
DEFAULT_PERIODS = "2023-09-01:2024-04-30,2024-09-01:2025-04-30"
PERIODS = [tuple(p.split(":")) for p in os.environ.get("DLR_NB_PERIODS", DEFAULT_PERIODS).split(",")]
LIDAR = os.environ.get("DLR_NB_LIDAR", "mplgr")   # phase from the MPL gradient method (whole record)
REBUILD = os.environ.get("DLR_NB_REBUILD", "0") == "1"
FIG_DIR = REPO / "figures" / "dlr_microphysics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SIGMA = SIGMA_SB_W_M2_K4  # W m-2 K-4

# --- analysis constants --------------------------------------------------------------
LWP_LOW_MAX_G_M2 = 30.0           # "low LWP" regime: below this, emissivity is not saturated
LWP_BIN_EDGES_G_M2 = np.array([-10.0, 5.0, 15.0, 30.0, 60.0, 120.0, 250.0, 600.0])
T_BASE_EDGES_K = np.arange(235.0, 281.0, 5.0)
MIN_N = 100                        # do not quote a statistic from fewer minutes than this
CF_WINDOW_MIN = 11                 # rolling window for the point-site "overcast" proxy
CF_OVERCAST = 0.9                  # fraction of cloudy minutes in the window
BOOT_N = 300                       # day-block bootstrap resamples
DOPPLER_SMOOTH_MIN = 5             # rolling median for the Doppler features (turbulence)

# --- plotting: fixed categorical order (validated reference palette), single-hue sequentials
PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
       "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
CLASS_COLORS = {"clear": "#9a9a96", "ice_only": PAL["blue"], "liquid_only": PAL["orange"],
                "mixed_column": PAL["aqua"], "unknown_contaminated": PAL["magenta"], "no_data": "#d6d6d2"}
CLASS_ORDER = ["clear", "ice_only", "mixed_column", "liquid_only"]
DENSITY_CMAP = "Blues"
TEMP_CMAP = "Oranges"
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":", "font.size": 9.5,
    "legend.frameon": False, "axes.titlesize": 10.5,
})


def savefig(fig, stem: str) -> None:
    """Save a figure into figures/dlr_microphysics/ (PNG)."""
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")


print("periods:", PERIODS, "| lidar:", LIDAR)
''')

# =============================================================================
md(r'''
## 2. Build / load the 1-min analysis table

Two preprocessing stages, both idempotent (existing outputs are skipped):

1. `arm_nsa.column_features.build_column_files` reduces every 30-s THERMOCLDPHASE profile to
   scalar column features (phase counts, liquid/ice geometry, layer structure, boundary temperatures
   from the interpolated sonde, radar and lidar summaries, MWR LWP). One file per day under
   `data/processed/dlr_columns/<lidar>/`.
2. `arm_nsa.dlr_dataset.build_dlr_dataset` aligns those with QCRAD fluxes, MET, the ground IRT skin
   temperature, CLDTYPE cloud type / precipitation and the sampled MICROBASE days on a 1-min grid
   (the :00 profile of each minute is used, tolerance 20 s).

Raw data come from `scripts/download_dlr_microphysics_winters.sh` (thermocldphase, qcrad, met, mwr,
gndirt, cldtype) and `scripts/reduce_microbase.py` (MICROBASE, every 5th day, reduced and deleted).
''')
code(r'''
pieces = []
for start, end in PERIODS:
    counts = build_column_files(start, end, lidar=LIDAR, verbose=False)
    print(f"{start}..{end}: column files built={counts['done']} existing={counts['skipped']} failed={counts['failed']}")
    try:
        part = build_dlr_dataset(start, end, lidar=LIDAR, force=REBUILD, verbose=False)
    except FileNotFoundError as err:
        print("  skipped (no data yet):", err)
        continue
    label = f"{start[:4]}/{end[2:4]}" if start[:4] != end[:4] else f"{start[:7]}..{end[:7]}"
    part["season"] = ("time", np.full(part.sizes["time"], label, dtype=object))
    print(f"  {label}: {part.sizes['time']:,} minutes, {part.attrs.get('n_column_days')} column days")
    pieces.append(part)
if not pieces:
    raise SystemExit("no data for any period -- run the download scripts first")
ds = xr.concat(pieces, dim="time", combine_attrs="drop_conflicts")
df = ds.to_dataframe()
df["month"] = df.index.month
df["day"] = df.index.floor("D")
print(f"\nmerged table: {len(df):,} minutes x {df.shape[1]} columns, "
      f"{df.index.min():%Y-%m-%d} .. {df.index.max():%Y-%m-%d}")
''')

md(r'''
### 2.1 Coverage and cloud-column class frequencies by month

`column_class` is the whole-column phase summary of the pixel mask: *clear* (valid profile, no
hydrometeor), *ice_only* (every hydrometeor pixel ice or snow), *liquid_only* (every hydrometeor
pixel liquid, drizzle or rain), *mixed_column* (liquid- and ice-containing pixels both present
somewhere in the column, including mixed-phase pixels). "Liquid-containing" below always means
`liquid_only ∪ mixed_column`, i.e. it is decided by the lidar/radar classifier, not by LWP.
''')
code(r'''
cls_name = df["column_class"].map(COLUMN_CLASS)
cov = pd.DataFrame({
    "minutes": df.groupby(["season", "month"]).size(),
    "phase_valid_%": 100 * df.groupby(["season", "month"])["n_valid"].apply(lambda s: (s > 0).mean()),
    "LWD_valid_%": 100 * df.groupby(["season", "month"])["lwdn_w_m2"].apply(lambda s: s.notna().mean()),
    "LWP_valid_%": 100 * df.groupby(["season", "month"])["lwp_g_m2"].apply(lambda s: s.notna().mean()),
})
frac = (pd.crosstab([df["season"], df["month"]], cls_name, normalize="index") * 100).round(1)
cov = cov.join(frac.add_prefix("% "))
# order months Sep..Apr
order = {m: i for i, m in enumerate([9, 10, 11, 12, 1, 2, 3, 4])}
cov = cov.reset_index().sort_values(["season", "month"], key=lambda s: s.map(order) if s.name == "month" else s)
display(cov.set_index(["season", "month"]).round(1))

fig, ax = plt.subplots(figsize=(9, 3.2))
sub = cov.set_index(["season", "month"])
x = np.arange(len(sub))
bottom = np.zeros(len(sub))
for c in CLASS_ORDER + ["unknown_contaminated", "no_data"]:
    col = f"% {c}"
    if col not in sub:
        continue
    ax.bar(x, sub[col].values, bottom=bottom, color=CLASS_COLORS[c], width=0.8, label=c, edgecolor="white", linewidth=0.8)
    bottom += sub[col].values
ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n{m:02d}" for s, m in sub.index], fontsize=8)
ax.set_ylabel("% of minutes")
ax.set_title("Column phase class by month (THERMOCLDPHASE pixel mask, %s)" % LIDAR)
ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
savefig(fig, "fig01_column_class_by_month")
plt.show()
''')

# =============================================================================
md(r'''
## 3. LWP quality: clear-sky offset and noise floor

MWRRET LWP has a retrieval offset that drifts (README: −47 g m⁻² median on 15 Nov 2025). Following
Hartig et al. (2026) the clear-sky retrievals are used as the zero point: for each season-month the
median LWP in clear columns (no hydrometeor pixel, no ARSCL layer) is subtracted from every sample,
and the robust spread of clear-sky LWP (1.4826 × MAD) is the effective noise floor. Months with an
offset larger than 15 g m⁻² in magnitude are flagged; they are kept but marked in the coverage table.
''')
code(r'''
clear_mask = (df["column_class"] == 1) & (df["n_layers"] == 0)
g = df.loc[clear_mask].groupby(["season", "month"])["lwp_g_m2"]
lwp_qc = pd.DataFrame({"n_clear": g.size(), "offset_g_m2": g.median(),
                       "noise_1sigma_g_m2": 1.4826 * g.apply(lambda s: (s - s.median()).abs().median())})
lwp_qc["flag"] = np.where(lwp_qc["offset_g_m2"].abs() > 15, "OFFSET>15", "")
display(lwp_qc.round(2))

key = pd.MultiIndex.from_arrays([df["season"], df["month"]])
df["lwp_offset_g_m2"] = lwp_qc["offset_g_m2"].reindex(key).fillna(0.0).values
df["lwp_noise_g_m2"] = lwp_qc["noise_1sigma_g_m2"].reindex(key).fillna(np.nan).values
df["lwp_adj_g_m2"] = df["lwp_g_m2"] - df["lwp_offset_g_m2"]
print(f"season-wide clear-sky noise floor (median of monthly 1-sigma): "
      f"{lwp_qc['noise_1sigma_g_m2'].median():.1f} g m-2")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
bins = np.arange(-60, 200, 4)
for c, col in [("clear", CLASS_COLORS["clear"]), ("ice_only", CLASS_COLORS["ice_only"]),
               ("mixed_column", CLASS_COLORS["mixed_column"]), ("liquid_only", CLASS_COLORS["liquid_only"])]:
    m = cls_name == c
    for ax, var in zip(axes, ["lwp_g_m2", "lwp_adj_g_m2"]):
        ax.hist(df.loc[m, var].dropna(), bins=bins, histtype="step", color=col, lw=1.6, label=c, density=True)
axes[0].set_title("MWR LWP as delivered")
axes[1].set_title("after monthly clear-sky offset")
for ax in axes:
    ax.set_xlabel("LWP [g m$^{-2}$]")
    ax.axvline(0, color="0.3", lw=0.8)
axes[0].set_ylabel("density")
axes[1].legend(fontsize=8)
savefig(fig, "fig02_lwp_clear_sky_offset")
plt.show()
''')

# =============================================================================
md(r'''
## 4. Derived quantities and analysis subsets

* `overcast_like`: fraction of cloudy minutes in an 11-min centred window ≥ 0.9 — the point-site
  analogue of the ERA5 "tcc ≥ 0.99" selection.
* `t_eff_K = (DLR/σ)^{1/4}`: the effective emission temperature of the sky.
* `sigT4_base`: blackbody emission at the cloud-base temperature, the ceiling for DLR from an opaque
  low cloud (neglecting the sub-cloud atmosphere, which at Barrow in winter adds a few W m⁻²).
* `fall_speed_*`: minus the mean Doppler velocity (file convention positive up), 5-min rolling median.
* `precip_ice_from_liq`: liquid-containing column with ≥ 2 ice-containing pixels *below* the liquid
  base or any snow pixel — the precipitating supercooled cloud MCT aims to create.
* `layer_class`: vertical arrangement of liquid and ice for liquid-containing columns (Section 7).
''')
code(r'''
df["cloudy"] = df["n_cloud"] > 0
df["cf_win"] = df["cloudy"].astype(float).rolling(CF_WINDOW_MIN, center=True, min_periods=6).mean()
df["overcast_like"] = df["cf_win"] >= CF_OVERCAST
df["liquid_containing"] = df["column_class"].isin([3, 4])
df["sigT4_base"] = SIGMA * df["t_cloud_base_K"] ** 4
df["sigT4_air"] = SIGMA * df["t_air_2m_K"] ** 4
df["t_eff_K"] = (df["lwdn_w_m2"] / SIGMA) ** 0.25
df["dT_base_skin_K"] = df["t_cloud_base_K"] - df["t_skin_irt_K"]
df["dT_base_air_K"] = df["t_cloud_base_K"] - df["t_air_2m_K"]
df["dT_eff_base_K"] = df["t_eff_K"] - df["t_cloud_base_K"]
df["twp_g_m2"] = df["lwp_adj_g_m2"].clip(lower=0) + df["iwp_proxy_g_m2"]
for v in ["mdv_ice_mean_m_s", "mdv_below_liq_mean_m_s", "mdv_liq_mean_m_s"]:
    df[v + "_sm"] = df[v].rolling(DOPPLER_SMOOTH_MIN, center=True, min_periods=2).median()
df["fall_speed_ice_m_s"] = -df["mdv_ice_mean_m_s_sm"]
df["fall_speed_below_liq_m_s"] = -df["mdv_below_liq_mean_m_s_sm"]
df["precip_ice_from_liq"] = df["liquid_containing"] & ((df["n_ice_below_liq"] >= 2) | (df["n_snow_px"] > 0))
df["nonprecip_liq"] = df["liquid_containing"] & (df["n_ice_below_liq"] == 0) & (df["n_snow_px"] == 0) & (df["n_liq_precip_px"] == 0)


def layer_class(d: pd.DataFrame) -> pd.Series:
    """Vertical arrangement of liquid and ice for liquid-containing columns."""
    out = pd.Series("not_liquid", index=d.index, dtype=object)
    lc = d["liquid_containing"]
    single = d["n_layers"] == 1
    multi = d["n_layers"] >= 2
    liq_only = d["column_class"] == 3
    ice_below = d["n_ice_below_liq"] > 0
    ice_above = d["n_ice_above_liq"] > 0
    out[lc & single & liq_only] = "1L liquid only"
    out[lc & single & ~liq_only & ice_below & ~ice_above] = "1L liquid-topped, ice below"
    out[lc & single & ~liq_only & ice_above & ~ice_below] = "1L ice above liquid"
    out[lc & single & ~liq_only & ice_above & ice_below] = "1L ice above and below"
    out[lc & single & ~liq_only & ~ice_above & ~ice_below] = "1L mixed pixels only"
    out[lc & multi & d["layer1_phase"].isin([1, 3])] = "multi-layer, liquid lowest"
    out[lc & multi & (d["layer1_phase"] == 2)] = "multi-layer, ice lowest"
    out[lc & (d["n_layers"] == 0)] = "liquid, no ARSCL layer"
    return out


df["layer_class"] = layer_class(df)
LAYER_ORDER = ["1L liquid only", "1L liquid-topped, ice below", "1L ice above liquid",
               "1L ice above and below", "1L mixed pixels only", "multi-layer, liquid lowest",
               "multi-layer, ice lowest"]
LAYER_COLORS = dict(zip(LAYER_ORDER, [PAL["orange"], PAL["aqua"], PAL["blue"], PAL["violet"],
                                      PAL["yellow"], PAL["magenta"], PAL["green"]]))

base = df["lwdn_w_m2"].notna() & (df["n_valid"] > 0)
liq = base & df["liquid_containing"] & df["lwp_adj_g_m2"].notna()
liq_low = liq & (df["lwp_adj_g_m2"] < LWP_LOW_MAX_G_M2)
ice_only = base & (df["column_class"] == 2)
clear_sky = base & (df["column_class"] == 1) & (df["n_layers"] == 0) & (df["cf_win"] == 0)
print(f"minutes with DLR and a valid phase profile : {base.sum():>8,}")
print(f"  liquid-containing (lidar/radar)          : {liq.sum():>8,}  ({100*liq.sum()/base.sum():.1f}%)")
print(f"    of which LWP_adj < {LWP_LOW_MAX_G_M2:.0f} g m-2            : {liq_low.sum():>8,}")
print(f"  ice-only                                 : {ice_only.sum():>8,}")
print(f"  clear (no layer, clear 11-min window)    : {clear_sky.sum():>8,}")
print("\nlayer classes among liquid-containing minutes:")
display(df.loc[liq, "layer_class"].value_counts().to_frame("minutes").assign(pct=lambda t: (100 * t["minutes"] / t["minutes"].sum()).round(1)))
''')

md(r'''
### 4.1 Statistical helpers

* `corr_table`: Pearson r and Spearman ρ of DLR with each candidate, with the effective sample size
  n_eff = n (1 − r₁ₓ r₁ᵧ)/(1 + r₁ₓ r₁ᵧ) from the lag-1 autocorrelations (Bretherton et al. 1999,
  *J. Climate* 12, 1990–2009) and the corresponding p-value.
* `ols`: OLS with standardised coefficients β = b·s_x/s_y, so predictors on different scales compare.
* `day_block_bootstrap`: percentile CIs by resampling whole days with replacement (seeded RNG).
* `binned`: median / IQR / count of y in bins of x.
''')
code(r'''
def lag1(s: pd.Series) -> float:
    """Lag-1 autocorrelation on the minute grid (NaN-pairs dropped)."""
    return float(s.autocorr(lag=1)) if s.notna().sum() > 10 else np.nan


def n_effective(x: pd.Series, y: pd.Series) -> float:
    r1x, r1y = lag1(x), lag1(y)
    n = int((x.notna() & y.notna()).sum())
    if not np.isfinite(r1x) or not np.isfinite(r1y):
        return float(n)
    return n * (1 - r1x * r1y) / (1 + r1x * r1y)


def corr_table(d: pd.DataFrame, y: str, candidates: dict, min_n: int = MIN_N) -> pd.DataFrame:
    """Correlations of d[y] with each candidate column (label -> column name)."""
    rows = []
    for label, col in candidates.items():
        if col not in d:
            continue
        m = d[y].notna() & d[col].notna()
        n = int(m.sum())
        if n < min_n:
            continue
        xs, ys = d.loc[m, col].astype(float), d.loc[m, y].astype(float)
        r, _ = stats.pearsonr(xs, ys)
        rho, _ = stats.spearmanr(xs, ys)
        # re-index onto the minute grid so the lag-1 autocorrelation is a true 1-min lag
        full_x = d[col].where(m)
        full_y = d[y].where(m)
        ne = n_effective(full_x, full_y)
        t = r * np.sqrt(max(ne - 2, 1) / max(1 - r**2, 1e-12))
        p = 2 * stats.t.sf(abs(t), max(ne - 2, 1))
        rows.append({"predictor": label, "column": col, "n": n, "n_eff": int(ne), "pearson_r": r,
                     "spearman_rho": rho, "r2": r**2, "p_neff": p})
    out = pd.DataFrame(rows).set_index("predictor")
    return out.reindex(out["pearson_r"].abs().sort_values(ascending=False).index)


def ols(d: pd.DataFrame, y: str, xcols: list, min_n: int = MIN_N):
    """OLS of d[y] on d[xcols] with intercept; returns dict with coefficients, standardised betas, R2."""
    m = d[y].notna()
    for c in xcols:
        m &= d[c].notna()
    if m.sum() < min_n:
        return None
    X = d.loc[m, xcols].astype(float).values
    yy = d.loc[m, y].astype(float).values
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    yhat = A @ beta
    ss_res = float(((yy - yhat) ** 2).sum())
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    std_beta = beta[1:] * X.std(axis=0) / yy.std()
    return {"n": int(m.sum()), "r2": r2, "rmse": float(np.sqrt(ss_res / len(yy))),
            "intercept": float(beta[0]), "coef": dict(zip(xcols, beta[1:])),
            "std_coef": dict(zip(xcols, std_beta)), "mask": m}


def residual_after(d: pd.DataFrame, y: str, controls: list) -> pd.Series:
    """Residual of d[y] after an OLS fit on the controls (NaN where not fitted)."""
    fit = ols(d, y, controls)
    out = pd.Series(np.nan, index=d.index)
    if fit is None:
        return out
    m = fit["mask"]
    X = d.loc[m, controls].astype(float).values
    out[m] = d.loc[m, y].values - (fit["intercept"] + X @ np.array([fit["coef"][c] for c in controls]))
    return out


def day_block_bootstrap(d: pd.DataFrame, stat, n_boot: int = BOOT_N, ci: float = 95.0):
    """Percentile CI of stat(frame) resampling whole days with replacement (seeded)."""
    groups = d.groupby(d["day"]).indices
    days = np.array(list(groups))
    if days.size < 8:
        return np.nan, np.nan, np.nan
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(days, size=days.size, replace=True)
        idx = np.concatenate([groups[k] for k in pick])
        vals.append(stat(d.iloc[idx]))
    vals = np.asarray(vals, dtype=float)
    lo, hi = np.nanpercentile(vals, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(stat(d)), float(lo), float(hi)


def binned(d: pd.DataFrame, x: str, y: str, edges, min_n: int = 30) -> pd.DataFrame:
    """Median, quartiles and count of y in bins of x."""
    cut = pd.cut(d[x], edges)
    g = d.groupby(cut, observed=True)[y]
    out = pd.DataFrame({"n": g.size(), "median": g.median(), "q25": g.quantile(0.25),
                        "q75": g.quantile(0.75), "mean": g.mean()})
    out["x_mid"] = [c.mid for c in out.index]
    return out[out["n"] >= min_n]


def fmt_ci(v, lo, hi, unit="", nd=2):
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]{unit}"


print("helpers defined")
''')

# =============================================================================
md(r'''
## 5. The ERA5 figure, reproduced with the ARM observations

DLR against LWP for overcast-like, liquid-containing minutes; the ERA5 Utqiaġvik panel gave
DLR = 0.443·LWP + 218 W m⁻² (r = +0.67, 2,685 cell-hours) for 2024/25–2025/26. Panel (b) colours the
same points by the cloud-base temperature — if H1 holds, the vertical spread at low LWP should be
ordered by colour. Panels (c) and (d) split by column class and show the ice-only clouds against
the Ze-based IWP proxy instead.
''')
code(r'''
ov = liq & df["overcast_like"]
sub = df.loc[ov]
fit_all = ols(sub, "lwdn_w_m2", ["lwp_adj_g_m2"])
slope_ci = day_block_bootstrap(sub, lambda t: ols(t, "lwdn_w_m2", ["lwp_adj_g_m2"])["coef"]["lwp_adj_g_m2"])
print(f"overcast-like liquid-containing minutes: n = {len(sub):,} on {sub['day'].nunique()} days")
print(f"DLR = {fit_all['coef']['lwp_adj_g_m2']:.3f} LWP + {fit_all['intercept']:.1f}   r2 = {fit_all['r2']:.3f}"
      f"   slope 95% CI (day blocks) {fmt_ci(*slope_ci, ' W m-2 per g m-2', 3)}")
print("ERA5 Utqiagvik panel for comparison: slope 0.443, intercept 218.3, r2 = 0.453")

fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
xmax = 350
ax = axes[0, 0]
hb = ax.hexbin(sub["lwp_adj_g_m2"], sub["lwdn_w_m2"], gridsize=70, extent=(-10, xmax, 100, 350),
               cmap=DENSITY_CMAP, bins="log", mincnt=1, linewidths=0.2)
xx = np.linspace(-10, xmax, 50)
ax.plot(xx, fit_all["intercept"] + fit_all["coef"]["lwp_adj_g_m2"] * xx, color="k", lw=1.6, label="OLS, this notebook")
ax.plot(xx, 218.29 + 0.443 * xx, color=PAL["red"], lw=1.4, ls="--", label="ERA5 Utqiaġvik panel")
ax.set_title(f"(a) overcast-like liquid-containing minutes, n = {len(sub):,}")
ax.legend(fontsize=8, loc="lower right")
fig.colorbar(hb, ax=ax, label="minutes (log)")

ax = axes[0, 1]
samp = sub.sample(min(len(sub), 25000), random_state=SEED)
sc = ax.scatter(samp["lwp_adj_g_m2"], samp["lwdn_w_m2"], c=samp["t_cloud_base_K"] - 273.15, s=3, cmap=TEMP_CMAP,
                vmin=-35, vmax=0, alpha=0.7, linewidths=0)
ax.set_title("(b) coloured by cloud-base temperature")
fig.colorbar(sc, ax=ax, label="cloud-base T [°C]")

ax = axes[1, 0]
for c in ["liquid_only", "mixed_column"]:
    m = ov & (cls_name == c)
    s2 = df.loc[m]
    if len(s2) < MIN_N:
        continue
    f = ols(s2, "lwdn_w_m2", ["lwp_adj_g_m2"])
    ax.scatter(s2["lwp_adj_g_m2"].sample(min(len(s2), 8000), random_state=SEED),
               s2["lwdn_w_m2"].sample(min(len(s2), 8000), random_state=SEED), s=2, color=CLASS_COLORS[c], alpha=0.35, linewidths=0)
    ax.plot(xx, f["intercept"] + f["coef"]["lwp_adj_g_m2"] * xx, color=CLASS_COLORS[c], lw=2,
            label=f"{c}: {f['coef']['lwp_adj_g_m2']:.3f} LWP + {f['intercept']:.0f}, r2 {f['r2']:.2f}, n {f['n']:,}")
ax.set_title("(c) by column class")
ax.legend(fontsize=8, loc="lower right")

ax = axes[1, 1]
s3 = df.loc[ice_only & df["overcast_like"] & (df["iwp_proxy_g_m2"] > 0)]
if len(s3) > MIN_N:
    hb2 = ax.hexbin(s3["iwp_proxy_g_m2"], s3["lwdn_w_m2"], gridsize=60, xscale="log", extent=(-1, 3, 100, 350),
                    cmap=DENSITY_CMAP, bins="log", mincnt=1, linewidths=0.2)
    b = binned(s3, "iwp_proxy_g_m2", "lwdn_w_m2", np.logspace(-1, 3, 13))
    ax.plot(b["x_mid"], b["median"], color="k", lw=1.6, marker="o", ms=4, label="median per IWP bin")
    fig.colorbar(hb2, ax=ax, label="minutes (log)")
    ax.legend(fontsize=8, loc="lower right")
ax.set_xscale("log")
ax.set_xlabel("IWP proxy from Ze [g m$^{-2}$]")
ax.set_title(f"(d) ice-only overcast-like minutes, n = {len(s3):,}")
for ax in axes.flat[:3]:
    ax.set_xlim(-10, xmax)
for ax in axes.flat:
    ax.set_ylim(100, 350)
    ax.set_ylabel("DLR [W m$^{-2}$]")
for ax in axes.flat[:3]:
    ax.set_xlabel("LWP (MWR, clear-sky corrected) [g m$^{-2}$]")
fig.suptitle(f"ARM NSA C1, {', '.join(sorted(df['season'].unique()))}, Sep–Apr, 1-min", y=1.0)
fig.tight_layout()
savefig(fig, "fig03_dlr_vs_lwp_arm")
plt.show()
''')

# =============================================================================
md(r'''
## 6. The low-LWP regime: what orders the DLR spread? (H1–H4, single predictors)

Liquid-containing minutes with LWP < 30 g m⁻². The table ranks candidate drivers by |r| with DLR;
`n_eff` is the autocorrelation-corrected sample size and `p_neff` the p-value computed with it.

**Read the surface temperatures with care.** In polar night the skin and 2-m temperatures are
*responses* to DLR on sub-daily time scales (about half of a DLR change goes into LWU within the
hour, Sledd et al. 2025), so their correlation with DLR is partly circular and they are marked
`[response]`. The cloud-base temperature, PWV, ice and layering are the upstream quantities; the
contrast `cloud base − skin T` mixes both.
The binned panels then show the conditional medians for the leading candidates, split into LWP
sub-bins so that the LWP dependence itself does not masquerade as a temperature dependence.
''')
code(r'''
CANDIDATES = {
    "cloud-base T (lowest hydrometeor)": "t_cloud_base_K",
    "liquid-base T": "t_liq_base_K",
    "liquid-top T": "t_liq_top_K",
    "cloud-top T (highest hydrometeor)": "t_cloud_top_K",
    "highest layer-top T": "t_top_max_K",
    "2-m air T [response]": "t_air_2m_K",
    "skin T (IRT) [response]": "t_skin_irt_K",
    "cloud base − skin T": "dT_base_skin_K",
    "cloud base − 2-m air T": "dT_base_air_K",
    "low-level inversion strength": "inversion_K",
    "PWV": "pwv_cm",
    "LWP (adjusted)": "lwp_adj_g_m2",
    "IWP proxy (whole column)": "iwp_proxy_g_m2",
    "IWP below liquid base": "iwp_below_liq_g_m2",
    "IWP above liquid top": "iwp_above_liq_g_m2",
    "ice fraction of pixels": "frc_ice",
    "number of ARSCL layers": "n_layers",
    "cloud-base height": "cloud_base_m",
    "highest layer top": "top_max_m",
    "liquid depth": "liq_depth_m",
    "max Ze": "ze_max_dbz",
    "ice fall speed": "fall_speed_ice_m_s",
    "MPL LDR in liquid": "mpl_ldr_liq_mean",
    "cloud fraction (11 min)": "cf_win",
}
low = df.loc[liq_low]
ct_low = corr_table(low, "lwdn_w_m2", CANDIDATES)
print(f"liquid-containing, LWP < {LWP_LOW_MAX_G_M2:.0f} g m-2: n = {len(low):,} minutes, "
      f"DLR 5-95% = {low['lwdn_w_m2'].quantile(0.05):.0f}–{low['lwdn_w_m2'].quantile(0.95):.0f} W m-2, "
      f"std = {low['lwdn_w_m2'].std():.1f}")
display(ct_low.round(3))

fig, ax = plt.subplots(figsize=(7, 6))
t = ct_low.iloc[::-1]
ax.barh(t.index, t["pearson_r"], color=[PAL["blue"] if v > 0 else PAL["orange"] for v in t["pearson_r"]], height=0.65)
ax.scatter(t["spearman_rho"], np.arange(len(t)), color="k", s=14, zorder=3, label="Spearman ρ")
ax.axvline(0, color="0.3", lw=0.8)
ax.set_xlabel("correlation with DLR")
ax.set_title(f"Single-predictor correlations, liquid-containing, LWP < {LWP_LOW_MAX_G_M2:.0f} g m$^{{-2}}$ (bars: Pearson r)")
ax.legend(fontsize=8, loc="lower right")
savefig(fig, "fig04_low_lwp_correlations")
plt.show()
''')
code(r'''
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
lwp_sub = [(-10, 5), (5, 15), (15, 30)]
sub_cols = [PAL["blue"], PAL["orange"], PAL["aqua"]]
panels = [("t_cloud_base_K", "cloud-base temperature [K]", T_BASE_EDGES_K),
          ("dT_base_skin_K", "cloud base − skin temperature [K]", np.arange(-20, 21, 2.5)),
          ("t_top_max_K", "highest layer-top temperature [K]", np.arange(210, 281, 5))]
for ax, (col, lab, edges) in zip(axes, panels):
    for (lo_, hi_), c in zip(lwp_sub, sub_cols):
        m = liq_low & (df["lwp_adj_g_m2"] >= lo_) & (df["lwp_adj_g_m2"] < hi_)
        b = binned(df.loc[m], col, "lwdn_w_m2", edges)
        if b.empty:
            continue
        ax.fill_between(b["x_mid"], b["q25"], b["q75"], color=c, alpha=0.15, linewidth=0)
        ax.plot(b["x_mid"], b["median"], color=c, lw=2, marker="o", ms=3.5, label=f"LWP {lo_}–{hi_} g m$^{{-2}}$")
    if col == "t_cloud_base_K":
        tt = np.linspace(edges[0], edges[-1], 50)
        ax.plot(tt, SIGMA * tt**4, color="0.2", lw=1, ls="--", label="σT$^4$ (opaque cloud at base T)")
    ax.set_xlabel(lab)
    ax.set_title(f"median and IQR by {lab.split(' [')[0]}", fontsize=9.5)
axes[0].set_ylabel("DLR [W m$^{-2}$]")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Low-LWP liquid-containing minutes: conditional DLR", y=1.02)
fig.tight_layout()
savefig(fig, "fig05_low_lwp_binned")
plt.show()
''')

md(r'''
### 6.1 Multiple regression: how much of the low-LWP spread is explained, and by what

Nested OLS models add predictors in the order of the hypotheses. Standardised coefficients (β)
are comparable across predictors; the R² ladder shows what each block adds. The blackbody term
σT_base⁴ is used instead of T_base so the coefficient is a dimensionless effective emissivity-like
weight. Day-block bootstrap intervals are given for the column-only model (all rows but the last);
the last row adds the skin temperature purely to show how much of the residual is tied to the
coupled surface state, which is not an explanation of DLR.
''')
code(r'''
# Column (atmospheric) predictors first; the surface skin temperature -- a response to DLR -- is
# added last and only as a reference for how much of the remaining variance the coupled surface
# state "explains" without being a cause.
LADDER = [
    ("LWP only", ["lwp_adj_g_m2"]),
    ("+ cloud-base σT⁴ (H1)", ["lwp_adj_g_m2", "sigT4_base"]),
    ("+ column vapour: PWV (H2, atmosphere)", ["lwp_adj_g_m2", "sigT4_base", "pwv_cm"]),
    ("+ ice: IWP proxy, ice fraction (H3)", ["lwp_adj_g_m2", "sigT4_base", "pwv_cm", "iwp_proxy_g_m2", "frc_ice"]),
    ("+ layering: n_layers, highest-top T (H4)", ["lwp_adj_g_m2", "sigT4_base", "pwv_cm", "iwp_proxy_g_m2", "frc_ice", "n_layers", "t_top_max_K"]),
    ("+ skin T [response, reference only]", ["lwp_adj_g_m2", "sigT4_base", "pwv_cm", "iwp_proxy_g_m2", "frc_ice", "n_layers", "t_top_max_K", "t_skin_irt_K"]),
]
rows = []
fits = {}
for name, cols in LADDER:
    f = ols(low, "lwdn_w_m2", cols)
    if f is None:
        continue
    fits[name] = f
    rows.append({"model": name, "n": f["n"], "R2": f["r2"], "RMSE_W_m2": f["rmse"]})
ladder = pd.DataFrame(rows).set_index("model")
ladder["ΔR2"] = ladder["R2"].diff().fillna(ladder["R2"])
display(ladder.round(3))

full_name, full_cols = LADDER[-2]  # the column-only model is the one to quote
if full_name in fits:
    f = fits[full_name]
    print("\nfull model, standardised coefficients with day-block bootstrap 95% CI:")
    for c in full_cols:
        v, lo_, hi_ = day_block_bootstrap(low, lambda t, c=c: (ols(t, "lwdn_w_m2", full_cols) or {"std_coef": {c: np.nan}})["std_coef"][c])
        print(f"  {c:20s} β = {fmt_ci(v, lo_, hi_)}    (raw {f['coef'][c]:+.4g} W m-2 per unit)")
    r2_ci = day_block_bootstrap(low, lambda t: (ols(t, "lwdn_w_m2", full_cols) or {"r2": np.nan})["r2"])
    print(f"  R2 = {fmt_ci(*r2_ci)}")
''')

# =============================================================================
md(r'''
## 7. Emission temperature, clear-sky baseline and effective emissivity

**Emission-temperature view (H1).** If a thin liquid cloud were opaque, DLR would equal σT_base⁴ plus
a small sub-cloud contribution. The panel shows T_eff = (DLR/σ)^{1/4} against T_base; the
fraction of minutes with T_eff within 3 K of T_base ("opaque") is then plotted against LWP.

**Clear-sky baseline.** Shupe & Intrieri (2004) computed clear-sky DLR with a radiative-transfer
model because Arctic clear periods are too few and unrepresentative to composite. Without a model
here, a Brunt-type fit on the observed clear minutes is used instead:
DLR_clear / σT_air⁴ = a + b·√e, with e the 2-m vapour pressure in hPa from MET temperature and RH
(Brunt 1932, *Q. J. R. Meteorol. Soc.* 58, 389–420; saturation vapour pressure after Bolton 1980).
The coefficients are fitted here, not taken from the literature, and the 2-m vapour pressure is
used rather than the MWR PWV because the MWR is missing in most clear minutes. Its RMSE on the
clear minutes is the baseline uncertainty. If the fitted b is *negative* (DLR/σT_air⁴ falling with
vapour pressure), the "clear" minutes are not radiatively clean: at the coldest temperatures ice
crystals below the lidar/radar detection limits (diamond dust, ice fog) raise DLR/σT⁴, and the
fitted emissivity is absorbing that covariance. The baseline is then an empirical reference for
the cloud-vs-clear contrast, not a physical clear-sky flux, and CRE values inherit its RMSE. The **cloud radiative effect** is then CRE_LW = DLR − DLR_clear(T_air, PWV), and
an **effective cloud emissivity** ε_eff = CRE_LW / (σT_base⁴ − DLR_clear) follows for cloud bases below
2 km when the denominator exceeds 10 W m⁻². The classic saturation curve ε = 1 − exp(−a·LWP) is fitted
(Stephens 1978 reports a ≈ 0.158 m² g⁻¹ for downward emissivity — moderately confident of that value;
the fitted a is what matters here). Note ε_eff absorbs everything the two-temperature picture
neglects: sub-cloud emission, ice, and the mismatch between the lidar base and the radiative base.
''')
code(r'''
# --- emission temperature view --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
ax = axes[0]
s = low.dropna(subset=["t_eff_K", "t_cloud_base_K"])
hb = ax.hexbin(s["t_cloud_base_K"], s["t_eff_K"], gridsize=60, extent=(235, 280, 215, 280), cmap=DENSITY_CMAP, bins="log", mincnt=1, linewidths=0.2)
ax.plot([235, 280], [235, 280], color="k", lw=1, ls="--", label="T$_{eff}$ = T$_{base}$")
ax.set_xlabel("cloud-base temperature [K]")
ax.set_ylabel("T$_{eff}$ = (DLR/σ)$^{1/4}$ [K]")
ax.set_title(f"low-LWP liquid-containing, n = {len(s):,}")
ax.legend(fontsize=8, loc="upper left")
fig.colorbar(hb, ax=ax, label="minutes (log)")

ax = axes[1]
lc = df.loc[liq & df["dT_eff_base_K"].notna() & (df["cloud_base_m"] < 2000)]
edges = np.array([-10, 0, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 250, 400])
cut = pd.cut(lc["lwp_adj_g_m2"], edges)
opaque = lc.groupby(cut, observed=True)["dT_eff_base_K"].apply(lambda v: (v.abs() <= 3).mean())
cnt = lc.groupby(cut, observed=True).size()
mid = np.array([c.mid for c in opaque.index])
ax.plot(mid[cnt >= MIN_N], 100 * opaque[cnt >= MIN_N], color=PAL["blue"], lw=2, marker="o", ms=4)
ax.set_xlabel("LWP (adjusted) [g m$^{-2}$]")
ax.set_ylabel("% minutes with |T$_{eff}$ − T$_{base}$| ≤ 3 K")
ax.set_title("how often the sky radiates like an opaque cloud at base T")
ax.set_xscale("symlog", linthresh=10)
fig.tight_layout()
savefig(fig, "fig06_emission_temperature")
plt.show()
''')
code(r'''
# --- clear-sky baseline fit --------------------------------------------------------------
def vapour_pressure_hpa(t_c, rh_pct):
    """2-m vapour pressure [hPa] from T [degC] and RH [%] w.r.t. water (Bolton 1980, MWR 108, 1046-1053)."""
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    return es * rh_pct / 100.0


df["e_2m_hpa"] = vapour_pressure_hpa(df["temp_2m_c"], df["rh_2m_pct"])
cs = df.loc[clear_sky].dropna(subset=["lwdn_w_m2", "t_air_2m_K", "e_2m_hpa"])
cs = cs[cs["e_2m_hpa"] > 0.05]
cs_ratio = cs["lwdn_w_m2"] / cs["sigT4_air"]
A = np.column_stack([np.ones(len(cs)), np.sqrt(cs["e_2m_hpa"].values)])
(a_cs, b_cs), *_ = np.linalg.lstsq(A, cs_ratio.values, rcond=None)
cs_pred = cs["sigT4_air"] * (a_cs + b_cs * np.sqrt(cs["e_2m_hpa"]))
cs_rmse = float(np.sqrt(((cs["lwdn_w_m2"] - cs_pred) ** 2).mean()))
cs_r2 = 1 - float(((cs["lwdn_w_m2"] - cs_pred) ** 2).sum() / ((cs["lwdn_w_m2"] - cs["lwdn_w_m2"].mean()) ** 2).sum())
print(f"clear-sky fit on n = {len(cs):,} minutes ({cs['day'].nunique()} days): "
      f"DLR_clear = σT_air^4 ({a_cs:.3f} + {b_cs:.4f} sqrt(e[hPa]));  R2 = {cs_r2:.2f}, RMSE = {cs_rmse:.1f} W m-2")
print(f"(clear minutes available: {int(clear_sky.sum()):,}; with MET T and RH: {len(cs):,})")
print(f"clear-sky DLR range: {cs['lwdn_w_m2'].quantile(0.05):.0f}–{cs['lwdn_w_m2'].quantile(0.95):.0f} W m-2 (5–95%)")

ok = df["t_air_2m_K"].notna() & (df["e_2m_hpa"] > 0.05)
df["dlr_clear_fit_w_m2"] = np.where(ok, df["sigT4_air"] * (a_cs + b_cs * np.sqrt(df["e_2m_hpa"].clip(lower=0.05))), np.nan)
df["cre_lw_w_m2"] = df["lwdn_w_m2"] - df["dlr_clear_fit_w_m2"]
den = df["sigT4_base"] - df["dlr_clear_fit_w_m2"]
df["eps_eff"] = np.where((den > 10) & (df["cloud_base_m"] < 2000), df["cre_lw_w_m2"] / den, np.nan)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
ax = axes[0]
ax.scatter(cs_pred, cs["lwdn_w_m2"], s=3, color=PAL["blue"], alpha=0.3, linewidths=0)
ax.plot([120, 300], [120, 300], color="k", lw=1, ls="--")
ax.set_xlabel("fitted clear-sky DLR [W m$^{-2}$]")
ax.set_ylabel("observed clear-sky DLR [W m$^{-2}$]")
ax.set_title(f"clear-sky baseline, RMSE {cs_rmse:.1f} W m$^{{-2}}$")

ax = axes[1]
le = df.loc[liq & df["eps_eff"].notna()]
edges = np.array([-10, 0, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 250, 400])
b = binned(le, "lwp_adj_g_m2", "eps_eff", edges)
ax.fill_between(b["x_mid"], b["q25"], b["q75"], color=PAL["blue"], alpha=0.15, linewidth=0)
ax.plot(b["x_mid"], b["median"], color=PAL["blue"], lw=2, marker="o", ms=4, label="median, IQR")
# fit eps = 1 - exp(-a LWP) on the pure single-layer liquid class only (mixed columns also
# carry ice emissivity, which would bias a toward saturation); least squares in eps space
pl = le[(le["layer_class"] == "1L liquid only") & (le["lwp_adj_g_m2"] > 0) & (le["eps_eff"] > -0.2) & (le["eps_eff"] < 1.2)]
if len(pl) >= MIN_N:
    def _sse(a):
        return float(((pl["eps_eff"] - (1 - np.exp(-a * pl["lwp_adj_g_m2"]))) ** 2).sum())
    a_grid = np.linspace(0.005, 0.5, 400)
    a_fit = float(a_grid[np.argmin([_sse(a) for a in a_grid])])
    bp = binned(pl, "lwp_adj_g_m2", "eps_eff", edges)
    ax.plot(bp["x_mid"], bp["median"], color=PAL["orange"], lw=2, marker="s", ms=4, label=f"1L liquid only, median (n = {len(pl):,})")
else:
    a_fit = np.nan
xx = np.linspace(0.5, 400, 200)
if np.isfinite(a_fit):
    ax.plot(xx, 1 - np.exp(-a_fit * xx), color="k", lw=1.2, ls="--", label=f"fit to 1L liquid only: a = {a_fit:.3f} m² g⁻¹")
ax.plot(xx, 1 - np.exp(-0.158 * xx), color=PAL["red"], lw=1, ls=":", label="a = 0.158 (Stephens 1978, downward)")
ax.set_xscale("symlog", linthresh=10)
ax.set_ylim(-0.2, 1.3)
ax.set_xlabel("LWP (adjusted) [g m$^{-2}$]")
ax.set_ylabel("ε$_{eff}$")
ax.set_title("effective emissivity vs LWP, liquid-containing")
ax.legend(fontsize=7.5, loc="lower right")

ax = axes[2]
for lc_name in LAYER_ORDER:
    m = le["layer_class"] == lc_name
    bb = binned(le.loc[m], "lwp_adj_g_m2", "eps_eff", edges)
    if bb.empty:
        continue
    ax.plot(bb["x_mid"], bb["median"], lw=1.8, marker="o", ms=3, color=LAYER_COLORS[lc_name], label=lc_name)
ax.set_xscale("symlog", linthresh=10)
ax.set_ylim(-0.2, 1.3)
ax.set_xlabel("LWP (adjusted) [g m$^{-2}$]")
ax.set_title("ε$_{eff}$ by liquid/ice layering (medians)")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
savefig(fig, "fig07_clear_sky_and_emissivity")
plt.show()
if np.isfinite(a_fit):
    print(f"fitted emissivity e-folding LWP (pure single-layer liquid) = {1/a_fit:.1f} g m-2 (a = {a_fit:.3f} m2/g)")
''')

# =============================================================================
md(r'''
## 8. Surface–cloud temperature contrast (H2)

The net longwave the surface feels is DLR − LWU; for the DLR itself, what matters is the cloud
temperature relative to the *air* whose emission the cloud replaces. The contrast to the skin
temperature is what decides whether a cloud warms the surface strongly (surface much colder than
cloud base under an inversion) or weakly. Panels: DLR anomaly relative to σT_air⁴ against the
cloud-base minus skin contrast, and the low-level inversion strength.
''')
code(r'''
l2 = df.loc[liq_low].copy()
l2["dlr_minus_sigT4air"] = l2["lwdn_w_m2"] - l2["sigT4_air"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
ax = axes[0]
s = l2.dropna(subset=["dT_base_skin_K", "dlr_minus_sigT4air"])
hb = ax.hexbin(s["dT_base_skin_K"], s["dlr_minus_sigT4air"], gridsize=60, extent=(-20, 20, -120, 40), cmap=DENSITY_CMAP, bins="log", mincnt=1, linewidths=0.2)
b = binned(s, "dT_base_skin_K", "dlr_minus_sigT4air", np.arange(-20, 21, 2.5))
ax.plot(b["x_mid"], b["median"], color="k", lw=1.6, marker="o", ms=3.5)
ax.axhline(0, color="0.3", lw=0.8); ax.axvline(0, color="0.3", lw=0.8)
ax.set_xlabel("cloud base − skin temperature [K]")
ax.set_ylabel("DLR − σT$_{air}^4$ [W m$^{-2}$]")
ax.set_title("low-LWP liquid-containing")
fig.colorbar(hb, ax=ax, label="minutes (log)")

ax = axes[1]
b = binned(s, "inversion_K", "lwdn_w_m2", np.arange(-2, 21, 2))
ax.fill_between(b["x_mid"], b["q25"], b["q75"], color=PAL["blue"], alpha=0.15, linewidth=0)
ax.plot(b["x_mid"], b["median"], color=PAL["blue"], lw=2, marker="o", ms=4)
ax.set_xlabel("low-level inversion strength, max T(z<2 km) − T(160 m) [K]")
ax.set_ylabel("DLR [W m$^{-2}$]")
ax.set_title("DLR vs inversion strength")

ax = axes[2]
r_tab = corr_table(l2, "lwdn_w_m2", {"cloud base − skin T": "dT_base_skin_K", "skin T": "t_skin_irt_K",
                                      "cloud-base T": "t_cloud_base_K", "inversion": "inversion_K"})
part_skin = residual_after(l2, "lwdn_w_m2", ["sigT4_base", "lwp_adj_g_m2"])
l2["resid_after_base_lwp"] = part_skin
b = binned(l2, "dT_base_skin_K", "resid_after_base_lwp", np.arange(-20, 21, 2.5))
ax.fill_between(b["x_mid"], b["q25"], b["q75"], color=PAL["orange"], alpha=0.15, linewidth=0)
ax.plot(b["x_mid"], b["median"], color=PAL["orange"], lw=2, marker="o", ms=4)
ax.axhline(0, color="0.3", lw=0.8)
ax.set_xlabel("cloud base − skin temperature [K]")
ax.set_ylabel("DLR residual after σT$_{base}^4$ and LWP [W m$^{-2}$]")
ax.set_title("does the surface contrast add information?")
fig.tight_layout()
savefig(fig, "fig08_surface_cloud_contrast")
plt.show()
display(r_tab.round(3))
''')

# =============================================================================
md(r'''
## 9. Ice (H3): IWP in mixed columns and the thin-to-opaque transition of ice-only clouds

Bertrand et al. (2025) found that most of the NSA winter cloud feedback comes from ice-only clouds
becoming more opaque (0.44 ± 0.06 W m⁻² K⁻¹). Here: (a) for ice-only columns, CRE_LW against the
Ze-based IWP proxy with the same saturation form; (b) for liquid-containing columns in fixed LWP
bins, the DLR residual (after σT_base⁴ and LWP) against IWP — the marginal effect of ice at fixed
liquid; (c) split of that ice into below-liquid (precipitating) and above-liquid.
''')
code(r'''
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
ax = axes[0]
io = df.loc[ice_only & df["cre_lw_w_m2"].notna() & (df["iwp_proxy_g_m2"] > 0)]
iwp_edges = np.logspace(-1, 3, 13)
b = binned(io, "iwp_proxy_g_m2", "cre_lw_w_m2", iwp_edges)
ax.fill_between(b["x_mid"], b["q25"], b["q75"], color=PAL["blue"], alpha=0.15, linewidth=0)
ax.plot(b["x_mid"], b["median"], color=PAL["blue"], lw=2, marker="o", ms=4, label=f"ice-only, n = {len(io):,}")
ax.set_xscale("log")
ax.set_xlabel("IWP proxy (0.1 Ze$^{0.63}$ over ice pixels) [g m$^{-2}$]")
ax.set_ylabel("CRE$_{LW}$ = DLR − DLR$_{clear}$ [W m$^{-2}$]")
ax.set_title("ice-only clouds: thin-to-opaque")
ax.legend(fontsize=8)
ct_ice = corr_table(io, "cre_lw_w_m2", {"log10 IWP": "iwp_proxy_g_m2", "cloud-base T": "t_cloud_base_K", "cloud-top T": "t_cloud_top_K", "ice fall speed": "fall_speed_ice_m_s", "n_layers": "n_layers"})

ax = axes[1]
lq = df.loc[liq].copy()
lq["resid"] = residual_after(lq, "lwdn_w_m2", ["sigT4_base", "lwp_adj_g_m2"])
for (lo_, hi_), c in zip([(-10, 5), (5, 15), (15, 30), (30, 60)], [PAL["blue"], PAL["orange"], PAL["aqua"], PAL["violet"]]):
    m = (lq["lwp_adj_g_m2"] >= lo_) & (lq["lwp_adj_g_m2"] < hi_) & (lq["iwp_proxy_g_m2"] > 0)
    b = binned(lq.loc[m], "iwp_proxy_g_m2", "resid", iwp_edges)
    if b.empty:
        continue
    ax.plot(b["x_mid"], b["median"], color=c, lw=2, marker="o", ms=3.5, label=f"LWP {lo_}–{hi_}")
ax.axhline(0, color="0.3", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("IWP proxy [g m$^{-2}$]")
ax.set_ylabel("DLR residual after σT$_{base}^4$, LWP [W m$^{-2}$]")
ax.set_title("marginal effect of ice at fixed LWP (liquid-containing)")
ax.legend(fontsize=8, title="g m$^{-2}$")

ax = axes[2]
m = lq["lwp_adj_g_m2"] < LWP_LOW_MAX_G_M2
for col, c, lab in [("iwp_below_liq_g_m2", PAL["aqua"], "ice below liquid base"), ("iwp_above_liq_g_m2", PAL["blue"], "ice above liquid top")]:
    b = binned(lq.loc[m & (lq[col] > 0)], col, "resid", iwp_edges)
    if b.empty:
        continue
    ax.plot(b["x_mid"], b["median"], color=c, lw=2, marker="o", ms=3.5, label=lab)
ax.axhline(0, color="0.3", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("IWP proxy of that part [g m$^{-2}$]")
ax.set_ylabel("DLR residual [W m$^{-2}$]")
ax.set_title(f"where the ice sits (LWP < {LWP_LOW_MAX_G_M2:.0f})")
ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, "fig09_ice_effects")
plt.show()
display(ct_ice.round(3))
''')

# =============================================================================
md(r'''
## 10. Vertical layering of liquid and ice, and multi-layer clouds (H4, H5)

DLR distributions by `layer_class` inside LWP bins. Because the classes also differ in cloud-base
temperature, the second panel compares them at matched T_base (medians of the residual after
σT_base⁴ and LWP). The third panel isolates the multi-layer effect: minutes with one ARSCL layer vs.
two or more, at matched LWP and cloud-base temperature, against the temperature of the highest top.
''')
code(r'''
lq = df.loc[liq].copy()
lq["resid"] = residual_after(lq, "lwdn_w_m2", ["sigT4_base", "lwp_adj_g_m2"])
lwp_cut = pd.cut(lq["lwp_adj_g_m2"], LWP_BIN_EDGES_G_M2)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
tab = lq.groupby([lwp_cut, "layer_class"], observed=True)["lwdn_w_m2"].agg(["median", "size"]).reset_index()
xpos = {iv: i for i, iv in enumerate(lwp_cut.cat.categories)}
width = 0.8 / len(LAYER_ORDER)
for j, lc_name in enumerate(LAYER_ORDER):
    t = tab[(tab["layer_class"] == lc_name) & (tab["size"] >= MIN_N)]
    ax.bar([xpos[iv] + (j - len(LAYER_ORDER) / 2 + 0.5) * width for iv in t["lwp_adj_g_m2"]], t["median"], width=width * 0.95, color=LAYER_COLORS[lc_name], label=lc_name)
ax.set_xticks(list(xpos.values()))
ax.set_xticklabels([f"{iv.left:.0f}–{iv.right:.0f}" for iv in xpos], fontsize=8)
ax.set_xlabel("LWP bin [g m$^{-2}$]")
ax.set_ylabel("median DLR [W m$^{-2}$]")
ax.set_ylim(150, 320)
ax.set_title("median DLR by layering and LWP (n ≥ %d)" % MIN_N)
ax.legend(fontsize=6.5, loc="upper left", ncol=2)

ax = axes[1]
tab2 = lq.groupby([lwp_cut, "layer_class"], observed=True)["resid"].agg(["median", "size"]).reset_index()
for j, lc_name in enumerate(LAYER_ORDER):
    t = tab2[(tab2["layer_class"] == lc_name) & (tab2["size"] >= MIN_N)]
    ax.bar([xpos[iv] + (j - len(LAYER_ORDER) / 2 + 0.5) * width for iv in t["lwp_adj_g_m2"]], t["median"], width=width * 0.95, color=LAYER_COLORS[lc_name])
ax.axhline(0, color="0.3", lw=0.8)
ax.set_xticks(list(xpos.values()))
ax.set_xticklabels([f"{iv.left:.0f}–{iv.right:.0f}" for iv in xpos], fontsize=8)
ax.set_xlabel("LWP bin [g m$^{-2}$]")
ax.set_ylabel("median DLR residual after σT$_{base}^4$, LWP [W m$^{-2}$]")
ax.set_title("same, at matched cloud-base temperature")

ax = axes[2]
mlow = lq["lwp_adj_g_m2"] < LWP_LOW_MAX_G_M2
for nl, c, lab in [(1, PAL["orange"], "1 layer"), (2, PAL["blue"], "≥ 2 layers")]:
    m = mlow & ((lq["n_layers"] == 1) if nl == 1 else (lq["n_layers"] >= 2))
    b = binned(lq.loc[m], "t_top_max_K", "resid", np.arange(205, 281, 5))
    if b.empty:
        continue
    ax.fill_between(b["x_mid"], b["q25"], b["q75"], color=c, alpha=0.12, linewidth=0)
    ax.plot(b["x_mid"], b["median"], color=c, lw=2, marker="o", ms=3.5, label=f"{lab}, n = {int(m.sum()):,}")
ax.axhline(0, color="0.3", lw=0.8)
ax.set_xlabel("temperature of the highest layer top [K]")
ax.set_ylabel("DLR residual [W m$^{-2}$]")
ax.set_title(f"multi-layer effect, LWP < {LWP_LOW_MAX_G_M2:.0f} g m$^{{-2}}$")
ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, "fig10_layering")
plt.show()

# matched-bin summary table: residual medians by class, low LWP only
summ = lq.loc[mlow].groupby("layer_class", observed=True).agg(n=("resid", "size"), n_days=("day", "nunique"), median_DLR=("lwdn_w_m2", "median"),
                                                              median_resid=("resid", "median"), median_Tbase=("t_cloud_base_K", "median"), median_LWP=("lwp_adj_g_m2", "median"))
display(summ.reindex([c for c in LAYER_ORDER if c in summ.index]).round(2))
multi_vs_single = {}
for lab, m in [("1 layer", mlow & (lq["n_layers"] == 1)), ("≥2 layers", mlow & (lq["n_layers"] >= 2))]:
    v, lo_, hi_ = day_block_bootstrap(lq.loc[m], lambda t: t["resid"].median())
    multi_vs_single[lab] = (v, lo_, hi_)
    print(f"median DLR residual, {lab:9s}: {fmt_ci(v, lo_, hi_, ' W m-2', 1)}  (n = {int(m.sum()):,})")
''')

# =============================================================================
md(r'''
## 11. Microphysical proxies (H6)

### 11.1 Droplet effective radius from KAZR reflectivity and MWR LWP (Frisch et al. 1995)

For a lognormal droplet spectrum with median radius r₀ and logarithmic width σ_x,
⟨rᵏ⟩ = r₀ᵏ exp(k²σ_x²/2), so (Frisch et al. 1995, Eqs. 3–6):

$$Z = 2^{6} N r_0^{6} e^{18\sigma_x^{2}}, \qquad
\mathrm{LWC} = \tfrac{4}{3}\pi\rho_w N r_0^{3} e^{4.5\sigma_x^{2}}, \qquad
r_e = \frac{\langle r^3\rangle}{\langle r^2\rangle} = r_0\, e^{2.5\sigma_x^{2}}$$

Eliminating N gives r₀ from the ratio Z/LWC alone, with σ_x the only assumption (0.35, as in
MICROBASE): $r_0 = \left[\frac{Z}{\mathrm{LWC}}\cdot\frac{4\pi\rho_w/3}{64\,e^{13.5\sigma_x^2}}\right]^{1/3}$.
The number concentration then follows as a check. LWC is taken as LWP / liquid depth (uniform); the
Frisch method distributes it as Z^{1/2}, which changes r_e by a few tens of percent at most for
shallow layers. Applicability: liquid-only, single-layer, non-precipitating columns, radar echo in
at least half of the liquid pixels, and max Ze < −17 dBZ to exclude drizzle. Hansen & Travis (1974,
Sect. 2.4) is the reference for r_eff and the effective variance v_eff = b as the two parameters that
control scattering; only r_eff is accessible here, v_eff is assumed through σ_x.

**How to read the residual-vs-r_e panel.** The physical expectation at fixed LWP is a *negative*
slope (more, smaller droplets → larger absorption optical depth per unit LWP) that vanishes once
the cloud is opaque. But r_e is inferred from Ze/LWC with LWC = LWP/depth, so MWR noise leaks in
with a definite sign: an LWP overestimate lowers the inferred r_e (∝ LWC^(−1/3)) *and* lowers the
residual (the LWP fit expects more DLR than the true LWP supports). LWP noise alone therefore
produces an apparent **positive** relation in the lowest LWP bins, strongest where the noise is a
large fraction of LWP. Only a negative slope that persists at LWP ≥ 15 g m⁻² and in the day-block
interval would count as evidence of a droplet-size effect.
''')
code(r'''
SIGMA_X = 0.35
RHO_W_G_MM3 = 1.0e-3   # water density [g mm-3]
ZE_DRIZZLE_DBZ = -17.0


def frisch_droplet_radius(ze_dbz, lwc_g_m3, sigma_x=SIGMA_X):
    """Lognormal-spectrum droplet r_e [um] and N [cm-3] from Ze [dBZ] and LWC [g m-3]."""
    z_mm6_m3 = 10.0 ** (np.asarray(ze_dbz, float) / 10.0)
    ratio = z_mm6_m3 / np.asarray(lwc_g_m3, float)            # mm^6 g^-1
    r0_mm = (ratio * (4.0 / 3.0) * np.pi * RHO_W_G_MM3 / (64.0 * np.exp(13.5 * sigma_x**2))) ** (1.0 / 3.0)
    re_um = r0_mm * np.exp(2.5 * sigma_x**2) * 1.0e3
    n_m3 = np.asarray(lwc_g_m3, float) / ((4.0 / 3.0) * np.pi * RHO_W_G_MM3 * r0_mm**3 * np.exp(4.5 * sigma_x**2))
    return re_um, n_m3 * 1.0e-6


# sanity check against a hand calculation: Ze = -25 dBZ, LWC = 0.1 g m-3 -> r_e ~ 9.9 um, N ~ 35 cm-3
print("check:", np.round(frisch_droplet_radius(-25.0, 0.1), 1))

fr_mask = (liq & (df["column_class"] == 3) & (df["n_layers"] == 1) & (df["n_liq_precip_px"] == 0)
           & (df["lwp_adj_g_m2"] > 5) & (df["liq_depth_m"] >= 60) & (df["n_liq_only_radar"] >= 2)
           & (df["n_liq_only_radar"] >= 0.5 * df["n_liq_only_px"]) & (df["ze_liq_max_dbz"] < ZE_DRIZZLE_DBZ))
fr = df.loc[fr_mask].copy()
fr["lwc_g_m3"] = fr["lwp_adj_g_m2"] / fr["liq_depth_m"]
fr["re_frisch_um"], fr["n_frisch_cm3"] = frisch_droplet_radius(fr["ze_liq_mean_dbz"], fr["lwc_g_m3"])
good = fr["re_frisch_um"].between(2, 30) & fr["n_frisch_cm3"].between(1, 1000)
print(f"Frisch-eligible minutes: {len(fr):,} on {fr['day'].nunique()} days; physically plausible (2-30 um, 1-1000 cm-3): {good.sum():,}")
fr = fr[good]
df["re_frisch_um"] = fr["re_frisch_um"]
df["n_frisch_cm3"] = fr["n_frisch_cm3"]
if len(fr) >= MIN_N:
    print(fr[["re_frisch_um", "n_frisch_cm3", "lwp_adj_g_m2", "liq_depth_m", "ze_liq_mean_dbz", "t_cloud_base_K"]].describe().round(2).T)
''')
code(r'''
if len(fr) >= MIN_N:
    fr["resid"] = residual_after(fr, "lwdn_w_m2", ["sigT4_base", "lwp_adj_g_m2"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    ax.hist(fr["re_frisch_um"], bins=np.arange(2, 30, 1), color=PAL["orange"], alpha=0.85)
    ax.set_xlabel("r$_e$ (Frisch, Ze + LWP) [µm]")
    ax.set_ylabel("minutes")
    ax.set_title(f"liquid-only non-drizzling, n = {len(fr):,}")
    ax = axes[1]
    sc = ax.scatter(fr["lwp_adj_g_m2"], fr["re_frisch_um"], c=fr["n_frisch_cm3"], s=4, cmap=DENSITY_CMAP, norm=mcolors.LogNorm(1, 1000), linewidths=0)
    ax.set_xscale("log")
    ax.set_xlabel("LWP (adjusted) [g m$^{-2}$]")
    ax.set_ylabel("r$_e$ [µm]")
    ax.set_title("r$_e$ vs LWP, colour = inferred N")
    fig.colorbar(sc, ax=ax, label="N [cm$^{-3}$]")
    ax = axes[2]
    for (lo_, hi_), c in zip([(5, 15), (15, 30), (30, 60), (60, 150)], [PAL["blue"], PAL["orange"], PAL["aqua"], PAL["violet"]]):
        m = (fr["lwp_adj_g_m2"] >= lo_) & (fr["lwp_adj_g_m2"] < hi_)
        b = binned(fr.loc[m], "re_frisch_um", "resid", np.arange(2, 26, 3), min_n=20)
        if b.empty:
            continue
        ax.plot(b["x_mid"], b["median"], color=c, lw=2, marker="o", ms=3.5, label=f"LWP {lo_}–{hi_}, n = {int(m.sum()):,}")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("r$_e$ (Frisch) [µm]")
    ax.set_ylabel("DLR residual after σT$_{base}^4$, LWP [W m$^{-2}$]")
    ax.set_title("droplet size at fixed LWP: expected negative slope")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    savefig(fig, "fig11_frisch_droplet_radius")
    plt.show()
    ct_fr = corr_table(fr, "resid", {"r_e (Frisch)": "re_frisch_um", "N (Frisch)": "n_frisch_cm3", "mean Ze in liquid": "ze_liq_mean_dbz", "liquid depth": "liq_depth_m"}, min_n=50)
    display(ct_fr.round(3))
    # partial slope of the residual on r_e within the low-LWP range, with a day-block CI
    frl = fr[fr["lwp_adj_g_m2"] < LWP_LOW_MAX_G_M2]
    if len(frl) >= MIN_N:
        v, lo_, hi_ = day_block_bootstrap(frl, lambda t: (ols(t, "resid", ["re_frisch_um"], min_n=30) or {"coef": {"re_frisch_um": np.nan}})["coef"]["re_frisch_um"])
        print(f"LWP < {LWP_LOW_MAX_G_M2:.0f}: d(DLR residual)/d r_e = {fmt_ci(v, lo_, hi_, ' W m-2 per um', 2)}  (n = {len(frl):,}, {frl['day'].nunique()} days)")
else:
    print("too few Frisch-eligible minutes for this slice")
''')

md(r'''
### 11.2 MICROBASE effective radii: what they are, and are not

On the sampled MICROBASE days, the LWC-weighted liquid r_e is plotted against the column-mean LWC,
and the IWC-weighted ice r_e against the ice-layer temperature. By construction (Wang et al. 2025,
Sect. 4.2.2 and 4.2.5) the first must lie on r_e ∝ LWC^{1/3} with N = 200 cm⁻³ and the second on
r_ei = (75.3 + 0.5895 T[°C])/2. If both hold, MICROBASE adds no size information beyond LWC
and temperature, and it must not be used as an independent test of H6. Its LWP is also compared
with the MWR LWP used elsewhere.
''')
code(r'''
mb_cols = [c for c in df.columns if c.startswith("mb_")]
mb = df.loc[df.get("mb_lwp_mb_g_m2", pd.Series(np.nan, index=df.index)).notna()] if mb_cols else df.iloc[0:0]
print(f"MICROBASE-sampled minutes in the table: {len(mb):,} on {mb['day'].nunique() if len(mb) else 0} days")
if len(mb) >= MIN_N:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    m = (mb["mb_n_liq_gates"] > 0) & (mb["mb_re_liq_lwcw_um"] > 0)
    lwc_mean = mb.loc[m, "mb_lwp_mb_g_m2"] / (mb.loc[m, "mb_n_liq_gates"] * 30.0)
    ax.scatter(lwc_mean, mb.loc[m, "mb_re_liq_lwcw_um"], s=3, color=PAL["orange"], alpha=0.4, linewidths=0)
    xx = np.logspace(-3, 0.5, 50)
    r0 = (3 * xx / (4 * np.pi * 1e-3 * 200e6 * np.exp(4.5 * 0.35**2))) ** (1 / 3) * 1e3 * 1.358  # MICROBASE formula: LWC g/m3, N=200 cm-3 = 2e8 m-3, rho 1e-3 g/mm3 -> r0 in mm... 
    # units: LWC [g m-3] / (rho_w [g mm-3] * N [m-3]) -> mm^3 ; sqrt3 -> mm ; *1e3 -> um
    ax.plot(xx, r0, color="k", lw=1.2, ls="--", label="MICROBASE formula (N = 200 cm⁻³, σ = 0.35)")
    ax.set_xscale("log"); ax.set_xlabel("column-mean LWC = LWP / (liquid gates × 30 m) [g m$^{-3}$]")
    ax.set_ylabel("MICROBASE liquid r$_e$ (LWC-weighted) [µm]")
    ax.set_title("liquid r$_e$ is LWC$^{1/3}$ by construction")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    m2 = (mb["mb_n_ice_gates"] > 0) & mb["t_ice_top_K"].notna()
    ax.scatter(mb.loc[m2, "t_ice_top_K"] - 273.15, mb.loc[m2, "mb_re_ice_iwcw_um"], s=3, color=PAL["blue"], alpha=0.4, linewidths=0)
    tt = np.linspace(-60, 0, 50)
    ax.plot(tt, (75.3 + 0.5895 * tt) / 2, color="k", lw=1.2, ls="--", label="Ivanova et al. (2001): (75.3 + 0.5895 T)/2")
    ax.set_xlabel("ice-top temperature [°C] (thermocldphase sonde)")
    ax.set_ylabel("MICROBASE ice r$_e$ (IWC-weighted) [µm]")
    ax.set_title("ice r$_e$ is temperature by construction")
    ax.legend(fontsize=7.5)
    ax = axes[2]
    ax.scatter(mb["lwp_adj_g_m2"], mb["mb_lwp_mb_g_m2"], s=3, color=PAL["aqua"], alpha=0.4, linewidths=0)
    ax.plot([0, 400], [0, 400], color="k", lw=1, ls="--")
    ax.set_xlim(-20, 400); ax.set_ylim(-20, 400)
    ax.set_xlabel("MWR LWP (adjusted) [g m$^{-2}$]"); ax.set_ylabel("MICROBASE integrated LWC [g m$^{-2}$]")
    ax.set_title("MICROBASE LWP vs MWR LWP")
    fig.tight_layout()
    savefig(fig, "fig12_microbase_reff")
    plt.show()
    if "re_frisch_um" in df and (mb["re_frisch_um"].notna() & (mb["mb_re_liq_lwcw_um"] > 0)).sum() > 30:
        both = mb[mb["re_frisch_um"].notna() & (mb["mb_re_liq_lwcw_um"] > 0)]
        print(f"minutes with both Frisch and MICROBASE liquid r_e: {len(both):,}; "
              f"median Frisch {both['re_frisch_um'].median():.1f} um vs MICROBASE {both['mb_re_liq_lwcw_um'].median():.1f} um; "
              f"Spearman rho = {stats.spearmanr(both['re_frisch_um'], both['mb_re_liq_lwcw_um'])[0]:.2f}")
else:
    print("no MICROBASE-sampled days in this slice yet (scripts/reduce_microbase.py fills them in)")
''')

md(r'''
### 11.3 Ice-particle proxies: Doppler fall speed, Z–V relation, depolarisation

The KAZR mean Doppler velocity in ice is the fall speed plus the vertical air motion; with a 5-min
median the air motion averages down and what remains scales with particle size and density
(pristine crystals 0.3–0.7 m s⁻¹, aggregates ≈ 1 m s⁻¹, rimed particles faster). Radar and lidar
LDR respond to habit and orientation. Panels: (a) fall speed vs Ze for ice-only columns (a Z–V
relation; its slope/offset is the crude size proxy); (b) DLR residual (after σT_base⁴ and IWP) vs
fall speed in ice-only columns — larger, faster particles carry less extinction per unit mass;
(c) DLR residual vs fall speed of ice *below the liquid* in precipitating supercooled clouds.
''')
code(r'''
io = df.loc[ice_only & (df["iwp_proxy_g_m2"] > 0) & df["fall_speed_ice_m_s"].notna()].copy()
io["log_iwp"] = np.log10(io["iwp_proxy_g_m2"].clip(lower=0.01))
io["resid"] = residual_after(io, "lwdn_w_m2", ["sigT4_base", "log_iwp"])
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
ax = axes[0]
s = io.dropna(subset=["ze_max_dbz"])
hb = ax.hexbin(s["ze_max_dbz"], s["fall_speed_ice_m_s"], gridsize=60, extent=(-45, 25, -1, 3), cmap=DENSITY_CMAP, bins="log", mincnt=1, linewidths=0.2)
b = binned(s, "ze_max_dbz", "fall_speed_ice_m_s", np.arange(-45, 26, 5))
ax.plot(b["x_mid"], b["median"], color="k", lw=1.6, marker="o", ms=3.5, label="median")
ax.axhline(0, color="0.3", lw=0.8)
ax.set_xlabel("max Ze in column [dBZ]"); ax.set_ylabel("ice fall speed (−MDV, 5-min median) [m s$^{-1}$]")
ax.set_title(f"Z–V relation, ice-only, n = {len(s):,}")
ax.legend(fontsize=8); fig.colorbar(hb, ax=ax, label="minutes (log)")
ax = axes[1]
b = binned(io, "fall_speed_ice_m_s", "resid", np.arange(-0.5, 2.6, 0.25))
ax.fill_between(b["x_mid"], b["q25"], b["q75"], color=PAL["blue"], alpha=0.15, linewidth=0)
ax.plot(b["x_mid"], b["median"], color=PAL["blue"], lw=2, marker="o", ms=4)
ax.axhline(0, color="0.3", lw=0.8)
ax.set_xlabel("ice fall speed [m s$^{-1}$]"); ax.set_ylabel("DLR residual after σT$_{base}^4$, log IWP [W m$^{-2}$]")
ax.set_title("ice-only: particle size at fixed IWP")
ax = axes[2]
pp = df.loc[df["precip_ice_from_liq"] & df["fall_speed_below_liq_m_s"].notna() & (df["lwp_adj_g_m2"] < 60)].copy()
pp["resid"] = residual_after(pp, "lwdn_w_m2", ["sigT4_base", "lwp_adj_g_m2"])
b = binned(pp, "fall_speed_below_liq_m_s", "resid", np.arange(-0.5, 2.6, 0.25))
ax.fill_between(b["x_mid"], b["q25"], b["q75"], color=PAL["aqua"], alpha=0.15, linewidth=0)
ax.plot(b["x_mid"], b["median"], color=PAL["aqua"], lw=2, marker="o", ms=4)
ax.axhline(0, color="0.3", lw=0.8)
ax.set_xlabel("fall speed of ice below the liquid base [m s$^{-1}$]"); ax.set_ylabel("DLR residual [W m$^{-2}$]")
ax.set_title(f"precipitating supercooled clouds (LWP < 60), n = {len(pp):,}")
fig.tight_layout()
savefig(fig, "fig13_ice_fall_speed")
plt.show()
ct_v = corr_table(io, "resid", {"ice fall speed": "fall_speed_ice_m_s", "spectral width (ice)": "sw_ice_mean_m_s", "radar LDR (ice)": "ldr_ice_mean_db", "MPL LDR (ice)": "mpl_ldr_ice_mean", "max Ze": "ze_max_dbz"})
display(ct_v.round(3))
''')

# =============================================================================
md(r'''
## 12. The MCT questions: glaciation and precipitation of supercooled liquid

**Glaciation analogue.** No experiment glaciates a cloud here, so the observational stand-in is a
matched comparison: liquid-containing columns and ice-only columns in the same cloud-base
temperature bin and the same total-water-path bin (LWP + Ze-based IWP for liquid-containing columns,
IWP for ice-only). The DLR difference, weighted by how often the liquid-containing population
occupies each bin, is the "if the same water were ice at the same temperature" estimate. Selection
caveats are real: ice-only columns at a given water path are different clouds (often deeper, colder
tops), and the IWP proxy carries a factor-2 uncertainty. Villanueva et al. (2022, Table 2) give the
LES reference: LW CRE 57 → 39 → 3 W m⁻² for 0 / 0.1 / 1 % h⁻¹ droplet freezing (Arctic stratocumulus,
top −15 °C).

**Precipitation.** Precipitating supercooled clouds (ice below the liquid base or snow pixels) are
compared with non-precipitating liquid-containing clouds: their LWP distributions (the depletion
MCT relies on), DLR at matched cloud-base temperature, and the LWP–DLR relation.
''')
code(r'''
twp_edges = np.array([1, 3, 10, 30, 100, 300, 1000, 3000])
lqm = df.loc[liq & df["t_cloud_base_K"].notna() & (df["twp_g_m2"] > 1)].copy()
iom = df.loc[ice_only & df["t_cloud_base_K"].notna() & (df["iwp_proxy_g_m2"] > 1)].copy()
lqm["tb"] = pd.cut(lqm["t_cloud_base_K"], T_BASE_EDGES_K)
iom["tb"] = pd.cut(iom["t_cloud_base_K"], T_BASE_EDGES_K)
lqm["wb"] = pd.cut(lqm["twp_g_m2"], twp_edges)
iom["wb"] = pd.cut(iom["iwp_proxy_g_m2"], twp_edges)
gl = lqm.groupby(["tb", "wb"], observed=True)["lwdn_w_m2"].agg(["median", "size"]).rename(columns={"median": "dlr_liq", "size": "n_liq"})
gi = iom.groupby(["tb", "wb"], observed=True)["lwdn_w_m2"].agg(["median", "size"]).rename(columns={"median": "dlr_ice", "size": "n_ice"})
match = gl.join(gi, how="inner")
match = match[(match["n_liq"] >= 50) & (match["n_ice"] >= 50)]
match["dDLR"] = match["dlr_ice"] - match["dlr_liq"]
if len(match):
    w = match["n_liq"] / match["n_liq"].sum()
    dglac = float((match["dDLR"] * w).sum())
    print(f"matched (T_base × water path) bins: {len(match)}; liquid minutes covered: {int(match['n_liq'].sum()):,} of {len(lqm):,}")
    print(f"frequency-weighted DLR change if liquid-containing columns were ice-only at the same T_base and water path: {dglac:+.1f} W m-2")
    display(match.round(1))
    fig, ax = plt.subplots(figsize=(7, 4))
    piv = match["dDLR"].unstack("wb")
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-60, vmax=60, aspect="auto", origin="lower")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{c.left:.0f}–{c.right:.0f}" for c in piv.columns], fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"{c.left:.0f}–{c.right:.0f}" for c in piv.index], fontsize=8)
    ax.set_xlabel("water path bin [g m$^{-2}$] (LWP + IWP for liquid-containing, IWP for ice-only)")
    ax.set_ylabel("cloud-base temperature bin [K]")
    ax.set_title("median DLR(ice-only) − DLR(liquid-containing), matched bins")
    for (i, j), v in np.ndenumerate(piv.values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=7.5, color="k")
    fig.colorbar(im, ax=ax, label="ΔDLR [W m$^{-2}$]")
    ax.grid(False)
    savefig(fig, "fig14_glaciation_matched_bins")
    plt.show()
else:
    dglac = np.nan
    print("not enough matched bins in this slice")
''')
code(r'''
pr = df.loc[df["precip_ice_from_liq"] & df["lwdn_w_m2"].notna()]
npr = df.loc[df["nonprecip_liq"] & df["lwdn_w_m2"].notna()]
print(f"precipitating supercooled (ice below liquid / snow): {len(pr):,} minutes; non-precipitating liquid-containing: {len(npr):,}")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
ax = axes[0]
edges = np.arange(-20, 300, 5)
for s, c, lab in [(npr, PAL["orange"], "non-precipitating"), (pr, PAL["aqua"], "precipitating ice from liquid")]:
    v = s["lwp_adj_g_m2"].dropna()
    ax.hist(v, bins=edges, histtype="step", lw=2, color=c, density=True, cumulative=True, label=f"{lab} (median {v.median():.0f})")
ax.set_xlabel("LWP (adjusted) [g m$^{-2}$]"); ax.set_ylabel("cumulative fraction"); ax.set_title("LWP depletion by precipitation?")
ax.legend(fontsize=8, loc="lower right")
ax = axes[1]
for s, c, lab in [(npr, PAL["orange"], "non-precipitating"), (pr, PAL["aqua"], "precipitating")]:
    b = binned(s, "t_cloud_base_K", "lwdn_w_m2", T_BASE_EDGES_K)
    ax.fill_between(b["x_mid"], b["q25"], b["q75"], color=c, alpha=0.12, linewidth=0)
    ax.plot(b["x_mid"], b["median"], color=c, lw=2, marker="o", ms=3.5, label=lab)
tt = np.linspace(235, 280, 50); ax.plot(tt, SIGMA * tt**4, color="0.2", lw=1, ls="--", label="σT$_{base}^4$")
ax.set_xlabel("cloud-base temperature [K]"); ax.set_ylabel("DLR [W m$^{-2}$]"); ax.set_title("DLR at matched cloud-base T")
ax.legend(fontsize=8)
ax = axes[2]
for s, c, lab in [(npr, PAL["orange"], "non-precipitating"), (pr, PAL["aqua"], "precipitating")]:
    b = binned(s, "lwp_adj_g_m2", "lwdn_w_m2", LWP_BIN_EDGES_G_M2)
    ax.plot(b["x_mid"], b["median"], color=c, lw=2, marker="o", ms=3.5, label=lab)
ax.set_xscale("symlog", linthresh=10)
ax.set_xlabel("LWP (adjusted) [g m$^{-2}$]"); ax.set_ylabel("median DLR [W m$^{-2}$]"); ax.set_title("DLR–LWP relation by precipitation state")
ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, "fig15_precipitating_supercooled")
plt.show()
# matched T_base comparison with day-block CI on the median difference
both = pd.concat([pr.assign(grp="pr"), npr.assign(grp="npr")])
both["tb"] = pd.cut(both["t_cloud_base_K"], T_BASE_EDGES_K)
def _dlr_gap(t):
    g = t.groupby(["tb", "grp"], observed=True)["lwdn_w_m2"].median().unstack("grp")
    n = t.groupby(["tb", "grp"], observed=True).size().unstack("grp")
    g = g[(n["pr"] >= 30) & (n["npr"] >= 30)]
    if g.empty:
        return np.nan
    w = n.loc[g.index, "pr"]
    return float(((g["pr"] - g["npr"]) * w).sum() / w.sum())
v, lo_, hi_ = day_block_bootstrap(both, _dlr_gap, n_boot=150)
dprecip = v
print(f"DLR(precipitating) − DLR(non-precipitating) at matched cloud-base T, weighted: {fmt_ci(v, lo_, hi_, ' W m-2', 1)}")
v2, lo2, hi2 = day_block_bootstrap(both, lambda t: float(t.loc[t.grp == 'pr', 'lwp_adj_g_m2'].median() - t.loc[t.grp == 'npr', 'lwp_adj_g_m2'].median()), n_boot=150)
print(f"median LWP(precipitating) − LWP(non-precipitating): {fmt_ci(v2, lo2, hi2, ' g m-2', 1)}")
''')

# =============================================================================
md(r'''
## 13. Summary of the numbers this run produced
''')
code(r'''
summary = {
    "seasons": ", ".join(sorted(df["season"].unique())),
    "minutes with DLR + phase": int(base.sum()),
    "liquid-containing minutes": int(liq.sum()),
    f"liquid-containing, LWP < {LWP_LOW_MAX_G_M2:.0f}": int(liq_low.sum()),
    "DLR–LWP slope, overcast-like liquid [W m-2 per g m-2]": round(fit_all["coef"]["lwp_adj_g_m2"], 3),
    "DLR–LWP r2 (overcast-like liquid)": round(fit_all["r2"], 3),
    "low-LWP DLR std [W m-2]": round(float(low["lwdn_w_m2"].std()), 1),
    "low-LWP: top single predictor": f"{ct_low.index[0]} (r = {ct_low['pearson_r'].iloc[0]:+.2f})",
    "low-LWP: r with cloud-base T": round(float(ct_low.loc["cloud-base T (lowest hydrometeor)", "pearson_r"]), 3) if "cloud-base T (lowest hydrometeor)" in ct_low.index else np.nan,
    "low-LWP: r with cloud base − skin T": round(float(ct_low.loc["cloud base − skin T", "pearson_r"]), 3) if "cloud base − skin T" in ct_low.index else np.nan,
    "low-LWP: r with IWP proxy": round(float(ct_low.loc["IWP proxy (whole column)", "pearson_r"]), 3) if "IWP proxy (whole column)" in ct_low.index else np.nan,
    "low-LWP: r with n_layers": round(float(ct_low.loc["number of ARSCL layers", "pearson_r"]), 3) if "number of ARSCL layers" in ct_low.index else np.nan,
    "R2 ladder": {k: round(v, 3) for k, v in ladder["R2"].items()},
    "clear-sky fit RMSE [W m-2]": round(cs_rmse, 1),
    "emissivity e-folding LWP [g m-2]": round(float(1 / a_fit), 1) if np.isfinite(a_fit) else np.nan,
    "multi-layer vs single, median residual [W m-2]": {k: round(v[0], 1) for k, v in multi_vs_single.items()},
    "Frisch r_e minutes": int(df["re_frisch_um"].notna().sum()) if "re_frisch_um" in df else 0,
    "Frisch r_e median [um]": round(float(df["re_frisch_um"].median()), 1) if "re_frisch_um" in df and df["re_frisch_um"].notna().any() else np.nan,
    "glaciation analogue dDLR [W m-2]": round(dglac, 1) if np.isfinite(dglac) else np.nan,
    "precipitating − non-precipitating DLR at matched T_base [W m-2]": round(dprecip, 1) if np.isfinite(dprecip) else np.nan,
}
for k, v in summary.items():
    print(f"{k:60s} {v}")
import json
(FIG_DIR / "summary_numbers.json").write_text(json.dumps(summary, indent=2, default=str))
''')

md(r'''
### 13.1 Results of the 2023/24 + 2024/25 run (recorded 2026-09-18; re-runs overwrite the cell above, not this text)

* **The DLR spread at low LWP is emission temperature, not microphysics.** For liquid-containing
  columns with LWP < 30 g m⁻² (155,449 minutes) DLR spans 174–304 W m⁻² (5–95 %). LWP alone explains
  9 % of the variance; adding σT⁴ at the lidar/radar cloud base raises R² to 0.77 and PWV to 0.83.
  Ice (IWP proxy, ice fraction) adds 0.004 and layering (layer count, highest-top temperature) adds
  nothing at the population level. The skin temperature adds a further 0.06 but is a response to DLR.
* **Emissivity saturates fast.** For pure single-layer liquid clouds ε_eff = 1 − exp(−0.118·LWP),
  an e-folding LWP of 8.5 g m⁻²; the sky radiates within 3 K of the cloud-base blackbody in 60 % of
  minutes at LWP ≈ 15 g m⁻² and 90 % at LWP ≈ 100 g m⁻². Above ~20 g m⁻² LWP no longer matters.
* **The ERA5 relation is not the observed one.** Overcast-like liquid-containing minutes give
  DLR = 0.104·LWP + 254 (r² = 0.16) against ERA5's 0.443·LWP + 218 (r² = 0.45) at the same site: the
  observed curve saturates by ~20 g m⁻² whereas ERA5 needs ~100 g m⁻² to reach the same DLR.
* **Ice matters where liquid is thin.** In ice-only columns CRE_LW rises from ≈ 0 to ≈ 60 W m⁻²
  between IWP proxies of 1 and 1000 g m⁻² (thin-to-opaque; r = 0.61 with log IWP). In liquid-containing
  columns the ice only adds DLR when LWP < 5 g m⁻² and the IWP proxy exceeds ~30 g m⁻².
* **Layering and multi-layer clouds are second order.** At matched cloud-base temperature and LWP,
  the class medians differ by ≤ 15 W m⁻²; two or more layers add about 3 W m⁻² over one layer.
* **No droplet-size signal is detectable.** The Frisch radius for 11,441 liquid-only non-drizzling
  minutes has median 8.7 µm (inferred N median 48 cm⁻³, plausible for the Arctic); the residual slope
  against r_e at LWP < 30 is 0.08 [−0.61, 0.82] W m⁻² µm⁻¹, and the positive tendency in the 5–15 g m⁻²
  bin is what LWP noise produces. MICROBASE r_e is LWC^(1/3) and temperature by construction and was not
  used as evidence.
* **Ice fall speed and depolarisation carry no clean DLR signal** beyond what Ze/IWP already gives
  (|r| ≤ 0.25 with the residual; no monotonic relation).
* **MCT analogues.** Replacing liquid-containing columns by ice-only columns at the same cloud-base
  temperature and water path lowers the median DLR by 26.5 W m⁻² (frequency-weighted over 50 matched
  bins; 20–46 W m⁻² for bases at 250–275 K and 10–300 g m⁻²), the observational counterpart of the
  −54 W m⁻² full-glaciation LES result of Villanueva et al. (2022, Table 2). Naturally precipitating
  supercooled clouds (57 % of liquid-containing minutes) have only 7.8 [1.5, 12.9] g m⁻² less LWP and
  1.4 [−0.2, 3.2] W m⁻² less DLR than non-precipitating ones at matched cloud-base temperature: at
  NSA, precipitation from supercooled layers does not by itself thin them enough to matter.
* Data quality: monthly MWR clear-sky offsets were −4 to +7.5 g m⁻² (noise floor 5 g m⁻²); 17 days of
  2024/25 are absent from the ARM archive (5–11 Sep 2024, 12–21 Apr 2025); the clear-sky baseline has
  RMSE 16 W m⁻² and a negative vapour coefficient (undetected ice crystals in "clear" minutes).
''')

md(r'''
## 14. Reading the results, caveats, and what to do next

**How to read the sections above** (the text here is generic on purpose; the numbers are in the cells and
in `figures/dlr_microphysics/summary_numbers.json`):

* If Section 6 ranks the cloud-base temperature first and Section 7 shows T_eff hugging T_base only
  above some LWP, then the low-LWP spread is mostly *emission temperature plus partial emissivity* —
  H1 — and microphysics can only act through the emissivity–LWP curve (Section 7, e-folding LWP)
  and through the ice contributions (Section 9).
* The R² ladder (6.1) says how much each hypothesis block adds *given the earlier ones*. A large
  jump at the ice block means low-LWP columns are radiatively ice clouds with a little liquid; a
  large jump at layering means upper clouds matter.
* Section 11.1 is the only place where a droplet-size signal can appear. A negative residual slope
  against r_e at fixed LWP is the Twomey-like longwave effect (more, smaller droplets → larger
  absorption optical depth per unit LWP). Because r_e here is inferred from Ze/LWC, a spurious
  correlation through LWC errors is possible: check that the sign persists across LWP bins and that
  N is plausible.
* Section 12 gives the observational envelope for MCT: the matched-bin ΔDLR is an upper-bound
  analogue of full glaciation, and the precipitating-vs-non-precipitating comparison shows whether
  natural precipitation already depletes LWP and DLR at NSA.

**Caveats that apply throughout**

1. Minutes are not independent; the day-block bootstrap intervals and n_eff are the honest
   uncertainties. Two seasons hold of order 100 independent weather situations.
2. MWR LWP: ±10–15 g m⁻² noise after the clear-sky correction; a month with an offset flag should be
   treated with suspicion, and negative adjusted LWP is retained (not clipped) so that the noise is
   symmetric.
3. The phase mask is a rule-based classification; lidar attenuation above ~ the first liquid layer
   means ice above a thick liquid cloud is radar-only and "ice above liquid" is under-counted.
4. The IWP proxy uses a single Z–IWC relation (SHEBA winter); relative comparisons are robust,
   absolute values are not.
5. The clear-sky baseline is a fit to observed clear minutes, not a radiative-transfer model
   (Shupe & Intrieri 2004 explain why that matters); CRE and ε_eff inherit its RMSE plus the
   representativeness bias of clear periods.
6. MICROBASE effective radii are parameterisations (11.2). Any "microphysics" signal built on them
   is a signal of LWC or temperature.

**Next steps that would turn proxies into measurements**

* Run an AERI-based retrieval (MIXCRA, Turner 2005) on `nsaaerich1C1.b1` for the thin-cloud
  minutes: liquid and ice r_e from the 8–13 µm window are exactly the sizes the longwave responds to.
* Order the Shupe–Turner product (2004–2019) and repeat Sections 9–12 with phase-aware LWC/IWC.
* Radiative closure with RRTM-LW driven by the column features (Bertrand et al. 2025 Methods) to
  replace the empirical clear-sky baseline and to attribute DLR to LWP, IWP, T_base and r_e formally.
* KAZR Doppler spectra (`nsakazrspeccmaskgecopolC1.a1`) for the liquid/drizzle/ice separation
  inside mixed pixels, which the 30-s moments cannot give.
''')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true", help="run the notebook with nbconvert after writing it")
    p.add_argument("--timeout", type=int, default=3600)
    a = p.parse_args()
    nb = new_notebook(cells=CELLS, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    nbformat.write(nb, NB_PATH)
    print(f"wrote {NB_PATH} ({len(CELLS)} cells)")
    if a.execute:
        env = dict(os.environ)
        env.setdefault("JUPYTER_CONFIG_DIR", str(REPO / ".jupyter_empty"))
        Path(env["JUPYTER_CONFIG_DIR"]).mkdir(exist_ok=True)
        cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
               f"--ExecutePreprocessor.timeout={a.timeout}", str(NB_PATH)]
        print(" ".join(cmd))
        return subprocess.call(cmd, cwd=REPO, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
