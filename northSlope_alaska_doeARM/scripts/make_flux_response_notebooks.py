#!/usr/bin/env python3
"""Generate (and optionally execute) the two surface-flux-response notebooks.

    seb_flux_response_to_DLR.ipynb           responders regressed on DLR
    seb_flux_response_to_DLR_and_SW.ipynb    responders regressed on DLR + SW_net

They are the observational twins of
ERA5/surface_energy_budget/turbulent_flux_response_to_DLR.ipynb and
..._to_DLR_and_SW.ipynb, built on arm_nsa/flux_response.py. The notebooks hold
no analysis of their own -- every number comes from the module -- so this
generator is the single place their structure is defined. Re-run it after
changing the module's API; re-execute the notebooks after rebuilding the
products.

    python scripts/make_flux_response_notebooks.py            # write only
    python scripts/make_flux_response_notebooks.py --execute  # write + run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parent.parent
KERNEL = {"display_name": "base", "language": "python", "name": "python3"}

# ----------------------------------------------------------------------------
# Shared cells
# ----------------------------------------------------------------------------

SETUP_CODE = """%load_ext autoreload
%autoreload 2
%matplotlib inline

import sys, warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path.cwd() if (Path.cwd() / "arm_nsa").is_dir() else Path.cwd().parent
sys.path.insert(0, str(REPO))
from arm_nsa import flux_response as fr

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 100          # on-screen; saved PNGs use fr.FIG_DPI

SEASON = "2025-10-01_2026-03-31"
PRODUCTS = REPO / "data" / "processed"
SAVE_DIR = REPO / "figures" / "seb_flux_response"   # None to only display
N_BOOT = 1000                                        # bootstrap draws (2000 in the ERA5 notebooks)
"""

LOAD_CODE = """# The 10-min product is the Sledd et al. (2025) cadence (their observations are
# 10-min means); the hourly product is the ERA5 cadence. Both were built by
# scripts/build_nsa_seb_hourly.py from the same 1-min streams.
OBS = fr.load_product(PRODUCTS / f"nsa_seb_10min_{SEASON}.nc")
OBS_H = fr.load_product(PRODUCTS / f"nsa_seb_hourly_{SEASON}.nc")
MASKS = fr.population_masks(OBS)
MASKS_H = fr.population_masks(OBS_H)

# ERA5 at the grid cell nearest NSA C1 (71.25 N, 156.5 W; a coastal cell, ERA5
# land fraction 0.62), on the same Eq. (1) terms via the ERA5 seb_terms module.
ERA5 = fr.load_era5_cell("2025-10-01", "2026-03-31")
ERA5_MASKS = fr.population_masks(ERA5) if ERA5 is not None else None

print(f"10-min samples: {OBS.sizes['time']:,}   hourly: {OBS_H.sizes['time']:,}   "
      f"ERA5 hours: {ERA5.sizes['time'] if ERA5 is not None else 'n/a'}")
print("populations (10-min):", {k: int(v.sum()) for k, v in MASKS.items()})
print()
print(fr.variable_check_table(OBS))
"""

VARIABLE_CHECK_MD = """## Do we have the right variables? Sledd et al. (2025) term by term

Sledd et al. define the budget (their Eq. 1, turbulent fluxes positive **away**
from the surface, $G$ positive toward it)

$$LWD - LWU + SWD - SWU - SWT - SH - LH + G = M$$

and evaluate the response of every term on the right-hand side of

$$\\frac{\\partial(LWD + SWN)}{\\partial(LWD + SWN)} = 1 =
\\frac{\\partial(LWU + SH + LH + M - SWT - G)}{\\partial(LWD + SWN)} \\qquad \\text{(their Eq. 9)}$$

with a least-squares slope per calendar month on 10-min data. What they had at
MOSAiC, and what NSA C1 supplies for the same term:

| term | MOSAiC (Sledd et al. Sect. 2.1) | NSA C1 in this notebook | status |
|---|---|---|---|
| LWD, LWU | pyrgeometers at 1 m (down) and 3 m (up); ±2.6 / ±1 W m⁻² | QCRAD 1-min pyrgeometers, LWU at 10 m; LWD is the two-pyrgeometer best estimate because pyrgeometer 1 failed for 44 days of Dec–Feb (README) | **measured** |
| SWD, SWU | pyranometers, ±3–4.5 W m⁻² | QCRAD 1-min | **measured** |
| $T_{skin}$ | inverted from LWU, ε = 0.985 | GNDIRT brightness temperature corrected with ε = 0.985 and the reflected LWD; LWU inversion as fallback and as a cross-check (they agree to −0.5 ± 0.4 K outside the calm-clear frost episodes) | **measured** (two routes) |
| SH, LH | eddy covariance (20-Hz sonic + gas analyser); SH ±4.8 W m⁻², LH ±50% | **no eddy-covariance system exists at Barrow** — bulk-aerodynamic parameterization (MOST, Grachev et al. 2007 stable functions, Andreas 1987 scalar roughness, $z_0$ = 3×10⁻⁴ m) from measured $T_{skin}$, $T_{2m}$, RH, $U_{10}$, $p$ | **parameterized** |
| G | conduction + storage in a 6-cm slab from ice-mass-balance buoys | **no soil/snow instrumentation at Barrow** — thermal-inertia estimate from the $T_{skin}$ history (Wang & Bras 1999), and the Eq. (5) residual | **estimated** |
| SWT | Beer's law through a 6-cm snow/ice slab (their Eq. 6) | zero: the snowpack is deeper than 6 cm all season and the tundra below is opaque, so penetrating shortwave is absorbed in the pack and is part of $G$ | assumed 0 |
| M | residual | zero (frozen surface, Oct–Mar) | 0 |
| $T_{2m}$, $U$ | tower | MET 1-min (2-m T/RH, 10-m wind) | **measured** |
| cadence | 10 min | 10 min (this product) and 1 h (for ERA5) | matched |

**So the radiative half of the framework — the forcer and the largest responder
(LWU), which carries half of every anomaly — is on the same observational
footing as MOSAiC. The turbulent and subsurface responders are not.** At
MOSAiC they were independent measurements and their sum closing to one
(Sledd's "total measured response", 0.94–1.08 in winter) was a *finding*. Here
SH and LH are functions of the measured skin-to-air temperature difference and
wind, and $G$ is a function of the measured skin temperature history: all
three responders are downstream of $T_{skin}$, so their sum closing to one is a
weaker statement — it says the parameterizations are mutually consistent with
the measured radiation, not that they are right. What *is* independently
testable is the comparison of the two routes to the subsurface response,
$dNA/dF$ (measured radiation minus parameterized turbulence) and $d(-G_{TI})/dF$
(thermal inertia), which is the analogue of Sledd's NA-versus-$-G$ comparison.
"""

REGRESSION_MD = """## What a regression is doing here

The ERA5 notebook of the same name carries a full primer; the short form:

- **A slope is an estimated derivative.** $b = \\mathrm{cov}(x, y)/\\mathrm{var}(x)$
  is $dy/dx$ averaged over the range of $x$ the data cover. Where the relation
  bends, it is the best single-number summary over that range.
- **The response goes on $y$.** $\\mathrm{slope}(y|x)\\cdot\\mathrm{slope}(x|y) = r^2$,
  so inverting the wrong fit overstates a derivative by $1/r^2$. Every panel
  below fits the responder on the forcer.
- **$r^2$ is a description of the scatter, not a verdict on the slope.** A slope
  can be well determined through a lot of scatter. Sledd et al. scale their
  markers by $r^2$ for exactly this reason, and so do the monthly figures here.
- **Every number is a covariance across the season's weather**, not a controlled
  experiment. High-DLR periods are cloudy, warm, windy periods. The plain slope
  (Sledd's estimator) is the headline; the multiple regression that holds
  $T_{2m}$ fixed is the other end of the confounder-versus-mediator bracket and
  is drawn beside it.
- **Confidence intervals are moving-block bootstrap intervals** (7-day blocks,
  the units the record supplies independently), never the textbook standard
  error, which assumes independent samples and is too small by one to two
  orders of magnitude on these data. One season holds 26 blocks, so the
  intervals are wide — and honest.
"""


def md(text: str):
    return new_markdown_cell(text)


def code(text: str):
    return new_code_cell(text)


# ----------------------------------------------------------------------------
# Notebook 1: DLR forcer
# ----------------------------------------------------------------------------


def notebook_dlr() -> nbformat.NotebookNode:
    cells = [
        md(
            """# Surface flux response to a downwelling-longwave forcing — NSA C1 observations

The observational twin of `ERA5/surface_energy_budget/turbulent_flux_response_to_DLR.ipynb`:
the same question — *if the downwelling longwave at the Arctic surface changes by
1 W m⁻², where does that watt go?* — answered from the ARM instruments at
Utqiaġvik (NSA C1) for the 2025/26 cold season instead of from ERA5. The
framework is Sledd et al. (2025), *JGR Atmospheres*, 130, e2024JD042578, after
Miller et al. (2017): regress each surface term on the radiative forcing and
read the slope as the fraction of the forcing that term disposes of.

## Sign convention

Everything in the observational product is already in the **Sledd** convention:
the turbulent fluxes `sh_up`, `lh_up` are **positive upward** (energy leaving the
surface), $G$ is positive **toward** the surface and its responder is $-G$, and
the radiative net fluxes are positive downward. Nothing here needs the sign
gymnastics of the ERA5 notebook (whose fluxes are stored positive downward); a
positive fraction always means *energy leaving the surface per W m⁻² arriving*.

Under the winter inversion the surface is usually colder than the 2-m air, so
`sh_up` is **negative** most of the time (heat flows down into the snow). A DLR
increase warms the skin, shrinks the skin-to-air deficit, and makes `sh_up`
*less negative*: the downward flux weakens. That is a positive $f_{SH}$ — the
turbulent term damps the forcing — even though the flux itself points down. The
sign of the flux is not the sign of the response.

## The important caveat

Nothing here is a controlled forcing experiment. Every slope is a covariance
across the natural synoptic variability of one season: high-DLR periods are
cloudy, and cloudy periods differ in wind, air mass and stability. Two
estimators are reported throughout — the plain regression (Sledd's) and the
regression holding $T_{2m}$ fixed — and they bracket the causal answer from
opposite sides.

## Contents

| section | figure |
|---|---|
| variables | which Sledd terms are measured, parameterized, or estimated here |
| 1 | every responder against DLR, all sky (Sledd Fig. 1 layout) |
| 2–3 | sensible and latent heat against DLR, by population |
| 4 | LWU, NA and $-G$ against DLR |
| 5 | the partition by population, with bootstrap intervals; the ledger |
| 6 | thermal freedom: $dT_{skin}/d(DLR)$ and $d(T_{skin}-T_{2m})/d(DLR)$ |
| 7 | the control ladder |
| 8 | month by month (Sledd Fig. 2a layout, DLR forcer) with the ERA5 cell |
| 9 | sensitivity to the bulk-flux free parameters |
| 10 | 10-min against hourly |
| 11 | DLR against LWP |
| 12 | the observed partition beside ERA5's at the same cell |

All analysis lives in `arm_nsa/flux_response.py`; this notebook holds none of
its own (autoreload is on).
"""
        ),
        md(REGRESSION_MD),
        md("## Setup"),
        code(SETUP_CODE),
        md("## Load the two products and the ERA5 cell"),
        code(LOAD_CODE),
        md(VARIABLE_CHECK_MD),
        md(
            """## Verify the algebra, then the numbers

`self_check` asserts that a variable regressed on itself gives exactly 1, that the
remainder of the partition equals the direct regression of NA (linearity), that
holding a variable fixed drives its own slope to zero, and that the two fit
directions multiply to $r^2$. `print_report` is the partition for every
population under both estimators — the numbers the figures draw.
"""
        ),
        code(
            """fr.self_check(OBS)
fr.print_report(OBS, MASKS, forcer="lwd")
"""
        ),
        md(
            """### 1. Every responder against DLR — all sky

The Sledd et al. (2025) Fig. 1 layout on the Barrow record: six density-coloured
scatters of the 10-min data, the unweighted least-squares line in black, its
slope and $r^2$ in the box. The slope of each panel *is* that term's response.
LWU is the term that behaves (a near-straight line, high $r^2$); SH and LH are
the bulk parameterization's responses; NA is the atmospheric-side remainder and
$-G_{TI}$ the thermal-inertia estimate of the same subsurface flux; SW$_{net}$ is
the shortwave co-variation that the DLR-only forcer leaves on the responder side.
"""
        ),
        code(
            """fr.fig_all_responders(OBS, MASKS["all"], forcer="lwd", population="all", out_dir=SAVE_DIR);"""
        ),
        md(
            """### 2. Sensible heat flux against DLR, by population

Populations: all sky; cloudy (ARSCL cloud fraction ≥ 0.95 within the averaging
window); clear (≤ 0.05). The liquid-bearing overcast population (cloudy and
MWR LWP ≥ 10 g m⁻²) is in the report above. Expect the clear-sky slope to be
the smallest: under clear skies DLR varies with water vapour and temperature
advection rather than cloud, and the skin follows the air.
"""
        ),
        code(
            """fr.fig_responder_vs_forcer(OBS, MASKS, "sh_up", forcer="lwd", out_dir=SAVE_DIR);"""
        ),
        md("### 3. Latent heat flux against DLR"),
        code(
            """fr.fig_responder_vs_forcer(OBS, MASKS, "lh_up", forcer="lwd", out_dir=SAVE_DIR);"""
        ),
        md(
            """### 4. Upwelling longwave, the net atmospheric flux, and $-G$ against DLR

LWU is the measured term and carries the largest share. NA = LWD − LWU + SWN −
SH − LH is what the atmosphere leaves for the subsurface (Sledd Eq. 5) and its
slope equals $1 - f_{LWU} - f_{SH} - f_{LH} - f_{SW}$ exactly. $-G_{TI}$ is the
independent-in-method (not independent-of-$T_{skin}$) estimate of the same
quantity; the difference between the last two panels' slopes is the closure
error of the observational budget.
"""
        ),
        code(
            """fr.fig_responder_vs_forcer(OBS, MASKS, "lwu", forcer="lwd", out_dir=SAVE_DIR);
fr.fig_responder_vs_forcer(OBS, MASKS, "na", forcer="lwd", out_dir=SAVE_DIR);
fr.fig_responder_vs_forcer(OBS, MASKS, "neg_g_ti", forcer="lwd", out_dir=SAVE_DIR);
"""
        ),
        md(
            """## 5. The partition, and how it is estimated

With every responder counted as energy leaving the surface, differentiating the
budget with respect to DLR gives

$$1 = f_{LWU} + f_{SH} + f_{LH} + f_{SW} + f_{res}, \\qquad
f_{SW} = -\\frac{d(SW_{net})}{d(DLR)}, \\quad f_{res} = \\frac{d(NA)}{d(DLR)}$$

which sums to one by construction ($f_{res}$ is the remainder). $f_G^{TI} =
d(-G_{TI})/d(DLR)$ is drawn beside it as the independent estimate of the same
subsurface response. Error bars are 95% moving-block bootstrap intervals; the
table prints the bootstrap standard error next to the textbook one so the size
of the independence assumption is a number rather than an assertion.
"""
        ),
        code(
            """POPS = ("all", "cloudy", "liquid", "clear")
PARTS_LWD = {p: fr.partition(OBS, MASKS[p], "lwd") for p in POPS}
CI_LWD = {p: fr.bootstrap_partition(OBS, MASKS[p], "lwd", n_boot=N_BOOT) for p in POPS}
fr.fig_partition_bars(PARTS_LWD, "lwd", intervals=CI_LWD, out_dir=SAVE_DIR);

print(f"{'population':<10}{'term':<10}{'value':>8}{'95% CI':>20}{'boot SE':>9}{'naive SE':>10}{'ratio':>7}")
naive_key = {"f_lwu": "lwu", "f_sh": "sh_up", "f_lh": "lh_up", "f_res": "na", "f_g_ti": "neg_g_ti"}
for p in POPS:
    for t in ("f_lwu", "f_sh", "f_lh", "f_sw", "f_res", "f_g_ti"):
        r = CI_LWD[p][t]
        nse = fr.naive_se(OBS, MASKS[p], naive_key[t], "lwd") if t in naive_key else float("nan")
        print(f"{p:<10}{t:<10}{r['value']:>8.3f}   [{r['lo']:+.3f}, {r['hi']:+.3f}]{r['se']:>9.3f}{nse:>10.4f}{r['se']/nse if nse == nse else float('nan'):>7.1f}")
"""
        ),
        md(
            """### The ledger

The fractions can be negative — a term that *arrives with* the DLR anomaly is a
source, not a sink — and a stacked bar hides that. The ledger of the ERA5
notebook normalises both sides by the gross energy in motion so supply and
disposal each sum to one for every population.
"""
        ),
        code("""fr.fig_ledger(PARTS_LWD, "lwd", out_dir=SAVE_DIR);"""),
        md(
            """## 6. Thermal freedom

$dT_{skin}/d(DLR)$ is how far the surface is free to warm per W m⁻²; it sets
$f_{LWU}$ through Stefan–Boltzmann ($4\\varepsilon\\sigma T^3 \\approx 3.6$ W m⁻² K⁻¹ at 250 K,
so 0.19 K per W m⁻² ≈ 0.68 of the anomaly re-emitted). The sign of the
turbulent response is decided by $d(T_{skin} - T_{2m})/d(DLR)$: positive means
the surface warms faster than the air and the upward flux strengthens (or the
downward one weakens).
"""
        ),
        code("""fr.fig_thermal_freedom(OBS, MASKS, forcer="lwd", out_dir=SAVE_DIR);"""),
        md(
            """## 7. The control ladder

Plain regression (Sledd), then holding $T_{2m}$ fixed, then $T_{2m}$ and wind.
Holding $T_{2m}$ fixed removes the air-mass confound *and* the genuine chain
DLR → $T_{skin}$ → $T_{2m}$; over a snow surface whose air is largely slaved to it
that deletes most of $f_{LWU}$, which is why the controlled estimate is a lower
bound on the radiative response and not a corrected one.
"""
        ),
        code(
            """fr.fig_control_ladder(OBS, MASKS, forcer="lwd", out_dir=SAVE_DIR);
fr.print_report(OBS, MASKS, forcer="lwd", controls=("none", "t2m", "t2m_wind"))
"""
        ),
        md(
            """## 8. Month by month, against the ERA5 cell

The Sledd et al. Fig. 2a layout with the DLR forcer (their figure uses
DLR + SW$_{net}$; that version is the companion notebook). Marker size scales
with the $r^2$ of each term's monthly regression. Dashed lines are ERA5 at the
grid cell nearest the site, from the same Eq. (1) terms (ERA5 has no $G$).
"""
        ),
        code(
            """fr.fig_monthly_response(OBS, MASKS, forcer="lwd", populations=("all", "cloudy"), era5=ERA5, show_sledd=False, out_dir=SAVE_DIR);
r = fr.monthly_partition(OBS, MASKS["all"], "lwd")
print(f"{'month':>5}{'f_LWU':>8}{'f_SH':>8}{'f_LH':>8}{'f_SW':>8}{'f_res':>8}{'f_G(TI)':>9}{'sum+G':>8}{'n':>7}{'Tskin C':>9}")
for i, m in enumerate(r["months"]):
    print(f"{fr.WINTER_LABELS[i]:>5}{r['f_lwu'][i]:>8.3f}{r['f_sh'][i]:>8.3f}{r['f_lh'][i]:>8.3f}{r['f_sw'][i]:>8.3f}"
          f"{r['f_res'][i]:>8.3f}{r['f_g_ti'][i]:>9.3f}{r['total_measured'][i]:>8.3f}{int(r['n'][i]):>7}{r['t_skin_mean'][i]-273.15:>9.1f}")
"""
        ),
        md(
            """## 9. Sensitivity of the bulk-flux responses to their free parameters

The roughness length is not measured. The bulk fluxes are recomputed from the
product-resolution state for $z_0$ = 10⁻⁴, 3×10⁻⁴ (default) and 10⁻³ m and for
both stable-stability schemes, and the partition is re-fitted each time. What
$z_0$ cannot change is $f_{LWU}$; what it moves between is $f_{SH}$ and $f_{res}$.
"""
        ),
        code(
            """Z0_ROWS = fr.z0_sensitivity(OBS, MASKS["all"], forcer="lwd")
fr.fig_z0_sensitivity(Z0_ROWS, forcer="lwd", out_dir=SAVE_DIR);
print(f"{'scheme':<14}{'z0 [m]':>8}{'f_LWU':>8}{'f_SH':>8}{'f_LH':>8}{'f_res':>8}{'mean SH':>9}")
for r in Z0_ROWS:
    print(f"{r['scheme']:<14}{r['z0_m']:>8.0e}{r['f_lwu']:>8.3f}{r['f_sh']:>8.3f}{r['f_lh']:>8.3f}{r['f_res']:>8.3f}{r['sh_up_mean']:>9.2f}")
"""
        ),
        md(
            """## 10. 10-min against hourly

Sledd et al. note that averaging their 10-min data to 6 h raises the LWU
response by ≤ 0.05 and lowers the SH response by a similar amount. The same
comparison here, between the two products built from the same 1-min streams.
"""
        ),
        code(
            """RES = {"10-min, all sky": fr.partition(OBS, MASKS["all"], "lwd"), "hourly, all sky": fr.partition(OBS_H, MASKS_H["all"], "lwd")}
fr.POP_LABELS.update({k: k for k in RES}); fr.POP_COLORS.update({"10-min, all sky": "#333333", "hourly, all sky": "#999999"})
fr.fig_resolution_comparison(RES, "lwd", out_dir=SAVE_DIR);
for k, v in RES.items():
    print(f"{k:<18}", fr.format_partition(v))
"""
        ),
        md(
            """## 11. DLR against LWP

The observational relation the ERA5 notebook compares to (its figure 7), on the
same axes, for cloudy 10-min samples. The dashed line is the published Barrow
fit quoted there, $y = 0.27x + 228.26$; the black line is this season's fit.
LWP is the MWRRET retrieval where available and the 3-channel MWR otherwise.
"""
        ),
        code("""fr.fig_lwp_vs_dlr(OBS, MASKS["cloudy"], out_dir=SAVE_DIR);"""),
        md(
            """## 12. The observed partition beside ERA5's at the same cell

Same season, same forcer, same populations (ERA5 cloud fraction and LWP gates
applied to ERA5's own `tcc` and `tclw`). ERA5's turbulent fluxes are its own
(negated to the Sledd sense); it has no $G$, so only the residual is comparable.
The ERA5 cell is 62% land on a 0.25° grid and mixes tundra with the coastal
sea, which is one reason to expect differences that are not model error.
"""
        ),
        code(
            """if ERA5 is not None:
    print(f"{'':<8}{'source':<8}{'f_LWU':>8}{'f_SH':>8}{'f_LH':>8}{'f_SW':>8}{'f_res':>8}{'dTs/dF':>9}{'n':>8}")
    for p in ("all", "cloudy", "clear"):
        for name, d, m in (("obs", OBS_H, MASKS_H[p]), ("ERA5", ERA5, ERA5_MASKS[p])):
            q = fr.partition(d, m, "lwd")
            print(f"{p:<8}{name:<8}{q['f_lwu']:>8.3f}{q['f_sh']:>8.3f}{q['f_lh']:>8.3f}{q['f_sw']:>8.3f}{q['f_res']:>8.3f}{q['dskt_dF']:>9.4f}{q['n']:>8}")
else:
    print("ERA5 files not found; skipping")
"""
        ),
        md(
            """## What this record shows

NSA C1, 1 Oct 2025 – 31 Mar 2026, 26,160 ten-minute samples, plain regression on
DLR unless noted; intervals are 95% moving-block bootstrap (26 blocks of 7 days,
1,000 replicates). Fractions are energy leaving the surface per W m⁻² of DLR.

**The partition, all sky.** $f_{LWU}$ = 0.69 [0.64, 0.73], $f_{SH}$ = 0.09
[0.06, 0.13], $f_{LH}$ = 0.04 [0.02, 0.05], $f_{SW}$ = −0.02, $f_{res}$ = 0.20
[0.16, 0.26]; the thermal-inertia route gives $f_G^{TI}$ = 0.11 [0.09, 0.15].
The surface warms 0.19 K per W m⁻² of DLR.

**By population.** Under liquid-bearing overcast the surface re-radiates 0.91 of
an extra watt and the bulk turbulent response is nil ($f_{SH}$ = 0.01 [−0.02,
0.04]): the skin and the air move together under the cloud
($d(T_{skin} - T_{2m})/d(DLR)$ = 0.006 K per W m⁻²). Under clear skies $f_{LWU}$
exceeds one (1.02 [0.93, 1.13]) and $f_{SH}$ is *negative* (−0.11 [−0.30,
+0.02]): clear-sky DLR variability is water vapour and advection, the air warms
faster than the surface, and the turbulent term delivers energy alongside the
radiative anomaly rather than removing it — the ERA5 notebook's "turbulence adds
to the warming" case, seen in the observations. The clear-sky interval is wide
because that population holds few independent weather situations.

**Two estimators.** Holding $T_{2m}$ fixed moves $f_{LWU}$ from 0.69 to 0.15 and
$f_{SH}$ from 0.09 to 0.38. That is the mediator effect: over snow the 2-m air
is largely slaved to the skin beneath it, so removing its co-variation removes
most of the genuine radiative response. Neither is the causal answer; the plain
slope is the headline (as in Sledd et al.) and the controlled one the lower
bound on the radiative pathway.

**What the free parameters do.** A factor of ten in $z_0$ (10⁻⁴ to 10⁻³ m) moves
$f_{SH}$ from 0.08 to 0.10 and $f_{res}$ the other way; the two stable-stability
schemes differ by 0.001. The bulk SH *response* is robust to the
parameterization's knobs — but it is still the response of a parameterization
(see the variables section and the companion notebook, where it is about half
the eddy-covariance response measured at MOSAiC).

**Cadence and the naive interval.** The 10-min and hourly products agree to
0.006 in every fraction. The textbook standard error is 7–18× smaller than the
bootstrap one; against it every fraction would look determined to three figures.

**ERA5 at the same cell (hourly, all sky).** $f_{LWU}$ 0.79 vs 0.70 observed,
$f_{SH}$ 0.08 vs 0.09, $f_{LH}$ 0.07 vs 0.04, $f_{res}$ 0.08 vs 0.19: ERA5
returns more of the anomaly as emission and leaves less than half as much for
the subsurface. Its cell is 62% land and includes coastal sea, so part of that
is not model error.
"""
        ),
    ]
    nb = new_notebook(
        cells=cells,
        metadata={"kernelspec": KERNEL, "language_info": {"name": "python"}},
    )
    return nb


# ----------------------------------------------------------------------------
# Notebook 2: DLR + SW_net forcer
# ----------------------------------------------------------------------------


def notebook_fnet() -> nbformat.NotebookNode:
    cells = [
        md(
            """# Surface flux response to the net radiative forcing, DLR + SW$_{net}$ — NSA C1 observations

Companion to `seb_flux_response_to_DLR.ipynb`, and the observational twin of
`ERA5/surface_energy_budget/turbulent_flux_response_to_DLR_and_SW.ipynb`. The
forcer is now $F = LWD + SW_{net}$, the one Sledd et al. (2025) use (their
Sect. 3.2, Eq. 9), so every number here is directly comparable with their
Figs. 1–3.

**Why the second forcer.** In polar night a thinner cloud changes DLR and nothing
else, and the two forcers coincide. In October and March a thinner cloud also
lets more shortwave through: regressing on DLR alone then books the shortwave
part of the response as a *responder* ($f_{SW}$) when it is really part of the
*forcing*. With $F = LWD + SW_{net}$ the partition has four terms,

$$1 = \\underbrace{\\frac{d(LWU)}{dF}}_{f_{LWU}}
    + \\underbrace{\\frac{d(SH_{up})}{dF}}_{f_{SH}}
    + \\underbrace{\\frac{d(LH_{up})}{dF}}_{f_{LH}}
    + \\underbrace{\\frac{d(NA)}{dF}}_{f_{res}},
\\qquad NA = F - LWU - SH - LH = M - SWT - G,$$

and Sledd's closure statement (their Eq. 9) is that the *measured* responders
$LWU + SH + LH - G$ (with $M = SWT = 0$ in winter) respond to $F$ with a total
slope of one. At MOSAiC the winter total was 0.94–1.08.

## Contents

| section | content |
|---|---|
| variables | the Sledd terms, what Barrow supplies, and why closure means less here |
| 1 | sign conventions, and the identity checked numerically |
| 2 | December: the Sledd Fig. 1 panels, with their December 2019 slopes |
| 3 | the four responders (and $-G$) against $F$, by population |
| 4 | the partition on the two forcers side by side, with bootstrap intervals |
| 5 | month by month against MOSAiC (Sledd Fig. 2a) and the ERA5 cell |
| 6 | closure: $dNA/dF$ against $d(-G_{TI})/dF$, and the "total measured response" |
| 7 | Sledd's winter window, and their Fig. 3a observational range |
| 8 | the control ladder and the bulk-flux free parameters |
| 9 | 10-min against hourly |
"""
        ),
        md("## Setup"),
        code(SETUP_CODE),
        md("## Load"),
        code(LOAD_CODE),
        md(VARIABLE_CHECK_MD),
        md(
            """## 1. Sign conventions, and the identity checked numerically

The product stores every term in the Sledd sense: `sh_up`, `lh_up` positive
upward, $G$ positive toward the surface, radiative fluxes as measured. The
ERA5 notebook needed a section on why a naive sum of slopes does not reach one
(ERA5 stores fluxes positive downward); here the sum of the four plain slopes
must equal one exactly, and the cell below checks it — together with the
identity $d(NA)/dF = 1 - f_{LWU} - f_{SH} - f_{LH}$ — for every population.
"""
        ),
        code(
            """fr.self_check(OBS)
for p in ("all", "cloudy", "liquid", "clear"):
    q = fr.partition(OBS, MASKS[p], "fnet")
    s4 = q["f_lwu"] + q["f_sh"] + q["f_lh"] + q["f_res"]
    print(f"{p:<8} f_LWU + f_SH + f_LH + f_res = {s4:.6f}   f_res (remainder) = {q['f_res']:+.4f}   d(NA)/dF (direct) = {q['f_res_direct']:+.4f}")
print()
fr.print_report(OBS, MASKS, forcer="fnet")
"""
        ),
        md(
            """## 2. December — the Sledd et al. Fig. 1 panels

Their Fig. 1 is December 2019 at MOSAiC, all sites, 10-min data: LWU slope 0.50
($r^2$ 0.79), upward SH 0.25 (0.48), upward LH 0.02 (0.37), $-G$ 0.20 (0.12).
The same month at Barrow, same layout, same cadence. The two clusters in their
LWU panel — radiatively clear and opaquely cloudy — are the bimodal winter
Arctic surface state and should be visible here too.
"""
        ),
        code(
            """DEC = MASKS["all"] & fr.month_mask(OBS, [12])
fr.fig_all_responders(OBS, DEC, forcer="fnet", population="all", out_dir=SAVE_DIR, stem="december_all_responders_vs_fnet");
q = fr.partition(OBS, DEC, "fnet"); r2 = fr.partition_r2(OBS, DEC, "fnet")
print(f"{'':<12}{'Barrow Dec 2025':>18}{'r2':>7}{'MOSAiC Dec 2019':>18}")
for k, sk in (("f_lwu", "lwu"), ("f_sh", "sh"), ("f_lh", "lh"), ("f_g_ti", "g")):
    print(f"{fr.TERM_STYLE[k][0]:<28}{q[k]:>8.3f}{r2[k]:>7.2f}{fr.SLEDD_FIG1_DECEMBER[sk]:>12.2f}")
print(f"{'NA (residual)':<28}{q['f_res']:>8.3f}{r2['f_res']:>7.2f}{'--':>12}")
"""
        ),
        md("""## 3. The four responders, and $-G_{TI}$, against $F$ by population"""),
        code(
            """for y in ("lwu", "sh_up", "lh_up", "na", "neg_g_ti"):
    fr.fig_responder_vs_forcer(OBS, MASKS, y, forcer="fnet", out_dir=SAVE_DIR);
"""
        ),
        md(
            """## 4. The partition on the two forcers, side by side

Left-to-right within each term: the four populations. Mean SW$_{net}$ over the
season is small (Oct and Mar carry it), so the two forcers differ mainly in
those months; in December and January they are the same numbers. Error bars:
95% moving-block bootstrap.
"""
        ),
        code(
            """POPS = ("all", "cloudy", "liquid", "clear")
PARTS = {f: {p: fr.partition(OBS, MASKS[p], f) for p in POPS} for f in ("lwd", "fnet")}
CI = {p: fr.bootstrap_partition(OBS, MASKS[p], "fnet", n_boot=N_BOOT) for p in POPS}
fr.fig_partition_bars(PARTS["fnet"], "fnet", intervals=CI, out_dir=SAVE_DIR);
fr.fig_partition_bars(PARTS["lwd"], "lwd", out_dir=SAVE_DIR, stem="partition_bars_lwd_for_comparison");
print(f"{'pop':<8}{'forcer':>7}{'f_LWU':>8}{'f_SH':>8}{'f_LH':>8}{'f_SW':>8}{'f_res':>8}{'f_G(TI)':>9}{'total':>8}{'dTs/dF':>9}")
for p in POPS:
    for f in ("lwd", "fnet"):
        q = PARTS[f][p]
        print(f"{p:<8}{f:>7}{q['f_lwu']:>8.3f}{q['f_sh']:>8.3f}{q['f_lh']:>8.3f}{q.get('f_sw', float('nan')):>8.3f}{q['f_res']:>8.3f}{q['f_g_ti']:>9.3f}{q['total_measured']:>8.3f}{q['dskt_dF']:>9.4f}")
print()
print(f"{'pop':<8}{'term':<8}{'value':>8}{'95% CI':>20}{'n_block':>8}")
for p in POPS:
    for t in ("f_lwu", "f_sh", "f_lh", "f_res", "f_g_ti", "total_measured"):
        r = CI[p][t]; print(f"{p:<8}{t:<8}{r['value']:>8.3f}   [{r['lo']:+.3f}, {r['hi']:+.3f}]{r['n_block']:>8}")
"""
        ),
        md(
            """## 5. Month by month, against MOSAiC and the ERA5 cell

The direct counterpart of Sledd et al. (2025) Fig. 2a for October–March.
Filled markers: this season at Barrow, sized by $r^2$; hollow markers: the
MOSAiC values digitised from their figure (±0.02); dashed: ERA5 at the nearest
grid cell. The dotted black line is the "total measured response"
$f_{LWU} + f_{SH} + f_{LH} + f_G^{TI}$, the analogue of their black diamonds
(with the caveat of the variables section: three of those four terms are
downstream of the same skin temperature here).

Two things to keep in view while reading it: MOSAiC is one winter over drifting
pack ice at 84–88 °N with a thin snow cover over 1–2 m of ice and an ocean
beneath; Barrow is one winter over snow-covered tundra on frozen ground at
71 °N. The surface types differ in thermal inertia and in what lies below, so
agreement in $f_{LWU}$ is expected (a snow surface radiates back about half of
what it receives either way) while the subsurface share need not match.
"""
        ),
        code(
            """fr.fig_monthly_response(OBS, MASKS, forcer="fnet", populations=("all", "cloudy"), era5=ERA5, show_sledd=True, out_dir=SAVE_DIR);
r = fr.monthly_partition(OBS, MASKS["all"], "fnet")
S = fr.SLEDD_FIG2A_WINTER
print(f"{'':>5}{'---- LWU ----':>18}{'---- SH ----':>18}{'---- LH ----':>18}{'-- NA / -G --':>26}{'-- total --':>18}")
print(f"{'':>5}{'obs':>9}{'MOSAiC':>9}{'obs':>9}{'MOSAiC':>9}{'obs':>9}{'MOSAiC':>9}{'NA obs':>9}{'-G(TI)':>8}{'NA MOS':>9}{'obs':>9}{'MOSAiC':>9}")
for i, m in enumerate(r["months"]):
    print(f"{fr.WINTER_LABELS[i]:>5}{r['f_lwu'][i]:>9.3f}{S['lwu'][m]:>9.3f}{r['f_sh'][i]:>9.3f}{S['sh'][m]:>9.3f}"
          f"{r['f_lh'][i]:>9.3f}{S['lh'][m]:>9.3f}{r['f_res'][i]:>9.3f}{r['f_g_ti'][i]:>8.3f}{S['na'][m]:>9.3f}"
          f"{r['total_measured'][i]:>9.3f}{S['total'][m]:>9.3f}")
"""
        ),
        md(
            """## 6. Closure

Two routes to the subsurface response: $dNA/dF$, which is what the *measured*
radiation leaves after the *parameterized* turbulence, and $d(-G_{TI})/dF$, the
thermal-inertia estimate from the skin-temperature history alone. Sledd et al.
compared their NA response with the buoy-measured $-G$ response and found them
"equal, or at least within measurement uncertainties" in winter. The gap here is
the closure error of the Barrow budget: it collects the LWU frost-bias episodes
(README), the bulk-flux parameterization error, and the thermal-inertia
assumptions (a homogeneous half-space with one thermal inertia for snow over
frozen soil), with no way from the data alone to apportion it.

The "total measured response" $f_{LWU} + f_{SH} + f_{LH} + f_G^{TI}$ is drawn
because it is what Sledd's black diamonds show, but at Barrow it is not an
independent test: see the variables section.
"""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(6.6, 4.2))
x = np.arange(len(fr.WINTER_MONTHS))
for p, ls in (("all", "-"), ("cloudy", "--")):
    r = fr.monthly_partition(OBS, MASKS[p], "fnet")
    ax.plot(x, r["f_res"], color=fr.TERM_STYLE["f_res"][1], ls=ls, marker="^", label=f"dNA/dF, {p}")
    ax.plot(x, r["f_g_ti"], color=fr.TERM_STYLE["f_g_ti"][1], ls=ls, marker=">", label=f"d(-G$_{{TI}}$)/dF, {p}")
    ax.plot(x, r["total_measured"], color="#000000", ls=ls, marker="D", ms=4, label=f"total measured, {p}")
ax.scatter(x + 0.12, [fr.SLEDD_FIG2A_WINTER["na"][m] for m in fr.WINTER_MONTHS], s=60, facecolor="none", edgecolor=fr.TERM_STYLE["f_res"][1], marker="^", label="MOSAiC NA")
ax.scatter(x + 0.12, [fr.SLEDD_FIG2A_WINTER["g"][m] for m in fr.WINTER_MONTHS], s=60, facecolor="none", edgecolor=fr.TERM_STYLE["f_g_ti"][1], marker=">", label="MOSAiC -G (buoys)")
ax.scatter(x + 0.12, [fr.SLEDD_FIG2A_WINTER["total"][m] for m in fr.WINTER_MONTHS], s=50, facecolor="none", edgecolor="#000000", marker="D", label="MOSAiC total")
ax.axhline(1, color="#333333", lw=0.8, ls=":"); ax.axhline(0, color="#333333", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(fr.WINTER_LABELS); ax.set_ylabel("response to DLR + SW$_{net}$"); ax.grid(alpha=0.2)
ax.legend(fontsize=6.6, frameon=False, ncol=2); ax.set_title("Closure: two routes to the subsurface response, and the total", fontsize=10, loc="left")
fig.tight_layout()
if SAVE_DIR is not None: fig.savefig(Path(SAVE_DIR) / "closure_monthly_fnet.png", dpi=fr.FIG_DPI, bbox_inches="tight")
"""
        ),
        md(
            """## 7. Sledd's winter window and their Fig. 3a observational range

Sledd et al. evaluate models over 15 Oct – 14 Mar, and their Fig. 3a gives the
observed winter responses at the Central Observatory for 1-h and 3-h averages
(stars) with the range across sites as grey rectangles: LWU ≈ 0.5–0.6, NA ≈
0.2–0.3, SH ≈ 0.2, LH ≈ 0.02. The same window at Barrow, at 10-min, 1-h and
3-h averaging.
"""
        ),
        code(
            """W = fr.sledd_winter_mask(OBS) & MASKS["all"]
W_H = fr.sledd_winter_mask(OBS_H) & MASKS_H["all"]
OBS_3H = OBS_H.resample(time="3h").mean()
OBS_3H["month"] = OBS_3H["time"].dt.month
W_3H = fr.sledd_winter_mask(OBS_3H) & np.isfinite(OBS_3H["lwd"].values)
rows = {"10-min": fr.partition(OBS, W, "fnet"), "1-h": fr.partition(OBS_H, W_H, "fnet"), "3-h": fr.partition(OBS_3H, W_3H, "fnet")}
print("15 Oct 2025 - 14 Mar 2026, all sky (Sledd et al. Fig. 3a window)")
print(f"{'avg':<8}{'f_LWU':>8}{'f_SH':>8}{'f_LH':>8}{'NA':>8}{'f_G(TI)':>9}{'total':>8}{'n':>8}")
for k, q in rows.items():
    print(f"{k:<8}{q['f_lwu']:>8.3f}{q['f_sh']:>8.3f}{q['f_lh']:>8.3f}{q['f_res']:>8.3f}{q['f_g_ti']:>9.3f}{q['total_measured']:>8.3f}{q['n']:>8}")
print("MOSAiC CO (Sledd Fig. 3a, read from the figure): LWU ~0.55-0.60, NA ~0.20-0.28, SH ~0.20, LH ~0.02")
if ERA5 is not None:
    q = fr.partition(ERA5, fr.sledd_winter_mask(ERA5) & ERA5_MASKS["all"], "fnet")
    print(f"{'ERA5 1-h':<8}{q['f_lwu']:>8.3f}{q['f_sh']:>8.3f}{q['f_lh']:>8.3f}{q['f_res']:>8.3f}{'--':>9}{'--':>8}{q['n']:>8}   (IFS in Sledd Fig. 3a: LWU ~0.40, SH ~0, NA ~0.60)")
"""
        ),
        md("""## 8. The control ladder and the bulk-flux free parameters"""),
        code(
            """fr.fig_control_ladder(OBS, MASKS, forcer="fnet", out_dir=SAVE_DIR);
Z0_ROWS = fr.z0_sensitivity(OBS, MASKS["all"], forcer="fnet")
fr.fig_z0_sensitivity(Z0_ROWS, forcer="fnet", out_dir=SAVE_DIR);
print(f"{'scheme':<14}{'z0 [m]':>8}{'f_LWU':>8}{'f_SH':>8}{'f_LH':>8}{'f_res':>8}{'mean SH':>9}")
for r in Z0_ROWS:
    print(f"{r['scheme']:<14}{r['z0_m']:>8.0e}{r['f_lwu']:>8.3f}{r['f_sh']:>8.3f}{r['f_lh']:>8.3f}{r['f_res']:>8.3f}{r['sh_up_mean']:>9.2f}")
"""
        ),
        md("""## 9. 10-min against hourly"""),
        code(
            """RES = {"10-min, all sky": fr.partition(OBS, MASKS["all"], "fnet"), "hourly, all sky": fr.partition(OBS_H, MASKS_H["all"], "fnet")}
fr.POP_LABELS.update({k: k for k in RES}); fr.POP_COLORS.update({"10-min, all sky": "#333333", "hourly, all sky": "#999999"})
fr.fig_resolution_comparison(RES, "fnet", out_dir=SAVE_DIR);
for k, v in RES.items():
    print(f"{k:<18}", fr.format_partition(v))
"""
        ),
        md(
            """## What this record shows against MOSAiC

Same season and estimator as the DLR notebook, forcer $F$ = DLR + SW$_{net}$,
10-min data, 95% moving-block bootstrap intervals (26 blocks).

**All sky, Oct–Mar.** $f_{LWU}$ = 0.67 [0.62, 0.71], $f_{SH}$ = 0.09 [0.06, 0.13],
$f_{LH}$ = 0.04, $dNA/dF$ = 0.20 [0.16, 0.26], $d(-G_{TI})/dF$ = 0.11 [0.08,
0.14]; the "total measured response" $f_{LWU} + f_{SH} + f_{LH} + f_G^{TI}$ is
0.91 [0.88, 0.93]. In Sledd's winter window (15 Oct – 14 Mar): LWU 0.66, SH
0.10, LH 0.03, NA 0.22 at 10-min, and within 0.02 of that at 1-h and 3-h.
MOSAiC's Central Observatory (their Fig. 3a): LWU ≈ 0.55–0.60, SH ≈ 0.20,
LH ≈ 0.02, NA ≈ 0.20–0.28.

**Month by month (Sledd Fig. 2a).** The LWU response at Barrow is 0.50–0.62
against 0.44–0.57 at MOSAiC — the same physics (a snow surface re-radiates
about half of what it receives) at a warmer surface (−5 to −28 °C monthly
means, against −16 to −31 °C). December matches their Fig. 1 on the radiative
and subsurface terms: LWU 0.56 (theirs 0.50), $-G$ 0.21 (0.20), LH 0.01
(0.02). The subsurface responses bracket theirs: $dNA/dF$ 0.18–0.39 (MOSAiC NA
0.12–0.34), $d(-G_{TI})/dF$ 0.11–0.21 (MOSAiC buoys 0.21–0.32).

**The turbulent response is the discrepancy, and it is the parameterized
term.** Bulk $f_{SH}$ is 0.04–0.23 by month, 0.09 over the season, against
MOSAiC's eddy-covariance 0.13–0.32; in December it is 0.04 with $r^2$ = 0.02,
i.e. no relation at all, where MOSAiC found 0.25 with $r^2$ = 0.48. Neither
$z_0$ (0.08–0.10 across a decade) nor the stable-stability scheme closes the
gap. Two readings are consistent with the data and cannot be separated here:
the bulk formula with SHEBA-type strongly damped stable functions
under-responds when the surface warms under cloud, or the tundra boundary
layer genuinely couples less than the one over a 1–2 m ice floe with an ocean
beneath. The Oliktok Point ECOR (`ecor_e10`, once screened for riming) is the
only measured turbulent flux on the North Slope that could arbitrate.

**Closure.** The two routes to the subsurface response differ by 0.05–0.17 by
month, with $dNA/dF$ the larger. Because SH, LH and $G_{TI}$ are all functions
of the measured skin temperature, the total of 0.83–0.96 is a consistency
statement, not the independent closure MOSAiC's buoys allowed. A budget that
closed on measured terms would need eddy-covariance fluxes and a snow/soil
heat-flux measurement at Barrow; neither exists (README).

**ERA5 at the same cell.** LWU 0.76, SH 0.09, LH 0.07, NA 0.07 in Sledd's
window. That is nothing like the IFS over MOSAiC ice in their Fig. 3a (LWU
≈ 0.40, SH ≈ 0, NA ≈ 0.60): a land cell with a snow scheme partitions the
anomaly the way the observations do, only with more emission and less
subsurface. The no-snow-on-sea-ice failure they diagnose does not apply to a
tundra cell.
"""
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": KERNEL, "language_info": {"name": "python"}},
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="execute the notebooks in place after writing them",
    )
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    targets = {
        REPO / "seb_flux_response_to_DLR.ipynb": notebook_dlr(),
        REPO / "seb_flux_response_to_DLR_and_SW.ipynb": notebook_fnet(),
    }
    for path, nb in targets.items():
        nbformat.write(nb, path)
        print(f"wrote {path}")
    if args.execute:
        from nbclient import NotebookClient

        for path in targets:
            nb = nbformat.read(path, as_version=4)
            client = NotebookClient(
                nb,
                timeout=args.timeout,
                kernel_name=KERNEL["name"],
                resources={"metadata": {"path": str(REPO)}},
            )
            client.execute()
            nbformat.write(nb, path)
            print(f"executed {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
