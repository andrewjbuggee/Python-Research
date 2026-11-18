Generated 2026-11-17 by make_era5_iceOnly_sensitivity.py (repo 6f5a22e-dirty)
Seasons: 11.

ERA5_iceOnlyHours_perMonth_2014_2025_definitionSensitivity.csv
    Ice-only hours per month by count method (per_cell_area, per_cell_uniform, any_cell), region, variant, month, season
Figures: Figure 7 layout, panels (a)-(d) UTQ, Land, Open ocean, Pack ice
ERA5_iceOnlyHours_perMonth_2014_2025_definitionSensitivity_areaWeighted.png
    Hours per month for a typical grid cell, area weighted (cos latitude)
ERA5_iceOnlyHours_perMonth_2014_2025_definitionSensitivity_anyCellCount.png
    Hours per month with at least one ice-only cell of the class, as Figure 7 counts

How many ERA5 hours count as ice-only cloud, and how does that depend on the definition?

WHY THIS EXISTS
===============
In Figure 7a of Genie's draft (OV_SLCFrequency_20261102), ERA5 has far more
ice-only hours at UTQ than the ARM observations: 44-198 h per month against
2-9 h. Her ARM rule is strict in time (below), the ERA5 rule is a snapshot at
the hour stamp. This script measures how much of the gap a more ARM-like ERA5
rule removes.

GENIE'S ARM RULE (as described by the user, 2026-11-17)
=======================================================
From the thermocldphase VAP (30-s columns): a column is ice-only when every
level is either clear or ice, with no liquid, mixed or precipitation at any
level. An hour is ice-only when at least 80% of its 30-s columns are. Figure 7a
then counts ice-only spells longer than 1 h.

THE ERA5 RULE IN FIGURE 7
=========================
make_era5_hourly_values_csvs.py, which wrote Genie's ERA5 files, calls a cell
ice-only at an hour stamp when ALL of these hold (SETTINGS of
make_era5_utqiagvik_filtered_netcdf.py, imported here):

    tcc >= 0.95                                   (overcast gate)
    IWP/CWP >= 0.90, with CWP = LWP + IWP and each path set to zero at or
        below its 0.01 g m-2 floor (cloud_classification.fraction_phase_masks)
    tp < 0.05 mm hr-1                             (no precipitation)

THE VARIANTS
============
    fig7              the Figure 7 rule above
    fig7_gt1h         fig7, and the same cell is also ice-only (fig7) at the hour
                      stamp before or after: an ice-only spell longer than 1 h
    tcc80_gt1h        as fig7_gt1h with tcc >= 0.80 instead of 0.95
    gt1h_ice99        as fig7_gt1h with IWP/CWP >= 0.99
    gt1h_tp001        as fig7_gt1h with tp < 0.01 mm hr-1
    gt1h_ice99_tp001  as fig7_gt1h with IWP/CWP >= 0.99 and tp < 0.01 mm hr-1

In every "> 1 h" variant the adjacent hour must pass that variant's own full
test. The last three (user's choices, 2026-11-17) follow from the UTQ fig7
ice-only hours: 71% have some liquid above the floor and 83% have tp >= 0.001
mm hr-1, whereas Genie's rule allows no liquid and no precipitation at any level.
Precipitation limits keep the project's form: an hour AT or above is dropped.

The adjacency test is the one in make_era5_clear_sky_sensitivity.py
(accumulate(), imported here, so both scripts use the same code): same season,
stamps exactly one hour apart. Reading "> 1 h" as "this and an adjacent stamp"
is my interpretation; check it against Genie's code.

Why these are only partly ARM-like (my inference, not tested):
  * tcc, tclw and tciw are instantaneous at the stamp (ERA5 documentation,
    Tables 2 and 6: no hourly-mean cloud cover exists), while Genie's 80% is a
    fraction of the hour. Two consecutive stamps that pass is the closest
    available match to "most of the hour".
  * tcc >= 0.80 is the user's sensitivity choice. It is a loose
    analogy to "80% of the columns": ARM's other 20% may be anything, including
    liquid, whereas ERA5's IWP/CWP test is applied to the whole cell.
  * ARM's ice-only column excludes any liquid level; ERA5's rule allows up to
    10% of the condensed water as liquid. Hours where ERA5 has a little liquid
    above its floor may be called mixed by ARM.
  * Lidar is attenuated by liquid and the radar misses small ice, so the ARM
    phase mask and ERA5's cloud scheme do not see the same ice.

HOW HOURS ARE COUNTED
=====================
Main figure: hours per month for a typical grid cell, area weighted,

        H = sum_t n(t) / N(t)     [h]

with N(t) the cos(lat)-weighted count of the class's valid cells at hour t and
n(t) that of the cells in the state. UTQ is one cell, so its numbers are the
same in every weighting. The CSV also holds equal-weight per-cell hours and
the Figure 7 count (hours with at least one cell of the class in the state).
All are hours in each month, October to March, mean over seasons; a class that
does not exist in a month contributes zero.

THE ARM REFERENCE
=================
ARM_ICE_GT1H_H and ERA5_FIG7_ICE_H are the numbers printed above the boxes of
Figure 7a in the 2026-11-02 draft, read off the figure, not computed here.
The ERA5 ones must match the fig7 variant at UTQ (printed as a check).

CHECK
=====
For the fig7 variant, the 'any cell' count must equal the number of filled
ice_only hours in the run's hourly-values files
(era5_hourly_values_csv*/ERA5_DLR_perHour_<span>_<class>.csv), for every
month, season and region. The run stops if it does not.

REGIONS
=======
As for make_era5_clear_sky_sensitivity.py: Barrow locally with no options,
the Rui region on Casper (raw data there only), e.g.

    ./submit.sh --region Rui-region --years 2014-2024 --csvs ice_only --out <run folder>

The UTQ panel is the same cell in every domain.

USAGE
=====
    python make_era5_iceOnly_sensitivity.py              # Barrow, ~1 min on the laptop
    python make_era5_iceOnly_sensitivity.py --plot-only  # redraw from the CSV

Nothing here is stochastic, so there is no random seed.
