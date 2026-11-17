Generated 2026-11-07 by make_era5_filter_stage_hours_lwp.py (repo 1cf6043-dirty)
Seasons: 11. Cell weighting: area.

ERA5_avgHoursPerCell_areaWeighted_perMonth_2014_2025_liquidContaining_filterStages.csv
    Hours per cell by month, class, stage and season [h]
ERA5_LWP_areaWeighted_perMonth_2014_2025_liquidContaining_filterStages.csv
    Mean LWP by month, class, stage and season [g m-2]
ERA5_avgHoursPerCell_areaWeighted_perDay_2014_2025_liquidContaining_filterStages_allDays.csv
    Hours per cell for every day, class and stage [h]
ERA5_avgHoursPerCell_areaWeighted_perDay_2014_2025_liquidContaining_filterStages_summary.csv
    Median and mean of those days by month [h]
ERA5_avgHoursPerCell_areaWeighted_perMonth_2014_2025_liquidContaining_filterStages.png
    Stacked hours per month, median of seasons
ERA5_avgHoursPerCell_areaWeighted_perDay_2014_2025_liquidContaining_filterStages.png
    Stacked hours per day, median of days
ERA5_LWP_areaWeighted_perMonth_2014_2025_liquidContaining_filterStages.png
    LWP, side-by-side bars, median of seasons

What do the overcast and no-precipitation filters do to hours and to LWP?

WHAT IT MAKES
=============
Three figures for October, November and December, for pack ice, open ocean, land
and the UTQ grid cell, each drawn with area weighting and with equal weighting:

    1. Hours per month   stacked bars, median over the seasons
    2. Hours per day     stacked bars, median over the days, the statistic in the
                         median line of Genie's daily box plots (her Figure 5)
    3. LWP               three bars per class: median over the seasons of the mean
                         liquid water path in the hours that pass each stage

In a stacked bar, the whole bar is the hours at the first stage and the segments
show the hours each filter removes.

THE THREE FILTER STAGES
=======================
All three require a liquid-containing column: cloud water above the 0.01 g m-2
floors and IWP/CWP < 0.90. They then tighten the cloud cover and precipitation
tests one at a time:

    any_cloud          tcc > 0, with no precipitation test
    overcast           tcc >= 0.95
    overcast_noprecip  tcc >= 0.95 and tp < 0.05 mm hr-1  (the project's tables)

tcc > 0 is needed as its own test because ERA5 does not guarantee it: from
October to December, 0.107% of liquid-containing cell-hours have tcc = 0 (mean
LWP 0.16 g m-2). Each stage is a subset of the one before. The order matters for
the labels. "Removed by the no-precipitation filter" means overcast hours that
precipitate. A precipitating hour that is not overcast is already gone at the
first step, so it counts as removed by the overcast filter.

HOW THE VALUES ARE DEFINED
==========================
For one class c and hour t, with weight w (1, or cos(lat) for area weighting):

    N(t)  = sum of w over the valid cells of class c
    n(t)  = sum of w over those cells that pass the stage
    L(t)  = sum of w * LWP over those same cells             [g m-2]

Over any period (one UTC day, or one month of one season):

    hours per cell  H = sum_t n(t) / N(t)                    [h]
    mean LWP          = [ sum_t L(t) / N(t) ] / H            [g m-2]

H is the same per_hour average as make_era5_class_per_cell_csvs.py. Hours with no
cells of the class add zero to H, so open ocean in December is small partly
because it often does not exist. Mean LWP uses the same hourly normalization, so
each hour counts in proportion to the share of the class that passes the stage.

Hours per month are summarized by the median of the seasons' monthly H. Hours per
day are summarized by the median of H over every day of that month, pooled across
the seasons. A season with H = 0 counts as zero for hours. It has no defined LWP,
so it is left out of the LWP median, and the figure says so.

WHY MEDIANS CAN STILL BE STACKED
================================
Medians do not add, but every day and season has any_cloud >= overcast >=
overcast_noprecip, so the medians of the three stages keep that order. A segment
is therefore a DIFFERENCE BETWEEN TWO MEDIANS, not the median of the hours that
filter removes. The bar tops and the purple segments are true medians.

WHY THE LWP BARS ARE NOT STACKED
================================
Mean LWP can rise or fall as a filter removes hours, so a stack of LWP values
would have no meaning.

COMPARING WITH GENIE'S FIGURES
==============================
The figures use all the seasons. Genie's Figure 5 leaves out the months with
incomplete ARM data, which here means October-December 2020. The tables also hold
that version of each median, in the column median_excl_arm_missing. With it,
the daily medians at her thresholds agree with her Figure 5 to about 0.1 h.

WHY THIS SCRIPT MAKES ITS OWN PASS
==================================
count_cell_hours does not record LWP, so this script reads the raw fields and
evaluates all three stages on the same hours. It calls the project's
classify_cells and fraction_phase_masks and takes its thresholds from SETTINGS,
so the definitions are not copied. check_against_csvs compares the last stage with
the monthly and daily tables, and every run tests that the stages are nested.

WHAT "SUPERCOOLED" MEANS HERE
=============================
As in the rest of the Ocean Visions tables, the phase test uses tclw and tciw and
contains no temperature test. make_era5_supercooled_check.py examines how much of
that liquid is below 0 deg C.

USAGE
=====
    python make_era5_filter_stage_hours_lwp.py                     # both weightings
    python make_era5_filter_stage_hours_lwp.py --cell-weighting uniform
    python make_era5_filter_stage_hours_lwp.py --plot-only         # no raw data read

Nothing here is stochastic, so there is no random seed.
