Generated 2026-11-17 by make_era5_clear_sky_sensitivity.py (repo 150cc11-dirty)
Seasons: 11.

ERA5_clearSkyHours_perMonth_2014_2025_definitionSensitivity.csv
    Clear-sky hours per month by count method, region, variant, month, season
Figures: Figure 7 layout, panels (a)-(d) UTQ, Land, Open ocean, Pack ice
ERA5_clearSkyHours_perMonth_2014_2025_definitionSensitivity.png
    Hours per month for a typical grid cell, equal weight per cell
ERA5_clearSkyHours_perMonth_2014_2025_definitionSensitivity_areaWeighted.png
    Hours per month for a typical grid cell, area weighted (cos latitude)
ERA5_clearSkyHours_perMonth_2014_2025_definitionSensitivity_anyCellCount.png
    Hours per month with at least one clear cell of the class, as Figure 7 counts

How many ERA5 hours count as clear sky, and how does that depend on the definition?

WHY THIS EXISTS
===============
In Figure 7 of Genie's draft (OV_SLCFrequency_20261102), ERA5 has far fewer
clear-sky hours than the ARM observations at UTQ: 1-17 h per month against
35-165 h. This script measures how the ERA5 count changes when each part of the
clear-sky definition is relaxed, so the strictness of the definition can be
separated from ERA5 simply being cloudier.

THE DEFINITION IN FIGURE 7
==========================
make_era5_hourly_values_csvs.py, which wrote Genie's ERA5 files, calls an hour
clear when ALL of these hold (plot_dlr_by_phase.DEFAULT_CLEAR_TCC_MAX = 0.05):

    LWP <= 0.01 g m-2 and IWP <= 0.01 g m-2   (both paths at or below the floors)
    tcc < 0.05
    tp  < 0.05 mm hr-1

So the cloud cover limit is already 0.05, not 0.

THE RELAXED DEFINITIONS
=======================
Each variant keeps tp < 0.05 mm hr-1. The precipitation test makes no
difference to clear hours (checked 2026-11-12: the counts with and without it
agree to 0.1 h per month), so it is not varied.

    fig7          the Figure 7 definition above
    tcc20         tcc < 0.20, paths unchanged
    tcc20_iwp01   tcc < 0.20, LWP <= 0.01, IWP <= 0.1 g m-2
    tcc20_paths1  tcc < 0.20, LWP and IWP <= 1 g m-2
    paths1        LWP and IWP <= 1 g m-2, no cloud cover test
    cre5          DLR - DLR_clear < 5 W m-2, no cloud cover or path test
    cre10         DLR - DLR_clear < 10 W m-2, no cloud cover or path test
    arm80         tcc <= 0.20, no path test (an analogue of Genie's ARM rule; below)
    arm80_gt1h    arm80, and the same cell also passes arm80 in the hour before
                  or the hour after (a clear spell longer than 1 h)

DLR is msdwlwrf and DLR_clear is msdwlwrfcs, ERA5's own clear-sky downward
longwave: the same radiation calculation with the clouds removed (ECMWF IFS
documentation, Part IV, chapter 2). Their difference is ERA5's longwave cloud
radiative effect at the surface. The cre variants therefore ask whether the
clouds ERA5 does have are too thin to matter radiatively. The thresholds 0.20, 0.1, 1, 5
and 10 are sensitivity choices of mine, not values from a paper.

AN ERA5 ANALOGUE OF GENIE'S ARM CLEAR-SKY RULE
==============================================
Genie calls an hour clear when at least 80% of the 30-s thermocldphase time
stamps in it have a column labelled clear, i.e. at most 20% of the samples see
any hydrometeor anywhere in the column. Her Figure 7a counts clear spells
longer than 1 h.

ERA5 has no sub-hourly samples, but tcc is the fraction of the grid box covered
by cloud at any level, seen from above. My reasoning (an analogy I have not seen
in a paper): an hour of 30-s columns samples the cloud field advected over the
site, roughly 20-30 km at 6-8 m/s, which is about one ERA5 cell (0.25 deg is
about 28 km north-south). If that time sample is representative of the cell
(Taylor's frozen-turbulence hypothesis plus an ergodic assumption), "at most 20%
of columns cloudy" corresponds to "cloud cover at most 0.20". Hence arm80:
tcc <= 0.20. It has no water-path test, because Genie's rule allows cloud in up
to 20% of the samples, so the grid-box mean paths need not be near zero.

arm80_gt1h adds persistence for the "> 1 h" in Figure 7a: an hour counts only if
the same cell also passes arm80 at an adjacent hour stamp (same season, one
hour apart). That is my reading of "> 1 h"; check it against Genie's code.

Where the analogy is weakest (my inference, not tested):
  * ERA5 tcc is a model diagnostic from the cloud scheme, not a detection by
    radar and lidar, so it can count cloud that a lidar would miss, and miss
    thin cloud that a lidar would see.
  * tcc is an instantaneous value at the hour stamp, while Genie's 80% is
    over the hour. arm80_gt1h, which needs two consecutive stamps, is the
    closer match to a clear hour.

Timing caveat: tcc, tclw, tciw are instantaneous at the time stamp, while tp,
msdwlwrf and msdwlwrfcs are means over the hour ending at it.

TWO WAYS TO COUNT HOURS FOR A SURFACE CLASS
===========================================
(a) Per grid cell, as in the class_normalized tables:

        H = sum_t n(t) / N(t)     [h]

    N(t) is the weight of the class's valid cells at hour t, and n(t) the weight
    of those that are clear (weight 1, or cos(lat) for area weighting). This is
    the clear-sky time of a typical cell of the class, so it compares with the
    single UTQ cell and with the ARM site.

(b) Hours with at least one clear cell of the class. This is what the numbers in
    Figure 7 count, because Genie's files hold the class-mean DLR over the cells
    in a state and are blank only when no cell is in it. For a class of hundreds
    of cells this is far larger than (a), and it is not comparable with a
    single site. For UTQ, (a) and (b) are the same.

Both are given as hours in each month, October to March, averaged over the
seasons. A class that does not exist in a month (open ocean in January)
contributes zero hours, as in the other tables.

THE FIGURES
===========
Three figures in the layout of Figure 7: panels (a)-(d) are UTQ, Land, Open
ocean and Pack ice, each with the six months on the x axis and one bar per
definition. They differ only in how a class is counted:

    ..._definitionSensitivity.png                count (a), equal weight per cell
    ..._definitionSensitivity_areaWeighted.png   count (a), cos(latitude) weight
    ..._definitionSensitivity_anyCellCount.png   count (b), as Figure 7 counts

UTQ is the same in all three. Its panel also marks the ARM clear-sky hours for
each month as a dashed line.

THE ARM REFERENCE
=================
ARM_CLEAR_GT1H_H holds the ARM clear-sky hours (> 1 h durations) printed above
the boxes of Figure 7a in the 2026-11-02 draft. They are read off the figure, not
computed here.

CHECK
=====
For the fig7 variant, count (b) must equal the number of filled clear_sky hours in
era5_hourly_values_csv/ERA5_DLR_perHour_<span>_<class>.csv. check_against_hourly_csvs
tests this for every month and class.

REGIONS: BARROW OR RUI
======================
The domain comes from ov_config, as for the other make_era5_*.py scripts. The
UTQ panel is the same grid cell in every domain, so its numbers do not depend
on the domain; Land, Open ocean and Pack ice do.

    Barrow (laptop, the local files)    python make_era5_clear_sky_sensitivity.py
    Rui region (Casper; GDEX only)       ./submit.sh --region Rui-region --years 2014-2024                                              --csvs clear_sky --out <run folder>

The Figure 7 check looks for the hourly-values folder of the same run
(era5_hourly_values_csv or era5_hourly_values_csv_areaWeighted beside the
hour-count folder), or --hourly-dir, and is skipped with a message if none is
found.

USAGE
=====
    python make_era5_clear_sky_sensitivity.py              # Barrow, both weightings, ~1 min
    python make_era5_clear_sky_sensitivity.py --plot-only  # redraw from the CSVs

Nothing here is stochastic, so there is no random seed.
