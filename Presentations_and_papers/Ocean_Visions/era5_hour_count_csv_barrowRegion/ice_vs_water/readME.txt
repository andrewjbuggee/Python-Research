ERA5 CLOUDS OVER OPEN WATER AND OVER SEA ICE, OCTOBER AND NOVEMBER
==================================================================

Generated 2026-11-06 by make_era5_ice_vs_water_clouds.py
(repo 086cc0f-dirty). Averages are not weighted (each grid cell counts equally). Data, filters and phase are as in
../readME.txt. The method is in the script's docstring.

  ERA5_iceVsWater_2014_2025.csv
      one row per thresholds x period x metric:
        class_mean_water, class_mean_ice      hourly class means, each hour equal;
                                              lc_kept equals the table quantity
        same_cell_water, same_cell_ice        same grid cell, same half month
        same_cell_water_minus_ice             their difference
        same_cell_jackknife_se                leave-one-season-out standard error
        n_cell_periods                        how many cell-periods were eligible
                                              (at least 72 h of each surface)
      thresholds: project (sea ice < 5%, > 95%); morrison (< 15%, > 80%).

  ERA5_iceVsWater_2014_2025_<thresholds>.png
      the figure, for October and November.

Fractions are of hours, except mean total and low cloud cover, which are of the
sky. ERA5 cloud cover is not what a lidar reports, and liquid containing here is
per column, not per height level as in Morrison et al. (2018).
