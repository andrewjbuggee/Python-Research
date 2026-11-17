AVERAGE CUMULATIVE CLOUD HOURS PER GRID CELL, BY SURFACE CLASS
==============================================================

Generated 2026-11-05 by make_era5_cumulative_hours_by_class.py
(repo c252567-dirty).

Data, filters, surface classes and cloud types are as in ../readME.txt.

  ERA5_cumulativeHoursPerCell_perDay_2014_2025_<TYPE>.csv
      rows = calendar days, 1 Oct to 31 Mar including 29 Feb; columns = classes.
      For each season: the running sum of n(t)/N(t) to the end of that day
      (hours per grid cell), then the mean over seasons. A class's hours with
      no cells, and 29 Feb in common years, add nothing. The 31 Mar row equals
      the mean row of ../precip_sensitivity/ERA5_avgHoursPerCell_perSeason_*_noPrecip.csv.
      <TYPE> = liquidContaining | liquidOnly | iceOnly.

  ERA5_classAreaPercent_perDay_2014_2025.csv
      area of the region held by each class (%), cos(latitude) weighted, mean
      over seasons. Cumulative hours above are NOT area weighted.

  ERA5_cumulativeHoursPerCell_2014_2025_<TYPE>.png
      the figure drawn from these two tables.
