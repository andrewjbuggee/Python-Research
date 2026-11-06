EFFECT OF THE NO-PRECIPITATION FILTER ON PER-CELL CLOUD HOURS
=============================================================

Generated 2026-11-05 by make_era5_precip_filter_sensitivity.py
(repo c252567-dirty).

Data, surface classes and cloud types are as in ../readME.txt. The average is
the per_hour method in ../class_normalized/readME_classNormalized.txt.

Two filters, differing only in the precipitation test:

  noPrecip       overcast (tcc >= 0.95), cloud water above the 0.01 g m-2
                 floors, and tp < 0.05 mm hr-1. These are the existing tables.
  precipAllowed  the same, with the tp test removed.

Every file has rows = seasons and columns = UTQ, Land, Coastal, OpenOcean,
MarginalSeaIce, PackIce. <TYPE> = liquidContaining | liquidOnly | iceOnly.

  ERA5_avgHoursPerCell_perSeason_2014_2025_<TYPE>_noPrecip.csv
  ERA5_avgHoursPerCell_perSeason_2014_2025_<TYPE>_precipAllowed.csv
      average hours per cell (h); last row = mean of the seasons

  ERA5_percentHoursPerCellLost_perSeason_2014_2025_<TYPE>.csv
      100 * (precipAllowed - noPrecip) / precipAllowed, from the tables above

  ERA5_percentCellHoursLost_perSeason_2014_2025_<TYPE>.csv
      the same from raw (cell, hour) counts

The percentage tables end in two summary rows:

  mean_of_seasons     arithmetic mean of the seasonal percentages
  pooled_all_seasons  computed from the sums over all seasons

Phase uses tclw and tciw only, and ERA5 holds falling snow and rain as separate
species, so precipitating columns may read as more liquid-rich than their total
condensate. This effect has not been quantified.
