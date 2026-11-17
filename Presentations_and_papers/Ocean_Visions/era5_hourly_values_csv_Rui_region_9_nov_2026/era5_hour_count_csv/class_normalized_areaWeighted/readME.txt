PER-GRID-CELL NORMALISATION OF THE SURFACE-CLASS TABLES
=======================================================

Generated 2026-11-07 by make_era5_class_per_cell_csvs.py
(repo 66e061b), method = per_hour, area-weighted by cos(latitude).

Companion to ../readME.txt, which describes the data, the filters and the
surface classes. Every file here is a normalised version of one of the
cell-hour tables in the folder above, for the five surface classes. UTQ is a
single grid cell, so its tables are already per cell and are not repeated.


WHY NORMALISE
-------------
The tables above count cell-hours summed over every cell of a class. That
total grows with the size of the class, and the size of the three ocean
classes changes every hour as the ice edge moves (their membership is decided
from that hour's sea ice concentration). Over these seasons the classes held:

  Land              1631 -  1631 cells (median 1631); absent in 0 of 48,120 hours
  Coastal           1318 -  1318 cells (median 1318); absent in 0 of 48,120 hours
  OpenOcean            1 -  5388 cells (median 1385); absent in 25,220 of 48,120 hours
  MarginalSeaIce       4 -  3939 cells (median 1132); absent in 0 of 48,120 hours
  PackIce              2 -  6868 cells (median 5580); absent in 46 of 48,120 hours


DEFINITIONS
-----------
For one class and hour t:

  N(t) = grid cells in the class at hour t with valid tcc, tclw and tciw
  n(t) = those cells that also pass every filter (cloudy, condensate,
         not precipitating) and hold the cloud type
  f(t) = n(t) / N(t), the share of the class with that cloud type at t

For a period (day, month or season), T = the number of hours in it during
which the class existed (N(t) > 0); hours without the class add nothing.

  per_hour (used here): divide by the number of cells at EACH hour,
      then sum over the hours.

        avg hours per cell = sum_t f(t)
        fraction of time   = sum_t f(t) / T

      Every hour counts equally, however many cells the class held.
      For Land and Coastal (fixed cells) this is simply the total
      cell-hours divided by the number of cells.

  AREA WEIGHTING: in these files every cell-hour in N(t) and n(t) counts
  as cos(latitude), not as 1, so an average is per unit area of the class
  rather than per grid cell. T still counts hours. "avg hours per cell"
  below therefore means hours at an average point of the class by area.
  The cell counts listed above are plain counts.

"avg hours per cell" is the number of hours an average grid cell of the
class spent under that cloud type, counting only the T hours the class
existed. "fraction of time" is the same as a share of T (0 to 1); multiply
by the full length of the period to express it as hours of a cell that
stayed in the class throughout, which is how the Ocean Visions class figures
present it. Cells are area-weighted by cos(latitude), as those figures are.

A fraction is blank (NaN) where the class did not exist at all in that
period; the matching average is 0.

The ice-only, liquid-only and liquid-containing definitions and the 12-hour
"no phase" note are as in ../readME.txt.


CAUTION: SMALL CLASSES
----------------------
When a class holds few cells, f(t) jumps in large steps (one cell of two is
f = 0.5), so daily values for the marginal ice zone and, in mid-winter,
open ocean are noisy. Monthly and seasonal values average this out.


FILES (5 classes: Land, Coastal, OpenOcean, MarginalSeaIce, PackIce)
-----
Same layouts as the cell-hour tables:

  ERA5_avgHoursPerCell_areaWeighted_perSeason_2014_2025_<CLASS>.csv
  ERA5_fractionOfTime_areaWeighted_perSeason_2014_2025_<CLASS>.csv
      rows = seasons; columns = liquid_containing, liquid_only, ice_only

  ERA5_avgHoursPerCell_areaWeighted_perMonth_2014_2025_<TYPE>_<CLASS>.csv
  ERA5_fractionOfTime_areaWeighted_perMonth_2014_2025_<TYPE>_<CLASS>.csv
      rows = Oct ... Mar; columns = seasons;
      <TYPE> = liquidContaining | liquidOnly | iceOnly

  ERA5_avgHoursPerCell_areaWeighted_perDay_2014_2025_<CLASS>.csv
  ERA5_fractionOfTime_areaWeighted_perDay_2014_2025_<CLASS>.csv
      one row per day; columns = date, liquid_containing, liquid_only, ice_only

  ERA5_hoursClassPresent_perSeason_2014_2025.csv   T per season (column per class)
  ERA5_hoursClassPresent_perMonth_2014_2025_<CLASS>.csv   T per month
  ERA5_hoursClassPresent_perDay_2014_2025.csv      T per day (column per class)
