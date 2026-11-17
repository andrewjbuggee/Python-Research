ERA5 CLOUD CELL-HOUR COUNTS, UTQIAGVIK AND THE BARROW DOMAIN
=============================================================

Andrew Buggee, Scripps Institution of Oceanography, UC San Diego
Generated 2026-10-02 by make_era5_hour_count_csvs.py (repo 5c425b0-dirty)

These tables are the ERA5 statistics behind figures 1, 2 and 4 of the Ocean
Visions research update (28 Sept. 2026), extended to the five surface classes
of the surrounding domain.


1. DATA
-------
ERA5 hourly data on single levels (Hersbach et al. 2020, Q. J. R. Meteorol.
Soc., doi:10.1002/qj.3803), 0.25 deg grid, from the Copernicus Climate Data
Store. Domain: 41 x 61 cells, 70-80 N, 165-150 W.
Fields used: total cloud cover (tcc), total column cloud liquid water (tclw),
total column cloud ice water (tciw), total precipitation (tp), sea ice area
fraction (siconc), and the ERA5 land-sea mask (lsm).

UTQ is the single ERA5 cell nearest the DOE ARM North Slope of Alaska central
facility (71.323 N, 156.609 W); its centre is 71.25 N,
156.50 W.

Time: hourly, UTC. A "day" is a UTC calendar day.


2. WHICH HOURS ARE COUNTED
--------------------------
Every cell-hour is tested independently. It is counted when ALL hold:

  a. Season window   1 Oct - 31 Mar (inclusive), seasons 2014/15 through
                     2024/25 (11 seasons).
  b. Cloudy          tcc >= 0.95.
  c. Cloud water     LWP = 1000*tclw and IWP = 1000*tciw (g m-2). Each path
                     at or below its floor (0.01 g m-2 for liquid,
                     0.01 g m-2 for ice) is set to zero, and the
                     hour must keep a condensed water path CWP = LWP + IWP > 0.
  d. No precipitation  tp < 0.05 mm hr-1 (tp is the
                     accumulation over the hour ending at the time stamp).

Each counted hour is then labelled by the share of CWP that is liquid or ice
(using the floored paths):

  liquid containing  IWP/CWP <  0.90   (liquid only + mixed phase)
  liquid only        LWP/CWP >= 0.90   (a subset of liquid containing)
  ice only           IWP/CWP >= 0.90

Liquid containing and ice only together make up every counted hour; mixed
phase = liquid containing - liquid only.


3. SURFACE CLASSES
------------------
Each cell is classified EVERY HOUR, from the static land fraction and that
hour's sea ice concentration:

  Land            lsm >= 0.99 (and siconc undefined or < 0.001)
  Coastal         0.01 < lsm < 0.99 (mixed land/sea cells)
  OpenOcean       lsm <= 0.01 and siconc < 0.05
  MarginalSeaIce  lsm <= 0.01 and 0.05 <= siconc <= 0.95
  PackIce         lsm <= 0.01 and siconc > 0.95

So a cell can move between the three ocean classes as the ice edge moves, and
the number of cells in each ocean class changes from hour to hour. The UTQ
cell is also counted within its own class (Land or Coastal) in the class
tables; it is not a separate area.


4. UNITS: CELL-HOURS (PLAIN COUNTS)
-----------------------------------
Every entry is the number of (grid cell, hour) pairs meeting the criteria.
For UTQ (one cell) this is simply hours. For a surface class it is summed
over every cell that belonged to the class in that hour: no area weighting,
no averaging over cells, no rescaling for missing data (there are no gaps in
the archive over these seasons).

A class total therefore grows with the size of the class. To compare classes,
divide by the class's total valid cell-hours in the same period, given in
normalization/ (same rows and layout; "all valid" = every cell-hour of the
class with finite tcc, tclw and tciw, before filters b-d). That gives the
unweighted fraction of the class's area-time in each state. Note that the
notebook's domain figures use cos(latitude)-weighted fractions, which differ
slightly from these unweighted ones because pack ice lies further north.


5. FILES
--------
Per season (6 files), rows = seasons, columns = liquid_containing,
liquid_only, ice_only:
    ERA5_totalHours_perSeason_2014_2025_<CLASS>.csv

Per month (18 files), rows = Oct ... Mar, columns = seasons
2014/15 ... 2024/25; one file per cloud type:
    ERA5_totalHours_perMonth_2014_2025_<TYPE>_<CLASS>.csv
    <TYPE> = liquidContaining | liquidOnly | iceOnly

Per day (6 files), one row per day of the record
(2005 days, days with no cloud included as 0), columns = date
('1 Oct. 2014'), liquid_containing, liquid_only, ice_only:
    ERA5_totalHours_perDay_2014_2025_<CLASS>.csv

<CLASS> = UTQ | Land | Coastal | OpenOcean | MarginalSeaIce | PackIce

normalization/ : all valid cell-hours per class, per season (one file,
one column per class), per month (one file per class, same layout as the
monthly tables) and per day (one file, one column per class).

The monthly and seasonal tables are sums of the daily ones.


6. RELATION TO THE FIGURES
--------------------------
UTQ totals over all 11 seasons: 17,491 h liquid containing,
3,362 h liquid only, 9,054 h ice only.

Figure 1 (hours per season, UTQ): the red bars are the liquid_containing
column of the UTQ per-season file. The blue ice-only bars additionally
include the few cloudy, non-precipitating hours with NO cloud water above
the floors (12 hours over all seasons), which these tables leave out.

Figure 2 (monthly box plots, UTQ): ERA5 boxes are built from the UTQ
liquidContaining monthly table (one value per season per month). The figure
omits the 8 season-months for which the ARM record is incomplete
(Dec. 2018, Jan. 2020, Oct. 2020, Nov. 2020, Dec. 2020, Nov. 2021, Feb. 2023, Oct. 2023) from both ERA5 and the observations; these tables include
every month.

Figure 4 (cloud duration / hours per day, UTQ): the daily-mode histogram is
the distribution of the liquid_containing column of the UTQ per-day file.

The thresholds are the same for the class tables; the class tables are not
shown in figures 1, 2 or 4.


7. PLOT COLORS
--------------
The colors used in the Ocean Visions figures and the surface-class figures,
so plots made from these tables can match. RGB is given on both the 0-255
and 0-1 scales; "matplotlib" is the color as written in the plotting code
(a name, a hex string, or a grey level between 0 and 1).

Cloud phase -- figures 1 and 2:
  element                        RGB (0-255)     RGB (0-1)              hex      matplotlib note
  ----------------------------------------------------------------------------------------------------------
  liquid containing              (255, 0, 0)     (1.000, 0.000, 0.000)  #ff0000  red        fig 1 red bars; fig 2 ERA5 boxes (solid fill)
  ice only                       (0, 0, 255)     (0.000, 0.000, 1.000)  #0000ff  blue       fig 1 blue bars
  ARM observed mean (fig 1)      (255, 0, 0)     (1.000, 0.000, 0.000)  #ff0000  red        dashed line, drawn on the halo below
    halo under the ARM line      (217, 217, 217) (0.850, 0.850, 0.850)  #d9d9d9  0.85       opacity (alpha) 0.7
  ARM observations (fig 2)       (255, 0, 0)     (1.000, 0.000, 0.000)  #ff0000  red        dotted box outline, white fill

Cloud phase -- working figures only (not in the talk):
  element                        RGB (0-255)     RGB (0-1)              hex      matplotlib note
  ----------------------------------------------------------------------------------------------------------
  liquid only                    (31, 95, 168)   (0.122, 0.373, 0.659)  #1f5fa8  #1f5fa8    SAME color as OpenOcean below -- pick another if both appear
  mixed phase                    (142, 94, 162)  (0.557, 0.369, 0.635)  #8e5ea2  #8e5ea2

Surface classes -- plot_surface_class_timeseries.ipynb and the talk:
  element                        RGB (0-255)     RGB (0-1)              hex      matplotlib note
  ----------------------------------------------------------------------------------------------------------
  Land                           (140, 109, 79)  (0.549, 0.427, 0.310)  #8c6d4f  #8c6d4f    labelled 'Land' in the figures
  Coastal                        (224, 130, 20)  (0.878, 0.510, 0.078)  #e08214  #e08214    labelled 'Coastal (mixed)' in the figures
  OpenOcean                      (31, 95, 168)   (0.122, 0.373, 0.659)  #1f5fa8  #1f5fa8    labelled 'Open ocean' in the figures
  MarginalSeaIce                 (78, 179, 211)  (0.306, 0.702, 0.827)  #4eb3d3  #4eb3d3    labelled 'Marginal ice zone' in the figures
  PackIce                        (154, 165, 173) (0.604, 0.647, 0.678)  #9aa5ad  #9aa5ad    labelled 'Sea ice' in the figures
  ARM site (my Py notebook)      (17, 17, 17)    (0.067, 0.067, 0.067)  #111111  #111111    dashed line

Figure 4:
  element                        RGB (0-255)     RGB (0-1)              hex      matplotlib note
  ----------------------------------------------------------------------------------------------------------
  ARM cell / cloud duration      (18, 57, 94)    (0.071, 0.224, 0.369)  #12395e  #12395E    duration histogram and its axis
  cloud-level wind speed         (42, 157, 143)  (0.165, 0.616, 0.561)  #2a9d8f  #2a9d8f    wind histogram and its axis
