HOURLY ERA5 LWP, CLOUD TEMPERATURE AND DLR BY SKY STATE AND SURFACE CLASS
=========================================================================

Andrew Buggee, Scripps Institution of Oceanography, UC San Diego
Generated 2026-10-09 by make_era5_hourly_values_csvs.py (repo 39b7a8b-dirty)

Companion to ../era5_hour_count_csv/readME.txt, which describes the ERA5
data, the filters, the surface classes and the plot colours. The hours and
cloud types here are the same ones counted there.


1. LAYOUT
---------
One row per hour, every hour of 1 Oct - 31 Mar, 2014/15
through 2024/25 (48,120 hours, UTC; the time stamp is the
end of the hour for the fluxes). A cell is blank when no grid cell of the class
was in that sky state in that hour.

  ERA5_LWP_perHour_2014_2025_<CLASS>.csv        g m-2
      date, liquid_containing, liquid_only, ice_only
  ERA5_CloudTemp_perHour_2014_2025_<CLASS>.csv  deg C
      date, liquid_containing, liquid_only, ice_only
  ERA5_DLR_perHour_2014_2025_<CLASS>.csv        W m-2
      date, liquid_containing, liquid_only, ice_only, clear_sky,
      clear_sky_calc_liquid_containing, clear_sky_calc_liquid_only,
      clear_sky_calc_ice_only

  <CLASS> = UTQ | Land | OpenOcean | PackIce

  cell_counts/ERA5_nCells_perHour_2014_2025_<CLASS>.csv
      number of grid cells in each sky state in each hour (the cells behind
      each value), and the number of those with a cloud temperature.


2. SKY STATES
-------------
  liquid containing  tcc >= 0.95, LWP or IWP above 0.01 g m-2,
                     tp < 0.05 mm hr-1, IWP/CWP < 0.90
  liquid only        as above with LWP/CWP >= 0.90 (a subset of liquid containing)
  ice only           as above with IWP/CWP >= 0.90
  clear sky          LWP <= 0.01 and IWP <= 0.01 g m-2, tcc < 0.05,
                     tp < 0.05 mm hr-1

CWP = LWP + IWP after setting each path at or below its floor to zero.
Clear sky is rare in ERA5 at UTQ: 496 of 48,120 hours.


3. QUANTITIES
-------------
  LWP        1000 x tclw, total column cloud liquid water (g m-2).
  DLR        msdwlwrf, surface downward longwave, mean over the hour (W m-2).
  clear_sky_calc_<state>
             msdwlwrfcs, ERA5's clear-sky surface downward longwave: the same
             radiation calculation with the clouds removed, for the hours of
             that state (W m-2). DLR - clear_sky_calc is the longwave cloud
             radiative effect at the surface.
  CloudTemp  from ERA5 pressure-level data (temperature, cloud liquid water
             content clwc and cloud ice water content ciwc on 23 levels,
             1000-200 hPa): the condensate-weighted mean temperature of the
             column,

                 T = sum_l (q_l dp_l T_l) / sum_l (q_l dp_l)

             with q = clwc (liquid-weighted) for liquid containing and liquid
             only, and q = ciwc (ice-weighted) for ice only; dp_l is the
             pressure thickness of level l, clipped at the surface. It is the
             temperature of the cloud's liquid (or ice), weighted toward where
             most of it is -- not cloud-base or cloud-top temperature. A column
             with less than 0.001 g m-2 of the weighting species on the
             pressure levels has no value.

             COVERAGE in this version: pressure-level data were available for
             2022/23 (10% of its hours); 2024/25 (100% of its hours).
             CloudTemp is blank for every other hour.


4. ONE VALUE PER HOUR
---------------------
UTQ is the single ERA5 grid cell containing the ARM site (71.25 N, 156.50 W):
its value is that cell's. For Land, OpenOcean and PackIce the value is the
unweighted mean over every grid cell of the class in that sky state in that
hour. A box plot of these hourly values gives each HOUR equal weight; to give
each grid-cell-hour equal weight instead, weight each value by its count in
cell_counts/. OpenOcean has no cells at all in most mid-winter hours, so its
Dec-Mar columns are mostly blank.
