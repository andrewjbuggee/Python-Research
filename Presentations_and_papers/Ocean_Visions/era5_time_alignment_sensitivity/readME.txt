ERA5 TIME-ALIGNMENT SENSITIVITY OF SKY-STATE DLR STATISTICS
===========================================================

Generated 2026-11-13 by era5_time_alignment_sensitivity.py (repo 1315dbb-dirty)

Cloud-state fields (tcc, tclw, tciw, siconc) are snapshots at the valid
time T; tp and msdwlwrf/msdwlwrfcs are integrated over (T-1 h, T].
Variants (my own construction for this test):

  A  as delivered: state X(T), tp(T), DLR(T-1,T]
  B  centred flux: state X(T), tp & DLR over (T-1,T+1]
  C  centred state: state from mean of X(T-1), X(T); tp(T), DLR(T-1,T]
  D  persistent: same state at T-1 and T; tp(T), DLR(T-1,T]

Seasons 2014 .. 2024, Oct 1 - Mar 31. All variants use
the same 48,113 hours (of 48,120 in window): those with
an archive hour on each side. Thresholds as delivered: tcc >= 0.95, floors 0.01/0.01 g m-2,
tp < 0.05 mm hr-1, phase share 0.9.
Statistics pool (cell, hour) pairs, unweighted. LW CRE = msdwlwrf - msdwlwrfcs.
Medians from 0.25 W m-2 histograms (+-0.125 W m-2).

Columns: cell_hours; cell_hours_change_vs_A_pct; pct_of_A_retained (share of
A's cell-hours in that state also in it under the variant); mean/median DLR
and LW CRE; d_*_vs_A = variant minus A.
