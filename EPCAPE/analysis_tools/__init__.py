"""Shared analysis tools used by instruments/ and comparisons/.

products.py   load_product(): one in-memory Dataset per configured product
qc.py         ARM qc_ bit fields -> "bad sample" masks
filters.py    named selection criteria and the filter "funnel" table
units.py      unit conversion driven by each variable's units attribute
stats.py      paired-comparison statistics (bias, RMSD, r, RMA fit)
derived.py    save_derived(): MATLAB-ready netCDF with settings and git commit
"""
