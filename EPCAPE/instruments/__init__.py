"""Per-instrument loading, filtering and plotting.

One sub-package per instrument whose data products come from that
instrument alone:

    mfrsr/          multifilter rotating shadowband radiometer (MFRSRCLDOD VAP)
    sunphotometer/  Cimel sunphotometer cloud mode (SPHOTCOD VAP)
    mwr/            2-channel microwave radiometer (MWRLOS)

Products that estimate one variable from several instruments belong under
``variables/<variable>/``. Analyses that set several products side by side
belong under ``comparisons/<topic>/``.
"""
