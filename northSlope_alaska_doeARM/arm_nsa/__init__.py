"""arm_nsa: data pipeline for DOE ARM North Slope of Alaska observations.

Built for the Ocean Visions project "Observation-Based Assessment of
Mixed-Phase Cloud Thinning for Reducing Sea Ice Loss in Northern Alaska
Communities" (PI: L. M. Russell). The instrument set and processing recipe
follow Hartig et al. (2026), doi:10.5194/egusphere-2026-2426; see README.md.

Typical use:

    from arm_nsa import download, sonde, mwr, radar, ceilometer, surface
    from arm_nsa.coordinate import build_library, save_library
    from arm_nsa.phase import classify_phase, phase_occurrence

    download.download_datastream("mwr", "2022-01-01", "2022-01-31")
    mwr_ds = mwr.read_mwr("2022-01-01", "2022-01-31")
"""

from . import (  # noqa: F401  (re-exported submodules)
    ceilometer,
    config,
    coordinate,
    credentials,
    download,
    mwr,
    phase,
    qc,
    radar,
    readers,
    shupe_turner,
    sonde,
    surface,
)

__version__ = "0.1.0"
