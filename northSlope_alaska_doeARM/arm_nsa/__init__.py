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

Surface energy budget (the observational twin of the ERA5 SEB pipeline;
see seb.py for the term-by-term account of what NSA C1 measures):

    from arm_nsa import seb, surface, bulk_flux, ground_flux, cloud_water

    qcrad = surface.read_qcrad("2025-12-01", "2025-12-31")
    irt = seb.read_gndirt("2025-12-01", "2025-12-31")
    met = surface.read_met("2025-12-01", "2025-12-31")
    terms = seb.compute_seb_terms(qcrad, gndirt=irt, met=met)   # ERA5-named
    fl = bulk_flux.bulk_fluxes(terms["t_skin_K"], met["temp_2m_c"] + 273.15,
                               met["rh_2m_pct"], met["wspd_m_s"],
                               met["pres_kpa"] * 1e3)             # SH, LH up
"""

from . import (  # noqa: F401  (re-exported submodules)
    bulk_flux,
    ceilometer,
    cloud_water,
    config,
    coordinate,
    credentials,
    download,
    flux_response,
    ground_flux,
    mwr,
    phase,
    qc,
    radar,
    readers,
    seb,
    shupe_turner,
    sonde,
    surface,
)

__version__ = "0.1.0"
