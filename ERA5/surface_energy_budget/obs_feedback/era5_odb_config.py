#!/usr/bin/env python3
"""Station, geometry, and ODB vocabulary for the Barrow assimilation check.

Everything in this module is a CONSTANT or a small pure helper. It exists so
that the fetch step and the analysis step agree on what "Barrow" means and on
how an ODB integer code maps to something readable.

TWO KINDS OF CONSTANT LIVE HERE, AND THEY HAVE DIFFERENT RELIABILITY
====================================================================
1. Geometry (lat/lon of Utqiagvik, the search box). Certain.
2. ODB code tables (varno, reportype, obstype, status bit positions). These are
   transcribed from ECMWF's ODB documentation and are the part most likely to
   be stale or wrong for a given ERA5 stream. Every lookup below therefore
   DEGRADES GRACEFULLY: an unrecognised code is reported as e.g. "varno_247"
   rather than dropped or silently mislabelled, and the analysis script prints
   a census of unknown codes so you can see immediately if a table is wrong.

   Authoritative check, if you have ODB tools:  ``odb header <file.odb>``
   prints the bitfield member names and offsets actually used in the file.

WHY A BOX AND NOT JUST A STATION ID
===================================
Station identifiers are the weak link. In ODB, ``statid@hdr`` is a packed
character string, sometimes blank-padded or right-justified, and a land SYNOP
may be archived as '70026', ' 70026' or '70026   ' depending on the source
feed. An equality test against the wrong padding silently returns zero rows --
which is indistinguishable from "the station was never assimilated", the exact
false negative this whole exercise is meant to avoid.

So the primary selector is a LAT/LON BOX, and the station identifier is only
ever used to LABEL what came back. Ask the archive "what reports exist near
71.3 N, 156.8 W" and let it tell you which identifiers it uses. That also
answers the 70027-vs-70026 question empirically instead of by assumption.
"""

from __future__ import annotations

from typing import Dict, Final, List, Tuple

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

# Utqiagvik (formerly Barrow), Alaska. PABR / Wiley Post-Will Rogers Memorial
# Airport, which is the WMO SYNOP and radiosonde site.
BARROW_LAT_DEG: Final[float] = 71.28
BARROW_LON_DEG: Final[float] = -156.79  # negative east longitude

# The NOAA Barrow Atmospheric Baseline Observatory (BRW) and the ARM North
# Slope of Alaska (NSA) site sit ~5-8 km NE of the airport. A half-degree box
# contains all three, plus Point Barrow itself. Widen with --box-deg if you
# want to sweep in Nuiqsut / Wainwright / Atqasuk for context.
DEFAULT_BOX_HALFWIDTH_LAT_DEG: Final[float] = 0.60
DEFAULT_BOX_HALFWIDTH_LON_DEG: Final[float] = 1.60  # ~57 km at 71 N

# Station identifiers to look for in whatever comes back. These are LABELS, not
# selectors -- see the module docstring. 70026 is the WMO index for
# BARROW/W. POST-W. ROGERS AP (PABR); 70027 is a neighbouring North Slope index
# whose exact identity I have not verified, which is precisely why the code
# reports every statid it finds in the box rather than testing only these two.
STATION_IDS_OF_INTEREST: Final[Tuple[str, ...]] = ("70026", "70027")


def barrow_box(
    half_lat_deg: float = DEFAULT_BOX_HALFWIDTH_LAT_DEG,
    half_lon_deg: float = DEFAULT_BOX_HALFWIDTH_LON_DEG,
) -> Dict[str, float]:
    """Return the lat/lon search box around Utqiagvik, in degrees.

    Longitudes are returned in BOTH conventions (-180..180 and 0..360) because
    ERA5 ODB feedback files are not consistent about which one ``lon@hdr``
    uses, and a query written for the wrong one returns an empty table that
    looks exactly like a real negative result.
    """
    lat_min = BARROW_LAT_DEG - half_lat_deg
    lat_max = BARROW_LAT_DEG + half_lat_deg
    lon_min = BARROW_LON_DEG - half_lon_deg
    lon_max = BARROW_LON_DEG + half_lon_deg
    return {
        "lat_min_deg": lat_min,
        "lat_max_deg": lat_max,
        "lon_min_deg": lon_min,          # negative-east convention
        "lon_max_deg": lon_max,
        "lon_min_360_deg": lon_min + 360.0,   # 0..360 convention
        "lon_max_360_deg": lon_max + 360.0,
    }


# ---------------------------------------------------------------------------
# ODB code tables  (see reliability note in the module docstring)
# ---------------------------------------------------------------------------

# varno@body -- what physical quantity a body row holds.
VARNO_NAMES: Final[Dict[int, str]] = {
    1: "z_geopotential_m2_s2",
    2: "t_upper_air_K",
    3: "u_wind_m_s",
    4: "v_wind_m_s",
    7: "q_specific_humidity_kg_kg",
    29: "rh_relative_humidity_frac",
    39: "t2m_K",
    40: "td2m_K",
    41: "u10m_m_s",
    42: "v10m_m_s",
    58: "rh2m_frac",
    59: "td_dewpoint_K",
    110: "ps_surface_pressure_Pa",
    111: "dd_wind_direction_deg",
    112: "ff_wind_speed_m_s",
    119: "bending_angle_rad",
    128: "apdss_Pa",
}

# The varnos that matter for a surface energy budget question at a land site.
SURFACE_VARNOS: Final[Tuple[int, ...]] = (39, 40, 41, 42, 58, 110)

# obstype@hdr -- the broad observing-system family.
OBSTYPE_NAMES: Final[Dict[int, str]] = {
    1: "synop_land_and_ship",
    2: "airep_aircraft",
    3: "satob_amv",
    4: "dribu_buoy",
    5: "temp_radiosonde",
    6: "pilot_wind_profile",
    7: "satem_satellite_sounder",
    8: "paob",
    9: "scatt_scatterometer",
    10: "limb",
}

# reportype@hdr -- the fine-grained report subtype. These codes are the least
# stable table in this file; treat an unrecognised value as informative, not as
# an error. Left deliberately partial.
REPORTYPE_HINTS: Final[Dict[int, str]] = {
    16001: "synop_land_manual",
    16002: "synop_land_auto",
    16005: "synop_land",
    16022: "ship",
    16045: "temp_land_radiosonde",
    16068: "temp_mobile_or_bufr",
}


def varno_label(varno: int) -> str:
    """Human-readable name for a varno, or ``varno_<n>`` if unrecognised."""
    return VARNO_NAMES.get(int(varno), f"varno_{int(varno)}")


def obstype_label(obstype: int) -> str:
    """Human-readable name for an obstype, or ``obstype_<n>`` if unrecognised."""
    return OBSTYPE_NAMES.get(int(obstype), f"obstype_{int(obstype)}")


def reportype_label(reportype: int) -> str:
    """Best-effort name for a reportype, or ``reportype_<n>`` if unrecognised."""
    return REPORTYPE_HINTS.get(int(reportype), f"reportype_{int(reportype)}")


# ---------------------------------------------------------------------------
# Status bitfields
# ---------------------------------------------------------------------------

# ODB stores usage flags as bitfields. The PREFERRED path in this package is to
# ask MARS/odb to expand the named members for us in SQL ("datum_status.active"
# etc.), which removes any dependence on bit positions. The offsets below are a
# FALLBACK for when a file arrives with the bitfield packed into one integer.
#
# Verify against your own file with ``odb header`` before trusting them.
DATUM_STATUS_BITS: Final[Dict[str, int]] = {
    "active": 0,
    "passive": 1,
    "rejected": 2,
    "blacklisted": 3,
}

REPORT_STATUS_BITS: Final[Dict[str, int]] = {
    "active": 0,
    "passive": 1,
    "rejected": 2,
    "blacklisted": 3,
}

# The four mutually-informative outcomes we ultimately want per observation.
# "active" is the only one that means the observation changed the analysis.
USAGE_CATEGORIES: Final[Tuple[str, ...]] = (
    "active",
    "passive",
    "rejected",
    "blacklisted",
)


def decode_bitfield(value: int, bits: Dict[str, int]) -> Dict[str, bool]:
    """Unpack a packed ODB status integer into named booleans.

    Only used on the fallback path; see the comment above ``DATUM_STATUS_BITS``.
    """
    return {name: bool(int(value) >> offset & 1) for name, offset in bits.items()}


# ---------------------------------------------------------------------------
# The ODB columns we ask for
# ---------------------------------------------------------------------------

# Requesting a narrow column set is not cosmetic: ERA5 conventional feedback for
# one 12-hour window is global, and pulling every column multiplies the transfer
# for no gain. These are the columns the analysis actually reads.
ODB_HEADER_COLUMNS: Final[List[str]] = [
    "andate",       # analysis (window) date
    "antime",       # analysis (window) time
    "date",         # observation date
    "time",         # observation time
    "statid",       # station identifier, packed string
    "lat",
    "lon",
    "stalt",        # station altitude, m
    "obstype",
    "codetype",
    "reportype",
    "report_status",
    "report_event1",
]

ODB_BODY_COLUMNS: Final[List[str]] = [
    "varno",
    "vertco_reference_1",   # pressure or height of the datum
    "obsvalue",
    "fg_depar",             # observation minus first guess  (o - b)
    "an_depar",             # observation minus analysis     (o - a)
    "obs_error",
    "final_obs_error",
    "biascorr",             # applied bias correction
    "biascorr_fg",
    "datum_status",
    "datum_event1",
    "datum_anflag",
]
