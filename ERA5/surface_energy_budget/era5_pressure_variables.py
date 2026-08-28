#!/usr/bin/env python3
"""ERA5 pressure-level variables for cloud water content, and what they cost.

WHAT IS ACTUALLY NEEDED TO GET ABSOLUTE WATER CONTENTS
=====================================================
ERA5's condensate on pressure levels is *specific* content: kilograms of the
species per kilogram of TOTAL MOIST AIR, where moist air = dry air + vapour +
cloud liquid + cloud ice + rain + snow. Two conversions matter, and they need
different inputs.

(1) LAYER MASS PATH, kg m-2 -- the robust one, needs NO temperature
------------------------------------------------------------------
Under hydrostatic balance the mass of air above unit area between two pressure
surfaces is dp/g, so the mass of species x in that layer is

    W_x = (1/g) * q_x * dp                        [kg m-2]

and the column path is the pressure integral, W_x = (1/g) * INT q_x dp. This
needs only the specific content and the pressure coordinate. It is exact to the
same degree ERA5's own hydrostatic model is, and it is how ERA5's own tclw/tciw
single-level fields are formed -- which makes it the check you should run.

(2) VOLUMETRIC DENSITY, kg m-3 -- needs temperature and humidity
----------------------------------------------------------------
    rho_moist = p / (R_d * T_v)
    rho_x     = q_x * rho_moist                   [kg m-3]

with a virtual temperature that carries both the vapour effect and condensate
loading. ERA5 gives SPECIFIC contents (per kg of moist air), not mixing ratios
(per kg of dry air), and the two take different forms. For specific contents,
starting from p = rho*T*(q_d*R_d + q_v*R_v) with q_d = 1 - q_v - q_c:

    T_v = T * (1 + 0.608*q_v - q_c),   q_c = q_l + q_i + q_r + q_s

The 0.608 is (1 - eps)/eps with eps = R_d/R_v = 0.622. Note the condensate term
is SUBTRACTED: suspended water is mass that adds no partial pressure, so it
makes the air denser. The mixing-ratio form, T*(1 + w_v/eps)/(1 + w_v + w_c), is
the one usually quoted and is NOT the right one to use on these fields.

So density needs T and q_v, plus every condensate species for the loading term.

REQUIRED, OPTIONAL, AND ONE THING THAT DOES NOT EXIST
====================================================
Required for both conversions:

    specific_cloud_liquid_water_content   clwc
    specific_cloud_ice_water_content      ciwc
    specific_rain_water_content           crwc
    specific_snow_water_content           cswc

Required for the volumetric conversion only:

    temperature                           t
    specific_humidity                     q

Strongly recommended, and NOT in the original list:

    fraction_of_cloud_cover               cc

    The specific contents are GRID-BOX MEANS -- they already include the clear
    part of the box. In-cloud content is q_x / cc. A ground-based instrument at
    the ARM site measures in-cloud values along its beam, so comparing ERA5's
    grid-box mean against it without cc compares two different quantities. In an
    Arctic winter, where layer cloud fractions of 0.3-0.6 are common, that is a
    factor-of-two error, not a rounding difference.

Requested, useful, kept:

    geopotential                          z

    Divide by g for geopotential height. Needed if you want profiles against
    altitude rather than pressure, or layer thickness in metres. Not needed for
    either conversion above.

Requested, but REDUNDANT:

    relative_humidity                     r

    Fully determined by t, q and the level pressure. Kept because it was asked
    for and it is convenient, but it is roughly 1/8 of the download for zero new
    information. Drop it with --var-set minimal if the download is too slow.

Requested, but DOES NOT EXIST on this dataset:

    pressure

    On reanalysis-era5-pressure-levels the pressure IS the vertical coordinate.
    Every field is returned on the fixed levels you request, so pressure is the
    'pressure_level' coordinate of the result, not a variable to download.

One thing you need that lives in the OTHER dataset:

    surface_pressure                      sp   (single levels)

    Pressure levels below the surface are filled by extrapolation, not measured,
    and must be masked out before any column integral. That test is p_level <=
    sp. You already download sp in the single-levels archive, so nothing new is
    needed -- convert_specific_to_absolute.py reads it from there.
"""

from __future__ import annotations

from typing import NamedTuple


class PlVar(NamedTuple):
    """One ERA5 pressure-level variable."""

    cds_name: str
    short: str
    units: str
    role: str


# ----------------------------------------------------------------------------
# The registry
# ----------------------------------------------------------------------------
CONDENSATE_VARS: tuple[PlVar, ...] = (
    PlVar("specific_cloud_liquid_water_content", "clwc", "kg kg-1", "condensate"),
    PlVar("specific_cloud_ice_water_content", "ciwc", "kg kg-1", "condensate"),
    PlVar("specific_rain_water_content", "crwc", "kg kg-1", "condensate"),
    PlVar("specific_snow_water_content", "cswc", "kg kg-1", "condensate"),
)

# Needed to turn a specific content into a density.
THERMO_VARS: tuple[PlVar, ...] = (
    PlVar("temperature", "t", "K", "thermodynamic state"),
    PlVar("specific_humidity", "q", "kg kg-1", "thermodynamic state"),
)

# Needed to turn a grid-box mean into an in-cloud value.
CLOUD_FRACTION_VARS: tuple[PlVar, ...] = (
    PlVar("fraction_of_cloud_cover", "cc", "0-1", "cloud geometry"),
)

# Requested; useful for altitude coordinates, not needed for the conversions.
GEOMETRY_VARS: tuple[PlVar, ...] = (
    PlVar("geopotential", "z", "m2 s-2", "geometry"),
)

# Requested; fully determined by t, q and the level pressure.
REDUNDANT_VARS: tuple[PlVar, ...] = (
    PlVar("relative_humidity", "r", "%", "redundant diagnostic"),
)

VARIABLE_SETS: dict[str, tuple[PlVar, ...]] = {
    # Everything needed for BOTH conversions, and nothing else. 6 variables.
    "minimal": CONDENSATE_VARS + THERMO_VARS,
    # Adds the cloud fraction, without which grid-box means cannot be turned
    # into the in-cloud values a ground instrument sees. 7 variables.
    "standard": CONDENSATE_VARS + THERMO_VARS + CLOUD_FRACTION_VARS,
    # Adds geopotential. 8 variables. Only worth the extra request-count if you
    # want altitude coordinates without deriving them.
    "extended": (CONDENSATE_VARS + THERMO_VARS + CLOUD_FRACTION_VARS
                 + GEOMETRY_VARS),
    # As requested, including the redundant relative humidity. 9 variables.
    "requested": (CONDENSATE_VARS + THERMO_VARS + CLOUD_FRACTION_VARS
                  + GEOMETRY_VARS + REDUNDANT_VARS),
}

# 'standard' is the default: geopotential and relative humidity are both
# derivable from what remains, and dropping them is not merely tidy -- it is a
# 33% cut in wall-clock time. At 7 variables x 23 levels a request holds 3 days
# instead of 2, which takes the season from 124 requests to 81. Since CDS queue
# time dominates and is independent of payload, request count IS the runtime.
#
#   z  geopotential       integrate the hypsometric equation upward from the
#                         surface using t and q, with the static single-level
#                         surface geopotential as the reference.
#   r  relative humidity  determined by t, q and the level pressure.
DEFAULT_VAR_SET = "standard"


# ----------------------------------------------------------------------------
# Pressure levels
# ----------------------------------------------------------------------------
# Every level ERA5 offers, hPa, descending in altitude order (highest pressure
# = lowest altitude first).
ALL_LEVELS: tuple[int, ...] = (
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600,
    550, 500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100,
    70, 50, 30, 20, 10, 7, 5, 3, 2, 1,
)

# The level axis MULTIPLIES the request cost: a request costs
# variables x levels x days x 24 fields, so 37 levels makes an 8-variable
# request 37 times more expensive than the same variables on single levels.
# Trimming the top of the atmosphere is by far the cheapest saving available,
# and for Arctic condensate it costs nothing: the winter tropopause sits near
# 250-300 hPa and there is no liquid above it at all.
LEVEL_SETS: dict[str, tuple[int, ...]] = {
    # Surface to 200 hPa. Covers the whole Arctic troposphere with headroom for
    # cirrus above the winter tropopause. 23 levels.
    "troposphere": tuple(p for p in ALL_LEVELS if p >= 200),
    # Surface to 500 hPa. Everything that carries liquid in this region, and
    # roughly half the cost. 16 levels.
    "lower": tuple(p for p in ALL_LEVELS if p >= 500),
    # Surface to 100 hPa, for stratospheric cloud or ozone work. 27 levels.
    "deep": tuple(p for p in ALL_LEVELS if p >= 100),
    "all": ALL_LEVELS,
}

DEFAULT_LEVEL_SET = "troposphere"


def resolve_levels(spec: str) -> tuple[int, ...]:
    """Turn a level-set name or an explicit comma list into hPa values.

    Accepts a name from LEVEL_SETS, or '1000,925,850', or a '1000-500' range
    meaning every ERA5 level within those bounds inclusive.
    """
    spec = str(spec).strip()
    if spec in LEVEL_SETS:
        return LEVEL_SETS[spec]

    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = sorted((int(a), int(b)))
            out.update(p for p in ALL_LEVELS if lo <= p <= hi)
        else:
            value = int(part)
            if value not in ALL_LEVELS:
                raise ValueError(
                    f"{value} hPa is not an ERA5 pressure level. "
                    f"Valid: {', '.join(str(p) for p in ALL_LEVELS)}"
                )
            out.add(value)
    if not out:
        raise ValueError(f"no pressure levels selected from {spec!r}")
    return tuple(sorted(out, reverse=True))


def variables_for(var_set: str) -> list[str]:
    """CDS variable names for a named set."""
    if var_set not in VARIABLE_SETS:
        raise ValueError(
            f"unknown --var-set {var_set!r}; "
            f"choose from {', '.join(VARIABLE_SETS)}"
        )
    return [v.cds_name for v in VARIABLE_SETS[var_set]]


def describe_set(var_set: str) -> str:
    """One line per variable, for the run header."""
    rows = []
    for v in VARIABLE_SETS[var_set]:
        rows.append(f"    {v.short:<6}{v.cds_name:<38}{v.units:<10}{v.role}")
    return "\n".join(rows)
