"""ERA5 single-level variable registry and Arctic spatial regions for the
surface energy budget (SEB) analysis.

The SEB framework follows Sledd et al. (2025), JGR Atmospheres, 130, e2024JD042578,
https://doi.org/10.1029/2024JD042578

Equation (1) of that paper, with all turbulent fluxes defined POSITIVE UPWARD
(away from the surface):

    LWD - LWU + SWD - SWU - SWT - SH - LH + G = M

    LWD  downwelling longwave radiative flux        [W m-2]
    LWU  upwelling longwave radiative flux          [W m-2]
    SWD  downwelling shortwave radiative flux       [W m-2]
    SWU  upwelling shortwave radiative flux         [W m-2]
    SWT  shortwave transmitted below the surface    [W m-2]
    SH   sensible heat flux                         [W m-2]
    LH   latent heat flux                           [W m-2]
    G    subsurface heat flux (conduction+storage)  [W m-2]
    M    melt energy, computed as a residual        [W m-2]

Equation (5), the "net atmospheric flux" form used for model evaluation:

    LWD - LWU + SWD - SWU - SH - LH = M - SWT - G = NA

The radiative forcing term of the Miller et al. (2017) framework used
throughout Sledd et al. (Section 3.2) is:

    forcing = LWD + SWN,    where SWN = SWD - SWU

IMPORTANT SIGN CONVENTION
-------------------------
ERA5 defines every surface flux as POSITIVE DOWNWARD (into the surface). This is
the OPPOSITE of the Sledd et al. convention for the turbulent terms. The mapping
is therefore:

    SH_sledd = -msshf      LH_sledd = -mslhf

so the group ``- SH - LH`` in Equation (1) becomes ``+ msshf + mslhf`` in ERA5
variables. See seb_terms.py, which applies this conversion for you.
"""

from __future__ import annotations

from typing import NamedTuple


# ----------------------------------------------------------------------------
# Variable registry
# ----------------------------------------------------------------------------
class Era5Var(NamedTuple):
    """One ERA5 single-level variable.

    Attributes
    ----------
    cds_name : CDS API request string (validated against the live CDS enum).
    short_name : ERA5/GRIB short name, i.e. the netCDF variable name.
    units : Physical units of the stored field.
    role : Which Equation (1) term (or supporting diagnostic) this serves.
    """

    cds_name: str
    short_name: str
    units: str
    role: str


# --- Group 1: the 17 variables originally requested -------------------------
# Radiative and turbulent flux terms of Equation (1), plus cloud/hydrometeor
# state. All "mean_*" fluxes are hourly-mean RATES in W m-2 (positive downward),
# which avoids the /3600 conversion needed by the accumulated J m-2 variables.
CORE_VARS: tuple[Era5Var, ...] = (
    # Longwave
    Era5Var("mean_surface_downward_long_wave_radiation_flux", "msdwlwrf", "W m-2", "LWD"),
    Era5Var("mean_surface_net_long_wave_radiation_flux", "msnlwrf", "W m-2", "LWD - LWU"),
    # Shortwave
    Era5Var("mean_surface_downward_short_wave_radiation_flux", "msdwswrf", "W m-2", "SWD"),
    Era5Var("mean_surface_net_short_wave_radiation_flux", "msnswrf", "W m-2", "SWN = SWD - SWU"),
    # Turbulent (ERA5 positive DOWNWARD -> negate for Sledd convention)
    Era5Var("mean_surface_sensible_heat_flux", "msshf", "W m-2", "-SH"),
    Era5Var("mean_surface_latent_heat_flux", "mslhf", "W m-2", "-LH"),
    # Cloud fraction
    Era5Var("low_cloud_cover", "lcc", "0-1", "cloud state"),
    Era5Var("medium_cloud_cover", "mcc", "0-1", "cloud state"),
    Era5Var("high_cloud_cover", "hcc", "0-1", "cloud state"),
    Era5Var("total_cloud_cover", "tcc", "0-1", "cloud state"),
    # Column-integrated hydrometeors
    Era5Var("total_column_cloud_ice_water", "tciw", "kg m-2", "cloud state"),
    Era5Var("total_column_cloud_liquid_water", "tclw", "kg m-2", "cloud state"),
    Era5Var("total_column_supercooled_liquid_water", "tcslw", "kg m-2", "cloud state"),
    Era5Var("total_column_snow_water", "tcsw", "kg m-2", "cloud state"),
    Era5Var("total_column_rain_water", "tcrw", "kg m-2", "cloud state"),
    # Precipitation and cloud geometry
    Era5Var("total_precipitation", "tp", "m (1 h accum.)", "surface mass flux"),
    Era5Var("cloud_base_height", "cbh", "m", "cloud state"),
)


# --- Group 2: additions needed to actually close / interpret Equation (1) ----
# Everything here is either (a) required to evaluate a term in Eq. (1) that the
# core list leaves undetermined, or (b) required to restrict the analysis to
# Arctic Ocean sea ice as intended.
ESSENTIAL_ADDITIONS: tuple[Era5Var, ...] = (
    # Skin temperature drives LWU. The entire growth-vs-melt regime distinction
    # in Sledd et al. hinges on whether T_skin is free to vary or is pinned at
    # the melting point, so this is not optional.
    Era5Var("skin_temperature", "skt", "K", "LWU / melt-regime diagnostic"),
    # Required to mask open ocean vs. sea ice. Sledd et al. is entirely over ice.
    Era5Var("sea_ice_cover", "siconc", "0-1", "sea-ice mask"),
    # ERA5's 4-layer sea-ice temperature profile: the only route to a subsurface
    # conduction estimate (G) from ERA5. See caveat in README.
    Era5Var("ice_temperature_layer_1", "istl1", "K", "G (conduction)"),
    Era5Var("ice_temperature_layer_2", "istl2", "K", "G (conduction)"),
    Era5Var("ice_temperature_layer_3", "istl3", "K", "G (conduction)"),
    Era5Var("ice_temperature_layer_4", "istl4", "K", "G (conduction)"),
    # Surface albedo: the control on SW absorption highlighted throughout the
    # paper. Redundant with SWU = SWD - SWN but far more convenient.
    Era5Var("forecast_albedo", "fal", "0-1", "SWU / albedo"),
    # Clear-sky counterparts -> surface cloud radiative effect. The forcing
    # framework is built on cloud-driven LWD variability, so CRE is the natural
    # decomposition of the forcing term.
    Era5Var("mean_surface_downward_long_wave_radiation_flux_clear_sky", "msdwlwrfcs", "W m-2", "CRE"),
    Era5Var("mean_surface_net_long_wave_radiation_flux_clear_sky", "msnlwrfcs", "W m-2", "CRE"),
    Era5Var("mean_surface_downward_short_wave_radiation_flux_clear_sky", "msdwswrfcs", "W m-2", "CRE"),
    Era5Var("mean_surface_net_short_wave_radiation_flux_clear_sky", "msnswrfcs", "W m-2", "CRE"),
    # Water vapour is the other first-order control on Arctic LWD besides cloud.
    Era5Var("total_column_water_vapour", "tcwv", "kg m-2", "LWD control"),
    # Near-surface state controlling the turbulent fluxes. Sledd et al. attribute
    # the SH response to near-surface stability (T_skin - T_2m) and wind speed.
    Era5Var("2m_temperature", "t2m", "K", "SH / stability"),
    Era5Var("2m_dewpoint_temperature", "d2m", "K", "LH / humidity"),
    Era5Var("10m_u_component_of_wind", "u10", "m s-1", "turbulent flux control"),
    Era5Var("10m_v_component_of_wind", "v10", "m s-1", "turbulent flux control"),
    Era5Var("surface_pressure", "sp", "Pa", "air density / thermodynamics"),
    Era5Var("sea_surface_temperature", "sst", "K", "open-ocean surface"),
)


# --- Group 3: useful diagnostics, not required to close Eq. (1) -------------
EXTENDED_ADDITIONS: tuple[Era5Var, ...] = (
    Era5Var("boundary_layer_height", "blh", "m", "stability diagnostic"),
    Era5Var("friction_velocity", "zust", "m s-1", "turbulent flux diagnostic"),
    # Rate forms of precipitation, and the snow/rain split (snowfall resets
    # albedo and insulates the ice; rain does the opposite).
    Era5Var("mean_total_precipitation_rate", "mtpr", "kg m-2 s-1", "surface mass flux"),
    Era5Var("mean_snowfall_rate", "msr", "kg m-2 s-1", "surface mass flux"),
    Era5Var("mean_snowmelt_rate", "msmr", "kg m-2 s-1", "M (land snow only)"),
    # Snow properties. NOTE: in ERA5 these are LAND-tile quantities; ERA5 carries
    # no snow on sea ice (see README caveat). Included for coastal/land contrast.
    Era5Var("snow_depth", "sd", "m water equiv.", "land snow (not sea ice)"),
    Era5Var("snow_density", "rsn", "kg m-3", "land snow (not sea ice)"),
    Era5Var("mean_surface_direct_short_wave_radiation_flux", "msdrswrf", "W m-2", "SW partitioning"),
    Era5Var("mean_top_net_long_wave_radiation_flux", "mtnlwrf", "W m-2", "TOA budget"),
    Era5Var("mean_top_net_short_wave_radiation_flux", "mtnswrf", "W m-2", "TOA budget"),
)


VARIABLE_SETS: dict[str, tuple[Era5Var, ...]] = {
    "core": CORE_VARS,
    "recommended": CORE_VARS + ESSENTIAL_ADDITIONS,
    "extended": CORE_VARS + ESSENTIAL_ADDITIONS + EXTENDED_ADDITIONS,
}

# Flat lookup by short name, for post-processing.
VAR_BY_SHORT_NAME: dict[str, Era5Var] = {
    v.short_name: v for v in VARIABLE_SETS["extended"]
}


# ----------------------------------------------------------------------------
# CDS netCDF name normalisation
# ----------------------------------------------------------------------------
# The current CDS netCDF backend does NOT write the ERA5 GRIB short names for
# the time-mean flux fields. It emits "avg_*" names instead, so a file straight
# from the CDS contains avg_sdlwrf rather than msdwlwrf. Every mapping below was
# verified against the GRIB_paramId attribute in a real downloaded file, not
# inferred from the abbreviations.
#
#   avg_ishf     235033  Time-mean surface sensible heat flux
#   avg_slhtf    235034  Time-mean surface latent heat flux
#   avg_sdswrf   235035  Time-mean surface downward short-wave radiation flux
#   avg_sdlwrf   235036  Time-mean surface downward long-wave radiation flux
#   avg_snswrf   235037  Time-mean surface net short-wave radiation flux
#   avg_snlwrf   235038  Time-mean surface net long-wave radiation flux
#   avg_snswrfcs 235051  ... net short-wave, clear sky
#   avg_snlwrfcs 235052  ... net long-wave, clear sky
#   avg_sdswrfcs 235068  ... downward short-wave, clear sky
#   avg_sdlwrfcs 235069  ... downward long-wave, clear sky
CDS_NC_RENAMES: dict[str, str] = {
    "avg_ishf": "msshf",
    "avg_slhtf": "mslhf",
    "avg_sdswrf": "msdwswrf",
    "avg_sdlwrf": "msdwlwrf",
    "avg_snswrf": "msnswrf",
    "avg_snlwrf": "msnlwrf",
    "avg_snswrfcs": "msnswrfcs",
    "avg_snlwrfcs": "msnlwrfcs",
    "avg_sdswrfcs": "msdwswrfcs",
    "avg_sdlwrfcs": "msdwlwrfcs",
    # Rate/diagnostic fields in the extended set follow the same pattern.
    "avg_tprate": "mtpr",
    "avg_sf": "msr",
    "avg_smr": "msmr",
    "avg_sdirswrf": "msdrswrf",
    "avg_tnlwrf": "mtnlwrf",
    "avg_tnswrf": "mtnswrf",
}


def normalise_names(ds):
    """Rename CDS ``avg_*`` flux fields to their canonical ERA5 short names.

    Safe to call on an already-normalised dataset: only keys actually present
    are renamed.
    """
    present = {old: new for old, new in CDS_NC_RENAMES.items() if old in ds.variables}
    return ds.rename(present) if present else ds


def cds_variable_list(var_set: str) -> list[str]:
    """Return the CDS API ``variable`` list for a named variable set.

    Parameters
    ----------
    var_set : One of "core", "recommended", or "extended".
    """
    if var_set not in VARIABLE_SETS:
        raise ValueError(
            f"Unknown variable set {var_set!r}. Choose from {sorted(VARIABLE_SETS)}."
        )
    return [v.cds_name for v in VARIABLE_SETS[var_set]]


# ----------------------------------------------------------------------------
# Spatial regions
# ----------------------------------------------------------------------------
# CDS "area" is ordered [North, West, South, East] in degrees, with longitude on
# [-180, 180]. A West/East pair of -180/180 requests the full longitude circle.
class Region(NamedTuple):
    """A named lat/lon box, stored in CDS ``area`` order."""

    north_deg: float
    west_deg: float
    south_deg: float
    east_deg: float
    description: str

    def as_area(self) -> list[float]:
        """Return ``[North, West, South, East]`` for the CDS request."""
        return [self.north_deg, self.west_deg, self.south_deg, self.east_deg]


REGIONS: dict[str, Region] = {
    # Everything poleward of the Arctic Circle (66.5 deg N).
    "arctic_circle": Region(90.0, -180.0, 66.5, 180.0, "North of the Arctic Circle (66.5N)"),
    # Tighter pan-Arctic box: excludes most of the Nordic/Bering marginal seas.
    "arctic_70n": Region(90.0, -180.0, 70.0, 180.0, "North of 70N"),
    # Central Arctic pack ice; roughly the MOSAiC drift latitudes.
    "central_arctic": Region(90.0, -180.0, 80.0, 180.0, "Central Arctic, north of 80N"),
    # Requested Utqiagvik/Barrow strip.
    "barrow": Region(80.0, -165.0, 70.0, -150.0, "Barrow strip: 70-80N, 165-150W"),
    # Beaufort/Chukchi shelf and slope, for context around the Barrow strip.
    "beaufort_chukchi": Region(80.0, -170.0, 68.0, -120.0, "Beaufort-Chukchi: 68-80N, 170-120W"),
}

# Edit this to define an ad hoc region, then run with --region custom.
# Alternatively pass --area N W S E on the command line, which overrides this.
CUSTOM_REGION = Region(
    north_deg=85.0,
    west_deg=-30.0,
    south_deg=75.0,
    east_deg=30.0,
    description="Custom region (edit CUSTOM_REGION or pass --area N W S E)",
)


def get_region(name: str) -> Region:
    """Look up a named region, including the editable ``custom`` entry."""
    if name == "custom":
        return CUSTOM_REGION
    if name not in REGIONS:
        raise ValueError(
            f"Unknown region {name!r}. Choose from {sorted(REGIONS) + ['custom']}."
        )
    return REGIONS[name]


def region_names() -> list[str]:
    """All selectable region names, including ``custom``."""
    return sorted(REGIONS) + ["custom"]
