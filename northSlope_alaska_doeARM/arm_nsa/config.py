"""Central configuration for the ARM NSA mixed-phase cloud pipeline.

Everything site-, datastream-, or analysis-constant-related lives here so the
science modules stay free of magic numbers. The datastream registry follows
two source papers:

* Hartig et al. (2026), "Cloud liquid water path at the North Slope of Alaska
  is largely insensitive to local meteorology in Arctic winter"
  (doi:10.5194/egusphere-2026-2426), hereafter "Hartig26" -- the sonde / KAZR /
  MWRRET / ceilometer set and its processing recipe (role="hartig26-core").
* Bertrand et al. (2025), "Increasing wintertime cloud opacity increases
  surface longwave radiation at a long-term Arctic observatory" (Nat. Commun.
  16, 9135, doi:10.1038/s41467-025-64441-8), hereafter "Bertrand25" -- the
  radiation-environment set: QCRAD broadband fluxes, MET/METTWR surface
  meteorology, and the Shupe-Turner multi-sensor cloud microphysics product
  whose vertically resolved hydrometeor phase classification underpins the
  clear / ice-only / mixed-phase / liquid-only scene analysis
  (role="bertrand25-core").

Entries with role="extension" (MPL cloud mask, interpolated sonde) are
registered for convenience but have no dedicated processing module yet.

ARM datastream naming convention: <site><instrument><facility>.<level>, e.g.
"nsamwrret1liljclouC1.c2" = NSA site, MWRRET v1 "liljclou" retrieval, Central
Facility C1 (Utqiagvik/Barrow), data level c2 (highest-QC value-added product).
Levels: a0/a1 raw, b1 calibrated + QC checks, c0/c1/c2 value-added products.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Site
# ---------------------------------------------------------------------------

SITE_CODE = "nsa"
FACILITY_CODE = "C1"  # Central Facility, Utqiagvik (formerly Barrow), AK
SITE_LAT_DEG = 71.323  # deg N
SITE_LON_DEG = -156.609  # deg E
SITE_ALT_M = 8.0  # site elevation above mean sea level [m]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Layout:
#   data/raw/<datastream>/<datastream>.YYYYMMDD.HHMMSS.(nc|cdf)   as downloaded
#   data/processed/...                                            pipeline output
# Override the root with the ARM_NSA_DATA_ROOT environment variable if the
# data should live on scratch/external storage instead of inside the repo.

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("ARM_NSA_DATA_ROOT", _REPO_ROOT / "data"))
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
FIGURE_DIR = _REPO_ROOT / "figures"

# ---------------------------------------------------------------------------
# Analysis period and season (Hartig26 Sect. 2)
# ---------------------------------------------------------------------------

# Full multi-instrument overlap period used by Hartig26.
ANALYSIS_START = "2011-11-12"
ANALYSIS_END = "2023-12-31"

# "Extended winter": November through March, when Arctic cloud fraction is in
# its annual minimum plateau (Shupe et al. 2011, cited in Hartig26 Sect. 2).
WINTER_MONTHS = (11, 12, 1, 2, 3)

# ---------------------------------------------------------------------------
# Sonde processing constants (Hartig26 Sect. 2)
# ---------------------------------------------------------------------------

# Common vertical grid for interpolated soundings: 8 m to 12,000 m every 5 m.
SONDE_GRID_BOTTOM_M = 8.0
SONDE_GRID_TOP_M = 12_000.0
SONDE_GRID_STEP_M = 5.0

# A sounding must report data up to at least this height to be usable.
SONDE_MIN_TOP_M = 1_000.0

# Saturated-layer detection: RH (w.r.t. liquid) threshold, gap-filling, and
# minimum layer depth. Hartig26 uses 95% (not 100%) because the Vaisala RS41
# combined RH uncertainty is ~3% and comparison against ceilometer cloud bases
# supports the lower threshold (their Fig. A1a; also Silber et al. 2020, 2021).
SATURATION_RH_THRESHOLD_PCT = 95.0
SATURATED_LAYER_MAX_GAP_M = 30.0  # fill sub-threshold gaps <= this depth
SATURATED_LAYER_MIN_DEPTH_M = 30.0  # discard layers thinner than this

# ---------------------------------------------------------------------------
# KAZR radar processing constants (Hartig26 Sect. 2)
# ---------------------------------------------------------------------------

# Common vertical grid for radar reflectivity: 105 m to 12,000 m every 30 m.
RADAR_GRID_BOTTOM_M = 105.0
RADAR_GRID_TOP_M = 12_000.0
RADAR_GRID_STEP_M = 30.0

# Exclude radar gates with signal-to-noise ratio below this value [dB].
RADAR_MIN_SNR_DB = -13.0

# Ice water content from reflectivity: IWC = a * Ze**b, IWC in g/m^3 and Ze in
# mm^6/m^3 (linear units, NOT dBZ). a matches the SHEBA winter average (Shupe
# et al. 2005); b is the average suggested by Matrosov (1999).
IWC_PREFACTOR_A = 0.1
IWC_EXPONENT_B = 0.63

# Clear-sky flag: a profile is "clear" when >= 99% of gates below 10 km have no
# detected reflectivity AND no contiguous detected region is deeper than 100 m.
CLEAR_SKY_MAX_HEIGHT_M = 10_000.0
CLEAR_SKY_MIN_CLEAR_FRACTION = 0.99
CLEAR_SKY_MAX_LAYER_DEPTH_M = 100.0

# ---------------------------------------------------------------------------
# Microwave radiometer constants (Hartig26 Sect. 2)
# ---------------------------------------------------------------------------

# LWP below this value is indistinguishable from clear sky given retrieval
# scatter (theoretical uncertainty ~25 g/m^2, but clear-sky retrievals cluster
# within a few g/m^2 of zero). Hartig26 always separates 0-10 g/m^2 cases.
LWP_CLEAR_SKY_THRESHOLD_G_M2 = 10.0

# ---------------------------------------------------------------------------
# Shupe-Turner cloud phase classification (Bertrand25 Methods)
# ---------------------------------------------------------------------------

# The MICROBASE2SHUPETURN product (Shupe 2007 classifier; Shupe et al. 2015)
# reports a per-volume hydrometeor phase code in `CloudPhaseMask`. The code
# groupings below are copied from the Bertrand25 analysis code (Zenodo
# doi:10.5281/zenodo.15786066, prep_basecase_nb44.py) for reproducibility:
#   liquid volumes: 3 (liquid), 5 (liquid + drizzle)
#   ice volumes:    1 (ice), 2 (snow)
#   mixed volumes:  7 (mixed-phase)
#   clear:          0
# Codes not listed (4 = drizzle, 6 = rain, and anything else) are liquid
# precipitation categories that Bertrand25 leaves out of all three groups --
# they are vanishingly rare in Arctic winter. Profiles containing ONLY such
# codes are classified "other_hydrometeor" here rather than polluting "clear".
ST_CLEAR_CODE = 0
ST_ICE_CODES = (1, 2)
ST_LIQUID_CODES = (3, 5)
ST_MIXED_CODES = (7,)

# Approximate availability of the Shupe-Turner product at NSA C1: Bertrand25
# uses "13 years (2004-2019) of comprehensive cloud measurements" (gaps exist
# around radar/lidar upgrades). Query the archive for exact file coverage.
SHUPETURN_APPROX_START = "2004-01-01"
SHUPETURN_APPROX_END = "2019-12-31"

# Bertrand25 restricts to December-March (their "winter"); Hartig26 uses the
# extended November-March season (WINTER_MONTHS above). The phase/radiation
# analysis script defaults to WINTER_MONTHS for project-wide consistency and
# takes --months 12,1,2,3 to reproduce Bertrand25 exactly.
BERTRAND_WINTER_MONTHS = (12, 1, 2, 3)

# ---------------------------------------------------------------------------
# THERMOCLDPHASE cloud phase VAP (routine replacement for the PI-only product)
# ---------------------------------------------------------------------------

# Archive coverage per ARM Data Discovery (checked 2026-07-27); the two data
# levels overlap, and c0 -- despite being the "intermediate" level -- currently
# runs much later than c1:
#   nsathermocldphaseC1.c1: 2011-11-11 .. 2020-01-31
#   nsathermocldphaseC1.c0: 2014-02-09 .. 2026-01-20
THERMOCLDPHASE_C1_START = "2011-11-11"
THERMOCLDPHASE_C1_END = "2020-01-31"
THERMOCLDPHASE_C0_START = "2014-02-09"
THERMOCLDPHASE_C0_END = "2026-01-20"

# Pixel-level phase codes, read from the flag_values / flag_meanings attributes
# of cloud_phase_mplgr / cloud_phase_hsrl in nsathermocldphaseC1.c0 files. They
# are NOT tabulated anywhere in the VAP report (all 17 pages of
# DOE/SC-ARM-TR-325 were checked -- Table 4 gives variable names only), so the
# file attributes are the authority.
#
# !! THESE DIFFER FROM THE ST_*_CODES ABOVE. Liquid and ice are swapped and
# snow moves from 2 to 7:
#     THERMOCLDPHASE: 0 clear, 1 liquid, 2 ice, 3 mixed, 4 drizzle,
#                     5 liquid_drizzle, 6 rain, 7 snow, 8 unknown
#     Shupe-Turner:   0 clear, 1 ice, 2 snow, 3 liquid, 4 drizzle,
#                     5 liquid+drizzle, 6 rain, 7 mixed
# Applying ST_LIQUID_CODES / ST_ICE_CODES to this product would silently
# exchange the liquid and ice populations. Keep the two sets separate.
THERMO_CLEAR_CODE = 0
THERMO_LIQUID_CODES = (1,)  # liquid only
THERMO_ICE_CODES = (2,)  # ice only
THERMO_MIXED_CODES = (3,)
THERMO_UNKNOWN_CODE = 8
# Precipitating liquid (drizzle, liquid+drizzle, rain) and frozen precipitation
# (snow) -- excluded from the strict "pure" definitions above, available for an
# inclusive one.
THERMO_LIQUID_PRECIP_CODES = (4, 5, 6)
THERMO_FROZEN_PRECIP_CODES = (7,)
THERMO_PHASE_MEANINGS = {
    0: "clear_sky",
    1: "liquid",
    2: "ice",
    3: "mixed_phase",
    4: "drizzle",
    5: "liquid_drizzle",
    6: "rain",
    7: "snow",
    8: "unknown",
}
# The VAP's own layer rule is driven by frc_ice, the fraction of ICE-CONTAINING
# pixels within a layer -- ice, mixed-phase, OR snow (report Sect. 2, following
# Wang et al. 2024) -- with 0.1 and 0.9 separating liquid / mixed / ice layers.
# The files publish only the resulting layer label, never frc_ice itself, so any
# finer split of the mixed range must recompute the fraction from the pixel mask.
THERMO_ICE_CONTAINING_CODES = (2, 3, 7)
THERMO_FRC_ICE_LIQUID_MAX = 0.1
THERMO_FRC_ICE_ICE_MIN = 0.9

# The per-LAYER fields (cloud_phase_layer_*) use a SHORTER, different scale:
# 0 clear_sky, 1 liquid, 2 ice, 3 mixed_phase -- assigned by the fraction of
# ice-containing pixels in the layer (report Sect. 2: liquid < 0.1, mixed
# 0.1-0.9, ice > 0.9). Do not mix these with the pixel codes above.
THERMO_LAYER_PHASE_MEANINGS = {0: "clear_sky", 1: "liquid", 2: "ice", 3: "mixed_phase"}

# Vertical grid of the pixel-level fields: 596 levels, 0.16-18.01 km, 30 m
# spacing. NOTE the height coordinate is stored in KILOMETRES, not metres.
THERMO_HEIGHT_UNITS = "km"
THERMO_TIME_STEP_S = 30.0

# Companion data DOI for the NSA record (report reference list):
# Zhang & Levin 2024, THERMOCLDPHASE 2017-03-01 to 2024-07-01, NSA C1,
# doi:10.5439/1871014

# ---------------------------------------------------------------------------
# ACRED cloud retrieval ensemble (DOE/SC-ARM-TR-099, Zhao et al. 2011)
# ---------------------------------------------------------------------------

# ACRED assembles NINE independent ground-based cloud retrievals onto one grid
# so the spread between them estimates retrieval uncertainty. At NSA C1 the
# ensemble members are MICROBASE, SHUPE_TURNER, WANG, and DENG (their Table 1)
# -- i.e. this product carries Shupe-Turner-derived microphysics through a
# routine, downloadable datastream, though only LWC/IWC/r_e/LWP/IWP, NOT the
# CloudPhaseMask that the scene classification needs.
ACRED_APPROX_START = "1999-01-01"  # nsaacredC1.c1 per ARM Data Discovery
ACRED_APPROX_END = "2008-12-31"

# Hourly means on 512 layers of 45 m. Every geophysical variable comes as a
# mean, a standard deviation, and a QC flag, and carries a retrieval-method
# dimension (3-D fields vary with time/height/method, 2-D with time/method).
ACRED_N_LAYERS = 512
ACRED_LAYER_THICKNESS_M = 45.0
ACRED_MISSING_VALUE = -9999.0

# ACRED QC flags are NOT ARM's bit-packed convention -- they encode what
# fraction of the sub-hourly samples were valid (report Sect. 4.5), so qc.py's
# apply_qc() does not apply to this product:
#    0 = more than 50% of the data valid over the hour
#   -1 = 30-50% valid      -2 = 10-30% valid
#   -3 = under 10% valid   -4 = missing data point
ACRED_QC_MEANINGS = {
    0: "more than 50% valid over the hour",
    -1: "30-50% valid",
    -2: "10-30% valid",
    -3: "less than 10% valid",
    -4: "missing data point",
}
ACRED_QC_GOOD = (0,)

# ---------------------------------------------------------------------------
# Sonde-coordinated averaging (Hartig26 Sect. 2)
# ---------------------------------------------------------------------------

# For each radiosonde launch, radar/MWR/ceilometer data are averaged over the
# hour FOLLOWING the launch time.
COORDINATION_WINDOW_S = 3600.0

# Radar hourly means are kept only if at least this fraction of radar time
# samples inside the window actually reported reflectivity.
RADAR_MIN_COVERAGE_FRACTION = 0.5

# ---------------------------------------------------------------------------
# Datastream registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatastreamSpec:
    """Description of one ARM datastream used by the pipeline.

    Attributes
    ----------
    key:
        Short pipeline-internal name ("sonde", "kazr", ...).
    datastreams:
        ARM datastream names, in the order they should be tried. KAZR has
        three entries because the product name changed twice over 2011-2023
        (Hartig26 Table 1); for a given date range, files may exist under any
        of them, and the downloader queries each one.
    description:
        Human-readable summary (instrument and role).
    variables:
        Mapping from the pipeline's canonical variable name to a tuple of
        candidate names inside the netCDF files, tried in order. Candidate
        lists absorb historical renames between product versions. If none of
        the candidates is present, readers raise an error that lists what IS
        in the file, so extending a candidate list is a one-line fix here.
    role:
        "hartig26-core" for Hartig26 Table 1 instruments, "bertrand25-core"
        for the Bertrand25 radiation-environment set (QCRAD, MET/METTWR,
        Shupe-Turner microphysics), "extension" for registered-but-unwrapped
        convenience datastreams.
    """

    key: str
    datastreams: Tuple[str, ...]
    description: str
    variables: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    role: str = "hartig26-core"


DATASTREAMS: Dict[str, DatastreamSpec] = {
    # -- Hartig26 Table 1 -----------------------------------------------------
    "sonde": DatastreamSpec(
        key="sonde",
        datastreams=("nsasondewnpnC1.b1",),
        description=(
            "Balloon-borne sounding system (Vaisala radiosonde), 0-4 launches "
            "per day; T, RH, p, winds at 5-8 m native vertical resolution."
        ),
        variables={
            # canonical -> candidates in file (units handled by readers)
            "pres_hpa": ("pres",),
            "tdry_c": ("tdry", "temp"),
            "dp_c": ("dp",),
            "rh_pct": ("rh",),
            "wspd_m_s": ("wspd",),
            "wdir_deg": ("deg", "wdir"),
            "u_wind_m_s": ("u_wind",),
            "v_wind_m_s": ("v_wind",),
            "alt_m": ("alt",),
        },
    ),
    "kazr": DatastreamSpec(
        key="kazr",
        datastreams=(
            # General ("ge") mode, co-polarized. Product renamed twice:
            "nsakazrcorgeC1.c1",  # ~2011-2014
            "nsakazrcorgeC1.c0",  # ~2014-2019
            "nsakazrcfrcorgeC1.c0",  # ~2019-2023 (KAZR-CFR reprocessing)
        ),
        description=(
            "Ka-band (35 GHz) zenith-pointing Doppler cloud radar, general "
            "mode; reflectivity profiles at ~4-5 s / 30 m resolution."
        ),
        variables={
            "reflectivity_dbz": ("reflectivity", "reflectivity_copol"),
            "snr_db": (
                "signal_to_noise_ratio",
                "signal_to_noise_ratio_copol",
                "snr",
                "snr_copol",
            ),
            "range_m": ("range",),
        },
    ),
    "mwr": DatastreamSpec(
        key="mwr",
        datastreams=("nsamwrret1liljclouC1.c2",),
        description=(
            "Microwave radiometer retrieval (MWRRET v1, Turner et al. 2007 "
            "physical retrieval): best-estimate LWP and PWV from 23.8/31.4 "
            "GHz brightness temperatures, ~20-30 s resolution."
        ),
        variables={
            "lwp_g_m2": ("be_lwp", "lwp", "liq"),
            "pwv_cm": ("be_pwv", "pwv", "vap"),
        },
    ),
    "ceil": DatastreamSpec(
        key="ceil",
        datastreams=("nsaceilC1.b1",),
        description=(
            "Vaisala CL31 laser ceilometer; lowest cloud base height at 15 s "
            "/ 10 m resolution (proprietary Vaisala detection)."
        ),
        variables={
            "first_cbh_m": ("first_cbh",),
            "detection_status": ("detection_status",),
        },
    ),
    # -- Bertrand25 radiation-environment set ---------------------------------
    "met": DatastreamSpec(
        key="met",
        datastreams=("nsametC1.b1",),
        description=(
            "Surface meteorology (MET): 1-min 2-m air temperature and "
            "humidity, winds, pressure. Available from late 2003; see "
            "'mettwr' for 1998-2003. Bertrand25 surface-warming record; "
            "source for the surface-temperature plots."
        ),
        variables={
            "temp_2m_c": ("temp_mean",),
            "rh_2m_pct": ("rh_mean",),
            "wspd_m_s": ("wspd_vec_mean", "wspd_arith_mean"),
            "wdir_deg": ("wdir_vec_mean",),
            "pres_kpa": ("atmos_pressure",),
        },
        role="bertrand25-core",
    ),
    "qcrad": DatastreamSpec(
        key="qcrad",
        datastreams=("nsaqcrad1longC1.c2", "nsaqcrad1longC1.c1"),
        description=(
            "QCRAD1LONG value-added product: quality-controlled broadband "
            "surface radiative fluxes (down/up SW and LW), 1998-present. "
            "Bertrand25's 26-year longwave record (they merge .c2 with .c1 "
            "to fill gaps -- the registry lists both, downloader gets both)."
        ),
        variables={
            "swdn_w_m2": (
                "BestEstimate_down_short_hemisp",
                "down_short_hemisp",
            ),
            "lwdn_w_m2": (
                "down_long_hemisp_shaded",
                "down_long_hemisp_shaded1",
                "down_long_hemisp",
            ),
            "swup_w_m2": ("up_short_hemisp",),
            "lwup_w_m2": ("up_long_hemisp",),
        },
        role="bertrand25-core",
    ),
    "shupeturn": DatastreamSpec(
        key="shupeturn",
        datastreams=("nsamicrobase2shupeturnC1.c1",),
        description=(
            "Shupe-Turner multi-sensor cloud microphysics (MICROBASE2 "
            "framework; Shupe 2007 phase classifier, Shupe et al. 2015): "
            "time-height hydrometeor phase classification plus LWC/IWC "
            "retrievals from KAZR + MPL depolarization + ceilometer + MWR + "
            "radiosondes. ~1-min resolution, approx. 2004-2019. The phase "
            "source for the clear/ice/mixed/liquid scene analysis "
            "(Bertrand25). Bertrand25 used the variable-coefficient variant."
        ),
        variables={
            "phase_code": ("CloudPhaseMask", "cloud_phase_mask", "CloudPhase"),
            "cloud_fraction": ("Avg_CloudFraction", "cloud_fraction"),
            "lwc_g_m3": ("Avg_Retrieved_LWC", "Avg_LWC", "lwc"),
            "iwc_g_m3": ("Avg_Retrieved_IWC", "Avg_IWC", "iwc"),
            "lwp_g_m2": ("Avg_LWP", "lwp"),
            "iwp_g_m2": ("Avg_IWP", "iwp"),
        },
        role="bertrand25-core",
    ),
    "thermocldphase": DatastreamSpec(
        key="thermocldphase",
        datastreams=(
            # Both levels are registered so the full 2011-2026 span is
            # reachable; they OVERLAP 2014-2020 (see the coverage constants
            # above). The readers concatenate every file of a spec and then
            # de-duplicate by timestamp, so a range inside the overlap silently
            # mixes two data levels -- pass a single raw datastream name to the
            # downloader/readers (both accept one, e.g.
            # "nsathermocldphaseC1.c1") when a run must be reproducible.
            "nsathermocldphaseC1.c1",
            "nsathermocldphaseC1.c0",
        ),
        description=(
            "THERMOCLDPHASE VAP (Zhang, Levin, Shupe & Goldberger 2025, "
            "DOE/SC-ARM-TR-325; doi:10.5439/3022391): vertically resolved "
            "thermodynamic cloud phase at 30 s / 30 m, classifying each pixel "
            "as liquid, drizzle, liquid+drizzle, rain, ice, snow, mixed-phase, "
            "unknown, or clear, plus a per-layer phase (liquid/mixed/ice) for "
            "up to 10 ARSCL cloud layers. Built from MPL or HSRL backscatter "
            "and depolarization, ARSCL radar moments, MWRRET LWP, and "
            "INTERPSONDE temperature. Applies the SAME Shupe (2007) multi-"
            "sensor classifier as the Shupe-Turner product, so this is the "
            "routine, ARM-Live-downloadable substitute for the PI-only "
            "'shupeturn' key -- but the two use different variable names AND "
            "apparently different integer phase codes, so they are not "
            "drop-in interchangeable."
        ),
        variables={
            # Table 4 of DOE/SC-ARM-TR-325. The MPL-gradient variant is listed
            # first because MPL runs at every site for the whole record, while
            # ARM operates only three HSRLs and the report states the HSRL
            # fields are simply absent where no HSRL is deployed. The two agree
            # closely (report Sect. 5); read "cloud_phase_hsrl" directly from
            # the file when the calibrated-lidar version is specifically wanted.
            "phase_code": ("cloud_phase_mplgr", "cloud_phase_hsrl"),
        },
        role="extension",
    ),
    "acred": DatastreamSpec(
        key="acred",
        datastreams=("nsaacredC1.c1",),
        description=(
            "ACRED, the ARM Cloud Retrieval Ensemble Data Set (Zhao et al. "
            "2011, DOE/SC-ARM-TR-099; doi:10.5439/1995948): NINE independent "
            "ground-based cloud microphysical retrievals assembled onto one "
            "common grid -- hourly means, 512 layers of 45 m -- so the spread "
            "across members estimates retrieval uncertainty rather than "
            "asserting one answer. Carries liquid/ice effective radius, LWC/"
            "IWC, LWP/IWP, liquid/ice optical depth, and cloud fraction, each "
            "as a mean, a standard deviation, and a QC flag. At NSA C1 the "
            "members are MICROBASE, SHUPE_TURNER, WANG, and DENG, so the "
            "Shupe-Turner microphysics are reachable here through a routine "
            "datastream -- but ACRED carries NO phase mask, so it cannot "
            "replace 'shupeturn'/'thermocldphase' for scene classification. "
            "Flagged by ARM as evaluation data; NSA coverage 1999-2008, and "
            "the SHUPE_TURNER member only spans roughly 2004-2007 (report "
            "Table 1). Note the retrieval-method dimension: these variables "
            "are NOT plain (time, height) fields."
        ),
        variables={
            # Names confirmed against the ARM Data Discovery variable list for
            # nsaacredC1.c1 -- NOT from DOE/SC-ARM-TR-099, which describes
            # ACRED's contents only in prose and never tabulates netCDF names
            # (its "lwc_orig"/"lwp_layer_orig" identifiers are pseudocode for
            # the INPUT retrievals, not ACRED's own fields). Note "liquid_re",
            # not the "liq_re" the surrounding code's abbreviations would
            # suggest. Canonical names deliberately match the "shupeturn"
            # entry so the two products stay comparable field-for-field.
            #
            # Two caveats before using these:
            #  * UNITS ARE UNVERIFIED. The g/m^3 and g/m^2 suffixes follow this
            #    registry's convention and the usual ARM one; the report does
            #    not state ACRED's units. Check the `units` attribute on first
            #    read and rename here if they differ.
            #  * These are ACRED's hourly MEANS. Per report Sect. 2 every
            #    variable also ships a standard deviation and a QC flag, and
            #    the files additionally hold liquid/ice optical depth and
            #    cloud fraction; those names are not yet confirmed, so they are
            #    left out rather than guessed. Every field carries the
            #    retrieval-method dimension, and missing values are -9999 under
            #    the non-ARM QC convention in ACRED_QC_MEANINGS above
            #    (qc.apply_qc does NOT apply to this product).
            "liquid_re_um": ("liquid_re",),
            "ice_re_um": ("ice_re",),
            "lwc_g_m3": ("lwc",),
            "iwc_g_m3": ("iwc",),
            "lwp_g_m2": ("lwp",),
            "iwp_g_m2": ("iwp",),
        },
        role="extension",
    ),
    "mettwr": DatastreamSpec(
        key="mettwr",
        datastreams=("nsamettwrC1.b1",),
        description=(
            "Surface and Tower Meteorological Instrumentation (METTWR): the "
            "MET predecessor covering ~1998-2003 (tower levels 2/10/20/40 m). "
            "Needed only to extend surface temperature records before MET; "
            "Bertrand25 splices METTWR + MET for their 26-year record."
        ),
        variables={
            # Tower products have per-level names; candidates cover the
            # 2-m level under the namings seen across ARM tower ingests.
            "temp_2m_c": ("temp_mean_2m", "temp_2m_mean", "T2m_mean", "temp_mean"),
            "rh_2m_pct": ("rh_mean_2m", "rh_2m_mean", "rh_mean"),
        },
        role="bertrand25-core",
    ),
    # -- Registered extensions (no dedicated module yet) ----------------------
    "mplcmask": DatastreamSpec(
        key="mplcmask",
        datastreams=("nsamplcmask1zwangC1.c1",),
        description=(
            "Micropulse lidar cloud mask (Wang): cloud detection profiles "
            "used by Bertrand25 (merged with the ceilometer) for total cloud "
            "cover. Registered for download convenience; no custom reader."
        ),
        variables={
            "cloud_mask": ("cloud_mask", "cloud_mask_tsi"),
            "cloud_base_m": ("cloud_base",),
            "num_cloud_layers": ("num_cloud_layers",),
        },
        role="extension",
    ),
    "interpsonde": DatastreamSpec(
        key="interpsonde",
        datastreams=("nsainterpolatedsondeC1.c1",),
        description=(
            "INTERPOLATEDSONDE value-added product: radiosonde thermodynamic "
            "profiles interpolated to a continuous 1-min time grid (used by "
            "Bertrand25 as RRTM input). Registered for download convenience; "
            "no custom reader."
        ),
        variables={
            "temp_c": ("temp",),
            "rh_pct": ("rh",),
            "pres_hpa": ("bar_pres", "pres"),
        },
        role="extension",
    ),
}


def get_spec(key: str) -> DatastreamSpec:
    """Return the DatastreamSpec for a pipeline key ('sonde', 'kazr', ...)."""
    try:
        return DATASTREAMS[key]
    except KeyError as err:
        valid = ", ".join(sorted(DATASTREAMS))
        raise KeyError(f"Unknown datastream key '{key}'. Valid keys: {valid}") from err


def raw_dir_for(datastream: str) -> Path:
    """Directory where raw files for an ARM datastream are stored locally."""
    return RAW_DATA_DIR / datastream


def set_data_root(root: "str | os.PathLike[str]") -> Path:
    """Repoint the pipeline's data tree at `root` at runtime.

    Recomputes DATA_ROOT / RAW_DATA_DIR / PROCESSED_DATA_DIR from `root` and
    rebinds these module globals in place. Because raw_dir_for() and the reader
    modules read these names at call time (not import time), any code that runs
    *after* this call -- downloads and reads alike -- uses the new location.

    This is the programmatic equivalent of exporting ARM_NSA_DATA_ROOT before
    import, and is what the scripts' --data-root flag calls to send data to (or
    read it back from) an external drive. FIGURE_DIR is deliberately left inside
    the repo, so figures stay with the code even when the data lives elsewhere.

    Returns the resolved DATA_ROOT.
    """
    global DATA_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR
    DATA_ROOT = Path(root).expanduser().resolve()
    RAW_DATA_DIR = DATA_ROOT / "raw"
    PROCESSED_DATA_DIR = DATA_ROOT / "processed"
    return DATA_ROOT
