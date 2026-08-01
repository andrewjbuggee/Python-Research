"""Map ERA5 single-level fields onto the surface energy budget terms of
Sledd et al. (2025), Equation (1), handling the sign-convention flip.

    Sledd, A., Shupe, M. D., Solomon, A., & Cox, C. J. (2025). JGR Atmospheres,
    130, e2024JD042578. https://doi.org/10.1029/2024JD042578

Why this module exists
----------------------
ERA5 stores every surface flux as POSITIVE DOWNWARD (into the surface). Sledd
et al. Equation (1) defines the turbulent fluxes as POSITIVE UPWARD and writes
them with leading minus signs:

    LWD - LWU + SWD - SWU - SWT - SH - LH + G = M

Substituting SH = -msshf and LH = -mslhf, the ERA5 form of the same equation is

    msnlwrf + msnswrf - SWT + msshf + mslhf + G = M

Getting that flip wrong silently reverses the sensible and latent heat terms,
which is why the conversion lives in one place rather than being repeated at
each call site.

Terms ERA5 cannot supply
------------------------
SWT (shortwave transmitted below the surface interface layer) is not an ERA5
output. Sledd et al. derive it from Beer's law (their Equation 6) using
MOSAiC-specific snow and ice optical properties. It is zero in polar night, so
Equation (1) reduces cleanly for a January analysis.

M (melt energy over sea ice) is likewise unavailable. ERA5 exposes
``mean_snowmelt_rate`` for the LAND tile only.

G is not archived directly. It can be approximated from the ERA5 four-layer sea
ice temperature profile (istl1..istl4), but see the caveats in README.md before
doing so -- ERA5 carries a fixed 1.5 m ice slab with no snow layer on sea ice.

Usage
-----
    import xarray as xr
    from seb_terms import compute_seb_terms

    ds = xr.open_dataset("era5_seb_barrow_20260101.nc")
    seb = compute_seb_terms(ds)
    seb["forcing_W_m2"]      # LWD + SWN, the Miller et al. radiative forcing
    seb["na_flux_W_m2"]      # Equation (5) net atmospheric flux
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    import xarray as xr

# Stefan-Boltzmann constant [W m-2 K-4]
SIGMA_SB_W_M2_K4 = 5.670374419e-8

# Surface emissivity assumed by Sledd et al. when inverting LWU for skin
# temperature over snow and sea ice (their Section 2.1).
EMISSIVITY_SNOW_ICE = 0.985


def _require(ds: "xr.Dataset", names: tuple[str, ...], term: str) -> None:
    """Raise a helpful error if any short name needed for ``term`` is absent."""
    missing = [n for n in names if n not in ds]
    if missing:
        raise KeyError(
            f"Cannot compute {term}: missing ERA5 variable(s) {missing}. "
            f"Re-download with a variable set that includes them "
            f"(see era5_seb_variables.py)."
        )


def compute_seb_terms(ds: "xr.Dataset") -> "xr.Dataset":
    """Return the Sledd et al. Equation (1) terms from an ERA5 SEB dataset.

    Every returned flux uses the SLEDD convention: radiative fluxes are signed
    as they appear in Equation (1), and the turbulent fluxes SH and LH are
    POSITIVE UPWARD (away from the surface).

    Parameters
    ----------
    ds : Dataset opened from a file written by download_era5_seb.py. Must at
        minimum contain msdwlwrf, msnlwrf, msdwswrf, msnswrf, msshf and mslhf.

    Returns
    -------
    Dataset of derived terms, each carrying ``units`` and ``long_name``
    attributes. Terms whose ERA5 inputs are absent are simply omitted rather
    than filled with NaN, so check membership before use.
    """
    import xarray as xr

    from era5_seb_variables import normalise_names

    # Files written by download_era5_seb.py already carry canonical short names,
    # but a file pulled straight from the CDS uses avg_sdlwrf, avg_ishf and so
    # on. Normalise first so either input works.
    ds = normalise_names(ds)

    _require(
        ds,
        ("msdwlwrf", "msnlwrf", "msdwswrf", "msnswrf", "msshf", "mslhf"),
        "Equation (1)",
    )

    out: dict[str, "xr.DataArray"] = {}

    # --- Radiative terms (ERA5 net fluxes are already down-minus-up) --------
    lwd_W_m2 = ds["msdwlwrf"]
    lw_net_W_m2 = ds["msnlwrf"]  # = LWD - LWU
    lwu_W_m2 = lwd_W_m2 - lw_net_W_m2

    swd_W_m2 = ds["msdwswrf"]
    sw_net_W_m2 = ds["msnswrf"]  # = SWD - SWU = SWN
    swu_W_m2 = swd_W_m2 - sw_net_W_m2

    out["lwd_W_m2"] = lwd_W_m2
    out["lwu_W_m2"] = lwu_W_m2
    out["lw_net_W_m2"] = lw_net_W_m2
    out["swd_W_m2"] = swd_W_m2
    out["swu_W_m2"] = swu_W_m2
    out["swn_W_m2"] = sw_net_W_m2

    # --- Turbulent terms: flip ERA5 down-positive to Sledd up-positive ------
    sh_up_W_m2 = -ds["msshf"]
    lh_up_W_m2 = -ds["mslhf"]
    out["sh_up_W_m2"] = sh_up_W_m2
    out["lh_up_W_m2"] = lh_up_W_m2

    # --- Radiative forcing of the Miller et al. (2017) framework ------------
    # Sledd et al. Section 3.2: forcing = LWD + SWN. Responding fluxes are
    # regressed against this.
    out["forcing_W_m2"] = lwd_W_m2 + sw_net_W_m2

    # --- Equation (5): net atmospheric flux ---------------------------------
    # NA = LWD - LWU + SWD - SWU - SH - LH = M - SWT - G
    out["na_flux_W_m2"] = lw_net_W_m2 + sw_net_W_m2 - sh_up_W_m2 - lh_up_W_m2

    # --- Surface cloud radiative effect, if clear-sky fields were downloaded -
    if "msnlwrfcs" in ds and "msnswrfcs" in ds:
        cre_lw_W_m2 = lw_net_W_m2 - ds["msnlwrfcs"]
        cre_sw_W_m2 = sw_net_W_m2 - ds["msnswrfcs"]
        out["cre_lw_W_m2"] = cre_lw_W_m2
        out["cre_sw_W_m2"] = cre_sw_W_m2
        out["cre_net_W_m2"] = cre_lw_W_m2 + cre_sw_W_m2
    if "msdwlwrfcs" in ds:
        out["lwd_cre_W_m2"] = lwd_W_m2 - ds["msdwlwrfcs"]

    # --- Skin temperature implied by LWU, for consistency checking ----------
    # Sledd et al. invert LWU for skin temperature with emissivity 0.985,
    # accounting for reflected LWD. Comparing this against ERA5's own skt
    # exposes any mismatch in the assumed emissivity.
    if "skt" in ds:
        out["t_skin_K"] = ds["skt"]
        t_skin_from_lwu_K = (
            (lwu_W_m2 - (1.0 - EMISSIVITY_SNOW_ICE) * lwd_W_m2)
            / (EMISSIVITY_SNOW_ICE * SIGMA_SB_W_M2_K4)
        ) ** 0.25
        out["t_skin_from_lwu_K"] = t_skin_from_lwu_K

    # --- Near-surface stability, the control on the SH response -------------
    if "skt" in ds and "t2m" in ds:
        out["dt_skin_minus_2m_K"] = ds["skt"] - ds["t2m"]

    if "u10" in ds and "v10" in ds:
        out["wind_speed_10m_m_s"] = (ds["u10"] ** 2 + ds["v10"] ** 2) ** 0.5

    result = xr.Dataset(out)

    long_names = {
        "lwd_W_m2": "Downwelling longwave radiative flux (LWD)",
        "lwu_W_m2": "Upwelling longwave radiative flux (LWU)",
        "lw_net_W_m2": "Net longwave, LWD - LWU, positive downward",
        "swd_W_m2": "Downwelling shortwave radiative flux (SWD)",
        "swu_W_m2": "Upwelling shortwave radiative flux (SWU)",
        "swn_W_m2": "Net shortwave, SWN = SWD - SWU, positive downward",
        "sh_up_W_m2": "Sensible heat flux, positive UPWARD (Sledd SH)",
        "lh_up_W_m2": "Latent heat flux, positive UPWARD (Sledd LH)",
        "forcing_W_m2": "Radiative forcing, LWD + SWN (Miller et al. framework)",
        "na_flux_W_m2": "Net atmospheric flux, Eq. (5) = M - SWT - G",
        "cre_lw_W_m2": "Surface longwave cloud radiative effect",
        "cre_sw_W_m2": "Surface shortwave cloud radiative effect",
        "cre_net_W_m2": "Surface net cloud radiative effect",
        "lwd_cre_W_m2": "Cloud contribution to LWD (all-sky minus clear-sky)",
        "t_skin_K": "ERA5 skin temperature",
        "t_skin_from_lwu_K": "Skin temperature inverted from LWU, emissivity 0.985",
        "dt_skin_minus_2m_K": "Skin minus 2 m temperature (near-surface stability)",
        "wind_speed_10m_m_s": "10 m wind speed",
    }
    for name, da in result.data_vars.items():
        da.attrs["long_name"] = long_names.get(name, name)
        da.attrs["units"] = _units_from_name(name)

    result.attrs["convention"] = (
        "Turbulent fluxes positive UPWARD (Sledd et al. 2025 Eq. 1). Radiative "
        "net fluxes positive DOWNWARD."
    )
    result.attrs["reference"] = "Sledd et al. (2025), JGR Atmos, 130, e2024JD042578"
    return result


def _units_from_name(name: str) -> str:
    """Recover units from the trailing unit token of a variable name."""
    if name.endswith("_W_m2"):
        return "W m-2"
    if name.endswith("_K"):
        return "K"
    if name.endswith("_m_s"):
        return "m s-1"
    return "1"


def open_seb_files(paths: list[str] | str) -> "xr.Dataset":
    """Open one or many downloaded SEB files as a single time-ordered dataset.

    Uses ``combine="by_coords"`` so daily chunks concatenate along time.
    """
    import xarray as xr

    if isinstance(paths, str):
        return xr.open_dataset(paths)
    return xr.open_mfdataset(paths, combine="by_coords")
