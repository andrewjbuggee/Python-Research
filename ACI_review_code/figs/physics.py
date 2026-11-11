"""Physics helpers for Section 7 figures. Every formula is taken from the cited source;
the __main__ block checks the numbers against the source tables/examples."""
import numpy as np

# ---------- constants ----------
RV = 461.5      # J kg-1 K-1, water vapor
RD = 287.0      # J kg-1 K-1, dry air (R' in Rogers & Yau)
CP = 1005.0     # J kg-1 K-1
EPS = 0.622
G0 = 9.81
RHO_L = 1000.0  # kg m-3

def latent_heat(T):            # J kg-1, linear fit adequate for 0-30 C
    return 2.501e6 - 2370.0 * (T - 273.15)

def es_water(T):               # Pa, Bolton (1980) form of Clausius-Clapeyron fit
    Tc = T - 273.15
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))

def K_air(T):                  # W m-1 K-1, linear interpolation of Rogers & Yau Table 7.1
    Tc = T - 273.15
    return np.interp(Tc, [-40, -30, -20, -10, 0, 10, 20, 30],
                     [2.07e-2, 2.16e-2, 2.24e-2, 2.32e-2, 2.40e-2, 2.48e-2, 2.55e-2, 2.63e-2])

def D_vapor(T, p):             # m2 s-1, Rogers & Yau Table 7.1 (given at 100 kPa, scales as 100/p)
    Tc = T - 273.15
    D100 = np.interp(Tc, [-40, -30, -20, -10, 0, 10, 20, 30],
                     [1.62e-5, 1.76e-5, 1.91e-5, 2.06e-5, 2.21e-5, 2.36e-5, 2.52e-5, 2.69e-5])
    return D100 * (1.0e5 / p)

# ---------- Rogers & Yau Ch. 6: Kelvin and Kohler ----------
def kohler_a_um(T):            # "a ~ 3.3e-5/T (cm)"  -> micrometers
    return 3.3e-5 / T * 1.0e4

def kohler_b_um3(mass_g, i=2.0, Ms=58.44):   # "b ~ 4.3 i M/ms (cm3)" -> um3
    return 4.3 * i * mass_g / Ms * 1.0e12

def kohler_S(r_um, T, mass_g, i=2.0, Ms=58.44):   # Eq. (6.6), approximate form
    a = kohler_a_um(T); b = kohler_b_um3(mass_g, i, Ms)
    return 1.0 + a / r_um - b / r_um**3

def kohler_crit(T, mass_g, i=2.0, Ms=58.44):      # Eqs. (6.7), (6.8)
    a = kohler_a_um(T); b = kohler_b_um3(mass_g, i, Ms)
    return np.sqrt(3 * b / a), np.sqrt(4 * a**3 / (27 * b))   # r* (um), S*-1 (fraction)

# ---------- Rogers & Yau Ch. 7: growth and supersaturation budget ----------
def growth_G(T, p):            # 1/(Fk+Fd), m2 s-1 ; Eq. (7.17)
    L = latent_heat(T)
    Fk = (L / (RV * T) - 1.0) * L * RHO_L / (K_air(T) * T)
    Fd = RHO_L * RV * T / (D_vapor(T, p) * es_water(T))
    return 1.0 / (Fk + Fd)

def Q1(T):                     # m-1 ; Eq. (7.23)
    L = latent_heat(T)
    return (1.0 / T) * (EPS * L * G0 / (RD * CP * T) - G0 / RD)

def Q2(T, p):                  # dimensionless (per unit mixing ratio) ; Eq. (7.24)
    L = latent_heat(T)
    rho = p / (RD * T)
    return rho * (RD * T / (EPS * es_water(T)) + EPS * L**2 / (p * T * CP))

def twomey_N(C, k, U_cms):     # Eq. (6.21): N in cm-3, U in cm/s
    return 0.88 * C**(2.0 / (k + 2.0)) * (7.0e-2 * U_cms**1.5)**(k / (k + 2.0))

def twomey_Smax(C, k, U_cms):  # Eq. (6.22): percent
    return 3.6 * ((1.6e-3 * U_cms**1.5) / C)**(1.0 / (k + 2.0))

def parcel(C, k, w, T=283.15, p=9.0e4, zmax=150.0, dt=0.02, nbin=600, smax_pct=5.0):
    """Adiabatic parcel with a CCN activity spectrum N = C s^k (s in percent), Eq. (6.20).
    Each CCN class activates when the ambient supersaturation first exceeds its critical value,
    then grows by r dr/dt = G s (Eq. 7.17, curvature and solute terms neglected after activation,
    as in Twomey's 1959 treatment). Supersaturation follows Eq. (7.22): ds/dt = Q1 w - Q2 dchi/dt.
    Returns z (m), s (%), N_activated (cm-3)."""
    Gg = growth_G(T, p); q1 = Q1(T); q2 = Q2(T, p); rho = p / (RD * T)
    edges = np.logspace(-4, np.log10(smax_pct), nbin + 1)       # percent
    sc = np.sqrt(edges[:-1] * edges[1:]) / 100.0                # critical s (fraction)
    Nbin = C * (edges[1:]**k - edges[:-1]**k) * 1.0e6 / rho     # per kg of air
    r2 = np.zeros(nbin); active = np.zeros(nbin, dtype=bool)
    s = 0.0; t = 0.0
    nstep = int(zmax / w / dt)
    zs = np.empty(nstep); ss = np.empty(nstep); Ns = np.empty(nstep)
    for n in range(nstep):
        active |= (sc < s)
        r = np.sqrt(r2)
        dchi = 4.0 * np.pi * RHO_L * np.sum(Nbin[active] * r[active]) * Gg * s   # kg/kg per s
        ds = q1 * w - q2 * dchi
        r2[active] += 2.0 * Gg * s * dt
        s += ds * dt
        t += dt
        zs[n] = w * t; ss[n] = s * 100.0
        Ns[n] = np.sum(Nbin[active]) * rho / 1.0e6
    return zs, ss, Ns

# ---------- Wood's notes on Twomey (1977): optical thickness ----------
def tau_uniform(N_cm3, L_gm3, h_m):
    N = N_cm3 * 1.0e6; L = L_gm3 * 1.0e-3
    r = (3.0 * L / (4.0 * np.pi * RHO_L * N))**(1.0 / 3.0)
    return 2.0 * np.pi * N * r**2 * h_m, r

# ---------- Bohren & Clothiaux Eqs. (5.51), (5.52), (5.66) ----------
def R_twostream(tau, g=0.85):
    x = tau * (1.0 - g) / 2.0
    return x / (1.0 + x)

def diffuse_below(tau, g=0.85):
    return 1.0 / (1.0 + tau * (1.0 - g) / 2.0) - np.exp(-tau)

# ---------- kappa-Kohler (Petters & Kreidenweis 2007) ----------
def A_kelvin(T, sigma=0.072):          # m ; A = 8.69251e-6 * sigma / T
    return 8.69251e-6 * sigma / T

def sc_kappa(D_m, kappa, T=288.15):    # fraction ; ln Sc = sqrt(4A^3 / (27 kappa D^3))
    A = A_kelvin(T)
    return np.exp(np.sqrt(4.0 * A**3 / (27.0 * kappa * D_m**3))) - 1.0

if __name__ == "__main__":
    print("== Rogers & Yau Table 6.1 (Kelvin critical radius at 0 C): source values in brackets")
    a = kohler_a_um(273.15)
    for S, src in [(1.01, 1.208e-1), (1.10, 1.261e-2), (1.5, 2.964e-3), (2, 1.734e-3), (10, 5.221e-4)]:
        print(f"  S={S:5.2f}  r_c={a/np.log(S):.4e} um   [{src:.4e}]")
    print("== Rogers & Yau Table 6.2 (NaCl, 273 K): r*, S*-1 (%)")
    for m, rs, sx in [(1e-16, 0.19, 0.42), (1e-15, 0.61, 0.13), (1e-14, 1.9, 0.042), (1e-13, 6.1, 0.013), (1e-12, 19, 0.0042)]:
        r, s = kohler_crit(273.0, m)
        print(f"  m={m:.0e} g  r*={r:.3g} um [{rs}]   S*-1={100*s:.3g}% [{sx}]")
    r, s = kohler_crit(273.0, 1e-16, i=2.0, Ms=132.14)
    print(f"== R&Y Fig 6.2 ammonium sulfate 1e-16 g (i=2): r*={r:.3f} um, S*-1={100*s:.2f}%  [text: 0.13 um, 0.6%]")
    r, s = kohler_crit(273.0, 1e-16, i=3.0, Ms=132.14)
    print(f"   same with i=3: r*={r:.3f} um, S*-1={100*s:.2f}%")
    print("== R&Y p.110 example: T=7C, p=80 kPa, r=5 um, U=5 m/s, N=300 cm-3 -> s_inf~0.5%, tau~2 s")
    T, p = 280.15, 8.0e4
    rho = p / (RD * T); nu0 = 300e6 / rho
    eta = 4 * np.pi * RHO_L * nu0 * 5e-6 * Q2(T, p) * growth_G(T, p)
    omega = 100 * Q1(T) * 5.0
    print(f"  Q1={Q1(T):.3e} m-1  Q2={Q2(T,p):.1f}  G={growth_G(T,p)*1e12:.1f} um2/s  s_inf={omega/eta:.3f}%  tau={1/eta:.2f} s")
    print(f"== R&Y p.104: growth parameter at 80 kPa, 0 C = 68.2 um2/s ; mine = {growth_G(273.15, 8e4)*1e12:.1f}")
    print("== Parcel model vs Twomey Eqs (6.21),(6.22)")
    for C, k, w in [(100, 0.5, 0.5), (1000, 0.5, 0.5), (650, 0.7, 0.5), (650, 0.7, 2.0), (300, 0.7, 0.2)]:
        z, s, N = parcel(C, k, w)
        print(f"  C={C:5d} k={k} w={w} m/s : parcel Smax={s.max():.3f}% N={N[-1]:.0f} | Twomey Smax={twomey_Smax(C,k,w*100):.3f}% N={twomey_N(C,k,w*100):.0f} | z(Smax)={z[s.argmax()]:.0f} m")
    print("== optical thickness example (Wood notes): N=100 cm-3, L=0.3 g m-3, h=300 m")
    t, r = tau_uniform(100, 0.3, 300); print(f"  tau={t:.1f}, r={r*1e6:.2f} um, check 3Lh/(2 rho r)={3*0.3e-3*300/(2*1000*r):.1f}")
    print(f"== B&C Eq 5.51: tau(1-g)=100 -> R={R_twostream(100/0.15):.3f} [text: 0.98]")
    g = 0.85; print(f"== B&C Eq 5.66 max near tau=(1/g)ln[2/(1-g)] = {np.log(2/(1-g))/g:.2f}")
    print("== kappa-Kohler vs Dedrick et al. (2026): D_HM 62-75 nm, kappa 0.42-0.50 -> s_kappa reported 0.28-0.40% (mean range)")
    for D, kap in [(62e-9, 0.50), (75e-9, 0.42), (69e-9, 0.45)]:
        print(f"  D={D*1e9:.0f} nm kappa={kap}: s_c={100*sc_kappa(D,kap):.3f}%")
    A = A_kelvin(288.15);
    for s_pct in [0.1, 0.2, 0.3]:
        Dc = (4*A**3/(27*0.45*np.log(1+s_pct/100)**2))**(1/3)
        print(f"  s={s_pct}% kappa=0.45 -> D_c={Dc*1e9:.0f} nm")
