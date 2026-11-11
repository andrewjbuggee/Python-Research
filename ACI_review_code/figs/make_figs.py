"""Figures for Section 7. Every curve is computed from an equation in a cited source
(see physics.py); tabulated source values are overlaid as markers where they exist."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib import font_manager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physics as ph
import miepython

OUT = os.path.dirname(os.path.abspath(__file__))
for f in font_manager.findSystemFonts():
    if "Carlito" in f:
        font_manager.fontManager.addfont(f)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#8a8984", "#e7e6e1", "#b9b8b2"
WASH_B, WASH_O = "#dbe9fb", "#fde3d8"

plt.rcParams.update({
    "font.family": ["Carlito", "DejaVu Sans"], "font.size": 9.5,
    "mathtext.default": "regular",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.size": 0, "ytick.minor.size": 0,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8.5, "text.color": INK,
    "figure.dpi": 100, "savefig.dpi": 220, "savefig.facecolor": "white",
})

from matplotlib.ticker import NullFormatter
def save(fig, name):
    for ax in fig.axes:
        ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("saved", name)

def dot(ax, x, y, color, size=7, z=5):
    ax.plot([x], [y], "o", ms=size, mfc=color, mec="white", mew=1.6, zorder=z)

def note(ax, x, y, s, ha="left", va="center", size=8.5, color=INK2, **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, **kw)

# ------------------------------------------------------------------ Fig 1: the chain
def fig_chain():
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")
    def box(x, y, w, h, title, sub, fc, ts=8.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.4", fc=fc, ec=AXIS, lw=0.9))
        ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", fontsize=ts, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=7.6, color=INK2)
    def arrow(x0, y0, x1, y1, color=INK2, rad=0.0):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, shrinkA=1, shrinkB=1,
                                    connectionstyle=f"arc3,rad={rad}"))
    w, h, y, gap = 16.2, 10.5, 27.5, 4.5
    xs = [0.5 + i * (w + gap) for i in range(5)]
    box(xs[0], y, w, h, "Aerosol", "size, number, κ", WASH_B)
    box(xs[1], y, w, h, "Droplet number", "N$_d$", WASH_B)
    box(xs[2], y, w, h, "Droplet size", r"r$_e$ $\propto$ (L/N$_d$)$^{1/3}$", WASH_B)
    box(xs[3], y, w, h, "Optical depth", r"τ $\propto$ N$_d$$^{1/3}$ LWP$^{5/6}$", WASH_B)
    box(xs[4], y, w, h, "Albedo", "α ≈ τ/(τ + c)", WASH_B)
    labs = ["activation\n7.2–7.4", "fixed water\n7.5, 7.7", "extinction\n7.7", "two-stream\n7.8"]
    for i in range(4):
        arrow(xs[i] + w + 0.4, y + h / 2, xs[i + 1] - 0.4, y + h / 2)
        ax.text(xs[i] + w + gap / 2, y + h + 1.0, labs[i], ha="center", va="bottom", fontsize=7.3, color=INK2, linespacing=1.05)
    y2, w2 = 3.0, 27.5
    xr, xe, xl = 8.0, 40.0, 72.0
    box(xr, y2, w2, h, "Rain formation", r"fall speed $\propto$ r$^2$, collisions (7.6)", WASH_O)
    box(xe, y2, w2, h, "Entrainment, evaporation", "sedimentation, mixing (7.10)", WASH_O)
    box(xl, y2, w2, h, "Liquid water path", "LWP rises or falls", WASH_O)
    xc = xs[2] + w / 2
    arrow(xc - 3, y - 0.5, xr + w2 * 0.62, y2 + h + 0.5)
    arrow(xc + 3, y - 0.5, xe + w2 * 0.45, y2 + h + 0.5)
    arrow(xe + w2 + 0.4, y2 + h / 2, xl - 0.4, y2 + h / 2)
    arrow(xr + w2 * 0.8, y2 - 0.3, xl + w2 * 0.3, y2 - 0.3, rad=0.16)
    arrow(xl + w2 * 0.35, y2 + h + 0.5, xs[3] + w * 0.55, y - 0.5, color=ORANGE)
    ax.text(xs[3] + w * 0.55 + 9.5, (y + y2 + h) / 2, "adjustments", ha="left", va="center", fontsize=7.8, color=INK2)
    save(fig, "fig01_chain.png")

# ------------------------------------------------------------------ Fig 2: Kohler
def fig_kohler():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw=dict(wspace=0.32))
    r = np.logspace(-2, 1.05, 800)
    T = 273.0
    # (a) ammonium sulfate 1e-16 g, as in R&Y Fig 6.2
    aa = ph.kohler_a_um(T); bb = ph.kohler_b_um3(1e-16, 2.0, 132.14)
    curv = 100 * aa / r; sol = -100 * bb / r**3; tot = curv + sol
    a.axhline(0, color=AXIS, lw=0.9, zorder=1)
    a.plot(r, curv, color=ORANGE, ls=(0, (4, 2)), lw=1.6, label="curvature term  +a/r")
    a.plot(r, sol, color=AQUA, ls=(0, (1.2, 1.6)), lw=1.8, label="solute term  −b/r$^3$")
    a.plot(r, tot, color=BLUE, label="sum (Köhler curve)")
    rs, ss = ph.kohler_crit(T, 1e-16, 2.0, 132.14)
    dot(a, rs, 100 * ss, BLUE)
    a.annotate(f"critical point\nr* = {rs:.2f} μm,  s* = {100*ss:.1f}%", xy=(rs, 100 * ss), xytext=(0.3, 1.12),
               fontsize=8.3, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), linespacing=1.1)
    a.text(0.0125, -0.82, "haze\n(stable)", fontsize=8.2, color=INK2, linespacing=1.05)
    a.text(1.3, 0.2, "activated\n(grows freely)", fontsize=8.2, color=INK2, linespacing=1.05)
    a.set_xscale("log"); a.set_xlim(0.01, 10); a.set_ylim(-1.0, 1.5)
    a.set_xlabel("Droplet radius (μm)"); a.set_ylabel("Equilibrium supersaturation (%)")
    a.set_title("(a) Two opposing terms")
    a.legend(loc="lower right", bbox_to_anchor=(1.03, 0.0), handlelength=2.2)
    a.set_xticks([0.01, 0.1, 1, 10]); a.set_xticklabels(["0.01", "0.1", "1", "10"])
    # (b) NaCl family, R&Y Table 6.2
    cols = [BLUE, ORANGE, AQUA]
    masses = [1e-16, 1e-15, 1e-14]
    table = {1e-16: (0.19, 0.42), 1e-15: (0.61, 0.13), 1e-14: (1.9, 0.042)}
    labs = ["10$^{-16}$ g", "10$^{-15}$ g", "10$^{-14}$ g"]
    b.axhline(0, color=AXIS, lw=0.9, zorder=1)
    for m, c, lab in zip(masses, cols, labs):
        s = 100 * (ph.kohler_S(r, T, m) - 1)
        b.plot(r, s, color=c, label=lab)
        b.plot([table[m][0]], [table[m][1]], "o", ms=7, mfc="white", mec=c, mew=1.8, zorder=6)
    b.text(0.0125, 0.47, "Open circles: r*, s* tabulated\nin Rogers & Yau, Table 6.2", fontsize=8.0, color=INK2, linespacing=1.1)
    b.set_xscale("log"); b.set_xlim(0.01, 10); b.set_ylim(-0.3, 0.6)
    b.set_xlabel("Droplet radius (μm)"); b.set_ylabel("Equilibrium supersaturation (%)")
    b.set_title("(b) Bigger nucleus, lower barrier")
    b.legend(title="NaCl mass", title_fontsize=8.5, loc="lower right", bbox_to_anchor=(1.03, 0.0))
    b.set_xticks([0.01, 0.1, 1, 10]); b.set_xticklabels(["0.01", "0.1", "1", "10"])
    save(fig, "fig02_kohler.png")

# ------------------------------------------------------------------ Fig 3: kappa
def sc_kappa_exact(Dd, kappa, T=298.15, sigma=0.072):
    """Maximum of S(D) = (D^3-Dd^3)/(D^3-Dd^3(1-kappa)) * exp(A/D)  (Petters & Kreidenweis 2007, Eq. 6)."""
    A = 8.69251e-6 * sigma / T
    D = Dd * np.logspace(0.002, 3.0, 6000)
    S = (D**3 - Dd**3) / (D**3 - Dd**3 * (1 - kappa)) * np.exp(A / D)
    return S.max() - 1.0

def fig_kappa():
    fig, ax = plt.subplots(figsize=(6.5, 3.15))
    Dd = np.logspace(np.log10(20e-9), np.log10(500e-9), 120)
    kaps = [0.1, 0.3, 0.6, 1.2]; cols = [YELLOW, AQUA, ORANGE, BLUE]
    for k, c in zip(kaps, cols):
        s = np.array([100 * sc_kappa_exact(d, k) for d in Dd])
        ax.plot(Dd * 1e9, s, color=c, label=f"κ = {k}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(20, 500); ax.set_ylim(0.02, 3)
    # EPCAPE annotations (Dedrick et al. 2026)
    ax.add_patch(Rectangle((62, 0.28), 13, 0.12, fc="none", ec=INK, lw=1.1, zorder=6))
    ax.annotate("Mean Hoppel-minimum diameters (62 and 75 nm) and\nmean supersaturation retrieved from the minimum\n(0.28–0.40%) at EPCAPE (Dedrick et al., 2026)", xy=(75, 0.34), xytext=(105, 1.3),
                fontsize=8.0, color=INK2, linespacing=1.1, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.axhspan(0.03, 0.13, color=WASH_B, zorder=0)
    ax.text(21.5, 0.062, "cloud-scene and parcel supersaturation at EPCAPE (0.03–0.13%)", fontsize=8.0, color=INK2, va="center")
    ax.set_xticks([20, 50, 100, 200, 500]); ax.set_xticklabels(["20", "50", "100", "200", "500"])
    ax.set_yticks([0.03, 0.1, 0.3, 1, 3]); ax.set_yticklabels(["0.03", "0.1", "0.3", "1", "3"])
    ax.set_xlabel("Dry particle diameter (nm)"); ax.set_ylabel("Critical supersaturation (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, columnspacing=1.6, handlelength=1.8)
    save(fig, "fig03_kappa.png")

# ------------------------------------------------------------------ Fig 4: size distribution + CCN spectrum
def lognormal(D, N, Dg, sg):
    return N / (np.sqrt(2 * np.pi) * np.log10(sg)) * np.exp(-(np.log10(D / Dg))**2 / (2 * np.log10(sg)**2))

def fig_ccn():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.95), gridspec_kw=dict(wspace=0.34))
    D = np.logspace(1, 3, 500)
    ait = lognormal(D, 300, 40, 1.55); acc = lognormal(D, 500, 160, 1.70); tot = ait + acc
    a.fill_between(D, 0, tot, where=D >= 150, color=WASH_B, lw=0, zorder=1)
    a.fill_between(D, 0, tot, where=(D >= 72) & (D <= 150), color=WASH_O, lw=0, zorder=1)
    a.plot(D, tot, color=BLUE, zorder=3)
    a.plot(D, ait, color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=2); a.plot(D, acc, color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=2)
    imin = np.argmin(np.where((D > 45) & (D < 140), tot, np.inf))
    a.annotate("Hoppel\nminimum", xy=(D[imin], tot[imin]), xytext=(62, 905), fontsize=8.2, color=INK2, ha="center", linespacing=1.05,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    a.text(13, 700, "Aitken\nmode", fontsize=8.2, color=INK2, linespacing=1.05)
    a.text(300, 760, "accumulation\nmode", fontsize=8.2, color=INK2, ha="left", linespacing=1.05)
    a.annotate("activates\nat 0.1%", xy=(215, 330), xytext=(420, 520), fontsize=8.0, color=INK2, ha="left", va="center", linespacing=1.05,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    a.text(104, 250, "also\nat\n0.3%", fontsize=7.6, color=INK2, ha="center", linespacing=1.0)
    a.set_xscale("log"); a.set_xlim(10, 1000); a.set_ylim(0, 1000)
    a.set_xticks([10, 100, 1000]); a.set_xticklabels(["10", "100", "1000"])
    a.set_xlabel("Dry particle diameter (nm)"); a.set_ylabel("dN/dlog$_{10}$D (cm$^{-3}$)")
    a.set_title("(a) Which particles activate")
    s = np.linspace(0.02, 2.0, 300)
    b.plot(s, 1500 * s**1.1, color=ORANGE, label="continental: C = 1500, k = 1.1")
    b.plot(s, 150 * s**0.6, color=BLUE, label="marine: C = 150, k = 0.6")
    b.set_yscale("log"); b.set_xlim(0, 2); b.set_ylim(10, 10000)
    b.set_yticks([10, 100, 1000, 10000]); b.set_yticklabels(["10", "100", "1,000", "10,000"])
    b.set_xlabel("Supersaturation s (%)"); b.set_ylabel("Activated CCN (cm$^{-3}$)")
    b.set_title("(b) Activity spectrum  N = C s$^k$")
    b.legend(loc="lower right")
    save(fig, "fig04_ccn.png")

# ------------------------------------------------------------------ Fig 5: parcel
def fig_parcel():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True, gridspec_kw=dict(wspace=0.12))
    res = {}
    for w, c, lab in [(0.5, BLUE, "updraft 0.5 m s$^{-1}$"), (2.0, ORANGE, "updraft 2 m s$^{-1}$")]:
        z, s, N = ph.parcel(650, 0.7, w, zmax=100.0)
        res[w] = (s.max(), N[-1])
        a.plot(s, z, color=c, label=lab); b.plot(N, z, color=c, label=lab)
        sm = ph.twomey_Smax(650, 0.7, w * 100); Nt = ph.twomey_N(650, 0.7, w * 100)
        a.plot([sm], [97], marker="v", ms=7, mfc=c, mec="white", mew=1.2, clip_on=False, zorder=6)
        b.plot([Nt], [97], marker="v", ms=7, mfc=c, mec="white", mew=1.2, clip_on=False, zorder=6)
        iz = np.argmax(s); dot(a, s[iz], z[iz], c)
    a.set_xlim(0, 1.0); a.set_ylim(0, 100); b.set_xlim(0, 800)
    a.set_xlabel("Supersaturation (%)"); a.set_ylabel("Height above cloud base (m)")
    b.set_xlabel("Activated droplets (cm$^{-3}$)")
    a.set_title("(a) Supersaturation peaks, then relaxes"); b.set_title("(b) Droplet number is set early")
    a.legend(loc="center right", bbox_to_anchor=(1.02, 0.45))
    a.text(0.62, 78, "▼ peak from\nTwomey's formula\n(R&Y Eq. 6.22)", fontsize=8.0, color=INK2, linespacing=1.1)
    b.text(480, 78, "▼ number from\nTwomey's formula\n(R&Y Eq. 6.21)", fontsize=8.0, color=INK2, linespacing=1.1)
    save(fig, "fig05_parcel.png")
    return res

# ------------------------------------------------------------------ Fig 6: Twomey formula
def fig_twomey():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw=dict(wspace=0.34))
    C = np.logspace(np.log10(30), np.log10(3000), 200); k = 0.7
    a.plot(C, C, color=MUTED, lw=1.0, ls=(0, (3, 2)))
    a.text(1500, 2050, "N$_d$ = C", fontsize=8.2, color=INK2, rotation=38, ha="center", va="center")
    for U, c in [(200, ORANGE), (50, BLUE), (10, AQUA)]:
        lab = f"{U/100:g} m s$^{{-1}}$"
        a.plot(C, ph.twomey_N(C, k, U), color=c, label=lab)
        b.plot(C, ph.twomey_Smax(C, k, U), color=c, label=lab)
    for ax in (a, b):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(30, 3000)
        ax.set_xticks([30, 100, 300, 1000, 3000]); ax.set_xticklabels(["30", "100", "300", "1,000", "3,000"])
        ax.set_xlabel("CCN active at 1% supersaturation, C (cm$^{-3}$)")
    a.set_ylim(10, 3000); a.set_yticks([10, 30, 100, 300, 1000, 3000]); a.set_yticklabels(["10", "30", "100", "300", "1,000", "3,000"])
    b.set_ylim(0.03, 2); b.set_yticks([0.03, 0.1, 0.3, 1]); b.set_yticklabels(["0.03", "0.1", "0.3", "1"])
    a.set_ylabel("Droplet number N$_d$ (cm$^{-3}$)"); b.set_ylabel("Peak supersaturation (%)")
    a.set_title("(a) More CCN, but not as many more droplets"); b.set_title("(b) Because the peak falls")
    a.legend(title="Updraft", title_fontsize=8.5, loc="upper left"); b.legend(title="Updraft", title_fontsize=8.5, loc="lower left")
    save(fig, "fig06_twomey.png")

# ------------------------------------------------------------------ Fig 7: condensational growth
def fig_growth():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw=dict(wspace=0.32))
    xi1 = ph.growth_G(273.0, 9.0e4) * 1e12      # um2/s
    xi = 0.0005 * xi1
    t = np.logspace(-2, 5, 400)
    a.plot(t, np.sqrt(0.75**2 + 2 * xi * t), color=BLUE, label="parabolic law, R&Y Eq. 7.19")
    radii = [1, 2, 4, 10, 20, 30, 50]
    cols = {"smallest nucleus": ([2.4, 130, 1000, 2700, 8500, 17500, 44500], "o"),
            "middle nucleus": ([0.15, 7.0, 320, 1800, 7400, 16000, 43500], "s"),
            "largest nucleus": ([0.013, 0.61, 62, 870, 5900, 14500, 41500], "^")}
    for lab, (tt, mk) in cols.items():
        a.plot(tt, radii, mk, ms=5.5, mfc="white", mec=INK2, mew=1.1, label=f"R&Y Table 7.2, {lab}", zorder=5)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1e-2, 1e5); a.set_ylim(0.12, 70)
    a.set_xticks([1e-2, 1, 1e2, 1e4]); a.set_xticklabels(["0.01", "1", "100", "10,000"])
    a.set_yticks([1, 3, 10, 30]); a.set_yticklabels(["1", "3", "10", "30"])
    a.set_xlabel("Time (s)"); a.set_ylabel("Droplet radius (μm)")
    a.set_title("(a) Growth at 0.05% supersaturation")
    a.legend(loc="lower right", fontsize=7.4, handlelength=1.4, bbox_to_anchor=(1.04, 0.0), borderaxespad=0.2)
    a.annotate("20 μm takes\nabout 2 hours", xy=(8500, 20), xytext=(0.02, 22), fontsize=8.0, color=INK2, linespacing=1.05,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    # (b) narrowing
    xi2 = 0.002 * xi1
    tt = np.linspace(0, 600, 200)
    r1 = np.sqrt(2.0**2 + 2 * xi2 * tt); r2 = np.sqrt(8.0**2 + 2 * xi2 * tt)
    b.fill_between(tt / 60, r1, r2, color=WASH_B, lw=0)
    b.plot(tt / 60, r2, color=BLUE, label="starts at 8 μm"); b.plot(tt / 60, r1, color=ORANGE, label="starts at 2 μm")
    b.annotate("", xy=(0.15, 8), xytext=(0.15, 2), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))
    b.text(0.75, 2.45, "6 μm apart", fontsize=8.2, color=INK2, va="center")
    b.annotate("", xy=(9.85, r2[-1]), xytext=(9.85, r1[-1]), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))
    b.text(9.55, 15.45, f"{r2[-1]-r1[-1]:.1f} μm apart", fontsize=8.2, color=INK2, va="center", ha="right")
    b.set_xlim(0, 10); b.set_ylim(0, 16)
    b.set_xlabel("Time (minutes)"); b.set_ylabel("Droplet radius (μm)")
    b.set_title("(b) Small droplets catch up (0.2%)")
    b.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08))
    save(fig, "fig07_growth.png")
    return xi1, r2[-1] - r1[-1]

# ------------------------------------------------------------------ Fig 8: fall speed + collision efficiency
def fig_fall():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw=dict(wspace=0.32))
    r1 = np.logspace(0, np.log10(30), 50)            # um, Stokes
    a.plot(r1, 1.19e6 * (r1 * 1e-4)**2, color=BLUE, label="Stokes: u = k$_1$r$^2$ (r < 30 μm)")
    r3 = np.logspace(np.log10(40), np.log10(600), 50)
    a.plot(r3, 8e3 * (r3 * 1e-4), color=ORANGE, label="u = k$_3$r (40 μm < r < 0.6 mm)")
    gk_d = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0, 4.0, 5.0])
    gk_u = np.array([0.27, 0.72, 1.17, 1.62, 2.06, 2.47, 3.27, 4.03, 5.17, 6.49, 8.06, 8.83, 9.09])
    a.plot(gk_d * 500, gk_u * 100, "o", ms=5.5, mfc="white", mec=INK2, mew=1.1, label="measured (Gunn & Kinzer, 1949)", zorder=5)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1, 3000); a.set_ylim(0.001, 2000)
    a.set_xticks([1, 10, 100, 1000]); a.set_xticklabels(["1", "10", "100", "1,000"])
    a.set_yticks([0.01, 1, 100]); a.set_yticklabels(["0.01", "1", "100"])
    dot(a, 10, 1.19e6 * (10e-4)**2, BLUE)
    a.annotate("cloud droplet\n10 μm: 1.2 cm s$^{-1}$", xy=(10, 1.19), xytext=(1.3, 30), fontsize=8.0, color=INK2, linespacing=1.05,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    a.set_xlabel("Drop radius (μm)"); a.set_ylabel("Terminal fall speed (cm s$^{-1}$)")
    a.set_title("(a) Fall speed")
    a.legend(loc="lower right", fontsize=7.4, bbox_to_anchor=(1.04, 0.0), handlelength=1.4, borderaxespad=0.2)
    # (b) R&Y Table 8.2, collector radius R = 10..40 um (rows whose column alignment is unambiguous)
    rr = [2, 3, 4, 6, 8, 10, 15, 20]
    E = {10: ([2, 3, 4, 6, 8], [0.017, 0.027, 0.037, 0.052, 0.052]),
         20: ([3, 4, 6, 8, 10, 15], [0.016, 0.027, 0.060, 0.12, 0.17, 0.17]),
         30: ([3, 4, 6, 8, 10, 15, 20], [0.020, 0.13, 0.28, 0.37, 0.54, 0.55, 0.47]),
         40: ([3, 4, 6, 8, 10, 15, 20], [0.020, 0.23, 0.40, 0.55, 0.70, 0.75, 0.75])}
    for R, c in [(40, BLUE), (30, ORANGE), (20, AQUA), (10, YELLOW)]:
        x, yv = E[R]
        b.plot(x, yv, "-o", color=c, ms=5.5, mec="white", mew=1.0, label=f"R = {R} μm")
    b.set_xlim(0, 35); b.set_ylim(0, 0.85); b.set_xticks([0, 5, 10, 15, 20])
    b.set_xlabel("Radius of collected droplet, r (μm)"); b.set_ylabel("Collision efficiency")
    b.set_title("(b) Chance of colliding")
    b.legend(title="Collector drop", title_fontsize=8.5, loc="center right", bbox_to_anchor=(1.04, 0.5), handlelength=1.4)
    save(fig, "fig08_fall.png")

# ------------------------------------------------------------------ Fig 9: same water, more droplets
def pack(n, rad, seed, L=10.0):
    rng = np.random.default_rng(seed); pts = []
    tries = 0
    while len(pts) < n and tries < 200000:
        tries += 1
        p = rng.uniform(rad + 0.15, L - rad - 0.15, 2)
        if all(np.hypot(*(p - q)) > 2 * rad + 0.12 for q in pts):
            pts.append(p)
    return pts

def fig_dots():
    fig = plt.figure(figsize=(6.5, 2.75))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.28, 1.15], wspace=0.16)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1]); c = fig.add_subplot(gs[3])
    for ax, n, rad, seed, ttl in [(a, 8, 1.0, 3, "(a) 8 droplets, radius r"), (b, 64, 0.5, 11, "(b) 64 droplets, r/2")]:
        for p in pack(n, rad, seed):
            ax.add_patch(Circle(p, rad, fc=BLUE, ec="white", lw=0.8))
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(AXIS)
        ax.set_title(ttl, fontsize=8.8)
    a.set_xlabel("volume 8·(4/3)πr$^3$\nshadow area 8πr$^2$", fontsize=8.4, color=INK2, linespacing=1.15)
    b.set_xlabel("same volume\nshadow area 16πr$^2$: doubled", fontsize=8.4, color=INK2, linespacing=1.15)
    x = np.logspace(0, 1, 100)
    c.plot(x, x**(1 / 3), color=BLUE)
    for xv in (2, 8):
        dot(c, xv, xv**(1 / 3), BLUE)
    c.annotate("double N$_d$:\nτ × 1.26", xy=(2, 2**(1 / 3)), xytext=(2.5, 1.02), fontsize=8.2, color=INK2, linespacing=1.05,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    c.annotate("8 × N$_d$:\nτ × 2", xy=(8, 2.0), xytext=(3.4, 2.08), fontsize=8.2, color=INK2, linespacing=1.05,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    c.set_xscale("log"); c.set_xlim(1, 10); c.set_ylim(0.9, 2.3)
    c.set_xticks([1, 2, 4, 8]); c.set_xticklabels(["1", "2", "4", "8"])
    c.set_xlabel("Droplet number relative to start"); c.set_ylabel("Optical depth relative to start")
    c.set_title(r"(c) τ $\propto$ N$_d$$^{1/3}$ at fixed water", fontsize=8.8)
    save(fig, "fig09_dots.png")

# ------------------------------------------------------------------ Fig 10: Mie extinction efficiency
def fig_mie():
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    lam = 0.55
    r = np.logspace(np.log10(0.03), np.log10(25), 2500)
    x = 2 * np.pi * r / lam
    q = np.array([miepython.efficiencies_mx(1.333 - 0j, xi)[0] for xi in x])
    ax.axvspan(5, 15, color=WASH_B, zorder=0)
    ax.axhline(2, color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.plot(r, q, color=BLUE, lw=1.4)
    ax.text(8.7, 0.45, "cloud\ndroplets", fontsize=8.4, color=INK2, ha="center", linespacing=1.05)
    ax.text(0.032, 2.12, "large-particle limit: Q$_{ext}$ = 2", fontsize=8.4, color=INK2)
    ax.text(0.085, 1.15, "aerosol sizes:\nextinction rises\nsteeply with size", fontsize=8.2, color=INK2, linespacing=1.08, ha="center")
    ax.set_xscale("log"); ax.set_xlim(0.03, 25); ax.set_ylim(0, 4.3)
    ax.set_xticks([0.1, 1, 10]); ax.set_xticklabels(["0.1", "1", "10"])
    ax.set_xlabel("Droplet radius (μm), for light of wavelength 0.55 μm"); ax.set_ylabel("Extinction efficiency Q$_{ext}$")
    save(fig, "fig10_mie.png")
    return float(np.mean(q[(r > 5) & (r < 15)]))

# ------------------------------------------------------------------ Fig 11: adiabatic cloud
def fig_adiabatic():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.9), sharey=True, gridspec_kw=dict(wspace=0.14))
    H = 300.0; Ltop = 0.6e-3
    z = np.linspace(0.5, H, 300); L = Ltop * z / H
    axs[0].plot(L * 1e3, z, color=INK2)
    out = {}
    for N, c in [(50, ORANGE), (200, BLUE)]:
        Nm = N * 1e6
        r = (3 * L / (4 * np.pi * ph.RHO_L * Nm))**(1 / 3)
        beta = 2 * np.pi * Nm * r**2
        tau = np.trapezoid(beta, z)
        out[N] = (r[-1] * 1e6, tau)
        axs[1].plot(r * 1e6, z, color=c, label=f"{N} cm$^{{-3}}$")
        axs[2].plot(beta * 1e3, z, color=c, label=f"N$_d$ = {N}:  τ = {tau:.0f}")
    axs[0].set_xlim(0, 0.7); axs[1].set_xlim(0, 16); axs[2].set_xlim(0, 120); axs[1].set_xticks([0, 5, 10, 15]); axs[0].set_xticks([0, 0.2, 0.4, 0.6]); axs[0].set_ylim(0, 300)
    axs[0].set_ylabel("Height above cloud base (m)")
    axs[0].set_xlabel("Liquid water (g m$^{-3}$)"); axs[1].set_xlabel("Droplet radius (μm)"); axs[2].set_xlabel("Extinction (km$^{-1}$)")
    axs[0].set_title("(a) Same water"); axs[1].set_title("(b) Smaller droplets"); axs[2].set_title("(c) More extinction")
    axs[1].legend(loc="lower right", fontsize=7.8, handlelength=1.2, bbox_to_anchor=(1.05, 0.0), title="N$_d$", title_fontsize=8.0)
    axs[2].legend(loc="lower right", fontsize=8.0, handlelength=1.4, bbox_to_anchor=(1.04, 0.0))
    save(fig, "fig11_adiabatic.png")
    return out

# ------------------------------------------------------------------ Fig 12: albedo + susceptibility
def fig_albedo():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.95), gridspec_kw=dict(wspace=0.32))
    tau = np.linspace(0, 60, 400)
    a.plot(tau, tau / (tau + 7), color=BLUE, label="α = τ/(τ + 7)  (Wood's notes)")
    a.plot(tau, ph.R_twostream(tau, 0.85), color=ORANGE, label="two-stream, g = 0.85  (B&C Eq. 5.51)")
    for t0, t1 in [(15.1, 19.0)]:
        for f, c in [(lambda t: t / (t + 7), BLUE), (lambda t: ph.R_twostream(t, 0.85), ORANGE)]:
            dot(a, t0, f(t0), c, size=6); dot(a, t1, f(t1), c, size=6)
    a.annotate("doubling N$_d$\n(τ from 15 to 19)", xy=(17, 0.62), xytext=(27, 0.33), fontsize=8.2, color=INK2, linespacing=1.05,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    a.set_xlim(0, 60); a.set_ylim(0, 1)
    a.set_xlabel("Cloud optical depth τ"); a.set_ylabel("Cloud albedo α")
    a.set_title("(a) Albedo saturates")
    a.legend(loc="lower right", fontsize=7.8, handlelength=1.6)
    al = np.linspace(0, 1, 200)
    b.plot(al, al * (1 - al) / 3, color=BLUE)
    dot(b, 0.5, 1 / 12, BLUE)
    b.annotate("most susceptible at α = 0.5:\ndα/dln N$_d$ = 1/12", xy=(0.5, 1 / 12), xytext=(0.5, 0.098), fontsize=8.2, color=INK2, ha="center", linespacing=1.05)
    b.set_xlim(0, 1); b.set_ylim(0, 0.115)
    b.set_xlabel("Cloud albedo α"); b.set_ylabel("dα / dln N$_d$  at fixed water")
    b.set_title("(b) Susceptibility  α(1 − α)/3")
    save(fig, "fig12_albedo.png")

# ------------------------------------------------------------------ Fig 13: transmission and diffuse light
def fig_transmit():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.95), gridspec_kw=dict(wspace=0.32))
    ts = np.linspace(0, 8, 300)
    a.plot(ts, 1 / (1 + ts / 2), color=BLUE, label="with multiple scattering (B&C Eq. 5.52)")
    a.plot(ts, np.exp(-ts / 2), color=MUTED, lw=1.4, ls=(0, (3, 2)), label="exponential (single scattering only)")
    a.set_xlim(0, 8); a.set_ylim(0, 1)
    a.set_xlabel("Scaled optical thickness τ(1 − g)"); a.set_ylabel("Transmissivity")
    a.set_title("(a) Scattered light gets second chances")
    a.legend(loc="upper right", fontsize=7.8, handlelength=1.6)
    tau = np.linspace(0, 20, 500)
    for g, c in [(0.95, BLUE), (0.85, ORANGE), (0.75, AQUA), (0.0, YELLOW)]:
        d = ph.diffuse_below(tau, g)
        b.plot(tau, d, color=c, label=f"g = {g:g}")
        i = np.argmax(d); dot(b, tau[i], d[i], c, size=6)
    b.set_xlim(0, 20); b.set_ylim(0, 1)
    b.set_xlabel("Cloud optical thickness τ"); b.set_ylabel("Diffuse irradiance below cloud / incident")
    b.set_title("(b) Diffuse light peaks for thin cloud")
    b.legend(loc="upper right", ncol=2, fontsize=8.0, handlelength=1.4, columnspacing=1.0)
    save(fig, "fig13_transmit.png")

# ------------------------------------------------------------------ Fig 14: evaporation + sedimentation
def fig_evap():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.95), gridspec_kw=dict(wspace=0.34))
    r0 = np.array([1, 3, 10, 30, 100, 150]); dist = np.array([2e-6, 0.17e-3, 2.1e-2, 1.69, 208, 1050])   # m
    rr = np.logspace(0, np.log10(150), 50)
    a.plot(rr, dist[2] * (rr / 10.0)**4, color=BLUE, label="fourth-power law through 10 μm")
    a.plot(r0, dist, "o", ms=6, mfc="white", mec=INK2, mew=1.2, label="Rogers & Yau, Table 7.3", zorder=5)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(0.8, 200); a.set_ylim(1e-6, 1e4)
    a.set_xticks([1, 10, 100]); a.set_xticklabels(["1", "10", "100"])
    a.set_yticks([1e-6, 1e-3, 1, 1e3]); a.set_yticklabels(["1 μm", "1 mm", "1 m", "1 km"])
    a.set_xlabel("Initial drop radius (μm)"); a.set_ylabel("Distance fallen before evaporating")
    a.set_title("(a) Survival in dry air (RH 80%)")
    a.legend(loc="upper left", fontsize=7.8, handlelength=1.6, bbox_to_anchor=(-0.02, 1.0))
    N = np.logspace(np.log10(20), np.log10(1000), 100)
    for L, c in [(0.6, ORANGE), (0.3, BLUE)]:
        r = (3 * L * 1e-3 / (4 * np.pi * ph.RHO_L * N * 1e6))**(1 / 3)     # m
        u = 1.19e6 * (r * 100)**2                                          # cm/s
        b.plot(N, u, color=c, label=f"{L} g m$^{{-3}}$")
    b.set_xscale("log"); b.set_xlim(20, 1000); b.set_ylim(0, 4.5)
    b.set_xticks([20, 50, 100, 200, 500, 1000]); b.set_xticklabels(["20", "50", "100", "200", "500", "1,000"])
    b.set_xlabel("Droplet number N$_d$ (cm$^{-3}$)"); b.set_ylabel("Fall speed of mean droplet (cm s$^{-1}$)")
    b.set_title("(b) Sedimentation slows as N$_d$ rises")
    b.legend(title="Liquid water", title_fontsize=8.5, loc="upper right")
    save(fig, "fig14_evap.png")

# ------------------------------------------------------------------ Fig 15: net susceptibility
def fig_net():
    fig, ax = plt.subplots(figsize=(6.5, 2.85))
    s = np.linspace(-1.3, 0.5, 200); f = 1 + 2.5 * s
    ax.axhspan(-2.6, 0, color=WASH_O, zorder=0); ax.axhspan(1, 2.6, color=WASH_B, zorder=0)
    ax.axhline(0, color=AXIS, lw=0.9); ax.axhline(1, color=AXIS, lw=0.9)
    ax.plot(s, f, color=BLUE)
    for sv, lab, xy in [(0.0, "no LWP response:\nTwomey effect alone", (0.06, 0.45)),
                        (-0.4, "slope −0.4:\nexact cancellation", (-0.36, -0.75)),
                        (-1.1, "slope −1.1 reported at EPCAPE\n(Zhang et al., 2025)", (-1.0, -2.3))]:
        dot(ax, sv, 1 + 2.5 * sv, BLUE)
        ax.annotate(lab, xy=(sv, 1 + 2.5 * sv), xytext=xy, fontsize=8.2, color=INK2, linespacing=1.05,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.text(-1.27, 2.2, "brighter than Twomey alone (cloud thickens)", fontsize=8.4, color=INK2, va="center")
    ax.text(0.47, -2.2, "net darkening (cloud thins faster than it brightens)", fontsize=8.4, color=INK2, va="center", ha="right")
    ax.text(-1.27, 0.5, "Twomey effect partly offset", fontsize=8.4, color=INK2, va="center")
    ax.set_xlim(-1.3, 0.5); ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel("LWP adjustment slope, dln LWP / dln N$_d$"); ax.set_ylabel("Net susceptibility ÷ Twomey susceptibility")
    save(fig, "fig15_net.png")

# ------------------------------------------------------------------ Fig: absorption in a thick cloud (B&C Eq. 5.72)
def Rinf(x, g=0.85):
    w = 1.0 - x
    return (np.sqrt(1 - w * g) - np.sqrt(1 - w)) / (np.sqrt(1 - w * g) + np.sqrt(1 - w))

def fig_absorb():
    fig, ax = plt.subplots(figsize=(6.5, 2.85))
    x = np.logspace(-7, -1, 400)
    ax.plot(x, Rinf(x), color=BLUE)
    xv = 0.1 * 1e-5 / 3.0          # kappa_w d / 3 with kappa_w = 0.1 m-1, d = 10 um (B&C Sec. 5.3.1)
    dot(ax, xv, Rinf(xv), BLUE)
    ax.annotate(f"visible light, 10-μm droplets:\nabsorbed fraction ≈ 3 × 10$^{{-7}}$, reflectivity {Rinf(xv):.3f}", xy=(xv, Rinf(xv)), xytext=(1.3e-7, 0.62),
                fontsize=8.2, color=INK2, linespacing=1.08, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    dot(ax, 1e-3, Rinf(1e-3), BLUE)
    ax.annotate(f"one photon in a thousand absorbed\nper scattering: reflectivity {Rinf(1e-3):.2f}", xy=(1e-3, Rinf(1e-3)), xytext=(8e-6, 0.36),
                fontsize=8.2, color=INK2, linespacing=1.08, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    xa, xb = 1.0e-2 / 3.0, 2.0e-2 / 3.0
    dot(ax, xa, Rinf(xa), ORANGE); dot(ax, xb, Rinf(xb), ORANGE)
    ax.annotate(f"droplet size doubled at an\nabsorbing wavelength:\n{Rinf(xa):.2f} falls to {Rinf(xb):.2f}", xy=(xb, Rinf(xb)), xytext=(1.6e-4, 0.08),
                fontsize=8.2, color=INK2, linespacing=1.08, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set_xscale("log"); ax.set_xlim(1e-7, 1e-1); ax.set_ylim(0, 1.05)
    ax.set_xticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_xticklabels(["10$^{-7}$", "10$^{-6}$", "10$^{-5}$", "10$^{-4}$", "10$^{-3}$", "10$^{-2}$", "10$^{-1}$"])
    ax.set_xlabel("Fraction of light absorbed at each scattering event (1 − single-scattering albedo)")
    ax.set_ylabel("Reflectivity of a very thick cloud")
    save(fig, "fig16_absorb.png")
    return Rinf(xv), Rinf(1e-3), Rinf(xa), Rinf(xb)

if __name__ == "__main__":
    fig_chain(); fig_kohler(); fig_kappa(); fig_ccn()
    res = fig_parcel(); print("parcel (Smax %, N):", res)
    fig_twomey()
    xi1, dr = fig_growth(); print(f"xi1(273K, 90kPa) = {xi1:.1f} um2/s ; narrowing final gap {dr:.2f} um")
    fig_fall(); fig_dots()
    print("mean Qext 5-15 um:", fig_mie())
    print("adiabatic (r_top um, tau):", fig_adiabatic())
    fig_albedo(); fig_transmit(); fig_evap(); fig_net()
    print("Rinf values:", fig_absorb())
    # numbers quoted in the text
    for D, k in [(100e-9, 0.6), (150e-9, 0.45), (72e-9, 0.45), (69e-9, 0.45)]:
        print(f"kappa-Kohler exact: D={D*1e9:.0f} nm kappa={k}: s_c={100*sc_kappa_exact(D,k):.3f}%  (approx {100*ph.sc_kappa(D,k,298.15):.3f}%)")
    for C in (100, 200, 1000, 2000):
        print(f"Twomey k=0.5 U=50: C={C} N={ph.twomey_N(C,0.5,50):.0f} Smax={ph.twomey_Smax(C,0.5,50):.3f}")
    print("EPCAPE-like: C=1000,k=0.7,U=20 cm/s ->", ph.twomey_N(1000, 0.7, 20), ph.twomey_Smax(1000, 0.7, 20))
    for N in (100, 200):
        t, r = ph.tau_uniform(N, 0.3, 300); print(f"uniform cloud N={N}: r={r*1e6:.2f} um tau={t:.1f} alpha7={t/(t+7):.3f} alphaBC={ph.R_twostream(t):.3f}")
