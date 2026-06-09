"""
plot_network_architecture_diagram.py — Schematic of the ProfileOnlyNetwork
retrieval architecture, styled for a slide / PhD-defense audience.

The diagram is hand-drawn with matplotlib patches (no torchviz) so it reads
as a clean conceptual figure rather than a computation graph.  It mirrors the
visual grammar of Segal-Rozenhaimer et al. (2022, Fig. 2) — input column →
hidden layers → output/target layer — but depicts THIS network faithfully:

  * Input  : 636 HySICS reflectance channels (~352-2297 nm), per-channel
             min-max normalized to [0, 1] (linear reflectance, NO log10)  +  4
             viewing-geometry angles (SZA, VZA, SAZ, VAZ), min-max normalized
             over fixed physical ranges  = 640 inputs.
             NOTE: raw spectrum, NO PCA (unlike the reference figure).
  * Encoder: 4 hidden layers x 256 units; each block is
             Linear -> LayerNorm -> GELU -> Dropout.
  * Output : a 7-level droplet effective-radius profile r_e(z) (cloud top ->
             cloud base), with a per-level 1-sigma uncertainty.  Two heads
             share the encoder: a sigmoid-bounded mean head ([1.5, 50] um) and
             a log-std head, trained with a Gaussian NLL + physics-informed
             penalties (monotonicity, adiabatic shape, smoothness).

Architecture source: models_profile_only.ProfileOnlyNetwork + RetrievalConfig
(n_wavelengths=636, n_geometry_inputs=4, n_levels=7, hidden_dims=(256,)*4).

Outputs (500 DPI PNG + vector PDF) are written to --output-dir.

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})

# Palette — blues echo the reference figure; the two brand accents (blue / green)
# match Andrew's result figures (RAW_BLUE retrieval, GREEN NN curves).
INPUT_FILL,  INPUT_EDGE  = "#BBD0EC", "#3A6DB0"   # input neurons
HIDDEN_FILL, HIDDEN_EDGE = "#8FB3DE", "#1F4E89"   # hidden neurons
GEOM_FILL,   GEOM_EDGE   = "#FCE3B4", "#C8922A"   # geometry neurons
MEAN_FILL,   MEAN_EDGE   = "#1F5FC2", "#143F84"   # mean (r_e) head
SIGMA_FILL,  SIGMA_EDGE  = "#10B981", "#0B7A57"   # uncertainty (sigma) head
BOX_FILL,    BOX_EDGE    = "#EEF3FA", "#3A6DB0"   # pipeline boxes
FLOW    = "#3A3A3A"   # bold layer-to-layer flow arrows
SYNAPSE = "#B7C2D0"   # faint all-to-all connection lines
INK     = "#1A1A1A"   # primary text
SUBTLE  = "#5A5A5A"   # secondary text

R = 0.26  # neuron radius (data units)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing primitives
# ─────────────────────────────────────────────────────────────────────────────
def neuron(ax, x, y, fill, edge, r=R, z=3):
    ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=edge,
                         linewidth=1.4, zorder=z))


def neuron_column(ax, x, ys, fill, edge, r=R):
    for y in ys:
        neuron(ax, x, y, fill, edge, r=r)
    return [(x, y) for y in ys]


def vdots(ax, x, y, color=INK):
    for dy in (-0.18, 0.0, 0.18):
        ax.add_patch(Circle((x, y + dy), 0.035, facecolor=color,
                             edgecolor="none", zorder=4))


def synapses(ax, left_pts, right_pts, color=SYNAPSE, alpha=0.45, lw=0.6):
    """Faint all-to-all lines suggesting dense connectivity."""
    for (x0, y0) in left_pts:
        for (x1, y1) in right_pts:
            ax.plot([x0 + R, x1 - R], [y0, y1], color=color,
                    alpha=alpha, lw=lw, zorder=1, solid_capstyle="round")


def flow_arrow(ax, x0, x1, y, color=FLOW, lw=2.4):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y),
                                 arrowstyle="-|>", mutation_scale=18,
                                 lw=lw, color=color, zorder=5,
                                 shrinkA=0, shrinkB=0))


def pill(ax, cx, cy, w, h, text, fill=BOX_FILL, edge=BOX_EDGE,
         fontsize=11, weight="normal", text_color=INK):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.18",
                                facecolor=fill, edgecolor=edge,
                                linewidth=1.6, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, weight=weight, zorder=6)


def layer_header(ax, x, text, y=8.45):
    ax.text(x, y, text, ha="center", va="center", fontsize=12.5,
            weight="bold", color=INK, zorder=6)
    ax.plot([x - 0.95, x + 0.95], [y - 0.32, y - 0.32], color=INK,
            lw=1.3, zorder=6)


# ─────────────────────────────────────────────────────────────────────────────
# Profile glyph: stylized retrieved r_e(z) +/- sigma, cloud top -> base
# ─────────────────────────────────────────────────────────────────────────────
def profile_glyph(ax, x0, x1, y0, y1):
    pill(ax, (x0 + x1) / 2, y1 + 0.55, (x1 - x0) + 0.5, 0.62,
         "Retrieved profile", fill="white", edge=BOX_EDGE,
         fontsize=11.5, weight="bold")

    n = 7
    yy = np.linspace(y1 - 0.25, y0 + 0.25, n)          # top -> base (top high)
    frac = np.linspace(0.0, 1.0, n)
    re = 0.12 + 0.80 * frac ** 0.85                    # r_e grows toward base
    sig = 0.10 + 0.05 * np.sin(np.pi * frac)           # per-level uncertainty
    xx = x0 + 0.25 + re * (x1 - x0 - 0.5)
    half = sig * (x1 - x0 - 0.5)

    ax.fill_betweenx(yy, xx - half, xx + half, color=SIGMA_FILL,
                     alpha=0.28, zorder=2)
    ax.plot(xx, yy, "-", color=MEAN_FILL, lw=2.2, zorder=4)
    ax.errorbar(xx, yy, xerr=half, fmt="s", ms=4.5, color=MEAN_FILL,
                ecolor=SIGMA_EDGE, elinewidth=1.3, capsize=2.5, zorder=5)

    ax.annotate("cloud top", (xx[0], yy[0]), (x0 - 0.05, y1 + 0.05),
                fontsize=8.5, color=SUBTLE, ha="right", va="center")
    ax.annotate("cloud base", (xx[-1], yy[-1]), (x0 - 0.05, y0 - 0.05),
                fontsize=8.5, color=SUBTLE, ha="right", va="center")
    ax.text((x0 + x1) / 2, y0 - 0.55, r"$r_e$ (μm)  $\rightarrow$",
            ha="center", va="center", fontsize=9, color=SUBTLE)


# ─────────────────────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────────────────────
def build_figure():
    fig, ax = plt.subplots(figsize=(17.0, 9.0))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    cy = 4.55  # vertical center for the network spine

    # ── Column x-positions ────────────────────────────────────────────────────
    x_src   = 1.55
    x_prep  = 3.85
    x_in    = 6.0
    x_h     = [7.85, 9.25, 10.65, 12.05]
    x_out   = 13.95
    x_glyph0, x_glyph1 = 15.05, 16.55

    # ── Input source boxes (spectrum + geometry) ──────────────────────────────
    pill(ax, x_src, cy + 1.65, 2.5, 1.5,
         "HySICS spectrum\n" r"$R(\lambda)$, 636 channels" "\n(≈352–2297 nm)",
         fontsize=10.5)
    pill(ax, x_src, cy - 1.75, 2.5, 1.35,
         "Viewing geometry\nSZA · VZA · SAZ · VAZ",
         fill="#FBF1DC", edge=GEOM_EDGE, fontsize=10.5)

    # ── Preprocessing (per-channel min-max normalize) — NOTE: no PCA ──────────
    pill(ax, x_prep, cy + 1.65, 1.9, 1.2,
         r"$R(\lambda)$" "\nmin-max\nnormalize", fontsize=10.5)
    ax.text(x_prep, cy + 0.55, "no PCA —\nfull spectrum", ha="center",
            va="center", fontsize=8.8, style="italic", color="#B23B3B")
    flow_arrow(ax, x_src + 1.25, x_prep - 0.95, cy + 1.65, lw=2.0)

    # ── Input layer neurons ───────────────────────────────────────────────────
    layer_header(ax, x_in, "Input layer\n(640)")
    spec_ys = [cy + 2.45, cy + 2.0, cy + 1.55]          # R(λ1..) sample + dots
    geom_ys = [cy - 0.55, cy - 1.1, cy - 1.65, cy - 2.2]  # 4 geometry angles
    in_spec = neuron_column(ax, x_in, spec_ys, INPUT_FILL, INPUT_EDGE)
    vdots(ax, x_in, cy + 0.95)
    in_spec_low = neuron_column(ax, x_in, [cy + 0.45], INPUT_FILL, INPUT_EDGE)
    in_geom = neuron_column(ax, x_in, geom_ys, GEOM_FILL, GEOM_EDGE)
    in_pts = in_spec + in_spec_low + in_geom

    ax.text(x_in - 0.5, cy + 2.45, r"$R(\lambda_1)$", ha="right",
            va="center", fontsize=8.6, color=SUBTLE)
    ax.text(x_in - 0.5, cy + 0.45, r"$R(\lambda_{636})$", ha="right",
            va="center", fontsize=8.6, color=SUBTLE)
    for yy, lbl in zip(geom_ys, ["SZA", "VZA", "SAZ", "VAZ"]):
        ax.text(x_in - 0.45, yy, lbl, ha="right", va="center",
                fontsize=8.3, color=SUBTLE)

    # arrows from the two preprocessing/source boxes into the input column
    flow_arrow(ax, x_prep + 0.95, x_in - R - 0.45, cy + 1.65, lw=2.0)
    flow_arrow(ax, x_src + 1.25, x_in - R - 0.45, cy - 1.75, lw=2.0)

    # ── Hidden layers (4 x 256, GELU) ─────────────────────────────────────────
    hid_ys = [cy + 1.85, cy + 1.1, cy + 0.35, cy - 0.95]
    hidden_cols = []
    for j, xh in enumerate(x_h):
        col = neuron_column(ax, xh, hid_ys, HIDDEN_FILL, HIDDEN_EDGE)
        vdots(ax, xh, cy - 0.35)
        hidden_cols.append(col)
        ax.text(xh, cy - 1.75, "256", ha="center", va="center",
                fontsize=9, color=SUBTLE)
    layer_header(ax, (x_h[0] + x_h[-1]) / 2,
                 "Hidden layers  (4 × 256, GELU)")

    # connections: input -> H1 -> H2 -> H3 -> H4
    synapses(ax, in_pts, hidden_cols[0])
    for a, b in zip(hidden_cols[:-1], hidden_cols[1:]):
        synapses(ax, a, b)

    # per-layer block recipe annotation
    ax.text((x_h[0] + x_h[-1]) / 2, cy - 2.45,
            r"each layer:  Linear $\rightarrow$ LayerNorm $\rightarrow$ "
            r"GELU $\rightarrow$ Dropout",
            ha="center", va="center", fontsize=9.5, color=SUBTLE,
            style="italic")

    # ── Output heads: mean r_e(z) (upper cluster) + per-level sigma (lower) ────
    layer_header(ax, x_out, "Output layer\n(7 levels)")
    mean_ys = [cy + 2.45 - 0.33 * i for i in range(7)]   # 7.0 -> 5.02
    sig_ys  = [cy - 0.55 - 0.33 * i for i in range(7)]   # 4.0 -> 2.02
    out_mean = neuron_column(ax, x_out, mean_ys, MEAN_FILL, MEAN_EDGE, r=0.17)
    out_sig  = neuron_column(ax, x_out, sig_ys,  SIGMA_FILL, SIGMA_EDGE, r=0.17)

    synapses(ax, hidden_cols[-1], out_mean, color="#9DB8D8", alpha=0.45, lw=0.5)
    synapses(ax, hidden_cols[-1], out_sig,  color="#8FD3BC", alpha=0.45, lw=0.5)

    ax.text(x_out + 0.42, (mean_ys[0] + mean_ys[-1]) / 2, r"$r_e(z)$",
            ha="left", va="center", fontsize=11, weight="bold",
            color=MEAN_EDGE)
    ax.text(x_out + 0.42, (sig_ys[0] + sig_ys[-1]) / 2, r"$\sigma(z)$",
            ha="left", va="center", fontsize=11, weight="bold",
            color=SIGMA_EDGE)

    # ── Profile glyph on the far right ─────────────────────────────────────────
    profile_glyph(ax, x_glyph0, x_glyph1, cy - 2.1, cy + 2.1)
    flow_arrow(ax, x_out + 0.30, x_glyph0 - 0.15, cy, lw=2.0)

    # ── Callout annotations (the distinctive features) ────────────────────────
    ax.annotate(
        "sigmoid-bounded\n" r"$r_e \in [1.5,\ 50]$ μm",
        xy=(x_out - R, mean_ys[2]), xytext=(x_out - 1.85, 7.85),
        fontsize=9.2, color=MEAN_EDGE, ha="center", va="center",
        arrowprops=dict(arrowstyle="-|>", color=MEAN_EDGE, lw=1.3,
                        connectionstyle="arc3,rad=-0.2"))
    ax.annotate(
        "aleatoric uncertainty\n(Gaussian NLL)",
        xy=(x_out - R, sig_ys[4]), xytext=(x_out - 1.95, 1.55),
        fontsize=9.2, color=SIGMA_EDGE, ha="center", va="center",
        arrowprops=dict(arrowstyle="-|>", color=SIGMA_EDGE, lw=1.3,
                        connectionstyle="arc3,rad=0.2"))

    # physics-informed training footer
    pill(ax, 9.0, 0.72, 12.6, 0.78,
         "Physics-informed loss:  Gaussian NLL  +  "
         r"$\lambda$(monotonicity · adiabatic shape · smoothness)",
         fill="#F3F0FA", edge="#7B5EA7", fontsize=10.5, weight="bold",
         text_color="#3F2E63")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(8.5, 8.95,
            "Droplet-profile retrieval network  "
            r"(spectral reflectance $\rightarrow$ vertical $r_e$ profile)",
            ha="center", va="center", fontsize=15.5, weight="bold",
            color=INK)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Defense variant — stripped & enlarged for projector legibility
# ─────────────────────────────────────────────────────────────────────────────
def build_figure_defense():
    """
    Slide-optimized version of build_figure() for the PhD defense.

    Differences from the paper figure:
      * No input source / preprocessing boxes (HySICS, min-max normalize, geometry)
        and no "no PCA" note — the freed space lets the network draw larger.
      * No retrieved-profile glyph, and no 'cloud top' / 'cloud base' /
        'r_e (μm)' captions.
      * No 'sigmoid-bounded r_e ∈ [1.5, 50] μm' callout.
      * Bigger neurons and fonts, ~2:1 aspect to match the slide region.

    Everything kept (input neurons + their labels, 4×256 encoder, the two
    output heads, the aleatoric-uncertainty callout, and the physics-loss
    footer) depicts the same ProfileOnlyNetwork.
    """
    fig, ax = plt.subplots(figsize=(14.0, 7.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.4)
    ax.set_aspect("equal")
    ax.axis("off")

    cy = 3.95
    r_main = 0.30   # input / hidden neuron radius
    r_out = 0.19    # output neuron radius

    # ── Column x-positions (spread to fill the freed width) ───────────────────
    x_in  = 1.7
    x_h   = [3.9, 5.8, 7.7, 9.6]
    x_out = 11.7

    def header(x, text, half=1.15, y=6.95, fs=15):
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                weight="bold", color=INK, zorder=6)
        ax.plot([x - half, x + half], [y - 0.40, y - 0.40], color=INK,
                lw=1.5, zorder=6)

    # ── Input layer ───────────────────────────────────────────────────────────
    header(x_in, "Input layer\n(640)")
    spec_ys = [cy + 2.05, cy + 1.50, cy + 0.95]
    spec_low = [cy - 0.05]
    geom_ys = [cy - 0.80, cy - 1.42, cy - 2.04, cy - 2.66]
    in_spec = neuron_column(ax, x_in, spec_ys, INPUT_FILL, INPUT_EDGE, r=r_main)
    vdots(ax, x_in, cy + 0.45)
    in_low = neuron_column(ax, x_in, spec_low, INPUT_FILL, INPUT_EDGE, r=r_main)
    in_geom = neuron_column(ax, x_in, geom_ys, GEOM_FILL, GEOM_EDGE, r=r_main)
    in_pts = in_spec + in_low + in_geom

    ax.text(x_in - 0.55, spec_ys[0], r"$R(\lambda_1)$", ha="right",
            va="center", fontsize=12, color=SUBTLE)
    ax.text(x_in - 0.55, spec_low[0], r"$R(\lambda_{636})$", ha="right",
            va="center", fontsize=12, color=SUBTLE)
    for yy, lbl in zip(geom_ys, ["SZA", "VZA", "SAZ", "VAZ"]):
        ax.text(x_in - 0.50, yy, lbl, ha="right", va="center",
                fontsize=11.5, color=SUBTLE)

    # ── Hidden layers ─────────────────────────────────────────────────────────
    hid_ys = [cy + 1.95, cy + 1.28, cy + 0.61, cy - 1.15]
    hidden_cols = []
    for xh in x_h:
        col = neuron_column(ax, xh, hid_ys, HIDDEN_FILL, HIDDEN_EDGE, r=r_main)
        vdots(ax, xh, cy - 0.30)
        hidden_cols.append(col)
        ax.text(xh, cy - 1.95, "256", ha="center", va="center",
                fontsize=11.5, color=SUBTLE)
    header((x_h[0] + x_h[-1]) / 2, "Hidden layers  (4 × 256, GELU)", half=2.7)

    synapses(ax, in_pts, hidden_cols[0])
    for a, b in zip(hidden_cols[:-1], hidden_cols[1:]):
        synapses(ax, a, b)

    ax.text((x_h[0] + x_h[-1]) / 2, cy - 2.70,
            r"each layer:  Linear $\rightarrow$ LayerNorm $\rightarrow$ "
            r"GELU $\rightarrow$ Dropout",
            ha="center", va="center", fontsize=12, color=SUBTLE,
            style="italic")

    # ── Output heads: mean r_e(z) (upper) + per-level sigma (lower) ────────────
    header(x_out, "Output layer\n(7 levels)", half=1.25)
    mean_ys = [cy + 2.05 - 0.36 * i for i in range(7)]
    sig_ys = [cy - 0.50 - 0.36 * i for i in range(7)]
    out_mean = neuron_column(ax, x_out, mean_ys, MEAN_FILL, MEAN_EDGE, r=r_out)
    out_sig = neuron_column(ax, x_out, sig_ys, SIGMA_FILL, SIGMA_EDGE, r=r_out)

    synapses(ax, hidden_cols[-1], out_mean, color="#9DB8D8", alpha=0.45, lw=0.5)
    synapses(ax, hidden_cols[-1], out_sig, color="#8FD3BC", alpha=0.45, lw=0.5)

    ax.text(x_out + 0.55, (mean_ys[0] + mean_ys[-1]) / 2, r"$r_e(z)$",
            ha="left", va="center", fontsize=15, weight="bold", color=MEAN_EDGE)
    ax.text(x_out + 0.55, (sig_ys[0] + sig_ys[-1]) / 2, r"$\sigma(z)$",
            ha="left", va="center", fontsize=15, weight="bold", color=SIGMA_EDGE)

    ax.annotate(
        "aleatoric uncertainty\n(Gaussian NLL)",
        xy=(x_out + r_out, sig_ys[2]), xytext=(x_out + 1.65, cy - 2.45),
        fontsize=11.5, color=SIGMA_EDGE, ha="center", va="center",
        arrowprops=dict(arrowstyle="-|>", color=SIGMA_EDGE, lw=1.4,
                        connectionstyle="arc3,rad=-0.25"))

    # ── Physics-informed training footer ──────────────────────────────────────
    pill(ax, 7.0, 0.48, 12.4, 0.80,
         "Physics-informed loss:  Gaussian NLL  +  "
         r"$\lambda$(monotonicity · adiabatic shape · smoothness)",
         fill="#F3F0FA", edge="#7B5EA7", fontsize=13, weight="bold",
         text_color="#3F2E63")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(7.0, 7.35,
            "Droplet-profile retrieval network  "
            r"(spectral reflectance $\rightarrow$ vertical $r_e$ profile)",
            ha="center", va="center", fontsize=16.5, weight="bold", color=INK)

    return fig


def parse_args():
    repo_root = Path(__file__).resolve().parent
    default_out = (repo_root.parent
                   / "Presentations_and_papers" / "PhD_defense")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variant", choices=["paper", "defense"], default="defense",
                   help="'paper' = full annotated figure; 'defense' = stripped "
                        "& enlarged for slides (default).")
    p.add_argument("--output-dir", type=str, default=str(default_out),
                   help="Directory for the PNG + PDF (default: PhD_defense/)")
    p.add_argument("--stem", type=str, default=None,
                   help="Output filename stem. Default depends on --variant.")
    p.add_argument("--dpi", type=int, default=500, help="PNG resolution")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.variant == "paper":
        fig = build_figure()
        stem = args.stem or "network_architecture_diagram"
    else:
        fig = build_figure_defense()
        stem = args.stem or "network_architecture_diagram_defense"

    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Wrote ({args.variant}):\n  {png}\n  {pdf}")


if __name__ == "__main__":
    main()
