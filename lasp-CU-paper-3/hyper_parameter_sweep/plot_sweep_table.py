"""
plot_sweep_table.py — Render the hyperparameter sweep design as a figure.

Reads sweep_configs/sweep_summary.json (produced by generate_sweep.py) and
produces sweep_design_table.png — a single matplotlib table summarizing
the independent variables, their type (categorical / continuous), the
sampled range or set, and the sampling strategy.

Usage:
    python plot_sweep_table.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt

SUMMARY = Path('sweep_configs/sweep_summary.json')
OUT     = Path('sweep_design_table.png')


def fmt_range(values, sci=False):
    lo, hi = min(values), max(values)
    if sci:
        return f"{lo:.2e} – {hi:.2e}"
    return f"{lo:.3f} – {hi:.3f}"


def main():
    with open(SUMMARY) as f:
        summary = json.load(f)

    n_runs = len(summary)

    # Categorical sets actually present in the sweep
    archs   = sorted({tuple(s['hidden_dims']) for s in summary},
                     key=lambda x: (len(x), x[0]))
    weights = sorted({s['level_weights_name'] for s in summary})
    batches = sorted({s['batch_size'] for s in summary})

    arch_str = ", ".join(
        f"{len(a)}x{a[0]}" if all(h == a[0] for h in a) else f"{a[0]}->{a[-1]} ({len(a)}L)"
        for a in archs
    )

    rows = [
        ["Learning rate",   "Continuous (log)",
         fmt_range([s['learning_rate'] for s in summary], sci=True),
         "Latin Hypercube",
         "Bengio 2012; Smith 2017 LR-range test"],
        ["Dropout",         "Continuous",
         fmt_range([s['dropout'] for s in summary]),
         "Latin Hypercube",
         "Srivastava et al. 2014"],
        ["Weight decay",    "Continuous (log)",
         fmt_range([s['weight_decay'] for s in summary], sci=True),
         "Latin Hypercube",
         "Loshchilov & Hutter 2019 (AdamW)"],
        ["Sigma floor",     "Continuous",
         fmt_range([s['sigma_floor'] for s in summary]),
         "Latin Hypercube",
         "Prevents sigma-collapse in NLL"],
        ["Architecture",    f"Categorical ({len(archs)})",
         arch_str,
         "Cycled (shuffled)",
         "He et al. 2016; Karniadakis 2021"],
        ["Level weights",   f"Categorical ({len(weights)})",
         ", ".join(weights),
         "Cycled (shuffled)",
         "Emphasize physically harder levels"],
        ["Batch size",      f"Categorical ({len(batches)})",
         ", ".join(str(b) for b in batches),
         "Cycled (shuffled)",
         "Keskar et al. 2017"],
    ]

    headers = ["Hyperparameter", "Type", "Values / Range", "Sampling", "Reference / Rationale"]

    fig, ax = plt.subplots(figsize=(14, 4.0))
    ax.axis('off')
    ax.set_title(
        f"Hyperparameter Sweep Design ({n_runs} configurations)",
        fontsize=13, fontweight='bold', pad=12,
    )

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='left',
        colLoc='left',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header row
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor('#1f3b70')
        cell.set_text_props(color='white', weight='bold')

    # Alternate row shading
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f5')

    # Column widths (proportional)
    col_widths = [0.13, 0.13, 0.32, 0.13, 0.29]
    for j, w in enumerate(col_widths):
        for i in range(len(rows) + 1):
            table[(i, j)].set_width(w)

    plt.savefig(OUT, dpi=500, bbox_inches='tight')
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
