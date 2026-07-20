"""Shared helpers for the plotting scripts (path bootstrap + figure saving)."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo importable when running scripts straight from a checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402

from arm_nsa import config  # noqa: E402


def finish_figure(fig, filename: str, show: bool) -> Path:
    """Save `fig` into figures/ (always) and optionally display it."""
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURE_DIR / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"figure saved: {out_path}")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return out_path


def use_headless_backend_if_needed(show: bool) -> None:
    """Select a non-interactive backend when the user only wants files."""
    if not show:
        matplotlib.use("Agg")
