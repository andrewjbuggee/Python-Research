"""Configuration: campaign dates, machine-specific paths and named products.

Everything machine-specific lives in config.yaml, so the same code runs on a
laptop, the UCSD Research Cluster, or ARM's JupyterHub/Cumulus.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def config_path() -> Path:
    return Path(os.environ.get("EPCAPE_CONFIG", REPO_ROOT / "config.yaml"))


def load_config(path=None) -> dict:
    path = Path(path) if path else config_path()
    with open(path) as f:
        return yaml.safe_load(f) or {}


def as_date(value) -> dt.date:
    """Accept a date, datetime or 'YYYY-MM-DD' string."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value).strip())


def campaign_dates(cfg: Optional[dict] = None) -> Tuple[dt.date, dt.date]:
    cfg = cfg if cfg is not None else load_config()
    c = cfg["campaign"]
    return as_date(c["start"]), as_date(c["end"])


@dataclass(frozen=True)
class Machine:
    name: str
    data_root: Path
    arm_archive: Optional[Path] = None

    def full_dir(self, datastream: str) -> Path:
        """Complete ARM files, named exactly as in the ARM archive."""
        return self.data_root / "arm" / datastream

    def subset_dir(self, product: str) -> Path:
        """Server-side variable subsets of a datastream, one folder per product."""
        return self.data_root / "arm_subset" / product

    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    def archive_dir(self, datastream: str) -> Optional[Path]:
        """Folder of this datastream in a mounted ARM archive, if there is one."""
        if self.arm_archive is None:
            return None
        site = datastream[:3]
        for candidate in (self.arm_archive / site / datastream, self.arm_archive / datastream):
            try:
                if candidate.is_dir() and any(candidate.iterdir()):
                    return candidate
            except OSError:  # exists but not readable
                continue
        return None


def active_machine(cfg: Optional[dict] = None) -> Machine:
    cfg = cfg if cfg is not None else load_config()
    machines = cfg.get("machines") or {}
    name = os.environ.get("EPCAPE_MACHINE", "local").strip()
    if name not in machines:
        raise KeyError(
            f"EPCAPE_MACHINE={name!r} is not defined in {config_path()}. "
            f"Defined machines: {', '.join(machines) or '(none)'}"
        )
    entry = machines[name] or {}
    root = os.environ.get("EPCAPE_DATA_ROOT") or entry.get("data_root") or "data"
    if "PROJECT_ID" in str(root):
        raise ValueError(
            f"data_root for machine {name!r} in {config_path()} still contains the "
            "placeholder PROJECT_ID; replace it with your ARM HPC project ID."
        )
    root = Path(os.path.expandvars(os.path.expanduser(str(root))))
    if not root.is_absolute():
        root = REPO_ROOT / root
    archive = entry.get("arm_archive")
    archive = Path(os.path.expanduser(str(archive))) if archive else None
    return Machine(name=name, data_root=root, arm_archive=archive)


@dataclass(frozen=True)
class Product:
    name: str
    datastream: str
    variables: List[str] = field(default_factory=list)
    description: str = ""
    # Kept when the datastream has them, skipped silently when it does not
    # (ancillary fields whose names or presence vary between VAP versions).
    optional_variables: List[str] = field(default_factory=list)


def get_product(name: str, cfg: Optional[dict] = None) -> Product:
    cfg = cfg if cfg is not None else load_config()
    products = cfg.get("products") or {}
    if name not in products:
        raise KeyError(
            f"Unknown product {name!r}. Products defined in {config_path()}: "
            f"{', '.join(products) or '(none)'}"
        )
    p = products[name] or {}
    if not p.get("datastream"):
        raise ValueError(f"Product {name!r} in {config_path()} has no datastream.")
    return Product(
        name=name,
        datastream=str(p["datastream"]).strip(),
        variables=[str(v).strip() for v in (p.get("variables") or [])],
        description=str(p.get("description", "")),
        optional_variables=[str(v).strip() for v in (p.get("optional_variables") or [])],
    )
