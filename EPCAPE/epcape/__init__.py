"""Tools for getting EPCAPE data from the DOE ARM archive.

Typical use from a notebook (the repository folder must be on sys.path):

    from epcape import campaign_dates, get_product, sync_product, combine_product
    start, end = campaign_dates()
    product = get_product("cbh_ceil_M1")
    sync_product(product, start, end)              # download (or find in ARM archive)
    path = combine_product(product, start, end)    # one netCDF in data/processed/
"""
from .config import active_machine, as_date, campaign_dates, get_product, load_config
from .sync import sync_datastream, sync_product
from .combine import combine_files, combine_product, summarize

__all__ = [
    "active_machine", "as_date", "campaign_dates", "get_product", "load_config",
    "sync_datastream", "sync_product", "combine_files", "combine_product", "summarize",
]
