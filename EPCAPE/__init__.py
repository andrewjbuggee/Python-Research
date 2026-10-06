"""Tools for getting EPCAPE data from the DOE ARM archive.

Typical use from a notebook (the folder that contains EPCAPE/ must be on sys.path):

    from EPCAPE import campaign_dates, get_product, sync_product, combine_product
    start, end = campaign_dates()
    product = get_product("cbh_ceil_M1")
    sync_product(product, start, end)              # download (or find in ARM archive)
    path = combine_product(product, start, end)    # one netCDF in data/processed/
"""
from .download_data.config import active_machine, as_date, campaign_dates, get_product, load_config
from .download_data.sync import sync_datastream, sync_product
from .download_data.combine import combine_files, combine_product, summarize

__all__ = [
    "active_machine", "as_date", "campaign_dates", "get_product", "load_config",
    "sync_datastream", "sync_product", "combine_files", "combine_product", "summarize",
]
