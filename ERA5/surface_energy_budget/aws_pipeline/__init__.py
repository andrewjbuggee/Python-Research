"""ERA5 from the NSF NCAR S3 bucket, as a drop-in ``--storage aws`` for the SEB analyses.

    era5_s3      lazy, windowed, cached reader for s3://nsf-ncar-era5
    s3_storage   the adapter that plugs it into seb_analysis_common /
                 surface_classification / cloud_level_wind as ``storage="aws"``
    run_analysis batch driver: the Ocean Visions figure set at any box and season
    verify_against_local   bit-for-bit checks against the local CDS archive
    remote/      EC2 (us-west-2) bootstrap and job scripts

Nothing here is imported by the analysis modules unless ``storage="aws"`` is
requested, so local runs are unaffected by this package being present.
"""

from . import era5_s3  # noqa: F401  (re-export for `from aws_pipeline import era5_s3`)

__all__ = ["era5_s3"]
