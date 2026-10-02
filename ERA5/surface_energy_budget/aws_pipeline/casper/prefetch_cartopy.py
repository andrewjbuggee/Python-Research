#!/usr/bin/env python3
"""Download the Natural Earth features the map figures use, once, on a login node.

Casper compute nodes are not guaranteed outbound internet, and cartopy fetches
shapefiles lazily the first time a feature is drawn -- which would fail inside
a batch job after the expensive work is already done. Staging them into
CARTOPY_DATA_DIR (set to a /glade/work path by setup_casper.sh) avoids that.

The scales and features here are exactly the ones the project draws:
map_liquid_hours uses COASTLINE/LAND at 50m and OCEAN/LAND at 110m;
cloud_spatial_extent and turbulent_flux_response reuse the same helpers.
"""
import os
import sys

WANTED = [("physical", "coastline", "50m"), ("physical", "land", "50m"),
          ("physical", "ocean", "110m"), ("physical", "land", "110m"),
          ("physical", "lakes", "50m"), ("cultural", "admin_0_boundary_lines_land", "50m")]

def main() -> int:
    try:
        import cartopy
        from cartopy.io import shapereader
    except ImportError as exc:
        print(f"cartopy not importable: {exc}")
        return 1
    target = os.environ.get("CARTOPY_DATA_DIR")
    if target:
        cartopy.config["data_dir"] = target
    print(f"cartopy data_dir: {cartopy.config['data_dir']}")
    failed = 0
    for category, name, scale in WANTED:
        try:
            path = shapereader.natural_earth(resolution=scale, category=category, name=name)
            print(f"  ok   {scale:>5} {category}/{name} -> {path}")
        except Exception as exc:  # noqa: BLE001 - any download failure matters equally
            print(f"  FAIL {scale:>5} {category}/{name}: {type(exc).__name__}: {exc}")
            failed += 1
    if failed:
        print(f"{failed} feature(s) could not be staged; map figures may fail on a compute node")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
