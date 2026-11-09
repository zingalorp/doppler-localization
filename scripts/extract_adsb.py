"""Cut one day's adsb.lol archive down to the flights near a point.

    uv run python scripts/extract_adsb.py data/adsb/2026-09-08 40.87682 -73.52635 80
"""

import sys
import time
from pathlib import Path

from doppler_loc.adsb import default_parts, extract_region

if __name__ == "__main__":
    day_dir, lat, lon, radius = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    out = Path(day_dir) / f"region_{lat:.3f}_{lon:.3f}_{radius:.0f}km.npz"
    t = time.time()
    n_files, n_kept = extract_region(default_parts(day_dir), lat, lon, radius, out)
    print(f"{n_files} aircraft scanned, {n_kept} within {radius} km -> {out} ({time.time() - t:.0f} s)")
