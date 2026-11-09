"""Ground elevation from AWS Terrain Tiles (Terrarium PNGs, ~30 m/pixel at zoom 12)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import threading
import urllib.request

import numpy as np
from PIL import Image

URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def zoom_for_spacing(spacing_m: float) -> int:
    """Tile zoom for a grid `spacing_m` apart: full detail (12, ~30 m) down to 200 m spacing, then
    pixels about a quarter of the spacing (400 m: zoom 10, 1 km: 8). A few metres of ground height
    change nothing kilometres away, and a 500 km area needs some 300 tiles at zoom 10, not 4700."""
    if spacing_m <= 200:
        return 12
    return int(np.clip(12 - np.floor(np.log2(spacing_m / 60.0)), 6, 12))


class TerrainTiles:
    """Elevation above mean sea level in metres; tiles are downloaded once into `cache_dir`.
    `progress(done, total)`, when set, is called as missing tiles download."""

    def __init__(self, cache_dir, zoom=12, progress=None):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.zoom = zoom
        self.progress = progress
        self._tiles = {}

    def _path(self, x, y):
        return self.cache / f"{self.zoom}_{x}_{y}.png"

    def _download(self, key):
        path = self._path(*key)
        with urllib.request.urlopen(URL.format(z=self.zoom, x=key[0], y=key[1]), timeout=60) as r:
            data = r.read()
        tmp = path.with_suffix(f".{threading.get_ident()}.part")
        tmp.write_bytes(data)
        tmp.replace(path)

    def prefetch(self, keys, workers=8):
        """Download the tiles among `keys` not yet cached, `workers` at a time."""
        missing = [k for k in keys if k not in self._tiles and not self._path(*k).exists()]
        if not missing:
            return
        done = 0
        with ThreadPoolExecutor(workers) as ex:
            for _ in ex.map(self._download, missing):
                done += 1
                if self.progress and (done % 20 == 0 or done == len(missing)):
                    self.progress(done, len(missing))

    def _tile(self, x, y):
        key = (x, y)
        if key not in self._tiles:
            path = self._path(x, y)
            if not path.exists():
                self._download(key)
            rgb = np.asarray(Image.open(io.BytesIO(path.read_bytes())).convert("RGB"), dtype=np.float64)
            self._tiles[key] = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
        return self._tiles[key]

    def elevation(self, lat, lon) -> np.ndarray:
        lat, lon = np.broadcast_arrays(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
        # A click on a copy of the world map left or right of the first gives lon beyond +-180, and
        # tiles stop at the Web Mercator limit: both would ask for tiles that don't exist.
        lon = (lon + 180.0) % 360.0 - 180.0
        lat = np.clip(lat, -85.05, 85.05)
        n = 2**self.zoom
        px = (lon + 180) / 360 * n * 256
        lr = np.radians(lat)
        py = (1 - np.log(np.tan(lr) + 1 / np.cos(lr)) / np.pi) / 2 * n * 256
        # Bilinear between pixel centres.
        px, py = px - 0.5, py - 0.5
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        fx, fy = px - x0, py - y0
        # Every tile any corner falls in, fetched in parallel up front rather than one by one.
        self.prefetch({(int(a), int(b)) for dx in (0, 1) for dy in (0, 1)
                       for a, b in np.unique(np.stack([(x0 + dx) // 256, (y0 + dy) // 256], -1).reshape(-1, 2), axis=0)})
        out = np.zeros(lat.shape)
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            gx, gy = x0 + dx, y0 + dy
            tx, ty = gx // 256, gy // 256
            vals = np.empty(lat.shape)
            for key in set(zip(tx.ravel().tolist(), ty.ravel().tolist())):
                sel = (tx == key[0]) & (ty == key[1])
                vals[sel] = self._tile(*key)[gy[sel] % 256, gx[sel] % 256]
            out += w * vals
        return out
