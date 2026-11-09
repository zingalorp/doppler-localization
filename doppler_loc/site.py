"""Where a recording might have been made: a local frame, ground heights and the geoid.

ADS-B geometric altitude is height above the WGS84 ellipsoid, while elevation models give height
above mean sea level (the geoid). The two differ by -100 m to +80 m around the world (-32 m near
New York, +45 m over central Europe), so microphone positions get the EGM96 geoid height added.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .dem import TerrainTiles, zoom_for_spacing
from .geo import LocalFrame


@lru_cache(maxsize=1)
def _to_geoid():
    import pyproj

    pyproj.network.set_network_enabled(True)  # fetches the EGM96 grid from cdn.proj.org once
    return pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4326+5773", always_xy=True)


def geoid_height(lat, lon) -> np.ndarray:
    """EGM96 geoid height above the WGS84 ellipsoid, m."""
    lat, lon = np.broadcast_arrays(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
    _, _, h = _to_geoid().transform(lon, lat, np.zeros(lat.shape))
    return -np.asarray(h, dtype=float)


class Site:
    """A local frame centred on (lat, lon), with microphones standing on the ground."""

    def __init__(self, lat, lon, dem_dir, *, mic_height=2.0, zoom=12, wide=False):
        self.lat, self.lon = float(lat), float(lon)
        self.frame = LocalFrame(self.lat, self.lon)
        self.dem = TerrainTiles(dem_dir, zoom)
        self.mic_height = mic_height
        self.geoid = float(geoid_height(self.lat, self.lon))  # varies by < 1 m over 50 km
        self.wide = wide  # over a country it varies by tens of metres: look it up for each point
        self._coarser = {zoom: self}

    def for_spacing(self, spacing_m) -> "Site":
        """This site with terrain as detailed as a grid `spacing_m` apart needs (`dem.zoom_for_spacing`):
        the same frame and geoid, coarser tiles for a coarse grid, so wide areas download far fewer."""
        zoom = zoom_for_spacing(spacing_m)
        if zoom not in self._coarser:
            other = object.__new__(Site)
            other.__dict__.update(self.__dict__)
            other.dem = TerrainTiles(self.dem.cache, zoom, self.dem.progress)
            self._coarser[zoom] = other
        return self._coarser[zoom]

    def on_download(self, progress):
        """Report tile downloads (done, total) through `progress`, at every level of detail."""
        for s in self._coarser.values():
            s.dem.progress = progress

    def place(self, x, y) -> np.ndarray:
        """Microphone positions (P, 3) in the frame for ground points at frame (x, y)."""
        lat, lon = self.frame.ground_latlon(x, y)
        geoid = geoid_height(lat, lon) if self.wide else self.geoid
        h = np.maximum(self.dem.elevation(lat, lon), 0.0) + geoid + self.mic_height  # water reads as <= 0
        return self.frame.to_enu(lat, lon, h)

    def to_latlon(self, xyz):
        return self.frame.to_latlon(xyz)

    def to_xy(self, lat, lon) -> np.ndarray:
        return self.frame.to_enu(lat, lon, np.zeros(np.shape(lat)))[..., :2]
