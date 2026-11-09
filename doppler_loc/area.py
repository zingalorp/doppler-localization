"""A search area that isn't a square: a shape drawn on the map, or a country or state.

A shape is a list of rings, each a list of (lat, lon) corners. The search still runs over the square
around it (`centre_and_half`), so flight data and grids stay as they were; what lies outside the
shape is left out (`AreaMask`).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from .geo import LocalFrame


def clean_rings(rings) -> list:
    """`rings` as [[lat, lon], ...] lists, or a ValueError saying what is wrong with them."""
    out = []
    try:
        for ring in rings:
            pts = [[float(a), float(b)] for a, b in ring]
            if len(pts) < 3:
                raise ValueError("a shape needs at least three corners")
            if any(abs(a) > 90 or abs(b) > 540 for a, b in pts):
                raise ValueError("a corner of the shape is off the map")
            out.append(pts)
    except TypeError as e:
        raise ValueError("a shape is a list of rings of [lat, lon] corners") from e
    if not out:
        raise ValueError("the shape is empty")
    if sum(len(r) for r in out) > 50000:
        raise ValueError("the shape has too many corners")
    return out


def _unwrapped(rings) -> list:
    """The rings with longitudes running on from the biggest ring's: a country across the 180th
    meridian (or a shape drawn past it) would otherwise span the whole world."""
    ref = float(np.mean(np.asarray(max(rings, key=len), dtype=float)[:, 1]))
    return [np.c_[r[:, 0], (r[:, 1] - ref + 180.0) % 360.0 - 180.0 + ref] for r in (np.asarray(x, dtype=float) for x in rings)]


def _along_edges(ring, step_deg=0.25):
    """`ring` with corners added along its long edges: an edge is straight on the map, and a straight
    line in the search's frame between far-apart corners runs kilometres away from it."""
    ends = np.roll(ring, -1, axis=0)
    out = [np.linspace(a, b, max(int(np.ceil(np.abs(b - a).max() / step_deg)), 1), endpoint=False) for a, b in zip(ring, ends)]
    return np.concatenate(out)


def centre_and_half(rings, margin_km=2.0) -> tuple[float, float, float]:
    """(lat, lon, half_km) of the smallest frame-aligned square around `rings`."""
    pts = np.concatenate(_unwrapped(rings))
    frame = LocalFrame(float((pts[:, 0].min() + pts[:, 0].max()) / 2), float((pts[:, 1].min() + pts[:, 1].max()) / 2))
    xy = frame.to_enu(pts[:, 0], pts[:, 1], np.zeros(len(pts)))[:, :2]
    mid = (xy.min(0) + xy.max(0)) / 2
    lat, lon = frame.ground_latlon(mid[0], mid[1])
    frame = LocalFrame(float(lat), float(lon))
    xy = frame.to_enu(pts[:, 0], pts[:, 1], np.zeros(len(pts)))[:, :2]
    half = float(np.abs(xy).max()) / 1000.0 + margin_km
    lon = (float(lon) + 180.0) % 360.0 - 180.0
    return round(float(lat), 5), round(lon, 5), round(half, 1)


class AreaMask:
    """Whether frame points (x, y) lie in the shape, or within `margin_m` of it (a candidate is
    refined a few hundred metres from where it was found), read from a raster of it: a country
    can have thousands of corners and a search asks about hundreds of millions of points."""

    def __init__(self, rings, frame: LocalFrame, half_m, cells=4096, margin_m=1000.0):
        self.reach = float(half_m) * 1.02 + margin_m
        self.cell = 2.0 * self.reach / cells
        image = Image.new("1", (cells, cells), 0)
        draw = ImageDraw.Draw(image)
        for r in _unwrapped(rings):
            r = _along_edges(r)
            xy = frame.to_enu(r[:, 0], r[:, 1], np.zeros(len(r)))[:, :2]
            draw.polygon([tuple(p) for p in (xy + self.reach) / self.cell], fill=1, outline=1)
        mask = np.array(image, dtype=bool)
        grow = int(np.ceil(margin_m / self.cell))
        self.mask = binary_dilation(mask, iterations=grow) if grow else mask

    def contains(self, x, y) -> np.ndarray:
        i = np.floor((np.asarray(x, dtype=float) + self.reach) / self.cell).astype(np.int64)
        j = np.floor((np.asarray(y, dtype=float) + self.reach) / self.cell).astype(np.int64)
        n = self.mask.shape[0]
        inside = (i >= 0) & (i < n) & (j >= 0) & (j < n)
        out = np.zeros(i.shape, bool)
        out[inside] = self.mask[j[inside], i[inside]]
        return out
