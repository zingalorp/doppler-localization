"""WGS84 coordinates to a local east/north/up frame in metres."""

from __future__ import annotations

import numpy as np

A = 6378137.0  # WGS84 semi-major axis
E2 = 6.69437999014e-3  # first eccentricity squared
FT = 0.3048
KT = 1852.0 / 3600.0


def mercator_to_latlon(x, y):
    """EPSG:3857 (web mercator, as QGIS shows) to degrees."""
    lon = np.degrees(np.asarray(x) / A)
    lat = np.degrees(2 * np.arctan(np.exp(np.asarray(y) / A)) - np.pi / 2)
    return lat, lon


def geodetic_to_ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.stack(
        [(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon), (n * (1 - E2) + h) * np.sin(lat)],
        axis=-1,
    )


def ecef_to_geodetic(xyz):
    """Latitude, longitude (degrees) and height for ECEF points (..., 3): Bowring's formula, then
    two more steps, which is well under a millimetre for anything near the ground."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(4):
        n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - E2 * n / (n + h)))
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(np.arctan2(y, x)), p / np.cos(lat) - n


def _enu_basis(lat, lon):
    """Rows: east, north, up unit vectors in ECEF, shape (..., 3, 3)."""
    lat, lon = np.radians(lat), np.radians(lon)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    z = np.zeros_like(sl)
    east = np.stack([-so, co, z], -1)
    north = np.stack([-sl * co, -sl * so, cl], -1)
    up = np.stack([cl * co, cl * so, sl], -1)
    return np.stack([east, north, up], -2)


class LocalFrame:
    """Tangent-plane frame at `origin`. Heights are ellipsoidal (as ADS-B geometric altitude is).

    Distances in this frame are true 3D distances, so Earth curvature (70 m of drop at 30 km)
    is handled without any flat-earth approximation.
    """

    def __init__(self, lat0, lon0, h0=0.0):
        self.lat0, self.lon0 = lat0, lon0
        self.origin = geodetic_to_ecef(lat0, lon0, h0)
        self.basis = _enu_basis(lat0, lon0)

    def to_enu(self, lat, lon, h):
        return (geodetic_to_ecef(lat, lon, h) - self.origin) @ self.basis.T

    def vel_to_enu(self, lat, lon, v_enu_local):
        """Rotate velocities given in each point's own east/north/up into this frame."""
        ecef = np.einsum("...ij,...i->...j", _enu_basis(lat, lon), v_enu_local)
        return ecef @ self.basis.T

    def to_latlon(self, xyz):
        """Where frame points (..., 3) are: the inverse of `to_enu`."""
        lat, lon, _ = ecef_to_geodetic(self.origin + np.asarray(xyz, dtype=float) @ self.basis)
        return lat, lon

    def height(self, xyz):
        """Height above the ellipsoid of frame points (..., 3): z alone falls away with distance from
        the frame's centre, by 450 km at 2,400 km."""
        return ecef_to_geodetic(self.origin + np.asarray(xyz, dtype=float) @ self.basis)[2]

    def ground_latlon(self, x, y):
        """The place on the ellipsoid at frame position (x, y): a frame's plane rises above the
        ground away from its centre, by 450 km at 2,400 km, so that height is found first."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        z = -(x**2 + y**2) / (2 * A)
        for _ in range(8):
            lat, lon, h = ecef_to_geodetic(self.origin + np.stack([x, y, z], axis=-1) @ self.basis)
            z = z - h
        return lat, lon
