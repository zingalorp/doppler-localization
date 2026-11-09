import numpy as np
import pytest

from doppler_loc.area import AreaMask, centre_and_half, clean_rings
from doppler_loc.geo import LocalFrame

# A triangle in Tennessee, about 300 km across.
RING = [[35.0, -87.0], [35.0, -84.0], [37.0, -85.5]]


def test_square_around_a_shape():
    lat, lon, half = centre_and_half([RING])
    assert 35.5 < lat < 36.5 and -86 < lon < -85
    frame = LocalFrame(lat, lon)
    xy = frame.to_enu(*np.array(RING).T, np.zeros(3))[:, :2]
    assert np.abs(xy).max() / 1000 <= half


def test_mask_keeps_what_is_inside():
    lat, lon, half = centre_and_half([RING])
    frame = LocalFrame(lat, lon)
    mask = AreaMask([RING], frame, half * 1000, margin_m=0.0)
    inside = frame.to_enu(35.5, -85.5, 0.0)[:2]
    outside = frame.to_enu(36.8, -86.8, 0.0)[:2]
    assert mask.contains(*inside) and not mask.contains(*outside)
    assert not mask.contains(1e9, 0.0)  # off the raster altogether


def test_margin_reaches_past_the_edge():
    lat, lon, half = centre_and_half([RING])
    frame = LocalFrame(lat, lon)
    just_south = frame.to_enu(35.0 - 800.0 / 111200.0, -85.5, 0.0)[:2]  # 800 m beyond the south side
    assert not AreaMask([RING], frame, half * 1000, margin_m=0.0).contains(*just_south)
    assert AreaMask([RING], frame, half * 1000, margin_m=1500.0).contains(*just_south)


def test_shape_across_the_date_line():
    ring = [[50.0, 175.0], [50.0, -175.0], [55.0, 180.0]]
    lat, lon, half = centre_and_half([ring])
    assert abs(abs(lon) - 180.0) < 3 and half < 800


def test_bad_shapes_are_refused():
    for bad in ([], [[[1, 2], [3, 4]]], [[[95, 0], [1, 1], [2, 2]]], [1, 2]):
        with pytest.raises(ValueError):
            clean_rings(bad)
