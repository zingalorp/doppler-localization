import numpy as np

from doppler_loc.adsb import coverage_gaps


def busy(t0, t1, n=6, every=5.0):
    """n aircraft each reporting every `every` s from t0 to t1: (times, aircraft)."""
    t = np.arange(t0, t1, every)
    return np.tile(t, n), np.repeat(np.arange(n), t.size)


def test_outage_in_busy_sky_is_a_gap():
    a, ia = busy(0, 3600)
    b, ib = busy(5400, 9000)  # nothing at all for 30 min in between
    gaps = coverage_gaps(np.r_[a, b], np.r_[ia, ib])
    assert len(gaps) == 1
    start, end = gaps[0]
    assert abs(start - 3600) <= 60 and abs(end - 5400) <= 60


def test_quiet_sky_is_not_a_gap():
    # One aircraft at a time, with long empty stretches between: a quiet area, not an outage,
    # however many positions each reports.
    a, _ = busy(0, 600, n=1, every=1.0)
    b, _ = busy(1800, 2400, n=1, every=1.0)
    c, _ = busy(3600, 4200, n=1, every=1.0)
    assert coverage_gaps(np.r_[a, b, c], np.r_[np.zeros(a.size), np.ones(b.size), np.full(c.size, 2)]) == []


def test_short_dropout_is_ignored():
    a, ia = busy(0, 3600)
    b, ib = busy(3720, 7200)  # 2 min without positions: under the 3 min that counts
    assert coverage_gaps(np.r_[a, b], np.r_[ia, ib]) == []
