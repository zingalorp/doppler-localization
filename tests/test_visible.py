import numpy as np

from doppler_loc.spectro import Spectrogram
from doppler_loc.visible import curve_visibility


def straight_curve(t, f0=200.0, t_cpa=300.0, speed=60.0, dist=800.0, c=343.0):
    """A straight pass's pitch over recording times t (enough for this test: no delay)."""
    x = speed * (t - t_cpa)
    return f0 / (1 + speed * x / np.hypot(x, dist) / c)


def spectrogram(lines=(), seed=0):
    t = np.arange(0.0, 600.0, 0.5)
    f = np.arange(20.0, 1000.0, 0.5)
    rng = np.random.default_rng(seed)
    score = np.clip(rng.normal(0, 3, (len(f), len(t))), 0, None).astype(np.float32)
    for ft in lines:  # a bright line along each pitch track
        rows = np.clip(np.searchsorted(f, ft(t)), 0, len(f) - 1)
        score[rows, np.arange(len(t))] = 15.0
    return Spectrogram(t, f, np.empty((0, 0), np.float32), score)


def test_curve_on_a_line_is_visible():
    t = np.linspace(120, 480, 361)
    spec = spectrogram([straight_curve])
    v = curve_visibility(spec, t, straight_curve(t))
    assert v["grade"] == "visible", v


def test_curve_where_nothing_is_is_absent():
    t = np.linspace(120, 480, 361)
    spec = spectrogram([lambda tt: straight_curve(tt, f0=320.0)])  # a line, but at another pitch
    v = curve_visibility(spec, t, straight_curve(t))
    assert v["grade"] == "absent", v


def test_flat_curve_cannot_be_told():
    t = np.linspace(120, 480, 361)
    v = curve_visibility(spectrogram(), t, np.full(t.size, 150.0))
    assert v["grade"] == "unsure" and "flat" in v["why"]
