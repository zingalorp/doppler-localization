"""Does a flight's modelled Doppler curve follow a line on the spectrogram?

Stacking (`stack.py`) finds flights "heard" at a place by how their curve piles up the spectrogram's
peaks with the emitted pitch left free. Over three minutes of peaks a free pitch finds some pile for
many curves, so a heard flight's curve often runs where no line is to be seen. This checks what one
checks by eye: along the curve, frame by frame, is there a bright line within about 1 Hz of it?

Only the curve's informative part is looked at: where its pitch changes (5-95% of its total change,
10 s either side), since a flat stretch matches any steady tone. A frame is a hit when the brightest
score within the tolerance is at least `floor` dB and beats, by `margin` dB, the median of the same
taken along copies of the curve slid up and down in pitch (3-15%): a busy patch of spectrogram
raises the bar for the curve and its copies alike. A line is seen for half a minute to a minute or
two, so the measure is the best 60 s stretch's share of hits (`best60`), set against the copies'
(their median, `chance`, and their best, `chance_max`).

On the test sites, curves of passes that flights explain well score 0.8-1.0 and their copies
0.1-0.3, while West Chicago's heard flights, which can't be seen, score about what their copies do.
"""

from __future__ import annotations

import numpy as np

from .spectro import Spectrogram

SLIDES = np.exp(np.r_[-np.linspace(0.03, 0.15, 5), np.linspace(0.03, 0.15, 5)])
GRADES = ("visible", "unsure", "absent")


def _band_max(spec: Spectrogram, frames, fc, tol_rel, tol_hz):
    """For each frame, the brightest score within the tolerance of pitch fc (tracks, frames)."""
    tol = np.maximum(tol_hz, tol_rel * fc)
    a = np.clip(np.searchsorted(spec.f, fc - tol), 0, len(spec.f) - 1)
    b = np.clip(np.searchsorted(spec.f, fc + tol, "right"), 1, len(spec.f))
    width = int(max((b - a).max(), 1))
    rows = a[..., None] + np.arange(width)
    inside = rows < b[..., None]
    vals = spec.score[np.clip(rows, 0, len(spec.f) - 1), frames[None, :, None]]
    return np.where(inside, vals, -np.inf).max(-1)


def curve_visibility(spec: Spectrogram, curve_t, curve_f, *, tol_rel=0.004, tol_hz=1.0, margin=3.0, floor=6.0,
                     run_s=60.0) -> dict:
    """How well the curve (recording times, Hz) follows a line on `spec`: `best60`, `chance`,
    `chance_max` (see the module) and a `grade`: "visible", "unsure" or "absent"."""
    t, f = np.asarray(curve_t, dtype=float), np.asarray(curve_f, dtype=float)
    ok = np.isfinite(f) & (f > 0)
    t, f = t[ok], f[ok]
    if len(t) < 10:
        return dict(grade="unsure", why="too little of the curve on the recording")
    lf = np.log(f)
    lo, hi = lf.min(), lf.max()
    if hi - lo < 0.015:
        return dict(grade="unsure", why="too flat to tell")
    mid = (lf > lo + 0.05 * (hi - lo)) & (lf < hi - 0.05 * (hi - lo))
    if not mid.any():
        mid[:] = True
    a, b = max(t[mid].min() - 10, t[0]), min(t[mid].max() + 10, t[-1])
    frames = np.flatnonzero((spec.t >= a) & (spec.t <= b))
    if frames.size < 20:
        return dict(grade="unsure", why="too little of the curve on the recording")
    fc = np.interp(spec.t[frames], t, f)
    tracks = _band_max(spec, frames, fc[None, :] * np.r_[1.0, SLIDES][:, None], tol_rel, tol_hz)
    hits = (tracks >= floor) & (tracks - np.median(tracks, axis=0) >= margin)
    w = min(int(round(run_s / spec.hop)), hits.shape[1])
    runs = np.array([np.convolve(h, np.ones(w) / w, "valid").max() for h in hits.astype(float)])
    best, chance, chance_max = float(runs[0]), float(np.median(runs[1:])), float(runs[1:].max())
    if best >= 0.7 and best - chance >= 0.35 and best >= chance_max:
        grade = "visible"
    elif best < 0.35 or best - chance < 0.15:
        grade = "absent"
    else:
        grade = "unsure"
    return dict(grade=grade, best60=round(best, 2), chance=round(chance, 2), chance_max=round(chance_max, 2))
