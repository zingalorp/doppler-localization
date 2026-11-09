"""Tell aircraft passes from other falling tones by how the traced line looks, not only how loud it is.

The detector's confidence (straight-pass fit share x strength) is high for anything loud with a
falling pitch: trains, cars and lorries, birds. Those differ from an aircraft in ways a person
looking at the spectrogram sees at once:

- a ground vehicle is close, so its pitch drops in a few seconds, and it is loud across the whole
  band as it goes by (tyres, wind, rails), while an aircraft a kilometre or two up sweeps over
  tens of seconds and barely lifts the broadband floor;
- an engine on the road changes speed and gear, so its line wobbles about any smooth curve;
- an aircraft's tone is a thin line well above the bins beside it, often with harmonics, and is
  heard on both sides of the closest approach.

`pass_features` measures these from the spectrogram and the traced line, and `PassModel` is a
logistic regression over them, trained on a labelled set of detected passes (not included here).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from scipy.ndimage import median_filter

from .measure import straight_flyby, trace_ridge
from .physics import SPEED_OF_SOUND
from .spectro import Spectrogram

MODEL_PATH = Path(__file__).with_name("pass_model.json")

FEATURES = (
    "fit_share",  # share of the line within 1.5 Hz of the straight-pass fit
    "fit_mad_hz",  # median distance of the line from that fit, Hz
    "log_strength",  # detection strength (sum along the matched curve, in noise units)
    "log_speed",  # fitted speed, m/s
    "log_dist",  # fitted closest approach, m
    "log_fall_s",  # seconds the fitted pitch takes from 10% to 90% of its fall
    "change",  # log(max/min) pitch of the line
    "roughness",  # median |second difference| of the line, Hz per frame^2
    "contrast_db",  # line brightness above the bins 4-12 Hz either side, dB
    "visible",  # share of the line's frames at least 3 dB above the bins beside it
    "both_sides",  # the lesser of the visible shares before and after closest approach
    "burst",  # broadband rise near closest approach over the minutes around it, in their spreads
    "rise",  # the most the pitch climbs (log ratio) from 25 s before the line to closest approach, x its visibility
    "lead",  # share of the 40 s before the line where a tone continues it, 3 dB above the bins beside
    "tail",  # the same for the 40 s after
    "stripes",  # share of the line's frames on a broadband click or burst (6 dB over the local floor)
    "harmonics",  # number of harmonics found with it
    "log_f0",  # emitted pitch, Hz
)


def _fall_seconds(speed, dist, c=SPEED_OF_SOUND):
    tau = np.linspace(-300.0, 300.0, 6001)
    y = np.log(straight_flyby(tau, 1.0, speed, dist, 0.0, c))
    top, bottom = np.log(c / (c - speed)), np.log(c / (c + speed))
    frac = (top - y) / (top - bottom)
    return float(tau[np.argmax(frac > 0.9)] - tau[np.argmax(frac > 0.1)])


def _above_beside(spec, fi, ti):
    """dB of bins (fi, ti) above the median of the bins 4-12 Hz either side."""
    k_lo, k_hi = int(round(4 / spec.df)), int(round(12 / spec.df))
    beside = [spec.db[np.clip(fi + s * k, 0, len(spec.f) - 1), ti] for s in (-1, 1) for k in range(k_lo, k_hi + 1)]
    return spec.db[fi, ti] - np.median(beside, axis=0)


def _follow(spec, ti, fi, step, n, max_jump_hz=1.5):
    """Greedily follow the brightest bin from (fi, ti) for n frames in direction step (+1 or -1):
    the (frequency bins, frame indices) visited, not including the start."""
    jump = max(int(round(max_jump_hz / spec.df)), 1)
    fs, ts = [], []
    for _ in range(n):
        ti += step
        if not 0 <= ti < len(spec.t):
            break
        lo, hi = max(fi - jump, 0), min(fi + jump + 1, len(spec.f))
        fi = lo + int(np.argmax(spec.score[lo:hi, ti]))
        fs.append(fi)
        ts.append(ti)
    return np.array(fs, int), np.array(ts, int)


def broadband(spec: Spectrogram) -> np.ndarray:
    """Median dB across the band, per frame: the floor a passing vehicle lifts."""
    return np.median(spec.db, axis=0)


def pass_features(spec: Spectrogram, p: dict, band_db: np.ndarray | None = None, c=SPEED_OF_SOUND) -> dict:
    """Features of one pass record (as `Project._pass_record` makes it) on its spectrogram."""
    times, freqs = np.asarray(p["line_t"], float), np.asarray(p["line_f"], float)
    fit = straight_flyby(times, p["f0"], p["speed"], p["dist"], p["t_heard"], c)
    resid = np.abs(freqs - fit)

    ti = np.clip(np.searchsorted(spec.t, times), 0, len(spec.t) - 1)
    fi = np.clip(np.rint((freqs - spec.f[0]) / spec.df).astype(int), 0, len(spec.f) - 1)
    above = _above_beside(spec, fi, ti)  # per frame, dB
    contrast = float(np.median(above))
    visible = above >= 3.0
    before, after = visible[times < p["t_heard"]], visible[times >= p["t_heard"]]
    both = min(before.mean() if len(before) else 0.0, after.mean() if len(after) else 0.0)

    # Beyond the traced (falling) line: an aircraft is heard approaching and leaving at nearly its end
    # pitches, while a bird, or a vehicle speeding up, rises into the fall and a short sound stops.
    n = int(round(40 / spec.hop))
    bf, bt = _follow(spec, ti[0], fi[0], -1, n)
    af, at = _follow(spec, ti[-1], fi[-1], +1, n)
    lead = float(np.mean(_above_beside(spec, bf, bt) >= 3.0)) if len(bf) else 0.0
    tail = float(np.mean(_above_beside(spec, af, at) >= 3.0)) if len(af) else 0.0
    # Rising into the fall: re-trace from 25 s before the line to closest approach, free to go up.
    rise = 0.0
    a, b = times[0] - 25.0, max(p["t_heard"], times[0] + 5.0)
    if a >= spec.t[0]:
        rt, rf, _ = trace_ridge(spec, (a, b), (freqs[0] * 0.75, freqs[0] * 1.3), max_jump_hz=2.0)
        rf = median_filter(rf, 5, mode="nearest")
        up = float(np.max(np.log(rf) - np.minimum.accumulate(np.log(rf))))
        rti = np.clip(np.searchsorted(spec.t, rt), 0, len(spec.t) - 1)
        rfi = np.clip(np.rint((rf - spec.f[0]) / spec.df).astype(int), 0, len(spec.f) - 1)
        rise = up * float(np.mean(_above_beside(spec, rfi, rti) >= 3.0))

    band_db = broadband(spec) if band_db is None else band_db
    floor_t = band_db[max(ti[0] - n, 0) : ti[-1] + n]
    stripes = float(np.mean(band_db[ti] - np.median(floor_t) > 6.0))
    near = np.abs(spec.t - p["t_heard"]) <= 10.0
    around = (np.abs(spec.t - p["t_heard"]) > 30.0) & (np.abs(spec.t - p["t_heard"]) <= 150.0)
    burst = 0.0
    if near.any() and around.any():
        floor = band_db[around]
        spread = np.median(np.abs(floor - np.median(floor))) + 0.1
        burst = float((np.median(band_db[near]) - np.median(floor)) / spread)

    return dict(
        fit_share=float(np.mean(resid < 1.5)),
        fit_mad_hz=float(np.median(resid)),
        log_strength=float(np.log(max(p.get("strength") or 10.0, 1.0))),
        log_speed=float(np.log(p["speed"])),
        log_dist=float(np.log(p["dist"])),
        log_fall_s=float(np.log(max(_fall_seconds(p["speed"], p["dist"], c), 0.5))),
        change=float(np.log(freqs.max() / freqs.min())),
        roughness=float(np.median(np.abs(np.diff(freqs, 2)))) if len(freqs) > 2 else 0.0,
        contrast_db=contrast,
        visible=float(visible.mean()),
        both_sides=float(both),
        burst=burst,
        rise=rise,
        lead=lead,
        tail=tail,
        stripes=stripes,
        harmonics=float(len(p.get("harmonics") or ())),
        log_f0=float(np.log(p["f0"])),
    )


def fit_logistic(X, y, *, l2=1.0, iters=50, weights=None):
    """Weights (bias first) of an L2-penalised logistic regression, by Newton's method."""
    X1 = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X1.shape[1])
    sw = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    pen = np.full(X1.shape[1], l2)
    pen[0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X1 @ w))
        grad = X1.T @ (sw * (p - y)) + pen * w
        H = (X1 * (sw * p * (1 - p))[:, None]).T @ X1 + np.diag(pen)
        step = np.linalg.solve(H, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


@dataclass
class PassModel:
    """Logistic regression over standardised `FEATURES`: the probability a pass is an aircraft."""

    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray  # bias first

    @classmethod
    def train(cls, rows: list[dict], y, *, names=FEATURES, l2=1.0, balance=True):
        X = np.array([[r[n] for n in names] for r in rows], float)
        y = np.asarray(y, float)
        mean, std = X.mean(0), X.std(0) + 1e-9
        # Aircraft are a small minority: weight the classes equally, so the model is not just "never".
        w = np.where(y > 0, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9)) if balance else None
        return cls(tuple(names), mean, std, fit_logistic((X - mean) / std, y, l2=l2, weights=w))

    def prob(self, rows: list[dict]) -> np.ndarray:
        X = (np.array([[r[n] for n in self.names] for r in rows], float) - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-(self.weights[0] + X @ self.weights[1:])))

    def save(self, path=MODEL_PATH):
        Path(path).write_text(json.dumps(dict(names=self.names, mean=self.mean.tolist(), std=self.std.tolist(),
                                              weights=self.weights.tolist()), indent=1))

    @classmethod
    def load(cls, path=MODEL_PATH):
        d = json.loads(Path(path).read_text())
        return cls(tuple(d["names"]), np.array(d["mean"]), np.array(d["std"]), np.array(d["weights"]))

