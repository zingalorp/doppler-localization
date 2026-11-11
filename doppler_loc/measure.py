"""Measure aircraft passes on a spectrogram: trace the tone, fit a straight-line (or turning) flyby."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from .physics import SPEED_OF_SOUND
from .spectro import Spectrogram


@dataclass
class Event:
    """One aircraft pass as measured on the recording.

    The ranges are deliberately generous: the straight-line fit is biased when the aircraft
    turns or changes speed, and the exact model in `score` takes over from there.
    """

    t_heard: float  # recording time (s) at which the heard pitch equals f0: the CPA sound
    f0: float  # emitted frequency of the traced line, Hz
    speed: float  # m/s
    dist: float  # slant distance at closest approach, m
    f0_tol: float = 0.02
    speed_tol: float = 0.3
    dist_tol: float = 0.5
    harmonics: tuple[int, ...] = (1,)
    line: tuple[np.ndarray, np.ndarray] | None = None  # the traced (times, freqs) the fit was made to

    @property
    def speed_range(self) -> tuple[float, float]:
        return self.speed * (1 - self.speed_tol), self.speed * (1 + self.speed_tol)

    @property
    def dist_range(self) -> tuple[float, float]:
        return self.dist * (1 - self.dist_tol), self.dist * (1 + self.dist_tol)

    def f0_grid(self, step=0.0025) -> np.ndarray:
        return self.f0 * (1.0 + np.arange(-self.f0_tol, self.f0_tol + 1e-9, step))


def straight_flyby(t, f0, speed, dist, t_heard, c=SPEED_OF_SOUND):
    """Heard frequency for a constant-velocity pass, including the propagation delay.

    Closed form: with u the emission time relative to CPA and w = t - t_heard + dist / c,
    u + sqrt(dist^2 + speed^2 u^2) / c = w is a quadratic in u.
    """
    w = np.asarray(t, dtype=float) - t_heard + dist / c
    a = c**2 - speed**2
    u = (c**2 * w - np.sqrt(c**2 * speed**2 * w**2 + a * dist**2)) / a
    r = c * (w - u)
    v_radial = -(speed**2) * u / r
    return f0 * c / (c - v_radial)


def turning_flyby(t, f0, speed, turn, d0, heading, height, t_ref, c=SPEED_OF_SOUND):
    """Heard frequency for a pass at constant speed, height and turn rate, including the delay.

    At emission time `t_ref` the source is `d0` m from the microphone horizontally (on the +y
    axis), `height` m above it, heading `heading` (rad from +x) and turning at `turn` rad/s. The
    path is integrated over emission times and the heard pitch read at recording times `t`; NaN
    where no sound emitted within 80 s before the first of them arrives then.
    """
    t = np.asarray(t, dtype=float)
    tau = np.arange(t.min() - 80.0, t.max() + 1.0, 0.25)
    th = heading + turn * (tau - t_ref)
    if abs(turn) < 1e-6:
        x = speed * np.cos(heading) * (tau - t_ref)
        y = d0 + speed * np.sin(heading) * (tau - t_ref)
    else:
        x = speed / turn * (np.sin(th) - np.sin(heading))
        y = d0 - speed / turn * (np.cos(th) - np.cos(heading))
    r = np.sqrt(x**2 + y**2 + height**2)
    v_radial = speed * (x * np.cos(th) + y * np.sin(th)) / r
    return np.interp(t, tau + r / c, f0 * c / (c + v_radial), left=np.nan, right=np.nan)


@dataclass
class TurnFit:
    """A `turning_flyby` fitted to a traced line (see `fit_turning`)."""

    params: tuple  # (f0, speed, turn, d0, heading, height)
    t_ref: float
    share: float  # of the line within tol_hz of the curve

    def curve(self, t, c=SPEED_OF_SOUND):
        return turning_flyby(t, *self.params, self.t_ref, c)


def fit_turning(times, freqs, f0, *, max_height=6000.0, speeds=(20.0, 150.0), max_turn=0.12, tol_hz=1.5,
                c=SPEED_OF_SOUND) -> TurnFit:
    """Fit a flight at constant speed, height and turn rate to a traced line.

    Real planes turn, which bends the line away from any straight pass: a plane turning towards the
    microphone even rises in pitch. With the turn rate free (up to `max_turn` rad/s, about 7 deg/s)
    such a line can be followed, which says whether an aircraft could have made it. Speed, distance
    and heading trade off against each other here, so only the curve and the emitted pitch should
    be trusted. Several starting turns, headings and distances are tried.
    """
    all_t = np.asarray(times, dtype=float)
    all_f = np.asarray(freqs, dtype=float)
    pick = np.unique(np.linspace(0, len(all_t) - 1, min(len(all_t), 80)).round().astype(int))
    times, freqs = all_t[pick], all_f[pick]  # enough to fit the shape, and much faster
    t_ref = float(np.median(times))
    lo = [f0 * 0.8, speeds[0], -max_turn, 0.0, -np.pi, 50.0]
    hi = [f0 * 1.2, speeds[1], max_turn, 15000.0, np.pi, max_height]

    def resid(x):
        m = turning_flyby(times, *x, t_ref, c)
        return np.where(np.isfinite(m), m - freqs, 50.0)

    best = None
    for turn in (-0.04, 0.04):
        for heading in np.linspace(-np.pi, np.pi, 6, endpoint=False):
            for d0 in (600.0, 2000.0):
                x0 = np.clip([f0, 60.0, turn, d0, heading, 500.0], lo, hi)
                res = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.5, max_nfev=100)
                if best is None or res.cost < best.cost:
                    best = res
    share = float(np.mean(np.abs(turning_flyby(all_t, *best.x, t_ref, c) - all_f) < tol_hz))
    return TurnFit(tuple(float(v) for v in best.x), t_ref, share)


def trace_ridge(
    spec: Spectrogram, t_range, f_range, *, max_jump_hz=2.0, jump_cost=0.5, guide=None, band_hz=6.0, falling=False,
    pull_db_per_hz=0.0, cap_db=None,
):
    """Follow the strongest continuous line through a box of the spectrogram.

    Dynamic programming over frequency bins: maximise the summed score along the path minus a
    penalty per bin of frequency jump between frames. Returns (times, freqs, strength).

    On real recordings steady tones cross the curve; `guide`, a few hand-picked (t, f) points
    along it, keeps the path within `band_hz` of the line through them. With `falling`, the path
    may only go down or stay level, as a straight pass's pitch does, so it cannot climb onto
    another line that crosses or joins it. `cap_db` caps how bright a frame can count and
    `pull_db_per_hz` takes that much off per Hz away from the guide, so that something much
    brighter than the line beside it can't pull the path off the guide (strengths are uncapped).
    """
    ti = np.flatnonzero((spec.t >= t_range[0]) & (spec.t <= t_range[1]))
    fi = np.flatnonzero((spec.f >= f_range[0]) & (spec.f <= f_range[1]))
    S = spec.score[np.ix_(fi, ti)].astype(np.float64)
    if guide is not None:  # band_hz: one width, or one per frame of t_range
        gt, gf = np.asarray(guide, dtype=float).T
        centre = np.interp(spec.t[ti], gt, gf)
        off = np.abs(spec.f[fi][:, None] - centre[None, :])
        S = np.where(off <= band_hz, S, -5.0)
    W = S if cap_db is None else np.minimum(S, cap_db)
    if guide is not None and pull_db_per_hz:
        W = W - pull_db_per_hz * off
    F, T = S.shape
    n = int(round(max_jump_hz / spec.df))
    jumps = np.arange(0 if falling else -n, n + 1)  # previous bin minus current bin
    bins = np.arange(F)

    acc = W[:, 0].copy()
    back = np.zeros((F, T), dtype=int)
    for k in range(1, T):
        prev = bins[None, :] + jumps[:, None]
        ok = (prev >= 0) & (prev < F)
        cand = np.where(ok, acc[np.clip(prev, 0, F - 1)] - jump_cost * np.abs(jumps)[:, None], -np.inf)
        best = cand.argmax(0)
        back[:, k] = prev[best, bins]
        acc = cand[best, bins] + W[:, k]

    path = np.empty(T, dtype=int)
    path[-1] = acc.argmax()
    for k in range(T - 1, 0, -1):
        path[k - 1] = back[path[k], k]
    return spec.t[ti], spec.f[fi][path], S[path, np.arange(T)]


def trace_by_hand(spec: Spectrogram, points, *, near_hz=1.5, far_hz=4.0, hz_per_s=0.15, min_strength=3.0,
                  cap_db=10.0, pull_db_per_hz=2.0):
    """Trace a line along hand-placed (t, f) points, trusting them over brighter things nearby.

    The points are joined by a smooth curve that cannot overshoot them (PCHIP), not straight lines,
    so it follows the bend between points placed far apart. The line is looked for within `near_hz`
    of that curve at each point, the band widening by `hz_per_s` per second away from the nearest
    one to at most `far_hz`: what was clicked on is where the line is. Within the band, brightness
    counts up to `cap_db` and less by `pull_db_per_hz` per Hz off the curve, so a much brighter tone
    crossing it can't hold the trace more than about a hertz away. Where the spectrogram has no
    line at least `min_strength` dB within the band, the hand curve itself stands in for it (at
    that strength, so the fit uses it) rather than the brightest speck there.
    Returns (times, freqs, strength) as `trace_ridge` does.
    """
    from scipy.interpolate import PchipInterpolator

    pts = np.asarray(sorted((float(t), float(f)) for t, f in points))
    gt, idx = np.unique(pts[:, 0], return_inverse=True)  # clicks at one time: their mean pitch
    gf = np.bincount(idx, pts[:, 1]) / np.bincount(idx)
    t_range = (gt[0], gt[-1])
    frames = spec.t[(spec.t >= t_range[0]) & (spec.t <= t_range[1])]
    curve = PchipInterpolator(gt, gf)(frames) if len(gt) > 2 else np.interp(frames, gt, gf)
    band = np.clip(near_hz + hz_per_s * np.abs(frames[:, None] - gt[None, :]).min(1), near_hz, far_hz)
    times, freqs, strength = trace_ridge(spec, t_range, (curve.min() - far_hz - 1, curve.max() + far_hz + 1),
                                         guide=list(zip(frames, curve)), band_hz=band, cap_db=cap_db,
                                         pull_db_per_hz=pull_db_per_hz)
    faint = strength < min_strength
    freqs = np.where(faint, curve, freqs)
    strength = np.where(faint, min_strength, strength)
    return times, freqs, strength


MIN_FIT_FRAMES = 8  # strong frames a line needs for the faint ones to be left out of its fit


def fit_flyby(times, freqs, strength=None, *, min_strength=3.0, c=SPEED_OF_SOUND) -> Event:
    """Fit `straight_flyby` to a traced line, ignoring frames where the line is too faint.

    A line with hardly any frames that strong is fitted whole: a handful of frames can't be fitted,
    and what was traced is then judged like any other pass rather than stopping detection.
    """
    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    if strength is not None:
        keep = np.asarray(strength) >= min_strength
        if keep.sum() >= MIN_FIT_FRAMES:
            times, freqs = times[keep], freqs[keep]

    k = max(len(freqs) // 5, 1)
    f_hi, f_lo = np.median(freqs[:k]), np.median(freqs[-k:])
    f0 = 2.0 / (1.0 / f_hi + 1.0 / f_lo)
    speed = float(np.clip(c * (f_hi - f_lo) / (f_hi + f_lo), 20.0, 250.0))
    t_mid = float(times[np.argmin(np.abs(freqs - f0))])

    def resid(x):
        return straight_flyby(times, *x, c=c) - freqs

    lo = [f0 * 0.8, 10.0, 50.0, times[0]]
    hi = [f0 * 1.2, 0.9 * c, 30000.0, times[-1]]
    best = None
    for d0 in (300.0, 1000.0, 3000.0):
        x0 = np.clip([f0, speed, d0, t_mid], lo, hi)
        res = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.5)
        if best is None or res.cost < best.cost:
            best = res
    f0, speed, dist, t_heard = best.x
    return Event(t_heard=float(t_heard), f0=float(f0), speed=float(speed), dist=float(dist), line=(times, freqs))


def implausible(event: Event, *, max_dist=10000.0, speeds=(20.0, 200.0), tol_hz=1.5, min_fit=0.5, c=SPEED_OF_SOUND):
    """Why a measured pass cannot be one a search would explain, or None if it can.

    The rules restate the search's own assumptions rather than describe any one recording:
    flights are only looked for within `max_dist` (a pass fitted farther away has no candidate),
    only propeller aircraft are searched (`speeds` spans what they fly), and a traced line the
    straight-line fit does not follow (less than `min_fit` of it within `tol_hz`) measured
    something other than a pass.
    """
    if event.dist > max_dist:
        return f"closest approach {event.dist / 1000:.1f} km, beyond the {max_dist / 1000:.0f} km searched"
    if not speeds[0] <= event.speed <= speeds[1]:
        return f"speed {event.speed:.0f} m/s, outside {speeds[0]:.0f}-{speeds[1]:.0f} m/s"
    if event.line is not None:
        times, freqs = event.line
        fit = np.mean(np.abs(straight_flyby(times, event.f0, event.speed, event.dist, event.t_heard, c) - freqs) < tol_hz)
        if fit < min_fit:
            return f"only {fit:.0%} of the traced line within {tol_hz} Hz of the fit"
    return None


# Ratios a line can have to another of the same aircraft: propellers and engines make several series
# of harmonics, so the line traced may itself be 3/2 or 4/3 of another.
HARMONIC_RATIOS = tuple(sorted({m / n for n in range(1, 5) for m in range(1, 7) if m != n and 0.2 <= m / n <= 6}))


def harmonics_of(spec: Spectrogram, t, f, *, ratios=HARMONIC_RATIOS, min_contrast=4.0, min_cover=0.3, tol=0.006):
    """The other lines of the aircraft that made a traced line (times `t`, pitches `f`): every ratio r
    with a line running along r x the trace, brighter than what lies beside it. Returns (r, that
    line's pitch / r at each frame, its strength there, or 0 where it strays) for each. On 45 passes
    matched with flights, ratios that mean nothing are found 0.2 times a pass at these settings,
    real ones 1.1 times, and on the same 12 passes nearly every time: so look for several."""
    t, f = np.asarray(t, dtype=float), np.asarray(f, dtype=float)
    window = min((len(f) - 1) // 2 * 2 + 1, 21)
    curve = savgol_filter(f, window, 2) if window >= 5 else f  # a trace jitters more than the ratio's tolerance
    found = []
    for r in ratios:
        target = r * curve
        if target.min() < spec.f[0] + 5.0 or target.max() > spec.f[-1] - 10.0:
            continue
        band = max(2.0, 1.5 * r)
        rt, rf, strength = trace_ridge(spec, (t[0], t[-1]), (target.min() - band - 1, target.max() + band + 1), guide=list(zip(t, target)),
                                       band_hz=band, cap_db=12.0, pull_db_per_hz=1.0 / r)
        if len(rt) != len(t):
            continue
        near = np.abs(np.log(rf / target)) < tol
        frames = np.clip(np.searchsorted(spec.t, t), 0, len(spec.t) - 1)
        beside = [spec.score[np.clip(np.searchsorted(spec.f, target + s * (band + 3.0)), 0, len(spec.f) - 1), frames] for s in (-1, 1)]
        strong = (strength >= 3.0) & near
        if strong.mean() >= min_cover and np.median(strength[strong]) - np.median(np.maximum(*beside)[strong]) >= min_contrast:
            found.append((float(r), rf / r, np.where(near, strength, 0.0)))
    return found


def sharpen_with_harmonics(spec: Spectrogram, t, f, strength=None, *, min_lines=2):
    """A traced line's pitches at each frame, taken from the line and its harmonics together, and the
    ratios found. Every harmonic has the same Doppler shape, and one r times higher moves r times as
    many Hz over the same bins, so it counts r^2 as much (a weighted median, so a stray line can't
    pull far). With fewer than `min_lines` others found the line stays as it is: one can be chance,
    enough to show a person but not to change the line."""
    f = np.asarray(f, dtype=float)
    found = harmonics_of(spec, t, f)
    if len(found) < min_lines:
        return f, [x[0] for x in found]
    if strength is None:
        frames = np.clip(np.searchsorted(spec.t, t), 0, len(spec.t) - 1)
        strength = spec.score[np.clip(np.searchsorted(spec.f, f), 0, len(spec.f) - 1), frames]
    pitch = np.array([f] + [x[1] for x in found])
    weight = np.array([np.maximum(strength, 1.0)] + [x[2] * x[0] ** 2 for x in found])
    order = np.argsort(pitch, axis=0)
    pitch, weight = np.take_along_axis(pitch, order, 0), np.take_along_axis(weight, order, 0)
    total = np.cumsum(weight, axis=0)
    return pitch[(total >= 0.5 * total[-1]).argmax(0), np.arange(f.size)], [x[0] for x in found]


def trim_tails(t, f, strength, *, tol_hz=3.0, floor_db=4.0, win=9, need=0.7, rounds=3, keep_at_least=0.5, min_fit=0.6, c=SPEED_OF_SOUND):
    """Where a detected line really ends: (first frame, one past the last) to keep. A trace must give a
    pitch at every frame of its stretch, so past where the plane fades it runs on along whatever is
    nearby. Brightness alone can't say where: real tails are often faint, and they are what fixes the
    speed. So each end is cut back to where, over a few seconds, most of the line follows the flyby
    fitted to what is left and isn't at the level of the noise (the median at or under `floor_db`).
    A line no straight pass follows (`min_fit`) is left alone: it may be a turn, not a bad tail."""
    t, f, strength = (np.asarray(x, dtype=float) for x in (t, f, strength))
    n = len(t)
    a, b = 0, n
    if n < 3 * win:
        return a, b
    pad = win // 2
    level = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(strength, pad, mode="edge"), win), axis=1)
    for _ in range(rounds):
        ev = fit_flyby(t[a:b], f[a:b], c=c)
        good = np.abs(straight_flyby(t, ev.f0, ev.speed, ev.dist, ev.t_heard, c) - f) < tol_hz
        if np.mean(good[a:b]) < min_fit:
            return (0, n) if a == 0 and b == n else (a, b)
        kernel = np.ones(win)
        run = np.convolve(good.astype(float), kernel, mode="same") / np.convolve(np.ones(n), kernel, mode="same")
        ok = (run >= need) & (level > floor_db)
        new_a, new_b = a, b
        while new_a < new_b and not ok[new_a]:
            new_a += 1
        while new_b > new_a and not ok[new_b - 1]:
            new_b -= 1
        if new_b - new_a < keep_at_least * n or (new_a, new_b) == (a, b):
            break
        a, b = new_a, new_b
    return a, b
