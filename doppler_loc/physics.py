"""Sound propagation from a moving aircraft to a fixed listener."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flight import FlightTrack

SPEED_OF_SOUND = 343.0  # m/s at about 20 °C


def emission_times(track: FlightTrack, observers, t_rx, c=SPEED_OF_SOUND, iters=3) -> np.ndarray:
    """When was the sound that reaches `observers` at `t_rx` emitted?

    Solves t_e + |p(t_e) - x| / c = t_rx with Newton's method; 3 steps get within microseconds
    at 10 km. `observers` (..., 3) broadcasts against `t_rx` (...). The heard pitch depends on
    the aircraft's motion at t_e, not at t_rx, and that is not the same as delaying the
    instantaneous curve by a constant.
    """
    obs = np.asarray(observers, dtype=float)
    t_rx = np.asarray(t_rx, dtype=float)
    t_e = t_rx - np.linalg.norm(track.position(t_rx) - obs, axis=-1) / c
    for _ in range(iters):
        d = track.position(t_e) - obs
        r = np.linalg.norm(d, axis=-1)
        r_dot = (track.velocity(t_e) * d).sum(-1) / r
        t_e = t_e - (t_e + r / c - t_rx) / (1.0 + r_dot / c)
    return t_e


def doppler_factor(track: FlightTrack, observers, t_rx, c=SPEED_OF_SOUND, *, retarded=True) -> np.ndarray:
    """Ratio of heard to emitted frequency at `observers` at times `t_rx`.

    NaN where the sound would have been emitted outside the track's time span. With
    `retarded=False` the aircraft's state at `t_rx` itself is used (the naive model).
    """
    obs = np.asarray(observers, dtype=float)
    t_rx = np.asarray(t_rx, dtype=float)
    t_e = emission_times(track, obs, t_rx, c) if retarded else t_rx
    d = obs - track.position(t_e)
    v_radial = (track.velocity(t_e) * d).sum(-1) / np.linalg.norm(d, axis=-1)  # + = approaching
    g = c / (c - v_radial)
    return np.where(track.contains(t_e), g, np.nan)


@dataclass
class ClosestApproach:
    t: np.ndarray  # emission time of closest approach (CPA)
    dist: np.ndarray  # slant distance at CPA, m
    t_heard: np.ndarray  # when the CPA sound reaches the observer (the "delayed CPA")
    speed: np.ndarray  # aircraft 3D speed at CPA, m/s
    valid: np.ndarray  # False when the closest point is an end of the track


def closest_approach(
    track: FlightTrack, observers, c=SPEED_OF_SOUND, *, dt=4.0, chunk=8192, max_dist=None, tile=5000.0
) -> ClosestApproach:
    """Closest point of approach of `track` to each of `observers` (P, 3).

    With `max_dist`, only approaches closer than that are wanted: observers are grouped into
    `tile`-sized squares and each square is compared only with the stretch of track that comes
    within `max_dist` of it, which is most of the work saved for a track that just clips the
    area. Observers the track never comes that close to are marked invalid.
    """
    obs = np.atleast_2d(np.asarray(observers, dtype=float))
    P = len(obs)
    n = max(int(np.ceil((track.t_end - track.t_start) / dt)), 2) + 1
    ts = np.linspace(track.t_start, track.t_end, n)
    step = ts[1] - ts[0]
    pts = track.position(ts)

    if max_dist is None:
        groups = [(np.arange(P), np.arange(n))]
    else:
        # The closest point can fall between samples, so allow one sample spacing extra.
        reach = max_dist + np.linalg.norm(np.diff(pts, axis=0), axis=1).max()
        cell = np.floor(obs[:, :2] / tile).astype(np.int64)
        cell -= cell.min(0)
        key = cell[:, 0] + cell[:, 1] * (cell[:, 0].max() + 1)
        order = np.argsort(key, kind="stable")
        groups = []
        for members in np.split(order, np.flatnonzero(np.diff(key[order])) + 1):
            lo = obs[members, :2].min(0) - reach
            hi = obs[members, :2].max(0) + reach
            near = np.flatnonzero(((pts[:, :2] >= lo) & (pts[:, :2] <= hi)).all(1))
            if near.size:
                groups.append((members, near))

    # Nearest sample for each observer, -1 where none is in reach.
    i = np.full(P, -1)
    for members, near in groups:
        p = pts[near]
        p_sq = (p**2).sum(1)
        for s in range(0, len(members), chunk):
            m = members[s : s + chunk]
            d2 = (obs[m] ** 2).sum(1)[:, None] + p_sq[None] - 2.0 * obs[m] @ p.T
            i[m] = near[d2.argmin(1)]

    # Parabola through the neighbouring samples, then Newton on d/dt |p(t) - x|^2 = 0 on the spline.
    todo = np.flatnonzero(i >= 0)
    o = obs[todo]
    ic = np.clip(i[todo], 1, n - 2)
    y0, y1, y2 = (((pts[ic + k] - o) ** 2).sum(1) for k in (-1, 0, 1))
    curv = y0 - 2.0 * y1 + y2
    off = np.where(curv > 0, 0.5 * (y0 - y2) / np.where(curv > 0, curv, 1.0), 0.0)
    t_sub = ts[ic] + np.clip(off, -1.0, 1.0) * step
    for _ in range(3):
        d = track.position(t_sub) - o
        v = track.velocity(t_sub)
        t_sub = np.clip(t_sub - (v * d).sum(1) / np.maximum((v**2).sum(1), 1e-6), track.t_start, track.t_end)

    t = np.full(P, np.nan)
    dist = np.full(P, np.inf)
    speed = np.full(P, np.nan)
    t[todo] = t_sub
    dist[todo] = np.linalg.norm(track.position(t_sub) - o, axis=1)
    speed[todo] = np.linalg.norm(track.velocity(t_sub), axis=1)
    valid = (i > 0) & (i < n - 1) & (t > track.t_start) & (t < track.t_end)
    if max_dist is not None:
        valid &= dist < max_dist
    return ClosestApproach(t=t, dist=dist, t_heard=t + dist / c, speed=speed, valid=valid)
