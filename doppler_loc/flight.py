"""Aircraft trajectories as continuous functions of time.

Everything is in a local east/north/up frame in metres. Altitude (`pos[:, 2]`) is measured
from the same datum as ground elevation: ADS-B reports height above sea level, so observers
need DEM heights rather than zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, CubicHermiteSpline, CubicSpline, make_smoothing_spline


class FlightTrack:
    """Position and velocity of one aircraft, interpolated from timestamped samples.

    With `smooth=True` the samples are treated as noisy (as ADS-B reports are) and fitted with
    a smoothing spline, so the velocity used for Doppler shifts is not dominated by GPS jitter.
    With `vel` (reported velocities at each sample) the interpolation is cubic Hermite instead.
    """

    def __init__(self, flight_id, t, pos, *, vel=None, smooth=False, typecode=""):
        self.flight_id = flight_id
        self.typecode = typecode
        self.t = np.asarray(t, dtype=float)
        self.pos = np.asarray(pos, dtype=float)
        # Splines run on time relative to the first sample to keep them well conditioned.
        self._t_ref = self.t[0]
        rel = self.t - self._t_ref
        # One vector-valued spline, so each evaluation is a single call rather than one per axis.
        if vel is not None:
            self._pos = CubicHermiteSpline(rel, self.pos, np.asarray(vel, dtype=float))
        elif smooth:
            # Each axis picks its own smoothing (GPS noise across, altitude steps up); the knots
            # are the sample times either way, so the coefficients stack into one spline.
            axes = [make_smoothing_spline(rel, self.pos[:, k]) for k in range(3)]
            self._pos = BSpline(axes[0].t, np.stack([s.c for s in axes], axis=-1), axes[0].k)
        else:
            self._pos = CubicSpline(rel, self.pos)
        self._vel = self._pos.derivative()

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    def position(self, t) -> np.ndarray:
        return self._pos(np.asarray(t, dtype=float) - self._t_ref)

    def velocity(self, t) -> np.ndarray:
        return self._vel(np.asarray(t, dtype=float) - self._t_ref)

    def contains(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return (t >= self.t_start) & (t <= self.t_end)


@dataclass(frozen=True)
class Maneuver:
    """A constant-rate change between `start` and `end` seconds after the track begins."""

    start: float
    end: float
    turn_rate: float = 0.0  # deg/s, positive turns right (clockwise seen from above)
    accel: float = 0.0  # m/s^2 along the flight path
    climb_rate: float = 0.0  # m/s, negative descends


def simulate_track(
    flight_id,
    start,
    heading_deg,
    speed,
    t_start,
    duration,
    maneuvers=(),
    *,
    dt=0.1,
    typecode="C172",
) -> FlightTrack:
    """Fly from `start` (x, y, altitude) on a compass heading, applying `maneuvers`."""
    n = int(round(duration / dt)) + 1
    rel = np.arange(n) * dt
    turn = np.zeros(n)
    accel = np.zeros(n)
    climb = np.zeros(n)
    for m in maneuvers:
        on = (rel >= m.start) & (rel < m.end)
        turn[on] += m.turn_rate
        accel[on] += m.accel
        climb[on] += m.climb_rate

    heading = np.deg2rad(heading_deg) + np.concatenate([[0.0], np.cumsum(np.deg2rad(turn[:-1]) * dt)])
    spd = speed + np.concatenate([[0.0], np.cumsum(accel[:-1] * dt)])
    vel = np.stack([spd * np.sin(heading), spd * np.cos(heading), climb], axis=1)
    steps = np.cumsum(0.5 * (vel[1:] + vel[:-1]) * dt, axis=0)
    pos = np.asarray(start, dtype=float) + np.concatenate([np.zeros((1, 3)), steps])
    return FlightTrack(flight_id, t_start + rel, pos, typecode=typecode)


def adsb_sample(
    track: FlightTrack,
    *,
    interval=1.0,
    jitter=0.3,
    pos_noise=5.0,
    alt_step=7.62,
    dropout=0.05,
    rng=None,
) -> FlightTrack:
    """Imitate an ADS-B feed of `track`: irregular reports, GPS noise, 25 ft altitude steps."""
    rng = np.random.default_rng(rng)
    n = int((track.t_end - track.t_start) / interval) + 1
    t = track.t_start + np.arange(n) * interval + rng.uniform(-jitter, jitter, n)
    t = np.unique(np.clip(t, track.t_start, track.t_end))
    keep = rng.random(len(t)) > dropout
    keep[[0, -1]] = True
    t = t[keep]

    pos = track.position(t)
    pos[:, :2] += rng.normal(0.0, pos_noise, (len(t), 2))
    pos[:, 2] = np.round(pos[:, 2] / alt_step) * alt_step
    return FlightTrack(track.flight_id, t, pos, smooth=True, typecode=track.typecode)
