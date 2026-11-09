"""Synthetic worlds with a known answer, for checking the search end to end."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .flight import FlightTrack, Maneuver, adsb_sample, simulate_track
from .physics import SPEED_OF_SOUND, closest_approach
from .synth import Source, synthesize

MIC_HEIGHT = 1.5


def hills(x, y):
    """Stand-in for a DEM: rolling terrain 60-210 m above sea level."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return 120 + 60 * np.sin(x / 3000) * np.cos(y / 4000) + 30 * np.sin((x + y) / 1700)


@dataclass
class Scenario:
    name: str
    truth: np.ndarray  # true microphone position (x, y, z)
    true_tracks: list[FlightTrack]  # what the aircraft really did
    adsb: list[FlightTrack]  # what the flight data says they did (the solver only sees this)
    audio: np.ndarray
    sr: int
    t_rec: float  # absolute time the recording started
    duration: float
    ground: Callable
    marks: list[tuple[tuple[float, float], tuple[float, float]]]  # (t_range, f_range) boxes around each pass
    f0s: list[float]  # true emitted frequency of each marked pass


def _pass_by(flight_id, truth, *, heading, speed, alt, side_offset, t_cpa, before=600.0, after=600.0,
             maneuvers=(), typecode="C172") -> FlightTrack:
    """A track whose straight-line CPA to `truth` is `side_offset` m to the right of its path at `t_cpa`.

    Maneuver times are relative to the CPA (negative = before).
    """
    h = np.deg2rad(heading)
    fwd = np.array([np.sin(h), np.cos(h)])
    right = np.array([fwd[1], -fwd[0]])
    cpa_xy = truth[:2] - side_offset * right
    start = np.r_[cpa_xy - fwd * speed * before, alt]
    shifted = [Maneuver(m.start + before, m.end + before, m.turn_rate, m.accel, m.climb_rate) for m in maneuvers]
    return simulate_track(flight_id, start, heading, speed, t_cpa - before, before + after, shifted, typecode=typecode)


def single_pass(*, maneuvering=True, duration=300.0, level=0.25, seed=0, sr=8000) -> Scenario:
    """One propeller aircraft passes a microphone whose clock is roughly right.

    A perfectly straight, steady pass is mirror-symmetric about the flight path, so it can only
    pin the listener to two spots. Turning or slowing near the pass breaks the symmetry.
    """
    rng = np.random.default_rng(seed)
    truth = np.array([1200.0, -800.0, 0.0])
    truth[2] = hills(*truth[:2]) + MIC_HEIGHT
    t_rec = 45_000.0
    maneuvers = (
        [Maneuver(-20, 0, turn_rate=2.0), Maneuver(-60, 40, climb_rate=-3.0), Maneuver(0, 25, accel=-0.6)]
        if maneuvering else []
    )
    tr = _pass_by("N4521P", truth, heading=60.0, speed=62.0, alt=1150.0, side_offset=1300.0,
                  t_cpa=t_rec + 150.0 - 5.0, maneuvers=maneuvers)
    f0 = 84.0
    audio = synthesize(truth, [Source(tr, f0, level=level)], t_rec, duration, sr=sr, seed=seed)
    return Scenario(
        name="single_pass_maneuvering" if maneuvering else "single_pass_straight",
        truth=truth, true_tracks=[tr], adsb=[adsb_sample(tr, rng=rng)], audio=audio, sr=sr,
        t_rec=t_rec, duration=duration, ground=hills,
        marks=[((60.0, 240.0), (f0 * 0.78, f0 * 1.25))], f0s=[f0],
    )


def two_passes(*, n_decoys=60, duration=1200.0, level=0.25, seed=0, sr=8000) -> Scenario:
    """Two passes about 15 minutes apart, among many other flights, at an unknown time."""
    rng = np.random.default_rng(seed)
    truth = np.array([3000.0, 2000.0, 0.0])
    truth[2] = hills(*truth[:2]) + MIC_HEIGHT
    t_rec = 50_000.0 + rng.uniform(0, 3600)
    c = SPEED_OF_SOUND

    a = _pass_by("N738QA", truth, heading=120.0, speed=58.0, alt=900.0, side_offset=-900.0,
                 t_cpa=t_rec + 190.0 - 3.0, maneuvers=[Maneuver(10, 40, turn_rate=-1.5)])
    b = _pass_by("N2209K", truth, heading=15.0, speed=75.0, alt=1500.0, side_offset=1600.0,
                 t_cpa=t_rec + 1105.0 - 5.0, maneuvers=[Maneuver(-90, 120, climb_rate=2.5)], typecode="PC12")
    tracks = [a, b]
    f0s = [96.0, 135.0]

    while len(tracks) < 2 + n_decoys:
        tr = _random_flight(rng, f"D{len(tracks):03d}", t_lo=t_rec - 5400, t_hi=t_rec + 5400)
        ca = closest_approach(tr, truth[None], c)
        in_recording = (ca.t_heard[0] > t_rec - 120) & (ca.t_heard[0] < t_rec + duration + 120)
        if ca.dist[0] < 8000 and in_recording:
            continue  # the recording only has the two passes in it
        tracks.append(tr)

    audio = synthesize(truth, [Source(a, f0s[0], level=level), Source(b, f0s[1], level=1.5 * level)],
                       t_rec, duration, sr=sr, seed=seed)
    marks = [((190.0 - 90, 190.0 + 90), (f0s[0] * 0.8, f0s[0] * 1.22)),
             ((1105.0 - 90, 1105.0 + 90), (f0s[1] * 0.75, f0s[1] * 1.3))]
    return Scenario(
        name="two_passes_unknown_time", truth=truth, true_tracks=tracks,
        adsb=[adsb_sample(t, rng=rng) for t in tracks], audio=audio, sr=sr,
        t_rec=t_rec, duration=duration, ground=hills, marks=marks, f0s=f0s,
    )


def _random_flight(rng, flight_id, *, t_lo, t_hi, radius=30_000.0) -> FlightTrack:
    jet = rng.random() < 0.15
    speed = rng.uniform(180, 240) if jet else rng.uniform(45, 95)
    alt = rng.uniform(3000, 9000) if jet else rng.uniform(500, 3000)
    ang = rng.uniform(0, 2 * np.pi)
    start = radius * np.array([np.cos(ang), np.sin(ang)])
    aim = rng.uniform(-20_000, 20_000, 2)
    d = aim - start
    heading = np.rad2deg(np.arctan2(d[0], d[1]))
    duration = 2 * radius / speed
    maneuvers = [
        Maneuver(s, s + rng.uniform(10, 60), turn_rate=rng.uniform(-3, 3),
                 accel=rng.uniform(-0.3, 0.3), climb_rate=rng.uniform(-4, 4))
        for s in rng.uniform(0, duration, rng.integers(0, 3))
    ]
    return simulate_track(flight_id, np.r_[start, alt], heading, speed, rng.uniform(t_lo, t_hi), duration,
                          maneuvers, typecode="B738" if jet else "C208")
