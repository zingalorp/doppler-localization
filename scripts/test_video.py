"""Run the pipeline on the video's test recording (youtube.com/watch?v=4CxVxuMVkMc).

Recorded 2026-09-08 somewhere in the continental US; the answer is 40.876382, -73.526214 by
Bayview Ave on Oyster Bay Harbor, Long Island.

    uv run python scripts/test_video.py verify   # which flights, at the known spot?
    uv run python scripts/test_video.py search   # blind search of a 50 x 50 km box
    uv run python scripts/test_video.py pairs    # the same, choosing which two passes to use

The passes are found on the spectrogram automatically. verify and search use the two strongest;
pairs drops detections the search could not explain and tries every pair of the rest.
Add --hand to measure the video's two passes from the hand-placed guides in MARKS instead.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from doppler_loc import plots
from doppler_loc.adsb import Archive, is_jet, load_type_table
from doppler_loc.dem import TerrainTiles
from doppler_loc.detect import detect, measure
from doppler_loc.geo import LocalFrame
from doppler_loc.measure import fit_flyby, implausible, straight_flyby, trace_ridge
from doppler_loc.physics import closest_approach
from doppler_loc.search import (
    observer_grid, refine_pair, score_pair, search_event_pairs, search_two_events,
)
from doppler_loc.spectro import Spectrogram

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out" / "test_video"
REGION = DATA / "adsb" / "2026-09-08" / "region_40.877_-73.526_80km.npz"

TRUE_LAT, TRUE_LON = 40.87638222902607, -73.52621377807661  # exact spot, from the person who set the test
GEOID = -32.5  # EGM96 geoid height near Oyster Bay, m: ADS-B geometric altitude is above the ellipsoid
MIC_HEIGHT = 4.0  # security camera on a pole

# Hand-placed guides along each curve, as read off the zoomed spectrogram (used with --hand).
MARKS = {
    "A": dict(t_range=(2088, 2240), f_range=(75, 190),
              guide=[(2090, 178), (2110, 166), (2130, 145), (2145, 125), (2160, 108), (2180, 98), (2200, 92), (2240, 85)]),
    "B": dict(t_range=(3040, 3135), f_range=(95, 180),
              guide=[(3042, 174), (3055, 165), (3065, 150), (3080, 124), (3095, 111), (3110, 106), (3135, 102)]),
}

HAND = False
frame = LocalFrame(TRUE_LAT, TRUE_LON)
dem = TerrainTiles(DATA / "dem")


def place(x, y):
    lat, lon = frame.to_latlon(np.stack([x, y, np.zeros_like(x)], axis=-1))
    h = np.maximum(dem.elevation(lat, lon), 0.0) + GEOID + MIC_HEIGHT  # water reads as <= 0
    return frame.to_enu(lat, lon, h)


def utc(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M:%S")


def find_events(spec, n):
    """The `n` strongest passes on the recording, measured, strongest first."""
    t0 = time.time()
    dets = detect(spec, max_events=n)
    print(f"passes detected on the recording ({time.time() - t0:.0f} s), strongest first:")
    for k, det in enumerate(dets):
        print(f"  D{k + 1}: heard at {det.t_heard:6.1f} s  f0 {det.f0:5.1f} Hz  ~{det.speed:3.0f} m/s  ~{det.dist:5.0f} m  "
              f"strength {det.strength:4.1f}" + (f"  harmonics x{det.harmonics}" if det.harmonics else ""))
    events = [measure(spec, det) for det in dets]

    cols = min(len(dets), 3)
    rows = -(-len(dets) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
    for k, (ax, det, ev) in enumerate(zip(axes.ravel(), dets, events)):
        t_range = (det.t_range[0] - 20, det.t_range[1] + 20)
        plots.spectrogram(ax, spec, t_range, (0.6 * ev.f0, 1.7 * ev.f0), vmax=14)
        t = np.linspace(*t_range, 400)
        ax.plot(t, det.curve(t), "c--", lw=1, label="detected shape")
        ax.plot(t, straight_flyby(t, ev.f0, ev.speed, ev.dist, ev.t_heard), "lime", lw=1,
                label=f"fit: {ev.speed:.0f} m/s, CPA {ev.dist:.0f} m")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"D{k + 1}, strength {det.strength:.0f}")
    for ax in axes.ravel()[len(dets):]:
        ax.set_visible(False)
    plots.save(fig, OUT / "detections.png")
    return events


def load(n_events=2):
    """Spectrogram, measured events with their names, and the day's flight legs.

    With --hand the two events come from the hand-placed guides; otherwise they are the
    `n_events` strongest detections.
    """
    d = np.load(DATA / "test_video" / "spec.npz")
    spec = Spectrogram(d["t"], d["f"], d["db"], d["score"])
    if HAND:
        events = [fit_flyby(*trace_ridge(spec, m["t_range"], m["f_range"], guide=m["guide"])) for m in MARKS.values()]
        names = list(MARKS)
    else:
        events = find_events(spec, n_events)
        names = [f"D{k + 1}" for k in range(len(events))]
    for k, ev in zip(names, events):
        ev.speed_tol, ev.dist_tol = 0.4, 0.7  # partial curves: speed and distance trade off
        print(f"event {k}: f0 {ev.f0:.1f} Hz, {ev.speed:.0f} m/s, CPA {ev.dist:.0f} m, heard at {ev.t_heard:.1f} s")
    t0 = time.time()
    tracks = Archive(REGION).tracks(frame)
    print(f"{len(tracks)} flight legs within 80 km ({time.time() - t0:.0f} s)")
    return spec, events, names, tracks


def describe(tr, ca, i):
    m = getattr(tr, "meta", {})
    alt = tr.position(ca.t[i])[2]
    return (f"{tr.flight_id:<10} {m.get('t') or '?':<5} {(m.get('desc') or '')[:28]:<28} "
            f"CPA {ca.dist[i]:5.0f} m  {ca.speed[i]:4.0f} m/s  alt {alt:5.0f} m  heard {utc(ca.t_heard[i])}Z")


def verify():
    spec, events, _, tracks = load()
    e1, e2 = sorted(events[:2], key=lambda ev: ev.t_heard)
    truth = place(np.array([0.0]), np.array([0.0]))
    delta = e2.t_heard - e1.t_heard
    cpas = [closest_approach(tr, truth) for tr in tracks]
    near = [(i, ca) for i, ca in enumerate(cpas) if ca.valid[0] and ca.dist[0] < 15000]
    print(f"{len(near)} legs pass within 15 km of the true spot during the day")

    shifts = np.arange(-8, 9)
    pairs = []
    for i, ca in near:
        for j, cb in near:
            if i != j and abs(cb.t_heard[0] - ca.t_heard[0] - delta) < 15:
                s1, s2 = score_pair(spec, tracks[i], tracks[j], truth, e1, e2, ca.t_heard, cb.t_heard, shifts)
                pairs.append((s1.score[0] + s2.score[0], i, j, ca, cb, s1, s2))
    pairs.sort(key=lambda p: -p[0])
    print(f"\n{len(pairs)} flight pairs heard {delta:.0f} ± 15 s apart at the true spot; best by curve score:")
    for tot, i, j, ca, cb, s1, s2 in pairs[:6]:
        print(f"  score {s1.score[0]:5.2f} + {s2.score[0]:5.2f} = {tot:5.2f}")
        print(f"    A: {describe(tracks[i], ca, 0)}")
        print(f"    B: {describe(tracks[j], cb, 0)}")
    if not pairs:
        return

    tot, i, j, ca, cb, s1, s2 = pairs[0]
    t0 = ca.t_heard[0] - e1.t_heard + s1.shift[0] * spec.hop
    print(f"\nrecording would have started at {utc(t0)}Z (ended {utc(t0 + spec.t[-1])}Z)")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, k, ev, tr, f0 in zip(axes, "AB", (e1, e2), (tracks[i], tracks[j]), (s1.f0[0], s2.f0[0])):
        t_range = (ev.t_heard - 120, ev.t_heard + 120)
        plots.spectrogram(ax, spec, t_range, (0.6 * ev.f0, 1.7 * ev.f0), vmax=14)
        plots.model_curve(ax, spec, tr, truth[0], t0, f0, t_range, color="lime", lw=1,
                          label=f"{tr.flight_id} at the true spot")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"event {k}")
    plots.save(fig, OUT / "verify_curves.png")


def candidate_flights(tracks, center):
    """Propeller (or unknown) aircraft that come within audible range of the box, low enough to hear."""
    types = load_type_table(DATA / "types" / "icao_aircraft_types2.js")
    keep = []
    for tr in tracks:
        if is_jet(tr.typecode, types):
            continue
        p = tr.pos
        inside = (np.abs(p[:, 0] - center[0]) < 35000) & (np.abs(p[:, 1] - center[1]) < 35000) & (p[:, 2] < 6000)
        if inside.sum() >= 5:
            keep.append(tr)
    print(f"{len(keep)} non-jet legs come near the box below 6 km")
    return keep


CENTER = np.array([-6000.0, -4000.0])  # of the search box; deliberately not the answer
SEARCH = dict(offset_tol=8.0, shift_tol=6.0, silence_dist=2500.0, silence_between_only=True)
AUDIBLE = 10000.0  # m, farthest a flight is looked for


def search_grid():
    grid = observer_grid(CENTER, 25000.0, 200.0, place=place)
    print(f"grid: {len(grid)} points, 50 x 50 km around ({CENTER[0] / 1000:+.0f}, {CENTER[1] / 1000:+.0f}) km")
    return grid


def search():
    """Blind search of a 50 x 50 km box with the two strongest detections (or the hand-measured pair)."""
    spec, events, _, tracks = load()
    e1, e2 = sorted(events[:2], key=lambda ev: ev.t_heard)
    grid = search_grid()
    keep = candidate_flights(tracks, CENTER)

    t_start = time.time()
    res = search_two_events(spec, keep, grid, e1, e2, spec.t[-1], audible_dist=AUDIBLE, workers=os.cpu_count(),
                            **SEARCH)
    print(f"search took {time.time() - t_start:.0f} s on {os.cpu_count()} cores: "
          + ", ".join(f"{k} {v:.0f} s" for k, v in res.seconds.items()))
    for name, n in res.funnel.items():
        print(f"  {name:>20}: {n}")
    if len(res.point) == 0:
        return

    locate(spec, keep, grid, e1, e2, res)


def locate(spec, keep, grid, e1, e2, res):
    """Print the best flight pairs of one two-event search, refine the top 5 and map the result."""
    ids = [t.flight_id for t in keep]
    top = res.best_per_pair()
    print("top flight pairs (200 m grid):")
    for r in top[:8]:
        p = grid[res.point[r]]
        print(f"  {ids[res.a[r]]:>10} -> {ids[res.b[r]]:<10} {res.score1[r]:5.2f} + {res.score2[r]:5.2f} = "
              f"{res.total[r]:5.2f}   at ({p[0] / 1000:+6.2f}, {p[1] / 1000:+6.2f}) km, "
              f"{np.hypot(*p[:2]) / 1000:5.2f} km from truth")

    refined = []
    for r in top[:5]:
        fa, fb = keep[res.a[r]], keep[res.b[r]]
        obs, s1, s2 = refine_pair(spec, fa, fb, grid[res.point[r], :2], e1, e2, place=place, shift_tol=6.0)
        tot = s1.score + s2.score
        k = int(np.argmax(tot))
        refined.append((tot[k], obs[k], fa, fb, obs, tot))
    refined.sort(key=lambda r: -r[0])
    best_tot, best, fa, fb, obs, tot = refined[0]
    lat, lon = frame.to_latlon(best)
    print(f"\nbest: {fa.flight_id} -> {fb.flight_id}, score {best_tot:.2f}, at {lat:.5f}, {lon:.5f}")
    print(f"distance from the answer: {np.hypot(*best[:2]):.0f} m")

    point_score = np.full(len(grid), np.nan)
    np.fmax.at(point_score, res.point, res.total)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    plots.score_map(axes[0], grid, point_score, spacing=200, highlight=[fa, fb], truth=np.zeros(3),
                    title="survivors of all filters (truth at 0, 0)")
    plots.score_map(axes[1], obs, tot, spacing=25, highlight=[fa, fb], truth=np.zeros(3), best=best,
                    title=f"refined: {fa.flight_id} -> {fb.flight_id}")
    plots.save(fig, OUT / "search_map.png")


def pairs(n_detect=10, n_events=6):
    """Blind search that also chooses which two passes to use.

    Of the `n_detect` strongest detections, those `implausible` for the search are dropped, and
    every pair of the `n_events` strongest left is searched. Event pairs are ranked by how much
    of both traced lines their best flights follow (see `search_event_pairs`), and the winner is
    refined as in `search`.
    """
    spec, events, names, tracks = load(n_detect)
    keep_events = []
    for name, ev in zip(names, events):
        why = implausible(ev, max_dist=AUDIBLE)
        if why:
            print(f"  {name} dropped: {why}")
        elif len(keep_events) < n_events:
            keep_events.append(name)
    use = [names.index(n) for n in keep_events]
    events, names = [events[k] for k in use], keep_events
    print(f"searching every pair of {', '.join(names)}")

    grid = search_grid()
    keep = candidate_flights(tracks, CENTER)
    t_start = time.time()
    found = search_event_pairs(spec, keep, grid, events, spec.t[-1], audible_dist=AUDIBLE, workers=os.cpu_count(),
                               **SEARCH)
    print(f"search of {len(found)} event pairs took {time.time() - t_start:.0f} s on {os.cpu_count()} cores")

    ids = [t.flight_id for t in keep]
    print("\nevent pairs, strongest evidence first. 'follows' = points of each traced line the best flights' "
          "curves pass within 1.5 Hz of;\nevidence = followed points above the median candidate's, both events:")
    for ep in found:
        res, e1, e2 = ep.result, events[ep.i], events[ep.j]
        label = f"{names[ep.i]:>3} -> {names[ep.j]:<3} {e2.t_heard - e1.t_heard:5.0f} s apart"
        if len(ep.rows) == 0:
            print(f"  {label}: no flight pair survives the filters")
            continue
        r = ep.rows[0]
        print(f"  {label}: evidence {ep.best:5.0f}  {ids[res.a[r]]:>10} -> {ids[res.b[r]]:<10} follows "
              f"{ep.followed[0, 0]:3.0f}/{len(e1.line[0]):<3d} {ep.followed[0, 1]:3.0f}/{len(e2.line[0]):<3d} "
              f"({len(ep.rows):3d} flight pairs)  {np.hypot(*grid[res.point[r], :2]) / 1000:5.1f} km from truth")

    win = found[0]
    print(f"\nchosen: {names[win.i]} -> {names[win.j]} (evidence {win.best:.0f} vs {found[1].best:.0f} for the next pair)")
    locate(spec, keep, grid, events[win.i], events[win.j], win.result)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    args = [a for a in sys.argv[1:] if a != "--hand"]
    HAND = "--hand" in sys.argv
    {"verify": verify, "search": search, "pairs": pairs}[args[0] if args else "verify"]()
