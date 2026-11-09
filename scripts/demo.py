"""Run the synthetic scenarios end to end and write figures, audio and a summary to out/.

    uv run python scripts/demo.py                # all scenarios
    uv run python scripts/demo.py two_passes     # just one
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from doppler_loc import plots, scenarios
from doppler_loc.measure import fit_flyby, straight_flyby, trace_ridge
from doppler_loc.physics import SPEED_OF_SOUND, closest_approach, doppler_factor
from doppler_loc.search import (
    observer_grid, refine_pair, refine_single, search_known_time, search_two_events,
)
from doppler_loc.spectro import compute_spectrogram
from doppler_loc.synth import write_wav

OUT = Path(__file__).resolve().parent.parent / "out"
C = SPEED_OF_SOUND


def measure(spec, mark):
    times, freqs, strength = trace_ridge(spec, *mark)
    return fit_flyby(times, freqs, strength), (times, freqs, strength)


def describe_truth(track, truth):
    ca = closest_approach(track, truth[None])
    return f"true CPA {ca.dist[0]:.0f} m at {ca.speed[0]:.1f} m/s"


def run_single(maneuvering: bool):
    t_start = time.time()
    sc = scenarios.single_pass(maneuvering=maneuvering)
    out = OUT / sc.name
    out.mkdir(parents=True, exist_ok=True)
    write_wav(out / "recording.wav", sc.audio, sc.sr)

    spec = compute_spectrogram(sc.audio, sc.sr)
    event, (tt, ff, ss) = measure(spec, sc.marks[0])
    print(f"\n== {sc.name}")
    print(f"measured: f0 {event.f0:.2f} Hz, {event.speed:.1f} m/s, CPA {event.dist:.0f} m, heard at {event.t_heard:.1f} s"
          f"  ({describe_truth(sc.true_tracks[0], sc.truth)}, f0 {sc.f0s[0]} Hz)")

    clock_error = 2.0  # the recording's timestamp is 2 s off and the solver doesn't know
    t0 = sc.t_rec + clock_error
    grid = observer_grid((0.0, 0.0), 8000.0, 200.0, sc.ground)
    res = search_known_time(spec, sc.adsb, grid, event, t0, clock_tol=4.0)
    scored = np.isfinite(res.score)
    print(f"coarse grid: {len(grid)} points, {scored.sum()} pass speed/distance/time filters")

    # Best peak, plus the best one more than 1 km from it (the mirror image, if there is one).
    order = np.argsort(-np.nan_to_num(res.score, nan=-np.inf))
    peaks = [order[0]]
    for i in order[1:]:
        if np.isfinite(res.score[i]) and np.linalg.norm(grid[i, :2] - grid[peaks[0], :2]) > 1000:
            peaks.append(i)
            break
    refined = []
    for i in peaks:
        obs, s = refine_single(spec, sc.adsb[0], grid[i, :2], event, t0, ground=sc.ground, clock_tol=4.0)
        j = int(np.argmax(s.score))
        refined.append((obs[j], s.score[j], s.f0[j], s.shift[j] * spec.hop, obs, s))
    refined.sort(key=lambda r: -r[1])
    best, best_score, best_f0, best_shift, fine_obs, fine = refined[0]
    err = np.linalg.norm(best[:2] - sc.truth[:2])
    print(f"best: ({best[0]:.0f}, {best[1]:.0f}) score {best_score:.2f}, error {err:.0f} m, "
          f"clock correction {best_shift:+.1f} s (true {-clock_error:+.1f})")
    if len(refined) > 1:
        other = refined[1]
        print(f"runner-up peak: ({other[0][0]:.0f}, {other[0][1]:.0f}) score {other[1]:.2f}"
              f"  -> ratio {other[1] / best_score:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plots.score_map(axes[0], grid, res.score, spacing=200, tracks=sc.adsb, truth=sc.truth, best=best,
                    title="coarse search (200 m)")
    plots.score_map(axes[1], fine_obs, fine.score, spacing=25, tracks=sc.adsb, truth=sc.truth, best=best,
                    title="refined around the best peak (25 m)")
    plots.save(fig, out / "map.png")

    t_range, f_range = sc.marks[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plots.spectrogram(axes[0], spec, t_range, f_range)
    axes[0].set_title("zoomed spectrogram")
    plots.spectrogram(axes[1], spec, t_range, f_range)
    axes[1].plot(tt, ff, ".", ms=3, color="cyan", label="traced line")
    axes[1].plot(tt, straight_flyby(tt, event.f0, event.speed, event.dist, event.t_heard), "--", color="white",
                 lw=1, label="straight-line fit")
    plots.model_curve(axes[1], spec, sc.adsb[0], best, t0 + best_shift, best_f0, t_range, color="lime", lw=1.2,
                      label="model at best candidate")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("measurement and best model")
    plots.save(fig, out / "spectrogram.png")

    if not maneuvering:
        compare_models(spec, sc, out / "model_comparison.png")
    print(f"took {time.time() - t_start:.0f} s; figures in {out}")
    return err


def compare_models(spec, sc, path):
    """The video's point: delaying the instantaneous curve is not the same as the retarded model."""
    track, truth, f0 = sc.true_tracks[0], sc.truth, sc.f0s[0]
    t = sc.t_rec + spec.t
    ca = closest_approach(track, truth[None])
    retarded = f0 * doppler_factor(track, truth, t)
    instant = f0 * doppler_factor(track, truth, t, retarded=False)
    delayed = f0 * doppler_factor(track, truth, t - ca.dist[0] / C, retarded=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    t_range, f_range = sc.marks[0]
    plots.spectrogram(ax, spec, t_range, f_range)
    ax.plot(spec.t, instant, color="tab:blue", lw=1.2, label="no delay")
    ax.plot(spec.t, delayed, color="tab:orange", lw=1.2, label="instantaneous, delayed by CPA travel time")
    ax.plot(spec.t, retarded, color="lime", lw=1.2, label="emission-time model")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("same flight, same listener, three models")
    plots.save(fig, path)


def run_two_passes():
    t_start = time.time()
    sc = scenarios.two_passes()
    out = OUT / sc.name
    out.mkdir(parents=True, exist_ok=True)
    write_wav(out / "recording.wav", sc.audio, sc.sr)
    spec = compute_spectrogram(sc.audio, sc.sr)

    print(f"\n== {sc.name}  ({len(sc.adsb)} flights in the area)")
    events = []
    for k, mark in enumerate(sc.marks):
        ev, _ = measure(spec, mark)
        events.append(ev)
        print(f"event {k + 1}: f0 {ev.f0:.2f} Hz, {ev.speed:.1f} m/s, CPA {ev.dist:.0f} m, heard at {ev.t_heard:.1f} s"
              f"  ({describe_truth(sc.true_tracks[k], sc.truth)}, f0 {sc.f0s[k]} Hz)")
    e1, e2 = events

    grid = observer_grid((0.0, 0.0), 25000.0, 200.0, sc.ground)
    res = search_two_events(spec, sc.adsb, grid, e1, e2, sc.duration)
    for name, n in res.funnel.items():
        print(f"  {name:>20}: {n}")

    ids = [f.flight_id for f in sc.adsb]
    top = res.best_per_pair()
    print("top flight pairs (coarse):")
    for r in top[:5]:
        print(f"  {ids[res.a[r]]} -> {ids[res.b[r]]}: {res.score1[r]:.2f} + {res.score2[r]:.2f} = {res.total[r]:.2f}")

    refined = []
    for r in top[:3]:
        fa, fb = sc.adsb[res.a[r]], sc.adsb[res.b[r]]
        obs, s1, s2 = refine_pair(spec, fa, fb, grid[res.point[r], :2], e1, e2, ground=sc.ground)
        tot = s1.score + s2.score
        j = int(np.argmax(tot))
        refined.append(dict(a=res.a[r], b=res.b[r], best=obs[j], total=tot[j], obs=obs, tot=tot,
                            f0=(s1.f0[j], s2.f0[j]),
                            t0=closest_approach(fa, obs[j][None]).t_heard[0] - e1.t_heard + s1.shift[j] * spec.hop))
    refined.sort(key=lambda d: -d["total"])
    win = refined[0]
    err = np.linalg.norm(win["best"][:2] - sc.truth[:2])
    print(f"best: {ids[win['a']]} -> {ids[win['b']]} at ({win['best'][0]:.0f}, {win['best'][1]:.0f}), "
          f"score {win['total']:.2f}, error {err:.0f} m")
    print(f"recording start: estimated {win['t0']:.1f}, true {sc.t_rec:.1f} ({win['t0'] - sc.t_rec:+.1f} s)")

    point_score = np.full(len(grid), np.nan)
    np.fmax.at(point_score, res.point, res.total)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plots.score_map(axes[0], grid, point_score, spacing=200, tracks=sc.adsb,
                    highlight=[sc.adsb[win["a"]], sc.adsb[win["b"]]], truth=sc.truth,
                    title="points surviving all filters, coloured by score")
    plots.score_map(axes[1], win["obs"], win["tot"], spacing=25, highlight=[sc.adsb[win["a"]], sc.adsb[win["b"]]],
                    truth=sc.truth, best=win["best"], title="refined, winning flight pair (25 m)")
    plots.save(fig, out / "map.png")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(res.score1, res.score2, s=3, color="0.75", label="all candidates")
    ax.scatter(res.score1[top], res.score2[top], s=14, color="tab:blue", label="best point per flight pair")
    r0 = top[0]
    ax.scatter([res.score1[r0]], [res.score2[r0]], s=60, facecolors="none", edgecolors="red", label="top pair")
    ax.set_xlabel("event 1 score")
    ax.set_ylabel("event 2 score")
    ax.legend(fontsize=8)
    plots.save(fig, out / "pair_scores.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for k, (ax, mark) in enumerate(zip(axes, sc.marks)):
        plots.spectrogram(ax, spec, *mark)
        track = sc.adsb[win["a"] if k == 0 else win["b"]]
        plots.model_curve(ax, spec, track, win["best"], win["t0"], win["f0"][k], mark[0], color="lime", lw=1.0,
                          label=f"model: {track.flight_id}")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"event {k + 1}")
    plots.save(fig, out / "spectrogram.png")
    print(f"took {time.time() - t_start:.0f} s; figures in {out}")
    return err


SCENARIOS = {
    "straight": lambda: run_single(maneuvering=False),
    "maneuvering": lambda: run_single(maneuvering=True),
    "two_passes": run_two_passes,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or SCENARIOS:
        SCENARIOS[name]()
