"""Searching with pairs of passes without a grid over the area: flights first.

`search.search_two_events` works out every flight's closest approach to every grid point and then
filters, so its time and memory grow with the area. But two passes heard `delta` seconds apart have
to come from two flights that were close to each other `delta` seconds apart, each no farther from
the microphone than its pass allows. So the flights alone, sampled every few seconds, say which
pairs can have made the two passes and roughly where (`_join_blocks`, `_partners_of`). That takes seconds for a whole
country, however fine the grid. Only those few places get a grid, with the same filters and scoring
as `search_two_events`. On a 200 km box the two give the same rows and scores, except for points
where a glitch in a flight's interpolated track reads several times its reported speed, which the
grid search lets through and this doesn't.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.ndimage import binary_dilation

from .dem import TerrainTiles
from .flight import FlightTrack
from .measure import Event, straight_flyby
from .physics import SPEED_OF_SOUND, closest_approach, doppler_factor
from .score import line_agreement, score_event
from .search import EventPair, PairResult, _map, _shift_frames, _within, _worker, explain_passes
from .spectro import Spectrogram

SAMPLE_S = 10.0  # how often flights are sampled to compare them with each other
QUIET_BIN_S, QUIET_CELL_M = 60.0, 5000.0  # the sample index behind the silence filter
OFF = 1 << 20  # keeps cell numbers positive inside a sort key
TILES_KEPT = 64  # terrain tiles a worker keeps between jobs (half a megabyte each, for 2,000 over a country)
EARTH_M = 6371000.0
POLISH_TOP = 3000  # flight pairs, by the lines they follow at their grid point, moved to where they fit best


class Legs:
    """Flights as flat arrays, each made into a `FlightTrack` only when it is needed: over a whole
    country there are too many to keep them all as splines."""

    def __init__(self, names, metas, offsets, t, pos, vel):
        self.names, self.metas = names, metas
        self.offsets, self.t, self.pos, self.vel = offsets, t, pos, vel

    @classmethod
    def from_tracks(cls, tracks):
        offsets = np.r_[0, np.cumsum([len(tr.t) for tr in tracks])]
        return cls([tr.flight_id for tr in tracks], [getattr(tr, "meta", {}) or {} for tr in tracks], offsets,
                   np.concatenate([tr.t for tr in tracks]), np.concatenate([tr.position(tr.t) for tr in tracks]),
                   np.concatenate([tr.velocity(tr.t) for tr in tracks]))

    @classmethod
    def from_parts(cls, parts):
        """From (name, meta, t, pos, vel) of each flight."""
        names, metas, ts, poss, vels = (list(c) for c in zip(*parts)) if parts else ([], [], [], [], [])
        offsets = np.r_[0, np.cumsum([len(t) for t in ts])]
        return cls(names, metas, offsets, np.concatenate(ts) if ts else np.zeros(0), np.concatenate(poss) if poss else np.zeros((0, 3)),
                   np.concatenate(vels) if vels else np.zeros((0, 3)))

    def __len__(self):
        return len(self.names)

    def track(self, i) -> FlightTrack:
        s = slice(self.offsets[i], self.offsets[i + 1])
        meta = self.metas[i]
        tr = FlightTrack(self.names[i], self.t[s], self.pos[s], vel=self.vel[s], typecode=meta.get("t") or "")
        tr.meta = meta
        return tr

    def samples(self, dt=SAMPLE_S) -> np.ndarray:
        """Every flight every `dt` s, as rows (flight, t, x, y, z, speed): straight lines between
        reports are plenty for comparing flights, whose margins allow for it."""
        speed = np.linalg.norm(self.vel, axis=1)
        out = []
        for i in range(len(self)):
            a, b = self.offsets[i], self.offsets[i + 1]
            t = self.t[a:b]
            ts = np.arange(t[0], t[-1] + dt / 2, dt)
            out.append(np.c_[np.full(ts.size, i), ts, *(np.interp(ts, t, self.pos[a:b, k]) for k in range(3)),
                             np.interp(ts, t, speed[a:b])])
        return np.concatenate(out) if out else np.zeros((0, 6))


class QuietIndex:
    """Which flights are near a place around a time: for the silence filter, and for checking a spot."""

    def __init__(self, samples):
        key = self._key(np.floor(samples[:, 1] / QUIET_BIN_S).astype(np.int64),
                        np.floor(samples[:, 2] / QUIET_CELL_M).astype(np.int64),
                        np.floor(samples[:, 3] / QUIET_CELL_M).astype(np.int64))
        order = np.argsort(key)
        self.key, self.leg = key[order], samples[order, 0].astype(np.int32)

    @staticmethod
    def _key(tb, cx, cy):
        return (tb << 42) + ((cx + OFF) << 21) + (cy + OFF)

    def near(self, x_lo, x_hi, y_lo, y_hi, t_lo, t_hi) -> np.ndarray:
        tb = np.arange(np.floor(t_lo / QUIET_BIN_S), np.floor(t_hi / QUIET_BIN_S) + 1).astype(np.int64)
        cx = np.arange(np.floor(x_lo / QUIET_CELL_M), np.floor(x_hi / QUIET_CELL_M) + 1).astype(np.int64)
        cy = np.arange(np.floor(y_lo / QUIET_CELL_M), np.floor(y_hi / QUIET_CELL_M) + 1).astype(np.int64)
        q = self._key(*np.meshgrid(tb, cx, cy, indexing="ij")).ravel()
        lo, hi = np.searchsorted(self.key, q, "left"), np.searchsorted(self.key, q, "right")
        hit = np.flatnonzero(hi > lo)
        if not hit.size:
            return np.zeros(0, np.int32)
        return np.unique(np.concatenate([self.leg[lo[k]:hi[k]] for k in hit]))


def _usable(samples, ev: Event):
    """Samples of flights fast or slow enough for `ev`, with room for a reading between samples."""
    lo, hi = ev.speed_range
    return samples[(samples[:, 5] >= lo * 0.9 - 5.0) & (samples[:, 5] <= hi * 1.1 + 5.0)], hi * 1.1 + 5.0


def _join_blocks(a, reach_a, b, reach_b, t_lo, t_hi, chunk=1 << 13):
    """Pairs of samples of `a` and `b`: b within [t_lo, t_hi] s after a, and close enough that a point
    can lie within reach_a of a and reach_b of b. Yields them a block at a time, as two arrays of rows
    (flight, x, y): over a country they can run to hundreds of millions, and the arrays to work each
    block out take several times the room of what is kept, so `chunk` samples of `a` at a time."""
    if not len(a) or not len(b):
        return
    reach = reach_a + reach_b
    cell, span = max(reach, 1000.0), max(t_hi - t_lo, 1.0)
    key = ((np.floor(b[:, 1] / span).astype(np.int64) << 42)
           + ((np.floor(b[:, 2] / cell).astype(np.int64) + OFF) << 21) + (np.floor(b[:, 3] / cell).astype(np.int64) + OFF))
    order = np.argsort(key)
    key, b_sorted = key[order], b[order]
    tb = np.floor((a[:, 1] + t_lo) / span).astype(np.int64)
    ax, ay = np.floor(a[:, 2] / cell).astype(np.int64), np.floor(a[:, 3] / cell).astype(np.int64)
    for s in range(0, len(a), chunk):
        a_s = a[s:s + chunk]
        for bin_ in (0, 1):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = ((tb[s:s + chunk] + bin_) << 42) + ((ax[s:s + chunk] + dx + OFF) << 21) + (ay[s:s + chunk] + dy + OFF)
                    lo = np.searchsorted(key, q, "left")
                    n = np.searchsorted(key, q, "right") - lo
                    m = np.flatnonzero(n)
                    if not m.size:
                        continue
                    ia = np.repeat(m, n[m])
                    ib = np.repeat(lo[m], n[m]) + (np.arange(n[m].sum()) - np.repeat(np.r_[0, np.cumsum(n[m])[:-1]], n[m]))
                    gap = b_sorted[ib, 1] - a_s[ia, 1]
                    d = b_sorted[ib, 2:5] - a_s[ia, 2:5]
                    ok = (gap >= t_lo) & (gap <= t_hi) & (b_sorted[ib, 0] != a_s[ia, 0]) & ((d**2).sum(1) <= reach**2)
                    if ok.any():
                        yield a_s[ia[ok]][:, [0, 2, 3]].astype(np.float32), b_sorted[ib[ok]][:, [0, 2, 3]].astype(np.float32)


def _join(a, reach_a, b, reach_b, t_lo, t_hi, chunk=1 << 13):
    """All of `_join_blocks` at once."""
    found = list(_join_blocks(a, reach_a, b, reach_b, t_lo, t_hi, chunk))
    if not found:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return np.concatenate([f[0] for f in found]), np.concatenate([f[1] for f in found])


def _rects(ma, mb, ra, rb, half, spacing):
    """For each pair of samples that matched, the lattice cells (rectangle ix0, ix1, iy0, iy1) where a
    point can lie within ra of a and rb of b, if any: (flight a, flight b, rectangles), all int32."""
    x0 = np.maximum(ma[:, 1] - ra, mb[:, 1] - rb)
    x1 = np.minimum(ma[:, 1] + ra, mb[:, 1] + rb)
    y0 = np.maximum(ma[:, 2] - ra, mb[:, 2] - rb)
    y1 = np.minimum(ma[:, 2] + ra, mb[:, 2] + rb)
    keep = (x0 <= x1) & (y0 <= y1) & (x1 >= -half) & (x0 <= half) & (y1 >= -half) & (y0 <= half)
    x0, x1, y0, y1 = (np.clip(v[keep], -half, half) for v in (x0, x1, y0, y1))
    rects = np.stack([np.ceil((x0 + half) / spacing), np.floor((x1 + half) / spacing),
                      np.ceil((y0 + half) / spacing), np.floor((y1 + half) / spacing)], 1).astype(np.int32)
    return ma[keep, 0].astype(np.int32), mb[keep, 0].astype(np.int32), rects


def _by_flight_pair(fa, fb, rects):
    """{flight a: [(flight b, its rectangles (n, 4))]}, in flight order."""
    order = np.lexsort((fb, fa))
    fa, fb, rects = fa[order], fb[order], rects[order]
    out = {}
    if len(fa):
        edges = np.r_[0, np.flatnonzero((np.diff(fa) != 0) | (np.diff(fb) != 0)) + 1, len(fa)]
        for s, e in zip(edges[:-1], edges[1:]):
            out.setdefault(int(fa[s]), []).append((int(fb[s]), rects[s:e]))
    return out


def _partners(ma, mb, ra, rb, half, spacing):
    """{flight a: [(flight b, rectangles (n, 4) of lattice cells ix0, ix1, iy0, iy1)]}: where a point
    can lie within ra of a and rb of b, for each pair of samples that matched."""
    return _by_flight_pair(*_rects(ma, mb, ra, rb, half, spacing))


def _partners_of(a, ra, b, rb, t_lo, t_hi, half, spacing):
    """`_partners` of what `_join` matches, a block at a time: only the rectangles are kept, a third
    of the room the matched samples and the arrays to get from them to the rectangles would take."""
    parts = [_rects(ma, mb, ra, rb, half, spacing) for ma, mb in _join_blocks(a, ra, b, rb, t_lo, t_hi)]
    if not parts:
        return {}
    fa, fb, rects = (np.concatenate([p[k] for p in parts]) for k in range(3))
    del parts
    return _by_flight_pair(fa, fb, rects)


def _cells(rects):
    """The lattice cells (ix, iy) in any of `rects`, each once."""
    x0, y0 = rects[:, 0].min(), rects[:, 2].min()
    mask = np.zeros((rects[:, 1].max() - x0 + 2, rects[:, 3].max() - y0 + 2), bool)
    for ix0, ix1, iy0, iy1 in rects:
        if ix1 >= ix0 and iy1 >= iy0:
            mask[ix0 - x0:ix1 - x0 + 1, iy0 - y0:iy1 - y0 + 1] = True
    ix, iy = np.nonzero(mask)
    return ix + x0, iy + y0


def _lattice_keys(rects, half, spacing, area=None):
    """The lattice cells in any of `rects`, as sortable keys, without those outside `area` (an `AreaMask`)."""
    ix, iy = _cells(rects)
    if area is not None:
        inside = area.contains(ix * spacing - half, iy * spacing - half)
        ix, iy = ix[inside], iy[inside]
    return (ix.astype(np.int64) << 32) + iy


def _tile_cells(rects, spacing):
    corners = np.concatenate([rects[:, [0, 2]], rects[:, [0, 3]], rects[:, [1, 2]], rects[:, [1, 3]]])
    return np.unique(np.floor(corners * spacing / 4000.0).astype(np.int64), axis=0)


def _prefetch_terrain(site, pairs, half, spacing, progress=None):
    """Download every terrain tile the workers will read, from here, with one set of connections."""
    if not pairs:
        return
    # The tiles are tens of kilometres across: a point every few kilometres finds them all. A batch of
    # rectangles at a time, since all at once take many times their own room.
    found, batch, size = [], [], 0
    for v in pairs.values():
        for _, r in v:
            batch.append(r)
            size += len(r)
            if size >= 1 << 20:
                found.append(_tile_cells(np.concatenate(batch), spacing))
                batch, size = [], 0
    if batch:
        found.append(_tile_cells(np.concatenate(batch), spacing))
    cells = np.unique(np.concatenate(found), axis=0)
    lat, lon = site.frame.ground_latlon(cells[:, 0] * 4000.0 - half, cells[:, 1] * 4000.0 - half)
    n = 2**site.dem.zoom
    lr = np.radians(np.clip(lat, -85.05, 85.05))
    x = ((lon + 180.0) % 360.0 / 360.0 * n).astype(int)
    y = ((1 - np.log(np.tan(lr) + 1 / np.cos(lr)) / np.pi) / 2 * n).astype(int)
    site.dem.progress = progress
    site.dem.prefetch(set(zip(x.tolist(), y.tolist())))
    site.dem.progress = None


def _near_any(t, times, within):
    """Whether each of `t` lies within `within` of any of the sorted `times`."""
    if not len(times):
        return np.zeros(len(t), bool)
    k = np.searchsorted(times, t)
    return (np.abs(times[np.clip(k, 0, len(times) - 1)] - t) <= within) | (np.abs(times[np.clip(k - 1, 0, len(times) - 1)] - t) <= within)


def _pair_job(job):
    """The filters and scores of `search._score_job`, for one first flight and its possible partners,
    at the lattice points where they can be heard together."""
    a, partners = job
    w = _worker
    legs, e1, e2, half, spacing, c = w["legs"], w["e1"], w["e2"], w["half"], w["spacing"], w["c"]
    dist, start = w["audible"], w["start"]
    keys_of = [_lattice_keys(rects, half, spacing, w["area"]) for _, rects in partners]
    keys = np.unique(np.concatenate(keys_of)) if keys_of else np.zeros(0, np.int64)
    out = dict(best=[], rows=[], checked=int(keys.size), measured=0, quiet=0)
    if not keys.size:
        return out
    site = w["site"]
    xyz = site.place((keys >> 32) * spacing - half, (keys & 0xFFFFFFFF) * spacing - half)
    if len(site.dem._tiles) > TILES_KEPT:
        site.dem._tiles.clear()
    fa = legs.track(a)
    ca = closest_approach(fa, xyz, c, max_dist=dist)
    ok_a = ca.valid & _within(ca.dist, e1.dist_range) & _within(ca.speed, e1.speed_range)
    if start is not None:
        ok_a &= np.abs(ca.t_heard - (start[0] + e1.t_heard)) < start[1] + w["offset_tol"]
    rows, tracks, delta = [], {}, e2.t_heard - e1.t_heard
    guard, quiet_dist = w["guard"], w["quiet_dist"]
    for (b, _), k in zip(partners, keys_of):
        sel = np.searchsorted(keys, k)
        sel = sel[ok_a[sel]]
        if not sel.size:
            continue
        fb = legs.track(b)
        cb = closest_approach(fb, xyz[sel], c, max_dist=dist)
        ok = cb.valid & _within(cb.dist, e2.dist_range) & _within(cb.speed, e2.speed_range)
        ok &= np.abs(cb.t_heard - ca.t_heard[sel] - delta) < w["offset_tol"]
        if not ok.any():
            continue
        idx, th_a, th_b = sel[ok], ca.t_heard[sel[ok]], cb.t_heard[ok]
        out["measured"] += idx.size
        # Silence: nothing else heard between the two, outside `guard` of each (as `search._HeardTimes`),
        # unless a pass was traced at that moment: then something was heard, as that flight would be.
        lo, hi = th_a + guard, th_b - guard
        recording_start = th_a - e1.t_heard
        pad = quiet_dist + QUIET_CELL_M
        quiet = np.ones(idx.size, bool)
        for o in w["index"].near(xyz[idx, 0].min() - pad, xyz[idx, 0].max() + pad, xyz[idx, 1].min() - pad, xyz[idx, 1].max() + pad,
                                 lo.min() - 120.0, hi.max() + 60.0):
            o = int(o)
            if o not in tracks:
                tracks[o] = legs.track(o)
            co = closest_approach(tracks[o], xyz[idx], c, max_dist=quiet_dist)
            quiet &= ~(co.valid & (co.t_heard > lo) & (co.t_heard < hi) & ~_near_any(co.t_heard - recording_start, w["heard"], guard))
            if not quiet.any():
                break
        if not quiet.any():
            continue
        out["quiet"] += int(quiet.sum())
        rows.append((b, fb, idx[quiet], th_a[quiet], th_b[quiet]))
    if not rows:
        return out
    # Event 1's score at a point is the same whichever flight made event 2: score it once per point.
    spec, shifts, window = w["spec"], w["shifts"], w["window"]
    pts, first = np.unique(np.concatenate([g[2] for g in rows]), return_index=True)
    th_first = np.concatenate([g[3] for g in rows])[first]
    s1 = score_event(spec, fa, xyz[pts], th_first - e1.t_heard, th_first, e1.f0_grid(), shifts, window=window,
                     harmonics=e1.harmonics, c=c)
    for b, fb, idx, th_a, th_b in rows:
        r = np.searchsorted(pts, idx)
        s2 = score_event(spec, fb, xyz[idx], th_a - e1.t_heard, th_b, e2.f0_grid(), s1.shift[r, None], window=window,
                         harmonics=e2.harmonics, c=c)
        t0 = th_a - e1.t_heard + s1.shift[r] * spec.hop
        total = s1.score[r] + s2.score
        if w["all_rows"]:
            out["rows"].append((np.full(idx.size, a), np.full(idx.size, b), keys[idx], s1.score[r], s2.score, s1.f0[r], s2.f0, t0))
        k = int(np.argmax(total))
        obs = xyz[idx[k]][None]
        followed = [line_agreement(fl, obs, t0[k:k + 1], [f0], ev, tol_hz=w["tol_hz"], c=c)[0] * len(ev.line[0])
                    for fl, f0, ev in ((fa, s1.f0[r][k], e1), (fb, s2.f0[k], e2))]
        out["best"].append(dict(a=a, b=b, key=int(keys[idx[k]]), xyz=xyz[idx[k]], score1=float(s1.score[r][k]), score2=float(s2.score[k]),
                                f0_1=float(s1.f0[r][k]), f0_2=float(s2.f0[k]), t0=float(t0[k]), followed=followed))
    return out


def search_two_events_wide(spec: Spectrogram, legs: Legs, samples, index, e1: Event, e2: Event, site, *, half, spacing, start=None,
                           offset_tol=8.0, shift_tol=6.0, silence_dist=2500.0, guard=20.0, audible_dist=10000.0, window=60.0,
                           tol_hz=1.5, c=SPEED_OF_SOUND, workers=1, sample_dt=SAMPLE_S, all_rows=False, progress=None, heard=(), area=None):
    """As `search.search_two_events` with `silence_between_only`, on a lattice over [-half, half]
    placed by `site` (at the terrain detail it has), flights first. `start`: (absolute recording
    start, uncertainty in s), if known. Returns (PairResult, one row per flight pair, its best
    point, unless `all_rows`; `point` is a lattice key; (n, 2) points of the two lines followed at
    the best points; their xyz (n, 3)), the best rows' columns being in the order of those two.
    `heard`: when other passes were traced, in s from the recording's start, which a flight then may explain."""
    clock = [time.perf_counter()]
    delta = e2.t_heard - e1.t_heard
    sa, va = _usable(samples, e1)
    sb, vb = _usable(samples, e2)
    d1, d2 = e1.dist_range, e2.dist_range
    d1_hi, d2_hi = min(d1[1], audible_dist), min(d2[1], audible_dist)
    if start is not None:  # when the first flight can have been at its closest
        when = start[0] + e1.t_heard
        sa = sa[(sa[:, 1] > when - start[1] - offset_tol - d1_hi / c - sample_dt) & (sa[:, 1] < when + start[1] + offset_tol + sample_dt)]
    # A sample can be half a step from the moment that counts, a lattice point up to a spacing from
    # where it was asked for, and, far from the frame's centre, tilted terrain moves it a little more.
    tilt = half * 1.5 / EARTH_M * 3000.0
    ra = d1_hi + va * sample_dt / 2 + spacing + 300.0 + tilt
    rb = d2_hi + vb * sample_dt / 2 + spacing + 300.0 + tilt
    t_lo = delta - offset_tol + (d1[0] - d2_hi) / c - sample_dt
    t_hi = delta + offset_tol + (d1_hi - d2[0]) / c + sample_dt
    pairs = _partners_of(sa, ra, sb, rb, t_lo, t_hi, half, spacing)
    del sa, sb
    joined = sum(len(v) for v in pairs.values())
    clock.append(time.perf_counter())
    _prefetch_terrain(site, pairs, half, spacing, progress)
    clock.append(time.perf_counter())
    light = Spectrogram(spec.t, spec.f, np.empty((0, 0), dtype=np.float32), spec.score)
    state = dict(spec=light, legs=legs, index=index, e1=e1, e2=e2, half=half, spacing=spacing, site=site, c=c, audible=audible_dist, start=start,
                 offset_tol=offset_tol, guard=guard, quiet_dist=silence_dist, shifts=_shift_frames(spec, shift_tol), window=window, tol_hz=tol_hz,
                 all_rows=all_rows, heard=np.sort(np.asarray(heard, dtype=float)), area=area)
    # Biggest first, so no worker is left with a large one at the end.
    jobs = sorted(pairs.items(), key=lambda kv: -sum(len(r) for _, r in kv[1]))
    res = _map(_pair_job, jobs, workers, state)
    clock.append(time.perf_counter())
    best = [b for x in res for b in x["best"]]
    funnel = {"grid points": int((2 * half / spacing + 1) ** 2), "flight pairs near each other": joined, "points checked": sum(x["checked"] for x in res),
              "+ second flight, at the right gap": sum(x["measured"] for x in res), "+ silence": sum(x["quiet"] for x in res), "flight pairs left": len(best)}
    seconds = dict(zip(("flights compared", "terrain", "points scored"), np.diff(clock)))
    followed = np.array([b["followed"] for b in best], float).reshape(-1, 2)
    xyz = np.array([b["xyz"] for b in best], float).reshape(-1, 3)
    if all_rows:
        rows = [r for x in res for r in x["rows"]]
        cols = [np.concatenate([r[k] for r in rows]) if rows else np.zeros(0) for k in range(8)]
        result = PairResult(point=cols[2].astype(np.int64), a=cols[0].astype(np.int64), b=cols[1].astype(np.int64), score1=cols[3], score2=cols[4],
                            f0_1=cols[5], f0_2=cols[6], t0=cols[7], funnel=funnel, seconds=seconds)
    else:
        col = lambda k, dtype=float: np.array([b[k] for b in best], dtype)  # noqa: E731
        result = PairResult(point=col("key", np.int64), a=col("a", np.int64), b=col("b", np.int64), score1=col("score1"), score2=col("score2"),
                            f0_1=col("f0_1"), f0_2=col("f0_2"), t0=col("t0"), funnel=funnel, seconds=seconds)
    return result, followed, xyz


def _polish_job(job):
    """Move a flight pair to where its two curves follow their traced lines best, near its grid
    point, and see how much of every pass can be explained from there at that start: a wrong pair
    can match two lines somewhere by chance, but rarely all of them. Returns (share of each event's
    line, best point, start)."""
    a, b, xyz, t0, i, j = job
    w = _worker
    legs, events, f0_tols, reach, step, span = w["legs"], w["events"], w["f0_tols"], w["audible"], w["step"], w["span"]
    ax = np.arange(-span, span + step / 2, step)
    x, y = (v.ravel() for v in np.meshgrid(xyz[0] + ax, xyz[1] + ax))
    site = w["site"]
    pts = site.place(x, y)
    if len(site.dem._tiles) > TILES_KEPT:
        site.dem._tiles.clear()
    fa, fb = legs.track(a), legs.track(b)
    shifts = (-4.0, -2.0, 0.0, 2.0, 4.0)
    best = (-1.0, xyz, t0)
    for p in pts:
        ea = explain_passes([fa], p, t0, [events[i]], max_dist=reach, f0_tol=f0_tols[i], shifts_s=shifts)[0]
        if ea.flight < 0:
            continue
        eb = explain_passes([fb], p, t0 + ea.shift, [events[j]], max_dist=reach, f0_tol=f0_tols[j], shifts_s=(-1.0, 0.0, 1.0))[0]
        if ea.share + eb.share > best[0]:
            best = (ea.share + eb.share, p, t0 + ea.shift)
    _, p, start = best
    shares = np.zeros(len(events))
    x_, y_ = p[:2]
    for n, ev in enumerate(events):
        if n == i:
            flights = [fa]
        elif n == j:
            flights = [fb]
        else:
            when = start + ev.t_heard
            flights = [legs.track(int(o)) for o in w["index"].near(x_ - reach, x_ + reach, y_ - reach, y_ + reach, when - 240.0, when + 240.0)]
        if flights:
            shares[n] = explain_passes(flights, p, start, [ev], max_dist=reach, f0_tol=f0_tols[n])[0].share
    return shares, p, start


def _settle_job(job):
    """Move a candidate to where all the passes it explains fit best together, each pass keeping the
    flight found for it. A pass close to the microphone pins the place to a hundred metres or so,
    so where the pair of passes it was found from fit best can lose others that fit a little way off."""
    xyz, t0, flights = job  # flights: [(event, leg)] of each pass explained
    w = _worker
    legs, events, f0_tols, step, span = w["legs"], w["events"], w["f0_tols"], w["step"], w["span"]
    ax = np.arange(-span, span + step / 2, step)
    x, y = (v.ravel() for v in np.meshgrid(xyz[0] + ax, xyz[1] + ax))
    pts = w["site"].place(x, y)
    if len(w["site"].dem._tiles) > TILES_KEPT:
        w["site"].dem._tiles.clear()
    each = np.zeros((len(pts), len(events)))
    starts = np.full(len(pts), t0)
    for rank, (e, leg) in enumerate(flights):
        ev = events[e]
        each[:, e], st, _ = follow_line(legs.track(leg), pts, starts, ev, _line_sample(ev), jitter=(-2.0, 0.0, 2.0) if rank else (-4.0, -2.0, 0.0, 2.0, 4.0),
                                        f0_tol=f0_tols[e])
        if not rank:  # the first flight, the anchor's, sets the start at each point
            starts = st
    k = int(np.argmax(each.sum(1)))
    return each[k], pts[k], float(starts[k])


SETTLE_TOP = 100  # of a pair of passes' best candidates, moved to where all their passes fit


def search_pairs_wide(spec, legs: Legs, events, site, *, half, spacing, workers=1, report=None, f0_tols=None, top=POLISH_TOP, pairs_of=None, turning=None, anchored_only=False, **kw):
    """`search.search_event_pairs` flights first: every pair of `events`. Flight pairs are ranked by
    the same evidence, then the `top` of them, polished (`_polish_job`), by how much of every pass
    they explain, each pass counting the same. Returns (EventPairs, best first, with `point`
    numbering the returned grid (N, 3) and one row per flight pair; the grid; (samples, index)).
    `pairs_of`: the events to make pairs of, when not all. `turning` (bool per event) marks passes
    whose straight-pass numbers can't be trusted: a pair of one of those and a straight one goes
    through `search_anchored_wide`, the straight one first. `anchored_only`: search only those pairs,
    for a search that has covered the rest its own way."""
    f0_tols = [0.15] * len(events) if f0_tols is None else f0_tols
    turning = np.zeros(len(events), bool) if turning is None else np.asarray(turning, bool)
    samples = legs.samples()
    index = QuietIndex(samples)
    among = np.arange(len(events)) if pairs_of is None else np.asarray(pairs_of)
    order = among[np.argsort([events[k].t_heard for k in among], kind="stable")]
    todo = [(i, j) for n, i in enumerate(order) for j in order[n + 1:] if not anchored_only or bool(turning[i]) != bool(turning[j])]
    out, grids, base = [], [], 0
    for n, (i, j) in enumerate(todo):
        if report:
            report(n, len(todo))
        anchored = bool(turning[i]) != bool(turning[j])
        if anchored:
            if turning[i]:
                i, j = j, i
            res, followed, xyz = search_anchored_wide(legs, samples, events[i], events[j], site, half=half, spacing=spacing, workers=workers,
                                                      f0_tol_a=f0_tols[i], f0_tol_b=f0_tols[j], **kw)
        else:
            res, followed, xyz = search_two_events_wide(spec, legs, samples, index, events[i], events[j], site, half=half, spacing=spacing, workers=workers,
                                                      heard=[ev.t_heard for ev in events], **kw)
        res.point = base + np.arange(len(xyz))
        base += len(xyz)
        evidence = (followed - np.median(followed, axis=0)).sum(1) if len(followed) else np.zeros(0)
        rank = np.argsort(-evidence, kind="stable")
        chosen = rank[:top]
        if len(chosen):
            # Finer than the grid, and as far as a grid point can be from the best place.
            state = dict(legs=legs, index=index, events=events, f0_tols=f0_tols, audible=kw.get("audible_dist", 10000.0), site=site,
                         span=spacing, step=spacing / 5.0)
            polished = _map(_polish_job, [(int(res.a[k]), int(res.b[k]), xyz[k], float(res.t0[k]), i, j) for k in chosen], workers, state)
            shares = np.array([x[0] for x in polished])
            xyz = xyz.copy()
            xyz[chosen] = np.array([x[1] for x in polished])
            res.t0 = res.t0.copy()
            res.t0[chosen] = [x[2] for x in polished]
            if anchored:
                # A second polish for the best: the same flights, but every pass counting.
                best = np.argsort(-shares.sum(1), kind="stable")[:SETTLE_TOP]
                jobs = []
                for m in best:
                    k, p, st = chosen[m], polished[m][1], polished[m][2]
                    # Which flight explains each other pass there (found again: `_polish_job` returns shares only).
                    flights = [(i, int(res.a[k])), (j, int(res.b[k]))]
                    reach = kw.get("audible_dist", 10000.0)
                    for e, ev in enumerate(events):
                        if e in (i, j):
                            continue
                        when = st + ev.t_heard
                        near = [int(o) for o in index.near(p[0] - reach, p[0] + reach, p[1] - reach, p[1] + reach, when - 240.0, when + 240.0)]
                        tracks = [legs.track(o) for o in near]
                        if tracks:
                            found = explain_passes(tracks, p, st, [ev], max_dist=reach, f0_tol=f0_tols[e])[0]
                            if found.flight >= 0:
                                flights.append((e, near[found.flight]))
                    jobs.append((p, st, flights))
                for m, (each, p, st) in zip(best, _map(_settle_job, jobs, workers, state)):
                    if each.sum() > shares[m].sum():
                        shares[m], xyz[chosen[m]], res.t0[chosen[m]] = each, p, st
            followed = followed.copy()
            followed[chosen] = np.c_[shares[:, i] * len(events[i].line[0]), shares[:, j] * len(events[j].line[0])]
            total = 100.0 * shares.sum(1)  # points of a hundred, each pass the same
            order = np.argsort(-total, kind="stable")
            rank = np.r_[chosen[order], rank[top:]]
            evidence = np.r_[total[order], np.full(len(rank) - len(chosen), -np.inf)]
            evidence_of = np.full(len(res.a), -np.inf)
            evidence_of[rank] = evidence
            evidence = evidence_of
        grids.append(xyz)
        out.append(EventPair(int(i), int(j), res, rank, followed[rank], evidence[rank], anchored=anchored))
    out.sort(key=lambda ep: -ep.best)
    return out, (np.concatenate(grids) if grids else np.zeros((0, 3))), (samples, index)


# --- pairs with a turning pass: one straight pass places the flight, the other is judged by its line ---


FOLLOW_LINE_POINTS = 32  # points of a traced line compared with a flight's curve while searching
FOLLOW_JITTER_S = (-4.0, -2.0, 0.0, 2.0, 4.0)  # around the start the straight pass implies
FOLLOW_MIN_SHARE = 0.4  # of a line, at a lattice point: the peak can lie half a step away
FOLLOW_SPEEDS = (20.0, 200.0)  # m/s: what a propeller aircraft can fly, whatever the pass's fit says
FOLLOW_SAMPLE_S = 30.0  # the follower's flights are compared this often: it is heard for longer than the anchor


def _thinned(samples, dt, every):
    """Every flight's samples `every` s apart instead of `dt`, counted from each flight's own first."""
    if not len(samples):
        return samples
    _, first, which = np.unique(samples[:, 0], return_index=True, return_inverse=True)
    steps = np.rint((samples[:, 1] - samples[first[which], 1]) / dt).astype(np.int64)
    return samples[steps % int(round(every / dt)) == 0]


def _line_sample(ev: Event, n=FOLLOW_LINE_POINTS):
    times, freqs = (np.asarray(x, dtype=float) for x in ev.line)
    pick = np.unique(np.linspace(0, len(times) - 1, min(len(times), n)).round().astype(int))
    return times[pick], freqs[pick]


class FitRegion:
    """Which (speed, distance at closest approach) can draw a straight pass's traced line at all.

    The curve of a straight flight depends only on those two, so this is worked out once per pass
    and read per point, where working out each point's curve would take hours over a country: the
    ranges on a fit (+-40% speed, +-70% distance) are ~16 times wider than what the line allows."""

    def __init__(self, ev: Event, line, *, jitter=FOLLOW_JITTER_S, tol_hz=1.5, min_share=FOLLOW_MIN_SHARE, n_speed=48, n_dist=64, c=SPEED_OF_SOUND):
        times, freqs = line
        self.v0, self.v1 = ev.speed_range
        self.d0, self.d1 = np.log(max(ev.dist_range[0], 1.0)), np.log(max(ev.dist_range[1], 2.0))
        self.n_speed, self.n_dist = n_speed, n_dist
        v = np.linspace(self.v0, self.v1, n_speed)[:, None, None, None]
        d = np.exp(np.linspace(self.d0, self.d1, n_dist))[None, :, None, None]
        jit = np.asarray(jitter, dtype=float)[None, None, :, None]
        g = straight_flyby(times[None, None, None, :] + jit, 1.0, v, d, ev.t_heard, c)
        f0 = np.median(freqs / g, axis=3, keepdims=True)
        share = np.mean(np.abs(f0 * g - freqs) < tol_hz, axis=3).max(2)
        self.table = binary_dilation(share >= min_share, iterations=1)  # a point can fall between two cells

    def allows(self, speed, dist) -> np.ndarray:
        i = np.clip(np.rint((speed - self.v0) / (self.v1 - self.v0) * (self.n_speed - 1)).astype(np.int64), 0, self.n_speed - 1)
        j = np.clip(np.rint((np.log(np.maximum(dist, 1.0)) - self.d0) / (self.d1 - self.d0) * (self.n_dist - 1)).astype(np.int64), 0, self.n_dist - 1)
        return self.table[i, j]


def follow_line(track: FlightTrack, xyz, starts, ev: Event, line, *, jitter=FOLLOW_JITTER_S, tol_hz=1.5, f0_tol=0.15, change_ratio=0.5,
                c=SPEED_OF_SOUND, chunk=512):
    """How much of `ev`'s traced line `track`'s curve follows from each of `xyz` (n, 3) with the
    recording starting at `starts` (n,) plus any of `jitter`, the emitted pitch fitted as
    `search.explain_passes` fits it. Needs no straight-pass numbers: speed, distance and the moment
    of closest approach all go wrong for a plane that turns, while the curve doesn't. Returns (share,
    start, f0), each (n,)."""
    times, freqs = line
    jit = np.asarray(jitter, dtype=float)
    n = len(xyz)
    line_change = np.log(freqs.max() / freqs.min())
    share, start, f0_out = np.zeros(n), np.array(starts, dtype=float), np.full(n, np.nan)
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        st = starts[sl, None] + jit  # (m, J)
        g = doppler_factor(track, xyz[sl, None, None, :], st[:, :, None] + times, c)  # (m, J, L)
        fin = np.isfinite(g)
        gm = np.where(fin, g, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows: the curve is off the track's span
            f0 = np.nanmedian(freqs / gm, axis=2)
            change = np.log(np.nanmax(gm, axis=2) / np.nanmin(gm, axis=2))
            sane = (fin.mean(2) > 0.5) & (np.abs(f0 / ev.f0 - 1.0) < f0_tol)
            if line_change > 0.03:
                sane &= (change >= change_ratio * line_change) & (change <= line_change / change_ratio)
        got = np.where(sane, np.mean(fin & (np.abs(f0[:, :, None] * g - freqs) < tol_hz), axis=2), 0.0)
        j = got.argmax(1)
        rows = np.arange(len(j))
        share[sl], start[sl], f0_out[sl] = got[rows, j], st[rows, j], f0[rows, j]
    return share, start, f0_out


def _anchored_job(job):
    """One anchor flight: where its closest approach fits the straight pass and its curve follows that
    line, then which of its possible partners' curves follow the other pass's line from there."""
    a, partners = job
    w = _worker
    legs, anchor, other, half, spacing, c = w["legs"], w["anchor"], w["other"], w["half"], w["spacing"], w["c"]
    start, min_share = w["start"], w["min_share"]
    keys_of = [_lattice_keys(rects, half, spacing, w["area"]) for _, rects in partners]
    keys = np.unique(np.concatenate(keys_of)) if keys_of else np.zeros(0, np.int64)
    out = dict(best=[], checked=int(keys.size), anchored=0, followed=0)
    if not keys.size:
        return out
    site = w["site"]
    xyz = site.place((keys >> 32) * spacing - half, (keys & 0xFFFFFFFF) * spacing - half)
    if len(site.dem._tiles) > TILES_KEPT:
        site.dem._tiles.clear()
    fa = legs.track(a)
    ca = closest_approach(fa, xyz, c, max_dist=w["audible"])
    ok = ca.valid & _within(ca.dist, anchor.dist_range) & _within(ca.speed, anchor.speed_range)
    if start is not None:
        ok &= np.abs(ca.t_heard - (start[0] + anchor.t_heard)) < start[1] + w["offset_tol"]
    out["in_range"] = int(ok.sum())
    ok[ok] = w["region"].allows(ca.speed[ok], ca.dist[ok])
    out["in_region"] = int(ok.sum())
    near = np.flatnonzero(ok)
    if not near.size:
        return out
    sa, st_a, f0_a = follow_line(fa, xyz[near], ca.t_heard[near] - anchor.t_heard, anchor, w["line_a"], f0_tol=w["f0_tol_a"], tol_hz=w["tol_hz"], c=c)
    keep = sa >= min_share
    good = near[keep]
    if not good.size:
        return out
    out["anchored"] = int(good.size)
    share_a, start_a, f0a = np.zeros(len(keys)), np.zeros(len(keys)), np.zeros(len(keys))
    share_a[good], start_a[good], f0a[good] = sa[keep], st_a[keep], f0_a[keep]
    is_good = np.zeros(len(keys), bool)
    is_good[good] = True
    n_a, n_b = len(anchor.line[0]), len(other.line[0])
    for (b, _), k in zip(partners, keys_of):
        sel = np.searchsorted(keys, k)
        sel = sel[is_good[sel]]
        if not sel.size:
            continue
        fb = legs.track(b)
        sb, st_b, f0_b = follow_line(fb, xyz[sel], start_a[sel], other, w["line_b"], f0_tol=w["f0_tol_b"], tol_hz=w["tol_hz"], c=c)
        hit = np.flatnonzero(sb >= min_share)
        if not hit.size:
            continue
        out["followed"] += int(hit.size)
        total = share_a[sel[hit]] * n_a + sb[hit] * n_b
        m = hit[int(np.argmax(total))]
        p = sel[m]
        out["best"].append(dict(a=a, b=b, key=int(keys[p]), xyz=xyz[p], share_a=float(share_a[p]), share_b=float(sb[m]), f0_a=float(f0a[p]),
                                f0_b=float(f0_b[m]), t0=float(st_b[m]), followed=[float(share_a[p]) * n_a, float(sb[m]) * n_b]))
    return out


def search_anchored_wide(legs: Legs, samples, anchor: Event, other: Event, site, *, half, spacing, start=None, offset_tol=8.0, audible_dist=10000.0,
                         tol_hz=1.5, f0_tol_a=0.15, f0_tol_b=0.3, min_share=FOLLOW_MIN_SHARE, c=SPEED_OF_SOUND, workers=1, sample_dt=SAMPLE_S, progress=None,
                         area=None):
    """A pair of passes of which only `anchor` is a straight pass that its fit can be trusted for.
    Its flight must fit it (speed, distance, closest approach, then the curve), which also fixes the
    recording's start. `other`'s flight need only be within earshot when it was heard and its curve
    follow the traced line. Returns (PairResult, one row per flight pair; followed (n, 2); xyz (n, 3))
    as `search_two_events_wide` does, with `anchor` first."""
    clock = [time.perf_counter()]
    sa, va = _usable(samples, anchor)
    sb = _thinned(samples, sample_dt, FOLLOW_SAMPLE_S)
    sb = sb[(sb[:, 5] >= FOLLOW_SPEEDS[0]) & (sb[:, 5] <= FOLLOW_SPEEDS[1])]
    vb = FOLLOW_SPEEDS[1]
    d1 = anchor.dist_range
    d1_hi = min(d1[1], audible_dist)
    if start is not None:
        when = start[0] + anchor.t_heard
        sa = sa[(sa[:, 1] > when - start[1] - offset_tol - d1_hi / c - sample_dt) & (sa[:, 1] < when + start[1] + offset_tol + sample_dt)]
    tilt = half * 1.5 / EARTH_M * 3000.0
    ra = d1_hi + va * sample_dt / 2 + spacing + 300.0 + tilt
    rb = audible_dist + vb * FOLLOW_SAMPLE_S / 2 + spacing + 300.0 + tilt
    mid = float(np.mean(other.line[0][[0, -1]]))
    gap = mid - anchor.t_heard
    slack = offset_tol + sample_dt + FOLLOW_SAMPLE_S / 2
    pairs = _partners_of(sa, ra, sb, rb, gap + (d1[0] - audible_dist) / c - slack, gap + d1_hi / c + slack, half, spacing)
    del sa, sb
    joined = sum(len(v) for v in pairs.values())
    clock.append(time.perf_counter())
    _prefetch_terrain(site, pairs, half, spacing, progress)
    clock.append(time.perf_counter())
    line_a = _line_sample(anchor)
    state = dict(legs=legs, anchor=anchor, other=other, half=half, spacing=spacing, site=site, c=c, audible=audible_dist, start=start, offset_tol=offset_tol,
                 tol_hz=tol_hz, f0_tol_a=f0_tol_a, f0_tol_b=f0_tol_b, min_share=min_share, line_a=line_a, line_b=_line_sample(other), area=area,
                 region=FitRegion(anchor, line_a, tol_hz=tol_hz, min_share=min_share, c=c))
    jobs = sorted(pairs.items(), key=lambda kv: -sum(len(r) for _, r in kv[1]))
    res = _map(_anchored_job, jobs, workers, state)
    clock.append(time.perf_counter())
    best = [b for x in res for b in x["best"]]
    funnel = {"flight pairs near each other": joined, "points checked": sum(x["checked"] for x in res), "speed & distance in range": sum(x.get("in_range", 0) for x in res),
              "could draw the line": sum(x.get("in_region", 0) for x in res), "anchor's line followed": sum(x["anchored"] for x in res),
              "+ the other line followed": sum(x["followed"] for x in res), "flight pairs left": len(best)}
    seconds = dict(zip(("flights compared", "terrain", "points scored"), np.diff(clock)))
    col = lambda k, dtype=float: np.array([b[k] for b in best], dtype)  # noqa: E731
    followed = np.array([b["followed"] for b in best], float).reshape(-1, 2)
    xyz = np.array([b["xyz"] for b in best], float).reshape(-1, 3)
    result = PairResult(point=col("key", np.int64), a=col("a", np.int64), b=col("b", np.int64), score1=col("share_a"), score2=col("share_b"), f0_1=col("f0_a"), f0_2=col("f0_b"),
                        t0=col("t0"), funnel=funnel, seconds=seconds)
    return result, followed, xyz
