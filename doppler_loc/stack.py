"""Locate a recording by stacking its spectrogram along every flight's predicted Doppler curve.

The other searches start from passes found on the spectrogram (detected or traced by hand). This
one needs none: for every flight, every place it passes and every recording start, it takes the
spectrogram's peaks heard around the flight's closest approach there and divides out the Doppler
factor the flight's real track predicts. If that flight was heard there, its tone's peaks all
imply one emitted pitch and pile up, whatever the pitch is and however the plane turned; peaks of
anything else scatter. The emitted pitch is unknown, so the piles are counted on a log-frequency
axis, where the Doppler shift is a slide (as pulsar searches "dedisperse").

Two things keep chance from winning. A far-away flight's curve is nearly flat, and a flat curve
piles up any steady tone (a horn, machinery, insects), so only what a curve gathers beyond the
best steady pitch counts. And one flight's curve fits a whole line of places (its start shifting
to match), so each flight's evidence is measured against its own spread (a z score) and capped:
several flights must agree on a place and a start.

    peaks = spectrogram_peaks(spec)
    sm = stack_search(flights, grid, peaks, (start, 60.0), passes=Passes(flights, grid), workers=8)
    obs, t0, score, heard = refine_place(flights, grid[sm.best()[0], :2], sm.start[sm.best()[0]], peaks, sm.stats,
                                         place=site.place)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import warnings

import numpy as np

from .physics import SPEED_OF_SOUND, doppler_factor
from .search import Passes, _map, _worker, observer_grid
from .spectro import Spectrogram


@dataclass
class Peaks:
    """The strongest peaks of a spectrogram, per 1-s frame, on a log-frequency axis."""

    t: np.ndarray  # recording time, s, ascending
    u: np.ndarray  # log-frequency bin: f = f_lo * exp(step * u)
    w: np.ndarray  # dB above the band's median
    step: float
    f_lo: float

    def freq(self, u):
        return self.f_lo * np.exp(self.step * np.asarray(u, dtype=float))

    def bin(self, f):
        return np.log(np.asarray(f, dtype=float) / self.f_lo) / self.step

    def frames(self):
        """The peaks as a table, one row per 1-s frame: (time of row 0, bins (frames, k), dB (frames, k)),
        dB 0 where a frame has fewer than k peaks; None unless the frames are whole seconds apart."""
        if "_frames" not in self.__dict__:
            out = None
            if self.t.size:
                m = np.rint(self.t - self.t[0]).astype(np.int64)
                if np.abs(self.t - self.t[0] - m).max() < 1e-6:
                    first = np.r_[0, np.flatnonzero(np.diff(m)) + 1]
                    col = np.arange(len(m)) - np.repeat(first, np.diff(np.r_[first, len(m)]))
                    U = np.zeros((m[-1] + 1, col.max() + 1), np.int32)
                    W = np.zeros(U.shape, np.float32)
                    U[m, col], W[m, col] = self.u, self.w
                    out = (float(self.t[0]), U, W)
            self.__dict__["_frames"] = out
        return self.__dict__["_frames"]


def spectrogram_peaks(spec: Spectrogram, *, step=0.004, f_lo=30.0, f_hi=600.0, per_s=8, min_db=6.0) -> Peaks:
    """The `per_s` strongest local maxima (at least `min_db` above the median) of each second of
    `spec.score`, resampled to log-frequency bins `step` wide (0.4%) between f_lo and f_hi."""
    s = spec.score
    n = s.shape[1] // 2
    s = np.maximum(s[:, 0:2 * n:2], s[:, 1:2 * n:2])  # 1-s frames: the brighter of each pair
    t = spec.t[0:2 * n:2] + spec.hop / 2
    U = int(np.log(f_hi / f_lo) / step) + 1
    fu = f_lo * np.exp(step * np.arange(U))
    lo = np.clip(np.floor((fu * np.exp(-step / 2) - spec.f[0]) / spec.df).astype(int), 0, len(spec.f) - 1)
    L = np.maximum.reduceat(s, lo, axis=0).T  # (frames, U): the brightest linear bin in each log bin
    peak = (L[:, 1:-1] > L[:, :-2]) & (L[:, 1:-1] >= L[:, 2:]) & (L[:, 1:-1] >= min_db)
    ti, ui = np.nonzero(peak)
    ui = ui + 1
    w = L[ti, ui]
    order = np.lexsort((-w, ti))
    ti, ui, w = ti[order], ui[order], w[order]
    first = np.r_[0, np.flatnonzero(np.diff(ti)) + 1]
    rank = np.arange(len(ti)) - np.repeat(first, np.diff(np.r_[first, len(ti)]))
    keep = rank < per_s
    return Peaks(t[ti[keep]], ui[keep].astype(np.int32), w[keep].astype(np.float32), step, f_lo)


def _pile3(c):
    """Each bin plus its two neighbours, along the last axis."""
    h = c.copy()
    h[:, :-1] += c[:, 1:]
    h[:, 1:] += c[:, :-1]
    return h


def flight_evidence(track, observers, th, t0, taus, peaks: Peaks, *, window=90.0, f0_range=(35.0, 500.0),
                    c=SPEED_OF_SOUND):
    """How much `track`'s Doppler curve at each observer gathers the spectrogram's peaks.

    observers (n, 3) with th (n,) the absolute delayed-CPA time there; t0 the absolute time of
    recording time 0 before the offsets `taus` (nτ,), one for all or one per observer. Peaks heard within `window` s of th are
    shifted by the curve and histogrammed by the emitted pitch they imply (piles 3 bins wide).
    Returns (E, B), both (n, nτ): mean dB per second the best pile holds beyond what the best
    steady pitch holds for the same peaks, and that pile's log-frequency bin.
    """
    taus = np.asarray(taus, dtype=float)
    n, nt = len(observers), len(taus)
    t0 = np.broadcast_to(np.asarray(t0, dtype=float), (n,))
    E = np.zeros((n, nt), np.float32)
    B = np.zeros((n, nt), np.int32)
    tol = float(np.abs(taus).max()) if nt else 0.0
    u_lo, u_hi = int(np.ceil(peaks.bin(f0_range[0]))), int(peaks.bin(f0_range[1]))
    nb = u_hi - u_lo + 1
    span = np.arange(-window - tol, window + tol + 1.0)  # absolute times around th, 1 s apart
    table = peaks.frames()
    if table is not None:
        # Span positions a start's window can cover, and each start's index (nτ, 1, 1).
        cover = np.arange(max(0, int(np.floor(tol)) - 1), min(len(span), int(np.ceil(2 * window + tol)) + 2))
        rows = np.arange(nt)[:, None, None]
    for s in range(0, n, 64):
        sl = slice(s, min(s + 64, n))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # log of NaN: curve off the track's span
            lng = np.round(np.log(doppler_factor(track, observers[sl, None, :], th[sl, None] + span, c)) / peaks.step)
        for j, k in enumerate(range(sl.start, sl.stop)):
            a = np.searchsorted(peaks.t, th[k] - t0[k] - window - tol)
            b = np.searchsorted(peaks.t, th[k] - t0[k] + window + tol)
            if b - a < 5:
                continue
            # Starts a block at a time: each block's arrays stay in the core's own cache, which
            # matters with every core reading at once (the starts don't depend on each other).
            # Starts with nothing to pile get bin u_lo, as from an empty histogram, if any start has some.
            blocks, empty = _blocks(nt, TAU_BLOCK) if table is not None else [slice(0, nt)], []
            for blk in blocks:
                tb = taus[blk]
                if table is not None:
                    got = _entries_table(table, lng[j], cover, rows[:len(tb)], th[k], t0[k], tb, window, tol, u_lo, nb)
                else:
                    got = _entries_list(peaks, a, b, lng[j], len(span), th[k], t0[k], tb, window, tol, u_lo, nb)
                if got is None:
                    empty.append(blk)
                    continue
                ti, ub, u0, wts = got
                ok0 = (u0 >= 1) & (u0 < nb - 1)
                # Piles 3 bins wide: count each bin once (bincount, far faster than np.add.at), then
                # add each bin's neighbours.
                H, H0 = (_pile3(np.bincount(r * nb + u, w_, len(tb) * nb).reshape(len(tb), nb))
                         for r, u, w_ in ((ti, ub, wts), (ti[ok0], u0[ok0], wts[ok0])))
                E[k, blk] = np.clip(H.max(1) - H0.max(1), 0, None) / (2 * window)
                B[k, blk] = H.argmax(1) + u_lo
            if len(empty) < len(blocks):
                for blk in empty:
                    B[k, blk] = u_lo
    return E, B


TAU_BLOCK = 32  # starts per block: ~2.4x faster than all at once with 8 workers


def _blocks(n, size):
    return [slice(a, min(a + size, n)) for a in range(0, n, size)]


def _entries_table(table, lng, cover, rows, th, t0, taus, window, tol, u_lo, nb):
    """(start index, shifted bin, bin, dB) of every peak within `window` s of the curve for each start,
    read from the peaks' frame table: a start's peaks are a run of whole frames, so all starts are
    one (nτ, positions, k) block rather than a list of start x peak pairs."""
    P0, U, W = table
    # Peak frame m lands at span position rel = phi + m (the curve's time, from -window - tol).
    phi = (t0 + taus + P0) - (th - window - tol)
    r = np.rint(phi)
    inside = np.abs(cover[None, :] + (phi - r)[:, None] - (window + tol)) <= window  # (nτ, positions)
    m = cover[None, :] - r.astype(np.int64)[:, None]
    inside &= (m >= 0) & (m < len(U))
    shift = lng[cover]
    inside &= np.isfinite(shift)[None, :]
    if not inside.any():
        return None
    mc = np.where(inside, m, 0)
    Um = U[mc]  # (nτ, positions, k)
    u0 = Um - u_lo
    ub = u0 - np.where(np.isfinite(shift), shift, 0).astype(np.int32)[None, :, None]
    ok = inside[:, :, None] & (Um > 0) & (ub >= 1) & (ub < nb - 1)  # bin 0 marks no peak
    if not ok.any():
        return None
    return np.broadcast_to(rows, ok.shape)[ok], ub[ok], u0[ok], W[mc][ok]


def _entries_list(peaks, a, b, lng, n_span, th, t0, taus, window, tol, u_lo, nb):
    """As `_entries_table`, for peaks not on whole-second frames: each start's run of the list."""
    pt, pu, pw = peaks.t[a:b], peaks.u[a:b], peaks.w[a:b]
    centre = th - t0 - taus
    lo = np.searchsorted(pt, centre - window - 1.0)
    n_in = np.searchsorted(pt, centre + window + 1.0, "right") - lo
    if not n_in.sum():
        return None
    ti = np.repeat(np.arange(len(taus)), n_in)
    qi = np.arange(n_in.sum()) - np.repeat(np.cumsum(n_in) - n_in - lo, n_in)
    rel = (t0 + taus[ti] + pt[qi]) - (th - window - tol)  # into span
    idx = np.rint(rel).astype(int)
    inside = (np.abs(rel - (window + tol)) <= window) & (idx >= 0) & (idx < n_span)
    ti, qi, idx = ti[inside], qi[inside], idx[inside]
    shift = lng[idx]
    fin = np.isfinite(shift)
    ti, qi, shift = ti[fin], qi[fin], shift[fin]
    u0 = pu[qi] - u_lo
    ub = u0 - shift.astype(int)
    fin = (ub >= 1) & (ub < nb - 1)
    if not fin.any():
        return None
    return ti[fin], ub[fin], u0[fin], pw[qi[fin]]


def _normalise(E, min_points=20):
    """(median, spread) of a flight's nonzero evidence, or None when it has too little to judge."""
    v = E[E > 0]
    if v.size < min_points:
        return None
    med = float(np.median(v))
    return med, max(float(np.median(np.abs(v - med))) * 1.4826, 0.05 * med, 0.05)


def contribution(E, stats, cap=6.0, floor=2.0):
    """A flight's capped share of the total: its z score above `floor`, at most `cap` - `floor`."""
    med, spread = stats
    z = np.where(E > 0, (E - med) / spread, 0.0)
    return np.clip(z - floor, 0.0, cap - floor)


@dataclass
class StackMap:
    """Stacked evidence over a grid (see `stack_search`)."""

    score: np.ndarray  # (P,) the best total over the starts tried
    start: np.ndarray  # (P,) absolute recording start at that best
    stats: dict  # flight index -> (median, spread) of its evidence, to score other places alike
    flights: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (P,) flights adding to the best
    seconds: float = 0.0

    def places(self, observers, n=5, apart=2000.0) -> list[int]:
        """The `n` best grid points at least `apart` m from each other."""
        out = []
        for k in np.argsort(-self.score):
            if self.score[k] <= 0 or len(out) == n:
                break
            if all(np.hypot(*(observers[k, :2] - observers[j, :2])) >= apart for j in out):
                out.append(int(k))
        return out


def _stack_job(i):
    w = _worker
    idx, th = w["passes"].idx[i], w["passes"].th[i]
    t0, taus, pk, window = w["t0"], w["taus"], w["peaks"], w["window"]
    tol = float(np.abs(taus).max())
    ok = (th > t0 + pk.t[0] - window - tol) & (th < t0 + pk.t[-1] + window + tol) if pk.t.size else th < -np.inf
    if not ok.any():
        return i, idx[:0], np.zeros((0, len(taus)), np.float32)
    E, _ = flight_evidence(w["flights"][i], w["observers"][idx[ok]], th[ok], t0, taus, pk, window=window,
                           f0_range=w["f0_range"], c=w["c"])
    return i, idx[ok], E


def stack_search(flights, observers, peaks: Peaks, start, *, passes: Passes, tau_step=2.0, window=90.0, cap=6.0,
                 floor=2.0, f0_range=(35.0, 500.0), c=SPEED_OF_SOUND, workers=1) -> StackMap:
    """Stack every flight's curve at every observer it passes (`passes`, which must cover at least
    the distance a plane is heard from) and every recording start within `start` = (absolute
    start, uncertainty in s), `tau_step` s apart; sum the flights' capped contributions."""
    t_begin = time.time()
    observers = np.asarray(observers, dtype=float)
    taus = np.arange(-start[1], start[1] + 1e-9, tau_step)
    state = dict(flights=flights, observers=observers, passes=passes, t0=float(start[0]), taus=taus, peaks=peaks,
                 window=window, f0_range=f0_range, c=c)
    total = np.zeros((len(observers), len(taus)))
    count = np.zeros((len(observers), len(taus)))
    stats = {}
    # Flights passing most places first, so that no worker is left with a big one at the end.
    order = sorted(range(len(flights)), key=lambda i: -passes.idx[i].size)
    for i, idx, E in _map(_stack_job, order, workers, state):
        st = _normalise(E) if len(idx) else None
        if st is None:
            continue
        stats[i] = st
        part = contribution(E, st, cap, floor)
        total[idx] += part
        count[idx] += part > 0
    best = total.argmax(1)
    rows = np.arange(len(observers))
    return StackMap(score=total[rows, best], start=start[0] + taus[best], stats=stats, flights=count[rows, best],
                    seconds=time.time() - t_begin)


def stack_at(flights, observers, t0, peaks: Peaks, stats, *, taus=np.arange(-4.0, 4.1, 1.0), window=90.0,
             cap=6.0, floor=2.0, max_dist=10000.0, f0_range=(35.0, 500.0), c=SPEED_OF_SOUND):
    """Stacked evidence at a few observers (n, 3) around absolute start t0 (+ `taus`; one for all or
    one per observer), using the flights' spreads from a `stack_search` (`stats`, by index into
    `flights`).

    Returns (total (n, nτ), heard), heard[k] listing (flight index, contribution, emitted pitch Hz,
    absolute delayed-CPA time) of the flights adding to observer k at its best start."""
    observers = np.atleast_2d(np.asarray(observers, dtype=float))
    taus = np.asarray(taus, dtype=float)
    t0 = np.broadcast_to(np.asarray(t0, dtype=float), (len(observers),))
    # Only flights in the air while the recording ran can be heard on it.
    lo, hi = t0.min() + peaks.t[0] - window - 600.0, t0.max() + peaks.t[-1] + window + 600.0
    sel = [i for i in stats if i < len(flights) and flights[i].t_end > lo and flights[i].t_start < hi]
    total = np.zeros((len(observers), len(taus)))
    parts = []
    if sel:
        passes = Passes([flights[i] for i in sel], observers, max_dist)
        for j, i in enumerate(sel):
            idx, th = passes.idx[j], passes.th[j]
            ok = (th > t0[idx] + peaks.t[0] - window - 10) & (th < t0[idx] + peaks.t[-1] + window + 10)
            if not ok.any():
                continue
            E, B = flight_evidence(flights[i], observers[idx[ok]], th[ok], t0[idx[ok]], taus, peaks, window=window,
                                   f0_range=f0_range, c=c)
            part = contribution(E, stats[i], cap, floor)
            total[idx[ok]] += part
            parts.append((i, idx[ok], th[ok], part, B))
    best = total.argmax(1)
    heard = [[] for _ in range(len(observers))]
    for i, idx, th, part, B in parts:
        for r, k in enumerate(idx):
            if part[r, best[k]] > 0:
                heard[k].append((i, float(part[r, best[k]]), float(peaks.freq(B[r, best[k]])), float(th[r])))
    for h in heard:
        h.sort(key=lambda x: -x[1])
    return total, heard


def refine_place(flights, centre, t0, peaks: Peaks, stats, *, place, steps=((1000.0, 200.0), (200.0, 25.0)),
                 cap=6.0, fine_cap=30.0, start_window=None, **kw):
    """Close in on the best place around `centre` (x, y): for each (half width, spacing) in `steps`,
    stack a grid there (starts within +-4 s of the best so far, and inside `start_window`, (earliest,
    latest) absolute start, if given) and move to its best point.

    The cap that makes flights agree on a place would make every point near it tie once each
    flight is capped, so the grids are scored with the much higher `fine_cap`, and the place found
    scored with `cap` again. Returns (observer (3,), absolute start, score, heard) as `stack_at`
    gives them."""
    centre = np.asarray(centre, dtype=float)[:2]
    # Only the flights heard around the starting point can move the best place: score just those.
    near = observer_grid(tuple(centre), steps[0][0], steps[0][0], place=place)
    _, heard_near = stack_at(flights, near, t0, peaks, stats, cap=cap, **kw)
    stats = {i: stats[i] for h in heard_near for i, *_ in h}
    obs = None
    for half, spacing in steps:
        grid = observer_grid(tuple(centre), half, spacing, place=place)
        taus = np.arange(-4.0, 4.1, 1.0)
        if start_window is not None:
            inside = (t0 + taus >= start_window[0] - 1e-6) & (t0 + taus <= start_window[1] + 1e-6)
            taus = taus[inside] if inside.any() else np.zeros(1)
        total, _ = stack_at(flights, grid, t0, peaks, stats, taus=taus, cap=fine_cap, **kw)
        k, j = np.unravel_index(int(total.argmax()), total.shape)
        obs, centre, t0 = grid[k], grid[k, :2], t0 + taus[j]
    total, heard = stack_at(flights, obs, t0, peaks, stats, taus=np.zeros(1), cap=cap, **kw)
    return obs, t0, float(total[0, 0]), heard[0]
