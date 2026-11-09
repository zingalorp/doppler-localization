"""Find where a recording was made: candidate grids, cheap filters, curve scoring, refinement."""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import mmap
import multiprocessing
import os
import pickle
import sys
import tempfile
import time
import warnings
import weakref

import numpy as np

from .flight import FlightTrack
from .measure import Event
from .physics import SPEED_OF_SOUND, closest_approach, doppler_factor
from .score import EventScore, _padded, line_agreement, score_event
from .spectro import Spectrogram


def flat_ground(x, y):
    return np.zeros(np.broadcast(x, y).shape)


def observer_grid(center, half_size, spacing, ground=flat_ground, mic_height=1.5, place=None) -> np.ndarray:
    """Square grid of candidate microphone positions (P, 3) standing on `ground(x, y)`.

    For real terrain pass `place(x, y) -> (P, 3)` instead, which can follow the DEM and the
    Earth's curvature (the ground is ~50 m below the tangent plane 25 km out).
    """
    ax = np.arange(-half_size, half_size + spacing / 2, spacing)
    X, Y = np.meshgrid(center[0] + ax, center[1] + ax)
    x, y = X.ravel(), Y.ravel()
    if place is not None:
        return place(x, y)
    return np.stack([x, y, ground(x, y) + mic_height], axis=1)


def _within(x, bounds):
    return (x >= bounds[0]) & (x <= bounds[1])


def _shift_frames(spec: Spectrogram, tol: float) -> np.ndarray:
    n = int(round(tol / spec.hop))
    return np.arange(-n, n + 1)


# --- one pass, recording time known -------------------------------------------------------


@dataclass
class SingleResult:
    observers: np.ndarray  # (P, 3) every candidate
    flight: np.ndarray  # (P,) index of the best-scoring flight, -1 if none passed the filters
    score: np.ndarray  # (P,) best score, NaN if never scored
    f0: np.ndarray
    clock_error: np.ndarray  # (P,) best correction to the stated start time, s

    @property
    def best(self) -> int:
        return int(np.nanargmax(self.score))


def search_known_time(
    spec: Spectrogram,
    flights: list[FlightTrack],
    observers,
    event: Event,
    t0: float,
    *,
    clock_tol=3.0,
    window=60.0,
    c=SPEED_OF_SOUND,
) -> SingleResult:
    """Locate one measured pass when the recording started at about `t0` (± `clock_tol`)."""
    observers = np.asarray(observers, dtype=float)
    P = len(observers)
    res = SingleResult(observers, np.full(P, -1), np.full(P, np.nan), np.zeros(P), np.zeros(P))
    shifts = _shift_frames(spec, clock_tol)
    expected = t0 + event.t_heard
    for i, fl in enumerate(flights):
        ca = closest_approach(fl, observers, c)
        ok = (
            ca.valid
            & _within(ca.dist, event.dist_range)
            & _within(ca.speed, event.speed_range)
            & (np.abs(ca.t_heard - expected) <= clock_tol + 2.0)  # + measurement error of t_heard
        )
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            continue
        sc = score_event(
            spec, fl, observers[idx], t0, ca.t_heard[idx], event.f0_grid(), shifts,
            window=window, harmonics=event.harmonics, c=c,
        )
        better = ~(sc.score <= res.score[idx])  # also true where nothing was scored yet
        j = idx[better]
        res.flight[j] = i
        res.score[j] = sc.score[better]
        res.f0[j] = sc.f0[better]
        res.clock_error[j] = sc.shift[better] * spec.hop
    return res


def refine_single(
    spec, track, center, event, t0, *, half_size=600.0, spacing=25.0, ground=flat_ground, place=None,
    clock_tol=3.0, window=60.0, c=SPEED_OF_SOUND,
) -> tuple[np.ndarray, EventScore]:
    """Re-score a fine grid around `center` for an already-identified flight."""
    obs = observer_grid(center, half_size, spacing, ground, place=place)
    ca = closest_approach(track, obs, c)
    sc = score_event(
        spec, track, obs, t0, ca.t_heard, event.f0_grid(), _shift_frames(spec, clock_tol),
        window=window, harmonics=event.harmonics, c=c,
    )
    return obs, sc


# --- two passes, recording time unknown ---------------------------------------------------


@dataclass
class PairResult:
    """One row per (candidate point, event-1 flight, event-2 flight) that survived the filters."""

    point: np.ndarray
    a: np.ndarray
    b: np.ndarray
    score1: np.ndarray
    score2: np.ndarray
    f0_1: np.ndarray
    f0_2: np.ndarray
    t0: np.ndarray  # implied absolute start time of the recording
    funnel: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)  # time spent in each stage

    @property
    def total(self) -> np.ndarray:
        return self.score1 + self.score2

    def best_per_pair(self) -> np.ndarray:
        """Row index of the top-scoring point for each flight pair, best pair first."""
        rows = []
        for key in set(zip(self.a.tolist(), self.b.tolist())):
            sel = np.flatnonzero((self.a == key[0]) & (self.b == key[1]))
            rows.append(sel[np.argmax(self.total[sel])])
        rows = np.array(rows, dtype=int)
        return rows[np.argsort(-self.total[rows])]


def score_pair(spec, fa, fb, observers, e1, e2, th_a, th_b, shifts, *, window=60.0, c=SPEED_OF_SOUND):
    """Score two events with their relative timing locked.

    Event 1 sets the clock: recording time 0 is its delayed CPA minus its measured position on
    the recording, then corrected by the best of `shifts`. Event 2 is not allowed its own time
    correction; if the geometry puts its curve in the wrong place, that counts against it.
    """
    t0 = th_a - e1.t_heard
    s1 = score_event(spec, fa, observers, t0, th_a, e1.f0_grid(), shifts, window=window, harmonics=e1.harmonics, c=c)
    s2 = score_event(
        spec, fb, observers, t0, th_b, e2.f0_grid(), s1.shift[:, None], window=window, harmonics=e2.harmonics, c=c
    )
    return s1, s2


_worker: dict = {}


def _init_worker(state):
    _worker.update(state)


def _pool(workers, state):
    """A pool of `workers` forked processes (Linux), each starting with `state` in `_worker`.

    Forks start at once and share memory. Python warns about forking a process that has threads,
    but the only threads here are OpenBLAS's and SciPy's FFT pool, and both stop and restart
    themselves around a fork (pthread_atfork).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "This process .* is multi-threaded", DeprecationWarning)
        return multiprocessing.get_context("fork").Pool(workers, _init_worker, (state,))


# Elsewhere fork is not safe (macOS system libraries) or not available (Windows), so workers are
# fresh interpreters, which take a second each to start and get everything pickled. A pool per
# call, with `state` pickled for each worker, spent half of a search pickling on one core while the
# rest waited (60 of 116 s, Windows, 16 workers). So one pool serves the whole process, and each
# call's `state` is pickled once, to a file each worker loads the first time it needs it. Its big
# arrays (flight tracks, the passes table, the spectrogram) go to a second file that the workers
# map into memory, copy-on-write, rather than each reading in its own copy: over a wide area
# those arrays run to gigabytes, and eight copies of them would not fit. Shared arrays are read-only
# in the workers, so jobs must not write into their inputs.
_spawned = None
_BIG = 1 << 16  # arrays from 64 kB up are shared through the mapped file
_stale: list[str] = []  # files the workers still had mapped when a call ended, deleted later
# Big arrays already written, by id: (weak reference, file, offset, dtype, shape). A search hands the
# same spectrogram, tracks and passes table to call after call (45 pairs of passes: 45 calls), and
# writing the 170 MB spectrogram out again each time was a quarter of a search. The weak reference
# says whether it is still the same array (an id is reused once its array is gone). So an array
# handed to workers must not be changed in place afterwards: they would keep reading the old values.
_written: dict[int, tuple] = {}
_data_files: list[str] = []  # files holding arrays in `_written`, oldest first
_maps: dict[str, object] = {}  # in a worker: the files its current state's arrays are mapped from


def _spawned_pool(workers):
    global _spawned
    if _spawned is None or _spawned[1] < workers:
        if _spawned is not None:
            _spawned[0].terminate()
        _spawned = (multiprocessing.get_context("spawn").Pool(workers), workers)
    return _spawned[0]


class _SharingPickler(pickle.Pickler):
    """Pickles big arrays as (file, offset, dtype, shape): ones written for an earlier call where
    they are, new ones into `data` (at `data_path`), 64-byte aligned."""

    def __init__(self, f, data, data_path):
        super().__init__(f, protocol=pickle.HIGHEST_PROTOCOL)
        self.data, self.data_path = data, data_path

    def persistent_id(self, obj):
        if type(obj) is not np.ndarray or obj.nbytes < _BIG or obj.dtype.hasobject:
            return None
        known = _written.get(id(obj))
        if known is None or known[0]() is not obj:
            arr = np.ascontiguousarray(obj)
            self.data.write(b"\0" * (-self.data.tell() % 64))
            known = (weakref.ref(obj), self.data_path, self.data.tell(), arr.dtype.str, arr.shape)
            _written[id(obj)] = known
            self.data.write(memoryview(arr).cast("B"))
        return ("nd",) + known[1:]


class _SharingUnpickler(pickle.Unpickler):
    def __init__(self, f, old_maps, new_maps):
        super().__init__(f)
        self.old, self.new = old_maps, new_maps

    def persistent_load(self, pid):
        _, path, offset, dtype, shape = pid
        if path not in self.new:
            if path in self.old:
                self.new[path] = self.old[path]
            else:
                with open(path, "rb") as d:
                    # Read-only: copy-on-write would reserve the whole view in every worker (Windows).
                    self.new[path] = mmap.mmap(d.fileno(), 0, access=mmap.ACCESS_READ)
        return np.frombuffer(self.new[path], np.dtype(dtype), count=int(np.prod(shape)), offset=offset).reshape(shape)


def _job_with_state(args):
    key, path, fn, x = args
    if _worker.get("_key") != key:
        _worker.clear()  # drop the last call's arrays, and so the mappings no longer used
        _padded.clear()  # and its padded spectrogram: this call's arrays are new objects anyway
        new = {}
        with open(path, "rb") as f:
            state = _SharingUnpickler(f, _maps, new).load()
        _maps.clear()
        _maps.update(new)
        _worker.update(state, _key=key)
    return fn(x)


def _delete_stale():
    for p in list(_stale):
        try:
            os.unlink(p)
            _stale.remove(p)
        except FileNotFoundError:
            _stale.remove(p)
        except PermissionError:
            pass  # still mapped by a worker


def _retire_data_files():
    """Mark for deletion the data files none of whose arrays are still alive."""
    for k in [k for k, v in _written.items() if v[0]() is None]:
        del _written[k]
    live = {v[1] for v in _written.values()}
    for p in [p for p in _data_files if p not in live]:
        _data_files.remove(p)
        _stale.append(p)


@atexit.register
def _shut_down():
    if _spawned is not None:
        _spawned[0].terminate()
        _spawned[0].join()
    _stale.extend(_data_files)
    _data_files.clear()
    _delete_stale()


def _map(fn, items, workers, state):
    """[fn(x) for x in items] in `workers` processes, each given the shared inputs `state` once
    (read by `fn` from `_worker`) rather than with every item. Everything is pickled to the
    workers, so items should carry only what their job needs."""
    if workers <= 1:
        _worker.update(state)
        try:
            return [fn(x) for x in items]
        finally:
            _worker.clear()
    if sys.platform.startswith("linux"):
        with _pool(workers, state) as pool:
            return pool.map(fn, items, chunksize=1)
    _delete_stale()
    fd, path = tempfile.mkstemp(prefix="doppler-state-", suffix=".pkl")
    data_path = path[:-4] + ".bin"
    try:
        _data_files.append(data_path)
        with os.fdopen(fd, "wb") as f, open(data_path, "wb") as d:
            _SharingPickler(f, d, data_path).dump(state)
        key = os.path.basename(path)
        return _spawned_pool(workers).map(_job_with_state, [(key, path, fn, x) for x in items], chunksize=1)
    finally:
        _stale.append(path)
        _retire_data_files()
        _delete_stale()


def _passes_job(flight):
    w = _worker
    ca = closest_approach(flight, w["observers"], w["c"], max_dist=w["max_dist"])
    k = np.flatnonzero(ca.valid)
    return k.astype(np.int32), ca.t_heard[k], ca.dist[k], ca.speed[k]  # over a wide area this table is gigabytes


class Passes:
    """Where each flight is audible: for flight i, the observers `idx[i]` it passes within
    `max_dist` of, with the delayed-CPA time, CPA distance and speed at each. Kept sparse, since
    any one flight is near only a small part of a large grid. It does not depend on the
    recording, so one can serve searches with several pairs of events."""

    def __init__(self, flights, observers, max_dist=10000.0, c=SPEED_OF_SOUND, workers=1):
        res = _map(_passes_job, flights, workers, dict(observers=observers, max_dist=max_dist, c=c))
        self.idx, self.th, self.dist, self.speed = (list(col) for col in zip(*res)) if res else ([], [], [], [])
        self.t_min = np.array([th.min() if th.size else np.inf for th in self.th])
        self.t_max = np.array([th.max() if th.size else -np.inf for th in self.th])


class _HeardTimes:
    """Every (observer, heard time) of a flight passing within `max_dist`, sorted so that the
    number heard at observer p between two times is a pair of binary searches."""

    def __init__(self, passes: Passes, max_dist, pad):
        k = [i[d < max_dist] for i, d in zip(passes.idx, passes.dist)]
        th = [t[d < max_dist] for t, d in zip(passes.th, passes.dist)]
        idx, th = np.concatenate(k), np.concatenate(th)
        self.t_ref = (th.min() if th.size else 0.0) - pad
        self.span = (th.max() - self.t_ref if th.size else 0.0) + pad
        self.keys = np.sort(idx * self.span + (th - self.t_ref))

    def count(self, idx, lo, hi):
        """How many are heard at observers `idx` strictly between `lo` and `hi` (arrays)."""
        base = idx * self.span - self.t_ref
        lo = base + np.clip(lo, self.t_ref, self.t_ref + self.span)
        hi = base + np.clip(hi, self.t_ref, self.t_ref + self.span)
        n = np.searchsorted(self.keys, hi, "left") - np.searchsorted(self.keys, lo, "right")
        return np.maximum(n, 0)


def _score_job(job):
    """Curve scores for every surviving (point, b) of one event-1 flight a."""
    groups, flights = job  # flights: {index: FlightTrack} for a and its partners
    w = _worker
    spec, observers, e1, e2 = w["spec"], w["observers"], w["e1"], w["e2"]
    a = groups[0][0]
    # Event 1's score at a point does not depend on which flight made event 2, so score it once
    # per point and reuse it for every partner.
    pts, first = np.unique(np.concatenate([g[2] for g in groups]), return_index=True)
    th_a = np.concatenate([g[3] for g in groups])[first]
    s1 = score_event(spec, flights[a], observers[pts], th_a - e1.t_heard, th_a, e1.f0_grid(), w["shifts"],
                     window=w["window"], harmonics=e1.harmonics, c=w["c"])
    cols = {k: [] for k in ("point", "a", "b", "score1", "score2", "f0_1", "f0_2", "t0")}
    for _, b, idx, th_a, th_b in groups:
        r = np.searchsorted(pts, idx)
        s2 = score_event(spec, flights[b], observers[idx], th_a - e1.t_heard, th_b, e2.f0_grid(),
                         s1.shift[r, None], window=w["window"], harmonics=e2.harmonics, c=w["c"])
        cols["point"].append(idx)
        cols["a"].append(np.full(idx.size, a))
        cols["b"].append(np.full(idx.size, b))
        cols["score1"].append(s1.score[r])
        cols["score2"].append(s2.score)
        cols["f0_1"].append(s1.f0[r])
        cols["f0_2"].append(s2.f0)
        cols["t0"].append(th_a - e1.t_heard + s1.shift[r] * spec.hop)
    return cols


def search_two_events(
    spec: Spectrogram,
    flights: list[FlightTrack],
    observers,
    e1: Event,
    e2: Event,
    duration: float,
    *,
    offset_tol=6.0,
    shift_tol=4.0,
    silence_dist=3000.0,
    guard=20.0,
    audible_dist=10000.0,
    silence_between_only=False,
    window=60.0,
    c=SPEED_OF_SOUND,
    workers=1,
    passes: Passes | None = None,
    start: tuple[float, float] | None = None,
) -> PairResult:
    """Locate a recording with two measured passes and no idea when it was made, or with `start`
    = (absolute recording start, uncertainty in s) only pairs whose first flight is heard then.

    Filters, cheapest first:
      1. speed and CPA distance of each flight match the straight-line fits, which rules out most
         flights before any pair of them is compared (over 500 km, nearly all the time otherwise);
      2. offset: such a flight is heard, then another one (e2.t_heard - e1.t_heard) later;
      3. silence: no other flight within `silence_dist` is heard during the recording (or, with
         `silence_between_only`, between the two events, for recordings with other passes in them);
    then the survivors get the full curve scoring. The closest approaches and the scoring are
    spread over `workers` processes. Pass `passes` (for these flights and observers) to reuse
    closest approaches already worked out; `audible_dist` is then theirs.
    """
    observers = np.asarray(observers, dtype=float)
    P = len(observers)
    clock = [time.perf_counter()]
    if passes is None:
        passes = Passes(flights, observers, audible_dist, c, workers)
    clock.append(time.perf_counter())
    quiet = _HeardTimes(passes, silence_dist, pad=duration + guard + 1.0)
    delta = e2.t_heard - e1.t_heard

    heard_any = np.zeros(P, dtype=bool)
    hit_first = np.zeros(P, dtype=bool)
    hit_measured = np.zeros(P, dtype=bool)
    hit_quiet = np.zeros(P, dtype=bool)
    slot = np.full(P, -1)  # position of each observer in the current flight a's lists
    groups = []
    if start is not None:  # when the first event must have been heard
        when = start[0] + e1.t_heard
        slack = start[1] + offset_tol

    def matching(ev):
        """Each flight's observers it passes at the event's measured speed and distance."""
        return [np.flatnonzero(_within(passes.dist[k], ev.dist_range) & _within(passes.speed[k], ev.speed_range))
                for k in range(len(flights))]

    first, second = matching(e1), matching(e2)
    b_times = [passes.th[k][m] for k, m in enumerate(second)]
    b_min = np.array([t.min() if t.size else np.inf for t in b_times])
    b_max = np.array([t.max() if t.size else -np.inf for t in b_times])
    for a in range(len(flights)):
        if passes.idx[a].size == 0:
            continue
        heard_any[passes.idx[a]] = True
        m = first[a]
        if m.size == 0:
            continue
        ia_all, th_all = passes.idx[a][m], passes.th[a][m]
        hit_first[ia_all] = True
        if start is not None and (th_all.max() < when - slack or th_all.min() > when + slack):
            continue
        slot[ia_all] = np.arange(ia_all.size)
        # Only flights heard around delta after this one can pass the offset filter.
        cands = np.flatnonzero((b_max > th_all.min() + delta - offset_tol) & (b_min < th_all.max() + delta + offset_tol))
        for b in cands[cands != a]:
            j = slot[passes.idx[b][second[b]]]
            ib = np.flatnonzero(j >= 0)
            ia = j[ib]
            th_a, th_b = th_all[ia], b_times[b][ib]
            ok = np.abs(th_b - th_a - delta) < offset_tol
            if start is not None:
                ok &= np.abs(th_a - when) < slack
            if not ok.any():
                continue
            idx, th_a, th_b = ia_all[ia[ok]], th_a[ok], th_b[ok]
            hit_measured[idx] = True
            # Silence: nothing else heard in (lo, hi), outside `guard` of the two events themselves.
            rec_start = th_a - e1.t_heard
            lo, hi = (th_a, th_b) if silence_between_only else (rec_start, rec_start + duration)
            w1, w2 = np.minimum(th_a, th_b), np.maximum(th_a, th_b)
            others = (
                quiet.count(idx, lo, np.minimum(hi, w1 - guard))
                + quiet.count(idx, np.maximum(lo, w1 + guard), np.minimum(hi, w2 - guard))
                + quiet.count(idx, np.maximum(lo, w2 + guard), hi)
            )
            keep = others == 0
            if keep.any():
                hit_quiet[idx[keep]] = True
                groups.append((a, b, idx[keep], th_a[keep], th_b[keep]))
        slot[ia_all] = -1

    funnel = {
        "grid points": P,
        "hear any flight": int(heard_any.sum()),
        "first flight's speed & distance": int(hit_first.sum()),
        "+ second flight, at the right gap": int(hit_measured.sum()),
        "+ silence": int(hit_quiet.sum()),
        "flight pairs left": len(groups),
    }
    clock.append(time.perf_counter())

    by_a = {}
    for g in groups:
        by_a.setdefault(g[0], []).append(g)
    # Scoring reads only the score image, not the dB one.
    light = Spectrogram(spec.t, spec.f, np.empty((0, 0), dtype=np.float32), spec.score)
    state = dict(spec=light, observers=observers, e1=e1, e2=e2, shifts=_shift_frames(spec, shift_tol),
                 window=window, c=c)
    # Biggest first, so no worker is left with a large one at the end.
    jobs = sorted(by_a.values(), key=lambda gs: -sum(g[2].size for g in gs))
    jobs = [(gs, {k: flights[k] for g in gs for k in g[:2]}) for gs in jobs]
    cols = {k: [] for k in ("point", "a", "b", "score1", "score2", "f0_1", "f0_2", "t0")}
    for res in _map(_score_job, jobs, workers, state):
        for k, v in res.items():
            cols[k].extend(v)
    arrays = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    for k in ("point", "a", "b"):
        arrays[k] = arrays[k].astype(int)
    clock.append(time.perf_counter())
    seconds = dict(zip(("closest approach", "filters", "scoring"), np.diff(clock)))
    return PairResult(**arrays, funnel=funnel, seconds=seconds)


def refine_pair(
    spec, fa, fb, center, e1, e2, *, half_size=600.0, spacing=25.0, ground=flat_ground, place=None,
    shift_tol=4.0, window=60.0, c=SPEED_OF_SOUND,
) -> tuple[np.ndarray, EventScore, EventScore]:
    """Re-score a fine grid around `center` for an already-identified pair of flights."""
    obs = observer_grid(center, half_size, spacing, ground, place=place)
    ca, cb = closest_approach(fa, obs, c), closest_approach(fb, obs, c)
    s1, s2 = score_pair(spec, fa, fb, obs, e1, e2, ca.t_heard, cb.t_heard, _shift_frames(spec, shift_tol),
                        window=window, c=c)
    return obs, s1, s2


# --- one event, recording start known --------------------------------------------------------


@dataclass
class OneResult:
    """One row per (candidate point, flight) for a single event with a known recording start."""

    point: np.ndarray
    flight: np.ndarray
    score: np.ndarray
    f0: np.ndarray
    start: np.ndarray  # recording start implied by the best clock shift, absolute
    funnel: dict = field(default_factory=dict)

    def best_per_flight(self) -> np.ndarray:
        """Row index of the top-scoring point for each flight, best flight first."""
        rows = []
        for f in np.unique(self.flight):
            sel = np.flatnonzero(self.flight == f)
            rows.append(sel[np.argmax(self.score[sel])])
        rows = np.array(rows, dtype=int)
        return rows[np.argsort(-self.score[rows])]


def search_known_start(
    spec: Spectrogram,
    flights: list[FlightTrack],
    observers,
    event: Event,
    start: tuple[float, float],
    *,
    passes: Passes,
    measure_tol=8.0,
    use_fit=True,
    window=60.0,
    c=SPEED_OF_SOUND,
) -> OneResult:
    """Locate one measured pass when the recording's start is known: `start` = (absolute start,
    uncertainty in s), as for the video's one-minute clip.

    A flight can only have made the pass if its delayed closest approach at a point falls when
    the pass was heard (start + event.t_heard, within the uncertainty plus `measure_tol` for the
    measurement), with a matching speed and distance. Its modelled curve is then laid at its own
    absolute time on the recording, sweeping the clock over the uncertainty, and scored. With
    `use_fit` False the speed and distance aren't checked: for a pass whose straight-line fit
    can't be trusted (a turning plane), the time alone narrows the flights down to a few.
    """
    return search_known_starts(spec, flights, observers, [event], start, passes=passes, measure_tol=measure_tol,
                               use_fit=use_fit, window=window, c=c)[0]


def _known_job(job):
    """Curve scores of one flight at the points where it could have made one event."""
    k, i, flight, idx, th = job
    w = _worker
    ev = w["events"][k]
    sc = score_event(w["spec"], flight, w["observers"][idx], w["t0"], th, ev.f0_grid(), w["shifts"],
                     window=w["window"], harmonics=ev.harmonics, c=w["c"])
    return k, i, idx, sc


def search_known_starts(spec, flights, observers, events, start, *, passes: Passes, measure_tol=8.0, use_fit=True,
                        window=60.0, c=SPEED_OF_SOUND, workers=1) -> list[OneResult]:
    """`search_known_start` for several events at once, the flights' scoring spread over `workers`
    processes (biggest first). `use_fit` is one for all events or one per event."""
    observers = np.asarray(observers, dtype=float)
    t0, tol = start
    n = int(np.ceil(tol / spec.hop))
    shifts = np.arange(-n, n + 1)  # frames; the true start is t0 + shift * hop
    fits = np.broadcast_to(np.asarray(use_fit, dtype=bool), (len(events),))
    jobs, heard = [], [0] * len(events)
    for k, event in enumerate(events):
        when, slack = t0 + event.t_heard, tol + measure_tol
        for i, f in enumerate(flights):
            if passes.idx[i].size == 0 or passes.t_max[i] < when - slack or passes.t_min[i] > when + slack:
                continue
            ok = np.abs(passes.th[i] - when) < slack
            heard[k] += int(ok.sum())
            if fits[k]:
                ok &= _within(passes.dist[i], event.dist_range) & _within(passes.speed[i], event.speed_range)
            if ok.any():
                jobs.append((k, i, f, passes.idx[i][ok], passes.th[i][ok]))
    # Scoring reads only the score image, not the dB one.
    light = Spectrogram(spec.t, spec.f, np.empty((0, 0), dtype=np.float32), spec.score)
    state = dict(spec=light, observers=observers, events=events, t0=t0, shifts=shifts, window=window, c=c)
    order = sorted(range(len(jobs)), key=lambda j: -jobs[j][3].size)
    done = {(k, i): (idx, sc) for k, i, idx, sc in _map(_known_job, [jobs[j] for j in order], workers, state)}
    out = []
    for k in range(len(events)):
        cols = {c_: [] for c_ in ("point", "flight", "score", "f0", "start")}
        for kk, i, *_ in jobs:  # in flight order, as when scored one by one
            if kk != k:
                continue
            idx, sc = done[(k, i)]
            cols["point"].append(idx)
            cols["flight"].append(np.full(idx.size, i))
            cols["score"].append(sc.score)
            cols["f0"].append(sc.f0)
            cols["start"].append(t0 + sc.shift * spec.hop)
        arrays = {c_: (np.concatenate(v) if v else np.zeros(0)) for c_, v in cols.items()}
        for c_ in ("point", "flight"):
            arrays[c_] = arrays[c_].astype(int)
        funnel = {"heard then": heard[k], "+ speed & distance": len(arrays["point"]),
                  "flights left": len(np.unique(arrays["flight"]))}
        out.append(OneResult(**arrays, funnel=funnel))
    return out


def refine_known_start(spec, track, centre, event, start, *, half_size=600.0, spacing=25.0, place=None,
                       ground=flat_ground, window=60.0, c=SPEED_OF_SOUND):
    """Re-score a fine grid around `centre` for one flight, the recording start known within its
    uncertainty. Returns (observers, EventScore); the start found is start[0] + shift * hop."""
    obs = observer_grid(centre, half_size, spacing, ground, place=place)
    return obs, _score_known_start(spec, track, obs, event, start, window, c)


def _score_known_start(spec, track, obs, event, start, window=60.0, c=SPEED_OF_SOUND):
    ca = closest_approach(track, obs, c)
    n = int(np.ceil(start[1] / spec.hop))
    return score_event(spec, track, obs, start[0], ca.t_heard, event.f0_grid(), np.arange(-n, n + 1),
                       window=window, harmonics=event.harmonics, c=c)


def _refine_job(job):
    kind, args = job
    spec = _worker["spec"]
    if kind == "pair":
        fa, fb, obs, e1, e2, shift_tol = args
        ca, cb = closest_approach(fa, obs), closest_approach(fb, obs)
        return score_pair(spec, fa, fb, obs, e1, e2, ca.t_heard, cb.t_heard, _shift_frames(spec, shift_tol))
    if kind == "curve":
        track, obs, t0, f0_grid, shifts = args
        return score_event(spec, track, obs, t0, closest_approach(track, obs).t_heard, f0_grid, shifts,
                           per_shift=True)
    track, obs, event, start = args
    return _score_known_start(spec, track, obs, event, start)


def refine_many(spec, jobs, workers=1) -> list:
    """Several refinements at once, spread over `workers` processes, results in order. Each job is
    ("pair", (fa, fb, observers, e1, e2, shift_tol)), scored as `refine_pair` does (giving (s1, s2)),
    or ("start", (track, observers, event, start)), as `refine_known_start` (giving its EventScore),
    or ("curve", (track, observers, t0, f0_grid, shifts)), one flight's curve scored at each shift
    (`score_event(per_shift=True)`, (P, nS)); the observers are a grid made beforehand (it needs the
    terrain)."""
    if not jobs:
        return []
    light = Spectrogram(spec.t, spec.f, np.empty((0, 0), dtype=np.float32), spec.score)
    order = sorted(range(len(jobs)), key=lambda j: -len(jobs[j][1][2 if jobs[j][0] == "pair" else 1]))  # biggest first
    out = _map(_refine_job, [jobs[j] for j in order], min(workers, len(jobs)), dict(spec=light))
    res = [None] * len(jobs)
    for j, r in zip(order, out):
        res[j] = r
    return res


# --- several detected events, unknown which two to use ------------------------------------


@dataclass
class EventPair:
    """The two-event search for events i and j (i heard first) of a list of measured events.

    `rows` holds the best point of each surviving flight pair (rows of `result`), strongest
    `evidence` first; see `search_event_pairs`.
    """

    i: int
    j: int
    result: PairResult
    rows: np.ndarray
    followed: np.ndarray  # (len(rows), 2) traced-line points each event's flight curve follows
    evidence: np.ndarray  # (len(rows),) followed points above the typical candidate's, both events
    anchored: bool = False  # found by `wide.search_anchored_wide`: its rows are already placed, and i is the straight pass

    @property
    def best(self) -> float:
        return float(self.evidence[0]) if len(self.evidence) else -np.inf


def search_event_pairs(
    spec: Spectrogram,
    flights: list[FlightTrack],
    observers,
    events: list[Event],
    duration: float,
    *,
    audible_dist=10000.0,
    tol_hz=1.5,
    c=SPEED_OF_SOUND,
    workers=1,
    progress=None,
    passes: Passes | None = None,
    **kw,
) -> list[EventPair]:
    """Run `search_two_events` for every pair of `events`, sharing the closest approaches, and
    rank the pairs against each other. Returned best pair first.

    Curve scores rank flight pairs within one pair of events but do not compare across pairs: a
    loud event scores high whichever flight explains it. So pairs are compared by evidence: how
    many points of each event's traced line (`Event.line`) a flight's modelled curve follows
    within `tol_hz`, minus what the median candidate flight for that event follows (points any
    flight would follow, like a flat stretch, prove nothing), summed over both events.

    Events should first pass `measure.implausible`: one that is not a pass at all (a tone, say)
    can still have a long line that many flights' curves follow. `progress(done, total)` is
    called after the closest approaches (done = 0) and after each pair. `kw` goes to
    `search_two_events`.
    """
    observers = np.asarray(observers, dtype=float)
    if passes is None:
        passes = Passes(flights, observers, audible_dist, c, workers)
    order = np.argsort([ev.t_heard for ev in events], kind="stable")
    todo = [(i, j) for n, i in enumerate(order) for j in order[n + 1 :]]
    if progress:
        progress(0, len(todo))
    out = []
    for i, j in todo:
        e1, e2 = events[i], events[j]
        res = search_two_events(spec, flights, observers, e1, e2, duration, c=c, workers=workers, passes=passes,
                                **kw)
        rows = res.best_per_pair()
        followed = np.zeros((len(rows), 2))
        for k, r in enumerate(rows):
            obs = observers[res.point[r]][None]
            for col, (fl, f0, ev) in enumerate(((res.a[r], res.f0_1[r], e1), (res.b[r], res.f0_2[r], e2))):
                share = line_agreement(flights[fl], obs, res.t0[r : r + 1], [f0], ev, tol_hz=tol_hz, c=c)[0]
                followed[k, col] = share * len(ev.line[0])
        evidence = (followed - np.median(followed, axis=0)).sum(1) if len(rows) else np.zeros(0)
        rank = np.argsort(-evidence, kind="stable")
        out.append(EventPair(int(i), int(j), res, rows[rank], followed[rank], evidence[rank]))
        if progress:
            progress(len(out), len(todo))
    out.sort(key=lambda ep: -ep.best)
    return out


# --- does one place and time explain all the passes? ---------------------------------------


@dataclass
class Explanation:
    """The flight that best explains one pass, for a given place and recording start."""

    flight: int  # index into the flights, -1 when none fits
    share: float  # fraction of the pass's traced line its modelled curve follows
    f0: float  # emitted frequency that fits best, Hz
    shift: float = 0.0  # clock jitter that fits best, s (the start was t0 + shift)


def explain_passes(flights, observer, t0, events, *, max_dist=10000.0, tol_hz=1.5, f0_tol=0.15, change_ratio=0.5,
                   shifts_s=(-2.0, -1.0, 0.0, 1.0, 2.0), c=SPEED_OF_SOUND) -> list[Explanation]:
    """For a candidate microphone position (3,) and absolute recording start `t0`, find for each
    event the flight whose modelled curve follows its traced line (`Event.line`) best.

    The right place and time should explain most real passes, not only the two a candidate was
    found from; a wrong one rarely explains more than those two. The emitted frequency is fitted
    per flight (within `f0_tol` of the pass's own; one value or one per event), and a flight's curve must change pitch by
    between `change_ratio` and 1/`change_ratio` times as much as the line does, so that a distant
    plane's nearly flat curve cannot claim a steep pass by matching its flat ends. Small clock
    jitter (`shifts_s`) is allowed since the start is only known to the spectrogram's hop.
    """
    obs = np.asarray(observer, dtype=float).reshape(3)
    f0_tols = np.broadcast_to(np.asarray(f0_tol, dtype=float), (len(events),))
    out = []
    for ev, f0_tol in zip(events, f0_tols):
        times, freqs = (np.asarray(x, dtype=float) for x in ev.line)
        t_rx = t0 + times
        lo, hi = t_rx.min() - 60.0, t_rx.max() + 60.0
        line_change = np.log(freqs.max() / freqs.min())
        best = Explanation(-1, 0.0, float("nan"))
        for i, tr in enumerate(flights):
            if tr.t_end < lo or tr.t_start > hi:
                continue
            ts = np.linspace(max(tr.t_start, lo), min(tr.t_end, hi), 24)
            if np.linalg.norm(tr.position(ts) - obs, axis=1).min() > max_dist:
                continue
            for shift in shifts_s:
                g = doppler_factor(tr, obs, t_rx + shift, c)
                ok = np.isfinite(g)
                if ok.sum() < 0.5 * len(g):
                    continue
                f0 = float(np.median(freqs[ok] / g[ok]))
                if abs(f0 / ev.f0 - 1.0) > f0_tol:
                    continue
                change = np.log(g[ok].max() / g[ok].min())
                if line_change > 0.03 and not change_ratio <= change / line_change <= 1.0 / change_ratio:
                    continue
                share = float(np.mean(ok & (np.abs(f0 * g - freqs) < tol_hz)))
                if share > best.share:
                    best = Explanation(i, share, f0, shift)
        out.append(best)
    return out


# --- every pass votes for the places and start times that explain it ----------------------


@dataclass
class Votes:
    """Places and recording starts ranked by how many passes they explain (see `vote_passes`)."""

    point: np.ndarray  # (K,) grid index, best first
    start: np.ndarray  # (K,) absolute recording start, s
    votes: np.ndarray  # (K,) summed shares of the passes explained there, one per pass and per flight
    passes: list[np.ndarray]  # for each, the indices of the events that voted
    flights: list[np.ndarray]  # and the flight that explains each
    seconds: float = 0.0
    # For every event, voting or not, its best match anywhere: (share, flight, point, start), or None
    # when no flight follows `min_share` of its line at any point.
    best: list = field(default_factory=list)


def _vote_job(k):
    """(point, flight, share, start) for every place and flight that follows event k's line."""
    w = _worker
    ev, flights, observers, passes = w["events"][k], w["flights"], w["observers"], w["passes"]
    t0, tol = w["start"]
    when, slack = t0 + ev.t_heard, tol + w["measure_tol"][k]
    times, freqs = (np.asarray(x, dtype=float) for x in ev.line)
    pick = np.unique(np.linspace(0, len(times) - 1, min(len(times), w["line_points"])).round().astype(int))
    times, freqs = times[pick], freqs[pick]
    line_change = np.log(freqs.max() / freqs.min())
    jitter = np.asarray(w["jitter_s"], dtype=float)
    cols = {c: [] for c in ("point", "flight", "share", "start")}
    for i, f in enumerate(flights):
        if passes.idx[i].size == 0 or passes.t_max[i] < when - slack or passes.t_min[i] > when + slack:
            continue
        ok = np.abs(passes.th[i] - when) < slack
        if w["use_fit"][k]:
            ok &= _within(passes.dist[i], ev.dist_range) & _within(passes.speed[i], ev.speed_range)
        if not ok.any():
            continue
        idx, th = passes.idx[i][ok], passes.th[i][ok]
        # If this flight made the pass here, its delayed CPA is when the pass was heard: that fixes
        # the start, up to how well t_heard was measured.
        implied = th - ev.t_heard
        for s in range(0, idx.size, 512):
            sl = slice(s, s + 512)
            starts = implied[sl, None] + jitter  # (n, J)
            g = doppler_factor(f, observers[idx[sl], None, None, :], starts[:, :, None] + times, w["c"])  # (n, J, L)
            fin = np.isfinite(g)
            gm = np.where(fin, g, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows: curves off the track's span
                f0 = np.nanmedian(freqs / gm, axis=2)  # (n, J)
                change = np.log(np.nanmax(gm, axis=2) / np.nanmin(gm, axis=2))
                sane = (fin.mean(2) > 0.5) & (np.abs(f0 / ev.f0 - 1.0) < w["f0_tol"][k])
                sane &= (change >= w["change_ratio"] * line_change) & (change <= line_change / w["change_ratio"])
                sane &= np.abs(starts - t0) <= tol
            share = np.where(sane, np.mean(fin & (np.abs(f0[:, :, None] * g - freqs) < w["tol_hz"]), axis=2), 0.0)
            j = share.argmax(1)
            best = share[np.arange(len(j)), j]
            keep = best >= w["min_share"]
            cols["point"].append(idx[sl][keep])
            cols["flight"].append(np.full(keep.sum(), i))
            cols["share"].append(best[keep])
            cols["start"].append(starts[np.arange(len(j)), j][keep])
    return {c: (np.concatenate(v) if v else np.zeros(0)) for c, v in cols.items()}


def vote_passes(
    flights: list[FlightTrack],
    observers,
    events: list[Event],
    start: tuple[float, float],
    *,
    passes: Passes,
    voting=None,
    use_fit=True,
    min_share=0.5,
    start_bin_s=4.0,
    measure_tol=8.0,
    jitter_s=tuple(np.arange(-10.0, 10.1, 2.0)),
    tol_hz=1.5,
    f0_tol=0.15,
    change_ratio=0.5,
    line_points=48,
    n_best=200,
    c=SPEED_OF_SOUND,
    workers=1,
) -> Votes:
    """Every event votes for each grid point and recording start at which some flight's modelled
    curve follows its traced line, and the places with most votes win.

    Detected passes include birds, vehicles and aircraft without ADS-B, which no audio test tells
    apart from real ones. They need not be weeded out first: a false pass votes for scattered,
    unrelated places, while every real pass votes for the true place and the same start. So all
    candidate passes can vote, not just a hand-picked two.

    `start` = (absolute recording start, uncertainty in s). For each event, a flight can only
    explain it at points where its delayed closest approach is heard when the pass is (`passes`
    gives those) and, with `use_fit`, where its speed and distance there are within the event's
    ranges (`Event.speed_range`, `dist_range`; widen them for passes whose fit is doubtful). That
    also fixes the start at each point; the curve is then compared with the line over
    `jitter_s` around that start, fitting the emitted pitch, as `explain_passes` does. A vote
    is the share of the line followed (at least `min_share`). Votes at one point count together
    when their starts agree within `start_bin_s`; each event counts once there and so does each
    flight, so that one aircraft's harmonics or a line traced twice add nothing.

    `use_fit`, `f0_tol` and `measure_tol` can be given per event, to trust less the fitted numbers of
    a pass that is not a straight one (a turning plane's speed, distance, pitch and time heard).

    `voting` (bool per event, default all) leaves some events out of the count; every event still
    gets its best match anywhere (`Votes.best`), which says whether any flight could have made
    that pass at all, turning or not, without trusting a straight-pass fit.
    """
    t_begin = time.time()
    observers = np.asarray(observers, dtype=float)

    def per_event(x, dtype=float):
        return np.broadcast_to(np.asarray(x, dtype=dtype), (len(events),))

    state = dict(events=events, flights=flights, observers=observers, passes=passes, start=start,
                 use_fit=per_event(use_fit, bool), measure_tol=per_event(measure_tol), jitter_s=jitter_s, tol_hz=tol_hz,
                 f0_tol=per_event(f0_tol), change_ratio=change_ratio,
                 line_points=line_points, min_share=min_share, c=c)
    res = _map(_vote_job, range(len(events)), workers, state)
    best = []
    for r in res:
        b = int(np.argmax(r["share"])) if r["share"].size else None
        best.append(None if b is None else (float(r["share"][b]), int(r["flight"][b]), int(r["point"][b]),
                                            float(r["start"][b])))
    vote = np.ones(len(events), bool) if voting is None else np.asarray(voting, bool)
    res = [r if vote[k] else {c: np.zeros(0) for c in r} for k, r in enumerate(res)]
    ev = np.concatenate([np.full(r["point"].size, k) for k, r in enumerate(res)]).astype(int)
    point, flight, share, st = (np.concatenate([r[c] for r in res]) for c in ("point", "flight", "share", "start"))
    point, flight = point.astype(int), flight.astype(int)
    empty = Votes(np.zeros(0, int), np.zeros(0), np.zeros(0), [], [], time.time() - t_begin, best)
    if not point.size:
        return empty

    # A start near a bin edge belongs as much to the next bin: count every vote in its own bin and
    # the nearer neighbour.
    origin = start[0] - start[1] - start_bin_s
    pos = (st - origin) / start_bin_s
    b0 = np.floor(pos).astype(int)
    b1 = np.where(pos - b0 >= 0.5, b0 + 1, b0 - 1)
    nb = int(np.ceil((2 * start[1] + 2 * start_bin_s) / start_bin_s)) + 2
    key = np.concatenate([point * nb + b0, point * nb + b1])
    ev, flight, share, st = (np.concatenate([x, x]) for x in (ev, flight, share, st))

    def best_of(groups, key, share):
        """Index of the largest share for each (key, group)."""
        order = np.lexsort((-share, groups, key))
        g = groups[order] + key[order] * (groups.max() + 1)
        first = np.r_[True, g[1:] != g[:-1]]
        return order[first]

    # One vote per event at each (point, start bin), then one per flight.
    keep = best_of(ev, key, share)
    keep = keep[best_of(flight[keep], key[keep], share[keep])]
    key, ev, flight, share, st = key[keep], ev[keep], flight[keep], share[keep], st[keep]
    uniq, inv = np.unique(key, return_inverse=True)
    total = np.bincount(inv, weights=share)
    top = np.argsort(-total, kind="stable")[:n_best]
    # Starts: the mean of the votes' own.
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_start = np.bincount(inv, weights=share * st) / total
    groups = [np.flatnonzero(inv == t) for t in top]
    return Votes(point=(uniq[top] // nb).astype(int), start=mean_start[top], votes=total[top],
                 passes=[ev[g] for g in groups], flights=[flight[g] for g in groups],
                 seconds=time.time() - t_begin, best=best)
