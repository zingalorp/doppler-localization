"""Find aircraft passes on a spectrogram without help: match a bank of flyby curve shapes.

On a log-frequency axis the emitted frequency f0 only moves a straight-line flyby's curve up or
down, and the time of closest approach only moves it sideways. So for each shape (speed, CPA
distance) one 2-D cross-correlation of the image with that curve scores every (time, f0) at
once, and the best-scoring shapes and places are the candidate passes.

Steady tones and broadband bursts would dominate such sums on a real recording, so the image is
first flattened: a running median along time removes anything that stays at one frequency for
much of half a minute, and a median across frequency removes anything that covers a wide band at
one moment. A pass's curve, which sweeps through each frequency in seconds, survives both.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import heapq
import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.ndimage import gaussian_filter, maximum_filter

from .measure import Event, TurnFit, fit_flyby, fit_turning, implausible, straight_flyby, trace_ridge, trim_tails
from .physics import SPEED_OF_SOUND
from .spectro import Spectrogram


@dataclass
class Detection:
    """A candidate pass: the best-matching flyby shape at one place on the recording."""

    t_heard: float  # recording time at which the heard pitch equals f0, s
    f0: float  # Hz
    speed: float  # m/s
    dist: float  # m
    strength: float  # how far the sum along the curve stands above noise, in standard deviations
    t_range: tuple[float, float]  # the stretch of recording the matched curve covers
    harmonics: list[float] = field(default_factory=list)  # weaker curves found at these multiples of f0

    def curve(self, t, c=SPEED_OF_SOUND):
        return straight_flyby(t, self.f0, self.speed, self.dist, self.t_heard, c)

    def guide(self, n=12, c=SPEED_OF_SOUND):
        """(t, f) points along the matched curve, in the form `trace_ridge` takes."""
        t = np.linspace(*self.t_range, n)
        return list(zip(t.tolist(), self.curve(t, c).tolist()))


def _running_median(x, width, step, axis, rows=128, workers=1):
    """Median over `width` samples along `axis`, computed every `step` samples and interpolated.
    Blocks of `rows` go to `workers` threads (NumPy's partition runs without the GIL)."""
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    half = width // 2
    centres = np.unique(np.append(np.arange(0, n, step), n - 1))
    out = np.empty_like(x)

    def block(s):
        padded = np.pad(x[s : s + rows], ((0, 0), (half, half)), mode="reflect")
        med = np.median(sliding_window_view(padded, width, axis=-1)[:, centres], axis=-1)
        for r, m in enumerate(med):
            out[s + r] = np.interp(np.arange(n), centres, m)

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(block, range(0, len(x), rows)))
    return np.moveaxis(out, -1, axis)


def flatten(spec: Spectrogram, *, time_s=30.0, band_hz=20.0, clip_db=15.0, workers=1) -> np.ndarray:
    """dB above the local background along both time and frequency, clipped to [0, clip_db]."""
    db = spec.db.astype(np.float32)
    w_t = int(round(time_s / spec.hop)) | 1
    resid = db - _running_median(db, w_t, max(w_t // 12, 1), axis=1, workers=workers)
    w_f = int(round(band_hz / spec.df)) | 1
    resid = resid - _running_median(resid, w_f, max(w_f // 8, 1), axis=0, workers=workers)
    return np.clip(gaussian_filter(resid, (0.7, 0.5)), 0.0, clip_db)


def log_frequency(image, f, fmin, fmax, step):
    """Resample (F, T) rows onto log-spaced frequencies, keeping the brightest bin in each."""
    y = np.exp(np.arange(np.log(fmin), np.log(fmax), step))
    edges = np.exp(np.log(y) + step / 2)
    hi = np.searchsorted(f, edges, side="right")
    lo = np.concatenate([[np.searchsorted(f, fmin * np.exp(-step / 2))], hi[:-1]])
    nearest = np.clip(np.searchsorted(f, y), 0, len(f) - 1)
    out = np.empty((len(y), image.shape[1]), dtype=np.float32)
    for k in range(len(y)):
        out[k] = image[lo[k] : hi[k]].max(0) if hi[k] > lo[k] else image[nearest[k]]
    return y, out


def shape_bank(speeds, dists, hop, log_step, *, max_half_s=90.0, c=SPEED_OF_SOUND):
    """Each flyby shape as (speed, dist, frame offsets, log-frequency bin offsets).

    A shape covers the part of the pass where the pitch is still changing: from 10% to 90% of its
    total fall in log-frequency, widened by half on each side, which on a quiet background is all
    of the curve worth summing.
    """
    bank = []
    for v in speeds:
        for d in dists:
            tau = np.arange(-max_half_s, max_half_s + hop / 2, hop)
            y = np.log(straight_flyby(tau, 1.0, v, d, 0.0, c))
            top, bottom = np.log(c / (c - v)), np.log(c / (c + v))
            frac = (top - y) / (top - bottom)
            a, b = tau[np.argmax(frac > 0.1)], tau[len(frac) - 1 - np.argmax(frac[::-1] < 0.9)]
            keep = (tau >= a - (b - a) / 2) & (tau <= b + (b - a) / 2)
            frames = np.rint(tau[keep] / hop).astype(int)
            bins = np.rint(y[keep] / log_step).astype(int)
            bank.append((v, d, frames, bins))
    return bank


def detect(
    spec: Spectrogram,
    *,
    fmin=40.0,
    fmax=600.0,
    log_step=0.004,
    speeds=None,
    dists=None,
    min_strength=10.0,
    max_events=30,
    claim=0.02,
    c=SPEED_OF_SOUND,
    workers=-1,
) -> list[Detection]:
    """Candidate passes on `spec`, strongest first, using `workers` threads (-1: every core)."""
    speeds = np.geomspace(40.0, 240.0, 12) if speeds is None else np.asarray(speeds)
    dists = np.geomspace(300.0, 12000.0, 14) if dists is None else np.asarray(dists)
    threads = (os.cpu_count() or 1) if workers == -1 else max(1, workers)
    y, img = log_frequency(flatten(spec, workers=threads), spec.f, fmin, fmax, log_step)
    # In noise units, so that a sum over n pixels has spread sqrt(n) whatever the shape's length.
    img = (img - img.mean()) / img.std()
    Y, T = img.shape
    bank = shape_bank(speeds, dists, spec.hop, log_step, c=c)

    # Correlate via FFT: pad so that no shape wraps around the edges.
    reach_t = max(np.abs(fr).max() for *_, fr, _ in bank)
    reach_y = max(np.abs(b).max() for *_, b in bank)
    shape = (next_fast_len(Y + 2 * reach_y + 1), next_fast_len(T + 2 * reach_t + 1))
    spectrum = rfft2(img, shape, workers=threads)
    best = np.full((Y, T), -np.inf, dtype=np.float32)
    which = np.zeros((Y, T), dtype=np.int16)
    better = np.empty((Y, T), dtype=bool)
    for k, (v, d, frames, bins) in enumerate(bank):
        kernel = np.zeros(shape, dtype=np.float32)
        np.add.at(kernel, ((-bins) % shape[0], (-frames) % shape[1]), 1.0 / np.sqrt(len(frames)))
        kernel = rfft2(kernel, workers=threads, overwrite_x=True)
        kernel *= spectrum
        corr = irfft2(kernel, shape, workers=threads, overwrite_x=True)[:Y, :T]
        np.greater(corr, best, out=better)  # in place: boolean indexing over every pixel took 8 s of 55
        np.maximum(best, corr, out=best)
        np.copyto(which, k, where=better)

    # One pass also matches neighbouring shapes and places (speed, distance and timing trade off
    # along a partial curve), and its harmonics match too. So take local maxima strongest first,
    # and once a curve is accepted its pixels are used up: every later candidate is re-scored
    # without them, and kept only if it still stands out on pixels of its own.
    peaks = (best == maximum_filter(best, size=(int(0.02 / log_step) | 1, int(5 / spec.hop) | 1))) & (
        best >= min_strength
    )
    yi, ti = np.nonzero(peaks)
    heap = [(-float(best[a, b]), int(a), int(b), int(which[a, b])) for a, b in zip(yi, ti)]
    heapq.heapify(heap)
    used = np.zeros((Y, T), dtype=bool)
    band = int(round(claim / log_step))
    out = []
    while heap and len(out) < max_events:
        neg, a, b, k = heapq.heappop(heap)
        v, d, frames, bins = bank[k]
        rows, cols = a + bins, b + frames
        inside = (rows >= 0) & (rows < Y) & (cols >= 0) & (cols < T)
        rows, cols = rows[inside], cols[inside]
        score = float(np.where(used[rows, cols], 0.0, img[rows, cols]).sum() / np.sqrt(len(frames)))
        if score < min_strength:
            continue
        if heap and score < -heap[0][0] - 1e-6:
            heapq.heappush(heap, (-score, a, b, k))  # no longer the best: try again in its new place
            continue
        for off in range(-band, band + 1):
            r = rows + off
            ok = (r >= 0) & (r < Y)
            used[r[ok], cols[ok]] = True
        t_heard = float(spec.t[b])
        det = Detection(
            t_heard=t_heard, f0=float(y[a]), speed=float(v), dist=float(d), strength=score,
            t_range=(t_heard + frames[0] * spec.hop, t_heard + frames[-1] * spec.hop),
        )
        for kept in out:
            ratio = _same_pass(kept, det, c)
            if ratio is not None:
                if ratio != 1 and ratio not in kept.harmonics:
                    kept.harmonics.append(ratio)
                break
        else:
            out.append(det)
    return out


def close_to_found(spec: Spectrogram, lines, *, mask_hz=1.3, min_strength=28.0, near=0.15, ridge_hz=3.0, ridge=0.03,
                   min_apart_s=10.0, max_touching=3.0, max_events=15, c=SPEED_OF_SOUND, workers=-1):
    """Curves hidden by others already found, which `detect` can't see: it treats what lies close to an
    accepted curve as used, since one pass also matches shapes beside it. So look again with only the
    traced `lines` ((times, freqs) each) taken away. That also turns up what's left of those lines, so
    a new line must be a curve of its own: not at a constant ratio to a known line (the same plane), and
    beside one in time, within `near` of its pitch, further from it than a ridge is wide for at least
    `min_apart_s`, without following it for over `max_touching` times that. Returns (Detection, Event,
    index of that line) for each."""
    score = spec.score.copy()
    for t, f in lines:
        frames = np.clip(np.searchsorted(spec.t, t), 0, len(spec.t) - 1)
        for k, frame in zip(np.asarray(f, dtype=float), frames):
            a, b = np.searchsorted(spec.f, [k - mask_hz, k + mask_hz])
            score[a:b, frame] = 0.0
    rest = Spectrogram(spec.t, spec.f, spec.db, score)
    known, out = [(np.asarray(t), np.asarray(f)) for t, f in lines], []
    for det in detect(rest, max_events=max_events, min_strength=min_strength, c=c, workers=workers):
        try:
            ev = measure(rest, det, c=c)
        except ValueError:  # too little of it left to fit
            continue
        t, f = np.asarray(ev.line[0]), np.asarray(ev.line[1])
        if len(t) < 10 or any(len(known) in group for group in harmonic_groups([*known, (t, f)], hop=spec.hop)):
            continue
        best, neighbour = (0.0, 0.0), None
        for i, (lt, lf) in enumerate(known[:len(lines)]):
            inside = (t >= lt[0]) & (t <= lt[-1])
            there = np.interp(t[inside], lt, lf)
            gap, wide = np.abs(f[inside] - there), np.maximum(ridge_hz, ridge * there)
            apart = ((gap >= wide) & (gap <= near * there)).sum() * spec.hop
            if apart > best[0]:
                best, neighbour = (apart, (gap < wide).sum() * spec.hop), i
        if best[0] >= min_apart_s and best[1] <= max_touching * best[0]:
            out.append((det, ev, neighbour))
            known.append((t, f))
    return out


def _same_pass(a: Detection, b: Detection, c, *, tol=0.03, ratios=(1, 2, 3, 4, 1 / 2, 1 / 3, 1 / 4)):
    """The ratio r for which b's curve is r times a's over most of their common stretch, else None."""
    lo, hi = max(a.t_range[0], b.t_range[0]), min(a.t_range[1], b.t_range[1])
    shorter = min(np.ptp(a.t_range), np.ptp(b.t_range))
    if hi - lo < 0.5 * shorter:
        return None
    t = np.linspace(lo, hi, 50)
    gap = np.log(b.curve(t, c)) - np.log(a.curve(t, c))
    for r in ratios:
        if np.median(np.abs(gap - np.log(r))) < tol:
            return r
    return None


def measure(spec: Spectrogram, det: Detection, *, band_hz=6.0, iters=3, c=SPEED_OF_SOUND) -> Event:
    """Trace the line near a detection's curve and fit a flyby to it, as for a hand-guided trace.

    The detected shape is only the nearest in a coarse bank, so the trace is repeated along each
    new fit, over the stretch where that fit's pitch is still changing.
    """
    f0, speed, dist, t_heard, t_range = det.f0, det.speed, det.dist, det.t_heard, det.t_range
    for _ in range(iters):
        t = np.linspace(*t_range, 15)
        f = straight_flyby(t, f0, speed, dist, t_heard, c)
        line = trace_ridge(spec, t_range, (f.min() - 2 * band_hz, f.max() + 2 * band_hz), guide=list(zip(t, f)),
                           band_hz=band_hz, falling=True)
        ev = fit_flyby(*line, c=c)
        f0, speed, dist, t_heard = ev.f0, ev.speed, ev.dist, ev.t_heard
        t_range = _changing(speed, dist, t_heard, spec.hop, c)
    times, freqs = ev.line
    frames = np.clip(np.searchsorted(spec.t, times), 0, len(spec.t) - 1)
    a, b = trim_tails(times, freqs, spec.score[np.clip(np.searchsorted(spec.f, freqs), 0, len(spec.f) - 1), frames], c=c)
    return fit_flyby(times[a:b], freqs[a:b], c=c) if b - a < len(times) else ev


def _changing(speed, dist, t_heard, hop, c, *, lo=0.05, hi=0.95, margin=0.25, max_half_s=150.0):
    """Recording times over which a flyby's pitch makes the middle lo..hi of its fall, plus a margin."""
    tau = np.arange(-max_half_s, max_half_s + hop / 2, hop)
    y = np.log(straight_flyby(tau, 1.0, speed, dist, 0.0, c))
    top, bottom = np.log(c / (c - speed)), np.log(c / (c + speed))
    frac = (top - y) / (top - bottom)
    a, b = tau[np.argmax(frac > lo)], tau[len(frac) - 1 - np.argmax(frac[::-1] < hi)]
    return t_heard + a - margin * (b - a), t_heard + b + margin * (b - a)


@dataclass
class Assessment:
    """How much a measured pass looks like an aircraft the search can explain, for a person to review."""

    kind: str  # "likely aircraft", "possible", "possibly turning" or "not a pass"
    confidence: float  # 0..1
    notes: list[str]
    speed_tol: float  # filter tolerances to search it with
    dist_tol: float
    turn: TurnFit | None = None  # a turning flight that follows the line better than any straight pass


def assess(event: Event, strength: float | None = None, harmonics=(), *, max_dist=10000.0, tol_hz=1.5, good_fit=0.75,
           turns=True, min_gain=0.1, c=SPEED_OF_SOUND) -> Assessment:
    """Classify a measured pass.

    How well the straight-line fit follows the traced line decides how far its numbers can be
    trusted:
    - a good fit (at least `good_fit` of the line within `tol_hz`): its distance and speed are
      checked against what the search can explain (`measure.implausible`), giving "not a pass"
      if beyond the area searched or too fast, else "likely aircraft" or "possible" by
      confidence = fit share x detection strength (out of 40), +0.15 for harmonics, which
      propellers make. Strength is missing for passes traced by hand.
    - a poorer fit: real planes turn, climb and change speed, which bends the line away from any
      straight pass. Scoring models each flight's real track, so such a line is kept for a
      person to judge ("possibly turning" when it is strong), searched with wider tolerances
      since its fitted speed and distance are uncertain. With `turns`, a flight at constant speed
      and turn rate is fitted too (`measure.fit_turning`); when it follows at least `min_gain` more
      of the line, it is kept (`Assessment.turn`), to draw, and its share is the fit that counts.
    """
    times, freqs = event.line
    fit = float(np.mean(np.abs(straight_flyby(times, event.f0, event.speed, event.dist, event.t_heard, c) - freqs)
                        < tol_hz))
    notes = [f"{fit:.0%} of the traced line within {tol_hz} Hz of a straight pass"]
    turn = None
    if turns and fit < good_fit and len(times) >= 20:
        turn = fit_turning(times, freqs, event.f0, tol_hz=tol_hz, c=c)
        if turn.share >= fit + min_gain:
            notes.append(f"{turn.share:.0%} within {tol_hz} Hz of a turning flight "
                         f"({abs(np.degrees(turn.params[2])):.1f} deg/s), emitting {turn.params[0]:.1f} Hz")
            fit = turn.share
        else:
            turn = None
    loud = 1.0 if strength is None else min(strength / 40.0, 1.0)
    if strength is not None:
        notes.append(f"detection strength {strength:.0f}")
    confidence = fit * loud
    if harmonics:
        confidence = min(confidence + 0.15, 1.0)
        notes.append("harmonics at x" + ", ".join(f"{h:.3g}" for h in harmonics))
    if fit < good_fit or turn is not None:
        notes.append("not a straight pass: speed and distance uncertain, searched with wider tolerances")
        return Assessment("possibly turning" if loud >= 0.6 else "possible", confidence, notes, 0.6, 0.9, turn)
    why = implausible(event, max_dist=max_dist, tol_hz=tol_hz, min_fit=0.0, c=c)
    if why:
        return Assessment("not a pass", 0.0, [why] + notes, event.speed_tol, event.dist_tol)
    return Assessment("likely aircraft" if confidence >= 0.6 else "possible", confidence, notes, 0.4, 0.7)


def harmonic_groups(lines, order=None, *, hop=0.5, max_spread=0.006, min_overlap_s=20.0, min_change=0.05
                    ) -> list[list[int]]:
    """Group traced lines that one aircraft made, as lists of indices into `lines` ((times, freqs) each),
    main line first.

    Every harmonic of a source is shifted by the same Doppler factor, so two lines from one
    aircraft keep a constant pitch ratio all along their common stretch, whatever the ratio
    (2, but also 3/2 or 5/4 when the fundamental itself is faint; 1 when two passes traced the
    same line). Two lines match when, over at least `min_overlap_s` together, their log-pitch
    difference wanders by less than `max_spread` (median absolute deviation; 0.6%) while each
    changes pitch by more than `min_change`, so that steady tones are not taken for one aircraft.

    Lines are taken in `order` (most trusted first; default as given): each one not yet grouped
    becomes a main line and collects the remaining lines that match it directly. Matching only
    through a third line is not enough, since a chain can link two aircraft.
    """
    order = list(range(len(lines))) if order is None else list(order)
    frames = [np.rint(np.asarray(t) / hop).astype(int) for t, _ in lines]

    def match(a, b):
        _, ia, ib = np.intersect1d(frames[a], frames[b], return_indices=True)
        if len(ia) * hop < min_overlap_s:
            return False
        fa, fb = np.asarray(lines[a][1])[ia], np.asarray(lines[b][1])[ib]
        if min(np.log(fa.max() / fa.min()), np.log(fb.max() / fb.min())) < min_change:
            return False
        gap = np.log(fb) - np.log(fa)
        return np.median(np.abs(gap - np.median(gap))) < max_spread

    taken, groups = set(), []
    for main in order:
        if main in taken:
            continue
        group = [main] + [b for b in order if b != main and b not in taken and match(main, b)]
        taken.update(group)
        if len(group) > 1:
            groups.append(group)
    return groups
