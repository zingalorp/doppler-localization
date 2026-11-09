"""Score candidate listening positions by laying modelled Doppler curves on the spectrogram."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flight import FlightTrack
from .measure import Event
from .physics import SPEED_OF_SOUND, doppler_factor
from .spectro import Spectrogram


_padded = {}  # the last padded score image: {"score": source, "border": L, "image": padded}


def _padded_score(score, L):
    """`score` with a zero border at least L frames wide either side and a zero row below, and that
    border's width. Kept for the next call: copying a 3-hour spectrogram takes ~0.1 s, and a search
    scores hundreds of flights against the same one. The source is held, so its id can't be reused."""
    c = _padded
    if c.get("score") is not score or c["border"] < L:
        c.clear()
        F, T = score.shape
        image = np.zeros((F + 1, T + 2 * L), dtype=np.float32)
        image[:F, L:L + T] = score
        c.update(score=score, border=L, image=image)
    return c["image"], c["border"]


@dataclass
class EventScore:
    score: np.ndarray  # mean spectrogram score under the modelled curve (dB above median)
    f0: np.ndarray  # best emitted frequency, Hz
    shift: np.ndarray  # best correction to t0, in spectrogram frames


def score_event(
    spec: Spectrogram,
    track: FlightTrack,
    observers,
    t0,
    t_heard,
    f0_grid,
    shifts,
    *,
    window=60.0,
    harmonics=(1,),
    c=SPEED_OF_SOUND,
    chunk=1024,
    per_shift=False,
) -> EventScore | np.ndarray:
    """How well does `track`'s modelled Doppler curve at each observer sit on the spectrogram?

    t0: (P,) absolute time taken as recording time 0 for each observer.
    t_heard: (P,) absolute delayed-CPA time of `track` at each observer; the curve is scored
        for `window` seconds either side of it.
    f0_grid: emitted frequencies to try (the measurement of f0 has error).
    shifts: (nS,) or (P, nS) integer frame corrections to t0 to try. The true recording start
        is t0 + shift * hop.

    The curve itself does not depend on t0, only where it lands on the recording does, so it is
    computed once per observer and the shifts just move where it is read from.

    With `per_shift`, returns instead the (P, nS) best score over f0 at each shift, for adding up
    several flights' scores at one shared start.
    """
    obs = np.atleast_2d(np.asarray(observers, dtype=float))
    P = len(obs)
    t0 = np.broadcast_to(np.asarray(t0, dtype=float), (P,))
    t_heard = np.broadcast_to(np.asarray(t_heard, dtype=float), (P,))
    shifts = np.asarray(shifts, dtype=int)
    if shifts.ndim == 1:
        shifts = np.broadcast_to(shifts, (P, len(shifts)))
    f0_grid = np.asarray(f0_grid, dtype=float)
    harmonics = np.asarray(harmonics, dtype=float)

    hop, T, F = spec.hop, len(spec.t), len(spec.f)
    m = np.arange(-int(round(window / hop)), int(round(window / hop)) + 1)
    out_score = np.full(P, -np.inf)
    out_f0 = np.zeros(P)
    out_shift = np.zeros(P, dtype=int)
    out_per_shift = np.full((P, shifts.shape[1]), -np.inf) if per_shift else None

    # Every shift reads the same frequency bins, just L frames apart, so read L-frame runs of
    # the spectrogram at once. A zero border L frames wide catches runs that fall off the
    # recording, and a zero row catches frequencies out of band.
    s_hi = shifts.max(1)
    L = int((s_hi - shifts.min(1)).max()) + 1
    padded, border = _padded_score(spec.score, L)  # border >= L frames of zeros either side
    runs = np.lib.stride_tricks.sliding_window_view(padded, L, axis=1)  # (F + 1, T + 2 * border - L + 1, L)
    HQ = len(harmonics) * len(f0_grid)
    chunk = max(1, min(chunk, 2**18 // (HQ * len(m))))  # ~64 points' curves: small arrays read faster
    # Points summed at a time: their (H, Q, n, L) sums stay ~256 kB, in the core's own cache. Reading
    # the runs is limited by memory bandwidth, not arithmetic: building all of (H, Q, n, M, L) and then
    # summing it made 8 workers each run 4x slower than one alone; this is ~9x faster with 8 at once.
    piece = max(1, 2**16 // (HQ * L))

    for s in range(0, P, chunk):
        sl = slice(s, s + chunk)
        n = len(obs[sl])
        rows = np.arange(n)
        g = doppler_factor(track, obs[sl, None, :], t_heard[sl, None] + m * hop, c)  # (n, M)
        freq = harmonics[:, None, None, None] * f0_grid[None, :, None, None] * g  # (H, Q, n, M)
        fbin = np.rint((freq - spec.f[0]) / spec.df)
        fbin = np.where(np.isfinite(fbin) & (fbin >= 0) & (fbin < F), fbin, F).astype(np.intp)
        # Frame of the curve under the largest shift; column j of a run is shift s_hi - j.
        base = np.rint((t_heard[sl] - t0[sl] - spec.t[0]) / hop).astype(np.intp)
        start = np.clip(base[:, None] - s_hi[sl, None] + m + border, 0, T + 2 * border - L)  # (n, M)
        total = np.zeros((len(harmonics), len(f0_grid), n, L), np.float32)
        for a in range(0, n, piece):
            b = min(a + piece, n)
            acc = np.zeros((len(harmonics), len(f0_grid), b - a, L), np.float32)
            fb, st = fbin[:, :, a:b], start[a:b]
            for k in range(len(m)):
                acc += runs[fb[..., k], st[:, k]]
            total[:, :, a:b] = acc
        sc = total.mean(0) / len(m)  # (Q, n, L)
        sc = sc[:, rows[:, None], s_hi[sl, None] - shifts[sl]]  # (Q, n, nS)
        if per_shift:
            out_per_shift[sl] = sc.max(0)
            continue
        sc = sc.transpose(1, 0, 2).reshape(n, -1)
        best = sc.argmax(1)
        q, j = np.divmod(best, shifts.shape[1])
        out_score[sl] = sc[rows, best]
        out_f0[sl] = f0_grid[q]
        out_shift[sl] = shifts[sl][rows, j]

    if per_shift:
        return out_per_shift
    return EventScore(score=out_score, f0=out_f0, shift=out_shift)


def line_agreement(track: FlightTrack, observers, t0, f0, event: Event, *, tol_hz=1.5, c=SPEED_OF_SOUND) -> np.ndarray:
    """Fraction of the event's traced line that `track`'s modelled curve passes within `tol_hz` of,
    for each observer (n, 3) with recording start t0 (n,) and emitted frequency f0 (n,).

    A flight that made the pass follows its line; a flight whose curve merely scores well by
    crossing other bright lines does not.
    """
    times, freqs = event.line
    obs = np.atleast_2d(np.asarray(observers, dtype=float))
    t0 = np.asarray(t0, dtype=float)[:, None]
    model = np.asarray(f0, dtype=float)[:, None] * doppler_factor(track, obs[:, None, :], t0 + times, c)
    return np.mean(np.abs(model - freqs) < tol_hz, axis=1)
