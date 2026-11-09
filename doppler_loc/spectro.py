"""Long-window, low-frequency spectrograms for finding faint aircraft tones."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter


@dataclass
class Spectrogram:
    t: np.ndarray  # frame centres, seconds from the start of the recording
    f: np.ndarray  # bin frequencies, Hz
    db: np.ndarray  # (F, T) power in dB
    score: np.ndarray  # (F, T) dB above each frequency's median, clipped and smoothed

    @property
    def hop(self) -> float:
        return float(self.t[1] - self.t[0])

    @property
    def df(self) -> float:
        return float(self.f[1] - self.f[0])


def compute_spectrogram(
    audio,
    sr,
    *,
    fmin=20.0,
    fmax=600.0,
    win_s=1.0,
    hop_s=0.5,
    pad=2,
    smooth=(1.0, 0.5),
    clip_db=30.0,
) -> Spectrogram:
    """Spectrogram cropped to [fmin, fmax].

    The `score` image subtracts each frequency's median over time, which removes steady hum
    and the noise floor's colour so that a sum along a curve compares fairly across bands.
    """
    audio = np.asarray(audio, dtype=np.float64)
    nper = int(round(win_s * sr))
    hop = int(round(hop_s * sr))
    nfft = nper * pad
    window = np.hanning(nper)
    frames = sliding_window_view(audio, nper)[::hop]
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    sel = (freqs >= fmin) & (freqs <= fmax)

    power = np.empty((sel.sum(), len(frames)), dtype=np.float32)
    for s in range(0, len(frames), 256):
        spec = np.fft.rfft(frames[s : s + 256] * window, n=nfft, axis=1)[:, sel]
        power[:, s : s + 256] = (np.abs(spec) ** 2).T

    db = 10.0 * np.log10(power + 1e-20)
    score = np.clip(db - np.median(db, axis=1, keepdims=True), 0.0, clip_db)
    score = gaussian_filter(score, sigma=smooth).astype(np.float32)
    t = (np.arange(len(frames)) * hop + nper / 2) / sr
    return Spectrogram(t=t, f=freqs[sel], db=db, score=score)
