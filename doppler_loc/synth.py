"""Synthesise what a microphone would record as propeller aircraft fly past."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter

from .flight import FlightTrack
from .physics import SPEED_OF_SOUND, emission_times


@dataclass
class Source:
    track: FlightTrack  # the true trajectory, not the ADS-B one
    f0: float  # propeller blade-pass frequency at the aircraft, Hz
    level: float = 0.05  # tone amplitude at 1 km, relative to unit-RMS background noise
    harmonics: tuple[float, ...] = (1.0, 0.5, 0.25)
    rpm_drift: float = 0.002  # slow relative wander of engine speed


def synthesize(
    observer,
    sources: list[Source],
    t_start,
    duration,
    *,
    sr=8000,
    c=SPEED_OF_SOUND,
    absorption_db_per_km=1.0,
    hum=0.05,
    seed=0,
) -> np.ndarray:
    """Audio heard at `observer` from `t_start` for `duration` seconds, as float32 in [-1, 1].

    Each aircraft is a harmonic tone whose phase at the microphone is the source phase at the
    emission time, so the Doppler shift comes out of the propagation delay exactly rather than
    from a frequency formula. Background is brown-ish plus white noise and mains hum.
    """
    rng = np.random.default_rng(seed)
    obs = np.asarray(observer, dtype=float)
    n = int(round(duration * sr))
    t_audio = t_start + np.arange(n) / sr
    t_coarse = t_start + np.arange(0.0, duration + 0.04, 0.02)
    signal = np.zeros(n)

    for src in sources:
        tr = src.track
        t_e = emission_times(tr, obs, t_coarse, c)
        r = np.linalg.norm(tr.position(t_e) - obs, axis=-1)
        amp = src.level * (1000.0 / r) * 10.0 ** (-absorption_db_per_km * r / 1000.0 / 20.0)
        amp = np.where(tr.contains(t_e), amp, 0.0)

        grid = np.arange(tr.t_start, tr.t_end + 0.02, 0.02)
        wander = gaussian_filter1d(rng.normal(size=len(grid)), 5.0 / 0.02)
        wander /= wander.std() or 1.0
        freq = src.f0 * (1.0 + src.rpm_drift * wander)
        phase_src = 2.0 * np.pi * np.concatenate([[0.0], np.cumsum(freq[:-1] * 0.02)]) + rng.uniform(0, 2 * np.pi)

        phase = np.interp(t_audio, t_coarse, np.interp(t_e, grid, phase_src))
        a = np.interp(t_audio, t_coarse, amp)
        for h, weight in enumerate(src.harmonics, start=1):
            signal += a * weight * np.sin(h * phase)

    white = rng.normal(size=n)
    brown = lfilter([1.0], [1.0, -0.995], rng.normal(size=n))
    brown /= brown.std()
    noise = (0.5 * white + brown) / np.sqrt(1.25)
    tt = np.arange(n) / sr
    mains = hum * (np.sin(2 * np.pi * 60 * tt) + 0.5 * np.sin(2 * np.pi * 120 * tt + 1.0))

    audio = signal + noise + mains
    return (0.9 * audio / np.abs(audio).max()).astype(np.float32)


def write_wav(path, audio, sr=8000):
    wavfile.write(path, sr, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
