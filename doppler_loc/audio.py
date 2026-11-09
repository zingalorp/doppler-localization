"""Load real recordings (any format FFmpeg reads) as mono float audio."""

from __future__ import annotations

import av
import numpy as np


def load_audio(path, sr=8000) -> np.ndarray:
    """Decode `path`, mix to mono and resample to `sr` Hz.

    Each decoded frame goes straight through FFmpeg's resampler, so only the `sr` Hz result is
    ever held in memory: hours of 44.1 kHz stereo would otherwise take several GB before being
    cut down. Packets that fail to decode are skipped, so a recording cut off mid-write (a
    stream still being saved, or stopped by a timeout) still loads up to where it ends.
    """
    resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
    chunks = []

    def take(frames):
        for out in frames:
            chunks.append(out.to_ndarray().reshape(-1).copy())

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                continue
            for frame in frames:
                frame.pts = None  # resample by sample count; gaps in timestamps don't matter here
                take(resampler.resample(frame))
        take(resampler.resample(None))  # flush the resampler's tail
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
